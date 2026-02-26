"""
TexQL - Natural Language to SQL/MongoDB Query Generator
Streamlit Frontend Application
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="TexQL - NL to Query Generator",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .query-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #1E88E5;
    }
    .success-box {
        background-color: #d4edda;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #28a745;
    }
    .example-query {
        background-color: #fff3cd;
        border-radius: 5px;
        padding: 10px;
        margin: 5px 0;
        cursor: pointer;
    }
    .example-query:hover {
        background-color: #ffe69c;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-size: 1.1rem;
        padding: 0.5rem 1rem;
        border-radius: 10px;
        border: none;
        margin-top: 1rem;
    }
    .stButton>button:hover {
        background-color: #1565C0;
    }
</style>
""", unsafe_allow_html=True)


def extract_columns_from_nl(natural_language_query):
    """Extract table name and column names from natural language query"""
    import re
    
    nl = natural_language_query.lower().strip()
    
    # Extract table name
    table_match = re.search(r'(?:table|collection)\s+(?:named|called)?\s*(\w+)', nl)
    table_name = table_match.group(1) if table_match else None
    
    # Extract column names - look for patterns like "columns as X, Y, Z" or "with X, Y, Z"
    columns = []
    
    # Pattern 1: "columns as/named X, Y, Z"
    col_match = re.search(r'columns?\s+(?:as|named|like|called)?\s*([^,]+(?:,\s*[^,]+)*)', nl)
    if col_match:
        col_text = col_match.group(1)
        # Split by comma or 'and'
        columns = re.split(r',|\s+and\s+', col_text)
        columns = [c.strip() for c in columns if c.strip()]
    
    # Pattern 2: "add columns X, Y, Z" 
    if not columns:
        col_match = re.search(r'(?:add|with)\s+(?:columns?)?\s*([^,]+(?:,\s*[^,]+)*)', nl)
        if col_match:
            col_text = col_match.group(1)
            columns = re.split(r',|\s+and\s+', col_text)
            columns = [c.strip() for c in columns if c.strip()]
    
    return table_name, columns


def fix_create_table_sql(generated_sql, table_name, requested_columns):
    """Replace hallucinated columns with actual requested columns in CREATE TABLE"""
    import re
    
    if not table_name or not requested_columns:
        return generated_sql
    
    # Check if it's a CREATE TABLE query
    if not re.search(r'CREATE\s+TABLE', generated_sql, re.IGNORECASE):
        return generated_sql
    
    # Default data types for common column patterns
    def infer_type(col_name):
        col_lower = col_name.lower()
        if 'id' in col_lower:
            return 'INT PRIMARY KEY'
        elif any(word in col_lower for word in ['name', 'title', 'description', 'address', 'city']):
            return 'VARCHAR(100)'
        elif any(word in col_lower for word in ['email']):
            return 'VARCHAR(100)'
        elif any(word in col_lower for word in ['phone', 'contact', 'mobile']):
            return 'VARCHAR(20)'
        elif any(word in col_lower for word in ['date', 'created', 'updated']):
            return 'DATE'
        elif any(word in col_lower for word in ['price', 'salary', 'amount', 'cost']):
            return 'DECIMAL(10,2)'
        elif any(word in col_lower for word in ['age', 'quantity', 'count', 'stock']):
            return 'INT'
        elif any(word in col_lower for word in ['status', 'type', 'category']):
            return 'VARCHAR(50)'
        else:
            return 'VARCHAR(100)'
    
    # Build column definitions
    col_defs = []
    for col in requested_columns:
        col_clean = col.strip()
        if col_clean:
            col_type = infer_type(col_clean)
            col_defs.append(f"{col_clean} {col_type}")
    
    # Replace the column definitions in the SQL
    # Pattern: CREATE TABLE table_name (...)
    new_sql = re.sub(
        r'(CREATE\s+TABLE\s+' + re.escape(table_name) + r'\s*\()[^)]+(\))',
        r'\1' + ', '.join(col_defs) + r'\2',
        generated_sql,
        flags=re.IGNORECASE
    )
    
    return new_sql


def fix_create_collection_mongo(generated_mongo, table_name, requested_columns):
    """Fix MongoDB createCollection to use correct collection name and sample document"""
    import re
    
    if not table_name:
        return generated_mongo
    
    # Check if it's a create operation
    if not any(word in generated_mongo.lower() for word in ['create', 'insert']):
        return generated_mongo
    
    # Build sample document with requested columns
    doc_fields = []
    for col in requested_columns:
        col_clean = col.strip()
        if col_clean:
            # Provide example values based on column name
            if 'id' in col_clean.lower():
                doc_fields.append(f'"{col_clean}": 1')
            elif any(word in col_clean.lower() for word in ['name', 'title']):
                doc_fields.append(f'"{col_clean}": "sample_name"')
            elif 'email' in col_clean.lower():
                doc_fields.append(f'"{col_clean}": "user@example.com"')
            elif any(word in col_clean.lower() for word in ['phone', 'contact']):
                doc_fields.append(f'"{col_clean}": "1234567890"')
            else:
                doc_fields.append(f'"{col_clean}": "sample_value"')
    
    # Create proper MongoDB command
    if doc_fields:
        fixed_mongo = f"db.{table_name}.insertOne({{{', '.join(doc_fields)}}});"
    else:
        fixed_mongo = f"db.createCollection('{table_name}');"
    
    return fixed_mongo


class TexQLModel:
    """Unified model wrapper for SQL/MongoDB generation"""
    
    def __init__(self, model_path):
        """Initialize the model for inference"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            self.loaded = True
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            self.loaded = False
    
    def generate_query(self, natural_language_query, target_type='sql', temperature=0.3, 
                      num_beams=10, repetition_penalty=1.2, length_penalty=0.8):
        """Generate SQL or MongoDB query from natural language
        
        Args:
            natural_language_query: The user's natural language query
            target_type: 'sql' or 'mongodb' to specify output format
            temperature: Sampling temperature (lower = more focused)
            num_beams: Number of beams for beam search
            repetition_penalty: Penalty for repeating tokens (>1.0 discourages repetition)
            length_penalty: Penalty for length (>1.0 encourages longer, <1.0 encourages shorter)
        """
        if not self.loaded:
            return "Model not loaded"
        
        input_text = f"translate to {target_type}: {natural_language_query}"
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=256,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=512,
                num_beams=num_beams,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                no_repeat_ngram_size=3,  # Prevent repeating 3-grams
                early_stopping=True,
                do_sample=False  # Use greedy/beam search (more deterministic)
            )
        
        generated_query = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # ✅ POST-PROCESSING: Fix hallucinated columns in CREATE queries
        if any(word in natural_language_query.lower() for word in ['create', 'add columns']):
            table_name, requested_columns = extract_columns_from_nl(natural_language_query)
            
            if table_name and requested_columns:
                if target_type == 'sql':
                    generated_query = fix_create_table_sql(generated_query, table_name, requested_columns)
                elif target_type == 'mongodb':
                    generated_query = fix_create_collection_mongo(generated_query, table_name, requested_columns)
        
        return generated_query


@st.cache_resource
def load_model(model_path):
    """Load the unified TexQL model (cached)"""
    model = None
    
    if os.path.exists(model_path):
        model = TexQLModel(model_path)
    
    return model


def save_query_history(nl_query, sql_query, mongodb_query):
    """Save query to history"""
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    st.session_state.history.append({
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'natural_language': nl_query,
        'sql': sql_query,
        'mongodb': mongodb_query
    })


def main():
    # Header
    st.markdown('<div class="main-header">🔍 TexQL</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Natural Language to SQL/MongoDB Query Generator</div>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model path
        st.subheader("Model Path")
        model_path = st.text_input(
            "TexQL Model Path",
            value="models",
            help="Path to the unified TexQL model (generates both SQL and MongoDB)"
        )
        
        # Generation parameters
        st.subheader("Generation Parameters")
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.0,
            value=0.3,  # ✅ Lower default = less hallucination
            step=0.1,
            help="Lower = more focused, Higher = more creative"
        )
        num_beams = st.slider(
            "Beam Search Width",
            min_value=1,
            max_value=10,
            value=10,  # ✅ Higher value = more accurate results
            help="Higher values improve accuracy (recommended: keep at 10)"
        )
        repetition_penalty = st.slider(
            "Repetition Penalty",
            min_value=1.0,
            max_value=2.0,
            value=1.2,  # ✅ Discourages adding extra unwanted columns
            step=0.1,
            help="Higher = less repetition (prevents hallucinating extra columns)"
        )
        length_penalty = st.slider(
            "Length Penalty",
            min_value=0.5,
            max_value=1.5,
            value=0.8,  # ✅ Prefer shorter outputs
            step=0.1,
            help="Lower = prefer shorter outputs, Higher = prefer longer outputs"
        )
        
        # Load models button
        if st.button("🔄 Load/Reload Models"):
            st.cache_resource.clear()
            st.rerun()
        
        # Database schema info
        st.subheader("📊 Database Schema")
        with st.expander("View Available Tables"):
            st.markdown("""
            **employees**
            - employee_id, name, email
            - department, salary, hire_date, age
            
            **departments**
            - department_id, department_name
            - manager_id, budget, location
            
            **projects**
            - project_id, project_name
            - start_date, end_date, budget, status
            
            **orders**
            - order_id, customer_name
            - product_name, quantity
            - order_date, total_amount
            
            **products**
            - product_id, product_name
            - category, price
            - stock_quantity, supplier
            """)
    
    # Load model
    with st.spinner("Loading model..."):
        model = load_model(model_path)
    
    # Model status
    if model and model.loaded:
        device_info = "🎮 GPU" if model.device == "cuda" else "💻 CPU"
        st.success(f"✅ TexQL Model Loaded ({device_info})")
        st.info("💡 This model generates both SQL and MongoDB queries")
    else:
        st.error("⚠️ Model Not Available - Please check the model path")
    
    # Main interface
    st.markdown("---")
    
    # Example queries
    st.subheader("💡 Example Queries")
    examples = [
        "Show all employees",
        "Find employees where salary is greater than 50000",
        "Get all departments with budget more than 100000",
        "Insert a new employee with name John Doe, email john@example.com, department Engineering",
        "Update employees set department to Sales where employee_id is 101",
        "Delete orders with total_amount less than 1000",
        "Count all products in Electronics category",
        "Show top 10 employees ordered by salary",
    ]
    
    example_cols = st.columns(4)
    for idx, example in enumerate(examples):
        with example_cols[idx % 4]:
            if st.button(f"📝 {example[:30]}...", key=f"ex_{idx}"):
                st.session_state.user_query = example
    
    # Query input
    st.markdown("---")
    st.subheader("🔤 Enter Your Query")
    
    user_query = st.text_area(
        "Natural Language Query",
        value=st.session_state.get('user_query', ''),
        height=100,
        placeholder="e.g., Show all employees with salary greater than 50000"
    )
    
    # Generate button
    if st.button("🚀 Generate Queries"):
        if not user_query.strip():
            st.warning("Please enter a query")
        elif not model or not model.loaded:
            st.error("Model is not loaded. Please check the model path and reload.")
        else:
            with st.spinner("Generating queries..."):
                # Generate both SQL and MongoDB from the same model
                sql_query = model.generate_query(
                    user_query,
                    target_type='sql',
                    temperature=temperature,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty
                )
                
                mongodb_query = model.generate_query(
                    user_query,
                    target_type='mongodb',
                    temperature=temperature,
                    num_beams=num_beams,
                    repetition_penalty=repetition_penalty,
                    length_penalty=length_penalty
                )
                
                # Save to history
                save_query_history(user_query, sql_query, mongodb_query)
                
                # Display results
                st.markdown("---")
                st.success("✅ Queries Generated Successfully!")
                
                # Input query
                st.markdown('<div class="query-box">', unsafe_allow_html=True)
                st.markdown("**📝 Your Query:**")
                st.code(user_query, language="text")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Results in columns
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🗄️ SQL Query")
                    st.code(sql_query, language="sql")
                    
                    # Copy button
                    if st.button("📋 Copy SQL", key="copy_sql"):
                        st.session_state.clipboard = sql_query
                        st.success("Copied to clipboard!")
                
                with col2:
                    st.markdown("### 🍃 MongoDB Query")
                    st.code(mongodb_query, language="javascript")
                    
                    # Copy button
                    if st.button("📋 Copy MongoDB", key="copy_mongo"):
                        st.session_state.clipboard = mongodb_query
                        st.success("Copied to clipboard!")
    
    # Query history
    if 'history' in st.session_state and st.session_state.history:
        st.markdown("---")
        st.subheader("📚 Query History")
        
        # Clear history button
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
        
        # Display history in expander
        for idx, entry in enumerate(reversed(st.session_state.history[-10:])):
            with st.expander(f"Query {len(st.session_state.history) - idx}: {entry['natural_language'][:50]}..."):
                st.markdown(f"**Timestamp:** {entry['timestamp']}")
                st.markdown(f"**Natural Language:** {entry['natural_language']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**SQL:**")
                    if entry['sql']:
                        st.code(entry['sql'], language="sql")
                
                with col2:
                    st.markdown("**MongoDB:**")
                    if entry['mongodb']:
                        st.code(entry['mongodb'], language="javascript")
        
        # Export history
        if st.button("💾 Export History"):
            history_json = json.dumps(st.session_state.history, indent=2)
            st.download_button(
                label="Download History (JSON)",
                data=history_json,
                file_name=f"texql_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>TexQL - Natural Language to Query Generator</p>
        <p>Powered by T5 Transformer Models | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
