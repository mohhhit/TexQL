# TexQL - Natural Language to SQL/MongoDB Query Generator

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Transformers](https://img.shields.io/badge/Transformers-4.36-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.29-red)
![License](https://img.shields.io/badge/License-MIT-green)

TexQL is an NLP-based system that converts natural language queries into SQL and MongoDB statements for CRUD operations. Built using T5 transformer models and trained on synthetic industry data.

## ⚡ Quick Commands

```bash
# Setup (one-time)
python -m venv venv
.\venv\Scripts\Activate.ps1          # Windows
pip install -r requirements-app.txt

# Generate training data
python data_generation.py

# Test trained model
python demo.py

# Run web app
streamlit run app.py

# CLI inference
python inference.py --query "your query here" --type both
```

## 🌟 Features

- **Unified Model Architecture**: Single model generates both SQL and MongoDB queries
- **CRUD Operations**: Supports Create, Read, Update, and Delete operations
- **Balanced Training**: Interleaved sampling prevents catastrophic forgetting
- **Intelligent Post-Processing**: Fixes column hallucination in CREATE queries
- **Synthetic Data Generation**: Automated generation of 12,000 balanced training samples
- **Google Colab Ready**: Training notebook optimized for Colab with GPU support
- **Interactive UI**: Beautiful Streamlit-based web interface with adjustable parameters
- **Query History**: Track and export your query history
- **CLI Tool**: Command-line interface for batch processing
- **CPU Compatible**: Runs on CPU for inference, no GPU required for deployment

## 📁 Project Structure

```
TexQL/
├── data_generation.py          # Synthetic data generation script
├── training_colab_balanced.ipynb  # Balanced training notebook (Colab)
├── app.py                      # Streamlit frontend application
├── demo.py                     # Test script for trained models
├── inference.py                # Command-line inference tool
├── requirements.txt            # Full dependencies (training + app)
├── requirements-app.txt        # Minimal dependencies (app only, CPU)
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
├── README.md                  # This file
├── QUICKSTART.md              # Quick start guide
├── LOCAL_SETUP.md             # Local setup instructions
├── LOCAL_TRAINING.md          # Local training guide
├── data/                      # Generated training data
│   ├── training_data.csv
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
└── models/                    # Trained models
    └── (your trained model files)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd TexQL

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# Install dependencies for running the app (CPU-only)
pip install -r requirements-app.txt

# OR install full dependencies (for training/development)
pip install -r requirements.txt
```

**Note**: `requirements-app.txt` contains minimal dependencies for running the Streamlit app, demo, and data generation (CPU-only). Use `requirements.txt` for full training capabilities.

### 2. Generate Training Data

```bash
python data_generation.py
```

This will generate:
- 12,000 balanced training samples (6,000 SQL + 6,000 MongoDB)
- 1,500 samples per operation (CREATE, READ, UPDATE, DELETE) per language
- Train/Validation/Test splits (80/10/10)
- Output in `data/` directory

### 3. Train Model on Google Colab

**Recommended**: Use Google Colab for faster training with GPU

1. Upload `training_colab_balanced.ipynb` to Google Colab
2. Upload your generated data files to Google Drive
3. Run all cells in the notebook
4. Download the trained unified model

**Training Details:**
- **Unified Model**: Single model trained on both SQL and MongoDB simultaneously
- **Balanced Training**: Interleaved sampling ensures each batch has 50% SQL + 50% MongoDB
- **Prevents Catastrophic Forgetting**: Model maintains quality on both languages
- **Expected Time**: ~2-3 hours on Colab T4 GPU
- **Expected Accuracy**: 70-85% per language with >80% balance score

### 4. Run the Application

```bash
# Make sure virtual environment is activated
# Windows:
.\venv\Scripts\Activate.ps1
# Linux/Mac:
source venv/bin/activate

# Run the Streamlit app
streamlit run app.py

# OR test with demo script
python demo.py

# OR use CLI tool
python inference.py --query "Show all employees"
```

The app will open in your browser at `http://localhost:8501`

**Note**: The model will automatically use GPU if available, otherwise it will use CPU. Performance on CPU is sufficient for inference.

## 📊 Data Generation Details

The synthetic data generator creates queries for 5 database tables:

- **employees**: employee_id, name, email, department, salary, hire_date, age
- **departments**: department_id, department_name, manager_id, budget, location
- **projects**: project_id, project_name, start_date, end_date, budget, status
- **orders**: order_id, customer_name, product_name, quantity, order_date, total_amount
- **products**: product_id, product_name, category, price, stock_quantity, supplier

### Generation Process:

1. **SELECT Queries**: Simple selects, filters, aggregations, sorting
2. **INSERT Queries**: Single record insertions with sample data
3. **UPDATE Queries**: Updates with conditions, increments
4. **DELETE Queries**: Deletions with various conditions

## 🎯 Training Process

### Model Architecture

- **Base Model**: FLAN-T5-base (248M parameters)
- **Task**: Sequence-to-sequence translation
- **Input Format**: `"translate to sql: <natural language>"`
- **Output**: SQL or MongoDB query

### Training Configuration

```python
Model: google/flan-t5-base
Epochs: 10
Batch Size: 8
Learning Rate: 3e-4
Max Input Length: 256
Max Output Length: 512
```

### Training Steps:

1. Load and tokenize data
2. Configure training arguments
3. Train with Seq2SeqTrainer
4. Evaluate on validation set
5. Calculate exact match accuracy on test set
6. Save model and tokenizer

### Expected Performance:

- **Training Loss**: Should converge to < 0.5
- **Validation Loss**: Around 0.3-0.5
- **Exact Match Accuracy**: 60-85% (varies by query complexity)

## 🖥️ Using the Streamlit App

### Features:

1. **Example Queries**: Click pre-defined examples to try the model
2. **Custom Input**: Enter your own natural language queries
3. **Dual Generation**: Get both SQL and MongoDB queries
4. **Parameter Control**: Adjust temperature and beam search
5. **Query History**: View and export previous queries
6. **Copy Queries**: One-click copy to clipboard

### Configuration:

In the sidebar, you can:
- Set model paths
- Adjust generation parameters
- View database schema
- Reload models

## 📝 Example Usage

### Python API

```python
from app import TexQLModel

# Load model
sql_model = TexQLModel("models/texql-sql-final", target_type="sql")

# Generate query
query = sql_model.generate_query("Show all employees with salary greater than 50000")
print(query)
# Output: SELECT * FROM employees WHERE salary > 50000;
```

### Command Line

```bash
# Generate data
python data_generation.py

# Run app
streamlit run app.py
```

## 🔧 Advanced Configuration

### Training Customization

Edit the training notebook to customize:

- Model size (t5-small, t5-base, t5-large, flan-t5-large)
- Training epochs
- Batch size
- Learning rate
- Generation parameters

### Data Customization

Edit `data_generation.py` to:

- Add new tables/collections
- Modify query templates
- Change sample value generation
- Adjust data distribution

### Environment Variables

Create a `.env` file from `.env.example`:

```bash
WANDB_API_KEY=your_key_here  # Optional: for training tracking
MODEL_NAME=google/flan-t5-base
MAX_LENGTH=512
BATCH_SIZE=8
```

## 📈 Performance Metrics

### Data Generation
- **Speed**: ~1000 queries/second
- **Diversity**: 4 operations × 5 tables × multiple templates
- **Quality**: Validated schema-compliant queries

### Model Training
- **GPU Memory**: ~6-8 GB (T4 GPU)
- **Training Time**: 30-60 minutes per model
- **Inference Speed**: ~100-200 queries/second (GPU)

### Query Accuracy
- **Simple SELECT**: 85-95%
- **Complex SELECT**: 70-85%
- **INSERT**: 80-90%
- **UPDATE**: 75-85%
- **DELETE**: 75-85%

## 🛠️ Troubleshooting

### Common Issues:

**1. CUDA Out of Memory**
```python
# Reduce batch size in training_args
per_device_train_batch_size=4
gradient_accumulation_steps=4
```

**2. Model Not Loading**
```bash
# Check model path
ls models/texql-sql-final/
# Should contain: config.json, pytorch_model.bin, tokenizer files
```

**3. Low Accuracy**
```python
# Generate more training data
python data_generation.py  # Increase total_samples
# Train for more epochs
num_train_epochs=15
```

**4. Streamlit Issues**
```bash
# Clear cache and restart
streamlit cache clear
streamlit run app.py
```

## 📚 Dependencies

### Running the App (requirements-app.txt):
Minimal dependencies for running the Streamlit app, demo, and data generation on CPU:
- `transformers>=4.36.0` - HuggingFace Transformers
- `torch>=2.0.0` - PyTorch (CPU version)
- `streamlit>=1.29.0` - Web interface
- `pandas>=2.1.0` - Data manipulation
- `numpy>=1.26.0` - Numerical operations
- `faker>=22.0.0` - Synthetic data generation
- `tqdm>=4.66.0` - Progress bars
- `python-dotenv>=1.0.0` - Environment variables

### Full Training (requirements.txt):
Complete dependencies including GPU support, training tools, and database connectors:
- All dependencies from requirements-app.txt
- `datasets>=2.16.0` - Dataset handling for training
- `accelerate>=0.25.0` - GPU acceleration
- `sentencepiece>=0.1.99` - Tokenization
- `wandb` - Training tracking (optional)
- `pymongo` - MongoDB integration (optional)
- `sqlalchemy` - SQL database integration (optional)

**Recommendation**: Use `requirements-app.txt` for deployment and inference, use `requirements.txt` for training and development.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Additional Databases**: PostgreSQL, MySQL specific syntax
2. **Complex Queries**: JOINs, subqueries, CTEs
3. **Query Optimization**: Suggest indexes, explain plans
4. **Schema Learning**: Automatic schema extraction
5. **Error Handling**: Query validation and correction

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

Created for NLP course project - Text to SQL/MongoDB Query Generation

## 🙏 Acknowledgments

- HuggingFace Transformers team
- Google T5/FLAN-T5 models
- Streamlit team
- Open source community

## 📞 Support

For issues, questions, or suggestions:
1. Check the troubleshooting section
2. Review the example queries
3. Examine the training logs
4. Open an issue on GitHub

## 🗺️ Roadmap

- [ ] Support for JOINs and complex queries
- [ ] Multi-table query generation
- [ ] Query explanation and visualization
- [ ] Support for more databases (PostgreSQL, MySQL)
- [ ] API endpoint deployment
- [ ] Docker containerization
- [ ] Query optimization suggestions
- [ ] Interactive schema builder

## 📊 Citation

If you use this project in your research, please cite:

```bibtex
@software{texql2024,
  title={TexQL: Natural Language to SQL/MongoDB Query Generator},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/texql}
}
```

---

**Happy Querying! 🚀**
