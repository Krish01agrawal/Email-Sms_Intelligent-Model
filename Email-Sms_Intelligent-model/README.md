# Email-SMS Intelligent Model

A comprehensive deep learning solution for classifying emails and SMS as financial or non-financial, and extracting structured financial transaction data. Built with **DistilBERT** (optimized for speed and efficiency), TensorFlow, and MongoDB integration.

## 🚀 **Key Advantages with DistilBERT**

- **60% faster** than BERT while maintaining 97% performance
- **40% smaller** model size for efficient deployment
- **Optimized training** with larger batch sizes and faster convergence
- **Production-ready** with excellent accuracy for financial classification

## 🎯 Project Overview

This project addresses the challenge of automatically processing large volumes of emails and SMS to:
1. **Classify** messages as financial or non-financial
2. **Extract** structured transaction data from financial messages
3. **Store** processed data in MongoDB for efficient querying
4. **Integrate** with Agno framework for your financial AI assistant

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Email/SMS     │───▶│  Preprocessing   │───▶│  Transformer    │
│   Input Data    │    │   & Feature      │    │     Model       │
│                 │    │   Extraction     │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   MongoDB       │◀───│  Data Extraction │◀───│  Classification │
│   Storage       │    │   & Structuring  │    │   Results       │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Features

### Core Functionality
- **Binary Classification**: Financial vs Non-financial messages
- **Data Extraction**: Amount, merchant, date, transaction type
- **MongoDB Integration**: Structured storage with indexes
- **Batch Processing**: Efficient handling of large datasets
- **Confidence Scoring**: Model confidence for predictions

### Advanced Features
- **Multi-format Support**: Email and SMS processing
- **Rule-based + ML Hybrid**: Combines regex patterns with transformer models
- **Service Categorization**: Automatic merchant categorization
- **Transaction Summaries**: Aggregated financial insights
- **Configurable Pipeline**: Easy customization via config files

## 📁 Project Structure

```
Email-Sms_Intelligent-model/
├── config.py                      # Configuration management
├── data_preprocessing.py          # Text preprocessing and feature extraction
├── model.py                       # Transformer-based classification model
├── train.py                       # Training script with evaluation
├── train_distilbert.py            # Optimized DistilBERT training script
├── distilbert_training_notebook.ipynb  # Interactive Jupyter notebook
├── mongodb_integration.py         # MongoDB operations and data storage
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── models/                        # Trained model artifacts
├── logs/                          # Training logs and metrics
└── cache/                         # Model cache and temporary files
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- MongoDB (local or cloud)
- CUDA-compatible GPU (optional, for faster training)

### Setup

1. **Clone and navigate to the project:**
```bash
cd Email-Sms_Intelligent-model
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download spaCy model:**
```bash
python -m spacy download en_core_web_sm
```

4. **Set up environment variables:**
```bash
# Create .env file
echo "MONGODB_URI=mongodb://localhost:27017" > .env
```

## 📊 Dataset Preparation

### Supported Formats
- **Email Data**: CSV with columns: `Message`, `Category`
- **Financial Transactions**: CSV with detailed transaction data
- **Custom Format**: Extensible for your specific data structure

### Data Processing
```python
from data_preprocessing import DatasetPreparator

# Prepare datasets
preparator = DatasetPreparator()
train_df, val_df, test_df = preparator.load_and_prepare_data()
```

## 🎯 Model Training

### Quick Start
```bash
# Train with default settings
python train.py

# Train with custom parameters
python train.py --model-name "microsoft/DialoGPT-medium" --epochs 15 --batch-size 32

# Train classifier only (skip extractor)
python train.py --skip-extractor

# Evaluate existing model
python train.py --evaluate-only
```

### Training Options
- `--model-name`: Transformer model to use
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--learning-rate`: Learning rate
- `--max-samples`: Maximum samples per class (for balancing)
- `--skip-extractor`: Skip extractor training
- `--evaluate-only`: Only evaluate, skip training

### Model Performance
Expected performance metrics:
- **Accuracy**: >95%
- **F1 Score**: >94%
- **Precision**: >93%
- **Recall**: >94%

## 🔧 Configuration

### Model Configuration (`config.py`)
```python
model_config = ModelConfig(
    model_name="microsoft/DialoGPT-medium",
    max_length=512,
    batch_size=16,
    learning_rate=2e-5,
    num_epochs=10
)
```

### Data Configuration
```python
data_config = DataConfig(
    train_file="../datasets/mail_data.csv",
    financial_transactions_file="../datasets/genai_gmail_chat.financial_transactions.csv",
    test_size=0.2,
    val_size=0.1
)
```

### MongoDB Configuration
```python
mongodb_config = MongoDBConfig(
    connection_string="mongodb://localhost:27017",
    database_name="financial_assistant",
    email_logs_collection="email_logs",
    financial_transactions_collection="financial_transactions"
)
```

## 🗄️ MongoDB Integration

### Collections Structure

#### Email Logs Collection
```json
{
  "_id": "ObjectId",
  "user_id": "string",
  "email_id": "string",
  "sender": "string",
  "recipient": "string",
  "subject": "string",
  "body": "string",
  "is_financial": "boolean",
  "financial_confidence": "float",
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

#### Financial Transactions Collection
```json
{
  "_id": "ObjectId",
  "user_id": "string",
  "email_id": "string",
  "transaction_type": "string",
  "amount": "float",
  "currency": "string",
  "merchant_canonical": "string",
  "service_category": "string",
  "confidence_score": "float",
  "created_at": "datetime",
  "updated_at": "datetime"
}
```

### Usage Examples

```python
from mongodb_integration import MongoDBManager, EmailSMSProcessor

# Initialize
mongodb = MongoDBManager()
processor = EmailSMSProcessor(mongodb_manager=mongodb)

# Process email
email_data = {
    "user_id": "user123",
    "email_id": "email456",
    "subject": "UPI Transaction Alert",
    "body": "Rs.500 debited for Zomato order"
}
result = processor.process_email(email_data)

# Get user transactions
transactions = mongodb.get_user_transactions("user123")

# Get transaction summary
summary = mongodb.get_transaction_summary("user123")
```

## 🚀 **Training Options**

### Option 1: DistilBERT Training (Recommended)
```bash
# Fast and efficient training with DistilBERT
python train_distilbert.py

# With custom parameters
python train_distilbert.py --batch-size 64 --epochs 3 --learning-rate 3e-5
```

### Option 2: Interactive Jupyter Notebook
```bash
# Launch interactive training notebook
jupyter notebook distilbert_training_notebook.ipynb
```

### Option 3: Standard Training
```bash
# Original training script
python train.py
```

## 🔍 Usage Examples

### Single Message Processing
```python
from model import EmailSMSModel

# Initialize model (now uses DistilBERT by default)
model = EmailSMSModel()

# Predict
text = "Your HDFC Bank account has been debited with Rs. 500 for UPI transaction"
result = model.predict(text)

print(f"Is Financial: {result['is_financial']}")
print(f"Confidence: {result['financial_confidence']:.2f}")
print(f"Extraction: {result['extraction']}")
```

### Batch Processing
```python
from mongodb_integration import EmailSMSProcessor

# Initialize processor
processor = EmailSMSProcessor()

# Process multiple emails
emails = [
    {"user_id": "user1", "subject": "Bank Alert", "body": "..."},
    {"user_id": "user2", "subject": "Newsletter", "body": "..."}
]

results = processor.batch_process_emails(emails)
```

### Data Retrieval
```python
from mongodb_integration import MongoDBManager

mongodb = MongoDBManager()

# Get financial transactions for last month
from datetime import datetime, timedelta
end_date = datetime.now()
start_date = end_date - timedelta(days=30)

transactions = mongodb.get_user_transactions(
    user_id="user123",
    start_date=start_date,
    end_date=end_date,
    transaction_type="debit"
)
```

## 📈 Performance Monitoring

### Training Metrics
- Real-time logging with Weights & Biases
- Confusion matrix visualization
- Training/validation curves
- Model performance reports

### Production Monitoring
- Prediction confidence tracking
- Extraction accuracy metrics
- Processing time monitoring
- Error rate tracking

## 🔧 Customization

### Adding New Financial Keywords
```python
# In config.py
FINANCIAL_KEYWORDS.extend([
    "your_new_keyword",
    "another_keyword"
])
```

### Custom Service Categories
```python
# In config.py
SERVICE_CATEGORIES.extend([
    "Your_Category",
    "Another_Category"
])
```

### Model Architecture Changes
```python
# In model.py, modify FinancialClassifier class
class CustomFinancialClassifier(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        # Your custom architecture
```

## 🚀 Integration with Agno Framework

### API Endpoints
```python
# Example Agno integration
from agno import Agno
from mongodb_integration import EmailSMSProcessor

app = Agno()

@app.post("/process-email")
async def process_email(email_data: dict):
    processor = EmailSMSProcessor()
    result = processor.process_email(email_data)
    return result

@app.get("/user-transactions/{user_id}")
async def get_transactions(user_id: str):
    mongodb = MongoDBManager()
    transactions = mongodb.get_user_transactions(user_id)
    return transactions
```

### Query Processing
```python
# Example query processing for your financial assistant
def process_financial_query(query: str, user_id: str):
    mongodb = MongoDBManager()
    
    if "total transactions" in query.lower():
        summary = mongodb.get_transaction_summary(user_id)
        return f"Total transactions: {summary['total_transactions']}"
    
    if "debit" in query.lower():
        transactions = mongodb.get_user_transactions(
            user_id, transaction_type="debit"
        )
        return f"Found {len(transactions)} debit transactions"
```

## 🧪 Testing

### Unit Tests
```bash
# Run tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_model.py
```

### Integration Tests
```bash
# Test MongoDB integration
python mongodb_integration.py

# Test data preprocessing
python data_preprocessing.py
```

## 📊 Model Evaluation

### Metrics
- **Classification**: Accuracy, F1, Precision, Recall
- **Extraction**: Entity-level F1, Precision, Recall
- **Performance**: Processing time, throughput

### Evaluation Script
```bash
python train.py --evaluate-only
```

## 🔒 Security Considerations

- **Data Privacy**: All user data is encrypted at rest
- **Access Control**: MongoDB authentication required
- **Input Validation**: Sanitized text processing
- **Error Handling**: Graceful failure handling

## 📝 Logging

### Log Levels
- **INFO**: General processing information
- **WARNING**: Non-critical issues
- **ERROR**: Processing failures
- **DEBUG**: Detailed debugging information

### Log Files
- `logs/training.log`: Training progress
- `logs/processing.log`: Data processing logs
- `logs/errors.log`: Error tracking

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size
   - Use gradient accumulation
   - Enable mixed precision training

2. **MongoDB Connection Issues**
   - Check connection string
   - Verify network connectivity
   - Check authentication credentials

3. **Model Performance Issues**
   - Increase training data
   - Adjust hyperparameters
   - Try different model architectures

### Getting Help
- Create an issue on GitHub
- Check the documentation
- Review the logs for error details

## 🎯 Roadmap

### Phase 1 (Current)
- ✅ Basic classification model
- ✅ Data extraction pipeline
- ✅ MongoDB integration
- ✅ Training framework

### Phase 2 (Next)
- 🔄 Advanced NER for extraction
- 🔄 Multi-language support
- 🔄 Real-time processing
- 🔄 API optimization

### Phase 3 (Future)
- 📋 Custom model fine-tuning
- 📋 Advanced analytics
- 📋 Mobile deployment
- 📋 Enterprise features

---

**Built with ❤️ for intelligent financial data processing** 