"""
Configuration file for Email-SMS Intelligent Model
"""
import os
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class ModelConfig:
    """Configuration for the transformer model"""
    model_name: str = "distilbert-base-uncased"  # Using DistilBERT for efficiency
    max_length: int = 512
    batch_size: int = 32  # Larger batch size for DistilBERT
    learning_rate: float = 2e-5
    num_epochs: int = 5   # DistilBERT converges faster
    warmup_steps: int = 300
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 2
    fp16: bool = True
    save_steps: int = 500
    eval_steps: int = 250
    logging_steps: int = 50
    save_total_limit: int = 3

@dataclass
class DataConfig:
    """Configuration for data processing"""
    financial_transactions_file: str = "../datasets/genai_gmail_chat.financial_transactions.csv"
    sms_data_file: str = "../datasets/pluto_money.sms_data.csv"
    email_logs_file: str = "../datasets/pluto_money.email_logs.csv"
    additional_emails_file: str = "../datasets/krishplutomoney all emails gmail_data_117454877979500520700_20250630_012957.csv"
    test_size: float = 0.2
    val_size: float = 0.1
    random_state: int = 42
    
@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing"""
    remove_urls: bool = True
    remove_emails: bool = True
    remove_numbers: bool = False
    remove_punctuation: bool = False
    lowercase: bool = True
    remove_stopwords: bool = False
    min_text_length: int = 10
    max_text_length: int = 1000

@dataclass
class MongoDBConfig:
    """Configuration for MongoDB connection"""
    connection_string: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    database_name: str = "financial_assistant"
    email_logs_collection: str = "email_logs"
    financial_transactions_collection: str = "financial_transactions"
    sms_logs_collection: str = "sms_logs"

@dataclass
class TrainingConfig:
    """Configuration for training"""
    output_dir: str = "./models"
    cache_dir: str = "./cache"
    log_dir: str = "./logs"
    use_wandb: bool = True
    wandb_project: str = "email-sms-classifier"
    wandb_entity: Optional[str] = None
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001

@dataclass
class ExtractionConfig:
    """Configuration for data extraction"""
    confidence_threshold: float = 0.8
    min_amount: float = 0.0
    max_amount: float = 1000000.0
    supported_currencies: List[str] = None
    
    def __post_init__(self):
        if self.supported_currencies is None:
            self.supported_currencies = ["INR", "USD", "EUR", "GBP", "JPY", "CAD", "AUD"]

# Create global config instances
model_config = ModelConfig()
data_config = DataConfig()
preprocessing_config = PreprocessingConfig()
mongodb_config = MongoDBConfig()
training_config = TrainingConfig()
extraction_config = ExtractionConfig()

# Financial keywords for classification
FINANCIAL_KEYWORDS = [
    # Banking
    "bank", "account", "balance", "transaction", "transfer", "payment", "debit", "credit",
    "upi", "atm", "card", "cheque", "deposit", "withdrawal", "loan", "emi", "interest",
    
    # Investment
    "investment", "mutual fund", "stocks", "shares", "portfolio", "dividend", "profit",
    "loss", "trading", "broker", "sebi", "nse", "bse", "ipo", "market",
    
    # Insurance
    "insurance", "premium", "policy", "claim", "coverage", "life insurance", "health insurance",
]

# Transaction types
TRANSACTION_TYPES = [
    "debit", "credit", "payment", "refund", "transfer", "withdrawal", "deposit",
    "purchase", "bill payment", "recharge", "subscription", "investment"
]

# Service categories
SERVICE_CATEGORIES = [
    "Banking", "Investment", "Insurance", "E-commerce", "Food_Delivery",
    "Transportation", "Utilities", "Entertainment", "Education", "Healthcare"
] 