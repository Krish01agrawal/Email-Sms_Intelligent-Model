"""
Data Preprocessing Module for Email-SMS Intelligent Model
Handles text preprocessing, dataset preparation, and feature engineering
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List, Dict, Optional
import logging
from tqdm import tqdm
import json
from datetime import datetime
import spacy

from config import (
    data_config, preprocessing_config, FINANCIAL_KEYWORDS, 
    TRANSACTION_TYPES, SERVICE_CATEGORIES
)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextPreprocessor:
    """Text preprocessing utilities"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        if preprocessing_config.lowercase:
            text = text.lower()
        
        # Remove URLs
        if preprocessing_config.remove_urls:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        if preprocessing_config.remove_emails:
            text = re.sub(r'\S+@\S+', '', text)
        
        # Remove numbers (optional)
        if preprocessing_config.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation (optional)
        if preprocessing_config.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords from text"""
        if not preprocessing_config.remove_stopwords:
            return text
        
        words = word_tokenize(text)
        filtered_words = [word for word in words if word.lower() not in self.stop_words]
        return ' '.join(filtered_words)
    
    def lemmatize_text(self, text: str) -> str:
        """Lemmatize text"""
        words = word_tokenize(text)
        lemmatized_words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)
    
    def extract_financial_features(self, text: str) -> Dict[str, any]:
        """Extract financial-related features from text"""
        features = {
            'has_financial_keywords': False,
            'financial_keyword_count': 0,
            'has_amount': False,
            'amounts': [],
            'has_currency': False,
            'currencies': [],
            'has_date': False,
            'dates': [],
            'has_transaction_type': False,
            'transaction_types': [],
            'has_merchant': False,
            'merchants': [],
            'text_length': len(text),
            'word_count': len(text.split())
        }
        
        # Check for financial keywords
        text_lower = text.lower()
        for keyword in FINANCIAL_KEYWORDS:
            if keyword.lower() in text_lower:
                features['has_financial_keywords'] = True
                features['financial_keyword_count'] += 1
        
        # Extract amounts (various formats)
        amount_patterns = [
            r'₹\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',  # ₹1,234.56
            r'rs\.?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)',  # Rs. 1,234.56
            r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:rs|inr)',  # 1,234.56 Rs
            r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*rupees?',  # 1,234.56 rupees
            r'\$(\d+(?:,\d{3})*(?:\.\d{2})?)',  # $1,234.56
            r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*dollars?',  # 1,234.56 dollars
        ]
        
        for pattern in amount_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    amount = float(match.replace(',', ''))
                    if extraction_config.min_amount <= amount <= extraction_config.max_amount:
                        features['amounts'].append(amount)
                        features['has_amount'] = True
                except ValueError:
                    continue
        
        # Extract currencies
        currency_patterns = [
            r'\b(inr|rs|rupees?)\b',
            r'\b(usd|\$|dollars?)\b',
            r'\b(eur|€|euros?)\b',
            r'\b(gbp|£|pounds?)\b'
        ]
        
        for pattern in currency_patterns:
            matches = re.findall(pattern, text_lower)
            features['currencies'].extend(matches)
            if matches:
                features['has_currency'] = True
        
        # Extract dates
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # DD/MM/YYYY
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',  # YYYY/MM/DD
            r'\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{2,4}',  # DD Month YYYY
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text_lower)
            features['dates'].extend(matches)
            if matches:
                features['has_date'] = True
        
        # Check for transaction types
        for txn_type in TRANSACTION_TYPES:
            if txn_type.lower() in text_lower:
                features['transaction_types'].append(txn_type)
                features['has_transaction_type'] = True
        
        # Extract merchants using spaCy NER
        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON']:
                features['merchants'].append(ent.text)
                features['has_merchant'] = True
        
        return features

class DatasetPreparator:
    """Prepare datasets for training"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        
    def load_and_prepare_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and prepare training data"""
        logger.info("Loading datasets...")
        
        # Load financial transactions (these are labeled as financial)
        financial_df = pd.read_csv(data_config.financial_transactions_file)
        logger.info(f"Loaded financial transactions: {len(financial_df)} samples")
        
        # Load SMS data
        sms_df = pd.read_csv("../datasets/pluto_money.sms_data.csv")
        logger.info(f"Loaded SMS data: {len(sms_df)} samples")
        
        # Load email logs
        email_logs_df = pd.read_csv("../datasets/pluto_money.email_logs.csv")
        logger.info(f"Loaded email logs: {len(email_logs_df)} samples")
        
        # Load additional emails
        additional_emails_df = pd.read_csv("../datasets/krishplutomoney all emails gmail_data_117454877979500520700_20250630_012957.csv")
        logger.info(f"Loaded additional emails: {len(additional_emails_df)} samples")
        
        # Process each dataset
        financial_processed = self._prepare_financial_data(financial_df)
        sms_processed = self._prepare_sms_data(sms_df)
        email_logs_processed = self._prepare_email_logs(email_logs_df)
        additional_emails_processed = self._prepare_additional_emails(additional_emails_df)
        
        # Combine all datasets
        combined_df = pd.concat([
            financial_processed,
            sms_processed,
            email_logs_processed,
            additional_emails_processed
        ], ignore_index=True)
        
        # Split into train/val/test
        train_df, temp_df = train_test_split(
            combined_df, 
            test_size=data_config.test_size + data_config.val_size,
            random_state=data_config.random_state,
            stratify=combined_df['label']
        )
        
        val_size_adjusted = data_config.val_size / (data_config.test_size + data_config.val_size)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1 - val_size_adjusted,
            random_state=data_config.random_state,
            stratify=temp_df['label']
        )
        
        logger.info(f"Dataset splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def _prepare_financial_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare financial transactions dataset"""
        processed_data = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing financial data"):
            # Get text from snippet
            text = str(row.get('snippet', ''))
            
            if not text or text == 'nan':
                continue
            
            # Clean text
            cleaned_text = self.preprocessor.clean_text(text)
            
            # Skip if text is too short
            if len(cleaned_text) < preprocessing_config.min_text_length:
                continue
            
            # All entries in financial transactions are financial
            processed_data.append({
                'text': cleaned_text,
                'original_text': text,
                'label': 1,  # Financial
                'confidence': 1.0
            })
        
        return pd.DataFrame(processed_data)
    
    def _prepare_sms_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare SMS dataset"""
        processed_data = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing SMS data"):
            text = str(row.get('message', ''))
            
            if not text or text == 'nan':
                continue
            
            # Clean text
            cleaned_text = self.preprocessor.clean_text(text)
            
            # Skip if text is too short
            if len(cleaned_text) < preprocessing_config.min_text_length:
                continue
            
            # Extract features to determine if financial
            features = self.preprocessor.extract_financial_features(cleaned_text)
            is_financial = features['has_financial_indicators']
            
            processed_data.append({
                'text': cleaned_text,
                'original_text': text,
                'label': 1 if is_financial else 0,
                'confidence': features['confidence_score']
            })
        
        return pd.DataFrame(processed_data)
    
    def _prepare_email_logs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare email logs dataset"""
        processed_data = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing email logs"):
            # Combine subject and body
            subject = str(row.get('subject', ''))
            body = str(row.get('body', ''))
            text = f"{subject} {body}".strip()
            
            if not text or text == 'nan':
                continue
            
            # Clean text
            cleaned_text = self.preprocessor.clean_text(text)
            
            # Skip if text is too short
            if len(cleaned_text) < preprocessing_config.min_text_length:
                continue
            
            # Extract features to determine if financial
            features = self.preprocessor.extract_financial_features(cleaned_text)
            is_financial = features['has_financial_indicators']
            
            processed_data.append({
                'text': cleaned_text,
                'original_text': text,
                'label': 1 if is_financial else 0,
                'confidence': features['confidence_score']
            })
        
        return pd.DataFrame(processed_data)
    
    def _prepare_additional_emails(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare additional emails dataset"""
        processed_data = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing additional emails"):
            text = str(row.get('snippet', ''))
            
            if not text or text == 'nan':
                continue
            
            # Clean text
            cleaned_text = self.preprocessor.clean_text(text)
            
            # Skip if text is too short
            if len(cleaned_text) < preprocessing_config.min_text_length:
                continue
            
            # Extract features to determine if financial
            features = self.preprocessor.extract_financial_features(cleaned_text)
            is_financial = features['has_financial_indicators']
            
            processed_data.append({
                'text': cleaned_text,
                'original_text': text,
                'label': 1 if is_financial else 0,
                'confidence': features['confidence_score']
            })
        
        return pd.DataFrame(processed_data)

class DataExtractor:
    """Extract structured data from financial emails/SMS"""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
    
    def extract_transaction_data(self, text: str, confidence_threshold: float = 0.8) -> Dict:
        """Extract transaction data from text"""
        text_lower = text.lower()
        
        # Initialize extraction result
        extraction = {
            'transaction_type': None,
            'amount': None,
            'currency': 'INR',  # Default to INR
            'merchant': None,
            'date': None,
            'confidence_score': 0.0,
            'extraction_confidence': 0.0,
            'extracted_fields': []
        }
        
        # Extract amount
        amount_patterns = [
            (r'₹\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', 1.0),
            (r'rs\.?\s*(\d+(?:,\d{3})*(?:\.\d{2})?)', 0.9),
            (r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:rs|inr)', 0.8),
            (r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*rupees?', 0.8),
        ]
        
        for pattern, confidence in amount_patterns:
            match = re.search(pattern, text_lower)
            if match:
                try:
                    amount = float(match.group(1).replace(',', ''))
                    if extraction_config.min_amount <= amount <= extraction_config.max_amount:
                        extraction['amount'] = amount
                        extraction['extracted_fields'].append('amount')
                        extraction['confidence_score'] += confidence
                        break
                except ValueError:
                    continue
        
        # Extract transaction type
        for txn_type in TRANSACTION_TYPES:
            if txn_type.lower() in text_lower:
                extraction['transaction_type'] = txn_type
                extraction['extracted_fields'].append('transaction_type')
                extraction['confidence_score'] += 0.7
                break
        
        # Extract merchant using spaCy NER
        doc = nlp(text)
        merchants = []
        for ent in doc.ents:
            if ent.label_ in ['ORG', 'PERSON']:
                merchants.append(ent.text)
        
        if merchants:
            extraction['merchant'] = merchants[0]  # Take the first merchant
            extraction['extracted_fields'].append('merchant')
            extraction['confidence_score'] += 0.6
        
        # Extract date
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text_lower)
            if match:
                extraction['date'] = match.group(0)
                extraction['extracted_fields'].append('date')
                extraction['confidence_score'] += 0.5
                break
        
        # Calculate final confidence
        if extraction['extracted_fields']:
            extraction['extraction_confidence'] = extraction['confidence_score'] / len(extraction['extracted_fields'])
        
        return extraction

if __name__ == "__main__":
    # Test the preprocessing
    preparator = DatasetPreparator()
    train_df, val_df, test_df = preparator.load_and_prepare_data()
    
    print("Dataset preparation completed!")
    print(f"Train samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Save processed datasets
    train_df.to_csv('processed_train.csv', index=False)
    val_df.to_csv('processed_val.csv', index=False)
    test_df.to_csv('processed_test.csv', index=False)
    
    print("Processed datasets saved!") 