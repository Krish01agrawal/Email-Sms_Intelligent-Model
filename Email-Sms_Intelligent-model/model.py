"""
Transformer-based Model for Email-SMS Classification and Data Extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding
)
from datasets import Dataset
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from typing import Dict, List, Tuple, Optional
import logging
import os
from tqdm import tqdm
import json
from datetime import datetime

from config import model_config, training_config
from data_preprocessing import TextPreprocessor, DataExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialClassifier(nn.Module):
    """DistilBERT-based model for financial vs non-financial classification"""
    
    def __init__(self, model_name: str, num_labels: int = 2, dropout: float = 0.1):
        super(FinancialClassifier, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT doesn't have pooler_output, so we use the first token [CLS]
        hidden_state = outputs.last_hidden_state
        pooled_output = hidden_state[:, 0]  # Use [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

class FinancialExtractor(nn.Module):
    """DistilBERT-based model for financial data extraction (NER-style)"""
    
    def __init__(self, model_name: str, num_labels: int, dropout: float = 0.1):
        super(FinancialExtractor, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.extractor = nn.Linear(self.transformer.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.extractor(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # Only calculate loss on non-padded tokens
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, logits.size(-1))
            active_labels = torch.where(
                active_loss,
                labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
        
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

class EmailSMSModel:
    """Main model class for email/SMS classification and extraction"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or model_config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add special tokens for financial entities
        special_tokens = {
            'additional_special_tokens': [
                '[AMOUNT]', '[CURRENCY]', '[MERCHANT]', '[DATE]', '[TRANSACTION_TYPE]'
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        # Initialize models
        self.classifier = FinancialClassifier(self.model_name)
        self.extractor = FinancialExtractor(self.model_name, num_labels=6)  # 6 entity types
        
        # Initialize preprocessing components
        self.preprocessor = TextPreprocessor()
        self.data_extractor = DataExtractor()
        
        # Move models to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier.to(self.device)
        self.extractor.to(self.device)
        
        logger.info(f"Model initialized on device: {self.device}")
    
    def tokenize_data(self, texts: List[str], max_length: int = None) -> Dict:
        """Tokenize input texts using DistilBERT tokenizer"""
        max_length = max_length or model_config.max_length
        
        # DistilBERT tokenizer optimization
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
            add_special_tokens=True  # Ensures [CLS] and [SEP] tokens are added
        )
        
        return tokenized
    
    def prepare_dataset(self, df: pd.DataFrame) -> Dataset:
        """Prepare dataset for training"""
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        # Tokenize texts
        tokenized = self.tokenize_data(texts)
        
        # Create dataset
        dataset_dict = {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels
        }
        
        return Dataset.from_dict(dataset_dict)
    
    def train_classifier(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Train the financial classifier"""
        logger.info("Training financial classifier...")
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_df)
        val_dataset = self.prepare_dataset(val_df)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{training_config.output_dir}/classifier",
            num_train_epochs=model_config.num_epochs,
            per_device_train_batch_size=model_config.batch_size,
            per_device_eval_batch_size=model_config.batch_size,
            warmup_steps=model_config.warmup_steps,
            weight_decay=model_config.weight_decay,
            logging_dir=f"{training_config.log_dir}/classifier",
            logging_steps=model_config.logging_steps,
            evaluation_strategy="steps",
            eval_steps=model_config.eval_steps,
            save_steps=model_config.save_steps,
            save_total_limit=model_config.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=model_config.fp16,
            gradient_accumulation_steps=model_config.gradient_accumulation_steps,
            report_to="wandb" if training_config.use_wandb else None,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.classifier,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=self.compute_metrics
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(f"{training_config.output_dir}/classifier")
        
        logger.info("Classifier training completed!")
    
    def train_extractor(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Train the financial data extractor"""
        logger.info("Training financial data extractor...")
        
        # Prepare extraction datasets (only financial texts)
        financial_train = train_df[train_df['label'] == 1]
        financial_val = val_df[val_df['label'] == 1]
        
        if len(financial_train) == 0 or len(financial_val) == 0:
            logger.warning("No financial data available for extractor training")
            return
        
        # Prepare extraction datasets
        train_dataset = self.prepare_extraction_dataset(financial_train)
        val_dataset = self.prepare_extraction_dataset(financial_val)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=f"{training_config.output_dir}/extractor",
            num_train_epochs=model_config.num_epochs,
            per_device_train_batch_size=model_config.batch_size,
            per_device_eval_batch_size=model_config.batch_size,
            warmup_steps=model_config.warmup_steps,
            weight_decay=model_config.weight_decay,
            logging_dir=f"{training_config.log_dir}/extractor",
            logging_steps=model_config.logging_steps,
            evaluation_strategy="steps",
            eval_steps=model_config.eval_steps,
            save_steps=model_config.save_steps,
            save_total_limit=model_config.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=model_config.fp16,
            gradient_accumulation_steps=model_config.gradient_accumulation_steps,
            report_to="wandb" if training_config.use_wandb else None,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.extractor,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=self.compute_extraction_metrics
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(f"{training_config.output_dir}/extractor")
        
        logger.info("Extractor training completed!")
    
    def prepare_extraction_dataset(self, df: pd.DataFrame) -> Dataset:
        """Prepare dataset for extraction training"""
        # This is a simplified version - in practice, you'd need to create NER-style labels
        # For now, we'll use the rule-based extractor to create labels
        
        texts = df['text'].tolist()
        labels = []
        
        for text in texts:
            # Use rule-based extractor to create labels
            extraction = self.data_extractor.extract_transaction_data(text)
            # Convert extraction to NER labels (simplified)
            label = self._create_ner_labels(text, extraction)
            labels.append(label)
        
        # Tokenize texts
        tokenized = self.tokenize_data(texts)
        
        # Create dataset
        dataset_dict = {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels
        }
        
        return Dataset.from_dict(dataset_dict)
    
    def _create_ner_labels(self, text: str, extraction: Dict) -> List[int]:
        """Create NER labels for extraction training (simplified)"""
        # This is a simplified implementation
        # In practice, you'd need more sophisticated labeling
        tokens = self.tokenizer.tokenize(text)
        labels = [0] * len(tokens)  # 0 = O (outside)
        
        # Simple labeling based on extraction results
        if extraction['amount']:
            # Find amount tokens and label them
            amount_str = str(extraction['amount'])
            for i, token in enumerate(tokens):
                if amount_str in token:
                    labels[i] = 1  # 1 = AMOUNT
        
        if extraction['merchant']:
            # Find merchant tokens and label them
            merchant_tokens = self.tokenizer.tokenize(extraction['merchant'])
            for i in range(len(tokens) - len(merchant_tokens) + 1):
                if tokens[i:i+len(merchant_tokens)] == merchant_tokens:
                    for j in range(len(merchant_tokens)):
                        labels[i+j] = 2  # 2 = MERCHANT
        
        return labels
    
    def predict(self, text: str) -> Dict:
        """Predict financial classification and extract data"""
        # Clean text
        cleaned_text = self.preprocessor.clean_text(text)
        
        # Tokenize
        tokenized = self.tokenize_data([cleaned_text])
        input_ids = tokenized['input_ids'].to(self.device)
        attention_mask = tokenized['attention_mask'].to(self.device)
        
        # Classify
        self.classifier.eval()
        with torch.no_grad():
            classifier_output = self.classifier(input_ids=input_ids, attention_mask=attention_mask)
            classifier_logits = classifier_output['logits']
            classifier_probs = F.softmax(classifier_logits, dim=-1)
            is_financial = classifier_probs[0][1].item() > 0.5
        
        # Extract data if financial
        extraction_result = None
        if is_financial:
            # Use rule-based extractor for now
            extraction_result = self.data_extractor.extract_transaction_data(cleaned_text)
            
            # In the future, you could use the trained extractor here
            # self.extractor.eval()
            # with torch.no_grad():
            #     extractor_output = self.extractor(input_ids=input_ids, attention_mask=attention_mask)
            #     extractor_logits = extractor_output['logits']
            #     # Process logits to extract entities
        
        return {
            'text': text,
            'cleaned_text': cleaned_text,
            'is_financial': is_financial,
            'financial_confidence': classifier_probs[0][1].item(),
            'extraction': extraction_result
        }
    
    def batch_predict(self, texts: List[str]) -> List[Dict]:
        """Batch prediction for multiple texts"""
        results = []
        for text in tqdm(texts, desc="Processing texts"):
            result = self.predict(text)
            results.append(result)
        return results
    
    def compute_metrics(self, pred):
        """Compute metrics for classification"""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        acc = accuracy_score(labels, preds)
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def compute_extraction_metrics(self, pred):
        """Compute metrics for extraction"""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        
        # Flatten predictions and labels
        preds_flat = preds.flatten()
        labels_flat = labels.flatten()
        
        # Remove padding tokens
        mask = labels_flat != -100
        preds_flat = preds_flat[mask]
        labels_flat = labels_flat[mask]
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels_flat, preds_flat, average='weighted')
        acc = accuracy_score(labels_flat, preds_flat)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def save_model(self, path: str):
        """Save the complete model"""
        os.makedirs(path, exist_ok=True)
        
        # Save classifier
        torch.save(self.classifier.state_dict(), os.path.join(path, 'classifier.pth'))
        
        # Save extractor
        torch.save(self.extractor.state_dict(), os.path.join(path, 'extractor.pth'))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(path)
        
        # Save config
        config = {
            'model_name': self.model_name,
            'device': str(self.device),
            'timestamp': datetime.now().isoformat()
        }
        with open(os.path.join(path, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load the complete model"""
        # Load classifier
        classifier_path = os.path.join(path, 'classifier.pth')
        if os.path.exists(classifier_path):
            self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
        
        # Load extractor
        extractor_path = os.path.join(path, 'extractor.pth')
        if os.path.exists(extractor_path):
            self.extractor.load_state_dict(torch.load(extractor_path, map_location=self.device))
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        
        logger.info(f"Model loaded from {path}")

if __name__ == "__main__":
    # Test the model
    model = EmailSMSModel()
    
    # Test prediction
    test_text = "Your HDFC Bank account has been debited with Rs. 500 for UPI transaction"
    result = model.predict(test_text)
    print(f"Test result: {result}") 