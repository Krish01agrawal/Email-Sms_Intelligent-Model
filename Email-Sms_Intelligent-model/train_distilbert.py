#!/usr/bin/env python3
"""
Specialized Training Script for DistilBERT-based Email-SMS Classification
Optimized for speed, efficiency, and better performance
"""

import argparse
import os
import sys
import logging
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    DistilBertConfig, TrainingArguments, Trainer,
    DataCollatorWithPadding, EarlyStoppingCallback
)
from datasets import Dataset
import wandb
from tqdm import tqdm

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import model_config, data_config, training_config
from data_preprocessing import DatasetPreparator, TextPreprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DistilBERTClassifier:
    """Optimized DistilBERT classifier for financial vs non-financial classification"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased"):
        self.model_name = model_name
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=2,
            problem_type="single_label_classification"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        logger.info(f"DistilBERT model loaded on {self.device}")
    
    def tokenize_texts(self, texts: list, max_length: int = 512):
        """Tokenize texts with DistilBERT tokenizer"""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
    
    def prepare_dataset(self, df: pd.DataFrame) -> Dataset:
        """Prepare dataset for DistilBERT training"""
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        
        # Tokenize
        tokenized = self.tokenize_texts(texts, model_config.max_length)
        
        # Create dataset
        dataset_dict = {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': labels
        }
        
        return Dataset.from_dict(dataset_dict)
    
    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        """Train the DistilBERT classifier with optimizations"""
        logger.info("Starting DistilBERT training...")
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_df)
        val_dataset = self.prepare_dataset(val_df)
        
        # Optimized training arguments for DistilBERT
        training_args = TrainingArguments(
            output_dir=f"{training_config.output_dir}/distilbert_classifier",
            num_train_epochs=model_config.num_epochs,
            per_device_train_batch_size=model_config.batch_size,
            per_device_eval_batch_size=model_config.batch_size * 2,  # Larger eval batch
            warmup_steps=model_config.warmup_steps,
            weight_decay=model_config.weight_decay,
            learning_rate=model_config.learning_rate,
            logging_dir=f"{training_config.log_dir}/distilbert",
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
            dataloader_num_workers=4,  # Optimize data loading
            remove_unused_columns=False,
            push_to_hub=False,
            # DistilBERT specific optimizations
            optim="adamw_torch",
            lr_scheduler_type="linear",
            warmup_ratio=0.1,
        )
        
        # Initialize trainer with early stopping
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=training_config.early_stopping_patience)]
        )
        
        # Train the model
        logger.info("Training started...")
        train_result = trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(f"{training_config.output_dir}/distilbert_classifier")
        
        # Log training results
        logger.info(f"Training completed! Loss: {train_result.training_loss:.4f}")
        
        return trainer, train_result
    
    def compute_metrics(self, pred):
        """Compute classification metrics"""
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def predict(self, text: str) -> dict:
        """Make prediction on single text"""
        self.model.eval()
        
        # Tokenize
        inputs = self.tokenize_texts([text], model_config.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(outputs.logits, dim=-1)
        
        return {
            'text': text,
            'is_financial': bool(prediction.item()),
            'confidence': probs[0][prediction.item()].item(),
            'probabilities': probs[0].cpu().numpy().tolist()
        }
    
    def evaluate(self, test_df: pd.DataFrame) -> dict:
        """Evaluate model on test set"""
        logger.info("Evaluating model...")
        
        predictions = []
        true_labels = []
        
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Evaluating"):
            pred = self.predict(row['text'])
            predictions.append(pred['is_financial'])
            true_labels.append(row['label'])
        
        # Generate classification report
        report = classification_report(true_labels, predictions, output_dict=True)
        
        # Generate confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix - DistilBERT Financial Classifier')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f"{training_config.output_dir}/distilbert_confusion_matrix.png")
        plt.close()
        
        return {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'accuracy': report['accuracy'],
            'f1_score': report['weighted avg']['f1-score']
        }

def setup_wandb():
    """Setup Weights & Biases logging"""
    if training_config.use_wandb:
        wandb.init(
            project=training_config.wandb_project,
            entity=training_config.wandb_entity,
            config={
                "model_name": model_config.model_name,
                "batch_size": model_config.batch_size,
                "learning_rate": model_config.learning_rate,
                "num_epochs": model_config.num_epochs,
                "max_length": model_config.max_length
            }
        )

def create_directories():
    """Create necessary directories"""
    Path(training_config.output_dir).mkdir(parents=True, exist_ok=True)
    Path(training_config.log_dir).mkdir(parents=True, exist_ok=True)
    Path(training_config.cache_dir).mkdir(parents=True, exist_ok=True)

def save_training_info(trainer, train_result, evaluation_results):
    """Save training information"""
    training_info = {
        "model_name": model_config.model_name,
        "training_loss": train_result.training_loss,
        "evaluation_results": evaluation_results,
        "training_config": {
            "batch_size": model_config.batch_size,
            "learning_rate": model_config.learning_rate,
            "num_epochs": model_config.num_epochs,
            "max_length": model_config.max_length
        },
        "timestamp": datetime.now().isoformat()
    }
    
    with open(f"{training_config.output_dir}/distilbert_training_info.json", 'w') as f:
        json.dump(training_info, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Train DistilBERT Email-SMS Classifier')
    parser.add_argument('--model-name', type=str, default=None, help='Model name to use')
    parser.add_argument('--batch-size', type=int, default=None, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=None, help='Learning rate')
    parser.add_argument('--evaluate-only', action='store_true', help='Only evaluate existing model')
    args = parser.parse_args()
    
    # Update config with command line arguments
    if args.model_name:
        model_config.model_name = args.model_name
    if args.batch_size:
        model_config.batch_size = args.batch_size
    if args.epochs:
        model_config.num_epochs = args.epochs
    if args.learning_rate:
        model_config.learning_rate = args.learning_rate
    
    # Create directories
    create_directories()
    
    # Setup wandb
    setup_wandb()
    
    # Initialize model
    classifier = DistilBERTClassifier(model_config.model_name)
    
    if args.evaluate_only:
        # Load existing model and evaluate
        model_path = f"{training_config.output_dir}/distilbert_classifier"
        if os.path.exists(model_path):
            classifier.model = DistilBertForSequenceClassification.from_pretrained(model_path)
            classifier.model.to(classifier.device)
            
            # Load test data
            preparator = DatasetPreparator()
            _, _, test_df = preparator.load_and_prepare_data()
            
            # Evaluate
            results = classifier.evaluate(test_df)
            print(f"Test Accuracy: {results['accuracy']:.4f}")
            print(f"Test F1 Score: {results['f1_score']:.4f}")
        else:
            logger.error(f"Model not found at {model_path}")
        return
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    preparator = DatasetPreparator()
    train_df, val_df, test_df = preparator.load_and_prepare_data()
    
    # Save processed datasets
    train_df.to_csv(f"{training_config.output_dir}/distilbert_train.csv", index=False)
    val_df.to_csv(f"{training_config.output_dir}/distilbert_val.csv", index=False)
    test_df.to_csv(f"{training_config.output_dir}/distilbert_test.csv", index=False)
    
    logger.info(f"Training set: {len(train_df)} samples")
    logger.info(f"Validation set: {len(val_df)} samples")
    logger.info(f"Test set: {len(test_df)} samples")
    
    # Train the model
    trainer, train_result = classifier.train(train_df, val_df)
    
    # Evaluate on test set
    evaluation_results = classifier.evaluate(test_df)
    
    # Save training information
    save_training_info(trainer, train_result, evaluation_results)
    
    # Log final results
    logger.info("=" * 50)
    logger.info("TRAINING COMPLETED!")
    logger.info("=" * 50)
    logger.info(f"Test Accuracy: {evaluation_results['accuracy']:.4f}")
    logger.info(f"Test F1 Score: {evaluation_results['f1_score']:.4f}")
    logger.info(f"Training Loss: {train_result.training_loss:.4f}")
    logger.info(f"Model saved to: {training_config.output_dir}/distilbert_classifier")
    
    if training_config.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    main() 