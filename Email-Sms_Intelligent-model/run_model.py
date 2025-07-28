#!/usr/bin/env python3
"""
Complete pipeline for training, testing and evaluating the financial classifier
"""

import os
import sys
import pandas as pd
import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from data_preprocessing import TextPreprocessor, DatasetPreparator
from config import model_config

def load_and_prepare_data():
    """Load and prepare training data"""
    print("Loading datasets...")
    preparator = DatasetPreparator()
    train_df, val_df, test_df = preparator.load_and_prepare_data()
    
    print(f"\nDataset sizes:")
    print(f"Training: {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df

def train_model(train_df, val_df):
    """Train the model"""
    print("\nInitializing model...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2
    )
    
    # Prepare datasets
    train_encodings = tokenizer(
        train_df['text'].tolist(),
        truncation=True,
        padding=True,
        max_length=model_config.max_length,
        return_tensors='pt'
    )
    val_encodings = tokenizer(
        val_df['text'].tolist(),
        truncation=True,
        padding=True,
        max_length=model_config.max_length,
        return_tensors='pt'
    )
    
    train_dataset = {
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': torch.tensor(train_df['is_financial'].tolist())
    }
    val_dataset = {
        'input_ids': val_encodings['input_ids'],
        'attention_mask': val_encodings['attention_mask'],
        'labels': torch.tensor(val_df['is_financial'].tolist())
    }
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=model_config.num_epochs,
        per_device_train_batch_size=model_config.batch_size,
        per_device_eval_batch_size=model_config.batch_size,
        warmup_steps=model_config.warmup_steps,
        weight_decay=model_config.weight_decay,
        logging_dir='./logs',
        logging_steps=model_config.logging_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    print("\nStarting training...")
    trainer.train()
    
    # Save model
    print("\nSaving model...")
    os.makedirs('models/distilbert', exist_ok=True)
    model.save_pretrained('models/distilbert')
    tokenizer.save_pretrained('models/distilbert')
    
    return model, tokenizer

def evaluate_model(model, tokenizer, test_df):
    """Evaluate model on test set"""
    print("\nEvaluating model...")
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    predictions = []
    
    for text in tqdm(test_df['text'].tolist(), desc="Evaluating"):
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=model_config.max_length,
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions.append(torch.argmax(outputs.logits).item())
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(test_df['is_financial'], predictions))
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        pd.crosstab(test_df['is_financial'], predictions),
        annot=True,
        fmt='d',
        cmap='Blues'
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

def test_custom_data(model, tokenizer, custom_file='custom_test_data.txt'):
    """Test model on custom data"""
    print(f"\nTesting custom data from {custom_file}...")
    
    with open(custom_file, 'r') as f:
        texts = f.readlines()
    
    texts = [text.strip() for text in texts if text.strip()]
    
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    results = []
    for text in texts:
        inputs = tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=model_config.max_length,
            return_tensors='pt'
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits).item()
            confidence = torch.softmax(logits, dim=1)[0][prediction].item()
        
        results.append({
            'text': text,
            'is_financial': prediction == 1,
            'confidence': confidence
        })
    
    print("\nCustom Test Results:")
    print("-" * 80)
    for result in results:
        print(f"Text: {result['text']}")
        print(f"Prediction: {'FINANCIAL' if result['is_financial'] else 'NON-FINANCIAL'}")
        print(f"Confidence: {result['confidence']:.3f}")
        print("-" * 80)

def main():
    """Main function"""
    # Check if we should load existing model
    if os.path.exists('models/distilbert') and '--retrain' not in sys.argv:
        print("Loading existing model...")
        model = DistilBertForSequenceClassification.from_pretrained('models/distilbert')
        tokenizer = DistilBertTokenizer.from_pretrained('models/distilbert')
    else:
        # Train new model
        train_df, val_df, test_df = load_and_prepare_data()
        model, tokenizer = train_model(train_df, val_df)
        evaluate_model(model, tokenizer, test_df)
    
    # Test custom data
    if os.path.exists('custom_test_data.txt'):
        test_custom_data(model, tokenizer)
    else:
        print("\nNo custom test data found. Create 'custom_test_data.txt' to test custom messages.")

if __name__ == "__main__":
    main() 