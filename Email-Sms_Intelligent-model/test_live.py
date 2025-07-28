#!/usr/bin/env python3
"""
Interactive testing script for the financial classifier
"""

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

def load_model():
    """Load the trained model"""
    print("Loading model...")
    model = DistilBertForSequenceClassification.from_pretrained('models/distilbert')
    tokenizer = DistilBertTokenizer.from_pretrained('models/distilbert')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    return model, tokenizer, device

def classify_text(text, model, tokenizer, device):
    """Classify a single piece of text"""
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        return_tensors='pt'
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits).item()
        confidence = torch.softmax(logits, dim=1)[0][prediction].item()
    
    return prediction == 1, confidence

def main():
    """Main interactive loop"""
    model, tokenizer, device = load_model()
    
    print("\nFinancial Message Classifier")
    print("Enter 'q' to quit")
    print("-" * 50)
    
    while True:
        text = input("\nEnter message to classify: ")
        if text.lower() == 'q':
            break
        
        is_financial, confidence = classify_text(text, model, tokenizer, device)
        
        print("\nResults:")
        print(f"Classification: {'FINANCIAL' if is_financial else 'NON-FINANCIAL'}")
        print(f"Confidence: {confidence:.3f}")

if __name__ == "__main__":
    main() 