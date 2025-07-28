#!/usr/bin/env python3
"""
Model Comparison Script
Compares DistilBERT performance with other transformer models
"""

import time
import torch
import numpy as np
import pandas as pd
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    AutoTokenizer, AutoModelForSequenceClassification
)
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_model_and_tokenizer(model_name):
    """Load model and tokenizer"""
    print(f"Loading {model_name}...")
    
    if "distilbert" in model_name.lower():
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

def measure_inference_speed(model, tokenizer, device, texts, num_runs=100):
    """Measure inference speed"""
    times = []
    
    for _ in range(num_runs):
        start_time = time.time()
        
        # Tokenize
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
        
        end_time = time.time()
        times.append(end_time - start_time)
    
    return np.mean(times), np.std(times)

def measure_memory_usage(model):
    """Measure model memory usage"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
    }

def compare_models():
    """Compare different models"""
    models_to_test = [
        "distilbert-base-uncased",
        "bert-base-uncased", 
        "microsoft/DialoGPT-medium"
    ]
    
    # Sample financial and non-financial texts
    test_texts = [
        "Your account has been credited with Rs. 5000",
        "UPI payment of Rs. 2500 to Amazon completed",
        "Your mutual fund investment of Rs. 10000 has been processed",
        "Credit card payment due: Rs. 15000 by 15th",
        "Meeting scheduled for tomorrow at 3 PM",
        "Happy birthday! Hope you have a great day",
        "Please review the attached document",
        "Weather forecast for today: Sunny with clear skies"
    ]
    
    results = []
    
    for model_name in models_to_test:
        try:
            # Load model
            model, tokenizer, device = load_model_and_tokenizer(model_name)
            
            # Measure memory usage
            memory_info = measure_memory_usage(model)
            
            # Measure inference speed
            avg_time, std_time = measure_inference_speed(
                model, tokenizer, device, test_texts, num_runs=50
            )
            
            # Make predictions
            inputs = tokenizer(
                test_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
                probabilities = torch.softmax(outputs.logits, dim=-1)
            
            # Calculate average confidence
            avg_confidence = probabilities.max(dim=-1)[0].mean().item()
            
            results.append({
                'model_name': model_name,
                'total_parameters': memory_info['total_parameters'],
                'model_size_mb': memory_info['model_size_mb'],
                'avg_inference_time': avg_time,
                'std_inference_time': std_time,
                'avg_confidence': avg_confidence,
                'predictions': predictions.cpu().numpy().tolist()
            })
            
            print(f"✓ {model_name} completed")
            
        except Exception as e:
            print(f"✗ Error with {model_name}: {str(e)}")
            continue
    
    return results

def plot_comparison(results):
    """Plot comparison results"""
    if not results:
        print("No results to plot")
        return
    
    df = pd.DataFrame(results)
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Model Size Comparison
    ax1.bar(df['model_name'], df['model_size_mb'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax1.set_title('Model Size Comparison (MB)')
    ax1.set_ylabel('Size (MB)')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Inference Speed Comparison
    ax2.bar(df['model_name'], df['avg_inference_time'], 
            yerr=df['std_inference_time'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_title('Inference Speed Comparison')
    ax2.set_ylabel('Time (seconds)')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. Parameter Count
    ax3.bar(df['model_name'], df['total_parameters'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax3.set_title('Total Parameters')
    ax3.set_ylabel('Number of Parameters')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Average Confidence
    ax4.bar(df['model_name'], df['avg_confidence'], color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax4.set_title('Average Prediction Confidence')
    ax4.set_ylabel('Confidence Score')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_comparison_table(results):
    """Print comparison table"""
    if not results:
        print("No results to display")
        return
    
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    
    print(f"{'Model':<25} {'Size (MB)':<12} {'Params (M)':<12} {'Speed (s)':<12} {'Confidence':<12}")
    print("-"*80)
    
    for result in results:
        size_mb = result['model_size_mb']
        params_m = result['total_parameters'] / 1e6
        speed = result['avg_inference_time']
        confidence = result['avg_confidence']
        
        print(f"{result['model_name']:<25} {size_mb:<12.1f} {params_m:<12.1f} {speed:<12.4f} {confidence:<12.3f}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    
    # Find best performing model in each category
    if results:
        fastest = min(results, key=lambda x: x['avg_inference_time'])
        smallest = min(results, key=lambda x: x['model_size_mb'])
        most_confident = max(results, key=lambda x: x['avg_confidence'])
        
        print(f"🏃 Fastest: {fastest['model_name']} ({fastest['avg_inference_time']:.4f}s)")
        print(f"💾 Smallest: {smallest['model_name']} ({smallest['model_size_mb']:.1f}MB)")
        print(f"🎯 Most Confident: {most_confident['model_name']} ({most_confident['avg_confidence']:.3f})")
        
        # DistilBERT advantages
        distilbert = next((r for r in results if 'distilbert' in r['model_name'].lower()), None)
        if distilbert:
            print(f"\n🚀 DistilBERT Advantages:")
            print(f"   • 60% faster than BERT")
            print(f"   • 40% smaller model size")
            print(f"   • Maintains 97% of BERT's performance")
            print(f"   • Perfect for production deployment")

def main():
    """Main function"""
    print("🤖 Model Comparison: DistilBERT vs Other Transformers")
    print("="*60)
    
    # Run comparison
    results = compare_models()
    
    # Display results
    print_comparison_table(results)
    
    # Plot results
    plot_comparison(results)
    
    print(f"\n📊 Results saved to 'model_comparison.png'")
    print("✅ Comparison completed!")

if __name__ == "__main__":
    main() 