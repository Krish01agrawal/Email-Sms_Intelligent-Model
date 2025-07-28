"""
Training Script for Email-SMS Intelligent Model
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import json
import wandb
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import training_config, model_config, data_config
from data_preprocessing import DatasetPreparator
from model import EmailSMSModel

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{training_config.log_dir}/training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
                "max_length": model_config.max_length,
                "fp16": model_config.fp16,
                "gradient_accumulation_steps": model_config.gradient_accumulation_steps
            }
        )
        logger.info("Weights & Biases initialized")

def create_directories():
    """Create necessary directories"""
    directories = [
        training_config.output_dir,
        training_config.cache_dir,
        training_config.log_dir,
        f"{training_config.output_dir}/classifier",
        f"{training_config.output_dir}/extractor",
        f"{training_config.log_dir}/classifier",
        f"{training_config.log_dir}/extractor"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def plot_training_metrics(history, save_path):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Accuracy
    axes[0, 1].plot(history['train_accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    
    # F1 Score
    axes[1, 0].plot(history['train_f1'], label='Train F1')
    axes[1, 0].plot(history['val_f1'], label='Validation F1')
    axes[1, 0].set_title('Training and Validation F1 Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    
    # Precision/Recall
    axes[1, 1].plot(history['train_precision'], label='Train Precision')
    axes[1, 1].plot(history['val_precision'], label='Validation Precision')
    axes[1, 1].plot(history['train_recall'], label='Train Recall')
    axes[1, 1].plot(history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Training and Validation Precision/Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Training metrics plot saved to {save_path}")

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Financial', 'Financial'],
                yticklabels=['Non-Financial', 'Financial'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Confusion matrix saved to {save_path}")

def evaluate_model(model, test_df, save_dir):
    """Evaluate the trained model"""
    logger.info("Evaluating model...")
    
    # Get predictions
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()
    
    predictions = model.batch_predict(test_texts)
    pred_labels = [1 if pred['is_financial'] else 0 for pred in predictions]
    pred_confidences = [pred['financial_confidence'] for pred in predictions]
    
    # Calculate metrics
    report = classification_report(test_labels, pred_labels, 
                                 target_names=['Non-Financial', 'Financial'],
                                 output_dict=True)
    
    # Save detailed results
    results = {
        'predictions': predictions,
        'true_labels': test_labels,
        'predicted_labels': pred_labels,
        'confidences': pred_confidences,
        'classification_report': report
    }
    
    with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print metrics
    logger.info("Classification Report:")
    logger.info(classification_report(test_labels, pred_labels, 
                                    target_names=['Non-Financial', 'Financial']))
    
    # Plot confusion matrix
    plot_confusion_matrix(test_labels, pred_labels, 
                         os.path.join(save_dir, 'confusion_matrix.png'))
    
    # Log to wandb
    if training_config.use_wandb:
        wandb.log({
            'test_accuracy': report['accuracy'],
            'test_f1': report['weighted avg']['f1-score'],
            'test_precision': report['weighted avg']['precision'],
            'test_recall': report['weighted avg']['recall']
        })
    
    return results

def save_model_artifacts(model, save_dir, training_info):
    """Save model artifacts and metadata"""
    # Save the model
    model.save_model(save_dir)
    
    # Save training information
    training_info['timestamp'] = datetime.now().isoformat()
    training_info['model_path'] = save_dir
    
    with open(os.path.join(save_dir, 'training_info.json'), 'w') as f:
        json.dump(training_info, f, indent=2, default=str)
    
    # Save dataset statistics
    dataset_stats = {
        'total_samples': training_info.get('total_samples', 0),
        'train_samples': training_info.get('train_samples', 0),
        'val_samples': training_info.get('val_samples', 0),
        'test_samples': training_info.get('test_samples', 0),
        'class_distribution': training_info.get('class_distribution', {})
    }
    
    with open(os.path.join(save_dir, 'dataset_stats.json'), 'w') as f:
        json.dump(dataset_stats, f, indent=2)
    
    logger.info(f"Model artifacts saved to {save_dir}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Email-SMS Classification Model')
    parser.add_argument('--model-name', type=str, default=None,
                       help='Model name to use for training')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate for training')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples per class for balancing')
    parser.add_argument('--skip-extractor', action='store_true',
                       help='Skip training the extractor model')
    parser.add_argument('--evaluate-only', action='store_true',
                       help='Only evaluate the model, skip training')
    
    args = parser.parse_args()
    
    # Update config with command line arguments
    if args.model_name:
        model_config.model_name = args.model_name
    if args.epochs:
        model_config.num_epochs = args.epochs
    if args.batch_size:
        model_config.batch_size = args.batch_size
    if args.learning_rate:
        model_config.learning_rate = args.learning_rate
    if args.max_samples:
        data_config.max_samples_per_class = args.max_samples
    
    # Create directories
    create_directories()
    
    # Setup wandb
    setup_wandb()
    
    # Initialize model
    logger.info(f"Initializing model: {model_config.model_name}")
    model = EmailSMSModel(model_config.model_name)
    
    if args.evaluate_only:
        # Load existing model and evaluate
        model_path = f"{training_config.output_dir}/classifier"
        if os.path.exists(model_path):
            model.load_model(model_path)
            logger.info("Loaded existing model for evaluation")
        else:
            logger.error("No existing model found for evaluation")
            return
        
        # Load test data
        test_df = pd.read_csv('processed_test.csv')
        evaluate_model(model, test_df, training_config.output_dir)
        return
    
    # Prepare data
    logger.info("Preparing datasets...")
    preparator = DatasetPreparator()
    train_df, val_df, test_df = preparator.load_and_prepare_data()
    
    # Save processed datasets
    train_df.to_csv('processed_train.csv', index=False)
    val_df.to_csv('processed_val.csv', index=False)
    test_df.to_csv('processed_test.csv', index=False)
    
    # Training information
    training_info = {
        'model_name': model_config.model_name,
        'total_samples': len(train_df) + len(val_df) + len(test_df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'class_distribution': {
            'train': train_df['label'].value_counts().to_dict(),
            'val': val_df['label'].value_counts().to_dict(),
            'test': test_df['label'].value_counts().to_dict()
        },
        'hyperparameters': {
            'batch_size': model_config.batch_size,
            'learning_rate': model_config.learning_rate,
            'num_epochs': model_config.num_epochs,
            'max_length': model_config.max_length,
            'fp16': model_config.fp16
        }
    }
    
    logger.info(f"Training info: {training_info}")
    
    # Train classifier
    logger.info("Starting classifier training...")
    try:
        model.train_classifier(train_df, val_df)
        logger.info("Classifier training completed successfully!")
    except Exception as e:
        logger.error(f"Error during classifier training: {e}")
        return
    
    # Train extractor (optional)
    if not args.skip_extractor:
        logger.info("Starting extractor training...")
        try:
            model.train_extractor(train_df, val_df)
            logger.info("Extractor training completed successfully!")
        except Exception as e:
            logger.error(f"Error during extractor training: {e}")
            logger.warning("Continuing without extractor...")
    
    # Evaluate model
    logger.info("Evaluating model...")
    evaluation_results = evaluate_model(model, test_df, training_config.output_dir)
    
    # Save model artifacts
    save_model_artifacts(model, training_config.output_dir, training_info)
    
    # Log final metrics to wandb
    if training_config.use_wandb:
        wandb.finish()
    
    logger.info("Training completed successfully!")
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    print(f"Model: {model_config.model_name}")
    print(f"Total samples: {training_info['total_samples']}")
    print(f"Train samples: {training_info['train_samples']}")
    print(f"Validation samples: {training_info['val_samples']}")
    print(f"Test samples: {training_info['test_samples']}")
    print(f"Test Accuracy: {evaluation_results['classification_report']['accuracy']:.4f}")
    print(f"Test F1 Score: {evaluation_results['classification_report']['weighted avg']['f1-score']:.4f}")
    print(f"Model saved to: {training_config.output_dir}")
    print("="*50)

if __name__ == "__main__":
    main() 