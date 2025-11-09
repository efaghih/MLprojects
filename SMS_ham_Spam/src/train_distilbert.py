"""
DistilBERT Model Training Module for SMS Spam Classification

This module implements a modern deep learning baseline using:
- DistilBERT: A lightweight, distilled version of BERT
- Hugging Face Transformers: For easy model training and evaluation

DistilBERT is 40% smaller and 60% faster than BERT while retaining
97% of its language understanding capabilities.
"""

import pandas as pd
import numpy as np
import os
import torch
from datetime import datetime
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(data_dir='data/processed'):
    """
    Load train, validation, and test datasets.
    
    Args:
        data_dir (str): Directory containing the processed CSV files
        
    Returns:
        DatasetDict: Hugging Face dataset with train/val/test splits
    """
    print(f"Loading data from '{data_dir}/'...")
    
    # Load CSV files
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    
    print(f"✓ Training samples: {len(train_df)}")
    print(f"✓ Validation samples: {len(val_df)}")
    print(f"✓ Test samples: {len(test_df)}")
    
    # Convert to Hugging Face Dataset format
    # Rename 'label' to 'labels' (required by Trainer)
    train_dataset = Dataset.from_pandas(train_df.rename(columns={'label': 'labels'}))
    val_dataset = Dataset.from_pandas(val_df.rename(columns={'label': 'labels'}))
    test_dataset = Dataset.from_pandas(test_df.rename(columns={'label': 'labels'}))
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    return dataset_dict


def prepare_tokenizer(model_name='distilbert-base-uncased', max_length=128):
    """
    Load and configure the tokenizer.
    
    Args:
        model_name (str): Name of the pre-trained model
        max_length (int): Maximum sequence length (default: 128)
        
    Returns:
        AutoTokenizer: Configured tokenizer
    """
    print(f"\nLoading tokenizer: {model_name}")
    print(f"  - Max length: {max_length} tokens")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print("✓ Tokenizer loaded")
    
    return tokenizer


def tokenize_dataset(dataset_dict, tokenizer, max_length=128):
    """
    Tokenize all text in the dataset.
    
    The tokenizer converts text to input IDs and attention masks:
    - input_ids: Token indices in the vocabulary
    - attention_mask: 1 for real tokens, 0 for padding
    
    Args:
        dataset_dict (DatasetDict): Raw datasets
        tokenizer: Hugging Face tokenizer
        max_length (int): Maximum sequence length
        
    Returns:
        DatasetDict: Tokenized datasets
    """
    print("\nTokenizing datasets...")
    
    def tokenize_function(examples):
        """Tokenize a batch of examples."""
        return tokenizer(
            examples['text'],
            padding='max_length',  # Pad to max_length
            truncation=True,       # Truncate longer sequences
            max_length=max_length
        )
    
    # Tokenize all splits
    tokenized_datasets = dataset_dict.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing"
    )
    
    print("✓ Tokenization complete")
    print(f"  - Features: {tokenized_datasets['train'].column_names}")
    
    return tokenized_datasets


def load_model(model_name='distilbert-base-uncased', num_labels=2):
    """
    Load pre-trained DistilBERT model for sequence classification.
    
    Args:
        model_name (str): Name of the pre-trained model
        num_labels (int): Number of output classes (default: 2 for binary)
        
    Returns:
        AutoModelForSequenceClassification: Pre-trained model
    """
    print(f"\nLoading model: {model_name}")
    print(f"  - Task: Binary sequence classification")
    print(f"  - Number of labels: {num_labels}")
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✓ Model loaded on device: {device}")
    
    # Print model size
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Total parameters: {num_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    return model


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics during training.
    
    This function is called by the Trainer after each evaluation.
    
    Args:
        eval_pred: Tuple of (predictions, labels)
        
    Returns:
        dict: Dictionary of computed metrics
    """
    predictions, labels = eval_pred
    
    # Get predicted class (argmax of logits)
    preds = np.argmax(predictions, axis=1)
    
    # Get probabilities for spam class
    probs = torch.softmax(torch.tensor(predictions), dim=1)[:, 1].numpy()
    
    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', pos_label=1
    )
    
    accuracy = accuracy_score(labels, preds)
    roc_auc = roc_auc_score(labels, probs)
    pr_auc = average_precision_score(labels, probs, pos_label=1)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc
    }


def create_training_args(output_dir='models/distilbert',
                        batch_size=16,
                        num_epochs=3,
                        learning_rate=2e-5):
    """
    Create training arguments for the Trainer.
    
    Args:
        output_dir (str): Directory to save model checkpoints
        batch_size (int): Training batch size (default: 16)
        num_epochs (int): Number of training epochs (default: 3)
        learning_rate (float): Learning rate (default: 2e-5)
        
    Returns:
        TrainingArguments: Configuration for training
    """
    print("\nConfiguring training arguments:")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Optimizer: AdamW")
    print(f"  - LR scheduler: Linear with warmup")
    
    args = TrainingArguments(
        output_dir=output_dir,
        
        # Training hyperparameters
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=0.01,           # L2 regularization
        warmup_steps=100,            # Linear warmup
        
        # Evaluation strategy
        eval_strategy="epoch",       # Evaluate after each epoch
        save_strategy="epoch",       # Save checkpoint after each epoch
        load_best_model_at_end=True, # Load best model at the end
        metric_for_best_model="f1",  # Use F1 score to select best model
        greater_is_better=True,
        
        # Logging
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=50,
        report_to="none",            # Disable wandb/tensorboard
        
        # Performance
        fp16=torch.cuda.is_available(),  # Mixed precision (if GPU available)
        dataloader_num_workers=0,        # Windows compatibility
        
        # Reproducibility
        seed=42,
        
        # Save space
        save_total_limit=2,          # Keep only 2 checkpoints
    )
    
    print("✓ Training arguments configured")
    
    return args


def train_model(model, tokenized_datasets, training_args, tokenizer):
    """
    Train the DistilBERT model using Hugging Face Trainer.
    
    Args:
        model: Pre-trained model
        tokenized_datasets (DatasetDict): Tokenized datasets
        training_args (TrainingArguments): Training configuration
        tokenizer: Tokenizer (for processing)
        
    Returns:
        Trainer: Trained model
    """
    print("\n" + "="*70)
    print("TRAINING DISTILBERT MODEL")
    print("="*70)
    
    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    print("\nStarting training...")
    print("-"*70)
    
    # Train the model
    trainer.train()
    
    print("\n✓ Training complete!")
    
    return trainer


def evaluate_model(trainer, tokenized_datasets, output_dir='models/distilbert'):
    """
    Evaluate the trained model on all datasets.
    
    Args:
        trainer (Trainer): Trained model
        tokenized_datasets (DatasetDict): Tokenized datasets
        output_dir (str): Directory to save results
        
    Returns:
        dict: Evaluation results for all splits
    """
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    results = {}
    
    for split_name in ['train', 'validation', 'test']:
        print(f"\n{'-'*70}")
        print(f"{split_name.upper()} SET PERFORMANCE")
        print(f"{'-'*70}")
        
        # Get predictions
        predictions = trainer.predict(tokenized_datasets[split_name])
        
        # Extract predictions and labels
        logits = predictions.predictions
        labels = predictions.label_ids
        preds = np.argmax(logits, axis=1)
        probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
        
        # Compute metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average='binary', pos_label=1
        )
        accuracy = accuracy_score(labels, preds)
        roc_auc = roc_auc_score(labels, probs)
        pr_auc = average_precision_score(labels, probs, pos_label=1)
        cm = confusion_matrix(labels, preds)
        
        # Print metrics
        print(f"\n📊 Metrics:")
        print(f"  • Accuracy:            {accuracy:.4f}")
        print(f"  • Precision (Spam):    {precision:.4f}")
        print(f"  • Recall (Spam):       {recall:.4f}")
        print(f"  • F1 Score (Spam):     {f1:.4f}")
        print(f"  • ROC-AUC:             {roc_auc:.4f}")
        print(f"  • PR-AUC:              {pr_auc:.4f}")
        
        print(f"\n📈 Confusion Matrix:")
        print(f"                  Predicted")
        print(f"                Ham    Spam")
        print(f"  Actual  Ham   {cm[0,0]:5d}  {cm[0,1]:5d}")
        print(f"          Spam  {cm[1,0]:5d}  {cm[1,1]:5d}")
        
        print(f"\n📋 Classification Report:")
        print(classification_report(labels, preds, 
                                    target_names=['Ham', 'Spam'],
                                    digits=4))
        
        # Store results
        results[split_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': cm,
            'predictions': preds,
            'probabilities': probs,
            'labels': labels
        }
    
    # Save results
    save_results(results, output_dir)
    
    return results


def save_results(results, output_dir):
    """
    Save evaluation results and visualizations.
    
    Args:
        results (dict): Evaluation results
        output_dir (str): Directory to save results
    """
    print("\n" + "-"*70)
    print("SAVING RESULTS")
    print("-"*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save metrics summary
    metrics_path = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DISTILBERT MODEL - PERFORMANCE METRICS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: DistilBERT (distilbert-base-uncased)\n\n")
        
        for split_name, metrics in results.items():
            f.write(f"{split_name.capitalize()} Set:\n")
            f.write(f"  - Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"  - Precision: {metrics['precision']:.4f}\n")
            f.write(f"  - Recall:    {metrics['recall']:.4f}\n")
            f.write(f"  - F1 Score:  {metrics['f1']:.4f}\n")
            f.write(f"  - ROC-AUC:   {metrics['roc_auc']:.4f}\n")
            f.write(f"  - PR-AUC:    {metrics['pr_auc']:.4f}\n\n")
    
    print(f"✓ Saved metrics to '{metrics_path}'")
    
    # Plot confusion matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (split_name, metrics) in enumerate(results.items()):
        cm = metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Ham', 'Spam'],
                    yticklabels=['Ham', 'Spam'],
                    ax=axes[idx])
        axes[idx].set_title(f'{split_name.capitalize()} Set')
        axes[idx].set_ylabel('Actual')
        axes[idx].set_xlabel('Predicted')
    
    plt.tight_layout()
    cm_path = os.path.join(output_dir, 'confusion_matrices.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved confusion matrices to '{cm_path}'")
    plt.close()


def train_distilbert_model(data_dir='data/processed',
                          output_dir='models/distilbert',
                          model_name='distilbert-base-uncased',
                          max_length=128,
                          batch_size=16,
                          num_epochs=3,
                          learning_rate=2e-5):
    """
    Main training pipeline for DistilBERT model.
    
    Args:
        data_dir (str): Directory containing processed data
        output_dir (str): Directory to save model and results
        model_name (str): Pre-trained model name
        max_length (int): Maximum sequence length
        batch_size (int): Training batch size
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate
    """
    print("\n" + "="*70)
    print("SMS SPAM CLASSIFICATION - DISTILBERT MODEL TRAINING")
    print("="*70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Load data
    print("\n[1/6] Loading data...")
    dataset_dict = load_data(data_dir)
    
    # Step 2: Prepare tokenizer
    print("\n[2/6] Preparing tokenizer...")
    tokenizer = prepare_tokenizer(model_name, max_length)
    
    # Step 3: Tokenize datasets
    print("\n[3/6] Tokenizing datasets...")
    tokenized_datasets = tokenize_dataset(dataset_dict, tokenizer, max_length)
    
    # Step 4: Load model
    print("\n[4/6] Loading model...")
    model = load_model(model_name, num_labels=2)
    
    # Step 5: Train model
    print("\n[5/6] Training model...")
    training_args = create_training_args(
        output_dir=output_dir,
        batch_size=batch_size,
        num_epochs=num_epochs,
        learning_rate=learning_rate
    )
    trainer = train_model(model, tokenized_datasets, training_args, tokenizer)
    
    # Step 6: Evaluate model
    print("\n[6/6] Evaluating model...")
    results = evaluate_model(trainer, tokenized_datasets, output_dir)
    
    print("\n" + "="*70)
    print("✅ DISTILBERT MODEL TRAINING COMPLETE!")
    print("="*70)
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📌 Key Results (Test Set):")
    print(f"   • Accuracy:  {results['test']['accuracy']:.4f}")
    print(f"   • F1 Score:  {results['test']['f1']:.4f}")
    print(f"   • ROC-AUC:   {results['test']['roc_auc']:.4f}")
    print(f"   • PR-AUC:    {results['test']['pr_auc']:.4f}")
    
    return trainer, results


if __name__ == "__main__":
    # Train the DistilBERT model
    trainer, results = train_distilbert_model(
        data_dir='../data/processed',
        output_dir='../models/distilbert',
        model_name='distilbert-base-uncased',
        max_length=128,
        batch_size=16,
        num_epochs=3,
        learning_rate=2e-5
    )

