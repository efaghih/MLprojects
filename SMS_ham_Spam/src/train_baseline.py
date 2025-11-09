"""
Baseline Model Training Module for SMS Spam Classification

This module implements a baseline classifier using:
- TF-IDF (Term Frequency-Inverse Document Frequency) for text vectorization
- Multinomial Naive Bayes for classification

The baseline establishes a performance benchmark for more complex models.
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns


def load_data(data_dir='data/processed'):
    """
    Load train and validation datasets.
    
    Args:
        data_dir (str): Directory containing the processed CSV files
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val)
    """
    print(f"Loading data from '{data_dir}/'...")
    
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    val_df = pd.read_csv(os.path.join(data_dir, 'val.csv'))
    
    # Split features (text) and labels
    X_train = train_df['text'].values
    y_train = train_df['label'].values
    
    X_val = val_df['text'].values
    y_val = val_df['label'].values
    
    print(f"✓ Training samples: {len(X_train)}")
    print(f"✓ Validation samples: {len(X_val)}")
    
    return X_train, y_train, X_val, y_val


def create_baseline_pipeline(ngram_range=(1, 2), min_df=2, alpha=0.5):
    """
    Create a baseline classification pipeline.
    
    Pipeline components:
    1. TfidfVectorizer: Converts text to TF-IDF features
       - ngram_range: Considers both unigrams and bigrams (1-2 word sequences)
       - min_df: Ignores terms appearing in fewer than 2 documents (reduces noise)
    
    2. MultinomialNB: Multinomial Naive Bayes classifier
       - alpha: Laplace smoothing parameter (prevents zero probabilities)
    
    Args:
        ngram_range (tuple): Range of n-grams to extract (default: (1,2))
        min_df (int): Minimum document frequency for terms (default: 2)
        alpha (float): Smoothing parameter for Naive Bayes (default: 0.5)
        
    Returns:
        Pipeline: Sklearn pipeline object
    """
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            ngram_range=ngram_range,  # Use 1-grams and 2-grams
            min_df=min_df,             # Ignore rare terms
            lowercase=True,            # Convert to lowercase
            strip_accents='unicode',   # Remove accents
            stop_words='english'       # Remove common English words
        )),
        ('classifier', MultinomialNB(
            alpha=alpha  # Laplace smoothing
        ))
    ])
    
    print("\n✓ Created baseline pipeline:")
    print(f"  - TF-IDF: ngram_range={ngram_range}, min_df={min_df}")
    print(f"  - Multinomial NB: alpha={alpha}")
    
    return pipeline


def train_model(pipeline, X_train, y_train):
    """
    Train the baseline model.
    
    Args:
        pipeline: Sklearn pipeline
        X_train: Training text data
        y_train: Training labels
        
    Returns:
        Pipeline: Trained pipeline
    """
    print("\n" + "="*70)
    print("TRAINING BASELINE MODEL")
    print("="*70)
    
    print("\nFitting pipeline on training data...")
    pipeline.fit(X_train, y_train)
    
    print("✓ Training complete!")
    
    # Get vocabulary size
    vocab_size = len(pipeline.named_steps['tfidf'].vocabulary_)
    print(f"  - Vocabulary size: {vocab_size:,} unique terms")
    
    return pipeline


def evaluate_model(pipeline, X_train, y_train, X_val, y_val):
    """
    Evaluate the trained model on both training and validation sets.
    
    Metrics computed:
    - F1 Score: Harmonic mean of precision and recall (spam as positive)
    - ROC-AUC: Area under ROC curve (measures discrimination ability)
    - PR-AUC: Area under Precision-Recall curve (better for imbalanced data)
    - Confusion Matrix: True/False Positives/Negatives
    
    Args:
        pipeline: Trained pipeline
        X_train: Training text data
        y_train: Training labels
        X_val: Validation text data
        y_val: Validation labels
        
    Returns:
        dict: Dictionary containing all evaluation metrics
    """
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    results = {}
    
    # Get predictions and probabilities
    print("\nMaking predictions...")
    
    # Training set predictions
    y_train_pred = pipeline.predict(X_train)
    y_train_proba = pipeline.predict_proba(X_train)[:, 1]  # Probability of spam
    
    # Validation set predictions
    y_val_pred = pipeline.predict(X_val)
    y_val_proba = pipeline.predict_proba(X_val)[:, 1]  # Probability of spam
    
    # ==================== TRAINING SET METRICS ====================
    print("\n" + "-"*70)
    print("TRAINING SET PERFORMANCE")
    print("-"*70)
    
    train_f1 = f1_score(y_train, y_train_pred, pos_label=1)
    train_roc_auc = roc_auc_score(y_train, y_train_proba)
    train_pr_auc = average_precision_score(y_train, y_train_proba, pos_label=1)
    train_cm = confusion_matrix(y_train, y_train_pred)
    
    print(f"\n📊 Metrics:")
    print(f"  • F1 Score (Spam):     {train_f1:.4f}")
    print(f"  • ROC-AUC:             {train_roc_auc:.4f}")
    print(f"  • PR-AUC:              {train_pr_auc:.4f}")
    
    print(f"\n📈 Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                Ham    Spam")
    print(f"  Actual  Ham   {train_cm[0,0]:5d}  {train_cm[0,1]:5d}")
    print(f"          Spam  {train_cm[1,0]:5d}  {train_cm[1,1]:5d}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_train, y_train_pred, 
                                target_names=['Ham', 'Spam'],
                                digits=4))
    
    results['train'] = {
        'f1_score': train_f1,
        'roc_auc': train_roc_auc,
        'pr_auc': train_pr_auc,
        'confusion_matrix': train_cm,
        'predictions': y_train_pred,
        'probabilities': y_train_proba
    }
    
    # ==================== VALIDATION SET METRICS ====================
    print("\n" + "-"*70)
    print("VALIDATION SET PERFORMANCE")
    print("-"*70)
    
    val_f1 = f1_score(y_val, y_val_pred, pos_label=1)
    val_roc_auc = roc_auc_score(y_val, y_val_proba)
    val_pr_auc = average_precision_score(y_val, y_val_proba, pos_label=1)
    val_cm = confusion_matrix(y_val, y_val_pred)
    
    print(f"\n📊 Metrics:")
    print(f"  • F1 Score (Spam):     {val_f1:.4f}")
    print(f"  • ROC-AUC:             {val_roc_auc:.4f}")
    print(f"  • PR-AUC:              {val_pr_auc:.4f}")
    
    print(f"\n📈 Confusion Matrix:")
    print(f"                  Predicted")
    print(f"                Ham    Spam")
    print(f"  Actual  Ham   {val_cm[0,0]:5d}  {val_cm[0,1]:5d}")
    print(f"          Spam  {val_cm[1,0]:5d}  {val_cm[1,1]:5d}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_val, y_val_pred, 
                                target_names=['Ham', 'Spam'],
                                digits=4))
    
    results['val'] = {
        'f1_score': val_f1,
        'roc_auc': val_roc_auc,
        'pr_auc': val_pr_auc,
        'confusion_matrix': val_cm,
        'predictions': y_val_pred,
        'probabilities': y_val_proba
    }
    
    # ==================== OVERFITTING CHECK ====================
    print("\n" + "-"*70)
    print("OVERFITTING ANALYSIS")
    print("-"*70)
    
    f1_diff = train_f1 - val_f1
    roc_diff = train_roc_auc - val_roc_auc
    
    print(f"\nPerformance Gap (Train - Val):")
    print(f"  • F1 Score difference:  {f1_diff:+.4f}")
    print(f"  • ROC-AUC difference:   {roc_diff:+.4f}")
    
    if f1_diff < 0.05 and roc_diff < 0.05:
        print(f"\n✅ Model generalizes well (minimal overfitting)")
    elif f1_diff < 0.10 and roc_diff < 0.10:
        print(f"\n⚠️  Slight overfitting detected")
    else:
        print(f"\n❌ Significant overfitting detected")
    
    return results


def plot_confusion_matrices(results, output_dir='models/baseline'):
    """
    Plot confusion matrices for train and validation sets.
    
    Args:
        results (dict): Dictionary containing evaluation results
        output_dir (str): Directory to save plots
    """
    print("\n" + "-"*70)
    print("GENERATING VISUALIZATIONS")
    print("-"*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Training confusion matrix
    sns.heatmap(results['train']['confusion_matrix'], 
                annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'],
                ax=axes[0])
    axes[0].set_title('Training Set - Confusion Matrix')
    axes[0].set_ylabel('Actual')
    axes[0].set_xlabel('Predicted')
    
    # Validation confusion matrix
    sns.heatmap(results['val']['confusion_matrix'], 
                annot=True, fmt='d', cmap='Greens',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'],
                ax=axes[1])
    axes[1].set_title('Validation Set - Confusion Matrix')
    axes[1].set_ylabel('Actual')
    axes[1].set_xlabel('Predicted')
    
    plt.tight_layout()
    
    cm_path = os.path.join(output_dir, 'confusion_matrices.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved confusion matrices to '{cm_path}'")
    plt.close()


def save_model(pipeline, results, output_dir='models/baseline'):
    """
    Save the trained model and evaluation results.
    
    Args:
        pipeline: Trained pipeline
        results (dict): Evaluation results
        output_dir (str): Directory to save model
    """
    print("\n" + "-"*70)
    print("SAVING MODEL")
    print("-"*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the pipeline
    model_path = os.path.join(output_dir, 'baseline_model.pkl')
    joblib.dump(pipeline, model_path)
    print(f"✓ Saved model to '{model_path}'")
    
    # Save metrics summary
    metrics_summary = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': 'TF-IDF + Multinomial Naive Bayes',
        'train_metrics': {
            'f1_score': float(results['train']['f1_score']),
            'roc_auc': float(results['train']['roc_auc']),
            'pr_auc': float(results['train']['pr_auc'])
        },
        'val_metrics': {
            'f1_score': float(results['val']['f1_score']),
            'roc_auc': float(results['val']['roc_auc']),
            'pr_auc': float(results['val']['pr_auc'])
        }
    }
    
    metrics_path = os.path.join(output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("BASELINE MODEL - PERFORMANCE METRICS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Timestamp: {metrics_summary['timestamp']}\n")
        f.write(f"Model: {metrics_summary['model_type']}\n\n")
        
        f.write("Training Set:\n")
        f.write(f"  - F1 Score: {metrics_summary['train_metrics']['f1_score']:.4f}\n")
        f.write(f"  - ROC-AUC:  {metrics_summary['train_metrics']['roc_auc']:.4f}\n")
        f.write(f"  - PR-AUC:   {metrics_summary['train_metrics']['pr_auc']:.4f}\n\n")
        
        f.write("Validation Set:\n")
        f.write(f"  - F1 Score: {metrics_summary['val_metrics']['f1_score']:.4f}\n")
        f.write(f"  - ROC-AUC:  {metrics_summary['val_metrics']['roc_auc']:.4f}\n")
        f.write(f"  - PR-AUC:   {metrics_summary['val_metrics']['pr_auc']:.4f}\n")
    
    print(f"✓ Saved metrics summary to '{metrics_path}'")


def train_baseline_model(data_dir='data/processed', output_dir='models/baseline'):
    """
    Main training pipeline for baseline model.
    
    Args:
        data_dir (str): Directory containing processed data
        output_dir (str): Directory to save model and results
    """
    print("\n" + "="*70)
    print("SMS SPAM CLASSIFICATION - BASELINE MODEL TRAINING")
    print("="*70)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Step 1: Load data
    print("\n[1/5] Loading data...")
    X_train, y_train, X_val, y_val = load_data(data_dir)
    
    # Step 2: Create pipeline
    print("\n[2/5] Creating pipeline...")
    pipeline = create_baseline_pipeline(
        ngram_range=(1, 2),
        min_df=2,
        alpha=0.5
    )
    
    # Step 3: Train model
    print("\n[3/5] Training model...")
    pipeline = train_model(pipeline, X_train, y_train)
    
    # Step 4: Evaluate model
    print("\n[4/5] Evaluating model...")
    results = evaluate_model(pipeline, X_train, y_train, X_val, y_val)
    
    # Step 5: Save model and results
    print("\n[5/5] Saving model and results...")
    plot_confusion_matrices(results, output_dir)
    save_model(pipeline, results, output_dir)
    
    print("\n" + "="*70)
    print("✅ BASELINE MODEL TRAINING COMPLETE!")
    print("="*70)
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📌 Key Results (Validation Set):")
    print(f"   • F1 Score: {results['val']['f1_score']:.4f}")
    print(f"   • ROC-AUC:  {results['val']['roc_auc']:.4f}")
    print(f"   • PR-AUC:   {results['val']['pr_auc']:.4f}")
    
    return pipeline, results


if __name__ == "__main__":
    # Train the baseline model
    pipeline, results = train_baseline_model(
        data_dir='../data/processed',
        output_dir='../models/baseline'
    )

