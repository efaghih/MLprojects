"""
Model Comparison Script

This script evaluates both baseline and DistilBERT models on the test set
and creates a comprehensive comparison report.

Usage:
    python compare_models.py
"""

import pandas as pd
import numpy as np
import joblib
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import os


def load_test_data(data_dir='data/processed'):
    """Load test dataset."""
    print("Loading test data...")
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    print(f"✓ Test samples: {len(test_df)}")
    return test_df


def evaluate_baseline(model_path='models/baseline/baseline_model.pkl', test_df=None):
    """Evaluate baseline TF-IDF + Naive Bayes model on test set."""
    print("\n" + "="*70)
    print("EVALUATING BASELINE MODEL (TF-IDF + NAIVE BAYES)")
    print("="*70)
    
    # Load model
    print(f"\nLoading baseline model from '{model_path}'...")
    baseline_model = joblib.load(model_path)
    print("✓ Model loaded")
    
    # Get predictions
    X_test = test_df['text'].values
    y_test = test_df['label'].values
    
    print("Making predictions...")
    y_pred = baseline_model.predict(X_test)
    y_proba = baseline_model.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary', pos_label=1
    )
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba, pos_label=1)
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results
    print("\n📊 Test Set Metrics:")
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
    
    return {
        'model_name': 'TF-IDF + Naive Bayes',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_proba
    }


def evaluate_distilbert(model_dir='models/distilbert', test_df=None):
    """Evaluate DistilBERT model on test set."""
    print("\n" + "="*70)
    print("EVALUATING DISTILBERT MODEL")
    print("="*70)
    
    # Load tokenizer and model
    # The tokenizer is loaded from the original pre-trained model
    # The fine-tuned model is loaded from the checkpoint
    print(f"\nLoading DistilBERT model from '{model_dir}'...")
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Find the best checkpoint (last one)
    checkpoint_dir = os.path.join(model_dir, 'checkpoint-837')
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    print("✓ Model loaded")
    
    # Prepare dataset
    test_dataset = Dataset.from_pandas(test_df.rename(columns={'label': 'labels'}))
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=128
        )
    
    print("Tokenizing test data...")
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    
    # Create trainer for prediction
    training_args = TrainingArguments(
        output_dir='./tmp',
        per_device_eval_batch_size=16,
        dataloader_num_workers=0
    )
    
    trainer = Trainer(
        model=model,
        args=training_args
    )
    
    print("Making predictions...")
    predictions = trainer.predict(tokenized_test)
    
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
    
    # Print results
    print("\n📊 Test Set Metrics:")
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
    
    return {
        'model_name': 'DistilBERT',
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'predictions': preds,
        'probabilities': probs
    }


def create_comparison_report(baseline_results, distilbert_results, output_dir='reports'):
    """Create comprehensive comparison report."""
    print("\n" + "="*70)
    print("CREATING COMPARISON REPORT")
    print("="*70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison dataframe
    comparison_df = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'PR-AUC'],
        'TF-IDF + NB': [
            baseline_results['accuracy'],
            baseline_results['precision'],
            baseline_results['recall'],
            baseline_results['f1'],
            baseline_results['roc_auc'],
            baseline_results['pr_auc']
        ],
        'DistilBERT': [
            distilbert_results['accuracy'],
            distilbert_results['precision'],
            distilbert_results['recall'],
            distilbert_results['f1'],
            distilbert_results['roc_auc'],
            distilbert_results['pr_auc']
        ]
    })
    
    # Calculate improvements
    comparison_df['Improvement'] = comparison_df['DistilBERT'] - comparison_df['TF-IDF + NB']
    comparison_df['Improvement %'] = (comparison_df['Improvement'] / comparison_df['TF-IDF + NB'] * 100)
    
    # Print comparison table
    print("\n" + "-"*70)
    print("SIDE-BY-SIDE COMPARISON (TEST SET)")
    print("-"*70)
    print("\n" + comparison_df.to_string(index=False, float_format='%.4f'))
    
    # Print key insights
    print("\n" + "-"*70)
    print("KEY INSIGHTS")
    print("-"*70)
    
    f1_improvement = (distilbert_results['f1'] - baseline_results['f1']) * 100
    roc_improvement = (distilbert_results['roc_auc'] - baseline_results['roc_auc']) * 100
    
    print(f"\n✅ DistilBERT Improvements:")
    print(f"   • F1 Score:  +{f1_improvement:.2f} percentage points")
    print(f"   • ROC-AUC:   +{roc_improvement:.2f} percentage points")
    
    # Confusion matrix comparison
    print(f"\n📊 False Positives (Ham marked as Spam):")
    print(f"   • Baseline:   {baseline_results['confusion_matrix'][0,1]}")
    print(f"   • DistilBERT: {distilbert_results['confusion_matrix'][0,1]}")
    
    print(f"\n📊 False Negatives (Spam marked as Ham):")
    print(f"   • Baseline:   {baseline_results['confusion_matrix'][1,0]}")
    print(f"   • DistilBERT: {distilbert_results['confusion_matrix'][1,0]}")
    
    # Save report as text file
    report_path = os.path.join(output_dir, 'model_comparison.txt')
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("MODEL COMPARISON REPORT - SMS SPAM CLASSIFICATION\n")
        f.write("="*70 + "\n\n")
        f.write("Test Set Evaluation (558 samples)\n\n")
        f.write(comparison_df.to_string(index=False, float_format='%.4f'))
        f.write("\n\n")
        f.write("="*70 + "\n")
        f.write("DETAILED METRICS\n")
        f.write("="*70 + "\n\n")
        
        for name, results in [('Baseline (TF-IDF + Naive Bayes)', baseline_results),
                              ('DistilBERT (Transformer)', distilbert_results)]:
            f.write(f"{name}:\n")
            f.write(f"  Accuracy:  {results['accuracy']:.4f}\n")
            f.write(f"  Precision: {results['precision']:.4f}\n")
            f.write(f"  Recall:    {results['recall']:.4f}\n")
            f.write(f"  F1 Score:  {results['f1']:.4f}\n")
            f.write(f"  ROC-AUC:   {results['roc_auc']:.4f}\n")
            f.write(f"  PR-AUC:    {results['pr_auc']:.4f}\n\n")
    
    print(f"\n✓ Saved text report to '{report_path}'")
    
    # Save comparison CSV
    csv_path = os.path.join(output_dir, 'comparison_metrics.csv')
    comparison_df.to_csv(csv_path, index=False)
    print(f"✓ Saved CSV to '{csv_path}'")
    
    # Create visualization
    create_comparison_plots(baseline_results, distilbert_results, comparison_df, output_dir)
    
    return comparison_df


def create_comparison_plots(baseline_results, distilbert_results, comparison_df, output_dir):
    """Create comparison visualizations."""
    
    # 1. Metrics bar chart
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Subplot 1: Metrics comparison
    ax1 = axes[0, 0]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC', 'PR-AUC']
    x = np.arange(len(metrics))
    width = 0.35
    
    baseline_values = comparison_df['TF-IDF + NB'].values
    distilbert_values = comparison_df['DistilBERT'].values
    
    ax1.bar(x - width/2, baseline_values, width, label='TF-IDF + NB', color='#3498db', alpha=0.8)
    ax1.bar(x + width/2, distilbert_values, width, label='DistilBERT', color='#e74c3c', alpha=0.8)
    
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim([0.85, 1.0])
    
    # Subplot 2: Improvement chart
    ax2 = axes[0, 1]
    improvements = comparison_df['Improvement'].values * 100
    colors = ['green' if x > 0 else 'red' for x in improvements]
    
    ax2.barh(metrics, improvements, color=colors, alpha=0.7)
    ax2.set_xlabel('Improvement (percentage points)', fontsize=11)
    ax2.set_title('DistilBERT vs Baseline Improvement', fontsize=12, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax2.grid(axis='x', alpha=0.3)
    
    # Subplot 3: Confusion matrices comparison
    ax3 = axes[1, 0]
    cm_baseline = baseline_results['confusion_matrix']
    sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'],
                ax=ax3, cbar=False)
    ax3.set_title('Baseline Confusion Matrix', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Actual')
    ax3.set_xlabel('Predicted')
    
    # Subplot 4: DistilBERT confusion matrix
    ax4 = axes[1, 1]
    cm_distilbert = distilbert_results['confusion_matrix']
    sns.heatmap(cm_distilbert, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Ham', 'Spam'],
                yticklabels=['Ham', 'Spam'],
                ax=ax4, cbar=False)
    ax4.set_title('DistilBERT Confusion Matrix', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Actual')
    ax4.set_xlabel('Predicted')
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved visualization to '{plot_path}'")
    plt.close()


def main():
    """Main comparison pipeline."""
    print("\n" + "="*70)
    print("SMS SPAM CLASSIFICATION - MODEL COMPARISON")
    print("="*70)
    
    # Step 1: Load test data
    print("\n[1/4] Loading test data...")
    test_df = load_test_data()
    
    # Step 2: Evaluate baseline
    print("\n[2/4] Evaluating baseline model...")
    baseline_results = evaluate_baseline(test_df=test_df)
    
    # Step 3: Evaluate DistilBERT
    print("\n[3/4] Evaluating DistilBERT model...")
    distilbert_results = evaluate_distilbert(test_df=test_df)
    
    # Step 4: Create comparison report
    print("\n[4/4] Creating comparison report...")
    comparison_df = create_comparison_report(baseline_results, distilbert_results)
    
    print("\n" + "="*70)
    print("✅ MODEL COMPARISON COMPLETE!")
    print("="*70)
    print("\n📁 Files created:")
    print("   • reports/model_comparison.txt")
    print("   • reports/comparison_metrics.csv")
    print("   • reports/model_comparison.png")


if __name__ == "__main__":
    main()

