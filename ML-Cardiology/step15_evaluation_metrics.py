"""
Step 15: Evaluation metrics
Purpose: Calculate patient-level metrics (ROC-AUC, F1, Sensitivity, Specificity)
"""

from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
import numpy as np

def calculate_patient_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Calculate patient-level classification metrics
    
    Args:
        y_true: true labels (0=Normal, 1=Abnormal)
        y_pred_proba: predicted probabilities
        threshold: decision threshold (default 0.5)
    
    Returns:
        dict with metrics
    """
    # Convert probabilities to binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred)
    
    # Confusion matrix: [TN, FP], [FN, TP]
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for Abnormal
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall for Normal
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'roc_auc': roc_auc,
        'f1': f1,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'confusion_matrix': cm
    }

# Test with sample predictions
print("Testing evaluation metrics...")

# Simulate: 10 patients (5 Normal=0, 5 Abnormal=1)
y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
y_pred_proba = np.array([0.2, 0.3, 0.4, 0.1, 0.3, 0.7, 0.8, 0.6, 0.9, 0.7])

metrics = calculate_patient_metrics(y_true, y_pred_proba, threshold=0.5)

print(f"\nTrue labels: {y_true}")
print(f"Predicted probabilities: {y_pred_proba}")
print(f"\nMetrics:")
print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
print(f"  F1 Score: {metrics['f1']:.3f}")
print(f"  Sensitivity (Abnormal): {metrics['sensitivity']:.3f}")
print(f"  Specificity (Normal): {metrics['specificity']:.3f}")
print(f"  Accuracy: {metrics['accuracy']:.3f}")

print("\n[OK] Step 15 complete: Evaluation metrics ready")

