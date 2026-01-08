"""
Step 14: Patient-level aggregation
Purpose: Aggregate clip predictions to patient-level predictions
"""

import numpy as np

def aggregate_clips_to_patient(clip_predictions, patient_ids, method='mean'):
    """
    Aggregate clip-level predictions to patient-level predictions
    
    Args:
        clip_predictions: array of probabilities for each clip
        patient_ids: list of patient IDs for each clip
        method: 'mean' (default) or 'max'
    
    Returns:
        dict: {patient_id: aggregated_probability}
    """
    patient_probs = {}
    
    # Group predictions by patient
    for pred, pid in zip(clip_predictions, patient_ids):
        if pid not in patient_probs:
            patient_probs[pid] = []
        patient_probs[pid].append(pred)
    
    # Aggregate using specified method
    aggregated = {}
    for pid, probs in patient_probs.items():
        if method == 'mean':
            aggregated[pid] = np.mean(probs)
        elif method == 'max':
            aggregated[pid] = np.max(probs)
        else:
            aggregated[pid] = np.mean(probs)  # Default to mean
    
    return aggregated

# Test with sample data
print("Testing patient-level aggregation...")

# Simulate: 3 clips from patient A, 2 clips from patient B
clip_preds = np.array([0.7, 0.8, 0.6, 0.3, 0.4])  # Clip probabilities
patient_ids = ['A', 'A', 'A', 'B', 'B']  # Which patient each clip belongs to

# Aggregate
patient_preds = aggregate_clips_to_patient(clip_preds, patient_ids, method='mean')

print(f"\nClip predictions: {clip_preds}")
print(f"Patient IDs: {patient_ids}")
print(f"\nAggregated patient predictions (mean):")
for pid, prob in patient_preds.items():
    print(f"  Patient {pid}: {prob:.3f}")

print("\n[OK] Step 14 complete: Patient aggregation ready")

