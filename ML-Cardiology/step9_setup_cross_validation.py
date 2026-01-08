"""
Step 9: Setup patient-level cross-validation
Purpose: Split patients (not clips) into train/test folds
"""

from step4_create_dataset_mapping import create_dataset_mapping
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Get all unique patients with their outcomes
dataset = create_dataset_mapping()

# Extract unique patients and their outcomes
patient_data = {}
for wav_file, patient_id, outcome in dataset:
    if patient_id not in patient_data:
        patient_data[patient_id] = outcome

# Create lists for splitting
patient_ids = list(patient_data.keys())
outcomes = [patient_data[pid] for pid in patient_ids]

print(f"Total unique patients: {len(patient_ids)}")
print(f"Normal: {outcomes.count('Normal')}, Abnormal: {outcomes.count('Abnormal')}")

# Setup 5-fold cross-validation (patient-level)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Show first fold split
train_idx, test_idx = next(skf.split(patient_ids, outcomes))
train_patients = [patient_ids[i] for i in train_idx]
test_patients = [patient_ids[i] for i in test_idx]

print(f"\nFold 1 split:")
print(f"  Train patients: {len(train_patients)}")
print(f"  Test patients: {len(test_patients)}")
print(f"  Sample train: {train_patients[:3]}")
print(f"  Sample test: {test_patients[:3]}")

print("\n[OK] Step 9 complete: Cross-validation setup ready")

