"""
Step 3: Load patient labels from .txt files
Purpose: Extract Outcome (Normal/Abnormal) for each patient
"""

import os
import glob

def get_patient_outcome(patient_id, data_dir="dataset/training_data"):
    """
    Read Outcome label from patient's .txt file
    Returns: 'Normal' or 'Abnormal' or None if not found
    """
    txt_file = os.path.join(data_dir, f"{patient_id}.txt")
    
    if not os.path.exists(txt_file):
        return None
    
    # Read the file and find Outcome line
    with open(txt_file, 'r') as f:
        for line in f:
            if line.startswith("#Outcome:"):
                # Extract value after colon
                outcome = line.split(":")[1].strip()
                return outcome
    
    return None

# Test with a few patients
test_patients = ["2530", "43852", "46532"]

print("Testing label loading:")
for patient_id in test_patients:
    outcome = get_patient_outcome(patient_id)
    print(f"  Patient {patient_id}: {outcome}")

print("\n[OK] Step 3 complete: Label loading function ready")

