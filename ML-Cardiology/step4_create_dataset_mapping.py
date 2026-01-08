"""
Step 4: Create dataset mapping (WAV files to labels)
Purpose: Build a list of all WAV files with their corresponding Outcome labels
"""

import os
import glob
from step3_load_labels import get_patient_outcome

def create_dataset_mapping(data_dir="dataset/training_data"):
    """
    Create mapping: list of (wav_file_path, patient_id, outcome)
    """
    dataset = []
    wav_files = glob.glob(os.path.join(data_dir, "*.wav"))
    
    for wav_file in wav_files:
        # Extract patient ID from filename (e.g., "2530_AV.wav" -> "2530")
        filename = os.path.basename(wav_file)
        patient_id = filename.split("_")[0]
        
        # Get outcome label for this patient
        outcome = get_patient_outcome(patient_id, data_dir)
        
        if outcome:  # Only include if we found a label
            dataset.append((wav_file, patient_id, outcome))
    
    return dataset

# Create the mapping
dataset = create_dataset_mapping()

print(f"Total WAV files with labels: {len(dataset)}")
print(f"\nSample entries:")
for wav, pid, outcome in dataset[:5]:
    print(f"  {os.path.basename(wav)} -> Patient {pid}: {outcome}")

print("\n[OK] Step 4 complete: Dataset mapping created")

