"""
Step 1: Explore the dataset structure
Purpose: Understand what audio files we have and their properties
"""

import os
import glob
import pandas as pd

# Define paths
data_dir = "dataset/training_data"
csv_file = "dataset/training_data.csv"

# Load patient metadata
print("Loading patient metadata...")
df = pd.read_csv(csv_file)
print(f"Total patients: {len(df)}")
print(f"Outcome distribution:\n{df['Outcome'].value_counts()}")

# Count WAV files
wav_files = glob.glob(os.path.join(data_dir, "*.wav"))
print(f"\nTotal WAV files: {len(wav_files)}")

# Sample a few files to understand naming
print("\nSample WAV files:")
for wav in wav_files[:5]:
    print(f"  {os.path.basename(wav)}")

print("\n[OK] Step 1 complete: Dataset structure understood")

