"""
Step 2: Check audio file properties
Purpose: Verify sampling rates from header files (.hea)
"""

import os

# Read header file to get sampling rate info
# Format: record_name num_signals sampling_freq num_samples
sample_hea = "dataset/training_data/2530_AV.hea"

with open(sample_hea, 'r') as f:
    lines = f.readlines()
    # First line: record_name num_signals sampling_freq num_samples
    header = lines[0].strip().split()
    
record_name = header[0]
sampling_rate = int(header[2])  # Third field is sampling rate
num_samples = int(header[3])

duration = num_samples / sampling_rate

print(f"Sample file: {record_name}")
print(f"Sampling rate: {sampling_rate} Hz")
print(f"Number of samples: {num_samples}")
print(f"Duration: {duration:.2f} seconds")

# Verify expected sampling rate
if sampling_rate == 4000:
    print("\n[OK] Sampling rate is 4000 Hz as expected")
else:
    print(f"\n[NOTE] Sampling rate is {sampling_rate} Hz, will standardize to 4000 Hz")

print("\n[OK] Step 2 complete: Audio properties checked")

