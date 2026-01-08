"""
Quick test to verify training pipeline works
Tests on small sample before running full dataset
"""

# Test with small sample first
import sys
sys.path.insert(0, '.')

# Modify train_complete to use small sample
USE_FULL_DATASET = False
MAX_PATIENTS_PER_FOLD = 5  # Very small for quick test

print("=" * 70)
print("QUICK TEST - Verifying training pipeline")
print("=" * 70)
print("This will test on 5 patients per fold to verify everything works.")
print("After this succeeds, run train_complete.py for full training.\n")

# Import and modify the training function
from train_complete import *

# Override config for testing
USE_FULL_DATASET = False
MAX_PATIENTS_PER_FOLD = 5
HYPERPARAMS['epochs'] = 2  # Fewer epochs for quick test

print("Running quick test...")
print("(This will take a few minutes)")

