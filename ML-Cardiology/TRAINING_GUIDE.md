# Complete Training Guide

## Overview
You now have a complete training pipeline that implements all requirements:
- ✅ TensorFlow installation
- ✅ Full dataset processing
- ✅ Cross-validation
- ✅ Hyperparameter configuration
- ✅ Mean vs Max aggregation comparison
- ✅ Comprehensive evaluation metrics

## Files Created

1. **train_complete.py** - Main training script (RECOMMENDED)
   - Processes full dataset
   - 5-fold cross-validation
   - Compares mean vs max aggregation
   - Reports all metrics

2. **train_model.py** - Original training script
   - Uses sample data (faster for testing)

3. **train_model_full.py** - Alternative full implementation

## How to Run

### Option 1: Quick Test (Recommended First)
Test on small sample to verify everything works:
```bash
# Edit train_complete.py, set at top:
USE_FULL_DATASET = False
MAX_PATIENTS_PER_FOLD = 10  # Small number for testing
HYPERPARAMS['epochs'] = 2   # Fewer epochs

# Then run:
myenv\Scripts\python.exe train_complete.py
```

### Option 2: Full Training (All Data)
Run on complete dataset:
```bash
# Edit train_complete.py, set at top:
USE_FULL_DATASET = True
MAX_PATIENTS_PER_FOLD = None  # Use all patients

# Then run:
myenv\Scripts\python.exe train_complete.py
```

**Note:** Full training will take several hours depending on your hardware.

## Configuration Options

In `train_complete.py`, you can adjust:

```python
# Dataset size
USE_FULL_DATASET = True  # True = all patients, False = limited
MAX_PATIENTS_PER_FOLD = None  # None = all, or set number

# Hyperparameters
HYPERPARAMS = {
    'epochs': 10,           # Training epochs (try: 5, 10, 20)
    'batch_size': 32,      # Batch size (try: 16, 32, 64)
    'learning_rate': 0.001 # Learning rate (try: 0.0001, 0.001, 0.01)
}
```

## What the Script Does

1. **Loads Dataset** - All 942 patients, 3,163 WAV files
2. **5-Fold Cross-Validation** - Patient-level splits
3. **For Each Fold:**
   - Prepares training clips from train patients
   - Prepares test clips from test patients
   - Trains CNN model
   - Makes predictions on test clips
   - Aggregates to patient level (mean and max)
   - Calculates metrics
4. **Final Report:**
   - Average metrics across all folds
   - Comparison of mean vs max aggregation
   - Recommendations for best method

## Expected Output

```
FINAL CROSS-VALIDATION RESULTS
======================================================================

MEAN AGGREGATION (5-fold CV average):
  Roc_auc       0.XXX (+/- 0.XXX)
  F1            0.XXX (+/- 0.XXX)
  Sensitivity   0.XXX (+/- 0.XXX)
  Specificity   0.XXX (+/- 0.XXX)
  Accuracy      0.XXX (+/- 0.XXX)

MAX AGGREGATION (5-fold CV average):
  ...

RECOMMENDATIONS:
  Best ROC-AUC: Mean/Max (0.XXX)
  Best Sensitivity: Mean/Max (0.XXX)
```

## Hyperparameter Tuning

To tune hyperparameters, modify `HYPERPARAMS` and compare results:

1. **Learning Rate:**
   - Lower (0.0001): Slower but more stable
   - Higher (0.01): Faster but may overshoot

2. **Epochs:**
   - More epochs: Better learning but longer training
   - Watch for overfitting (val accuracy < train accuracy)

3. **Batch Size:**
   - Smaller (16): More updates, slower
   - Larger (64): Faster, less memory efficient

## Troubleshooting

**Out of Memory:**
- Reduce `MAX_PATIENTS_PER_FOLD`
- Reduce `batch_size`
- Process fewer patients at once

**Training Too Slow:**
- Reduce `epochs` for testing
- Use `MAX_PATIENTS_PER_FOLD` to limit data
- Reduce `batch_size` if memory allows

**Poor Results:**
- Try different hyperparameters
- Increase `epochs`
- Check data quality (some files may have errors)

## Next Steps After Training

1. Analyze which aggregation method works best
2. Tune hyperparameters based on results
3. Train final model on all data
4. Save model for inference
5. Create inference script for new patients

## Time Estimates

- Quick test (10 patients): ~5-10 minutes
- Medium run (100 patients): ~30-60 minutes  
- Full dataset (942 patients): ~3-6 hours

*Times vary based on hardware*

