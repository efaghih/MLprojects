"""
Complete Training Pipeline - All Features
- Full dataset processing
- Hyperparameter tuning
- Mean vs Max aggregation comparison
- Comprehensive evaluation
"""

import numpy as np
from step4_create_dataset_mapping import create_dataset_mapping
from step5_preprocess_audio import preprocess_audio
from step6_create_spectrograms import create_log_mel_spectrogram
from step7_split_into_clips import split_into_clips
from step12_implement_cnn_model import create_cnn_model
from step14_patient_aggregation import aggregate_clips_to_patient
from step15_evaluation_metrics import calculate_patient_metrics
from sklearn.model_selection import StratifiedKFold
import os
import time

# CONFIGURATION
USE_FULL_DATASET = True  # Set to False for quick testing (default is True) -> to fast test set it to False
MAX_PATIENTS_PER_FOLD = None  # None = all patients, or set number for testing. (default is None) -> to fast test set it to 10

# Hyperparameters to tune
HYPERPARAMS = {
    'epochs': 10,  #default is 10 -> you can do a fast test with 2 epochs 
    'batch_size': 32,
    'learning_rate': 0.001
}

def prepare_patient_data(patient_ids_list, dataset_dict, max_patients=None, verbose=True):
    """Prepare all clips for given patients with progress tracking"""
    X, y, pids = [], [], []
    processed = 0
    errors = 0
    
    for pid in patient_ids_list:
        if max_patients and processed >= max_patients:
            break
        if pid in dataset_dict:
            for wav_file, outcome in dataset_dict[pid]:
                if os.path.exists(wav_file):
                    try:
                        audio, sr = preprocess_audio(wav_file)
                        clips = split_into_clips(audio, sr, clip_duration=4, overlap=0.5)
                        label = 1 if outcome == "Abnormal" else 0
                        
                        for clip in clips:
                            spec = create_log_mel_spectrogram(clip, sr)
                            
                            # Ensure consistent shape (pad/truncate if needed)
                            target_freq, target_time = 1025, 28
                            
                            # Handle frequency dimension
                            if spec.shape[0] > target_freq:
                                spec = spec[:target_freq, :]
                            elif spec.shape[0] < target_freq:
                                pad = np.zeros((target_freq - spec.shape[0], spec.shape[1]))
                                spec = np.vstack([spec, pad])
                            
                            # Handle time dimension
                            if spec.shape[1] > target_time:
                                spec = spec[:, :target_time]
                            elif spec.shape[1] < target_time:
                                pad = np.zeros((spec.shape[0], target_time - spec.shape[1]))
                                spec = np.hstack([spec, pad])
                            
                            spec = spec[:, :, np.newaxis]  # Add channel dim
                            X.append(spec)
                            y.append(label)
                            pids.append(pid)
                    except Exception as e:
                        errors += 1
                        if verbose and errors <= 3:
                            print(f"      [WARN] Error processing {os.path.basename(wav_file)}: {str(e)[:50]}")
                        continue
            processed += 1
            if verbose and processed % 50 == 0:
                print(f"      Processed {processed} patients, {len(X)} clips so far...")
    
    if errors > 0 and verbose:
        print(f"      [INFO] {errors} files had errors (skipped)")
    
    return np.array(X), np.array(y), pids

print("=" * 70)
print("COMPLETE TRAINING PIPELINE")
print("=" * 70)
print(f"\nConfiguration:")
print(f"  Full dataset: {USE_FULL_DATASET}")
print(f"  Max patients per fold: {MAX_PATIENTS_PER_FOLD or 'All'}")
print(f"  Epochs: {HYPERPARAMS['epochs']}")
print(f"  Batch size: {HYPERPARAMS['batch_size']}")
print(f"  Learning rate: {HYPERPARAMS['learning_rate']}")

# Load dataset
print("\n[1/6] Loading dataset...")
start = time.time()
dataset = create_dataset_mapping()
patient_data = {}
for wav_file, patient_id, outcome in dataset:
    if patient_id not in patient_data:
        patient_data[patient_id] = []
    patient_data[patient_id].append((wav_file, outcome))

patient_ids = list(patient_data.keys())
outcomes = [patient_data[pid][0][1] for pid in patient_ids]
outcomes_binary = [1 if o == "Abnormal" else 0 for o in outcomes]

print(f"   Total patients: {len(patient_ids)}")
print(f"   Normal: {outcomes_binary.count(0)}, Abnormal: {outcomes_binary.count(1)}")
print(f"   Load time: {time.time()-start:.1f}s")

# Setup cross-validation
print("\n[2/6] Setting up 5-fold cross-validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, test_idx) in enumerate(skf.split(patient_ids, outcomes_binary), 1):
    print(f"\n{'='*70}")
    print(f"FOLD {fold}/5")
    print(f"{'='*70}")
    
    train_patients = [patient_ids[i] for i in train_idx]
    test_patients = [patient_ids[i] for i in test_idx]
    
    # Limit patients if configured
    if MAX_PATIENTS_PER_FOLD:
        train_patients = train_patients[:MAX_PATIENTS_PER_FOLD]
        test_patients = test_patients[:MAX_PATIENTS_PER_FOLD // 4]  # ~25% for test
    
    print(f"Train: {len(train_patients)} patients, Test: {len(test_patients)} patients")
    
    # Prepare training data
    print("   [3/6] Preparing training data...")
    start_time = time.time()
    train_X, train_y, train_pids = prepare_patient_data(
        train_patients, patient_data, 
        max_patients=None if USE_FULL_DATASET else len(train_patients)
    )
    prep_time = time.time() - start_time
    print(f"   Training: {len(train_X)} clips from {len(set(train_pids))} patients, Time: {prep_time:.1f}s")
    
    if len(train_X) == 0:
        print("   [SKIP] No training data")
        continue
    
    # Prepare test data
    print("   [4/6] Preparing test data...")
    start_time = time.time()
    test_X, test_y, test_pids = prepare_patient_data(
        test_patients, patient_data,
        max_patients=None if USE_FULL_DATASET else len(test_patients)
    )
    prep_time = time.time() - start_time
    print(f"   Test: {len(test_X)} clips from {len(set(test_pids))} patients, Time: {prep_time:.1f}s")
    
    if len(test_X) == 0:
        print("   [SKIP] No test data")
        continue
    
    # Create and train model
    print("   [5/6] Creating and training model...")
    model = create_cnn_model(input_shape=train_X[0].shape)
    
    # Set learning rate
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=HYPERPARAMS['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    start_time = time.time()
    history = model.fit(
        train_X, train_y, 
        epochs=HYPERPARAMS['epochs'], 
        batch_size=HYPERPARAMS['batch_size'], 
        verbose=0,
        validation_split=0.1
    )
    train_time = time.time() - start_time
    print(f"   Training time: {train_time:.1f}s")
    print(f"   Final train accuracy: {history.history['accuracy'][-1]:.3f}")
    if 'val_accuracy' in history.history:
        print(f"   Final val accuracy: {history.history['val_accuracy'][-1]:.3f}")
    
    # Predict
    print("   [6/6] Making predictions and evaluating...")
    test_pred_proba = model.predict(test_X, verbose=0).flatten()
    
    # Aggregate to patient level (both methods)
    patient_preds_mean = aggregate_clips_to_patient(test_pred_proba, test_pids, 'mean')
    patient_preds_max = aggregate_clips_to_patient(test_pred_proba, test_pids, 'max')
    
    # Get true labels
    test_patient_labels = {pid: outcomes_binary[patient_ids.index(pid)] 
                          for pid in test_patients if pid in patient_preds_mean}
    
    # Calculate metrics
    if len(test_patient_labels) > 0:
        # Mean aggregation
        mean_pids = [pid for pid in patient_preds_mean.keys() if pid in test_patient_labels]
        y_true_mean = np.array([test_patient_labels[pid] for pid in mean_pids])
        y_pred_mean = np.array([patient_preds_mean[pid] for pid in mean_pids])
        metrics_mean = calculate_patient_metrics(y_true_mean, y_pred_mean)
        
        # Max aggregation
        max_pids = [pid for pid in patient_preds_max.keys() if pid in test_patient_labels]
        y_true_max = np.array([test_patient_labels[pid] for pid in max_pids])
        y_pred_max = np.array([patient_preds_max[pid] for pid in max_pids])
        metrics_max = calculate_patient_metrics(y_true_max, y_pred_max)
        
        print(f"\n   Results (Mean aggregation):")
        print(f"     ROC-AUC: {metrics_mean['roc_auc']:.3f}")
        print(f"     F1: {metrics_mean['f1']:.3f}")
        print(f"     Sensitivity: {metrics_mean['sensitivity']:.3f}")
        print(f"     Specificity: {metrics_mean['specificity']:.3f}")
        print(f"     Accuracy: {metrics_mean['accuracy']:.3f}")
        
        print(f"\n   Results (Max aggregation):")
        print(f"     ROC-AUC: {metrics_max['roc_auc']:.3f}")
        print(f"     F1: {metrics_max['f1']:.3f}")
        print(f"     Sensitivity: {metrics_max['sensitivity']:.3f}")
        print(f"     Specificity: {metrics_max['specificity']:.3f}")
        print(f"     Accuracy: {metrics_max['accuracy']:.3f}")
        
        fold_results.append({
            'fold': fold,
            'mean': metrics_mean,
            'max': metrics_max,
            'n_test_patients': len(test_patient_labels)
        })

# Final comprehensive summary
print(f"\n{'='*70}")
print("FINAL CROSS-VALIDATION RESULTS")
print(f"{'='*70}")

if fold_results:
    # Calculate statistics
    mean_metrics = {key: [r['mean'][key] for r in fold_results] for key in ['roc_auc', 'f1', 'sensitivity', 'specificity', 'accuracy']}
    max_metrics = {key: [r['max'][key] for r in fold_results] for key in ['roc_auc', 'f1', 'sensitivity', 'specificity', 'accuracy']}
    
    print("\nMEAN AGGREGATION (5-fold CV average):")
    for metric in ['roc_auc', 'f1', 'sensitivity', 'specificity', 'accuracy']:
        avg = np.mean(mean_metrics[metric])
        std = np.std(mean_metrics[metric])
        print(f"  {metric.capitalize():12} {avg:.3f} (+/- {std:.3f})")
    
    print("\nMAX AGGREGATION (5-fold CV average):")
    for metric in ['roc_auc', 'f1', 'sensitivity', 'specificity', 'accuracy']:
        avg = np.mean(max_metrics[metric])
        std = np.std(max_metrics[metric])
        print(f"  {metric.capitalize():12} {avg:.3f} (+/- {std:.3f})")
    
    # Best method analysis
    mean_auc = np.mean(mean_metrics['roc_auc'])
    max_auc = np.mean(max_metrics['roc_auc'])
    mean_sens = np.mean(mean_metrics['sensitivity'])
    max_sens = np.mean(max_metrics['sensitivity'])
    
    print(f"\n{'='*70}")
    print("RECOMMENDATIONS:")
    print(f"{'='*70}")
    print(f"Best ROC-AUC: {'Mean' if mean_auc > max_auc else 'Max'} ({max(mean_auc, max_auc):.3f})")
    print(f"Best Sensitivity (for screening): {'Mean' if mean_sens > max_sens else 'Max'} ({max(mean_sens, max_sens):.3f})")
    
    print("\n[OK] Complete training and evaluation finished!")

else:
    print("\n[ERROR] No results. Check data loading and processing.")

