"""
Complete Training Pipeline - Full Implementation
Purpose: Train CNN with cross-validation, hyperparameter tuning, and evaluation
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

def prepare_patient_data(patient_ids_list, dataset_dict, max_patients=None):
    """Prepare all clips for given patients"""
    X, y, pids = [], [], []
    processed = 0
    
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
                            # Pad/truncate to consistent size
                            target_freq, target_time = 1025, 28
                            if spec.shape[0] != target_freq:
                                # Truncate or pad frequency dimension
                                if spec.shape[0] > target_freq:
                                    spec = spec[:target_freq, :]
                                else:
                                    pad = np.zeros((target_freq - spec.shape[0], spec.shape[1]))
                                    spec = np.vstack([spec, pad])
                            
                            if spec.shape[1] != target_time:
                                # Truncate or pad time dimension
                                if spec.shape[1] > target_time:
                                    spec = spec[:, :target_time]
                                else:
                                    pad = np.zeros((spec.shape[0], target_time - spec.shape[1]))
                                    spec = np.hstack([spec, pad])
                            
                            spec = spec[:, :, np.newaxis]  # Add channel dim
                            X.append(spec)
                            y.append(label)
                            pids.append(pid)
                    except Exception as e:
                        print(f"      [WARN] Error processing {wav_file}: {e}")
                        continue
            processed += 1
    
    return np.array(X), np.array(y), pids

print("=" * 70)
print("COMPLETE TRAINING PIPELINE - FULL DATASET")
print("=" * 70)

# Load dataset
print("\n[1/6] Loading dataset...")
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

# Setup cross-validation
print("\n[2/6] Setting up 5-fold cross-validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

# Hyperparameters to test
hyperparams = {
    'epochs': 10,  # Can tune: 5, 10, 20
    'batch_size': 32,  # Can tune: 16, 32, 64
    'learning_rate': 0.001  # Can tune: 0.0001, 0.001, 0.01
}

print(f"   Hyperparameters:")
print(f"     Epochs: {hyperparams['epochs']}")
print(f"     Batch size: {hyperparams['batch_size']}")
print(f"     Learning rate: {hyperparams['learning_rate']}")

for fold, (train_idx, test_idx) in enumerate(skf.split(patient_ids, outcomes_binary), 1):
    print(f"\n{'='*70}")
    print(f"FOLD {fold}/5")
    print(f"{'='*70}")
    
    train_patients = [patient_ids[i] for i in train_idx]
    test_patients = [patient_ids[i] for i in test_idx]
    
    print(f"Train: {len(train_patients)} patients, Test: {len(test_patients)} patients")
    
    # Prepare training data
    print("   [3/6] Preparing training data...")
    start_time = time.time()
    train_X, train_y, train_pids = prepare_patient_data(train_patients, patient_data)
    print(f"   Training clips: {len(train_X)}, Time: {time.time()-start_time:.1f}s")
    
    if len(train_X) == 0:
        print("   [SKIP] No training data")
        continue
    
    # Prepare test data
    print("   [4/6] Preparing test data...")
    start_time = time.time()
    test_X, test_y, test_pids = prepare_patient_data(test_patients, patient_data)
    print(f"   Test clips: {len(test_X)}, Time: {time.time()-start_time:.1f}s")
    
    if len(test_X) == 0:
        print("   [SKIP] No test data")
        continue
    
    # Create and train model
    print("   [5/6] Creating and training model...")
    model = create_cnn_model(input_shape=train_X[0].shape)
    
    # Update learning rate if needed
    from tensorflow.keras.optimizers import Adam
    model.compile(
        optimizer=Adam(learning_rate=hyperparams['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    start_time = time.time()
    history = model.fit(
        train_X, train_y, 
        epochs=hyperparams['epochs'], 
        batch_size=hyperparams['batch_size'], 
        verbose=0,
        validation_split=0.1  # Use 10% for validation during training
    )
    print(f"   Training time: {time.time()-start_time:.1f}s")
    print(f"   Final train accuracy: {history.history['accuracy'][-1]:.3f}")
    
    # Predict
    print("   [6/6] Making predictions and evaluating...")
    test_pred_proba = model.predict(test_X, verbose=0).flatten()
    
    # Aggregate to patient level (both methods)
    patient_preds_mean = aggregate_clips_to_patient(test_pred_proba, test_pids, 'mean')
    patient_preds_max = aggregate_clips_to_patient(test_pred_proba, test_pids, 'max')
    
    # Get true labels for test patients
    test_patient_labels = {pid: outcomes_binary[patient_ids.index(pid)] 
                          for pid in test_patients if pid in patient_preds_mean}
    
    # Calculate metrics for both aggregation methods
    if len(test_patient_labels) > 0:
        # Mean aggregation
        mean_pids = [pid for pid in patient_preds_mean.keys() if pid in test_patient_labels]
        y_true_mean = [test_patient_labels[pid] for pid in mean_pids]
        y_pred_mean = [patient_preds_mean[pid] for pid in mean_pids]
        metrics_mean = calculate_patient_metrics(y_true_mean, y_pred_mean)
        
        # Max aggregation
        max_pids = [pid for pid in patient_preds_max.keys() if pid in test_patient_labels]
        y_true_max = [test_patient_labels[pid] for pid in max_pids]
        y_pred_max = [patient_preds_max[pid] for pid in max_pids]
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
    # Mean aggregation results
    mean_metrics = {
        'roc_auc': [r['mean']['roc_auc'] for r in fold_results],
        'f1': [r['mean']['f1'] for r in fold_results],
        'sensitivity': [r['mean']['sensitivity'] for r in fold_results],
        'specificity': [r['mean']['specificity'] for r in fold_results],
        'accuracy': [r['mean']['accuracy'] for r in fold_results]
    }
    
    # Max aggregation results
    max_metrics = {
        'roc_auc': [r['max']['roc_auc'] for r in fold_results],
        'f1': [r['max']['f1'] for r in fold_results],
        'sensitivity': [r['max']['sensitivity'] for r in fold_results],
        'specificity': [r['max']['specificity'] for r in fold_results],
        'accuracy': [r['max']['accuracy'] for r in fold_results]
    }
    
    print("\nMEAN AGGREGATION (Average across 5 folds):")
    print(f"  ROC-AUC:     {np.mean(mean_metrics['roc_auc']):.3f} (+/- {np.std(mean_metrics['roc_auc']):.3f})")
    print(f"  F1 Score:    {np.mean(mean_metrics['f1']):.3f} (+/- {np.std(mean_metrics['f1']):.3f})")
    print(f"  Sensitivity: {np.mean(mean_metrics['sensitivity']):.3f} (+/- {np.std(mean_metrics['sensitivity']):.3f})")
    print(f"  Specificity: {np.mean(mean_metrics['specificity']):.3f} (+/- {np.std(mean_metrics['specificity']):.3f})")
    print(f"  Accuracy:    {np.mean(mean_metrics['accuracy']):.3f} (+/- {np.std(mean_metrics['accuracy']):.3f})")
    
    print("\nMAX AGGREGATION (Average across 5 folds):")
    print(f"  ROC-AUC:     {np.mean(max_metrics['roc_auc']):.3f} (+/- {np.std(max_metrics['roc_auc']):.3f})")
    print(f"  F1 Score:    {np.mean(max_metrics['f1']):.3f} (+/- {np.std(max_metrics['f1']):.3f})")
    print(f"  Sensitivity: {np.mean(max_metrics['sensitivity']):.3f} (+/- {np.std(max_metrics['sensitivity']):.3f})")
    print(f"  Specificity: {np.mean(max_metrics['specificity']):.3f} (+/- {np.std(max_metrics['specificity']):.3f})")
    print(f"  Accuracy:    {np.mean(max_metrics['accuracy']):.3f} (+/- {np.std(max_metrics['accuracy']):.3f})")
    
    # Determine best method
    mean_auc = np.mean(mean_metrics['roc_auc'])
    max_auc = np.mean(max_metrics['roc_auc'])
    
    print(f"\n{'='*70}")
    print("COMPARISON:")
    print(f"{'='*70}")
    print(f"Best ROC-AUC: {'Mean' if mean_auc > max_auc else 'Max'} aggregation ({max(mean_auc, max_auc):.3f})")
    print(f"Best for Screening (Sensitivity): ", end="")
    mean_sens = np.mean(mean_metrics['sensitivity'])
    max_sens = np.mean(max_metrics['sensitivity'])
    print(f"{'Mean' if mean_sens > max_sens else 'Max'} aggregation ({max(mean_sens, max_sens):.3f})")
    
    print("\n[OK] Training and evaluation complete!")

else:
    print("\n[ERROR] No results to report. Check data loading.")

