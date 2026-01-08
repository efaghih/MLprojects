"""
Complete Training Pipeline
Purpose: Train CNN model with cross-validation and evaluate
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

def prepare_patient_data(patient_ids_list, dataset_dict):
    """Prepare all clips for given patients"""
    X, y, pids = [], [], []
    for pid in patient_ids_list:
        if pid in dataset_dict:
            for wav_file, outcome in dataset_dict[pid]:
                if os.path.exists(wav_file):
                    audio, sr = preprocess_audio(wav_file)
                    clips = split_into_clips(audio, sr, clip_duration=4, overlap=0.5)
                    label = 1 if outcome == "Abnormal" else 0
                    for clip in clips:
                        spec = create_log_mel_spectrogram(clip, sr)
                        spec = spec[:, :, np.newaxis]  # Add channel dim
                        X.append(spec)
                        y.append(label)
                        pids.append(pid)
    return np.array(X), np.array(y), pids

print("=" * 70)
print("TRAINING CNN MODEL WITH CROSS-VALIDATION")
print("=" * 70)

# Load dataset
print("\n1. Loading dataset...")
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
print("\n2. Setting up 5-fold cross-validation...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, test_idx) in enumerate(skf.split(patient_ids, outcomes_binary), 1):
    print(f"\n{'='*70}")
    print(f"FOLD {fold}/5")
    print(f"{'='*70}")
    
    train_patients = [patient_ids[i] for i in train_idx]
    test_patients = [patient_ids[i] for i in test_idx]
    
    print(f"Train: {len(train_patients)} patients, Test: {len(test_patients)} patients")
    
    # Prepare data (sample first 10 patients per fold for speed)
    print("   Preparing training data (sample: 10 patients)...")
    train_X, train_y, train_pids = prepare_patient_data(train_patients[:10], patient_data)
    print(f"   Training clips: {len(train_X)}")
    
    print("   Preparing test data (sample: 5 patients)...")
    test_X, test_y, test_pids = prepare_patient_data(test_patients[:5], patient_data)
    print(f"   Test clips: {len(test_X)}")
    
    if len(train_X) == 0 or len(test_X) == 0:
        print("   [SKIP] No data for this fold")
        continue
    
    # Create and train model
    print("   Creating model...")
    model = create_cnn_model(input_shape=train_X[0].shape)
    
    print("   Training model (5 epochs)...")
    model.fit(train_X, train_y, epochs=5, batch_size=32, verbose=0)
    
    # Predict
    print("   Making predictions...")
    test_pred_proba = model.predict(test_X, verbose=0).flatten()
    
    # Aggregate to patient level
    print("   Aggregating to patient level...")
    patient_preds_mean = aggregate_clips_to_patient(test_pred_proba, test_pids, 'mean')
    patient_preds_max = aggregate_clips_to_patient(test_pred_proba, test_pids, 'max')
    
    # Get true labels
    test_patient_labels = {pid: outcomes_binary[patient_ids.index(pid)] 
                          for pid in test_patients[:5] if pid in patient_preds_mean}
    
    # Calculate metrics for both aggregation methods
    if len(test_patient_labels) > 0:
        y_true = [test_patient_labels[pid] for pid in patient_preds_mean.keys()]
        y_pred_mean = [patient_preds_mean[pid] for pid in patient_preds_mean.keys()]
        y_pred_max = [patient_preds_max[pid] for pid in patient_preds_max.keys()]
        
        metrics_mean = calculate_patient_metrics(y_true, y_pred_mean)
        metrics_max = calculate_patient_metrics(y_true, y_pred_max)
        
        print(f"\n   Results (Mean aggregation):")
        print(f"     ROC-AUC: {metrics_mean['roc_auc']:.3f}")
        print(f"     F1: {metrics_mean['f1']:.3f}")
        print(f"     Sensitivity: {metrics_mean['sensitivity']:.3f}")
        print(f"     Specificity: {metrics_mean['specificity']:.3f}")
        
        print(f"\n   Results (Max aggregation):")
        print(f"     ROC-AUC: {metrics_max['roc_auc']:.3f}")
        print(f"     F1: {metrics_max['f1']:.3f}")
        print(f"     Sensitivity: {metrics_max['sensitivity']:.3f}")
        print(f"     Specificity: {metrics_max['specificity']:.3f}")
        
        fold_results.append({
            'mean': metrics_mean,
            'max': metrics_max
        })

# Final summary
print(f"\n{'='*70}")
print("CROSS-VALIDATION SUMMARY")
print(f"{'='*70}")

if fold_results:
    mean_auc = np.mean([r['mean']['roc_auc'] for r in fold_results])
    max_auc = np.mean([r['max']['roc_auc'] for r in fold_results])
    
    print(f"\nAverage ROC-AUC (Mean aggregation): {mean_auc:.3f}")
    print(f"Average ROC-AUC (Max aggregation): {max_auc:.3f}")
    print(f"\nBest method: {'Mean' if mean_auc > max_auc else 'Max'} aggregation")

print("\n[OK] Training complete!")

