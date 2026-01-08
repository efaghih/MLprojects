"""
Step 13: Prepare training data
Purpose: Process all WAV files into clips + spectrograms with labels
"""

from step4_create_dataset_mapping import create_dataset_mapping
from step5_preprocess_audio import preprocess_audio
from step6_create_spectrograms import create_log_mel_spectrogram
from step7_split_into_clips import split_into_clips
import numpy as np

def prepare_all_clips(dataset, max_files=None):
    """
    Process all WAV files into clips with labels
    Returns: (spectrograms_list, labels_list, patient_ids_list)
    """
    X = []  # Spectrograms
    y = []  # Labels (0=Normal, 1=Abnormal)
    patient_ids = []  # Track which patient each clip belongs to
    
    # Process each WAV file
    for i, (wav_file, patient_id, outcome) in enumerate(dataset):
        if max_files and i >= max_files:
            break
        
        # Convert outcome to binary (0=Normal, 1=Abnormal)
        label = 1 if outcome == "Abnormal" else 0
        
        # Process WAV: load -> split into clips
        audio, sr = preprocess_audio(wav_file)
        clips = split_into_clips(audio, sr, clip_duration=4, overlap=0.5)
        
        # Convert each clip to spectrogram
        for clip in clips:
            spec = create_log_mel_spectrogram(clip, sr)
            # Reshape for CNN: add channel dimension (freq, time, 1)
            spec = spec[:, :, np.newaxis]
            X.append(spec)
            y.append(label)
            patient_ids.append(patient_id)
    
    return np.array(X), np.array(y), patient_ids

# Test with small sample (3 files)
print("Preparing training data (sample: 3 files)...")
dataset = create_dataset_mapping()
X, y, patient_ids = prepare_all_clips(dataset, max_files=3)

print(f"\nData prepared:")
print(f"  Spectrograms shape: {X.shape}")
print(f"  Labels shape: {y.shape}")
print(f"  Normal clips: {np.sum(y == 0)}, Abnormal clips: {np.sum(y == 1)}")
print(f"  Total clips: {len(patient_ids)}")

print("\n[OK] Step 13 complete: Training data preparation ready")

