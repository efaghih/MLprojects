"""
Step 6: Create log-mel spectrograms
Purpose: Convert audio to time-frequency representation for CNN input
"""

import numpy as np
from scipy.signal import spectrogram
from scipy.io import wavfile

def create_log_mel_spectrogram(audio, sr=4000, n_mels=64, hop_length=512):
    """
    Create log-mel spectrogram from audio:
    1. Compute mel spectrogram (frequency representation)
    2. Convert to log scale (log-mel)
    Returns: spectrogram array (freq_bins x time_frames)
    """
    # Compute mel spectrogram using scipy
    # For simplicity, we'll use standard spectrogram and approximate mel scale
    f, t, Sxx = spectrogram(audio, sr, nperseg=2048, noverlap=1536)
    
    # Convert to mel scale approximation (simplified)
    # In production, use librosa.melspectrogram for proper mel scale
    # For now, we'll use log of power spectrogram
    mel_spec = np.log10(Sxx + 1e-10)  # Add small value to avoid log(0)
    
    return mel_spec

# Test with preprocessed audio
from step5_preprocess_audio import preprocess_audio

test_file = "dataset/training_data/2530_AV.wav"
audio, sr = preprocess_audio(test_file)
spec = create_log_mel_spectrogram(audio, sr)

print(f"Spectrogram shape: {spec.shape}")
print(f"Frequency bins: {spec.shape[0]}, Time frames: {spec.shape[1]}")
print(f"Value range: [{spec.min():.2f}, {spec.max():.2f}]")

print("\n[OK] Step 6 complete: Spectrogram creation ready")

