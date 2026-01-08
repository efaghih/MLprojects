"""
Step 5: Preprocess audio files
Purpose: Load WAV files, ensure consistent sampling rate, normalize amplitude
"""

from scipy.io import wavfile
import numpy as np

def preprocess_audio(wav_file, target_sr=4000):
    """
    Load audio file and preprocess:
    1. Load audio data
    2. Resample if needed (to target_sr)
    3. Normalize amplitude to [-1, 1] range
    Returns: audio array, actual sampling rate
    """
    # Load audio file (scipy returns sampling rate and audio data)
    sr, audio = wavfile.read(wav_file)
    
    # Convert to float and normalize to [-1, 1] range
    # WAV files are typically 16-bit integers, so divide by max possible value
    audio = audio.astype(np.float32)
    if audio.dtype == np.int16:
        audio = audio / 32768.0  # 16-bit audio range
    else:
        # Normalize by max absolute value
        audio = audio / np.max(np.abs(audio))
    
    # Note: We'll handle resampling in next step if needed
    # For now, assume all files are 4000 Hz (verified in Step 2)
    
    return audio, sr

# Test with a sample file
test_file = "dataset/training_data/2530_AV.wav"
audio, sr = preprocess_audio(test_file)

print(f"Loaded: {test_file}")
print(f"Audio shape: {audio.shape}")
print(f"Sampling rate: {sr} Hz")
print(f"Duration: {len(audio)/sr:.2f} seconds")
print(f"Amplitude range: [{audio.min():.4f}, {audio.max():.4f}]")

print("\n[OK] Step 5 complete: Audio preprocessing function ready")

