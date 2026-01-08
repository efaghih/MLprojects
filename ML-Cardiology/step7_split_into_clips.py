"""
Step 7: Split recordings into fixed-length clips
Purpose: Create uniform-length clips (4-5 seconds) for CNN training
"""

import numpy as np
from step5_preprocess_audio import preprocess_audio
from step6_create_spectrograms import create_log_mel_spectrogram

def split_into_clips(audio, sr=4000, clip_duration=4, overlap=0.5):
    """
    Split audio into overlapping clips:
    1. Calculate clip length in samples
    2. Create overlapping windows
    3. Return list of clip arrays
    """
    clip_samples = int(clip_duration * sr)  # 4 seconds * 4000 Hz = 16000 samples
    hop_samples = int(clip_samples * (1 - overlap))  # 50% overlap = 8000 samples
    
    clips = []
    start = 0
    
    # Extract clips with overlap
    while start + clip_samples <= len(audio):
        clip = audio[start:start + clip_samples]
        clips.append(clip)
        start += hop_samples
    
    # Include remaining audio if significant (>2 seconds)
    if len(audio) - start > 2 * sr:
        last_clip = audio[-clip_samples:]
        clips.append(last_clip)
    
    return clips

# Test: split a recording into clips
test_file = "dataset/training_data/2530_AV.wav"
audio, sr = preprocess_audio(test_file)
clips = split_into_clips(audio, sr, clip_duration=4, overlap=0.5)

print(f"Original duration: {len(audio)/sr:.2f} seconds")
print(f"Number of clips created: {len(clips)}")
print(f"Each clip duration: {len(clips[0])/sr:.2f} seconds")
print(f"Total clip duration: {len(clips) * len(clips[0])/sr:.2f} seconds")

print("\n[OK] Step 7 complete: Clip splitting ready")

