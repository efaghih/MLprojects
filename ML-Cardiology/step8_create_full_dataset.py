"""
Step 8: Create full dataset pipeline
Purpose: Process all WAV files → clips → spectrograms → labels
"""

from step4_create_dataset_mapping import create_dataset_mapping
from step5_preprocess_audio import preprocess_audio
from step6_create_spectrograms import create_log_mel_spectrogram
from step7_split_into_clips import split_into_clips

def process_wav_to_clips(wav_file, clip_duration=4, overlap=0.5):
    """
    Complete pipeline: WAV → audio → clips → spectrograms
    Returns: list of spectrograms and their label
    """
    # Load and preprocess audio
    audio, sr = preprocess_audio(wav_file)
    
    # Split into clips
    clips = split_into_clips(audio, sr, clip_duration, overlap)
    
    # Convert each clip to spectrogram
    spectrograms = []
    for clip in clips:
        spec = create_log_mel_spectrogram(clip, sr)
        spectrograms.append(spec)
    
    return spectrograms

# Test with a few files (small sample for now)
dataset = create_dataset_mapping()
print(f"Processing sample of 3 WAV files...")

total_clips = 0
for wav_file, patient_id, outcome in dataset[:3]:
    specs = process_wav_to_clips(wav_file)
    total_clips += len(specs)
    print(f"  {patient_id}: {len(specs)} clips, label: {outcome}")

print(f"\nTotal clips from 3 files: {total_clips}")
print(f"Average clips per file: {total_clips/3:.1f}")

print("\n[OK] Step 8 complete: Full dataset pipeline ready")

