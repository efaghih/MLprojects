"""
Step 10: Build simple CNN model
Purpose: Create a small CNN to classify spectrograms (Normal/Abnormal)
"""

import numpy as np

def create_simple_cnn(input_shape=(1025, 181, 1)):
    """
    Create a simple CNN architecture:
    - Input: log-mel spectrogram
    - Output: probability of Abnormal
    Note: This is a model structure definition (not full training code)
    """
    # Model architecture (conceptual - will use TensorFlow/Keras later)
    layers = [
        "Input: spectrogram (1025 x 181)",
        "Conv2D: 32 filters, 3x3 kernel",
        "MaxPooling2D: 2x2",
        "Conv2D: 64 filters, 3x3 kernel",
        "MaxPooling2D: 2x2",
        "Flatten",
        "Dense: 128 units",
        "Dropout: 0.5",
        "Dense: 1 unit (sigmoid) -> Abnormal probability"
    ]
    
    return layers

# Show model structure
model_layers = create_simple_cnn()
print("CNN Model Architecture:")
for i, layer in enumerate(model_layers, 1):
    print(f"  {i}. {layer}")

print("\nModel will:")
print("  - Input: Spectrogram images (1025 x 181)")
print("  - Output: Single probability (0=Normal, 1=Abnormal)")
print("  - Loss: Binary cross-entropy")
print("  - Optimizer: Adam")

print("\n[OK] Step 10 complete: CNN model structure defined")

