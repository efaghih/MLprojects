"""
Step 12: Implement CNN model with TensorFlow/Keras
Purpose: Create actual trainable CNN model for spectrogram classification
"""

try:
    from tensorflow import keras
    from tensorflow.keras import layers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("[NOTE] TensorFlow not installed - showing model structure only")

def create_cnn_model(input_shape=(1025, 181, 1)):
    """
    Create CNN model for binary classification (Normal/Abnormal)
    Input: log-mel spectrogram
    Output: probability of Abnormal
    """
    if not TENSORFLOW_AVAILABLE:
        return None
    
    model = keras.Sequential([
        # First conv block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        
        # Second conv block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Test model creation
if TENSORFLOW_AVAILABLE:
    model = create_cnn_model()
    print("CNN Model created successfully!")
    print(f"Total parameters: {model.count_params():,}")
    model.summary()
else:
    print("Model structure:")
    print("  Input: (1025, 181, 1) - log-mel spectrogram")
    print("  Conv2D(32) -> MaxPool -> Conv2D(64) -> MaxPool")
    print("  Flatten -> Dense(128) -> Dropout(0.5) -> Dense(1, sigmoid)")
    print("\n[NOTE] Install TensorFlow to create actual model:")
    print("  pip install tensorflow")

print("\n[OK] Step 12 complete: CNN model implementation ready")

