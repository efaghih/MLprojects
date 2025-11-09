"""
DistilBERT Model Training Script

Run this script to train the DistilBERT transformer model for SMS spam classification.

This provides a modern deep learning baseline to compare against the classical
TF-IDF + Naive Bayes model.

Usage:
    python train_distilbert.py
    
Note: Training will take 10-30 minutes depending on your hardware.
      GPU is recommended but not required.
"""

from src.train_distilbert import train_distilbert_model


if __name__ == "__main__":
    print("🤖 Starting DistilBERT training...")
    print("\nNote: This will take 10-30 minutes depending on your hardware.")
    print("GPU will significantly speed up training if available.\n")
    
    # Train the DistilBERT model with specified hyperparameters
    trainer, results = train_distilbert_model(
        data_dir='data/processed',
        output_dir='models/distilbert',
        model_name='distilbert-base-uncased',
        max_length=128,        # Truncate at 128 tokens
        batch_size=16,         # Batch size 16
        num_epochs=3,          # 3 epochs
        learning_rate=2e-5     # Small learning rate
    )
    
    print("\n🎉 DistilBERT model training complete!")
    print("\n📁 Files created:")
    print("   • models/distilbert/ (model checkpoints)")
    print("   • models/distilbert/metrics.txt")
    print("   • models/distilbert/confusion_matrices.png")

