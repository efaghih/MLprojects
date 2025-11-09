"""
Baseline Model Training Script

Run this script to train the TF-IDF + Naive Bayes baseline model.

Usage:
    python train_baseline.py
"""

from src.train_baseline import train_baseline_model


if __name__ == "__main__":
    # Train the baseline model
    # This will load data, train, evaluate, and save the model
    pipeline, results = train_baseline_model(
        data_dir='data/processed',
        output_dir='models/baseline'
    )
    
    print("\n🎉 Your baseline model is ready!")
    print("\n📁 Files created:")
    print("   • models/baseline/baseline_model.pkl")
    print("   • models/baseline/metrics.txt")
    print("   • models/baseline/confusion_matrices.png")

