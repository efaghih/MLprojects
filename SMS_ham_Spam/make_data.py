"""
Data Preparation Script for SMS Spam Classification

Run this script to process the raw SMS data and create train/val/test splits.

Usage:
    python make_data.py
"""

from src.data import process_data
import os


if __name__ == "__main__":
    # Define paths
    input_path = "Dataset/SMSSpamCollection"
    output_dir = "data/processed"
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"❌ Error: Input file not found at '{input_path}'")
        exit(1)
    
    # Process the data with fixed random state for reproducibility
    train_df, val_df, test_df = process_data(
        input_path=input_path,
        output_dir=output_dir,
        random_state=42  # Fixed seed ensures same split every time
    )
    
    print("\n📊 Quick Data Preview:")
    print("\nFirst 3 training examples:")
    print(train_df.head(3).to_string(index=False))
    
    print("\n✅ All done! You can now use the processed data for model training.")

