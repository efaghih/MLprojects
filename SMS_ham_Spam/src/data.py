"""
Data Ingestion and Preprocessing Module for SMS Spam Classification

This module handles:
- Loading raw SMS spam data
- Creating binary labels (0=ham, 1=spam)
- Stratified train/validation/test splitting
- Saving processed datasets
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os


def load_raw_data(filepath):
    """
    Load the raw SMS Spam Collection dataset.
    
    Args:
        filepath (str): Path to the SMSSpamCollection file
        
    Returns:
        pd.DataFrame: DataFrame with 'label' and 'text' columns
    """
    # Load the tab-separated file
    # The file has no header, so we specify column names
    df = pd.read_csv(
        filepath,
        sep='\t',
        header=None,
        names=['label', 'text'],
        encoding='utf-8'
    )
    
    print(f"✓ Loaded {len(df)} SMS messages")
    print(f"  - Ham messages: {(df['label'] == 'ham').sum()}")
    print(f"  - Spam messages: {(df['label'] == 'spam').sum()}")
    
    return df


def create_binary_labels(df):
    """
    Convert string labels ('ham'/'spam') to binary integers (0/1).
    
    Args:
        df (pd.DataFrame): DataFrame with 'label' column
        
    Returns:
        pd.DataFrame: DataFrame with binary 'label' column (0=ham, 1=spam)
    """
    # Create a copy to avoid modifying original
    df_copy = df.copy()
    
    # Convert: spam -> 1, ham -> 0
    df_copy['label'] = df_copy['label'].apply(lambda x: 1 if x == 'spam' else 0)
    
    print(f"\n✓ Created binary labels")
    print(f"  - Class 0 (ham): {(df_copy['label'] == 0).sum()}")
    print(f"  - Class 1 (spam): {(df_copy['label'] == 1).sum()}")
    print(f"  - Class balance: {df_copy['label'].mean():.2%} spam")
    
    return df_copy


def stratified_split(df, train_size=0.8, val_size=0.1, test_size=0.1, random_state=42):
    """
    Perform stratified train/validation/test split.
    
    Stratified splitting ensures that the proportion of spam/ham messages
    is maintained across all three splits.
    
    Args:
        df (pd.DataFrame): DataFrame with 'label' and 'text' columns
        train_size (float): Proportion of data for training (default: 0.8)
        val_size (float): Proportion of data for validation (default: 0.1)
        test_size (float): Proportion of data for testing (default: 0.1)
        random_state (int): Random seed for reproducibility (default: 42)
        
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    # Verify split sizes sum to 1
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, \
        "Train, validation, and test sizes must sum to 1.0"
    
    # First split: separate train from (val + test)
    # The test_size for first split is val_size + test_size
    temp_test_size = val_size + test_size
    
    train_df, temp_df = train_test_split(
        df,
        test_size=temp_test_size,
        stratify=df['label'],
        random_state=random_state
    )
    
    # Second split: separate val from test
    # Calculate relative size of test within the temp set
    relative_test_size = test_size / temp_test_size
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df['label'],
        random_state=random_state
    )
    
    print(f"\n✓ Stratified split completed (random_state={random_state})")
    print(f"\n  Training set: {len(train_df)} samples ({len(train_df)/len(df):.1%})")
    print(f"    - Ham: {(train_df['label'] == 0).sum()}, Spam: {(train_df['label'] == 1).sum()}")
    print(f"    - Spam ratio: {train_df['label'].mean():.2%}")
    
    print(f"\n  Validation set: {len(val_df)} samples ({len(val_df)/len(df):.1%})")
    print(f"    - Ham: {(val_df['label'] == 0).sum()}, Spam: {(val_df['label'] == 1).sum()}")
    print(f"    - Spam ratio: {val_df['label'].mean():.2%}")
    
    print(f"\n  Test set: {len(test_df)} samples ({len(test_df)/len(df):.1%})")
    print(f"    - Ham: {(test_df['label'] == 0).sum()}, Spam: {(test_df['label'] == 1).sum()}")
    print(f"    - Spam ratio: {test_df['label'].mean():.2%}")
    
    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df, output_dir='data/processed'):
    """
    Save train, validation, and test splits as CSV files.
    
    Args:
        train_df (pd.DataFrame): Training data
        val_df (pd.DataFrame): Validation data
        test_df (pd.DataFrame): Test data
        output_dir (str): Directory to save the CSV files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each split
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\n✓ Saved datasets to '{output_dir}/'")
    print(f"  - train.csv: {len(train_df)} samples")
    print(f"  - val.csv: {len(val_df)} samples")
    print(f"  - test.csv: {len(test_df)} samples")


def process_data(input_path, output_dir='data/processed', random_state=42):
    """
    Main pipeline: Load, process, split, and save data.
    
    Args:
        input_path (str): Path to raw SMSSpamCollection file
        output_dir (str): Directory to save processed splits
        random_state (int): Random seed for reproducibility
    """
    print("="*70)
    print("SMS SPAM CLASSIFICATION - DATA INGESTION PIPELINE")
    print("="*70)
    
    # Step 1: Load raw data
    print("\n[1/4] Loading raw data...")
    df = load_raw_data(input_path)
    
    # Step 2: Create binary labels
    print("\n[2/4] Creating binary labels...")
    df = create_binary_labels(df)
    
    # Step 3: Stratified split (80/10/10)
    print("\n[3/4] Performing stratified split...")
    train_df, val_df, test_df = stratified_split(
        df,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
        random_state=random_state
    )
    
    # Step 4: Save splits
    print("\n[4/4] Saving splits...")
    save_splits(train_df, val_df, test_df, output_dir)
    
    print("\n" + "="*70)
    print("✅ DATA INGESTION COMPLETE!")
    print("="*70)
    
    return train_df, val_df, test_df


if __name__ == "__main__":
    # If this script is run directly, process the data
    input_file = "../Dataset/SMSSpamCollection"
    output_directory = "../data/processed"
    
    process_data(input_file, output_directory, random_state=42)

