"""
Example: Dataset Splitting with DatasetSplitter

This example demonstrates how to use the DatasetSplitter class to split
the ES12 dataset into train/validation/test sets using a hybrid strategy.
"""

from src.data_preparation.dataset_splitter import split_dataset

def main():
    """Main function to demonstrate dataset splitting."""
    
    print("="*70)
    print("Dataset Splitting Example")
    print("="*70)
    
    # Split the dataset using default configuration
    print("\n1. Splitting with default configuration...")
    train_df, val_df, test_df = split_dataset(
        input_path='output/features/es12_features_with_labels.csv',
        output_dir='output/features'
    )
    
    print("\n2. Verifying splits...")
    print(f"   Train: {len(train_df)} samples")
    print(f"   Val:   {len(val_df)} samples")
    print(f"   Test:  {len(test_df)} samples")
    
    print("\n3. Checking capacitor distribution...")
    print(f"   Train capacitors: {sorted(train_df['capacitor_id'].unique())}")
    print(f"   Val capacitors:   {sorted(val_df['capacitor_id'].unique())}")
    print(f"   Test capacitors:  {sorted(test_df['capacitor_id'].unique())}")
    
    print("\n4. Checking cycle ranges...")
    print(f"   Train cycles: {train_df['cycle'].min()}-{train_df['cycle'].max()}")
    print(f"   Val cycles:   {val_df['cycle'].min()}-{val_df['cycle'].max()}")
    print(f"   Test cycles:  {test_df['cycle'].min()}-{test_df['cycle'].max()}")
    
    print("\n" + "="*70)
    print("✅ Dataset splitting completed successfully!")
    print("="*70)
    
    return train_df, val_df, test_df


if __name__ == '__main__':
    main()
