"""
Verification script for feature scaling.

This script verifies that:
1. Scaled features have mean ≈ 0 and std ≈ 1
2. Metadata columns are unchanged
3. All files are correctly saved
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path


def verify_scaling():
    """Verify that feature scaling was done correctly."""
    
    print("=" * 80)
    print("Feature Scaling Verification")
    print("=" * 80)
    print()
    
    # Load datasets
    print("[1/4] Loading datasets...")
    train_df = pd.read_csv("output/features/train.csv")
    val_df = pd.read_csv("output/features/val.csv")
    test_df = pd.read_csv("output/features/test.csv")
    
    print(f"  ✓ Train: {train_df.shape}")
    print(f"  ✓ Val:   {val_df.shape}")
    print(f"  ✓ Test:  {test_df.shape}")
    print()
    
    # Load scaler
    print("[2/4] Loading scaler...")
    with open("output/models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    print(f"  ✓ Scaler loaded")
    print()
    
    # Identify feature columns
    metadata_columns = ['capacitor_id', 'cycle', 'is_abnormal', 'rul']
    feature_columns = [col for col in train_df.columns if col not in metadata_columns]
    
    print(f"[3/4] Verifying scaled features...")
    print(f"  Feature columns: {len(feature_columns)}")
    print(f"  Metadata columns: {len(metadata_columns)}")
    print()
    
    # Check training set statistics
    train_features = train_df[feature_columns]
    train_mean = train_features.mean()
    train_std = train_features.std()
    
    # Check if mean is close to 0 and std is close to 1
    mean_close_to_zero = np.allclose(train_mean, 0, atol=1e-10)
    std_close_to_one = np.allclose(train_std, 1, atol=1e-2)
    
    print(f"  Training set feature statistics:")
    print(f"    Mean close to 0: {mean_close_to_zero} (max abs: {abs(train_mean).max():.2e})")
    print(f"    Std close to 1:  {std_close_to_one} (range: [{train_std.min():.4f}, {train_std.max():.4f}])")
    print()
    
    # Check metadata columns are unchanged
    print("[4/4] Verifying metadata columns...")
    
    # Load unscaled data for comparison
    train_unscaled = pd.read_csv("output/features/train_unscaled.csv")
    
    metadata_unchanged = True
    for col in metadata_columns:
        if not train_df[col].equals(train_unscaled[col]):
            print(f"  ✗ {col} was modified!")
            metadata_unchanged = False
    
    if metadata_unchanged:
        print(f"  ✓ All metadata columns unchanged")
    print()
    
    # Summary
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print()
    
    all_checks_passed = mean_close_to_zero and std_close_to_one and metadata_unchanged
    
    if all_checks_passed:
        print("✓ All checks passed!")
        print()
        print("Feature scaling completed successfully:")
        print(f"  - Scaled datasets: train.csv, val.csv, test.csv")
        print(f"  - Scaler saved: output/models/scaler.pkl")
        print(f"  - Dataset summary: output/features/dataset_summary.txt")
        print(f"  - Backup (unscaled): train_unscaled.csv, val_unscaled.csv, test_unscaled.csv")
    else:
        print("✗ Some checks failed. Please review the output above.")
    
    print()
    print("=" * 80)
    
    return all_checks_passed


if __name__ == "__main__":
    verify_scaling()
