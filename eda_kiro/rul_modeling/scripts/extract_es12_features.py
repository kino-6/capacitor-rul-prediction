#!/usr/bin/env python3
"""
Extract features from all ES12 capacitors (ES12C1-ES12C8).

This script uses parallel processing to extract features from all 200 cycles
of each capacitor, without historical features (to save processing time).

Expected output: output/features/es12_features.csv (1600 rows × ~30 columns)
Expected processing time: 3-4 minutes with parallel processing
"""

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_preparation.parallel_extractor import extract_es12_features


def main():
    """Extract features from ES12 dataset."""
    
    # Configuration
    es12_path = "../data/raw/ES12.mat"
    output_path = "output/features/es12_features.csv"
    
    # All ES12 capacitors
    capacitor_ids = [f"ES12C{i}" for i in range(1, 9)]  # ES12C1 to ES12C8
    
    # Extract features
    print("\n" + "="*70)
    print("Task 1.2: Extract Features from ES12 Dataset")
    print("="*70)
    print(f"Input: {es12_path}")
    print(f"Output: {output_path}")
    print(f"Capacitors: {capacitor_ids}")
    print(f"Cycles per capacitor: 200")
    print(f"Historical features: No (Phase 1)")
    print("="*70 + "\n")
    
    # Run extraction
    features_df = extract_es12_features(
        es12_path=es12_path,
        output_path=output_path,
        capacitor_ids=capacitor_ids,
        max_cycles=200,
        n_processes=None,  # Use all available cores
        include_history=False,  # No history for Phase 1
        progress_interval=20  # Report every 20 cycles
    )
    
    # Summary
    print("\n" + "="*70)
    print("Extraction Summary")
    print("="*70)
    print(f"Total samples: {len(features_df)}")
    print(f"Total features: {len(features_df.columns) - 2}")  # Exclude metadata
    print(f"Shape: {features_df.shape}")
    print(f"Capacitors: {features_df['capacitor_id'].nunique()}")
    print(f"Cycles per capacitor: {features_df.groupby('capacitor_id')['cycle'].count().values}")
    print("\nFirst few rows:")
    print(features_df.head())
    print("\nFeature columns:")
    feature_cols = [col for col in features_df.columns if col not in ['capacitor_id', 'cycle']]
    print(f"  {', '.join(feature_cols)}")
    print("="*70 + "\n")
    
    print(f"✓ Task 1.2 completed successfully!")
    print(f"✓ Features saved to: {output_path}")


if __name__ == "__main__":
    main()
