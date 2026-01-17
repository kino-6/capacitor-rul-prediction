"""
Test script for parallel feature extraction.
"""

from pathlib import Path
from src.data_preparation.parallel_extractor import extract_es12_features

# Configuration
ES12_PATH = "../data/raw/ES12.mat"
OUTPUT_PATH = "output/features/es12_features.csv"

# Test with a single capacitor first
print("Testing with ES12C1 only (first 50 cycles)...")
features_df = extract_es12_features(
    es12_path=ES12_PATH,
    output_path="output/features/test_es12c1_features.csv",
    capacitor_ids=["ES12C1"],
    max_cycles=50,
    n_processes=1,
    include_history=False,
    progress_interval=10
)

print("\nFeature extraction test completed!")
print(f"Shape: {features_df.shape}")
print(f"\nFirst few rows:")
print(features_df.head())
print(f"\nFeature columns:")
print(features_df.columns.tolist())
