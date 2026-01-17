"""
Test script for parallel feature extraction.
"""

from pathlib import Path
from src.data_preparation.parallel_extractor import ParallelFeatureExtractor

# Configuration
ES12_PATH = "../data/raw/ES12.mat"
OUTPUT_PATH = "output/features/test_es12_features.csv"

# Test with just 2 capacitors and 10 cycles for quick verification
TEST_CAPACITORS = ["ES12C1", "ES12C2"]
TEST_CYCLES = 10

def test_parallel_extraction():
    """Test the parallel feature extraction with a small subset."""
    print("Testing Parallel Feature Extraction")
    print("=" * 70)
    
    # Check if ES12 file exists
    if not Path(ES12_PATH).exists():
        print(f"ERROR: ES12 data file not found at {ES12_PATH}")
        print("Please ensure the data file is in the correct location.")
        return False
    
    try:
        # Initialize extractor
        extractor = ParallelFeatureExtractor(
            es12_path=ES12_PATH,
            n_processes=2,  # Use 2 processes for testing
            include_history=False
        )
        
        # Extract features
        features_df = extractor.extract_all_capacitors(
            capacitor_ids=TEST_CAPACITORS,
            max_cycles=TEST_CYCLES,
            progress_interval=5
        )
        
        # Display results
        print("\nExtraction Results:")
        print(f"Shape: {features_df.shape}")
        print(f"Columns: {list(features_df.columns)}")
        print(f"\nFirst few rows:")
        print(features_df.head())
        
        # Save results
        extractor.save_features(features_df, OUTPUT_PATH)
        
        print("\n✓ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_parallel_extraction()
    exit(0 if success else 1)
