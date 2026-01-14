#!/usr/bin/env python3
"""
Test script for ES12 data loader functionality.

This script tests the specialized ES12 data loader to ensure it can
properly load and process the ES12.mat file.
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nasa_pcoe_eda.data.es12_loader import ES12DataLoader
from nasa_pcoe_eda.data.loader import DataLoader

def test_es12_loader():
    """Test the ES12 specialized loader."""
    print("=== Testing ES12 Specialized Loader ===")
    
    file_path = Path("data/raw/ES12.mat")
    
    if not file_path.exists():
        print(f"ES12 data file not found: {file_path}")
        return False
    
    try:
        # Test specialized ES12 loader
        loader = ES12DataLoader()
        df = loader.load_dataset(file_path)
        
        print(f"Successfully loaded ES12 data!")
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"Capacitors: {sorted(df['capacitor'].unique())}")
        print(f"Cycle range: {df['cycle'].min()} to {df['cycle'].max()}")
        
        # Show sample data
        print("\nSample data (first 5 rows):")
        print(df.head())
        
        # Test validation
        validation = loader.validate_data(df)
        print(f"\nValidation result: {'PASSED' if validation.is_valid else 'FAILED'}")
        if validation.errors:
            print(f"Errors: {validation.errors}")
        if validation.warnings:
            print(f"Warnings: {validation.warnings}")
        
        # Test metadata
        metadata = loader.get_metadata(df)
        print(f"\nMetadata:")
        print(f"  Records: {metadata.n_records}")
        print(f"  Features: {metadata.n_features}")
        print(f"  Memory usage: {metadata.memory_usage:.2f} MB")
        if metadata.date_range:
            print(f"  Date range: {metadata.date_range[0]} to {metadata.date_range[1]}")
        
        # Test capacitor-specific data
        print(f"\nTesting capacitor-specific data access:")
        cap_data = loader.get_capacitor_data('ES12C1')
        if cap_data is not None:
            print(f"ES12C1 data shape: {cap_data.shape}")
            print(f"ES12C1 cycles: {len(cap_data)}")
        
        # Test raw data access
        raw_data = loader.get_raw_transient_data('ES12C1')
        if raw_data is not None:
            print(f"ES12C1 raw VL shape: {raw_data['VL'].shape}")
            print(f"ES12C1 raw VO shape: {raw_data['VO'].shape}")
        
        return True
        
    except Exception as e:
        print(f"ES12 loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generic_loader():
    """Test the generic loader with ES12 detection."""
    print("\n=== Testing Generic Loader with ES12 Detection ===")
    
    file_path = Path("data/raw/ES12.mat")
    
    if not file_path.exists():
        print(f"ES12 data file not found: {file_path}")
        return False
    
    try:
        # Test generic loader (should auto-detect ES12)
        loader = DataLoader()
        df = loader.load_dataset(file_path)
        
        print(f"Successfully loaded data via generic loader!")
        print(f"DataFrame shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        # Check if it looks like ES12 data
        if 'capacitor' in df.columns:
            print(f"Capacitors detected: {sorted(df['capacitor'].unique())}")
            print("✓ ES12 auto-detection worked!")
        else:
            print("⚠ ES12 auto-detection may not have worked")
        
        return True
        
    except Exception as e:
        print(f"Generic loader test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("Testing ES12 data loading functionality...")
    
    success1 = test_es12_loader()
    success2 = test_generic_loader()
    
    if success1 and success2:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())