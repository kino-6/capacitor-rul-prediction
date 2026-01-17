"""
Debug script to identify the extraction error at cycle 139+.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import load_es12_cycle_data
from src.feature_extraction.response_extractor import ResponseFeatureExtractor

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR.parent / "data" / "raw"
ES12_PATH = DATA_DIR / "ES12.mat"

def debug_cycle_extraction(cap_id: str, cycle: int):
    """Debug extraction for a specific cycle."""
    print(f"\n{'='*80}")
    print(f"Debugging {cap_id} Cycle {cycle}")
    print('='*80)
    
    try:
        # Load data
        print("\n1. Loading data...")
        vl, vo = load_es12_cycle_data(str(ES12_PATH), cap_id, cycle)
        print(f"   VL shape: {vl.shape}, dtype: {vl.dtype}, type: {type(vl)}")
        print(f"   VO shape: {vo.shape}, dtype: {vo.dtype}, type: {type(vo)}")
        print(f"   VL sample: {vl[:5]}")
        print(f"   VO sample: {vo[:5]}")
        
        # Check if arrays are actually arrays
        print(f"\n2. Type checks:")
        print(f"   isinstance(vl, np.ndarray): {isinstance(vl, np.ndarray)}")
        print(f"   isinstance(vo, np.ndarray): {isinstance(vo, np.ndarray)}")
        print(f"   type(vl): {type(vl)}")
        print(f"   type(vo): {type(vo)}")
        
        # Try basic operations
        print(f"\n3. Basic operations:")
        print(f"   len(vl): {len(vl)}")
        print(f"   len(vo): {len(vo)}")
        print(f"   vl > 0: {(vl > 0).sum()} elements")
        print(f"   vo > 0: {(vo > 0).sum()} elements")
        
        # Extract features
        print(f"\n4. Extracting features...")
        extractor = ResponseFeatureExtractor()
        features = extractor.extract_features(vl, vo, cap_id, cycle, include_advanced=True)
        
        print(f"   ✓ Features extracted successfully!")
        print(f"   Features: {list(features.keys())}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print(f"   Error type: {type(e)}")
        import traceback
        print("\nFull traceback:")
        traceback.print_exc()
        return False


def main():
    """Main debug execution."""
    print("="*80)
    print("EXTRACTION ERROR DEBUG")
    print("="*80)
    
    # Test cycles around the failure point
    test_cases = [
        ("ES12C1", 1),
        ("ES12C1", 50),
        ("ES12C1", 138),
        ("ES12C1", 139),
        ("ES12C1", 140),
        ("ES12C1", 150),
        ("ES12C1", 200),
    ]
    
    results = {}
    for cap_id, cycle in test_cases:
        success = debug_cycle_extraction(cap_id, cycle)
        results[(cap_id, cycle)] = success
    
    # Summary
    print("\n" + "="*80)
    print("DEBUG SUMMARY")
    print("="*80)
    
    for (cap_id, cycle), success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {cap_id} Cycle {cycle}")
    
    # Identify failure point
    failures = [k for k, v in results.items() if not v]
    if failures:
        print(f"\n⚠️ First failure at: {failures[0]}")
    else:
        print(f"\n✅ All test cases passed!")


if __name__ == "__main__":
    main()
