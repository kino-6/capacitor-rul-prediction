#!/usr/bin/env python3
"""
Test script for FeatureExtractor
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from true_rul.data_loader import DataLoader
from true_rul.feature_extractor import FeatureExtractor
from true_rul.feature_normalizer import FeatureNormalizer
from true_rul.config import setup_logging

def main():
    """Test feature extraction"""
    setup_logging("test_feature_extractor.log")
    
    # Path to ES12 data
    data_path = Path("../data/raw/ES12.mat")
    
    if not data_path.exists():
        print(f"❌ ES12 data file not found: {data_path}")
        return
    
    print("🔄 Loading ES12 dataset...")
    loader = DataLoader()
    capacitor_data = loader.load_es12_dataset(data_path)
    
    print(f"✅ Loaded {len(capacitor_data)} capacitors")
    print()
    
    # Initialize feature extractor
    print("🔄 Initializing feature extractor...")
    extractor = FeatureExtractor(include_advanced=True, rolling_window=5)
    print(f"✅ Feature extractor ready ({extractor.n_features} features)")
    print()
    
    # Test on first capacitor
    cap_id = list(capacitor_data.keys())[0]
    cap_data = capacitor_data[cap_id]
    
    print(f"📊 Testing on {cap_id} ({cap_data.total_cycles} cycles)")
    print()
    
    # Extract features for first few cycles
    all_features = []
    feature_names = extractor.get_feature_names()
    
    for i in range(min(10, cap_data.total_cycles)):
        cycle = cap_data.get_cycle(i + 1)
        
        # Get history for rolling features
        history = cap_data.get_cycles_range(1, i) if i > 0 else []
        
        # Extract features
        features = extractor.extract_features(cycle, cap_id, history)
        all_features.append(list(features.values()))
        
        if i == 0:
            print(f"Cycle {i + 1} features:")
            for name, value in list(features.items())[:10]:
                print(f"  {name}: {value:.6f}")
            print(f"  ... ({len(features)} total features)")
            print()
    
    # Convert to numpy array
    features_array = np.array(all_features)
    print(f"✅ Extracted features shape: {features_array.shape}")
    print()
    
    # Test normalization
    print("🔄 Testing feature normalization...")
    normalizer = FeatureNormalizer(method="standard")
    
    # Fit on first 5 cycles
    normalizer.fit(features_array[:5], cap_id)
    
    # Transform all cycles
    normalized = normalizer.transform(features_array, cap_id)
    
    print(f"✅ Normalized features shape: {normalized.shape}")
    print(f"   Mean: {np.mean(normalized, axis=0)[:5]}")
    print(f"   Std:  {np.std(normalized, axis=0)[:5]}")
    print()
    
    # Feature statistics
    print("📈 Feature statistics:")
    print(f"   Min:  {np.min(features_array, axis=0)[:5]}")
    print(f"   Max:  {np.max(features_array, axis=0)[:5]}")
    print(f"   Mean: {np.mean(features_array, axis=0)[:5]}")
    print()
    
    print("✅ Feature extractor test passed!")

if __name__ == "__main__":
    main()
