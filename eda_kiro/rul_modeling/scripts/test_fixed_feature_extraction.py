#!/usr/bin/env python3
"""
Test script to verify that the feature extraction fixes work correctly.
"""

import sys
from pathlib import Path
import logging
import numpy as np

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from true_rul.data_loader import DataLoader
from true_rul.feature_extractor import FeatureExtractor
from true_rul.config import setup_logging

logger = logging.getLogger(__name__)


def test_feature_extraction_consistency():
    """Test that feature extraction returns consistent feature counts"""
    
    # Set up logging
    setup_logging(log_file="test_feature_extraction.log", level=logging.INFO)
    logger.info("Testing feature extraction consistency...")
    
    # Define paths
    data_path = Path("~/work/CapacitorElectricalStress/eda_kiro/data/raw/ES12.mat").expanduser()
    
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return False
    
    try:
        # Initialize data loader and feature extractor
        data_loader = DataLoader()
        feature_extractor = FeatureExtractor(include_advanced=True, rolling_window=5)
        
        # Load ES12 dataset
        capacitor_data = data_loader.load_es12_dataset(data_path)
        logger.info(f"Loaded {len(capacitor_data)} capacitors")
        
        # Test feature extraction on first capacitor
        cap_id = "ES12C1"
        if cap_id not in capacitor_data:
            cap_id = list(capacitor_data.keys())[0]
        
        cap_data = capacitor_data[cap_id]
        logger.info(f"Testing feature extraction on {cap_id} ({cap_data.total_cycles} cycles)")
        
        feature_counts = []
        feature_names_list = []
        
        # Test first 20 cycles to see the pattern
        test_cycles = min(20, cap_data.total_cycles)
        
        for i in range(test_cycles):
            cycle = cap_data.cycles[i]
            
            # Get history for rolling features
            history_start = max(0, cycle.cycle_number - feature_extractor.rolling_window)
            history = [c for c in cap_data.cycles 
                      if history_start <= c.cycle_number < cycle.cycle_number]
            
            try:
                # Extract features
                features_dict = feature_extractor.extract_features(cycle, cap_id, history)
                feature_count = len(features_dict)
                feature_names = list(features_dict.keys())
                
                feature_counts.append(feature_count)
                feature_names_list.append(feature_names)
                
                logger.info(f"Cycle {cycle.cycle_number}: {feature_count} features")
                
                # Check for NaN or infinite values
                nan_count = sum(1 for v in features_dict.values() if not np.isfinite(v))
                if nan_count > 0:
                    logger.warning(f"Cycle {cycle.cycle_number}: {nan_count} non-finite features")
                
            except Exception as e:
                logger.error(f"Failed to extract features for cycle {cycle.cycle_number}: {e}")
                return False
        
        # Check consistency
        unique_counts = set(feature_counts)
        logger.info(f"Feature counts: {feature_counts}")
        logger.info(f"Unique feature counts: {unique_counts}")
        
        if len(unique_counts) == 1:
            logger.info("✅ Feature extraction is consistent!")
            logger.info(f"All cycles have {list(unique_counts)[0]} features")
            
            # Show feature names from last cycle
            if feature_names_list:
                logger.info("Feature names:")
                for name in sorted(feature_names_list[-1]):
                    logger.info(f"  - {name}")
            
            return True
        else:
            logger.error("❌ Feature extraction is inconsistent!")
            
            # Show differences
            for i, (count, names) in enumerate(zip(feature_counts, feature_names_list)):
                if count != feature_counts[0]:
                    logger.error(f"Cycle {i+1}: {count} features (expected {feature_counts[0]})")
                    
                    # Show missing/extra features
                    if i > 0:
                        expected_names = set(feature_names_list[0])
                        actual_names = set(names)
                        missing = expected_names - actual_names
                        extra = actual_names - expected_names
                        
                        if missing:
                            logger.error(f"  Missing features: {missing}")
                        if extra:
                            logger.error(f"  Extra features: {extra}")
            
            return False
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def main():
    """Main test function"""
    success = test_feature_extraction_consistency()
    
    if success:
        print("🎉 Feature extraction test PASSED!")
    else:
        print("❌ Feature extraction test FAILED!")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)