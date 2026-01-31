#!/usr/bin/env python3
"""
Quick test for feature extraction consistency
"""

import sys
from pathlib import Path
import logging
import numpy as np

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from true_rul.data_loader import DataLoader
from true_rul.feature_extractor import FeatureExtractor
from true_rul.config import ES12_CONFIG, MODEL_CONFIG, setup_logging

# Set up logging
setup_logging(log_file="test_feature_extraction.log", level=logging.INFO)
logger = logging.getLogger(__name__)

def test_feature_extraction():
    """Test feature extraction on a single capacitor"""
    
    # Define paths
    data_path = Path("~/work/CapacitorElectricalStress/eda_kiro/data/raw/ES12.mat").expanduser()
    
    logger.info("Loading ES12 data...")
    data_loader = DataLoader()
    capacitor_data = data_loader.load_es12_dataset(data_path)
    
    # Test on first capacitor only
    cap_id = list(capacitor_data.keys())[0]
    cap_data = capacitor_data[cap_id]
    
    logger.info(f"Testing feature extraction on {cap_id} with {cap_data.total_cycles} cycles")
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor(
        include_advanced=MODEL_CONFIG["feature_extraction"]["include_advanced"],
        rolling_window=MODEL_CONFIG["feature_extraction"]["rolling_window"]
    )
    
    # Test first 10 cycles
    feature_lengths = []
    for i, cycle in enumerate(cap_data.cycles[:10]):
        # Get history for rolling features
        history_start = max(0, cycle.cycle_number - feature_extractor.rolling_window)
        history = [c for c in cap_data.cycles 
                  if history_start <= c.cycle_number < cycle.cycle_number]
        
        try:
            # Extract features
            features_dict = feature_extractor.extract_features(cycle, cap_id, history)
            features = np.array(list(features_dict.values()))
            feature_lengths.append(len(features))
            
            logger.info(f"Cycle {cycle.cycle_number}: {len(features)} features")
            if i == 0:
                logger.info(f"Feature names: {list(features_dict.keys())}")
            
        except Exception as e:
            logger.error(f"Failed to extract features for cycle {cycle.cycle_number}: {e}")
            feature_lengths.append(0)
    
    logger.info(f"Feature lengths: {feature_lengths}")
    logger.info(f"Unique lengths: {set(feature_lengths)}")
    
    if len(set(feature_lengths)) > 1:
        logger.error("Inconsistent feature lengths detected!")
        return False
    else:
        logger.info("Feature lengths are consistent!")
        return True

if __name__ == "__main__":
    success = test_feature_extraction()
    sys.exit(0 if success else 1)