#!/usr/bin/env python3
"""
Debug feature extraction issue
"""

import sys
from pathlib import Path
import logging
import numpy as np
import traceback

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from true_rul.data_loader import DataLoader
from true_rul.feature_extractor import FeatureExtractor
from true_rul.config import ES12_CONFIG, MODEL_CONFIG, setup_logging

# Set up logging
setup_logging(log_file="debug_feature_extraction.log", level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_feature_extraction():
    """Debug feature extraction on a single cycle"""
    
    # Define paths
    data_path = Path("~/work/CapacitorElectricalStress/eda_kiro/data/raw/ES12.mat").expanduser()
    
    logger.info("Loading ES12 data...")
    data_loader = DataLoader()
    capacitor_data = data_loader.load_es12_dataset(data_path)
    
    # Test on first capacitor only
    cap_id = list(capacitor_data.keys())[0]
    cap_data = capacitor_data[cap_id]
    
    logger.info(f"Testing feature extraction on {cap_id}")
    
    # Initialize feature extractor
    feature_extractor = FeatureExtractor(
        include_advanced=MODEL_CONFIG["feature_extraction"]["include_advanced"],
        rolling_window=MODEL_CONFIG["feature_extraction"]["rolling_window"]
    )
    
    # Test cycle 6 specifically (where the error occurs)
    cycle = cap_data.cycles[5]  # 0-based indexing, so cycle 6
    history = cap_data.cycles[:5]  # cycles 1-5
    
    logger.info(f"Testing cycle {cycle.cycle_number} with {len(history)} history cycles")
    
    try:
        # Extract features step by step
        logger.info("Extracting responsiveness features...")
        response_features = feature_extractor.extract_responsiveness_features(cycle, cap_id)
        logger.info(f"Responsiveness features: {len(response_features)}")
        
        logger.info("Extracting statistical features...")
        stat_features = feature_extractor.extract_statistical_features(cycle)
        logger.info(f"Statistical features: {len(stat_features)}")
        
        logger.info("Extracting frequency features...")
        freq_features = feature_extractor.extract_frequency_features(cycle)
        logger.info(f"Frequency features: {len(freq_features)}")
        
        logger.info("Extracting trend features...")
        trend_features = feature_extractor.extract_trend_features(cycle)
        logger.info(f"Trend features: {len(trend_features)}")
        
        logger.info("Extracting rolling features...")
        rolling_features = feature_extractor.extract_rolling_features(cycle, history)
        logger.info(f"Rolling features: {len(rolling_features)}")
        
        # Combine all features
        all_features = {}
        all_features.update(response_features)
        all_features.update(stat_features)
        all_features.update(freq_features)
        all_features.update(trend_features)
        all_features.update(rolling_features)
        
        logger.info(f"Total features: {len(all_features)}")
        logger.info("SUCCESS: Feature extraction completed")
        
        return True
        
    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = debug_feature_extraction()
    sys.exit(0 if success else 1)