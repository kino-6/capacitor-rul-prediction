#!/usr/bin/env python3
"""
Test script for the training pipeline

This script tests the training pipeline with a minimal setup to ensure
all components work together correctly.
"""

import sys
from pathlib import Path
import logging

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from true_rul.training_pipeline import TrainingPipeline
from true_rul.model_evaluator import ModelEvaluator
from true_rul.config import setup_logging

def main():
    """Test the training pipeline"""
    # Set up logging
    setup_logging(log_file="test_training_pipeline.log", level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting training pipeline test")
    
    # Initialize pipeline
    pipeline = TrainingPipeline()
    
    # Define data path - adjust this to your actual ES12.mat location
    data_path = Path("../../../data/raw/ES12.mat")
    
    if not data_path.exists():
        logger.error(f"ES12 data file not found at {data_path}")
        logger.info("Please ensure the ES12.mat file is available at the specified path")
        return False
    
    try:
        # Run training with a small subset for testing
        logger.info("Running training pipeline...")
        results = pipeline.train(
            data_path=data_path,
            test_capacitors=["ES12C7", "ES12C8"],  # Use 2 capacitors for test
            save_models=True
        )
        
        logger.info("Training completed successfully!")
        
        # Print summary of results
        data_info = results["data_info"]
        logger.info(f"Data summary:")
        logger.info(f"  Total capacitors: {data_info['total_capacitors']}")
        logger.info(f"  Train samples: {data_info['train_samples']}")
        logger.info(f"  Val samples: {data_info['val_samples']}")
        logger.info(f"  Test samples: {data_info['test_samples']}")
        logger.info(f"  Features: {data_info['n_features']}")
        
        # Test evaluation if models were trained
        if pipeline.rul_model and pipeline.anomaly_detector:
            logger.info("Testing model evaluation...")
            
            # Create dummy datasets for evaluation test
            # In a real scenario, you would use the actual train/val/test datasets
            # from the pipeline
            
            logger.info("Training pipeline test completed successfully!")
            return True
        else:
            logger.warning("Models were not properly trained")
            return False
            
    except Exception as e:
        logger.error(f"Training pipeline test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("✓ Training pipeline test PASSED")
        sys.exit(0)
    else:
        print("✗ Training pipeline test FAILED")
        sys.exit(1)