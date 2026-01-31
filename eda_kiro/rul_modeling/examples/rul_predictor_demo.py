"""
RUL Predictor Demo

This script demonstrates the usage of the RULPredictor class with comprehensive
error handling, input validation, and fallback mechanisms.
"""

import numpy as np
from datetime import datetime
import logging
from unittest.mock import Mock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from true_rul.data_structures import CycleData, PredictionResult
from true_rul.rul_predictor import RULPredictor
from true_rul.exceptions import InputValidationError, ModelNotReadyError


def create_mock_trained_models():
    """Create mock trained models for demonstration"""
    
    # Mock RUL regression model
    mock_rul_model = Mock()
    mock_rul_model.is_trained = True
    mock_rul_model.model_type = "xgboost"
    mock_rul_model.predict_with_confidence.return_value = (
        np.array([45.0]),  # prediction
        np.array([35.0]),  # lower bound
        np.array([55.0])   # upper bound
    )
    
    # Mock anomaly detector
    mock_anomaly_detector = Mock()
    mock_anomaly_detector.is_fitted = True
    mock_anomaly_detector.predict.return_value = (
        np.array([False]),  # anomaly flag
        np.array([0.3]),    # anomaly score
        {"feature_1": 0.4, "feature_2": 0.3, "feature_3": 0.3}  # feature importance
    )
    
    return mock_rul_model, mock_anomaly_detector


def demo_basic_prediction():
    """Demonstrate basic prediction functionality"""
    print("\n=== Basic Prediction Demo ===")
    
    # Create mock models
    rul_model, anomaly_detector = create_mock_trained_models()
    
    # Initialize predictor
    predictor = RULPredictor(
        rul_model=rul_model,
        anomaly_detector=anomaly_detector,
        prediction_timeout=2.0
    )
    
    print(f"Predictor status: {predictor}")
    print(f"Model status: {predictor.get_model_status()}")
    
    # Create sample cycle data
    cycle_data = CycleData(
        cycle_number=50,
        vl_series=np.random.normal(5.0, 0.1, 100),
        vo_series=np.random.normal(4.8, 0.1, 100),
        timestamp=datetime.now()
    )
    
    try:
        # Make prediction
        result = predictor.predict_with_error_handling(
            cycle_data=cycle_data,
            capacitor_id="ES12C1"
        )
        
        print(f"\nPrediction Result:")
        print(f"  RUL: {result.rul_cycles} cycles")
        print(f"  Confidence Interval: [{result.rul_confidence_lower}, {result.rul_confidence_upper}]")
        print(f"  Degradation Score: {result.degradation_score:.3f}")
        print(f"  Degradation Stage: {result.degradation_stage}")
        print(f"  Anomaly Flag: {result.anomaly_flag}")
        print(f"  Anomaly Score: {result.anomaly_score:.3f}")
        print(f"  Model Version: {result.model_version}")
        
    except Exception as e:
        print(f"Prediction failed: {e}")


def demo_input_validation():
    """Demonstrate input validation"""
    print("\n=== Input Validation Demo ===")
    
    predictor = RULPredictor()
    
    # Test various invalid inputs
    test_cases = [
        ("Invalid cycle data type", "not_cycle_data", "C1"),
        ("Empty capacitor ID", CycleData(1, np.array([1.0]), np.array([1.1])), ""),
        ("NaN in VL series", CycleData(1, np.array([1.0, np.nan]), np.array([1.1, 1.2])), "C1"),
    ]
    
    for description, cycle_data, capacitor_id in test_cases:
        try:
            if isinstance(cycle_data, str):
                predictor._validate_input(cycle_data, capacitor_id, None)
            else:
                predictor._validate_input(cycle_data, capacitor_id, None)
            print(f"  {description}: PASSED (unexpected)")
        except (InputValidationError, ValueError) as e:
            print(f"  {description}: FAILED as expected - {type(e).__name__}")
        except Exception as e:
            print(f"  {description}: FAILED with unexpected error - {e}")


def demo_error_handling():
    """Demonstrate error handling and fallback mechanisms"""
    print("\n=== Error Handling Demo ===")
    
    # Test with no models loaded
    predictor = RULPredictor()
    
    cycle_data = CycleData(
        cycle_number=1,
        vl_series=np.array([1.0, 2.0, 3.0]),
        vo_series=np.array([1.1, 2.1, 3.1])
    )
    
    try:
        result = predictor.predict_with_error_handling(cycle_data, "C1")
        print("  Model not ready test: FAILED (unexpected success)")
    except ModelNotReadyError as e:
        print(f"  Model not ready test: PASSED - {e.code}")
    
    # Test fallback mechanisms
    print("\n  Testing fallback mechanisms:")
    
    # Test basic feature extraction fallback
    basic_features = predictor._extract_basic_features_fallback(cycle_data)
    print(f"  Basic features fallback: {len(basic_features)} features extracted")
    
    # Test RUL prediction emergency fallback
    pred, lower, upper = predictor._predict_rul_with_fallback(np.array([[1, 2, 3]]))
    print(f"  RUL emergency fallback: {pred} cycles [{lower}, {upper}]")
    
    # Test anomaly detection fallback
    flag, score, importance = predictor._detect_anomaly_with_fallback(np.array([[1, 2, 3]]))
    print(f"  Anomaly fallback: flag={flag}, score={score:.3f}")


def demo_prediction_history():
    """Demonstrate prediction history management"""
    print("\n=== Prediction History Demo ===")
    
    predictor = RULPredictor()
    
    # Add some history
    for i in range(5):
        predictor._update_prediction_history("C1", 0.1 * i)
        predictor._update_prediction_history("C2", 0.2 * i)
    
    print(f"  History for C1: {predictor.prediction_history['C1']}")
    print(f"  History for C2: {predictor.prediction_history['C2']}")
    
    # Test history limit
    for i in range(25):
        predictor._update_prediction_history("C3", float(i))
    
    print(f"  History for C3 (limited to 20): {len(predictor.prediction_history['C3'])} entries")
    print(f"  First entry: {predictor.prediction_history['C3'][0]}")
    print(f"  Last entry: {predictor.prediction_history['C3'][-1]}")
    
    # Clear history
    predictor.clear_prediction_history("C1")
    print(f"  After clearing C1: {'C1' in predictor.prediction_history}")
    
    predictor.clear_prediction_history()
    print(f"  After clearing all: {len(predictor.prediction_history)} capacitors")


def demo_timeout_handling():
    """Demonstrate timeout handling"""
    print("\n=== Timeout Handling Demo ===")
    
    predictor = RULPredictor(prediction_timeout=0.001)  # Very short timeout
    
    print(f"  Initial timeout: {predictor.prediction_timeout}s")
    
    # Change timeout
    predictor.set_prediction_timeout(1.0)
    print(f"  Updated timeout: {predictor.prediction_timeout}s")
    
    # Test invalid timeout
    try:
        predictor.set_prediction_timeout(-1.0)
        print("  Invalid timeout test: FAILED (unexpected success)")
    except ValueError:
        print("  Invalid timeout test: PASSED")


def main():
    """Run all demonstrations"""
    print("RUL Predictor Comprehensive Demo")
    print("=" * 50)
    
    demo_basic_prediction()
    demo_input_validation()
    demo_error_handling()
    demo_prediction_history()
    demo_timeout_handling()
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")


if __name__ == "__main__":
    main()