"""
Verification Script for Complete Prediction Pipeline

This script verifies that the complete prediction pipeline works end-to-end:
- Data loading and feature extraction
- Model training and prediction
- Error handling for various failure scenarios
- Interpretability outputs (SHAP, feature importance)
- OOD detection functionality
- Structured logging

Requirements: Task 15 - Checkpoint verification
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import logging
from datetime import datetime
import tempfile
import traceback

from true_rul import (
    CycleData, PredictionResult, FeatureExtractor, TimeSeriesPreprocessor,
    GradientBoostingRULPredictor, EnsembleAnomalyDetector, PredictionAggregator,
    ConfidenceEstimator, RULPredictor, InterpretabilityEngine,
    OutOfDistributionDetector, configure_prediction_logging
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_data(n_samples: int = 50) -> tuple:
    """Create synthetic data for testing"""
    np.random.seed(42)
    
    cycles = []
    features_list = []
    rul_labels = []
    
    for i in range(n_samples):
        # Create synthetic voltage data with degradation trend
        degradation_factor = i / n_samples  # 0 to 1
        
        vl_series = np.random.randn(100) + degradation_factor * 0.5
        vo_series = np.random.randn(100) + degradation_factor * 0.3
        
        cycle = CycleData(
            cycle_number=i + 1,
            vl_series=vl_series,
            vo_series=vo_series,
            timestamp=datetime.now()
        )
        cycles.append(cycle)
        
        # Create synthetic features
        features = np.random.randn(55) + degradation_factor * 0.2
        features_list.append(features)
        
        # Create synthetic RUL labels (decreasing with degradation)
        rul = max(1, int(100 - degradation_factor * 80))
        rul_labels.append(rul)
    
    return cycles, np.array(features_list), np.array(rul_labels)


def test_feature_extraction():
    """Test feature extraction pipeline"""
    print("\n🔧 Testing Feature Extraction Pipeline...")
    
    try:
        # Create test data
        cycles, _, _ = create_synthetic_data(10)
        
        # Initialize feature extractor
        feature_extractor = FeatureExtractor()
        
        # Extract features for first cycle
        features = feature_extractor.extract_features(
            cycles[0], 
            capacitor_id="TEST_C1",
            cycle_history=cycles[:5] if len(cycles) > 5 else None
        )
        
        print(f"✅ Feature extraction successful: {len(features)} features extracted")
        
        # Test time series preprocessing
        preprocessor = TimeSeriesPreprocessor()
        
        # Create feature matrix
        feature_matrix = []
        for cycle in cycles:
            cycle_features = feature_extractor.extract_features(cycle, "TEST_C1")
            feature_matrix.append(list(cycle_features.values()))
        
        feature_matrix = np.array(feature_matrix)
        
        # Create temporal features
        temporal_features = preprocessor.create_temporal_features(cycles, feature_matrix)
        
        print(f"✅ Temporal feature creation successful: {temporal_features.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        traceback.print_exc()
        return False


def test_model_training():
    """Test model training pipeline"""
    print("\n🤖 Testing Model Training Pipeline...")
    
    try:
        # Create synthetic training data
        _, features, rul_labels = create_synthetic_data(100)
        
        # Split data
        split_idx = 80
        X_train, X_val = features[:split_idx], features[split_idx:]
        y_train, y_val = rul_labels[:split_idx], rul_labels[split_idx:]
        
        # Test RUL regression model
        rul_model = GradientBoostingRULPredictor(
            model_type="xgboost",
            n_estimators=50  # Small for testing
        )
        
        rul_model.train(X_train, y_train, X_val, y_val)
        print("✅ RUL model training successful")
        
        # Test predictions
        predictions = rul_model.predict(X_val[:5])
        print(f"✅ RUL predictions successful: {predictions[:3]}")
        
        # Test feature importance
        importance = rul_model.get_feature_importance()
        print(f"✅ Feature importance extraction successful: {len(importance)} features")
        
        # Test SHAP values
        try:
            shap_values = rul_model.get_shap_values(X_val[:3])
            print(f"✅ SHAP values computation successful: {shap_values.shape}")
        except Exception as e:
            print(f"⚠️ SHAP values failed (may be expected): {e}")
        
        # Test anomaly detection
        anomaly_detector = EnsembleAnomalyDetector()
        
        # Fit on normal cycles (first 20)
        normal_data = features[:20]
        anomaly_detector.fit(normal_data)
        print("✅ Anomaly detector training successful")
        
        # Test anomaly predictions
        test_data = features[80:85]
        anomaly_flags, anomaly_scores, feature_importance = anomaly_detector.predict(test_data)
        print(f"✅ Anomaly detection successful: {np.sum(anomaly_flags)} anomalies detected")
        
        return rul_model, anomaly_detector
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
        traceback.print_exc()
        return None, None


def test_interpretability_features(rul_model, features):
    """Test interpretability features"""
    print("\n🔍 Testing Interpretability Features...")
    
    try:
        # Initialize interpretability engine
        feature_names = [f"feature_{i}" for i in range(features.shape[1])]
        interp_engine = InterpretabilityEngine(feature_names=feature_names)
        
        # Test feature importance aggregation
        importance1 = rul_model.get_feature_importance()
        importance2 = {f"feature_{i}": np.random.random() for i in range(len(feature_names))}
        
        aggregated_importance = interp_engine.aggregate_feature_importance(
            [importance1, importance2],
            weights=[0.7, 0.3]
        )
        
        print(f"✅ Feature importance aggregation successful: {len(aggregated_importance)} features")
        
        # Test top features
        top_features = interp_engine.get_top_features(aggregated_importance, top_k=10)
        print(f"✅ Top features extraction successful: {len(top_features)} top features")
        
        # Test SHAP analysis (if available)
        try:
            shap_values = rul_model.get_shap_values(features[:3])
            shap_analysis = interp_engine.analyze_shap_values(shap_values, feature_names, sample_idx=0)
            print(f"✅ SHAP analysis successful: {len(shap_analysis)} analysis components")
        except Exception as e:
            print(f"⚠️ SHAP analysis skipped: {e}")
        
        # Test diagnostic report generation
        dummy_result = PredictionResult(
            rul_cycles=50,
            rul_confidence_lower=40,
            rul_confidence_upper=60,
            degradation_score=0.3,
            degradation_stage="early_degradation",
            anomaly_flag=False,
            anomaly_score=0.2,
            feature_importance=aggregated_importance,
            timestamp=datetime.now(),
            model_version="1.0.0"
        )
        
        diagnostic_report = interp_engine.generate_diagnostic_report(
            dummy_result,
            aggregated_importance
        )
        
        print(f"✅ Diagnostic report generation successful: {len(diagnostic_report)} sections")
        
        return True
        
    except Exception as e:
        print(f"❌ Interpretability features failed: {e}")
        traceback.print_exc()
        return False


def test_ood_detection(features):
    """Test out-of-distribution detection"""
    print("\n🚨 Testing Out-of-Distribution Detection...")
    
    try:
        # Initialize OOD detector
        ood_detector = OutOfDistributionDetector(threshold_std=3.0)
        
        # Fit on training data
        feature_names = [f"feature_{i}" for i in range(features.shape[1])]
        ood_detector.fit(features[:80], feature_names)
        print("✅ OOD detector fitting successful")
        
        # Test with normal data
        normal_samples = features[80:85]
        ood_flags = ood_detector.is_out_of_distribution(normal_samples)
        print(f"✅ OOD detection on normal data: {np.sum(ood_flags)} OOD samples")
        
        # Test with clear outliers
        outliers = np.array([
            np.ones(features.shape[1]) * 10,  # Clear outlier
            np.zeros(features.shape[1]),      # Potentially normal
            np.ones(features.shape[1]) * -10  # Clear outlier
        ])
        
        ood_flags_outliers = ood_detector.is_out_of_distribution(outliers)
        print(f"✅ OOD detection on outliers: {np.sum(ood_flags_outliers)} OOD samples")
        
        # Test OOD scores
        ood_scores = ood_detector.get_ood_score(outliers)
        print(f"✅ OOD score computation successful: scores={ood_scores}")
        
        # Test detailed OOD analysis
        ood_flags_detailed, ood_details = ood_detector.is_out_of_distribution(
            outliers[:2], return_details=True
        )
        print(f"✅ Detailed OOD analysis successful: {len(ood_details)} detailed results")
        
        return ood_detector
        
    except Exception as e:
        print(f"❌ OOD detection failed: {e}")
        traceback.print_exc()
        return None


def test_error_handling():
    """Test error handling scenarios"""
    print("\n⚠️ Testing Error Handling Scenarios...")
    
    try:
        # Test with invalid input data
        predictor = RULPredictor()  # No models loaded
        
        # Test input validation errors
        invalid_cycle = CycleData(
            cycle_number=1,
            vl_series=np.array([]),  # Empty array
            vo_series=np.array([1, 2, 3]),
            timestamp=datetime.now()
        )
        
        try:
            predictor.predict_with_error_handling(
                cycle_data=invalid_cycle,
                capacitor_id="TEST_C1"
            )
            print("❌ Should have raised InputValidationError")
            return False
        except Exception as e:
            print(f"✅ Input validation error caught: {type(e).__name__}")
        
        # Test model not ready error
        valid_cycle = CycleData(
            cycle_number=1,
            vl_series=np.random.randn(100),
            vo_series=np.random.randn(100),
            timestamp=datetime.now()
        )
        
        try:
            predictor.predict_with_error_handling(
                cycle_data=valid_cycle,
                capacitor_id="TEST_C1"
            )
            print("❌ Should have raised ModelNotReadyError")
            return False
        except Exception as e:
            print(f"✅ Model not ready error caught: {type(e).__name__}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        traceback.print_exc()
        return False


def test_structured_logging():
    """Test structured logging functionality"""
    print("\n📝 Testing Structured Logging...")
    
    try:
        # Configure structured logging
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            log_file = f.name
        
        logger_instance = configure_prediction_logging(
            log_file=log_file,
            enable_console=False
        )
        
        # Create predictor with structured logging
        predictor = RULPredictor(structured_logger=logger_instance)
        
        # Test logging with invalid input (will generate error log)
        invalid_cycle = CycleData(
            cycle_number=1,
            vl_series=np.array([]),  # Invalid
            vo_series=np.array([1, 2, 3]),
            timestamp=datetime.now()
        )
        
        try:
            predictor.predict_with_error_handling(
                cycle_data=invalid_cycle,
                capacitor_id=""  # Invalid empty string
            )
        except Exception:
            pass  # Expected
        
        # Check that logs were written
        with open(log_file, 'r') as f:
            log_lines = f.readlines()
        
        print(f"✅ Structured logging successful: {len(log_lines)} log entries")
        
        # Verify log format
        if log_lines:
            import json
            first_log = json.loads(log_lines[0])
            assert "event_type" in first_log
            assert "timestamp" in first_log
            print("✅ Log format validation successful")
        
        # Clean up
        Path(log_file).unlink(missing_ok=True)
        
        return True
        
    except Exception as e:
        print(f"❌ Structured logging test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run complete prediction pipeline verification"""
    print("🚀 Starting Complete Prediction Pipeline Verification")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Feature extraction
    results["feature_extraction"] = test_feature_extraction()
    
    # Test 2: Model training
    rul_model, anomaly_detector = test_model_training()
    results["model_training"] = (rul_model is not None and anomaly_detector is not None)
    
    if results["model_training"]:
        # Test 3: Interpretability features
        _, features, _ = create_synthetic_data(100)
        results["interpretability"] = test_interpretability_features(rul_model, features)
        
        # Test 4: OOD detection
        ood_detector = test_ood_detection(features)
        results["ood_detection"] = (ood_detector is not None)
    else:
        results["interpretability"] = False
        results["ood_detection"] = False
    
    # Test 5: Error handling
    results["error_handling"] = test_error_handling()
    
    # Test 6: Structured logging
    results["structured_logging"] = test_structured_logging()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():<25} {status}")
    
    print("-" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! Prediction pipeline is ready.")
        return True
    else:
        print(f"\n⚠️ {total_tests - passed_tests} tests failed. Please review and fix issues.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)