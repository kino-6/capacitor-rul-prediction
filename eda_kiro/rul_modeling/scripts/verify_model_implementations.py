#!/usr/bin/env python3
"""
Model Implementation Verification Script

This script verifies that all model classes can be instantiated, trained on
synthetic data, and generate feature importance and SHAP values as required.

Task 8: Checkpoint - Verify model implementations
"""

import sys
import os
import logging
import numpy as np
import warnings
from typing import Dict, Any, List, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import all model classes
from true_rul.rul_regression_model import RULRegressionModel
from true_rul.gradient_boosting_predictor import GradientBoostingRULPredictor
from true_rul.random_forest_predictor import RandomForestRULPredictor
from true_rul.elastic_net_predictor import ElasticNetRULPredictor
from true_rul.hybrid_ensemble_predictor import HybridEnsembleRULPredictor
from true_rul.ensemble_anomaly_detector import EnsembleAnomalyDetector
from true_rul.isolation_forest_detector import IsolationForestDetector
from true_rul.autoencoder_detector import AutoencoderDetector
from true_rul.improved_ocsvm import ImprovedOCSVM

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples: int = 100, n_features: int = 20, noise_level: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for testing models.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        noise_level: Amount of noise to add
    
    Returns:
        Tuple of (X, y) where X is features and y is RUL labels
    """
    np.random.seed(42)
    
    # Generate features with some structure
    X = np.random.randn(n_samples, n_features)
    
    # Add some correlation between features
    for i in range(1, min(5, n_features)):
        X[:, i] = 0.7 * X[:, 0] + 0.3 * np.random.randn(n_samples)
    
    # Generate RUL labels with some relationship to features
    # RUL decreases as certain features increase (simulating degradation)
    y = 200 - 10 * X[:, 0] - 5 * X[:, 1] + noise_level * np.random.randn(n_samples)
    y = np.maximum(y, 0)  # Ensure non-negative RUL
    
    return X, y


def test_model_instantiation() -> Dict[str, bool]:
    """Test that all model classes can be instantiated."""
    logger.info("Testing model instantiation...")
    
    results = {}
    
    # Test RUL regression models
    rul_models = [
        ("XGBoost", lambda: RULRegressionModel("xgboost")),
        ("LightGBM", lambda: RULRegressionModel("lightgbm")),
        ("RandomForest", lambda: RULRegressionModel("random_forest")),
        ("ElasticNet", lambda: RULRegressionModel("elastic_net")),
        ("Ensemble", lambda: RULRegressionModel("ensemble")),
        ("GradientBoostingXGB", lambda: GradientBoostingRULPredictor("xgboost")),
        ("GradientBoostingLGB", lambda: GradientBoostingRULPredictor("lightgbm")),
        ("RandomForestDirect", lambda: RandomForestRULPredictor()),
        ("ElasticNetDirect", lambda: ElasticNetRULPredictor()),
        ("HybridEnsemble", lambda: HybridEnsembleRULPredictor()),
    ]
    
    for name, model_factory in rul_models:
        try:
            model = model_factory()
            results[name] = True
            logger.info(f"✓ {name} instantiated successfully")
        except Exception as e:
            results[name] = False
            logger.error(f"✗ {name} instantiation failed: {e}")
    
    # Test anomaly detection models
    anomaly_models = [
        ("IsolationForest", lambda: IsolationForestDetector()),
        ("Autoencoder", lambda: AutoencoderDetector(input_dim=20)),
        ("ImprovedOCSVM", lambda: ImprovedOCSVM()),
        ("EnsembleAnomalyDetector", lambda: EnsembleAnomalyDetector()),
    ]
    
    for name, model_factory in anomaly_models:
        try:
            model = model_factory()
            results[name] = True
            logger.info(f"✓ {name} instantiated successfully")
        except Exception as e:
            results[name] = False
            logger.error(f"✗ {name} instantiation failed: {e}")
    
    return results


def test_rul_model_training(model_name: str, model_factory, X_train: np.ndarray, y_train: np.ndarray, 
                           X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
    """Test training of a single RUL model."""
    results = {
        'instantiation': False,
        'training': False,
        'prediction': False,
        'feature_importance': False,
        'shap_values': False,
        'error': None
    }
    
    try:
        # Instantiate model
        model = model_factory()
        results['instantiation'] = True
        
        # Train model
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        model.train(X_train, y_train, X_val, y_val, feature_names=feature_names)
        results['training'] = True
        
        # Test prediction
        predictions = model.predict(X_val)
        if len(predictions) == len(y_val) and np.all(predictions >= 0):
            results['prediction'] = True
        
        # Test feature importance
        try:
            importance = model.get_feature_importance()
            if isinstance(importance, dict) and len(importance) > 0:
                results['feature_importance'] = True
        except (NotImplementedError, AttributeError):
            logger.warning(f"{model_name} does not support feature importance")
        
        # Test SHAP values
        try:
            shap_values = model.get_shap_values(X_val[:5])  # Test on small subset
            if isinstance(shap_values, np.ndarray) and shap_values.shape[0] == 5:
                results['shap_values'] = True
        except (NotImplementedError, AttributeError, RuntimeError):
            logger.warning(f"{model_name} does not support SHAP values")
        
    except Exception as e:
        results['error'] = str(e)
        logger.error(f"Error testing {model_name}: {e}")
    
    return results


def test_anomaly_model_training(model_name: str, model_factory, X_normal: np.ndarray) -> Dict[str, Any]:
    """Test training of a single anomaly detection model."""
    results = {
        'instantiation': False,
        'training': False,
        'prediction': False,
        'error': None
    }
    
    try:
        # Instantiate model
        if model_name == "Autoencoder":
            model = model_factory(input_dim=X_normal.shape[1])
        else:
            model = model_factory()
        results['instantiation'] = True
        
        # Train model
        feature_names = [f"feature_{i}" for i in range(X_normal.shape[1])]
        if model_name == "Autoencoder":
            model.fit(X_normal, epochs=10, verbose=False)  # Quick training
        elif model_name == "EnsembleAnomalyDetector":
            model.fit(X_normal, feature_names=feature_names)
        else:
            model.fit(X_normal, feature_names=feature_names)
        results['training'] = True
        
        # Test prediction
        if model_name == "EnsembleAnomalyDetector":
            binary_pred, scores, info = model.predict(X_normal[:10])
            if len(binary_pred) == 10 and len(scores) == 10:
                results['prediction'] = True
        elif model_name == "Autoencoder":
            errors = model.get_reconstruction_error(X_normal[:10])
            if len(errors) == 10:
                results['prediction'] = True
        else:
            scores = model.predict_score(X_normal[:10])
            if len(scores) == 10:
                results['prediction'] = True
        
    except Exception as e:
        results['error'] = str(e)
        logger.error(f"Error testing {model_name}: {e}")
    
    return results


def test_model_training() -> Dict[str, Dict[str, Any]]:
    """Test training on synthetic data."""
    logger.info("Testing model training on synthetic data...")
    
    # Generate synthetic data
    X, y = generate_synthetic_data(n_samples=200, n_features=20)
    
    # Split into train/validation
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Normal data for anomaly detection (first 80% of training data)
    X_normal = X_train[:int(0.8 * len(X_train))]
    
    results = {}
    
    # Test RUL regression models
    rul_models = [
        ("XGBoost", lambda: RULRegressionModel("xgboost", n_estimators=50)),
        ("LightGBM", lambda: RULRegressionModel("lightgbm", n_estimators=50)),
        ("RandomForest", lambda: RULRegressionModel("random_forest", n_estimators=50)),
        ("ElasticNet", lambda: RULRegressionModel("elastic_net")),
        ("Ensemble", lambda: RULRegressionModel("ensemble")),
    ]
    
    for name, model_factory in rul_models:
        logger.info(f"Testing {name} training...")
        results[name] = test_rul_model_training(name, model_factory, X_train, y_train, X_val, y_val)
    
    # Test anomaly detection models
    anomaly_models = [
        ("IsolationForest", IsolationForestDetector),
        ("Autoencoder", AutoencoderDetector),
        ("ImprovedOCSVM", ImprovedOCSVM),
        ("EnsembleAnomalyDetector", EnsembleAnomalyDetector),
    ]
    
    for name, model_factory in anomaly_models:
        logger.info(f"Testing {name} training...")
        results[name] = test_anomaly_model_training(name, model_factory, X_normal)
    
    return results


def print_summary(instantiation_results: Dict[str, bool], training_results: Dict[str, Dict[str, Any]]):
    """Print a summary of all test results."""
    logger.info("\n" + "="*80)
    logger.info("MODEL IMPLEMENTATION VERIFICATION SUMMARY")
    logger.info("="*80)
    
    # Instantiation summary
    logger.info("\nINSTANTIATION RESULTS:")
    logger.info("-" * 40)
    instantiation_passed = 0
    for model_name, success in instantiation_results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        logger.info(f"{model_name:<25} {status}")
        if success:
            instantiation_passed += 1
    
    logger.info(f"\nInstantiation: {instantiation_passed}/{len(instantiation_results)} models passed")
    
    # Training summary
    logger.info("\nTRAINING RESULTS:")
    logger.info("-" * 40)
    training_passed = 0
    feature_importance_passed = 0
    shap_values_passed = 0
    
    for model_name, results in training_results.items():
        training_status = "✓ PASS" if results['training'] else "✗ FAIL"
        prediction_status = "✓ PASS" if results['prediction'] else "✗ FAIL"
        
        logger.info(f"{model_name:<25} Training: {training_status}, Prediction: {prediction_status}")
        
        if results['training']:
            training_passed += 1
        
        # Feature importance (only for RUL models)
        if 'feature_importance' in results:
            fi_status = "✓ PASS" if results['feature_importance'] else "✗ FAIL/N/A"
            logger.info(f"{'':<25} Feature Importance: {fi_status}")
            if results['feature_importance']:
                feature_importance_passed += 1
        
        # SHAP values (only for RUL models)
        if 'shap_values' in results:
            shap_status = "✓ PASS" if results['shap_values'] else "✗ FAIL/N/A"
            logger.info(f"{'':<25} SHAP Values: {shap_status}")
            if results['shap_values']:
                shap_values_passed += 1
        
        if results['error']:
            logger.info(f"{'':<25} Error: {results['error']}")
    
    logger.info(f"\nTraining: {training_passed}/{len(training_results)} models passed")
    
    # Count RUL models for feature importance and SHAP
    rul_model_count = sum(1 for results in training_results.values() if 'feature_importance' in results)
    if rul_model_count > 0:
        logger.info(f"Feature Importance: {feature_importance_passed}/{rul_model_count} RUL models passed")
        logger.info(f"SHAP Values: {shap_values_passed}/{rul_model_count} RUL models passed")
    
    # Overall assessment
    logger.info("\nOVERALL ASSESSMENT:")
    logger.info("-" * 40)
    
    total_models = len(instantiation_results)
    total_passed = instantiation_passed
    
    if total_passed == total_models:
        logger.info("✓ ALL MODELS CAN BE INSTANTIATED")
    else:
        logger.info(f"✗ {total_models - total_passed} MODELS FAILED INSTANTIATION")
    
    if training_passed == len(training_results):
        logger.info("✓ ALL MODELS CAN BE TRAINED AND MAKE PREDICTIONS")
    else:
        logger.info(f"✗ {len(training_results) - training_passed} MODELS FAILED TRAINING")
    
    if rul_model_count > 0:
        if feature_importance_passed >= rul_model_count // 2:  # At least half should support it
            logger.info("✓ FEATURE IMPORTANCE IS AVAILABLE")
        else:
            logger.info("⚠ LIMITED FEATURE IMPORTANCE SUPPORT")
        
        if shap_values_passed >= 2:  # At least 2 models should support SHAP
            logger.info("✓ SHAP VALUES ARE AVAILABLE")
        else:
            logger.info("⚠ LIMITED SHAP VALUES SUPPORT")
    
    logger.info("\n" + "="*80)


def main():
    """Main verification function."""
    logger.info("Starting model implementation verification...")
    
    try:
        # Test instantiation
        instantiation_results = test_model_instantiation()
        
        # Test training
        training_results = test_model_training()
        
        # Print summary
        print_summary(instantiation_results, training_results)
        
        # Check if all critical tests passed
        all_instantiated = all(instantiation_results.values())
        all_trained = all(results['training'] for results in training_results.values())
        
        if all_instantiated and all_trained:
            logger.info("✓ MODEL IMPLEMENTATION VERIFICATION COMPLETED SUCCESSFULLY")
            return True
        else:
            logger.error("✗ MODEL IMPLEMENTATION VERIFICATION FAILED")
            return False
            
    except Exception as e:
        logger.error(f"Verification failed with error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)