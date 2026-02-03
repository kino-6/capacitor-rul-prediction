#!/usr/bin/env python3
"""
Test script for advanced ensemble techniques

This script tests the advanced ensemble methods including stacking,
dynamic weighting, boosting anomaly detection, and mixture of experts.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression, make_classification
from sklearn.metrics import mean_squared_error, accuracy_score

from true_rul.advanced_ensemble_techniques import (
    StackingEnsembleRULPredictor,
    DynamicEnsembleWeighting,
    BoostingAnomalyDetector,
    MixtureOfExpertsRUL
)
from true_rul.rul_regression_model import RULRegressionModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_test_data():
    """Generate test data for ensemble techniques"""
    logger.info("Generating test data...")
    
    # RUL regression data
    X_rul, y_rul = make_regression(
        n_samples=300,
        n_features=15,
        noise=0.1,
        random_state=42
    )
    
    # Ensure RUL values are positive
    y_rul = np.abs(y_rul) + 1
    
    # Split RUL data
    X_train, X_test, y_train, y_test = train_test_split(
        X_rul, y_rul, test_size=0.3, random_state=42
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42
    )
    
    # Anomaly detection data
    X_anom, y_anom = make_classification(
        n_samples=200,
        n_features=15,
        n_classes=2,
        n_redundant=0,
        random_state=42
    )
    
    X_anom_train, X_anom_test, y_anom_train, y_anom_test = train_test_split(
        X_anom, y_anom, test_size=0.3, random_state=42
    )
    
    return {
        'rul': {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        },
        'anomaly': {
            'X_train': X_anom_train,
            'y_train': y_anom_train,
            'X_test': X_anom_test,
            'y_test': y_anom_test
        }
    }


def test_stacking_ensemble():
    """Test stacking ensemble RUL predictor"""
    logger.info("Testing stacking ensemble RUL predictor...")
    
    data = generate_test_data()
    rul_data = data['rul']
    
    # Create base models
    base_models = [
        RULRegressionModel(model_type="xgboost"),
        RULRegressionModel(model_type="random_forest")
    ]
    
    # Initialize stacking ensemble
    stacking_ensemble = StackingEnsembleRULPredictor(
        base_models=base_models,
        cv_folds=3  # Reduced for testing
    )
    
    # Train ensemble
    stacking_ensemble.fit(
        rul_data['X_train'],
        rul_data['y_train'],
        rul_data['X_val'],
        rul_data['y_val']
    )
    
    # Make predictions
    y_pred = stacking_ensemble.predict(rul_data['X_test'])
    
    # Evaluate
    rmse = np.sqrt(mean_squared_error(rul_data['y_test'], y_pred))
    
    logger.info(f"Stacking ensemble results:")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  Prediction shape: {y_pred.shape}")
    logger.info(f"  Prediction range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    
    # Test feature importance
    try:
        importance = stacking_ensemble.get_feature_importance()
        logger.info(f"  Feature importance keys: {len(importance)}")
    except Exception as e:
        logger.warning(f"  Feature importance failed: {e}")
    
    # Test model info
    info = stacking_ensemble.get_model_info()
    logger.info(f"  Model info: {info['ensemble_type']}, {info['n_base_models']} base models")
    
    return stacking_ensemble


def test_dynamic_ensemble_weighting():
    """Test dynamic ensemble weighting"""
    logger.info("Testing dynamic ensemble weighting...")
    
    data = generate_test_data()
    rul_data = data['rul']
    
    # Create base models
    base_models = [
        RULRegressionModel(model_type="xgboost"),
        RULRegressionModel(model_type="random_forest")
    ]
    
    # Test cluster-based weighting
    dynamic_ensemble = DynamicEnsembleWeighting(
        base_models=base_models,
        n_clusters=3,
        weighting_strategy="cluster_based"
    )
    
    # Train ensemble
    dynamic_ensemble.fit(
        rul_data['X_train'],
        rul_data['y_train'],
        rul_data['X_val'],
        rul_data['y_val']
    )
    
    # Make predictions
    y_pred = dynamic_ensemble.predict(rul_data['X_test'])
    
    # Evaluate
    rmse = np.sqrt(mean_squared_error(rul_data['y_test'], y_pred))
    
    logger.info(f"Dynamic ensemble (cluster-based) results:")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  Prediction shape: {y_pred.shape}")
    
    # Test model info
    info = dynamic_ensemble.get_model_info()
    logger.info(f"  Model info: {info['ensemble_type']}, strategy: {info['weighting_strategy']}")
    
    # Test regression-based weighting
    dynamic_ensemble_reg = DynamicEnsembleWeighting(
        base_models=[
            RULRegressionModel(model_type="xgboost"),
            RULRegressionModel(model_type="random_forest")
        ],
        weighting_strategy="regression_based"
    )
    
    dynamic_ensemble_reg.fit(
        rul_data['X_train'],
        rul_data['y_train'],
        rul_data['X_val'],
        rul_data['y_val']
    )
    
    y_pred_reg = dynamic_ensemble_reg.predict(rul_data['X_test'])
    rmse_reg = np.sqrt(mean_squared_error(rul_data['y_test'], y_pred_reg))
    
    logger.info(f"Dynamic ensemble (regression-based) results:")
    logger.info(f"  RMSE: {rmse_reg:.4f}")
    
    return dynamic_ensemble


def test_boosting_anomaly_detector():
    """Test boosting anomaly detector"""
    logger.info("Testing boosting anomaly detector...")
    
    data = generate_test_data()
    anom_data = data['anomaly']
    
    # Separate normal and anomalous samples
    normal_mask = anom_data['y_train'] == 0
    anomaly_mask = anom_data['y_train'] == 1
    
    X_normal = anom_data['X_train'][normal_mask]
    X_anomaly = anom_data['X_train'][anomaly_mask]
    
    # Initialize boosting detector
    boosting_detector = BoostingAnomalyDetector(
        n_estimators=5,  # Reduced for testing
        learning_rate=1.0
    )
    
    # Train detector
    boosting_detector.fit(X_normal, X_anomaly)
    
    # Make predictions
    y_pred = boosting_detector.predict(anom_data['X_test'])
    y_proba = boosting_detector.predict_proba(anom_data['X_test'])
    
    # Evaluate
    accuracy = accuracy_score(anom_data['y_test'], y_pred)
    
    logger.info(f"Boosting anomaly detector results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Prediction shape: {y_pred.shape}")
    logger.info(f"  Probability shape: {y_proba.shape}")
    logger.info(f"  Anomaly rate: {np.mean(y_pred):.2%}")
    
    # Test model info
    info = boosting_detector.get_model_info()
    logger.info(f"  Model info: {info['detector_type']}, {info['n_trained_detectors']} detectors")
    
    return boosting_detector


def test_mixture_of_experts():
    """Test mixture of experts RUL predictor"""
    logger.info("Testing mixture of experts RUL predictor...")
    
    data = generate_test_data()
    rul_data = data['rul']
    
    # Create experts
    experts = [
        RULRegressionModel(model_type="xgboost"),
        RULRegressionModel(model_type="random_forest"),
        RULRegressionModel(model_type="elastic_net")
    ]
    
    # Test data-driven specialization
    moe_model = MixtureOfExpertsRUL(
        experts=experts,
        expert_specialization="data_driven"
    )
    
    # Train model
    moe_model.fit(
        rul_data['X_train'],
        rul_data['y_train'],
        rul_data['X_val'],
        rul_data['y_val']
    )
    
    # Make predictions
    y_pred = moe_model.predict(rul_data['X_test'])
    
    # Evaluate
    rmse = np.sqrt(mean_squared_error(rul_data['y_test'], y_pred))
    
    logger.info(f"Mixture of experts (data-driven) results:")
    logger.info(f"  RMSE: {rmse:.4f}")
    logger.info(f"  Prediction shape: {y_pred.shape}")
    
    # Test expert weights
    expert_weights = moe_model.get_expert_weights(rul_data['X_test'])
    logger.info(f"  Expert weights shape: {expert_weights.shape}")
    logger.info(f"  Average expert weights: {np.mean(expert_weights, axis=0)}")
    
    # Test model info
    info = moe_model.get_model_info()
    logger.info(f"  Model info: {info['model_type']}, {info['n_experts']} experts")
    
    # Test manual specialization
    moe_manual = MixtureOfExpertsRUL(
        experts=[
            RULRegressionModel(model_type="xgboost"),
            RULRegressionModel(model_type="random_forest")
        ],
        expert_specialization="manual"
    )
    
    moe_manual.fit(
        rul_data['X_train'],
        rul_data['y_train'],
        rul_data['X_val'],
        rul_data['y_val']
    )
    
    y_pred_manual = moe_manual.predict(rul_data['X_test'])
    rmse_manual = np.sqrt(mean_squared_error(rul_data['y_test'], y_pred_manual))
    
    logger.info(f"Mixture of experts (manual) results:")
    logger.info(f"  RMSE: {rmse_manual:.4f}")
    
    return moe_model


def main():
    """Run all advanced ensemble technique tests"""
    logger.info("Starting advanced ensemble techniques tests...")
    
    try:
        # Test each advanced ensemble technique
        logger.info("\n" + "="*50)
        test_stacking_ensemble()
        
        logger.info("\n" + "="*50)
        test_dynamic_ensemble_weighting()
        
        logger.info("\n" + "="*50)
        test_boosting_anomaly_detector()
        
        logger.info("\n" + "="*50)
        test_mixture_of_experts()
        
        logger.info("\n" + "="*50)
        logger.info("All advanced ensemble technique tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()