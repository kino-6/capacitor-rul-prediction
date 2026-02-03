#!/usr/bin/env python3
"""
Test script for hyperparameter optimization functionality

This script tests the automated hyperparameter optimization system
including single-objective, multi-objective, and ensemble weight optimization.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

from true_rul.hyperparameter_optimizer import HyperparameterOptimizer, BayesianEnsembleOptimizer
from true_rul.rul_regression_model import RULRegressionModel
from true_rul.ensemble_anomaly_detector import EnsembleAnomalyDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_data():
    """Generate synthetic data for testing"""
    logger.info("Generating synthetic data...")
    
    # Generate RUL regression data
    X_rul, y_rul = make_regression(
        n_samples=1000,
        n_features=20,
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
    
    # Generate anomaly detection data
    # Normal data (first 80% of training data)
    n_normal = int(0.8 * len(X_train))
    X_normal = X_train[:n_normal]
    
    # Anomaly data (add noise to make anomalous)
    X_anomaly = X_train[n_normal:] + np.random.normal(0, 2, X_train[n_normal:].shape)
    
    # Create validation set for anomaly detection
    X_ad_val = np.vstack([X_val[:len(X_val)//2], X_anomaly])
    y_ad_val = np.hstack([
        np.zeros(len(X_val)//2),  # Normal
        np.ones(len(X_anomaly))   # Anomaly
    ])
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'X_normal': X_normal,
        'X_ad_val': X_ad_val,
        'y_ad_val': y_ad_val
    }


def test_rul_model_optimization():
    """Test RUL model hyperparameter optimization"""
    logger.info("Testing RUL model hyperparameter optimization...")
    
    data = generate_synthetic_data()
    
    # Initialize optimizer with reduced trials for testing
    optimizer = HyperparameterOptimizer(
        study_name="test_rul_optimization",
        n_trials=10,  # Reduced for testing
        timeout=300   # 5 minutes max
    )
    
    # Test XGBoost optimization
    results = optimizer.optimize_rul_model(
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_val=data['X_val'],
        y_val=data['y_val'],
        model_type="xgboost",
        scoring="rmse"
    )
    
    logger.info(f"XGBoost optimization results:")
    logger.info(f"  Best RMSE: {results['best_score']:.4f}")
    logger.info(f"  Best params: {results['best_params']}")
    logger.info(f"  Number of trials: {results['n_trials']}")
    
    return results


def test_anomaly_detector_optimization():
    """Test anomaly detector hyperparameter optimization"""
    logger.info("Testing anomaly detector hyperparameter optimization...")
    
    data = generate_synthetic_data()
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(
        study_name="test_anomaly_optimization",
        n_trials=10,  # Reduced for testing
        timeout=300
    )
    
    # Test anomaly detector optimization
    results = optimizer.optimize_anomaly_detector(
        X_normal=data['X_normal'],
        X_val=data['X_ad_val'],
        y_val=data['y_ad_val'],
        target_fpr=0.05
    )
    
    logger.info(f"Anomaly detector optimization results:")
    logger.info(f"  Best score: {results['best_score']:.4f}")
    logger.info(f"  Best params: {results['best_params']}")
    logger.info(f"  Number of trials: {results['n_trials']}")
    
    return results


def test_multi_objective_optimization():
    """Test multi-objective optimization"""
    logger.info("Testing multi-objective optimization...")
    
    data = generate_synthetic_data()
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(
        study_name="test_multi_objective",
        n_trials=15,  # Reduced for testing
        timeout=400
    )
    
    # Test multi-objective optimization
    results = optimizer.multi_objective_optimization(
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_val=data['X_val'],
        y_val=data['y_val'],
        X_normal=data['X_normal'],
        y_anomaly=data['y_ad_val'],
        model_type="xgboost"
    )
    
    logger.info(f"Multi-objective optimization results:")
    logger.info(f"  Number of Pareto solutions: {len(results['pareto_solutions'])}")
    logger.info(f"  Number of trials: {results['n_trials']}")
    
    # Show first few Pareto solutions
    for i, solution in enumerate(results['pareto_solutions'][:3]):
        logger.info(f"  Solution {i+1}: RMSE={solution['rul_rmse']:.4f}, FPR={solution['fpr']:.4f}")
    
    return results


def test_ensemble_weight_optimization():
    """Test ensemble weight optimization"""
    logger.info("Testing ensemble weight optimization...")
    
    data = generate_synthetic_data()
    
    # Train multiple models for ensemble
    models = []
    model_types = ["xgboost", "lightgbm", "random_forest"]
    
    for model_type in model_types:
        logger.info(f"Training {model_type} model...")
        model = RULRegressionModel(model_type=model_type)
        model.train(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val']
        )
        models.append(model)
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(
        study_name="test_ensemble_weights",
        n_trials=20,  # Reduced for testing
        timeout=300
    )
    
    # Test ensemble weight optimization
    results = optimizer.optimize_ensemble_weights(
        models=models,
        X_val=data['X_val'],
        y_val=data['y_val'],
        scoring="rmse"
    )
    
    logger.info(f"Ensemble weight optimization results:")
    logger.info(f"  Optimal weights: {[f'{w:.3f}' for w in results['optimal_weights']]}")
    logger.info(f"  Best RMSE: {results['best_score']:.4f}")
    logger.info(f"  Number of trials: {results['n_trials']}")
    
    return results


def test_bayesian_ensemble_optimizer():
    """Test Bayesian ensemble optimizer"""
    logger.info("Testing Bayesian ensemble optimizer...")
    
    data = generate_synthetic_data()
    
    # Train multiple models
    models = []
    model_types = ["xgboost", "lightgbm"]
    
    for model_type in model_types:
        logger.info(f"Training {model_type} model...")
        model = RULRegressionModel(model_type=model_type)
        model.train(
            data['X_train'], data['y_train'],
            data['X_val'], data['y_val']
        )
        models.append(model)
    
    # Initialize Bayesian optimizer
    bayesian_optimizer = BayesianEnsembleOptimizer(
        acquisition_function="expected_improvement",
        n_initial_points=5,
        random_state=42
    )
    
    # Test Bayesian optimization
    results = bayesian_optimizer.optimize_weights(
        models=models,
        X_val=data['X_val'],
        y_val=data['y_val'],
        n_calls=15  # Reduced for testing
    )
    
    logger.info(f"Bayesian ensemble optimization results:")
    logger.info(f"  Optimal weights: {[f'{w:.3f}' for w in results['optimal_weights']]}")
    logger.info(f"  Best RMSE: {results['best_score']:.4f}")
    logger.info(f"  Number of calls: {results['n_calls']}")
    
    return results


def test_automated_model_selection():
    """Test automated model selection pipeline"""
    logger.info("Testing automated model selection...")
    
    data = generate_synthetic_data()
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(
        study_name="test_model_selection",
        n_trials=8,  # Reduced for testing
        timeout=600  # 10 minutes max
    )
    
    # Test automated model selection
    results = optimizer.automated_model_selection(
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_val=data['X_val'],
        y_val=data['y_val'],
        model_types=["xgboost", "lightgbm", "random_forest"]  # Reduced set
    )
    
    logger.info(f"Automated model selection results:")
    logger.info(f"  Best model type: {results['best_model_type']}")
    logger.info(f"  Best RMSE: {results['best_score']:.4f}")
    logger.info(f"  Model ranking:")
    
    for i, (model_type, score) in enumerate(results['model_ranking']):
        logger.info(f"    {i+1}. {model_type}: {score:.4f}")
    
    return results


def main():
    """Run all hyperparameter optimization tests"""
    logger.info("Starting hyperparameter optimization tests...")
    
    try:
        # Test individual components
        logger.info("\n" + "="*50)
        test_rul_model_optimization()
        
        logger.info("\n" + "="*50)
        test_anomaly_detector_optimization()
        
        logger.info("\n" + "="*50)
        test_multi_objective_optimization()
        
        logger.info("\n" + "="*50)
        test_ensemble_weight_optimization()
        
        logger.info("\n" + "="*50)
        test_bayesian_ensemble_optimizer()
        
        logger.info("\n" + "="*50)
        test_automated_model_selection()
        
        logger.info("\n" + "="*50)
        logger.info("All hyperparameter optimization tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()