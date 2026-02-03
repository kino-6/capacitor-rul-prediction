#!/usr/bin/env python3
"""
Simple test script for hyperparameter optimization functionality

This script tests the core hyperparameter optimization features
with reduced complexity for faster execution.
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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_simple_data():
    """Generate simple synthetic data for testing"""
    logger.info("Generating simple synthetic data...")
    
    # Generate RUL regression data
    X_rul, y_rul = make_regression(
        n_samples=200,  # Smaller dataset
        n_features=10,  # Fewer features
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
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test
    }


def test_rul_model_optimization_simple():
    """Test RUL model hyperparameter optimization with reduced trials"""
    logger.info("Testing RUL model hyperparameter optimization (simple)...")
    
    data = generate_simple_data()
    
    # Initialize optimizer with minimal trials
    optimizer = HyperparameterOptimizer(
        study_name="test_rul_simple",
        n_trials=5,  # Very few trials for speed
        timeout=120   # 2 minutes max
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
    logger.info(f"  Best params keys: {list(results['best_params'].keys())}")
    logger.info(f"  Number of trials: {results['n_trials']}")
    
    # Verify we can create a model with the optimized parameters
    best_model = RULRegressionModel(model_type="xgboost", **results['best_params'])
    best_model.train(data['X_train'], data['y_train'], data['X_val'], data['y_val'])
    
    # Test prediction
    y_pred = best_model.predict(data['X_test'])
    logger.info(f"  Test prediction shape: {y_pred.shape}")
    logger.info(f"  Test prediction range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    
    return results


def test_ensemble_weight_optimization_simple():
    """Test ensemble weight optimization with simple models"""
    logger.info("Testing ensemble weight optimization (simple)...")
    
    data = generate_simple_data()
    
    # Train two simple models for ensemble
    models = []
    model_types = ["xgboost", "random_forest"]
    
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
        study_name="test_ensemble_simple",
        n_trials=10,  # Reduced for testing
        timeout=120
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
    
    # Verify weights sum to 1
    weight_sum = sum(results['optimal_weights'])
    logger.info(f"  Weight sum: {weight_sum:.6f}")
    assert abs(weight_sum - 1.0) < 1e-6, f"Weights should sum to 1, got {weight_sum}"
    
    return results


def test_bayesian_ensemble_optimizer_simple():
    """Test Bayesian ensemble optimizer with simple setup"""
    logger.info("Testing Bayesian ensemble optimizer (simple)...")
    
    data = generate_simple_data()
    
    # Train two simple models
    models = []
    model_types = ["xgboost", "random_forest"]
    
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
        n_initial_points=3,
        random_state=42
    )
    
    # Test Bayesian optimization
    results = bayesian_optimizer.optimize_weights(
        models=models,
        X_val=data['X_val'],
        y_val=data['y_val'],
        n_calls=8  # Reduced for testing
    )
    
    logger.info(f"Bayesian ensemble optimization results:")
    logger.info(f"  Optimal weights: {[f'{w:.3f}' for w in results['optimal_weights']]}")
    logger.info(f"  Best RMSE: {results['best_score']:.4f}")
    logger.info(f"  Number of calls: {results['n_calls']}")
    
    # Verify weights sum to 1
    weight_sum = sum(results['optimal_weights'])
    logger.info(f"  Weight sum: {weight_sum:.6f}")
    assert abs(weight_sum - 1.0) < 1e-6, f"Weights should sum to 1, got {weight_sum}"
    
    return results


def test_automated_model_selection_simple():
    """Test automated model selection with reduced scope"""
    logger.info("Testing automated model selection (simple)...")
    
    data = generate_simple_data()
    
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(
        study_name="test_model_selection_simple",
        n_trials=3,  # Very few trials per model
        timeout=300  # 5 minutes max
    )
    
    # Test automated model selection with limited models
    results = optimizer.automated_model_selection(
        X_train=data['X_train'],
        y_train=data['y_train'],
        X_val=data['X_val'],
        y_val=data['y_val'],
        model_types=["xgboost", "random_forest"]  # Only two models
    )
    
    logger.info(f"Automated model selection results:")
    logger.info(f"  Best model type: {results['best_model_type']}")
    logger.info(f"  Best RMSE: {results['best_score']:.4f}")
    logger.info(f"  Model ranking:")
    
    for i, (model_type, score) in enumerate(results['model_ranking']):
        logger.info(f"    {i+1}. {model_type}: {score:.4f}")
    
    # Verify best model exists and can predict
    if results['best_model'] is not None:
        y_pred = results['best_model'].predict(data['X_test'])
        logger.info(f"  Best model test prediction shape: {y_pred.shape}")
    
    return results


def main():
    """Run simple hyperparameter optimization tests"""
    logger.info("Starting simple hyperparameter optimization tests...")
    
    try:
        # Test individual components with reduced complexity
        logger.info("\n" + "="*50)
        test_rul_model_optimization_simple()
        
        logger.info("\n" + "="*50)
        test_ensemble_weight_optimization_simple()
        
        logger.info("\n" + "="*50)
        test_bayesian_ensemble_optimizer_simple()
        
        logger.info("\n" + "="*50)
        test_automated_model_selection_simple()
        
        logger.info("\n" + "="*50)
        logger.info("All simple hyperparameter optimization tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()