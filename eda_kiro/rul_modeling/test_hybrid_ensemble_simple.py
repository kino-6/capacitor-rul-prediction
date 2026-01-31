#!/usr/bin/env python3
"""
Simple test for HybridEnsembleRULPredictor without complex dependencies
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.datasets import make_regression

# Add src to path and import directly
src_path = Path(__file__).parent / "src" / "true_rul"
sys.path.insert(0, str(src_path))

# Import modules directly to avoid __init__.py issues
import gradient_boosting_predictor
import random_forest_predictor
import hybrid_ensemble_predictor

GradientBoostingRULPredictor = gradient_boosting_predictor.GradientBoostingRULPredictor
RandomForestRULPredictor = random_forest_predictor.RandomForestRULPredictor
HybridEnsembleRULPredictor = hybrid_ensemble_predictor.HybridEnsembleRULPredictor


def test_hybrid_ensemble_basic():
    """Test basic functionality of HybridEnsembleRULPredictor"""
    print("Testing HybridEnsembleRULPredictor...")
    
    # Generate synthetic data
    X, y = make_regression(
        n_samples=200,
        n_features=20,
        n_informative=15,
        noise=10.0,
        random_state=42
    )
    # Ensure non-negative RUL values
    y = np.abs(y)
    y = (y - y.min()) / (y.max() - y.min()) * 200  # Scale to 0-200 range
    
    # Split data
    train_size = 120
    val_size = 40
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Test initialization
    print("1. Testing initialization...")
    predictor = HybridEnsembleRULPredictor(
        xgboost_params={'n_estimators': 50},
        lightgbm_params={'n_estimators': 50},
        random_forest_params={'n_estimators': 50}
    )
    
    assert predictor.is_trained is False
    assert len(predictor.models) == 3
    assert 'xgboost' in predictor.models
    assert 'lightgbm' in predictor.models
    assert 'random_forest' in predictor.models
    print("✓ Initialization successful")
    
    # Test training
    print("2. Testing training...")
    predictor.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names
    )
    
    assert predictor.is_trained is True
    assert predictor.feature_names == feature_names
    print("✓ Training successful")
    
    # Test prediction
    print("3. Testing prediction...")
    predictions = predictor.predict(X_test)
    
    assert predictions.shape == (X_test.shape[0],)
    assert np.all(predictions >= 0), "All predictions should be non-negative"
    assert np.all(np.isfinite(predictions)), "All predictions should be finite"
    print(f"✓ Prediction successful. Shape: {predictions.shape}, Range: [{predictions.min():.2f}, {predictions.max():.2f}]")
    
    # Test prediction with confidence
    print("4. Testing prediction with confidence...")
    pred, lower, upper = predictor.predict_with_confidence(X_test)
    
    assert pred.shape == predictions.shape
    assert lower.shape == predictions.shape
    assert upper.shape == predictions.shape
    assert np.all(lower <= pred), "Lower bounds should be <= predictions"
    assert np.all(upper >= pred), "Upper bounds should be >= predictions"
    print("✓ Confidence intervals successful")
    
    # Test feature importance
    print("5. Testing feature importance...")
    importance = predictor.get_aggregated_feature_importance()
    
    assert isinstance(importance, dict)
    assert len(importance) == len(feature_names)
    assert np.isclose(sum(importance.values()), 1.0, atol=1e-6)
    print(f"✓ Feature importance successful. Top 3 features: {list(importance.keys())[:3]}")
    
    # Test individual predictions
    print("6. Testing individual predictions...")
    individual_preds = predictor.get_individual_predictions(X_test)
    
    assert len(individual_preds) == 3
    assert 'xgboost' in individual_preds
    assert 'lightgbm' in individual_preds
    assert 'random_forest' in individual_preds
    print("✓ Individual predictions successful")
    
    # Test prediction variance
    print("7. Testing prediction variance...")
    variance = predictor.get_prediction_variance(X_test)
    
    assert variance.shape == (X_test.shape[0],)
    assert np.all(variance >= 0), "Variance should be non-negative"
    print(f"✓ Prediction variance successful. Mean variance: {variance.mean():.4f}")
    
    # Test model info
    print("8. Testing model info...")
    info = predictor.get_model_info()
    
    assert info['model_type'] == 'hybrid_ensemble'
    assert info['is_trained'] is True
    assert info['n_features'] == len(feature_names)
    print("✓ Model info successful")
    
    print("\n🎉 All tests passed! HybridEnsembleRULPredictor is working correctly.")
    
    return predictor, X_test, y_test


if __name__ == "__main__":
    try:
        predictor, X_test, y_test = test_hybrid_ensemble_basic()
        
        # Additional performance check
        print("\n📊 Performance Summary:")
        predictions = predictor.predict(X_test)
        mae = np.mean(np.abs(predictions - y_test))
        rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
        print(f"MAE: {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        
        # Show weights effect
        print(f"\nModel weights: {predictor.weights}")
        individual_preds = predictor.get_individual_predictions(X_test)
        for name, pred in individual_preds.items():
            mae_individual = np.mean(np.abs(pred - y_test))
            print(f"{name} MAE: {mae_individual:.2f}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)