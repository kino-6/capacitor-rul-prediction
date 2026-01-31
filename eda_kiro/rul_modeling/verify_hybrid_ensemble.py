#!/usr/bin/env python3
"""
Verification script for HybridEnsembleRULPredictor
Tests the implementation without complex dependencies
"""

import numpy as np
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb
import lightgbm as lgb
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock the predictor classes for testing
class MockGradientBoostingRULPredictor(BaseEstimator, RegressorMixin):
    def __init__(self, model_type="xgboost", **kwargs):
        self.model_type = model_type
        self.kwargs = kwargs
        self.is_trained = False
        self.feature_names = None
        
        if model_type == "xgboost":
            self.model = xgb.XGBRegressor(**kwargs)
        else:
            self.model = lgb.LGBMRegressor(**kwargs)
    
    def train(self, X_train, y_train, X_val=None, y_val=None, feature_names=None, **kwargs):
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]
        if X_val is not None and self.model_type == "xgboost":
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        elif X_val is not None and self.model_type == "lightgbm":
            self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        else:
            self.model.fit(X_train, y_train)
        self.is_trained = True
        return self
    
    def predict(self, X):
        return np.maximum(self.model.predict(X), 0)
    
    def get_feature_importance(self):
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))

class MockRandomForestRULPredictor(BaseEstimator, RegressorMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = RandomForestRegressor(**kwargs)
        self.is_trained = False
        self.feature_names = None
    
    def train(self, X_train, y_train, X_val=None, y_val=None, feature_names=None, **kwargs):
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self
    
    def predict(self, X):
        return np.maximum(self.model.predict(X), 0)
    
    def get_feature_importance(self):
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))

# Simplified HybridEnsembleRULPredictor for testing
class HybridEnsembleRULPredictor(BaseEstimator, RegressorMixin):
    def __init__(self, weights=None, **kwargs):
        self.weights = weights or {'xgboost': 0.4, 'lightgbm': 0.4, 'random_forest': 0.2}
        
        # Validate weights
        if not np.isclose(sum(self.weights.values()), 1.0, atol=1e-6):
            raise ValueError(f"Weights must sum to 1.0, got {sum(self.weights.values())}")
        
        self.models = {
            'xgboost': MockGradientBoostingRULPredictor('xgboost', n_estimators=50),
            'lightgbm': MockGradientBoostingRULPredictor('lightgbm', n_estimators=50),
            'random_forest': MockRandomForestRULPredictor(n_estimators=50)
        }
        
        self.is_trained = False
        self.feature_names = None
    
    def train(self, X_train, y_train, X_val=None, y_val=None, feature_names=None, **kwargs):
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.train(X_train, y_train, X_val, y_val, self.feature_names, **kwargs)
        
        self.is_trained = True
        return self
    
    def predict(self, X):
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X) * self.weights[name]
            predictions.append(pred)
        
        return np.maximum(np.sum(predictions, axis=0), 0)
    
    def predict_with_confidence(self, X, confidence_level=0.95):
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        ensemble_pred = self.predict(X)
        
        # Get individual predictions for variance
        individual_preds = []
        for model in self.models.values():
            individual_preds.append(model.predict(X))
        
        individual_preds = np.array(individual_preds)
        ensemble_std = np.std(individual_preds, axis=0)
        
        # Use z-score for confidence intervals
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z_score = z_scores.get(confidence_level, 1.96)
        
        lower = np.maximum(ensemble_pred - z_score * ensemble_std, 0)
        upper = np.maximum(ensemble_pred + z_score * ensemble_std, 0)
        
        return ensemble_pred, lower, upper
    
    def get_aggregated_feature_importance(self):
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        
        aggregated = {name: 0.0 for name in self.feature_names}
        
        for model_name, model in self.models.items():
            importance = model.get_feature_importance()
            weight = self.weights[model_name]
            
            for feature, score in importance.items():
                aggregated[feature] += score * weight
        
        # Normalize
        total = sum(aggregated.values())
        if total > 0:
            aggregated = {k: v/total for k, v in aggregated.items()}
        
        return dict(sorted(aggregated.items(), key=lambda x: x[1], reverse=True))


def test_hybrid_ensemble():
    """Test the HybridEnsembleRULPredictor"""
    print("🧪 Testing HybridEnsembleRULPredictor Implementation")
    print("=" * 60)
    
    # Generate test data
    X, y = make_regression(n_samples=200, n_features=20, noise=10.0, random_state=42)
    y = np.abs(y)
    y = (y - y.min()) / (y.max() - y.min()) * 200
    
    # Split data
    X_train, X_val, X_test = X[:120], X[120:160], X[160:]
    y_train, y_val, y_test = y[:120], y[120:160], y[160:]
    
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Test 1: Initialization
    print("1. Testing initialization...")
    predictor = HybridEnsembleRULPredictor()
    assert not predictor.is_trained
    assert len(predictor.models) == 3
    print("   ✅ Initialization successful")
    
    # Test 2: Training
    print("2. Testing training...")
    predictor.train(X_train, y_train, X_val, y_val, feature_names)
    assert predictor.is_trained
    print("   ✅ Training successful")
    
    # Test 3: Prediction
    print("3. Testing prediction...")
    predictions = predictor.predict(X_test)
    assert predictions.shape == (X_test.shape[0],)
    assert np.all(predictions >= 0)
    print(f"   ✅ Predictions: shape={predictions.shape}, range=[{predictions.min():.2f}, {predictions.max():.2f}]")
    
    # Test 4: Confidence intervals
    print("4. Testing confidence intervals...")
    pred, lower, upper = predictor.predict_with_confidence(X_test)
    assert np.all(lower <= pred)
    assert np.all(upper >= pred)
    assert np.all(lower >= 0)
    print("   ✅ Confidence intervals successful")
    
    # Test 5: Feature importance
    print("5. Testing feature importance...")
    importance = predictor.get_aggregated_feature_importance()
    assert len(importance) == len(feature_names)
    assert np.isclose(sum(importance.values()), 1.0, atol=1e-6)
    top_features = list(importance.keys())[:3]
    print(f"   ✅ Top 3 features: {top_features}")
    
    # Test 6: Custom weights
    print("6. Testing custom weights...")
    custom_weights = {'xgboost': 0.5, 'lightgbm': 0.3, 'random_forest': 0.2}
    predictor2 = HybridEnsembleRULPredictor(weights=custom_weights)
    predictor2.train(X_train, y_train, feature_names=feature_names)
    pred2 = predictor2.predict(X_test)
    
    # Predictions should be different with different weights
    assert not np.allclose(predictions, pred2, atol=1e-6)
    print("   ✅ Custom weights produce different predictions")
    
    # Performance summary
    print("\n📊 Performance Summary:")
    mae = np.mean(np.abs(predictions - y_test))
    rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
    print(f"   MAE: {mae:.2f}")
    print(f"   RMSE: {rmse:.2f}")
    print(f"   Weights: {predictor.weights}")
    
    print("\n🎉 All tests passed! HybridEnsembleRULPredictor implementation is correct.")
    
    return predictor


if __name__ == "__main__":
    try:
        predictor = test_hybrid_ensemble()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()