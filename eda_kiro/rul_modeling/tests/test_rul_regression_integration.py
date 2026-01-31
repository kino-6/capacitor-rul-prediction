"""
Integration tests for RULRegressionModel unified interface

This module tests the unified interface with real model training and prediction
to ensure all model types work correctly through the same interface.
"""

import pytest
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from true_rul.rul_regression_model import RULRegressionModel


class TestRULRegressionIntegration:
    """Integration tests for RULRegressionModel"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample training data"""
        np.random.seed(42)
        n_samples = 50  # Smaller dataset for faster tests
        n_features = 5
        
        X = np.random.randn(n_samples, n_features)
        # Create realistic RUL labels (positive integers)
        y = np.random.randint(10, 100, size=n_samples).astype(float)
        
        # Split into train/val
        split_idx = int(0.8 * n_samples)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        feature_names = [f"feature_{i}" for i in range(n_features)]
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'feature_names': feature_names
        }
    
    @pytest.mark.parametrize("model_type", ["xgboost", "lightgbm", "random_forest", "elastic_net"])
    def test_unified_interface_all_models(self, model_type, sample_data):
        """Test that all model types work through the unified interface"""
        # Create model with small parameters for fast testing
        if model_type in ["xgboost", "lightgbm"]:
            model = RULRegressionModel(model_type=model_type, n_estimators=5)
        elif model_type == "random_forest":
            model = RULRegressionModel(model_type=model_type, n_estimators=5)
        else:  # elastic_net
            model = RULRegressionModel(model_type=model_type, degree=1)
        
        # Test training
        model.train(
            X_train=sample_data['X_train'],
            y_train=sample_data['y_train'],
            X_val=sample_data['X_val'],
            y_val=sample_data['y_val'],
            feature_names=sample_data['feature_names']
        )
        
        assert model.is_trained
        assert model.feature_names == sample_data['feature_names']
        
        # Test prediction
        predictions = model.predict(sample_data['X_val'])
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (sample_data['X_val'].shape[0],)
        assert np.all(predictions >= 0)  # RUL should be non-negative
        
        # Test feature importance
        importance = model.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) == len(sample_data['feature_names'])
        
        # Test model info
        info = model.get_model_info()
        assert info['model_type'] == model_type or info['wrapper_model_type'] == model_type
        assert info['is_trained'] is True
        assert info['n_features'] == len(sample_data['feature_names'])
    
    def test_ensemble_model_integration(self, sample_data):
        """Test ensemble model through unified interface"""
        model = RULRegressionModel(
            model_type="ensemble",
            # Use small parameters for fast testing
            xgboost_params={'n_estimators': 5},
            lightgbm_params={'n_estimators': 5},
            random_forest_params={'n_estimators': 5}
        )
        
        # Train ensemble
        model.train(
            X_train=sample_data['X_train'],
            y_train=sample_data['y_train'],
            X_val=sample_data['X_val'],
            y_val=sample_data['y_val'],
            feature_names=sample_data['feature_names']
        )
        
        assert model.is_trained
        
        # Test predictions
        predictions = model.predict(sample_data['X_val'])
        assert isinstance(predictions, np.ndarray)
        assert np.all(predictions >= 0)
        
        # Test confidence intervals
        pred, lower, upper = model.predict_with_confidence(sample_data['X_val'])
        assert np.all(lower <= pred)
        assert np.all(pred <= upper)
        assert np.all(lower >= 0)
        
        # Test aggregated feature importance
        importance = model.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) > 0
    
    def test_confidence_intervals_comparison(self, sample_data):
        """Test confidence intervals across different models"""
        models_with_confidence = ["random_forest", "ensemble"]
        
        for model_type in models_with_confidence:
            if model_type == "ensemble":
                model = RULRegressionModel(
                    model_type=model_type,
                    xgboost_params={'n_estimators': 5},
                    lightgbm_params={'n_estimators': 5},
                    random_forest_params={'n_estimators': 5}
                )
            else:
                model = RULRegressionModel(model_type=model_type, n_estimators=5)
            
            model.train(
                X_train=sample_data['X_train'],
                y_train=sample_data['y_train'],
                feature_names=sample_data['feature_names']
            )
            
            pred, lower, upper = model.predict_with_confidence(sample_data['X_val'])
            
            # Verify confidence interval properties
            assert np.all(lower <= pred), f"Lower bounds exceed predictions for {model_type}"
            assert np.all(pred <= upper), f"Predictions exceed upper bounds for {model_type}"
            assert np.all(lower >= 0), f"Negative lower bounds for {model_type}"
            assert np.all(upper > lower), f"Invalid interval widths for {model_type}"
    
    def test_feature_importance_consistency(self, sample_data):
        """Test that feature importance is consistent across models"""
        model_types = ["xgboost", "lightgbm", "random_forest"]
        importance_results = {}
        
        for model_type in model_types:
            model = RULRegressionModel(model_type=model_type, n_estimators=10)
            model.train(
                X_train=sample_data['X_train'],
                y_train=sample_data['y_train'],
                feature_names=sample_data['feature_names']
            )
            
            importance = model.get_feature_importance()
            importance_results[model_type] = importance
            
            # Check that all features are present
            assert set(importance.keys()) == set(sample_data['feature_names'])
            
            # Check that importance values are non-negative
            assert all(v >= 0 for v in importance.values())
        
        # All models should return the same feature names
        feature_sets = [set(imp.keys()) for imp in importance_results.values()]
        assert all(fs == feature_sets[0] for fs in feature_sets)
    
    def test_parameter_passing_integration(self, sample_data):
        """Test that model-specific parameters are correctly passed and used"""
        # Test XGBoost with specific parameters
        xgb_model = RULRegressionModel(
            model_type="xgboost",
            n_estimators=10,
            max_depth=3,
            learning_rate=0.1
        )
        
        xgb_model.train(
            X_train=sample_data['X_train'],
            y_train=sample_data['y_train'],
            feature_names=sample_data['feature_names']
        )
        
        # Verify parameters were set
        assert xgb_model.model.n_estimators == 10
        assert xgb_model.model.max_depth == 3
        assert xgb_model.model.learning_rate == 0.1
        
        # Test Elastic Net with specific parameters
        en_model = RULRegressionModel(
            model_type="elastic_net",
            degree=2,
            alpha=0.5,
            l1_ratio=0.3
        )
        
        en_model.train(
            X_train=sample_data['X_train'],
            y_train=sample_data['y_train'],
            feature_names=sample_data['feature_names']
        )
        
        # Verify parameters were set
        assert en_model.model.degree == 2
        assert en_model.model.alpha == 0.5
        assert en_model.model.l1_ratio == 0.3
    
    def test_error_handling_integration(self, sample_data):
        """Test error handling across different model types"""
        model_types = ["xgboost", "random_forest", "elastic_net"]
        
        for model_type in model_types:
            # Use appropriate parameters for each model type
            if model_type == "elastic_net":
                model = RULRegressionModel(model_type=model_type, degree=1)
            else:
                model = RULRegressionModel(model_type=model_type, n_estimators=5)
            
            # Test prediction before training
            with pytest.raises(RuntimeError, match="has not been trained"):
                model.predict(sample_data['X_train'])
            
            # Train model
            model.train(
                X_train=sample_data['X_train'],
                y_train=sample_data['y_train'],
                feature_names=sample_data['feature_names']
            )
            
            # Test prediction with wrong input shape
            with pytest.raises(ValueError, match="must match training features"):
                wrong_input = sample_data['X_val'][:, :-1]  # Remove one feature
                model.predict(wrong_input)


if __name__ == "__main__":
    pytest.main([__file__])