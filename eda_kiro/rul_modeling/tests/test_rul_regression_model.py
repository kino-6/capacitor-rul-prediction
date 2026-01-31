"""
Tests for RULRegressionModel unified interface

This module tests the unified interface for all RUL regression models,
including factory method, training, prediction, and interpretability features.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch
import logging

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from true_rul.rul_regression_model import RULRegressionModel


class TestRULRegressionModel:
    """Test cases for RULRegressionModel unified interface"""
    
    @pytest.fixture
    def sample_data(self):
        """Generate sample training data"""
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        # Create realistic RUL labels (positive integers)
        y = np.random.randint(1, 200, size=n_samples).astype(float)
        
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
    
    def test_factory_method_xgboost(self):
        """Test factory method creates XGBoost model correctly"""
        model = RULRegressionModel(model_type="xgboost")
        
        assert model.model_type == "xgboost"
        assert hasattr(model.model, 'train')
        assert hasattr(model.model, 'predict')
        assert not model.is_trained
    
    def test_factory_method_lightgbm(self):
        """Test factory method creates LightGBM model correctly"""
        model = RULRegressionModel(model_type="lightgbm")
        
        assert model.model_type == "lightgbm"
        assert hasattr(model.model, 'train')
        assert hasattr(model.model, 'predict')
        assert not model.is_trained
    
    def test_factory_method_random_forest(self):
        """Test factory method creates Random Forest model correctly"""
        model = RULRegressionModel(model_type="random_forest")
        
        assert model.model_type == "random_forest"
        assert hasattr(model.model, 'train')
        assert hasattr(model.model, 'predict')
        assert not model.is_trained
    
    def test_factory_method_elastic_net(self):
        """Test factory method creates Elastic Net model correctly"""
        model = RULRegressionModel(model_type="elastic_net")
        
        assert model.model_type == "elastic_net"
        assert hasattr(model.model, 'train')
        assert hasattr(model.model, 'predict')
        assert not model.is_trained
    
    def test_factory_method_ensemble(self):
        """Test factory method creates ensemble model correctly"""
        model = RULRegressionModel(model_type="ensemble")
        
        assert model.model_type == "ensemble"
        assert hasattr(model.model, 'train')
        assert hasattr(model.model, 'predict')
        assert not model.is_trained
    
    def test_factory_method_invalid_type(self):
        """Test factory method raises error for invalid model type"""
        with pytest.raises(ValueError, match="Unknown model type"):
            RULRegressionModel(model_type="invalid_model")
    
    def test_get_available_models(self):
        """Test get_available_models class method"""
        available = RULRegressionModel.get_available_models()
        
        assert isinstance(available, dict)
        assert "xgboost" in available
        assert "lightgbm" in available
        assert "random_forest" in available
        assert "elastic_net" in available
        assert "ensemble" in available
        
        # Check descriptions are strings
        for model_type, description in available.items():
            assert isinstance(description, str)
            assert len(description) > 0
    
    def test_train_input_validation(self, sample_data):
        """Test training input validation"""
        model = RULRegressionModel(model_type="xgboost")
        
        # Test mismatched X_train and y_train shapes
        with pytest.raises(ValueError, match="must have same number of samples"):
            model.train(
                X_train=sample_data['X_train'],
                y_train=sample_data['y_train'][:-1]  # One less sample
            )
        
        # Test mismatched X_val and y_val shapes
        with pytest.raises(ValueError, match="must have same number of samples"):
            model.train(
                X_train=sample_data['X_train'],
                y_train=sample_data['y_train'],
                X_val=sample_data['X_val'],
                y_val=sample_data['y_val'][:-1]  # One less sample
            )
        
        # Test mismatched feature dimensions
        with pytest.raises(ValueError, match="must have same number of features"):
            model.train(
                X_train=sample_data['X_train'],
                y_train=sample_data['y_train'],
                X_val=sample_data['X_val'][:, :-1],  # One less feature
                y_val=sample_data['y_val']
            )
        
        # Test mismatched feature names
        with pytest.raises(ValueError, match="must match number of features"):
            model.train(
                X_train=sample_data['X_train'],
                y_train=sample_data['y_train'],
                feature_names=sample_data['feature_names'][:-1]  # One less name
            )
    
    def test_train_success(self, sample_data):
        """Test successful training"""
        model = RULRegressionModel(model_type="xgboost", n_estimators=10)
        
        # Train model
        result = model.train(
            X_train=sample_data['X_train'],
            y_train=sample_data['y_train'],
            X_val=sample_data['X_val'],
            y_val=sample_data['y_val'],
            feature_names=sample_data['feature_names']
        )
        
        # Check return value and state
        assert result is model  # Should return self
        assert model.is_trained
        assert model.feature_names == sample_data['feature_names']
    
    def test_predict_before_training(self, sample_data):
        """Test prediction fails before training"""
        model = RULRegressionModel(model_type="xgboost")
        
        with pytest.raises(RuntimeError, match="has not been trained"):
            model.predict(sample_data['X_train'])
    
    def test_predict_input_validation(self, sample_data):
        """Test prediction input validation"""
        model = RULRegressionModel(model_type="xgboost", n_estimators=10)
        model.train(
            X_train=sample_data['X_train'],
            y_train=sample_data['y_train'],
            feature_names=sample_data['feature_names']
        )
        
        # Test wrong number of features
        with pytest.raises(ValueError, match="must match training features"):
            wrong_features = sample_data['X_train'][:, :-1]  # One less feature
            model.predict(wrong_features)
    
    def test_predict_success(self, sample_data):
        """Test successful prediction"""
        model = RULRegressionModel(model_type="xgboost", n_estimators=10)
        model.train(
            X_train=sample_data['X_train'],
            y_train=sample_data['y_train'],
            feature_names=sample_data['feature_names']
        )
        
        predictions = model.predict(sample_data['X_val'])
        
        assert isinstance(predictions, np.ndarray)
        assert predictions.shape == (sample_data['X_val'].shape[0],)
        assert np.all(predictions >= 0)  # RUL should be non-negative
    
    def test_predict_with_confidence_before_training(self, sample_data):
        """Test confidence prediction fails before training"""
        model = RULRegressionModel(model_type="xgboost")
        
        with pytest.raises(RuntimeError, match="has not been trained"):
            model.predict_with_confidence(sample_data['X_train'])
    
    def test_predict_with_confidence_input_validation(self, sample_data):
        """Test confidence prediction input validation"""
        model = RULRegressionModel(model_type="xgboost", n_estimators=10)
        model.train(
            X_train=sample_data['X_train'],
            y_train=sample_data['y_train'],
            feature_names=sample_data['feature_names']
        )
        
        # Test invalid confidence level
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            model.predict_with_confidence(sample_data['X_val'], confidence_level=1.5)
        
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            model.predict_with_confidence(sample_data['X_val'], confidence_level=0.0)
    
    def test_predict_with_confidence_success(self, sample_data):
        """Test successful confidence prediction"""
        model = RULRegressionModel(model_type="random_forest", n_estimators=10)
        model.train(
            X_train=sample_data['X_train'],
            y_train=sample_data['y_train'],
            feature_names=sample_data['feature_names']
        )
        
        pred, lower, upper = model.predict_with_confidence(sample_data['X_val'])
        
        assert isinstance(pred, np.ndarray)
        assert isinstance(lower, np.ndarray)
        assert isinstance(upper, np.ndarray)
        assert pred.shape == lower.shape == upper.shape
        assert np.all(lower <= pred)
        assert np.all(pred <= upper)
        assert np.all(lower >= 0)  # All bounds should be non-negative
    
    def test_predict_with_confidence_fallback(self, sample_data):
        """Test fallback confidence intervals for models without native support"""
        model = RULRegressionModel(model_type="elastic_net")
        model.train(
            X_train=sample_data['X_train'],
            y_train=sample_data['y_train'],
            feature_names=sample_data['feature_names']
        )
        
        pred, lower, upper = model.predict_with_confidence(sample_data['X_val'])
        
        assert isinstance(pred, np.ndarray)
        assert isinstance(lower, np.ndarray)
        assert isinstance(upper, np.ndarray)
        assert pred.shape == lower.shape == upper.shape
        assert np.all(lower <= pred)
        assert np.all(pred <= upper)
    
    def test_get_feature_importance_before_training(self, sample_data):
        """Test feature importance fails before training"""
        model = RULRegressionModel(model_type="xgboost")
        
        with pytest.raises(RuntimeError, match="has not been trained"):
            model.get_feature_importance()
    
    def test_get_feature_importance_success(self, sample_data):
        """Test successful feature importance extraction"""
        model = RULRegressionModel(model_type="xgboost", n_estimators=10)
        model.train(
            X_train=sample_data['X_train'],
            y_train=sample_data['y_train'],
            feature_names=sample_data['feature_names']
        )
        
        importance = model.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) > 0
        
        # Check all feature names are present
        for feature_name in sample_data['feature_names']:
            assert feature_name in importance
        
        # Check all values are non-negative
        for value in importance.values():
            assert value >= 0
    
    def test_get_shap_values_before_training(self, sample_data):
        """Test SHAP values fail before training"""
        model = RULRegressionModel(model_type="xgboost")
        
        with pytest.raises(RuntimeError, match="has not been trained"):
            model.get_shap_values(sample_data['X_train'])
    
    def test_get_shap_values_success(self, sample_data):
        """Test successful SHAP values extraction"""
        model = RULRegressionModel(model_type="xgboost", n_estimators=10)
        model.train(
            X_train=sample_data['X_train'],
            y_train=sample_data['y_train'],
            feature_names=sample_data['feature_names']
        )
        
        shap_values = model.get_shap_values(sample_data['X_val'])
        
        assert isinstance(shap_values, np.ndarray)
        assert shap_values.shape == sample_data['X_val'].shape
    
    def test_get_shap_values_not_supported(self, sample_data):
        """Test SHAP values not supported for some models"""
        model = RULRegressionModel(model_type="elastic_net")
        model.train(
            X_train=sample_data['X_train'],
            y_train=sample_data['y_train'],
            feature_names=sample_data['feature_names']
        )
        
        with pytest.raises(NotImplementedError, match="SHAP values not available"):
            model.get_shap_values(sample_data['X_val'])
    
    def test_get_model_info(self, sample_data):
        """Test model info extraction"""
        model = RULRegressionModel(model_type="xgboost", n_estimators=10)
        
        # Test before training
        info_before = model.get_model_info()
        assert info_before['is_trained'] is False
        assert info_before['wrapper_model_type'] == "xgboost"
        assert 'supported_methods' in info_before
        
        # Test after training
        model.train(
            X_train=sample_data['X_train'],
            y_train=sample_data['y_train'],
            feature_names=sample_data['feature_names']
        )
        
        info_after = model.get_model_info()
        assert info_after['is_trained'] is True
        assert info_after['n_features'] == len(sample_data['feature_names'])
        assert info_after['feature_names'] == sample_data['feature_names']
    
    def test_supported_methods(self):
        """Test supported methods detection"""
        # Test XGBoost (supports most methods)
        xgb_model = RULRegressionModel(model_type="xgboost")
        xgb_info = xgb_model.get_model_info()
        xgb_methods = xgb_info['supported_methods']
        
        assert xgb_methods['predict'] is True
        assert xgb_methods['get_feature_importance'] is True
        assert xgb_methods['get_shap_values'] is True
        
        # Test Random Forest (supports confidence intervals)
        rf_model = RULRegressionModel(model_type="random_forest")
        rf_info = rf_model.get_model_info()
        rf_methods = rf_info['supported_methods']
        
        assert rf_methods['predict'] is True
        assert rf_methods['predict_with_confidence'] is True
        assert rf_methods['get_feature_importance'] is True
    
    def test_repr(self, sample_data):
        """Test string representation"""
        model = RULRegressionModel(model_type="xgboost")
        
        # Before training
        repr_before = repr(model)
        assert "xgboost" in repr_before
        assert "untrained" in repr_before
        
        # After training
        model.train(
            X_train=sample_data['X_train'],
            y_train=sample_data['y_train'],
            feature_names=sample_data['feature_names']
        )
        
        repr_after = repr(model)
        assert "xgboost" in repr_after
        assert "trained" in repr_after
        assert str(len(sample_data['feature_names'])) in repr_after
    
    def test_parameter_passing(self):
        """Test that parameters are correctly passed to underlying models"""
        # Test XGBoost parameters
        xgb_model = RULRegressionModel(
            model_type="xgboost",
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1
        )
        assert xgb_model.model.n_estimators == 100
        assert xgb_model.model.max_depth == 8
        assert xgb_model.model.learning_rate == 0.1
        
        # Test Random Forest parameters
        rf_model = RULRegressionModel(
            model_type="random_forest",
            n_estimators=200,
            max_depth=10
        )
        assert rf_model.model.n_estimators == 200
        assert rf_model.model.max_depth == 10
        
        # Test Elastic Net parameters
        en_model = RULRegressionModel(
            model_type="elastic_net",
            degree=3,
            alpha=2.0,
            l1_ratio=0.7
        )
        assert en_model.model.degree == 3
        assert en_model.model.alpha == 2.0
        assert en_model.model.l1_ratio == 0.7
    
    def test_ensemble_model_integration(self, sample_data):
        """Test ensemble model works through unified interface"""
        model = RULRegressionModel(
            model_type="ensemble",
            weights={'xgboost': 0.5, 'lightgbm': 0.3, 'random_forest': 0.2}
        )
        
        # Train ensemble
        model.train(
            X_train=sample_data['X_train'],
            y_train=sample_data['y_train'],
            X_val=sample_data['X_val'],
            y_val=sample_data['y_val'],
            feature_names=sample_data['feature_names']
        )
        
        # Test predictions
        predictions = model.predict(sample_data['X_val'])
        assert isinstance(predictions, np.ndarray)
        assert np.all(predictions >= 0)
        
        # Test confidence intervals
        pred, lower, upper = model.predict_with_confidence(sample_data['X_val'])
        assert np.all(lower <= pred)
        assert np.all(pred <= upper)
        
        # Test feature importance
        importance = model.get_feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) > 0


if __name__ == "__main__":
    pytest.main([__file__])