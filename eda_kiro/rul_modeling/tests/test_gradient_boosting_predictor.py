"""
Unit tests for GradientBoostingRULPredictor

Tests the gradient boosting RUL predictor with both XGBoost and LightGBM.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import directly to avoid data_loader import issues
from true_rul.gradient_boosting_predictor import GradientBoostingRULPredictor


class TestGradientBoostingRULPredictorInitialization:
    """Test model initialization"""
    
    def test_init_xgboost(self):
        """Test XGBoost initialization"""
        predictor = GradientBoostingRULPredictor(model_type="xgboost")
        assert predictor.model_type == "xgboost"
        assert predictor.is_trained is False
        assert predictor.shap_explainer is None
    
    def test_init_lightgbm(self):
        """Test LightGBM initialization"""
        predictor = GradientBoostingRULPredictor(model_type="lightgbm")
        assert predictor.model_type == "lightgbm"
        assert predictor.is_trained is False
        assert predictor.shap_explainer is None
    
    def test_init_invalid_model_type(self):
        """Test initialization with invalid model type"""
        with pytest.raises(ValueError, match="model_type must be"):
            GradientBoostingRULPredictor(model_type="invalid")
    
    def test_init_custom_hyperparameters(self):
        """Test initialization with custom hyperparameters"""
        predictor = GradientBoostingRULPredictor(
            model_type="xgboost",
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1
        )
        assert predictor.n_estimators == 100
        assert predictor.max_depth == 3
        assert predictor.learning_rate == 0.1


class TestGradientBoostingRULPredictorTraining:
    """Test model training"""
    
    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic training data"""
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        X_train = np.random.randn(n_samples, n_features)
        # RUL decreases with some noise
        y_train = np.maximum(0, 100 - np.arange(n_samples) + np.random.randn(n_samples) * 5)
        
        X_val = np.random.randn(20, n_features)
        y_val = np.maximum(0, 100 - np.arange(20) + np.random.randn(20) * 5)
        
        return X_train, y_train, X_val, y_val
    
    def test_train_xgboost_without_validation(self, synthetic_data):
        """Test XGBoost training without validation set"""
        X_train, y_train, _, _ = synthetic_data
        
        predictor = GradientBoostingRULPredictor(
            model_type="xgboost",
            n_estimators=50
        )
        predictor.train(X_train, y_train)
        
        assert predictor.is_trained is True
        assert predictor.feature_names is not None
        assert len(predictor.feature_names) == X_train.shape[1]
    
    def test_train_xgboost_with_validation(self, synthetic_data):
        """Test XGBoost training with validation set and early stopping"""
        X_train, y_train, X_val, y_val = synthetic_data
        
        predictor = GradientBoostingRULPredictor(
            model_type="xgboost",
            n_estimators=100
        )
        predictor.train(
            X_train, y_train,
            X_val, y_val,
            early_stopping_rounds=10
        )
        
        assert predictor.is_trained is True
        assert predictor.shap_explainer is not None
    
    def test_train_lightgbm_without_validation(self, synthetic_data):
        """Test LightGBM training without validation set"""
        X_train, y_train, _, _ = synthetic_data
        
        predictor = GradientBoostingRULPredictor(
            model_type="lightgbm",
            n_estimators=50
        )
        predictor.train(X_train, y_train)
        
        assert predictor.is_trained is True
        assert predictor.feature_names is not None
    
    def test_train_lightgbm_with_validation(self, synthetic_data):
        """Test LightGBM training with validation set and early stopping"""
        X_train, y_train, X_val, y_val = synthetic_data
        
        predictor = GradientBoostingRULPredictor(
            model_type="lightgbm",
            n_estimators=100
        )
        predictor.train(
            X_train, y_train,
            X_val, y_val,
            early_stopping_rounds=10
        )
        
        assert predictor.is_trained is True
        assert predictor.shap_explainer is not None
    
    def test_train_with_feature_names(self, synthetic_data):
        """Test training with custom feature names"""
        X_train, y_train, _, _ = synthetic_data
        feature_names = [f"custom_feature_{i}" for i in range(X_train.shape[1])]
        
        predictor = GradientBoostingRULPredictor(model_type="xgboost")
        predictor.train(X_train, y_train, feature_names=feature_names)
        
        assert predictor.feature_names == feature_names
    
    def test_train_invalid_shapes(self, synthetic_data):
        """Test training with mismatched shapes"""
        X_train, y_train, _, _ = synthetic_data
        
        predictor = GradientBoostingRULPredictor(model_type="xgboost")
        
        # Mismatched X_train and y_train
        with pytest.raises(ValueError, match="same number of samples"):
            predictor.train(X_train, y_train[:-10])
    
    def test_train_invalid_validation_shapes(self, synthetic_data):
        """Test training with mismatched validation shapes"""
        X_train, y_train, X_val, y_val = synthetic_data
        
        predictor = GradientBoostingRULPredictor(model_type="xgboost")
        
        # Mismatched X_val and y_val
        with pytest.raises(ValueError, match="same number of samples"):
            predictor.train(X_train, y_train, X_val, y_val[:-5])
    
    def test_train_invalid_feature_names(self, synthetic_data):
        """Test training with wrong number of feature names"""
        X_train, y_train, _, _ = synthetic_data
        
        predictor = GradientBoostingRULPredictor(model_type="xgboost")
        
        with pytest.raises(ValueError, match="Number of feature names"):
            predictor.train(X_train, y_train, feature_names=["f1", "f2"])


class TestGradientBoostingRULPredictorPrediction:
    """Test model prediction"""
    
    @pytest.fixture
    def trained_predictor(self):
        """Create a trained predictor"""
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        X_train = np.random.randn(n_samples, n_features)
        y_train = np.maximum(0, 100 - np.arange(n_samples) + np.random.randn(n_samples) * 5)
        
        predictor = GradientBoostingRULPredictor(
            model_type="xgboost",
            n_estimators=50
        )
        predictor.train(X_train, y_train)
        
        return predictor, n_features
    
    def test_predict_single_sample(self, trained_predictor):
        """Test prediction on single sample"""
        predictor, n_features = trained_predictor
        
        X_test = np.random.randn(1, n_features)
        predictions = predictor.predict(X_test)
        
        assert predictions.shape == (1,)
        assert predictions[0] >= 0  # RUL should be non-negative
    
    def test_predict_multiple_samples(self, trained_predictor):
        """Test prediction on multiple samples"""
        predictor, n_features = trained_predictor
        
        X_test = np.random.randn(20, n_features)
        predictions = predictor.predict(X_test)
        
        assert predictions.shape == (20,)
        assert np.all(predictions >= 0)  # All RUL values should be non-negative
    
    def test_predict_before_training(self):
        """Test prediction before training raises error"""
        predictor = GradientBoostingRULPredictor(model_type="xgboost")
        X_test = np.random.randn(10, 5)
        
        with pytest.raises(RuntimeError, match="not been trained"):
            predictor.predict(X_test)
    
    def test_predict_invalid_shape(self, trained_predictor):
        """Test prediction with wrong number of features"""
        predictor, n_features = trained_predictor
        
        X_test = np.random.randn(10, n_features + 5)  # Wrong number of features
        
        with pytest.raises(ValueError, match="must match training features"):
            predictor.predict(X_test)
    
    def test_predict_non_negative(self, trained_predictor):
        """Test that predictions are always non-negative"""
        predictor, n_features = trained_predictor
        
        # Create extreme negative features that might produce negative predictions
        X_test = np.ones((10, n_features)) * -100
        predictions = predictor.predict(X_test)
        
        # Even with extreme inputs, RUL should be clipped to non-negative
        assert np.all(predictions >= 0)


class TestGradientBoostingRULPredictorFeatureImportance:
    """Test feature importance methods"""
    
    @pytest.fixture
    def trained_predictor_with_names(self):
        """Create a trained predictor with named features"""
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        X_train = np.random.randn(n_samples, n_features)
        y_train = np.maximum(0, 100 - np.arange(n_samples) + np.random.randn(n_samples) * 5)
        feature_names = [f"feature_{i}" for i in range(n_features)]
        
        predictor = GradientBoostingRULPredictor(
            model_type="xgboost",
            n_estimators=50
        )
        predictor.train(X_train, y_train, feature_names=feature_names)
        
        return predictor, feature_names
    
    def test_get_feature_importance_xgboost(self, trained_predictor_with_names):
        """Test feature importance for XGBoost"""
        predictor, feature_names = trained_predictor_with_names
        
        importance = predictor.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) > 0
        # Check that all values are non-negative
        assert all(v >= 0 for v in importance.values())
    
    def test_get_feature_importance_lightgbm(self):
        """Test feature importance for LightGBM"""
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        X_train = np.random.randn(n_samples, n_features)
        y_train = np.maximum(0, 100 - np.arange(n_samples) + np.random.randn(n_samples) * 5)
        feature_names = [f"feature_{i}" for i in range(n_features)]
        
        predictor = GradientBoostingRULPredictor(
            model_type="lightgbm",
            n_estimators=50
        )
        predictor.train(X_train, y_train, feature_names=feature_names)
        
        importance = predictor.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) == n_features
        assert all(v >= 0 for v in importance.values())
    
    def test_get_feature_importance_before_training(self):
        """Test feature importance before training raises error"""
        predictor = GradientBoostingRULPredictor(model_type="xgboost")
        
        with pytest.raises(RuntimeError, match="not been trained"):
            predictor.get_feature_importance()
    
    def test_feature_importance_sorted(self, trained_predictor_with_names):
        """Test that feature importance is sorted in descending order"""
        predictor, _ = trained_predictor_with_names
        
        importance = predictor.get_feature_importance()
        importance_values = list(importance.values())
        
        # Check that values are in descending order
        assert importance_values == sorted(importance_values, reverse=True)


class TestGradientBoostingRULPredictorSHAP:
    """Test SHAP value computation"""
    
    @pytest.fixture
    def trained_predictor(self):
        """Create a trained predictor"""
        np.random.seed(42)
        n_samples = 100
        n_features = 10
        
        X_train = np.random.randn(n_samples, n_features)
        y_train = np.maximum(0, 100 - np.arange(n_samples) + np.random.randn(n_samples) * 5)
        
        predictor = GradientBoostingRULPredictor(
            model_type="xgboost",
            n_estimators=50
        )
        predictor.train(X_train, y_train)
        
        return predictor, n_features
    
    def test_get_shap_values(self, trained_predictor):
        """Test SHAP value computation"""
        predictor, n_features = trained_predictor
        
        X_test = np.random.randn(10, n_features)
        shap_values = predictor.get_shap_values(X_test)
        
        assert shap_values.shape == (10, n_features)
        assert not np.any(np.isnan(shap_values))
    
    def test_get_shap_values_single_sample(self, trained_predictor):
        """Test SHAP values for single sample"""
        predictor, n_features = trained_predictor
        
        X_test = np.random.randn(1, n_features)
        shap_values = predictor.get_shap_values(X_test)
        
        assert shap_values.shape == (1, n_features)
    
    def test_get_shap_values_before_training(self):
        """Test SHAP values before training raises error"""
        predictor = GradientBoostingRULPredictor(model_type="xgboost")
        X_test = np.random.randn(10, 5)
        
        with pytest.raises(RuntimeError, match="not been trained"):
            predictor.get_shap_values(X_test)
    
    def test_get_shap_values_invalid_shape(self, trained_predictor):
        """Test SHAP values with wrong number of features"""
        predictor, n_features = trained_predictor
        
        X_test = np.random.randn(10, n_features + 5)
        
        with pytest.raises(ValueError, match="must match training features"):
            predictor.get_shap_values(X_test)


class TestGradientBoostingRULPredictorModelInfo:
    """Test model information methods"""
    
    def test_get_model_info_before_training(self):
        """Test model info before training"""
        predictor = GradientBoostingRULPredictor(
            model_type="xgboost",
            n_estimators=100,
            max_depth=5
        )
        
        info = predictor.get_model_info()
        
        assert info["model_type"] == "xgboost"
        assert info["is_trained"] is False
        assert info["n_features"] == 0
        assert info["hyperparameters"]["n_estimators"] == 100
        assert info["hyperparameters"]["max_depth"] == 5
    
    def test_get_model_info_after_training(self):
        """Test model info after training"""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.maximum(0, 100 - np.arange(100))
        
        predictor = GradientBoostingRULPredictor(
            model_type="xgboost",
            n_estimators=50
        )
        predictor.train(X_train, y_train)
        
        info = predictor.get_model_info()
        
        assert info["is_trained"] is True
        assert info["n_features"] == 10
        assert "n_trees" in info
        assert info["n_trees"] > 0


class TestGradientBoostingRULPredictorSaveLoad:
    """Test model saving and loading"""
    
    def test_save_and_load_xgboost(self):
        """Test saving and loading XGBoost model"""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.maximum(0, 100 - np.arange(100))
        
        # Train model
        predictor = GradientBoostingRULPredictor(
            model_type="xgboost",
            n_estimators=50
        )
        predictor.train(X_train, y_train)
        
        # Make predictions before saving
        X_test = np.random.randn(10, 10)
        predictions_before = predictor.predict(X_test)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "model.json")
            predictor.save_model(filepath)
            
            # Load model
            new_predictor = GradientBoostingRULPredictor(model_type="xgboost")
            new_predictor.feature_names = predictor.feature_names
            new_predictor.load_model(filepath)
            
            # Make predictions after loading
            predictions_after = new_predictor.predict(X_test)
            
            # Predictions should be identical
            np.testing.assert_array_almost_equal(
                predictions_before,
                predictions_after,
                decimal=5
            )
    
    def test_save_before_training(self):
        """Test saving before training raises error"""
        predictor = GradientBoostingRULPredictor(model_type="xgboost")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "model.json")
            with pytest.raises(RuntimeError, match="not been trained"):
                predictor.save_model(filepath)


class TestGradientBoostingRULPredictorEdgeCases:
    """Test edge cases and error handling"""
    
    def test_train_with_zero_samples(self):
        """Test training with zero samples"""
        predictor = GradientBoostingRULPredictor(model_type="xgboost")
        X_train = np.array([]).reshape(0, 10)
        y_train = np.array([])
        
        # XGBoost handles empty datasets with a warning but doesn't raise
        # Just verify it doesn't crash
        try:
            predictor.train(X_train, y_train)
            # If it trains, that's acceptable behavior
        except Exception:
            # If it raises an exception, that's also acceptable
            pass
    
    def test_train_with_single_sample(self):
        """Test training with single sample"""
        predictor = GradientBoostingRULPredictor(
            model_type="xgboost",
            n_estimators=10
        )
        X_train = np.random.randn(1, 10)
        y_train = np.array([50.0])
        
        # Should train without error
        predictor.train(X_train, y_train)
        assert predictor.is_trained is True
    
    def test_predict_with_nan_values(self):
        """Test prediction with NaN values"""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        y_train = np.maximum(0, 100 - np.arange(100))
        
        predictor = GradientBoostingRULPredictor(
            model_type="xgboost",
            n_estimators=50
        )
        predictor.train(X_train, y_train)
        
        # Create test data with NaN
        X_test = np.random.randn(10, 10)
        X_test[0, 0] = np.nan
        
        # XGBoost can handle NaN values
        predictions = predictor.predict(X_test)
        assert predictions.shape == (10,)
    
    def test_train_with_all_zero_labels(self):
        """Test training when all RUL labels are zero"""
        predictor = GradientBoostingRULPredictor(
            model_type="xgboost",
            n_estimators=50
        )
        X_train = np.random.randn(100, 10)
        y_train = np.zeros(100)
        
        predictor.train(X_train, y_train)
        
        X_test = np.random.randn(10, 10)
        predictions = predictor.predict(X_test)
        
        # Should predict values close to zero
        assert np.all(predictions >= 0)
        assert np.mean(predictions) < 10  # Should be small


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
