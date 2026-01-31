"""
Unit tests for RandomForestRULPredictor

Tests the Random Forest RUL predictor implementation including:
- Model initialization
- Training with various configurations
- Prediction with confidence intervals
- Feature importance extraction
- Error handling
- Model persistence

Requirements: 1.1, 1.3
"""

import pytest
import numpy as np
from sklearn.datasets import make_regression
from pathlib import Path
import sys
import importlib.util

# Import the module directly without triggering __init__.py
spec = importlib.util.spec_from_file_location(
    "random_forest_predictor",
    Path(__file__).parent.parent / "src" / "true_rul" / "random_forest_predictor.py"
)
random_forest_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(random_forest_module)
RandomForestRULPredictor = random_forest_module.RandomForestRULPredictor


class TestRandomForestRULPredictorInitialization:
    """Test model initialization"""
    
    def test_default_initialization(self):
        """Test initialization with default parameters"""
        predictor = RandomForestRULPredictor()
        
        assert predictor.n_estimators == 500
        assert predictor.max_depth == 15
        assert predictor.min_samples_split == 5
        assert predictor.min_samples_leaf == 2
        assert predictor.max_features == "sqrt"
        assert predictor.random_state == 42
        assert predictor.n_jobs == -1
        assert not predictor.is_trained
        assert predictor.feature_names is None
    
    def test_custom_initialization(self):
        """Test initialization with custom parameters"""
        predictor = RandomForestRULPredictor(
            n_estimators=300,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features="log2",
            random_state=123,
            n_jobs=4
        )
        
        assert predictor.n_estimators == 300
        assert predictor.max_depth == 10
        assert predictor.min_samples_split == 10
        assert predictor.min_samples_leaf == 5
        assert predictor.max_features == "log2"
        assert predictor.random_state == 123
        assert predictor.n_jobs == 4
    
    def test_model_components_initialized(self):
        """Test that all model components are initialized"""
        predictor = RandomForestRULPredictor()
        
        assert predictor.model is not None
        assert 'lower' in predictor.quantile_models
        assert 'upper' in predictor.quantile_models
        assert predictor.quantile_models['lower'] is not None
        assert predictor.quantile_models['upper'] is not None


class TestRandomForestRULPredictorTraining:
    """Test model training"""
    
    @pytest.fixture
    def synthetic_data(self):
        """Generate synthetic regression data"""
        X, y = make_regression(
            n_samples=200,
            n_features=20,
            n_informative=15,
            noise=10.0,
            random_state=42
        )
        # Make y non-negative (RUL values)
        y = np.abs(y)
        
        # Split into train and validation
        split_idx = 160
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        return X_train, y_train, X_val, y_val
    
    def test_basic_training(self, synthetic_data):
        """Test basic training functionality"""
        X_train, y_train, X_val, y_val = synthetic_data
        
        predictor = RandomForestRULPredictor(n_estimators=50)
        predictor.train(X_train, y_train)
        
        assert predictor.is_trained
        assert predictor.feature_names is not None
        assert len(predictor.feature_names) == X_train.shape[1]
    
    def test_training_with_feature_names(self, synthetic_data):
        """Test training with custom feature names"""
        X_train, y_train, _, _ = synthetic_data
        feature_names = [f"custom_feature_{i}" for i in range(X_train.shape[1])]
        
        predictor = RandomForestRULPredictor(n_estimators=50)
        predictor.train(X_train, y_train, feature_names=feature_names)
        
        assert predictor.feature_names == feature_names
    
    def test_training_with_validation_data(self, synthetic_data):
        """Test training with validation data (should not fail)"""
        X_train, y_train, X_val, y_val = synthetic_data
        
        predictor = RandomForestRULPredictor(n_estimators=50)
        predictor.train(X_train, y_train, X_val, y_val)
        
        assert predictor.is_trained
    
    def test_training_invalid_shapes(self, synthetic_data):
        """Test training with mismatched X and y shapes"""
        X_train, y_train, _, _ = synthetic_data
        
        predictor = RandomForestRULPredictor()
        
        with pytest.raises(ValueError, match="same number of samples"):
            predictor.train(X_train, y_train[:-10])
    
    def test_training_invalid_feature_names(self, synthetic_data):
        """Test training with wrong number of feature names"""
        X_train, y_train, _, _ = synthetic_data
        feature_names = ["feature_1", "feature_2"]  # Too few
        
        predictor = RandomForestRULPredictor()
        
        with pytest.raises(ValueError, match="Number of feature names"):
            predictor.train(X_train, y_train, feature_names=feature_names)
    
    def test_training_creates_default_feature_names(self, synthetic_data):
        """Test that default feature names are created"""
        X_train, y_train, _, _ = synthetic_data
        
        predictor = RandomForestRULPredictor(n_estimators=50)
        predictor.train(X_train, y_train)
        
        assert all(name.startswith("feature_") for name in predictor.feature_names)
        assert len(predictor.feature_names) == X_train.shape[1]


class TestRandomForestRULPredictorPrediction:
    """Test model prediction"""
    
    @pytest.fixture
    def trained_predictor(self):
        """Create a trained predictor"""
        X, y = make_regression(
            n_samples=200,
            n_features=20,
            n_informative=15,
            noise=10.0,
            random_state=42
        )
        y = np.abs(y)
        
        predictor = RandomForestRULPredictor(n_estimators=50, random_state=42)
        predictor.train(X, y)
        
        return predictor, X
    
    def test_basic_prediction(self, trained_predictor):
        """Test basic prediction functionality"""
        predictor, X = trained_predictor
        
        predictions = predictor.predict(X[:10])
        
        assert predictions.shape == (10,)
        assert np.all(predictions >= 0)  # RUL should be non-negative
    
    def test_prediction_without_training(self):
        """Test prediction before training raises error"""
        predictor = RandomForestRULPredictor()
        X = np.random.randn(10, 20)
        
        with pytest.raises(RuntimeError, match="not been trained"):
            predictor.predict(X)
    
    def test_prediction_invalid_shape(self, trained_predictor):
        """Test prediction with wrong number of features"""
        predictor, _ = trained_predictor
        X_wrong = np.random.randn(10, 15)  # Wrong number of features
        
        with pytest.raises(ValueError, match="must match training features"):
            predictor.predict(X_wrong)
    
    def test_prediction_non_negative(self, trained_predictor):
        """Test that predictions are always non-negative"""
        predictor, X = trained_predictor
        
        # Even with extreme negative inputs, predictions should be >= 0
        X_extreme = np.full((10, X.shape[1]), -1000.0)
        predictions = predictor.predict(X_extreme)
        
        assert np.all(predictions >= 0)
    
    def test_prediction_deterministic(self, trained_predictor):
        """Test that predictions are deterministic"""
        predictor, X = trained_predictor
        
        pred1 = predictor.predict(X[:5])
        pred2 = predictor.predict(X[:5])
        
        # Use almost_equal to handle floating point precision
        np.testing.assert_array_almost_equal(pred1, pred2, decimal=10)


class TestRandomForestRULPredictorConfidenceIntervals:
    """Test prediction with confidence intervals"""
    
    @pytest.fixture
    def trained_predictor(self):
        """Create a trained predictor"""
        X, y = make_regression(
            n_samples=200,
            n_features=20,
            n_informative=15,
            noise=10.0,
            random_state=42
        )
        y = np.abs(y)
        
        predictor = RandomForestRULPredictor(n_estimators=100, random_state=42)
        predictor.train(X, y)
        
        return predictor, X
    
    def test_predict_with_confidence_basic(self, trained_predictor):
        """Test basic confidence interval prediction"""
        predictor, X = trained_predictor
        
        pred, lower, upper = predictor.predict_with_confidence(X[:10])
        
        assert pred.shape == (10,)
        assert lower.shape == (10,)
        assert upper.shape == (10,)
    
    def test_confidence_interval_ordering(self, trained_predictor):
        """Test that lower <= prediction <= upper"""
        predictor, X = trained_predictor
        
        pred, lower, upper = predictor.predict_with_confidence(X[:20])
        
        assert np.all(lower <= pred)
        assert np.all(pred <= upper)
    
    def test_confidence_interval_non_negative(self, trained_predictor):
        """Test that confidence bounds are non-negative"""
        predictor, X = trained_predictor
        
        pred, lower, upper = predictor.predict_with_confidence(X[:20])
        
        assert np.all(lower >= 0)
        assert np.all(upper >= 0)
        assert np.all(pred >= 0)
    
    def test_confidence_level_parameter(self, trained_predictor):
        """Test different confidence levels"""
        predictor, X = trained_predictor
        
        # 95% confidence interval
        pred95, lower95, upper95 = predictor.predict_with_confidence(
            X[:10], confidence_level=0.95
        )
        
        # 80% confidence interval (should be narrower)
        pred80, lower80, upper80 = predictor.predict_with_confidence(
            X[:10], confidence_level=0.80
        )
        
        # Predictions should be the same
        np.testing.assert_array_almost_equal(pred95, pred80)
        
        # 80% CI should be narrower than 95% CI
        width_95 = upper95 - lower95
        width_80 = upper80 - lower80
        assert np.all(width_80 <= width_95 + 1e-6)  # Small tolerance for numerical errors
    
    def test_confidence_invalid_level(self, trained_predictor):
        """Test invalid confidence level raises error"""
        predictor, X = trained_predictor
        
        with pytest.raises(ValueError, match="confidence_level must be between"):
            predictor.predict_with_confidence(X[:10], confidence_level=1.5)
        
        with pytest.raises(ValueError, match="confidence_level must be between"):
            predictor.predict_with_confidence(X[:10], confidence_level=0.0)
    
    def test_confidence_without_training(self):
        """Test confidence prediction before training raises error"""
        predictor = RandomForestRULPredictor()
        X = np.random.randn(10, 20)
        
        with pytest.raises(RuntimeError, match="not been trained"):
            predictor.predict_with_confidence(X)
    
    def test_confidence_consistency_with_predict(self, trained_predictor):
        """Test that predict_with_confidence returns same predictions as predict"""
        predictor, X = trained_predictor
        
        pred_simple = predictor.predict(X[:10])
        pred_conf, _, _ = predictor.predict_with_confidence(X[:10])
        
        np.testing.assert_array_almost_equal(pred_simple, pred_conf, decimal=5)


class TestRandomForestRULPredictorFeatureImportance:
    """Test feature importance extraction"""
    
    @pytest.fixture
    def trained_predictor(self):
        """Create a trained predictor with known feature importance"""
        # Create data where first few features are more important
        X, y = make_regression(
            n_samples=200,
            n_features=20,
            n_informative=5,
            noise=10.0,
            random_state=42
        )
        y = np.abs(y)
        
        feature_names = [f"feature_{i}" for i in range(20)]
        
        predictor = RandomForestRULPredictor(n_estimators=100, random_state=42)
        predictor.train(X, y, feature_names=feature_names)
        
        return predictor
    
    def test_get_feature_importance_basic(self, trained_predictor):
        """Test basic feature importance extraction"""
        importance = trained_predictor.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) == 20
        assert all(isinstance(v, (int, float)) for v in importance.values())
    
    def test_feature_importance_non_negative(self, trained_predictor):
        """Test that feature importance values are non-negative"""
        importance = trained_predictor.get_feature_importance()
        
        assert all(v >= 0 for v in importance.values())
    
    def test_feature_importance_sorted(self, trained_predictor):
        """Test that feature importance is sorted in descending order"""
        importance = trained_predictor.get_feature_importance()
        
        values = list(importance.values())
        assert values == sorted(values, reverse=True)
    
    def test_feature_importance_without_training(self):
        """Test feature importance before training raises error"""
        predictor = RandomForestRULPredictor()
        
        with pytest.raises(RuntimeError, match="not been trained"):
            predictor.get_feature_importance()
    
    def test_feature_importance_uses_feature_names(self, trained_predictor):
        """Test that feature importance uses correct feature names"""
        importance = trained_predictor.get_feature_importance()
        
        assert all(name.startswith("feature_") for name in importance.keys())


class TestRandomForestRULPredictorVariance:
    """Test prediction variance calculation"""
    
    @pytest.fixture
    def trained_predictor(self):
        """Create a trained predictor"""
        X, y = make_regression(
            n_samples=200,
            n_features=20,
            n_informative=15,
            noise=10.0,
            random_state=42
        )
        y = np.abs(y)
        
        predictor = RandomForestRULPredictor(n_estimators=100, random_state=42)
        predictor.train(X, y)
        
        return predictor, X
    
    def test_get_prediction_variance_basic(self, trained_predictor):
        """Test basic variance calculation"""
        predictor, X = trained_predictor
        
        variance = predictor.get_prediction_variance(X[:10])
        
        assert variance.shape == (10,)
        assert np.all(variance >= 0)  # Variance is always non-negative
    
    def test_variance_without_training(self):
        """Test variance calculation before training raises error"""
        predictor = RandomForestRULPredictor()
        X = np.random.randn(10, 20)
        
        with pytest.raises(RuntimeError, match="not been trained"):
            predictor.get_prediction_variance(X)
    
    def test_variance_invalid_shape(self, trained_predictor):
        """Test variance with wrong number of features"""
        predictor, _ = trained_predictor
        X_wrong = np.random.randn(10, 15)
        
        with pytest.raises(ValueError, match="must match training features"):
            predictor.get_prediction_variance(X_wrong)


class TestRandomForestRULPredictorModelInfo:
    """Test model information retrieval"""
    
    def test_get_model_info_untrained(self):
        """Test model info for untrained model"""
        predictor = RandomForestRULPredictor(
            n_estimators=300,
            max_depth=10,
            random_state=123
        )
        
        info = predictor.get_model_info()
        
        assert info['model_type'] == 'random_forest'
        assert info['is_trained'] is False
        assert info['n_features'] == 0
        assert info['hyperparameters']['n_estimators'] == 300
        assert info['hyperparameters']['max_depth'] == 10
        assert info['hyperparameters']['random_state'] == 123
    
    def test_get_model_info_trained(self):
        """Test model info for trained model"""
        X, y = make_regression(n_samples=100, n_features=15, random_state=42)
        y = np.abs(y)
        
        predictor = RandomForestRULPredictor(n_estimators=50)
        predictor.train(X, y)
        
        info = predictor.get_model_info()
        
        assert info['is_trained'] is True
        assert info['n_features'] == 15
        assert info['n_trees'] == 50
        assert len(info['feature_names']) == 15


class TestRandomForestRULPredictorPersistence:
    """Test model saving and loading"""
    
    @pytest.fixture
    def trained_predictor_and_data(self, tmp_path):
        """Create a trained predictor and test data"""
        X, y = make_regression(
            n_samples=100,
            n_features=15,
            noise=10.0,
            random_state=42
        )
        y = np.abs(y)
        
        feature_names = [f"feature_{i}" for i in range(15)]
        
        predictor = RandomForestRULPredictor(n_estimators=50, random_state=42)
        predictor.train(X, y, feature_names=feature_names)
        
        filepath = tmp_path / "rf_model.joblib"
        
        return predictor, X, filepath
    
    def test_save_model(self, trained_predictor_and_data):
        """Test model saving"""
        predictor, _, filepath = trained_predictor_and_data
        
        predictor.save_model(str(filepath))
        
        assert filepath.exists()
    
    def test_save_untrained_model_raises_error(self, tmp_path):
        """Test saving untrained model raises error"""
        predictor = RandomForestRULPredictor()
        filepath = tmp_path / "model.joblib"
        
        with pytest.raises(RuntimeError, match="not been trained"):
            predictor.save_model(str(filepath))
    
    def test_load_model(self, trained_predictor_and_data):
        """Test model loading"""
        predictor, X, filepath = trained_predictor_and_data
        
        # Save model
        predictor.save_model(str(filepath))
        
        # Create new predictor and load
        new_predictor = RandomForestRULPredictor()
        new_predictor.load_model(str(filepath))
        
        assert new_predictor.is_trained
        assert new_predictor.feature_names == predictor.feature_names
        assert new_predictor.n_estimators == predictor.n_estimators
    
    def test_loaded_model_predictions_match(self, trained_predictor_and_data):
        """Test that loaded model produces same predictions"""
        predictor, X, filepath = trained_predictor_and_data
        
        # Get predictions from original model
        original_pred = predictor.predict(X[:10])
        original_conf = predictor.predict_with_confidence(X[:10])
        
        # Save and load model
        predictor.save_model(str(filepath))
        new_predictor = RandomForestRULPredictor()
        new_predictor.load_model(str(filepath))
        
        # Get predictions from loaded model
        loaded_pred = new_predictor.predict(X[:10])
        loaded_conf = new_predictor.predict_with_confidence(X[:10])
        
        np.testing.assert_array_almost_equal(original_pred, loaded_pred)
        np.testing.assert_array_almost_equal(original_conf[0], loaded_conf[0])
        np.testing.assert_array_almost_equal(original_conf[1], loaded_conf[1])
        np.testing.assert_array_almost_equal(original_conf[2], loaded_conf[2])


class TestRandomForestRULPredictorEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_single_sample_prediction(self):
        """Test prediction with single sample"""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        y = np.abs(y)
        
        predictor = RandomForestRULPredictor(n_estimators=50)
        predictor.train(X, y)
        
        single_sample = X[0:1]
        pred = predictor.predict(single_sample)
        
        assert pred.shape == (1,)
        assert pred[0] >= 0
    
    def test_large_batch_prediction(self):
        """Test prediction with large batch"""
        X, y = make_regression(n_samples=200, n_features=10, random_state=42)
        y = np.abs(y)
        
        predictor = RandomForestRULPredictor(n_estimators=50)
        predictor.train(X[:100], y[:100])
        
        # Predict on large batch
        large_batch = X[100:]
        pred = predictor.predict(large_batch)
        
        assert pred.shape == (100,)
        assert np.all(pred >= 0)
    
    def test_zero_rul_values(self):
        """Test training with zero RUL values"""
        X = np.random.randn(100, 10)
        y = np.zeros(100)  # All zeros
        
        predictor = RandomForestRULPredictor(n_estimators=50)
        predictor.train(X, y)
        
        pred = predictor.predict(X[:10])
        
        # Should predict values close to zero
        assert np.all(pred >= 0)
        assert np.all(pred < 10)  # Should be small
    
    def test_high_variance_data(self):
        """Test with high variance data"""
        X = np.random.randn(100, 10) * 100  # High variance
        y = np.abs(np.random.randn(100) * 1000)  # High variance RUL
        
        predictor = RandomForestRULPredictor(n_estimators=50)
        predictor.train(X, y)
        
        pred, lower, upper = predictor.predict_with_confidence(X[:10])
        
        # Should still produce valid predictions
        assert np.all(pred >= 0)
        assert np.all(lower >= 0)
        assert np.all(upper >= 0)
        assert np.all(lower <= pred)
        assert np.all(pred <= upper)
