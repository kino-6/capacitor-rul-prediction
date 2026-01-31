"""
Unit tests for ElasticNetRULPredictor

Tests the Elastic Net RUL predictor with polynomial features for
interpretable linear RUL predictions.
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import the module directly without going through __init__.py
import importlib.util
spec = importlib.util.spec_from_file_location(
    "elastic_net_predictor",
    Path(__file__).parent.parent / "src" / "true_rul" / "elastic_net_predictor.py"
)
elastic_net_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(elastic_net_module)
ElasticNetRULPredictor = elastic_net_module.ElasticNetRULPredictor


class TestElasticNetRULPredictorInitialization:
    """Test ElasticNetRULPredictor initialization"""
    
    def test_default_initialization(self):
        """Test initialization with default parameters"""
        predictor = ElasticNetRULPredictor()
        
        assert predictor.degree == 2
        assert predictor.alpha == 1.0
        assert predictor.l1_ratio == 0.5
        assert predictor.max_iter == 10000
        assert predictor.is_trained is False
        assert predictor.feature_names is None
        assert predictor.poly_feature_names is None
    
    def test_custom_initialization(self):
        """Test initialization with custom parameters"""
        predictor = ElasticNetRULPredictor(
            degree=3,
            alpha=0.5,
            l1_ratio=0.7,
            max_iter=5000,
            random_state=123
        )
        
        assert predictor.degree == 3
        assert predictor.alpha == 0.5
        assert predictor.l1_ratio == 0.7
        assert predictor.max_iter == 5000
        assert predictor.random_state == 123


class TestElasticNetRULPredictorTraining:
    """Test ElasticNetRULPredictor training"""
    
    def test_train_basic(self):
        """Test basic training functionality"""
        # Create synthetic data
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = 100 - 2 * X_train[:, 0] + 3 * X_train[:, 1] + np.random.randn(100) * 0.1
        y_train = np.maximum(y_train, 0)  # Ensure non-negative
        
        predictor = ElasticNetRULPredictor(degree=2, alpha=0.1)
        predictor.train(X_train, y_train)
        
        assert predictor.is_trained is True
        assert predictor.feature_names is not None
        assert len(predictor.feature_names) == 5
        assert predictor.poly_feature_names is not None
        # With degree=2 and 5 features, we get 5 + 5*6/2 = 20 polynomial features
        assert len(predictor.poly_feature_names) == 20
    
    def test_train_with_feature_names(self):
        """Test training with custom feature names"""
        np.random.seed(42)
        X_train = np.random.randn(100, 3)
        y_train = np.random.rand(100) * 100
        
        feature_names = ['voltage_mean', 'voltage_std', 'frequency']
        predictor = ElasticNetRULPredictor()
        predictor.train(X_train, y_train, feature_names=feature_names)
        
        assert predictor.feature_names == feature_names
        # Check that polynomial feature names include original names
        assert any('voltage_mean' in name for name in predictor.poly_feature_names)
        assert any('voltage_std' in name for name in predictor.poly_feature_names)
    
    def test_train_with_validation_data(self):
        """Test training with validation data (should not fail)"""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.rand(100) * 100
        X_val = np.random.randn(20, 5)
        y_val = np.random.rand(20) * 100
        
        predictor = ElasticNetRULPredictor()
        predictor.train(X_train, y_train, X_val, y_val)
        
        assert predictor.is_trained is True
    
    def test_train_invalid_shapes(self):
        """Test training with mismatched shapes"""
        X_train = np.random.randn(100, 5)
        y_train = np.random.rand(50)  # Wrong size
        
        predictor = ElasticNetRULPredictor()
        
        with pytest.raises(ValueError, match="must have same number of samples"):
            predictor.train(X_train, y_train)
    
    def test_train_invalid_feature_names(self):
        """Test training with wrong number of feature names"""
        X_train = np.random.randn(100, 5)
        y_train = np.random.rand(100) * 100
        feature_names = ['f1', 'f2', 'f3']  # Only 3 names for 5 features
        
        predictor = ElasticNetRULPredictor()
        
        with pytest.raises(ValueError, match="must match number of features"):
            predictor.train(X_train, y_train, feature_names=feature_names)
    
    def test_train_linear_model(self):
        """Test training with degree=1 (linear model)"""
        np.random.seed(42)
        X_train = np.random.randn(100, 3)
        y_train = 50 + 2 * X_train[:, 0] - 3 * X_train[:, 1] + np.random.randn(100) * 0.1
        y_train = np.maximum(y_train, 0)
        
        predictor = ElasticNetRULPredictor(degree=1, alpha=0.01)
        predictor.train(X_train, y_train)
        
        assert predictor.is_trained is True
        # With degree=1 and 3 features, we get exactly 3 features (no interactions)
        assert len(predictor.poly_feature_names) == 3


class TestElasticNetRULPredictorPrediction:
    """Test ElasticNetRULPredictor prediction"""
    
    @pytest.fixture
    def trained_predictor(self):
        """Create a trained predictor for testing"""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = 100 - 2 * X_train[:, 0] + 3 * X_train[:, 1] + np.random.randn(100) * 0.1
        y_train = np.maximum(y_train, 0)
        
        predictor = ElasticNetRULPredictor(degree=2, alpha=0.1)
        predictor.train(X_train, y_train)
        return predictor
    
    def test_predict_basic(self, trained_predictor):
        """Test basic prediction functionality"""
        X_test = np.random.randn(10, 5)
        predictions = trained_predictor.predict(X_test)
        
        assert predictions.shape == (10,)
        assert np.all(predictions >= 0)  # RUL should be non-negative
        assert np.all(np.isfinite(predictions))  # No NaN or inf
    
    def test_predict_single_sample(self, trained_predictor):
        """Test prediction on single sample"""
        X_test = np.random.randn(1, 5)
        predictions = trained_predictor.predict(X_test)
        
        assert predictions.shape == (1,)
        assert predictions[0] >= 0
    
    def test_predict_before_training(self):
        """Test prediction before training raises error"""
        predictor = ElasticNetRULPredictor()
        X_test = np.random.randn(10, 5)
        
        with pytest.raises(RuntimeError, match="has not been trained"):
            predictor.predict(X_test)
    
    def test_predict_wrong_shape(self, trained_predictor):
        """Test prediction with wrong input shape"""
        X_test = np.random.randn(10, 3)  # Wrong number of features
        
        with pytest.raises(ValueError, match="must match training features"):
            trained_predictor.predict(X_test)
    
    def test_predict_non_negative(self, trained_predictor):
        """Test that predictions are always non-negative"""
        # Create extreme negative inputs that might produce negative predictions
        X_test = np.ones((10, 5)) * -100
        predictions = trained_predictor.predict(X_test)
        
        assert np.all(predictions >= 0)


class TestElasticNetRULPredictorInterpretability:
    """Test ElasticNetRULPredictor interpretability features"""
    
    @pytest.fixture
    def trained_predictor(self):
        """Create a trained predictor with known relationships"""
        np.random.seed(42)
        X_train = np.random.randn(200, 3)
        # Create clear relationship: RUL = 100 - 5*f0 + 3*f1 - 2*f2
        y_train = 100 - 5 * X_train[:, 0] + 3 * X_train[:, 1] - 2 * X_train[:, 2]
        y_train = np.maximum(y_train, 0)
        
        feature_names = ['voltage_mean', 'voltage_std', 'frequency']
        predictor = ElasticNetRULPredictor(degree=2, alpha=0.01, l1_ratio=0.1)
        predictor.train(X_train, y_train, feature_names=feature_names)
        return predictor
    
    def test_get_feature_coefficients(self, trained_predictor):
        """Test getting feature coefficients"""
        coefficients = trained_predictor.get_feature_coefficients()
        
        assert isinstance(coefficients, dict)
        assert len(coefficients) > 0
        # All values should be floats
        assert all(isinstance(v, (float, np.floating)) for v in coefficients.values())
    
    def test_get_feature_coefficients_exclude_zero(self, trained_predictor):
        """Test getting coefficients excluding zeros"""
        coef_with_zero = trained_predictor.get_feature_coefficients(include_zero=True)
        coef_without_zero = trained_predictor.get_feature_coefficients(include_zero=False)
        
        # Without zeros should have fewer or equal entries
        assert len(coef_without_zero) <= len(coef_with_zero)
        # All non-zero coefficients should be non-zero
        assert all(v != 0 for v in coef_without_zero.values())
    
    def test_get_feature_coefficients_top_k(self, trained_predictor):
        """Test getting top k coefficients"""
        top_5 = trained_predictor.get_feature_coefficients(top_k=5)
        
        assert len(top_5) == 5
        # Should be sorted by absolute value
        abs_values = [abs(v) for v in top_5.values()]
        assert abs_values == sorted(abs_values, reverse=True)
    
    def test_get_feature_coefficients_before_training(self):
        """Test getting coefficients before training raises error"""
        predictor = ElasticNetRULPredictor()
        
        with pytest.raises(RuntimeError, match="has not been trained"):
            predictor.get_feature_coefficients()
    
    def test_get_feature_importance(self, trained_predictor):
        """Test getting feature importance"""
        importance = trained_predictor.get_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) == 3  # Original features
        assert set(importance.keys()) == {'voltage_mean', 'voltage_std', 'frequency'}
        
        # All importance values should be non-negative
        assert all(v >= 0 for v in importance.values())
        
        # Importance should sum to approximately 1.0
        assert abs(sum(importance.values()) - 1.0) < 1e-6
    
    def test_get_feature_importance_sorted(self, trained_predictor):
        """Test that feature importance is sorted"""
        importance = trained_predictor.get_feature_importance()
        
        values = list(importance.values())
        assert values == sorted(values, reverse=True)
    
    def test_get_intercept(self, trained_predictor):
        """Test getting model intercept"""
        intercept = trained_predictor.get_intercept()
        
        assert isinstance(intercept, float)
        assert np.isfinite(intercept)
    
    def test_get_intercept_before_training(self):
        """Test getting intercept before training raises error"""
        predictor = ElasticNetRULPredictor()
        
        with pytest.raises(RuntimeError, match="has not been trained"):
            predictor.get_intercept()


class TestElasticNetRULPredictorModelInfo:
    """Test ElasticNetRULPredictor model information"""
    
    def test_get_model_info_before_training(self):
        """Test getting model info before training"""
        predictor = ElasticNetRULPredictor(degree=3, alpha=0.5)
        info = predictor.get_model_info()
        
        assert info['model_type'] == 'elastic_net'
        assert info['is_trained'] is False
        assert info['n_features'] == 0
        assert info['hyperparameters']['degree'] == 3
        assert info['hyperparameters']['alpha'] == 0.5
    
    def test_get_model_info_after_training(self):
        """Test getting model info after training"""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.rand(100) * 100
        
        predictor = ElasticNetRULPredictor(degree=2, alpha=1.0, l1_ratio=0.8)
        predictor.train(X_train, y_train)
        info = predictor.get_model_info()
        
        assert info['model_type'] == 'elastic_net'
        assert info['is_trained'] is True
        assert info['n_features'] == 5
        assert info['n_poly_features'] == 20  # 5 + 5*6/2
        assert 'n_iterations' in info
        assert 'intercept' in info
        assert 'n_nonzero_coefs' in info
        assert 'sparsity' in info
        assert 0 <= info['sparsity'] <= 1


class TestElasticNetRULPredictorSaveLoad:
    """Test ElasticNetRULPredictor save and load functionality"""
    
    def test_save_and_load(self):
        """Test saving and loading a trained model"""
        # Train a model
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.rand(100) * 100
        feature_names = [f'feature_{i}' for i in range(5)]
        
        predictor = ElasticNetRULPredictor(degree=2, alpha=0.5)
        predictor.train(X_train, y_train, feature_names=feature_names)
        
        # Get predictions before saving
        X_test = np.random.randn(10, 5)
        predictions_before = predictor.predict(X_test)
        
        # Save model
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'elastic_net_model.pkl')
            predictor.save_model(filepath)
            
            # Load model
            new_predictor = ElasticNetRULPredictor()
            new_predictor.load_model(filepath)
            
            # Check that loaded model has same properties
            assert new_predictor.is_trained is True
            assert new_predictor.feature_names == feature_names
            assert new_predictor.degree == 2
            assert new_predictor.alpha == 0.5
            
            # Check that predictions are identical
            predictions_after = new_predictor.predict(X_test)
            np.testing.assert_array_almost_equal(predictions_before, predictions_after)
    
    def test_save_before_training(self):
        """Test saving before training raises error"""
        predictor = ElasticNetRULPredictor()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'model.pkl')
            with pytest.raises(RuntimeError, match="has not been trained"):
                predictor.save_model(filepath)


class TestElasticNetRULPredictorRegularization:
    """Test ElasticNetRULPredictor regularization effects"""
    
    def test_l1_regularization_sparsity(self):
        """Test that L1 regularization creates sparse models"""
        np.random.seed(42)
        X_train = np.random.randn(100, 10)
        # Only first 3 features are relevant
        y_train = 100 - 2 * X_train[:, 0] + 3 * X_train[:, 1] - X_train[:, 2]
        y_train = np.maximum(y_train, 0)
        
        # High L1 ratio should create sparse model
        predictor = ElasticNetRULPredictor(degree=1, alpha=1.0, l1_ratio=0.95)
        predictor.train(X_train, y_train)
        
        info = predictor.get_model_info()
        # Should have some zero coefficients due to L1 regularization
        assert info['sparsity'] > 0
    
    def test_different_alpha_values(self):
        """Test that different alpha values affect model complexity"""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.random.rand(100) * 100
        
        # Low alpha (less regularization)
        predictor_low = ElasticNetRULPredictor(degree=2, alpha=0.01)
        predictor_low.train(X_train, y_train)
        
        # High alpha (more regularization)
        predictor_high = ElasticNetRULPredictor(degree=2, alpha=10.0)
        predictor_high.train(X_train, y_train)
        
        # High alpha should have more sparsity
        info_low = predictor_low.get_model_info()
        info_high = predictor_high.get_model_info()
        
        assert info_high['sparsity'] >= info_low['sparsity']


class TestElasticNetRULPredictorEdgeCases:
    """Test ElasticNetRULPredictor edge cases"""
    
    def test_single_feature(self):
        """Test with single feature"""
        np.random.seed(42)
        X_train = np.random.randn(100, 1)
        y_train = 50 - 2 * X_train[:, 0] + np.random.randn(100) * 0.1
        y_train = np.maximum(y_train, 0)
        
        predictor = ElasticNetRULPredictor(degree=2)
        predictor.train(X_train, y_train)
        
        X_test = np.random.randn(10, 1)
        predictions = predictor.predict(X_test)
        
        assert predictions.shape == (10,)
        assert np.all(predictions >= 0)
    
    def test_many_features(self):
        """Test with many features"""
        np.random.seed(42)
        X_train = np.random.randn(200, 50)
        y_train = np.random.rand(200) * 100
        
        predictor = ElasticNetRULPredictor(degree=1, alpha=1.0)  # Use degree=1 to avoid explosion
        predictor.train(X_train, y_train)
        
        X_test = np.random.randn(10, 50)
        predictions = predictor.predict(X_test)
        
        assert predictions.shape == (10,)
        assert np.all(predictions >= 0)
    
    def test_constant_target(self):
        """Test with constant target values"""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.ones(100) * 50  # All same value
        
        predictor = ElasticNetRULPredictor()
        predictor.train(X_train, y_train)
        
        X_test = np.random.randn(10, 5)
        predictions = predictor.predict(X_test)
        
        # Predictions should be close to 50
        assert np.all(np.abs(predictions - 50) < 10)
    
    def test_zero_target(self):
        """Test with zero target values"""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = np.zeros(100)
        
        predictor = ElasticNetRULPredictor()
        predictor.train(X_train, y_train)
        
        X_test = np.random.randn(10, 5)
        predictions = predictor.predict(X_test)
        
        # Predictions should be close to 0
        assert np.all(predictions < 10)
