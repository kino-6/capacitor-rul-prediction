"""
Integration tests for ElasticNetRULPredictor

Tests the Elastic Net predictor with synthetic feature data
to verify integration behavior.
"""

import pytest
import numpy as np
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


class TestElasticNetIntegration:
    """Integration tests for ElasticNet with synthetic feature data"""
    
    @pytest.fixture
    def synthetic_features_and_labels(self):
        """Create synthetic feature data simulating extracted features"""
        np.random.seed(42)
        n_samples = 50
        n_features = 15  # Simulating 15 responsiveness features
        
        # Create features with degradation pattern
        features = []
        rul_labels = []
        
        for i in range(n_samples):
            # Simulate degrading features
            degradation_factor = 1 - (i / n_samples)
            feat = np.random.randn(n_features) * degradation_factor + np.array([
                degradation_factor * 10,  # Feature 0: strongly correlated with RUL
                degradation_factor * 5,   # Feature 1: moderately correlated
                np.random.randn(),        # Feature 2: noise
                degradation_factor * 3,   # Feature 3: weakly correlated
                *np.random.randn(n_features - 4)  # Rest: mostly noise
            ])
            features.append(feat)
            rul_labels.append(n_samples - i)
        
        return np.array(features), np.array(rul_labels)
    
    def test_end_to_end_prediction(self, synthetic_features_and_labels):
        """Test end-to-end prediction with synthetic features"""
        X, y = synthetic_features_and_labels
        
        # Split into train and test
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train ElasticNet predictor
        predictor = ElasticNetRULPredictor(degree=2, alpha=0.1)
        predictor.train(X_train, y_train)
        
        # Make predictions
        predictions = predictor.predict(X_test)
        
        # Verify predictions
        assert predictions.shape == y_test.shape
        assert np.all(predictions >= 0)
        assert np.all(np.isfinite(predictions))
        
        # Check that predictions are reasonable (within 2x of actual)
        assert np.all(predictions < y_test.max() * 2)
    
    def test_feature_importance_with_synthetic_features(self, synthetic_features_and_labels):
        """Test feature importance with synthetic features"""
        X, y = synthetic_features_and_labels
        
        # Train predictor
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        predictor = ElasticNetRULPredictor(degree=1, alpha=0.5)
        predictor.train(X, y, feature_names=feature_names)
        
        # Get feature importance
        importance = predictor.get_feature_importance()
        
        # Verify importance
        assert len(importance) == 15
        assert all(v >= 0 for v in importance.values())
        assert abs(sum(importance.values()) - 1.0) < 1e-6
        
        # Feature 0 should have high importance (strongly correlated)
        assert importance['feature_0'] > 0.1
    
    def test_interpretability_coefficients(self, synthetic_features_and_labels):
        """Test getting interpretable coefficients"""
        X, y = synthetic_features_and_labels
        X = X[:30]  # Use fewer samples for faster test
        y = y[:30]
        
        # Train predictor with degree=1 for simpler interpretation
        predictor = ElasticNetRULPredictor(degree=1, alpha=0.1)
        predictor.train(X, y)
        
        # Get coefficients
        coefficients = predictor.get_feature_coefficients()
        
        # Verify coefficients
        assert len(coefficients) > 0
        assert all(isinstance(v, (float, np.floating)) for v in coefficients.values())
        
        # Get top 5 coefficients
        top_5 = predictor.get_feature_coefficients(top_k=5)
        assert len(top_5) == 5
    
    def test_polynomial_features_with_synthetic_data(self, synthetic_features_and_labels):
        """Test polynomial feature expansion with synthetic data"""
        X, y = synthetic_features_and_labels
        X = X[:30]
        y = y[:30]
        
        # Train with degree=2 (quadratic features)
        predictor = ElasticNetRULPredictor(degree=2, alpha=0.5)
        predictor.train(X, y)
        
        # Verify polynomial features were created
        info = predictor.get_model_info()
        assert info['n_features'] == 15  # Original features
        # With degree=2 and 15 features: 15 + 15*16/2 = 135 polynomial features
        assert info['n_poly_features'] == 135
        
        # Make predictions
        predictions = predictor.predict(X)
        assert predictions.shape == (len(X),)
        assert np.all(predictions >= 0)
    
    def test_regularization_with_many_features(self, synthetic_features_and_labels):
        """Test that regularization works with polynomial features"""
        X, y = synthetic_features_and_labels
        X = X[:30]
        y = y[:30]
        
        # Train with high regularization
        predictor = ElasticNetRULPredictor(degree=2, alpha=5.0, l1_ratio=0.9)
        predictor.train(X, y)
        
        # Check that regularization created sparsity
        info = predictor.get_model_info()
        assert info['sparsity'] > 0  # Some coefficients should be zero
        
        # Predictions should still work
        predictions = predictor.predict(X)
        assert np.all(predictions >= 0)
    
    def test_comparison_with_baseline(self, synthetic_features_and_labels):
        """Test that ElasticNet produces reasonable results"""
        X, y = synthetic_features_and_labels
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train ElasticNet
        elastic_net = ElasticNetRULPredictor(degree=2, alpha=0.1)
        elastic_net.train(X_train, y_train)
        en_predictions = elastic_net.predict(X_test)
        
        # Calculate RMSE
        en_rmse = np.sqrt(np.mean((en_predictions - y_test) ** 2))
        
        # RMSE should be reasonable (less than 2x mean RUL)
        # This is a loose check since we're using synthetic data with noise
        assert en_rmse < np.mean(y_test) * 2
        
        # Predictions should be in reasonable range
        assert np.all(en_predictions <= y_test.max() * 1.5)
        assert np.all(en_predictions >= 0)


class TestElasticNetScaling:
    """Test ElasticNet with different data scales"""
    
    def test_feature_scaling_effect(self):
        """Test that feature scaling is applied correctly"""
        np.random.seed(42)
        
        # Create features with different scales
        X_train = np.random.randn(100, 5)
        X_train[:, 0] *= 1000  # Large scale
        X_train[:, 1] *= 0.001  # Small scale
        y_train = 50 + 2 * X_train[:, 0] / 1000 - 3 * X_train[:, 1] / 0.001
        y_train = np.maximum(y_train, 0)
        
        # Train predictor (should handle scaling internally)
        predictor = ElasticNetRULPredictor(degree=1, alpha=0.01)
        predictor.train(X_train, y_train)
        
        # Make predictions
        X_test = np.random.randn(10, 5)
        X_test[:, 0] *= 1000
        X_test[:, 1] *= 0.001
        predictions = predictor.predict(X_test)
        
        # Predictions should be reasonable despite different scales
        assert np.all(predictions >= 0)
        assert np.all(np.isfinite(predictions))
    
    def test_consistent_predictions_after_scaling(self):
        """Test that predictions are consistent with scaled features"""
        np.random.seed(42)
        X_train = np.random.randn(100, 5) * 10
        y_train = np.random.rand(100) * 100
        
        predictor = ElasticNetRULPredictor(degree=1)
        predictor.train(X_train, y_train)
        
        # Same input should give same output
        X_test = np.random.randn(5, 5) * 10
        pred1 = predictor.predict(X_test)
        pred2 = predictor.predict(X_test)
        
        np.testing.assert_array_almost_equal(pred1, pred2)


class TestElasticNetEdgeCasesIntegration:
    """Test edge cases in integration scenarios"""
    
    def test_all_zero_features(self):
        """Test with all zero features"""
        X_train = np.zeros((50, 5))
        y_train = np.ones(50) * 50
        
        predictor = ElasticNetRULPredictor(degree=1)
        predictor.train(X_train, y_train)
        
        X_test = np.zeros((10, 5))
        predictions = predictor.predict(X_test)
        
        # Should predict close to mean
        assert np.all(np.abs(predictions - 50) < 10)
    
    def test_highly_correlated_features(self):
        """Test with highly correlated features"""
        np.random.seed(42)
        X_base = np.random.randn(100, 1)
        # Create correlated features
        X_train = np.hstack([X_base, X_base + np.random.randn(100, 1) * 0.01,
                            X_base + np.random.randn(100, 1) * 0.01])
        y_train = 100 - 2 * X_base.flatten()
        y_train = np.maximum(y_train, 0)
        
        # ElasticNet should handle correlated features with regularization
        predictor = ElasticNetRULPredictor(degree=1, alpha=1.0, l1_ratio=0.5)
        predictor.train(X_train, y_train)
        
        X_test = np.random.randn(10, 3)
        predictions = predictor.predict(X_test)
        
        assert np.all(predictions >= 0)
        assert np.all(np.isfinite(predictions))
