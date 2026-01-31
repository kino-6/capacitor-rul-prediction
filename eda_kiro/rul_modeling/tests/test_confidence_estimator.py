"""
Tests for ConfidenceEstimator class
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from true_rul.confidence_estimator import (
    ConfidenceEstimator,
    EnsembleVarianceMethod,
    MonteCarloDropoutMethod
)


class MockModel:
    """Mock model for testing"""
    
    def __init__(self, base_prediction=50.0):
        self.base_prediction = base_prediction
    
    def predict(self, x):
        """Mock predict method"""
        if len(x.shape) > 1:
            return np.array([self.base_prediction] * x.shape[0])
        return np.array([self.base_prediction])


class TestEnsembleVarianceMethod:
    """Test cases for EnsembleVarianceMethod"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.method = EnsembleVarianceMethod(confidence_level=0.95)
        self.mock_model = MockModel(base_prediction=50.0)
        self.test_input = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    def test_initialization(self):
        """Test method initialization"""
        assert self.method.confidence_level == 0.95
        assert self.method.z_score == 1.96
    
    def test_z_score_calculation(self):
        """Test z-score calculation for different confidence levels"""
        method_90 = EnsembleVarianceMethod(confidence_level=0.90)
        assert method_90.z_score == 1.645
        
        method_99 = EnsembleVarianceMethod(confidence_level=0.99)
        assert method_99.z_score == 2.576
    
    def test_single_model_prediction(self):
        """Test prediction from single model"""
        pred = self.method._single_prediction(self.mock_model, self.test_input)
        assert pred == 50.0
    
    def test_bootstrap_predictions(self):
        """Test bootstrap prediction generation"""
        predictions = self.method._bootstrap_predictions(
            self.mock_model, self.test_input, n_samples=10
        )
        
        assert len(predictions) == 10
        assert all(isinstance(p, float) for p in predictions)
        # Predictions should vary around the base prediction
        mean_pred = np.mean(predictions)
        assert abs(mean_pred - 50.0) < 10.0  # Should be close to base prediction
    
    def test_confidence_interval_computation(self):
        """Test confidence interval computation"""
        predictions = [45.0, 50.0, 55.0, 48.0, 52.0]
        lower, upper = self.method._compute_confidence_interval(predictions)
        
        assert lower < upper
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        
        # Mean should be between bounds
        mean_pred = np.mean(predictions)
        assert lower <= mean_pred <= upper
    
    def test_ensemble_with_multiple_models(self):
        """Test ensemble estimation with multiple models"""
        models = [
            MockModel(45.0),
            MockModel(50.0),
            MockModel(55.0)
        ]
        
        lower, upper = self.method.estimate(models, self.test_input)
        
        assert lower < upper
        assert isinstance(lower, float)
        assert isinstance(upper, float)
    
    def test_empty_predictions_error(self):
        """Test error handling for empty predictions"""
        with pytest.raises(ValueError):
            self.method._compute_confidence_interval([])


class TestConfidenceEstimator:
    """Test cases for ConfidenceEstimator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.estimator = ConfidenceEstimator(method="ensemble", confidence_level=0.95)
        self.mock_model = MockModel(base_prediction=75.0)
        self.test_input = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    def test_initialization_ensemble(self):
        """Test initialization with ensemble method"""
        assert self.estimator.method == "ensemble"
        assert self.estimator.confidence_level == 0.95
        assert isinstance(self.estimator.estimator, EnsembleVarianceMethod)
    
    def test_initialization_invalid_method(self):
        """Test initialization with invalid method"""
        with pytest.raises(ValueError):
            ConfidenceEstimator(method="invalid_method")
    
    def test_estimate_with_single_model(self):
        """Test estimation with single model"""
        lower, upper = self.estimator.estimate(
            self.mock_model, self.test_input, n_samples=20
        )
        
        assert lower < upper
        assert isinstance(lower, float)
        assert isinstance(upper, float)
    
    def test_estimate_with_multiple_models(self):
        """Test estimation with multiple models"""
        models = [
            MockModel(70.0),
            MockModel(75.0),
            MockModel(80.0)
        ]
        
        lower, upper = self.estimator.estimate(models, self.test_input)
        
        assert lower < upper
        # Mean should be around 75.0
        mean_pred = (lower + upper) / 2
        assert abs(mean_pred - 75.0) < 10.0
    
    def test_estimate_confidence_ensemble_direct(self):
        """Test direct ensemble confidence estimation"""
        predictions = [70.0, 75.0, 80.0, 72.0, 78.0]
        lower, upper = self.estimator.estimate_confidence_ensemble(predictions)
        
        assert lower < upper
        mean_pred = np.mean(predictions)
        assert lower <= mean_pred <= upper
    
    def test_get_method_info(self):
        """Test getting method information"""
        info = self.estimator.get_method_info()
        
        assert info['method'] == 'ensemble'
        assert info['confidence_level'] == 0.95
        assert info['z_score'] == 1.96
        assert 'torch_available' in info
    
    def test_set_method(self):
        """Test changing estimation method"""
        # Start with ensemble
        assert self.estimator.method == "ensemble"
        
        # Change to ensemble (should not change)
        self.estimator.set_method("ensemble")
        assert self.estimator.method == "ensemble"
        
        # Test invalid method
        with pytest.raises(ValueError):
            self.estimator.set_method("invalid")
    
    def test_z_score_calculation(self):
        """Test z-score calculation for different confidence levels"""
        assert self.estimator._get_z_score(0.90) == 1.645
        assert self.estimator._get_z_score(0.95) == 1.96
        assert self.estimator._get_z_score(0.99) == 2.576
        assert self.estimator._get_z_score(0.85) == 1.96  # Default fallback
    
    def test_model_without_predict_method(self):
        """Test error handling for model without predict method"""
        invalid_model = "not_a_model"
        
        with pytest.raises(ValueError):
            self.estimator.estimate(invalid_model, self.test_input)
    
    def test_confidence_level_override(self):
        """Test confidence level override in methods"""
        predictions = [70.0, 75.0, 80.0, 72.0, 78.0]
        
        # Test with different confidence level
        lower_95, upper_95 = self.estimator.estimate_confidence_ensemble(
            predictions, confidence_level=0.95
        )
        lower_90, upper_90 = self.estimator.estimate_confidence_ensemble(
            predictions, confidence_level=0.90
        )
        
        # 90% CI should be narrower than 95% CI
        assert (upper_90 - lower_90) < (upper_95 - lower_95)


class TestMonteCarloDropoutMethod:
    """Test cases for MonteCarloDropoutMethod (if PyTorch available)"""
    
    def test_initialization_without_torch(self):
        """Test initialization when PyTorch is not available"""
        # This test assumes PyTorch might not be available
        # The actual behavior depends on the import
        try:
            method = MonteCarloDropoutMethod(confidence_level=0.95)
            assert method.confidence_level == 0.95
        except ImportError:
            # Expected if PyTorch is not available
            pass
    
    def test_estimate_without_torch_model(self):
        """Test estimation with non-PyTorch model"""
        try:
            method = MonteCarloDropoutMethod(confidence_level=0.95)
            mock_model = MockModel()
            test_input = np.array([1.0, 2.0, 3.0])
            
            with pytest.raises(ValueError):
                method.estimate(mock_model, test_input)
        except ImportError:
            # Expected if PyTorch is not available
            pass


class TestIntegration:
    """Integration tests for confidence estimation"""
    
    def test_end_to_end_ensemble_estimation(self):
        """Test end-to-end ensemble confidence estimation"""
        # Create multiple mock models with different predictions
        models = [
            MockModel(48.0),
            MockModel(50.0),
            MockModel(52.0),
            MockModel(49.0),
            MockModel(51.0)
        ]
        
        estimator = ConfidenceEstimator(method="ensemble")
        test_input = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        lower, upper = estimator.estimate(models, test_input)
        
        # Verify reasonable confidence interval
        assert lower < upper
        assert 45.0 <= lower <= 55.0
        assert 45.0 <= upper <= 55.0
        
        # Mean should be around 50.0
        mean_pred = (lower + upper) / 2
        assert abs(mean_pred - 50.0) < 5.0
    
    def test_confidence_interval_width_scaling(self):
        """Test that confidence interval width scales with variance"""
        # Low variance models
        low_var_models = [MockModel(50.0), MockModel(50.1), MockModel(49.9)]
        
        # High variance models  
        high_var_models = [MockModel(40.0), MockModel(50.0), MockModel(60.0)]
        
        estimator = ConfidenceEstimator(method="ensemble")
        test_input = np.array([1.0, 2.0, 3.0])
        
        low_lower, low_upper = estimator.estimate(low_var_models, test_input)
        high_lower, high_upper = estimator.estimate(high_var_models, test_input)
        
        # High variance should have wider confidence interval
        low_width = low_upper - low_lower
        high_width = high_upper - high_lower
        
        assert high_width > low_width