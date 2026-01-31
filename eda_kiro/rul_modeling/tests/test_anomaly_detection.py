"""
Tests for anomaly detection models.

This module contains unit tests for the anomaly detection components:
- IsolationForestDetector
- AutoencoderDetector  
- ImprovedOCSVM
- EnsembleAnomalyDetector
"""

import pytest
import numpy as np
import torch
from unittest.mock import patch

from true_rul.isolation_forest_detector import IsolationForestDetector
from true_rul.autoencoder_detector import AutoencoderDetector
from true_rul.improved_ocsvm import ImprovedOCSVM
from true_rul.ensemble_anomaly_detector import EnsembleAnomalyDetector


class TestIsolationForestDetector:
    """Test cases for IsolationForestDetector."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = IsolationForestDetector(contamination=0.1)
        assert detector.contamination == 0.1
        assert not detector.is_fitted
        assert detector.feature_names is None
    
    def test_invalid_contamination(self):
        """Test initialization with invalid contamination."""
        with pytest.raises(ValueError, match="contamination must be between 0 and 0.5"):
            IsolationForestDetector(contamination=0.6)
    
    def test_fit_and_predict(self):
        """Test fitting and prediction."""
        # Generate synthetic normal data
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (100, 5))
        
        detector = IsolationForestDetector(contamination=0.05)
        detector.fit(normal_data)
        
        assert detector.is_fitted
        
        # Test prediction
        test_data = np.random.normal(0, 1, (10, 5))
        scores = detector.predict_score(test_data)
        
        assert len(scores) == 10
        assert isinstance(scores, np.ndarray)
    
    def test_fit_empty_data(self):
        """Test fitting with empty data."""
        detector = IsolationForestDetector()
        
        with pytest.raises(ValueError, match="normal_data cannot be empty"):
            detector.fit(np.array([]))
    
    def test_predict_before_fit(self):
        """Test prediction before fitting."""
        detector = IsolationForestDetector()
        test_data = np.random.normal(0, 1, (10, 5))
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            detector.predict_score(test_data)


class TestAutoencoderDetector:
    """Test cases for AutoencoderDetector."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = AutoencoderDetector(input_dim=10, encoding_dim=5)
        assert detector.input_dim == 10
        assert detector.encoding_dim == 5
        assert not detector.is_fitted
    
    def test_invalid_dimensions(self):
        """Test initialization with invalid dimensions."""
        with pytest.raises(ValueError, match="input_dim must be positive"):
            AutoencoderDetector(input_dim=0, encoding_dim=5)
        
        with pytest.raises(ValueError, match="encoding_dim must be positive"):
            AutoencoderDetector(input_dim=10, encoding_dim=15)
    
    def test_forward_pass(self):
        """Test forward pass through autoencoder."""
        detector = AutoencoderDetector(input_dim=10, encoding_dim=5)
        
        # Test forward pass
        x = torch.randn(32, 10)
        output = detector.forward(x)
        
        assert output.shape == (32, 10)
    
    def test_fit_and_predict(self):
        """Test fitting and prediction."""
        # Generate synthetic normal data
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (100, 10))
        
        detector = AutoencoderDetector(input_dim=10, encoding_dim=5)
        detector.fit(normal_data, epochs=5, verbose=False)  # Quick training for test
        
        assert detector.is_fitted
        assert detector.reconstruction_threshold is not None
        
        # Test prediction
        test_data = np.random.normal(0, 1, (10, 10))
        errors = detector.get_reconstruction_error(test_data)
        
        assert len(errors) == 10
        assert isinstance(errors, np.ndarray)
        assert all(errors >= 0)  # Reconstruction errors should be non-negative
    
    def test_fit_wrong_dimensions(self):
        """Test fitting with wrong input dimensions."""
        detector = AutoencoderDetector(input_dim=10, encoding_dim=5)
        wrong_data = np.random.normal(0, 1, (100, 5))  # Wrong number of features
        
        with pytest.raises(ValueError, match="Expected 10 features, got 5"):
            detector.fit(wrong_data)


class TestImprovedOCSVM:
    """Test cases for ImprovedOCSVM."""
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = ImprovedOCSVM(nu=0.1, kernel='rbf')
        assert detector.nu == 0.1
        assert detector.kernel == 'rbf'
        assert not detector.is_fitted
    
    def test_invalid_nu(self):
        """Test initialization with invalid nu."""
        with pytest.raises(ValueError, match="nu must be between 0 and 1"):
            ImprovedOCSVM(nu=1.5)
    
    def test_invalid_kernel(self):
        """Test initialization with invalid kernel."""
        with pytest.raises(ValueError, match="kernel must be one of"):
            ImprovedOCSVM(kernel='invalid')
    
    def test_fit_and_predict(self):
        """Test fitting and prediction."""
        # Generate synthetic normal data
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (100, 5))
        
        detector = ImprovedOCSVM(nu=0.05, auto_tune=False)  # Disable tuning for speed
        detector.fit(normal_data)
        
        assert detector.is_fitted
        
        # Test prediction
        test_data = np.random.normal(0, 1, (10, 5))
        scores = detector.predict_score(test_data)
        
        assert len(scores) == 10
        assert isinstance(scores, np.ndarray)
    
    def test_get_support_vectors(self):
        """Test getting support vectors."""
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (50, 3))
        
        detector = ImprovedOCSVM(nu=0.1, auto_tune=False)
        detector.fit(normal_data)
        
        support_vectors = detector.get_support_vectors()
        assert support_vectors.shape[1] == 3  # Same number of features
        assert len(support_vectors) > 0  # Should have some support vectors


class TestEnsembleAnomalyDetector:
    """Test cases for EnsembleAnomalyDetector."""
    
    def test_initialization(self):
        """Test ensemble initialization."""
        ensemble = EnsembleAnomalyDetector()
        assert len(ensemble.weights) == 3
        assert np.isclose(sum(ensemble.weights), 1.0)
        assert not ensemble.is_fitted
    
    def test_invalid_weights(self):
        """Test initialization with invalid weights."""
        with pytest.raises(ValueError, match="weights must contain exactly 3 values"):
            EnsembleAnomalyDetector(weights=[0.5, 0.5])
        
        with pytest.raises(ValueError, match="weights must sum to 1.0"):
            EnsembleAnomalyDetector(weights=[0.5, 0.5, 0.2])
    
    def test_fit_and_predict(self):
        """Test fitting and prediction."""
        # Generate synthetic normal data
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (100, 10))
        
        # Use smaller autoencoder for faster testing
        ensemble = EnsembleAnomalyDetector(
            autoencoder_params={'encoding_dim': 5}
        )
        
        # Mock the autoencoder training to be faster
        with patch.object(AutoencoderDetector, 'fit') as mock_fit:
            mock_ae = AutoencoderDetector(input_dim=10, encoding_dim=5)
            mock_ae.is_fitted = True
            mock_ae.reconstruction_threshold = 0.1
            mock_ae.scaler.fit(normal_data)
            mock_fit.return_value = mock_ae
            
            ensemble.fit(normal_data)
        
        assert ensemble.is_fitted
        
        # Test prediction
        test_data = np.random.normal(0, 1, (20, 10))
        
        # Mock the autoencoder prediction
        with patch.object(AutoencoderDetector, 'get_reconstruction_error') as mock_predict:
            mock_predict.return_value = np.random.uniform(0, 0.2, 20)
            
            binary_pred, scores, info = ensemble.predict(test_data)
        
        assert len(binary_pred) == 20
        assert len(scores) == 20
        assert isinstance(info, dict)
        assert 'feature_importance' in info
        assert 'threshold' in info
    
    def test_predict_before_fit(self):
        """Test prediction before fitting."""
        ensemble = EnsembleAnomalyDetector()
        test_data = np.random.normal(0, 1, (10, 5))
        
        with pytest.raises(ValueError, match="Ensemble must be fitted"):
            ensemble.predict(test_data)
    
    def test_get_model_info(self):
        """Test getting model information."""
        ensemble = EnsembleAnomalyDetector()
        info = ensemble.get_model_info()
        
        assert 'is_fitted' in info
        assert 'weights' in info
        assert 'detector_names' in info
        assert info['n_detectors'] == 3


if __name__ == "__main__":
    pytest.main([__file__])