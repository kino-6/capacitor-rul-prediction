"""
Confidence Estimator for True RUL Prediction System

This module implements confidence estimation methods for RUL predictions,
including ensemble variance and Monte Carlo dropout approaches.
"""

from typing import List, Tuple, Any, Optional, Union, Callable
import numpy as np
from abc import ABC, abstractmethod

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class BaseConfidenceMethod(ABC):
    """Base class for confidence estimation methods"""
    
    @abstractmethod
    def estimate(self, model: Any, x: np.ndarray, **kwargs) -> Tuple[float, float]:
        """
        Estimate confidence interval for prediction
        
        Args:
            model: Trained model
            x: Input data
            **kwargs: Method-specific parameters
            
        Returns:
            Tuple of (lower_bound, upper_bound) for 95% confidence interval
        """
        pass


class EnsembleVarianceMethod(BaseConfidenceMethod):
    """
    Confidence estimation using ensemble variance
    
    This method estimates confidence intervals by computing the variance
    across predictions from multiple models or bootstrap samples.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize ensemble variance method
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        """
        self.confidence_level = confidence_level
        self.z_score = self._get_z_score(confidence_level)
    
    def _get_z_score(self, confidence_level: float) -> float:
        """Get z-score for given confidence level"""
        # Common z-scores for confidence intervals
        z_scores = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }
        return z_scores.get(confidence_level, 1.96)
    
    def estimate(
        self,
        model: Any,
        x: np.ndarray,
        n_samples: int = 100,
        **kwargs
    ) -> Tuple[float, float]:
        """
        Estimate confidence interval using ensemble variance
        
        Args:
            model: Model or list of models
            x: Input data (single sample)
            n_samples: Number of bootstrap samples (if single model)
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        predictions = self._get_ensemble_predictions(model, x, n_samples)
        return self._compute_confidence_interval(predictions)
    
    def _get_ensemble_predictions(
        self,
        model: Any,
        x: np.ndarray,
        n_samples: int
    ) -> List[float]:
        """Get predictions from ensemble or bootstrap samples"""
        predictions = []
        
        if isinstance(model, (list, tuple)):
            # Multiple models - use all of them
            for m in model:
                pred = self._single_prediction(m, x)
                predictions.append(pred)
        elif hasattr(model, 'predict'):
            # Single model - use bootstrap sampling
            predictions = self._bootstrap_predictions(model, x, n_samples)
        else:
            raise ValueError("Model must have 'predict' method or be a list of models")
        
        return predictions
    
    def _single_prediction(self, model: Any, x: np.ndarray) -> float:
        """Get single prediction from model"""
        if hasattr(model, 'predict'):
            pred = model.predict(x.reshape(1, -1))
            return float(pred[0]) if hasattr(pred, '__len__') else float(pred)
        else:
            raise ValueError("Model must have 'predict' method")
    
    def _bootstrap_predictions(
        self,
        model: Any,
        x: np.ndarray,
        n_samples: int
    ) -> List[float]:
        """Generate bootstrap predictions (placeholder implementation)"""
        # For now, add small random noise to simulate bootstrap variance
        # In practice, this would use actual bootstrap resampling during training
        base_pred = self._single_prediction(model, x)
        predictions = []
        
        # Estimate model uncertainty as a fraction of the prediction
        uncertainty_factor = 0.1  # 10% uncertainty
        std_dev = abs(base_pred) * uncertainty_factor + 1.0  # Add minimum uncertainty
        
        for _ in range(n_samples):
            noise = np.random.normal(0, std_dev)
            predictions.append(base_pred + noise)
        
        return predictions
    
    def _compute_confidence_interval(self, predictions: List[float]) -> Tuple[float, float]:
        """Compute confidence interval from predictions"""
        if not predictions:
            raise ValueError("No predictions provided")
        
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        margin = self.z_score * std_pred
        lower = mean_pred - margin
        upper = mean_pred + margin
        
        return float(lower), float(upper)


class MonteCarloDropoutMethod(BaseConfidenceMethod):
    """
    Confidence estimation using Monte Carlo Dropout
    
    This method estimates confidence by performing multiple forward passes
    through a neural network with dropout enabled during inference.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize Monte Carlo Dropout method
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Monte Carlo Dropout")
        
        self.confidence_level = confidence_level
        self.z_score = self._get_z_score(confidence_level)
    
    def _get_z_score(self, confidence_level: float) -> float:
        """Get z-score for given confidence level"""
        z_scores = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }
        return z_scores.get(confidence_level, 1.96)
    
    def estimate(
        self,
        model: Any,
        x: np.ndarray,
        n_samples: int = 100,
        **kwargs
    ) -> Tuple[float, float]:
        """
        Estimate confidence using Monte Carlo Dropout
        
        Args:
            model: PyTorch model with dropout layers
            x: Input data
            n_samples: Number of forward passes
            **kwargs: Additional parameters
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Monte Carlo Dropout")
        
        if not isinstance(model, nn.Module):
            raise ValueError("Model must be a PyTorch nn.Module for MC Dropout")
        
        predictions = self._mc_dropout_predictions(model, x, n_samples)
        return self._compute_confidence_interval(predictions)
    
    def _mc_dropout_predictions(
        self,
        model: nn.Module,
        x: np.ndarray,
        n_samples: int
    ) -> List[float]:
        """Generate predictions using Monte Carlo Dropout"""
        # Convert to tensor
        if isinstance(x, np.ndarray):
            x_tensor = torch.FloatTensor(x).unsqueeze(0)
        else:
            x_tensor = x
        
        # Enable dropout for inference
        model.train()
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = model(x_tensor)
                predictions.append(float(pred.item()))
        
        # Restore eval mode
        model.eval()
        
        return predictions
    
    def _compute_confidence_interval(self, predictions: List[float]) -> Tuple[float, float]:
        """Compute confidence interval from predictions"""
        if not predictions:
            raise ValueError("No predictions provided")
        
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        margin = self.z_score * std_pred
        lower = mean_pred - margin
        upper = mean_pred + margin
        
        return float(lower), float(upper)


class ConfidenceEstimator:
    """
    Unified interface for confidence estimation
    
    Supports multiple confidence estimation methods:
    - Ensemble variance
    - Monte Carlo dropout
    - Custom methods
    """
    
    def __init__(self, method: str = "ensemble", confidence_level: float = 0.95):
        """
        Initialize confidence estimator
        
        Args:
            method: Confidence estimation method ("ensemble" or "mcdropout")
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
        """
        self.method = method
        self.confidence_level = confidence_level
        
        # Initialize the appropriate method
        if method == "ensemble":
            self.estimator = EnsembleVarianceMethod(confidence_level)
        elif method == "mcdropout":
            self.estimator = MonteCarloDropoutMethod(confidence_level)
        else:
            raise ValueError(f"Unknown confidence method: {method}")
    
    def estimate(
        self,
        model: Any,
        x: np.ndarray,
        n_samples: int = 100,
        **kwargs
    ) -> Tuple[float, float]:
        """
        Estimate confidence interval for prediction
        
        Args:
            model: Trained model (single model, list of models, or PyTorch module)
            x: Input data (single sample)
            n_samples: Number of samples for estimation
            **kwargs: Method-specific parameters
            
        Returns:
            Tuple of (lower_bound, upper_bound) for confidence interval
        """
        return self.estimator.estimate(model, x, n_samples, **kwargs)
    
    def estimate_confidence_ensemble(
        self,
        predictions: List[float],
        confidence_level: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Estimate confidence interval from ensemble predictions
        
        Args:
            predictions: List of predictions from different models/bootstraps
            confidence_level: Optional confidence level override
            
        Returns:
            Tuple of (lower_bound, upper_bound) for confidence interval
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        z_score = self._get_z_score(confidence_level)
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        margin = z_score * std_pred
        lower = mean_pred - margin
        upper = mean_pred + margin
        
        return float(lower), float(upper)
    
    def estimate_confidence_mcdropout(
        self,
        model: Any,
        x: np.ndarray,
        n_samples: int = 100,
        confidence_level: Optional[float] = None
    ) -> Tuple[float, float]:
        """
        Estimate confidence using Monte Carlo Dropout
        
        Args:
            model: PyTorch model with dropout layers
            x: Input tensor or numpy array
            n_samples: Number of forward passes
            confidence_level: Optional confidence level override
            
        Returns:
            Tuple of (lower_bound, upper_bound) for confidence interval
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Monte Carlo Dropout")
        
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        # Create temporary MC Dropout estimator if needed
        if self.method != "mcdropout":
            mc_estimator = MonteCarloDropoutMethod(confidence_level)
            return mc_estimator.estimate(model, x, n_samples)
        else:
            return self.estimator.estimate(model, x, n_samples)
    
    def _get_z_score(self, confidence_level: float) -> float:
        """Get z-score for given confidence level"""
        z_scores = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }
        return z_scores.get(confidence_level, 1.96)
    
    def set_method(self, method: str):
        """
        Change the confidence estimation method
        
        Args:
            method: New method ("ensemble" or "mcdropout")
        """
        if method != self.method:
            self.method = method
            if method == "ensemble":
                self.estimator = EnsembleVarianceMethod(self.confidence_level)
            elif method == "mcdropout":
                self.estimator = MonteCarloDropoutMethod(self.confidence_level)
            else:
                raise ValueError(f"Unknown confidence method: {method}")
    
    def get_method_info(self) -> dict:
        """
        Get information about the current confidence estimation method
        
        Returns:
            Dictionary with method information
        """
        return {
            'method': self.method,
            'confidence_level': self.confidence_level,
            'z_score': self._get_z_score(self.confidence_level),
            'torch_available': TORCH_AVAILABLE
        }