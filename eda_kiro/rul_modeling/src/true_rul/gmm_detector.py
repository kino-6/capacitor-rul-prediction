"""
Gaussian Mixture Model (GMM) Anomaly Detector

This module implements GMM-based anomaly detection for RUL prediction.
GMM models the normal data distribution and identifies anomalies as
samples with low likelihood under the learned distribution.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


@dataclass
class GMMConfig:
    """Configuration for GMM detector."""
    n_components: int = 3
    covariance_type: str = "full"  # "full", "tied", "diag", "spherical"
    max_iter: int = 100
    n_init: int = 1
    init_params: str = "kmeans"  # "kmeans", "random"
    random_state: int = 42
    tol: float = 1e-3
    reg_covar: float = 1e-6
    contamination: float = 0.1  # Expected proportion of outliers
    normalize_features: bool = True
    confidence_level: float = 0.95  # For threshold computation


class GMMDetector:
    """
    Gaussian Mixture Model anomaly detector.
    
    This detector models normal data using a mixture of Gaussian distributions
    and identifies anomalies as samples with low likelihood under the model.
    """
    
    def __init__(self, config: GMMConfig):
        self.config = config
        self.gmm_model: Optional[GaussianMixture] = None
        self.scaler: Optional[StandardScaler] = None
        self.threshold: Optional[float] = None
        self.is_fitted = False
        self.feature_names: Optional[List[str]] = None
        
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'GMMDetector':
        """
        Fit the GMM detector on normal data.
        
        Args:
            X: Training data (normal samples only)
            feature_names: Optional feature names for interpretability
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training GMM detector on {X.shape[0]} samples with {X.shape[1]} features...")
        
        self.feature_names = feature_names
        
        # Initialize scaler if normalization is enabled
        if self.config.normalize_features:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.copy()
        
        # Initialize GMM model
        self.gmm_model = GaussianMixture(
            n_components=self.config.n_components,
            covariance_type=self.config.covariance_type,
            max_iter=self.config.max_iter,
            n_init=self.config.n_init,
            init_params=self.config.init_params,
            random_state=self.config.random_state,
            tol=self.config.tol,
            reg_covar=self.config.reg_covar
        )
        
        # Fit the model
        self.gmm_model.fit(X_scaled)
        
        # Compute threshold based on training data
        self._compute_threshold(X_scaled)
        
        self.is_fitted = True
        
        # Log model information
        logger.info(f"GMM training completed with {self.config.n_components} components")
        logger.info(f"Model converged: {self.gmm_model.converged_}")
        logger.info(f"Final log-likelihood: {self.gmm_model.score(X_scaled):.4f}")
        logger.info(f"Anomaly threshold: {self.threshold:.4f}")
        
        return self
    
    def _compute_threshold(self, X_scaled: np.ndarray) -> None:
        """Compute anomaly detection threshold based on training data."""
        # Compute log-likelihood scores for training data
        log_likelihoods = self.gmm_model.score_samples(X_scaled)
        
        # Method 1: Percentile-based threshold
        percentile_threshold = np.percentile(log_likelihoods, 
                                           self.config.contamination * 100)
        
        # Method 2: Statistical threshold based on chi-squared distribution
        # For multivariate Gaussian, -2 * log_likelihood follows chi-squared distribution
        chi2_threshold = chi2.ppf(self.config.confidence_level, X_scaled.shape[1])
        statistical_threshold = -chi2_threshold / 2
        
        # Use the more conservative threshold
        self.threshold = min(percentile_threshold, statistical_threshold)
        
        logger.info(f"Percentile threshold ({self.config.contamination*100}%): {percentile_threshold:.4f}")
        logger.info(f"Statistical threshold ({self.config.confidence_level*100}% confidence): {statistical_threshold:.4f}")
        logger.info(f"Selected threshold: {self.threshold:.4f}")
    
    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores for input data.
        
        Args:
            X: Input data
            
        Returns:
            Anomaly scores (higher = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
        
        # Scale features if normalization was used during training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.copy()
        
        # Compute log-likelihood scores
        log_likelihoods = self.gmm_model.score_samples(X_scaled)
        
        # Convert to anomaly scores (higher = more anomalous)
        anomaly_scores = -log_likelihoods
        
        return anomaly_scores
    
    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict anomalies in input data.
        
        Args:
            X: Input data
            threshold: Decision threshold (if None, uses computed threshold)
            
        Returns:
            Binary predictions (1 = anomaly, 0 = normal)
        """
        # Scale features if normalization was used during training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.copy()
        
        # Compute log-likelihood scores
        log_likelihoods = self.gmm_model.score_samples(X_scaled)
        
        # Use provided threshold or computed threshold
        if threshold is None:
            threshold = self.threshold
        
        # Predict anomalies (log-likelihood below threshold)
        predictions = (log_likelihoods < threshold).astype(int)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input data.
        
        Args:
            X: Input data
            
        Returns:
            Class probabilities [P(normal), P(anomaly)]
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
        
        # Scale features if normalization was used during training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.copy()
        
        # Compute log-likelihood scores
        log_likelihoods = self.gmm_model.score_samples(X_scaled)
        
        # Convert to probabilities using sigmoid-like transformation
        # Higher log-likelihood -> higher probability of being normal
        normalized_scores = (log_likelihoods - self.threshold) / np.std(log_likelihoods)
        prob_normal = 1 / (1 + np.exp(-normalized_scores))
        prob_anomaly = 1 - prob_normal
        
        return np.column_stack([prob_normal, prob_anomaly])
    
    def get_component_responsibilities(self, X: np.ndarray) -> np.ndarray:
        """
        Get the responsibility of each component for each sample.
        
        Args:
            X: Input data
            
        Returns:
            Component responsibilities (n_samples, n_components)
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
        
        # Scale features if normalization was used during training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.copy()
        
        return self.gmm_model.predict_proba(X_scaled)
    
    def get_feature_importance(self, X: np.ndarray, method: str = "gradient") -> np.ndarray:
        """
        Compute feature importance for anomaly detection.
        
        Args:
            X: Input data
            method: Method for computing importance ("gradient" or "permutation")
            
        Returns:
            Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before computing feature importance")
        
        if method == "gradient":
            return self._compute_gradient_importance(X)
        elif method == "permutation":
            return self._compute_permutation_importance(X)
        else:
            raise ValueError(f"Unknown importance method: {method}")
    
    def _compute_gradient_importance(self, X: np.ndarray) -> np.ndarray:
        """Compute feature importance using gradient-based method."""
        # Scale features if normalization was used during training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.copy()
        
        # Compute numerical gradients of log-likelihood w.r.t. features
        epsilon = 1e-6
        gradients = np.zeros_like(X_scaled)
        
        baseline_scores = self.gmm_model.score_samples(X_scaled)
        
        for feature_idx in range(X_scaled.shape[1]):
            # Perturb feature slightly
            X_perturbed = X_scaled.copy()
            X_perturbed[:, feature_idx] += epsilon
            
            # Compute perturbed scores
            perturbed_scores = self.gmm_model.score_samples(X_perturbed)
            
            # Compute gradient
            gradients[:, feature_idx] = (perturbed_scores - baseline_scores) / epsilon
        
        # Feature importance as mean absolute gradient
        importance_scores = np.mean(np.abs(gradients), axis=0)
        
        # Normalize importance scores
        if np.sum(importance_scores) > 0:
            importance_scores = importance_scores / np.sum(importance_scores)
        
        return importance_scores
    
    def _compute_permutation_importance(self, X: np.ndarray) -> np.ndarray:
        """Compute feature importance using permutation method."""
        baseline_scores = self.predict_score(X)
        baseline_mean = np.mean(baseline_scores)
        
        importance_scores = np.zeros(X.shape[1])
        
        for feature_idx in range(X.shape[1]):
            # Create permuted version
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, feature_idx])
            
            # Compute scores with permuted feature
            permuted_scores = self.predict_score(X_permuted)
            permuted_mean = np.mean(permuted_scores)
            
            # Importance is the change in mean score
            importance_scores[feature_idx] = abs(permuted_mean - baseline_mean)
        
        # Normalize importance scores
        if np.sum(importance_scores) > 0:
            importance_scores = importance_scores / np.sum(importance_scores)
        
        return importance_scores
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the fitted model."""
        if not self.is_fitted:
            return {'is_fitted': False}
        
        info = {
            'is_fitted': True,
            'n_components': self.config.n_components,
            'covariance_type': self.config.covariance_type,
            'converged': self.gmm_model.converged_,
            'n_iter': self.gmm_model.n_iter_,
            'threshold': self.threshold,
            'contamination': self.config.contamination,
            'feature_normalization': self.config.normalize_features,
            'weights': self.gmm_model.weights_.tolist(),
            'means_shape': self.gmm_model.means_.shape,
            'covariances_shape': self.gmm_model.covariances_.shape
        }
        
        if self.feature_names:
            info['feature_names'] = self.feature_names
        
        return info
    
    def plot_components_2d(self, X: np.ndarray, feature_indices: Tuple[int, int] = (0, 1), 
                          figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot GMM components in 2D space.
        
        Args:
            X: Input data
            feature_indices: Indices of features to plot
            figsize: Figure size
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before plotting")
        
        if X.shape[1] < 2:
            raise ValueError("Need at least 2 features for 2D plotting")
        
        # Scale features if normalization was used during training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.copy()
        
        # Extract 2D features
        X_2d = X_scaled[:, list(feature_indices)]
        
        # Get predictions
        predictions = self.predict(X)
        responsibilities = self.get_component_responsibilities(X)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: Data points colored by anomaly prediction
        colors = ['blue' if pred == 0 else 'red' for pred in predictions]
        ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, alpha=0.6)
        ax1.set_title('Anomaly Detection Results')
        ax1.set_xlabel(f'Feature {feature_indices[0]}')
        ax1.set_ylabel(f'Feature {feature_indices[1]}')
        
        # Plot 2: Data points colored by dominant component
        dominant_components = np.argmax(responsibilities, axis=1)
        scatter = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=dominant_components, 
                            cmap='viridis', alpha=0.6)
        ax2.set_title('GMM Component Assignment')
        ax2.set_xlabel(f'Feature {feature_indices[0]}')
        ax2.set_ylabel(f'Feature {feature_indices[1]}')
        plt.colorbar(scatter, ax=ax2)
        
        # Plot component centers
        if self.config.covariance_type in ['full', 'tied']:
            means_2d = self.gmm_model.means_[:, list(feature_indices)]
            ax2.scatter(means_2d[:, 0], means_2d[:, 1], 
                       c='red', marker='x', s=100, linewidths=3)
        
        plt.tight_layout()
        plt.show()


def create_gmm_detector(
    n_components: int = 3,
    covariance_type: str = "full",
    contamination: float = 0.1,
    **kwargs
) -> GMMDetector:
    """
    Factory function to create a GMM detector with sensible defaults.
    
    Args:
        n_components: Number of mixture components
        covariance_type: Type of covariance parameters
        contamination: Expected proportion of outliers
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured GMM detector
    """
    config = GMMConfig(
        n_components=n_components,
        covariance_type=covariance_type,
        contamination=contamination,
        **kwargs
    )
    return GMMDetector(config)