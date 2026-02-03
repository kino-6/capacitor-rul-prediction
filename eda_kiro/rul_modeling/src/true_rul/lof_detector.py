"""
Local Outlier Factor (LOF) Anomaly Detector

This module implements LOF for anomaly detection in RUL prediction.
LOF identifies anomalies by measuring the local deviation of a data point
with respect to its neighbors.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


@dataclass
class LOFConfig:
    """Configuration for LOF detector."""
    n_neighbors: int = 20
    algorithm: str = "auto"  # "auto", "ball_tree", "kd_tree", "brute"
    leaf_size: int = 30
    metric: str = "minkowski"
    p: int = 2  # Parameter for Minkowski metric
    contamination: float = 0.1  # Expected proportion of outliers
    novelty: bool = True  # Set to True for novelty detection
    n_jobs: Optional[int] = None
    normalize_features: bool = True


class LOFDetector:
    """
    Local Outlier Factor anomaly detector.
    
    LOF identifies anomalies by measuring the local deviation of a data point
    with respect to its neighbors. It considers the density of the local
    neighborhood to determine if a point is an outlier.
    """
    
    def __init__(self, config: LOFConfig):
        self.config = config
        self.lof_model: Optional[LocalOutlierFactor] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_fitted = False
        self.feature_names: Optional[List[str]] = None
        
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'LOFDetector':
        """
        Fit the LOF detector on normal data.
        
        Args:
            X: Training data (normal samples only)
            feature_names: Optional feature names for interpretability
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training LOF detector on {X.shape[0]} samples with {X.shape[1]} features...")
        
        self.feature_names = feature_names
        
        # Initialize scaler if normalization is enabled
        if self.config.normalize_features:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.copy()
        
        # Initialize LOF model
        self.lof_model = LocalOutlierFactor(
            n_neighbors=self.config.n_neighbors,
            algorithm=self.config.algorithm,
            leaf_size=self.config.leaf_size,
            metric=self.config.metric,
            p=self.config.p,
            contamination=self.config.contamination,
            novelty=self.config.novelty,
            n_jobs=self.config.n_jobs
        )
        
        # Fit the model
        if self.config.novelty:
            # For novelty detection, fit without predicting
            self.lof_model.fit(X_scaled)
        else:
            # For outlier detection, fit and predict
            predictions = self.lof_model.fit_predict(X_scaled)
            logger.info(f"Training completed. Found {np.sum(predictions == -1)} outliers in training data")
        
        self.is_fitted = True
        logger.info("LOF training completed")
        return self
    
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
        
        if self.config.novelty:
            # For novelty detection, use decision_function
            scores = -self.lof_model.decision_function(X_scaled)
        else:
            # For outlier detection, use negative_outlier_factor_
            # Note: This requires the model to be fitted on the same data
            scores = -self.lof_model.negative_outlier_factor_
            if len(scores) != X.shape[0]:
                raise ValueError("For outlier detection mode, X must be the same as training data")
        
        return scores
    
    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict anomalies in input data.
        
        Args:
            X: Input data
            threshold: Decision threshold (if None, uses model's internal threshold)
            
        Returns:
            Binary predictions (1 = anomaly, 0 = normal)
        """
        if not self.config.novelty and threshold is not None:
            # For outlier detection with custom threshold
            scores = self.predict_score(X)
            return (scores > threshold).astype(int)
        
        # Scale features if normalization was used during training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.copy()
        
        if self.config.novelty:
            # For novelty detection
            predictions = self.lof_model.predict(X_scaled)
        else:
            # For outlier detection
            predictions = self.lof_model.fit_predict(X_scaled)
        
        # Convert sklearn format (-1, 1) to (1, 0)
        return (predictions == -1).astype(int)
    
    def get_feature_importance(self, X: np.ndarray, method: str = "permutation") -> np.ndarray:
        """
        Compute feature importance for anomaly detection.
        
        Args:
            X: Input data
            method: Method for computing importance ("permutation" or "distance")
            
        Returns:
            Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before computing feature importance")
        
        if method == "permutation":
            return self._compute_permutation_importance(X)
        elif method == "distance":
            return self._compute_distance_importance(X)
        else:
            raise ValueError(f"Unknown importance method: {method}")
    
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
    
    def _compute_distance_importance(self, X: np.ndarray) -> np.ndarray:
        """Compute feature importance based on distance contributions."""
        if not self.config.novelty:
            raise ValueError("Distance importance only available for novelty detection")
        
        # Scale features if normalization was used during training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.copy()
        
        # Get k-nearest neighbors for each point
        distances, indices = self.lof_model.kneighbors(X_scaled)
        
        # Compute feature-wise distance contributions
        importance_scores = np.zeros(X.shape[1])
        
        for i in range(X.shape[0]):
            neighbors = X_scaled[indices[i]]
            point = X_scaled[i:i+1]
            
            # Compute feature-wise distances to neighbors
            feature_distances = np.mean(np.abs(neighbors - point), axis=0)
            importance_scores += feature_distances
        
        # Normalize by number of samples
        importance_scores = importance_scores / X.shape[0]
        
        # Normalize to sum to 1
        if np.sum(importance_scores) > 0:
            importance_scores = importance_scores / np.sum(importance_scores)
        
        return importance_scores
    
    def get_neighbor_analysis(self, X: np.ndarray, sample_idx: int) -> Dict[str, Any]:
        """
        Analyze neighbors for a specific sample to understand anomaly detection.
        
        Args:
            X: Input data
            sample_idx: Index of sample to analyze
            
        Returns:
            Dictionary with neighbor analysis results
        """
        if not self.is_fitted or not self.config.novelty:
            raise ValueError("Neighbor analysis only available for fitted novelty detection")
        
        # Scale features if normalization was used during training
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.copy()
        
        sample = X_scaled[sample_idx:sample_idx+1]
        
        # Get neighbors
        distances, indices = self.lof_model.kneighbors(sample)
        
        # Get LOF score
        lof_score = -self.lof_model.decision_function(sample)[0]
        
        analysis = {
            'sample_index': sample_idx,
            'lof_score': lof_score,
            'is_anomaly': lof_score > 0,
            'neighbor_distances': distances[0],
            'neighbor_indices': indices[0],
            'mean_neighbor_distance': np.mean(distances[0]),
            'std_neighbor_distance': np.std(distances[0]),
            'max_neighbor_distance': np.max(distances[0]),
            'min_neighbor_distance': np.min(distances[0])
        }
        
        return analysis
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the fitted model."""
        if not self.is_fitted:
            return {'is_fitted': False}
        
        info = {
            'is_fitted': True,
            'n_neighbors': self.config.n_neighbors,
            'algorithm': self.config.algorithm,
            'metric': self.config.metric,
            'contamination': self.config.contamination,
            'novelty_detection': self.config.novelty,
            'feature_normalization': self.config.normalize_features,
            'n_features': self.lof_model.n_features_in_ if hasattr(self.lof_model, 'n_features_in_') else None
        }
        
        if self.feature_names:
            info['feature_names'] = self.feature_names
        
        return info


def create_lof_detector(
    n_neighbors: int = 20,
    contamination: float = 0.1,
    novelty: bool = True,
    **kwargs
) -> LOFDetector:
    """
    Factory function to create a LOF detector with sensible defaults.
    
    Args:
        n_neighbors: Number of neighbors to consider
        contamination: Expected proportion of outliers
        novelty: Whether to use novelty detection mode
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured LOF detector
    """
    config = LOFConfig(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=novelty,
        **kwargs
    )
    return LOFDetector(config)