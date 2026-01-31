"""
Out-of-Distribution (OOD) Detector for RUL Prediction System

This module implements out-of-distribution detection to identify when input
data significantly differs from the training distribution.

Requirements: 8.4
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


class OutOfDistributionDetector:
    """
    Detector for out-of-distribution samples
    
    This class computes training data statistics and flags samples that
    deviate significantly from the training distribution.
    
    Attributes:
        feature_stats: Dictionary containing training data statistics
        threshold_std: Number of standard deviations for OOD threshold
        is_fitted: Whether the detector has been fitted on training data
    """
    
    def __init__(
        self,
        threshold_std: float = 3.0,
        min_samples_for_stats: int = 10
    ):
        """
        Initialize OOD detector
        
        Args:
            threshold_std: Number of standard deviations for OOD threshold
            min_samples_for_stats: Minimum samples needed to compute reliable stats
        """
        self.threshold_std = threshold_std
        self.min_samples_for_stats = min_samples_for_stats
        
        # Training data statistics
        self.feature_stats: Dict[str, Dict[str, float]] = {}
        self.global_stats: Dict[str, float] = {}
        self.is_fitted = False
        
        # Feature names for interpretability
        self.feature_names: List[str] = []
        
        logger.info(f"OOD detector initialized with threshold={threshold_std} std")
    
    def fit(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> None:
        """
        Fit the OOD detector on training data
        
        Args:
            X: Training features (n_samples, n_features)
            feature_names: Optional feature names for interpretability
        """
        if X.shape[0] < self.min_samples_for_stats:
            raise ValueError(
                f"Need at least {self.min_samples_for_stats} samples for reliable statistics, "
                f"got {X.shape[0]}"
            )
        
        n_samples, n_features = X.shape
        
        # Set feature names
        if feature_names is not None:
            if len(feature_names) != n_features:
                raise ValueError(
                    f"Number of feature names ({len(feature_names)}) must match "
                    f"number of features ({n_features})"
                )
            self.feature_names = feature_names.copy()
        else:
            self.feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Compute per-feature statistics
        self.feature_stats = {}
        
        for i, feature_name in enumerate(self.feature_names):
            feature_values = X[:, i]
            
            # Remove NaN values for statistics computation
            valid_values = feature_values[~np.isnan(feature_values)]
            
            if len(valid_values) < self.min_samples_for_stats:
                logger.warning(
                    f"Feature {feature_name} has only {len(valid_values)} valid values. "
                    "Using global fallback statistics."
                )
                # Use global statistics as fallback
                self.feature_stats[feature_name] = {
                    "mean": 0.0,
                    "std": 1.0,
                    "min": -np.inf,
                    "max": np.inf,
                    "q25": -1.0,
                    "q75": 1.0,
                    "n_valid": len(valid_values)
                }
            else:
                self.feature_stats[feature_name] = {
                    "mean": float(np.mean(valid_values)),
                    "std": float(np.std(valid_values)),
                    "min": float(np.min(valid_values)),
                    "max": float(np.max(valid_values)),
                    "q25": float(np.percentile(valid_values, 25)),
                    "q75": float(np.percentile(valid_values, 75)),
                    "n_valid": len(valid_values)
                }
        
        # Compute global statistics for fallback
        all_valid_values = X[~np.isnan(X)]
        if len(all_valid_values) > 0:
            self.global_stats = {
                "mean": float(np.mean(all_valid_values)),
                "std": float(np.std(all_valid_values)),
                "min": float(np.min(all_valid_values)),
                "max": float(np.max(all_valid_values))
            }
        else:
            self.global_stats = {
                "mean": 0.0,
                "std": 1.0,
                "min": -np.inf,
                "max": np.inf
            }
        
        self.is_fitted = True
        
        logger.info(
            f"OOD detector fitted on {n_samples} samples with {n_features} features"
        )
    
    def is_out_of_distribution(
        self,
        X: np.ndarray,
        return_details: bool = False
    ) -> np.ndarray:
        """
        Check if samples are out of distribution
        
        Args:
            X: Input features (n_samples, n_features)
            return_details: Whether to return detailed OOD information
            
        Returns:
            Boolean array indicating OOD samples, or tuple with details if return_details=True
        """
        if not self.is_fitted:
            raise RuntimeError("OOD detector has not been fitted. Call fit() first.")
        
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Input features ({X.shape[1]}) must match fitted features "
                f"({len(self.feature_names)})"
            )
        
        n_samples = X.shape[0]
        ood_flags = np.zeros(n_samples, dtype=bool)
        
        if return_details:
            ood_details = []
        
        for sample_idx in range(n_samples):
            sample = X[sample_idx]
            sample_ood = False
            sample_details = {
                "sample_idx": sample_idx,
                "ood_features": [],
                "feature_deviations": {}
            }
            
            for feature_idx, (feature_name, feature_value) in enumerate(
                zip(self.feature_names, sample)
            ):
                # Skip NaN values
                if np.isnan(feature_value):
                    continue
                
                stats = self.feature_stats[feature_name]
                
                # Check if feature value is within acceptable range
                if stats["std"] > 0:
                    # Use z-score based detection
                    z_score = abs(feature_value - stats["mean"]) / stats["std"]
                    is_ood = z_score > self.threshold_std
                else:
                    # Use range-based detection if std is 0
                    is_ood = (
                        feature_value < stats["min"] or 
                        feature_value > stats["max"]
                    )
                    z_score = float('inf') if is_ood else 0.0
                
                if is_ood:
                    sample_ood = True
                    sample_details["ood_features"].append(feature_name)
                
                sample_details["feature_deviations"][feature_name] = {
                    "value": float(feature_value),
                    "z_score": float(z_score),
                    "is_ood": is_ood,
                    "expected_range": [
                        stats["mean"] - self.threshold_std * stats["std"],
                        stats["mean"] + self.threshold_std * stats["std"]
                    ]
                }
            
            ood_flags[sample_idx] = sample_ood
            
            if return_details:
                sample_details["is_ood"] = sample_ood
                ood_details.append(sample_details)
        
        if return_details:
            return ood_flags, ood_details
        else:
            return ood_flags
    
    def get_ood_score(self, X: np.ndarray) -> np.ndarray:
        """
        Get OOD scores for samples (higher = more out of distribution)
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            OOD scores array (n_samples,)
        """
        if not self.is_fitted:
            raise RuntimeError("OOD detector has not been fitted. Call fit() first.")
        
        n_samples = X.shape[0]
        ood_scores = np.zeros(n_samples)
        
        for sample_idx in range(n_samples):
            sample = X[sample_idx]
            max_z_score = 0.0
            
            for feature_idx, (feature_name, feature_value) in enumerate(
                zip(self.feature_names, sample)
            ):
                # Skip NaN values
                if np.isnan(feature_value):
                    continue
                
                stats = self.feature_stats[feature_name]
                
                # Compute z-score
                if stats["std"] > 0:
                    z_score = abs(feature_value - stats["mean"]) / stats["std"]
                else:
                    # Use normalized distance from range if std is 0
                    if feature_value < stats["min"]:
                        z_score = abs(feature_value - stats["min"])
                    elif feature_value > stats["max"]:
                        z_score = abs(feature_value - stats["max"])
                    else:
                        z_score = 0.0
                
                max_z_score = max(max_z_score, z_score)
            
            # Normalize score to [0, 1] range
            ood_scores[sample_idx] = min(1.0, max_z_score / self.threshold_std)
        
        return ood_scores
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """
        Get feature statistics summary
        
        Returns:
            Dictionary with feature statistics
        """
        if not self.is_fitted:
            return {"error": "Detector not fitted"}
        
        return {
            "n_features": len(self.feature_names),
            "feature_names": self.feature_names.copy(),
            "feature_stats": self.feature_stats.copy(),
            "global_stats": self.global_stats.copy(),
            "threshold_std": self.threshold_std,
            "is_fitted": self.is_fitted
        }
    
    def save(self, filepath: str) -> None:
        """
        Save OOD detector to disk
        
        Args:
            filepath: Path to save the detector
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted detector")
        
        detector_data = {
            "threshold_std": self.threshold_std,
            "min_samples_for_stats": self.min_samples_for_stats,
            "feature_stats": self.feature_stats,
            "global_stats": self.global_stats,
            "feature_names": self.feature_names,
            "is_fitted": self.is_fitted
        }
        
        # Ensure directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(detector_data, f)
        
        logger.info(f"OOD detector saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'OutOfDistributionDetector':
        """
        Load OOD detector from disk
        
        Args:
            filepath: Path to load the detector from
            
        Returns:
            Loaded OOD detector instance
        """
        with open(filepath, 'rb') as f:
            detector_data = pickle.load(f)
        
        # Create new instance
        detector = cls(
            threshold_std=detector_data["threshold_std"],
            min_samples_for_stats=detector_data["min_samples_for_stats"]
        )
        
        # Restore state
        detector.feature_stats = detector_data["feature_stats"]
        detector.global_stats = detector_data["global_stats"]
        detector.feature_names = detector_data["feature_names"]
        detector.is_fitted = detector_data["is_fitted"]
        
        logger.info(f"OOD detector loaded from {filepath}")
        return detector
    
    def reset(self) -> None:
        """Reset the detector to unfitted state"""
        self.feature_stats.clear()
        self.global_stats.clear()
        self.feature_names.clear()
        self.is_fitted = False
        
        logger.info("OOD detector reset")
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"OutOfDistributionDetector(fitted={self.is_fitted}, "
            f"n_features={len(self.feature_names)}, "
            f"threshold_std={self.threshold_std})"
        )