"""
Feature normalization with capacitor-specific scalers
"""

from typing import Dict, Optional
import logging

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logger = logging.getLogger(__name__)


class FeatureNormalizer:
    """
    Feature normalizer with capacitor-specific scaling
    
    Supports both StandardScaler and MinMaxScaler with fallback to global scaler
    """
    
    def __init__(self, method: str = "standard"):
        """
        Initialize feature normalizer
        
        Args:
            method: Normalization method ("standard" or "minmax")
        """
        if method not in ["standard", "minmax"]:
            raise ValueError(f"Unknown normalization method: {method}")
        
        self.method = method
        self.scalers: Dict[str, any] = {}
        self.global_scaler: Optional[any] = None
        
        logger.info(f"FeatureNormalizer initialized with method={method}")
    
    def fit(
        self,
        features: np.ndarray,
        capacitor_id: str
    ) -> None:
        """
        Fit scaler for a specific capacitor
        
        Args:
            features: Feature array (n_samples, n_features)
            capacitor_id: Capacitor identifier
        """
        if len(features) == 0:
            logger.warning(f"Empty features for {capacitor_id}, skipping fit")
            return
        
        # Create scaler
        if self.method == "standard":
            scaler = StandardScaler()
        else:  # minmax
            scaler = MinMaxScaler()
        
        # Fit scaler
        scaler.fit(features)
        self.scalers[capacitor_id] = scaler
        
        logger.debug(f"Fitted scaler for {capacitor_id} with {len(features)} samples")
    
    def fit_global(self, features: np.ndarray) -> None:
        """
        Fit global scaler (fallback for unknown capacitors)
        
        Args:
            features: Feature array (n_samples, n_features)
        """
        if len(features) == 0:
            logger.warning("Empty features for global scaler, skipping fit")
            return
        
        # Create scaler
        if self.method == "standard":
            scaler = StandardScaler()
        else:  # minmax
            scaler = MinMaxScaler()
        
        # Fit scaler
        scaler.fit(features)
        self.global_scaler = scaler
        
        logger.info(f"Fitted global scaler with {len(features)} samples")
    
    def transform(
        self,
        features: np.ndarray,
        capacitor_id: str
    ) -> np.ndarray:
        """
        Transform features using capacitor-specific or global scaler
        
        Args:
            features: Feature array (n_samples, n_features)
            capacitor_id: Capacitor identifier
            
        Returns:
            Normalized feature array
        """
        if len(features) == 0:
            return features
        
        # Try capacitor-specific scaler first
        if capacitor_id in self.scalers:
            scaler = self.scalers[capacitor_id]
            logger.debug(f"Using capacitor-specific scaler for {capacitor_id}")
        elif self.global_scaler is not None:
            scaler = self.global_scaler
            logger.debug(f"Using global scaler for {capacitor_id}")
        else:
            raise ValueError(
                f"No scaler available for {capacitor_id} and no global scaler fitted"
            )
        
        return scaler.transform(features)
    
    def fit_transform(
        self,
        features: np.ndarray,
        capacitor_id: str
    ) -> np.ndarray:
        """
        Fit and transform features in one step
        
        Args:
            features: Feature array (n_samples, n_features)
            capacitor_id: Capacitor identifier
            
        Returns:
            Normalized feature array
        """
        self.fit(features, capacitor_id)
        return self.transform(features, capacitor_id)
    
    def has_scaler(self, capacitor_id: str) -> bool:
        """
        Check if scaler exists for capacitor
        
        Args:
            capacitor_id: Capacitor identifier
            
        Returns:
            True if scaler exists
        """
        return capacitor_id in self.scalers
    
    def has_global_scaler(self) -> bool:
        """Check if global scaler exists"""
        return self.global_scaler is not None
    
    @property
    def n_scalers(self) -> int:
        """Get number of fitted scalers"""
        return len(self.scalers)


def normalize_features(
    features: np.ndarray,
    capacitor_id: str,
    normalizer: Optional[FeatureNormalizer] = None,
    method: str = "standard",
    fit: bool = False
) -> tuple[np.ndarray, FeatureNormalizer]:
    """
    Convenience function to normalize features
    
    Args:
        features: Feature array
        capacitor_id: Capacitor identifier
        normalizer: Existing normalizer (optional)
        method: Normalization method
        fit: Whether to fit the normalizer
        
    Returns:
        Tuple of (normalized_features, normalizer)
    """
    if normalizer is None:
        normalizer = FeatureNormalizer(method=method)
    
    if fit:
        normalized = normalizer.fit_transform(features, capacitor_id)
    else:
        normalized = normalizer.transform(features, capacitor_id)
    
    return normalized, normalizer
