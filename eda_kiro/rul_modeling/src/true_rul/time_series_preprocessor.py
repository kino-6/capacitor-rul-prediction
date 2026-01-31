"""
Time-Series Preprocessor for True RUL Prediction

This module provides temporal feature creation from cycle history,
including rolling statistics and trend features.
"""

import logging
from typing import List, Dict, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from .data_structures import CycleData
from .config import MODEL_CONFIG

logger = logging.getLogger(__name__)


class TimeSeriesPreprocessor:
    """
    Prepare time-series data and create temporal features
    
    This class creates temporal features from cycle history including:
    - Rolling statistics (mean, std, min, max)
    - Recent trend (current - previous cycle)
    - Long-term trend (current - 5 cycles ago)
    
    Attributes:
        rolling_window: Window size for rolling statistics
        normalization: Normalization method ("standard" or "minmax")
        scalers: Dictionary of capacitor-specific scalers
    """
    
    def __init__(
        self,
        rolling_window: int = 5,
        normalization: str = "standard"
    ):
        """
        Initialize TimeSeriesPreprocessor
        
        Args:
            rolling_window: Window size for rolling statistics (default: 5)
            normalization: Normalization method - "standard" or "minmax" (default: "standard")
        """
        self.rolling_window = rolling_window
        self.normalization = normalization
        self.scalers: Dict[str, StandardScaler] = {}
        
        logger.info(
            f"TimeSeriesPreprocessor initialized: "
            f"window={rolling_window}, normalization={normalization}"
        )
    
    def create_temporal_features(
        self,
        cycles: List[CycleData],
        features: np.ndarray
    ) -> np.ndarray:
        """
        Create temporal features from cycle history
        
        This method computes:
        - Rolling mean, std, min, max over the rolling window
        - Recent trend: difference between current and previous cycle
        - Long-term trend: difference between current and 5 cycles ago
        
        Args:
            cycles: List of CycleData objects (ordered by cycle number)
            features: Extracted features for each cycle (n_cycles, n_features)
            
        Returns:
            Enhanced feature array with temporal features (n_cycles, n_features * 7)
            where 7 = 1 (original) + 4 (rolling stats) + 2 (trends)
            
        Raises:
            ValueError: If cycles and features have mismatched lengths
        """
        if len(cycles) != len(features):
            raise ValueError(
                f"Cycles length ({len(cycles)}) must match features length ({len(features)})"
            )
        
        if len(cycles) == 0:
            raise ValueError("Cannot create temporal features from empty cycle list")
        
        n_cycles = len(cycles)
        n_features = features.shape[1] if len(features.shape) > 1 else features.shape[0]
        
        temporal_features = []
        
        for i in range(n_cycles):
            # Get rolling window of previous cycles (including current)
            window_start = max(0, i - self.rolling_window + 1)
            window_features = features[window_start:i+1]
            
            # Compute rolling statistics
            rolling_mean = np.mean(window_features, axis=0)
            rolling_std = np.std(window_features, axis=0)
            rolling_min = np.min(window_features, axis=0)
            rolling_max = np.max(window_features, axis=0)
            
            # Compute trend features
            if i >= 1:
                # Recent trend: current - previous cycle
                recent_trend = features[i] - features[i-1]
            else:
                # No previous cycle, use zeros
                recent_trend = np.zeros_like(features[i])
            
            if i >= 5:
                # Long-term trend: current - 5 cycles ago
                long_trend = features[i] - features[i-5]
            else:
                # Not enough history, use zeros
                long_trend = np.zeros_like(features[i])
            
            # Concatenate all temporal features
            # Order: original, rolling_mean, rolling_std, rolling_min, rolling_max, recent_trend, long_trend
            temp_feat = np.concatenate([
                features[i],
                rolling_mean,
                rolling_std,
                rolling_min,
                rolling_max,
                recent_trend,
                long_trend
            ])
            
            temporal_features.append(temp_feat)
        
        result = np.array(temporal_features)
        
        logger.debug(
            f"Created temporal features: input shape {features.shape}, "
            f"output shape {result.shape}"
        )
        
        return result
    
    def normalize_features(
        self,
        features: np.ndarray,
        capacitor_id: str,
        fit: bool = False
    ) -> np.ndarray:
        """
        Normalize features using capacitor-specific statistics
        
        Args:
            features: Feature array to normalize (n_samples, n_features)
            capacitor_id: Capacitor identifier for scaler lookup
            fit: Whether to fit the scaler (training) or use existing (inference)
            
        Returns:
            Normalized feature array with same shape as input
            
        Raises:
            ValueError: If no scaler is available and fit=False
        """
        if features.shape[0] == 0:
            raise ValueError("Cannot normalize empty feature array")
        
        # Fit scaler if requested
        if fit:
            if self.normalization == "standard":
                scaler = StandardScaler()
            elif self.normalization == "minmax":
                scaler = MinMaxScaler()
            else:
                raise ValueError(
                    f"Unknown normalization method: {self.normalization}. "
                    f"Must be 'standard' or 'minmax'"
                )
            
            self.scalers[capacitor_id] = scaler.fit(features)
            logger.info(f"Fitted {self.normalization} scaler for capacitor {capacitor_id}")
        
        # Get scaler for this capacitor
        if capacitor_id not in self.scalers:
            # Fallback to global scaler
            if "global" in self.scalers:
                logger.warning(
                    f"No scaler found for capacitor {capacitor_id}, using global scaler"
                )
                capacitor_id = "global"
            else:
                raise ValueError(
                    f"No scaler available for capacitor {capacitor_id} and no global scaler. "
                    f"Call normalize_features with fit=True first."
                )
        
        # Transform features
        normalized = self.scalers[capacitor_id].transform(features)
        
        logger.debug(
            f"Normalized features for {capacitor_id}: "
            f"shape {features.shape} -> {normalized.shape}"
        )
        
        return normalized
    
    def fit_global_scaler(self, features: np.ndarray) -> None:
        """
        Fit a global scaler on all training data
        
        This is useful as a fallback when capacitor-specific scalers are not available.
        
        Args:
            features: Feature array from all capacitors (n_samples, n_features)
        """
        if features.shape[0] == 0:
            raise ValueError("Cannot fit global scaler on empty feature array")
        
        if self.normalization == "standard":
            scaler = StandardScaler()
        elif self.normalization == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(
                f"Unknown normalization method: {self.normalization}. "
                f"Must be 'standard' or 'minmax'"
            )
        
        self.scalers["global"] = scaler.fit(features)
        logger.info(f"Fitted global {self.normalization} scaler on {features.shape[0]} samples")
    
    def get_scaler(self, capacitor_id: str) -> Optional[StandardScaler]:
        """
        Get the scaler for a specific capacitor
        
        Args:
            capacitor_id: Capacitor identifier
            
        Returns:
            Scaler object or None if not found
        """
        return self.scalers.get(capacitor_id)
    
    def has_scaler(self, capacitor_id: str) -> bool:
        """
        Check if a scaler exists for a capacitor
        
        Args:
            capacitor_id: Capacitor identifier
            
        Returns:
            True if scaler exists, False otherwise
        """
        return capacitor_id in self.scalers
    
    def reset_scalers(self) -> None:
        """
        Reset all scalers
        
        This is useful when retraining models from scratch.
        """
        self.scalers.clear()
        logger.info("All scalers have been reset")
    
    def get_temporal_feature_names(self, base_feature_names: List[str]) -> List[str]:
        """
        Get names for temporal features
        
        Args:
            base_feature_names: List of base feature names
            
        Returns:
            List of all feature names including temporal features
        """
        feature_names = []
        
        # Original features
        feature_names.extend(base_feature_names)
        
        # Rolling statistics
        for stat in ['rolling_mean', 'rolling_std', 'rolling_min', 'rolling_max']:
            feature_names.extend([f"{name}_{stat}" for name in base_feature_names])
        
        # Trend features
        for trend in ['recent_trend', 'long_trend']:
            feature_names.extend([f"{name}_{trend}" for name in base_feature_names])
        
        return feature_names
    
    @property
    def n_scalers(self) -> int:
        """Get number of fitted scalers"""
        return len(self.scalers)
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"TimeSeriesPreprocessor("
            f"rolling_window={self.rolling_window}, "
            f"normalization='{self.normalization}', "
            f"n_scalers={self.n_scalers})"
        )
