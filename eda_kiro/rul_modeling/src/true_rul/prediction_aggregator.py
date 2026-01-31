"""
Prediction Aggregator for True RUL Prediction System

This module implements the PredictionAggregator class that combines RUL predictions
and anomaly detection results into a unified prediction result.
"""

from typing import List, Dict, Optional
from datetime import datetime
import numpy as np

from .data_structures import PredictionResult


class PredictionAggregator:
    """
    Aggregates predictions from RUL regression and anomaly detection models
    
    This class combines:
    - RUL predictions with confidence intervals
    - Anomaly detection results (binary flag and continuous score)
    - Feature importance from both models
    - Historical degradation context
    
    The aggregator maps continuous scores to discrete degradation stages
    and provides a unified prediction interface.
    """
    
    def __init__(self, model_version: str = "1.0.0"):
        """
        Initialize the prediction aggregator
        
        Args:
            model_version: Version identifier for the model ensemble
        """
        self.model_version = model_version
        
        # Degradation stage thresholds
        self.degradation_thresholds = {
            'healthy': (0.0, 0.25),
            'early_degradation': (0.25, 0.5),
            'advanced_degradation': (0.5, 0.75),
            'critical': (0.75, 1.0)
        }
    
    def aggregate(
        self,
        rul_pred: float,
        rul_confidence_lower: float,
        rul_confidence_upper: float,
        anomaly_flag: bool,
        anomaly_score: float,
        feature_importance: Dict[str, float],
        degradation_history: Optional[List[float]] = None,
        capacitor_id: Optional[str] = None,
        cycle_number: Optional[int] = None
    ) -> PredictionResult:
        """
        Aggregate predictions from multiple models into a unified result
        
        Args:
            rul_pred: RUL prediction from regression model (cycles)
            rul_confidence_lower: Lower bound of confidence interval
            rul_confidence_upper: Upper bound of confidence interval
            anomaly_flag: Binary anomaly flag from ensemble detector
            anomaly_score: Continuous anomaly score (0-1)
            feature_importance: Dictionary of feature importance scores
            degradation_history: Historical degradation scores for context
            capacitor_id: Optional capacitor identifier
            cycle_number: Optional cycle number
            
        Returns:
            PredictionResult object with aggregated information
        """
        # Compute degradation score by combining RUL and anomaly information
        degradation_score = self._compute_degradation_score(
            rul_pred, anomaly_score, degradation_history
        )
        
        # Determine degradation stage
        degradation_stage = self.compute_degradation_stage(
            rul_pred, anomaly_score, degradation_score
        )
        
        # Ensure RUL values are non-negative integers
        rul_cycles = max(0, int(round(rul_pred)))
        rul_conf_lower = max(0, int(round(rul_confidence_lower)))
        rul_conf_upper = max(rul_cycles, int(round(rul_confidence_upper)))
        
        # Create prediction result
        return PredictionResult(
            rul_cycles=rul_cycles,
            rul_confidence_lower=rul_conf_lower,
            rul_confidence_upper=rul_conf_upper,
            degradation_score=degradation_score,
            degradation_stage=degradation_stage,
            anomaly_flag=anomaly_flag,
            anomaly_score=anomaly_score,
            feature_importance=feature_importance,
            timestamp=datetime.now(),
            model_version=self.model_version,
            capacitor_id=capacitor_id,
            cycle_number=cycle_number
        )
    
    def compute_degradation_stage(
        self,
        rul: float,
        anomaly_score: float,
        degradation_score: Optional[float] = None
    ) -> str:
        """
        Compute degradation stage based on RUL and anomaly score
        
        The degradation stage is determined by:
        1. Primary: degradation_score (if provided)
        2. Secondary: combination of RUL and anomaly_score
        
        Args:
            rul: Remaining useful life in cycles
            anomaly_score: Continuous anomaly score (0-1)
            degradation_score: Optional pre-computed degradation score
            
        Returns:
            One of: "healthy", "early_degradation", "advanced_degradation", "critical"
        """
        if degradation_score is None:
            # Compute degradation score from RUL and anomaly score
            degradation_score = self._compute_degradation_score(rul, anomaly_score)
        
        # Map degradation score to stage
        for stage, (min_thresh, max_thresh) in self.degradation_thresholds.items():
            if min_thresh <= degradation_score < max_thresh:
                return stage
        
        # Handle edge case where score equals 1.0
        if degradation_score >= 1.0:
            return "critical"
        
        # Fallback (should not happen with valid inputs)
        return "healthy"
    
    def _compute_degradation_score(
        self,
        rul: float,
        anomaly_score: float,
        degradation_history: Optional[List[float]] = None
    ) -> float:
        """
        Compute continuous degradation score (0-1) from RUL and anomaly information
        
        The degradation score combines:
        1. RUL-based component: Higher degradation for lower RUL
        2. Anomaly-based component: Direct anomaly score
        3. Trend component: Rate of degradation increase (if history available)
        
        Args:
            rul: Remaining useful life in cycles
            anomaly_score: Continuous anomaly score (0-1)
            degradation_history: Optional historical degradation scores
            
        Returns:
            Degradation score in range [0, 1]
        """
        # RUL-based component (inverse relationship)
        # Assume maximum expected RUL is around 200 cycles for ES12 dataset
        max_expected_rul = 200.0
        rul_component = max(0.0, 1.0 - (rul / max_expected_rul))
        rul_component = min(1.0, rul_component)  # Clamp to [0, 1]
        
        # Anomaly-based component (direct relationship)
        anomaly_component = max(0.0, min(1.0, anomaly_score))
        
        # Trend component (if history available)
        trend_component = 0.0
        if degradation_history and len(degradation_history) >= 2:
            # Compute recent trend (positive = increasing degradation)
            recent_trend = degradation_history[-1] - degradation_history[-2]
            trend_component = max(0.0, min(0.2, recent_trend))  # Cap at 0.2
        
        # Weighted combination
        # RUL: 40%, Anomaly: 50%, Trend: 10%
        degradation_score = (
            0.4 * rul_component +
            0.5 * anomaly_component +
            0.1 * trend_component
        )
        
        # Ensure score is in valid range
        return max(0.0, min(1.0, degradation_score))
    
    def update_degradation_thresholds(
        self,
        healthy: tuple = (0.0, 0.25),
        early_degradation: tuple = (0.25, 0.5),
        advanced_degradation: tuple = (0.5, 0.75),
        critical: tuple = (0.75, 1.0)
    ):
        """
        Update degradation stage thresholds
        
        Args:
            healthy: (min, max) thresholds for healthy stage
            early_degradation: (min, max) thresholds for early degradation
            advanced_degradation: (min, max) thresholds for advanced degradation
            critical: (min, max) thresholds for critical stage
        """
        self.degradation_thresholds = {
            'healthy': healthy,
            'early_degradation': early_degradation,
            'advanced_degradation': advanced_degradation,
            'critical': critical
        }
    
    def get_stage_info(self, stage: str) -> Dict[str, float]:
        """
        Get threshold information for a degradation stage
        
        Args:
            stage: Degradation stage name
            
        Returns:
            Dictionary with 'min' and 'max' threshold values
        """
        if stage not in self.degradation_thresholds:
            raise ValueError(f"Unknown degradation stage: {stage}")
        
        min_thresh, max_thresh = self.degradation_thresholds[stage]
        return {'min': min_thresh, 'max': max_thresh}