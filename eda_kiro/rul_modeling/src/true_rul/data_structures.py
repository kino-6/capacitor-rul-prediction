"""
Data structures for True RUL Prediction System
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
import numpy as np


@dataclass
class CycleData:
    """
    Data for a single charge-discharge cycle
    
    Attributes:
        cycle_number: Cycle number (1-based indexing)
        vl_series: Input voltage time-series data
        vo_series: Output voltage time-series data
        timestamp: Timestamp of the cycle measurement
    """
    cycle_number: int
    vl_series: np.ndarray
    vo_series: np.ndarray
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate data after initialization"""
        if self.cycle_number < 1:
            raise ValueError(f"cycle_number must be >= 1, got {self.cycle_number}")
        
        if len(self.vl_series) != len(self.vo_series):
            raise ValueError(
                f"VL and VO series must have same length: "
                f"VL={len(self.vl_series)}, VO={len(self.vo_series)}"
            )
        
        if len(self.vl_series) == 0:
            raise ValueError("VL and VO series cannot be empty")


@dataclass
class CapacitorData:
    """
    Data for a single capacitor across all cycles
    
    Attributes:
        capacitor_id: Capacitor identifier (e.g., "ES12C1")
        cycles: List of CycleData objects ordered by cycle number
        total_cycles: Total number of cycles
    """
    capacitor_id: str
    cycles: List[CycleData]
    total_cycles: int
    
    def __post_init__(self):
        """Validate data after initialization"""
        if not self.capacitor_id:
            raise ValueError("capacitor_id cannot be empty")
        
        if self.total_cycles != len(self.cycles):
            raise ValueError(
                f"total_cycles ({self.total_cycles}) does not match "
                f"number of cycles ({len(self.cycles)})"
            )
        
        # Verify cycles are ordered
        for i, cycle in enumerate(self.cycles):
            expected_cycle_num = i + 1
            if cycle.cycle_number != expected_cycle_num:
                raise ValueError(
                    f"Cycles must be ordered: expected cycle {expected_cycle_num}, "
                    f"got {cycle.cycle_number}"
                )
    
    def get_cycle(self, cycle_number: int) -> Optional[CycleData]:
        """
        Get data for a specific cycle
        
        Args:
            cycle_number: Cycle number (1-based)
            
        Returns:
            CycleData object or None if not found
        """
        if 1 <= cycle_number <= self.total_cycles:
            return self.cycles[cycle_number - 1]
        return None
    
    def get_cycles_range(self, start: int, end: int) -> List[CycleData]:
        """
        Get data for a range of cycles
        
        Args:
            start: Start cycle number (inclusive, 1-based)
            end: End cycle number (inclusive, 1-based)
            
        Returns:
            List of CycleData objects
        """
        if start < 1 or end > self.total_cycles or start > end:
            raise ValueError(
                f"Invalid cycle range: start={start}, end={end}, "
                f"total_cycles={self.total_cycles}"
            )
        return self.cycles[start - 1:end]


@dataclass
class PredictionResult:
    """
    Result of RUL prediction for a single cycle
    
    Attributes:
        rul_cycles: Predicted remaining useful life in cycles
        rul_confidence_lower: Lower bound of 95% confidence interval
        rul_confidence_upper: Upper bound of 95% confidence interval
        degradation_score: Continuous degradation score (0-1)
        degradation_stage: Degradation stage (healthy/early/advanced/critical)
        anomaly_flag: Binary anomaly flag (True if anomalous)
        anomaly_score: Continuous anomaly score
        feature_importance: Dictionary of feature names to importance scores
        timestamp: Timestamp of prediction
        model_version: Version of the model used
        capacitor_id: Capacitor identifier
        cycle_number: Cycle number
    """
    rul_cycles: int
    rul_confidence_lower: int
    rul_confidence_upper: int
    degradation_score: float
    degradation_stage: str
    anomaly_flag: bool
    anomaly_score: float
    feature_importance: Dict[str, float]
    timestamp: datetime
    model_version: str
    capacitor_id: Optional[str] = None
    cycle_number: Optional[int] = None
    
    def __post_init__(self):
        """Validate data after initialization"""
        if self.rul_cycles < 0:
            raise ValueError(f"rul_cycles must be >= 0, got {self.rul_cycles}")
        
        if self.rul_confidence_lower < 0:
            raise ValueError(
                f"rul_confidence_lower must be >= 0, got {self.rul_confidence_lower}"
            )
        
        if self.rul_confidence_upper < self.rul_cycles:
            raise ValueError(
                f"rul_confidence_upper ({self.rul_confidence_upper}) must be >= "
                f"rul_cycles ({self.rul_cycles})"
            )
        
        if not 0 <= self.degradation_score <= 1:
            raise ValueError(
                f"degradation_score must be in [0, 1], got {self.degradation_score}"
            )
        
        valid_stages = {"healthy", "early_degradation", "advanced_degradation", "critical"}
        if self.degradation_stage not in valid_stages:
            raise ValueError(
                f"degradation_stage must be one of {valid_stages}, "
                f"got {self.degradation_stage}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "rul_cycles": self.rul_cycles,
            "rul_confidence_lower": self.rul_confidence_lower,
            "rul_confidence_upper": self.rul_confidence_upper,
            "degradation_score": self.degradation_score,
            "degradation_stage": self.degradation_stage,
            "anomaly_flag": self.anomaly_flag,
            "anomaly_score": self.anomaly_score,
            "feature_importance": self.feature_importance,
            "timestamp": self.timestamp.isoformat(),
            "model_version": self.model_version,
            "capacitor_id": self.capacitor_id,
            "cycle_number": self.cycle_number,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        import json
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class TrainingDataset:
    """
    Dataset for model training
    
    Attributes:
        capacitor_ids: List of capacitor identifiers
        features: Feature array (n_samples, n_features)
        rul_labels: RUL labels (n_samples,)
        cycle_numbers: Cycle numbers (n_samples,)
        anomaly_labels: Optional anomaly labels for validation (n_samples,)
    """
    capacitor_ids: List[str]
    features: np.ndarray
    rul_labels: np.ndarray
    cycle_numbers: np.ndarray
    anomaly_labels: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Validate data after initialization"""
        n_samples = len(self.features)
        
        if len(self.capacitor_ids) != n_samples:
            raise ValueError(
                f"capacitor_ids length ({len(self.capacitor_ids)}) must match "
                f"features length ({n_samples})"
            )
        
        if len(self.rul_labels) != n_samples:
            raise ValueError(
                f"rul_labels length ({len(self.rul_labels)}) must match "
                f"features length ({n_samples})"
            )
        
        if len(self.cycle_numbers) != n_samples:
            raise ValueError(
                f"cycle_numbers length ({len(self.cycle_numbers)}) must match "
                f"features length ({n_samples})"
            )
        
        if self.anomaly_labels is not None and len(self.anomaly_labels) != n_samples:
            raise ValueError(
                f"anomaly_labels length ({len(self.anomaly_labels)}) must match "
                f"features length ({n_samples})"
            )
    
    def split_by_capacitor(
        self, test_capacitors: List[str]
    ) -> tuple["TrainingDataset", "TrainingDataset"]:
        """
        Split dataset by capacitor for cross-validation
        
        Args:
            test_capacitors: List of capacitor IDs for test set
            
        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        # Create masks
        test_mask = np.isin(self.capacitor_ids, test_capacitors)
        train_mask = ~test_mask
        
        # Split data
        train_dataset = TrainingDataset(
            capacitor_ids=[cid for cid, m in zip(self.capacitor_ids, train_mask) if m],
            features=self.features[train_mask],
            rul_labels=self.rul_labels[train_mask],
            cycle_numbers=self.cycle_numbers[train_mask],
            anomaly_labels=self.anomaly_labels[train_mask] if self.anomaly_labels is not None else None,
        )
        
        test_dataset = TrainingDataset(
            capacitor_ids=[cid for cid, m in zip(self.capacitor_ids, test_mask) if m],
            features=self.features[test_mask],
            rul_labels=self.rul_labels[test_mask],
            cycle_numbers=self.cycle_numbers[test_mask],
            anomaly_labels=self.anomaly_labels[test_mask] if self.anomaly_labels is not None else None,
        )
        
        return train_dataset, test_dataset
    
    def get_normal_cycles(self, max_cycle: int = 10) -> np.ndarray:
        """
        Get features from early cycles assumed to be normal
        
        Args:
            max_cycle: Maximum cycle number to consider as normal
            
        Returns:
            Feature array for normal cycles
        """
        normal_mask = self.cycle_numbers <= max_cycle
        return self.features[normal_mask]
    
    @property
    def n_samples(self) -> int:
        """Number of samples in the dataset"""
        return len(self.features)
    
    @property
    def n_features(self) -> int:
        """Number of features"""
        return self.features.shape[1] if len(self.features.shape) > 1 else 0
