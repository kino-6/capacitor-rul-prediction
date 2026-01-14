"""
Data models for the NASA PCOE EDA system.

This module defines all data structures used throughout the EDA system:
- Dataset metadata and validation results
- Statistical analysis results and reports
- Outlier detection summaries
- Time series analysis results
- Correlation and multicollinearity reports
- RUL and fault analysis results
- Preprocessing recommendations
- Comprehensive analysis results container

All models use dataclasses for type safety and automatic serialization support.
The models are designed to be immutable and provide clear interfaces between
different analysis components.

Key Model Categories:
- Data Loading: DatasetMetadata, ValidationResult
- Statistics: Stats, MissingValueReport
- Outliers: OutlierSummary
- Time Series: TrendReport, SeasonalityResult
- Correlations: MulticollinearityReport
- Fault Analysis: DistributionComparison
- Preprocessing: ScalingRecommendation, FeatureSuggestion, DataSplitStrategy
- Results: AnalysisResults (comprehensive container)

Example usage:
    metadata = DatasetMetadata(
        n_records=1000,
        n_features=50,
        feature_names=list(df.columns),
        data_types=df.dtypes.to_dict(),
        memory_usage=df.memory_usage(deep=True).sum(),
        date_range=None
    )
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DatasetMetadata:
    """Metadata about a loaded dataset."""

    n_records: int
    n_features: int
    feature_names: List[str]
    data_types: Dict[str, str]
    memory_usage: float
    date_range: Optional[Tuple[datetime, datetime]]


@dataclass
class ValidationResult:
    """Result of data validation operations."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]


@dataclass
class Stats:
    """Descriptive statistics for a feature."""

    mean: float
    median: float
    std: float
    min: float
    max: float
    q25: float
    q75: float


@dataclass
class MissingValueReport:
    """Report on missing values in the dataset."""

    missing_counts: Dict[str, int]
    missing_percentages: Dict[str, float]
    total_missing: int


@dataclass
class OutlierSummary:
    """Summary of outlier detection results."""

    outlier_counts: Dict[str, int]
    outlier_percentages: Dict[str, float]
    outlier_indices: Dict[str, np.ndarray]


@dataclass
class TrendReport:
    """Report on time series trends."""

    trends: Dict[str, Dict[str, float]]
    trend_directions: Dict[str, str]


@dataclass
class SeasonalityResult:
    """Result of seasonality detection."""

    has_seasonality: bool
    period: Optional[float]
    strength: Optional[float]


@dataclass
class MulticollinearityReport:
    """Report on multicollinearity detection."""

    high_vif_features: List[Tuple[str, float]]
    correlated_groups: List[List[str]]


@dataclass
class DistributionComparison:
    """Comparison of distributions between classes."""

    feature_distributions: Dict[str, Dict[str, Any]]
    statistical_tests: Dict[str, Dict[str, float]]


@dataclass
class ScalingRecommendation:
    """Recommendation for feature scaling."""

    method: str
    features: List[str]
    reason: str


@dataclass
class FeatureSuggestion:
    """Suggestion for feature engineering."""

    feature_name: str
    operation: str
    source_features: List[str]
    rationale: str


@dataclass
class DataSplitStrategy:
    """Strategy for splitting data into train/test sets."""

    method: str
    train_ratio: float
    validation_ratio: Optional[float]
    test_ratio: float
    preserve_temporal_order: bool
    rationale: str


@dataclass
class AnalysisResults:
    """Comprehensive results from EDA analysis."""

    metadata: DatasetMetadata
    statistics: Dict[str, Stats]
    missing_values: MissingValueReport
    correlation_matrix: pd.DataFrame
    outliers: OutlierSummary
    time_series_trends: Optional[TrendReport]
    rul_features: List[Tuple[str, float]]
    fault_features: List[str]
    preprocessing_recommendations: Dict[str, Any]
    visualization_paths: List[Path]
