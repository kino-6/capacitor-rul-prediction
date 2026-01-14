"""Unit tests for data models and exception classes."""

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nasa_pcoe_eda.exceptions import (
    AnalysisError,
    DataLoadError,
    DataValidationError,
    EDAError,
    VisualizationError,
)
from nasa_pcoe_eda.models import (
    AnalysisResults,
    DatasetMetadata,
    DataSplitStrategy,
    DistributionComparison,
    FeatureSuggestion,
    MissingValueReport,
    MulticollinearityReport,
    OutlierSummary,
    ScalingRecommendation,
    SeasonalityResult,
    Stats,
    TrendReport,
    ValidationResult,
)


class TestExceptions:
    """Test custom exception classes."""

    def test_eda_error_is_base_exception(self) -> None:
        """Test that EDAError is the base exception."""
        error = EDAError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_data_load_error_inherits_from_eda_error(self) -> None:
        """Test that DataLoadError inherits from EDAError."""
        error = DataLoadError("Load failed")
        assert isinstance(error, EDAError)
        assert isinstance(error, Exception)

    def test_data_validation_error_inherits_from_eda_error(self) -> None:
        """Test that DataValidationError inherits from EDAError."""
        error = DataValidationError("Validation failed")
        assert isinstance(error, EDAError)

    def test_analysis_error_inherits_from_eda_error(self) -> None:
        """Test that AnalysisError inherits from EDAError."""
        error = AnalysisError("Analysis failed")
        assert isinstance(error, EDAError)

    def test_visualization_error_inherits_from_eda_error(self) -> None:
        """Test that VisualizationError inherits from EDAError."""
        error = VisualizationError("Visualization failed")
        assert isinstance(error, EDAError)


class TestDataModels:
    """Test data model classes."""

    def test_dataset_metadata_creation(self) -> None:
        """Test DatasetMetadata can be created with all fields."""
        metadata = DatasetMetadata(
            n_records=100,
            n_features=5,
            feature_names=["a", "b", "c", "d", "e"],
            data_types={"a": "int", "b": "float"},
            memory_usage=1024.0,
            date_range=(datetime(2020, 1, 1), datetime(2020, 12, 31)),
        )
        assert metadata.n_records == 100
        assert metadata.n_features == 5
        assert len(metadata.feature_names) == 5
        assert metadata.date_range is not None

    def test_validation_result_creation(self) -> None:
        """Test ValidationResult can be created."""
        result = ValidationResult(
            is_valid=True, errors=[], warnings=["Minor warning"]
        )
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 1

    def test_stats_creation(self) -> None:
        """Test Stats dataclass can be created."""
        stats = Stats(
            mean=10.0, median=9.5, std=2.0, min=5.0, max=15.0, q25=8.0, q75=12.0
        )
        assert stats.mean == 10.0
        assert stats.median == 9.5
        assert stats.std == 2.0

    def test_missing_value_report_creation(self) -> None:
        """Test MissingValueReport can be created."""
        report = MissingValueReport(
            missing_counts={"a": 5, "b": 10},
            missing_percentages={"a": 5.0, "b": 10.0},
            total_missing=15,
        )
        assert report.total_missing == 15
        assert report.missing_counts["a"] == 5

    def test_outlier_summary_creation(self) -> None:
        """Test OutlierSummary can be created."""
        summary = OutlierSummary(
            outlier_counts={"a": 3},
            outlier_percentages={"a": 3.0},
            outlier_indices={"a": np.array([1, 5, 10])},
        )
        assert summary.outlier_counts["a"] == 3
        assert len(summary.outlier_indices["a"]) == 3

    def test_analysis_results_creation(self) -> None:
        """Test AnalysisResults can be created with all fields."""
        metadata = DatasetMetadata(
            n_records=100,
            n_features=3,
            feature_names=["a", "b", "c"],
            data_types={"a": "int"},
            memory_usage=512.0,
            date_range=None,
        )
        stats = Stats(
            mean=10.0, median=9.5, std=2.0, min=5.0, max=15.0, q25=8.0, q75=12.0
        )
        missing = MissingValueReport(
            missing_counts={}, missing_percentages={}, total_missing=0
        )
        outliers = OutlierSummary(
            outlier_counts={}, outlier_percentages={}, outlier_indices={}
        )
        corr_matrix = pd.DataFrame([[1.0, 0.5], [0.5, 1.0]])

        results = AnalysisResults(
            metadata=metadata,
            statistics={"a": stats},
            missing_values=missing,
            correlation_matrix=corr_matrix,
            outliers=outliers,
            time_series_trends=None,
            rul_features=[("feature1", 0.8)],
            fault_features=["feature2"],
            preprocessing_recommendations={},
            visualization_paths=[Path("output/plot.png")],
        )

        assert results.metadata.n_records == 100
        assert "a" in results.statistics
        assert len(results.rul_features) == 1
        assert len(results.visualization_paths) == 1
