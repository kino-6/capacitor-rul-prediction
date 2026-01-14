"""Unit tests for AnalysisOrchestrator."""

import tempfile
import logging
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import pandas as pd
import numpy as np

from nasa_pcoe_eda.orchestrator import AnalysisOrchestrator
from nasa_pcoe_eda.models import (
    AnalysisResults,
    DatasetMetadata,
    Stats,
    MissingValueReport,
    OutlierSummary,
    TrendReport,
    ValidationResult
)
from nasa_pcoe_eda.exceptions import EDAError, DataLoadError, AnalysisError


class TestAnalysisOrchestrator:
    """Test cases for AnalysisOrchestrator."""

    def setup_method(self):
        """Set up test fixtures."""
        # Use temporary directory for output
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir)
        self.orchestrator = AnalysisOrchestrator(output_dir=self.output_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_default_output_dir(self):
        """Test AnalysisOrchestrator initialization with default output directory."""
        orchestrator = AnalysisOrchestrator()
        
        assert orchestrator.output_dir == Path("output")
        assert orchestrator.output_dir.exists()
        assert orchestrator.logger is not None
        assert orchestrator.data_loader is not None
        assert orchestrator.stats_analyzer is not None
        assert orchestrator.correlation_analyzer is not None
        assert orchestrator.outlier_detector is not None
        assert orchestrator.timeseries_analyzer is not None
        assert orchestrator.quality_analyzer is not None
        assert orchestrator.rul_analyzer is not None
        assert orchestrator.fault_analyzer is not None
        assert orchestrator.preprocessing_recommender is not None
        assert orchestrator.visualization_engine is not None
        assert orchestrator.report_generator is not None

    def test_init_custom_output_dir(self):
        """Test AnalysisOrchestrator initialization with custom output directory."""
        custom_dir = self.output_dir / "custom"
        orchestrator = AnalysisOrchestrator(output_dir=custom_dir)
        
        assert orchestrator.output_dir == custom_dir
        assert custom_dir.exists()

    def test_init_creates_output_directory(self):
        """Test that initialization creates output directory if it doesn't exist."""
        non_existent_dir = self.output_dir / "non_existent"
        assert not non_existent_dir.exists()
        
        orchestrator = AnalysisOrchestrator(output_dir=non_existent_dir)
        
        assert non_existent_dir.exists()
        assert orchestrator.output_dir == non_existent_dir

    def test_setup_logging(self):
        """Test logging setup."""
        orchestrator = AnalysisOrchestrator(output_dir=self.output_dir)
        
        # Check logger configuration
        assert orchestrator.logger.name == "nasa_pcoe_eda"
        assert orchestrator.logger.level == logging.INFO
        assert len(orchestrator.logger.handlers) == 2  # File and console handlers
        
        # Check log directory creation
        log_dir = self.output_dir / "logs"
        assert log_dir.exists()

    def test_initial_state(self):
        """Test initial state of orchestrator."""
        assert self.orchestrator._data is None
        assert self.orchestrator._metadata is None
        assert self.orchestrator._analysis_results is None
        assert self.orchestrator.get_analysis_results() is None
        assert self.orchestrator.get_loaded_data() is None
        assert self.orchestrator.get_metadata() is None

    def test_load_and_validate_data_success(self):
        """Test successful data loading and validation."""
        # Setup mocks
        mock_data = self._create_mock_dataframe()
        mock_validation_result = ValidationResult(is_valid=True, errors=[], warnings=[])
        mock_metadata = self._create_mock_metadata()

        self.orchestrator.data_loader.load_dataset = Mock(return_value=mock_data)
        self.orchestrator.data_loader.validate_data = Mock(return_value=mock_validation_result)
        self.orchestrator.data_loader.get_metadata = Mock(return_value=mock_metadata)

        # Test
        data_path = Path("test_data.mat")
        self.orchestrator._load_and_validate_data(data_path)

        # Verify
        self.orchestrator.data_loader.load_dataset.assert_called_once_with(data_path)
        self.orchestrator.data_loader.validate_data.assert_called_once_with(mock_data)
        self.orchestrator.data_loader.get_metadata.assert_called_once_with(mock_data)

        assert self.orchestrator._data.equals(mock_data)
        assert self.orchestrator._metadata == mock_metadata

    def test_load_and_validate_data_validation_warnings(self):
        """Test data loading with validation warnings."""
        # Setup mocks
        mock_data = self._create_mock_dataframe()
        mock_validation_result = ValidationResult(
            is_valid=False, 
            errors=["Missing column"], 
            warnings=["Data quality issue"]
        )
        mock_metadata = self._create_mock_metadata()

        self.orchestrator.data_loader.load_dataset = Mock(return_value=mock_data)
        self.orchestrator.data_loader.validate_data = Mock(return_value=mock_validation_result)
        self.orchestrator.data_loader.get_metadata = Mock(return_value=mock_metadata)

        # Test - should not raise exception but log warnings
        data_path = Path("test_data.mat")
        self.orchestrator._load_and_validate_data(data_path)

        # Verify data is still loaded despite warnings
        assert self.orchestrator._data.equals(mock_data)
        assert self.orchestrator._metadata == mock_metadata

    def test_load_and_validate_data_load_error(self):
        """Test data loading with DataLoadError."""
        self.orchestrator.data_loader.load_dataset = Mock(side_effect=DataLoadError("File not found"))

        data_path = Path("missing_data.mat")
        
        with pytest.raises(DataLoadError):
            self.orchestrator._load_and_validate_data(data_path)

    def test_load_and_validate_data_unexpected_error(self):
        """Test data loading with unexpected error."""
        self.orchestrator.data_loader.load_dataset = Mock(side_effect=Exception("Unexpected error"))

        data_path = Path("test_data.mat")
        
        with pytest.raises(AnalysisError, match="Data loading failed"):
            self.orchestrator._load_and_validate_data(data_path)

    def test_compute_statistics_success(self):
        """Test successful statistics computation."""
        # Setup
        self.orchestrator._data = self._create_mock_dataframe()
        mock_stats = self._create_mock_statistics()
        mock_missing_report = self._create_mock_missing_values()
        mock_data_types = {'feature1': 'float64', 'feature2': 'float64'}

        self.orchestrator.stats_analyzer.compute_descriptive_stats = Mock(return_value=mock_stats['descriptive_stats'])
        self.orchestrator.stats_analyzer.analyze_missing_values = Mock(return_value=mock_missing_report)
        self.orchestrator.stats_analyzer.identify_data_types = Mock(return_value=mock_data_types)

        # Test
        result = self.orchestrator._compute_statistics()

        # Verify
        assert 'descriptive_stats' in result
        assert 'missing_values' in result
        assert 'data_types' in result
        assert result['descriptive_stats'] == mock_stats['descriptive_stats']
        assert result['missing_values'] == mock_missing_report
        assert result['data_types'] == mock_data_types

    def test_compute_statistics_no_numeric_columns(self):
        """Test statistics computation with no numeric columns."""
        # Setup data with no numeric columns
        self.orchestrator._data = pd.DataFrame({
            'text_col': ['a', 'b', 'c'],
            'category_col': ['x', 'y', 'z']
        })

        # Test
        result = self.orchestrator._compute_statistics()

        # Should return empty dict for descriptive stats
        assert result == {}

    def test_compute_statistics_error(self):
        """Test statistics computation with error."""
        self.orchestrator._data = self._create_mock_dataframe()
        self.orchestrator.stats_analyzer.compute_descriptive_stats = Mock(side_effect=Exception("Stats failed"))

        with pytest.raises(AnalysisError, match="Statistical analysis failed"):
            self.orchestrator._compute_statistics()

    def test_get_analysis_results_none(self):
        """Test getting analysis results when none exist."""
        result = self.orchestrator.get_analysis_results()
        assert result is None

    def test_get_analysis_results_with_results(self):
        """Test getting analysis results when they exist."""
        mock_results = Mock()
        self.orchestrator._analysis_results = mock_results
        
        result = self.orchestrator.get_analysis_results()
        assert result == mock_results

    def test_get_loaded_data_none(self):
        """Test getting loaded data when none exists."""
        result = self.orchestrator.get_loaded_data()
        assert result is None

    def test_get_loaded_data_with_data(self):
        """Test getting loaded data when it exists."""
        mock_data = self._create_mock_dataframe()
        self.orchestrator._data = mock_data
        
        result = self.orchestrator.get_loaded_data()
        assert result.equals(mock_data)

    def test_get_metadata_none(self):
        """Test getting metadata when none exists."""
        result = self.orchestrator.get_metadata()
        assert result is None

    def test_get_metadata_with_metadata(self):
        """Test getting metadata when it exists."""
        mock_metadata = self._create_mock_metadata()
        self.orchestrator._metadata = mock_metadata
        
        result = self.orchestrator.get_metadata()
        assert result == mock_metadata

    def test_logging_integration(self):
        """Test that logging works correctly during analysis."""
        # This test verifies that the logger is properly configured and can be used
        assert self.orchestrator.logger is not None
        assert self.orchestrator.logger.name == "nasa_pcoe_eda"
        
        # Test that we can log messages without errors
        self.orchestrator.logger.info("Test log message")
        self.orchestrator.logger.warning("Test warning message")
        self.orchestrator.logger.error("Test error message")

    # Helper methods for creating mock objects

    def _create_mock_dataframe(self):
        """Create a mock DataFrame for testing."""
        return pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'feature2': [2.0, 4.0, 6.0, 8.0, 10.0],
            'feature3': [0.5, 1.0, 1.5, 2.0, 2.5]
        })

    def _create_mock_dataframe_with_rul(self):
        """Create a mock DataFrame with RUL column for testing."""
        df = self._create_mock_dataframe()
        df['rul'] = [100, 80, 60, 40, 20]
        return df

    def _create_mock_dataframe_with_fault(self):
        """Create a mock DataFrame with fault column for testing."""
        df = self._create_mock_dataframe()
        df['fault'] = [0, 0, 1, 1, 2]
        return df

    def _create_mock_metadata(self):
        """Create mock DatasetMetadata for testing."""
        return DatasetMetadata(
            n_records=100,
            n_features=3,
            feature_names=['feature1', 'feature2', 'feature3'],
            data_types={'feature1': 'float64', 'feature2': 'float64', 'feature3': 'float64'},
            memory_usage=1024.0,
            date_range=None
        )

    def _create_mock_statistics(self):
        """Create mock statistics for testing."""
        return {
            'descriptive_stats': {
                'feature1': Stats(1.0, 1.0, 1.0, 0.0, 2.0, 0.5, 1.5),
                'feature2': Stats(2.0, 2.0, 1.0, 1.0, 3.0, 1.5, 2.5)
            },
            'missing_values': self._create_mock_missing_values(),
            'data_types': {'feature1': 'float64', 'feature2': 'float64'}
        }

    def _create_mock_missing_values(self):
        """Create mock MissingValueReport for testing."""
        return MissingValueReport(
            missing_counts={'feature1': 0, 'feature2': 0},
            missing_percentages={'feature1': 0.0, 'feature2': 0.0},
            total_missing=0
        )

    def _create_mock_correlation_matrix(self):
        """Create mock correlation matrix for testing."""
        return pd.DataFrame({
            'feature1': [1.0, 0.5, 0.3],
            'feature2': [0.5, 1.0, 0.2],
            'feature3': [0.3, 0.2, 1.0]
        }, index=['feature1', 'feature2', 'feature3'])

    def _create_mock_outliers(self):
        """Create mock OutlierSummary for testing."""
        return OutlierSummary(
            outlier_counts={'feature1': 2, 'feature2': 1},
            outlier_percentages={'feature1': 2.0, 'feature2': 1.0},
            outlier_indices={'feature1': np.array([1, 4]), 'feature2': np.array([2])}
        )

    def _create_mock_trends(self):
        """Create mock TrendReport for testing."""
        return TrendReport(
            trends={'feature1': {'slope': 0.5, 'r_squared': 0.8}},
            trend_directions={'feature1': 'increasing'}
        )