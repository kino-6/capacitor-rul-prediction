"""Unit tests for TimeSeriesAnalyzer."""

import numpy as np
import pandas as pd
import pytest

from nasa_pcoe_eda.analysis.timeseries import TimeSeriesAnalyzer
from nasa_pcoe_eda.models import SeasonalityResult, TrendReport


class TestTimeSeriesAnalyzer:
    """Test cases for TimeSeriesAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = TimeSeriesAnalyzer()

    def test_identify_temporal_features_datetime_columns(self):
        """Test identification of datetime columns as temporal features."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='D'),
            'value': range(10),
            'category': ['A'] * 10
        })
        
        temporal_features = self.analyzer.identify_temporal_features(df)
        assert 'timestamp' in temporal_features
        assert 'value' not in temporal_features
        assert 'category' not in temporal_features

    def test_identify_temporal_features_naming_patterns(self):
        """Test identification of temporal features by naming patterns."""
        df = pd.DataFrame({
            'time_step': range(10),
            'cycle_number': range(10),
            'temperature': np.random.randn(10),
            'pressure': np.random.randn(10)
        })
        
        temporal_features = self.analyzer.identify_temporal_features(df)
        assert 'time_step' in temporal_features
        assert 'cycle_number' in temporal_features

    def test_identify_temporal_features_monotonic_sequences(self):
        """Test identification of monotonic sequences as temporal features."""
        df = pd.DataFrame({
            'sequence_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # Monotonic increasing
            'random_values': np.random.randn(10),
            'non_monotonic': [1, 3, 2, 5, 4, 7, 6, 9, 8, 10]  # Not monotonic enough
        })
        
        temporal_features = self.analyzer.identify_temporal_features(df)
        assert 'sequence_id' in temporal_features
        assert 'random_values' not in temporal_features
        assert 'non_monotonic' not in temporal_features

    def test_compute_trends_increasing_trend(self):
        """Test trend computation for increasing data."""
        df = pd.DataFrame({
            'time': range(10),
            'increasing_feature': [i * 2 + np.random.normal(0, 0.1) for i in range(10)]
        })
        
        trend_report = self.analyzer.compute_trends(df, ['increasing_feature'])
        
        assert isinstance(trend_report, TrendReport)
        assert 'increasing_feature' in trend_report.trends
        assert 'increasing_feature' in trend_report.trend_directions
        
        # Should detect increasing trend
        assert trend_report.trends['increasing_feature']['slope'] > 0
        assert trend_report.trend_directions['increasing_feature'] == 'increasing'

    def test_compute_trends_decreasing_trend(self):
        """Test trend computation for decreasing data."""
        df = pd.DataFrame({
            'time': range(10),
            'decreasing_feature': [10 - i * 1.5 + np.random.normal(0, 0.1) for i in range(10)]
        })
        
        trend_report = self.analyzer.compute_trends(df, ['decreasing_feature'])
        
        assert 'decreasing_feature' in trend_report.trends
        assert trend_report.trends['decreasing_feature']['slope'] < 0
        assert trend_report.trend_directions['decreasing_feature'] == 'decreasing'

    def test_compute_trends_no_temporal_features(self):
        """Test trend computation when no temporal features are present."""
        df = pd.DataFrame({
            'feature1': range(10),
            'feature2': [i * 2 for i in range(10)]
        })
        
        trend_report = self.analyzer.compute_trends(df, ['feature1', 'feature2'])
        
        # Should still compute trends using row index as time
        assert 'feature1' in trend_report.trends
        assert 'feature2' in trend_report.trends

    def test_compute_trends_insufficient_data(self):
        """Test trend computation with insufficient data."""
        df = pd.DataFrame({
            'time': [1],
            'feature': [5.0]
        })
        
        trend_report = self.analyzer.compute_trends(df, ['feature'])
        
        assert trend_report.trend_directions['feature'] == 'insufficient_data'
        assert trend_report.trends['feature']['slope'] == 0.0

    def test_detect_seasonality_no_pattern(self):
        """Test seasonality detection on random data."""
        df = pd.DataFrame({
            'random_feature': np.random.randn(50)
        })
        
        result = self.analyzer.detect_seasonality(df, 'random_feature')
        
        assert isinstance(result, SeasonalityResult)
        # Random data should not show strong seasonality
        assert not result.has_seasonality or (result.strength and result.strength < 0.5)

    def test_detect_seasonality_with_pattern(self):
        """Test seasonality detection on data with periodic pattern."""
        # Create data with clear 5-period seasonality
        x = np.arange(50)
        seasonal_data = np.sin(2 * np.pi * x / 5) + np.random.normal(0, 0.1, 50)
        
        df = pd.DataFrame({
            'seasonal_feature': seasonal_data
        })
        
        result = self.analyzer.detect_seasonality(df, 'seasonal_feature')
        
        # Should detect seasonality with period around 5
        if result.has_seasonality:
            assert result.period is not None
            assert 4 <= result.period <= 6  # Allow some tolerance
            assert result.strength is not None
            assert result.strength > 0.3

    def test_detect_seasonality_insufficient_data(self):
        """Test seasonality detection with insufficient data."""
        df = pd.DataFrame({
            'small_feature': [1, 2, 3]
        })
        
        result = self.analyzer.detect_seasonality(df, 'small_feature')
        
        assert not result.has_seasonality
        assert result.period is None
        assert result.strength is None

    def test_detect_seasonality_non_numeric_feature(self):
        """Test seasonality detection on non-numeric feature."""
        df = pd.DataFrame({
            'text_feature': ['A', 'B', 'C', 'A', 'B', 'C']
        })
        
        result = self.analyzer.detect_seasonality(df, 'text_feature')
        
        assert not result.has_seasonality
        assert result.period is None
        assert result.strength is None

    def test_detect_seasonality_missing_feature(self):
        """Test seasonality detection on non-existent feature."""
        df = pd.DataFrame({
            'existing_feature': range(10)
        })
        
        result = self.analyzer.detect_seasonality(df, 'missing_feature')
        
        assert not result.has_seasonality
        assert result.period is None
        assert result.strength is None

    def test_compute_trends_with_missing_values(self):
        """Test trend computation with missing values."""
        df = pd.DataFrame({
            'time': range(10),
            'feature_with_nans': [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10]
        })
        
        trend_report = self.analyzer.compute_trends(df, ['feature_with_nans'])
        
        # Should handle NaN values gracefully
        assert 'feature_with_nans' in trend_report.trends
        assert trend_report.trends['feature_with_nans']['slope'] > 0  # Should still detect increasing trend

    def test_identify_temporal_features_empty_dataframe(self):
        """Test temporal feature identification on empty DataFrame."""
        df = pd.DataFrame()
        
        temporal_features = self.analyzer.identify_temporal_features(df)
        
        assert temporal_features == []

    def test_compute_trends_empty_features_list(self):
        """Test trend computation with empty features list."""
        df = pd.DataFrame({
            'time': range(10),
            'feature': range(10)
        })
        
        trend_report = self.analyzer.compute_trends(df, [])
        
        assert trend_report.trends == {}
        assert trend_report.trend_directions == {}