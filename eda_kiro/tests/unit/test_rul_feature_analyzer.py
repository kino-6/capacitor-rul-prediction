"""Unit tests for RULFeatureAnalyzer."""

import numpy as np
import pandas as pd
import pytest
import matplotlib.figure

from nasa_pcoe_eda.analysis.rul_features import RULFeatureAnalyzer


class TestRULFeatureAnalyzer:
    """Test cases for RULFeatureAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = RULFeatureAnalyzer()

    def test_identify_degradation_features_empty_dataframe(self):
        """Test degradation feature identification with empty DataFrame."""
        df = pd.DataFrame()
        result = self.analyzer.identify_degradation_features(df)
        assert result == []

    def test_identify_degradation_features_with_trend(self):
        """Test identification of features with clear degradation trends."""
        # Create synthetic data with degradation trend
        time_values = np.arange(100)
        degrading_feature = 100 - 0.5 * time_values + np.random.normal(0, 1, 100)
        stable_feature = np.random.normal(50, 1, 100)
        
        df = pd.DataFrame({
            'time': time_values,
            'degrading': degrading_feature,
            'stable': stable_feature
        })
        
        result = self.analyzer.identify_degradation_features(df)
        
        # Should identify the degrading feature
        assert 'degrading' in result
        assert 'stable' not in result

    def test_compute_degradation_rates_empty_dataframe(self):
        """Test degradation rate computation with empty DataFrame."""
        df = pd.DataFrame()
        result = self.analyzer.compute_degradation_rates(df, ['feature1'])
        assert result == {}

    def test_compute_degradation_rates_empty_features(self):
        """Test degradation rate computation with empty feature list."""
        df = pd.DataFrame({'feature1': [1, 2, 3]})
        result = self.analyzer.compute_degradation_rates(df, [])
        assert result == {}

    def test_compute_degradation_rates_with_linear_trend(self):
        """Test degradation rate computation with linear trend."""
        time_values = np.arange(10)
        # Feature with slope of -2
        feature_values = 100 - 2 * time_values
        
        df = pd.DataFrame({
            'time': time_values,
            'feature': feature_values
        })
        
        result = self.analyzer.compute_degradation_rates(df, ['feature'])
        
        assert 'feature' in result
        # Should be close to -2 (allowing for numerical precision)
        assert abs(result['feature'] - (-2.0)) < 0.1

    def test_rank_features_for_rul_empty_dataframe(self):
        """Test RUL ranking with empty DataFrame."""
        df = pd.DataFrame()
        result = self.analyzer.rank_features_for_rul(df, 'rul')
        assert result == []

    def test_rank_features_for_rul_missing_rul_column(self):
        """Test RUL ranking with missing RUL column."""
        df = pd.DataFrame({'feature1': [1, 2, 3]})
        result = self.analyzer.rank_features_for_rul(df, 'rul')
        assert result == []

    def test_rank_features_for_rul_with_correlations(self):
        """Test RUL ranking with correlated features."""
        # Create features with different correlations to RUL
        rul_values = np.arange(100, 0, -1)  # Decreasing RUL
        high_corr_feature = rul_values + np.random.normal(0, 1, 100)  # High positive correlation
        low_corr_feature = np.random.normal(50, 10, 100)  # Low correlation
        
        df = pd.DataFrame({
            'rul': rul_values,
            'high_corr': high_corr_feature,
            'low_corr': low_corr_feature
        })
        
        result = self.analyzer.rank_features_for_rul(df, 'rul')
        
        # Should return features sorted by absolute correlation
        assert len(result) == 2
        feature_names = [item[0] for item in result]
        assert 'high_corr' in feature_names
        assert 'low_corr' in feature_names
        
        # High correlation feature should be ranked first
        assert result[0][0] == 'high_corr'
        assert abs(result[0][1]) > abs(result[1][1])

    def test_visualize_degradation_patterns_empty_dataframe(self):
        """Test visualization with empty DataFrame."""
        df = pd.DataFrame()
        result = self.analyzer.visualize_degradation_patterns(df, ['feature1'])
        assert result == []

    def test_visualize_degradation_patterns_empty_features(self):
        """Test visualization with empty feature list."""
        df = pd.DataFrame({'feature1': [1, 2, 3]})
        result = self.analyzer.visualize_degradation_patterns(df, [])
        assert result == []

    def test_visualize_degradation_patterns_single_feature(self):
        """Test visualization with single feature."""
        time_values = np.arange(10)
        feature_values = 100 - 2 * time_values
        
        df = pd.DataFrame({
            'time': time_values,
            'feature': feature_values
        })
        
        result = self.analyzer.visualize_degradation_patterns(df, ['feature'])
        
        # Should return one figure for the single feature
        assert len(result) == 1
        assert isinstance(result[0], matplotlib.figure.Figure)

    def test_visualize_degradation_patterns_multiple_features(self):
        """Test visualization with multiple features."""
        time_values = np.arange(10)
        feature1_values = 100 - 2 * time_values
        feature2_values = 50 - 1 * time_values
        
        df = pd.DataFrame({
            'time': time_values,
            'feature1': feature1_values,
            'feature2': feature2_values
        })
        
        result = self.analyzer.visualize_degradation_patterns(df, ['feature1', 'feature2'])
        
        # Should return individual plots plus summary plot
        assert len(result) == 3  # 2 individual + 1 summary
        for fig in result:
            assert isinstance(fig, matplotlib.figure.Figure)

    def test_identify_time_column_datetime(self):
        """Test time column identification with datetime column."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=10),
            'feature': np.arange(10)
        })
        
        result = self.analyzer._identify_time_column(df)
        assert result == 'timestamp'

    def test_identify_time_column_by_name(self):
        """Test time column identification by name patterns."""
        df = pd.DataFrame({
            'cycle': np.arange(10),
            'feature': np.arange(10)
        })
        
        result = self.analyzer._identify_time_column(df)
        assert result == 'cycle'

    def test_identify_time_column_none_found(self):
        """Test time column identification when none found."""
        df = pd.DataFrame({
            'feature1': np.arange(10),
            'feature2': np.arange(10)
        })
        
        result = self.analyzer._identify_time_column(df)
        assert result is None

    def test_has_degradation_trend_insufficient_data(self):
        """Test degradation trend detection with insufficient data."""
        values = np.array([1, 2])  # Only 2 points
        time_values = np.array([0, 1])
        
        result = self.analyzer._has_degradation_trend(values, time_values)
        assert result is False

    def test_has_degradation_trend_with_nan(self):
        """Test degradation trend detection with NaN values."""
        values = np.array([1, np.nan, 3, 4, 5])
        time_values = np.array([0, 1, 2, 3, 4])
        
        # Should handle NaN values gracefully
        result = self.analyzer._has_degradation_trend(values, time_values)
        assert isinstance(result, (bool, np.bool_))

    def test_has_degradation_trend_strong_correlation(self):
        """Test degradation trend detection with strong correlation."""
        time_values = np.arange(100)
        # Strong negative trend
        values = 100 - 0.8 * time_values + np.random.normal(0, 1, 100)
        
        result = self.analyzer._has_degradation_trend(values, time_values)
        assert result == True

    def test_compute_feature_degradation_rate_insufficient_data(self):
        """Test degradation rate computation with insufficient data."""
        values = np.array([1])  # Only 1 point
        time_values = np.array([0])
        
        result = self.analyzer._compute_feature_degradation_rate(values, time_values)
        assert result == 0.0

    def test_compute_feature_degradation_rate_with_nan(self):
        """Test degradation rate computation with NaN values."""
        values = np.array([1, np.nan, 3, 4, 5])
        time_values = np.array([0, 1, 2, 3, 4])
        
        result = self.analyzer._compute_feature_degradation_rate(values, time_values)
        assert isinstance(result, float)

    def test_compute_feature_degradation_rate_linear_trend(self):
        """Test degradation rate computation with perfect linear trend."""
        time_values = np.array([0, 1, 2, 3, 4])
        values = np.array([10, 8, 6, 4, 2])  # Slope of -2
        
        result = self.analyzer._compute_feature_degradation_rate(values, time_values)
        assert abs(result - (-2.0)) < 0.001  # Should be very close to -2