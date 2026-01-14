"""Unit tests for CorrelationAnalyzer."""

import numpy as np
import pandas as pd
import pytest

from nasa_pcoe_eda.analysis.correlation import CorrelationAnalyzer
from nasa_pcoe_eda.models import MulticollinearityReport


class TestCorrelationAnalyzer:
    """Test cases for CorrelationAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = CorrelationAnalyzer()

    def test_compute_correlation_matrix_basic(self):
        """Test basic correlation matrix computation."""
        # Create test data with known correlations
        np.random.seed(42)
        x = np.random.randn(100)
        y = 2 * x + np.random.randn(100) * 0.1  # Strong positive correlation
        z = -x + np.random.randn(100) * 0.1     # Strong negative correlation
        w = np.random.randn(100)                # No correlation with others
        
        df = pd.DataFrame({
            'x': x,
            'y': y,
            'z': z,
            'w': w
        })
        
        corr_matrix = self.analyzer.compute_correlation_matrix(df)
        
        # Check matrix properties
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.shape == (4, 4)
        assert list(corr_matrix.index) == ['x', 'y', 'z', 'w']
        assert list(corr_matrix.columns) == ['x', 'y', 'z', 'w']
        
        # Check diagonal is 1.0 (self-correlation)
        np.testing.assert_array_almost_equal(np.diag(corr_matrix), [1.0, 1.0, 1.0, 1.0])
        
        # Check symmetry
        np.testing.assert_array_almost_equal(corr_matrix.values, corr_matrix.T.values)
        
        # Check expected correlations
        assert corr_matrix.loc['x', 'y'] > 0.8  # Strong positive
        assert corr_matrix.loc['x', 'z'] < -0.8  # Strong negative
        assert abs(corr_matrix.loc['x', 'w']) < 0.3  # Weak correlation

    def test_compute_correlation_matrix_empty_dataframe(self):
        """Test correlation matrix with empty DataFrame."""
        df = pd.DataFrame()
        corr_matrix = self.analyzer.compute_correlation_matrix(df)
        
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.empty

    def test_compute_correlation_matrix_no_numeric_columns(self):
        """Test correlation matrix with no numeric columns."""
        df = pd.DataFrame({
            'text': ['a', 'b', 'c'],
            'category': ['cat1', 'cat2', 'cat1']
        })
        
        corr_matrix = self.analyzer.compute_correlation_matrix(df)
        
        assert isinstance(corr_matrix, pd.DataFrame)
        assert corr_matrix.empty

    def test_compute_correlation_matrix_with_nan(self):
        """Test correlation matrix with NaN values."""
        df = pd.DataFrame({
            'a': [1, 2, np.nan, 4, 5],
            'b': [2, 4, 6, np.nan, 10],
            'c': [np.nan, np.nan, np.nan, np.nan, np.nan]  # All NaN
        })
        
        corr_matrix = self.analyzer.compute_correlation_matrix(df)
        
        # Should exclude the all-NaN column
        assert 'c' not in corr_matrix.columns
        assert 'a' in corr_matrix.columns
        assert 'b' in corr_matrix.columns

    def test_identify_high_correlations_basic(self):
        """Test identification of high correlations."""
        # Create correlation matrix with known values
        corr_data = {
            'a': [1.0, 0.9, 0.1],
            'b': [0.9, 1.0, 0.2],
            'c': [0.1, 0.2, 1.0]
        }
        corr_matrix = pd.DataFrame(corr_data, index=['a', 'b', 'c'])
        
        high_corrs = self.analyzer.identify_high_correlations(corr_matrix, threshold=0.8)
        
        assert len(high_corrs) == 1
        assert high_corrs[0] == ('a', 'b', 0.9)

    def test_identify_high_correlations_multiple_pairs(self):
        """Test identification of multiple high correlation pairs."""
        corr_data = {
            'a': [1.0, 0.9, 0.85],
            'b': [0.9, 1.0, 0.82],
            'c': [0.85, 0.82, 1.0]
        }
        corr_matrix = pd.DataFrame(corr_data, index=['a', 'b', 'c'])
        
        high_corrs = self.analyzer.identify_high_correlations(corr_matrix, threshold=0.8)
        
        assert len(high_corrs) == 3
        # Should be sorted by absolute correlation value (descending)
        assert high_corrs[0][2] == 0.9  # Highest correlation
        assert abs(high_corrs[0][2]) >= abs(high_corrs[1][2])
        assert abs(high_corrs[1][2]) >= abs(high_corrs[2][2])

    def test_identify_high_correlations_negative_correlation(self):
        """Test identification of high negative correlations."""
        corr_data = {
            'a': [1.0, -0.9, 0.1],
            'b': [-0.9, 1.0, 0.2],
            'c': [0.1, 0.2, 1.0]
        }
        corr_matrix = pd.DataFrame(corr_data, index=['a', 'b', 'c'])
        
        high_corrs = self.analyzer.identify_high_correlations(corr_matrix, threshold=0.8)
        
        assert len(high_corrs) == 1
        assert high_corrs[0] == ('a', 'b', -0.9)

    def test_identify_high_correlations_empty_matrix(self):
        """Test high correlation identification with empty matrix."""
        corr_matrix = pd.DataFrame()
        high_corrs = self.analyzer.identify_high_correlations(corr_matrix)
        
        assert high_corrs == []

    def test_detect_multicollinearity_basic(self):
        """Test basic multicollinearity detection."""
        # Create data with multicollinearity
        np.random.seed(42)
        x1 = np.random.randn(100)
        x2 = 2 * x1 + np.random.randn(100) * 0.01  # Almost perfectly correlated
        x3 = np.random.randn(100)  # Independent
        
        df = pd.DataFrame({
            'x1': x1,
            'x2': x2,
            'x3': x3
        })
        
        report = self.analyzer.detect_multicollinearity(df)
        
        assert isinstance(report, MulticollinearityReport)
        assert isinstance(report.high_vif_features, list)
        assert isinstance(report.correlated_groups, list)

    def test_detect_multicollinearity_empty_dataframe(self):
        """Test multicollinearity detection with empty DataFrame."""
        df = pd.DataFrame()
        report = self.analyzer.detect_multicollinearity(df)
        
        assert isinstance(report, MulticollinearityReport)
        assert report.high_vif_features == []
        assert report.correlated_groups == []

    def test_detect_multicollinearity_single_column(self):
        """Test multicollinearity detection with single column."""
        df = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
        report = self.analyzer.detect_multicollinearity(df)
        
        assert isinstance(report, MulticollinearityReport)
        assert report.high_vif_features == []
        assert report.correlated_groups == []

    def test_detect_multicollinearity_constant_column(self):
        """Test multicollinearity detection with constant column."""
        df = pd.DataFrame({
            'x1': [1, 2, 3, 4, 5],
            'x2': [1, 1, 1, 1, 1],  # Constant column
            'x3': [2, 4, 6, 8, 10]
        })
        
        report = self.analyzer.detect_multicollinearity(df)
        
        # Should handle constant columns gracefully
        assert isinstance(report, MulticollinearityReport)

    def test_identify_correlated_groups(self):
        """Test identification of correlated feature groups."""
        # Create correlation matrix with two groups
        corr_data = {
            'a': [1.0, 0.9, 0.85, 0.1, 0.2],
            'b': [0.9, 1.0, 0.88, 0.15, 0.1],
            'c': [0.85, 0.88, 1.0, 0.05, 0.25],
            'd': [0.1, 0.15, 0.05, 1.0, 0.9],
            'e': [0.2, 0.1, 0.25, 0.9, 1.0]
        }
        corr_matrix = pd.DataFrame(corr_data, index=['a', 'b', 'c', 'd', 'e'])
        
        groups = self.analyzer._identify_correlated_groups(corr_matrix, threshold=0.8)
        
        # Should identify two groups: {a, b, c} and {d, e}
        assert len(groups) == 2
        
        # Check that groups contain expected features
        group_sets = [set(group) for group in groups]
        assert {'a', 'b', 'c'} in group_sets
        assert {'d', 'e'} in group_sets

    def test_correlation_matrix_range_constraint(self):
        """Test that correlation values are within [-1, 1] range."""
        np.random.seed(42)
        df = pd.DataFrame({
            'x': np.random.randn(50),
            'y': np.random.randn(50),
            'z': np.random.randn(50)
        })
        
        corr_matrix = self.analyzer.compute_correlation_matrix(df)
        
        # All correlation values should be in [-1, 1]
        assert (corr_matrix.values >= -1.0).all()
        assert (corr_matrix.values <= 1.0).all()

    def test_correlation_matrix_symmetry(self):
        """Test that correlation matrix is symmetric."""
        np.random.seed(42)
        df = pd.DataFrame({
            'a': np.random.randn(50),
            'b': np.random.randn(50),
            'c': np.random.randn(50)
        })
        
        corr_matrix = self.analyzer.compute_correlation_matrix(df)
        
        # Matrix should be symmetric
        np.testing.assert_array_almost_equal(
            corr_matrix.values, 
            corr_matrix.T.values
        )

    def test_high_correlation_threshold_consistency(self):
        """Test that high correlation identification respects threshold."""
        corr_data = {
            'a': [1.0, 0.7, 0.9],
            'b': [0.7, 1.0, 0.6],
            'c': [0.9, 0.6, 1.0]
        }
        corr_matrix = pd.DataFrame(corr_data, index=['a', 'b', 'c'])
        
        # With threshold 0.8, only a-c correlation (0.9) should be identified
        high_corrs = self.analyzer.identify_high_correlations(corr_matrix, threshold=0.8)
        
        assert len(high_corrs) == 1
        assert all(abs(corr[2]) >= 0.8 for corr in high_corrs)
        
        # With threshold 0.6, all pairs should be identified
        high_corrs_low = self.analyzer.identify_high_correlations(corr_matrix, threshold=0.6)
        
        assert len(high_corrs_low) == 3
        assert all(abs(corr[2]) >= 0.6 for corr in high_corrs_low)