"""Unit tests for OutlierDetector."""

import numpy as np
import pandas as pd
import pytest

from nasa_pcoe_eda.analysis.outliers import OutlierDetector
from nasa_pcoe_eda.models import OutlierSummary


class TestOutlierDetector:
    """Test cases for OutlierDetector class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = OutlierDetector()
        
        # Create test data with known outliers
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 100)
        
        # Add some clear outliers
        outlier_data = normal_data.copy()
        outlier_data[0] = 10  # Clear outlier
        outlier_data[1] = -10  # Clear outlier
        
        self.test_df = pd.DataFrame({
            'normal': normal_data,
            'with_outliers': outlier_data,
            'constant': np.ones(100),  # No outliers possible
            'text': ['text'] * 100  # Non-numeric column
        })

    def test_detect_outliers_iqr_basic(self):
        """Test basic IQR outlier detection."""
        outliers = self.detector.detect_outliers_iqr(self.test_df)
        
        # Should only process numeric columns
        assert 'normal' in outliers
        assert 'with_outliers' in outliers
        assert 'constant' in outliers
        assert 'text' not in outliers
        
        # The 'with_outliers' column should detect the extreme values
        assert len(outliers['with_outliers']) >= 2  # At least the two extreme outliers
        assert 0 in outliers['with_outliers']  # Index 0 has value 10
        assert 1 in outliers['with_outliers']  # Index 1 has value -10

    def test_detect_outliers_iqr_threshold(self):
        """Test IQR outlier detection with different thresholds."""
        # Stricter threshold should detect fewer outliers
        strict_outliers = self.detector.detect_outliers_iqr(self.test_df, threshold=3.0)
        loose_outliers = self.detector.detect_outliers_iqr(self.test_df, threshold=1.0)
        
        # Stricter threshold should detect fewer or equal outliers
        for column in strict_outliers:
            if column in loose_outliers:
                assert len(strict_outliers[column]) <= len(loose_outliers[column])

    def test_detect_outliers_zscore_basic(self):
        """Test basic Z-score outlier detection."""
        outliers = self.detector.detect_outliers_zscore(self.test_df)
        
        # Should only process numeric columns
        assert 'normal' in outliers
        assert 'with_outliers' in outliers
        assert 'constant' in outliers
        assert 'text' not in outliers
        
        # The 'with_outliers' column should detect the extreme values
        assert len(outliers['with_outliers']) >= 2  # At least the two extreme outliers
        assert 0 in outliers['with_outliers']  # Index 0 has value 10
        assert 1 in outliers['with_outliers']  # Index 1 has value -10

    def test_detect_outliers_zscore_threshold(self):
        """Test Z-score outlier detection with different thresholds."""
        # Stricter threshold should detect fewer outliers
        strict_outliers = self.detector.detect_outliers_zscore(self.test_df, threshold=4.0)
        loose_outliers = self.detector.detect_outliers_zscore(self.test_df, threshold=2.0)
        
        # Stricter threshold should detect fewer or equal outliers
        for column in strict_outliers:
            if column in loose_outliers:
                assert len(strict_outliers[column]) <= len(loose_outliers[column])

    def test_summarize_outliers(self):
        """Test outlier summary generation."""
        outliers = self.detector.detect_outliers_iqr(self.test_df)
        summary = self.detector.summarize_outliers(outliers)
        
        # Check that summary is of correct type
        assert isinstance(summary, OutlierSummary)
        
        # Check that all numeric columns are included
        assert 'normal' in summary.outlier_counts
        assert 'with_outliers' in summary.outlier_counts
        assert 'constant' in summary.outlier_counts
        
        # Check that counts and percentages are consistent
        for feature in summary.outlier_counts:
            count = summary.outlier_counts[feature]
            percentage = summary.outlier_percentages[feature]
            indices = summary.outlier_indices[feature]
            
            # Count should match length of indices
            assert count == len(indices)
            
            # Percentage should be non-negative
            assert percentage >= 0

    def test_empty_dataframe(self):
        """Test behavior with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        iqr_outliers = self.detector.detect_outliers_iqr(empty_df)
        zscore_outliers = self.detector.detect_outliers_zscore(empty_df)
        
        assert iqr_outliers == {}
        assert zscore_outliers == {}
        
        summary = self.detector.summarize_outliers(iqr_outliers)
        assert summary.outlier_counts == {}
        assert summary.outlier_percentages == {}

    def test_all_nan_column(self):
        """Test behavior with column containing all NaN values."""
        nan_df = pd.DataFrame({
            'all_nan': [np.nan] * 10,
            'normal': np.random.normal(0, 1, 10)
        })
        
        iqr_outliers = self.detector.detect_outliers_iqr(nan_df)
        zscore_outliers = self.detector.detect_outliers_zscore(nan_df)
        
        # All NaN column should have no outliers
        assert len(iqr_outliers['all_nan']) == 0
        assert len(zscore_outliers['all_nan']) == 0

    def test_single_value_column(self):
        """Test behavior with column containing single unique value."""
        constant_df = pd.DataFrame({
            'constant': [5.0] * 10,
            'normal': np.random.normal(0, 1, 10)
        })
        
        iqr_outliers = self.detector.detect_outliers_iqr(constant_df)
        zscore_outliers = self.detector.detect_outliers_zscore(constant_df)
        
        # Constant column should have no outliers (IQR = 0)
        assert len(iqr_outliers['constant']) == 0
        # Z-score method might have issues with zero std, but should handle gracefully
        assert len(zscore_outliers['constant']) == 0