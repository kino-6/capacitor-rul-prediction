"""Unit tests for DataQualityAnalyzer."""

import pandas as pd
import numpy as np
import pytest

from nasa_pcoe_eda.analysis.quality import DataQualityAnalyzer


class TestDataQualityAnalyzer:
    """Unit tests for DataQualityAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = DataQualityAnalyzer()

    def test_evaluate_completeness_empty_dataframe(self):
        """Test completeness evaluation with empty DataFrame."""
        df = pd.DataFrame()
        result = self.analyzer.evaluate_completeness(df)
        assert result == {}

    def test_evaluate_completeness_no_missing_values(self):
        """Test completeness evaluation with no missing values."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c'],
            'C': [1.1, 2.2, 3.3]
        })
        result = self.analyzer.evaluate_completeness(df)
        
        expected = {'A': 100.0, 'B': 100.0, 'C': 100.0}
        assert result == expected

    def test_evaluate_completeness_with_missing_values(self):
        """Test completeness evaluation with missing values."""
        df = pd.DataFrame({
            'A': [1, 2, np.nan],
            'B': ['a', np.nan, 'c'],
            'C': [np.nan, np.nan, np.nan]
        })
        result = self.analyzer.evaluate_completeness(df)
        
        # Use approximate comparison for floating point values
        assert abs(result['A'] - 66.66666666666667) < 1e-10
        assert abs(result['B'] - 66.66666666666667) < 1e-10
        assert result['C'] == 0.0

    def test_detect_duplicates_empty_dataframe(self):
        """Test duplicate detection with empty DataFrame."""
        df = pd.DataFrame()
        duplicate_records, duplicate_count = self.analyzer.detect_duplicates(df)
        
        assert duplicate_records.empty
        assert duplicate_count == 0

    def test_detect_duplicates_no_duplicates(self):
        """Test duplicate detection with no duplicates."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c']
        })
        duplicate_records, duplicate_count = self.analyzer.detect_duplicates(df)
        
        assert duplicate_records.empty
        assert duplicate_count == 0

    def test_detect_duplicates_with_duplicates(self):
        """Test duplicate detection with duplicates."""
        df = pd.DataFrame({
            'A': [1, 2, 1, 3],
            'B': ['a', 'b', 'a', 'c']
        })
        duplicate_records, duplicate_count = self.analyzer.detect_duplicates(df)
        
        # Should find 2 rows that are duplicates (rows 0 and 2)
        assert len(duplicate_records) == 2
        assert duplicate_count == 1  # Only count extra duplicates
        
        # Verify the duplicate records are correct
        expected_duplicates = df.iloc[[0, 2]]
        pd.testing.assert_frame_equal(
            duplicate_records.sort_index(),
            expected_duplicates.sort_index()
        )

    def test_detect_duplicates_with_nan_values(self):
        """Test duplicate detection with NaN values."""
        df = pd.DataFrame({
            'A': [1, np.nan, 1, np.nan],
            'B': ['a', 'b', 'a', 'b']
        })
        duplicate_records, duplicate_count = self.analyzer.detect_duplicates(df)
        
        # Should find 4 rows (2 pairs of duplicates)
        assert len(duplicate_records) == 4
        assert duplicate_count == 2  # Two extra duplicates

    def test_verify_data_type_consistency_empty_dataframe(self):
        """Test type consistency verification with empty DataFrame."""
        df = pd.DataFrame()
        result = self.analyzer.verify_data_type_consistency(df)
        assert result == {}

    def test_verify_data_type_consistency_consistent_types(self):
        """Test type consistency verification with consistent types."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c'],
            'C': [1.1, 2.2, 3.3]
        })
        result = self.analyzer.verify_data_type_consistency(df)
        assert result == {}

    def test_verify_data_type_consistency_mixed_types(self):
        """Test type consistency verification with mixed types."""
        df = pd.DataFrame({
            'A': [1, 'string', 3.14],  # Mixed int, str, float
            'B': ['a', 'b', 'c']       # Consistent strings
        })
        result = self.analyzer.verify_data_type_consistency(df)
        
        # Should detect mixed types in column A
        assert 'A' in result
        assert len(result['A']) > 0
        assert 'Mixed data types found' in result['A'][0]
        
        # Column B should be fine
        assert 'B' not in result

    def test_generate_quality_report_empty_dataframe(self):
        """Test quality report generation with empty DataFrame."""
        df = pd.DataFrame()
        result = self.analyzer.generate_quality_report(df)
        
        expected = {
            'completeness': {},
            'duplicates': {'records': pd.DataFrame(), 'count': 0},
            'type_consistency': {},
            'summary': {
                'total_records': 0,
                'total_features': 0,
                'quality_score': 100.0
            }
        }
        
        assert result['completeness'] == expected['completeness']
        assert result['duplicates']['count'] == expected['duplicates']['count']
        assert result['duplicates']['records'].empty
        assert result['type_consistency'] == expected['type_consistency']
        assert result['summary'] == expected['summary']

    def test_generate_quality_report_perfect_quality(self):
        """Test quality report generation with perfect quality data."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c'],
            'C': [1.1, 2.2, 3.3]
        })
        result = self.analyzer.generate_quality_report(df)
        
        # Should have perfect completeness
        assert all(comp == 100.0 for comp in result['completeness'].values())
        
        # Should have no duplicates
        assert result['duplicates']['count'] == 0
        assert result['duplicates']['records'].empty
        
        # Should have no type issues
        assert result['type_consistency'] == {}
        
        # Should have perfect quality score
        assert result['summary']['quality_score'] == 100.0
        assert result['summary']['total_records'] == 3
        assert result['summary']['total_features'] == 3

    def test_generate_quality_report_with_issues(self):
        """Test quality report generation with quality issues."""
        df = pd.DataFrame({
            'A': [1, 2, 1, np.nan],     # Has missing value, but no duplicates (1,2,1,nan are not all identical)
            'B': ['a', 'b', 'a', 'c'],  # Has missing value in A row, but no duplicates
            'C': [1, 'string', 3, 4]    # Has mixed types
        })
        result = self.analyzer.generate_quality_report(df)
        
        # Should have some completeness issues
        assert result['completeness']['A'] == 75.0  # 3/4 * 100
        assert result['completeness']['B'] == 100.0
        assert result['completeness']['C'] == 100.0
        
        # Should have no duplicates (no complete row duplicates)
        assert result['duplicates']['count'] == 0
        assert len(result['duplicates']['records']) == 0
        
        # Should have type consistency issues
        assert 'C' in result['type_consistency']
        
        # Quality score should be less than 100
        assert result['summary']['quality_score'] < 100.0
        assert result['summary']['total_records'] == 4
        assert result['summary']['total_features'] == 3

    def test_generate_quality_report_with_duplicates(self):
        """Test quality report generation with actual duplicates."""
        df = pd.DataFrame({
            'A': [1, 2, 1, 2],     # Has duplicates: rows 0&2 and 1&3 are identical
            'B': ['a', 'b', 'a', 'b'],  # Has duplicates
            'C': [1.0, 2.0, 1.0, 2.0]    # Has duplicates
        })
        result = self.analyzer.generate_quality_report(df)
        
        # Should have perfect completeness
        assert all(comp == 100.0 for comp in result['completeness'].values())
        
        # Should have duplicates
        assert result['duplicates']['count'] == 2  # Two extra duplicates
        assert len(result['duplicates']['records']) == 4  # All rows are duplicates
        
        # Should have no type consistency issues
        assert result['type_consistency'] == {}
        
        # Quality score should be less than 100 due to duplicates
        assert result['summary']['quality_score'] < 100.0
        assert result['summary']['total_records'] == 4
        assert result['summary']['total_features'] == 3