"""
Unit tests for FeatureQualityChecker.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.data_preparation.quality_checker import FeatureQualityChecker


@pytest.fixture
def sample_features_df():
    """Create a sample features DataFrame for testing."""
    np.random.seed(42)
    
    # Create sample data with 100 samples
    data = {
        'capacitor_id': ['ES12C1'] * 50 + ['ES12C2'] * 50,
        'cycle': list(range(1, 51)) + list(range(1, 51)),
        'cycle_number': list(range(1, 51)) + list(range(1, 51)),
        'cycle_normalized': [i/200 for i in range(1, 51)] + [i/200 for i in range(1, 51)],
        'vl_mean': np.random.normal(0, 1, 100),
        'vo_mean': np.random.normal(0, 1, 100),
        'voltage_ratio': np.random.uniform(0.5, 1.5, 100),
        'response_efficiency': np.random.uniform(0.8, 1.2, 100),
    }
    
    # Add some outliers
    data['vl_mean'][0] = 10.0  # Outlier
    data['vo_mean'][1] = -10.0  # Outlier
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_csv_file(sample_features_df, temp_dir):
    """Create a temporary CSV file with sample data."""
    csv_path = Path(temp_dir) / 'test_features.csv'
    sample_features_df.to_csv(csv_path, index=False)
    return str(csv_path)


class TestFeatureQualityChecker:
    """Test cases for FeatureQualityChecker."""
    
    def test_initialization(self, sample_csv_file):
        """Test checker initialization."""
        checker = FeatureQualityChecker(sample_csv_file)
        assert checker.features_path == Path(sample_csv_file)
        assert checker.df is None
        assert checker.report == []
    
    def test_load_data(self, sample_csv_file):
        """Test data loading."""
        checker = FeatureQualityChecker(sample_csv_file)
        checker.load_data()
        
        assert checker.df is not None
        assert len(checker.df) == 100
        assert 'capacitor_id' in checker.df.columns
        assert 'vl_mean' in checker.df.columns
    
    def test_check_missing_values_no_missing(self, sample_csv_file):
        """Test missing value check with no missing values."""
        checker = FeatureQualityChecker(sample_csv_file)
        checker.load_data()
        
        missing = checker.check_missing_values()
        
        assert len(missing) == 0
        assert any('No missing values' in msg for msg in checker.report)
    
    def test_check_missing_values_with_missing(self, sample_features_df, temp_dir):
        """Test missing value check with missing values."""
        # Add missing values
        sample_features_df.loc[0, 'vl_mean'] = np.nan
        sample_features_df.loc[1, 'vo_mean'] = np.nan
        
        csv_path = Path(temp_dir) / 'test_missing.csv'
        sample_features_df.to_csv(csv_path, index=False)
        
        checker = FeatureQualityChecker(str(csv_path))
        checker.load_data()
        
        missing = checker.check_missing_values()
        
        assert len(missing) == 2
        assert 'vl_mean' in missing
        assert 'vo_mean' in missing
        assert missing['vl_mean'] == 1
        assert missing['vo_mean'] == 1
    
    def test_detect_outliers_iqr(self, sample_csv_file):
        """Test IQR outlier detection."""
        checker = FeatureQualityChecker(sample_csv_file)
        checker.load_data()
        
        outliers, lower, upper = checker.detect_outliers_iqr('vl_mean')
        
        assert isinstance(outliers, list)
        assert len(outliers) > 0  # We added an outlier
        assert 0 in outliers  # First row has outlier value 10.0
        assert lower < upper
    
    def test_detect_outliers_zscore(self, sample_csv_file):
        """Test Z-score outlier detection."""
        checker = FeatureQualityChecker(sample_csv_file)
        checker.load_data()
        
        outliers = checker.detect_outliers_zscore('vl_mean', threshold=3.0)
        
        assert isinstance(outliers, list)
        assert len(outliers) > 0  # We added an outlier
        assert 0 in outliers  # First row has outlier value 10.0
    
    def test_check_outliers(self, sample_csv_file):
        """Test comprehensive outlier check."""
        checker = FeatureQualityChecker(sample_csv_file)
        checker.load_data()
        
        outlier_summary = checker.check_outliers()
        
        assert isinstance(outlier_summary, dict)
        assert 'vl_mean' in outlier_summary
        assert 'vo_mean' in outlier_summary
        assert 'iqr_outliers' in outlier_summary['vl_mean']
        assert 'zscore_outliers' in outlier_summary['vl_mean']
    
    def test_generate_statistical_summary(self, sample_csv_file):
        """Test statistical summary generation."""
        checker = FeatureQualityChecker(sample_csv_file)
        checker.load_data()
        
        summary = checker.generate_statistical_summary()
        
        assert isinstance(summary, pd.DataFrame)
        assert 'mean' in summary.columns
        assert 'std' in summary.columns
        assert 'skewness' in summary.columns
        assert 'kurtosis' in summary.columns
        assert 'cv' in summary.columns
        assert 'vl_mean' in summary.index
    
    def test_check_data_distribution(self, sample_csv_file):
        """Test data distribution check."""
        checker = FeatureQualityChecker(sample_csv_file)
        checker.load_data()
        
        distribution = checker.check_data_distribution()
        
        assert isinstance(distribution, dict)
        assert 'capacitor_counts' in distribution
        assert 'total_samples' in distribution
        assert 'total_capacitors' in distribution
        assert distribution['total_samples'] == 100
        assert distribution['total_capacitors'] == 2
        assert distribution['capacitor_counts']['ES12C1'] == 50
        assert distribution['capacitor_counts']['ES12C2'] == 50
    
    def test_check_feature_ranges(self, sample_csv_file):
        """Test feature range check."""
        checker = FeatureQualityChecker(sample_csv_file)
        checker.load_data()
        
        ranges = checker.check_feature_ranges()
        
        assert isinstance(ranges, dict)
        assert 'vl_mean' in ranges
        assert 'vo_mean' in ranges
        assert isinstance(ranges['vl_mean'], tuple)
        assert len(ranges['vl_mean']) == 2
        assert ranges['vl_mean'][0] < ranges['vl_mean'][1]  # min < max
    
    def test_save_report(self, sample_csv_file, temp_dir):
        """Test report saving."""
        checker = FeatureQualityChecker(sample_csv_file)
        checker.load_data()
        checker.check_missing_values()
        
        output_path = Path(temp_dir) / 'test_report.txt'
        checker.save_report(str(output_path))
        
        assert output_path.exists()
        
        # Read and verify report content
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assert 'missing values' in content.lower()
        assert len(content) > 0
    
    def test_run_all_checks(self, sample_csv_file):
        """Test running all checks together."""
        checker = FeatureQualityChecker(sample_csv_file)
        checker.run_all_checks()
        
        # Verify all checks were run
        assert checker.df is not None
        assert len(checker.report) > 0
        
        # Check that report contains key sections
        report_text = '\n'.join(checker.report)
        assert 'FEATURE QUALITY REPORT' in report_text
        assert 'missing values' in report_text.lower()
        assert 'Data quality is GOOD' in report_text or 'Data quality issues found' in report_text


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self, temp_dir):
        """Test with empty DataFrame."""
        # Create empty CSV
        empty_df = pd.DataFrame(columns=['capacitor_id', 'cycle', 'vl_mean'])
        csv_path = Path(temp_dir) / 'empty.csv'
        empty_df.to_csv(csv_path, index=False)
        
        checker = FeatureQualityChecker(str(csv_path))
        checker.load_data()
        
        assert len(checker.df) == 0
    
    def test_single_column(self, temp_dir):
        """Test with single numeric column."""
        df = pd.DataFrame({
            'capacitor_id': ['ES12C1'] * 10,
            'value': np.random.normal(0, 1, 10)
        })
        csv_path = Path(temp_dir) / 'single_col.csv'
        df.to_csv(csv_path, index=False)
        
        checker = FeatureQualityChecker(str(csv_path))
        checker.load_data()
        
        # Should not raise error
        missing = checker.check_missing_values()
        assert isinstance(missing, dict)
    
    def test_all_missing_column(self, temp_dir):
        """Test with column that has all missing values."""
        df = pd.DataFrame({
            'capacitor_id': ['ES12C1'] * 10,
            'all_missing': [np.nan] * 10,
            'normal': np.random.normal(0, 1, 10)
        })
        csv_path = Path(temp_dir) / 'all_missing.csv'
        df.to_csv(csv_path, index=False)
        
        checker = FeatureQualityChecker(str(csv_path))
        checker.load_data()
        
        missing = checker.check_missing_values()
        
        assert 'all_missing' in missing
        assert missing['all_missing'] == 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
