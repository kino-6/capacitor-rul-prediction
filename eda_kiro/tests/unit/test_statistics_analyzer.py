"""Unit tests for StatisticsAnalyzer."""

import numpy as np
import pandas as pd
import pytest

from nasa_pcoe_eda.analysis.statistics import StatisticsAnalyzer
from nasa_pcoe_eda.models import MissingValueReport, Stats


class TestStatisticsAnalyzer:
    """Test suite for StatisticsAnalyzer."""

    @pytest.fixture
    def analyzer(self):
        """Create a StatisticsAnalyzer instance."""
        return StatisticsAnalyzer()

    @pytest.fixture
    def sample_numeric_df(self):
        """Create a sample DataFrame with numeric data."""
        return pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature2": [10.0, 20.0, 30.0, 40.0, 50.0],
                "feature3": [100.0, 200.0, 300.0, 400.0, 500.0],
            }
        )

    @pytest.fixture
    def df_with_missing(self):
        """Create a DataFrame with missing values."""
        return pd.DataFrame(
            {
                "col1": [1.0, 2.0, np.nan, 4.0, 5.0],
                "col2": [10.0, np.nan, np.nan, 40.0, 50.0],
                "col3": [100.0, 200.0, 300.0, 400.0, 500.0],
            }
        )

    @pytest.fixture
    def mixed_type_df(self):
        """Create a DataFrame with mixed data types."""
        return pd.DataFrame(
            {
                "numeric": [1, 2, 3, 4, 5],
                "float": [1.1, 2.2, 3.3, 4.4, 5.5],
                "string": ["a", "b", "c", "d", "e"],
                "bool": [True, False, True, False, True],
            }
        )

    def test_compute_descriptive_stats_basic(self, analyzer, sample_numeric_df):
        """Test basic descriptive statistics computation."""
        stats = analyzer.compute_descriptive_stats(sample_numeric_df)

        # Check that we have stats for all numeric columns
        assert len(stats) == 3
        assert "feature1" in stats
        assert "feature2" in stats
        assert "feature3" in stats

        # Verify feature1 statistics
        assert stats["feature1"].mean == 3.0
        assert stats["feature1"].median == 3.0
        assert stats["feature1"].min == 1.0
        assert stats["feature1"].max == 5.0
        assert stats["feature1"].q25 == 2.0
        assert stats["feature1"].q75 == 4.0

    def test_compute_descriptive_stats_with_known_values(self, analyzer):
        """Test statistics with known values."""
        df = pd.DataFrame({"values": [1.0, 2.0, 3.0, 4.0, 5.0]})
        stats = analyzer.compute_descriptive_stats(df)

        assert stats["values"].mean == 3.0
        assert stats["values"].median == 3.0
        assert abs(stats["values"].std - 1.5811388300841898) < 1e-10
        assert stats["values"].min == 1.0
        assert stats["values"].max == 5.0

    def test_compute_descriptive_stats_with_missing_values(
        self, analyzer, df_with_missing
    ):
        """Test that statistics handle missing values correctly."""
        stats = analyzer.compute_descriptive_stats(df_with_missing)

        # col1 has 1 NaN, should compute stats on remaining 4 values
        assert stats["col1"].mean == 3.0  # (1+2+4+5)/4
        assert stats["col1"].min == 1.0
        assert stats["col1"].max == 5.0

        # col2 has 2 NaN, should compute stats on remaining 3 values
        assert abs(stats["col2"].mean - 33.333333) < 0.001

    def test_compute_descriptive_stats_mixed_types(self, analyzer, mixed_type_df):
        """Test that only numeric columns are processed."""
        stats = analyzer.compute_descriptive_stats(mixed_type_df)

        # Should only have numeric and float columns
        assert len(stats) == 2
        assert "numeric" in stats
        assert "float" in stats
        assert "string" not in stats
        assert "bool" not in stats

    def test_compute_descriptive_stats_returns_stats_objects(
        self, analyzer, sample_numeric_df
    ):
        """Test that the return type is correct."""
        stats = analyzer.compute_descriptive_stats(sample_numeric_df)

        for feature_name, feature_stats in stats.items():
            assert isinstance(feature_stats, Stats)
            assert hasattr(feature_stats, "mean")
            assert hasattr(feature_stats, "median")
            assert hasattr(feature_stats, "std")
            assert hasattr(feature_stats, "min")
            assert hasattr(feature_stats, "max")
            assert hasattr(feature_stats, "q25")
            assert hasattr(feature_stats, "q75")

    def test_analyze_missing_values_no_missing(self, analyzer, sample_numeric_df):
        """Test missing value analysis with no missing values."""
        report = analyzer.analyze_missing_values(sample_numeric_df)

        assert isinstance(report, MissingValueReport)
        assert report.total_missing == 0
        assert all(count == 0 for count in report.missing_counts.values())
        assert all(pct == 0.0 for pct in report.missing_percentages.values())

    def test_analyze_missing_values_with_missing(self, analyzer, df_with_missing):
        """Test missing value analysis with missing values."""
        report = analyzer.analyze_missing_values(df_with_missing)

        assert isinstance(report, MissingValueReport)
        assert report.total_missing == 3  # 1 in col1, 2 in col2
        assert report.missing_counts["col1"] == 1
        assert report.missing_counts["col2"] == 2
        assert report.missing_counts["col3"] == 0

        # Check percentages (5 rows total)
        assert report.missing_percentages["col1"] == 20.0
        assert report.missing_percentages["col2"] == 40.0
        assert report.missing_percentages["col3"] == 0.0

    def test_analyze_missing_values_all_missing(self, analyzer):
        """Test with a column that is entirely missing."""
        df = pd.DataFrame(
            {
                "all_missing": [np.nan, np.nan, np.nan],
                "no_missing": [1.0, 2.0, 3.0],
            }
        )
        report = analyzer.analyze_missing_values(df)

        assert report.missing_counts["all_missing"] == 3
        assert report.missing_percentages["all_missing"] == 100.0
        assert report.missing_counts["no_missing"] == 0
        assert report.total_missing == 3

    def test_identify_data_types_basic(self, analyzer, mixed_type_df):
        """Test data type identification."""
        data_types = analyzer.identify_data_types(mixed_type_df)

        assert len(data_types) == 4
        assert "numeric" in data_types
        assert "float" in data_types
        assert "string" in data_types
        assert "bool" in data_types

        # Check that types are strings
        assert isinstance(data_types["numeric"], str)
        assert isinstance(data_types["float"], str)
        assert isinstance(data_types["string"], str)
        assert isinstance(data_types["bool"], str)

    def test_identify_data_types_numeric_only(self, analyzer, sample_numeric_df):
        """Test data type identification with numeric-only DataFrame."""
        data_types = analyzer.identify_data_types(sample_numeric_df)

        assert len(data_types) == 3
        for col in ["feature1", "feature2", "feature3"]:
            assert col in data_types
            assert "float" in data_types[col].lower()

    def test_identify_data_types_consistency(self, analyzer):
        """Test that data type identification is consistent."""
        df = pd.DataFrame(
            {
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "str_col": ["a", "b", "c"],
            }
        )

        types1 = analyzer.identify_data_types(df)
        types2 = analyzer.identify_data_types(df)

        # Should return the same types
        assert types1 == types2

    def test_empty_dataframe(self, analyzer):
        """Test handling of empty DataFrame."""
        df = pd.DataFrame()

        stats = analyzer.compute_descriptive_stats(df)
        assert len(stats) == 0

        report = analyzer.analyze_missing_values(df)
        assert report.total_missing == 0
        assert len(report.missing_counts) == 0

        data_types = analyzer.identify_data_types(df)
        assert len(data_types) == 0

    def test_single_value_column(self, analyzer):
        """Test with a column containing a single repeated value."""
        df = pd.DataFrame({"constant": [5.0, 5.0, 5.0, 5.0, 5.0]})
        stats = analyzer.compute_descriptive_stats(df)

        assert stats["constant"].mean == 5.0
        assert stats["constant"].median == 5.0
        assert stats["constant"].std == 0.0
        assert stats["constant"].min == 5.0
        assert stats["constant"].max == 5.0
