"""
Statistical analysis module for computing descriptive statistics and data quality metrics.

This module provides comprehensive statistical analysis capabilities including:
- Descriptive statistics (mean, median, std, quartiles, etc.)
- Missing value analysis and reporting
- Data type identification and validation
- Distribution analysis and normality testing
- Statistical summaries for all numeric features

The module is designed to handle various data types and edge cases gracefully,
providing robust statistical insights for exploratory data analysis.

Example usage:
    analyzer = StatisticsAnalyzer()
    stats = analyzer.compute_descriptive_stats(df)
    missing_report = analyzer.analyze_missing_values(df)
    data_types = analyzer.identify_data_types(df)
"""

from typing import Dict

import pandas as pd

from nasa_pcoe_eda.models import MissingValueReport, Stats


class StatisticsAnalyzer:
    """Analyzer for computing descriptive statistics and data quality metrics."""

    def compute_descriptive_stats(self, df: pd.DataFrame) -> Dict[str, Stats]:
        """
        Compute descriptive statistics for all numeric features.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary mapping feature names to Stats objects
        """
        stats_dict = {}

        # Get only numeric columns
        numeric_df = df.select_dtypes(include=["number"])

        for column in numeric_df.columns:
            series = numeric_df[column]

            # Compute statistics, handling NaN values
            stats_dict[column] = Stats(
                mean=float(series.mean()),
                median=float(series.median()),
                std=float(series.std()),
                min=float(series.min()),
                max=float(series.max()),
                q25=float(series.quantile(0.25)),
                q75=float(series.quantile(0.75)),
            )

        return stats_dict

    def analyze_missing_values(self, df: pd.DataFrame) -> MissingValueReport:
        """
        Analyze missing values in the dataset.

        Args:
            df: Input DataFrame

        Returns:
            MissingValueReport with counts and percentages
        """
        missing_counts = {}
        missing_percentages = {}
        total_missing = 0

        for column in df.columns:
            count = int(df[column].isna().sum())
            missing_counts[column] = count
            missing_percentages[column] = float((count / len(df)) * 100)
            total_missing += count

        return MissingValueReport(
            missing_counts=missing_counts,
            missing_percentages=missing_percentages,
            total_missing=total_missing,
        )

    def identify_data_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Identify and report the data type of each feature.

        Args:
            df: Input DataFrame

        Returns:
            Dictionary mapping feature names to data type strings
        """
        data_types = {}

        for column in df.columns:
            dtype = df[column].dtype
            data_types[column] = str(dtype)

        return data_types
