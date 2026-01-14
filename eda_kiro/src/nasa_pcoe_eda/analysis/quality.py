"""Data quality evaluation analyzer for the NASA PCOE EDA system."""

from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np

from ..models import ValidationResult


class DataQualityAnalyzer:
    """Analyzer for evaluating data quality metrics."""

    def __init__(self):
        """Initialize the DataQualityAnalyzer."""
        pass

    def evaluate_completeness(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Evaluate the completeness of each feature.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping feature names to completeness percentages (0-100)
        """
        if df.empty:
            return {}
        
        completeness = {}
        for column in df.columns:
            non_null_count = df[column].notna().sum()
            total_count = len(df)
            completeness_pct = (non_null_count / total_count) * 100 if total_count > 0 else 0
            completeness[column] = completeness_pct
        
        return completeness

    def detect_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """
        Detect duplicate records in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (duplicate_records_dataframe, duplicate_count)
            where duplicate_records_dataframe contains all rows that are duplicates
            (including the first occurrence) and duplicate_count is the number of duplicate rows
        """
        if df.empty:
            return pd.DataFrame(), 0
        
        # Find all duplicate rows (including first occurrence)
        duplicate_mask = df.duplicated(keep=False)
        duplicate_records = df[duplicate_mask].copy()
        
        # Count only the extra duplicates (excluding first occurrence)
        duplicate_count = df.duplicated().sum()
        
        return duplicate_records, duplicate_count

    def verify_data_type_consistency(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Verify data type consistency within each column.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary mapping feature names to lists of inconsistency issues
        """
        if df.empty:
            return {}
        
        inconsistencies = {}
        
        for column in df.columns:
            issues = []
            series = df[column]
            
            # Skip if all values are null
            if series.isna().all():
                continue
            
            # Check for mixed types in object columns
            if series.dtype == 'object':
                non_null_series = series.dropna()
                if len(non_null_series) > 0:
                    # Get unique types of non-null values
                    types_found = set(type(val).__name__ for val in non_null_series)
                    if len(types_found) > 1:
                        issues.append(f"Mixed data types found: {', '.join(sorted(types_found))}")
            
            # Check for numeric columns that might have string representations
            elif pd.api.types.is_numeric_dtype(series):
                # Check if there are any string-like values that couldn't be converted
                # This is mainly for validation purposes
                pass
            
            if issues:
                inconsistencies[column] = issues
        
        return inconsistencies

    def generate_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive data quality report.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing quality metrics and issues
        """
        if df.empty:
            return {
                'completeness': {},
                'duplicates': {'records': pd.DataFrame(), 'count': 0},
                'type_consistency': {},
                'summary': {
                    'total_records': 0,
                    'total_features': 0,
                    'quality_score': 100.0
                }
            }
        
        # Evaluate completeness
        completeness = self.evaluate_completeness(df)
        
        # Detect duplicates
        duplicate_records, duplicate_count = self.detect_duplicates(df)
        
        # Verify type consistency
        type_issues = self.verify_data_type_consistency(df)
        
        # Calculate overall quality score
        avg_completeness = np.mean(list(completeness.values())) if completeness else 100.0
        duplicate_penalty = (duplicate_count / len(df)) * 100 if len(df) > 0 else 0
        type_penalty = len(type_issues) * 5  # 5% penalty per column with type issues
        
        quality_score = max(0, avg_completeness - duplicate_penalty - type_penalty)
        
        return {
            'completeness': completeness,
            'duplicates': {
                'records': duplicate_records,
                'count': duplicate_count
            },
            'type_consistency': type_issues,
            'summary': {
                'total_records': len(df),
                'total_features': len(df.columns),
                'quality_score': quality_score
            }
        }