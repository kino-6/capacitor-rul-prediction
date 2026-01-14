"""
Outlier detection functionality for the NASA PCOE EDA system.

This module provides comprehensive outlier detection using multiple statistical methods:
- Interquartile Range (IQR) method for robust outlier detection
- Z-score method for parametric outlier detection
- Configurable thresholds for different sensitivity levels
- Comprehensive outlier summarization and reporting
- Support for multiple features simultaneously

The module is designed to handle various data distributions and provides
detailed outlier statistics including counts, percentages, and indices
for further analysis or data cleaning.

Example usage:
    detector = OutlierDetector()
    iqr_outliers = detector.detect_outliers_iqr(df, threshold=1.5)
    zscore_outliers = detector.detect_outliers_zscore(df, threshold=3.0)
    summary = detector.summarize_outliers(iqr_outliers, total_records=len(df))
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..models import OutlierSummary


class OutlierDetector:
    """Detects outliers and anomalies in datasets using various statistical methods."""

    def detect_outliers_iqr(
        self, df: pd.DataFrame, threshold: float = 1.5
    ) -> Dict[str, np.ndarray]:
        """
        Detect outliers using the Interquartile Range (IQR) method.
        
        Args:
            df: DataFrame to analyze
            threshold: IQR multiplier threshold (default: 1.5)
            
        Returns:
            Dictionary mapping feature names to arrays of outlier indices
        """
        outliers = {}
        
        # Only process numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            series = df[column].dropna()
            if len(series) == 0:
                outliers[column] = np.array([], dtype=int)
                continue
                
            # Calculate quartiles and IQR
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            
            # Calculate bounds
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            # Find outliers
            outlier_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
            outlier_indices = df[outlier_mask].index.to_numpy()
            
            outliers[column] = outlier_indices
            
        return outliers

    def detect_outliers_zscore(
        self, df: pd.DataFrame, threshold: float = 3.0
    ) -> Dict[str, np.ndarray]:
        """
        Detect outliers using the Z-score method.
        
        Args:
            df: DataFrame to analyze
            threshold: Z-score threshold (default: 3.0)
            
        Returns:
            Dictionary mapping feature names to arrays of outlier indices
        """
        outliers = {}
        
        # Only process numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            series = df[column].dropna()
            if len(series) == 0:
                outliers[column] = np.array([], dtype=int)
                continue
                
            # Check if standard deviation is zero (constant values)
            if series.std() == 0:
                # No outliers possible in constant data
                outliers[column] = np.array([], dtype=int)
                continue
                
            # Calculate z-scores
            z_scores = np.abs(stats.zscore(series))
            
            # Find outliers
            outlier_mask = z_scores > threshold
            
            # Map back to original DataFrame indices
            outlier_indices = series[outlier_mask].index.to_numpy()
            
            outliers[column] = outlier_indices
            
        return outliers

    def summarize_outliers(
        self, outliers: Dict[str, np.ndarray], total_records: Optional[int] = None
    ) -> OutlierSummary:
        """
        Summarize outlier detection results.
        
        Args:
            outliers: Dictionary mapping feature names to outlier indices
            total_records: Total number of records in the dataset (optional)
            
        Returns:
            OutlierSummary containing counts, percentages, and indices
        """
        outlier_counts = {}
        outlier_percentages = {}
        
        # Determine total number of records
        if total_records is None:
            if outliers:
                # Estimate from the maximum index + 1
                max_index = 0
                for indices in outliers.values():
                    if len(indices) > 0:
                        max_index = max(max_index, indices.max())
                total_records = max_index + 1 if max_index > 0 else 1
            else:
                total_records = 1
            
        for feature, indices in outliers.items():
            count = len(indices)
            percentage = (count / total_records) * 100 if total_records > 0 else 0.0
            
            outlier_counts[feature] = count
            outlier_percentages[feature] = percentage
            
        return OutlierSummary(
            outlier_counts=outlier_counts,
            outlier_percentages=outlier_percentages,
            outlier_indices=outliers
        )