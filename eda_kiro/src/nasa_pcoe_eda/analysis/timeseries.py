"""Time series analysis module for identifying temporal patterns and trends."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

from nasa_pcoe_eda.models import SeasonalityResult, TrendReport


class TimeSeriesAnalyzer:
    """Analyzer for time series patterns, trends, and seasonality detection."""

    def identify_temporal_features(self, df: pd.DataFrame) -> List[str]:
        """
        Identify time-based features in the dataset.

        Args:
            df: Input DataFrame

        Returns:
            List of column names that are identified as temporal features
        """
        temporal_features = []

        for column in df.columns:
            # Check if column is datetime type
            if pd.api.types.is_datetime64_any_dtype(df[column]):
                temporal_features.append(column)
                continue

            # Check for temporal naming patterns
            column_lower = column.lower()
            temporal_keywords = [
                'time', 'date', 'timestamp', 'cycle', 'step', 'epoch',
                'hour', 'day', 'month', 'year', 'second', 'minute',
                'period', 'sequence', 'index'
            ]

            if any(keyword in column_lower for keyword in temporal_keywords):
                # Additional check: if it's numeric and monotonic, likely temporal
                if pd.api.types.is_numeric_dtype(df[column]):
                    # Check if values are mostly monotonic (allowing some noise)
                    series = df[column].dropna()
                    if len(series) > 1:
                        # Calculate monotonicity ratio
                        diff = series.diff().dropna()
                        if len(diff) > 0:
                            positive_changes = (diff > 0).sum()
                            negative_changes = (diff < 0).sum()
                            total_changes = len(diff)
                            
                            # If more than 70% of changes are in one direction, consider it temporal
                            monotonic_ratio = max(positive_changes, negative_changes) / total_changes
                            if monotonic_ratio > 0.7:
                                temporal_features.append(column)

        return temporal_features

    def compute_trends(self, df: pd.DataFrame, features: List[str]) -> TrendReport:
        """
        Compute trend statistics for specified features.

        Args:
            df: Input DataFrame
            features: List of feature names to analyze for trends

        Returns:
            TrendReport containing trend statistics and directions
        """
        trends = {}
        trend_directions = {}

        # First, identify temporal features to use as x-axis
        temporal_features = self.identify_temporal_features(df)
        
        # Use the first temporal feature as time axis, or row index if none found
        if temporal_features:
            time_column = temporal_features[0]
            x_values = df[time_column].dropna()
        else:
            # Use row index as time proxy
            x_values = pd.Series(range(len(df)), index=df.index)

        for feature in features:
            if feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
                # Get non-null values aligned with time values
                feature_series = df[feature]
                
                # Align time and feature values (remove rows where either is null)
                combined = pd.DataFrame({'time': x_values, 'feature': feature_series}).dropna()
                
                if len(combined) < 2:
                    # Not enough data for trend analysis
                    trends[feature] = {
                        'slope': 0.0,
                        'intercept': 0.0,
                        'r_value': 0.0,
                        'p_value': 1.0,
                        'std_err': 0.0
                    }
                    trend_directions[feature] = 'insufficient_data'
                    continue

                x = combined['time'].values
                y = combined['feature'].values

                # Convert datetime to numeric if needed
                if pd.api.types.is_datetime64_any_dtype(x):
                    # Convert datetime to timestamp (seconds since epoch)
                    x = (pd.Series(x).astype('int64') // 10**9).values

                # Ensure we have numpy arrays for linregress
                x_array = np.asarray(x, dtype=float)
                y_array = np.asarray(y, dtype=float)

                # Perform linear regression
                linregress_result = stats.linregress(x_array, y_array)
                slope = linregress_result.slope
                intercept = linregress_result.intercept
                r_value = linregress_result.rvalue
                p_value = linregress_result.pvalue
                std_err = linregress_result.stderr

                trends[feature] = {
                    'slope': float(slope),
                    'intercept': float(intercept),
                    'r_value': float(r_value),
                    'p_value': float(p_value),
                    'std_err': float(std_err)
                }

                # Determine trend direction
                if p_value < 0.05:  # Statistically significant
                    if slope > 0:
                        trend_directions[feature] = 'increasing'
                    elif slope < 0:
                        trend_directions[feature] = 'decreasing'
                    else:
                        trend_directions[feature] = 'stable'
                else:
                    trend_directions[feature] = 'no_significant_trend'

        return TrendReport(trends=trends, trend_directions=trend_directions)

    def detect_seasonality(self, df: pd.DataFrame, feature: str) -> SeasonalityResult:
        """
        Detect seasonality or periodic patterns in a feature.

        Args:
            df: Input DataFrame
            feature: Feature name to analyze for seasonality

        Returns:
            SeasonalityResult indicating presence of seasonality
        """
        if feature not in df.columns or not pd.api.types.is_numeric_dtype(df[feature]):
            return SeasonalityResult(
                has_seasonality=False,
                period=None,
                strength=None
            )

        series = df[feature].dropna()
        
        if len(series) < 10:  # Need minimum data points
            return SeasonalityResult(
                has_seasonality=False,
                period=None,
                strength=None
            )

        # Simple autocorrelation-based seasonality detection
        try:
            # Compute autocorrelation for different lags
            max_lag = min(len(series) // 4, 50)  # Don't check more than 1/4 of data or 50 lags
            autocorrelations = []
            lags = []

            for lag in range(1, max_lag + 1):
                if len(series) > lag:
                    # Calculate autocorrelation at this lag
                    corr = series.autocorr(lag=lag)
                    if not pd.isna(corr):
                        autocorrelations.append(abs(corr))
                        lags.append(lag)

            if not autocorrelations:
                return SeasonalityResult(
                    has_seasonality=False,
                    period=None,
                    strength=None
                )

            autocorr_array = np.array(autocorrelations)
            
            # Find peaks in autocorrelation
            # Use a minimum height threshold and minimum distance between peaks
            min_height = 0.3  # Minimum correlation strength to consider
            min_distance = 2   # Minimum distance between peaks
            
            peaks, properties = find_peaks(
                autocorr_array, 
                height=min_height, 
                distance=min_distance
            )

            if len(peaks) > 0:
                # Get the lag with highest autocorrelation among peaks
                peak_idx = peaks[np.argmax(autocorr_array[peaks])]
                period = lags[peak_idx]
                strength = autocorr_array[peak_idx]

                return SeasonalityResult(
                    has_seasonality=True,
                    period=float(period),
                    strength=float(strength)
                )
            else:
                # Check if there's any significant autocorrelation even without clear peaks
                max_corr = np.max(autocorr_array)
                if max_corr > 0.5:  # Strong correlation threshold
                    max_idx = np.argmax(autocorr_array)
                    period = lags[max_idx]
                    
                    return SeasonalityResult(
                        has_seasonality=True,
                        period=float(period),
                        strength=float(max_corr)
                    )

        except Exception:
            # If any error occurs in seasonality detection, return no seasonality
            pass

        return SeasonalityResult(
            has_seasonality=False,
            period=None,
            strength=None
        )