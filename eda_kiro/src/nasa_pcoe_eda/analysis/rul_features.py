"""
RUL (Remaining Useful Life) feature analysis module for identifying degradation patterns.

This module specializes in analyzing features for their utility in RUL prediction:
- Degradation trend identification using statistical tests
- Degradation rate computation over time
- Feature ranking based on correlation with RUL
- Degradation pattern visualization
- Time-series analysis for prognostics applications

The module is specifically designed for prognostics applications where
understanding feature degradation over time is crucial for accurate
RUL prediction model development.

Example usage:
    analyzer = RULFeatureAnalyzer()
    degradation_features = analyzer.identify_degradation_features(df)
    degradation_rates = analyzer.compute_degradation_rates(df, degradation_features)
    feature_ranking = analyzer.rank_features_for_rul(df, 'rul_column')
    visualizations = analyzer.visualize_degradation_patterns(df, degradation_features)
"""

from typing import Dict, List, Tuple, Optional
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
from scipy import stats

from ..models import TrendReport
from ..exceptions import AnalysisError


class RULFeatureAnalyzer:
    """Analyzer for identifying features useful for RUL prediction."""

    def __init__(self):
        """Initialize the RUL feature analyzer."""
        pass

    def identify_degradation_features(self, df: pd.DataFrame) -> List[str]:
        """
        Identify features that show degradation trends over time.
        
        Args:
            df: Input DataFrame with time-indexed data
            
        Returns:
            List of feature names that show significant degradation trends
        """
        if df.empty:
            return []
            
        degradation_features = []
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Try to identify time column
        time_column = self._identify_time_column(df)
        if time_column is None:
            # If no time column, use index as time
            time_values = np.arange(len(df))
        else:
            time_values = df[time_column].values
            numeric_columns = numeric_columns.drop(time_column, errors='ignore')
        
        for feature in numeric_columns:
            if self._has_degradation_trend(df[feature].values, time_values):
                # Ensure feature name is always a string
                feature_name = str(feature)
                degradation_features.append(feature_name)
                
        return degradation_features

    def compute_degradation_rates(
        self, df: pd.DataFrame, features: List[str]
    ) -> Dict[str, float]:
        """
        Compute degradation rates for specified features.
        
        Args:
            df: Input DataFrame
            features: List of feature names to analyze
            
        Returns:
            Dictionary mapping feature names to degradation rates
        """
        if df.empty or not features:
            return {}
            
        degradation_rates = {}
        
        # Try to identify time column
        time_column = self._identify_time_column(df)
        if time_column is None:
            time_values = np.arange(len(df))
        else:
            time_values = df[time_column].values
            
        for feature in features:
            # Handle both string and integer column names
            actual_column = None
            if feature in df.columns:
                actual_column = feature
            elif feature.isdigit() and int(feature) in df.columns:
                actual_column = int(feature)
            elif isinstance(feature, str):
                # Try to find matching column by converting types
                for col in df.columns:
                    if str(col) == feature:
                        actual_column = col
                        break
            
            if actual_column is None:
                continue
                
            rate = self._compute_feature_degradation_rate(
                df[actual_column].values, time_values
            )
            # Ensure feature name is always a string
            feature_name = str(feature)
            degradation_rates[feature_name] = rate
            
        return degradation_rates

    def rank_features_for_rul(
        self, df: pd.DataFrame, rul_column: str
    ) -> List[Tuple[str, float]]:
        """
        Rank features by their correlation with RUL.
        
        Args:
            df: Input DataFrame
            rul_column: Name of the RUL column
            
        Returns:
            List of tuples (feature_name, correlation) sorted by absolute correlation
        """
        if df.empty or rul_column not in df.columns:
            return []
            
        feature_correlations = []
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        # Remove RUL column from features
        feature_columns = numeric_columns.drop(rul_column, errors='ignore')
        
        rul_values = df[rul_column].dropna()
        
        for feature in feature_columns:
            # Get overlapping non-null values
            common_idx = df[feature].notna() & df[rul_column].notna()
            if common_idx.sum() < 2:  # Need at least 2 points for correlation
                continue
                
            feature_values = df.loc[common_idx, feature]
            rul_subset = df.loc[common_idx, rul_column]
            
            try:
                correlation, p_value = stats.pearsonr(feature_values, rul_subset)
                if not np.isnan(correlation):
                    # Ensure feature name is always a string
                    feature_name = str(feature)
                    feature_correlations.append((feature_name, float(correlation)))
            except Exception:
                continue
                
        # Sort by absolute correlation (descending)
        feature_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return feature_correlations

    def visualize_degradation_patterns(
        self, df: pd.DataFrame, features: List[str]
    ) -> List[matplotlib.figure.Figure]:
        """
        Visualize degradation patterns for specified features.
        
        Args:
            df: Input DataFrame
            features: List of feature names to visualize
            
        Returns:
            List of matplotlib Figure objects
        """
        if df.empty or not features:
            return []
            
        figures = []
        
        # Try to identify time column
        time_column = self._identify_time_column(df)
        if time_column is None:
            time_values = np.arange(len(df))
            time_label = "Time Index"
        else:
            time_values = df[time_column].values
            time_label = time_column
            
        # Create individual plots for each feature
        for feature in features:
            if feature not in df.columns:
                continue
                
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot the feature values over time
            valid_idx = df[feature].notna()
            ax.plot(
                time_values[valid_idx], 
                df[feature][valid_idx], 
                'o-', 
                alpha=0.7,
                label=f'{feature} 実測値'
            )
            
            # Add trend line
            if valid_idx.sum() > 1:
                try:
                    # Fit linear trend using numpy polyfit
                    X = time_values[valid_idx]
                    y = df[feature][valid_idx].values
                    
                    # Fit linear polynomial (degree 1)
                    coeffs = np.polyfit(X, y, 1)
                    trend_y = np.polyval(coeffs, X)
                    
                    ax.plot(
                        time_values[valid_idx], 
                        trend_y, 
                        '--', 
                        color='red',
                        alpha=0.8,
                        label=f'トレンド線 (傾き: {coeffs[0]:.4f})'
                    )
                except Exception:
                    pass
            
            ax.set_xlabel(time_label)
            ax.set_ylabel(feature)
            ax.set_title(f'{feature} の劣化パターン')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            figures.append(fig)
            
        # Create summary plot if multiple features
        if len(features) > 1:
            fig, axes = plt.subplots(
                len(features), 1, 
                figsize=(12, 4 * len(features)),
                sharex=True
            )
            
            if len(features) == 1:
                axes = [axes]
                
            for i, feature in enumerate(features):
                if feature not in df.columns:
                    continue
                    
                ax = axes[i]
                valid_idx = df[feature].notna()
                
                # Normalize values for comparison
                values = df[feature][valid_idx].values
                if len(values) > 0:
                    normalized_values = (values - values.min()) / (values.max() - values.min() + 1e-8)
                    
                    ax.plot(
                        time_values[valid_idx], 
                        normalized_values, 
                        'o-', 
                        alpha=0.7,
                        label=f'{feature} (正規化)'
                    )
                
                ax.set_ylabel(f'{feature}\n(正規化)')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
            axes[-1].set_xlabel(time_label)
            fig.suptitle('劣化特徴量の比較 (正規化済み)', fontsize=14)
            plt.tight_layout()
            
            figures.append(fig)
            
        return figures

    def _identify_time_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Try to identify a time-based column in the DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Name of the time column if found, None otherwise
        """
        # Look for datetime columns
        datetime_columns = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_columns) > 0:
            return str(datetime_columns[0])
            
        # Look for columns with time-related names
        time_keywords = ['time', 'date', 'timestamp', 'cycle', 'step', 'index']
        for col in df.columns:
            # Convert column name to string and then to lowercase for comparison
            col_str = str(col).lower()
            if any(keyword in col_str for keyword in time_keywords):
                return str(col)
                
        return None

    def _has_degradation_trend(
        self, values: np.ndarray, time_values: np.ndarray
    ) -> bool:
        """
        Check if a feature shows a significant degradation trend.
        
        Args:
            values: Feature values
            time_values: Time values
            
        Returns:
            True if significant trend is detected
        """
        # Remove NaN values
        valid_idx = ~np.isnan(values)
        if valid_idx.sum() < 3:  # Need at least 3 points
            return False
            
        clean_values = values[valid_idx]
        clean_time = time_values[valid_idx]
        
        try:
            # Compute correlation with time
            correlation, p_value = stats.pearsonr(clean_time, clean_values)
            
            # Consider significant if |correlation| > 0.3 and p < 0.05
            return abs(correlation) > 0.3 and p_value < 0.05
            
        except Exception:
            return False

    def _compute_feature_degradation_rate(
        self, values: np.ndarray, time_values: np.ndarray
    ) -> float:
        """
        Compute the degradation rate for a feature.
        
        Args:
            values: Feature values
            time_values: Time values
            
        Returns:
            Degradation rate (slope of linear fit)
        """
        # Remove NaN values
        valid_idx = ~np.isnan(values)
        if valid_idx.sum() < 2:
            return 0.0
            
        clean_values = values[valid_idx]
        clean_time = time_values[valid_idx]
        
        try:
            # Fit linear regression using numpy polyfit
            coeffs = np.polyfit(clean_time, clean_values, 1)
            return float(coeffs[0])  # Return slope
            
        except Exception:
            return 0.0