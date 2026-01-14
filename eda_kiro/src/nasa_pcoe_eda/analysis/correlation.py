"""
Correlation analysis module for computing feature correlations and detecting multicollinearity.

This module provides advanced correlation analysis capabilities including:
- Pairwise correlation matrix computation using Pearson correlation
- High correlation identification with configurable thresholds
- Multicollinearity detection using Variance Inflation Factor (VIF)
- Correlation group identification for feature selection
- Robust handling of missing values and edge cases

The analysis helps identify redundant features and understand relationships
between variables, which is crucial for feature selection and model building.

Example usage:
    analyzer = CorrelationAnalyzer()
    corr_matrix = analyzer.compute_correlation_matrix(df)
    high_corrs = analyzer.identify_high_correlations(corr_matrix, threshold=0.8)
    multicollinearity = analyzer.detect_multicollinearity(df)
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from statsmodels.stats.outliers_influence import variance_inflation_factor  # type: ignore

from nasa_pcoe_eda.models import MulticollinearityReport


class CorrelationAnalyzer:
    """Analyzer for computing correlations and detecting multicollinearity."""

    def compute_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute pairwise correlation matrix for numeric features.

        Args:
            df: Input DataFrame

        Returns:
            Correlation matrix as DataFrame
        """
        # Get only numeric columns
        numeric_df = df.select_dtypes(include=["number"])
        
        if numeric_df.empty:
            return pd.DataFrame()
        
        # Remove columns with all NaN values
        numeric_df = numeric_df.dropna(axis=1, how='all')
        
        if numeric_df.empty:
            return pd.DataFrame()
        
        # Compute correlation matrix using Pearson correlation
        correlation_matrix = numeric_df.corr(method='pearson')
        
        return correlation_matrix

    def identify_high_correlations(
        self, corr_matrix: pd.DataFrame, threshold: float = 0.8
    ) -> List[Tuple[str, str, float]]:
        """
        Identify feature pairs with high correlation.

        Args:
            corr_matrix: Correlation matrix
            threshold: Correlation threshold (default: 0.8)

        Returns:
            List of tuples (feature1, feature2, correlation_value)
        """
        high_correlations: List[Tuple[str, str, float]] = []
        
        if corr_matrix.empty:
            return high_correlations
        
        # Get upper triangle of correlation matrix (avoid duplicates and self-correlation)
        upper_triangle = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        # Find correlations above threshold
        high_corr_mask = (np.abs(corr_matrix) >= threshold) & upper_triangle
        
        # Extract feature pairs and their correlation values
        for i in range(len(corr_matrix.index)):
            for j in range(len(corr_matrix.columns)):
                if high_corr_mask.iloc[i, j]:
                    feature1 = corr_matrix.index[i]
                    feature2 = corr_matrix.columns[j]
                    correlation_value = float(corr_matrix.iloc[i, j])  # type: ignore
                    high_correlations.append((feature1, feature2, correlation_value))
        
        # Sort by absolute correlation value (descending)
        high_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return high_correlations

    def detect_multicollinearity(self, df: pd.DataFrame) -> MulticollinearityReport:
        """
        Detect multicollinearity using Variance Inflation Factor (VIF).

        Args:
            df: Input DataFrame

        Returns:
            MulticollinearityReport with VIF values and correlated groups
        """
        # Get only numeric columns
        numeric_df = df.select_dtypes(include=["number"])
        
        if numeric_df.empty:
            return MulticollinearityReport(
                high_vif_features=[],
                correlated_groups=[]
            )
        
        # Remove columns with all NaN values and drop rows with any NaN
        numeric_df = numeric_df.dropna(axis=1, how='all').dropna()
        
        if numeric_df.empty or len(numeric_df.columns) < 2:
            return MulticollinearityReport(
                high_vif_features=[],
                correlated_groups=[]
            )
        
        # Remove constant columns (zero variance)
        constant_columns = []
        for col in numeric_df.columns:
            if numeric_df[col].nunique() <= 1:
                constant_columns.append(col)
        
        numeric_df = numeric_df.drop(columns=constant_columns)
        
        if numeric_df.empty or len(numeric_df.columns) < 2:
            return MulticollinearityReport(
                high_vif_features=[],
                correlated_groups=[]
            )
        
        # Calculate VIF for each feature
        high_vif_features = []
        vif_threshold = 10.0  # Common threshold for high multicollinearity
        
        try:
            for i, feature in enumerate(numeric_df.columns):
                vif_value = variance_inflation_factor(numeric_df.values, i)
                if vif_value >= vif_threshold:
                    high_vif_features.append((feature, float(vif_value)))
        except Exception:
            # If VIF calculation fails (e.g., due to perfect multicollinearity),
            # fall back to correlation-based detection
            pass
        
        # Sort by VIF value (descending)
        high_vif_features.sort(key=lambda x: x[1], reverse=True)
        
        # Identify correlated groups using correlation matrix
        corr_matrix = self.compute_correlation_matrix(numeric_df)
        correlated_groups = self._identify_correlated_groups(corr_matrix, threshold=0.8)
        
        return MulticollinearityReport(
            high_vif_features=high_vif_features,
            correlated_groups=correlated_groups
        )

    def _identify_correlated_groups(
        self, corr_matrix: pd.DataFrame, threshold: float = 0.8
    ) -> List[List[str]]:
        """
        Identify groups of highly correlated features.

        Args:
            corr_matrix: Correlation matrix
            threshold: Correlation threshold

        Returns:
            List of feature groups (each group is a list of feature names)
        """
        if corr_matrix.empty:
            return []
        
        # Create adjacency matrix for highly correlated features
        high_corr_mask = np.abs(corr_matrix) >= threshold
        high_corr_mask_df = pd.DataFrame(high_corr_mask, index=corr_matrix.index, columns=corr_matrix.columns)
        np.fill_diagonal(high_corr_mask_df.values, False)  # Remove self-correlations
        
        # Find connected components (groups of correlated features)
        features = list(corr_matrix.index)
        visited: set[str] = set()
        groups = []
        
        for feature in features:
            if feature not in visited:
                group = self._dfs_correlated_group(
                    feature, high_corr_mask_df, features, visited
                )
                if len(group) > 1:  # Only include groups with more than one feature
                    groups.append(group)
        
        return groups

    def _dfs_correlated_group(
        self, 
        feature: str, 
        adjacency_matrix: pd.DataFrame, 
        all_features: List[str], 
        visited: set[str]
    ) -> List[str]:
        """
        Depth-first search to find connected components in correlation graph.

        Args:
            feature: Starting feature
            adjacency_matrix: Boolean matrix indicating high correlations
            all_features: List of all feature names
            visited: Set of already visited features

        Returns:
            List of features in the same correlated group
        """
        visited.add(feature)
        group = [feature]
        
        feature_idx = all_features.index(feature)
        
        # Find all features highly correlated with current feature
        for i, other_feature in enumerate(all_features):
            if other_feature not in visited and adjacency_matrix.iloc[feature_idx, i]:
                group.extend(
                    self._dfs_correlated_group(
                        other_feature, adjacency_matrix, all_features, visited
                    )
                )
        
        return group