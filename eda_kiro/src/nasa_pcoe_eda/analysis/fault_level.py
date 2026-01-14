"""Fault level analysis module for identifying features that distinguish between fault states."""

from typing import Dict, List, Tuple, Optional, Any
import warnings

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

from ..models import DistributionComparison
from ..exceptions import AnalysisError


class FaultLevelAnalyzer:
    """Analyzer for identifying features useful for fault level identification."""

    def __init__(self):
        """Initialize the fault level analyzer."""
        pass

    def identify_discriminative_features(
        self, df: pd.DataFrame, fault_column: str
    ) -> List[str]:
        """
        Identify features that can distinguish between different fault states.
        
        Args:
            df: Input DataFrame
            fault_column: Name of the column containing fault labels
            
        Returns:
            List of feature names that are discriminative for fault classification
            
        Raises:
            AnalysisError: If fault column is not found or has insufficient data
        """
        if df.empty:
            return []
            
        if fault_column not in df.columns:
            raise AnalysisError(f"Fault column '{fault_column}' not found in DataFrame")
            
        # Get numeric features (excluding the fault column)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        feature_columns = numeric_columns.drop(fault_column, errors='ignore')
        
        if len(feature_columns) == 0:
            return []
            
        # Remove rows with missing fault labels
        clean_df = df.dropna(subset=[fault_column])
        if len(clean_df) < 2:
            return []
            
        # Get unique fault levels
        fault_levels = clean_df[fault_column].unique()
        if len(fault_levels) < 2:
            return []
            
        discriminative_features = []
        
        for feature in feature_columns:
            if self._is_discriminative_feature(clean_df, feature, fault_column):
                discriminative_features.append(feature)
                
        return discriminative_features

    def compare_distributions(
        self, df: pd.DataFrame, fault_column: str, features: List[str]
    ) -> DistributionComparison:
        """
        Compare distributions of features between normal and abnormal states.
        
        Args:
            df: Input DataFrame
            fault_column: Name of the column containing fault labels
            features: List of feature names to analyze
            
        Returns:
            DistributionComparison object with statistical comparisons
            
        Raises:
            AnalysisError: If fault column is not found
        """
        if fault_column not in df.columns:
            raise AnalysisError(f"Fault column '{fault_column}' not found in DataFrame")
            
        # Remove rows with missing fault labels
        clean_df = df.dropna(subset=[fault_column])
        if len(clean_df) < 2:
            return DistributionComparison(
                feature_distributions={},
                statistical_tests={}
            )
            
        fault_levels = clean_df[fault_column].unique()
        feature_distributions = {}
        statistical_tests = {}
        
        for feature in features:
            if feature not in clean_df.columns:
                continue
                
            # Get feature values for each fault level
            feature_by_fault = {}
            for fault_level in fault_levels:
                mask = clean_df[fault_column] == fault_level
                values = clean_df.loc[mask, feature].dropna()
                if len(values) > 0:
                    feature_by_fault[str(fault_level)] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'median': float(values.median()),
                        'count': len(values),
                        'values': values.values
                    }
                    
            feature_distributions[feature] = feature_by_fault
            
            # Perform statistical tests
            if len(fault_levels) == 2:
                # Two-sample t-test for binary classification
                groups = [
                    clean_df.loc[clean_df[fault_column] == level, feature].dropna()
                    for level in fault_levels
                ]
                
                if all(len(group) > 1 for group in groups):
                    try:
                        t_stat, p_value = stats.ttest_ind(groups[0], groups[1])
                        statistical_tests[feature] = {
                            'test': 't-test',
                            't_statistic': float(t_stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05
                        }
                    except Exception:
                        statistical_tests[feature] = {
                            'test': 't-test',
                            'error': 'Failed to compute t-test'
                        }
            else:
                # ANOVA for multi-class classification
                groups = [
                    clean_df.loc[clean_df[fault_column] == level, feature].dropna()
                    for level in fault_levels
                ]
                
                if all(len(group) > 1 for group in groups):
                    try:
                        f_stat, p_value = stats.f_oneway(*groups)
                        statistical_tests[feature] = {
                            'test': 'ANOVA',
                            'f_statistic': float(f_stat),
                            'p_value': float(p_value),
                            'significant': p_value < 0.05
                        }
                    except Exception:
                        statistical_tests[feature] = {
                            'test': 'ANOVA',
                            'error': 'Failed to compute ANOVA'
                        }
                        
        return DistributionComparison(
            feature_distributions=feature_distributions,
            statistical_tests=statistical_tests
        )

    def compute_class_separability(
        self, df: pd.DataFrame, fault_column: str, features: List[str]
    ) -> Dict[str, float]:
        """
        Compute class separability metrics for features.
        
        Args:
            df: Input DataFrame
            fault_column: Name of the column containing fault labels
            features: List of feature names to analyze
            
        Returns:
            Dictionary mapping feature names to separability scores
            
        Raises:
            AnalysisError: If fault column is not found
        """
        if fault_column not in df.columns:
            raise AnalysisError(f"Fault column '{fault_column}' not found in DataFrame")
            
        # Remove rows with missing fault labels
        clean_df = df.dropna(subset=[fault_column])
        if len(clean_df) < 2:
            return {}
            
        fault_levels = clean_df[fault_column].unique()
        if len(fault_levels) < 2:
            return {}
            
        separability_scores = {}
        
        for feature in features:
            if feature not in clean_df.columns:
                continue
                
            # Remove rows with missing feature values
            feature_df = clean_df[[feature, fault_column]].dropna()
            if len(feature_df) < 2:
                continue
                
            score = self._compute_fisher_score(
                feature_df[feature].values,
                feature_df[fault_column].values
            )
            
            separability_scores[feature] = score
            
        return separability_scores

    def _is_discriminative_feature(
        self, df: pd.DataFrame, feature: str, fault_column: str
    ) -> bool:
        """
        Check if a feature is discriminative for fault classification.
        
        Args:
            df: Input DataFrame
            feature: Feature name to check
            fault_column: Name of the fault column
            
        Returns:
            True if the feature is discriminative
        """
        # Remove rows with missing values
        clean_data = df[[feature, fault_column]].dropna()
        if len(clean_data) < 4:  # Need at least 4 samples
            return False
            
        fault_levels = clean_data[fault_column].unique()
        if len(fault_levels) < 2:
            return False
            
        # Check if each class has at least 2 samples
        class_counts = clean_data[fault_column].value_counts()
        if (class_counts < 2).any():
            return False
            
        try:
            if len(fault_levels) == 2:
                # Two-sample t-test
                groups = [
                    clean_data.loc[clean_data[fault_column] == level, feature]
                    for level in fault_levels
                ]
                _, p_value = stats.ttest_ind(groups[0], groups[1])
                return p_value < 0.05
            else:
                # ANOVA
                groups = [
                    clean_data.loc[clean_data[fault_column] == level, feature]
                    for level in fault_levels
                ]
                _, p_value = stats.f_oneway(*groups)
                return p_value < 0.05
                
        except Exception:
            return False

    def _compute_fisher_score(
        self, feature_values: np.ndarray, labels: np.ndarray
    ) -> float:
        """
        Compute Fisher score for a feature.
        
        The Fisher score measures the ratio of between-class variance to
        within-class variance. Higher scores indicate better class separability.
        
        Args:
            feature_values: Feature values
            labels: Class labels
            
        Returns:
            Fisher score (non-negative value)
        """
        try:
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                return 0.0
                
            # Compute overall mean
            overall_mean = np.mean(feature_values)
            
            # Compute between-class variance
            between_class_var = 0.0
            total_samples = len(feature_values)
            
            class_means = {}
            class_counts = {}
            
            for label in unique_labels:
                mask = labels == label
                class_values = feature_values[mask]
                if len(class_values) == 0:
                    continue
                    
                class_mean = np.mean(class_values)
                class_count = len(class_values)
                
                class_means[label] = class_mean
                class_counts[label] = class_count
                
                between_class_var += class_count * (class_mean - overall_mean) ** 2
                
            between_class_var /= total_samples
            
            # Compute within-class variance
            within_class_var = 0.0
            
            for label in unique_labels:
                mask = labels == label
                class_values = feature_values[mask]
                if len(class_values) <= 1:
                    continue
                    
                class_mean = class_means[label]
                class_var = np.sum((class_values - class_mean) ** 2)
                within_class_var += class_var
                
            within_class_var /= total_samples
            
            # Compute Fisher score
            if within_class_var == 0:
                # Perfect separation
                return float('inf') if between_class_var > 0 else 0.0
            else:
                fisher_score = between_class_var / within_class_var
                return max(0.0, fisher_score)  # Ensure non-negative
                
        except Exception:
            return 0.0