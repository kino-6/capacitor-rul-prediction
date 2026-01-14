"""Preprocessing recommendation module for model building preparation."""

from typing import Dict, List, Optional, Any, Tuple
import warnings

import pandas as pd
import numpy as np
from scipy import stats

from ..models import (
    MissingValueReport,
    ScalingRecommendation,
    FeatureSuggestion,
    DataSplitStrategy,
    AnalysisResults
)
from ..exceptions import AnalysisError


class PreprocessingRecommender:
    """Recommender for data preprocessing strategies for model building."""

    def __init__(self):
        """Initialize the preprocessing recommender."""
        pass

    def recommend_missing_value_strategy(
        self, missing_report: MissingValueReport
    ) -> Dict[str, str]:
        """
        Recommend missing value handling strategies for each feature.
        
        Args:
            missing_report: Report containing missing value information
            
        Returns:
            Dictionary mapping feature names to recommended strategies
            
        Raises:
            AnalysisError: If missing report is invalid
        """
        if not isinstance(missing_report, MissingValueReport):
            raise AnalysisError("Invalid missing value report provided")
            
        recommendations = {}
        
        for feature, missing_count in missing_report.missing_counts.items():
            missing_percentage = missing_report.missing_percentages.get(feature, 0.0)
            
            if missing_percentage == 0.0:
                recommendations[feature] = "no_action"
            elif missing_percentage < 5.0:
                # Low missing rate: use simple imputation
                recommendations[feature] = "mean_imputation"
            elif missing_percentage < 20.0:
                # Moderate missing rate: use more sophisticated imputation
                recommendations[feature] = "median_imputation"
            elif missing_percentage < 50.0:
                # High missing rate: consider advanced imputation or feature engineering
                recommendations[feature] = "forward_fill"
            else:
                # Very high missing rate: consider dropping the feature
                recommendations[feature] = "drop_feature"
                
        return recommendations

    def recommend_scaling(self, df: pd.DataFrame) -> ScalingRecommendation:
        """
        Recommend scaling method based on feature distributions and ranges.
        
        Args:
            df: Input DataFrame
            
        Returns:
            ScalingRecommendation object with method and rationale
            
        Raises:
            AnalysisError: If DataFrame is empty or has no numeric features
        """
        if df.empty:
            raise AnalysisError("Cannot recommend scaling for empty DataFrame")
            
        # Get numeric features only
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_features:
            return ScalingRecommendation(
                method="none",
                features=[],
                reason="No numeric features found"
            )
            
        # Analyze feature ranges and distributions
        feature_ranges = {}
        feature_skewness = {}
        
        for feature in numeric_features:
            values = df[feature].dropna()
            if len(values) > 0:
                feature_ranges[feature] = values.max() - values.min()
                if len(values) > 2:
                    feature_skewness[feature] = abs(stats.skew(values))
                else:
                    feature_skewness[feature] = 0.0
                    
        if not feature_ranges:
            return ScalingRecommendation(
                method="none",
                features=[],
                reason="No valid numeric data found"
            )
            
        # Calculate scale differences
        max_range = max(feature_ranges.values())
        min_range = min(feature_ranges.values())
        scale_ratio = max_range / min_range if min_range > 0 else float('inf')
        
        # Calculate average skewness
        avg_skewness = np.mean(list(feature_skewness.values()))
        
        # Determine scaling method
        if scale_ratio > 100:
            if avg_skewness > 2.0:
                # High scale difference and high skewness
                method = "robust_scaling"
                reason = f"Large scale differences (ratio: {scale_ratio:.1f}) and high skewness (avg: {avg_skewness:.2f})"
            else:
                # High scale difference but low skewness
                method = "standard_scaling"
                reason = f"Large scale differences (ratio: {scale_ratio:.1f}) between features"
        elif scale_ratio > 10:
            if avg_skewness > 1.5:
                # Moderate scale difference and moderate skewness
                method = "min_max_scaling"
                reason = f"Moderate scale differences (ratio: {scale_ratio:.1f}) and some skewness (avg: {avg_skewness:.2f})"
            else:
                # Moderate scale difference but low skewness
                method = "standard_scaling"
                reason = f"Moderate scale differences (ratio: {scale_ratio:.1f}) between features"
        else:
            # Small scale differences
            method = "none"
            reason = f"Small scale differences (ratio: {scale_ratio:.1f}) - scaling not necessary"
            
        return ScalingRecommendation(
            method=method,
            features=numeric_features,
            reason=reason
        )

    def suggest_feature_engineering(
        self, df: pd.DataFrame, analysis_results: AnalysisResults
    ) -> List[FeatureSuggestion]:
        """
        Suggest feature engineering operations based on analysis results.
        
        Args:
            df: Input DataFrame
            analysis_results: Results from comprehensive analysis
            
        Returns:
            List of feature engineering suggestions
            
        Raises:
            AnalysisError: If inputs are invalid
        """
        if df.empty:
            raise AnalysisError("Cannot suggest feature engineering for empty DataFrame")
            
        if not isinstance(analysis_results, AnalysisResults):
            raise AnalysisError("Invalid analysis results provided")
            
        suggestions = []
        
        # Get numeric features
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # 1. Suggest polynomial features for highly correlated pairs
        if hasattr(analysis_results, 'correlation_matrix') and not analysis_results.correlation_matrix.empty:
            corr_matrix = analysis_results.correlation_matrix
            
            for i, feature1 in enumerate(corr_matrix.columns):
                for j, feature2 in enumerate(corr_matrix.columns):
                    if i < j:  # Avoid duplicates
                        correlation = abs(corr_matrix.loc[feature1, feature2])
                        if 0.3 < correlation < 0.8:  # Moderate correlation
                            suggestions.append(FeatureSuggestion(
                                feature_name=f"{feature1}_{feature2}_interaction",
                                operation="multiply",
                                source_features=[feature1, feature2],
                                rationale=f"Moderate correlation ({correlation:.3f}) suggests potential interaction"
                            ))
                            
        # 2. Suggest ratio features for related measurements
        for i, feature1 in enumerate(numeric_features):
            for j, feature2 in enumerate(numeric_features):
                if i < j and feature2 in df.columns:
                    # Check if feature2 has non-zero values for ratio calculation
                    non_zero_mask = df[feature2] != 0
                    if non_zero_mask.sum() > len(df) * 0.8:  # At least 80% non-zero
                        suggestions.append(FeatureSuggestion(
                            feature_name=f"{feature1}_{feature2}_ratio",
                            operation="divide",
                            source_features=[feature1, feature2],
                            rationale=f"Ratio may capture relative relationship between {feature1} and {feature2}"
                        ))
                        
        # 3. Suggest log transformation for highly skewed features
        for feature in numeric_features:
            if feature in df.columns:
                values = df[feature].dropna()
                if len(values) > 2 and values.min() > 0:  # Positive values only
                    skewness = abs(stats.skew(values))
                    if skewness > 2.0:
                        suggestions.append(FeatureSuggestion(
                            feature_name=f"{feature}_log",
                            operation="log_transform",
                            source_features=[feature],
                            rationale=f"High skewness ({skewness:.2f}) suggests log transformation may help"
                        ))
                        
        # 4. Suggest squared features for potential non-linear relationships
        for feature in numeric_features[:3]:  # Limit to first 3 features to avoid explosion
            if feature in df.columns:
                suggestions.append(FeatureSuggestion(
                    feature_name=f"{feature}_squared",
                    operation="square",
                    source_features=[feature],
                    rationale=f"Squared term may capture non-linear patterns in {feature}"
                ))
                
        # Limit total suggestions to avoid overwhelming the user
        return suggestions[:10]

    def recommend_data_split(
        self, df: pd.DataFrame, is_time_series: bool
    ) -> DataSplitStrategy:
        """
        Recommend data splitting strategy based on data characteristics.
        
        Args:
            df: Input DataFrame
            is_time_series: Whether the data has temporal ordering
            
        Returns:
            DataSplitStrategy object with recommended approach
            
        Raises:
            AnalysisError: If DataFrame is empty
        """
        if df.empty:
            raise AnalysisError("Cannot recommend data split for empty DataFrame")
            
        n_samples = len(df)
        
        if is_time_series:
            # Time series data requires temporal ordering preservation
            if n_samples < 100:
                # Small dataset: simple train/test split
                return DataSplitStrategy(
                    method="temporal_split",
                    train_ratio=0.8,
                    validation_ratio=None,
                    test_ratio=0.2,
                    preserve_temporal_order=True,
                    rationale=f"Small time series dataset ({n_samples} samples) - simple temporal split"
                )
            else:
                # Larger dataset: train/validation/test split
                return DataSplitStrategy(
                    method="temporal_split_with_validation",
                    train_ratio=0.6,
                    validation_ratio=0.2,
                    test_ratio=0.2,
                    preserve_temporal_order=True,
                    rationale=f"Time series dataset ({n_samples} samples) with validation set for model tuning"
                )
        else:
            # Non-time series data: can use random splitting
            if n_samples < 100:
                # Small dataset: simple train/test split
                return DataSplitStrategy(
                    method="random_split",
                    train_ratio=0.8,
                    validation_ratio=None,
                    test_ratio=0.2,
                    preserve_temporal_order=False,
                    rationale=f"Small dataset ({n_samples} samples) - simple random split"
                )
            elif n_samples < 1000:
                # Medium dataset: stratified split if possible
                return DataSplitStrategy(
                    method="stratified_split",
                    train_ratio=0.7,
                    validation_ratio=0.15,
                    test_ratio=0.15,
                    preserve_temporal_order=False,
                    rationale=f"Medium dataset ({n_samples} samples) - stratified split to maintain class balance"
                )
            else:
                # Large dataset: standard train/validation/test split
                return DataSplitStrategy(
                    method="random_split_with_validation",
                    train_ratio=0.6,
                    validation_ratio=0.2,
                    test_ratio=0.2,
                    preserve_temporal_order=False,
                    rationale=f"Large dataset ({n_samples} samples) - standard three-way split"
                )

    def generate_preprocessing_pipeline(
        self,
        df: pd.DataFrame,
        missing_report: MissingValueReport,
        analysis_results: AnalysisResults,
        is_time_series: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Generate a complete preprocessing pipeline as an ordered list of steps.
        
        Args:
            df: Input DataFrame
            missing_report: Missing value analysis report
            analysis_results: Comprehensive analysis results
            is_time_series: Whether data has temporal ordering
            
        Returns:
            Ordered list of preprocessing steps
            
        Raises:
            AnalysisError: If inputs are invalid
        """
        if df.empty:
            raise AnalysisError("Cannot generate pipeline for empty DataFrame")
            
        pipeline = []
        
        # Step 1: Handle missing values
        missing_strategies = self.recommend_missing_value_strategy(missing_report)
        if any(strategy != "no_action" for strategy in missing_strategies.values()):
            pipeline.append({
                "step": "missing_value_handling",
                "order": 1,
                "strategies": missing_strategies,
                "description": "Handle missing values based on missing rate analysis"
            })
            
        # Step 2: Feature engineering (before scaling)
        feature_suggestions = self.suggest_feature_engineering(df, analysis_results)
        if feature_suggestions:
            pipeline.append({
                "step": "feature_engineering",
                "order": 2,
                "suggestions": feature_suggestions[:5],  # Limit to top 5
                "description": "Create derived features to capture relationships and patterns"
            })
            
        # Step 3: Scaling
        scaling_rec = self.recommend_scaling(df)
        if scaling_rec.method != "none":
            pipeline.append({
                "step": "feature_scaling",
                "order": 3,
                "method": scaling_rec.method,
                "features": scaling_rec.features,
                "reason": scaling_rec.reason,
                "description": f"Apply {scaling_rec.method} to normalize feature scales"
            })
            
        # Step 4: Data splitting
        split_strategy = self.recommend_data_split(df, is_time_series)
        pipeline.append({
            "step": "data_splitting",
            "order": 4,
            "strategy": split_strategy,
            "description": f"Split data using {split_strategy.method}"
        })
        
        # Sort by order to ensure correct sequence
        pipeline.sort(key=lambda x: x["order"])
        
        return pipeline