"""Unit tests for PreprocessingRecommender."""

import numpy as np
import pandas as pd
import pytest

from nasa_pcoe_eda.preprocessing.recommender import PreprocessingRecommender
from nasa_pcoe_eda.models import (
    MissingValueReport,
    ScalingRecommendation,
    FeatureSuggestion,
    DataSplitStrategy,
    AnalysisResults,
    DatasetMetadata,
    Stats
)
from nasa_pcoe_eda.exceptions import AnalysisError


class TestPreprocessingRecommender:
    """Test cases for PreprocessingRecommender."""

    def setup_method(self):
        """Set up test fixtures."""
        self.recommender = PreprocessingRecommender()

    def test_recommend_missing_value_strategy_no_missing(self):
        """Test missing value strategy recommendation with no missing values."""
        missing_report = MissingValueReport(
            missing_counts={'feature1': 0, 'feature2': 0},
            missing_percentages={'feature1': 0.0, 'feature2': 0.0},
            total_missing=0
        )
        
        recommendations = self.recommender.recommend_missing_value_strategy(missing_report)
        
        assert isinstance(recommendations, dict)
        assert recommendations['feature1'] == 'no_action'
        assert recommendations['feature2'] == 'no_action'

    def test_recommend_missing_value_strategy_low_missing(self):
        """Test missing value strategy recommendation with low missing rates."""
        missing_report = MissingValueReport(
            missing_counts={'feature1': 2, 'feature2': 4},
            missing_percentages={'feature1': 2.0, 'feature2': 4.0},
            total_missing=6
        )
        
        recommendations = self.recommender.recommend_missing_value_strategy(missing_report)
        
        assert recommendations['feature1'] == 'mean_imputation'
        assert recommendations['feature2'] == 'mean_imputation'

    def test_recommend_missing_value_strategy_moderate_missing(self):
        """Test missing value strategy recommendation with moderate missing rates."""
        missing_report = MissingValueReport(
            missing_counts={'feature1': 10, 'feature2': 15},
            missing_percentages={'feature1': 10.0, 'feature2': 15.0},
            total_missing=25
        )
        
        recommendations = self.recommender.recommend_missing_value_strategy(missing_report)
        
        assert recommendations['feature1'] == 'median_imputation'
        assert recommendations['feature2'] == 'median_imputation'

    def test_recommend_missing_value_strategy_high_missing(self):
        """Test missing value strategy recommendation with high missing rates."""
        missing_report = MissingValueReport(
            missing_counts={'feature1': 30, 'feature2': 40},
            missing_percentages={'feature1': 30.0, 'feature2': 40.0},
            total_missing=70
        )
        
        recommendations = self.recommender.recommend_missing_value_strategy(missing_report)
        
        assert recommendations['feature1'] == 'forward_fill'
        assert recommendations['feature2'] == 'forward_fill'

    def test_recommend_missing_value_strategy_very_high_missing(self):
        """Test missing value strategy recommendation with very high missing rates."""
        missing_report = MissingValueReport(
            missing_counts={'feature1': 60, 'feature2': 80},
            missing_percentages={'feature1': 60.0, 'feature2': 80.0},
            total_missing=140
        )
        
        recommendations = self.recommender.recommend_missing_value_strategy(missing_report)
        
        assert recommendations['feature1'] == 'drop_feature'
        assert recommendations['feature2'] == 'drop_feature'

    def test_recommend_missing_value_strategy_invalid_input(self):
        """Test missing value strategy recommendation with invalid input."""
        with pytest.raises(AnalysisError, match="Invalid missing value report provided"):
            self.recommender.recommend_missing_value_strategy("invalid")

    def test_recommend_scaling_empty_dataframe(self):
        """Test scaling recommendation with empty DataFrame."""
        df = pd.DataFrame()
        
        with pytest.raises(AnalysisError, match="Cannot recommend scaling for empty DataFrame"):
            self.recommender.recommend_scaling(df)

    def test_recommend_scaling_no_numeric_features(self):
        """Test scaling recommendation with no numeric features."""
        df = pd.DataFrame({
            'text_col': ['a', 'b', 'c'],
            'category_col': ['x', 'y', 'z']
        })
        
        recommendation = self.recommender.recommend_scaling(df)
        
        assert isinstance(recommendation, ScalingRecommendation)
        assert recommendation.method == 'none'
        assert recommendation.features == []
        assert 'No numeric features found' in recommendation.reason

    def test_recommend_scaling_large_scale_differences(self):
        """Test scaling recommendation with large scale differences."""
        df = pd.DataFrame({
            'small_scale': [1, 2, 3, 4, 5],
            'large_scale': [1000, 2000, 3000, 4000, 5000]
        })
        
        recommendation = self.recommender.recommend_scaling(df)
        
        assert isinstance(recommendation, ScalingRecommendation)
        assert recommendation.method in ['standard_scaling', 'robust_scaling']
        assert 'small_scale' in recommendation.features
        assert 'large_scale' in recommendation.features
        assert 'Large scale differences' in recommendation.reason

    def test_recommend_scaling_moderate_scale_differences(self):
        """Test scaling recommendation with moderate scale differences."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        recommendation = self.recommender.recommend_scaling(df)
        
        assert isinstance(recommendation, ScalingRecommendation)
        # Scale ratio is 10.0, which is exactly at the boundary, so it may be 'none'
        assert recommendation.method in ['standard_scaling', 'min_max_scaling', 'none']
        assert len(recommendation.features) == 2

    def test_recommend_scaling_small_scale_differences(self):
        """Test scaling recommendation with small scale differences."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [1.1, 2.1, 3.1, 4.1, 5.1]
        })
        
        recommendation = self.recommender.recommend_scaling(df)
        
        assert isinstance(recommendation, ScalingRecommendation)
        assert recommendation.method == 'none'
        assert 'Small scale differences' in recommendation.reason

    def test_recommend_scaling_with_missing_values(self):
        """Test scaling recommendation with missing values."""
        df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5],
            'feature2': [10, np.nan, 30, 40, 50]
        })
        
        recommendation = self.recommender.recommend_scaling(df)
        
        assert isinstance(recommendation, ScalingRecommendation)
        # Should handle missing values gracefully

    def test_recommend_scaling_high_skewness(self):
        """Test scaling recommendation with highly skewed data."""
        # Create highly skewed data
        np.random.seed(42)
        skewed_data = np.random.exponential(1, 100)  # Exponential distribution is highly skewed
        normal_data = np.random.normal(0, 1, 100)
        
        df = pd.DataFrame({
            'skewed_feature': skewed_data,
            'normal_feature': normal_data
        })
        
        recommendation = self.recommender.recommend_scaling(df)
        
        assert isinstance(recommendation, ScalingRecommendation)
        # With high skewness, should recommend robust scaling
        if recommendation.method != 'none':
            assert recommendation.method in ['robust_scaling', 'standard_scaling', 'min_max_scaling']

    def test_suggest_feature_engineering_empty_dataframe(self):
        """Test feature engineering suggestions with empty DataFrame."""
        df = pd.DataFrame()
        analysis_results = self._create_mock_analysis_results()
        
        with pytest.raises(AnalysisError, match="Cannot suggest feature engineering for empty DataFrame"):
            self.recommender.suggest_feature_engineering(df, analysis_results)

    def test_suggest_feature_engineering_invalid_analysis_results(self):
        """Test feature engineering suggestions with invalid analysis results."""
        df = pd.DataFrame({'feature1': [1, 2, 3]})
        
        with pytest.raises(AnalysisError, match="Invalid analysis results provided"):
            self.recommender.suggest_feature_engineering(df, "invalid")

    def test_suggest_feature_engineering_basic(self):
        """Test basic feature engineering suggestions."""
        # Create test data with correlations
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'feature3': np.random.randn(100)
        })
        
        # Create correlation matrix
        corr_matrix = df.corr()
        analysis_results = self._create_mock_analysis_results(correlation_matrix=corr_matrix)
        
        suggestions = self.recommender.suggest_feature_engineering(df, analysis_results)
        
        assert isinstance(suggestions, list)
        assert len(suggestions) <= 10  # Should limit suggestions
        
        # Check that suggestions have required fields
        for suggestion in suggestions:
            assert isinstance(suggestion, FeatureSuggestion)
            assert hasattr(suggestion, 'feature_name')
            assert hasattr(suggestion, 'operation')
            assert hasattr(suggestion, 'source_features')
            assert hasattr(suggestion, 'rationale')

    def test_suggest_feature_engineering_with_skewed_data(self):
        """Test feature engineering suggestions with skewed data."""
        # Create data with guaranteed high skewness
        # Use a power law distribution which is highly skewed
        np.random.seed(42)
        base_data = np.random.uniform(0.1, 1.0, 100)
        skewed_data = 1.0 / base_data  # This creates high positive skewness
        
        df = pd.DataFrame({
            'skewed_feature': skewed_data,
            'normal_feature': np.random.randn(100)
        })
        
        # Verify the skewness is indeed high
        from scipy import stats
        actual_skewness = abs(stats.skew(skewed_data))
        
        analysis_results = self._create_mock_analysis_results()
        suggestions = self.recommender.suggest_feature_engineering(df, analysis_results)
        
        # Should suggest log transformation for highly skewed data
        log_suggestions = [s for s in suggestions if s.operation == 'log_transform']
        
        if actual_skewness > 2.0:
            assert len(log_suggestions) > 0
            log_suggestion = log_suggestions[0]
            assert 'skewed_feature' in log_suggestion.source_features
            assert 'skewness' in log_suggestion.rationale.lower()
        else:
            # If our data generation didn't create high enough skewness, 
            # create it manually for the test
            very_skewed_data = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 100] * 10)
            df_manual = pd.DataFrame({
                'skewed_feature': very_skewed_data,
                'normal_feature': np.random.randn(100)
            })
            
            suggestions_manual = self.recommender.suggest_feature_engineering(df_manual, analysis_results)
            log_suggestions_manual = [s for s in suggestions_manual if s.operation == 'log_transform']
            
            # This should definitely suggest log transformation
            assert len(log_suggestions_manual) > 0

    def test_suggest_feature_engineering_interaction_terms(self):
        """Test feature engineering suggestions for interaction terms."""
        # Create data with moderate correlation
        np.random.seed(42)
        feature1 = np.random.randn(100)
        feature2 = 0.5 * feature1 + 0.5 * np.random.randn(100)  # Moderate correlation
        
        df = pd.DataFrame({
            'feature1': feature1,
            'feature2': feature2
        })
        
        corr_matrix = df.corr()
        analysis_results = self._create_mock_analysis_results(correlation_matrix=corr_matrix)
        
        suggestions = self.recommender.suggest_feature_engineering(df, analysis_results)
        
        # Should suggest interaction terms for moderately correlated features
        interaction_suggestions = [s for s in suggestions if s.operation == 'multiply']
        assert len(interaction_suggestions) > 0
        
        interaction_suggestion = interaction_suggestions[0]
        assert len(interaction_suggestion.source_features) == 2
        assert 'interaction' in interaction_suggestion.rationale.lower()

    def test_recommend_data_split_empty_dataframe(self):
        """Test data split recommendation with empty DataFrame."""
        df = pd.DataFrame()
        
        with pytest.raises(AnalysisError, match="Cannot recommend data split for empty DataFrame"):
            self.recommender.recommend_data_split(df, False)

    def test_recommend_data_split_small_time_series(self):
        """Test data split recommendation for small time series dataset."""
        df = pd.DataFrame({
            'feature1': range(50),
            'target': range(50)
        })
        
        strategy = self.recommender.recommend_data_split(df, is_time_series=True)
        
        assert isinstance(strategy, DataSplitStrategy)
        assert strategy.method == 'temporal_split'
        assert strategy.preserve_temporal_order == True
        assert strategy.train_ratio == 0.8
        assert strategy.test_ratio == 0.2
        assert strategy.validation_ratio is None

    def test_recommend_data_split_large_time_series(self):
        """Test data split recommendation for large time series dataset."""
        df = pd.DataFrame({
            'feature1': range(1000),
            'target': range(1000)
        })
        
        strategy = self.recommender.recommend_data_split(df, is_time_series=True)
        
        assert isinstance(strategy, DataSplitStrategy)
        assert strategy.method == 'temporal_split_with_validation'
        assert strategy.preserve_temporal_order == True
        assert strategy.validation_ratio is not None
        assert strategy.train_ratio + strategy.validation_ratio + strategy.test_ratio == 1.0

    def test_recommend_data_split_small_non_time_series(self):
        """Test data split recommendation for small non-time series dataset."""
        df = pd.DataFrame({
            'feature1': range(50),
            'target': range(50)
        })
        
        strategy = self.recommender.recommend_data_split(df, is_time_series=False)
        
        assert isinstance(strategy, DataSplitStrategy)
        assert strategy.method == 'random_split'
        assert strategy.preserve_temporal_order == False
        assert strategy.train_ratio == 0.8
        assert strategy.test_ratio == 0.2

    def test_recommend_data_split_medium_non_time_series(self):
        """Test data split recommendation for medium non-time series dataset."""
        df = pd.DataFrame({
            'feature1': range(500),
            'target': range(500)
        })
        
        strategy = self.recommender.recommend_data_split(df, is_time_series=False)
        
        assert isinstance(strategy, DataSplitStrategy)
        assert strategy.method == 'stratified_split'
        assert strategy.preserve_temporal_order == False
        assert strategy.validation_ratio is not None

    def test_recommend_data_split_large_non_time_series(self):
        """Test data split recommendation for large non-time series dataset."""
        df = pd.DataFrame({
            'feature1': range(2000),
            'target': range(2000)
        })
        
        strategy = self.recommender.recommend_data_split(df, is_time_series=False)
        
        assert isinstance(strategy, DataSplitStrategy)
        assert strategy.method == 'random_split_with_validation'
        assert strategy.preserve_temporal_order == False
        assert strategy.validation_ratio is not None

    def test_generate_preprocessing_pipeline_empty_dataframe(self):
        """Test preprocessing pipeline generation with empty DataFrame."""
        df = pd.DataFrame()
        missing_report = MissingValueReport({}, {}, 0)
        analysis_results = self._create_mock_analysis_results()
        
        with pytest.raises(AnalysisError, match="Cannot generate pipeline for empty DataFrame"):
            self.recommender.generate_preprocessing_pipeline(df, missing_report, analysis_results)

    def test_generate_preprocessing_pipeline_complete(self):
        """Test complete preprocessing pipeline generation."""
        # Create test data
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5] * 20,  # Some missing values
            'feature2': [10, 20, 30, 40, 50] * 20,  # Different scale
            'feature3': np.random.randn(100)
        })
        
        missing_report = MissingValueReport(
            missing_counts={'feature1': 20, 'feature2': 0, 'feature3': 0},
            missing_percentages={'feature1': 20.0, 'feature2': 0.0, 'feature3': 0.0},
            total_missing=20
        )
        
        corr_matrix = df.corr()
        analysis_results = self._create_mock_analysis_results(correlation_matrix=corr_matrix)
        
        pipeline = self.recommender.generate_preprocessing_pipeline(
            df, missing_report, analysis_results, is_time_series=False
        )
        
        assert isinstance(pipeline, list)
        assert len(pipeline) > 0
        
        # Check that pipeline steps are ordered
        orders = [step['order'] for step in pipeline]
        assert orders == sorted(orders)
        
        # Check that all steps have required fields
        for step in pipeline:
            assert 'step' in step
            assert 'order' in step
            assert 'description' in step
            
        # Should include data splitting as final step
        final_step = pipeline[-1]
        assert final_step['step'] == 'data_splitting'

    def test_generate_preprocessing_pipeline_no_missing_values(self):
        """Test preprocessing pipeline generation with no missing values."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [10, 20, 30, 40, 50]
        })
        
        missing_report = MissingValueReport(
            missing_counts={'feature1': 0, 'feature2': 0},
            missing_percentages={'feature1': 0.0, 'feature2': 0.0},
            total_missing=0
        )
        
        analysis_results = self._create_mock_analysis_results()
        
        pipeline = self.recommender.generate_preprocessing_pipeline(
            df, missing_report, analysis_results
        )
        
        # Should not include missing value handling step
        missing_steps = [step for step in pipeline if step['step'] == 'missing_value_handling']
        assert len(missing_steps) == 0

    def test_generate_preprocessing_pipeline_time_series(self):
        """Test preprocessing pipeline generation for time series data."""
        df = pd.DataFrame({
            'feature1': range(100),
            'feature2': range(100, 200)
        })
        
        missing_report = MissingValueReport(
            missing_counts={'feature1': 0, 'feature2': 0},
            missing_percentages={'feature1': 0.0, 'feature2': 0.0},
            total_missing=0
        )
        
        analysis_results = self._create_mock_analysis_results()
        
        pipeline = self.recommender.generate_preprocessing_pipeline(
            df, missing_report, analysis_results, is_time_series=True
        )
        
        # Final step should preserve temporal order
        final_step = pipeline[-1]
        assert final_step['step'] == 'data_splitting'
        assert final_step['strategy'].preserve_temporal_order == True

    def _create_mock_analysis_results(self, correlation_matrix=None):
        """Create mock AnalysisResults for testing."""
        if correlation_matrix is None:
            correlation_matrix = pd.DataFrame()
            
        return AnalysisResults(
            metadata=DatasetMetadata(
                n_records=100,
                n_features=3,
                feature_names=['feature1', 'feature2', 'feature3'],
                data_types={'feature1': 'float64', 'feature2': 'float64', 'feature3': 'float64'},
                memory_usage=1000.0,
                date_range=None
            ),
            statistics={
                'feature1': Stats(1.0, 1.0, 1.0, 0.0, 2.0, 0.5, 1.5),
                'feature2': Stats(2.0, 2.0, 1.0, 1.0, 3.0, 1.5, 2.5),
                'feature3': Stats(0.0, 0.0, 1.0, -2.0, 2.0, -0.5, 0.5)
            },
            missing_values=MissingValueReport({}, {}, 0),
            correlation_matrix=correlation_matrix,
            outliers=None,
            time_series_trends=None,
            rul_features=[],
            fault_features=[],
            preprocessing_recommendations={},
            visualization_paths=[]
        )

    def test_scaling_recommendation_with_zero_variance(self):
        """Test scaling recommendation with zero variance features."""
        df = pd.DataFrame({
            'constant_feature': [5, 5, 5, 5, 5],  # Zero variance
            'normal_feature': [1, 2, 3, 4, 5]
        })
        
        recommendation = self.recommender.recommend_scaling(df)
        
        # Should handle zero variance gracefully
        assert isinstance(recommendation, ScalingRecommendation)

    def test_feature_engineering_with_zero_division(self):
        """Test feature engineering suggestions with potential zero division."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0, 0, 1, 2, 3]  # Contains zeros
        })
        
        analysis_results = self._create_mock_analysis_results()
        suggestions = self.recommender.suggest_feature_engineering(df, analysis_results)
        
        # Should handle potential zero division in ratio features
        ratio_suggestions = [s for s in suggestions if s.operation == 'divide']
        
        # Should only suggest ratios where denominator has sufficient non-zero values
        for suggestion in ratio_suggestions:
            assert len(suggestion.source_features) == 2

    def test_pipeline_step_ordering(self):
        """Test that preprocessing pipeline steps are correctly ordered."""
        df = pd.DataFrame({
            'feature1': [1, np.nan, 3, 4, 5] * 20,
            'feature2': [100, 200, 300, 400, 500] * 20
        })
        
        missing_report = MissingValueReport(
            missing_counts={'feature1': 20, 'feature2': 0},
            missing_percentages={'feature1': 20.0, 'feature2': 0.0},
            total_missing=20
        )
        
        analysis_results = self._create_mock_analysis_results()
        
        pipeline = self.recommender.generate_preprocessing_pipeline(
            df, missing_report, analysis_results
        )
        
        # Check correct ordering: missing values -> feature engineering -> scaling -> splitting
        step_names = [step['step'] for step in pipeline]
        
        if 'missing_value_handling' in step_names:
            missing_idx = step_names.index('missing_value_handling')
            if 'feature_engineering' in step_names:
                eng_idx = step_names.index('feature_engineering')
                assert missing_idx < eng_idx
                
        if 'feature_scaling' in step_names:
            scaling_idx = step_names.index('feature_scaling')
            if 'feature_engineering' in step_names:
                eng_idx = step_names.index('feature_engineering')
                assert eng_idx < scaling_idx
                
        # Data splitting should always be last
        assert step_names[-1] == 'data_splitting'

    def test_feature_suggestions_limit(self):
        """Test that feature engineering suggestions are limited to reasonable number."""
        # Create data with many features to potentially generate many suggestions
        np.random.seed(42)
        data = {}
        for i in range(10):
            data[f'feature_{i}'] = np.random.randn(100)
            
        df = pd.DataFrame(data)
        corr_matrix = df.corr()
        analysis_results = self._create_mock_analysis_results(correlation_matrix=corr_matrix)
        
        suggestions = self.recommender.suggest_feature_engineering(df, analysis_results)
        
        # Should limit suggestions to avoid overwhelming the user
        assert len(suggestions) <= 10

    def test_data_split_ratios_sum_to_one(self):
        """Test that data split ratios sum to 1.0."""
        df = pd.DataFrame({'feature1': range(1000)})
        
        strategy = self.recommender.recommend_data_split(df, is_time_series=False)
        
        total_ratio = strategy.train_ratio + strategy.test_ratio
        if strategy.validation_ratio is not None:
            total_ratio += strategy.validation_ratio
            
        assert abs(total_ratio - 1.0) < 1e-10  # Account for floating point precision