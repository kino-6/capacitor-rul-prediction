"""Property-based tests for PreprocessingRecommender."""

import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings

from nasa_pcoe_eda.preprocessing.recommender import PreprocessingRecommender


class TestPreprocessingRecommenderProperties:
    """Property-based tests for PreprocessingRecommender."""

    def setup_method(self):
        """Set up test fixtures."""
        self.recommender = PreprocessingRecommender()

    # Feature: nasa-pcoe-eda, Property 24: スケーリング推奨の論理性
    @given(st.data())
    @settings(max_examples=100)
    def test_scaling_recommendation_logic(self, data):
        """
        任意のデータセットに対して、特徴量間のスケールの差が大きい（例：最大値/最小値の比が100以上）場合、
        システムは正規化を推奨する
        
        Property 24: Scaling Recommendation Logic
        For any dataset, if the scale difference between features is large 
        (e.g., max/min ratio > 100), the system should recommend normalization.
        """
        # Generate a DataFrame with numeric columns that have different scales
        num_columns = data.draw(st.integers(min_value=2, max_value=8))
        num_rows = data.draw(st.integers(min_value=10, max_value=100))
        
        # Generate DataFrame data with controlled scale differences
        df_data = {}
        feature_ranges = []
        
        for i in range(num_columns):
            col_name = f"feature_{i}"
            
            # Generate base values
            base_values = data.draw(st.lists(
                st.floats(
                    min_value=0.1, 
                    max_value=10.0, 
                    allow_nan=False, 
                    allow_infinity=False
                ),
                min_size=num_rows,
                max_size=num_rows
            ))
            
            # Apply different scales to create scale differences
            if i == 0:
                # First feature: small scale (0.1 to 10)
                scale_factor = 1.0
            elif i == 1:
                # Second feature: large scale to ensure ratio > 100
                scale_factor = data.draw(st.floats(min_value=1000.0, max_value=10000.0))
            else:
                # Other features: random scales
                scale_factor = data.draw(st.floats(min_value=1.0, max_value=1000.0))
            
            scaled_values = [val * scale_factor for val in base_values]
            df_data[col_name] = scaled_values
            
            # Calculate range for this feature
            feature_range = max(scaled_values) - min(scaled_values)
            feature_ranges.append(feature_range)
        
        df = pd.DataFrame(df_data)
        
        # Skip if DataFrame is empty or has no numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty or len(numeric_df.columns) < 2:
            return
        
        # Remove columns with zero variance (constant columns)
        valid_features = []
        valid_ranges = []
        for col, feature_range in zip(numeric_df.columns, feature_ranges):
            if numeric_df[col].nunique() > 1 and feature_range > 0:
                valid_features.append(col)
                valid_ranges.append(feature_range)
        
        # Skip if we don't have at least 2 valid columns
        if len(valid_features) < 2:
            return
        
        # Filter DataFrame to only valid features
        filtered_df = numeric_df[valid_features]
        
        # Calculate actual scale ratio
        max_range = max(valid_ranges)
        min_range = min(valid_ranges)
        scale_ratio = max_range / min_range if min_range > 0 else float('inf')
        
        # Get scaling recommendation
        scaling_rec = self.recommender.recommend_scaling(filtered_df)
        
        # Verify the scaling recommendation logic
        if scale_ratio > 100:
            # Large scale difference: should recommend some form of scaling
            assert scaling_rec.method != "none", (
                f"Large scale difference detected (ratio: {scale_ratio:.2f}) "
                f"but no scaling was recommended. Method: '{scaling_rec.method}'. "
                f"When scale ratio > 100, some form of scaling should be recommended. "
                f"This violates the scaling recommendation logic property."
            )
            
            # Should recommend one of the valid scaling methods
            valid_scaling_methods = [
                "standard_scaling", 
                "min_max_scaling", 
                "robust_scaling"
            ]
            assert scaling_rec.method in valid_scaling_methods, (
                f"Large scale difference detected (ratio: {scale_ratio:.2f}) "
                f"but invalid scaling method '{scaling_rec.method}' was recommended. "
                f"Valid methods are: {valid_scaling_methods}. "
                f"This violates the scaling recommendation logic property."
            )
            
            # The reason should mention scale differences
            assert "scale" in scaling_rec.reason.lower(), (
                f"Scaling was recommended (method: '{scaling_rec.method}') "
                f"but the reason '{scaling_rec.reason}' does not mention scale differences. "
                f"The rationale should explain why scaling is needed. "
                f"This violates the scaling recommendation logic property."
            )
            
            # Features list should contain the numeric features
            assert len(scaling_rec.features) > 0, (
                f"Scaling method '{scaling_rec.method}' was recommended "
                f"but no features were specified for scaling. "
                f"The recommendation should specify which features to scale. "
                f"This violates the scaling recommendation logic property."
            )
            
            # All recommended features should be numeric
            for feature in scaling_rec.features:
                assert feature in filtered_df.columns, (
                    f"Feature '{feature}' was recommended for scaling "
                    f"but is not present in the input DataFrame columns: {list(filtered_df.columns)}. "
                    f"Only existing features should be recommended for scaling. "
                    f"This violates the scaling recommendation logic property."
                )
                
                assert pd.api.types.is_numeric_dtype(filtered_df[feature]), (
                    f"Feature '{feature}' was recommended for scaling "
                    f"but is not numeric (dtype: {filtered_df[feature].dtype}). "
                    f"Only numeric features should be recommended for scaling. "
                    f"This violates the scaling recommendation logic property."
                )
        
        elif scale_ratio <= 10:
            # Small scale difference: scaling may not be necessary
            # This is not a strict requirement, but if no scaling is recommended,
            # the reason should reflect the small scale difference
            if scaling_rec.method == "none":
                assert "scale" in scaling_rec.reason.lower(), (
                    f"No scaling was recommended for small scale difference (ratio: {scale_ratio:.2f}) "
                    f"but the reason '{scaling_rec.reason}' does not mention scale differences. "
                    f"The rationale should explain why scaling is not needed. "
                    f"This violates the scaling recommendation logic property."
                )
        
        # Additional consistency checks regardless of scale ratio
        
        # Method and reason should be consistent
        if scaling_rec.method == "none":
            assert len(scaling_rec.features) == 0 or all(
                pd.api.types.is_numeric_dtype(filtered_df[f]) for f in scaling_rec.features
            ), (
                f"No scaling method recommended but features list is not empty or contains non-numeric features. "
                f"Method: '{scaling_rec.method}', Features: {scaling_rec.features}. "
                f"This is inconsistent behavior."
            )
        else:
            assert len(scaling_rec.features) > 0, (
                f"Scaling method '{scaling_rec.method}' was recommended "
                f"but no features were specified. "
                f"A scaling method should always specify which features to scale."
            )
        
        # Reason should not be empty
        assert len(scaling_rec.reason.strip()) > 0, (
            f"Scaling recommendation has empty or whitespace-only reason. "
            f"Method: '{scaling_rec.method}', Reason: '{scaling_rec.reason}'. "
            f"All recommendations should provide a clear rationale."
        )
        
        # Method should be a valid string
        assert isinstance(scaling_rec.method, str), (
            f"Scaling method should be a string, got {type(scaling_rec.method)}: {scaling_rec.method}"
        )
        
        # Features should be a list
        assert isinstance(scaling_rec.features, list), (
            f"Features should be a list, got {type(scaling_rec.features)}: {scaling_rec.features}"
        )
        
        # Reason should be a string
        assert isinstance(scaling_rec.reason, str), (
            f"Reason should be a string, got {type(scaling_rec.reason)}: {scaling_rec.reason}"
        )

    # Feature: nasa-pcoe-eda, Property 23: 前処理推奨の完全性
    @given(st.data())
    @settings(max_examples=100)
    def test_preprocessing_recommendation_completeness(self, data):
        """
        任意のデータセットに対して、前処理パイプラインの推奨には、欠損値処理、スケーリング、
        特徴量エンジニアリング、データ分割の全てのステップが含まれる
        
        Property 23: Preprocessing Recommendation Completeness
        For any dataset, the preprocessing pipeline recommendation should include all steps:
        missing value handling, scaling, feature engineering, and data splitting.
        """
        # Generate a DataFrame with various characteristics
        num_columns = data.draw(st.integers(min_value=2, max_value=6))
        num_rows = data.draw(st.integers(min_value=20, max_value=100))
        
        # Generate DataFrame with mixed data types and some missing values
        df_data = {}
        missing_counts = {}
        missing_percentages = {}
        
        for i in range(num_columns):
            col_name = f"feature_{i}"
            
            # Generate base numeric data
            base_values = data.draw(st.lists(
                st.floats(
                    min_value=-1000.0, 
                    max_value=1000.0, 
                    allow_nan=False, 
                    allow_infinity=False
                ),
                min_size=num_rows,
                max_size=num_rows
            ))
            
            # Introduce some missing values randomly
            missing_rate = data.draw(st.floats(min_value=0.0, max_value=0.3))
            num_missing = int(num_rows * missing_rate)
            
            if num_missing > 0:
                # Randomly select indices to make missing
                missing_indices = data.draw(st.lists(
                    st.integers(min_value=0, max_value=num_rows-1),
                    min_size=num_missing,
                    max_size=num_missing,
                    unique=True
                ))
                
                for idx in missing_indices:
                    base_values[idx] = np.nan
            
            df_data[col_name] = base_values
            missing_counts[col_name] = num_missing
            missing_percentages[col_name] = (num_missing / num_rows) * 100.0
        
        df = pd.DataFrame(df_data)
        
        # Skip if DataFrame is empty
        if df.empty:
            return
        
        # Create missing value report
        from nasa_pcoe_eda.models import MissingValueReport
        missing_report = MissingValueReport(
            missing_counts=missing_counts,
            missing_percentages=missing_percentages,
            total_missing=sum(missing_counts.values())
        )
        
        # Create minimal analysis results for feature engineering
        from nasa_pcoe_eda.models import (
            AnalysisResults, DatasetMetadata, Stats, OutlierSummary
        )
        
        # Create correlation matrix
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            correlation_matrix = numeric_df.corr().fillna(0.0)
        else:
            correlation_matrix = pd.DataFrame()
        
        # Create minimal analysis results
        analysis_results = AnalysisResults(
            metadata=DatasetMetadata(
                n_records=len(df),
                n_features=len(df.columns),
                feature_names=list(df.columns),
                data_types={col: str(df[col].dtype) for col in df.columns},
                memory_usage=df.memory_usage(deep=True).sum(),
                date_range=None
            ),
            statistics={},
            missing_values=missing_report,
            correlation_matrix=correlation_matrix,
            outliers=OutlierSummary(
                outlier_counts={},
                outlier_percentages={},
                outlier_indices={}
            ),
            time_series_trends=None,
            rul_features=[],
            fault_features=[],
            preprocessing_recommendations={},
            visualization_paths=[]
        )
        
        # Test both time series and non-time series scenarios
        for is_time_series in [True, False]:
            # Generate preprocessing pipeline
            pipeline = self.recommender.generate_preprocessing_pipeline(
                df=df,
                missing_report=missing_report,
                analysis_results=analysis_results,
                is_time_series=is_time_series
            )
            
            # Verify pipeline completeness - all required steps should be considered
            pipeline_steps = {step["step"] for step in pipeline}
            
            # Required step categories that should always be evaluated
            required_step_categories = {
                "missing_value_handling",
                "feature_scaling", 
                "feature_engineering",
                "data_splitting"
            }
            
            # Data splitting should ALWAYS be present
            assert "data_splitting" in pipeline_steps, (
                f"Data splitting step is missing from preprocessing pipeline. "
                f"Pipeline steps: {pipeline_steps}. "
                f"Data splitting should always be included as it's essential for model validation. "
                f"This violates the preprocessing recommendation completeness property."
            )
            
            # Verify data splitting step has proper structure
            data_split_steps = [step for step in pipeline if step["step"] == "data_splitting"]
            assert len(data_split_steps) == 1, (
                f"Expected exactly one data splitting step, found {len(data_split_steps)}. "
                f"This violates the preprocessing recommendation completeness property."
            )
            
            data_split_step = data_split_steps[0]
            assert "strategy" in data_split_step, (
                f"Data splitting step missing 'strategy' field. "
                f"Step content: {data_split_step}. "
                f"This violates the preprocessing recommendation completeness property."
            )
            
            # Verify the strategy has required fields
            strategy = data_split_step["strategy"]
            required_strategy_fields = ["method", "train_ratio", "test_ratio", "preserve_temporal_order", "rationale"]
            for field in required_strategy_fields:
                assert hasattr(strategy, field), (
                    f"Data splitting strategy missing required field '{field}'. "
                    f"Available fields: {[attr for attr in dir(strategy) if not attr.startswith('_')]}. "
                    f"This violates the preprocessing recommendation completeness property."
                )
            
            # Check if missing value handling is included when there are missing values
            total_missing = sum(missing_counts.values())
            if total_missing > 0:
                assert "missing_value_handling" in pipeline_steps, (
                    f"Missing value handling step is absent despite {total_missing} missing values. "
                    f"Pipeline steps: {pipeline_steps}. "
                    f"Missing value handling should be included when missing values are present. "
                    f"This violates the preprocessing recommendation completeness property."
                )
                
                # Verify missing value handling step structure
                missing_steps = [step for step in pipeline if step["step"] == "missing_value_handling"]
                assert len(missing_steps) == 1, (
                    f"Expected exactly one missing value handling step, found {len(missing_steps)}. "
                    f"This violates the preprocessing recommendation completeness property."
                )
                
                missing_step = missing_steps[0]
                assert "strategies" in missing_step, (
                    f"Missing value handling step missing 'strategies' field. "
                    f"Step content: {missing_step}. "
                    f"This violates the preprocessing recommendation completeness property."
                )
            
            # Check scaling recommendation - should be evaluated even if not needed
            # The scaling step might not be included if scaling is not recommended,
            # but we should verify that the decision was made consciously
            scaling_rec = self.recommender.recommend_scaling(df)
            if scaling_rec.method != "none":
                assert "feature_scaling" in pipeline_steps, (
                    f"Feature scaling step is missing despite scaling method '{scaling_rec.method}' being recommended. "
                    f"Pipeline steps: {pipeline_steps}. "
                    f"When scaling is recommended, it should be included in the pipeline. "
                    f"This violates the preprocessing recommendation completeness property."
                )
                
                # Verify scaling step structure
                scaling_steps = [step for step in pipeline if step["step"] == "feature_scaling"]
                assert len(scaling_steps) == 1, (
                    f"Expected exactly one feature scaling step, found {len(scaling_steps)}. "
                    f"This violates the preprocessing recommendation completeness property."
                )
                
                scaling_step = scaling_steps[0]
                required_scaling_fields = ["method", "features", "reason"]
                for field in required_scaling_fields:
                    assert field in scaling_step, (
                        f"Feature scaling step missing required field '{field}'. "
                        f"Step content: {scaling_step}. "
                        f"This violates the preprocessing recommendation completeness property."
                    )
            
            # Check feature engineering - should be evaluated
            # Feature engineering might not be included if no suggestions are made,
            # but the evaluation should happen
            feature_suggestions = self.recommender.suggest_feature_engineering(df, analysis_results)
            if feature_suggestions:
                assert "feature_engineering" in pipeline_steps, (
                    f"Feature engineering step is missing despite {len(feature_suggestions)} suggestions being made. "
                    f"Pipeline steps: {pipeline_steps}. "
                    f"When feature engineering suggestions exist, they should be included in the pipeline. "
                    f"This violates the preprocessing recommendation completeness property."
                )
                
                # Verify feature engineering step structure
                fe_steps = [step for step in pipeline if step["step"] == "feature_engineering"]
                assert len(fe_steps) == 1, (
                    f"Expected exactly one feature engineering step, found {len(fe_steps)}. "
                    f"This violates the preprocessing recommendation completeness property."
                )
                
                fe_step = fe_steps[0]
                assert "suggestions" in fe_step, (
                    f"Feature engineering step missing 'suggestions' field. "
                    f"Step content: {fe_step}. "
                    f"This violates the preprocessing recommendation completeness property."
                )
            
            # Verify pipeline ordering
            orders = [step["order"] for step in pipeline]
            assert orders == sorted(orders), (
                f"Pipeline steps are not properly ordered. "
                f"Orders: {orders}, Expected: {sorted(orders)}. "
                f"Pipeline steps should be in logical order for preprocessing. "
                f"This violates the preprocessing recommendation completeness property."
            )
            
            # Verify each step has required common fields
            for step in pipeline:
                required_common_fields = ["step", "order", "description"]
                for field in required_common_fields:
                    assert field in step, (
                        f"Pipeline step missing required field '{field}'. "
                        f"Step: {step}. "
                        f"All pipeline steps should have consistent structure. "
                        f"This violates the preprocessing recommendation completeness property."
                    )
                
                # Verify field types
                assert isinstance(step["step"], str), (
                    f"Step 'step' field should be string, got {type(step['step'])}: {step['step']}. "
                    f"This violates the preprocessing recommendation completeness property."
                )
                
                assert isinstance(step["order"], int), (
                    f"Step 'order' field should be integer, got {type(step['order'])}: {step['order']}. "
                    f"This violates the preprocessing recommendation completeness property."
                )
                
                assert isinstance(step["description"], str), (
                    f"Step 'description' field should be string, got {type(step['description'])}: {step['description']}. "
                    f"This violates the preprocessing recommendation completeness property."
                )
                
                # Description should not be empty
                assert len(step["description"].strip()) > 0, (
                    f"Step description is empty or whitespace-only. "
                    f"Step: {step['step']}, Description: '{step['description']}'. "
                    f"All steps should have meaningful descriptions. "
                    f"This violates the preprocessing recommendation completeness property."
                )
            
            # Verify pipeline is not empty
            assert len(pipeline) > 0, (
                f"Preprocessing pipeline is empty. "
                f"At minimum, data splitting should always be included. "
                f"This violates the preprocessing recommendation completeness property."
            )
            
            # Verify time series consideration is reflected in data splitting
            data_split_step = next(step for step in pipeline if step["step"] == "data_splitting")
            strategy = data_split_step["strategy"]
            
            if is_time_series:
                assert strategy.preserve_temporal_order == True, (
                    f"Time series data should preserve temporal order in splitting. "
                    f"preserve_temporal_order: {strategy.preserve_temporal_order}. "
                    f"This violates the preprocessing recommendation completeness property."
                )
                
                assert "temporal" in strategy.method.lower(), (
                    f"Time series data should use temporal splitting method. "
                    f"Method: '{strategy.method}'. "
                    f"This violates the preprocessing recommendation completeness property."
                )
            else:
                assert strategy.preserve_temporal_order == False, (
                    f"Non-time series data should not preserve temporal order in splitting. "
                    f"preserve_temporal_order: {strategy.preserve_temporal_order}. "
                    f"This violates the preprocessing recommendation completeness property."
                )
            
            # Verify ratios sum to approximately 1.0
            total_ratio = strategy.train_ratio + strategy.test_ratio
            if strategy.validation_ratio is not None:
                total_ratio += strategy.validation_ratio
            
            assert abs(total_ratio - 1.0) < 0.01, (
                f"Data split ratios should sum to approximately 1.0. "
                f"Train: {strategy.train_ratio}, Validation: {strategy.validation_ratio}, "
                f"Test: {strategy.test_ratio}, Total: {total_ratio}. "
                f"This violates the preprocessing recommendation completeness property."
            )