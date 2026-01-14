"""Property-based tests for TimeSeriesAnalyzer."""

import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis.extra.pandas import data_frames, columns

from nasa_pcoe_eda.analysis.timeseries import TimeSeriesAnalyzer


class TestTimeSeriesAnalyzerProperties:
    """Property-based tests for TimeSeriesAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = TimeSeriesAnalyzer()

    # Feature: nasa-pcoe-eda, Property 13: 時間特徴量識別の一貫性
    @given(st.data())
    @settings(max_examples=100)
    def test_temporal_feature_identification_consistency(self, data):
        """
        任意のデータセットに対して、時間ベース特徴量として識別される列は、
        datetime型または時間的な命名規則を持つ
        
        Property 13: Time Feature Identification Consistency
        For any dataset, columns identified as temporal features should have
        either datetime data type or temporal naming patterns.
        """
        # Generate column names - mix of temporal and non-temporal
        temporal_keywords = [
            'time', 'date', 'timestamp', 'cycle', 'step', 'epoch',
            'hour', 'day', 'month', 'year', 'second', 'minute',
            'period', 'sequence', 'index'
        ]
        
        non_temporal_keywords = [
            'temperature', 'pressure', 'voltage', 'current', 'value',
            'measurement', 'sensor', 'data', 'feature', 'signal'
        ]
        
        # Generate a mix of column types
        num_columns = data.draw(st.integers(min_value=1, max_value=10))
        column_specs = []
        
        for i in range(num_columns):
            col_type = data.draw(st.sampled_from(['datetime', 'temporal_numeric', 'non_temporal']))
            
            if col_type == 'datetime':
                # Create datetime column
                col_name = f"datetime_col_{i}"
                column_specs.append((col_name, 'datetime'))
            elif col_type == 'temporal_numeric':
                # Create numeric column with temporal naming
                temporal_keyword = data.draw(st.sampled_from(temporal_keywords))
                col_name = f"{temporal_keyword}_{i}"
                column_specs.append((col_name, 'numeric_temporal'))
            else:
                # Create non-temporal column
                non_temporal_keyword = data.draw(st.sampled_from(non_temporal_keywords))
                col_name = f"{non_temporal_keyword}_{i}"
                column_specs.append((col_name, 'numeric_non_temporal'))
        
        # Generate DataFrame with specified column types
        num_rows = data.draw(st.integers(min_value=2, max_value=20))
        df_data = {}
        
        for col_name, col_type in column_specs:
            if col_type == 'datetime':
                # Generate datetime data
                start_date = pd.Timestamp('2020-01-01')
                dates = pd.date_range(start_date, periods=num_rows, freq='D')
                df_data[col_name] = dates
            elif col_type == 'numeric_temporal':
                # Generate monotonic numeric data (to increase chance of being identified as temporal)
                base_values = np.arange(num_rows)
                # Add some small random noise but keep mostly monotonic
                noise = data.draw(st.lists(
                    st.floats(min_value=-0.1, max_value=0.1, allow_nan=False),
                    min_size=num_rows, max_size=num_rows
                ))
                df_data[col_name] = base_values + np.array(noise)
            else:
                # Generate random numeric data (non-temporal)
                random_values = data.draw(st.lists(
                    st.floats(min_value=-100, max_value=100, allow_nan=False),
                    min_size=num_rows, max_size=num_rows
                ))
                df_data[col_name] = random_values
        
        df = pd.DataFrame(df_data)
        
        # Test the property
        identified_temporal_features = self.analyzer.identify_temporal_features(df)
        
        # Verify the property: each identified temporal feature should either be:
        # 1. A datetime column, OR
        # 2. Have temporal naming patterns
        for feature in identified_temporal_features:
            # Check if it's datetime type
            is_datetime = pd.api.types.is_datetime64_any_dtype(df[feature])
            
            # Check if it has temporal naming patterns
            feature_lower = feature.lower()
            has_temporal_naming = any(keyword in feature_lower for keyword in temporal_keywords)
            
            # The property: identified temporal features must satisfy at least one condition
            assert is_datetime or has_temporal_naming, (
                f"Column '{feature}' was identified as temporal but has neither "
                f"datetime type nor temporal naming patterns. "
                f"Column type: {df[feature].dtype}, "
                f"Column name: {feature}"
            )
        
        # Additional consistency check: all datetime columns should be identified as temporal
        for col_name in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col_name]):
                assert col_name in identified_temporal_features, (
                    f"Datetime column '{col_name}' was not identified as temporal feature"
                )