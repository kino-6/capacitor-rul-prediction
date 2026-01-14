"""Property-based tests for StatisticsAnalyzer."""

import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings

from nasa_pcoe_eda.analysis.statistics import StatisticsAnalyzer


class TestStatisticsAnalyzerProperties:
    """Property-based tests for StatisticsAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = StatisticsAnalyzer()

    # Feature: nasa-pcoe-eda, Property 6: 統計計算の正確性
    @given(st.data())
    @settings(max_examples=100)
    def test_statistical_calculation_accuracy(self, data):
        """
        任意の数値データセットに対して、計算された平均値、中央値、標準偏差、最小値、最大値は、
        数学的定義に従った正しい値である
        
        Property 6: Statistical Calculation Accuracy
        For any numerical dataset, the calculated mean, median, standard deviation, minimum, 
        and maximum values should be correct values according to mathematical definitions.
        """
        # Generate a DataFrame with numeric columns
        num_columns = data.draw(st.integers(min_value=1, max_value=8))
        num_rows = data.draw(st.integers(min_value=5, max_value=100))
        
        # Generate DataFrame data manually
        df_data = {}
        for i in range(num_columns):
            col_name = f"feature_{i}"
            # Generate numeric data for this column
            column_data = data.draw(st.lists(
                st.floats(
                    min_value=-1000, 
                    max_value=1000, 
                    allow_nan=False, 
                    allow_infinity=False
                ),
                min_size=num_rows,
                max_size=num_rows
            ))
            df_data[col_name] = column_data
        
        df = pd.DataFrame(df_data)
        
        # Skip if DataFrame is empty or has no numeric columns
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return
        
        # Compute descriptive statistics using our analyzer
        stats_dict = self.analyzer.compute_descriptive_stats(df)
        
        # Verify that statistics are computed for all numeric columns
        expected_columns = set(numeric_df.columns)
        actual_columns = set(stats_dict.keys())
        assert expected_columns == actual_columns, (
            f"Statistics computed for columns {actual_columns}, "
            f"but expected columns {expected_columns}. "
            f"Statistics should be computed for all numeric columns."
        )
        
        # Verify accuracy of each statistic for each column
        for column in numeric_df.columns:
            series = numeric_df[column]
            computed_stats = stats_dict[column]
            
            # Skip columns that are all NaN (pandas will handle these gracefully)
            if series.isna().all():
                continue
            
            # Verify mean accuracy
            expected_mean = float(series.mean())
            computed_mean = computed_stats.mean
            if not (pd.isna(expected_mean) and pd.isna(computed_mean)):
                assert abs(computed_mean - expected_mean) < 1e-10, (
                    f"Mean calculation incorrect for column '{column}': "
                    f"computed {computed_mean}, expected {expected_mean}. "
                    f"Difference: {abs(computed_mean - expected_mean)}. "
                    f"This violates the statistical accuracy property."
                )
            
            # Verify median accuracy
            expected_median = float(series.median())
            computed_median = computed_stats.median
            if not (pd.isna(expected_median) and pd.isna(computed_median)):
                assert abs(computed_median - expected_median) < 1e-10, (
                    f"Median calculation incorrect for column '{column}': "
                    f"computed {computed_median}, expected {expected_median}. "
                    f"Difference: {abs(computed_median - expected_median)}. "
                    f"This violates the statistical accuracy property."
                )
            
            # Verify standard deviation accuracy
            expected_std = float(series.std())
            computed_std = computed_stats.std
            if not (pd.isna(expected_std) and pd.isna(computed_std)):
                assert abs(computed_std - expected_std) < 1e-10, (
                    f"Standard deviation calculation incorrect for column '{column}': "
                    f"computed {computed_std}, expected {expected_std}. "
                    f"Difference: {abs(computed_std - expected_std)}. "
                    f"This violates the statistical accuracy property."
                )
            
            # Verify minimum accuracy
            expected_min = float(series.min())
            computed_min = computed_stats.min
            if not (pd.isna(expected_min) and pd.isna(computed_min)):
                assert abs(computed_min - expected_min) < 1e-10, (
                    f"Minimum calculation incorrect for column '{column}': "
                    f"computed {computed_min}, expected {expected_min}. "
                    f"Difference: {abs(computed_min - expected_min)}. "
                    f"This violates the statistical accuracy property."
                )
            
            # Verify maximum accuracy
            expected_max = float(series.max())
            computed_max = computed_stats.max
            if not (pd.isna(expected_max) and pd.isna(computed_max)):
                assert abs(computed_max - expected_max) < 1e-10, (
                    f"Maximum calculation incorrect for column '{column}': "
                    f"computed {computed_max}, expected {expected_max}. "
                    f"Difference: {abs(computed_max - expected_max)}. "
                    f"This violates the statistical accuracy property."
                )
            
            # Verify 25th percentile (Q1) accuracy
            expected_q25 = float(series.quantile(0.25))
            computed_q25 = computed_stats.q25
            if not (pd.isna(expected_q25) and pd.isna(computed_q25)):
                assert abs(computed_q25 - expected_q25) < 1e-10, (
                    f"25th percentile calculation incorrect for column '{column}': "
                    f"computed {computed_q25}, expected {expected_q25}. "
                    f"Difference: {abs(computed_q25 - expected_q25)}. "
                    f"This violates the statistical accuracy property."
                )
            
            # Verify 75th percentile (Q3) accuracy
            expected_q75 = float(series.quantile(0.75))
            computed_q75 = computed_stats.q75
            if not (pd.isna(expected_q75) and pd.isna(computed_q75)):
                assert abs(computed_q75 - expected_q75) < 1e-10, (
                    f"75th percentile calculation incorrect for column '{column}': "
                    f"computed {computed_q75}, expected {expected_q75}. "
                    f"Difference: {abs(computed_q75 - expected_q75)}. "
                    f"This violates the statistical accuracy property."
                )
            
            # Verify logical relationships between statistics
            if not pd.isna(computed_min) and not pd.isna(computed_max):
                assert computed_min <= computed_max, (
                    f"Minimum value ({computed_min}) is greater than maximum value ({computed_max}) "
                    f"for column '{column}'. This violates basic statistical logic."
                )
            
            if not pd.isna(computed_q25) and not pd.isna(computed_q75):
                assert computed_q25 <= computed_q75, (
                    f"25th percentile ({computed_q25}) is greater than 75th percentile ({computed_q75}) "
                    f"for column '{column}'. This violates basic statistical logic."
                )
            
            if not pd.isna(computed_min) and not pd.isna(computed_q25):
                assert computed_min <= computed_q25, (
                    f"Minimum value ({computed_min}) is greater than 25th percentile ({computed_q25}) "
                    f"for column '{column}'. This violates basic statistical logic."
                )
            
            if not pd.isna(computed_q75) and not pd.isna(computed_max):
                assert computed_q75 <= computed_max, (
                    f"75th percentile ({computed_q75}) is greater than maximum value ({computed_max}) "
                    f"for column '{column}'. This violates basic statistical logic."
                )
            
            # Verify standard deviation is non-negative
            if not pd.isna(computed_std):
                assert computed_std >= 0, (
                    f"Standard deviation ({computed_std}) is negative for column '{column}'. "
                    f"Standard deviation must be non-negative."
                )
            
            # For single-value columns, standard deviation should be 0
            if series.nunique() == 1 and not pd.isna(computed_std):
                assert abs(computed_std) < 1e-10, (
                    f"Standard deviation ({computed_std}) is not zero for constant column '{column}'. "
                    f"Standard deviation of constant values should be zero."
                )

    # Feature: nasa-pcoe-eda, Property 7: 欠損値カウントの正確性
    @given(st.data())
    @settings(max_examples=100)
    def test_missing_value_count_accuracy(self, data):
        """
        任意のデータセットに対して、各特徴量について報告される欠損値数は、実際のNaN/null値の数と一致する
        
        Property 7: Missing Value Count Accuracy
        For any dataset, the number of missing values reported for each feature 
        should match the actual number of NaN/null values.
        """
        # Generate a DataFrame with various data types and missing values
        num_columns = data.draw(st.integers(min_value=1, max_value=8))
        num_rows = data.draw(st.integers(min_value=5, max_value=100))
        
        # Generate DataFrame data with intentional missing values
        df_data = {}
        expected_missing_counts = {}
        
        for i in range(num_columns):
            col_name = f"feature_{i}"
            
            # Choose column type randomly
            col_type = data.draw(st.sampled_from(['numeric', 'string', 'boolean']))
            
            if col_type == 'numeric':
                # Generate numeric data with some NaN values
                base_data = data.draw(st.lists(
                    st.floats(
                        min_value=-1000, 
                        max_value=1000, 
                        allow_nan=False, 
                        allow_infinity=False
                    ),
                    min_size=num_rows,
                    max_size=num_rows
                ))
                
                # Randomly introduce NaN values
                missing_indices = data.draw(st.sets(
                    st.integers(min_value=0, max_value=num_rows-1),
                    max_size=min(num_rows // 2, 10)  # At most half the rows, max 10
                ))
                
                column_data = base_data.copy()
                for idx in missing_indices:
                    column_data[idx] = np.nan
                    
                expected_missing_counts[col_name] = len(missing_indices)
                
            elif col_type == 'string':
                # Generate string data with some None values
                base_data = data.draw(st.lists(
                    st.text(min_size=1, max_size=10),
                    min_size=num_rows,
                    max_size=num_rows
                ))
                
                # Randomly introduce None values
                missing_indices = data.draw(st.sets(
                    st.integers(min_value=0, max_value=num_rows-1),
                    max_size=min(num_rows // 2, 10)  # At most half the rows, max 10
                ))
                
                column_data = base_data.copy()
                for idx in missing_indices:
                    column_data[idx] = None
                    
                expected_missing_counts[col_name] = len(missing_indices)
                
            else:  # boolean
                # Generate boolean data with some None values
                base_data = data.draw(st.lists(
                    st.booleans(),
                    min_size=num_rows,
                    max_size=num_rows
                ))
                
                # Randomly introduce None values
                missing_indices = data.draw(st.sets(
                    st.integers(min_value=0, max_value=num_rows-1),
                    max_size=min(num_rows // 2, 10)  # At most half the rows, max 10
                ))
                
                column_data = base_data.copy()
                for idx in missing_indices:
                    column_data[idx] = None
                    
                expected_missing_counts[col_name] = len(missing_indices)
            
            df_data[col_name] = column_data
        
        df = pd.DataFrame(df_data)
        
        # Analyze missing values using our analyzer
        missing_report = self.analyzer.analyze_missing_values(df)
        
        # Verify that missing value counts are accurate for all columns
        assert set(missing_report.missing_counts.keys()) == set(df.columns), (
            f"Missing value report should include all columns. "
            f"Expected columns: {set(df.columns)}, "
            f"Got columns: {set(missing_report.missing_counts.keys())}"
        )
        
        # Verify accuracy of missing value counts for each column
        for column in df.columns:
            # Calculate actual missing values using pandas
            actual_missing = int(df[column].isna().sum())
            reported_missing = missing_report.missing_counts[column]
            
            assert actual_missing == reported_missing, (
                f"Missing value count incorrect for column '{column}': "
                f"reported {reported_missing}, actual {actual_missing}. "
                f"The reported missing value count must exactly match the actual count. "
                f"This violates the missing value count accuracy property."
            )
            
            # Verify that our expected count matches the actual count
            # (this validates our test data generation)
            expected_count = expected_missing_counts[column]
            assert actual_missing == expected_count, (
                f"Test data generation error for column '{column}': "
                f"expected {expected_count} missing values, but pandas found {actual_missing}. "
                f"This indicates an issue with the test data generation logic."
            )
        
        # Verify accuracy of missing value percentages
        for column in df.columns:
            actual_missing = int(df[column].isna().sum())
            expected_percentage = (actual_missing / len(df)) * 100
            reported_percentage = missing_report.missing_percentages[column]
            
            assert abs(reported_percentage - expected_percentage) < 1e-10, (
                f"Missing value percentage incorrect for column '{column}': "
                f"reported {reported_percentage}%, expected {expected_percentage}%. "
                f"Difference: {abs(reported_percentage - expected_percentage)}. "
                f"The reported percentage must be accurate to within floating point precision."
            )
        
        # Verify accuracy of total missing count
        expected_total = sum(int(df[col].isna().sum()) for col in df.columns)
        reported_total = missing_report.total_missing
        
        assert expected_total == reported_total, (
            f"Total missing value count incorrect: "
            f"reported {reported_total}, expected {expected_total}. "
            f"The total missing count should be the sum of missing values across all columns. "
            f"This violates the missing value count accuracy property."
        )
        
        # Verify that total equals sum of individual counts
        sum_of_individual = sum(missing_report.missing_counts.values())
        assert reported_total == sum_of_individual, (
            f"Total missing count ({reported_total}) does not equal sum of individual counts ({sum_of_individual}). "
            f"This indicates an internal consistency error in the missing value report."
        )

    # Feature: nasa-pcoe-eda, Property 8: データ型識別の正確性
    @given(st.data())
    @settings(max_examples=100)
    def test_data_type_identification_accuracy(self, data):
        """
        任意のデータセットに対して、各特徴量について識別されたデータ型は、実際のデータ型と一致する
        
        Property 8: Data Type Identification Accuracy
        For any dataset, the identified data type for each feature should match the actual data type.
        """
        # Generate a DataFrame with various data types
        num_columns = data.draw(st.integers(min_value=1, max_value=8))
        num_rows = data.draw(st.integers(min_value=5, max_value=100))
        
        # Generate DataFrame data with different data types
        df_data = {}
        expected_dtypes = {}
        
        for i in range(num_columns):
            col_name = f"feature_{i}"
            
            # Choose column type randomly
            col_type = data.draw(st.sampled_from([
                'int64', 'float64', 'object', 'bool', 'datetime64[ns]', 'category'
            ]))
            
            if col_type == 'int64':
                # Generate integer data
                column_data = data.draw(st.lists(
                    st.integers(min_value=-1000, max_value=1000),
                    min_size=num_rows,
                    max_size=num_rows
                ))
                df_data[col_name] = column_data
                expected_dtypes[col_name] = 'int64'
                
            elif col_type == 'float64':
                # Generate float data
                column_data = data.draw(st.lists(
                    st.floats(
                        min_value=-1000.0, 
                        max_value=1000.0, 
                        allow_nan=False, 
                        allow_infinity=False
                    ),
                    min_size=num_rows,
                    max_size=num_rows
                ))
                df_data[col_name] = column_data
                expected_dtypes[col_name] = 'float64'
                
            elif col_type == 'object':
                # Generate string data
                column_data = data.draw(st.lists(
                    st.text(min_size=1, max_size=20),
                    min_size=num_rows,
                    max_size=num_rows
                ))
                df_data[col_name] = column_data
                expected_dtypes[col_name] = 'object'
                
            elif col_type == 'bool':
                # Generate boolean data
                column_data = data.draw(st.lists(
                    st.booleans(),
                    min_size=num_rows,
                    max_size=num_rows
                ))
                df_data[col_name] = column_data
                expected_dtypes[col_name] = 'bool'
                
            elif col_type == 'datetime64[ns]':
                # Generate datetime data
                base_timestamp = pd.Timestamp('2020-01-01')
                column_data = [
                    base_timestamp + pd.Timedelta(days=data.draw(st.integers(min_value=0, max_value=365)))
                    for _ in range(num_rows)
                ]
                df_data[col_name] = column_data
                expected_dtypes[col_name] = 'datetime64[ns]'
                
            else:  # category
                # Generate categorical data
                categories = data.draw(st.lists(
                    st.text(min_size=1, max_size=10),
                    min_size=2,
                    max_size=5,
                    unique=True
                ))
                column_data = data.draw(st.lists(
                    st.sampled_from(categories),
                    min_size=num_rows,
                    max_size=num_rows
                ))
                df_data[col_name] = pd.Categorical(column_data)
                expected_dtypes[col_name] = 'category'
        
        # Create DataFrame and ensure proper dtypes
        df = pd.DataFrame(df_data)
        
        # Explicitly set dtypes to ensure they match our expectations
        for col_name, expected_dtype in expected_dtypes.items():
            if expected_dtype == 'int64':
                df[col_name] = df[col_name].astype('int64')
            elif expected_dtype == 'float64':
                df[col_name] = df[col_name].astype('float64')
            elif expected_dtype == 'object':
                df[col_name] = df[col_name].astype('object')
            elif expected_dtype == 'bool':
                df[col_name] = df[col_name].astype('bool')
            elif expected_dtype == 'datetime64[ns]':
                df[col_name] = pd.to_datetime(df[col_name])
            elif expected_dtype == 'category':
                df[col_name] = df[col_name].astype('category')
        
        # Identify data types using our analyzer
        identified_types = self.analyzer.identify_data_types(df)
        
        # Verify that data types are identified for all columns
        assert set(identified_types.keys()) == set(df.columns), (
            f"Data type identification should include all columns. "
            f"Expected columns: {set(df.columns)}, "
            f"Got columns: {set(identified_types.keys())}"
        )
        
        # Verify accuracy of data type identification for each column
        for column in df.columns:
            # Get the actual pandas dtype
            actual_dtype = str(df[column].dtype)
            identified_dtype = identified_types[column]
            
            assert actual_dtype == identified_dtype, (
                f"Data type identification incorrect for column '{column}': "
                f"identified '{identified_dtype}', actual '{actual_dtype}'. "
                f"The identified data type must exactly match the actual pandas dtype. "
                f"This violates the data type identification accuracy property."
            )
            
            # Verify that the identified type is a valid pandas dtype string
            assert isinstance(identified_dtype, str), (
                f"Identified data type for column '{column}' should be a string, "
                f"but got {type(identified_dtype)}. "
                f"Data types should be returned as string representations."
            )
            
            # Verify that the identified type is not empty
            assert len(identified_dtype.strip()) > 0, (
                f"Identified data type for column '{column}' is empty or whitespace-only: '{identified_dtype}'. "
                f"Data type identification should return meaningful type strings."
            )
        
        # Verify consistency: running the method multiple times should give same results
        identified_types_second = self.analyzer.identify_data_types(df)
        assert identified_types == identified_types_second, (
            f"Data type identification is not consistent. "
            f"First run: {identified_types}, "
            f"Second run: {identified_types_second}. "
            f"The method should return identical results for the same input."
        )
        
        # Verify that common pandas dtypes are correctly identified
        for column in df.columns:
            dtype_str = identified_types[column]
            actual_dtype = df[column].dtype
            
            # Check that numeric types are properly identified
            if pd.api.types.is_integer_dtype(actual_dtype):
                assert 'int' in dtype_str.lower(), (
                    f"Integer column '{column}' should have 'int' in its dtype string, "
                    f"but got '{dtype_str}'. This suggests incorrect type identification."
                )
            elif pd.api.types.is_float_dtype(actual_dtype):
                assert 'float' in dtype_str.lower(), (
                    f"Float column '{column}' should have 'float' in its dtype string, "
                    f"but got '{dtype_str}'. This suggests incorrect type identification."
                )
            elif pd.api.types.is_bool_dtype(actual_dtype):
                assert 'bool' in dtype_str.lower(), (
                    f"Boolean column '{column}' should have 'bool' in its dtype string, "
                    f"but got '{dtype_str}'. This suggests incorrect type identification."
                )
            elif pd.api.types.is_datetime64_any_dtype(actual_dtype):
                assert 'datetime' in dtype_str.lower(), (
                    f"Datetime column '{column}' should have 'datetime' in its dtype string, "
                    f"but got '{dtype_str}'. This suggests incorrect type identification."
                )
            elif pd.api.types.is_categorical_dtype(actual_dtype):
                assert 'category' in dtype_str.lower(), (
                    f"Categorical column '{column}' should have 'category' in its dtype string, "
                    f"but got '{dtype_str}'. This suggests incorrect type identification."
                )
            elif pd.api.types.is_object_dtype(actual_dtype):
                assert 'object' in dtype_str.lower(), (
                    f"Object column '{column}' should have 'object' in its dtype string, "
                    f"but got '{dtype_str}'. This suggests incorrect type identification."
                )