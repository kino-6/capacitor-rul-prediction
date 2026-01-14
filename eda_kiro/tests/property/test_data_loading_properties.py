"""Property-based tests for data loading completeness."""

import tempfile
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings
from scipy.io import savemat

from nasa_pcoe_eda.data.loader import DataLoader
from nasa_pcoe_eda.exceptions import DataLoadError


class TestDataLoadingProperties:
    """Property-based tests for data loading functionality."""

    # Feature: nasa-pcoe-eda, Property 2: データ読み込みの完全性
    @given(st.data())
    @settings(max_examples=100)
    def test_data_loading_completeness(self, data):
        """
        任意の有効なデータセットファイルに対して、読み込み後に報告されるレコード数と特徴量数は、
        実際のファイル内容と一致する
        
        Property 2: Data loading completeness
        For any valid dataset file, the reported record count and feature count after loading
        should match the actual file contents.
        """
        # Generate random data dimensions
        n_records = data.draw(st.integers(min_value=1, max_value=100))
        n_features = data.draw(st.integers(min_value=1, max_value=20))
        
        # Generate random data structure type
        data_structure_type = data.draw(st.sampled_from(['regular_array', 'structured_array']))
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as tmp_file:
            mat_path = Path(tmp_file.name)
        
        try:
            # Generate test data based on structure type
            if data_structure_type == 'regular_array':
                # Create regular 2D array
                test_data = data.draw(st.lists(
                    st.lists(
                        st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
                        min_size=n_features,
                        max_size=n_features
                    ),
                    min_size=n_records,
                    max_size=n_records
                ))
                
                # Convert to numpy array
                np_data = np.array(test_data)
                
                # Save to .mat file
                mat_data = {"test_data": np_data}
                savemat(str(mat_path), mat_data)
                
                # Expected dimensions
                expected_records = n_records
                expected_features = n_features
                
            else:  # structured_array
                # Generate field names (ASCII only for MATLAB compatibility)
                field_names = []
                for i in range(n_features):
                    field_name = data.draw(st.text(
                        alphabet='abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_',
                        min_size=1,
                        max_size=10
                    ).filter(lambda x: x and x[0].isalpha()))
                    # Ensure unique field names
                    while field_name in field_names:
                        field_name += '_' + str(len(field_names))
                    field_names.append(field_name)
                
                # Create structured array dtype
                dt = np.dtype([(name, 'f8') for name in field_names])
                
                # Generate structured data
                structured_data = []
                for _ in range(n_records):
                    record = []
                    for _ in range(n_features):
                        value = data.draw(st.floats(
                            min_value=-1000.0, 
                            max_value=1000.0, 
                            allow_nan=False, 
                            allow_infinity=False
                        ))
                        record.append(value)
                    structured_data.append(tuple(record))
                
                # Convert to numpy structured array
                np_data = np.array(structured_data, dtype=dt)
                
                # Save to .mat file
                mat_data = {"structured_data": np_data}
                savemat(str(mat_path), mat_data)
                
                # Expected dimensions
                expected_records = n_records
                expected_features = n_features
            
            # Load the data using DataLoader
            loader = DataLoader()
            df = loader.load_dataset(mat_path)
            
            # Get metadata to check reported dimensions
            metadata = loader.get_metadata(df)
            
            # Verify completeness property: reported dimensions match actual dimensions
            assert metadata.n_records == expected_records, (
                f"Data loading completeness violated: reported record count {metadata.n_records} "
                f"does not match actual record count {expected_records}. "
                f"Data structure type: {data_structure_type}, "
                f"DataFrame shape: {df.shape}"
            )
            
            assert metadata.n_features == expected_features, (
                f"Data loading completeness violated: reported feature count {metadata.n_features} "
                f"does not match actual feature count {expected_features}. "
                f"Data structure type: {data_structure_type}, "
                f"DataFrame shape: {df.shape}"
            )
            
            # Additional verification: DataFrame dimensions should match metadata
            assert len(df) == metadata.n_records, (
                f"Inconsistency between DataFrame and metadata: "
                f"DataFrame has {len(df)} records but metadata reports {metadata.n_records}"
            )
            
            assert len(df.columns) == metadata.n_features, (
                f"Inconsistency between DataFrame and metadata: "
                f"DataFrame has {len(df.columns)} features but metadata reports {metadata.n_features}"
            )
            
            # Verify that the actual data content is preserved
            # For regular arrays, check that all data is numeric and finite
            if data_structure_type == 'regular_array':
                # All columns should be numeric
                for col in df.columns:
                    assert pd.api.types.is_numeric_dtype(df[col]), (
                        f"Column {col} should be numeric but has dtype {df[col].dtype}"
                    )
                    # All values should be finite (no NaN or inf)
                    assert df[col].isna().sum() == 0, (
                        f"Column {col} contains {df[col].isna().sum()} NaN values, "
                        f"but input data had no NaN values"
                    )
            
            # For structured arrays, verify field names are preserved
            elif data_structure_type == 'structured_array':
                # Check that we have the expected number of columns
                assert len(df.columns) == len(field_names), (
                    f"Expected {len(field_names)} columns from structured array, "
                    f"but got {len(df.columns)}"
                )
        
        finally:
            # Clean up temporary file
            if mat_path.exists():
                mat_path.unlink()

    # Feature: nasa-pcoe-eda, Property 2: データ読み込みの完全性 (Edge case testing)
    @given(st.data())
    @settings(max_examples=50)
    def test_data_loading_completeness_edge_cases(self, data):
        """
        Test data loading completeness for edge cases like single row/column data.
        
        This extends Property 2 to cover edge cases that might be handled differently.
        """
        # Generate edge case dimensions
        edge_case_type = data.draw(st.sampled_from(['single_row', 'single_column', 'single_cell']))
        
        if edge_case_type == 'single_row':
            n_records = 1
            n_features = data.draw(st.integers(min_value=1, max_value=10))
        elif edge_case_type == 'single_column':
            n_records = data.draw(st.integers(min_value=1, max_value=10))
            n_features = 1
        else:  # single_cell
            n_records = 1
            n_features = 1
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as tmp_file:
            mat_path = Path(tmp_file.name)
        
        try:
            # Generate test data
            test_data = []
            for _ in range(n_records):
                row = []
                for _ in range(n_features):
                    value = data.draw(st.floats(
                        min_value=-100.0, 
                        max_value=100.0, 
                        allow_nan=False, 
                        allow_infinity=False
                    ))
                    row.append(value)
                test_data.append(row)
            
            # Convert to numpy array
            np_data = np.array(test_data)
            
            # Handle 1D case (single column)
            if n_features == 1 and n_records > 1:
                np_data = np_data.flatten()
            
            # Save to .mat file
            mat_data = {"edge_case_data": np_data}
            savemat(str(mat_path), mat_data)
            
            # Load the data using DataLoader
            loader = DataLoader()
            df = loader.load_dataset(mat_path)
            
            # Get metadata
            metadata = loader.get_metadata(df)
            
            # For edge cases, the loader might reshape the data
            # But the total number of data points should be preserved
            total_expected_points = n_records * n_features
            total_actual_points = metadata.n_records * metadata.n_features
            
            assert total_actual_points == total_expected_points, (
                f"Data loading completeness violated for edge case '{edge_case_type}': "
                f"expected {total_expected_points} total data points, "
                f"but got {total_actual_points} "
                f"(records: {metadata.n_records}, features: {metadata.n_features})"
            )
            
            # Verify DataFrame is not empty
            assert not df.empty, (
                f"DataFrame should not be empty for edge case '{edge_case_type}' "
                f"with {n_records} records and {n_features} features"
            )
            
            # Verify all data is preserved (no NaN introduced)
            total_nan_count = df.isna().sum().sum()
            assert total_nan_count == 0, (
                f"Edge case '{edge_case_type}' introduced {total_nan_count} NaN values "
                f"when none were expected"
            )
        
        finally:
            # Clean up temporary file
            if mat_path.exists():
                mat_path.unlink()

    # Feature: nasa-pcoe-eda, Property 5: データ永続性
    @given(st.data())
    @settings(max_examples=100)
    def test_data_persistence(self, data):
        """
        任意のデータセットに対して、読み込み後にデータがメモリに保持され、
        後続の分析操作でアクセス可能である
        
        Property 5: Data persistence
        For any dataset, after loading, the data should be retained in memory
        and accessible for subsequent analysis operations.
        """
        # Generate random data dimensions
        n_records = data.draw(st.integers(min_value=1, max_value=50))
        n_features = data.draw(st.integers(min_value=1, max_value=10))
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as tmp_file:
            mat_path = Path(tmp_file.name)
        
        try:
            # Generate test data
            test_data = []
            for _ in range(n_records):
                row = []
                for _ in range(n_features):
                    value = data.draw(st.floats(
                        min_value=-100.0, 
                        max_value=100.0, 
                        allow_nan=False, 
                        allow_infinity=False
                    ))
                    row.append(value)
                test_data.append(row)
            
            # Convert to numpy array
            np_data = np.array(test_data)
            
            # Save to .mat file
            mat_data = {"persistence_test_data": np_data}
            savemat(str(mat_path), mat_data)
            
            # Create DataLoader instance
            loader = DataLoader()
            
            # Verify initially no data is loaded
            assert loader.data is None, (
                "DataLoader should have no data initially"
            )
            
            # Load the data
            df_loaded = loader.load_dataset(mat_path)
            
            # Property 5: Verify data persists in memory after loading
            # 1. The loader should retain the data internally
            assert loader.data is not None, (
                "Data persistence violated: DataLoader.data should not be None after loading"
            )
            
            # 2. The retained data should be the same as what was returned
            assert loader.data is df_loaded, (
                "Data persistence violated: DataLoader.data should be the same object as returned DataFrame"
            )
            
            # 3. The data should be accessible for subsequent operations
            # Test various analysis operations that would be performed later
            
            # Basic DataFrame operations should work
            assert len(loader.data) == n_records, (
                f"Data persistence violated: persisted data has {len(loader.data)} records, "
                f"expected {n_records}"
            )
            
            assert len(loader.data.columns) == n_features, (
                f"Data persistence violated: persisted data has {len(loader.data.columns)} features, "
                f"expected {n_features}"
            )
            
            # Statistical operations should work on persisted data
            try:
                # These operations should succeed if data is properly persisted
                mean_values = loader.data.mean()
                assert len(mean_values) == n_features, (
                    "Data persistence violated: cannot compute statistics on persisted data"
                )
                
                std_values = loader.data.std()
                assert len(std_values) == n_features, (
                    "Data persistence violated: cannot compute standard deviation on persisted data"
                )
                
                # Correlation matrix should be computable
                if n_features > 1:
                    corr_matrix = loader.data.corr()
                    assert corr_matrix.shape == (n_features, n_features), (
                        f"Data persistence violated: correlation matrix has shape {corr_matrix.shape}, "
                        f"expected ({n_features}, {n_features})"
                    )
                
            except Exception as e:
                raise AssertionError(
                    f"Data persistence violated: subsequent analysis operations failed on persisted data: {e}"
                )
            
            # 4. Data should remain accessible after multiple accesses
            first_access = loader.data
            second_access = loader.data
            
            assert first_access is second_access, (
                "Data persistence violated: multiple accesses to loader.data return different objects"
            )
            
            # 5. Data content should remain unchanged after access
            original_shape = df_loaded.shape
            original_dtypes = df_loaded.dtypes.copy()
            
            # Access the data multiple times
            for _ in range(3):
                accessed_data = loader.data
                assert accessed_data.shape == original_shape, (
                    f"Data persistence violated: data shape changed from {original_shape} "
                    f"to {accessed_data.shape} after access"
                )
                
                # Check that dtypes haven't changed
                for col in original_dtypes.index:
                    if col in accessed_data.columns:
                        assert accessed_data[col].dtype == original_dtypes[col], (
                            f"Data persistence violated: column '{col}' dtype changed from "
                            f"{original_dtypes[col]} to {accessed_data[col].dtype}"
                        )
            
            # 6. Verify data can be used for typical EDA operations
            try:
                # Missing value analysis
                missing_counts = loader.data.isnull().sum()
                assert len(missing_counts) == n_features, (
                    "Data persistence violated: cannot perform missing value analysis"
                )
                
                # Data type identification
                data_types = loader.data.dtypes
                assert len(data_types) == n_features, (
                    "Data persistence violated: cannot identify data types"
                )
                
                # Basic indexing and slicing
                if n_records > 1:
                    subset = loader.data.iloc[0:min(2, n_records)]
                    assert len(subset) <= 2, (
                        "Data persistence violated: cannot perform indexing operations"
                    )
                
                if n_features > 1:
                    col_subset = loader.data.iloc[:, 0:min(2, n_features)]
                    assert len(col_subset.columns) <= 2, (
                        "Data persistence violated: cannot perform column selection"
                    )
                
            except Exception as e:
                raise AssertionError(
                    f"Data persistence violated: typical EDA operations failed: {e}"
                )
            
            # 7. Test that loading new data replaces the old data
            # Create a second dataset with different dimensions
            new_n_records = data.draw(st.integers(min_value=1, max_value=30))
            new_n_features = data.draw(st.integers(min_value=1, max_value=8))
            
            # Create second temporary file
            with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as tmp_file2:
                mat_path2 = Path(tmp_file2.name)
            
            try:
                # Generate different test data
                new_test_data = []
                for _ in range(new_n_records):
                    row = []
                    for _ in range(new_n_features):
                        value = data.draw(st.floats(
                            min_value=200.0,  # Different range to ensure different data
                            max_value=300.0, 
                            allow_nan=False, 
                            allow_infinity=False
                        ))
                        row.append(value)
                    new_test_data.append(row)
                
                # Convert to numpy array
                new_np_data = np.array(new_test_data)
                
                # Save to second .mat file
                new_mat_data = {"new_persistence_test_data": new_np_data}
                savemat(str(mat_path2), new_mat_data)
                
                # Load the new data
                df_new = loader.load_dataset(mat_path2)
                
                # Verify the persisted data is now the new data
                assert loader.data is df_new, (
                    "Data persistence violated: loader should persist the most recently loaded data"
                )
                
                assert loader.data.shape == (new_n_records, new_n_features), (
                    f"Data persistence violated: persisted data has shape {loader.data.shape}, "
                    f"expected ({new_n_records}, {new_n_features}) for new dataset"
                )
                
            finally:
                # Clean up second temporary file
                if mat_path2.exists():
                    mat_path2.unlink()
        
        finally:
            # Clean up first temporary file
            if mat_path.exists():
                mat_path.unlink()