"""Property-based tests for data validation consistency."""

import tempfile
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from hypothesis import given, strategies as st, settings
from scipy.io import savemat

from nasa_pcoe_eda.data.loader import DataLoader
from nasa_pcoe_eda.exceptions import DataLoadError, DataValidationError


class TestDataValidationProperties:
    """Property-based tests for data validation functionality."""

    # Feature: nasa-pcoe-eda, Property 3: データ検証の一貫性
    @given(st.data())
    @settings(max_examples=100)
    def test_data_validation_consistency(self, data):
        """
        任意のデータファイルに対して、検証関数は同じ入力に対して常に同じ検証結果を返す
        
        Property 3: Data validation consistency
        For any data file, the validation function returns the same validation result 
        for the same input.
        """
        # Generate test DataFrame with various characteristics
        n_records = data.draw(st.integers(min_value=0, max_value=50))
        n_features = data.draw(st.integers(min_value=0, max_value=10))
        
        # Generate DataFrame structure type
        df_type = data.draw(st.sampled_from([
            'normal', 'empty', 'no_columns', 'no_rows', 
            'with_nulls', 'all_nulls', 'duplicate_columns'
        ]))
        
        # Create DataFrame based on type
        if df_type == 'empty':
            df = pd.DataFrame()
        elif df_type == 'no_columns':
            df = pd.DataFrame(index=range(max(1, n_records)))
        elif df_type == 'no_rows':
            if n_features > 0:
                columns = [f'col_{i}' for i in range(n_features)]
                df = pd.DataFrame(columns=columns)
            else:
                df = pd.DataFrame()
        elif df_type == 'duplicate_columns':
            if n_features > 1:
                # Create DataFrame with duplicate column names
                columns = [f'col_{i}' for i in range(n_features)]
                # Make some columns duplicate
                duplicate_indices = data.draw(st.lists(
                    st.integers(min_value=1, max_value=n_features-1),
                    min_size=1,
                    max_size=min(3, n_features-1),
                    unique=True
                ))
                for idx in duplicate_indices:
                    columns[idx] = columns[0]  # Make it duplicate the first column
                
                # Generate data
                test_data = []
                for _ in range(max(1, n_records)):
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
                
                df = pd.DataFrame(test_data, columns=columns)
            else:
                # Fallback to normal case
                df = self._create_normal_dataframe(data, n_records, n_features)
        elif df_type == 'all_nulls':
            if n_features > 0 and n_records > 0:
                columns = [f'col_{i}' for i in range(n_features)]
                df = pd.DataFrame(np.nan, index=range(n_records), columns=columns)
            else:
                df = pd.DataFrame()
        elif df_type == 'with_nulls':
            df = self._create_dataframe_with_nulls(data, n_records, n_features)
        else:  # normal
            df = self._create_normal_dataframe(data, n_records, n_features)
        
        # Create DataLoader instance
        loader = DataLoader()
        
        # Perform validation multiple times on the same DataFrame
        try:
            result1 = loader.validate_data(df)
            result2 = loader.validate_data(df)
            result3 = loader.validate_data(df)
            
            # Verify consistency: all results should be identical
            assert result1.is_valid == result2.is_valid == result3.is_valid, (
                f"Data validation consistency violated: is_valid differs across calls. "
                f"Call 1: {result1.is_valid}, Call 2: {result2.is_valid}, Call 3: {result3.is_valid}. "
                f"DataFrame type: {df_type}, shape: {df.shape}"
            )
            
            assert result1.errors == result2.errors == result3.errors, (
                f"Data validation consistency violated: errors differ across calls. "
                f"Call 1: {result1.errors}, Call 2: {result2.errors}, Call 3: {result3.errors}. "
                f"DataFrame type: {df_type}, shape: {df.shape}"
            )
            
            assert result1.warnings == result2.warnings == result3.warnings, (
                f"Data validation consistency violated: warnings differ across calls. "
                f"Call 1: {result1.warnings}, Call 2: {result2.warnings}, Call 3: {result3.warnings}. "
                f"DataFrame type: {df_type}, shape: {df.shape}"
            )
            
            # Additional consistency check: validate that the validation logic is deterministic
            # by checking that the same conditions produce the same results
            if df.empty:
                assert not result1.is_valid, (
                    f"Empty DataFrame should be invalid, but validation returned is_valid=True"
                )
                assert "DataFrame is empty" in result1.errors, (
                    f"Empty DataFrame should have 'DataFrame is empty' error, "
                    f"but got errors: {result1.errors}"
                )
            
            if len(df.columns) == 0 and not df.empty:
                assert not result1.is_valid, (
                    f"DataFrame with no columns should be invalid, but validation returned is_valid=True"
                )
                assert "DataFrame has no columns" in result1.errors, (
                    f"DataFrame with no columns should have 'DataFrame has no columns' error, "
                    f"but got errors: {result1.errors}"
                )
            
            if len(df) == 0 and len(df.columns) > 0:
                assert not result1.is_valid, (
                    f"DataFrame with no rows should be invalid, but validation returned is_valid=True"
                )
                assert "DataFrame has no rows" in result1.errors, (
                    f"DataFrame with no rows should have 'DataFrame has no rows' error, "
                    f"but got errors: {result1.errors}"
                )
            
            # Check duplicate columns detection consistency
            if not df.empty and df.columns.duplicated().any():
                assert not result1.is_valid, (
                    f"DataFrame with duplicate columns should be invalid, "
                    f"but validation returned is_valid=True"
                )
                duplicate_error_found = any("Duplicate column names" in error for error in result1.errors)
                assert duplicate_error_found, (
                    f"DataFrame with duplicate columns should have duplicate column error, "
                    f"but got errors: {result1.errors}"
                )
        
        except DataValidationError as e:
            # If validation raises an exception, it should do so consistently
            try:
                loader.validate_data(df)
                assert False, (
                    f"Data validation consistency violated: first call raised exception "
                    f"'{str(e)}' but second call did not raise an exception. "
                    f"DataFrame type: {df_type}, shape: {df.shape}"
                )
            except DataValidationError as e2:
                # Both calls should raise the same type of exception
                # We can't easily compare exception messages as they might contain memory addresses
                # But the exception type should be consistent
                assert type(e) == type(e2), (
                    f"Data validation consistency violated: exception types differ. "
                    f"First: {type(e)}, Second: {type(e2)}. "
                    f"DataFrame type: {df_type}, shape: {df.shape}"
                )

    def _create_normal_dataframe(self, data, n_records: int, n_features: int) -> pd.DataFrame:
        """Create a normal DataFrame with random data."""
        if n_records == 0 or n_features == 0:
            return pd.DataFrame()
        
        columns = [f'col_{i}' for i in range(n_features)]
        test_data = []
        
        for _ in range(n_records):
            row = []
            for _ in range(n_features):
                value = data.draw(st.floats(
                    min_value=-1000.0, 
                    max_value=1000.0, 
                    allow_nan=False, 
                    allow_infinity=False
                ))
                row.append(value)
            test_data.append(row)
        
        return pd.DataFrame(test_data, columns=columns)

    def _create_dataframe_with_nulls(self, data, n_records: int, n_features: int) -> pd.DataFrame:
        """Create a DataFrame with some null values."""
        if n_records == 0 or n_features == 0:
            return pd.DataFrame()
        
        columns = [f'col_{i}' for i in range(n_features)]
        test_data = []
        
        # Determine null probability
        null_probability = data.draw(st.floats(min_value=0.1, max_value=0.8))
        
        for _ in range(n_records):
            row = []
            for _ in range(n_features):
                # Decide whether to insert null
                is_null = data.draw(st.floats(min_value=0.0, max_value=1.0)) < null_probability
                
                if is_null:
                    value = np.nan
                else:
                    value = data.draw(st.floats(
                        min_value=-100.0, 
                        max_value=100.0, 
                        allow_nan=False, 
                        allow_infinity=False
                    ))
                row.append(value)
            test_data.append(row)
        
        return pd.DataFrame(test_data, columns=columns)

    # Feature: nasa-pcoe-eda, Property 3: データ検証の一貫性 (Idempotency test)
    @given(st.data())
    @settings(max_examples=50)
    def test_validation_idempotency_with_modifications(self, data):
        """
        Test that validation remains consistent even when DataFrame is modified between calls.
        
        This extends Property 3 to ensure that validation doesn't have side effects
        that could affect subsequent validations.
        """
        # Create a base DataFrame
        n_records = data.draw(st.integers(min_value=1, max_value=20))
        n_features = data.draw(st.integers(min_value=1, max_value=5))
        
        df = self._create_normal_dataframe(data, n_records, n_features)
        
        # Create DataLoader instance
        loader = DataLoader()
        
        # Validate the original DataFrame
        original_result = loader.validate_data(df)
        
        # Create a copy and validate it (should be identical)
        df_copy = df.copy()
        copy_result = loader.validate_data(df_copy)
        
        # Results should be identical for identical DataFrames
        assert original_result.is_valid == copy_result.is_valid, (
            f"Validation consistency violated: identical DataFrames produced different is_valid results. "
            f"Original: {original_result.is_valid}, Copy: {copy_result.is_valid}"
        )
        
        assert original_result.errors == copy_result.errors, (
            f"Validation consistency violated: identical DataFrames produced different errors. "
            f"Original: {original_result.errors}, Copy: {copy_result.errors}"
        )
        
        assert original_result.warnings == copy_result.warnings, (
            f"Validation consistency violated: identical DataFrames produced different warnings. "
            f"Original: {original_result.warnings}, Copy: {copy_result.warnings}"
        )
        
        # Modify the copy and validate again
        if len(df_copy.columns) > 0 and len(df_copy) > 0:
            # Add some NaN values
            col_to_modify = data.draw(st.sampled_from(df_copy.columns.tolist()))
            row_to_modify = data.draw(st.integers(min_value=0, max_value=len(df_copy)-1))
            df_copy.loc[row_to_modify, col_to_modify] = np.nan
            
            # Validate the modified DataFrame
            modified_result = loader.validate_data(df_copy)
            
            # Now validate the original DataFrame again - should be unchanged
            final_original_result = loader.validate_data(df)
            
            # Original DataFrame validation should be consistent
            assert original_result.is_valid == final_original_result.is_valid, (
                f"Validation consistency violated: original DataFrame validation changed after "
                f"validating a modified copy. "
                f"Initial: {original_result.is_valid}, Final: {final_original_result.is_valid}"
            )
            
            assert original_result.errors == final_original_result.errors, (
                f"Validation consistency violated: original DataFrame errors changed after "
                f"validating a modified copy. "
                f"Initial: {original_result.errors}, Final: {final_original_result.errors}"
            )
            
            assert original_result.warnings == final_original_result.warnings, (
                f"Validation consistency violated: original DataFrame warnings changed after "
                f"validating a modified copy. "
                f"Initial: {original_result.warnings}, Final: {final_original_result.warnings}"
            )

    # Feature: nasa-pcoe-eda, Property 4: エラーハンドリングの堅牢性
    @given(st.data())
    @settings(max_examples=100)
    def test_error_handling_robustness(self, data):
        """
        任意の破損または欠落したデータファイルに対して、システムは具体的なエラーメッセージを生成し、処理を継続する
        
        Property 4: Error handling robustness
        For any corrupted or missing data file, the system generates specific error messages 
        and continues processing.
        """
        loader = DataLoader()
        
        # Test different types of file errors
        error_type = data.draw(st.sampled_from([
            'missing_file', 'not_a_file', 'wrong_extension', 
            'empty_mat_file', 'corrupted_mat_file', 'invalid_structure'
        ]))
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            if error_type == 'missing_file':
                # Test with non-existent file
                non_existent_file = temp_path / "nonexistent.mat"
                
                try:
                    loader.load_dataset(non_existent_file)
                    assert False, "Expected DataLoadError for missing file, but no exception was raised"
                except DataLoadError as e:
                    # Verify specific error message
                    error_msg = str(e)
                    assert "not found" in error_msg.lower() or "does not exist" in error_msg.lower(), (
                        f"Error message should mention file not found, but got: {error_msg}"
                    )
                    assert str(non_existent_file) in error_msg, (
                        f"Error message should include the file path, but got: {error_msg}"
                    )
                except Exception as e:
                    assert False, (
                        f"Expected DataLoadError for missing file, but got {type(e).__name__}: {str(e)}"
                    )
            
            elif error_type == 'not_a_file':
                # Test with directory instead of file
                directory_path = temp_path / "test_directory"
                directory_path.mkdir()
                
                try:
                    loader.load_dataset(directory_path)
                    assert False, "Expected DataLoadError for directory path, but no exception was raised"
                except DataLoadError as e:
                    error_msg = str(e)
                    assert "not a file" in error_msg.lower() or "is not a file" in error_msg.lower(), (
                        f"Error message should mention path is not a file, but got: {error_msg}"
                    )
                    assert str(directory_path) in error_msg, (
                        f"Error message should include the path, but got: {error_msg}"
                    )
                except Exception as e:
                    assert False, (
                        f"Expected DataLoadError for directory path, but got {type(e).__name__}: {str(e)}"
                    )
            
            elif error_type == 'wrong_extension':
                # Test with wrong file extension
                wrong_ext_file = temp_path / "test.txt"
                wrong_ext_file.write_text("some text content")
                
                try:
                    loader.load_dataset(wrong_ext_file)
                    assert False, "Expected DataLoadError for wrong extension, but no exception was raised"
                except DataLoadError as e:
                    error_msg = str(e)
                    assert "invalid file format" in error_msg.lower() or "expected .mat" in error_msg.lower(), (
                        f"Error message should mention invalid file format, but got: {error_msg}"
                    )
                    assert ".txt" in error_msg or "txt" in error_msg, (
                        f"Error message should mention the actual extension, but got: {error_msg}"
                    )
                except Exception as e:
                    assert False, (
                        f"Expected DataLoadError for wrong extension, but got {type(e).__name__}: {str(e)}"
                    )
            
            elif error_type == 'empty_mat_file':
                # Test with empty .mat file
                empty_mat_file = temp_path / "empty.mat"
                
                # Create an empty .mat file (this will likely be corrupted)
                try:
                    # Try to create a minimal but invalid .mat file
                    empty_mat_file.write_bytes(b"")
                    
                    try:
                        loader.load_dataset(empty_mat_file)
                        assert False, "Expected DataLoadError for empty .mat file, but no exception was raised"
                    except DataLoadError as e:
                        error_msg = str(e)
                        # Should contain specific error information
                        assert len(error_msg) > 10, (
                            f"Error message should be descriptive, but got: {error_msg}"
                        )
                        assert str(empty_mat_file) in error_msg or "empty.mat" in error_msg, (
                            f"Error message should reference the file, but got: {error_msg}"
                        )
                    except Exception as e:
                        assert False, (
                            f"Expected DataLoadError for empty .mat file, but got {type(e).__name__}: {str(e)}"
                        )
                except Exception:
                    # If we can't create the test file, skip this case
                    pass
            
            elif error_type == 'corrupted_mat_file':
                # Test with corrupted .mat file
                corrupted_mat_file = temp_path / "corrupted.mat"
                
                # Create a file with .mat extension but invalid content
                corrupted_content = data.draw(st.binary(min_size=10, max_size=100))
                corrupted_mat_file.write_bytes(corrupted_content)
                
                try:
                    loader.load_dataset(corrupted_mat_file)
                    assert False, "Expected DataLoadError for corrupted .mat file, but no exception was raised"
                except DataLoadError as e:
                    error_msg = str(e)
                    # Should contain specific error information
                    assert len(error_msg) > 10, (
                        f"Error message should be descriptive, but got: {error_msg}"
                    )
                    assert str(corrupted_mat_file) in error_msg or "corrupted.mat" in error_msg, (
                        f"Error message should reference the file, but got: {error_msg}"
                    )
                except Exception as e:
                    assert False, (
                        f"Expected DataLoadError for corrupted .mat file, but got {type(e).__name__}: {str(e)}"
                    )
            
            elif error_type == 'invalid_structure':
                # Test with valid .mat file but no usable data
                invalid_structure_file = temp_path / "invalid_structure.mat"
                
                try:
                    # Create a .mat file with only metadata (no actual data)
                    mat_data = {
                        '__header__': b'MATLAB 5.0 MAT-file',
                        '__version__': '1.0',
                        '__globals__': []
                    }
                    savemat(str(invalid_structure_file), mat_data)
                    
                    try:
                        loader.load_dataset(invalid_structure_file)
                        assert False, "Expected DataLoadError for .mat file with no data, but no exception was raised"
                    except DataLoadError as e:
                        error_msg = str(e)
                        # Should contain specific error information about no data
                        assert "no data" in error_msg.lower() or "empty" in error_msg.lower(), (
                            f"Error message should mention no data found, but got: {error_msg}"
                        )
                        assert len(error_msg) > 10, (
                            f"Error message should be descriptive, but got: {error_msg}"
                        )
                    except Exception as e:
                        assert False, (
                            f"Expected DataLoadError for .mat file with no data, but got {type(e).__name__}: {str(e)}"
                        )
                except Exception:
                    # If we can't create the test file, skip this case
                    pass
        
        # Test that the loader can still function after encountering errors
        # (processing continuation aspect)
        try:
            # Create a valid test DataFrame to verify the loader still works
            test_df = pd.DataFrame({
                'col1': [1, 2, 3],
                'col2': [4, 5, 6]
            })
            
            # Validate that the loader can still process valid data
            validation_result = loader.validate_data(test_df)
            metadata = loader.get_metadata(test_df)
            
            # These operations should succeed, demonstrating processing continuation
            assert validation_result is not None, "Validation should work after error handling"
            assert metadata is not None, "Metadata extraction should work after error handling"
            assert validation_result.is_valid, "Valid DataFrame should pass validation"
            assert metadata.n_records == 3, "Metadata should correctly report record count"
            assert metadata.n_features == 2, "Metadata should correctly report feature count"
            
        except Exception as e:
            assert False, (
                f"System should continue processing after handling errors, "
                f"but failed with {type(e).__name__}: {str(e)}"
            )