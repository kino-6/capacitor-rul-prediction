"""
Data loading module for NASA PCOE datasets.

This module provides robust data loading capabilities for NASA PCOE datasets,
with special support for MATLAB .mat files including both v7.0 and v7.3 formats.
It includes comprehensive validation, error handling, and metadata extraction.

Key features:
- Support for MATLAB .mat files (both scipy.io and h5py backends)
- Automatic format detection and appropriate loader selection
- Data validation and integrity checking
- Metadata extraction (dimensions, types, memory usage)
- Robust error handling with detailed error messages
- Cross-platform path handling

Example usage:
    loader = DataLoader()
    df = loader.load_dataset(Path("data/ES12.mat"))
    metadata = loader.get_metadata(df)
    validation = loader.validate_data(df)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.io import loadmat

from ..exceptions import DataLoadError, DataValidationError
from ..models import DatasetMetadata, ValidationResult
from .es12_loader import ES12DataLoader


class DataLoader:
    """Loads and validates NASA PCOE dataset files."""

    def __init__(self) -> None:
        """Initialize the DataLoader."""
        self._loaded_data: Optional[pd.DataFrame] = None

    def load_dataset(self, path: Path) -> pd.DataFrame:
        """
        Load a dataset from a MATLAB .mat file.

        Args:
            path: Path to the .mat file

        Returns:
            DataFrame containing the loaded data

        Raises:
            DataLoadError: If the file cannot be loaded or is invalid
        """
        # Convert to Path object if string
        if isinstance(path, str):
            path = Path(path)

        # Check if file exists
        if not path.exists():
            raise DataLoadError(f"Data file not found: {path}")

        # Check if it's a file
        if not path.is_file():
            raise DataLoadError(f"Path is not a file: {path}")

        # Check file extension
        if path.suffix.lower() != ".mat":
            raise DataLoadError(
                f"Invalid file format: {path.suffix}. Expected .mat file"
            )

        # Check if this is an ES12 dataset and use specialized loader
        if 'ES12' in path.name or 'es12' in path.name.lower():
            try:
                es12_loader = ES12DataLoader()
                df = es12_loader.load_dataset(path)
                self._loaded_data = df
                return df
            except Exception as e:
                # Fall back to generic loader if ES12 loader fails
                print(f"ES12 loader failed, falling back to generic loader: {e}")

        try:
            # Try loading with scipy first (for older MATLAB files)
            try:
                mat_data = loadmat(str(path))
                data_keys = [k for k in mat_data.keys() if not k.startswith("__")]
                
                if not data_keys:
                    raise DataLoadError("No data found in MATLAB file")
                
                df = self._extract_dataframe(mat_data, data_keys)
                
            except NotImplementedError:
                # MATLAB v7.3 files need h5py
                import h5py
                
                with h5py.File(str(path), 'r') as f:
                    # Get all dataset keys
                    data_keys = [k for k in f.keys()]
                    
                    if not data_keys:
                        raise DataLoadError("No data found in MATLAB file")
                    
                    df = self._extract_dataframe_from_hdf5(f, data_keys)

            # Store the loaded data
            self._loaded_data = df

            return df

        except Exception as e:
            if isinstance(e, DataLoadError):
                raise
            raise DataLoadError(f"Failed to load data from {path}: {str(e)}")

    def _extract_dataframe_from_hdf5(self, h5file, data_keys: list) -> pd.DataFrame:
        """
        Extract a DataFrame from HDF5/MATLAB v7.3 file.

        Args:
            h5file: h5py File object
            data_keys: List of dataset keys

        Returns:
            DataFrame with the extracted data
        """
        import h5py
        
        # Try to extract data from the HDF5 structure
        data_dict = {}
        
        def extract_datasets(name, obj):
            """Recursively extract datasets"""
            if isinstance(obj, h5py.Dataset):
                try:
                    data = obj[()]
                    # Skip very small datasets (likely metadata)
                    if data.size < 2:
                        return
                    
                    # Handle different data shapes
                    if data.ndim == 1:
                        data_dict[name] = data
                    elif data.ndim == 2:
                        # If it's a column vector, flatten it
                        if data.shape[1] == 1:
                            data_dict[name] = data.flatten()
                        elif data.shape[0] == 1:
                            data_dict[name] = data.flatten()
                        else:
                            # Multiple columns - add each as separate feature
                            for i in range(data.shape[1]):
                                data_dict[f"{name}_{i}"] = data[:, i]
                except Exception:
                    pass
        
        # Visit all items in the file
        h5file.visititems(extract_datasets)
        
        if not data_dict:
            raise DataLoadError("Could not extract data from HDF5 file")
        
        # Find the most common length
        lengths = [len(v) for v in data_dict.values()]
        if not lengths:
            raise DataLoadError("No valid data arrays found")
        
        # Use the most common length
        from collections import Counter
        most_common_length = Counter(lengths).most_common(1)[0][0]
        
        # Filter to only include arrays with the most common length
        filtered_dict = {
            k: v for k, v in data_dict.items() 
            if len(v) == most_common_length
        }
        
        if not filtered_dict:
            raise DataLoadError("Could not find arrays with consistent length")
        
        # Create DataFrame
        df = pd.DataFrame(filtered_dict)
        
        return df

    def _extract_dataframe(
        self, mat_data: dict, data_keys: list
    ) -> pd.DataFrame:
        """
        Extract a DataFrame from MATLAB data structure.

        Args:
            mat_data: Dictionary from loadmat
            data_keys: List of data keys (non-metadata)

        Returns:
            DataFrame with the extracted data
        """
        # Try to find the main data structure
        # NASA PCOE datasets typically have a main structure
        if len(data_keys) == 1:
            main_key = data_keys[0]
            data = mat_data[main_key]
        else:
            # If multiple keys, try to find the largest array
            main_key = max(
                data_keys,
                key=lambda k: (
                    mat_data[k].size if isinstance(mat_data[k], np.ndarray) else 0
                ),
            )
            data = mat_data[main_key]

        # Handle different data structures
        if isinstance(data, np.ndarray):
            # Check if it's a structured array
            if data.dtype.names:
                # Structured array with named fields
                df_dict = {}
                for name in data.dtype.names:
                    field_data = data[name]
                    # Flatten if needed
                    if field_data.ndim > 1:
                        field_data = field_data.flatten()
                    df_dict[name] = field_data
                df = pd.DataFrame(df_dict)
            elif data.ndim == 2:
                # Regular 2D array
                df = pd.DataFrame(data)
            elif data.ndim == 1:
                # 1D array - make it a single column
                df = pd.DataFrame(data, columns=["value"])
            else:
                # Try to reshape to 2D
                try:
                    reshaped = data.reshape(-1, data.shape[-1])
                    df = pd.DataFrame(reshaped)
                except Exception:
                    raise DataLoadError(
                        f"Cannot convert data structure to DataFrame: shape={data.shape}"
                    )
        else:
            raise DataLoadError(
                f"Unsupported data type in MATLAB file: {type(data)}"
            )

        return df

    def validate_data(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate the integrity of loaded data.

        Args:
            df: DataFrame to validate

        Returns:
            ValidationResult with validation status and messages

        Raises:
            DataValidationError: If validation encounters critical errors
        """
        errors = []
        warnings = []

        try:
            # Check if DataFrame is empty
            if df.empty:
                errors.append("DataFrame is empty")

            # Check if DataFrame has no columns
            if len(df.columns) == 0:
                errors.append("DataFrame has no columns")

            # Check if DataFrame has no rows
            if len(df) == 0:
                errors.append("DataFrame has no rows")

            # Check for all-null columns
            all_null_cols = df.columns[df.isnull().all()].tolist()
            if all_null_cols:
                warnings.append(
                    f"Columns with all null values: {', '.join(map(str, all_null_cols))}"
                )

            # Check for duplicate column names
            if df.columns.duplicated().any():
                dup_cols = df.columns[df.columns.duplicated()].tolist()
                errors.append(
                    f"Duplicate column names found: {', '.join(map(str, dup_cols))}"
                )

            # Check for extremely high missing value percentage
            missing_pct = (df.isnull().sum() / len(df) * 100).max()
            if missing_pct > 90:
                warnings.append(
                    f"Some columns have >90% missing values (max: {missing_pct:.1f}%)"
                )

            # Check data types - warn if all columns are object type
            # Only check if there are no duplicate columns
            if not df.columns.duplicated().any():
                if all(df[col].dtype == object for col in df.columns):
                    warnings.append("All columns are object type - may need type conversion")

            # Determine if valid
            is_valid = len(errors) == 0

            return ValidationResult(
                is_valid=is_valid, errors=errors, warnings=warnings
            )

        except Exception as e:
            raise DataValidationError(f"Validation failed: {str(e)}")

    def get_metadata(self, df: pd.DataFrame) -> DatasetMetadata:
        """
        Extract metadata from the dataset.

        Args:
            df: DataFrame to extract metadata from

        Returns:
            DatasetMetadata with dataset information
        """
        # Get basic dimensions
        n_records = len(df)
        n_features = len(df.columns)
        feature_names = df.columns.tolist()

        # Get data types
        data_types = {str(col): str(df[col].dtype) for col in df.columns}

        # Calculate memory usage in MB
        memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)

        # Try to identify date range if temporal columns exist
        date_range = None
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    min_date = df[col].min()
                    max_date = df[col].max()
                    if pd.notna(min_date) and pd.notna(max_date):
                        date_range = (min_date.to_pydatetime(), max_date.to_pydatetime())
                        break
                except Exception:
                    continue

        return DatasetMetadata(
            n_records=n_records,
            n_features=n_features,
            feature_names=feature_names,
            data_types=data_types,
            memory_usage=memory_usage,
            date_range=date_range,
        )

    @property
    def data(self) -> Optional[pd.DataFrame]:
        """
        Get the currently loaded data.

        Returns:
            The loaded DataFrame, or None if no data is loaded
        """
        return self._loaded_data
