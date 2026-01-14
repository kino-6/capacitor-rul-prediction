"""Unit tests for DataLoader class."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

from nasa_pcoe_eda.data import DataLoader
from nasa_pcoe_eda.exceptions import DataLoadError, DataValidationError
from nasa_pcoe_eda.models import DatasetMetadata, ValidationResult


class TestDataLoader:
    """Test DataLoader class."""

    @pytest.fixture
    def loader(self) -> DataLoader:
        """Create a DataLoader instance."""
        return DataLoader()

    @pytest.fixture
    def sample_mat_file(self, tmp_path: Path) -> Path:
        """Create a sample .mat file for testing."""
        mat_path = tmp_path / "test_data.mat"
        
        # Create sample data
        data = {
            "test_data": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        }
        savemat(str(mat_path), data)
        
        return mat_path

    @pytest.fixture
    def structured_mat_file(self, tmp_path: Path) -> Path:
        """Create a structured array .mat file for testing."""
        mat_path = tmp_path / "structured_data.mat"
        
        # Create structured array
        dt = np.dtype([("feature1", "f8"), ("feature2", "f8"), ("feature3", "f8")])
        data = np.array(
            [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)], dtype=dt
        )
        savemat(str(mat_path), {"structured_data": data})
        
        return mat_path

    def test_load_dataset_success(
        self, loader: DataLoader, sample_mat_file: Path
    ) -> None:
        """Test successful loading of a .mat file."""
        df = loader.load_dataset(sample_mat_file)
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert df.shape[0] == 3  # 3 rows
        assert df.shape[1] == 3  # 3 columns

    def test_load_dataset_file_not_found(self, loader: DataLoader) -> None:
        """Test loading a non-existent file raises DataLoadError."""
        with pytest.raises(DataLoadError, match="Data file not found"):
            loader.load_dataset(Path("nonexistent.mat"))

    def test_load_dataset_invalid_extension(
        self, loader: DataLoader, tmp_path: Path
    ) -> None:
        """Test loading a file with wrong extension raises DataLoadError."""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("test data")
        
        with pytest.raises(DataLoadError, match="Invalid file format"):
            loader.load_dataset(txt_file)

    def test_load_dataset_directory_path(
        self, loader: DataLoader, tmp_path: Path
    ) -> None:
        """Test loading a directory path raises DataLoadError."""
        with pytest.raises(DataLoadError, match="Path is not a file"):
            loader.load_dataset(tmp_path)

    def test_load_dataset_stores_data(
        self, loader: DataLoader, sample_mat_file: Path
    ) -> None:
        """Test that loaded data is stored in the loader."""
        df = loader.load_dataset(sample_mat_file)
        
        assert loader.data is not None
        assert loader.data.equals(df)

    def test_load_dataset_structured_array(
        self, loader: DataLoader, structured_mat_file: Path
    ) -> None:
        """Test loading a structured array .mat file."""
        df = loader.load_dataset(structured_mat_file)
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "feature1" in df.columns or 0 in df.columns

    def test_validate_data_valid_dataframe(self, loader: DataLoader) -> None:
        """Test validation of a valid DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = loader.validate_data(df)
        
        assert isinstance(result, ValidationResult)
        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_data_empty_dataframe(self, loader: DataLoader) -> None:
        """Test validation of an empty DataFrame."""
        df = pd.DataFrame()
        result = loader.validate_data(df)
        
        assert result.is_valid is False
        assert any("empty" in error.lower() for error in result.errors)

    def test_validate_data_all_null_column(self, loader: DataLoader) -> None:
        """Test validation warns about all-null columns."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]})
        result = loader.validate_data(df)
        
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert any("null" in warning.lower() for warning in result.warnings)

    def test_validate_data_duplicate_columns(self, loader: DataLoader) -> None:
        """Test validation detects duplicate column names."""
        df = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "a"])
        result = loader.validate_data(df)
        
        assert result.is_valid is False
        assert any("duplicate" in error.lower() for error in result.errors)

    def test_validate_data_high_missing_percentage(self, loader: DataLoader) -> None:
        """Test validation warns about high missing value percentage."""
        df = pd.DataFrame({"a": [1] + [None] * 99})
        result = loader.validate_data(df)
        
        assert result.is_valid is True
        assert len(result.warnings) > 0
        assert any("90%" in warning or "missing" in warning.lower() for warning in result.warnings)

    def test_get_metadata_basic(self, loader: DataLoader) -> None:
        """Test metadata extraction from a DataFrame."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0], "c": ["x", "y", "z"]})
        metadata = loader.get_metadata(df)
        
        assert isinstance(metadata, DatasetMetadata)
        assert metadata.n_records == 3
        assert metadata.n_features == 3
        assert set(metadata.feature_names) == {"a", "b", "c"}
        assert len(metadata.data_types) == 3
        assert metadata.memory_usage > 0

    def test_get_metadata_with_datetime(self, loader: DataLoader) -> None:
        """Test metadata extraction with datetime column."""
        df = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "date": pd.to_datetime(["2020-01-01", "2020-06-01", "2020-12-31"]),
            }
        )
        metadata = loader.get_metadata(df)
        
        assert metadata.date_range is not None
        assert len(metadata.date_range) == 2

    def test_get_metadata_no_datetime(self, loader: DataLoader) -> None:
        """Test metadata extraction without datetime column."""
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        metadata = loader.get_metadata(df)
        
        assert metadata.date_range is None

    def test_data_property_before_loading(self, loader: DataLoader) -> None:
        """Test data property returns None before loading."""
        assert loader.data is None

    def test_load_dataset_with_string_path(
        self, loader: DataLoader, sample_mat_file: Path
    ) -> None:
        """Test loading dataset with string path."""
        df = loader.load_dataset(str(sample_mat_file))
        
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_load_dataset_empty_mat_file(
        self, loader: DataLoader, tmp_path: Path
    ) -> None:
        """Test loading an empty .mat file raises DataLoadError."""
        mat_path = tmp_path / "empty.mat"
        # Create a .mat file with only metadata
        savemat(str(mat_path), {})
        
        with pytest.raises(DataLoadError, match="No data found"):
            loader.load_dataset(mat_path)
