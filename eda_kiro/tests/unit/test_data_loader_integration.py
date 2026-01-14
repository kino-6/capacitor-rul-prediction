"""Integration tests for DataLoader with realistic scenarios."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from nasa_pcoe_eda.data import DataLoader
from nasa_pcoe_eda.exceptions import DataLoadError


class TestDataLoaderIntegration:
    """Integration tests for DataLoader."""

    @pytest.fixture
    def loader(self) -> DataLoader:
        """Create a DataLoader instance."""
        return DataLoader()

    @pytest.fixture
    def realistic_mat_file(self, tmp_path: Path) -> Path:
        """Create a realistic .mat file similar to NASA PCOE data."""
        mat_path = tmp_path / "ES12_sample.mat"
        
        # Simulate NASA PCOE capacitor data structure
        # Typically has time series measurements
        n_samples = 100
        data = {
            "ES12": {
                "time": np.linspace(0, 1000, n_samples),
                "voltage": np.random.randn(n_samples) * 10 + 100,
                "current": np.random.randn(n_samples) * 0.5 + 2,
                "temperature": np.random.randn(n_samples) * 5 + 25,
                "capacitance": np.linspace(100, 80, n_samples) + np.random.randn(n_samples) * 2,
            }
        }
        savemat(str(mat_path), data)
        
        return mat_path

    def test_complete_workflow(
        self, loader: DataLoader, realistic_mat_file: Path
    ) -> None:
        """Test complete workflow: load, validate, get metadata."""
        # Load dataset
        df = loader.load_dataset(realistic_mat_file)
        assert df is not None
        assert not df.empty
        
        # Validate data
        validation = loader.validate_data(df)
        assert validation.is_valid
        
        # Get metadata
        metadata = loader.get_metadata(df)
        assert metadata.n_records > 0
        assert metadata.n_features > 0
        assert metadata.memory_usage > 0
        
        # Verify data is stored
        assert loader.data is not None
        assert loader.data.equals(df)

    def test_error_handling_corrupted_file(
        self, loader: DataLoader, tmp_path: Path
    ) -> None:
        """Test error handling with corrupted file."""
        corrupted_file = tmp_path / "corrupted.mat"
        corrupted_file.write_bytes(b"This is not a valid MATLAB file")
        
        with pytest.raises(DataLoadError):
            loader.load_dataset(corrupted_file)

    def test_error_handling_missing_file(self, loader: DataLoader) -> None:
        """Test error handling with missing file."""
        missing_file = Path("/nonexistent/path/to/file.mat")
        
        with pytest.raises(DataLoadError, match="Data file not found"):
            loader.load_dataset(missing_file)

    def test_data_persistence_requirement(
        self, loader: DataLoader, realistic_mat_file: Path
    ) -> None:
        """Test that data persists in memory after loading (Requirement 2.5)."""
        df = loader.load_dataset(realistic_mat_file)
        
        # Data should be accessible through the data property
        assert loader.data is not None
        
        # Should be the same DataFrame
        assert loader.data.equals(df)
        
        # Should be available for subsequent analysis
        metadata = loader.get_metadata(loader.data)
        assert metadata.n_records == len(df)

    def test_record_and_feature_reporting(
        self, loader: DataLoader, realistic_mat_file: Path
    ) -> None:
        """Test that loader reports correct record and feature counts (Requirement 2.3)."""
        df = loader.load_dataset(realistic_mat_file)
        metadata = loader.get_metadata(df)
        
        # Verify reported counts match actual data
        assert metadata.n_records == len(df)
        assert metadata.n_features == len(df.columns)
        assert len(metadata.feature_names) == len(df.columns)
