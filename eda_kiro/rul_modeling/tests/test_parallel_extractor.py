"""
Unit tests for parallel feature extraction.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.data_preparation.parallel_extractor import (
    ParallelFeatureExtractor,
    extract_es12_features
)


class TestParallelFeatureExtractor:
    """Test suite for ParallelFeatureExtractor."""
    
    @pytest.fixture
    def es12_path(self):
        """Path to ES12 data file."""
        return "../data/raw/ES12.mat"
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_initialization(self, es12_path):
        """Test ParallelFeatureExtractor initialization."""
        extractor = ParallelFeatureExtractor(
            es12_path=es12_path,
            n_processes=2,
            include_history=False
        )
        
        assert extractor.es12_path == es12_path
        assert extractor.n_processes == 2
        assert extractor.include_history is False
    
    def test_initialization_with_invalid_path(self):
        """Test initialization with non-existent file."""
        with pytest.raises(FileNotFoundError):
            ParallelFeatureExtractor(
                es12_path="nonexistent.mat",
                n_processes=2
            )
    
    def test_extract_single_capacitor(self, es12_path):
        """Test feature extraction from a single capacitor."""
        extractor = ParallelFeatureExtractor(
            es12_path=es12_path,
            n_processes=1,
            include_history=False
        )
        
        # Extract features from first 10 cycles of ES12C1
        features_df = extractor.extract_capacitor_features(
            capacitor_id="ES12C1",
            max_cycles=10,
            progress_interval=5
        )
        
        # Verify shape
        assert len(features_df) == 10, "Should have 10 cycles"
        assert len(features_df.columns) >= 26, "Should have at least 26 features"
        
        # Verify metadata columns
        assert 'capacitor_id' in features_df.columns
        assert 'cycle' in features_df.columns
        assert all(features_df['capacitor_id'] == 'ES12C1')
        assert list(features_df['cycle']) == list(range(1, 11))
        
        # Verify feature columns exist
        expected_features = [
            'vl_mean', 'vl_std', 'vo_mean', 'vo_std',
            'voltage_ratio', 'response_efficiency',
            'vl_trend', 'vo_trend', 'vl_cv', 'vo_cv',
            'cycle_number', 'cycle_normalized'
        ]
        for feature in expected_features:
            assert feature in features_df.columns, f"Missing feature: {feature}"
        
        # Verify no NaN values
        assert not features_df.isnull().any().any(), "Should not have NaN values"
    
    def test_parallel_extraction_multiple_capacitors(self, es12_path):
        """Test parallel extraction from multiple capacitors."""
        extractor = ParallelFeatureExtractor(
            es12_path=es12_path,
            n_processes=2,
            include_history=False
        )
        
        # Extract from 2 capacitors, 10 cycles each
        features_df = extractor.extract_all_capacitors(
            capacitor_ids=["ES12C1", "ES12C2"],
            max_cycles=10,
            progress_interval=5
        )
        
        # Verify shape
        assert len(features_df) == 20, "Should have 20 total cycles (2 caps × 10 cycles)"
        
        # Verify both capacitors are present
        cap_ids = features_df['capacitor_id'].unique()
        assert len(cap_ids) == 2
        assert 'ES12C1' in cap_ids
        assert 'ES12C2' in cap_ids
        
        # Verify each capacitor has 10 cycles
        for cap_id in ['ES12C1', 'ES12C2']:
            cap_data = features_df[features_df['capacitor_id'] == cap_id]
            assert len(cap_data) == 10
            assert list(cap_data['cycle']) == list(range(1, 11))
    
    def test_progress_reporting(self, es12_path, capsys):
        """Test that progress is reported correctly."""
        extractor = ParallelFeatureExtractor(
            es12_path=es12_path,
            n_processes=1,
            include_history=False
        )
        
        # Extract with progress interval of 5
        extractor.extract_capacitor_features(
            capacitor_id="ES12C1",
            max_cycles=10,
            progress_interval=5
        )
        
        # Capture output
        captured = capsys.readouterr()
        
        # Verify progress messages
        assert "Starting feature extraction" in captured.out
        assert "Cycle 5/10" in captured.out
        assert "50.0%" in captured.out
        assert "Cycle 10/10" in captured.out
        assert "100.0%" in captured.out
        assert "Completed" in captured.out
    
    def test_save_features(self, es12_path, temp_output_dir):
        """Test saving features to CSV."""
        extractor = ParallelFeatureExtractor(
            es12_path=es12_path,
            n_processes=1,
            include_history=False
        )
        
        # Extract features
        features_df = extractor.extract_capacitor_features(
            capacitor_id="ES12C1",
            max_cycles=5,
            progress_interval=5
        )
        
        # Save to CSV
        output_path = Path(temp_output_dir) / "test_features.csv"
        extractor.save_features(features_df, str(output_path))
        
        # Verify file exists
        assert output_path.exists()
        
        # Load and verify
        loaded_df = pd.read_csv(output_path)
        assert len(loaded_df) == 5
        
        # Verify metadata columns are first (save_features reorders columns)
        assert loaded_df.columns[0] == 'capacitor_id'
        assert loaded_df.columns[1] == 'cycle'
        
        # Verify all columns are present (order may differ due to sorting)
        assert set(loaded_df.columns) == set(features_df.columns)
    
    def test_convenience_function(self, es12_path, temp_output_dir):
        """Test the convenience function extract_es12_features."""
        output_path = Path(temp_output_dir) / "features.csv"
        
        features_df = extract_es12_features(
            es12_path=es12_path,
            output_path=str(output_path),
            capacitor_ids=["ES12C1"],
            max_cycles=5,
            n_processes=1,
            include_history=False,
            progress_interval=5
        )
        
        # Verify DataFrame
        assert len(features_df) == 5
        assert 'capacitor_id' in features_df.columns
        
        # Verify file was saved
        assert output_path.exists()
    
    def test_feature_values_are_numeric(self, es12_path):
        """Test that all feature values are numeric."""
        extractor = ParallelFeatureExtractor(
            es12_path=es12_path,
            n_processes=1,
            include_history=False
        )
        
        features_df = extractor.extract_capacitor_features(
            capacitor_id="ES12C1",
            max_cycles=5,
            progress_interval=5
        )
        
        # Exclude metadata columns
        feature_cols = [col for col in features_df.columns 
                       if col not in ['capacitor_id', 'cycle']]
        
        # Verify all feature columns are numeric
        for col in feature_cols:
            assert pd.api.types.is_numeric_dtype(features_df[col]), \
                f"Column {col} should be numeric"
    
    def test_cycle_normalized_values(self, es12_path):
        """Test that cycle_normalized is correctly calculated."""
        extractor = ParallelFeatureExtractor(
            es12_path=es12_path,
            n_processes=1,
            include_history=False
        )
        
        features_df = extractor.extract_capacitor_features(
            capacitor_id="ES12C1",
            max_cycles=10,
            progress_interval=5
        )
        
        # Verify cycle_normalized = cycle / 10
        expected_normalized = features_df['cycle'] / 10
        np.testing.assert_array_almost_equal(
            features_df['cycle_normalized'],
            expected_normalized
        )
    
    def test_multiprocessing_utilization(self, es12_path):
        """Test that multiprocessing is properly configured."""
        # Test with different process counts
        for n_proc in [1, 2, 4]:
            extractor = ParallelFeatureExtractor(
                es12_path=es12_path,
                n_processes=n_proc,
                include_history=False
            )
            assert extractor.n_processes == n_proc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
