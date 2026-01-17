"""Tests for DatasetSplitter class."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from src.data_preparation.dataset_splitter import DatasetSplitter, split_dataset


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    data = []
    for cap_num in range(1, 9):
        cap_id = f'ES12C{cap_num}'
        for cycle in range(1, 201):
            data.append({
                'capacitor_id': cap_id,
                'cycle': cycle,
                'cycle_normalized': cycle / 200,
                'vl_mean': 1.0 + np.random.randn() * 0.1,
                'vo_mean': 0.8 + np.random.randn() * 0.1,
                'voltage_ratio': 0.8 + np.random.randn() * 0.05,
                'is_abnormal': 1 if cycle > 100 else 0,
                'rul': 200 - cycle
            })
    return pd.DataFrame(data)


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


class TestDatasetSplitter:
    """Test suite for DatasetSplitter class."""
    
    def test_initialization_default(self):
        """Test default initialization."""
        splitter = DatasetSplitter()
        assert splitter.train_capacitors == ['ES12C1', 'ES12C2', 'ES12C3', 'ES12C4', 'ES12C5']
        assert splitter.val_capacitors == ['ES12C6']
        assert splitter.test_capacitors == ['ES12C7', 'ES12C8']
        assert splitter.train_cycle_range == (1, 150)
        assert splitter.val_cycle_range == (1, 150)
        assert splitter.test_cycle_range == (1, 200)
    
    def test_split_basic(self, sample_dataset):
        """Test basic splitting functionality."""
        splitter = DatasetSplitter()
        train_df, val_df, test_df = splitter.split(sample_dataset)
        assert len(train_df) == 750
        assert len(val_df) == 150
        assert len(test_df) == 400
    
    def test_split_capacitor_assignment(self, sample_dataset):
        """Test that capacitors are correctly assigned to splits."""
        splitter = DatasetSplitter()
        train_df, val_df, test_df = splitter.split(sample_dataset)
        train_caps = set(train_df['capacitor_id'].unique())
        val_caps = set(val_df['capacitor_id'].unique())
        test_caps = set(test_df['capacitor_id'].unique())
        assert train_caps == {'ES12C1', 'ES12C2', 'ES12C3', 'ES12C4', 'ES12C5'}
        assert val_caps == {'ES12C6'}
        assert test_caps == {'ES12C7', 'ES12C8'}
    
    def test_split_cycle_ranges(self, sample_dataset):
        """Test that cycle ranges are correctly applied."""
        splitter = DatasetSplitter()
        train_df, val_df, test_df = splitter.split(sample_dataset)
        assert train_df['cycle'].min() == 1
        assert train_df['cycle'].max() == 150
        assert val_df['cycle'].min() == 1
        assert val_df['cycle'].max() == 150
        assert test_df['cycle'].min() == 1
        assert test_df['cycle'].max() == 200
    
    def test_split_no_overlap(self, sample_dataset):
        """Test that there is no data overlap between splits."""
        splitter = DatasetSplitter()
        train_df, val_df, test_df = splitter.split(sample_dataset)
        train_ids = set(train_df['capacitor_id'] + '_' + train_df['cycle'].astype(str))
        val_ids = set(val_df['capacitor_id'] + '_' + val_df['cycle'].astype(str))
        test_ids = set(test_df['capacitor_id'] + '_' + test_df['cycle'].astype(str))
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0
    
    def test_get_split_statistics(self, sample_dataset):
        """Test statistics generation."""
        splitter = DatasetSplitter()
        train_df, val_df, test_df = splitter.split(sample_dataset)
        stats = splitter.get_split_statistics(train_df, val_df, test_df)
        assert stats['train']['n_samples'] == 750
        assert stats['train']['n_capacitors'] == 5
        assert stats['val']['n_samples'] == 150
        assert stats['test']['n_samples'] == 400
    
    def test_save_splits(self, sample_dataset, temp_output_dir):
        """Test saving splits to files."""
        splitter = DatasetSplitter()
        train_df, val_df, test_df = splitter.split(sample_dataset)
        paths = splitter.save_splits(train_df, val_df, test_df, temp_output_dir)
        assert Path(paths['train']).exists()
        assert Path(paths['val']).exists()
        assert Path(paths['test']).exists()
        loaded_train = pd.read_csv(paths['train'])
        assert len(loaded_train) == len(train_df)
