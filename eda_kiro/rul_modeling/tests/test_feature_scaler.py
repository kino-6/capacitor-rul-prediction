"""
Tests for FeatureScaler class.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import pickle

from src.data_preparation.feature_scaler import FeatureScaler, scale_and_save_datasets


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    
    # Create sample features with different scales
    data = {
        'capacitor_id': ['C1'] * 50 + ['C2'] * 50,
        'cycle': list(range(1, 51)) * 2,
        'feature1': np.random.normal(100, 20, 100),
        'feature2': np.random.normal(0.5, 0.1, 100),
        'feature3': np.random.normal(1000, 200, 100),
        'is_abnormal': [0] * 60 + [1] * 40,
        'rul': list(range(50, 0, -1)) * 2
    }
    
    df = pd.DataFrame(data)
    
    # Split into train/val/test
    train_df = df.iloc[:60]
    val_df = df.iloc[60:80]
    test_df = df.iloc[80:]
    
    return train_df, val_df, test_df


def test_feature_scaler_initialization():
    """Test FeatureScaler initialization."""
    scaler = FeatureScaler()
    
    assert scaler.scaler is not None
    assert scaler.feature_columns is None
    assert 'capacitor_id' in scaler.metadata_columns
    assert 'cycle' in scaler.metadata_columns
    assert 'is_abnormal' in scaler.metadata_columns
    assert 'rul' in scaler.metadata_columns


def test_feature_scaler_fit(sample_data):
    """Test fitting the scaler."""
    train_df, _, _ = sample_data
    
    scaler = FeatureScaler()
    scaler.fit(train_df)
    
    # Check that feature columns were identified
    assert scaler.feature_columns is not None
    assert len(scaler.feature_columns) == 3  # feature1, feature2, feature3
    assert 'feature1' in scaler.feature_columns
    assert 'feature2' in scaler.feature_columns
    assert 'feature3' in scaler.feature_columns
    
    # Check that metadata columns are excluded
    assert 'capacitor_id' not in scaler.feature_columns
    assert 'cycle' not in scaler.feature_columns
    assert 'is_abnormal' not in scaler.feature_columns
    assert 'rul' not in scaler.feature_columns


def test_feature_scaler_transform(sample_data):
    """Test transforming data."""
    train_df, val_df, _ = sample_data
    
    scaler = FeatureScaler()
    scaler.fit(train_df)
    
    # Transform validation data
    val_scaled = scaler.transform(val_df)
    
    # Check shape is preserved
    assert val_scaled.shape == val_df.shape
    
    # Check that metadata columns are unchanged
    pd.testing.assert_series_equal(val_scaled['capacitor_id'], val_df['capacitor_id'])
    pd.testing.assert_series_equal(val_scaled['cycle'], val_df['cycle'])
    pd.testing.assert_series_equal(val_scaled['is_abnormal'], val_df['is_abnormal'])
    pd.testing.assert_series_equal(val_scaled['rul'], val_df['rul'])
    
    # Check that feature columns are scaled
    assert not np.allclose(val_scaled['feature1'].values, val_df['feature1'].values)
    assert not np.allclose(val_scaled['feature2'].values, val_df['feature2'].values)
    assert not np.allclose(val_scaled['feature3'].values, val_df['feature3'].values)


def test_feature_scaler_fit_transform(sample_data):
    """Test fit_transform method."""
    train_df, _, _ = sample_data
    
    scaler = FeatureScaler()
    train_scaled = scaler.fit_transform(train_df)
    
    # Check shape is preserved
    assert train_scaled.shape == train_df.shape
    
    # Check that training features have mean ≈ 0 and std ≈ 1
    feature_cols = ['feature1', 'feature2', 'feature3']
    for col in feature_cols:
        assert np.abs(train_scaled[col].mean()) < 1e-10
        assert np.abs(train_scaled[col].std() - 1.0) < 0.01


def test_feature_scaler_transform_before_fit(sample_data):
    """Test that transform raises error if called before fit."""
    _, val_df, _ = sample_data
    
    scaler = FeatureScaler()
    
    with pytest.raises(ValueError, match="Scaler must be fitted before transform"):
        scaler.transform(val_df)


def test_feature_scaler_save_load(sample_data):
    """Test saving and loading the scaler."""
    train_df, val_df, _ = sample_data
    
    scaler = FeatureScaler()
    scaler.fit(train_df)
    
    # Save scaler
    with tempfile.TemporaryDirectory() as tmpdir:
        scaler_path = Path(tmpdir) / "scaler.pkl"
        scaler.save(str(scaler_path))
        
        # Check file exists
        assert scaler_path.exists()
        
        # Load scaler
        new_scaler = FeatureScaler()
        new_scaler.feature_columns = scaler.feature_columns  # Need to set this manually
        new_scaler.load(str(scaler_path))
        
        # Transform with both scalers and compare
        val_scaled1 = scaler.transform(val_df)
        val_scaled2 = new_scaler.transform(val_df)
        
        pd.testing.assert_frame_equal(val_scaled1, val_scaled2)


def test_feature_scaler_get_feature_stats(sample_data):
    """Test getting feature statistics."""
    train_df, _, _ = sample_data
    
    scaler = FeatureScaler()
    scaler.fit(train_df)
    
    stats_df = scaler.get_feature_stats()
    
    # Check structure
    assert 'feature' in stats_df.columns
    assert 'mean' in stats_df.columns
    assert 'std' in stats_df.columns
    
    # Check number of features
    assert len(stats_df) == 3
    
    # Check that means and stds are reasonable
    assert all(stats_df['std'] > 0)


def test_scale_and_save_datasets(sample_data):
    """Test the complete scaling pipeline."""
    train_df, val_df, test_df = sample_data
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Save input datasets
        train_path = tmpdir / "train.csv"
        val_path = tmpdir / "val.csv"
        test_path = tmpdir / "test.csv"
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        # Run scaling pipeline
        output_dir = tmpdir / "output"
        scaler_path = tmpdir / "scaler.pkl"
        
        train_scaled, val_scaled, test_scaled = scale_and_save_datasets(
            train_path=str(train_path),
            val_path=str(val_path),
            test_path=str(test_path),
            output_dir=str(output_dir),
            scaler_path=str(scaler_path)
        )
        
        # Check that files were created
        assert (output_dir / "train_scaled.csv").exists()
        assert (output_dir / "val_scaled.csv").exists()
        assert (output_dir / "test_scaled.csv").exists()
        assert scaler_path.exists()
        
        # Check that returned dataframes have correct shapes
        assert train_scaled.shape == train_df.shape
        assert val_scaled.shape == val_df.shape
        assert test_scaled.shape == test_df.shape
        
        # Check that training features are standardized
        feature_cols = ['feature1', 'feature2', 'feature3']
        for col in feature_cols:
            assert np.abs(train_scaled[col].mean()) < 1e-10
            assert np.abs(train_scaled[col].std() - 1.0) < 0.01


def test_metadata_preservation(sample_data):
    """Test that metadata columns are never modified."""
    train_df, val_df, test_df = sample_data
    
    scaler = FeatureScaler()
    train_scaled = scaler.fit_transform(train_df)
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)
    
    # Check all metadata columns
    metadata_cols = ['capacitor_id', 'cycle', 'is_abnormal', 'rul']
    
    for col in metadata_cols:
        pd.testing.assert_series_equal(train_scaled[col], train_df[col], check_names=True)
        pd.testing.assert_series_equal(val_scaled[col], val_df[col], check_names=True)
        pd.testing.assert_series_equal(test_scaled[col], test_df[col], check_names=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
