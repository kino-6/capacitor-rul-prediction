"""Unit tests for Secondary Model."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression

from src.models.secondary_model import SecondaryModel


@pytest.fixture
def sample_data():
    """Create sample regression data for testing."""
    X, y = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=8,
        noise=10.0,
        random_state=42
    )
    
    # Make y positive (RUL values)
    y = np.abs(y)
    
    feature_names = [f"feature_{i}" for i in range(10)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="rul")
    
    # Split into train/val/test
    X_train = X_df[:120]
    y_train = y_series[:120]
    X_val = X_df[120:160]
    y_val = y_series[120:160]
    X_test = X_df[160:]
    y_test = y_series[160:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def test_secondary_model_initialization():
    """Test SecondaryModel initialization."""
    model = SecondaryModel()
    assert model.model_type == "random_forest"
    assert model.model is None
    assert model.feature_names is None


def test_secondary_model_train(sample_data):
    """Test SecondaryModel training."""
    X_train, y_train, X_val, y_val, _, _ = sample_data
    
    model = SecondaryModel()
    metrics = model.train(X_train, y_train, X_val, y_val)
    
    assert model.model is not None
    assert model.feature_names == list(X_train.columns)
    assert "train" in metrics
    assert "val" in metrics
    assert "mae" in metrics["train"]
    assert "rmse" in metrics["train"]
    assert "r2" in metrics["train"]
    assert "mape" in metrics["train"]


def test_secondary_model_predict(sample_data):
    """Test SecondaryModel prediction."""
    X_train, y_train, _, _, X_test, _ = sample_data
    
    model = SecondaryModel()
    model.train(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    assert len(predictions) == len(X_test)
    assert all(predictions >= 0)  # RUL should be non-negative


def test_secondary_model_evaluate(sample_data):
    """Test SecondaryModel evaluation."""
    X_train, y_train, _, _, X_test, y_test = sample_data
    
    model = SecondaryModel()
    model.train(X_train, y_train)
    
    metrics = model.evaluate(X_test, y_test)
    
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert "mape" in metrics
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0


def test_secondary_model_feature_importance(sample_data):
    """Test feature importance extraction."""
    X_train, y_train, _, _, _, _ = sample_data
    
    model = SecondaryModel()
    model.train(X_train, y_train)
    
    importance_df = model.get_feature_importance()
    
    assert len(importance_df) == len(X_train.columns)
    assert "feature" in importance_df.columns
    assert "importance" in importance_df.columns
    assert importance_df["importance"].sum() > 0


def test_secondary_model_save_load(sample_data):
    """Test model saving and loading."""
    X_train, y_train, _, _, X_test, _ = sample_data
    
    model = SecondaryModel()
    model.train(X_train, y_train)
    
    predictions_before = model.predict(X_test)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pkl"
        model.save(model_path)
        
        # Load model
        loaded_model = SecondaryModel()
        loaded_model.load(model_path)
        
        predictions_after = loaded_model.predict(X_test)
        
        assert np.allclose(predictions_before, predictions_after)
        assert loaded_model.model_type == model.model_type
        assert loaded_model.feature_names == model.feature_names


def test_secondary_model_predict_without_training():
    """Test that prediction fails without training."""
    model = SecondaryModel()
    X_dummy = pd.DataFrame(np.random.rand(10, 5))
    
    with pytest.raises(ValueError, match="Model has not been trained yet"):
        model.predict(X_dummy)


def test_secondary_model_save_without_training():
    """Test that saving fails without training."""
    model = SecondaryModel()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pkl"
        
        with pytest.raises(ValueError, match="Model has not been trained yet"):
            model.save(model_path)


def test_secondary_model_custom_hyperparameters(sample_data):
    """Test training with custom hyperparameters."""
    X_train, y_train, _, _, _, _ = sample_data
    
    model = SecondaryModel()
    metrics = model.train(
        X_train,
        y_train,
        n_estimators=50,
        max_depth=5
    )
    
    assert model.model.n_estimators == 50
    assert model.model.max_depth == 5
    assert "train" in metrics


def test_mape_calculation():
    """Test MAPE calculation."""
    y_true = np.array([100, 200, 300])
    y_pred = np.array([110, 190, 310])
    
    mape = SecondaryModel._calculate_mape(y_true, y_pred)
    
    expected_mape = np.mean([10/100, 10/200, 10/300]) * 100
    assert np.isclose(mape, expected_mape)


def test_mape_with_zero_values():
    """Test MAPE calculation with zero values."""
    y_true = np.array([0, 100, 200])
    y_pred = np.array([10, 110, 190])
    
    mape = SecondaryModel._calculate_mape(y_true, y_pred)
    
    # Should only calculate MAPE for non-zero values
    expected_mape = np.mean([10/100, 10/200]) * 100
    assert np.isclose(mape, expected_mape)
