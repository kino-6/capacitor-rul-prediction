"""Unit tests for Primary Model."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.models.primary_model import PrimaryModel


@pytest.fixture
def sample_data():
    """Create sample classification data for testing."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        random_state=42
    )
    
    feature_names = [f"feature_{i}" for i in range(10)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name="is_abnormal")
    
    # Split into train/val/test
    X_train = X_df[:120]
    y_train = y_series[:120]
    X_val = X_df[120:160]
    y_val = y_series[120:160]
    X_test = X_df[160:]
    y_test = y_series[160:]
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def test_primary_model_initialization():
    """Test PrimaryModel initialization."""
    model = PrimaryModel()
    assert model.model_type == "random_forest"
    assert model.model is None
    assert model.feature_names is None


def test_primary_model_train(sample_data):
    """Test PrimaryModel training."""
    X_train, y_train, X_val, y_val, _, _ = sample_data
    
    model = PrimaryModel()
    metrics = model.train(X_train, y_train, X_val, y_val)
    
    assert model.model is not None
    assert model.feature_names == list(X_train.columns)
    assert "train" in metrics
    assert "val" in metrics
    assert "accuracy" in metrics["train"]
    assert "f1_score" in metrics["train"]
    assert "roc_auc" in metrics["train"]


def test_primary_model_predict(sample_data):
    """Test PrimaryModel prediction."""
    X_train, y_train, _, _, X_test, _ = sample_data
    
    model = PrimaryModel()
    model.train(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    assert len(predictions) == len(X_test)
    assert set(predictions).issubset({0, 1})


def test_primary_model_predict_proba(sample_data):
    """Test PrimaryModel probability prediction."""
    X_train, y_train, _, _, X_test, _ = sample_data
    
    model = PrimaryModel()
    model.train(X_train, y_train)
    
    probabilities = model.predict_proba(X_test)
    
    assert probabilities.shape == (len(X_test), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_primary_model_evaluate(sample_data):
    """Test PrimaryModel evaluation."""
    X_train, y_train, _, _, X_test, y_test = sample_data
    
    model = PrimaryModel()
    model.train(X_train, y_train)
    
    metrics = model.evaluate(X_test, y_test)
    
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert "confusion_matrix" in metrics
    assert "classification_report" in metrics


def test_primary_model_feature_importance(sample_data):
    """Test feature importance extraction."""
    X_train, y_train, _, _, _, _ = sample_data
    
    model = PrimaryModel()
    model.train(X_train, y_train)
    
    importance_df = model.get_feature_importance()
    
    assert len(importance_df) == len(X_train.columns)
    assert "feature" in importance_df.columns
    assert "importance" in importance_df.columns
    assert importance_df["importance"].sum() > 0


def test_primary_model_save_load(sample_data):
    """Test model saving and loading."""
    X_train, y_train, _, _, X_test, _ = sample_data
    
    model = PrimaryModel()
    model.train(X_train, y_train)
    
    predictions_before = model.predict(X_test)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pkl"
        model.save(model_path)
        
        # Load model
        loaded_model = PrimaryModel()
        loaded_model.load(model_path)
        
        predictions_after = loaded_model.predict(X_test)
        
        assert np.array_equal(predictions_before, predictions_after)
        assert loaded_model.model_type == model.model_type
        assert loaded_model.feature_names == model.feature_names


def test_primary_model_predict_without_training():
    """Test that prediction fails without training."""
    model = PrimaryModel()
    X_dummy = pd.DataFrame(np.random.rand(10, 5))
    
    with pytest.raises(ValueError, match="Model has not been trained yet"):
        model.predict(X_dummy)


def test_primary_model_save_without_training():
    """Test that saving fails without training."""
    model = PrimaryModel()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "test_model.pkl"
        
        with pytest.raises(ValueError, match="Model has not been trained yet"):
            model.save(model_path)


def test_primary_model_custom_hyperparameters(sample_data):
    """Test training with custom hyperparameters."""
    X_train, y_train, _, _, _, _ = sample_data
    
    model = PrimaryModel()
    metrics = model.train(
        X_train,
        y_train,
        n_estimators=50,
        max_depth=5
    )
    
    assert model.model.n_estimators == 50
    assert model.model.max_depth == 5
    assert "train" in metrics
