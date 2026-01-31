"""
Unit tests for HybridEnsembleRULPredictor

Tests the hybrid ensemble model that combines XGBoost, LightGBM, and Random Forest
for robust RUL predictions with confidence intervals.
"""

import pytest
import numpy as np
from sklearn.datasets import make_regression

from true_rul.hybrid_ensemble_predictor import HybridEnsembleRULPredictor


@pytest.fixture
def synthetic_data():
    """Generate synthetic regression data for testing"""
    X, y = make_regression(
        n_samples=200,
        n_features=20,
        n_informative=15,
        noise=10.0,
        random_state=42
    )
    # Ensure non-negative RUL values
    y = np.abs(y)
    y = (y - y.min()) / (y.max() - y.min()) * 200  # Scale to 0-200 range
    
    # Split into train/val/test
    train_size = 120
    val_size = 40
    
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'feature_names': feature_names
    }


class TestHybridEnsembleRULPredictorInitialization:
    """Test initialization and configuration"""
    
    def test_default_initialization(self):
        """Test initialization with default parameters"""
        predictor = HybridEnsembleRULPredictor()
        
        assert predictor.is_trained is False
        assert predictor.feature_names is None
        assert len(predictor.models) == 3
        assert 'xgboost' in predictor.models
        assert 'lightgbm' in predictor.models
        assert 'random_forest' in predictor.models
        
        # Check default weights
        assert predictor.weights['xgboost'] == 0.4
        assert predictor.weights['lightgbm'] == 0.4
        assert predictor.weights['random_forest'] == 0.2
        assert np.isclose(sum(predictor.weights.values()), 1.0)
    
    def test_custom_weights(self):
        """Test initialization with custom weights"""
        custom_weights = {
            'xgboost': 0.5,
            'lightgbm': 0.3,
            'random_forest': 0.2
        }
        predictor = HybridEnsembleRULPredictor(weights=custom_weights)
        
        assert predictor.weights == custom_weights
        assert np.isclose(sum(predictor.weights.values()), 1.0)
    
    def test_invalid_weights_sum(self):
        """Test that invalid weights raise ValueError"""
        invalid_weights = {
            'xgboost': 0.5,
            'lightgbm': 0.3,
            'random_forest': 0.3  # Sum = 1.1
        }
        
        with pytest.raises(ValueError, match="must sum to 1.0"):
            HybridEnsembleRULPredictor(weights=invalid_weights)
    
    def test_custom_model_parameters(self):
        """Test initialization with custom model parameters"""
        xgboost_params = {'n_estimators': 100, 'max_depth': 4}
        lightgbm_params = {'n_estimators': 150, 'max_depth': 5}
        rf_params = {'n_estimators': 200, 'max_depth': 10}
        
        predictor = HybridEnsembleRULPredictor(
            xgboost_params=xgboost_params,
            lightgbm_params=lightgbm_params,
            random_forest_params=rf_params
        )
        
        assert predictor.models['xgboost'].n_estimators == 100
        assert predictor.models['lightgbm'].n_estimators == 150
        assert predictor.models['random_forest'].n_estimators == 200


class TestHybridEnsembleRULPredictorTraining:
    """Test training functionality"""
    
    def test_train_basic(self, synthetic_data):
        """Test basic training without validation data"""
        predictor = HybridEnsembleRULPredictor(
            xgboost_params={'n_estimators': 50},
            lightgbm_params={'n_estimators': 50},
            random_forest_params={'n_estimators': 50}
        )
        
        predictor.train(
            X_train=synthetic_data['X_train'],
            y_train=synthetic_data['y_train'],
            feature_names=synthetic_data['feature_names']
        )
        
        assert predictor.is_trained is True
        assert predictor.feature_names == synthetic_data['feature_names']
        
        # Check that all models are trained
        for model in predictor.models.values():
            assert model.is_trained is True
    
    def test_train_with_validation(self, synthetic_data):
        """Test training with validation data for early stopping"""
        predictor = HybridEnsembleRULPredictor(
            xgboost_params={'n_estimators': 100},
            lightgbm_params={'n_estimators': 100},
            random_forest_params={'n_estimators': 50}
        )
        
        predictor.train(
            X_train=synthetic_data['X_train'],
            y_train=synthetic_data['y_train'],
            X_val=synthetic_data['X_val'],
            y_val=synthetic_data['y_val'],
            feature_names=synthetic_data['feature_names'],
            early_stopping_rounds=10
        )
        
        assert predictor.is_trained is True
    
    def test_train_invalid_shapes(self, synthetic_data):
        """Test that training with mismatched shapes raises ValueError"""
        predictor = HybridEnsembleRULPredictor()
        
        # Mismatched X_train and y_train
        with pytest.raises(ValueError, match="same number of samples"):
            predictor.train(
                X_train=synthetic_data['X_train'],
                y_train=synthetic_data['y_train'][:50]  # Wrong size
            )
        
        # Mismatched X_val and y_val
        with pytest.raises(ValueError, match="same number of samples"):
            predictor.train(
                X_train=synthetic_data['X_train'],
                y_train=synthetic_data['y_train'],
                X_val=synthetic_data['X_val'],
                y_val=synthetic_data['y_val'][:20]  # Wrong size
            )
    
    def test_train_invalid_feature_names(self, synthetic_data):
        """Test that wrong number of feature names raises ValueError"""
        predictor = HybridEnsembleRULPredictor()
        
        with pytest.raises(ValueError, match="must match"):
            predictor.train(
                X_train=synthetic_data['X_train'],
                y_train=synthetic_data['y_train'],
                feature_names=['f1', 'f2']  # Wrong number
            )


class TestHybridEnsembleRULPredictorPrediction:
    """Test prediction functionality"""
    
    @pytest.fixture
    def trained_predictor(self, synthetic_data):
        """Create a trained predictor for testing"""
        predictor = HybridEnsembleRULPredictor(
            xgboost_params={'n_estimators': 50},
            lightgbm_params={'n_estimators': 50},
            random_forest_params={'n_estimators': 50}
        )
        
        predictor.train(
            X_train=synthetic_data['X_train'],
            y_train=synthetic_data['y_train'],
            X_val=synthetic_data['X_val'],
            y_val=synthetic_data['y_val'],
            feature_names=synthetic_data['feature_names']
        )
        
        return predictor
    
    def test_predict_basic(self, trained_predictor, synthetic_data):
        """Test basic prediction"""
        predictions = trained_predictor.predict(synthetic_data['X_test'])
        
        assert predictions.shape == (synthetic_data['X_test'].shape[0],)
        assert np.all(predictions >= 0), "All predictions should be non-negative"
        assert np.all(np.isfinite(predictions)), "All predictions should be finite"
    
    def test_predict_untrained(self, synthetic_data):
        """Test that prediction without training raises RuntimeError"""
        predictor = HybridEnsembleRULPredictor()
        
        with pytest.raises(RuntimeError, match="not been trained"):
            predictor.predict(synthetic_data['X_test'])
    
    def test_predict_invalid_shape(self, trained_predictor, synthetic_data):
        """Test that prediction with wrong shape raises ValueError"""
        X_wrong = synthetic_data['X_test'][:, :10]  # Wrong number of features
        
        with pytest.raises(ValueError, match="must match"):
            trained_predictor.predict(X_wrong)
    
    def test_predict_with_confidence(self, trained_predictor, synthetic_data):
        """Test prediction with confidence intervals"""
        pred, lower, upper = trained_predictor.predict_with_confidence(
            synthetic_data['X_test']
        )
        
        assert pred.shape == (synthetic_data['X_test'].shape[0],)
        assert lower.shape == pred.shape
        assert upper.shape == pred.shape
        
        # Check confidence interval properties
        assert np.all(lower >= 0), "Lower bounds should be non-negative"
        assert np.all(upper >= 0), "Upper bounds should be non-negative"
        assert np.all(lower <= pred), "Lower bounds should be <= predictions"
        assert np.all(upper >= pred), "Upper bounds should be >= predictions"
        assert np.all(np.isfinite(lower)), "Lower bounds should be finite"
        assert np.all(np.isfinite(upper)), "Upper bounds should be finite"
    
    def test_predict_with_confidence_custom_level(self, trained_predictor, synthetic_data):
        """Test prediction with custom confidence level"""
        pred_90, lower_90, upper_90 = trained_predictor.predict_with_confidence(
            synthetic_data['X_test'],
            confidence_level=0.90
        )
        
        pred_99, lower_99, upper_99 = trained_predictor.predict_with_confidence(
            synthetic_data['X_test'],
            confidence_level=0.99
        )
        
        # 99% CI should be wider than 90% CI
        width_90 = upper_90 - lower_90
        width_99 = upper_99 - lower_99
        assert np.all(width_99 >= width_90), "99% CI should be wider than 90% CI"
    
    def test_predict_with_confidence_invalid_level(self, trained_predictor, synthetic_data):
        """Test that invalid confidence level raises ValueError"""
        with pytest.raises(ValueError, match="must be between 0 and 1"):
            trained_predictor.predict_with_confidence(
                synthetic_data['X_test'],
                confidence_level=1.5
            )
    
    def test_get_individual_predictions(self, trained_predictor, synthetic_data):
        """Test getting predictions from individual models"""
        individual_preds = trained_predictor.get_individual_predictions(
            synthetic_data['X_test']
        )
        
        assert len(individual_preds) == 3
        assert 'xgboost' in individual_preds
        assert 'lightgbm' in individual_preds
        assert 'random_forest' in individual_preds
        
        for name, pred in individual_preds.items():
            assert pred is not None
            assert pred.shape == (synthetic_data['X_test'].shape[0],)
            assert np.all(pred >= 0)
    
    def test_get_prediction_variance(self, trained_predictor, synthetic_data):
        """Test getting prediction variance"""
        variance = trained_predictor.get_prediction_variance(synthetic_data['X_test'])
        
        assert variance.shape == (synthetic_data['X_test'].shape[0],)
        assert np.all(variance >= 0), "Variance should be non-negative"
        assert np.all(np.isfinite(variance)), "Variance should be finite"


class TestHybridEnsembleRULPredictorInterpretability:
    """Test interpretability features"""
    
    @pytest.fixture
    def trained_predictor(self, synthetic_data):
        """Create a trained predictor for testing"""
        predictor = HybridEnsembleRULPredictor(
            xgboost_params={'n_estimators': 50},
            lightgbm_params={'n_estimators': 50},
            random_forest_params={'n_estimators': 50}
        )
        
        predictor.train(
            X_train=synthetic_data['X_train'],
            y_train=synthetic_data['y_train'],
            feature_names=synthetic_data['feature_names']
        )
        
        return predictor
    
    def test_get_aggregated_feature_importance(self, trained_predictor, synthetic_data):
        """Test aggregated feature importance"""
        importance = trained_predictor.get_aggregated_feature_importance()
        
        assert isinstance(importance, dict)
        assert len(importance) == len(synthetic_data['feature_names'])
        
        # Check that all feature names are present
        for name in synthetic_data['feature_names']:
            assert name in importance
        
        # Check that importance values are non-negative
        for value in importance.values():
            assert value >= 0
        
        # Check that importance values sum to approximately 1.0 (normalized)
        total = sum(importance.values())
        assert np.isclose(total, 1.0, atol=1e-6)
    
    def test_get_aggregated_feature_importance_unnormalized(self, trained_predictor):
        """Test aggregated feature importance without normalization"""
        importance = trained_predictor.get_aggregated_feature_importance(normalize=False)
        
        assert isinstance(importance, dict)
        
        # Without normalization, sum may not be 1.0
        total = sum(importance.values())
        assert total > 0
    
    def test_get_aggregated_feature_importance_untrained(self, synthetic_data):
        """Test that feature importance without training raises RuntimeError"""
        predictor = HybridEnsembleRULPredictor()
        
        with pytest.raises(RuntimeError, match="not been trained"):
            predictor.get_aggregated_feature_importance()


class TestHybridEnsembleRULPredictorModelInfo:
    """Test model information and metadata"""
    
    def test_get_model_info_untrained(self):
        """Test getting model info before training"""
        predictor = HybridEnsembleRULPredictor()
        info = predictor.get_model_info()
        
        assert info['model_type'] == 'hybrid_ensemble'
        assert info['is_trained'] is False
        assert info['n_features'] == 0
        assert info['feature_names'] is None
        assert 'weights' in info
        assert 'models' in info
    
    def test_get_model_info_trained(self, synthetic_data):
        """Test getting model info after training"""
        predictor = HybridEnsembleRULPredictor(
            xgboost_params={'n_estimators': 50},
            lightgbm_params={'n_estimators': 50},
            random_forest_params={'n_estimators': 50}
        )
        
        predictor.train(
            X_train=synthetic_data['X_train'],
            y_train=synthetic_data['y_train'],
            feature_names=synthetic_data['feature_names']
        )
        
        info = predictor.get_model_info()
        
        assert info['is_trained'] is True
        assert info['n_features'] == len(synthetic_data['feature_names'])
        assert info['feature_names'] == synthetic_data['feature_names']
        
        # Check that info for each model is present
        assert 'xgboost' in info['models']
        assert 'lightgbm' in info['models']
        assert 'random_forest' in info['models']


class TestHybridEnsembleRULPredictorSaveLoad:
    """Test model persistence"""
    
    def test_save_load_model(self, synthetic_data, tmp_path):
        """Test saving and loading the ensemble model"""
        # Train a model
        predictor = HybridEnsembleRULPredictor(
            xgboost_params={'n_estimators': 50},
            lightgbm_params={'n_estimators': 50},
            random_forest_params={'n_estimators': 50}
        )
        
        predictor.train(
            X_train=synthetic_data['X_train'],
            y_train=synthetic_data['y_train'],
            feature_names=synthetic_data['feature_names']
        )
        
        # Get predictions before saving
        pred_before = predictor.predict(synthetic_data['X_test'])
        
        # Save model
        filepath_prefix = str(tmp_path / "test_ensemble")
        predictor.save_model(filepath_prefix)
        
        # Create new predictor and load
        predictor_loaded = HybridEnsembleRULPredictor()
        predictor_loaded.load_model(filepath_prefix)
        
        # Check that loaded model is trained
        assert predictor_loaded.is_trained is True
        assert predictor_loaded.feature_names == synthetic_data['feature_names']
        
        # Get predictions after loading
        pred_after = predictor_loaded.predict(synthetic_data['X_test'])
        
        # Predictions should be identical
        np.testing.assert_array_almost_equal(pred_before, pred_after, decimal=5)
    
    def test_save_untrained_model(self, tmp_path):
        """Test that saving untrained model raises RuntimeError"""
        predictor = HybridEnsembleRULPredictor()
        filepath_prefix = str(tmp_path / "test_ensemble")
        
        with pytest.raises(RuntimeError, match="not been trained"):
            predictor.save_model(filepath_prefix)


class TestHybridEnsembleRULPredictorEdgeCases:
    """Test edge cases and error handling"""
    
    def test_single_sample_prediction(self, synthetic_data):
        """Test prediction on a single sample"""
        predictor = HybridEnsembleRULPredictor(
            xgboost_params={'n_estimators': 50},
            lightgbm_params={'n_estimators': 50},
            random_forest_params={'n_estimators': 50}
        )
        
        predictor.train(
            X_train=synthetic_data['X_train'],
            y_train=synthetic_data['y_train'],
            feature_names=synthetic_data['feature_names']
        )
        
        # Predict on single sample
        X_single = synthetic_data['X_test'][:1]
        pred = predictor.predict(X_single)
        
        assert pred.shape == (1,)
        assert pred[0] >= 0
    
    def test_ensemble_weights_effect(self, synthetic_data):
        """Test that different weights produce different predictions"""
        # Train with default weights
        predictor1 = HybridEnsembleRULPredictor(
            xgboost_params={'n_estimators': 50},
            lightgbm_params={'n_estimators': 50},
            random_forest_params={'n_estimators': 50}
        )
        predictor1.train(
            X_train=synthetic_data['X_train'],
            y_train=synthetic_data['y_train']
        )
        
        # Train with different weights
        predictor2 = HybridEnsembleRULPredictor(
            xgboost_params={'n_estimators': 50},
            lightgbm_params={'n_estimators': 50},
            random_forest_params={'n_estimators': 50},
            weights={'xgboost': 0.7, 'lightgbm': 0.2, 'random_forest': 0.1}
        )
        predictor2.train(
            X_train=synthetic_data['X_train'],
            y_train=synthetic_data['y_train']
        )
        
        # Predictions should be different
        pred1 = predictor1.predict(synthetic_data['X_test'])
        pred2 = predictor2.predict(synthetic_data['X_test'])
        
        # At least some predictions should differ
        assert not np.allclose(pred1, pred2, atol=1e-6)
