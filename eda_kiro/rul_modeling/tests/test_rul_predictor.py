"""
Tests for RULPredictor main class
"""

import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from true_rul.data_structures import CycleData, PredictionResult
from true_rul.exceptions import (
    InputValidationError, ModelNotReadyError, FeatureExtractionError, TimeoutError
)
from true_rul.rul_predictor import RULPredictor


class TestRULPredictor:
    """Test cases for RULPredictor class"""
    
    def test_init_default(self):
        """Test initialization with default parameters"""
        predictor = RULPredictor()
        
        assert predictor.rul_model is None
        assert predictor.anomaly_detector is None
        assert predictor.feature_extractor is not None
        assert predictor.prediction_aggregator is not None
        assert predictor.confidence_estimator is not None
        assert predictor.prediction_timeout == 1.0
        assert not predictor.is_ready
    
    def test_init_with_models(self):
        """Test initialization with mock models"""
        mock_rul_model = Mock()
        mock_rul_model.is_trained = True
        
        mock_anomaly_detector = Mock()
        mock_anomaly_detector.is_fitted = True
        
        predictor = RULPredictor(
            rul_model=mock_rul_model,
            anomaly_detector=mock_anomaly_detector,
            prediction_timeout=2.0
        )
        
        assert predictor.rul_model is mock_rul_model
        assert predictor.anomaly_detector is mock_anomaly_detector
        assert predictor.prediction_timeout == 2.0
        assert predictor.is_ready
    
    def test_validate_input_valid(self):
        """Test input validation with valid data"""
        predictor = RULPredictor()
        
        cycle_data = CycleData(
            cycle_number=1,
            vl_series=np.array([1.0, 2.0, 3.0]),
            vo_series=np.array([1.1, 2.1, 3.1])
        )
        
        # Should not raise any exception
        predictor._validate_input(cycle_data, "C1", None)
    
    def test_validate_input_invalid_cycle_data(self):
        """Test input validation with invalid cycle data"""
        predictor = RULPredictor()
        
        # Test with non-CycleData object
        with pytest.raises(InputValidationError) as exc_info:
            predictor._validate_input("not_cycle_data", "C1", None)
        
        assert "cycle_data must be a CycleData instance" in str(exc_info.value)
    
    def test_validate_input_empty_series(self):
        """Test input validation with empty voltage series"""
        predictor = RULPredictor()
        
        # CycleData constructor will raise ValueError for empty series
        with pytest.raises(ValueError) as exc_info:
            cycle_data = CycleData(
                cycle_number=1,
                vl_series=np.array([]),
                vo_series=np.array([])
            )
        
        assert "VL and VO series cannot be empty" in str(exc_info.value)
    
    def test_validate_input_nan_values(self):
        """Test input validation with NaN values"""
        predictor = RULPredictor()
        
        cycle_data = CycleData(
            cycle_number=1,
            vl_series=np.array([1.0, np.nan, 3.0]),
            vo_series=np.array([1.1, 2.1, 3.1])
        )
        
        with pytest.raises(InputValidationError) as exc_info:
            predictor._validate_input(cycle_data, "C1", None)
        
        assert "VL series contains NaN or infinite values" in str(exc_info.value)
    
    def test_validate_input_mismatched_lengths(self):
        """Test input validation with mismatched series lengths"""
        predictor = RULPredictor()
        
        # CycleData constructor will raise ValueError for mismatched lengths
        with pytest.raises(ValueError) as exc_info:
            cycle_data = CycleData(
                cycle_number=1,
                vl_series=np.array([1.0, 2.0]),
                vo_series=np.array([1.1, 2.1, 3.1])
            )
        
        assert "VL and VO series must have same length" in str(exc_info.value)
    
    def test_validate_input_invalid_capacitor_id(self):
        """Test input validation with invalid capacitor ID"""
        predictor = RULPredictor()
        
        cycle_data = CycleData(
            cycle_number=1,
            vl_series=np.array([1.0, 2.0, 3.0]),
            vo_series=np.array([1.1, 2.1, 3.1])
        )
        
        with pytest.raises(InputValidationError) as exc_info:
            predictor._validate_input(cycle_data, "", None)
        
        assert "capacitor_id must be a non-empty string" in str(exc_info.value)
    
    def test_predict_with_error_handling_model_not_ready(self):
        """Test prediction with models not ready"""
        predictor = RULPredictor()  # No models loaded
        
        cycle_data = CycleData(
            cycle_number=1,
            vl_series=np.array([1.0, 2.0, 3.0]),
            vo_series=np.array([1.1, 2.1, 3.1])
        )
        
        with pytest.raises(ModelNotReadyError) as exc_info:
            predictor.predict_with_error_handling(cycle_data, "C1")
        
        assert "Models are not ready for prediction" in str(exc_info.value)
    
    def test_extract_basic_features_fallback(self):
        """Test basic feature extraction fallback"""
        predictor = RULPredictor()
        
        cycle_data = CycleData(
            cycle_number=1,
            vl_series=np.array([1.0, 2.0, 3.0]),
            vo_series=np.array([1.1, 2.1, 3.1])
        )
        
        features = predictor._extract_basic_features_fallback(cycle_data)
        
        assert len(features) == 55  # Expected feature count
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
    
    def test_predict_rul_with_fallback_emergency(self):
        """Test RUL prediction emergency fallback"""
        predictor = RULPredictor()
        
        # Mock model that raises exception
        mock_model = Mock()
        mock_model.predict.side_effect = Exception("Model failed")
        predictor.rul_model = mock_model
        
        features = np.array([[1.0, 2.0, 3.0]])
        
        pred, lower, upper = predictor._predict_rul_with_fallback(features)
        
        # Should return conservative estimates
        assert pred == 50.0
        assert lower == 30.0
        assert upper == 70.0
    
    def test_detect_anomaly_with_fallback(self):
        """Test anomaly detection fallback"""
        predictor = RULPredictor()
        
        # Mock detector that raises exception
        mock_detector = Mock()
        mock_detector.predict.side_effect = Exception("Detector failed")
        predictor.anomaly_detector = mock_detector
        
        features = np.array([[1.0, 2.0, 3.0]])
        
        anomaly_flag, anomaly_score, feature_importance = predictor._detect_anomaly_with_fallback(features)
        
        assert isinstance(anomaly_flag, bool)
        assert 0 <= anomaly_score <= 1
        assert isinstance(feature_importance, dict)
    
    def test_update_prediction_history(self):
        """Test prediction history update"""
        predictor = RULPredictor()
        
        predictor._update_prediction_history("C1", 0.5)
        predictor._update_prediction_history("C1", 0.6)
        predictor._update_prediction_history("C2", 0.3)
        
        assert len(predictor.prediction_history["C1"]) == 2
        assert len(predictor.prediction_history["C2"]) == 1
        assert predictor.prediction_history["C1"] == [0.5, 0.6]
        assert predictor.prediction_history["C2"] == [0.3]
    
    def test_update_prediction_history_limit(self):
        """Test prediction history size limit"""
        predictor = RULPredictor()
        
        # Add more than 20 predictions
        for i in range(25):
            predictor._update_prediction_history("C1", float(i))
        
        # Should keep only last 20
        assert len(predictor.prediction_history["C1"]) == 20
        assert predictor.prediction_history["C1"][0] == 5.0  # First kept value
        assert predictor.prediction_history["C1"][-1] == 24.0  # Last value
    
    def test_get_model_status(self):
        """Test model status reporting"""
        mock_rul_model = Mock()
        mock_rul_model.is_trained = True
        mock_rul_model.model_type = "xgboost"
        
        mock_anomaly_detector = Mock()
        mock_anomaly_detector.is_fitted = True
        
        predictor = RULPredictor(
            rul_model=mock_rul_model,
            anomaly_detector=mock_anomaly_detector
        )
        
        status = predictor.get_model_status()
        
        assert status["is_ready"] is True
        assert status["rul_model"]["loaded"] is True
        assert status["rul_model"]["trained"] is True
        assert status["rul_model"]["type"] == "xgboost"
        assert status["anomaly_detector"]["loaded"] is True
        assert status["anomaly_detector"]["fitted"] is True
    
    def test_clear_prediction_history(self):
        """Test clearing prediction history"""
        predictor = RULPredictor()
        
        predictor._update_prediction_history("C1", 0.5)
        predictor._update_prediction_history("C2", 0.6)
        
        # Clear specific capacitor
        predictor.clear_prediction_history("C1")
        assert "C1" not in predictor.prediction_history
        assert "C2" in predictor.prediction_history
        
        # Clear all
        predictor.clear_prediction_history()
        assert len(predictor.prediction_history) == 0
    
    def test_set_prediction_timeout(self):
        """Test setting prediction timeout"""
        predictor = RULPredictor()
        
        predictor.set_prediction_timeout(2.5)
        assert predictor.prediction_timeout == 2.5
        
        with pytest.raises(ValueError):
            predictor.set_prediction_timeout(-1.0)
    
    def test_repr(self):
        """Test string representation"""
        predictor = RULPredictor()
        predictor._update_prediction_history("C1", 0.5)
        
        repr_str = repr(predictor)
        
        assert "RULPredictor" in repr_str
        assert "ready=False" in repr_str
        assert "timeout=1.0s" in repr_str
        assert "history_size=1" in repr_str


if __name__ == "__main__":
    pytest.main([__file__])