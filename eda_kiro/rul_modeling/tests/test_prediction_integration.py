"""
Integration tests for prediction aggregation and confidence estimation
"""

import pytest
import numpy as np
from unittest.mock import Mock

from true_rul.prediction_aggregator import PredictionAggregator
from true_rul.confidence_estimator import ConfidenceEstimator
from true_rul.data_structures import PredictionResult


class MockRULModel:
    """Mock RUL regression model"""
    
    def __init__(self, base_prediction=50.0, variance=5.0):
        self.base_prediction = base_prediction
        self.variance = variance
    
    def predict(self, x):
        """Mock predict method with some variance"""
        if len(x.shape) > 1:
            n_samples = x.shape[0]
        else:
            n_samples = 1
        
        predictions = np.random.normal(
            self.base_prediction, 
            self.variance, 
            n_samples
        )
        return predictions if n_samples > 1 else predictions[0]


class MockAnomalyDetector:
    """Mock anomaly detection model"""
    
    def __init__(self, anomaly_score=0.3, anomaly_flag=False):
        self.anomaly_score = anomaly_score
        self.anomaly_flag = anomaly_flag
    
    def predict(self, x):
        """Mock predict method"""
        return {
            'anomaly_flag': self.anomaly_flag,
            'anomaly_score': self.anomaly_score,
            'feature_importance': {
                'feature_1': 0.25,
                'feature_2': 0.20,
                'feature_3': 0.15,
                'feature_4': 0.10,
                'feature_5': 0.30
            }
        }


class TestPredictionIntegration:
    """Integration tests for prediction pipeline"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.aggregator = PredictionAggregator(model_version="integration-test-1.0")
        self.confidence_estimator = ConfidenceEstimator(method="ensemble")
        
        # Create mock models
        self.rul_models = [
            MockRULModel(base_prediction=48.0, variance=3.0),
            MockRULModel(base_prediction=50.0, variance=2.0),
            MockRULModel(base_prediction=52.0, variance=4.0)
        ]
        
        self.anomaly_detector = MockAnomalyDetector(
            anomaly_score=0.25, 
            anomaly_flag=False
        )
        
        self.test_input = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    def test_end_to_end_prediction_pipeline(self):
        """Test complete prediction pipeline"""
        # Step 1: Get RUL predictions from ensemble
        rul_predictions = []
        for model in self.rul_models:
            pred = model.predict(self.test_input)
            rul_predictions.append(pred)
        
        # Step 2: Estimate confidence intervals
        rul_mean = np.mean(rul_predictions)
        rul_lower, rul_upper = self.confidence_estimator.estimate_confidence_ensemble(
            rul_predictions
        )
        
        # Step 3: Get anomaly detection results
        anomaly_result = self.anomaly_detector.predict(self.test_input)
        
        # Step 4: Aggregate all predictions
        final_result = self.aggregator.aggregate(
            rul_pred=rul_mean,
            rul_confidence_lower=rul_lower,
            rul_confidence_upper=rul_upper,
            anomaly_flag=anomaly_result['anomaly_flag'],
            anomaly_score=anomaly_result['anomaly_score'],
            feature_importance=anomaly_result['feature_importance'],
            capacitor_id="TEST_C1",
            cycle_number=25
        )
        
        # Verify the final result
        assert isinstance(final_result, PredictionResult)
        assert final_result.capacitor_id == "TEST_C1"
        assert final_result.cycle_number == 25
        assert final_result.model_version == "integration-test-1.0"
        
        # Verify RUL predictions are reasonable
        assert 40 <= final_result.rul_cycles <= 60  # Should be around 50
        assert final_result.rul_confidence_lower <= final_result.rul_cycles
        assert final_result.rul_cycles <= final_result.rul_confidence_upper
        
        # Verify anomaly detection results
        assert final_result.anomaly_flag is False
        assert 0.0 <= final_result.anomaly_score <= 1.0
        
        # Verify degradation stage is computed
        assert final_result.degradation_stage in [
            "healthy", "early_degradation", "advanced_degradation", "critical"
        ]
        
        # Verify feature importance is preserved
        assert len(final_result.feature_importance) == 5
        assert sum(final_result.feature_importance.values()) == 1.0
    
    def test_high_degradation_scenario(self):
        """Test prediction pipeline with high degradation scenario"""
        # Create models for high degradation scenario
        high_degradation_rul_models = [
            MockRULModel(base_prediction=8.0, variance=2.0),
            MockRULModel(base_prediction=10.0, variance=1.5),
            MockRULModel(base_prediction=12.0, variance=3.0)
        ]
        
        high_degradation_anomaly = MockAnomalyDetector(
            anomaly_score=0.85,
            anomaly_flag=True
        )
        
        # Get predictions
        rul_predictions = [model.predict(self.test_input) for model in high_degradation_rul_models]
        rul_mean = np.mean(rul_predictions)
        rul_lower, rul_upper = self.confidence_estimator.estimate_confidence_ensemble(rul_predictions)
        
        anomaly_result = high_degradation_anomaly.predict(self.test_input)
        
        # Aggregate predictions
        result = self.aggregator.aggregate(
            rul_pred=rul_mean,
            rul_confidence_lower=rul_lower,
            rul_confidence_upper=rul_upper,
            anomaly_flag=anomaly_result['anomaly_flag'],
            anomaly_score=anomaly_result['anomaly_score'],
            feature_importance=anomaly_result['feature_importance'],
            capacitor_id="HIGH_DEG_C1",
            cycle_number=180
        )
        
        # Verify high degradation is detected
        assert result.anomaly_flag is True
        assert result.anomaly_score > 0.8
        assert result.degradation_stage in ["advanced_degradation", "critical"]
        assert result.rul_cycles < 20  # Low RUL
    
    def test_healthy_scenario(self):
        """Test prediction pipeline with healthy scenario"""
        # Create models for healthy scenario
        healthy_rul_models = [
            MockRULModel(base_prediction=180.0, variance=10.0),
            MockRULModel(base_prediction=175.0, variance=8.0),
            MockRULModel(base_prediction=185.0, variance=12.0)
        ]
        
        healthy_anomaly = MockAnomalyDetector(
            anomaly_score=0.05,
            anomaly_flag=False
        )
        
        # Get predictions
        rul_predictions = [model.predict(self.test_input) for model in healthy_rul_models]
        rul_mean = np.mean(rul_predictions)
        rul_lower, rul_upper = self.confidence_estimator.estimate_confidence_ensemble(rul_predictions)
        
        anomaly_result = healthy_anomaly.predict(self.test_input)
        
        # Aggregate predictions
        result = self.aggregator.aggregate(
            rul_pred=rul_mean,
            rul_confidence_lower=rul_lower,
            rul_confidence_upper=rul_upper,
            anomaly_flag=anomaly_result['anomaly_flag'],
            anomaly_score=anomaly_result['anomaly_score'],
            feature_importance=anomaly_result['feature_importance'],
            capacitor_id="HEALTHY_C1",
            cycle_number=15
        )
        
        # Verify healthy state is detected
        assert result.anomaly_flag is False
        assert result.anomaly_score < 0.1
        assert result.degradation_stage == "healthy"
        assert result.rul_cycles > 150  # High RUL
    
    def test_confidence_interval_consistency(self):
        """Test that confidence intervals are consistent across scenarios"""
        # Test with different variance levels
        low_variance_models = [
            MockRULModel(base_prediction=50.0, variance=1.0),
            MockRULModel(base_prediction=50.0, variance=1.0),
            MockRULModel(base_prediction=50.0, variance=1.0)
        ]
        
        high_variance_models = [
            MockRULModel(base_prediction=30.0, variance=5.0),
            MockRULModel(base_prediction=50.0, variance=5.0),
            MockRULModel(base_prediction=70.0, variance=5.0)
        ]
        
        # Low variance scenario
        low_var_preds = [model.predict(self.test_input) for model in low_variance_models]
        low_var_lower, low_var_upper = self.confidence_estimator.estimate_confidence_ensemble(low_var_preds)
        
        # High variance scenario
        high_var_preds = [model.predict(self.test_input) for model in high_variance_models]
        high_var_lower, high_var_upper = self.confidence_estimator.estimate_confidence_ensemble(high_var_preds)
        
        # High variance should have wider confidence intervals
        low_var_width = low_var_upper - low_var_lower
        high_var_width = high_var_upper - high_var_lower
        
        assert high_var_width > low_var_width
    
    def test_degradation_history_impact(self):
        """Test impact of degradation history on predictions"""
        # Scenario with increasing degradation trend
        increasing_history = [0.1, 0.15, 0.25, 0.35, 0.45]
        
        # Scenario with stable degradation
        stable_history = [0.2, 0.21, 0.19, 0.20, 0.22]
        
        base_rul = 75.0
        base_anomaly_score = 0.3
        
        # Test with increasing trend
        result_increasing = self.aggregator.aggregate(
            rul_pred=base_rul,
            rul_confidence_lower=base_rul - 5,
            rul_confidence_upper=base_rul + 5,
            anomaly_flag=False,
            anomaly_score=base_anomaly_score,
            feature_importance={'feature_1': 1.0},
            degradation_history=increasing_history
        )
        
        # Test with stable trend
        result_stable = self.aggregator.aggregate(
            rul_pred=base_rul,
            rul_confidence_lower=base_rul - 5,
            rul_confidence_upper=base_rul + 5,
            anomaly_flag=False,
            anomaly_score=base_anomaly_score,
            feature_importance={'feature_1': 1.0},
            degradation_history=stable_history
        )
        
        # Increasing trend should result in higher degradation score
        assert result_increasing.degradation_score >= result_stable.degradation_score
    
    def test_result_serialization(self):
        """Test that prediction results can be serialized"""
        # Create a sample result
        result = self.aggregator.aggregate(
            rul_pred=100.0,
            rul_confidence_lower=95.0,
            rul_confidence_upper=105.0,
            anomaly_flag=False,
            anomaly_score=0.2,
            feature_importance={'feature_1': 0.5, 'feature_2': 0.5},
            capacitor_id="SERIAL_TEST_C1",
            cycle_number=50
        )
        
        # Test dictionary conversion
        result_dict = result.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict['capacitor_id'] == "SERIAL_TEST_C1"
        assert result_dict['cycle_number'] == 50
        assert result_dict['rul_cycles'] == 100
        
        # Test JSON conversion
        result_json = result.to_json()
        assert isinstance(result_json, str)
        assert "SERIAL_TEST_C1" in result_json
        assert "100" in result_json