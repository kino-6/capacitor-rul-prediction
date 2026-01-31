"""
Tests for PredictionAggregator class
"""

import pytest
import numpy as np
from datetime import datetime

from true_rul.prediction_aggregator import PredictionAggregator
from true_rul.data_structures import PredictionResult


class TestPredictionAggregator:
    """Test cases for PredictionAggregator"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.aggregator = PredictionAggregator(model_version="test-1.0")
        
        # Sample feature importance
        self.feature_importance = {
            'feature_1': 0.3,
            'feature_2': 0.2,
            'feature_3': 0.15,
            'feature_4': 0.1,
            'feature_5': 0.25
        }
    
    def test_initialization(self):
        """Test aggregator initialization"""
        assert self.aggregator.model_version == "test-1.0"
        assert len(self.aggregator.degradation_thresholds) == 4
        assert 'healthy' in self.aggregator.degradation_thresholds
        assert 'critical' in self.aggregator.degradation_thresholds
    
    def test_aggregate_basic(self):
        """Test basic aggregation functionality"""
        result = self.aggregator.aggregate(
            rul_pred=50.0,
            rul_confidence_lower=45.0,
            rul_confidence_upper=55.0,
            anomaly_flag=False,
            anomaly_score=0.1,
            feature_importance=self.feature_importance,
            capacitor_id="C1",
            cycle_number=10
        )
        
        assert isinstance(result, PredictionResult)
        assert result.rul_cycles == 50
        assert result.rul_confidence_lower == 45
        assert result.rul_confidence_upper == 55
        assert result.anomaly_flag is False
        assert result.anomaly_score == 0.1
        assert result.capacitor_id == "C1"
        assert result.cycle_number == 10
        assert result.model_version == "test-1.0"
        assert isinstance(result.timestamp, datetime)
    
    def test_compute_degradation_stage_healthy(self):
        """Test degradation stage computation for healthy state"""
        stage = self.aggregator.compute_degradation_stage(
            rul=150.0,
            anomaly_score=0.05
        )
        assert stage == "healthy"
    
    def test_compute_degradation_stage_early(self):
        """Test degradation stage computation for early degradation"""
        stage = self.aggregator.compute_degradation_stage(
            rul=100.0,
            anomaly_score=0.3
        )
        assert stage == "early_degradation"
    
    def test_compute_degradation_stage_advanced(self):
        """Test degradation stage computation for advanced degradation"""
        stage = self.aggregator.compute_degradation_stage(
            rul=50.0,
            anomaly_score=0.6
        )
        assert stage == "advanced_degradation"
    
    def test_compute_degradation_stage_critical(self):
        """Test degradation stage computation for critical state"""
        stage = self.aggregator.compute_degradation_stage(
            rul=10.0,
            anomaly_score=0.9
        )
        assert stage == "critical"
    
    def test_degradation_score_computation(self):
        """Test degradation score computation"""
        # Test with low RUL and high anomaly score
        score = self.aggregator._compute_degradation_score(
            rul=20.0,
            anomaly_score=0.8
        )
        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should be high degradation
        
        # Test with high RUL and low anomaly score
        score = self.aggregator._compute_degradation_score(
            rul=180.0,
            anomaly_score=0.1
        )
        assert 0.0 <= score <= 1.0
        assert score < 0.3  # Should be low degradation
    
    def test_degradation_score_with_history(self):
        """Test degradation score computation with history"""
        history = [0.1, 0.15, 0.2, 0.3]  # Increasing trend
        
        score = self.aggregator._compute_degradation_score(
            rul=100.0,
            anomaly_score=0.4,
            degradation_history=history
        )
        
        assert 0.0 <= score <= 1.0
        # Score should be higher due to increasing trend
    
    def test_negative_rul_handling(self):
        """Test handling of negative RUL predictions"""
        result = self.aggregator.aggregate(
            rul_pred=-5.0,  # Negative RUL
            rul_confidence_lower=-10.0,
            rul_confidence_upper=0.0,
            anomaly_flag=True,
            anomaly_score=0.9,
            feature_importance=self.feature_importance
        )
        
        # Should clamp to non-negative values
        assert result.rul_cycles == 0
        assert result.rul_confidence_lower == 0
        assert result.rul_confidence_upper == 0
    
    def test_confidence_interval_consistency(self):
        """Test that confidence intervals are consistent"""
        result = self.aggregator.aggregate(
            rul_pred=50.0,
            rul_confidence_lower=60.0,  # Invalid: lower > prediction
            rul_confidence_upper=40.0,  # Invalid: upper < prediction
            anomaly_flag=False,
            anomaly_score=0.2,
            feature_importance=self.feature_importance
        )
        
        # Should ensure upper >= prediction
        assert result.rul_confidence_upper >= result.rul_cycles
    
    def test_update_degradation_thresholds(self):
        """Test updating degradation thresholds"""
        new_thresholds = {
            'healthy': (0.0, 0.3),
            'early_degradation': (0.3, 0.6),
            'advanced_degradation': (0.6, 0.8),
            'critical': (0.8, 1.0)
        }
        
        self.aggregator.update_degradation_thresholds(**new_thresholds)
        
        assert self.aggregator.degradation_thresholds['healthy'] == (0.0, 0.3)
        assert self.aggregator.degradation_thresholds['critical'] == (0.8, 1.0)
    
    def test_get_stage_info(self):
        """Test getting stage information"""
        info = self.aggregator.get_stage_info('healthy')
        assert 'min' in info
        assert 'max' in info
        assert info['min'] == 0.0
        assert info['max'] == 0.25
        
        # Test invalid stage
        with pytest.raises(ValueError):
            self.aggregator.get_stage_info('invalid_stage')
    
    def test_edge_case_degradation_score_1_0(self):
        """Test edge case where degradation score equals 1.0"""
        stage = self.aggregator.compute_degradation_stage(
            rul=0.0,
            anomaly_score=1.0,
            degradation_score=1.0
        )
        assert stage == "critical"
    
    def test_feature_importance_preservation(self):
        """Test that feature importance is preserved in result"""
        result = self.aggregator.aggregate(
            rul_pred=75.0,
            rul_confidence_lower=70.0,
            rul_confidence_upper=80.0,
            anomaly_flag=False,
            anomaly_score=0.3,
            feature_importance=self.feature_importance
        )
        
        assert result.feature_importance == self.feature_importance
        assert len(result.feature_importance) == 5
        assert result.feature_importance['feature_1'] == 0.3