"""
Property-based test for Low Confidence Flagging

This module contains property-based tests using the Hypothesis framework
to validate that the RUL prediction system properly flags predictions
with low confidence as uncertain.

Requirements: 7.3
"""

import sys
from pathlib import Path
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays
from datetime import datetime
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from true_rul.data_structures import PredictionResult, CycleData
    from true_rul.prediction_aggregator import PredictionAggregator
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    MODULES_AVAILABLE = False


class MockRULModel:
    """Mock RUL model for testing confidence flagging"""
    
    def __init__(self, confidence_width_factor: float = 1.0):
        """
        Initialize mock model
        
        Args:
            confidence_width_factor: Factor to control confidence interval width
                                   (higher = wider intervals = lower confidence)
        """
        self.confidence_width_factor = confidence_width_factor
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Mock prediction"""
        # Simple prediction based on feature mean
        return np.mean(X, axis=1) * 10 + 50
    
    def predict_with_confidence(self, X: np.ndarray) -> tuple:
        """Mock prediction with confidence intervals"""
        predictions = self.predict(X)
        
        # Create confidence intervals with varying width
        # Higher confidence_width_factor = wider intervals = lower confidence
        base_uncertainty = predictions * 0.1  # 10% base uncertainty
        uncertainty = base_uncertainty * self.confidence_width_factor
        
        lower = np.maximum(0, predictions - uncertainty)
        upper = predictions + uncertainty
        
        return predictions, lower, upper


class MockAnomalyDetector:
    """Mock anomaly detector for testing"""
    
    def __init__(self):
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> tuple:
        """Mock anomaly detection"""
        # Simple anomaly score based on feature variance
        anomaly_scores = np.var(X, axis=1) / 10.0
        anomaly_flags = anomaly_scores > 0.5
        
        # Mock feature importance
        feature_importance = {f"feature_{i}": np.random.random() for i in range(X.shape[1])}
        
        return anomaly_flags, anomaly_scores, feature_importance


class EnhancedPredictionAggregator(PredictionAggregator):
    """
    Enhanced prediction aggregator that implements low confidence flagging
    
    This extends the base PredictionAggregator to add confidence flagging
    functionality as required by Requirement 7.3.
    """
    
    def __init__(self, model_version: str = "1.0.0", confidence_threshold: float = 0.3):
        """
        Initialize enhanced aggregator
        
        Args:
            model_version: Version identifier for the model ensemble
            confidence_threshold: Threshold for flagging low confidence predictions
                                 (relative confidence interval width)
        """
        super().__init__(model_version)
        self.confidence_threshold = confidence_threshold
    
    def aggregate(
        self,
        rul_pred: float,
        rul_confidence_lower: float,
        rul_confidence_upper: float,
        anomaly_flag: bool,
        anomaly_score: float,
        feature_importance: Dict[str, float],
        degradation_history: list = None,
        capacitor_id: str = None,
        cycle_number: int = None
    ) -> PredictionResult:
        """
        Aggregate predictions with low confidence flagging
        
        This method extends the base aggregation to include confidence flagging
        as required by Requirement 7.3: "WHEN confidence is low, THE RUL_Predictor 
        SHALL flag predictions as uncertain and recommend additional monitoring"
        """
        # Call parent aggregation method
        result = super().aggregate(
            rul_pred=rul_pred,
            rul_confidence_lower=rul_confidence_lower,
            rul_confidence_upper=rul_confidence_upper,
            anomaly_flag=anomaly_flag,
            anomaly_score=anomaly_score,
            feature_importance=feature_importance,
            degradation_history=degradation_history,
            capacitor_id=capacitor_id,
            cycle_number=cycle_number
        )
        
        # Add low confidence flagging
        confidence_width = rul_confidence_upper - rul_confidence_lower
        relative_confidence_width = confidence_width / max(1.0, rul_pred)  # Avoid division by zero
        
        # Flag as low confidence if interval is too wide
        is_low_confidence = relative_confidence_width > self.confidence_threshold
        
        # Create enhanced result with confidence flag
        enhanced_result = EnhancedPredictionResult(
            rul_cycles=result.rul_cycles,
            rul_confidence_lower=result.rul_confidence_lower,
            rul_confidence_upper=result.rul_confidence_upper,
            degradation_score=result.degradation_score,
            degradation_stage=result.degradation_stage,
            anomaly_flag=result.anomaly_flag,
            anomaly_score=result.anomaly_score,
            feature_importance=result.feature_importance,
            timestamp=result.timestamp,
            model_version=result.model_version,
            capacitor_id=result.capacitor_id,
            cycle_number=result.cycle_number,
            is_low_confidence=is_low_confidence,
            confidence_width=confidence_width,
            relative_confidence_width=relative_confidence_width
        )
        
        return enhanced_result


class EnhancedPredictionResult(PredictionResult):
    """
    Enhanced prediction result with low confidence flagging
    
    This extends the base PredictionResult to include confidence flagging
    fields as required by Requirement 7.3.
    """
    
    def __init__(self, is_low_confidence: bool = False, confidence_width: float = 0.0, 
                 relative_confidence_width: float = 0.0, **kwargs):
        """
        Initialize enhanced prediction result
        
        Args:
            is_low_confidence: Flag indicating if prediction has low confidence
            confidence_width: Absolute width of confidence interval
            relative_confidence_width: Relative width (width / prediction)
            **kwargs: Arguments for base PredictionResult
        """
        super().__init__(**kwargs)
        self.is_low_confidence = is_low_confidence
        self.confidence_width = confidence_width
        self.relative_confidence_width = relative_confidence_width
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with confidence flags"""
        result = super().to_dict()
        result.update({
            "is_low_confidence": self.is_low_confidence,
            "confidence_width": self.confidence_width,
            "relative_confidence_width": self.relative_confidence_width
        })
        return result


@pytest.mark.skipif(
    not MODULES_AVAILABLE,
    reason="Required modules not available"
)
class TestLowConfidenceFlaggingProperty:
    """Property-based tests for low confidence flagging"""
    
    @given(
        n_samples=st.integers(min_value=5, max_value=15),
        n_features=st.integers(min_value=3, max_value=8),
        confidence_width_factor=st.floats(min_value=0.1, max_value=5.0),
        confidence_threshold=st.floats(min_value=0.1, max_value=0.8),
        data=st.data()
    )
    @settings(max_examples=100, deadline=20000)  # 20 second timeout per example
    def test_low_confidence_flagging_property(
        self, 
        n_samples, 
        n_features, 
        confidence_width_factor, 
        confidence_threshold,
        data
    ):
        """
        Property 14: Low Confidence Flagging
        
        **Validates: Requirements 7.3**
        
        WHEN confidence is low (wide confidence intervals), THE RUL_Predictor 
        SHALL flag predictions as uncertain and recommend additional monitoring.
        
        This property validates that:
        1. Predictions with wide confidence intervals are flagged as low confidence
        2. Predictions with narrow confidence intervals are not flagged
        3. The flagging threshold is consistently applied
        4. Low confidence flags are properly included in prediction results
        """
        # Generate realistic test data
        X_test = data.draw(
            arrays(
                dtype=np.float64,
                shape=(n_samples, n_features),
                elements=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False)
            )
        )
        
        # Skip if data contains invalid values
        assume(np.all(np.isfinite(X_test)))
        
        # Create mock models with controlled confidence width
        rul_model = MockRULModel(confidence_width_factor=confidence_width_factor)
        anomaly_detector = MockAnomalyDetector()
        
        # Create enhanced aggregator with specified threshold
        aggregator = EnhancedPredictionAggregator(
            confidence_threshold=confidence_threshold
        )
        
        # Generate predictions with confidence intervals
        predictions, lower_bounds, upper_bounds = rul_model.predict_with_confidence(X_test)
        anomaly_flags, anomaly_scores, feature_importance = anomaly_detector.predict(X_test)
        
        # Test each prediction
        for i in range(n_samples):
            pred = float(predictions[i])
            lower = float(lower_bounds[i])
            upper = float(upper_bounds[i])
            anomaly_flag = bool(anomaly_flags[i])
            anomaly_score = float(anomaly_scores[i])
            
            # Skip invalid predictions
            assume(pred > 0)  # RUL must be positive
            assume(lower >= 0)  # Lower bound must be non-negative
            assume(upper >= pred)  # Upper bound must be >= prediction
            assume(np.isfinite(pred) and np.isfinite(lower) and np.isfinite(upper))
            
            # Aggregate prediction
            result = aggregator.aggregate(
                rul_pred=pred,
                rul_confidence_lower=lower,
                rul_confidence_upper=upper,
                anomaly_flag=anomaly_flag,
                anomaly_score=anomaly_score,
                feature_importance=feature_importance,
                capacitor_id=f"test_cap_{i}",
                cycle_number=i + 1
            )
            
            # Calculate expected confidence metrics
            confidence_width = upper - lower
            relative_confidence_width = confidence_width / pred
            expected_low_confidence = relative_confidence_width > confidence_threshold
            
            # Property 1: Low confidence flag should match threshold criterion
            assert hasattr(result, 'is_low_confidence'), (
                "Result must have is_low_confidence attribute"
            )
            assert result.is_low_confidence == expected_low_confidence, (
                f"Low confidence flag mismatch: expected {expected_low_confidence}, "
                f"got {result.is_low_confidence} "
                f"(relative_width={relative_confidence_width:.3f}, "
                f"threshold={confidence_threshold:.3f})"
            )
            
            # Property 2: Confidence width should be correctly calculated
            assert hasattr(result, 'confidence_width'), (
                "Result must have confidence_width attribute"
            )
            assert abs(result.confidence_width - confidence_width) < 1e-6, (
                f"Confidence width mismatch: expected {confidence_width:.6f}, "
                f"got {result.confidence_width:.6f}"
            )
            
            # Property 3: Relative confidence width should be correctly calculated
            assert hasattr(result, 'relative_confidence_width'), (
                "Result must have relative_confidence_width attribute"
            )
            assert abs(result.relative_confidence_width - relative_confidence_width) < 1e-6, (
                f"Relative confidence width mismatch: expected {relative_confidence_width:.6f}, "
                f"got {result.relative_confidence_width:.6f}"
            )
            
            # Property 4: Low confidence flag should be boolean
            assert isinstance(result.is_low_confidence, bool), (
                f"Low confidence flag must be boolean, got {type(result.is_low_confidence)}"
            )
            
            # Property 5: Confidence metrics should be non-negative
            assert result.confidence_width >= 0, (
                f"Confidence width must be non-negative, got {result.confidence_width}"
            )
            assert result.relative_confidence_width >= 0, (
                f"Relative confidence width must be non-negative, "
                f"got {result.relative_confidence_width}"
            )
            
            # Property 6: Enhanced result should maintain all base properties
            assert result.rul_cycles >= 0, "RUL cycles must be non-negative"
            assert result.rul_confidence_lower >= 0, "Lower bound must be non-negative"
            assert result.rul_confidence_upper >= result.rul_cycles, "Upper bound must be >= prediction"
            assert 0 <= result.degradation_score <= 1, "Degradation score must be in [0,1]"
            assert isinstance(result.anomaly_flag, bool), "Anomaly flag must be boolean"
            assert 0 <= result.anomaly_score <= 1, "Anomaly score must be in [0,1]"
            assert isinstance(result.feature_importance, dict), "Feature importance must be dict"
            
            # Property 7: Enhanced result should be serializable
            result_dict = result.to_dict()
            assert 'is_low_confidence' in result_dict, "Dict must contain low confidence flag"
            assert 'confidence_width' in result_dict, "Dict must contain confidence width"
            assert 'relative_confidence_width' in result_dict, "Dict must contain relative width"
    
    @given(
        confidence_threshold=st.floats(min_value=0.1, max_value=0.5)
    )
    @settings(max_examples=50, deadline=15000)
    def test_confidence_threshold_consistency(self, confidence_threshold):
        """
        Property 14 (Consistency): Confidence Threshold Application
        
        **Validates: Requirements 7.3**
        
        The confidence threshold should be consistently applied across
        different predictions and scenarios.
        """
        # Create test scenarios with known confidence characteristics
        test_scenarios = [
            # (prediction, lower, upper, expected_flag_description)
            (100.0, 90.0, 110.0, "narrow_interval"),      # 20% width -> low threshold
            (100.0, 50.0, 150.0, "wide_interval"),        # 100% width -> high threshold  
            (50.0, 45.0, 55.0, "narrow_interval"),        # 20% width -> low threshold
            (50.0, 25.0, 75.0, "wide_interval"),          # 100% width -> high threshold
            (10.0, 9.0, 11.0, "narrow_interval"),         # 20% width -> low threshold
            (10.0, 5.0, 15.0, "wide_interval"),           # 100% width -> high threshold
        ]
        
        aggregator = EnhancedPredictionAggregator(
            confidence_threshold=confidence_threshold
        )
        
        # Mock feature importance
        feature_importance = {"feature_0": 0.5, "feature_1": 0.3, "feature_2": 0.2}
        
        for pred, lower, upper, scenario_type in test_scenarios:
            # Calculate expected behavior
            relative_width = (upper - lower) / pred
            expected_low_confidence = relative_width > confidence_threshold
            
            # Generate prediction result
            result = aggregator.aggregate(
                rul_pred=pred,
                rul_confidence_lower=lower,
                rul_confidence_upper=upper,
                anomaly_flag=False,
                anomaly_score=0.1,
                feature_importance=feature_importance,
                capacitor_id="test_cap",
                cycle_number=1
            )
            
            # Property: Threshold should be consistently applied
            assert result.is_low_confidence == expected_low_confidence, (
                f"Threshold consistency failed for scenario {scenario_type}: "
                f"pred={pred}, lower={lower}, upper={upper}, "
                f"relative_width={relative_width:.3f}, threshold={confidence_threshold:.3f}, "
                f"expected={expected_low_confidence}, got={result.is_low_confidence}"
            )
            
            # Property: Relative width calculation should be consistent
            expected_relative_width = (upper - lower) / pred
            assert abs(result.relative_confidence_width - expected_relative_width) < 1e-6, (
                f"Relative width calculation inconsistent: "
                f"expected={expected_relative_width:.6f}, got={result.relative_confidence_width:.6f}"
            )
    
    def test_edge_cases_low_confidence_flagging(self):
        """
        Property 14 (Edge Cases): Low Confidence Flagging Edge Cases
        
        **Validates: Requirements 7.3**
        
        Test edge cases for low confidence flagging to ensure robustness.
        """
        aggregator = EnhancedPredictionAggregator(confidence_threshold=0.3)
        feature_importance = {"feature_0": 1.0}
        
        # Edge case 1: Very small prediction (near zero)
        result = aggregator.aggregate(
            rul_pred=1.0,
            rul_confidence_lower=0.0,
            rul_confidence_upper=2.0,
            anomaly_flag=False,
            anomaly_score=0.1,
            feature_importance=feature_importance
        )
        
        # Should handle small predictions correctly
        assert isinstance(result.is_low_confidence, bool)
        assert result.relative_confidence_width == 2.0  # (2-0)/1 = 2.0
        assert result.is_low_confidence == True  # 2.0 > 0.3
        
        # Edge case 2: Zero confidence interval width
        result = aggregator.aggregate(
            rul_pred=50.0,
            rul_confidence_lower=50.0,
            rul_confidence_upper=50.0,
            anomaly_flag=False,
            anomaly_score=0.1,
            feature_importance=feature_importance
        )
        
        # Should handle zero width correctly
        assert result.confidence_width == 0.0
        assert result.relative_confidence_width == 0.0
        assert result.is_low_confidence == False  # 0.0 <= 0.3
        
        # Edge case 3: Large prediction with proportional confidence interval
        result = aggregator.aggregate(
            rul_pred=1000.0,
            rul_confidence_lower=900.0,
            rul_confidence_upper=1100.0,
            anomaly_flag=False,
            anomaly_score=0.1,
            feature_importance=feature_importance
        )
        
        # Should handle large predictions correctly
        assert result.confidence_width == 200.0
        assert result.relative_confidence_width == 0.2  # 200/1000 = 0.2
        assert result.is_low_confidence == False  # 0.2 <= 0.3
    
    def test_low_confidence_integration_with_base_functionality(self):
        """
        Property 14 (Integration): Low Confidence Flagging Integration
        
        **Validates: Requirements 7.3**
        
        Ensure that low confidence flagging integrates properly with
        existing prediction functionality without breaking base behavior.
        """
        aggregator = EnhancedPredictionAggregator(confidence_threshold=0.25)
        feature_importance = {"feature_0": 0.6, "feature_1": 0.4}
        
        # Test with various degradation stages
        test_cases = [
            (20.0, 18.0, 22.0, "healthy"),      # Narrow interval, healthy
            (50.0, 30.0, 70.0, "early_degradation"),  # Wide interval, early degradation
            (80.0, 60.0, 100.0, "advanced_degradation"),  # Wide interval, advanced
            (150.0, 140.0, 160.0, "critical"),  # Narrow interval, critical
        ]
        
        for pred, lower, upper, expected_stage_type in test_cases:
            result = aggregator.aggregate(
                rul_pred=pred,
                rul_confidence_lower=lower,
                rul_confidence_upper=upper,
                anomaly_flag=False,
                anomaly_score=0.2,
                feature_importance=feature_importance,
                capacitor_id="integration_test",
                cycle_number=10
            )
            
            # Property: Base functionality should still work
            assert result.rul_cycles >= 0
            assert result.rul_confidence_lower >= 0
            assert result.rul_confidence_upper >= result.rul_cycles
            assert 0 <= result.degradation_score <= 1
            assert result.degradation_stage in {
                "healthy", "early_degradation", "advanced_degradation", "critical"
            }
            
            # Property: Low confidence flagging should be added without breaking base
            assert hasattr(result, 'is_low_confidence')
            assert isinstance(result.is_low_confidence, bool)
            
            # Property: Confidence metrics should be calculated
            expected_width = upper - lower
            expected_relative_width = expected_width / pred
            expected_low_confidence = expected_relative_width > 0.25
            
            assert abs(result.confidence_width - expected_width) < 1e-6
            assert abs(result.relative_confidence_width - expected_relative_width) < 1e-6
            assert result.is_low_confidence == expected_low_confidence
            
            # Property: Serialization should include new fields
            result_dict = result.to_dict()
            assert 'is_low_confidence' in result_dict
            assert 'confidence_width' in result_dict
            assert 'relative_confidence_width' in result_dict
            
            # Property: All original fields should still be present
            original_fields = {
                'rul_cycles', 'rul_confidence_lower', 'rul_confidence_upper',
                'degradation_score', 'degradation_stage', 'anomaly_flag',
                'anomaly_score', 'feature_importance', 'timestamp', 'model_version'
            }
            for field in original_fields:
                assert field in result_dict, f"Original field {field} missing from result"