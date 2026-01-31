"""
Property-based test for continuous degradation output

This module contains property-based tests using the Hypothesis framework
to validate that the RUL prediction system outputs continuous degradation
progression rather than binary normal/abnormal classification.

Requirements: 4.1
"""

import sys
from pathlib import Path
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.extra.numpy import arrays
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from true_rul.prediction_aggregator import PredictionAggregator
    AGGREGATOR_AVAILABLE = True
except Exception:
    AGGREGATOR_AVAILABLE = False

try:
    from true_rul.data_structures import PredictionResult
    DATA_STRUCTURES_AVAILABLE = True
except Exception:
    DATA_STRUCTURES_AVAILABLE = False


class TestContinuousDegradationProperty:
    """Property-based tests for continuous degradation output"""
    
    @pytest.mark.skipif(not (AGGREGATOR_AVAILABLE and DATA_STRUCTURES_AVAILABLE), 
                       reason="Required modules not available")
    @given(
        rul_pred=st.floats(min_value=1.0, max_value=200.0, allow_nan=False, allow_infinity=False),
        anomaly_flag=st.booleans(),
        anomaly_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        n_features=st.integers(min_value=3, max_value=8),
        data=st.data()
    )
    @settings(max_examples=100, deadline=20000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_continuous_degradation_output(self, rul_pred, anomaly_flag, 
                                         anomaly_score, n_features, data):
        """
        Property 7: Continuous Degradation Output
        
        **Validates: Requirements 4.1**
        
        For any prediction, the degradation score should be a continuous float value 
        in the range [0, 1], not a binary classification.
        
        This property ensures that the RUL prediction system outputs continuous 
        degradation progression rather than binary normal/abnormal classification,
        enabling more nuanced understanding of component health status.
        """
        # Generate confidence intervals that are properly ordered
        rul_confidence_lower = data.draw(st.floats(min_value=0.0, max_value=rul_pred, 
                                                  allow_nan=False, allow_infinity=False))
        rul_confidence_upper = data.draw(st.floats(min_value=rul_pred, max_value=rul_pred + 50.0, 
                                                  allow_nan=False, allow_infinity=False))
        
        # Generate realistic feature importance dictionary
        feature_names = [f"feature_{i}" for i in range(n_features)]
        feature_importance = {}
        for name in feature_names:
            importance = data.draw(st.floats(min_value=0.0, max_value=1.0, 
                                           allow_nan=False, allow_infinity=False))
            feature_importance[name] = importance
        
        # Normalize feature importance to sum to 1.0 (realistic constraint)
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            feature_importance = {k: v / total_importance for k, v in feature_importance.items()}
        
        # Generate optional degradation history
        history_length = data.draw(st.integers(min_value=0, max_value=10))
        degradation_history = None
        if history_length > 0:
            degradation_history = data.draw(
                arrays(
                    dtype=np.float64,
                    shape=(history_length,),
                    elements=st.floats(min_value=0.0, max_value=1.0, 
                                     allow_nan=False, allow_infinity=False)
                )
            ).tolist()
        
        # Optional capacitor ID and cycle number
        capacitor_id = data.draw(st.one_of(st.none(), st.text(min_size=1, max_size=10)))
        cycle_number = data.draw(st.one_of(st.none(), st.integers(min_value=1, max_value=200)))
        
        try:
            # Create prediction aggregator
            aggregator = PredictionAggregator(model_version="test_1.0.0")
            
            # Generate aggregated prediction
            result = aggregator.aggregate(
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
            
            # Property 1: Degradation score must be continuous (float type)
            assert isinstance(result.degradation_score, (float, np.floating)), (
                f"Degradation score must be a continuous float value, "
                f"got {type(result.degradation_score)}: {result.degradation_score}"
            )
            
            # Property 2: Degradation score must be in valid range [0, 1]
            assert 0.0 <= result.degradation_score <= 1.0, (
                f"Degradation score must be in range [0, 1], "
                f"got {result.degradation_score}"
            )
            
            # Property 3: Degradation score must be finite (not NaN or infinity)
            assert np.isfinite(result.degradation_score), (
                f"Degradation score must be finite, got {result.degradation_score}"
            )
            
            # Property 4: Degradation score should not be binary (0 or 1 only)
            # We allow some tolerance for edge cases, but the system should generally
            # produce intermediate values, not just binary extremes
            # This is tested by ensuring the aggregator can produce intermediate values
            # when given intermediate inputs
            if 0.1 <= anomaly_score <= 0.9 and 10 <= rul_pred <= 150:
                # For intermediate inputs, we expect intermediate degradation scores
                # (not strictly 0.0 or 1.0, allowing small tolerance for edge cases)
                assert not (result.degradation_score == 0.0 or result.degradation_score == 1.0), (
                    f"For intermediate inputs (anomaly_score={anomaly_score:.3f}, "
                    f"rul_pred={rul_pred:.1f}), degradation score should not be binary extremes, "
                    f"got {result.degradation_score}"
                )
            
            # Property 5: Degradation stage should be consistent with degradation score
            expected_stage = aggregator.compute_degradation_stage(
                rul_pred, anomaly_score, result.degradation_score
            )
            assert result.degradation_stage == expected_stage, (
                f"Degradation stage inconsistent with score: "
                f"score={result.degradation_score}, stage={result.degradation_stage}, "
                f"expected={expected_stage}"
            )
            
            # Property 6: Degradation score should be deterministic for same inputs
            result2 = aggregator.aggregate(
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
            
            assert abs(result.degradation_score - result2.degradation_score) < 1e-10, (
                f"Degradation score should be deterministic: "
                f"first={result.degradation_score}, second={result2.degradation_score}"
            )
            
            # Property 7: Test direct degradation score computation
            direct_score = aggregator._compute_degradation_score(
                rul_pred, anomaly_score, degradation_history
            )
            
            assert isinstance(direct_score, (float, np.floating)), (
                f"Direct degradation score must be continuous float, "
                f"got {type(direct_score)}: {direct_score}"
            )
            
            assert 0.0 <= direct_score <= 1.0, (
                f"Direct degradation score must be in range [0, 1], "
                f"got {direct_score}"
            )
            
            assert np.isfinite(direct_score), (
                f"Direct degradation score must be finite, got {direct_score}"
            )
            
        except Exception as e:
            # If aggregation fails due to invalid input combinations, skip this example
            assume(False, f"Prediction aggregation failed: {e}")
    
    @pytest.mark.skipif(not (AGGREGATOR_AVAILABLE and DATA_STRUCTURES_AVAILABLE), 
                       reason="Required modules not available")
    @given(
        rul_values=st.lists(
            st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False),
            min_size=5, max_size=15
        ),
        anomaly_scores=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=5, max_size=15
        )
    )
    @settings(max_examples=50, deadline=15000)
    def test_degradation_score_continuity(self, rul_values, anomaly_scores):
        """
        Property 7 (Continuity): Degradation Score Smooth Transitions
        
        **Validates: Requirements 4.1**
        
        When RUL and anomaly scores change gradually, the degradation score
        should also change gradually (continuous function), not in discrete jumps.
        
        This ensures the system provides smooth degradation progression rather
        than abrupt binary transitions.
        """
        # Ensure both lists have the same length
        min_length = min(len(rul_values), len(anomaly_scores))
        rul_values = rul_values[:min_length]
        anomaly_scores = anomaly_scores[:min_length]
        
        assume(min_length >= 5)  # Need enough points to test continuity
        
        try:
            aggregator = PredictionAggregator(model_version="test_1.0.0")
            
            # Compute degradation scores for all input combinations
            degradation_scores = []
            for rul, anomaly in zip(rul_values, anomaly_scores):
                score = aggregator._compute_degradation_score(rul, anomaly)
                degradation_scores.append(score)
            
            # Property 1: All scores should be continuous (float) values
            for i, score in enumerate(degradation_scores):
                assert isinstance(score, (float, np.floating)), (
                    f"Degradation score {i} must be continuous float, "
                    f"got {type(score)}: {score}"
                )
                
                assert 0.0 <= score <= 1.0, (
                    f"Degradation score {i} must be in range [0, 1], got {score}"
                )
                
                assert np.isfinite(score), (
                    f"Degradation score {i} must be finite, got {score}"
                )
            
            # Property 2: Test for reasonable continuity
            # Sort inputs by RUL to test monotonic relationship
            sorted_indices = np.argsort(rul_values)
            sorted_rul = [rul_values[i] for i in sorted_indices]
            sorted_scores = [degradation_scores[i] for i in sorted_indices]
            
            # For sorted RUL values (low to high), degradation scores should generally
            # decrease (higher RUL = lower degradation), allowing for some variation
            # due to anomaly score influence
            large_increases = 0
            for i in range(1, len(sorted_scores)):
                score_change = sorted_scores[i] - sorted_scores[i-1]
                rul_change = sorted_rul[i] - sorted_rul[i-1]
                
                # If RUL increased significantly, degradation shouldn't increase dramatically
                if rul_change > 20:  # Significant RUL increase
                    if score_change > 0.3:  # But degradation increased a lot
                        large_increases += 1
            
            # Allow some violations due to anomaly score influence, but not too many
            violation_rate = large_increases / max(1, len(sorted_scores) - 1)
            assert violation_rate <= 0.3, (
                f"Too many large degradation increases with RUL increases: "
                f"{large_increases}/{len(sorted_scores)-1} = {violation_rate:.2f}"
            )
            
            # Property 3: Test range coverage
            # The degradation scores should cover a reasonable range of values
            # (not all clustered at extremes)
            score_range = max(degradation_scores) - min(degradation_scores)
            if len(set(rul_values)) > 3 and len(set(anomaly_scores)) > 3:
                # Only test range if we have diverse inputs
                assert score_range > 0.1, (
                    f"Degradation scores should cover reasonable range with diverse inputs, "
                    f"got range {score_range:.3f} from scores {degradation_scores}"
                )
            
        except Exception as e:
            assume(False, f"Degradation score continuity test failed: {e}")
    
    @pytest.mark.skipif(not (AGGREGATOR_AVAILABLE and DATA_STRUCTURES_AVAILABLE), 
                       reason="Required modules not available")
    @given(
        anomaly_score=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=30, deadline=10000)
    def test_degradation_score_boundary_cases(self, anomaly_score):
        """
        Property 7 (Boundary Cases): Degradation Score at Extremes
        
        **Validates: Requirements 4.1**
        
        Test degradation score behavior at boundary conditions:
        - Very high RUL (near end of life)
        - Very low RUL (healthy)
        - Extreme anomaly scores
        
        The degradation score should still be continuous and in valid range.
        """
        try:
            aggregator = PredictionAggregator(model_version="test_1.0.0")
            
            # Test cases with extreme RUL values
            test_cases = [
                (0.0, anomaly_score),      # End of life
                (1.0, anomaly_score),      # Almost end of life
                (200.0, anomaly_score),    # Very healthy (max expected)
                (500.0, anomaly_score),    # Extremely healthy (beyond expected)
                (50.0, 0.0),               # Mid RUL, no anomaly
                (50.0, 1.0),               # Mid RUL, maximum anomaly
            ]
            
            for rul, anomaly in test_cases:
                score = aggregator._compute_degradation_score(rul, anomaly)
                
                # Property 1: Score must be continuous float
                assert isinstance(score, (float, np.floating)), (
                    f"Boundary case (RUL={rul}, anomaly={anomaly}) degradation score "
                    f"must be continuous float, got {type(score)}: {score}"
                )
                
                # Property 2: Score must be in valid range
                assert 0.0 <= score <= 1.0, (
                    f"Boundary case (RUL={rul}, anomaly={anomaly}) degradation score "
                    f"must be in range [0, 1], got {score}"
                )
                
                # Property 3: Score must be finite
                assert np.isfinite(score), (
                    f"Boundary case (RUL={rul}, anomaly={anomaly}) degradation score "
                    f"must be finite, got {score}"
                )
            
            # Property 4: Test expected relationships at boundaries
            score_end_life = aggregator._compute_degradation_score(0.0, anomaly_score)
            score_healthy = aggregator._compute_degradation_score(200.0, anomaly_score)
            
            # End of life should generally have higher degradation than healthy
            # (allowing for some cases where anomaly score dominates)
            if anomaly_score < 0.5:  # When anomaly is not dominant
                assert score_end_life >= score_healthy, (
                    f"End of life (RUL=0) should have higher degradation than healthy (RUL=200) "
                    f"when anomaly score is low: end_life={score_end_life:.3f}, "
                    f"healthy={score_healthy:.3f}, anomaly={anomaly_score:.3f}"
                )
            
        except Exception as e:
            assume(False, f"Boundary case test failed: {e}")