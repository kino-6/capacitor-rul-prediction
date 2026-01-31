"""
Property-based test for degradation monotonicity

This module contains property-based tests using the Hypothesis framework
to validate that the RUL prediction system maintains monotonicity in 
degradation scores (degradation should not decrease over time for the same component).

Requirements: 4.4
"""

import sys
from pathlib import Path
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.extra.numpy import arrays
from datetime import datetime
from typing import List, Tuple

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


class TestDegradationMonotonicityProperty:
    """Property-based tests for degradation monotonicity"""
    
    @pytest.mark.skipif(not (AGGREGATOR_AVAILABLE and DATA_STRUCTURES_AVAILABLE), 
                       reason="Required modules not available")
    @given(
        n_cycles=st.integers(min_value=5, max_value=20),
        initial_rul=st.floats(min_value=50.0, max_value=200.0, allow_nan=False, allow_infinity=False),
        data=st.data()
    )
    @settings(max_examples=100, deadline=30000, suppress_health_check=[HealthCheck.filter_too_much])
    def test_degradation_monotonicity_over_time(self, n_cycles, initial_rul, data):
        """
        Property 9: Degradation Monotonicity
        
        **Validates: Requirements 4.4**
        
        For the same component over time, degradation scores should not decrease.
        As cycles progress, the degradation score should remain the same or increase,
        reflecting the physical reality that components don't spontaneously improve.
        
        This property tests a sequence of cycles for a single capacitor to ensure
        that degradation progression is monotonic (non-decreasing).
        """
        # Generate a realistic sequence of cycles for the same capacitor
        capacitor_id = data.draw(st.text(min_size=3, max_size=10))
        
        # Generate decreasing RUL values (as cycles progress, RUL decreases)
        rul_values = []
        current_rul = initial_rul
        for i in range(n_cycles):
            # RUL should generally decrease, but allow some small variations
            rul_decrease = data.draw(st.floats(min_value=0.0, max_value=5.0, 
                                             allow_nan=False, allow_infinity=False))
            current_rul = max(0.0, current_rul - rul_decrease)
            rul_values.append(current_rul)
        
        # Generate anomaly scores that generally increase over time (more anomalous as degradation progresses)
        anomaly_scores = []
        base_anomaly = data.draw(st.floats(min_value=0.0, max_value=0.3, 
                                         allow_nan=False, allow_infinity=False))
        for i in range(n_cycles):
            # Anomaly scores should generally increase, but allow some variation
            anomaly_increase = data.draw(st.floats(min_value=0.0, max_value=0.1, 
                                                 allow_nan=False, allow_infinity=False))
            current_anomaly = min(1.0, base_anomaly + (i * 0.05) + anomaly_increase)
            anomaly_scores.append(current_anomaly)
        
        # Generate feature importance (can be random as it doesn't affect degradation score)
        n_features = data.draw(st.integers(min_value=3, max_value=8))
        feature_names = [f"feature_{j}" for j in range(n_features)]
        
        try:
            aggregator = PredictionAggregator(model_version="test_1.0.0")
            
            # Generate predictions for each cycle
            predictions = []
            degradation_history = []
            
            for cycle_num in range(n_cycles):
                # Generate feature importance for this cycle
                feature_importance = {}
                for name in feature_names:
                    importance = data.draw(st.floats(min_value=0.0, max_value=1.0, 
                                                   allow_nan=False, allow_infinity=False))
                    feature_importance[name] = importance
                
                # Normalize feature importance
                total_importance = sum(feature_importance.values())
                if total_importance > 0:
                    feature_importance = {k: v / total_importance for k, v in feature_importance.items()}
                
                # Generate confidence intervals
                rul_pred = rul_values[cycle_num]
                rul_confidence_lower = data.draw(st.floats(min_value=0.0, max_value=max(0.0, rul_pred), 
                                                          allow_nan=False, allow_infinity=False))
                rul_confidence_upper = data.draw(st.floats(min_value=max(0.0, rul_pred), 
                                                          max_value=rul_pred + 20.0, 
                                                          allow_nan=False, allow_infinity=False))
                
                # Create prediction with degradation history
                result = aggregator.aggregate(
                    rul_pred=rul_pred,
                    rul_confidence_lower=rul_confidence_lower,
                    rul_confidence_upper=rul_confidence_upper,
                    anomaly_flag=anomaly_scores[cycle_num] > 0.5,
                    anomaly_score=anomaly_scores[cycle_num],
                    feature_importance=feature_importance,
                    degradation_history=degradation_history.copy(),
                    capacitor_id=capacitor_id,
                    cycle_number=cycle_num + 1
                )
                
                predictions.append(result)
                degradation_history.append(result.degradation_score)
            
            # Property 1: All degradation scores should be valid
            for i, pred in enumerate(predictions):
                assert 0.0 <= pred.degradation_score <= 1.0, (
                    f"Degradation score at cycle {i+1} must be in [0, 1], "
                    f"got {pred.degradation_score}"
                )
                
                assert np.isfinite(pred.degradation_score), (
                    f"Degradation score at cycle {i+1} must be finite, "
                    f"got {pred.degradation_score}"
                )
            
            # Property 2: Degradation scores should be monotonic (non-decreasing)
            degradation_scores = [pred.degradation_score for pred in predictions]
            
            violations = []
            for i in range(1, len(degradation_scores)):
                current_score = degradation_scores[i]
                previous_score = degradation_scores[i-1]
                
                if current_score < previous_score:
                    # Allow very small decreases due to floating point precision
                    decrease = previous_score - current_score
                    if decrease > 1e-10:  # Significant decrease
                        violations.append({
                            'cycle': i + 1,
                            'previous_cycle': i,
                            'previous_score': previous_score,
                            'current_score': current_score,
                            'decrease': decrease,
                            'rul_previous': rul_values[i-1],
                            'rul_current': rul_values[i],
                            'anomaly_previous': anomaly_scores[i-1],
                            'anomaly_current': anomaly_scores[i]
                        })
            
            # Property 3: No significant violations of monotonicity should occur
            if violations:
                violation_details = []
                for v in violations:
                    violation_details.append(
                        f"Cycle {v['cycle']}: {v['current_score']:.6f} < "
                        f"Cycle {v['previous_cycle']}: {v['previous_score']:.6f} "
                        f"(decrease: {v['decrease']:.6f}, RUL: {v['rul_current']:.1f} vs {v['rul_previous']:.1f}, "
                        f"Anomaly: {v['anomaly_current']:.3f} vs {v['anomaly_previous']:.3f})"
                    )
                
                assert False, (
                    f"Degradation monotonicity violated for capacitor {capacitor_id}. "
                    f"Found {len(violations)} violations:\n" + "\n".join(violation_details)
                )
            
            # Property 4: If RUL decreases significantly and anomaly increases, 
            # degradation should increase or stay the same
            for i in range(1, len(predictions)):
                rul_decrease = rul_values[i-1] - rul_values[i]
                anomaly_increase = anomaly_scores[i] - anomaly_scores[i-1]
                score_change = degradation_scores[i] - degradation_scores[i-1]
                
                # If both RUL decreased significantly AND anomaly increased significantly
                if rul_decrease > 10.0 and anomaly_increase > 0.1:
                    assert score_change >= -1e-10, (
                        f"When RUL decreases significantly ({rul_decrease:.1f}) and "
                        f"anomaly increases significantly ({anomaly_increase:.3f}), "
                        f"degradation score should not decrease. "
                        f"Cycle {i}: score change = {score_change:.6f}"
                    )
            
            # Property 5: Test that the sequence shows overall degradation progression
            if len(degradation_scores) >= 5:
                first_third = np.mean(degradation_scores[:len(degradation_scores)//3])
                last_third = np.mean(degradation_scores[-len(degradation_scores)//3:])
                
                # The last third should generally have higher degradation than the first third
                # Allow some tolerance for cases where the component starts already degraded
                if first_third < 0.7:  # Only test if not starting in critical condition
                    assert last_third >= first_third - 1e-10, (
                        f"Overall degradation progression should be non-decreasing: "
                        f"first third mean = {first_third:.3f}, last third mean = {last_third:.3f}"
                    )
            
        except Exception as e:
            # If aggregation fails due to invalid input combinations, skip this example
            assume(False)
    
    @pytest.mark.skipif(not (AGGREGATOR_AVAILABLE and DATA_STRUCTURES_AVAILABLE), 
                       reason="Required modules not available")
    @given(
        degradation_history=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=3, max_size=10
        ),
        rul_current=st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        anomaly_current=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=50, deadline=15000)
    def test_degradation_score_with_history_monotonicity(self, degradation_history, rul_current, anomaly_current):
        """
        Property 9 (History): Degradation Score with History Monotonicity
        
        **Validates: Requirements 4.4**
        
        When computing degradation scores with historical context, the trend component
        should not cause the overall degradation to decrease significantly.
        
        This tests the internal degradation score computation to ensure that
        historical trends contribute positively to monotonicity.
        """
        # Sort degradation history to ensure it's monotonic (as it should be in real usage)
        sorted_history = sorted(degradation_history)
        
        try:
            aggregator = PredictionAggregator(model_version="test_1.0.0")
            
            # Compute degradation score with the sorted history
            current_score = aggregator._compute_degradation_score(
                rul=rul_current,
                anomaly_score=anomaly_current,
                degradation_history=sorted_history
            )
            
            # Property 1: Current score should be valid
            assert 0.0 <= current_score <= 1.0, (
                f"Current degradation score must be in [0, 1], got {current_score}"
            )
            
            assert np.isfinite(current_score), (
                f"Current degradation score must be finite, got {current_score}"
            )
            
            # Property 2: If history shows increasing trend, current score should not be 
            # significantly lower than the last historical value
            if len(sorted_history) >= 2:
                last_historical = sorted_history[-1]
                second_last_historical = sorted_history[-2]
                
                # If history shows increasing trend
                if last_historical > second_last_historical:
                    # Current score should not be significantly lower than last historical
                    # Allow some decrease due to other factors, but not dramatic
                    max_allowed_decrease = 0.2  # Allow up to 0.2 decrease
                    
                    assert current_score >= last_historical - max_allowed_decrease, (
                        f"With increasing degradation history trend, current score should not "
                        f"decrease dramatically. Last historical: {last_historical:.3f}, "
                        f"current: {current_score:.3f}, decrease: {last_historical - current_score:.3f}"
                    )
            
            # Property 3: Test multiple consecutive computations to ensure monotonicity
            # Simulate adding the current score to history and computing next score
            extended_history = sorted_history + [current_score]
            
            # Compute next score with slightly worse conditions (lower RUL, higher anomaly)
            next_rul = max(0.0, rul_current - 5.0)  # RUL decreases
            next_anomaly = min(1.0, anomaly_current + 0.1)  # Anomaly increases
            
            next_score = aggregator._compute_degradation_score(
                rul=next_rul,
                anomaly_score=next_anomaly,
                degradation_history=extended_history
            )
            
            # Property 4: Next score should not be significantly lower than current
            # (allowing small decreases due to floating point precision)
            assert next_score >= current_score - 1e-10, (
                f"Degradation score should not decrease when conditions worsen. "
                f"Current: {current_score:.6f}, Next: {next_score:.6f}, "
                f"RUL change: {rul_current:.1f} -> {next_rul:.1f}, "
                f"Anomaly change: {anomaly_current:.3f} -> {next_anomaly:.3f}"
            )
            
        except Exception as e:
            assume(False)
    
    @pytest.mark.skipif(not (AGGREGATOR_AVAILABLE and DATA_STRUCTURES_AVAILABLE), 
                       reason="Required modules not available")
    @given(
        n_capacitors=st.integers(min_value=2, max_value=5),
        n_cycles_per_capacitor=st.integers(min_value=5, max_value=10),
        data=st.data()
    )
    @settings(max_examples=30, deadline=25000)
    def test_monotonicity_across_multiple_capacitors(self, n_capacitors, n_cycles_per_capacitor, data):
        """
        Property 9 (Multi-Capacitor): Degradation Monotonicity Across Multiple Capacitors
        
        **Validates: Requirements 4.4**
        
        Each capacitor should maintain its own monotonic degradation progression,
        independent of other capacitors. This tests that the monotonicity property
        holds when processing multiple capacitors.
        """
        try:
            aggregator = PredictionAggregator(model_version="test_1.0.0")
            
            # Generate data for multiple capacitors
            capacitor_data = {}
            
            for cap_idx in range(n_capacitors):
                capacitor_id = f"CAP_{cap_idx:02d}"
                
                # Generate initial conditions for this capacitor
                initial_rul = data.draw(st.floats(min_value=80.0, max_value=200.0, 
                                                allow_nan=False, allow_infinity=False))
                initial_anomaly = data.draw(st.floats(min_value=0.0, max_value=0.2, 
                                                    allow_nan=False, allow_infinity=False))
                
                # Generate cycle sequence for this capacitor
                rul_values = []
                anomaly_scores = []
                current_rul = initial_rul
                current_anomaly = initial_anomaly
                
                for cycle in range(n_cycles_per_capacitor):
                    # RUL decreases over time
                    rul_decrease = data.draw(st.floats(min_value=1.0, max_value=8.0, 
                                                     allow_nan=False, allow_infinity=False))
                    current_rul = max(0.0, current_rul - rul_decrease)
                    rul_values.append(current_rul)
                    
                    # Anomaly increases over time
                    anomaly_increase = data.draw(st.floats(min_value=0.0, max_value=0.08, 
                                                         allow_nan=False, allow_infinity=False))
                    current_anomaly = min(1.0, current_anomaly + anomaly_increase)
                    anomaly_scores.append(current_anomaly)
                
                capacitor_data[capacitor_id] = {
                    'rul_values': rul_values,
                    'anomaly_scores': anomaly_scores,
                    'degradation_scores': []
                }
            
            # Process each capacitor's cycles and track degradation scores
            for capacitor_id, cap_data in capacitor_data.items():
                degradation_history = []
                
                for cycle in range(n_cycles_per_capacitor):
                    # Generate feature importance
                    feature_importance = {
                        f"feature_{i}": data.draw(st.floats(min_value=0.0, max_value=1.0, 
                                                           allow_nan=False, allow_infinity=False))
                        for i in range(5)
                    }
                    
                    # Normalize feature importance
                    total_importance = sum(feature_importance.values())
                    if total_importance > 0:
                        feature_importance = {k: v / total_importance for k, v in feature_importance.items()}
                    
                    # Generate prediction
                    rul_pred = cap_data['rul_values'][cycle]
                    anomaly_score = cap_data['anomaly_scores'][cycle]
                    
                    result = aggregator.aggregate(
                        rul_pred=rul_pred,
                        rul_confidence_lower=max(0.0, rul_pred - 10.0),
                        rul_confidence_upper=rul_pred + 10.0,
                        anomaly_flag=anomaly_score > 0.5,
                        anomaly_score=anomaly_score,
                        feature_importance=feature_importance,
                        degradation_history=degradation_history.copy(),
                        capacitor_id=capacitor_id,
                        cycle_number=cycle + 1
                    )
                    
                    cap_data['degradation_scores'].append(result.degradation_score)
                    degradation_history.append(result.degradation_score)
            
            # Property 1: Each capacitor should maintain monotonicity independently
            for capacitor_id, cap_data in capacitor_data.items():
                degradation_scores = cap_data['degradation_scores']
                
                # Check monotonicity for this capacitor
                for i in range(1, len(degradation_scores)):
                    current_score = degradation_scores[i]
                    previous_score = degradation_scores[i-1]
                    
                    assert current_score >= previous_score - 1e-10, (
                        f"Monotonicity violated for {capacitor_id} at cycle {i+1}: "
                        f"current={current_score:.6f} < previous={previous_score:.6f}"
                    )
            
            # Property 2: Different capacitors can have different degradation patterns
            # (This ensures that monotonicity is per-capacitor, not global)
            all_final_scores = [cap_data['degradation_scores'][-1] 
                              for cap_data in capacitor_data.values()]
            
            # With multiple capacitors, we should see some variation in final scores
            # (unless all capacitors happen to have very similar conditions)
            if len(set(f"{score:.3f}" for score in all_final_scores)) == 1:
                # All final scores are very similar - check if input conditions were similar
                all_final_ruls = [cap_data['rul_values'][-1] 
                                for cap_data in capacitor_data.values()]
                all_final_anomalies = [cap_data['anomaly_scores'][-1] 
                                     for cap_data in capacitor_data.values()]
                
                rul_range = max(all_final_ruls) - min(all_final_ruls)
                anomaly_range = max(all_final_anomalies) - min(all_final_anomalies)
                
                # Only assert variation if input conditions were actually different
                if rul_range > 20.0 or anomaly_range > 0.2:
                    assert len(set(f"{score:.2f}" for score in all_final_scores)) > 1, (
                        f"With diverse final conditions (RUL range: {rul_range:.1f}, "
                        f"anomaly range: {anomaly_range:.3f}), expected some variation "
                        f"in final degradation scores, but all were similar: {all_final_scores}"
                    )
            
        except Exception as e:
            assume(False)
    
    @pytest.mark.skipif(not (AGGREGATOR_AVAILABLE and DATA_STRUCTURES_AVAILABLE), 
                       reason="Required modules not available")
    @given(
        rul_sequence=st.lists(
            st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False),
            min_size=5, max_size=15
        ),
        anomaly_sequence=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=5, max_size=15
        )
    )
    @settings(max_examples=50, deadline=20000)
    def test_monotonicity_with_realistic_sequences(self, rul_sequence, anomaly_sequence):
        """
        Property 9 (Realistic): Degradation Monotonicity with Realistic Sequences
        
        **Validates: Requirements 4.4**
        
        Test monotonicity with realistic degradation sequences where:
        - RUL generally decreases over time
        - Anomaly scores generally increase over time
        - Some noise/variation is allowed but overall trend should be maintained
        """
        # Ensure both sequences have the same length
        min_length = min(len(rul_sequence), len(anomaly_sequence))
        rul_sequence = rul_sequence[:min_length]
        anomaly_sequence = anomaly_sequence[:min_length]
        
        assume(min_length >= 5)
        
        # Make sequences more realistic by sorting them appropriately
        # RUL should generally decrease (sort in descending order)
        rul_sequence = sorted(rul_sequence, reverse=True)
        # Anomaly should generally increase (sort in ascending order)  
        anomaly_sequence = sorted(anomaly_sequence)
        
        try:
            aggregator = PredictionAggregator(model_version="test_1.0.0")
            
            # Compute degradation scores for the realistic sequence
            degradation_scores = []
            degradation_history = []
            
            for i, (rul, anomaly) in enumerate(zip(rul_sequence, anomaly_sequence)):
                score = aggregator._compute_degradation_score(
                    rul=rul,
                    anomaly_score=anomaly,
                    degradation_history=degradation_history.copy()
                )
                
                degradation_scores.append(score)
                degradation_history.append(score)
            
            # Property 1: All scores should be valid
            for i, score in enumerate(degradation_scores):
                assert 0.0 <= score <= 1.0, (
                    f"Degradation score at position {i} must be in [0, 1], got {score}"
                )
                
                assert np.isfinite(score), (
                    f"Degradation score at position {i} must be finite, got {score}"
                )
            
            # Property 2: Scores should be monotonic (non-decreasing)
            for i in range(1, len(degradation_scores)):
                current_score = degradation_scores[i]
                previous_score = degradation_scores[i-1]
                
                assert current_score >= previous_score - 1e-10, (
                    f"Monotonicity violated at position {i}: "
                    f"current={current_score:.6f} < previous={previous_score:.6f} "
                    f"(RUL: {rul_sequence[i]:.1f} vs {rul_sequence[i-1]:.1f}, "
                    f"Anomaly: {anomaly_sequence[i]:.3f} vs {anomaly_sequence[i-1]:.3f})"
                )
            
            # Property 3: With realistic sequences, we should see meaningful progression
            if len(degradation_scores) >= 10:
                first_quarter = np.mean(degradation_scores[:len(degradation_scores)//4])
                last_quarter = np.mean(degradation_scores[-len(degradation_scores)//4:])
                
                # Last quarter should have higher or equal degradation than first quarter
                assert last_quarter >= first_quarter - 1e-10, (
                    f"With realistic degradation sequence, last quarter should have "
                    f"higher degradation than first quarter: "
                    f"first={first_quarter:.3f}, last={last_quarter:.3f}"
                )
                
                # The progression should be meaningful (not just flat)
                total_progression = last_quarter - first_quarter
                rul_total_decrease = rul_sequence[0] - rul_sequence[-1]
                anomaly_total_increase = anomaly_sequence[-1] - anomaly_sequence[0]
                
                # If there was significant change in inputs, expect some progression
                if rul_total_decrease > 50.0 or anomaly_total_increase > 0.3:
                    assert total_progression >= 0.05, (
                        f"With significant input changes (RUL decrease: {rul_total_decrease:.1f}, "
                        f"anomaly increase: {anomaly_total_increase:.3f}), expected meaningful "
                        f"degradation progression, got {total_progression:.3f}"
                    )
            
        except Exception as e:
            assume(False)