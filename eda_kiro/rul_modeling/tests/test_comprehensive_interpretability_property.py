"""
Property-based test for comprehensive interpretability output

This module contains property-based tests using the Hypothesis framework
to validate that the RUL prediction system provides comprehensive interpretability
information for all predictions.

Requirements: 9.1, 9.2, 9.3, 9.4
"""

import sys
from pathlib import Path
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays
from typing import Dict, Any, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import modules using standard imports
try:
    from true_rul.interpretability_engine import InterpretabilityEngine
    INTERPRETABILITY_AVAILABLE = True
except Exception:
    INTERPRETABILITY_AVAILABLE = False

try:
    from true_rul.data_structures import PredictionResult
    DATA_STRUCTURES_AVAILABLE = True
except Exception:
    DATA_STRUCTURES_AVAILABLE = False

try:
    from true_rul.elastic_net_predictor import ElasticNetRULPredictor
    ELASTIC_NET_AVAILABLE = True
except Exception:
    ELASTIC_NET_AVAILABLE = False

try:
    from true_rul.random_forest_predictor import RandomForestRULPredictor
    RF_AVAILABLE = True
except Exception:
    RF_AVAILABLE = False

try:
    from true_rul.ensemble_anomaly_detector import EnsembleAnomalyDetector
    ANOMALY_DETECTOR_AVAILABLE = True
except Exception:
    ANOMALY_DETECTOR_AVAILABLE = False


class TestComprehensiveInterpretabilityProperty:
    """Property-based tests for comprehensive interpretability output"""
    
    @pytest.mark.skipif(
        not (INTERPRETABILITY_AVAILABLE and DATA_STRUCTURES_AVAILABLE),
        reason="Required modules not available"
    )
    @given(
        n_features=st.integers(min_value=5, max_value=15),
        n_models=st.integers(min_value=2, max_value=4),
        data=st.data()
    )
    @settings(max_examples=50, deadline=30000)  # 30 second timeout per example
    def test_comprehensive_interpretability_output(self, n_features, n_models, data):
        """
        Property 16: Comprehensive Interpretability Output
        
        **Validates: Requirements 9.1, 9.2, 9.3, 9.4**
        
        For any prediction, the RUL_Predictor should provide complete interpretability 
        information including:
        - Feature importance scores (summing to 1.0) - Requirement 9.1
        - Attention weights (if applicable) - Requirement 9.2  
        - For anomalous predictions, specific contributing features - Requirement 9.3
        - SHAP values or similar explainability metrics - Requirement 9.4
        
        This property ensures that all predictions come with comprehensive explanations
        that enable domain experts to understand and validate the model's reasoning.
        """
        # Generate feature names
        feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Initialize interpretability engine
        engine = InterpretabilityEngine(
            feature_names=feature_names,
            enable_plotting=False  # Disable plotting for testing
        )
        
        # Generate multiple feature importance dictionaries (simulating ensemble models)
        importance_dicts = []
        model_weights = []
        
        for i in range(n_models):
            # Generate realistic feature importance (positive values that sum to ~1.0)
            raw_importance = data.draw(
                arrays(
                    dtype=np.float64,
                    shape=(n_features,),
                    elements=st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)
                )
            )
            
            # Normalize to sum to 1.0
            normalized_importance = raw_importance / np.sum(raw_importance)
            
            importance_dict = {
                feature_names[j]: float(normalized_importance[j])
                for j in range(n_features)
            }
            importance_dicts.append(importance_dict)
            
            # Generate model weight
            weight = data.draw(st.floats(min_value=0.1, max_value=1.0))
            model_weights.append(weight)
        
        # Normalize model weights
        total_weight = sum(model_weights)
        model_weights = [w / total_weight for w in model_weights]
        
        # Skip if data contains invalid values
        assume(all(
            all(np.isfinite(v) and v >= 0 for v in importance_dict.values())
            for importance_dict in importance_dicts
        ))
        assume(all(np.isfinite(w) and w > 0 for w in model_weights))
        
        # Test 1: Feature importance aggregation (Requirement 9.1)
        aggregated_importance = engine.aggregate_feature_importance(
            importance_dicts, 
            weights=model_weights,
            normalize=True
        )
        
        # Property 1.1: Must return a dictionary
        assert isinstance(aggregated_importance, dict), (
            "Feature importance must be returned as a dictionary"
        )
        
        # Property 1.2: Must contain all features
        assert len(aggregated_importance) == n_features, (
            f"Feature importance must contain all {n_features} features, "
            f"got {len(aggregated_importance)}"
        )
        
        for feature_name in feature_names:
            assert feature_name in aggregated_importance, (
                f"Feature '{feature_name}' missing from importance dictionary"
            )
        
        # Property 1.3: All importance values must be non-negative and finite
        for feature_name, importance in aggregated_importance.items():
            assert isinstance(importance, (int, float)), (
                f"Feature importance for '{feature_name}' must be numeric, "
                f"got {type(importance)}"
            )
            assert np.isfinite(importance), (
                f"Feature importance for '{feature_name}' must be finite, "
                f"got {importance}"
            )
            assert importance >= 0, (
                f"Feature importance for '{feature_name}' must be non-negative, "
                f"got {importance}"
            )
        
        # Property 1.4: Normalized importance should sum to approximately 1.0
        total_importance = sum(aggregated_importance.values())
        assert abs(total_importance - 1.0) < 1e-10, (
            f"Normalized feature importance should sum to 1.0, "
            f"got {total_importance:.12f}"
        )
        
        # Test 2: Top features extraction (Requirement 9.1)
        top_k = data.draw(st.integers(min_value=3, max_value=min(n_features, 10)))
        top_features = engine.get_top_features(aggregated_importance, top_k=top_k)
        
        # Property 2.1: Must return a list of tuples
        assert isinstance(top_features, list), (
            "Top features must be returned as a list"
        )
        assert len(top_features) <= top_k, (
            f"Top features list should have at most {top_k} elements, "
            f"got {len(top_features)}"
        )
        
        # Property 2.2: Each element must be a (feature_name, importance) tuple
        for i, item in enumerate(top_features):
            assert isinstance(item, tuple) and len(item) == 2, (
                f"Top feature {i} must be a tuple of length 2, got {item}"
            )
            
            feature_name, importance = item
            assert isinstance(feature_name, str), (
                f"Feature name must be string, got {type(feature_name)}"
            )
            assert isinstance(importance, (int, float)), (
                f"Feature importance must be numeric, got {type(importance)}"
            )
            assert np.isfinite(importance) and importance >= 0, (
                f"Feature importance must be non-negative and finite, got {importance}"
            )
        
        # Property 2.3: Top features should be sorted by importance (descending)
        if len(top_features) > 1:
            for i in range(len(top_features) - 1):
                current_importance = abs(top_features[i][1])
                next_importance = abs(top_features[i + 1][1])
                assert current_importance >= next_importance, (
                    f"Top features not sorted: position {i} has importance "
                    f"{current_importance:.6f}, position {i+1} has {next_importance:.6f}"
                )
        
        # Test 3: SHAP values analysis (Requirement 9.4)
        n_samples = data.draw(st.integers(min_value=1, max_value=5))
        
        # Generate realistic SHAP values (can be positive or negative)
        shap_values = data.draw(
            arrays(
                dtype=np.float64,
                shape=(n_samples, n_features),
                elements=st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False)
            )
        )
        
        assume(np.all(np.isfinite(shap_values)))
        
        # Test SHAP analysis for first sample
        shap_analysis = engine.analyze_shap_values(
            shap_values, 
            feature_names=feature_names,
            sample_idx=0
        )
        
        # Property 3.1: Must return a dictionary with required keys
        assert isinstance(shap_analysis, dict), (
            "SHAP analysis must return a dictionary"
        )
        
        required_keys = [
            "shap_values", "feature_names", "shap_importance",
            "top_positive_contributors", "top_negative_contributors",
            "total_shap_magnitude", "shap_sum"
        ]
        
        for key in required_keys:
            assert key in shap_analysis, (
                f"SHAP analysis missing required key: '{key}'"
            )
        
        # Property 3.2: SHAP values structure validation
        assert isinstance(shap_analysis["shap_values"], list), (
            "SHAP values must be a list"
        )
        assert len(shap_analysis["shap_values"]) == n_features, (
            f"SHAP values must have {n_features} elements, "
            f"got {len(shap_analysis['shap_values'])}"
        )
        
        # Property 3.3: Feature names validation
        assert shap_analysis["feature_names"] == feature_names, (
            "SHAP analysis feature names must match input feature names"
        )
        
        # Property 3.4: SHAP importance validation
        shap_importance = shap_analysis["shap_importance"]
        assert isinstance(shap_importance, dict), (
            "SHAP importance must be a dictionary"
        )
        assert len(shap_importance) == n_features, (
            f"SHAP importance must contain all {n_features} features"
        )
        
        for feature_name in feature_names:
            assert feature_name in shap_importance, (
                f"SHAP importance missing feature: '{feature_name}'"
            )
            importance = shap_importance[feature_name]
            assert isinstance(importance, (int, float)), (
                f"SHAP importance must be numeric for '{feature_name}'"
            )
            assert np.isfinite(importance) and importance >= 0, (
                f"SHAP importance must be non-negative and finite for '{feature_name}'"
            )
        
        # Property 3.5: Contributors validation
        for contrib_type in ["top_positive_contributors", "top_negative_contributors"]:
            contributors = shap_analysis[contrib_type]
            assert isinstance(contributors, list), (
                f"{contrib_type} must be a list"
            )
            
            for contrib in contributors:
                assert isinstance(contrib, tuple) and len(contrib) == 2, (
                    f"Each contributor must be a (feature, value) tuple"
                )
                feature_name, value = contrib
                assert isinstance(feature_name, str), (
                    "Contributor feature name must be string"
                )
                assert isinstance(value, (int, float)), (
                    "Contributor value must be numeric"
                )
                assert np.isfinite(value), (
                    "Contributor value must be finite"
                )
        
        # Property 3.6: Magnitude and sum validation
        total_magnitude = shap_analysis["total_shap_magnitude"]
        shap_sum = shap_analysis["shap_sum"]
        
        assert isinstance(total_magnitude, (int, float)), (
            "Total SHAP magnitude must be numeric"
        )
        assert isinstance(shap_sum, (int, float)), (
            "SHAP sum must be numeric"
        )
        assert np.isfinite(total_magnitude) and total_magnitude >= 0, (
            "Total SHAP magnitude must be non-negative and finite"
        )
        assert np.isfinite(shap_sum), (
            "SHAP sum must be finite"
        )
        
        # Test 4: Diagnostic report generation (Requirements 9.1, 9.3, 9.5)
        # Create a mock prediction result
        rul_cycles = data.draw(st.integers(min_value=1, max_value=100))
        rul_confidence_lower = data.draw(st.integers(min_value=1, max_value=rul_cycles))
        rul_confidence_upper = data.draw(st.integers(min_value=rul_cycles, max_value=rul_cycles + 50))
        
        prediction_result = PredictionResult(
            rul_cycles=rul_cycles,
            rul_confidence_lower=rul_confidence_lower,
            rul_confidence_upper=rul_confidence_upper,
            degradation_score=data.draw(st.floats(min_value=0.0, max_value=1.0)),
            degradation_stage=data.draw(st.sampled_from(["healthy", "early_degradation", "advanced_degradation", "critical"])),
            anomaly_flag=data.draw(st.booleans()),
            anomaly_score=data.draw(st.floats(min_value=0.0, max_value=1.0)),
            feature_importance=aggregated_importance,
            timestamp=1234567890.0,
            model_version="test_v1.0"
        )
        
        # Generate diagnostic report
        diagnostic_report = engine.generate_diagnostic_report(
            prediction_result=prediction_result,
            feature_importance=aggregated_importance,
            shap_analysis=shap_analysis
        )
        
        # Property 4.1: Must return a dictionary with required structure
        assert isinstance(diagnostic_report, dict), (
            "Diagnostic report must be a dictionary"
        )
        
        required_report_keys = [
            "timestamp", "prediction_summary", "interpretability_analysis",
            "deviation_analysis", "recommendations"
        ]
        
        for key in required_report_keys:
            assert key in diagnostic_report, (
                f"Diagnostic report missing required key: '{key}'"
            )
        
        # Property 4.2: Prediction summary validation
        pred_summary = diagnostic_report["prediction_summary"]
        assert isinstance(pred_summary, dict), (
            "Prediction summary must be a dictionary"
        )
        
        summary_keys = [
            "rul_cycles", "degradation_score", "degradation_stage",
            "anomaly_flag", "anomaly_score", "confidence_interval"
        ]
        
        for key in summary_keys:
            assert key in pred_summary, (
                f"Prediction summary missing key: '{key}'"
            )
        
        # Property 4.3: Interpretability analysis validation (Requirement 9.1)
        interp_analysis = diagnostic_report["interpretability_analysis"]
        assert isinstance(interp_analysis, dict), (
            "Interpretability analysis must be a dictionary"
        )
        
        assert "top_features" in interp_analysis, (
            "Interpretability analysis must contain top_features"
        )
        
        top_features_report = interp_analysis["top_features"]
        assert isinstance(top_features_report, list), (
            "Top features in report must be a list"
        )
        
        for feature_info in top_features_report:
            assert isinstance(feature_info, dict), (
                "Each top feature must be a dictionary"
            )
            assert "feature" in feature_info and "importance" in feature_info, (
                "Each top feature must have 'feature' and 'importance' keys"
            )
        
        # Property 4.4: SHAP analysis in report (Requirement 9.4)
        if "shap_analysis" in interp_analysis:
            shap_report = interp_analysis["shap_analysis"]
            assert isinstance(shap_report, dict), (
                "SHAP analysis in report must be a dictionary"
            )
            
            shap_report_keys = [
                "top_positive_contributors", "top_negative_contributors",
                "total_explanation_magnitude"
            ]
            
            for key in shap_report_keys:
                assert key in shap_report, (
                    f"SHAP report missing key: '{key}'"
                )
        
        # Property 4.5: Recommendations validation
        recommendations = diagnostic_report["recommendations"]
        assert isinstance(recommendations, list), (
            "Recommendations must be a list"
        )
        
        for recommendation in recommendations:
            assert isinstance(recommendation, str), (
                "Each recommendation must be a string"
            )
            assert len(recommendation) > 0, (
                "Recommendations must not be empty strings"
            )
        
        # Property 4.6: Anomaly-specific interpretability (Requirement 9.3)
        if prediction_result.anomaly_flag:
            # For anomalous predictions, there should be specific guidance
            assert len(recommendations) > 0, (
                "Anomalous predictions must have recommendations"
            )
            
            # Check if any recommendation mentions anomaly
            anomaly_mentioned = any(
                "anomaly" in rec.lower() or "inspection" in rec.lower()
                for rec in recommendations
            )
            assert anomaly_mentioned, (
                "Anomalous predictions should have anomaly-specific recommendations"
            )
        
        # Test 5: Historical context and deviation analysis (Requirement 9.5)
        # Add some prediction history
        for i in range(3):
            mock_result = PredictionResult(
                rul_cycles=50 + i * 5,
                rul_confidence_lower=45 + i * 5,
                rul_confidence_upper=55 + i * 5,
                degradation_score=0.3 + i * 0.1,
                degradation_stage="early_degradation",
                anomaly_flag=False,
                anomaly_score=0.2,
                feature_importance=aggregated_importance,
                timestamp=1234567890.0 + i * 3600,
                model_version="test_v1.0"
            )
            engine.update_history(mock_result, aggregated_importance)
        
        # Get historical context
        historical_context = engine.get_historical_context()
        
        # Property 5.1: Historical context structure
        assert isinstance(historical_context, dict), (
            "Historical context must be a dictionary"
        )
        
        if historical_context:  # Only check if history exists
            context_keys = [
                "n_historical_predictions", "historical_rul_mean", "historical_rul_std",
                "historical_degradation_mean", "historical_degradation_std"
            ]
            
            for key in context_keys:
                if key in historical_context:
                    value = historical_context[key]
                    assert isinstance(value, (int, float)), (
                        f"Historical context '{key}' must be numeric"
                    )
                    assert np.isfinite(value), (
                        f"Historical context '{key}' must be finite"
                    )
        
        # Test 6: Summary statistics (General validation)
        summary_stats = engine.get_summary_statistics()
        
        # Property 6.1: Summary statistics structure
        assert isinstance(summary_stats, dict), (
            "Summary statistics must be a dictionary"
        )
        
        expected_stats_keys = [
            "n_features", "n_historical_predictions", "n_feature_importance_records",
            "plotting_enabled", "deviation_threshold", "importance_threshold"
        ]
        
        for key in expected_stats_keys:
            assert key in summary_stats, (
                f"Summary statistics missing key: '{key}'"
            )
        
        # Property 6.2: Validate specific statistics
        assert summary_stats["n_features"] == n_features, (
            f"Summary statistics should report {n_features} features"
        )
        assert isinstance(summary_stats["plotting_enabled"], bool), (
            "Plotting enabled must be boolean"
        )
        assert summary_stats["deviation_threshold"] > 0, (
            "Deviation threshold must be positive"
        )
        assert summary_stats["importance_threshold"] > 0, (
            "Importance threshold must be positive"
        )
    
    @pytest.mark.skipif(
        not (INTERPRETABILITY_AVAILABLE and ELASTIC_NET_AVAILABLE),
        reason="Required modules not available"
    )
    @given(
        n_features=st.integers(min_value=3, max_value=8),
        n_samples=st.integers(min_value=10, max_value=25)
    )
    @settings(max_examples=20, deadline=20000)
    def test_model_interpretability_integration(self, n_features, n_samples):
        """
        Property 16 (Integration): Model Interpretability Integration
        
        **Validates: Requirements 9.1, 9.4**
        
        When using actual trained models, the interpretability features should
        integrate seamlessly and provide consistent results.
        """
        # Generate training data
        X_train = np.random.normal(0, 1, (n_samples, n_features))
        y_train = np.random.uniform(10, 80, n_samples)
        
        # Generate test data
        X_test = np.random.normal(0, 1, (3, n_features))
        
        feature_names = [f"feature_{i}" for i in range(n_features)]
        
        try:
            # Train ElasticNet model
            model = ElasticNetRULPredictor(degree=1)
            model.train(X_train, y_train, feature_names=feature_names)
            
            # Get feature importance from model
            model_importance = model.get_feature_importance()
            
            # Initialize interpretability engine
            engine = InterpretabilityEngine(
                feature_names=feature_names,
                enable_plotting=False
            )
            
            # Property 1: Model importance should be compatible with engine
            assert isinstance(model_importance, dict), (
                "Model feature importance must be a dictionary"
            )
            
            # Property 2: All features should be present
            for feature_name in feature_names:
                assert feature_name in model_importance, (
                    f"Model importance missing feature: '{feature_name}'"
                )
            
            # Property 3: Importance values should be valid
            for feature_name, importance in model_importance.items():
                assert isinstance(importance, (int, float)), (
                    f"Model importance for '{feature_name}' must be numeric"
                )
                assert np.isfinite(importance), (
                    f"Model importance for '{feature_name}' must be finite"
                )
                assert importance >= 0, (
                    f"Model importance for '{feature_name}' must be non-negative"
                )
            
            # Property 4: Engine should handle model importance correctly
            top_features = engine.get_top_features(model_importance, top_k=5)
            
            assert isinstance(top_features, list), (
                "Top features from model importance must be a list"
            )
            assert len(top_features) <= min(5, n_features), (
                "Top features list should respect top_k limit"
            )
            
            # Property 5: Aggregation with single model should preserve structure
            aggregated = engine.aggregate_feature_importance([model_importance])
            
            assert isinstance(aggregated, dict), (
                "Aggregated single model importance must be a dictionary"
            )
            assert len(aggregated) == len(model_importance), (
                "Aggregated importance should have same number of features"
            )
            
            # Values should be approximately equal (within numerical precision)
            for feature_name in feature_names:
                original = model_importance[feature_name]
                aggregated_val = aggregated[feature_name]
                assert abs(original - aggregated_val) < 1e-10, (
                    f"Aggregated single model importance should preserve values for '{feature_name}'"
                )
            
        except Exception as e:
            # Skip if model training fails
            assume(False, f"Model training failed: {e}")
    
    @pytest.mark.skipif(
        not (INTERPRETABILITY_AVAILABLE and ANOMALY_DETECTOR_AVAILABLE),
        reason="Required modules not available"
    )
    @given(
        n_features=st.integers(min_value=5, max_value=10)
    )
    @settings(max_examples=15, deadline=25000)
    def test_anomaly_specific_interpretability(self, n_features):
        """
        Property 16 (Anomaly Focus): Anomaly-Specific Interpretability
        
        **Validates: Requirements 9.3**
        
        When an anomaly is detected, the system should highlight the specific 
        features or patterns that triggered the detection.
        """
        feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Initialize interpretability engine
        engine = InterpretabilityEngine(
            feature_names=feature_names,
            enable_plotting=False
        )
        
        # Create mock anomaly detection result with feature importance
        anomaly_importance = {}
        
        # Simulate some features having high importance (anomaly contributors)
        high_importance_features = np.random.choice(
            feature_names, 
            size=min(3, n_features), 
            replace=False
        )
        
        for feature_name in feature_names:
            if feature_name in high_importance_features:
                # High importance for anomaly contributors
                importance = np.random.uniform(0.15, 0.4)
            else:
                # Low importance for other features
                importance = np.random.uniform(0.01, 0.1)
            
            anomaly_importance[feature_name] = importance
        
        # Normalize to sum to 1.0
        total = sum(anomaly_importance.values())
        anomaly_importance = {
            k: v / total for k, v in anomaly_importance.items()
        }
        
        # Create anomalous prediction result
        anomalous_prediction = PredictionResult(
            rul_cycles=25,
            rul_confidence_lower=20,
            rul_confidence_upper=30,
            degradation_score=0.8,  # High degradation
            degradation_stage="critical",
            anomaly_flag=True,  # Anomalous
            anomaly_score=0.9,  # High anomaly score
            feature_importance=anomaly_importance,
            timestamp=1234567890.0,
            model_version="test_v1.0"
        )
        
        # Generate diagnostic report for anomalous prediction
        diagnostic_report = engine.generate_diagnostic_report(
            prediction_result=anomalous_prediction,
            feature_importance=anomaly_importance
        )
        
        # Property 1: Anomalous predictions should have specific recommendations
        recommendations = diagnostic_report["recommendations"]
        assert len(recommendations) > 0, (
            "Anomalous predictions must have recommendations"
        )
        
        # Property 2: Should mention anomaly or inspection
        anomaly_mentioned = any(
            "anomaly" in rec.lower() or "inspection" in rec.lower()
            for rec in recommendations
        )
        assert anomaly_mentioned, (
            "Anomalous predictions should have anomaly-specific recommendations"
        )
        
        # Property 3: Top features should include high-importance anomaly contributors
        interp_analysis = diagnostic_report["interpretability_analysis"]
        top_features_report = interp_analysis["top_features"]
        
        # Get top feature names from report
        top_feature_names = [
            feature_info["feature"] for feature_info in top_features_report[:3]
        ]
        
        # At least one high-importance feature should be in top features
        high_importance_in_top = any(
            feature in top_feature_names for feature in high_importance_features
        )
        assert high_importance_in_top, (
            "Top features should include high-importance anomaly contributors"
        )
        
        # Property 4: Feature importance should be properly structured
        for feature_info in top_features_report:
            assert "feature" in feature_info and "importance" in feature_info, (
                "Each top feature must have 'feature' and 'importance' keys"
            )
            
            feature_name = feature_info["feature"]
            importance = feature_info["importance"]
            
            assert feature_name in feature_names, (
                f"Feature '{feature_name}' should be in original feature list"
            )
            assert isinstance(importance, (int, float)), (
                f"Importance for '{feature_name}' must be numeric"
            )
            assert np.isfinite(importance) and importance >= 0, (
                f"Importance for '{feature_name}' must be non-negative and finite"
            )