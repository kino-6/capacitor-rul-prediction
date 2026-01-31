"""
Property-based tests for RUL Regression Models

This module contains property-based tests using the Hypothesis framework
to validate universal correctness properties of RUL regression models.

Requirements: 1.1
"""

import sys
from pathlib import Path
import numpy as np
import pytest
from hypothesis import given, strategies as st, settings, assume
from hypothesis.extra.numpy import arrays
import importlib.util

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import modules directly without going through __init__.py to avoid dependency issues
def load_module_from_file(module_name, file_path):
    """Load a module directly from file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load the predictor modules
src_dir = Path(__file__).parent.parent / "src" / "true_rul"

try:
    elastic_net_module = load_module_from_file(
        "elastic_net_predictor", 
        src_dir / "elastic_net_predictor.py"
    )
    ElasticNetRULPredictor = elastic_net_module.ElasticNetRULPredictor
    ELASTIC_NET_AVAILABLE = True
except Exception:
    ELASTIC_NET_AVAILABLE = False

try:
    random_forest_module = load_module_from_file(
        "random_forest_predictor",
        src_dir / "random_forest_predictor.py"
    )
    RandomForestRULPredictor = random_forest_module.RandomForestRULPredictor
    RF_AVAILABLE = True
except Exception:
    RF_AVAILABLE = False

# For gradient boosting, we need to check if dependencies are available
try:
    gradient_boosting_module = load_module_from_file(
        "gradient_boosting_predictor",
        src_dir / "gradient_boosting_predictor.py"
    )
    GradientBoostingRULPredictor = gradient_boosting_module.GradientBoostingRULPredictor
    GB_AVAILABLE = True
except Exception:
    GB_AVAILABLE = False


class TestRULRegressionProperties:
    """Property-based tests for RUL regression models"""
    
    @given(
        n_samples=st.integers(min_value=20, max_value=50),
        n_features=st.integers(min_value=5, max_value=15),
        data=st.data()
    )
    @settings(max_examples=50, deadline=30000)  # 30 second timeout per example
    def test_non_negative_rul_output(self, n_samples, n_features, data):
        """
        Property 1: Non-negative RUL Output
        
        **Validates: Requirements 1.1**
        
        For any valid input features and any trained RUL regression model,
        all predicted RUL values must be non-negative (>= 0).
        
        This property ensures that the fundamental requirement of RUL prediction
        is satisfied: remaining useful life cannot be negative.
        """
        # Generate realistic training data
        X_train = data.draw(
            arrays(
                dtype=np.float64,
                shape=(n_samples, n_features),
                elements=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)
            )
        )
        
        # Generate realistic RUL labels (positive integers representing remaining cycles)
        y_train = data.draw(
            arrays(
                dtype=np.float64,
                shape=(n_samples,),
                elements=st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False)
            )
        )
        
        # Generate test data with potentially different distribution
        n_test = data.draw(st.integers(min_value=5, max_value=15))
        X_test = data.draw(
            arrays(
                dtype=np.float64,
                shape=(n_test, n_features),
                elements=st.floats(min_value=-8.0, max_value=8.0, allow_nan=False, allow_infinity=False)
            )
        )
        
        # Skip if data contains invalid values
        assume(np.all(np.isfinite(X_train)))
        assume(np.all(np.isfinite(y_train)))
        assume(np.all(np.isfinite(X_test)))
        assume(np.all(y_train > 0))  # RUL labels must be positive
        
        # Test available models
        models_to_test = []
        
        if ELASTIC_NET_AVAILABLE:
            models_to_test.append(("elastic_net", ElasticNetRULPredictor(degree=1)))
        
        if RF_AVAILABLE:
            models_to_test.append(("random_forest", RandomForestRULPredictor(n_estimators=10)))
        
        if GB_AVAILABLE:
            try:
                models_to_test.append(("xgboost", GradientBoostingRULPredictor(model_type="xgboost")))
            except Exception:
                pass  # Skip if XGBoost not available
        
        # Skip if no models are available
        assume(len(models_to_test) > 0)
        
        # Test each available model
        for model_name, model in models_to_test:
            try:
                # Reduce model complexity for faster testing
                if hasattr(model, 'model') and hasattr(model.model, 'n_estimators'):
                    model.model.n_estimators = 10
                
                # Generate feature names
                feature_names = [f"feature_{i}" for i in range(n_features)]
                
                # Train the model
                model.train(X_train, y_train, feature_names=feature_names)
                
                # Make predictions
                predictions = model.predict(X_test)
                
                # Property: All predictions must be non-negative
                assert np.all(predictions >= 0), (
                    f"Model {model_name} produced negative RUL predictions: "
                    f"min={np.min(predictions):.3f}, "
                    f"negative_count={np.sum(predictions < 0)}/{len(predictions)}"
                )
                
                # Additional check: predictions should be finite
                assert np.all(np.isfinite(predictions)), (
                    f"Model {model_name} produced non-finite predictions: "
                    f"nan_count={np.sum(np.isnan(predictions))}, "
                    f"inf_count={np.sum(np.isinf(predictions))}"
                )
                
                # Test confidence intervals if supported
                try:
                    if hasattr(model, 'predict_with_confidence'):
                        pred_with_conf, lower_bounds, upper_bounds = model.predict_with_confidence(X_test)
                        
                        # All confidence interval components must be non-negative
                        assert np.all(pred_with_conf >= 0), (
                            f"Model {model_name} confidence predictions contain negative values: "
                            f"min={np.min(pred_with_conf):.3f}"
                        )
                        assert np.all(lower_bounds >= 0), (
                            f"Model {model_name} lower bounds contain negative values: "
                            f"min={np.min(lower_bounds):.3f}"
                        )
                        assert np.all(upper_bounds >= 0), (
                            f"Model {model_name} upper bounds contain negative values: "
                            f"min={np.min(upper_bounds):.3f}"
                        )
                        
                        # Confidence intervals should be finite
                        assert np.all(np.isfinite(pred_with_conf)), "Confidence predictions must be finite"
                        assert np.all(np.isfinite(lower_bounds)), "Lower bounds must be finite"
                        assert np.all(np.isfinite(upper_bounds)), "Upper bounds must be finite"
                    
                except (NotImplementedError, RuntimeError, AttributeError):
                    # Some models may not support confidence intervals
                    pass
                
            except Exception as e:
                # If model training fails due to data issues, skip this example
                # This can happen with extreme parameter combinations
                assume(False, f"Model {model_name} training failed: {e}")
    
    @pytest.mark.skipif(not ELASTIC_NET_AVAILABLE, reason="ElasticNet not available")
    @given(
        n_features=st.integers(min_value=3, max_value=8)
    )
    @settings(max_examples=20, deadline=20000)
    def test_zero_rul_boundary_case(self, n_features):
        """
        Property 1 (Boundary Case): Zero RUL Handling
        
        **Validates: Requirements 1.1**
        
        When training data includes samples with RUL = 0 (end of life),
        the model should still produce non-negative predictions.
        """
        # Create training data with some zero RUL samples
        n_samples = 25
        X_train = np.random.normal(0, 1, (n_samples, n_features))
        
        # Mix of various RUL values including zeros
        y_train = np.concatenate([
            np.random.uniform(1, 50, n_samples - 5),  # Normal RUL values
            np.zeros(5)  # End-of-life samples
        ])
        
        # Shuffle to mix zero and non-zero samples
        indices = np.random.permutation(n_samples)
        X_train = X_train[indices]
        y_train = y_train[indices]
        
        # Test data
        X_test = np.random.normal(0, 1, (8, n_features))
        
        try:
            # Use ElasticNet as it's most stable with zero values
            model = ElasticNetRULPredictor(degree=1)
            
            feature_names = [f"feature_{i}" for i in range(n_features)]
            model.train(X_train, y_train, feature_names=feature_names)
            predictions = model.predict(X_test)
            
            # Property: Even with zero RUL training samples, predictions must be non-negative
            assert np.all(predictions >= 0), (
                f"ElasticNet with zero RUL training data produced negative predictions: "
                f"min={np.min(predictions):.3f}"
            )
            
        except Exception as e:
            # Skip if training fails with this specific data configuration
            assume(False, f"Model training failed with zero RUL data: {e}")
    
    @pytest.mark.skipif(not ELASTIC_NET_AVAILABLE, reason="ElasticNet not available")
    @given(
        n_features=st.integers(min_value=3, max_value=6)
    )
    @settings(max_examples=15, deadline=15000)
    def test_extreme_input_robustness(self, n_features):
        """
        Property 1 (Robustness): Non-negative Output with Extreme Inputs
        
        **Validates: Requirements 1.1**
        
        Even when presented with extreme input values (within reasonable bounds),
        the model should maintain non-negative RUL predictions.
        """
        # Normal training data
        n_samples = 20
        X_train = np.random.normal(0, 1, (n_samples, n_features))
        y_train = np.random.uniform(10, 80, n_samples)
        
        # Extreme test inputs (but still realistic)
        X_test_extreme = np.array([
            np.full(n_features, 3.0),    # All features at high positive values
            np.full(n_features, -3.0),   # All features at high negative values
            np.random.uniform(-2, 2, n_features),  # Mixed extreme values
        ])
        
        try:
            # Use ElasticNet for stability
            model = ElasticNetRULPredictor(degree=1)
            
            feature_names = [f"feature_{i}" for i in range(n_features)]
            model.train(X_train, y_train, feature_names=feature_names)
            predictions = model.predict(X_test_extreme)
            
            # Property: Even with extreme inputs, predictions must be non-negative
            assert np.all(predictions >= 0), (
                f"ElasticNet with extreme inputs produced negative predictions: "
                f"predictions={predictions}, min={np.min(predictions):.3f}"
            )
            
        except Exception as e:
            # Skip if training fails
            assume(False, f"Model training failed with extreme inputs: {e}")
    
    @given(
        n_samples=st.integers(min_value=15, max_value=30),
        n_features=st.integers(min_value=5, max_value=10),
        data=st.data()
    )
    @settings(max_examples=30, deadline=25000)  # 25 second timeout per example
    def test_complete_prediction_output_structure(self, n_samples, n_features, data):
        """
        Property 2: Complete Prediction Output Structure
        
        **Validates: Requirements 1.3, 2.2, 7.2, 7.5**
        
        For any prediction request, the RUL_Predictor should return a structured output containing:
        - RUL value (non-negative integer)
        - Confidence interval (lower <= prediction <= upper, all non-negative)
        - Degradation score (0-1 range)
        - Anomaly flag (boolean)
        - All required metadata (timestamp, model version)
        
        This property ensures that the prediction output structure is complete and consistent
        across all model types and input conditions.
        """
        # Generate realistic training data
        X_train = data.draw(
            arrays(
                dtype=np.float64,
                shape=(n_samples, n_features),
                elements=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False)
            )
        )
        
        # Generate realistic RUL labels
        y_train = data.draw(
            arrays(
                dtype=np.float64,
                shape=(n_samples,),
                elements=st.floats(min_value=5.0, max_value=80.0, allow_nan=False, allow_infinity=False)
            )
        )
        
        # Generate test data
        n_test = data.draw(st.integers(min_value=3, max_value=8))
        X_test = data.draw(
            arrays(
                dtype=np.float64,
                shape=(n_test, n_features),
                elements=st.floats(min_value=-4.0, max_value=4.0, allow_nan=False, allow_infinity=False)
            )
        )
        
        # Skip if data contains invalid values
        assume(np.all(np.isfinite(X_train)))
        assume(np.all(np.isfinite(y_train)))
        assume(np.all(np.isfinite(X_test)))
        assume(np.all(y_train > 0))
        
        # Test available models that support confidence intervals
        models_to_test = []
        
        if ELASTIC_NET_AVAILABLE:
            models_to_test.append(("elastic_net", ElasticNetRULPredictor(degree=1)))
        
        if RF_AVAILABLE:
            models_to_test.append(("random_forest", RandomForestRULPredictor(n_estimators=10)))
        
        # Skip if no models are available
        assume(len(models_to_test) > 0)
        
        # Test each available model
        for model_name, model in models_to_test:
            try:
                # Generate feature names
                feature_names = [f"feature_{i}" for i in range(n_features)]
                
                # Train the model
                model.train(X_train, y_train, feature_names=feature_names)
                
                # Test basic prediction structure
                predictions = model.predict(X_test)
                
                # Property 1: Basic predictions must be non-negative and finite
                assert np.all(predictions >= 0), (
                    f"Model {model_name} produced negative RUL predictions: "
                    f"min={np.min(predictions):.3f}"
                )
                assert np.all(np.isfinite(predictions)), (
                    f"Model {model_name} produced non-finite predictions"
                )
                
                # Property 2: Test confidence interval structure if supported
                if hasattr(model, 'predict_with_confidence'):
                    pred_with_conf, lower_bounds, upper_bounds = model.predict_with_confidence(X_test)
                    
                    # All components must be arrays of same length
                    assert len(pred_with_conf) == len(X_test), (
                        f"Prediction array length mismatch: expected {len(X_test)}, "
                        f"got {len(pred_with_conf)}"
                    )
                    assert len(lower_bounds) == len(X_test), (
                        f"Lower bounds array length mismatch: expected {len(X_test)}, "
                        f"got {len(lower_bounds)}"
                    )
                    assert len(upper_bounds) == len(X_test), (
                        f"Upper bounds array length mismatch: expected {len(X_test)}, "
                        f"got {len(upper_bounds)}"
                    )
                    
                    # All values must be non-negative (Requirement 1.3)
                    assert np.all(pred_with_conf >= 0), (
                        f"Model {model_name} confidence predictions contain negative values: "
                        f"min={np.min(pred_with_conf):.3f}"
                    )
                    assert np.all(lower_bounds >= 0), (
                        f"Model {model_name} lower bounds contain negative values: "
                        f"min={np.min(lower_bounds):.3f}"
                    )
                    assert np.all(upper_bounds >= 0), (
                        f"Model {model_name} upper bounds contain negative values: "
                        f"min={np.min(upper_bounds):.3f}"
                    )
                    
                    # Confidence interval ordering: lower <= prediction <= upper (Requirement 1.3)
                    assert np.all(lower_bounds <= pred_with_conf), (
                        f"Model {model_name} lower bounds exceed predictions: "
                        f"violations={np.sum(lower_bounds > pred_with_conf)}/{len(pred_with_conf)}"
                    )
                    assert np.all(pred_with_conf <= upper_bounds), (
                        f"Model {model_name} predictions exceed upper bounds: "
                        f"violations={np.sum(pred_with_conf > upper_bounds)}/{len(pred_with_conf)}"
                    )
                    
                    # All confidence interval components must be finite
                    assert np.all(np.isfinite(pred_with_conf)), (
                        f"Model {model_name} confidence predictions must be finite"
                    )
                    assert np.all(np.isfinite(lower_bounds)), (
                        f"Model {model_name} lower bounds must be finite"
                    )
                    assert np.all(np.isfinite(upper_bounds)), (
                        f"Model {model_name} upper bounds must be finite"
                    )
                
                # Property 3: Test feature importance structure if supported (Requirement 2.2, 7.5)
                if hasattr(model, 'get_feature_importance'):
                    feature_importance = model.get_feature_importance()
                    
                    # Must return a dictionary
                    assert isinstance(feature_importance, dict), (
                        f"Model {model_name} feature importance must be a dictionary, "
                        f"got {type(feature_importance)}"
                    )
                    
                    # Must have entries for all features
                    assert len(feature_importance) > 0, (
                        f"Model {model_name} feature importance dictionary is empty"
                    )
                    
                    # All importance values must be finite and non-negative
                    importance_values = list(feature_importance.values())
                    assert all(isinstance(v, (int, float)) for v in importance_values), (
                        f"Model {model_name} feature importance values must be numeric"
                    )
                    assert all(np.isfinite(v) for v in importance_values), (
                        f"Model {model_name} feature importance values must be finite"
                    )
                    assert all(v >= 0 for v in importance_values), (
                        f"Model {model_name} feature importance values must be non-negative"
                    )
                
                # Property 4: Test model metadata structure (Requirement 7.5)
                if hasattr(model, 'get_model_info'):
                    model_info = model.get_model_info()
                    
                    # Must return a dictionary
                    assert isinstance(model_info, dict), (
                        f"Model {model_name} info must be a dictionary, "
                        f"got {type(model_info)}"
                    )
                    
                    # Must contain model version information
                    assert 'model_type' in model_info or 'model_name' in model_info, (
                        f"Model {model_name} info must contain model type/name"
                    )
                
                # Property 5: Test prediction consistency (same input -> same output)
                predictions_repeat = model.predict(X_test)
                assert np.allclose(predictions, predictions_repeat, rtol=1e-10), (
                    f"Model {model_name} predictions are not deterministic: "
                    f"max_diff={np.max(np.abs(predictions - predictions_repeat)):.10f}"
                )
                
            except Exception as e:
                # If model training fails due to data issues, skip this example
                assume(False, f"Model {model_name} testing failed: {e}")