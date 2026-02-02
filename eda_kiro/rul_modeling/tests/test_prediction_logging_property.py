"""
Property-based test for Prediction Logging

This module contains property-based tests using the Hypothesis framework
to validate that the RUL prediction system implements comprehensive logging
of all predictions, inputs, and performance metrics for monitoring.

Requirements: 10.3
"""

import sys
from pathlib import Path
import numpy as np
import pytest
import json
import tempfile
import logging
from datetime import datetime
from hypothesis import given, strategies as st, settings, assume, HealthCheck
from hypothesis.extra.numpy import arrays
from typing import Dict, Any, List
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from true_rul.data_structures import PredictionResult, CycleData
    from true_rul.structured_logger import PredictionLogger, JsonFormatter, get_prediction_logger, configure_prediction_logging
    from true_rul.rul_predictor import RULPredictor
    from true_rul.exceptions import PredictionError, InputValidationError, FeatureExtractionError
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Import error: {e}")
    MODULES_AVAILABLE = False


class MockRULModel:
    """Mock RUL model for testing logging"""
    
    def __init__(self):
        self.is_trained = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Mock prediction"""
        return np.mean(X, axis=1) * 10 + 50
    
    def predict_with_confidence(self, X: np.ndarray) -> tuple:
        """Mock prediction with confidence intervals"""
        predictions = self.predict(X)
        uncertainty = predictions * 0.1
        lower = np.maximum(0, predictions - uncertainty)
        upper = predictions + uncertainty
        return predictions, lower, upper


class MockAnomalyDetector:
    """Mock anomaly detector for testing"""
    
    def __init__(self):
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> tuple:
        """Mock anomaly detection"""
        anomaly_scores = np.var(X, axis=1) / 10.0
        anomaly_flags = anomaly_scores > 0.5
        feature_importance = {f"feature_{i}": np.random.random() for i in range(X.shape[1])}
        return anomaly_flags, anomaly_scores, feature_importance


class MockFeatureExtractor:
    """Mock feature extractor for testing"""
    
    def extract_all_features(self, cycle_data: CycleData, history: List[CycleData] = None) -> np.ndarray:
        """Mock feature extraction"""
        # Create features based on voltage data statistics
        vl_mean = np.mean(cycle_data.vl_series) if cycle_data.vl_series is not None else 0.0
        vo_mean = np.mean(cycle_data.vo_series) if cycle_data.vo_series is not None else 0.0
        vl_std = np.std(cycle_data.vl_series) if cycle_data.vl_series is not None else 0.0
        vo_std = np.std(cycle_data.vo_series) if cycle_data.vo_series is not None else 0.0
        
        return np.array([vl_mean, vo_mean, vl_std, vo_std, np.random.random()])


class LogCapturingHandler(logging.Handler):
    """Custom logging handler to capture log records for testing"""
    
    def __init__(self):
        super().__init__()
        self.records = []
        self.log_data = []
    
    def emit(self, record):
        """Capture log record"""
        self.records.append(record)
        
        # Parse structured data if present
        if hasattr(record, 'structured_data'):
            # Use the JsonFormatter to properly serialize the data
            formatter = JsonFormatter()
            formatted_message = formatter.format(record)
            try:
                log_entry = json.loads(formatted_message)
                self.log_data.append(log_entry)
            except (json.JSONDecodeError, ValueError):
                # Fallback to structured data as-is
                self.log_data.append(record.structured_data)
        else:
            # Try to parse JSON from message
            try:
                log_entry = json.loads(record.getMessage())
                self.log_data.append(log_entry)
            except (json.JSONDecodeError, ValueError):
                # Not JSON, store as plain message
                self.log_data.append({"message": record.getMessage()})
    
    def clear(self):
        """Clear captured records"""
        self.records.clear()
        self.log_data.clear()


@pytest.mark.skipif(
    not MODULES_AVAILABLE,
    reason="Required modules not available"
)
class TestPredictionLoggingProperty:
    """Property-based tests for prediction logging"""
    
    @pytest.fixture
    def temp_log_file(self):
        """Create temporary log file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def log_handler(self):
        """Create log capturing handler"""
        return LogCapturingHandler()
    
    @pytest.fixture
    def logger_with_handler(self, temp_log_file, log_handler):
        """Create logger with capturing handler"""
        logger = PredictionLogger(
            logger_name="test_prediction_logger",
            log_file=temp_log_file,
            enable_console=False
        )
        logger.logger.addHandler(log_handler)
        return logger, log_handler
    
    @given(
        n_samples=st.integers(min_value=1, max_value=5),
        n_features=st.integers(min_value=3, max_value=8),
        capacitor_id=st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'))),
        cycle_number=st.integers(min_value=1, max_value=200),
        data=st.data()
    )
    @settings(max_examples=50, deadline=30000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_prediction_logging_property(
        self, 
        n_samples, 
        n_features, 
        capacitor_id,
        cycle_number,
        data
    ):
        """
        Property 18: Prediction Logging
        
        **Validates: Requirements 10.3**
        
        THE RUL_Predictor SHALL implement logging of all predictions, inputs, 
        and performance metrics for monitoring.
        
        This property validates that:
        1. All prediction requests are logged with input summary
        2. All prediction responses are logged with output and performance metrics
        3. All prediction errors are logged with full context and stack traces
        4. Log entries contain required fields and proper structure
        5. Logging is consistent across different prediction scenarios
        """
        # Generate realistic test data
        vl_series = data.draw(
            arrays(
                dtype=np.float64,
                shape=(100,),
                elements=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)
            )
        )
        
        vo_series = data.draw(
            arrays(
                dtype=np.float64,
                shape=(100,),
                elements=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)
            )
        )
        
        # Skip if data contains invalid values
        assume(np.all(np.isfinite(vl_series)))
        assume(np.all(np.isfinite(vo_series)))
        assume(len(capacitor_id.strip()) > 0)
        
        # Create temporary log file and handler for this test
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            temp_log_file = f.name
        
        log_handler = LogCapturingHandler()
        
        try:
            # Create logger with capturing handler
            logger = PredictionLogger(
                logger_name="test_prediction_logger",
                log_file=temp_log_file,
                enable_console=False
            )
            logger.logger.addHandler(log_handler)
            
            # Set model metadata
            model_metadata = {
                "predictor_version": "1.0.0",
                "models": {
                    "rul_model": {"type": "xgboost", "version": "1.0"},
                    "anomaly_detector": {"type": "ensemble", "version": "1.0"}
                }
            }
            logger.set_model_metadata(model_metadata)
            
            # Create cycle data
            cycle_data = CycleData(
                cycle_number=cycle_number,
                vl_series=vl_series,
                vo_series=vo_series,
                timestamp=datetime.now()
            )
            
            # Clear previous log entries
            log_handler.clear()
            
            # Test successful prediction logging
            request_id = logger.log_prediction_request(
                capacitor_id=capacitor_id,
                cycle_number=cycle_number,
                cycle_data=cycle_data
            )
            
            # Property 1: Prediction request must be logged with required fields
            assert len(log_handler.log_data) >= 1, "Prediction request must be logged"
            
            request_log = log_handler.log_data[0]
            
            # Validate request log structure
            required_request_fields = [
                "event_type", "request_id", "timestamp", "input_summary", "model_metadata"
            ]
            for field in required_request_fields:
                assert field in request_log, f"Request log must contain {field}"
            
            assert request_log["event_type"] == "prediction_request", (
                f"Request event type must be 'prediction_request', got {request_log['event_type']}"
            )
            
            assert request_log["request_id"] == request_id, (
                f"Request ID mismatch: expected {request_id}, got {request_log['request_id']}"
            )
            
            # Validate input summary structure
            input_summary = request_log["input_summary"]
            required_input_fields = [
                "capacitor_id", "cycle_number", "vl_series_length", "vo_series_length",
                "vl_mean", "vl_std", "vo_mean", "vo_std"
            ]
            for field in required_input_fields:
                assert field in input_summary, f"Input summary must contain {field}"
            
            assert input_summary["capacitor_id"] == capacitor_id, (
                f"Capacitor ID mismatch: expected {capacitor_id}, got {input_summary['capacitor_id']}"
            )
            
            assert input_summary["cycle_number"] == cycle_number, (
                f"Cycle number mismatch: expected {cycle_number}, got {input_summary['cycle_number']}"
            )
            
            # Property 2: Test successful response logging
            log_handler.clear()
            
            # Create mock prediction result
            prediction_result = PredictionResult(
                rul_cycles=50,
                rul_confidence_lower=40,
                rul_confidence_upper=60,
                degradation_score=0.3,
                degradation_stage="early_degradation",
                anomaly_flag=False,
                anomaly_score=0.2,
                feature_importance={"feature_1": 0.5, "feature_2": 0.3, "feature_3": 0.2},
                timestamp=datetime.now(),
                model_version="1.0.0"
            )
            
            performance_metrics = {
                "elapsed_time_seconds": 0.5,
                "within_timeout": True,
                "feature_extraction_success": True,
                "memory_usage_mb": 50.0
            }
            
            logger.log_prediction_response(
                request_id=request_id,
                result=prediction_result,
                performance_metrics=performance_metrics
            )
            
            # Validate response logging
            assert len(log_handler.log_data) >= 1, "Prediction response must be logged"
            
            response_log = log_handler.log_data[0]
            assert response_log["event_type"] == "prediction_response"
            assert "output_summary" in response_log
            assert "performance_metrics" in response_log
            
            # Property 3: Test error logging
            log_handler.clear()
            
            test_error = PredictionError(
                "Test prediction error",
                code="TEST_ERROR",
                details={"test_field": "test_value"}
            )
            
            logger.log_prediction_error(
                request_id=request_id,
                error=test_error,
                capacitor_id=capacitor_id,
                cycle_number=cycle_number,
                context={"error_category": "test_error"}
            )
            
            # Validate error logging
            assert len(log_handler.log_data) >= 1, "Prediction error must be logged"
            
            error_log = log_handler.log_data[0]
            assert error_log["event_type"] == "prediction_error"
            assert "error_summary" in error_log
            
            # Property 4: All log entries must have valid timestamps
            for log_entry in log_handler.log_data:
                assert "timestamp" in log_entry, "All log entries must have timestamp"
                timestamp_str = log_entry["timestamp"]
                assert timestamp_str.endswith("Z"), f"Timestamp must end with 'Z': {timestamp_str}"
            
            # Property 5: Log entries must be JSON serializable
            for log_entry in log_handler.log_data:
                try:
                    json.dumps(log_entry)
                except (TypeError, ValueError) as e:
                    pytest.fail(f"Log entry must be JSON serializable: {e}")
        
        finally:
            # Clean up temporary file
            Path(temp_log_file).unlink(missing_ok=True)
    
    @given(
        batch_size=st.integers(min_value=2, max_value=10),
        success_rate=st.floats(min_value=0.0, max_value=1.0),
        data=st.data()
    )
    @settings(max_examples=30, deadline=20000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_batch_prediction_logging_property(
        self, 
        batch_size, 
        success_rate,
        data
    ):
        """
        Property 18 (Batch): Batch Prediction Logging
        
        **Validates: Requirements 10.3**
        
        Batch predictions must also be comprehensively logged with
        aggregated metrics and individual prediction tracking.
        """
        # Create temporary log file and handler for this test
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            temp_log_file = f.name
        
        log_handler = LogCapturingHandler()
        
        try:
            # Create logger with capturing handler
            logger = PredictionLogger(
                logger_name="test_batch_logger",
                log_file=temp_log_file,
                enable_console=False
            )
            logger.logger.addHandler(log_handler)
            
            # Generate capacitor IDs
            capacitor_ids = [f"C{i}" for i in range(batch_size)]
            
            # Calculate success/error counts
            success_count = int(batch_size * success_rate)
            error_count = batch_size - success_count
            
            # Clear previous logs
            log_handler.clear()
            
            # Test batch start logging
            from true_rul.structured_logger import log_batch_prediction_start
            
            batch_context = {"batch_id": "test_batch_001", "user_id": "test_user"}
            
            log_batch_prediction_start(
                batch_size=batch_size,
                capacitor_ids=capacitor_ids,
                context=batch_context
            )
            
            # Property 1: Batch start must be logged
            assert len(log_handler.log_data) >= 1, "Batch start must be logged"
            
            start_log = log_handler.log_data[0]
            assert start_log["event_type"] == "model_batch_prediction_start", (
                f"Batch start event type must be 'model_batch_prediction_start', got {start_log['event_type']}"
            )
            
            assert start_log["event_data"]["batch_size"] == batch_size, (
                f"Batch size mismatch: expected {batch_size}, got {start_log['event_data']['batch_size']}"
            )
            
            # Property 2: Test batch completion logging
            log_handler.clear()
            
            from true_rul.structured_logger import log_batch_prediction_complete
            
            total_time = data.draw(st.floats(min_value=1.0, max_value=60.0))
            
            log_batch_prediction_complete(
                batch_size=batch_size,
                success_count=success_count,
                error_count=error_count,
                total_time=total_time,
                context=batch_context
            )
            
            # Validate batch completion logging
            assert len(log_handler.log_data) >= 1, "Batch completion must be logged"
            
            complete_log = log_handler.log_data[0]
            assert complete_log["event_type"] == "model_batch_prediction_complete"
            
            event_data = complete_log["event_data"]
            assert event_data["batch_size"] == batch_size
            assert event_data["success_count"] == success_count
        
        finally:
            # Clean up temporary file
            Path(temp_log_file).unlink(missing_ok=True)
    
    def test_performance_metrics_logging_property(self, temp_log_file, log_handler):
        """
        Property 18 (Performance): Performance Metrics Logging
        
        **Validates: Requirements 10.3**
        
        Performance metrics must be comprehensively logged for monitoring.
        """
        # Create logger with capturing handler
        logger = PredictionLogger(
            logger_name="test_performance_logger",
            log_file=temp_log_file,
            enable_console=False
        )
        logger.logger.addHandler(log_handler)
        
        # Clear previous logs
        log_handler.clear()
        
        # Test performance metrics logging
        test_metrics = {
            "total_predictions": 100,
            "average_latency_ms": 250.5,
            "success_rate": 0.95,
            "memory_usage_mb": 128.7,
            "cpu_usage_percent": 45.2,
            "error_rate": 0.05
        }
        
        metrics_context = {
            "time_period": "last_hour",
            "server_id": "rul_server_01"
        }
        
        logger.log_performance_metrics(test_metrics, metrics_context)
        
        # Property: Performance metrics must be logged with required structure
        assert len(log_handler.log_data) >= 1, "Performance metrics must be logged"
        
        metrics_log = log_handler.log_data[0]
        
        # Validate metrics log structure
        required_fields = ["event_type", "timestamp", "metrics", "context"]
        for field in required_fields:
            assert field in metrics_log, f"Metrics log must contain {field}"
        
        assert metrics_log["event_type"] == "performance_metrics", (
            f"Metrics event type must be 'performance_metrics', got {metrics_log['event_type']}"
        )
        
        assert metrics_log["metrics"] == test_metrics, (
            "Performance metrics must be logged exactly as provided"
        )
        
        assert metrics_log["context"] == metrics_context, (
            "Performance metrics context must be logged"
        )
    
    def test_model_event_logging_property(self, temp_log_file, log_handler):
        """
        Property 18 (Model Events): Model Event Logging
        
        **Validates: Requirements 10.3**
        
        Model-related events (loading, training, etc.) must be logged.
        """
        # Configure global logger to use our test handler
        from true_rul.structured_logger import configure_prediction_logging
        
        # Configure global logger with our test file
        global_logger = configure_prediction_logging(
            log_file=temp_log_file,
            enable_console=False
        )
        global_logger.logger.addHandler(log_handler)
        
        # Clear previous logs
        log_handler.clear()
        
        # Test model loading event
        from true_rul.structured_logger import log_model_loading
        
        log_model_loading(
            model_type="xgboost",
            model_path="/path/to/model.pkl",
            load_time=2.5,
            success=True
        )
        
        # Property: Model events must be logged with required structure
        assert len(log_handler.log_data) >= 1, "Model loading must be logged"
        
        loading_log = log_handler.log_data[0]
        
        assert loading_log["event_type"] == "model_loading", (
            f"Model loading event type must be 'model_loading', got {loading_log['event_type']}"
        )
        
        event_data = loading_log["event_data"]
        required_loading_fields = ["model_type", "model_path", "load_time_seconds", "success", "error"]
        for field in required_loading_fields:
            assert field in event_data, f"Model loading event must contain {field}"
        
        assert event_data["model_type"] == "xgboost"
        assert event_data["model_path"] == "/path/to/model.pkl"
        assert event_data["load_time_seconds"] == 2.5
        assert event_data["success"] is True
        assert event_data["error"] is None
    
    def test_logging_consistency_across_scenarios(self, temp_log_file, log_handler):
        """
        Property 18 (Consistency): Logging Consistency Across Scenarios
        
        **Validates: Requirements 10.3**
        
        Logging must be consistent across different prediction scenarios
        and error conditions.
        """
        # Create logger with capturing handler
        logger = PredictionLogger(
            logger_name="test_consistency_logger",
            log_file=temp_log_file,
            enable_console=False
        )
        logger.logger.addHandler(log_handler)
        
        # Test multiple prediction scenarios
        scenarios = [
            {
                "capacitor_id": "C1",
                "cycle_number": 10,
                "vl_length": 100,
                "vo_length": 100,
                "should_succeed": True
            },
            {
                "capacitor_id": "C2", 
                "cycle_number": 50,
                "vl_length": 150,
                "vo_length": 150,
                "should_succeed": True
            },
            {
                "capacitor_id": "C3",
                "cycle_number": 100,
                "vl_length": 80,
                "vo_length": 80,
                "should_succeed": False  # Will test error logging
            }
        ]
        
        for i, scenario in enumerate(scenarios):
            log_handler.clear()
            
            # Create cycle data
            cycle_data = CycleData(
                cycle_number=scenario["cycle_number"],
                vl_series=np.random.randn(scenario["vl_length"]),
                vo_series=np.random.randn(scenario["vo_length"]),
                timestamp=datetime.now()
            )
            
            # Log request
            request_id = logger.log_prediction_request(
                capacitor_id=scenario["capacitor_id"],
                cycle_number=scenario["cycle_number"],
                cycle_data=cycle_data
            )
            
            # Property: Each request must generate consistent log structure
            assert len(log_handler.log_data) >= 1, f"Scenario {i}: Request must be logged"
            
            request_log = log_handler.log_data[0]
            
            # Validate consistent structure across scenarios
            assert request_log["event_type"] == "prediction_request"
            assert "request_id" in request_log
            assert "timestamp" in request_log
            assert "input_summary" in request_log
            
            input_summary = request_log["input_summary"]
            assert input_summary["capacitor_id"] == scenario["capacitor_id"]
            assert input_summary["cycle_number"] == scenario["cycle_number"]
            assert input_summary["vl_series_length"] == scenario["vl_length"]
            assert input_summary["vo_series_length"] == scenario["vo_length"]
            
            # Test response or error logging
            if scenario["should_succeed"]:
                # Log successful response
                log_handler.records.clear()  # Clear but keep log_data for validation
                
                result = PredictionResult(
                    rul_cycles=50 + i * 10,
                    rul_confidence_lower=40 + i * 10,
                    rul_confidence_upper=60 + i * 10,
                    degradation_score=0.2 + i * 0.1,
                    degradation_stage="early_degradation",
                    anomaly_flag=False,
                    anomaly_score=0.1 + i * 0.05,
                    feature_importance={"feature_1": 0.5},
                    timestamp=datetime.now(),
                    model_version="1.0.0"
                )
                
                logger.log_prediction_response(
                    request_id=request_id,
                    result=result,
                    performance_metrics={"elapsed_time_seconds": 0.5 + i * 0.1}
                )
                
                # Validate response log structure
                response_log = log_handler.log_data[-1]  # Last logged entry
                assert response_log["event_type"] == "prediction_response"
                assert response_log["request_id"] == request_id
                assert "output_summary" in response_log
                assert "performance_metrics" in response_log
                
            else:
                # Log error
                log_handler.records.clear()  # Clear but keep log_data for validation
                
                error = InputValidationError(f"Invalid input for scenario {i}")
                
                logger.log_prediction_error(
                    request_id=request_id,
                    error=error,
                    capacitor_id=scenario["capacitor_id"],
                    cycle_number=scenario["cycle_number"],
                    context={"scenario": i}
                )
                
                # Validate error log structure
                error_log = log_handler.log_data[-1]  # Last logged entry
                assert error_log["event_type"] == "prediction_error"
                assert error_log["request_id"] == request_id
                assert "error_summary" in error_log
                
                error_summary = error_log["error_summary"]
                assert error_summary["capacitor_id"] == scenario["capacitor_id"]
                assert error_summary["cycle_number"] == scenario["cycle_number"]
                assert "context" in error_summary
        
        # Property: All log entries across scenarios must be valid JSON
        for log_entry in log_handler.log_data:
            try:
                json.dumps(log_entry)
            except (TypeError, ValueError) as e:
                pytest.fail(f"Log entry must be JSON serializable across all scenarios: {e}")


if __name__ == "__main__":
    pytest.main([__file__])