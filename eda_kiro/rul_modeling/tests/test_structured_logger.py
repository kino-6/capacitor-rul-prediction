"""
Unit tests for structured logging functionality
"""

import pytest
import json
import tempfile
import logging
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from true_rul.structured_logger import (
    PredictionLogger, JsonFormatter, get_prediction_logger,
    configure_prediction_logging, log_batch_prediction_start,
    log_batch_prediction_complete, log_model_loading
)
from true_rul.data_structures import CycleData, PredictionResult
from true_rul.exceptions import PredictionError


class TestPredictionLogger:
    """Test suite for PredictionLogger"""
    
    @pytest.fixture
    def temp_log_file(self):
        """Create temporary log file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def logger(self, temp_log_file):
        """Create logger instance"""
        return PredictionLogger(
            logger_name="test_logger",
            log_file=temp_log_file,
            enable_console=False
        )
    
    @pytest.fixture
    def sample_cycle_data(self):
        """Create sample cycle data"""
        return CycleData(
            cycle_number=10,
            vl_series=np.random.randn(100),
            vo_series=np.random.randn(100),
            timestamp=datetime.now()
        )
    
    @pytest.fixture
    def sample_prediction_result(self):
        """Create sample prediction result"""
        return PredictionResult(
            rul_cycles=50,
            rul_confidence_lower=40,
            rul_confidence_upper=60,
            degradation_score=0.3,
            degradation_stage="early_degradation",
            anomaly_flag=False,
            anomaly_score=0.2,
            feature_importance={"feature_1": 0.5, "feature_2": 0.3},
            timestamp=datetime.now(),
            model_version="1.0.0"
        )
    
    def test_logger_initialization(self, temp_log_file):
        """Test logger initialization"""
        logger = PredictionLogger(
            logger_name="test_init",
            log_file=temp_log_file,
            log_level=logging.DEBUG,
            enable_console=True
        )
        
        assert logger.logger.name == "test_init"
        assert logger.logger.level == logging.DEBUG
        assert len(logger.logger.handlers) == 2  # Console + file
        assert logger.model_metadata == {}
        assert logger.active_requests == {}
    
    def test_set_model_metadata(self, logger):
        """Test setting model metadata"""
        metadata = {
            "predictor_version": "1.0.0",
            "models": {"rul_model": {"type": "xgboost"}}
        }
        
        logger.set_model_metadata(metadata)
        assert logger.model_metadata == metadata
    
    def test_log_prediction_request(self, logger, sample_cycle_data, temp_log_file):
        """Test logging prediction request"""
        request_id = logger.log_prediction_request(
            capacitor_id="C1",
            cycle_number=10,
            cycle_data=sample_cycle_data
        )
        
        # Check request ID format
        assert isinstance(request_id, str)
        assert len(request_id) == 36  # UUID format
        
        # Check request tracking
        assert request_id in logger.active_requests
        assert logger.active_requests[request_id]["capacitor_id"] == "C1"
        assert logger.active_requests[request_id]["cycle_number"] == 10
        
        # Check log file content
        with open(temp_log_file, 'r') as f:
            log_line = f.readline().strip()
            log_data = json.loads(log_line)
            
            assert log_data["event_type"] == "prediction_request"
            assert log_data["request_id"] == request_id
            assert log_data["input_summary"]["capacitor_id"] == "C1"
            assert log_data["input_summary"]["cycle_number"] == 10
            assert log_data["input_summary"]["vl_series_length"] == 100
            assert log_data["input_summary"]["vo_series_length"] == 100
    
    def test_log_prediction_response(self, logger, sample_prediction_result, temp_log_file):
        """Test logging prediction response"""
        # First create a request
        request_id = logger.log_prediction_request(
            capacitor_id="C1",
            cycle_number=10,
            cycle_data=CycleData(
                cycle_number=10,
                vl_series=np.random.randn(100),
                vo_series=np.random.randn(100),
                timestamp=datetime.now()
            )
        )
        
        # Clear log file to isolate response log
        open(temp_log_file, 'w').close()
        
        # Log response
        performance_metrics = {
            "elapsed_time_seconds": 0.5,
            "within_timeout": True,
            "feature_extraction_success": True
        }
        
        logger.log_prediction_response(
            request_id=request_id,
            result=sample_prediction_result,
            performance_metrics=performance_metrics
        )
        
        # Check request cleanup
        assert request_id not in logger.active_requests
        
        # Check log file content
        with open(temp_log_file, 'r') as f:
            log_line = f.readline().strip()
            log_data = json.loads(log_line)
            
            assert log_data["event_type"] == "prediction_response"
            assert log_data["request_id"] == request_id
            assert log_data["output_summary"]["rul_cycles"] == 50
            assert log_data["output_summary"]["degradation_stage"] == "early_degradation"
            assert log_data["performance_metrics"]["elapsed_time_seconds"] == 0.5
    
    def test_log_prediction_error(self, logger, temp_log_file):
        """Test logging prediction error"""
        # Create a request
        request_id = logger.log_prediction_request(
            capacitor_id="C1",
            cycle_number=10,
            cycle_data=CycleData(
                cycle_number=10,
                vl_series=np.random.randn(100),
                vo_series=np.random.randn(100),
                timestamp=datetime.now()
            )
        )
        
        # Clear log file
        open(temp_log_file, 'w').close()
        
        # Create error
        error = PredictionError(
            "Test error",
            code="TEST_ERROR",
            details={"test": "value"}
        )
        
        context = {"error_category": "test_error"}
        
        logger.log_prediction_error(
            request_id=request_id,
            error=error,
            capacitor_id="C1",
            cycle_number=10,
            context=context
        )
        
        # Check request cleanup
        assert request_id not in logger.active_requests
        
        # Check log file content
        with open(temp_log_file, 'r') as f:
            log_line = f.readline().strip()
            log_data = json.loads(log_line)
            
            assert log_data["event_type"] == "prediction_error"
            assert log_data["request_id"] == request_id
            assert log_data["error_summary"]["error_type"] == "PredictionError"
            assert log_data["error_summary"]["error_message"] == "Test error"
            assert log_data["error_summary"]["error_code"] == "TEST_ERROR"
            assert log_data["error_summary"]["context"]["error_category"] == "test_error"
    
    def test_log_performance_metrics(self, logger, temp_log_file):
        """Test logging performance metrics"""
        metrics = {
            "total_predictions": 100,
            "average_latency": 0.5,
            "success_rate": 0.95
        }
        
        context = {"time_period": "last_hour"}
        
        logger.log_performance_metrics(metrics, context)
        
        # Check log file content
        with open(temp_log_file, 'r') as f:
            log_line = f.readline().strip()
            log_data = json.loads(log_line)
            
            assert log_data["event_type"] == "performance_metrics"
            assert log_data["metrics"]["total_predictions"] == 100
            assert log_data["context"]["time_period"] == "last_hour"
    
    def test_log_model_event(self, logger, temp_log_file):
        """Test logging model events"""
        event_data = {
            "model_type": "xgboost",
            "training_time": 120.5,
            "accuracy": 0.92
        }
        
        context = {"dataset": "ES12"}
        
        logger.log_model_event("training", event_data, context)
        
        # Check log file content
        with open(temp_log_file, 'r') as f:
            log_line = f.readline().strip()
            log_data = json.loads(log_line)
            
            assert log_data["event_type"] == "model_training"
            assert log_data["event_data"]["model_type"] == "xgboost"
            assert log_data["context"]["dataset"] == "ES12"


class TestJsonFormatter:
    """Test suite for JsonFormatter"""
    
    @pytest.fixture
    def formatter(self):
        """Create formatter instance"""
        return JsonFormatter()
    
    def test_basic_formatting(self, formatter):
        """Test basic log record formatting"""
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test_logger"
        assert log_data["message"] == "Test message"
        assert log_data["module"] == "test"
        assert log_data["line"] == 10
        assert "timestamp" in log_data
    
    def test_structured_data_formatting(self, formatter):
        """Test formatting with structured data"""
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Add structured data
        record.structured_data = {
            "event_type": "test_event",
            "custom_field": "custom_value"
        }
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["event_type"] == "test_event"
        assert log_data["custom_field"] == "custom_value"
    
    def test_exception_formatting(self, formatter):
        """Test formatting with exception info"""
        try:
            raise ValueError("Test exception")
        except ValueError as e:
            import sys
            record = logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=10,
                msg="Error occurred",
                args=(),
                exc_info=sys.exc_info()
            )
        
        formatted = formatter.format(record)
        log_data = json.loads(formatted)
        
        assert log_data["level"] == "ERROR"
        assert "exception" in log_data
        assert "ValueError: Test exception" in log_data["exception"]


class TestGlobalLoggerFunctions:
    """Test suite for global logger functions"""
    
    def test_get_prediction_logger(self):
        """Test getting global logger instance"""
        # Reset global logger
        import true_rul.structured_logger as sl
        sl._prediction_logger = None
        
        logger1 = get_prediction_logger()
        logger2 = get_prediction_logger()
        
        # Should return same instance
        assert logger1 is logger2
        assert isinstance(logger1, PredictionLogger)
    
    @pytest.fixture
    def temp_log_file(self):
        """Create temporary log file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)
    
    def test_configure_prediction_logging(self, temp_log_file):
        """Test configuring global logging"""
        logger = configure_prediction_logging(
            log_file=temp_log_file,
            log_level=logging.DEBUG,
            enable_console=False
        )
        
        assert isinstance(logger, PredictionLogger)
        assert logger.logger.level == logging.DEBUG
    
    @patch('true_rul.structured_logger.get_prediction_logger')
    def test_log_batch_prediction_start(self, mock_get_logger):
        """Test batch prediction start logging"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        capacitor_ids = ["C1", "C2", "C1", "C3"]
        context = {"batch_id": "batch_001"}
        
        log_batch_prediction_start(
            batch_size=4,
            capacitor_ids=capacitor_ids,
            context=context
        )
        
        mock_logger.log_model_event.assert_called_once()
        args, kwargs = mock_logger.log_model_event.call_args
        
        assert args[0] == "batch_prediction_start"
        assert args[1]["batch_size"] == 4
        assert args[1]["unique_capacitors"] == 3
        assert args[2] == context
    
    @patch('true_rul.structured_logger.get_prediction_logger')
    def test_log_batch_prediction_complete(self, mock_get_logger):
        """Test batch prediction complete logging"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        log_batch_prediction_complete(
            batch_size=10,
            success_count=8,
            error_count=2,
            total_time=5.5
        )
        
        mock_logger.log_model_event.assert_called_once()
        args, kwargs = mock_logger.log_model_event.call_args
        
        assert args[0] == "batch_prediction_complete"
        assert args[1]["batch_size"] == 10
        assert args[1]["success_count"] == 8
        assert args[1]["error_count"] == 2
        assert args[1]["success_rate"] == 0.8
        assert args[1]["total_time_seconds"] == 5.5
        assert args[1]["average_time_per_prediction"] == 0.55
    
    @patch('true_rul.structured_logger.get_prediction_logger')
    def test_log_model_loading(self, mock_get_logger):
        """Test model loading logging"""
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger
        
        log_model_loading(
            model_type="xgboost",
            model_path="/path/to/model.pkl",
            load_time=2.5,
            success=True
        )
        
        mock_logger.log_model_event.assert_called_once()
        args, kwargs = mock_logger.log_model_event.call_args
        
        assert args[0] == "loading"
        assert args[1]["model_type"] == "xgboost"
        assert args[1]["model_path"] == "/path/to/model.pkl"
        assert args[1]["load_time_seconds"] == 2.5
        assert args[1]["success"] is True
        assert args[1]["error"] is None


class TestIntegrationWithRULPredictor:
    """Integration tests with RULPredictor"""
    
    @pytest.fixture
    def temp_log_file(self):
        """Create temporary log file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)
    
    def test_rul_predictor_logging_integration(self, temp_log_file):
        """Test that RULPredictor uses structured logging correctly"""
        from true_rul.rul_predictor import RULPredictor
        
        # Configure logging
        logger = configure_prediction_logging(
            log_file=temp_log_file,
            enable_console=False
        )
        
        # Create predictor (without models for this test)
        predictor = RULPredictor(structured_logger=logger)
        
        # Check that logger is set
        assert predictor.prediction_logger is logger
        
        # Check that model metadata is set
        assert "predictor_version" in predictor.prediction_logger.model_metadata
        assert "initialization_time" in predictor.prediction_logger.model_metadata


if __name__ == "__main__":
    pytest.main([__file__])