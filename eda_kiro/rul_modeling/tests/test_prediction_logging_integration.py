"""
Integration tests for prediction logging functionality
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from true_rul.structured_logger import configure_prediction_logging
from true_rul.data_structures import CycleData
from true_rul.exceptions import InputValidationError


class TestPredictionLoggingIntegration:
    """Integration tests for prediction logging"""
    
    @pytest.fixture
    def temp_log_file(self):
        """Create temporary log file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
            yield f.name
        Path(f.name).unlink(missing_ok=True)
    
    @pytest.fixture
    def sample_cycle_data(self):
        """Create sample cycle data"""
        return CycleData(
            cycle_number=10,
            vl_series=np.random.randn(100),
            vo_series=np.random.randn(100),
            timestamp=datetime.now()
        )
    
    def test_rul_predictor_input_validation_logging(self, temp_log_file, sample_cycle_data):
        """Test that input validation errors are logged properly"""
        from true_rul.rul_predictor import RULPredictor
        
        # Configure logging
        logger = configure_prediction_logging(
            log_file=temp_log_file,
            enable_console=False
        )
        
        # Create predictor without models (will fail readiness check)
        predictor = RULPredictor(structured_logger=logger)
        
        # Try prediction with invalid input (empty capacitor_id)
        try:
            predictor.predict_with_error_handling(
                cycle_data=sample_cycle_data,
                capacitor_id="",  # Invalid empty string
                cycle_history=None
            )
        except InputValidationError:
            pass  # Expected
        
        # Check that error was logged
        with open(temp_log_file, 'r') as f:
            lines = f.readlines()
            
        # Should have request log and error log
        assert len(lines) >= 2
        
        # Check request log
        request_log = json.loads(lines[0])
        assert request_log["event_type"] == "prediction_request"
        
        # Check error log
        error_log = json.loads(lines[1])
        assert error_log["event_type"] == "prediction_error"
        assert error_log["error_summary"]["error_type"] == "InputValidationError"
        assert "capacitor_id must be a non-empty string" in error_log["error_summary"]["error_message"]
    
    def test_structured_logging_json_format(self, temp_log_file):
        """Test that all log entries are valid JSON"""
        from true_rul.structured_logger import get_prediction_logger
        
        logger = configure_prediction_logging(
            log_file=temp_log_file,
            enable_console=False
        )
        
        # Log various types of events
        logger.log_performance_metrics({"test_metric": 1.0})
        logger.log_model_event("test_event", {"test_data": "value"})
        
        # Read and validate JSON format
        with open(temp_log_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        log_data = json.loads(line)
                        assert isinstance(log_data, dict)
                        assert "timestamp" in log_data
                        assert "event_type" in log_data
                    except json.JSONDecodeError as e:
                        pytest.fail(f"Invalid JSON on line {line_num}: {e}")
    
    def test_logging_performance_impact(self, temp_log_file):
        """Test that logging doesn't significantly impact performance"""
        import time
        from true_rul.structured_logger import get_prediction_logger
        
        logger = configure_prediction_logging(
            log_file=temp_log_file,
            enable_console=False
        )
        
        # Measure time for multiple log entries
        start_time = time.time()
        
        for i in range(100):
            logger.log_performance_metrics({
                "iteration": i,
                "test_value": np.random.random()
            })
        
        elapsed_time = time.time() - start_time
        
        # Should complete 100 log entries in reasonable time (< 1 second)
        assert elapsed_time < 1.0, f"Logging took too long: {elapsed_time:.3f}s"
        
        # Verify all entries were logged
        with open(temp_log_file, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 100


if __name__ == "__main__":
    pytest.main([__file__])