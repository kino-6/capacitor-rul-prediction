"""
Structured Logging for RUL Predictions

This module implements structured JSON logging for the RUL prediction system,
providing comprehensive logging of predictions, errors, and performance metrics.

Requirements: 10.3
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Union
from pathlib import Path
import traceback

from .data_structures import CycleData, PredictionResult
from .exceptions import PredictionError


class PredictionLogger:
    """
    Structured logger for RUL predictions
    
    This class provides structured JSON logging for:
    - Prediction requests and responses
    - Error logging with stack traces
    - Performance metrics
    - Model metadata
    
    All logs are formatted as JSON for easy parsing and analysis.
    """
    
    def __init__(
        self,
        logger_name: str = "rul_prediction",
        log_level: int = logging.INFO,
        log_file: Optional[str] = None,
        enable_console: bool = True
    ):
        """
        Initialize structured logger
        
        Args:
            logger_name: Name for the logger
            log_level: Logging level (default: INFO)
            log_file: Optional log file path
            enable_console: Whether to enable console logging
        """
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(log_level)
        
        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Create JSON formatter
        formatter = JsonFormatter()
        
        # Add console handler if enabled
        if enable_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Add file handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Model metadata for context
        self.model_metadata: Dict[str, Any] = {}
        
        # Request tracking
        self.active_requests: Dict[str, Dict[str, Any]] = {}
    
    def set_model_metadata(self, metadata: Dict[str, Any]) -> None:
        """
        Set model metadata for logging context
        
        Args:
            metadata: Model metadata dictionary
        """
        self.model_metadata = metadata.copy()
    
    def log_prediction_request(
        self,
        capacitor_id: str,
        cycle_number: int,
        cycle_data: CycleData
    ) -> str:
        """
        Log prediction request
        
        Args:
            capacitor_id: Capacitor identifier
            cycle_number: Cycle number
            cycle_data: Cycle data for prediction
            
        Returns:
            Request ID for tracking
        """
        request_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        # Create input summary (avoid logging full voltage arrays)
        input_summary = {
            "capacitor_id": capacitor_id,
            "cycle_number": cycle_number,
            "vl_series_length": len(cycle_data.vl_series) if cycle_data.vl_series is not None else 0,
            "vo_series_length": len(cycle_data.vo_series) if cycle_data.vo_series is not None else 0,
            "vl_mean": float(cycle_data.vl_series.mean()) if cycle_data.vl_series is not None else None,
            "vl_std": float(cycle_data.vl_series.std()) if cycle_data.vl_series is not None else None,
            "vo_mean": float(cycle_data.vo_series.mean()) if cycle_data.vo_series is not None else None,
            "vo_std": float(cycle_data.vo_series.std()) if cycle_data.vo_series is not None else None,
            "timestamp": getattr(cycle_data, 'timestamp', None)
        }
        
        log_entry = {
            "event_type": "prediction_request",
            "request_id": request_id,
            "timestamp": timestamp,
            "input_summary": input_summary,
            "model_metadata": self.model_metadata
        }
        
        # Store request for later correlation
        self.active_requests[request_id] = {
            "start_time": timestamp,
            "capacitor_id": capacitor_id,
            "cycle_number": cycle_number
        }
        
        self.logger.info("Prediction request", extra={"structured_data": log_entry})
        
        return request_id
    
    def log_prediction_response(
        self,
        request_id: str,
        result: PredictionResult,
        performance_metrics: Dict[str, Any]
    ) -> None:
        """
        Log prediction response
        
        Args:
            request_id: Request ID from log_prediction_request
            result: Prediction result
            performance_metrics: Performance metrics dictionary
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        # Get request context
        request_context = self.active_requests.get(request_id, {})
        
        # Create output summary
        output_summary = {
            "rul_cycles": result.rul_cycles,
            "rul_confidence_lower": result.rul_confidence_lower,
            "rul_confidence_upper": result.rul_confidence_upper,
            "degradation_score": result.degradation_score,
            "degradation_stage": result.degradation_stage,
            "anomaly_flag": result.anomaly_flag,
            "anomaly_score": result.anomaly_score,
            "feature_importance_count": len(result.feature_importance) if result.feature_importance else 0,
            "model_version": result.model_version
        }
        
        log_entry = {
            "event_type": "prediction_response",
            "request_id": request_id,
            "timestamp": timestamp,
            "request_context": request_context,
            "output_summary": output_summary,
            "performance_metrics": performance_metrics,
            "model_metadata": self.model_metadata
        }
        
        self.logger.info("Prediction response", extra={"structured_data": log_entry})
        
        # Clean up request tracking
        if request_id in self.active_requests:
            del self.active_requests[request_id]
    
    def log_prediction_error(
        self,
        request_id: str,
        error: Exception,
        capacitor_id: str,
        cycle_number: int,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log prediction error with full context
        
        Args:
            request_id: Request ID from log_prediction_request
            error: Exception that occurred
            capacitor_id: Capacitor identifier
            cycle_number: Cycle number
            context: Additional context information
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        # Get request context
        request_context = self.active_requests.get(request_id, {})
        
        # Create error summary
        error_summary = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "error_code": getattr(error, 'code', None),
            "error_details": getattr(error, 'details', {}),
            "stack_trace": traceback.format_exc(),
            "capacitor_id": capacitor_id,
            "cycle_number": cycle_number
        }
        
        # Add context if provided
        if context:
            error_summary["context"] = context
        
        log_entry = {
            "event_type": "prediction_error",
            "request_id": request_id,
            "timestamp": timestamp,
            "request_context": request_context,
            "error_summary": error_summary,
            "model_metadata": self.model_metadata
        }
        
        self.logger.error("Prediction error", extra={"structured_data": log_entry})
        
        # Clean up request tracking
        if request_id in self.active_requests:
            del self.active_requests[request_id]
    
    def log_performance_metrics(
        self,
        metrics: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log performance metrics
        
        Args:
            metrics: Performance metrics dictionary
            context: Additional context information
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        log_entry = {
            "event_type": "performance_metrics",
            "timestamp": timestamp,
            "metrics": metrics,
            "context": context or {},
            "model_metadata": self.model_metadata
        }
        
        self.logger.info("Performance metrics", extra={"structured_data": log_entry})
    
    def log_model_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log model-related events (training, loading, etc.)
        
        Args:
            event_type: Type of model event
            event_data: Event-specific data
            context: Additional context information
        """
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        log_entry = {
            "event_type": f"model_{event_type}",
            "timestamp": timestamp,
            "event_data": event_data,
            "context": context or {},
            "model_metadata": self.model_metadata
        }
        
        self.logger.info(f"Model {event_type}", extra={"structured_data": log_entry})


class JsonFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log string
        """
        # Base log entry
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add structured data if present
        if hasattr(record, 'structured_data'):
            log_entry.update(record.structured_data)
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info', 'structured_data']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=self._json_serializer, separators=(',', ':'))
    
    def _json_serializer(self, obj: Any) -> Any:
        """
        Custom JSON serializer for non-serializable objects
        
        Args:
            obj: Object to serialize
            
        Returns:
            Serializable representation
        """
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)


# Global logger instance
_prediction_logger: Optional[PredictionLogger] = None


def get_prediction_logger(
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    enable_console: bool = True
) -> PredictionLogger:
    """
    Get or create global prediction logger instance
    
    Args:
        log_file: Optional log file path
        log_level: Logging level
        enable_console: Whether to enable console logging
        
    Returns:
        PredictionLogger instance
    """
    global _prediction_logger
    
    if _prediction_logger is None:
        # Default log file location
        if log_file is None:
            log_file = "logs/rul_predictions.jsonl"
        
        _prediction_logger = PredictionLogger(
            log_file=log_file,
            log_level=log_level,
            enable_console=enable_console
        )
    
    return _prediction_logger


def configure_prediction_logging(
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    enable_console: bool = True
) -> PredictionLogger:
    """
    Configure global prediction logging
    
    Args:
        log_file: Optional log file path
        log_level: Logging level
        enable_console: Whether to enable console logging
        
    Returns:
        Configured PredictionLogger instance
    """
    global _prediction_logger
    
    # Reset global logger
    _prediction_logger = None
    
    return get_prediction_logger(
        log_file=log_file,
        log_level=log_level,
        enable_console=enable_console
    )


# Utility functions for common logging patterns
def log_batch_prediction_start(
    batch_size: int,
    capacitor_ids: list,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log start of batch prediction
    
    Args:
        batch_size: Number of predictions in batch
        capacitor_ids: List of capacitor IDs
        context: Additional context
    """
    logger = get_prediction_logger()
    
    event_data = {
        "batch_size": batch_size,
        "capacitor_ids": capacitor_ids,
        "unique_capacitors": len(set(capacitor_ids))
    }
    
    logger.log_model_event("batch_prediction_start", event_data, context)


def log_batch_prediction_complete(
    batch_size: int,
    success_count: int,
    error_count: int,
    total_time: float,
    context: Optional[Dict[str, Any]] = None
) -> None:
    """
    Log completion of batch prediction
    
    Args:
        batch_size: Number of predictions in batch
        success_count: Number of successful predictions
        error_count: Number of failed predictions
        total_time: Total processing time
        context: Additional context
    """
    logger = get_prediction_logger()
    
    event_data = {
        "batch_size": batch_size,
        "success_count": success_count,
        "error_count": error_count,
        "success_rate": success_count / batch_size if batch_size > 0 else 0,
        "total_time_seconds": total_time,
        "average_time_per_prediction": total_time / batch_size if batch_size > 0 else 0
    }
    
    logger.log_model_event("batch_prediction_complete", event_data, context)


def log_model_loading(
    model_type: str,
    model_path: str,
    load_time: float,
    success: bool,
    error: Optional[str] = None
) -> None:
    """
    Log model loading event
    
    Args:
        model_type: Type of model being loaded
        model_path: Path to model file
        load_time: Time taken to load model
        success: Whether loading was successful
        error: Error message if loading failed
    """
    logger = get_prediction_logger()
    
    event_data = {
        "model_type": model_type,
        "model_path": model_path,
        "load_time_seconds": load_time,
        "success": success,
        "error": error
    }
    
    logger.log_model_event("loading", event_data)