"""
Exception classes for True RUL Prediction System
"""

from typing import Optional, Dict, Any


class PredictionError(Exception):
    """
    Base exception class for prediction errors
    
    Attributes:
        code: Error code for programmatic handling
        message: Human-readable error message
        details: Additional error details
    """
    
    def __init__(self, message: str, code: str = "PREDICTION_ERROR", details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary"""
        return {
            "error_type": self.__class__.__name__,
            "code": self.code,
            "message": self.message,
            "details": self.details
        }


class InputValidationError(PredictionError):
    """
    Exception raised when input validation fails
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="INPUT_VALIDATION_ERROR", details=details)


class ModelNotReadyError(PredictionError):
    """
    Exception raised when model is not ready for prediction
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="MODEL_NOT_READY_ERROR", details=details)


class FeatureExtractionError(PredictionError):
    """
    Exception raised when feature extraction fails
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, code="FEATURE_EXTRACTION_ERROR", details=details)


class TimeoutError(PredictionError):
    """
    Exception raised when prediction times out
    """
    
    def __init__(self, message: str, timeout_seconds: float, details: Optional[Dict[str, Any]] = None):
        details = details or {}
        details["timeout_seconds"] = timeout_seconds
        super().__init__(message, code="TIMEOUT_ERROR", details=details)


class ModelCompressionError(Exception):
    """
    Exception raised when model compression fails
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class CachingError(Exception):
    """
    Exception raised when caching operations fail
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class MonitoringError(Exception):
    """
    Exception raised when monitoring operations fail
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}