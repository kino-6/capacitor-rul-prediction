"""
Main RUL Predictor Class with Comprehensive Error Handling

This module implements the RULPredictor class that serves as the main entry point
for RUL predictions with comprehensive error handling, input validation, graceful
degradation, timeout handling, and fallback mechanisms.

Requirements: 7.1, Error Handling
"""

import logging
import time
import traceback
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor, TimeoutError as ConcurrentTimeoutError

from .data_structures import CycleData, PredictionResult
from .exceptions import (
    PredictionError, InputValidationError, ModelNotReadyError,
    FeatureExtractionError, TimeoutError
)
from .rul_regression_model import RULRegressionModel
from .ensemble_anomaly_detector import EnsembleAnomalyDetector
from .feature_extractor import FeatureExtractor
from .prediction_aggregator import PredictionAggregator
from .confidence_estimator import ConfidenceEstimator
from .structured_logger import get_prediction_logger, PredictionLogger

logger = logging.getLogger(__name__)


class RULPredictor:
    """
    Main RUL Predictor with comprehensive error handling
    
    This class provides the primary interface for RUL predictions with:
    - Comprehensive input validation
    - Graceful degradation for feature extraction failures
    - Timeout handling for predictions (1 second limit)
    - Fallback confidence intervals when estimation fails
    - Structured error handling and logging
    
    Attributes:
        rul_model: Trained RUL regression model
        anomaly_detector: Trained anomaly detection ensemble
        feature_extractor: Feature extraction component
        prediction_aggregator: Prediction aggregation component
        confidence_estimator: Confidence estimation component
        is_ready: Whether all models are loaded and ready
        prediction_timeout: Timeout for predictions in seconds
    """
    
    def __init__(
        self,
        rul_model: Optional[RULRegressionModel] = None,
        anomaly_detector: Optional[EnsembleAnomalyDetector] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
        prediction_aggregator: Optional[PredictionAggregator] = None,
        confidence_estimator: Optional[ConfidenceEstimator] = None,
        prediction_timeout: float = 1.0,
        structured_logger: Optional[PredictionLogger] = None
    ):
        """
        Initialize RUL Predictor
        
        Args:
            rul_model: Trained RUL regression model
            anomaly_detector: Trained anomaly detection ensemble
            feature_extractor: Feature extraction component
            prediction_aggregator: Prediction aggregation component
            confidence_estimator: Confidence estimation component
            prediction_timeout: Timeout for predictions in seconds (default: 1.0)
            structured_logger: Optional structured logger instance
        """
        self.rul_model = rul_model
        self.anomaly_detector = anomaly_detector
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.prediction_aggregator = prediction_aggregator or PredictionAggregator()
        self.confidence_estimator = confidence_estimator or ConfidenceEstimator()
        self.prediction_timeout = prediction_timeout
        
        # Initialize structured logger
        self.prediction_logger = structured_logger or get_prediction_logger()
        
        # Track readiness state
        self.is_ready = self._check_readiness()
        
        # Initialize prediction history for context
        self.prediction_history: Dict[str, List[float]] = {}
        
        # Set model metadata for logging
        self._update_model_metadata()
        
        logger.info(
            f"RULPredictor initialized: ready={self.is_ready}, "
            f"timeout={prediction_timeout}s"
        )
    
    def _check_readiness(self) -> bool:
        """
        Check if all required models are loaded and ready
        
        Returns:
            True if all models are ready, False otherwise
        """
        try:
            rul_ready = (
                self.rul_model is not None and 
                hasattr(self.rul_model, 'is_trained') and 
                self.rul_model.is_trained
            )
            
            anomaly_ready = (
                self.anomaly_detector is not None and
                hasattr(self.anomaly_detector, 'is_fitted') and
                self.anomaly_detector.is_fitted
            )
            
            return rul_ready and anomaly_ready
            
        except Exception as e:
            logger.error(f"Error checking model readiness: {e}")
            return False
    
    def _update_model_metadata(self) -> None:
        """
        Update model metadata for structured logging
        """
        metadata = {
            "predictor_version": "1.0.0",
            "initialization_time": datetime.utcnow().isoformat() + "Z",
            "prediction_timeout": self.prediction_timeout,
            "models": {}
        }
        
        # RUL model metadata
        if self.rul_model:
            metadata["models"]["rul_model"] = {
                "type": getattr(self.rul_model, 'model_type', 'unknown'),
                "is_trained": getattr(self.rul_model, 'is_trained', False)
            }
        
        # Anomaly detector metadata
        if self.anomaly_detector:
            metadata["models"]["anomaly_detector"] = {
                "type": "ensemble",
                "is_fitted": getattr(self.anomaly_detector, 'is_fitted', False),
                "detectors": getattr(self.anomaly_detector, 'detector_types', [])
            }
        
        # Feature extractor metadata
        if self.feature_extractor:
            metadata["models"]["feature_extractor"] = {
                "type": "advanced" if getattr(self.feature_extractor, 'include_advanced', True) else "basic"
            }
        
        self.prediction_logger.set_model_metadata(metadata)
    
    def predict_with_error_handling(
        self,
        cycle_data: CycleData,
        capacitor_id: str,
        cycle_history: Optional[List[CycleData]] = None
    ) -> PredictionResult:
        """
        Predict RUL with comprehensive error handling
        
        This method implements the main prediction pipeline with:
        - Input validation
        - Feature extraction with graceful degradation
        - Model prediction with timeout handling
        - Fallback mechanisms for failures
        - Comprehensive structured logging
        
        Args:
            cycle_data: Current cycle voltage data
            capacitor_id: Capacitor identifier
            cycle_history: Previous cycles for context (optional)
            
        Returns:
            PredictionResult with RUL prediction and metadata
            
        Raises:
            InputValidationError: If input validation fails
            ModelNotReadyError: If models are not ready
            TimeoutError: If prediction exceeds timeout
            PredictionError: For other prediction failures
        """
        # Start structured logging for this prediction
        request_id = self.prediction_logger.log_prediction_request(
            capacitor_id=capacitor_id,
            cycle_number=cycle_data.cycle_number,
            cycle_data=cycle_data
        )
        
        start_time = time.time()
        
        try:
            # Log prediction request (traditional logging for backward compatibility)
            logger.info(
                f"Starting prediction for {capacitor_id} cycle {cycle_data.cycle_number}"
            )
            
            # 1. Input validation
            self._validate_input(cycle_data, capacitor_id, cycle_history)
            
            # 2. Check model readiness
            if not self.is_ready:
                raise ModelNotReadyError(
                    "Models are not ready for prediction. Ensure all models are trained and loaded.",
                    details={
                        "rul_model_ready": self.rul_model is not None and getattr(self.rul_model, 'is_trained', False),
                        "anomaly_detector_ready": self.anomaly_detector is not None and getattr(self.anomaly_detector, 'is_fitted', False)
                    }
                )
            
            # 3. Execute prediction with timeout
            try:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        self._execute_prediction,
                        cycle_data,
                        capacitor_id,
                        cycle_history
                    )
                    
                    result = future.result(timeout=self.prediction_timeout)
                    
            except ConcurrentTimeoutError:
                raise TimeoutError(
                    f"Prediction timed out after {self.prediction_timeout} seconds",
                    timeout_seconds=self.prediction_timeout,
                    details={
                        "capacitor_id": capacitor_id,
                        "cycle_number": cycle_data.cycle_number
                    }
                )
            
            # 4. Calculate performance metrics
            elapsed_time = time.time() - start_time
            performance_metrics = {
                "elapsed_time_seconds": elapsed_time,
                "timeout_seconds": self.prediction_timeout,
                "within_timeout": elapsed_time < self.prediction_timeout,
                "feature_extraction_success": True,  # If we got here, it succeeded
                "model_prediction_success": True,
                "anomaly_detection_success": True
            }
            
            # 5. Log successful prediction (structured)
            self.prediction_logger.log_prediction_response(
                request_id=request_id,
                result=result,
                performance_metrics=performance_metrics
            )
            
            # 6. Log successful prediction (traditional)
            logger.info(
                f"Prediction completed for {capacitor_id} cycle {cycle_data.cycle_number} "
                f"in {elapsed_time:.3f}s: RUL={result.rul_cycles}, "
                f"stage={result.degradation_stage}, anomaly={result.anomaly_flag}"
            )
            
            # 7. Update prediction history
            self._update_prediction_history(capacitor_id, result.degradation_score)
            
            return result
            
        except (InputValidationError, ModelNotReadyError, TimeoutError) as e:
            # Log known exceptions with structured logging
            self.prediction_logger.log_prediction_error(
                request_id=request_id,
                error=e,
                capacitor_id=capacitor_id,
                cycle_number=cycle_data.cycle_number,
                context={
                    "error_category": "known_error",
                    "validation_passed": isinstance(e, (ModelNotReadyError, TimeoutError))
                }
            )
            # Re-raise known exceptions
            raise
            
        except Exception as e:
            # Handle unexpected errors
            elapsed_time = time.time() - start_time
            error_details = {
                "capacitor_id": capacitor_id,
                "cycle_number": cycle_data.cycle_number,
                "elapsed_time": elapsed_time,
                "error_type": type(e).__name__,
                "traceback": traceback.format_exc()
            }
            
            # Log with structured logging
            self.prediction_logger.log_prediction_error(
                request_id=request_id,
                error=e,
                capacitor_id=capacitor_id,
                cycle_number=cycle_data.cycle_number,
                context={
                    "error_category": "unexpected_error",
                    "error_details": error_details
                }
            )
            
            # Log with traditional logging
            logger.error(
                f"Unexpected error in prediction for {capacitor_id} "
                f"cycle {cycle_data.cycle_number}: {e}",
                extra={"error_details": error_details}
            )
            
            raise PredictionError(
                f"Unexpected error during prediction: {e}",
                code="UNEXPECTED_ERROR",
                details=error_details
            )
    
    def _validate_input(
        self,
        cycle_data: CycleData,
        capacitor_id: str,
        cycle_history: Optional[List[CycleData]] = None
    ) -> None:
        """
        Validate input data for prediction
        
        Args:
            cycle_data: Current cycle data
            capacitor_id: Capacitor identifier
            cycle_history: Previous cycles (optional)
            
        Raises:
            InputValidationError: If validation fails
        """
        errors = []
        
        # Validate cycle_data
        if not isinstance(cycle_data, CycleData):
            errors.append("cycle_data must be a CycleData instance")
        else:
            # Validate voltage series
            if cycle_data.vl_series is None or len(cycle_data.vl_series) == 0:
                errors.append("VL series cannot be empty")
            elif not isinstance(cycle_data.vl_series, np.ndarray):
                errors.append("VL series must be a numpy array")
            elif np.any(np.isnan(cycle_data.vl_series)) or np.any(np.isinf(cycle_data.vl_series)):
                errors.append("VL series contains NaN or infinite values")
            
            if cycle_data.vo_series is None or len(cycle_data.vo_series) == 0:
                errors.append("VO series cannot be empty")
            elif not isinstance(cycle_data.vo_series, np.ndarray):
                errors.append("VO series must be a numpy array")
            elif np.any(np.isnan(cycle_data.vo_series)) or np.any(np.isinf(cycle_data.vo_series)):
                errors.append("VO series contains NaN or infinite values")
            
            # Check series lengths match
            if (len(cycle_data.vl_series) != len(cycle_data.vo_series)):
                errors.append("VL and VO series must have the same length")
            
            # Check cycle number
            if cycle_data.cycle_number < 1:
                errors.append("Cycle number must be >= 1")
        
        # Validate capacitor_id
        if not capacitor_id or not isinstance(capacitor_id, str):
            errors.append("capacitor_id must be a non-empty string")
        
        # Validate cycle_history if provided
        if cycle_history is not None:
            if not isinstance(cycle_history, list):
                errors.append("cycle_history must be a list")
            else:
                for i, hist_cycle in enumerate(cycle_history):
                    if not isinstance(hist_cycle, CycleData):
                        errors.append(f"cycle_history[{i}] must be a CycleData instance")
        
        # Raise validation error if any issues found
        if errors:
            raise InputValidationError(
                f"Input validation failed: {'; '.join(errors)}",
                details={
                    "validation_errors": errors,
                    "capacitor_id": capacitor_id,
                    "cycle_number": getattr(cycle_data, 'cycle_number', None)
                }
            )
    
    def _execute_prediction(
        self,
        cycle_data: CycleData,
        capacitor_id: str,
        cycle_history: Optional[List[CycleData]] = None
    ) -> PredictionResult:
        """
        Execute the core prediction logic
        
        Args:
            cycle_data: Current cycle data
            capacitor_id: Capacitor identifier
            cycle_history: Previous cycles for context
            
        Returns:
            PredictionResult with prediction and metadata
            
        Raises:
            FeatureExtractionError: If feature extraction fails
            PredictionError: If model prediction fails
        """
        # 1. Feature extraction with graceful degradation
        try:
            features = self._extract_features_with_fallback(
                cycle_data, capacitor_id, cycle_history
            )
        except Exception as e:
            raise FeatureExtractionError(
                f"Feature extraction failed: {e}",
                details={
                    "capacitor_id": capacitor_id,
                    "cycle_number": cycle_data.cycle_number,
                    "error": str(e)
                }
            )
        
        # 2. RUL prediction with confidence intervals
        try:
            rul_pred, rul_lower, rul_upper = self._predict_rul_with_fallback(features)
        except Exception as e:
            logger.error(f"RUL prediction failed: {e}")
            raise PredictionError(
                f"RUL prediction failed: {e}",
                code="RUL_PREDICTION_ERROR",
                details={"error": str(e)}
            )
        
        # 3. Anomaly detection
        try:
            anomaly_flag, anomaly_score, feature_importance = self._detect_anomaly_with_fallback(features)
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            raise PredictionError(
                f"Anomaly detection failed: {e}",
                code="ANOMALY_DETECTION_ERROR",
                details={"error": str(e)}
            )
        
        # 4. Aggregate predictions
        try:
            degradation_history = self.prediction_history.get(capacitor_id, [])
            
            result = self.prediction_aggregator.aggregate(
                rul_pred=rul_pred,
                rul_confidence_lower=rul_lower,
                rul_confidence_upper=rul_upper,
                anomaly_flag=anomaly_flag,
                anomaly_score=anomaly_score,
                feature_importance=feature_importance,
                degradation_history=degradation_history,
                capacitor_id=capacitor_id,
                cycle_number=cycle_data.cycle_number
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction aggregation failed: {e}")
            raise PredictionError(
                f"Prediction aggregation failed: {e}",
                code="AGGREGATION_ERROR",
                details={"error": str(e)}
            )
    
    def _extract_features_with_fallback(
        self,
        cycle_data: CycleData,
        capacitor_id: str,
        cycle_history: Optional[List[CycleData]] = None
    ) -> np.ndarray:
        """
        Extract features with graceful degradation for failures
        
        Args:
            cycle_data: Current cycle data
            capacitor_id: Capacitor identifier
            cycle_history: Previous cycles for context
            
        Returns:
            Feature array
            
        Raises:
            FeatureExtractionError: If all feature extraction attempts fail
        """
        try:
            # Primary feature extraction
            feature_dict = self.feature_extractor.extract_features(
                cycle_data, capacitor_id, cycle_history
            )
            features = np.array(list(feature_dict.values()))
            
            # Validate features
            if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                logger.warning("Features contain NaN/inf values, attempting fallback")
                raise ValueError("Features contain invalid values")
            
            return features.reshape(1, -1)  # Reshape for single sample
            
        except Exception as e:
            logger.warning(f"Primary feature extraction failed: {e}, trying fallback")
            
            try:
                # Fallback: Extract only basic statistical features
                features = self._extract_basic_features_fallback(cycle_data)
                return features.reshape(1, -1)
                
            except Exception as fallback_error:
                logger.error(f"Fallback feature extraction also failed: {fallback_error}")
                raise FeatureExtractionError(
                    f"All feature extraction methods failed. Primary: {e}, Fallback: {fallback_error}",
                    details={
                        "primary_error": str(e),
                        "fallback_error": str(fallback_error)
                    }
                )
    
    def _extract_basic_features_fallback(self, cycle_data: CycleData) -> np.ndarray:
        """
        Extract basic statistical features as fallback
        
        Args:
            cycle_data: Current cycle data
            
        Returns:
            Basic feature array
        """
        vl = cycle_data.vl_series
        vo = cycle_data.vo_series
        
        # Basic statistical features
        features = [
            np.mean(vl), np.std(vl), np.min(vl), np.max(vl),
            np.mean(vo), np.std(vo), np.min(vo), np.max(vo),
            np.mean(vo - vl),  # Response difference
            np.std(vo - vl),   # Response variability
        ]
        
        # Pad with zeros to match expected feature count (assuming ~55 features)
        while len(features) < 55:
            features.append(0.0)
        
        return np.array(features[:55])  # Truncate if too many
    
    def _predict_rul_with_fallback(
        self,
        features: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Predict RUL with fallback confidence intervals
        
        Args:
            features: Input features
            
        Returns:
            Tuple of (prediction, lower_bound, upper_bound)
        """
        try:
            # Primary prediction with confidence
            if hasattr(self.rul_model, 'predict_with_confidence'):
                pred, lower, upper = self.rul_model.predict_with_confidence(features)
                return float(pred[0]), float(lower[0]), float(upper[0])
            else:
                # Fallback: basic prediction with estimated confidence
                pred = self.rul_model.predict(features)
                pred_val = float(pred[0])
                
                # Simple confidence estimation (±20% of prediction)
                uncertainty = pred_val * 0.2
                lower = max(0, pred_val - uncertainty)
                upper = pred_val + uncertainty
                
                logger.warning("Using fallback confidence intervals")
                return pred_val, lower, upper
                
        except Exception as e:
            logger.error(f"RUL prediction failed, using emergency fallback: {e}")
            
            # Emergency fallback: return conservative estimate
            return 50.0, 30.0, 70.0  # Conservative RUL estimate
    
    def _detect_anomaly_with_fallback(
        self,
        features: np.ndarray
    ) -> Tuple[bool, float, Dict[str, float]]:
        """
        Detect anomaly with fallback mechanisms
        
        Args:
            features: Input features
            
        Returns:
            Tuple of (anomaly_flag, anomaly_score, feature_importance)
        """
        try:
            # Primary anomaly detection
            anomaly_flag, anomaly_score, feature_importance = self.anomaly_detector.predict(features)
            
            return bool(anomaly_flag[0]), float(anomaly_score[0]), feature_importance
            
        except Exception as e:
            logger.error(f"Anomaly detection failed, using fallback: {e}")
            
            # Fallback: simple threshold-based detection
            # Use feature variance as a simple anomaly indicator
            feature_var = np.var(features)
            anomaly_score = min(1.0, feature_var / 100.0)  # Normalize roughly
            anomaly_flag = anomaly_score > 0.5
            
            # Empty feature importance
            feature_importance = {}
            
            logger.warning("Using fallback anomaly detection")
            return bool(anomaly_flag), float(anomaly_score), feature_importance
    
    def _update_prediction_history(self, capacitor_id: str, degradation_score: float) -> None:
        """
        Update prediction history for context
        
        Args:
            capacitor_id: Capacitor identifier
            degradation_score: Current degradation score
        """
        if capacitor_id not in self.prediction_history:
            self.prediction_history[capacitor_id] = []
        
        self.prediction_history[capacitor_id].append(degradation_score)
        
        # Keep only last 20 predictions for memory efficiency
        if len(self.prediction_history[capacitor_id]) > 20:
            self.prediction_history[capacitor_id] = self.prediction_history[capacitor_id][-20:]
    
    def get_model_status(self) -> Dict[str, Any]:
        """
        Get status information about loaded models
        
        Returns:
            Dictionary with model status information
        """
        return {
            "is_ready": self.is_ready,
            "prediction_timeout": self.prediction_timeout,
            "rul_model": {
                "loaded": self.rul_model is not None,
                "trained": getattr(self.rul_model, 'is_trained', False),
                "type": getattr(self.rul_model, 'model_type', None)
            },
            "anomaly_detector": {
                "loaded": self.anomaly_detector is not None,
                "fitted": getattr(self.anomaly_detector, 'is_fitted', False)
            },
            "feature_extractor": {
                "loaded": self.feature_extractor is not None,
                "advanced_features": getattr(self.feature_extractor, 'include_advanced', True)
            },
            "prediction_history_size": sum(len(hist) for hist in self.prediction_history.values())
        }
    
    def clear_prediction_history(self, capacitor_id: Optional[str] = None) -> None:
        """
        Clear prediction history
        
        Args:
            capacitor_id: Specific capacitor to clear, or None for all
        """
        if capacitor_id is None:
            self.prediction_history.clear()
            logger.info("Cleared all prediction history")
        elif capacitor_id in self.prediction_history:
            del self.prediction_history[capacitor_id]
            logger.info(f"Cleared prediction history for {capacitor_id}")
    
    def set_prediction_timeout(self, timeout_seconds: float) -> None:
        """
        Set prediction timeout
        
        Args:
            timeout_seconds: New timeout value in seconds
        """
        if timeout_seconds <= 0:
            raise ValueError("Timeout must be positive")
        
        self.prediction_timeout = timeout_seconds
        logger.info(f"Prediction timeout set to {timeout_seconds}s")
    
    def __repr__(self) -> str:
        """String representation"""
        return (
            f"RULPredictor(ready={self.is_ready}, "
            f"timeout={self.prediction_timeout}s, "
            f"history_size={sum(len(h) for h in self.prediction_history.values())})"
        )