"""
True RUL Prediction System

A comprehensive system for predicting remaining useful life (RUL) of capacitors
using interpretable machine learning models.
"""

__version__ = "0.2.0"

from .data_structures import CycleData, CapacitorData, PredictionResult, TrainingDataset
# from .data_loader import DataLoader, load_es12_data  # Temporarily disabled due to import issue
from .feature_extractor import FeatureExtractor
from .feature_normalizer import FeatureNormalizer, normalize_features
from .time_series_preprocessor import TimeSeriesPreprocessor
from .gradient_boosting_predictor import GradientBoostingRULPredictor
from .random_forest_predictor import RandomForestRULPredictor
from .elastic_net_predictor import ElasticNetRULPredictor
from .isolation_forest_detector import IsolationForestDetector
from .autoencoder_detector import AutoencoderDetector
from .improved_ocsvm import ImprovedOCSVM
from .ensemble_anomaly_detector import EnsembleAnomalyDetector
from .prediction_aggregator import PredictionAggregator
from .confidence_estimator import ConfidenceEstimator
from .rul_regression_model import RULRegressionModel
from .hybrid_ensemble_predictor import HybridEnsembleRULPredictor
from .exceptions import (
    PredictionError, InputValidationError, ModelNotReadyError,
    FeatureExtractionError, TimeoutError
)
from .rul_predictor import RULPredictor
from .structured_logger import (
    PredictionLogger, get_prediction_logger, configure_prediction_logging,
    log_batch_prediction_start, log_batch_prediction_complete, log_model_loading
)
from .interpretability_engine import InterpretabilityEngine
from .ood_detector import OutOfDistributionDetector

__all__ = [
    "CycleData",
    "CapacitorData",
    "PredictionResult",
    "TrainingDataset",
    # "DataLoader",
    # "load_es12_data",
    "FeatureExtractor",
    "FeatureNormalizer",
    "normalize_features",
    "TimeSeriesPreprocessor",
    "GradientBoostingRULPredictor",
    "RandomForestRULPredictor",
    "ElasticNetRULPredictor",
    "RULRegressionModel",
    "HybridEnsembleRULPredictor",
    "IsolationForestDetector",
    "AutoencoderDetector",
    "ImprovedOCSVM",
    "EnsembleAnomalyDetector",
    "PredictionAggregator",
    "ConfidenceEstimator",
    "PredictionError",
    "InputValidationError",
    "ModelNotReadyError",
    "FeatureExtractionError",
    "TimeoutError",
    "RULPredictor",
    "PredictionLogger",
    "get_prediction_logger",
    "configure_prediction_logging",
    "log_batch_prediction_start",
    "log_batch_prediction_complete",
    "log_model_loading",
    "InterpretabilityEngine",
    "OutOfDistributionDetector",
]
