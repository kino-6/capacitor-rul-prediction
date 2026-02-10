"""
Fast Ensemble Anomaly Detector for testing purposes

This is a simplified version optimized for speed during testing.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
import time
from tqdm import tqdm

from .isolation_forest_detector import IsolationForestDetector
from .fast_autoencoder_detector import FastAutoencoderDetector
from .improved_ocsvm import ImprovedOCSVM

logger = logging.getLogger(__name__)


class FastEnsembleAnomalyDetector:
    """
    Fast ensemble anomaly detector for testing
    
    Uses simplified components and faster training for development/testing.
    """
    
    def __init__(self, weights: Optional[List[float]] = None):
        """Initialize fast ensemble detector"""
        if weights is None:
            self.weights = [0.4, 0.4, 0.2]  # IF, FastAE, OCSVM
        else:
            self.weights = weights
        
        self.detectors = [
            IsolationForestDetector(contamination=0.05),
            None,  # Will be initialized with input_dim
            ImprovedOCSVM(nu=0.05, kernel='rbf')
        ]
        
        self.detector_names = ['IsolationForest', 'FastAutoencoder', 'OCSVM']
        self.threshold = 0.0
        self.is_fitted = False
        self.feature_names = None
        
        logger.info(f"Initialized FastEnsembleAnomalyDetector with weights: {self.weights}")
    
    def fit(self, normal_data: np.ndarray, 
            feature_names: Optional[List[str]] = None) -> 'FastEnsembleAnomalyDetector':
        """Fast fit on normal data"""
        if normal_data.size == 0:
            raise ValueError("normal_data cannot be empty")
        
        n_samples, n_features = normal_data.shape
        logger.info(f"Fast fitting ensemble on {n_samples} normal samples with {n_features} features")
        start_time = time.time()
        
        self.feature_names = feature_names
        
        # Initialize fast autoencoder
        self.detectors[1] = FastAutoencoderDetector(
            input_dim=n_features, 
            encoding_dim=max(4, n_features // 8)  # Very small encoding
        )
        
        # Fit detectors with progress bar
        detector_scores = []
        
        for i, (detector, name) in enumerate(zip(self.detectors, self.detector_names)):
            detector_start = time.time()
            logger.info(f"Training {name}...")
            
            try:
                if isinstance(detector, FastAutoencoderDetector):
                    detector.fit(normal_data, epochs=2, verbose=False)  # Very fast training
                else:
                    detector.fit(normal_data, feature_names)
                
                # Get scores
                if isinstance(detector, FastAutoencoderDetector):
                    scores = detector.get_reconstruction_error(normal_data)
                else:
                    scores = detector.predict_score(normal_data)
                    scores = -scores  # Convert to anomaly scores
                
                detector_scores.append(scores)
                detector_time = time.time() - detector_start
                logger.info(f"{name} training completed in {detector_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                raise
        
        # Set threshold
        self.is_fitted = True
        ensemble_scores = self._compute_ensemble_scores(normal_data)
        self.threshold = np.percentile(ensemble_scores, 95)
        
        total_time = time.time() - start_time
        logger.info(f"Fast ensemble training completed in {total_time:.2f}s. Threshold: {self.threshold:.4f}")
        
        return self
    
    def _compute_ensemble_scores(self, x: np.ndarray) -> np.ndarray:
        """Compute weighted ensemble scores"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted first")
        
        scores = []
        for detector, weight in zip(self.detectors, self.weights):
            if isinstance(detector, FastAutoencoderDetector):
                detector_scores = detector.get_reconstruction_error(x)
            else:
                detector_scores = detector.predict_score(x)
                detector_scores = -detector_scores  # Convert to anomaly scores
            
            scores.append(detector_scores * weight)
        
        return np.sum(scores, axis=0)
    
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Predict anomalies"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted first")
        
        ensemble_scores = self._compute_ensemble_scores(x)
        binary_predictions = (ensemble_scores > self.threshold).astype(int)
        
        info = {
            'ensemble_scores': ensemble_scores,
            'threshold': self.threshold,
            'detector_weights': self.weights
        }
        
        return binary_predictions, ensemble_scores, info
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'detector_names': self.detector_names,
            'weights': self.weights,
            'threshold': self.threshold,
            'is_fitted': self.is_fitted,
            'model_type': 'FastEnsemble'
        }