"""
Advanced Ensemble Anomaly Detector

This module implements an advanced ensemble that combines multiple
anomaly detection algorithms including Deep SVDD, LOF, GMM, and
traditional methods with confidence-weighted voting.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib
from pathlib import Path

from .deep_svdd_detector import DeepSVDDDetector, create_deep_svdd_detector
from .lof_detector import LOFDetector, create_lof_detector
from .gmm_detector import GMMDetector, create_gmm_detector
from .isolation_forest_detector import IsolationForestDetector
from .improved_ocsvm import ImprovedOCSVM
from .autoencoder_detector import AutoencoderDetector

logger = logging.getLogger(__name__)


@dataclass
class AdvancedEnsembleConfig:
    """Configuration for Advanced Ensemble detector."""
    # Detector selection
    use_deep_svdd: bool = True
    use_lof: bool = True
    use_gmm: bool = True
    use_isolation_forest: bool = True
    use_ocsvm: bool = True
    use_autoencoder: bool = False  # Requires more setup
    
    # Ensemble weights (will be normalized)
    weights: Dict[str, float] = None
    
    # Voting strategy
    voting_strategy: str = "weighted_average"  # "weighted_average", "majority", "confidence_weighted"
    confidence_threshold: float = 0.7
    
    # Training configuration
    parallel_training: bool = True
    n_jobs: int = -1
    
    # Individual detector configurations
    deep_svdd_config: Dict[str, Any] = None
    lof_config: Dict[str, Any] = None
    gmm_config: Dict[str, Any] = None
    isolation_forest_config: Dict[str, Any] = None
    ocsvm_config: Dict[str, Any] = None
    autoencoder_config: Dict[str, Any] = None


class AdvancedEnsembleDetector:
    """
    Advanced ensemble anomaly detector combining multiple algorithms.
    
    This detector combines Deep SVDD, LOF, GMM, Isolation Forest, OCSVM,
    and optionally Autoencoder with sophisticated voting mechanisms.
    """
    
    def __init__(self, config: AdvancedEnsembleConfig):
        self.config = config
        self.detectors: Dict[str, Any] = {}
        self.detector_weights: Dict[str, float] = {}
        self.is_fitted = False
        self.feature_names: Optional[List[str]] = None
        
        # Set default weights if not provided
        if config.weights is None:
            self.config.weights = {
                'deep_svdd': 0.25,
                'lof': 0.20,
                'gmm': 0.20,
                'isolation_forest': 0.15,
                'ocsvm': 0.15,
                'autoencoder': 0.05
            }
        
        self._initialize_detectors()
    
    def _initialize_detectors(self) -> None:
        """Initialize individual detectors based on configuration."""
        logger.info("Initializing advanced ensemble detectors...")
        
        # Deep SVDD
        if self.config.use_deep_svdd:
            deep_svdd_config = self.config.deep_svdd_config or {}
            self.detectors['deep_svdd'] = create_deep_svdd_detector(
                input_dim=None,  # Will be set during fit
                **deep_svdd_config
            )
            logger.info("✓ Deep SVDD detector initialized")
        
        # LOF
        if self.config.use_lof:
            lof_config = self.config.lof_config or {}
            self.detectors['lof'] = create_lof_detector(**lof_config)
            logger.info("✓ LOF detector initialized")
        
        # GMM
        if self.config.use_gmm:
            gmm_config = self.config.gmm_config or {}
            self.detectors['gmm'] = create_gmm_detector(**gmm_config)
            logger.info("✓ GMM detector initialized")
        
        # Isolation Forest
        if self.config.use_isolation_forest:
            iso_config = self.config.isolation_forest_config or {}
            self.detectors['isolation_forest'] = IsolationForestDetector(**iso_config)
            logger.info("✓ Isolation Forest detector initialized")
        
        # OCSVM
        if self.config.use_ocsvm:
            ocsvm_config = self.config.ocsvm_config or {}
            self.detectors['ocsvm'] = ImprovedOCSVM(**ocsvm_config)
            logger.info("✓ OCSVM detector initialized")
        
        # Autoencoder (optional)
        if self.config.use_autoencoder:
            ae_config = self.config.autoencoder_config or {}
            self.detectors['autoencoder'] = AutoencoderDetector(**ae_config)
            logger.info("✓ Autoencoder detector initialized")
        
        # Normalize weights for active detectors
        self._normalize_weights()
    
    def _normalize_weights(self) -> None:
        """Normalize weights for active detectors."""
        active_detectors = list(self.detectors.keys())
        total_weight = sum(self.config.weights.get(name, 0) for name in active_detectors)
        
        if total_weight > 0:
            self.detector_weights = {
                name: self.config.weights.get(name, 0) / total_weight
                for name in active_detectors
            }
        else:
            # Equal weights if no weights specified
            weight = 1.0 / len(active_detectors)
            self.detector_weights = {name: weight for name in active_detectors}
        
        logger.info(f"Normalized detector weights: {self.detector_weights}")
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> 'AdvancedEnsembleDetector':
        """
        Fit all detectors in the ensemble.
        
        Args:
            X: Training data (normal samples only)
            feature_names: Optional feature names for interpretability
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training advanced ensemble on {X.shape[0]} samples with {X.shape[1]} features...")
        
        self.feature_names = feature_names
        
        # Update Deep SVDD input dimension
        if 'deep_svdd' in self.detectors:
            deep_svdd_config = self.config.deep_svdd_config or {}
            self.detectors['deep_svdd'] = create_deep_svdd_detector(
                input_dim=X.shape[1],
                **deep_svdd_config
            )
        
        if self.config.parallel_training:
            self._fit_parallel(X)
        else:
            self._fit_sequential(X)
        
        self.is_fitted = True
        logger.info("Advanced ensemble training completed")
        return self
    
    def _fit_sequential(self, X: np.ndarray) -> None:
        """Fit detectors sequentially."""
        for name, detector in self.detectors.items():
            logger.info(f"Training {name} detector...")
            try:
                if hasattr(detector, 'fit'):
                    if name in ['lof', 'gmm']:
                        detector.fit(X, self.feature_names)
                    else:
                        detector.fit(X)
                logger.info(f"✓ {name} detector trained successfully")
            except Exception as e:
                logger.error(f"✗ Failed to train {name} detector: {e}")
                # Remove failed detector
                del self.detectors[name]
                if name in self.detector_weights:
                    del self.detector_weights[name]
        
        # Renormalize weights after removing failed detectors
        if self.detectors:
            self._normalize_weights()
    
    def _fit_parallel(self, X: np.ndarray) -> None:
        """Fit detectors in parallel."""
        def fit_detector(name_detector_pair):
            name, detector = name_detector_pair
            try:
                logger.info(f"Training {name} detector...")
                if hasattr(detector, 'fit'):
                    if name in ['lof', 'gmm']:
                        detector.fit(X, self.feature_names)
                    else:
                        detector.fit(X)
                return name, detector, None
            except Exception as e:
                logger.error(f"Failed to train {name} detector: {e}")
                return name, None, e
        
        # Use ThreadPoolExecutor for parallel training
        max_workers = min(len(self.detectors), self.config.n_jobs if self.config.n_jobs > 0 else 4)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(fit_detector, (name, detector)): name
                for name, detector in self.detectors.items()
            }
            
            successful_detectors = {}
            
            for future in as_completed(futures):
                name, detector, error = future.result()
                if error is None and detector is not None:
                    successful_detectors[name] = detector
                    logger.info(f"✓ {name} detector trained successfully")
                else:
                    logger.error(f"✗ {name} detector training failed")
        
        # Update detectors with successful ones
        self.detectors = successful_detectors
        
        # Update weights
        self.detector_weights = {
            name: weight for name, weight in self.detector_weights.items()
            if name in self.detectors
        }
        
        # Renormalize weights
        if self.detectors:
            self._normalize_weights()
    
    def predict_score(self, X: np.ndarray) -> np.ndarray:
        """
        Compute ensemble anomaly scores.
        
        Args:
            X: Input data
            
        Returns:
            Ensemble anomaly scores
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        if not self.detectors:
            raise ValueError("No detectors available for prediction")
        
        # Collect scores from all detectors
        detector_scores = {}
        
        for name, detector in self.detectors.items():
            try:
                if hasattr(detector, 'predict_score'):
                    scores = detector.predict_score(X)
                    # Normalize scores to [0, 1] range
                    scores_normalized = self._normalize_scores(scores)
                    detector_scores[name] = scores_normalized
                else:
                    logger.warning(f"Detector {name} does not have predict_score method")
            except Exception as e:
                logger.error(f"Error computing scores for {name}: {e}")
        
        if not detector_scores:
            raise ValueError("No detector scores available")
        
        # Combine scores based on voting strategy
        if self.config.voting_strategy == "weighted_average":
            ensemble_scores = self._weighted_average_scores(detector_scores)
        elif self.config.voting_strategy == "confidence_weighted":
            ensemble_scores = self._confidence_weighted_scores(detector_scores, X)
        else:
            raise ValueError(f"Unknown voting strategy: {self.config.voting_strategy}")
        
        return ensemble_scores
    
    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predict anomalies using ensemble voting.
        
        Args:
            X: Input data
            threshold: Decision threshold for ensemble scores
            
        Returns:
            Binary predictions (1 = anomaly, 0 = normal)
        """
        if self.config.voting_strategy == "majority":
            return self._majority_voting(X)
        else:
            # Use score-based prediction
            ensemble_scores = self.predict_score(X)
            
            if threshold is None:
                # Use adaptive threshold based on score distribution
                threshold = np.percentile(ensemble_scores, 90)
            
            return (ensemble_scores > threshold).astype(int)
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range."""
        if len(scores) == 0:
            return scores
        
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score > min_score:
            return (scores - min_score) / (max_score - min_score)
        else:
            return np.zeros_like(scores)
    
    def _weighted_average_scores(self, detector_scores: Dict[str, np.ndarray]) -> np.ndarray:
        """Compute weighted average of detector scores."""
        ensemble_scores = np.zeros(len(next(iter(detector_scores.values()))))
        
        for name, scores in detector_scores.items():
            weight = self.detector_weights.get(name, 0)
            ensemble_scores += weight * scores
        
        return ensemble_scores
    
    def _confidence_weighted_scores(self, detector_scores: Dict[str, np.ndarray], X: np.ndarray) -> np.ndarray:
        """Compute confidence-weighted ensemble scores."""
        # Compute confidence for each detector based on score consistency
        detector_confidences = {}
        
        for name, scores in detector_scores.items():
            # Confidence based on score variance (lower variance = higher confidence)
            score_std = np.std(scores)
            confidence = 1.0 / (1.0 + score_std)
            detector_confidences[name] = confidence
        
        # Normalize confidences
        total_confidence = sum(detector_confidences.values())
        if total_confidence > 0:
            detector_confidences = {
                name: conf / total_confidence
                for name, conf in detector_confidences.items()
            }
        
        # Combine scores with confidence weighting
        ensemble_scores = np.zeros(len(next(iter(detector_scores.values()))))
        
        for name, scores in detector_scores.items():
            base_weight = self.detector_weights.get(name, 0)
            confidence_weight = detector_confidences.get(name, 0)
            combined_weight = base_weight * confidence_weight
            ensemble_scores += combined_weight * scores
        
        return ensemble_scores
    
    def _majority_voting(self, X: np.ndarray) -> np.ndarray:
        """Perform majority voting across detectors."""
        detector_predictions = {}
        
        for name, detector in self.detectors.items():
            try:
                if hasattr(detector, 'predict'):
                    predictions = detector.predict(X)
                    detector_predictions[name] = predictions
            except Exception as e:
                logger.error(f"Error getting predictions from {name}: {e}")
        
        if not detector_predictions:
            raise ValueError("No detector predictions available")
        
        # Stack predictions and compute majority vote
        all_predictions = np.stack(list(detector_predictions.values()), axis=0)
        majority_predictions = (np.mean(all_predictions, axis=0) > 0.5).astype(int)
        
        return majority_predictions
    
    def get_feature_importance(self, X: np.ndarray, method: str = "ensemble_average") -> np.ndarray:
        """
        Compute ensemble feature importance.
        
        Args:
            X: Input data
            method: Method for combining importance ("ensemble_average" or "weighted_average")
            
        Returns:
            Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before computing feature importance")
        
        detector_importances = {}
        
        for name, detector in self.detectors.items():
            try:
                if hasattr(detector, 'get_feature_importance'):
                    importance = detector.get_feature_importance(X)
                    detector_importances[name] = importance
            except Exception as e:
                logger.warning(f"Could not compute feature importance for {name}: {e}")
        
        if not detector_importances:
            # Fallback to uniform importance
            return np.ones(X.shape[1]) / X.shape[1]
        
        # Combine importances
        if method == "weighted_average":
            ensemble_importance = np.zeros(X.shape[1])
            for name, importance in detector_importances.items():
                weight = self.detector_weights.get(name, 0)
                ensemble_importance += weight * importance
        else:
            # Simple average
            all_importances = np.stack(list(detector_importances.values()), axis=0)
            ensemble_importance = np.mean(all_importances, axis=0)
        
        # Normalize
        if np.sum(ensemble_importance) > 0:
            ensemble_importance = ensemble_importance / np.sum(ensemble_importance)
        
        return ensemble_importance
    
    def get_detector_contributions(self, X: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed contributions from each detector.
        
        Args:
            X: Input data
            
        Returns:
            Dictionary with detector contributions
        """
        contributions = {}
        
        for name, detector in self.detectors.items():
            try:
                detector_info = {
                    'weight': self.detector_weights.get(name, 0),
                    'scores': detector.predict_score(X) if hasattr(detector, 'predict_score') else None,
                    'predictions': detector.predict(X) if hasattr(detector, 'predict') else None,
                }
                
                # Add detector-specific information
                if hasattr(detector, 'get_model_info'):
                    detector_info['model_info'] = detector.get_model_info()
                
                contributions[name] = detector_info
                
            except Exception as e:
                logger.error(f"Error getting contributions from {name}: {e}")
                contributions[name] = {'error': str(e)}
        
        return contributions
    
    def save_ensemble(self, filepath: str) -> None:
        """Save the entire ensemble to disk."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted ensemble")
        
        ensemble_data = {
            'config': self.config,
            'detector_weights': self.detector_weights,
            'feature_names': self.feature_names,
            'is_fitted': self.is_fitted
        }
        
        # Save ensemble metadata
        joblib.dump(ensemble_data, f"{filepath}_metadata.pkl")
        
        # Save individual detectors
        detector_dir = Path(filepath).parent / f"{Path(filepath).stem}_detectors"
        detector_dir.mkdir(exist_ok=True)
        
        for name, detector in self.detectors.items():
            detector_path = detector_dir / f"{name}.pkl"
            try:
                if hasattr(detector, 'save_model'):
                    detector.save_model(str(detector_path))
                else:
                    joblib.dump(detector, detector_path)
            except Exception as e:
                logger.error(f"Failed to save {name} detector: {e}")
        
        logger.info(f"Advanced ensemble saved to {filepath}")
    
    def load_ensemble(self, filepath: str) -> 'AdvancedEnsembleDetector':
        """Load a saved ensemble from disk."""
        # Load ensemble metadata
        ensemble_data = joblib.load(f"{filepath}_metadata.pkl")
        
        self.config = ensemble_data['config']
        self.detector_weights = ensemble_data['detector_weights']
        self.feature_names = ensemble_data['feature_names']
        self.is_fitted = ensemble_data['is_fitted']
        
        # Load individual detectors
        detector_dir = Path(filepath).parent / f"{Path(filepath).stem}_detectors"
        self.detectors = {}
        
        for detector_file in detector_dir.glob("*.pkl"):
            name = detector_file.stem
            try:
                if name == 'deep_svdd':
                    detector = create_deep_svdd_detector(input_dim=len(self.feature_names) if self.feature_names else 50)
                    detector.load_model(str(detector_file))
                else:
                    detector = joblib.load(detector_file)
                
                self.detectors[name] = detector
            except Exception as e:
                logger.error(f"Failed to load {name} detector: {e}")
        
        logger.info(f"Advanced ensemble loaded from {filepath}")
        return self


def create_advanced_ensemble_detector(**kwargs) -> AdvancedEnsembleDetector:
    """
    Factory function to create an advanced ensemble detector.
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        Configured advanced ensemble detector
    """
    config = AdvancedEnsembleConfig(**kwargs)
    return AdvancedEnsembleDetector(config)