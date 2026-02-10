"""
Ensemble Anomaly Detector for RUL prediction system.

This module implements an ensemble approach combining multiple anomaly detection
algorithms to achieve robust anomaly detection with FPR < 5%. The ensemble
combines Isolation Forest, Autoencoder, and One-Class SVM detectors.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import time

from .isolation_forest_detector import IsolationForestDetector
from .autoencoder_detector import AutoencoderDetector
from .improved_ocsvm import ImprovedOCSVM

logger = logging.getLogger(__name__)


class EnsembleAnomalyDetector:
    """
    Ensemble anomaly detector combining multiple detection algorithms.
    
    This detector combines three complementary anomaly detection approaches:
    1. Isolation Forest (35% weight): Tree-based isolation approach
    2. Autoencoder (40% weight): Neural network reconstruction approach  
    3. One-Class SVM (25% weight): Support vector boundary approach
    
    The ensemble provides robust anomaly detection by leveraging the strengths
    of different algorithms and reducing individual model weaknesses.
    
    Attributes:
        detectors (List): List of individual detector instances
        weights (List[float]): Weights for combining detector scores
        threshold (float): Decision threshold for binary classification
        is_fitted (bool): Whether the ensemble has been trained
        feature_names (Optional[List[str]]): Names of input features
    """
    
    def __init__(self, 
                 weights: Optional[List[float]] = None,
                 isolation_forest_params: Optional[Dict[str, Any]] = None,
                 autoencoder_params: Optional[Dict[str, Any]] = None,
                 ocsvm_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the ensemble anomaly detector.
        
        Args:
            weights (Optional[List[float]]): Weights for [IsolationForest, Autoencoder, OCSVM].
                                           Default: [0.35, 0.40, 0.25]
            isolation_forest_params (Optional[Dict]): Parameters for Isolation Forest
            autoencoder_params (Optional[Dict]): Parameters for Autoencoder
            ocsvm_params (Optional[Dict]): Parameters for One-Class SVM
        """
        # Set default weights
        if weights is None:
            self.weights = [0.35, 0.40, 0.25]  # IF, AE, OCSVM
        else:
            if len(weights) != 3:
                raise ValueError("weights must contain exactly 3 values")
            if not np.isclose(sum(weights), 1.0):
                raise ValueError("weights must sum to 1.0")
            if any(w < 0 for w in weights):
                raise ValueError("all weights must be non-negative")
            self.weights = list(weights)
        
        # Set default parameters
        if_params = isolation_forest_params or {'contamination': 0.05}
        ae_params = autoencoder_params or {'encoding_dim': 16}
        ocsvm_params = ocsvm_params or {'nu': 0.05, 'kernel': 'rbf'}
        
        # Initialize detectors
        self.detectors = [
            IsolationForestDetector(**if_params),
            None,  # Autoencoder will be initialized when we know input_dim
            ImprovedOCSVM(**ocsvm_params)
        ]
        
        self.autoencoder_params = ae_params
        self.threshold = 0.0  # Will be set during training
        self.is_fitted = False
        self.feature_names: Optional[List[str]] = None
        self.training_scores = None
        self.detector_names = ['IsolationForest', 'Autoencoder', 'OCSVM']
        
        logger.info(f"Initialized EnsembleAnomalyDetector with weights: {self.weights}")
    
    def fit(self, 
            normal_data: np.ndarray, 
            feature_names: Optional[List[str]] = None,
            validation_data: Optional[np.ndarray] = None,
            validation_labels: Optional[np.ndarray] = None,
            target_fpr: float = 0.05) -> 'EnsembleAnomalyDetector':
        """
        Fit all detectors in the ensemble on normal cycles data.
        
        Args:
            normal_data (np.ndarray): Feature vectors from normal cycles.
                                    Shape: (n_samples, n_features)
            feature_names (Optional[List[str]]): Names of the features
            validation_data (Optional[np.ndarray]): Validation data for threshold tuning
            validation_labels (Optional[np.ndarray]): Validation labels (0=normal, 1=anomaly)
            target_fpr (float): Target false positive rate for threshold tuning
            
        Returns:
            EnsembleAnomalyDetector: Self for method chaining
            
        Raises:
            ValueError: If normal_data is empty or has invalid shape
        """
        if normal_data.size == 0:
            raise ValueError("normal_data cannot be empty")
        
        if len(normal_data.shape) != 2:
            raise ValueError("normal_data must be 2D array with shape (n_samples, n_features)")
        
        n_samples, n_features = normal_data.shape
        if n_samples < 2:
            raise ValueError("normal_data must contain at least 2 samples")
        
        logger.info(f"Fitting ensemble on {n_samples} normal samples with {n_features} features")
        start_time = time.time()
        
        # Store feature names
        self.feature_names = feature_names
        
        # Initialize autoencoder now that we know input_dim
        self.detectors[1] = AutoencoderDetector(
            input_dim=n_features, 
            **self.autoencoder_params
        )
        
        # Fit each detector with progress tracking
        detector_scores = []
        
        # Create progress bar for detectors
        detector_pbar = tqdm(
            zip(self.detectors, self.detector_names), 
            total=len(self.detectors),
            desc="Training detectors"
        )
        
        for i, (detector, name) in enumerate(detector_pbar):
            detector_pbar.set_description(f"Training {name}")
            detector_start = time.time()
            
            try:
                if isinstance(detector, AutoencoderDetector):
                    # Autoencoder needs special training parameters (heavily reduced for faster training)
                    detector.fit(
                        normal_data, 
                        epochs=5,  # Further reduced from 20 to 5
                        batch_size=min(16, max(4, n_samples // 2)),  # Smaller batch size
                        learning_rate=0.01,  # Higher learning rate
                        validation_split=0.1,  # Smaller validation split
                        early_stopping_patience=2,  # Very early stopping
                        verbose=False
                    )
                else:
                    # Other detectors use standard fit
                    detector.fit(normal_data, feature_names)
                
                # Get scores on training data for ensemble calibration
                if isinstance(detector, AutoencoderDetector):
                    scores = detector.get_reconstruction_error(normal_data)
                    # Convert to anomaly scores (higher = more anomalous)
                    # For autoencoder, higher reconstruction error = more anomalous
                else:
                    scores = detector.predict_score(normal_data)
                    # Convert to anomaly scores (higher = more anomalous)
                    # For IF and OCSVM, lower scores = more anomalous, so negate
                    scores = -scores
                
                detector_scores.append(scores)
                detector_time = time.time() - detector_start
                detector_pbar.set_postfix({'time': f'{detector_time:.1f}s'})
                logger.info(f"{name} training completed in {detector_time:.2f}s")
                
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
                raise
        
        detector_pbar.close()
        
        # Store training scores for analysis
        self.training_scores = np.array(detector_scores).T  # Shape: (n_samples, n_detectors)
        
        # Set decision threshold
        if validation_data is not None and validation_labels is not None:
            self._tune_threshold(validation_data, validation_labels, target_fpr)
        else:
            # Use default threshold based on training data
            # Temporarily set is_fitted to True to compute ensemble scores
            self.is_fitted = True
            ensemble_scores = self._compute_ensemble_scores(normal_data)
            # Set threshold at 95th percentile of normal data scores
            self.threshold = np.percentile(ensemble_scores, 95)
        
        self.is_fitted = True
        total_time = time.time() - start_time
        logger.info(f"Ensemble training completed in {total_time:.2f}s. Decision threshold: {self.threshold:.4f}")
        
        return self
    
    def _compute_ensemble_scores(self, x: np.ndarray) -> np.ndarray:
        """
        Compute weighted ensemble anomaly scores.
        
        Args:
            x (np.ndarray): Input feature vectors
            
        Returns:
            np.ndarray: Ensemble anomaly scores (higher = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before computing scores")
        
        detector_scores = []
        
        for detector in self.detectors:
            if isinstance(detector, AutoencoderDetector):
                scores = detector.get_reconstruction_error(x)
                # Higher reconstruction error = more anomalous (already correct)
            else:
                scores = detector.predict_score(x)
                # Convert to anomaly scores (higher = more anomalous)
                scores = -scores
            
            detector_scores.append(scores)
        
        # Normalize scores to [0, 1] range for each detector
        normalized_scores = []
        for i, scores in enumerate(detector_scores):
            if hasattr(self, 'training_scores') and self.training_scores is not None:
                # Use training data statistics for normalization
                train_scores = self.training_scores[:, i]
                min_score = np.min(train_scores)
                max_score = np.max(train_scores)
            else:
                # Fallback to current batch statistics
                min_score = np.min(scores)
                max_score = np.max(scores)
            
            if max_score > min_score:
                normalized = (scores - min_score) / (max_score - min_score)
            else:
                normalized = np.zeros_like(scores)
            
            normalized_scores.append(normalized)
        
        # Compute weighted ensemble score
        ensemble_scores = np.zeros(len(x))
        for i, (scores, weight) in enumerate(zip(normalized_scores, self.weights)):
            ensemble_scores += weight * scores
        
        return ensemble_scores
    
    def _tune_threshold(self, 
                       validation_data: np.ndarray, 
                       validation_labels: np.ndarray, 
                       target_fpr: float) -> None:
        """
        Tune the decision threshold to achieve target FPR.
        
        Args:
            validation_data (np.ndarray): Validation feature vectors
            validation_labels (np.ndarray): Validation labels (0=normal, 1=anomaly)
            target_fpr (float): Target false positive rate
        """
        logger.info(f"Tuning threshold for target FPR: {target_fpr}")
        
        # Get ensemble scores on validation data
        val_scores = self._compute_ensemble_scores(validation_data)
        
        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(validation_labels, val_scores)
        
        # Find threshold that achieves target FPR
        target_idx = np.argmax(fpr >= target_fpr)
        if target_idx == 0 and fpr[0] > target_fpr:
            # If even the lowest threshold gives FPR > target, use highest threshold
            self.threshold = thresholds[0]
            actual_fpr = fpr[0]
        else:
            self.threshold = thresholds[target_idx]
            actual_fpr = fpr[target_idx]
        
        # Compute AUC for reference
        auc_score = auc(fpr, tpr)
        
        logger.info(f"Threshold tuned to {self.threshold:.4f} (actual FPR: {actual_fpr:.4f}, AUC: {auc_score:.4f})")
    
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Predict anomaly status with scores and feature importance.
        
        Args:
            x (np.ndarray): Input feature vectors. Shape: (n_samples, n_features)
            
        Returns:
            Tuple containing:
            - np.ndarray: Binary predictions (0=normal, 1=anomaly)
            - np.ndarray: Anomaly scores (higher = more anomalous)
            - Dict[str, Any]: Additional information including feature importance
            
        Raises:
            ValueError: If ensemble is not fitted or input has wrong shape
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions. Call fit() first.")
        
        if x.size == 0:
            raise ValueError("Input x cannot be empty")
        
        # Handle single sample case
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        if len(x.shape) != 2:
            raise ValueError("Input x must be 1D or 2D array")
        
        n_samples = x.shape[0]
        
        # Compute ensemble scores
        ensemble_scores = self._compute_ensemble_scores(x)
        
        # Make binary predictions
        binary_predictions = (ensemble_scores > self.threshold).astype(int)
        
        # Compute feature importance for anomalous samples
        feature_importance = self._compute_feature_importance(x, binary_predictions)
        
        # Prepare additional information
        info = {
            'feature_importance': feature_importance,
            'threshold': self.threshold,
            'detector_weights': dict(zip(self.detector_names, self.weights)),
            'n_anomalies': np.sum(binary_predictions),
            'anomaly_rate': np.mean(binary_predictions)
        }
        
        logger.debug(f"Processed {n_samples} samples. Anomalies detected: {info['n_anomalies']} ({info['anomaly_rate']:.2%})")
        
        return binary_predictions, ensemble_scores, info
    
    def _compute_feature_importance(self, 
                                  x: np.ndarray, 
                                  anomaly_predictions: np.ndarray) -> Dict[str, float]:
        """
        Compute feature importance for anomalous samples.
        
        This method analyzes which features contribute most to anomaly detection
        by examining the deviation of anomalous samples from normal patterns.
        
        Args:
            x (np.ndarray): Input feature vectors
            anomaly_predictions (np.ndarray): Binary anomaly predictions
            
        Returns:
            Dict[str, float]: Feature importance scores
        """
        if not hasattr(self, 'training_scores') or self.training_scores is None:
            # Fallback: return uniform importance
            n_features = x.shape[1]
            feature_names = self.feature_names or [f'feature_{i}' for i in range(n_features)]
            return {name: 1.0 / n_features for name in feature_names}
        
        # Get anomalous samples
        anomalous_indices = np.where(anomaly_predictions == 1)[0]
        
        if len(anomalous_indices) == 0:
            # No anomalies detected, return uniform importance
            n_features = x.shape[1]
            feature_names = self.feature_names or [f'feature_{i}' for i in range(n_features)]
            return {name: 0.0 for name in feature_names}
        
        anomalous_samples = x[anomalous_indices]
        
        # Compute feature importance based on deviation from normal patterns
        # Use training data statistics as baseline for normal behavior
        if hasattr(self, 'training_scores') and self.training_scores is not None:
            # Get normal data statistics from training (approximate)
            # This is a simplified approach - in practice, we'd store training data statistics
            normal_mean = np.mean(x[anomaly_predictions == 0], axis=0) if np.any(anomaly_predictions == 0) else np.zeros(x.shape[1])
            normal_std = np.std(x[anomaly_predictions == 0], axis=0) if np.any(anomaly_predictions == 0) else np.ones(x.shape[1])
            
            # Compute normalized deviation for anomalous samples
            deviations = np.abs(anomalous_samples - normal_mean) / (normal_std + 1e-8)
            
            # Average deviation across anomalous samples
            avg_deviation = np.mean(deviations, axis=0)
            
            # Normalize to get importance scores
            total_deviation = np.sum(avg_deviation)
            if total_deviation > 0:
                importance_scores = avg_deviation / total_deviation
            else:
                importance_scores = np.ones(len(avg_deviation)) / len(avg_deviation)
        else:
            # Fallback: uniform importance
            importance_scores = np.ones(x.shape[1]) / x.shape[1]
        
        # Create feature importance dictionary
        feature_names = self.feature_names or [f'feature_{i}' for i in range(x.shape[1])]
        feature_importance = dict(zip(feature_names, importance_scores))
        
        return feature_importance
    
    def get_detector_scores(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get individual detector scores for analysis.
        
        Args:
            x (np.ndarray): Input feature vectors
            
        Returns:
            Dict[str, np.ndarray]: Scores from each detector
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before getting detector scores")
        
        scores = {}
        
        for detector, name in zip(self.detectors, self.detector_names):
            if isinstance(detector, AutoencoderDetector):
                detector_scores = detector.get_reconstruction_error(x)
            else:
                detector_scores = detector.predict_score(x)
                # Convert to anomaly scores (higher = more anomalous)
                detector_scores = -detector_scores
            
            scores[name] = detector_scores
        
        return scores
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the ensemble and its components.
        
        Returns:
            Dict[str, Any]: Comprehensive model information
        """
        info = {
            'is_fitted': self.is_fitted,
            'weights': dict(zip(self.detector_names, self.weights)),
            'threshold': self.threshold,
            'n_detectors': len(self.detectors),
            'detector_names': self.detector_names
        }
        
        if self.is_fitted:
            info['n_features'] = len(self.feature_names) if self.feature_names else None
            
            # Get individual detector info
            detector_info = {}
            for detector, name in zip(self.detectors, self.detector_names):
                if hasattr(detector, 'get_model_info'):
                    detector_info[name] = detector.get_model_info()
            
            info['detector_info'] = detector_info
        
        return info
    
    def __repr__(self) -> str:
        """String representation of the ensemble detector."""
        status = "fitted" if self.is_fitted else "not fitted"
        weights_str = f"[{', '.join(f'{w:.2f}' for w in self.weights)}]"
        return f"EnsembleAnomalyDetector(weights={weights_str}, status={status})"