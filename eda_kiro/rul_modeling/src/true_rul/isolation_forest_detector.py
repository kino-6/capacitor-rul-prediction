"""
Isolation Forest Detector for anomaly detection in RUL prediction system.

This module implements an Isolation Forest-based anomaly detector that identifies
abnormal behavior patterns in capacitor voltage data. The detector is trained on
normal cycles (1-10) and provides anomaly scores for new samples.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class IsolationForestDetector:
    """
    Isolation Forest-based anomaly detector for capacitor degradation detection.
    
    This detector uses the Isolation Forest algorithm to identify anomalous patterns
    in feature vectors extracted from voltage time-series data. It's trained on
    normal cycles (typically cycles 1-10) and provides anomaly scores for new samples.
    
    Attributes:
        contamination (float): Expected proportion of outliers in the dataset
        model (IsolationForest): The underlying scikit-learn Isolation Forest model
        is_fitted (bool): Whether the model has been trained
        feature_names (Optional[List[str]]): Names of input features for interpretability
    """
    
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        """
        Initialize the Isolation Forest detector.
        
        Args:
            contamination (float): Expected proportion of outliers in the dataset.
                                 Should be between 0 and 0.5. Default is 0.05 (5%).
            random_state (int): Random state for reproducibility
        """
        if not 0 < contamination <= 0.5:
            raise ValueError("contamination must be between 0 and 0.5")
            
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples='auto',
            max_features=1.0,
            bootstrap=False,
            n_jobs=-1
        )
        self.is_fitted = False
        self.feature_names: Optional[list] = None
        
        logger.info(f"Initialized IsolationForestDetector with contamination={contamination}")
    
    def fit(self, normal_data: np.ndarray, feature_names: Optional[list] = None) -> 'IsolationForestDetector':
        """
        Fit the Isolation Forest on normal cycles data.
        
        This method trains the detector on data from normal cycles (typically cycles 1-10)
        to learn the patterns of normal behavior. The detector will then identify
        deviations from these patterns as anomalies.
        
        Args:
            normal_data (np.ndarray): Feature vectors from normal cycles.
                                    Shape: (n_samples, n_features)
            feature_names (Optional[list]): Names of the features for interpretability
            
        Returns:
            IsolationForestDetector: Self for method chaining
            
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
        
        logger.info(f"Fitting IsolationForest on {n_samples} normal samples with {n_features} features")
        
        # Store feature names for interpretability
        self.feature_names = feature_names
        
        # Fit the model
        self.model.fit(normal_data)
        self.is_fitted = True
        
        # Log training statistics
        train_scores = self.model.decision_function(normal_data)
        logger.info(f"Training completed. Normal data score range: [{train_scores.min():.3f}, {train_scores.max():.3f}]")
        
        return self
    
    def predict_score(self, x: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores for input samples.
        
        The anomaly score is based on the decision function of the Isolation Forest.
        Higher scores indicate more normal behavior, while lower (more negative) scores
        indicate more anomalous behavior.
        
        Args:
            x (np.ndarray): Input feature vectors. Shape: (n_samples, n_features)
            
        Returns:
            np.ndarray: Anomaly scores for each sample. Shape: (n_samples,)
                       Higher scores = more normal, lower scores = more anomalous
                       
        Raises:
            ValueError: If model is not fitted or input has wrong shape
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")
        
        if x.size == 0:
            raise ValueError("Input x cannot be empty")
        
        # Handle single sample case
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        if len(x.shape) != 2:
            raise ValueError("Input x must be 1D or 2D array")
        
        n_samples, n_features = x.shape
        
        # Get decision function scores (higher = more normal)
        scores = self.model.decision_function(x)
        
        logger.debug(f"Computed anomaly scores for {n_samples} samples. Score range: [{scores.min():.3f}, {scores.max():.3f}]")
        
        return scores
    
    def predict_binary(self, x: np.ndarray) -> np.ndarray:
        """
        Return binary anomaly predictions.
        
        Args:
            x (np.ndarray): Input feature vectors. Shape: (n_samples, n_features)
            
        Returns:
            np.ndarray: Binary predictions. Shape: (n_samples,)
                       1 = normal, -1 = anomaly (following sklearn convention)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")
        
        # Handle single sample case
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        predictions = self.model.predict(x)
        return predictions
    
    def get_anomaly_threshold(self) -> float:
        """
        Get the decision threshold used for binary classification.
        
        Returns:
            float: The threshold value. Scores below this are considered anomalous.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get threshold")
        
        return self.model.offset_
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the fitted model.
        
        Returns:
            Dict[str, Any]: Model information including parameters and statistics
        """
        info = {
            'contamination': self.contamination,
            'is_fitted': self.is_fitted,
            'n_estimators': self.model.n_estimators,
            'max_samples': self.model.max_samples,
            'max_features': self.model.max_features
        }
        
        if self.is_fitted:
            info['threshold'] = self.model.offset_
            info['n_features'] = len(self.feature_names) if self.feature_names else None
        
        return info
    
    def __repr__(self) -> str:
        """String representation of the detector."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"IsolationForestDetector(contamination={self.contamination}, status={status})"