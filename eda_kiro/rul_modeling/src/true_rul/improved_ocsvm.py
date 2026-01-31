"""
Improved One-Class SVM Detector for anomaly detection in RUL prediction system.

This module implements an enhanced One-Class SVM for anomaly detection in
capacitor voltage data. The detector is trained on normal cycles and provides
anomaly scores based on the decision function.
"""

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ImprovedOCSVM:
    """
    Improved One-Class SVM anomaly detector for capacitor degradation detection.
    
    This detector uses One-Class SVM to learn the boundary of normal behavior
    from voltage time-series features. It includes improvements such as:
    - Automatic hyperparameter tuning
    - Feature scaling
    - Multiple kernel options
    - Robust threshold selection
    
    Attributes:
        nu (float): Upper bound on the fraction of training errors and lower bound
                   of the fraction of support vectors
        kernel (str): Kernel type for SVM
        model (OneClassSVM): The underlying scikit-learn One-Class SVM model
        scaler (StandardScaler): Feature scaler for normalization
        is_fitted (bool): Whether the model has been trained
        feature_names (Optional[List[str]]): Names of input features for interpretability
    """
    
    def __init__(self, 
                 kernel: str = "rbf", 
                 nu: float = 0.05, 
                 gamma: str = "scale",
                 auto_tune: bool = True,
                 random_state: int = 42):
        """
        Initialize the Improved One-Class SVM detector.
        
        Args:
            kernel (str): Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            nu (float): Upper bound on fraction of training errors and lower bound
                       of fraction of support vectors. Should be between 0 and 1.
            gamma (str or float): Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
            auto_tune (bool): Whether to automatically tune hyperparameters
            random_state (int): Random state for reproducibility
        """
        if not 0 < nu <= 1:
            raise ValueError("nu must be between 0 and 1")
        
        if kernel not in ['linear', 'poly', 'rbf', 'sigmoid']:
            raise ValueError("kernel must be one of: 'linear', 'poly', 'rbf', 'sigmoid'")
        
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.auto_tune = auto_tune
        self.random_state = random_state
        
        # Initialize model with default parameters
        self.model = OneClassSVM(
            kernel=kernel,
            nu=nu,
            gamma=gamma
        )
        
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names: Optional[List[str]] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.decision_threshold = 0.0  # Default threshold for One-Class SVM
        
        logger.info(f"Initialized ImprovedOCSVM with kernel={kernel}, nu={nu}, auto_tune={auto_tune}")
    
    def _tune_hyperparameters(self, normal_data: np.ndarray) -> Dict[str, Any]:
        """
        Automatically tune hyperparameters using grid search.
        
        Args:
            normal_data (np.ndarray): Normalized training data
            
        Returns:
            Dict[str, Any]: Best parameters found
        """
        logger.info("Tuning hyperparameters...")
        
        # Define parameter grid based on kernel
        if self.kernel == 'rbf':
            param_grid = {
                'nu': [0.01, 0.05, 0.1, 0.2],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]
            }
        elif self.kernel == 'linear':
            param_grid = {
                'nu': [0.01, 0.05, 0.1, 0.2]
            }
        elif self.kernel == 'poly':
            param_grid = {
                'nu': [0.01, 0.05, 0.1, 0.2],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'degree': [2, 3, 4]
            }
        else:  # sigmoid
            param_grid = {
                'nu': [0.01, 0.05, 0.1, 0.2],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            }
        
        # Custom scoring function for One-Class SVM
        def ocsvm_scorer(estimator, X):
            """Score based on the fraction of samples within the decision boundary."""
            predictions = estimator.predict(X)
            # We want most training samples to be classified as normal (1)
            # But not all (to avoid overfitting)
            normal_fraction = np.mean(predictions == 1)
            # Optimal fraction should be around (1 - nu)
            target_fraction = 1 - estimator.nu
            score = 1 - abs(normal_fraction - target_fraction)
            return score
        
        # Perform grid search
        grid_search = GridSearchCV(
            OneClassSVM(kernel=self.kernel),
            param_grid,
            scoring=ocsvm_scorer,
            cv=3,  # 3-fold CV
            n_jobs=-1,
            verbose=0
        )
        
        grid_search.fit(normal_data)
        
        best_params = grid_search.best_params_
        logger.info(f"Best parameters found: {best_params}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return best_params
    
    def fit(self, normal_data: np.ndarray, feature_names: Optional[List[str]] = None) -> 'ImprovedOCSVM':
        """
        Fit the One-Class SVM on normal cycles data.
        
        Args:
            normal_data (np.ndarray): Feature vectors from normal cycles.
                                    Shape: (n_samples, n_features)
            feature_names (Optional[List[str]]): Names of the features for interpretability
            
        Returns:
            ImprovedOCSVM: Self for method chaining
            
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
        
        logger.info(f"Fitting ImprovedOCSVM on {n_samples} normal samples with {n_features} features")
        
        # Store feature names for interpretability
        self.feature_names = feature_names
        
        # Scale the data
        normal_data_scaled = self.scaler.fit_transform(normal_data)
        
        # Tune hyperparameters if requested
        if self.auto_tune and n_samples >= 10:  # Need sufficient samples for tuning
            try:
                self.best_params = self._tune_hyperparameters(normal_data_scaled)
                
                # Update model with best parameters
                self.model = OneClassSVM(
                    kernel=self.kernel,
                    **self.best_params
                )
            except Exception as e:
                logger.warning(f"Hyperparameter tuning failed: {e}. Using default parameters.")
                self.best_params = None
        
        # Fit the model
        self.model.fit(normal_data_scaled)
        self.is_fitted = True
        
        # Compute decision scores on training data for threshold analysis
        train_scores = self.model.decision_function(normal_data_scaled)
        
        # Set decision threshold (default is 0 for One-Class SVM)
        # We could adjust this based on training data distribution
        self.decision_threshold = 0.0
        
        # Log training statistics
        n_support_vectors = len(self.model.support_)
        support_fraction = n_support_vectors / n_samples
        
        logger.info(f"Training completed. Support vectors: {n_support_vectors}/{n_samples} ({support_fraction:.2%})")
        logger.info(f"Training score range: [{train_scores.min():.3f}, {train_scores.max():.3f}]")
        logger.info(f"Decision threshold: {self.decision_threshold}")
        
        return self
    
    def predict_score(self, x: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores using decision function.
        
        The decision function returns the signed distance to the separating hyperplane.
        Higher scores indicate more normal behavior, while lower (more negative) scores
        indicate more anomalous behavior.
        
        Args:
            x (np.ndarray): Input feature vectors. Shape: (n_samples, n_features)
            
        Returns:
            np.ndarray: Decision function scores for each sample. Shape: (n_samples,)
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
        
        # Scale the input
        x_scaled = self.scaler.transform(x)
        
        # Get decision function scores
        scores = self.model.decision_function(x_scaled)
        
        logger.debug(f"Computed decision scores for {n_samples} samples. "
                    f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")
        
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
        
        # Scale the input
        x_scaled = self.scaler.transform(x)
        
        # Get binary predictions
        predictions = self.model.predict(x_scaled)
        return predictions
    
    def get_support_vectors(self) -> np.ndarray:
        """
        Get the support vectors from the fitted model.
        
        Returns:
            np.ndarray: Support vectors in the original feature space
            
        Raises:
            ValueError: If model is not fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get support vectors")
        
        # Transform support vectors back to original space
        support_vectors_scaled = self.model.support_vectors_
        support_vectors = self.scaler.inverse_transform(support_vectors_scaled)
        
        return support_vectors
    
    def get_decision_threshold(self) -> float:
        """
        Get the decision threshold used for binary classification.
        
        Returns:
            float: The threshold value. Scores below this are considered anomalous.
        """
        return self.decision_threshold
    
    def set_decision_threshold(self, threshold: float) -> None:
        """
        Set a custom decision threshold.
        
        Args:
            threshold (float): New threshold value
        """
        self.decision_threshold = threshold
        logger.info(f"Decision threshold updated to: {threshold}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the fitted model.
        
        Returns:
            Dict[str, Any]: Model information including parameters and statistics
        """
        info = {
            'kernel': self.kernel,
            'nu': self.nu,
            'gamma': self.gamma,
            'auto_tune': self.auto_tune,
            'is_fitted': self.is_fitted,
            'decision_threshold': self.decision_threshold
        }
        
        if self.is_fitted:
            info['n_support_vectors'] = len(self.model.support_)
            info['support_fraction'] = len(self.model.support_) / self.model.shape_fit_[0]
            info['n_features'] = len(self.feature_names) if self.feature_names else None
            
            if self.best_params:
                info['best_params'] = self.best_params
        
        return info
    
    def __repr__(self) -> str:
        """String representation of the detector."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"ImprovedOCSVM(kernel={self.kernel}, nu={self.nu}, status={status})"