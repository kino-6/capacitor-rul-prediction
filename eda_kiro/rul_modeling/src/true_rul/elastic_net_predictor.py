"""
Elastic Net RUL Predictor

This module implements the ElasticNetRULPredictor class that uses
Elastic Net regression with polynomial features for fully interpretable
linear RUL predictions.

Requirements: 1.1, 9.1
"""

import logging
from typing import Dict, Optional, Any
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin

logger = logging.getLogger(__name__)


class ElasticNetRULPredictor(BaseEstimator, RegressorMixin):
    """
    Elastic Net model for RUL prediction with polynomial features
    
    Uses Elastic Net regression (combination of L1 and L2 regularization)
    with polynomial feature expansion for interpretable linear predictions.
    Provides:
    - Fully interpretable linear coefficients
    - Regularization to prevent overfitting
    - Fast training and inference
    - Easy to understand feature contributions
    
    Attributes:
        poly: PolynomialFeatures transformer
        scaler: StandardScaler for feature normalization
        model: Elastic Net regressor
        feature_names: List of original feature names
        poly_feature_names: List of polynomial feature names
        is_trained: Whether the model has been trained
    """
    
    def __init__(
        self,
        degree: int = 2,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 10000,
        tol: float = 1e-4,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize ElasticNetRULPredictor
        
        Args:
            degree: Degree of polynomial features (1=linear, 2=quadratic, etc.)
            alpha: Regularization strength (higher = more regularization)
            l1_ratio: Mix of L1 and L2 regularization (0=L2 only, 1=L1 only, 0.5=equal mix)
            max_iter: Maximum number of iterations for optimization
            tol: Tolerance for optimization convergence
            random_state: Random seed for reproducibility
            **kwargs: Additional ElasticNet parameters
        """
        self.degree = degree
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.kwargs = kwargs
        
        # Initialize components
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.scaler = StandardScaler()
        self.model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            **kwargs
        )
        
        self.feature_names: Optional[list] = None
        self.poly_feature_names: Optional[list] = None
        self.is_trained: bool = False
        
        logger.info(
            f"Initialized Elastic Net RUL predictor with "
            f"degree={degree}, alpha={alpha}, l1_ratio={l1_ratio}"
        )
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None,
        verbose: bool = False
    ) -> "ElasticNetRULPredictor":
        """
        Train the Elastic Net model with polynomial features and feature scaling
        
        Training pipeline:
        1. Create polynomial features from input
        2. Scale features using StandardScaler
        3. Fit Elastic Net model
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training RUL labels (n_samples,)
            X_val: Validation features (optional, not used but kept for API consistency)
            y_val: Validation RUL labels (optional, not used but kept for API consistency)
            feature_names: List of feature names (optional)
            verbose: Whether to print training progress
        
        Returns:
            Self for method chaining
        
        Raises:
            ValueError: If input shapes are invalid
        """
        # Validate inputs
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"X_train and y_train must have same number of samples: "
                f"X_train={X_train.shape[0]}, y_train={y_train.shape[0]}"
            )
        
        # Store feature names
        if feature_names is not None:
            if len(feature_names) != X_train.shape[1]:
                raise ValueError(
                    f"Number of feature names ({len(feature_names)}) must match "
                    f"number of features ({X_train.shape[1]})"
                )
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        logger.info(
            f"Training Elastic Net model on {X_train.shape[0]} samples "
            f"with {X_train.shape[1]} features"
        )
        
        # Step 1: Create polynomial features
        if verbose:
            logger.info(f"Creating polynomial features (degree={self.degree})...")
        X_poly = self.poly.fit_transform(X_train)
        
        # Get polynomial feature names
        self.poly_feature_names = self.poly.get_feature_names_out(self.feature_names)
        
        if verbose:
            logger.info(
                f"Expanded to {X_poly.shape[1]} polynomial features "
                f"(from {X_train.shape[1]} original features)"
            )
        
        # Step 2: Scale features
        if verbose:
            logger.info("Scaling features...")
        X_scaled = self.scaler.fit_transform(X_poly)
        
        # Step 3: Fit Elastic Net model
        if verbose:
            logger.info("Fitting Elastic Net model...")
        self.model.fit(X_scaled, y_train)
        
        # Check convergence
        if self.model.n_iter_ >= self.max_iter:
            logger.warning(
                f"Elastic Net did not converge after {self.max_iter} iterations. "
                "Consider increasing max_iter or adjusting alpha."
            )
        elif verbose:
            logger.info(f"Model converged after {self.model.n_iter_} iterations")
        
        # Log sparsity information (how many coefficients are zero due to L1 regularization)
        n_nonzero = np.sum(self.model.coef_ != 0)
        n_total = len(self.model.coef_)
        sparsity = 1 - (n_nonzero / n_total)
        
        if verbose:
            logger.info(
                f"Model sparsity: {sparsity:.2%} "
                f"({n_total - n_nonzero}/{n_total} coefficients set to zero)"
            )
        
        self.is_trained = True
        logger.info("Training completed. Model is ready for predictions.")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict RUL for input features
        
        Applies the same transformation pipeline as training:
        1. Create polynomial features
        2. Scale features
        3. Apply linear model
        
        Args:
            X: Input features (n_samples, n_features)
        
        Returns:
            Predicted RUL values (n_samples,)
        
        Raises:
            RuntimeError: If model has not been trained
            ValueError: If input shape is invalid
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model has not been trained. Call train() before predict()."
            )
        
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Input features ({X.shape[1]}) must match training features "
                f"({len(self.feature_names)})"
            )
        
        # Apply transformation pipeline
        X_poly = self.poly.transform(X)
        X_scaled = self.scaler.transform(X_poly)
        predictions = self.model.predict(X_scaled)
        
        # Ensure non-negative predictions (RUL cannot be negative)
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def get_feature_coefficients(
        self,
        include_zero: bool = False,
        top_k: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Get model coefficients for interpretability
        
        Returns the linear coefficients for each polynomial feature,
        which directly show how each feature contributes to the prediction.
        Positive coefficients increase RUL, negative coefficients decrease it.
        
        Args:
            include_zero: Whether to include features with zero coefficients
            top_k: If specified, return only top k features by absolute coefficient value
        
        Returns:
            Dictionary mapping polynomial feature names to coefficient values
        
        Raises:
            RuntimeError: If model has not been trained
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model has not been trained. Call train() before get_feature_coefficients()."
            )
        
        # Create coefficient dictionary
        coef_dict = dict(zip(self.poly_feature_names, self.model.coef_))
        
        # Filter zero coefficients if requested
        if not include_zero:
            coef_dict = {k: v for k, v in coef_dict.items() if v != 0}
        
        # Sort by absolute value (descending)
        coef_dict = dict(
            sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        )
        
        # Return top k if specified
        if top_k is not None:
            coef_dict = dict(list(coef_dict.items())[:top_k])
        
        return coef_dict
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores based on absolute coefficient values
        
        For linear models, feature importance is derived from the absolute
        value of coefficients (after scaling). This shows which features
        have the strongest influence on predictions.
        
        Returns:
            Dictionary mapping original feature names to importance scores
        
        Raises:
            RuntimeError: If model has not been trained
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model has not been trained. Call train() before get_feature_importance()."
            )
        
        # Get absolute coefficient values
        abs_coefs = np.abs(self.model.coef_)
        
        # Map polynomial features back to original features
        # For each original feature, sum the absolute coefficients of all
        # polynomial terms that include that feature
        importance_dict = {name: 0.0 for name in self.feature_names}
        
        for poly_name, abs_coef in zip(self.poly_feature_names, abs_coefs):
            # Parse polynomial feature name to find contributing original features
            # e.g., "feature_0^2" or "feature_0 feature_1"
            for orig_name in self.feature_names:
                if orig_name in poly_name:
                    importance_dict[orig_name] += abs_coef
        
        # Normalize to sum to 1.0
        total = sum(importance_dict.values())
        if total > 0:
            importance_dict = {k: v / total for k, v in importance_dict.items()}
        
        # Sort by importance (descending)
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        return importance_dict
    
    def get_intercept(self) -> float:
        """
        Get the model intercept (bias term)
        
        Returns:
            Intercept value
        
        Raises:
            RuntimeError: If model has not been trained
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model has not been trained. Call train() before get_intercept()."
            )
        
        return float(self.model.intercept_)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata
        
        Returns:
            Dictionary with model information
        """
        info = {
            "model_type": "elastic_net",
            "is_trained": self.is_trained,
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "n_poly_features": len(self.poly_feature_names) if self.poly_feature_names is not None else 0,
            "feature_names": self.feature_names,
            "hyperparameters": {
                "degree": self.degree,
                "alpha": self.alpha,
                "l1_ratio": self.l1_ratio,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "random_state": self.random_state,
            }
        }
        
        if self.is_trained:
            info["n_iterations"] = self.model.n_iter_
            info["intercept"] = float(self.model.intercept_)
            info["n_nonzero_coefs"] = int(np.sum(self.model.coef_ != 0))
            info["sparsity"] = float(1 - (np.sum(self.model.coef_ != 0) / len(self.model.coef_)))
        
        return info
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk using joblib
        
        Args:
            filepath: Path to save the model
        
        Raises:
            RuntimeError: If model has not been trained
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model has not been trained. Call train() before save_model()."
            )
        
        import joblib
        
        model_data = {
            'model': self.model,
            'poly': self.poly,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'poly_feature_names': self.poly_feature_names,
            'hyperparameters': {
                'degree': self.degree,
                'alpha': self.alpha,
                'l1_ratio': self.l1_ratio,
                'max_iter': self.max_iter,
                'tol': self.tol,
                'random_state': self.random_state,
            }
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> "ElasticNetRULPredictor":
        """
        Load a trained model from disk
        
        Args:
            filepath: Path to the saved model
        
        Returns:
            Self for method chaining
        """
        import joblib
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.poly = model_data['poly']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.poly_feature_names = model_data['poly_feature_names']
        
        # Restore hyperparameters
        hyperparams = model_data['hyperparameters']
        self.degree = hyperparams['degree']
        self.alpha = hyperparams['alpha']
        self.l1_ratio = hyperparams['l1_ratio']
        self.max_iter = hyperparams['max_iter']
        self.tol = hyperparams['tol']
        self.random_state = hyperparams['random_state']
        
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
        
        return self
