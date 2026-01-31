"""
Unified RUL Regression Model Interface

This module implements the RULRegressionModel class that provides a unified
interface for all RUL regression models with factory method for model selection.

Requirements: 1.1
"""

import logging
from typing import Dict, Optional, Tuple, Any, Union
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from .gradient_boosting_predictor import GradientBoostingRULPredictor
from .random_forest_predictor import RandomForestRULPredictor
from .elastic_net_predictor import ElasticNetRULPredictor
from .hybrid_ensemble_predictor import HybridEnsembleRULPredictor

logger = logging.getLogger(__name__)


class RULRegressionModel(BaseEstimator, RegressorMixin):
    """
    Unified interface for RUL regression models
    
    Provides a factory method for model selection and unified interface
    for training, prediction, and interpretability across all model types.
    
    Supported model types:
    - "xgboost": XGBoost gradient boosting
    - "lightgbm": LightGBM gradient boosting
    - "random_forest": Random Forest regression
    - "elastic_net": Elastic Net linear regression
    - "ensemble": Hybrid ensemble of multiple models
    
    Attributes:
        model_type: Type of the underlying model
        model: The actual model instance
        feature_names: List of feature names
        is_trained: Whether the model has been trained
    """
    
    def __init__(self, model_type: str = "xgboost", **kwargs):
        """
        Initialize RUL regression model
        
        Args:
            model_type: One of "xgboost", "lightgbm", "random_forest", "elastic_net", "ensemble"
            **kwargs: Model-specific parameters passed to the underlying model
        
        Raises:
            ValueError: If model_type is not supported
        """
        self.model_type = model_type
        self.model = self._build_model(model_type, **kwargs)
        self.feature_names: Optional[list] = None
        self.is_trained: bool = False
        
        logger.info(f"Initialized RUL regression model with type: {model_type}")
    
    def _build_model(self, model_type: str, **kwargs) -> Union[
        GradientBoostingRULPredictor,
        RandomForestRULPredictor,
        ElasticNetRULPredictor,
        HybridEnsembleRULPredictor
    ]:
        """
        Factory method to build the specified model type
        
        Args:
            model_type: Type of model to build
            **kwargs: Model-specific parameters
        
        Returns:
            Instance of the specified model type
        
        Raises:
            ValueError: If model_type is not supported
        """
        if model_type == "xgboost":
            return GradientBoostingRULPredictor(model_type="xgboost", **kwargs)
        elif model_type == "lightgbm":
            return GradientBoostingRULPredictor(model_type="lightgbm", **kwargs)
        elif model_type == "random_forest":
            return RandomForestRULPredictor(**kwargs)
        elif model_type == "elastic_net":
            return ElasticNetRULPredictor(**kwargs)
        elif model_type == "ensemble":
            return HybridEnsembleRULPredictor(**kwargs)
        else:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Supported types: xgboost, lightgbm, random_forest, elastic_net, ensemble"
            )
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None,
        **kwargs
    ) -> "RULRegressionModel":
        """
        Train the RUL regression model
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training RUL labels (n_samples,)
            X_val: Validation features (optional)
            y_val: Validation RUL labels (optional)
            feature_names: List of feature names (optional)
            **kwargs: Additional training parameters passed to underlying model
        
        Returns:
            Self for method chaining
        
        Raises:
            ValueError: If input shapes are invalid
            RuntimeError: If training fails
        """
        # Validate inputs
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError(
                f"X_train and y_train must have same number of samples: "
                f"X_train={X_train.shape[0]}, y_train={y_train.shape[0]}"
            )
        
        if X_val is not None and y_val is not None:
            if X_val.shape[0] != y_val.shape[0]:
                raise ValueError(
                    f"X_val and y_val must have same number of samples: "
                    f"X_val={X_val.shape[0]}, y_val={y_val.shape[0]}"
                )
            if X_val.shape[1] != X_train.shape[1]:
                raise ValueError(
                    f"X_val and X_train must have same number of features: "
                    f"X_val={X_val.shape[1]}, X_train={X_train.shape[1]}"
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
            f"Training {self.model_type} model on {X_train.shape[0]} samples "
            f"with {X_train.shape[1]} features"
        )
        
        try:
            # Train the underlying model
            self.model.train(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                feature_names=self.feature_names,
                **kwargs
            )
            
            self.is_trained = True
            logger.info(f"{self.model_type} model training completed successfully")
            
        except Exception as e:
            logger.error(f"Training failed for {self.model_type} model: {e}")
            raise RuntimeError(f"Training failed for {self.model_type} model: {e}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict RUL for input features
        
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
        
        try:
            predictions = self.model.predict(X)
            return predictions
        except Exception as e:
            logger.error(f"Prediction failed for {self.model_type} model: {e}")
            raise RuntimeError(f"Prediction failed for {self.model_type} model: {e}")
    
    def predict_with_confidence(
        self,
        X: np.ndarray,
        confidence_level: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict RUL with confidence intervals
        
        Args:
            X: Input features (n_samples, n_features)
            confidence_level: Confidence level for intervals (default: 0.95)
        
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
            - predictions: Point predictions (n_samples,)
            - lower_bounds: Lower confidence bounds (n_samples,)
            - upper_bounds: Upper confidence bounds (n_samples,)
        
        Raises:
            RuntimeError: If model has not been trained
            ValueError: If input shape is invalid or confidence_level is invalid
            NotImplementedError: If underlying model doesn't support confidence intervals
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model has not been trained. Call train() before predict_with_confidence()."
            )
        
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Input features ({X.shape[1]}) must match training features "
                f"({len(self.feature_names)})"
            )
        
        if not 0 < confidence_level < 1:
            raise ValueError(
                f"confidence_level must be between 0 and 1, got {confidence_level}"
            )
        
        try:
            # Check if model supports confidence intervals
            if hasattr(self.model, 'predict_with_confidence'):
                return self.model.predict_with_confidence(X, confidence_level)
            else:
                # Fallback: use simple std-based confidence for models without native support
                logger.warning(
                    f"{self.model_type} model doesn't support native confidence intervals. "
                    f"Using fallback method."
                )
                return self._fallback_confidence_intervals(X, confidence_level)
                
        except Exception as e:
            logger.error(f"Confidence prediction failed for {self.model_type} model: {e}")
            raise RuntimeError(f"Confidence prediction failed for {self.model_type} model: {e}")
    
    def _fallback_confidence_intervals(
        self,
        X: np.ndarray,
        confidence_level: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fallback method for confidence intervals when model doesn't support them natively
        
        Uses a simple approach based on prediction uncertainty estimation.
        
        Args:
            X: Input features (n_samples, n_features)
            confidence_level: Confidence level for intervals
        
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        predictions = self.predict(X)
        
        # Simple fallback: use a fixed percentage of the prediction as uncertainty
        # This is a rough approximation and should be replaced with better methods
        # when possible (e.g., bootstrap, cross-validation, etc.)
        uncertainty_factor = 0.1  # 10% of prediction as uncertainty
        uncertainty = predictions * uncertainty_factor
        
        # Calculate z-score for confidence level
        z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z_score = z_scores.get(confidence_level, 1.96)
        
        lower_bounds = predictions - z_score * uncertainty
        upper_bounds = predictions + z_score * uncertainty
        
        # Ensure bounds are non-negative
        lower_bounds = np.maximum(lower_bounds, 0)
        upper_bounds = np.maximum(upper_bounds, 0)
        
        # Ensure lower <= prediction <= upper
        lower_bounds = np.minimum(lower_bounds, predictions)
        upper_bounds = np.maximum(upper_bounds, predictions)
        
        logger.warning(
            f"Using fallback confidence intervals with {uncertainty_factor:.1%} uncertainty factor"
        )
        
        return predictions, lower_bounds, upper_bounds
    
    def get_feature_importance(self, X: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Get feature importance for interpretability
        
        Args:
            X: Optional input for SHAP-based importance (used by some models)
        
        Returns:
            Dictionary mapping feature names to importance scores
        
        Raises:
            RuntimeError: If model has not been trained
            NotImplementedError: If underlying model doesn't support feature importance
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model has not been trained. Call train() before get_feature_importance()."
            )
        
        try:
            # Check for different feature importance methods
            if hasattr(self.model, 'get_aggregated_feature_importance'):
                # Ensemble models
                return self.model.get_aggregated_feature_importance(X)
            elif hasattr(self.model, 'get_feature_importance'):
                # Most models
                return self.model.get_feature_importance()
            else:
                raise NotImplementedError(
                    f"Feature importance not available for {self.model_type} model"
                )
                
        except Exception as e:
            logger.error(f"Feature importance failed for {self.model_type} model: {e}")
            raise RuntimeError(f"Feature importance failed for {self.model_type} model: {e}")
    
    def get_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Get SHAP values for detailed explanations
        
        Args:
            X: Input features (n_samples, n_features)
        
        Returns:
            SHAP values array (n_samples, n_features)
        
        Raises:
            RuntimeError: If model has not been trained
            NotImplementedError: If underlying model doesn't support SHAP values
            ValueError: If input shape is invalid
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model has not been trained. Call train() before get_shap_values()."
            )
        
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Input features ({X.shape[1]}) must match training features "
                f"({len(self.feature_names)})"
            )
        
        try:
            if hasattr(self.model, 'get_shap_values'):
                return self.model.get_shap_values(X)
            else:
                raise NotImplementedError(
                    f"SHAP values not available for {self.model_type} model"
                )
                
        except Exception as e:
            logger.error(f"SHAP values failed for {self.model_type} model: {e}")
            raise RuntimeError(f"SHAP values failed for {self.model_type} model: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata
        
        Returns:
            Dictionary with model information including unified interface metadata
        """
        # Get base info from underlying model
        if hasattr(self.model, 'get_model_info'):
            info = self.model.get_model_info()
        else:
            info = {}
        
        # Add unified interface metadata
        info.update({
            "unified_interface_version": "1.0",
            "wrapper_model_type": self.model_type,
            "is_trained": self.is_trained,
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "feature_names": self.feature_names,
            "supported_methods": self._get_supported_methods()
        })
        
        return info
    
    def _get_supported_methods(self) -> Dict[str, bool]:
        """
        Get information about which methods are supported by the underlying model
        
        Returns:
            Dictionary indicating which methods are available
        """
        return {
            "predict": True,  # All models support basic prediction
            "predict_with_confidence": hasattr(self.model, 'predict_with_confidence'),
            "get_feature_importance": (
                hasattr(self.model, 'get_feature_importance') or
                hasattr(self.model, 'get_aggregated_feature_importance')
            ),
            "get_shap_values": hasattr(self.model, 'get_shap_values'),
            "save_model": hasattr(self.model, 'save_model'),
            "load_model": hasattr(self.model, 'load_model')
        }
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk
        
        Args:
            filepath: Path to save the model
        
        Raises:
            RuntimeError: If model has not been trained
            NotImplementedError: If underlying model doesn't support saving
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model has not been trained. Call train() before save_model()."
            )
        
        try:
            if hasattr(self.model, 'save_model'):
                self.model.save_model(filepath)
                logger.info(f"Model saved to {filepath}")
            else:
                raise NotImplementedError(
                    f"Model saving not available for {self.model_type} model"
                )
                
        except Exception as e:
            logger.error(f"Model saving failed for {self.model_type} model: {e}")
            raise RuntimeError(f"Model saving failed for {self.model_type} model: {e}")
    
    def load_model(self, filepath: str) -> "RULRegressionModel":
        """
        Load a trained model from disk
        
        Args:
            filepath: Path to the saved model
        
        Returns:
            Self for method chaining
        
        Raises:
            NotImplementedError: If underlying model doesn't support loading
        """
        try:
            if hasattr(self.model, 'load_model'):
                self.model.load_model(filepath)
                self.is_trained = True
                
                # Update feature names from loaded model
                if hasattr(self.model, 'feature_names'):
                    self.feature_names = self.model.feature_names
                
                logger.info(f"Model loaded from {filepath}")
            else:
                raise NotImplementedError(
                    f"Model loading not available for {self.model_type} model"
                )
                
        except Exception as e:
            logger.error(f"Model loading failed for {self.model_type} model: {e}")
            raise RuntimeError(f"Model loading failed for {self.model_type} model: {e}")
        
        return self
    
    @classmethod
    def get_available_models(cls) -> Dict[str, str]:
        """
        Get list of available model types and their descriptions
        
        Returns:
            Dictionary mapping model types to descriptions
        """
        return {
            "xgboost": "XGBoost gradient boosting with SHAP interpretability",
            "lightgbm": "LightGBM gradient boosting with fast training",
            "random_forest": "Random Forest with quantile-based confidence intervals",
            "elastic_net": "Elastic Net linear regression with polynomial features",
            "ensemble": "Hybrid ensemble combining XGBoost, LightGBM, and Random Forest"
        }
    
    def __repr__(self) -> str:
        """String representation of the model"""
        status = "trained" if self.is_trained else "untrained"
        n_features = len(self.feature_names) if self.feature_names else "unknown"
        return (
            f"RULRegressionModel(model_type='{self.model_type}', "
            f"status='{status}', n_features={n_features})"
        )