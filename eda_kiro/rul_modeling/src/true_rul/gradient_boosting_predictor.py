"""
Gradient Boosting RUL Predictor

This module implements the GradientBoostingRULPredictor class that supports
both XGBoost and LightGBM for RUL regression with interpretability features.

Requirements: 1.1, 1.2, 9.1, 9.4
"""

import logging
from typing import Dict, Optional, Tuple, Any
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import shap
from sklearn.base import BaseEstimator, RegressorMixin

logger = logging.getLogger(__name__)


class GradientBoostingRULPredictor(BaseEstimator, RegressorMixin):
    """
    Gradient boosting model for RUL prediction with interpretability
    
    Supports both XGBoost and LightGBM with:
    - Early stopping during training
    - Feature importance analysis
    - SHAP values for detailed explanations
    
    Attributes:
        model_type: Type of model ("xgboost" or "lightgbm")
        model: Trained gradient boosting model
        shap_explainer: SHAP TreeExplainer for interpretability
        feature_names: List of feature names
        is_trained: Whether the model has been trained
    """
    
    def __init__(
        self,
        model_type: str = "xgboost",
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
        **kwargs
    ):
        """
        Initialize GradientBoostingRULPredictor
        
        Args:
            model_type: Type of model ("xgboost" or "lightgbm")
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate (eta)
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            random_state: Random seed for reproducibility
            **kwargs: Additional model-specific parameters
        
        Raises:
            ValueError: If model_type is not "xgboost" or "lightgbm"
        """
        if model_type not in ["xgboost", "lightgbm"]:
            raise ValueError(
                f"model_type must be 'xgboost' or 'lightgbm', got '{model_type}'"
            )
        
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.kwargs = kwargs
        
        # Initialize model
        self.model = self._build_model()
        self.shap_explainer: Optional[shap.TreeExplainer] = None
        self.feature_names: Optional[list] = None
        self.is_trained: bool = False
        
        logger.info(
            f"Initialized {model_type} RUL predictor with "
            f"n_estimators={n_estimators}, max_depth={max_depth}, "
            f"learning_rate={learning_rate}"
        )
    
    def _build_model(self) -> Any:
        """
        Build the gradient boosting model
        
        Returns:
            XGBoost or LightGBM regressor
        """
        if self.model_type == "xgboost":
            return xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                objective='reg:squarederror',
                random_state=self.random_state,
                **self.kwargs
            )
        else:  # lightgbm
            return lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_state,
                verbose=-1,  # Suppress warnings
                **self.kwargs
            )
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = False
    ) -> "GradientBoostingRULPredictor":
        """
        Train the gradient boosting model with early stopping
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training RUL labels (n_samples,)
            X_val: Validation features (optional)
            y_val: Validation RUL labels (optional)
            feature_names: List of feature names (optional)
            early_stopping_rounds: Number of rounds for early stopping
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
        
        # Train with early stopping if validation data provided
        if X_val is not None and y_val is not None:
            if self.model_type == "xgboost":
                # XGBoost 2.0+ uses callbacks for early stopping
                self.model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=verbose
                )
            else:  # lightgbm
                self.model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=early_stopping_rounds),
                        lgb.log_evaluation(period=0 if not verbose else 100)
                    ]
                )
        else:
            # Train without early stopping
            self.model.fit(X_train, y_train)
        
        # Initialize SHAP explainer after training
        try:
            self.shap_explainer = shap.TreeExplainer(self.model)
            logger.info("SHAP explainer initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize SHAP explainer: {e}")
            self.shap_explainer = None
        
        self.is_trained = True
        logger.info(f"Training completed. Model is ready for predictions.")
        
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
        
        predictions = self.model.predict(X)
        
        # Ensure non-negative predictions (RUL cannot be negative)
        predictions = np.maximum(predictions, 0)
        
        return predictions
    
    def get_feature_importance(
        self,
        importance_type: str = "gain"
    ) -> Dict[str, float]:
        """
        Get feature importance scores
        
        Args:
            importance_type: Type of importance for XGBoost
                ("gain", "weight", "cover", "total_gain", "total_cover")
                For LightGBM, uses "split" importance
        
        Returns:
            Dictionary mapping feature names to importance scores
        
        Raises:
            RuntimeError: If model has not been trained
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model has not been trained. Call train() before get_feature_importance()."
            )
        
        if self.model_type == "xgboost":
            # XGBoost supports multiple importance types
            importance = self.model.get_booster().get_score(
                importance_type=importance_type
            )
            # Map feature indices to names
            importance_dict = {}
            for key, value in importance.items():
                # XGBoost uses f0, f1, ... format
                if key.startswith('f'):
                    idx = int(key[1:])
                    if idx < len(self.feature_names):
                        importance_dict[self.feature_names[idx]] = value
                else:
                    importance_dict[key] = value
        else:  # lightgbm
            # LightGBM uses feature_importances_
            importance_values = self.model.feature_importances_
            importance_dict = dict(zip(self.feature_names, importance_values))
        
        # Sort by importance (descending)
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        return importance_dict
    
    def get_shap_values(
        self,
        X: np.ndarray,
        check_additivity: bool = False
    ) -> np.ndarray:
        """
        Get SHAP values for interpretability using TreeExplainer
        
        SHAP (SHapley Additive exPlanations) values explain the contribution
        of each feature to the prediction for each sample.
        
        Args:
            X: Input features (n_samples, n_features)
            check_additivity: Whether to check SHAP value additivity
        
        Returns:
            SHAP values array (n_samples, n_features)
        
        Raises:
            RuntimeError: If model has not been trained or SHAP explainer not initialized
            ValueError: If input shape is invalid
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model has not been trained. Call train() before get_shap_values()."
            )
        
        if self.shap_explainer is None:
            raise RuntimeError(
                "SHAP explainer not initialized. This may happen if SHAP "
                "initialization failed during training."
            )
        
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Input features ({X.shape[1]}) must match training features "
                f"({len(self.feature_names)})"
            )
        
        try:
            shap_values = self.shap_explainer.shap_values(
                X,
                check_additivity=check_additivity
            )
            return shap_values
        except Exception as e:
            logger.error(f"Failed to compute SHAP values: {e}")
            raise RuntimeError(f"SHAP value computation failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata
        
        Returns:
            Dictionary with model information
        """
        info = {
            "model_type": self.model_type,
            "is_trained": self.is_trained,
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "feature_names": self.feature_names,
            "hyperparameters": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "subsample": self.subsample,
                "colsample_bytree": self.colsample_bytree,
                "random_state": self.random_state,
            }
        }
        
        if self.is_trained:
            if self.model_type == "xgboost":
                info["n_trees"] = self.model.get_booster().num_boosted_rounds()
            else:  # lightgbm
                info["n_trees"] = self.model.booster_.num_trees()
        
        return info
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk
        
        Args:
            filepath: Path to save the model
        
        Raises:
            RuntimeError: If model has not been trained
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model has not been trained. Call train() before save_model()."
            )
        
        if self.model_type == "xgboost":
            self.model.save_model(filepath)
        else:  # lightgbm
            self.model.booster_.save_model(filepath)
        
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> "GradientBoostingRULPredictor":
        """
        Load a trained model from disk
        
        Args:
            filepath: Path to the saved model
        
        Returns:
            Self for method chaining
        """
        if self.model_type == "xgboost":
            self.model.load_model(filepath)
        else:  # lightgbm
            self.model = lgb.Booster(model_file=filepath)
        
        self.is_trained = True
        
        # Reinitialize SHAP explainer
        try:
            self.shap_explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            logger.warning(f"Failed to initialize SHAP explainer after loading: {e}")
            self.shap_explainer = None
        
        logger.info(f"Model loaded from {filepath}")
        
        return self
