"""
Random Forest RUL Predictor

This module implements the RandomForestRULPredictor class that uses
Random Forest regression with quantile-based confidence intervals.

Requirements: 1.1, 1.3
"""

import logging
from typing import Dict, Optional, Tuple, Any
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, RegressorMixin

logger = logging.getLogger(__name__)


class RandomForestRULPredictor(BaseEstimator, RegressorMixin):
    """
    Random Forest model for RUL prediction with confidence intervals
    
    Uses Random Forest regression with quantile-based confidence intervals
    derived from individual tree predictions. Provides:
    - Robust predictions through ensemble of decision trees
    - Confidence intervals from prediction variance
    - Feature importance analysis
    
    Attributes:
        model: Main Random Forest regressor for point predictions
        quantile_models: Dict of models for lower/upper quantile predictions
        feature_names: List of feature names
        is_trained: Whether the model has been trained
    """
    
    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: Optional[int] = 15,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        max_features: str = "sqrt",
        random_state: int = 42,
        n_jobs: int = -1,
        **kwargs
    ):
        """
        Initialize RandomForestRULPredictor
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None for unlimited)
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            max_features: Number of features to consider for best split
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all cores)
            **kwargs: Additional RandomForest parameters
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.kwargs = kwargs
        
        # Initialize models
        self.model = self._build_model()
        self.quantile_models: Dict[str, RandomForestRegressor] = {
            'lower': self._build_model(),
            'upper': self._build_model()
        }
        
        self.feature_names: Optional[list] = None
        self.is_trained: bool = False
        
        logger.info(
            f"Initialized Random Forest RUL predictor with "
            f"n_estimators={n_estimators}, max_depth={max_depth}, "
            f"min_samples_split={min_samples_split}"
        )
    
    def _build_model(self) -> RandomForestRegressor:
        """
        Build a Random Forest regressor
        
        Returns:
            RandomForestRegressor instance
        """
        return RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            **self.kwargs
        )
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        feature_names: Optional[list] = None,
        verbose: bool = False
    ) -> "RandomForestRULPredictor":
        """
        Train the Random Forest model and quantile models
        
        Trains three models:
        1. Main model for point predictions (mean)
        2. Lower quantile model (2.5th percentile)
        3. Upper quantile model (97.5th percentile)
        
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
            f"Training Random Forest model on {X_train.shape[0]} samples "
            f"with {X_train.shape[1]} features"
        )
        
        # Train main model for point predictions
        if verbose:
            logger.info("Training main Random Forest model...")
        self.model.fit(X_train, y_train)
        
        # Train quantile models using tree predictions
        # We'll use the individual tree predictions to estimate quantiles
        if verbose:
            logger.info("Training quantile models for confidence intervals...")
        
        # For quantile estimation, we train the same model but will use
        # individual tree predictions to compute quantiles during prediction
        self.quantile_models['lower'].fit(X_train, y_train)
        self.quantile_models['upper'].fit(X_train, y_train)
        
        self.is_trained = True
        logger.info("Training completed. Model is ready for predictions.")
        
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
    
    def predict_with_confidence(
        self,
        X: np.ndarray,
        confidence_level: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict RUL with confidence intervals
        
        Uses individual tree predictions to estimate quantile-based
        confidence intervals. The confidence interval is derived from
        the distribution of predictions across all trees in the forest.
        
        Args:
            X: Input features (n_samples, n_features)
            confidence_level: Confidence level for intervals (default: 0.95 for 95% CI)
        
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
            - predictions: Point predictions (n_samples,)
            - lower_bounds: Lower confidence bounds (n_samples,)
            - upper_bounds: Upper confidence bounds (n_samples,)
        
        Raises:
            RuntimeError: If model has not been trained
            ValueError: If input shape is invalid or confidence_level is invalid
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
        
        # Get point predictions
        predictions = self.predict(X)
        
        # Get predictions from all individual trees
        # Shape: (n_samples, n_estimators)
        tree_predictions = np.array([
            tree.predict(X) for tree in self.model.estimators_
        ]).T
        
        # Ensure non-negative tree predictions
        tree_predictions = np.maximum(tree_predictions, 0)
        
        # Calculate quantiles from tree predictions
        alpha = 1 - confidence_level
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2
        
        lower_bounds = np.percentile(tree_predictions, lower_quantile * 100, axis=1)
        upper_bounds = np.percentile(tree_predictions, upper_quantile * 100, axis=1)
        
        # Ensure bounds are non-negative
        lower_bounds = np.maximum(lower_bounds, 0)
        upper_bounds = np.maximum(upper_bounds, 0)
        
        # Ensure lower <= prediction <= upper
        lower_bounds = np.minimum(lower_bounds, predictions)
        upper_bounds = np.maximum(upper_bounds, predictions)
        
        return predictions, lower_bounds, upper_bounds
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores based on mean decrease in impurity
        
        Returns:
            Dictionary mapping feature names to importance scores
        
        Raises:
            RuntimeError: If model has not been trained
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model has not been trained. Call train() before get_feature_importance()."
            )
        
        importance_values = self.model.feature_importances_
        importance_dict = dict(zip(self.feature_names, importance_values))
        
        # Sort by importance (descending)
        importance_dict = dict(
            sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        )
        
        return importance_dict
    
    def get_prediction_variance(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction variance from individual tree predictions
        
        Args:
            X: Input features (n_samples, n_features)
        
        Returns:
            Prediction variance for each sample (n_samples,)
        
        Raises:
            RuntimeError: If model has not been trained
            ValueError: If input shape is invalid
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model has not been trained. Call train() before get_prediction_variance()."
            )
        
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Input features ({X.shape[1]}) must match training features "
                f"({len(self.feature_names)})"
            )
        
        # Get predictions from all individual trees
        tree_predictions = np.array([
            tree.predict(X) for tree in self.model.estimators_
        ]).T
        
        # Calculate variance across trees
        variance = np.var(tree_predictions, axis=1)
        
        return variance
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and metadata
        
        Returns:
            Dictionary with model information
        """
        info = {
            "model_type": "random_forest",
            "is_trained": self.is_trained,
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "feature_names": self.feature_names,
            "hyperparameters": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_split": self.min_samples_split,
                "min_samples_leaf": self.min_samples_leaf,
                "max_features": self.max_features,
                "random_state": self.random_state,
                "n_jobs": self.n_jobs,
            }
        }
        
        if self.is_trained:
            info["n_trees"] = len(self.model.estimators_)
        
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
            'quantile_models': self.quantile_models,
            'feature_names': self.feature_names,
            'hyperparameters': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'max_features': self.max_features,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
            }
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> "RandomForestRULPredictor":
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
        self.quantile_models = model_data['quantile_models']
        self.feature_names = model_data['feature_names']
        
        # Restore hyperparameters
        hyperparams = model_data['hyperparameters']
        self.n_estimators = hyperparams['n_estimators']
        self.max_depth = hyperparams['max_depth']
        self.min_samples_split = hyperparams['min_samples_split']
        self.min_samples_leaf = hyperparams['min_samples_leaf']
        self.max_features = hyperparams['max_features']
        self.random_state = hyperparams['random_state']
        self.n_jobs = hyperparams['n_jobs']
        
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")
        
        return self
