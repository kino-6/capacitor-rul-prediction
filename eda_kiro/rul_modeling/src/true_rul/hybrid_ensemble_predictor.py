"""
Hybrid Ensemble RUL Predictor

This module implements the HybridEnsembleRULPredictor class that combines
multiple interpretable models (XGBoost, LightGBM, Random Forest) for robust
RUL predictions with ensemble-based confidence intervals.

Requirements: 1.1, 1.3, 9.1
"""

import logging
from typing import Dict, Optional, Tuple, Any, List
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from .gradient_boosting_predictor import GradientBoostingRULPredictor
from .random_forest_predictor import RandomForestRULPredictor

logger = logging.getLogger(__name__)


class HybridEnsembleRULPredictor(BaseEstimator, RegressorMixin):
    """
    Hybrid ensemble combining interpretable models for RUL prediction
    
    Combines multiple models with weighted voting:
    - XGBoost (40% weight): Best overall performance, native feature importance
    - LightGBM (40% weight): Fast training, handles large feature sets well
    - Random Forest (20% weight): Provides quantile-based confidence intervals
    
    Provides:
    - Robust predictions through ensemble of diverse models
    - Confidence intervals from ensemble variance
    - Aggregated feature importance across models
    - Multiple interpretability methods (SHAP, feature importance)
    
    Attributes:
        models: Dictionary of base models
        weights: Dictionary of model weights (must sum to 1.0)
        feature_names: List of feature names
        is_trained: Whether all models have been trained
    """
    
    def __init__(
        self,
        xgboost_params: Optional[Dict[str, Any]] = None,
        lightgbm_params: Optional[Dict[str, Any]] = None,
        random_forest_params: Optional[Dict[str, Any]] = None,
        weights: Optional[Dict[str, float]] = None,
        random_state: int = 42
    ):
        """
        Initialize HybridEnsembleRULPredictor
        
        Args:
            xgboost_params: Parameters for XGBoost model (optional)
            lightgbm_params: Parameters for LightGBM model (optional)
            random_forest_params: Parameters for Random Forest model (optional)
            weights: Custom weights for models (must sum to 1.0)
                Default: {'xgboost': 0.4, 'lightgbm': 0.4, 'random_forest': 0.2}
            random_state: Random seed for reproducibility
        
        Raises:
            ValueError: If weights don't sum to 1.0
        """
        self.random_state = random_state
        
        # Set default weights
        if weights is None:
            self.weights = {
                'xgboost': 0.4,
                'lightgbm': 0.4,
                'random_forest': 0.2
            }
        else:
            # Validate weights
            weight_sum = sum(weights.values())
            if not np.isclose(weight_sum, 1.0, atol=1e-6):
                raise ValueError(
                    f"Model weights must sum to 1.0, got {weight_sum}. "
                    f"Weights: {weights}"
                )
            self.weights = weights
        
        # Set default parameters for each model
        default_xgboost_params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state
        }
        default_lightgbm_params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': random_state
        }
        default_random_forest_params = {
            'n_estimators': 300,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': random_state
        }
        
        # Merge with user-provided parameters
        xgboost_params = {**default_xgboost_params, **(xgboost_params or {})}
        lightgbm_params = {**default_lightgbm_params, **(lightgbm_params or {})}
        random_forest_params = {**default_random_forest_params, **(random_forest_params or {})}
        
        # Initialize models
        self.models = {
            'xgboost': GradientBoostingRULPredictor(
                model_type='xgboost',
                **xgboost_params
            ),
            'lightgbm': GradientBoostingRULPredictor(
                model_type='lightgbm',
                **lightgbm_params
            ),
            'random_forest': RandomForestRULPredictor(
                **random_forest_params
            )
        }
        
        self.feature_names: Optional[list] = None
        self.is_trained: bool = False
        
        logger.info(
            f"Initialized Hybrid Ensemble RUL predictor with weights: "
            f"XGBoost={self.weights['xgboost']:.1%}, "
            f"LightGBM={self.weights['lightgbm']:.1%}, "
            f"Random Forest={self.weights['random_forest']:.1%}"
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
    ) -> "HybridEnsembleRULPredictor":
        """
        Train all base models in the ensemble
        
        Trains each model independently on the same training data.
        Models that support early stopping will use validation data.
        
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
            f"Training ensemble on {X_train.shape[0]} samples "
            f"with {X_train.shape[1]} features"
        )
        
        # Train each model
        for name, model in self.models.items():
            if verbose:
                logger.info(f"Training {name} model...")
            
            try:
                # Prepare training arguments based on model type
                train_args = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_val': X_val,
                    'y_val': y_val,
                    'feature_names': self.feature_names,
                    'verbose': verbose
                }
                
                # Only add early_stopping_rounds for models that support it
                if name in ['xgboost', 'lightgbm']:
                    train_args['early_stopping_rounds'] = early_stopping_rounds
                
                model.train(**train_args)
                
                if verbose:
                    logger.info(f"{name} training completed successfully")
                    
            except Exception as e:
                logger.error(f"Failed to train {name} model: {e}")
                raise RuntimeError(f"Training failed for {name} model: {e}")
        
        self.is_trained = True
        logger.info("All ensemble models trained successfully")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict RUL using weighted ensemble
        
        Combines predictions from all models using configured weights.
        
        Args:
            X: Input features (n_samples, n_features)
        
        Returns:
            Weighted ensemble predictions (n_samples,)
        
        Raises:
            RuntimeError: If models have not been trained
            ValueError: If input shape is invalid
        """
        if not self.is_trained:
            raise RuntimeError(
                "Models have not been trained. Call train() before predict()."
            )
        
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Input features ({X.shape[1]}) must match training features "
                f"({len(self.feature_names)})"
            )
        
        # Get predictions from each model
        predictions = []
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                weighted_pred = pred * self.weights[name]
                predictions.append(weighted_pred)
            except Exception as e:
                logger.error(f"Prediction failed for {name} model: {e}")
                raise RuntimeError(f"Prediction failed for {name} model: {e}")
        
        # Combine weighted predictions
        ensemble_pred = np.sum(predictions, axis=0)
        
        # Ensure non-negative predictions
        ensemble_pred = np.maximum(ensemble_pred, 0)
        
        return ensemble_pred
    
    def predict_with_confidence(
        self,
        X: np.ndarray,
        confidence_level: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict RUL with confidence intervals using ensemble variance
        
        Confidence intervals are derived from the variance of predictions
        across different models in the ensemble. Higher variance indicates
        higher uncertainty.
        
        Args:
            X: Input features (n_samples, n_features)
            confidence_level: Confidence level for intervals (default: 0.95)
        
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
            - predictions: Weighted ensemble predictions (n_samples,)
            - lower_bounds: Lower confidence bounds (n_samples,)
            - upper_bounds: Upper confidence bounds (n_samples,)
        
        Raises:
            RuntimeError: If models have not been trained
            ValueError: If input shape is invalid or confidence_level is invalid
        """
        if not self.is_trained:
            raise RuntimeError(
                "Models have not been trained. Call train() before predict_with_confidence()."
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
        
        # Get weighted ensemble prediction
        ensemble_pred = self.predict(X)
        
        # Get individual model predictions (unweighted for variance calculation)
        individual_predictions = []
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                individual_predictions.append(pred)
            except Exception as e:
                logger.error(f"Prediction failed for {name} model: {e}")
                raise RuntimeError(f"Prediction failed for {name} model: {e}")
        
        # Calculate ensemble variance
        # Shape: (n_models, n_samples) -> (n_samples,)
        individual_predictions = np.array(individual_predictions)
        ensemble_std = np.std(individual_predictions, axis=0)
        
        # Calculate confidence intervals using normal approximation
        # For 95% CI: z = 1.96
        z_score = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576
        }.get(confidence_level, 1.96)
        
        # If exact confidence level not in lookup, calculate z-score
        if confidence_level not in [0.90, 0.95, 0.99]:
            try:
                from scipy import stats
                z_score = stats.norm.ppf((1 + confidence_level) / 2)
            except ImportError:
                logger.warning("scipy not available, using z=1.96 for confidence intervals")
                z_score = 1.96
        
        lower_bounds = ensemble_pred - z_score * ensemble_std
        upper_bounds = ensemble_pred + z_score * ensemble_std
        
        # Ensure bounds are non-negative
        lower_bounds = np.maximum(lower_bounds, 0)
        upper_bounds = np.maximum(upper_bounds, 0)
        
        # Ensure lower <= prediction <= upper
        lower_bounds = np.minimum(lower_bounds, ensemble_pred)
        upper_bounds = np.maximum(upper_bounds, ensemble_pred)
        
        return ensemble_pred, lower_bounds, upper_bounds
    
    def get_aggregated_feature_importance(
        self,
        X: Optional[np.ndarray] = None,
        normalize: bool = True
    ) -> Dict[str, float]:
        """
        Get feature importance aggregated across all models
        
        Combines feature importance from each model using the ensemble weights.
        This provides a unified view of which features are most important
        across the entire ensemble.
        
        Args:
            X: Optional input for SHAP-based importance (not used currently)
            normalize: Whether to normalize importance scores to sum to 1.0
        
        Returns:
            Dictionary mapping feature names to aggregated importance scores
        
        Raises:
            RuntimeError: If models have not been trained
        """
        if not self.is_trained:
            raise RuntimeError(
                "Models have not been trained. Call train() before get_aggregated_feature_importance()."
            )
        
        # Initialize aggregated importance scores
        aggregated_importance = {name: 0.0 for name in self.feature_names}
        
        # Aggregate importance from each model
        for model_name, model in self.models.items():
            try:
                # Get feature importance from model
                importance = model.get_feature_importance()
                
                # Weight by model weight and add to aggregated scores
                weight = self.weights[model_name]
                for feature_name, score in importance.items():
                    if feature_name in aggregated_importance:
                        aggregated_importance[feature_name] += score * weight
                    else:
                        logger.warning(
                            f"Feature '{feature_name}' from {model_name} not found "
                            f"in feature_names. Skipping."
                        )
                        
            except Exception as e:
                logger.warning(
                    f"Failed to get feature importance from {model_name}: {e}. "
                    f"Skipping this model."
                )
                continue
        
        # Normalize if requested
        if normalize:
            total = sum(aggregated_importance.values())
            if total > 0:
                aggregated_importance = {
                    k: v / total for k, v in aggregated_importance.items()
                }
        
        # Sort by importance (descending)
        aggregated_importance = dict(
            sorted(aggregated_importance.items(), key=lambda x: x[1], reverse=True)
        )
        
        return aggregated_importance
    
    def get_individual_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions from each individual model
        
        Useful for debugging and understanding model disagreement.
        
        Args:
            X: Input features (n_samples, n_features)
        
        Returns:
            Dictionary mapping model names to their predictions
        
        Raises:
            RuntimeError: If models have not been trained
            ValueError: If input shape is invalid
        """
        if not self.is_trained:
            raise RuntimeError(
                "Models have not been trained. Call train() before get_individual_predictions()."
            )
        
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Input features ({X.shape[1]}) must match training features "
                f"({len(self.feature_names)})"
            )
        
        individual_predictions = {}
        for name, model in self.models.items():
            try:
                individual_predictions[name] = model.predict(X)
            except Exception as e:
                logger.error(f"Prediction failed for {name} model: {e}")
                individual_predictions[name] = None
        
        return individual_predictions
    
    def get_prediction_variance(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction variance across ensemble models
        
        Higher variance indicates higher disagreement between models,
        which suggests higher uncertainty in the prediction.
        
        Args:
            X: Input features (n_samples, n_features)
        
        Returns:
            Prediction variance for each sample (n_samples,)
        
        Raises:
            RuntimeError: If models have not been trained
            ValueError: If input shape is invalid
        """
        if not self.is_trained:
            raise RuntimeError(
                "Models have not been trained. Call train() before get_prediction_variance()."
            )
        
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Input features ({X.shape[1]}) must match training features "
                f"({len(self.feature_names)})"
            )
        
        # Get predictions from all models
        predictions = []
        for name, model in self.models.items():
            try:
                pred = model.predict(X)
                predictions.append(pred)
            except Exception as e:
                logger.error(f"Prediction failed for {name} model: {e}")
                raise RuntimeError(f"Prediction failed for {name} model: {e}")
        
        # Calculate variance across models
        predictions = np.array(predictions)
        variance = np.var(predictions, axis=0)
        
        return variance
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the ensemble and its models
        
        Returns:
            Dictionary with ensemble information
        """
        info = {
            "model_type": "hybrid_ensemble",
            "is_trained": self.is_trained,
            "n_features": len(self.feature_names) if self.feature_names else 0,
            "feature_names": self.feature_names,
            "weights": self.weights,
            "models": {}
        }
        
        # Get info from each model
        for name, model in self.models.items():
            try:
                info["models"][name] = model.get_model_info()
            except Exception as e:
                logger.warning(f"Failed to get info from {name} model: {e}")
                info["models"][name] = {"error": str(e)}
        
        return info
    
    def save_model(self, filepath_prefix: str) -> None:
        """
        Save all ensemble models to disk
        
        Saves each model with a suffix indicating its type.
        
        Args:
            filepath_prefix: Prefix for model files (e.g., "models/ensemble")
                Will create: ensemble_xgboost.model, ensemble_lightgbm.model, etc.
        
        Raises:
            RuntimeError: If models have not been trained
        """
        if not self.is_trained:
            raise RuntimeError(
                "Models have not been trained. Call train() before save_model()."
            )
        
        import joblib
        
        # Save each model
        for name, model in self.models.items():
            try:
                if name in ['xgboost', 'lightgbm']:
                    filepath = f"{filepath_prefix}_{name}.model"
                else:  # random_forest
                    filepath = f"{filepath_prefix}_{name}.joblib"
                
                model.save_model(filepath)
                logger.info(f"Saved {name} model to {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to save {name} model: {e}")
                raise RuntimeError(f"Failed to save {name} model: {e}")
        
        # Save ensemble metadata
        metadata = {
            'weights': self.weights,
            'feature_names': self.feature_names,
            'random_state': self.random_state
        }
        metadata_filepath = f"{filepath_prefix}_metadata.joblib"
        joblib.dump(metadata, metadata_filepath)
        logger.info(f"Saved ensemble metadata to {metadata_filepath}")
    
    def load_model(self, filepath_prefix: str) -> "HybridEnsembleRULPredictor":
        """
        Load all ensemble models from disk
        
        Args:
            filepath_prefix: Prefix for model files (same as used in save_model)
        
        Returns:
            Self for method chaining
        """
        import joblib
        
        # Load ensemble metadata
        metadata_filepath = f"{filepath_prefix}_metadata.joblib"
        try:
            metadata = joblib.load(metadata_filepath)
            self.weights = metadata['weights']
            self.feature_names = metadata['feature_names']
            self.random_state = metadata['random_state']
            logger.info(f"Loaded ensemble metadata from {metadata_filepath}")
        except Exception as e:
            logger.error(f"Failed to load ensemble metadata: {e}")
            raise RuntimeError(f"Failed to load ensemble metadata: {e}")
        
        # Load each model
        for name, model in self.models.items():
            try:
                if name in ['xgboost', 'lightgbm']:
                    filepath = f"{filepath_prefix}_{name}.model"
                else:  # random_forest
                    filepath = f"{filepath_prefix}_{name}.joblib"
                
                model.load_model(filepath)
                logger.info(f"Loaded {name} model from {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to load {name} model: {e}")
                raise RuntimeError(f"Failed to load {name} model: {e}")
        
        self.is_trained = True
        logger.info("All ensemble models loaded successfully")
        
        return self
