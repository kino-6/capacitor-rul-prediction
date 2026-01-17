"""Secondary Model for RUL prediction (Remaining Useful Life regression)."""

import pickle
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


class SecondaryModel:
    """
    Secondary Model for RUL prediction.
    
    Predicts the Remaining Useful Life (RUL) of capacitors using
    Random Forest Regressor.
    
    Attributes:
        model_type: Type of model ('random_forest')
        model: Trained regressor model
        feature_names: List of feature names used for training
    """
    
    def __init__(self, model_type: str = "random_forest"):
        """
        Initialize Secondary Model.
        
        Args:
            model_type: Type of model to use (default: 'random_forest')
        """
        self.model_type = model_type
        self.model: Optional[RandomForestRegressor] = None
        self.feature_names: Optional[list] = None
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        **kwargs
    ) -> dict:
        """
        Train the Secondary Model.
        
        Args:
            X_train: Training features
            y_train: Training RUL values
            X_val: Validation features (optional)
            y_val: Validation RUL values (optional)
            **kwargs: Additional hyperparameters for the model
        
        Returns:
            Dictionary containing training metrics
        """
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Default hyperparameters
        default_params = {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            "n_jobs": -1,
        }
        
        # Update with user-provided parameters
        params = {**default_params, **kwargs}
        
        # Initialize and train model
        if self.model_type == "random_forest":
            self.model = RandomForestRegressor(**params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        self.model.fit(X_train, y_train)
        
        # Evaluate on training set
        y_train_pred = self.model.predict(X_train)
        
        train_metrics = {
            "mae": mean_absolute_error(y_train, y_train_pred),
            "rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "r2": r2_score(y_train, y_train_pred),
            "mape": self._calculate_mape(y_train, y_train_pred),
        }
        
        # Evaluate on validation set if provided
        val_metrics = {}
        if X_val is not None and y_val is not None:
            y_val_pred = self.model.predict(X_val)
            
            val_metrics = {
                "mae": mean_absolute_error(y_val, y_val_pred),
                "rmse": np.sqrt(mean_squared_error(y_val, y_val_pred)),
                "r2": r2_score(y_val, y_val_pred),
                "mape": self._calculate_mape(y_val, y_val_pred),
            }
        
        return {"train": train_metrics, "val": val_metrics}
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict RUL values.
        
        Args:
            X: Features to predict
        
        Returns:
            Predicted RUL values
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test RUL values
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        y_pred = self.predict(X_test)
        
        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
            "mape": self._calculate_mape(y_test, y_pred),
        }
        
        return metrics
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Returns:
            DataFrame with features and their importance scores
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        if not hasattr(self.model, "feature_importances_"):
            raise ValueError("Model does not support feature importance")
        
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_
        })
        
        return importance_df.sort_values("importance", ascending=False)
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            "model": self.model,
            "model_type": self.model_type,
            "feature_names": self.feature_names,
        }
        
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
    
    def load(self, path: Union[str, Path]) -> None:
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        with open(path, "rb") as f:
            model_data = pickle.load(f)
        
        self.model = model_data["model"]
        self.model_type = model_data["model_type"]
        self.feature_names = model_data["feature_names"]
    
    @staticmethod
    def _calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate Mean Absolute Percentage Error (MAPE).
        
        Args:
            y_true: True values
            y_pred: Predicted values
        
        Returns:
            MAPE value (percentage)
        """
        # Avoid division by zero
        mask = y_true != 0
        if not mask.any():
            return 0.0
        
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
