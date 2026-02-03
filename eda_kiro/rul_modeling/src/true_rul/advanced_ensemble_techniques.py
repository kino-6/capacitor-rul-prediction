"""
Advanced Ensemble Techniques for RUL Prediction System

This module implements advanced ensemble methods including:
- Stacking ensemble with meta-learner
- Dynamic ensemble weighting based on input characteristics
- Boosting-based ensemble for anomaly detection
- Mixture of experts architecture

Requirements: 1.1, 2.1
"""

import logging
import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
from pathlib import Path

from .rul_regression_model import RULRegressionModel
from .ensemble_anomaly_detector import EnsembleAnomalyDetector

logger = logging.getLogger(__name__)


class StackingEnsembleRULPredictor(BaseEstimator, RegressorMixin):
    """
    Stacking ensemble with meta-learner for RUL prediction
    
    This ensemble uses a two-level approach:
    1. Base models make predictions on the input
    2. Meta-learner combines base model predictions to make final prediction
    
    The meta-learner is trained using cross-validation to avoid overfitting.
    
    Attributes:
        base_models: List of base models for level-1 predictions
        meta_learner: Model that combines base model predictions
        cv_folds: Number of cross-validation folds for meta-learner training
        is_fitted: Whether the ensemble has been trained
        feature_names: Names of input features
    """
    
    def __init__(self,
                 base_models: Optional[List[Any]] = None,
                 meta_learner: Optional[Any] = None,
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize stacking ensemble
        
        Args:
            base_models: List of base models. If None, uses default models
            meta_learner: Meta-learner model. If None, uses LinearRegression
            cv_folds: Number of CV folds for meta-learner training
            random_state: Random seed for reproducibility
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.is_fitted = False
        self.feature_names: Optional[List[str]] = None
        
        # Initialize base models
        if base_models is None:
            self.base_models = [
                RULRegressionModel(model_type="xgboost"),
                RULRegressionModel(model_type="lightgbm"),
                RULRegressionModel(model_type="random_forest"),
                RULRegressionModel(model_type="elastic_net")
            ]
        else:
            self.base_models = base_models
        
        # Initialize meta-learner
        if meta_learner is None:
            self.meta_learner = LinearRegression()
        else:
            self.meta_learner = meta_learner
        
        logger.info(f"Initialized StackingEnsembleRULPredictor with {len(self.base_models)} base models")
    
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            feature_names: Optional[List[str]] = None) -> "StackingEnsembleRULPredictor":
        """
        Fit the stacking ensemble
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            feature_names: Feature names (optional)
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training stacking ensemble on {X.shape[0]} samples with {X.shape[1]} features")
        
        self.feature_names = feature_names
        
        # Step 1: Train base models and get cross-validation predictions
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        meta_features = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, base_model in enumerate(self.base_models):
            logger.info(f"Training base model {i+1}/{len(self.base_models)}: {type(base_model).__name__}")
            
            # Manual cross-validation for custom models
            cv_predictions = np.zeros(X.shape[0])
            
            for train_idx, val_idx in cv.split(X):
                X_fold_train, X_fold_val = X[train_idx], X[val_idx]
                y_fold_train, y_fold_val = y[train_idx], y[val_idx]
                
                # Create a copy of the base model for this fold
                if hasattr(base_model, 'model_type'):
                    fold_model = RULRegressionModel(model_type=base_model.model_type)
                else:
                    # For sklearn models, create a copy
                    from sklearn.base import clone
                    fold_model = clone(base_model)
                
                # Train on fold
                if hasattr(fold_model, 'train'):
                    fold_model.train(X_fold_train, y_fold_train, feature_names=feature_names)
                else:
                    fold_model.fit(X_fold_train, y_fold_train)
                
                # Predict on validation fold
                cv_predictions[val_idx] = fold_model.predict(X_fold_val)
            
            meta_features[:, i] = cv_predictions
            
            # Train base model on full training set
            if hasattr(base_model, 'train'):
                base_model.train(X, y, X_val, y_val, feature_names)
            else:
                base_model.fit(X, y)
        
        # Step 2: Train meta-learner on base model predictions
        logger.info("Training meta-learner...")
        self.meta_learner.fit(meta_features, y)
        
        self.is_fitted = True
        logger.info("Stacking ensemble training completed")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the stacking ensemble
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Get base model predictions
        base_predictions = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, base_model in enumerate(self.base_models):
            if hasattr(base_model, 'predict'):
                base_predictions[:, i] = base_model.predict(X)
            else:
                base_predictions[:, i] = base_model.predict(X)
        
        # Meta-learner makes final prediction
        final_predictions = self.meta_learner.predict(base_predictions)
        
        return final_predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from base models
        
        Returns:
            Aggregated feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before getting feature importance")
        
        importance_scores = {}
        total_weight = 0
        
        for i, base_model in enumerate(self.base_models):
            if hasattr(base_model, 'get_feature_importance'):
                model_importance = base_model.get_feature_importance()
                weight = 1.0 / len(self.base_models)  # Equal weighting
                
                for feature, score in model_importance.items():
                    importance_scores[feature] = importance_scores.get(feature, 0) + score * weight
                
                total_weight += weight
        
        # Normalize scores
        if total_weight > 0:
            for feature in importance_scores:
                importance_scores[feature] /= total_weight
        
        return importance_scores
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the stacking ensemble"""
        return {
            "ensemble_type": "stacking",
            "n_base_models": len(self.base_models),
            "base_model_types": [type(model).__name__ for model in self.base_models],
            "meta_learner_type": type(self.meta_learner).__name__,
            "cv_folds": self.cv_folds,
            "is_fitted": self.is_fitted
        }


class DynamicEnsembleWeighting(BaseEstimator, RegressorMixin):
    """
    Dynamic ensemble weighting based on input characteristics
    
    This ensemble adapts the weights of base models based on the characteristics
    of the input data. Different models may perform better on different types
    of inputs, and this ensemble learns to weight them accordingly.
    
    Attributes:
        base_models: List of base models
        weighting_model: Model that predicts optimal weights for each input
        input_clusterer: Clustering model to identify input characteristics
        n_clusters: Number of clusters for input characterization
        cluster_weights: Learned weights for each cluster
        is_fitted: Whether the ensemble has been trained
    """
    
    def __init__(self,
                 base_models: Optional[List[Any]] = None,
                 n_clusters: int = 5,
                 weighting_strategy: str = "cluster_based",
                 random_state: int = 42):
        """
        Initialize dynamic ensemble weighting
        
        Args:
            base_models: List of base models
            n_clusters: Number of clusters for input characterization
            weighting_strategy: Strategy for dynamic weighting ("cluster_based" or "regression_based")
            random_state: Random seed
        """
        self.n_clusters = n_clusters
        self.weighting_strategy = weighting_strategy
        self.random_state = random_state
        self.is_fitted = False
        
        # Initialize base models
        if base_models is None:
            self.base_models = [
                RULRegressionModel(model_type="xgboost"),
                RULRegressionModel(model_type="lightgbm"),
                RULRegressionModel(model_type="random_forest")
            ]
        else:
            self.base_models = base_models
        
        # Initialize components for dynamic weighting
        self.input_clusterer = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.cluster_weights = {}
        self.weighting_model = None
        self.scaler = StandardScaler()
        
        logger.info(f"Initialized DynamicEnsembleWeighting with {len(self.base_models)} base models")
    
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            feature_names: Optional[List[str]] = None) -> "DynamicEnsembleWeighting":
        """
        Fit the dynamic ensemble
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: Feature names
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training dynamic ensemble on {X.shape[0]} samples")
        
        # Step 1: Train base models
        base_predictions = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, base_model in enumerate(self.base_models):
            logger.info(f"Training base model {i+1}/{len(self.base_models)}")
            
            if hasattr(base_model, 'train'):
                base_model.train(X, y, X_val, y_val, feature_names)
            else:
                base_model.fit(X, y)
            
            base_predictions[:, i] = base_model.predict(X)
        
        # Step 2: Learn input characteristics
        X_scaled = self.scaler.fit_transform(X)
        cluster_labels = self.input_clusterer.fit_predict(X_scaled)
        
        # Step 3: Learn optimal weights for each cluster/input type
        if self.weighting_strategy == "cluster_based":
            self._learn_cluster_weights(cluster_labels, base_predictions, y)
        elif self.weighting_strategy == "regression_based":
            self._learn_regression_weights(X_scaled, base_predictions, y)
        
        self.is_fitted = True
        logger.info("Dynamic ensemble training completed")
        
        return self
    
    def _learn_cluster_weights(self,
                              cluster_labels: np.ndarray,
                              base_predictions: np.ndarray,
                              y: np.ndarray) -> None:
        """
        Learn optimal weights for each cluster
        
        Args:
            cluster_labels: Cluster assignments for each sample
            base_predictions: Predictions from base models
            y: True labels
        """
        for cluster_id in range(self.n_clusters):
            cluster_mask = cluster_labels == cluster_id
            
            if np.sum(cluster_mask) < 2:  # Skip clusters with too few samples
                # Use equal weights as fallback
                self.cluster_weights[cluster_id] = np.ones(len(self.base_models)) / len(self.base_models)
                continue
            
            cluster_predictions = base_predictions[cluster_mask]
            cluster_y = y[cluster_mask]
            
            # Find optimal weights for this cluster using least squares
            try:
                # Solve for weights that minimize prediction error
                weights, _, _, _ = np.linalg.lstsq(cluster_predictions, cluster_y, rcond=None)
                
                # Ensure weights are non-negative and sum to 1
                weights = np.maximum(weights, 0)
                if np.sum(weights) > 0:
                    weights = weights / np.sum(weights)
                else:
                    weights = np.ones(len(self.base_models)) / len(self.base_models)
                
                self.cluster_weights[cluster_id] = weights
                
            except np.linalg.LinAlgError:
                # Fallback to equal weights
                self.cluster_weights[cluster_id] = np.ones(len(self.base_models)) / len(self.base_models)
    
    def _learn_regression_weights(self,
                                 X_scaled: np.ndarray,
                                 base_predictions: np.ndarray,
                                 y: np.ndarray) -> None:
        """
        Learn regression model to predict optimal weights
        
        Args:
            X_scaled: Scaled input features
            base_predictions: Predictions from base models
            y: True labels
        """
        # Create training data for weighting model
        # Features: input characteristics + base model predictions
        weighting_features = np.hstack([X_scaled, base_predictions])
        
        # Target: optimal weights (computed using local optimization)
        optimal_weights = np.zeros((X_scaled.shape[0], len(self.base_models)))
        
        for i in range(X_scaled.shape[0]):
            # For each sample, find weights that would minimize error
            sample_predictions = base_predictions[i]
            sample_y = y[i]
            
            # Simple approach: weight inversely proportional to prediction error
            errors = np.abs(sample_predictions - sample_y)
            if np.sum(errors) > 0:
                weights = 1.0 / (errors + 1e-8)  # Add small epsilon to avoid division by zero
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(len(self.base_models)) / len(self.base_models)
            
            optimal_weights[i] = weights
        
        # Train regression model to predict weights
        self.weighting_model = RandomForestRegressor(
            n_estimators=100,
            random_state=self.random_state
        )
        
        # Train separate model for each weight component
        self.weight_models = []
        for j in range(len(self.base_models)):
            model = RandomForestRegressor(n_estimators=50, random_state=self.random_state)
            model.fit(weighting_features, optimal_weights[:, j])
            self.weight_models.append(model)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using dynamic weighting
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Get base model predictions
        base_predictions = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, base_model in enumerate(self.base_models):
            base_predictions[:, i] = base_model.predict(X)
        
        # Get dynamic weights
        if self.weighting_strategy == "cluster_based":
            weights = self._get_cluster_weights(X)
        elif self.weighting_strategy == "regression_based":
            weights = self._get_regression_weights(X, base_predictions)
        else:
            # Fallback to equal weights
            weights = np.ones((X.shape[0], len(self.base_models))) / len(self.base_models)
        
        # Compute weighted predictions
        final_predictions = np.sum(base_predictions * weights, axis=1)
        
        return final_predictions
    
    def _get_cluster_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Get weights based on cluster assignments
        
        Args:
            X: Input features
            
        Returns:
            Weight matrix (n_samples, n_models)
        """
        X_scaled = self.scaler.transform(X)
        cluster_labels = self.input_clusterer.predict(X_scaled)
        
        weights = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, cluster_id in enumerate(cluster_labels):
            if cluster_id in self.cluster_weights:
                weights[i] = self.cluster_weights[cluster_id]
            else:
                # Fallback to equal weights
                weights[i] = np.ones(len(self.base_models)) / len(self.base_models)
        
        return weights
    
    def _get_regression_weights(self, X: np.ndarray, base_predictions: np.ndarray) -> np.ndarray:
        """
        Get weights using regression model
        
        Args:
            X: Input features
            base_predictions: Base model predictions
            
        Returns:
            Weight matrix (n_samples, n_models)
        """
        X_scaled = self.scaler.transform(X)
        weighting_features = np.hstack([X_scaled, base_predictions])
        
        weights = np.zeros((X.shape[0], len(self.base_models)))
        
        for j, model in enumerate(self.weight_models):
            weights[:, j] = model.predict(weighting_features)
        
        # Ensure weights are non-negative and sum to 1
        weights = np.maximum(weights, 0)
        row_sums = np.sum(weights, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        weights = weights / row_sums
        
        return weights
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the dynamic ensemble"""
        return {
            "ensemble_type": "dynamic_weighting",
            "n_base_models": len(self.base_models),
            "base_model_types": [type(model).__name__ for model in self.base_models],
            "weighting_strategy": self.weighting_strategy,
            "n_clusters": self.n_clusters,
            "is_fitted": self.is_fitted
        }


class BoostingAnomalyDetector(BaseEstimator, ClassifierMixin):
    """
    Boosting-based ensemble for anomaly detection
    
    This detector uses gradient boosting to sequentially improve
    anomaly detection performance by focusing on previously
    misclassified samples.
    
    Attributes:
        base_detectors: List of base anomaly detectors
        boosting_weights: Weights for each base detector
        detector_errors: Training errors for each detector
        n_estimators: Number of boosting rounds
        learning_rate: Learning rate for boosting
        is_fitted: Whether the ensemble has been trained
    """
    
    def __init__(self,
                 base_detectors: Optional[List[Any]] = None,
                 n_estimators: int = 10,
                 learning_rate: float = 1.0,
                 random_state: int = 42):
        """
        Initialize boosting anomaly detector
        
        Args:
            base_detectors: List of base anomaly detectors
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate for boosting
            random_state: Random seed
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.is_fitted = False
        
        # Initialize base detectors
        if base_detectors is None:
            # Use simple gradient boosting classifiers as base detectors
            self.base_detectors = [
                GradientBoostingClassifier(
                    n_estimators=50,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=random_state + i
                )
                for i in range(n_estimators)
            ]
        else:
            self.base_detectors = base_detectors
        
        self.boosting_weights = []
        self.detector_errors = []
        
        logger.info(f"Initialized BoostingAnomalyDetector with {len(self.base_detectors)} base detectors")
    
    def fit(self,
            X_normal: np.ndarray,
            X_anomaly: Optional[np.ndarray] = None,
            feature_names: Optional[List[str]] = None) -> "BoostingAnomalyDetector":
        """
        Fit the boosting anomaly detector
        
        Args:
            X_normal: Normal training samples
            X_anomaly: Anomalous training samples (optional)
            feature_names: Feature names (optional)
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training boosting anomaly detector on {X_normal.shape[0]} normal samples")
        
        # Create training data
        if X_anomaly is not None:
            X_train = np.vstack([X_normal, X_anomaly])
            y_train = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_anomaly))])
        else:
            # Semi-supervised: only normal data available
            # Create synthetic anomalies by adding noise
            np.random.seed(self.random_state)
            X_synthetic_anomaly = X_normal + np.random.normal(0, 0.5, X_normal.shape)
            
            X_train = np.vstack([X_normal, X_synthetic_anomaly])
            y_train = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_synthetic_anomaly))])
        
        # Initialize sample weights
        sample_weights = np.ones(len(X_train)) / len(X_train)
        
        # Boosting iterations
        for i in range(min(self.n_estimators, len(self.base_detectors))):
            logger.info(f"Boosting iteration {i+1}/{self.n_estimators}")
            
            # Train base detector with current sample weights
            detector = self.base_detectors[i]
            
            # Fit detector (some detectors may not support sample weights)
            try:
                if hasattr(detector, 'fit') and 'sample_weight' in detector.fit.__code__.co_varnames:
                    detector.fit(X_train, y_train, sample_weight=sample_weights)
                else:
                    detector.fit(X_train, y_train)
            except Exception as e:
                logger.warning(f"Failed to train detector {i}: {e}")
                continue
            
            # Get predictions
            try:
                y_pred = detector.predict(X_train)
            except:
                # Some detectors might not have predict method
                if hasattr(detector, 'decision_function'):
                    scores = detector.decision_function(X_train)
                    y_pred = (scores > 0).astype(int)
                else:
                    logger.warning(f"Detector {i} doesn't support prediction")
                    continue
            
            # Calculate error
            error = np.average(y_pred != y_train, weights=sample_weights)
            
            # Avoid division by zero
            if error >= 0.5:
                error = 0.5 - 1e-10
            
            self.detector_errors.append(error)
            
            # Calculate detector weight
            detector_weight = self.learning_rate * np.log((1 - error) / error)
            self.boosting_weights.append(detector_weight)
            
            # Update sample weights
            sample_weights *= np.exp(detector_weight * (y_pred != y_train))
            sample_weights /= np.sum(sample_weights)  # Normalize
        
        self.is_fitted = True
        logger.info(f"Boosting training completed with {len(self.boosting_weights)} detectors")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make anomaly predictions using boosting ensemble
        
        Args:
            X: Input features
            
        Returns:
            Binary anomaly predictions (0=normal, 1=anomaly)
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before making predictions")
        
        # Get weighted predictions from all detectors
        ensemble_scores = np.zeros(X.shape[0])
        
        for i, (detector, weight) in enumerate(zip(self.base_detectors[:len(self.boosting_weights)], 
                                                  self.boosting_weights)):
            try:
                if hasattr(detector, 'predict'):
                    pred = detector.predict(X)
                elif hasattr(detector, 'decision_function'):
                    scores = detector.decision_function(X)
                    pred = (scores > 0).astype(int)
                else:
                    continue
                
                # Convert to {-1, 1} for boosting
                pred_boosting = 2 * pred - 1
                ensemble_scores += weight * pred_boosting
                
            except Exception as e:
                logger.warning(f"Detector {i} failed during prediction: {e}")
                continue
        
        # Convert back to {0, 1}
        final_predictions = (ensemble_scores > 0).astype(int)
        
        return final_predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities
        
        Args:
            X: Input features
            
        Returns:
            Class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before making predictions")
        
        # Get weighted scores
        ensemble_scores = np.zeros(X.shape[0])
        
        for detector, weight in zip(self.base_detectors[:len(self.boosting_weights)], 
                                   self.boosting_weights):
            try:
                if hasattr(detector, 'predict_proba'):
                    proba = detector.predict_proba(X)[:, 1]  # Probability of anomaly
                elif hasattr(detector, 'decision_function'):
                    scores = detector.decision_function(X)
                    # Convert to probabilities using sigmoid
                    proba = 1 / (1 + np.exp(-scores))
                else:
                    pred = detector.predict(X)
                    proba = pred.astype(float)
                
                ensemble_scores += weight * proba
                
            except Exception as e:
                logger.warning(f"Detector failed during probability prediction: {e}")
                continue
        
        # Normalize scores to probabilities
        if len(self.boosting_weights) > 0:
            ensemble_scores /= np.sum(self.boosting_weights)
        
        # Ensure probabilities are in [0, 1]
        ensemble_scores = np.clip(ensemble_scores, 0, 1)
        
        # Return probabilities for both classes
        proba_normal = 1 - ensemble_scores
        proba_anomaly = ensemble_scores
        
        return np.column_stack([proba_normal, proba_anomaly])
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the boosting detector"""
        return {
            "detector_type": "boosting_anomaly",
            "n_base_detectors": len(self.base_detectors),
            "n_trained_detectors": len(self.boosting_weights),
            "learning_rate": self.learning_rate,
            "detector_weights": self.boosting_weights,
            "detector_errors": self.detector_errors,
            "is_fitted": self.is_fitted
        }


class MixtureOfExpertsRUL(BaseEstimator, RegressorMixin):
    """
    Mixture of Experts architecture for RUL prediction
    
    This architecture uses a gating network to determine which expert
    (specialized model) should handle each input. Different experts
    can specialize in different types of degradation patterns.
    
    Attributes:
        experts: List of expert models
        gating_network: Network that determines expert weights
        n_experts: Number of expert models
        expert_specialization: Strategy for expert specialization
        is_fitted: Whether the model has been trained
    """
    
    def __init__(self,
                 experts: Optional[List[Any]] = None,
                 n_experts: int = 3,
                 expert_specialization: str = "data_driven",
                 gating_network: Optional[Any] = None,
                 random_state: int = 42):
        """
        Initialize mixture of experts
        
        Args:
            experts: List of expert models
            n_experts: Number of experts (if experts not provided)
            expert_specialization: How experts specialize ("data_driven" or "manual")
            gating_network: Gating network model
            random_state: Random seed
        """
        self.n_experts = n_experts
        self.expert_specialization = expert_specialization
        self.random_state = random_state
        self.is_fitted = False
        
        # Initialize experts
        if experts is None:
            # Create diverse experts with different characteristics
            self.experts = [
                RULRegressionModel(model_type="xgboost"),  # Good for non-linear patterns
                RULRegressionModel(model_type="random_forest"),  # Good for feature interactions
                RULRegressionModel(model_type="elastic_net")  # Good for linear relationships
            ][:n_experts]
        else:
            self.experts = experts
            self.n_experts = len(experts)
        
        # Initialize gating network
        if gating_network is None:
            # Use softmax regression for gating
            self.gating_network = LogisticRegression(
                solver='lbfgs',
                random_state=random_state,
                max_iter=1000
            )
        else:
            self.gating_network = gating_network
        
        self.expert_assignments = None
        
        logger.info(f"Initialized MixtureOfExpertsRUL with {self.n_experts} experts")
    
    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            X_val: Optional[np.ndarray] = None,
            y_val: Optional[np.ndarray] = None,
            feature_names: Optional[List[str]] = None) -> "MixtureOfExpertsRUL":
        """
        Fit the mixture of experts model
        
        Args:
            X: Training features
            y: Training labels
            X_val: Validation features
            y_val: Validation labels
            feature_names: Feature names
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Training mixture of experts on {X.shape[0]} samples")
        
        # Step 1: Determine expert specialization
        if self.expert_specialization == "data_driven":
            expert_assignments = self._assign_experts_data_driven(X, y)
        else:
            # Manual assignment based on data characteristics
            expert_assignments = self._assign_experts_manual(X, y)
        
        self.expert_assignments = expert_assignments
        
        # Step 2: Train experts on their specialized data
        for i, expert in enumerate(self.experts):
            expert_mask = expert_assignments == i
            
            if np.sum(expert_mask) < 2:  # Skip experts with too few samples
                logger.warning(f"Expert {i} has too few samples ({np.sum(expert_mask)}), using all data")
                expert_mask = np.ones(len(X), dtype=bool)
            
            X_expert = X[expert_mask]
            y_expert = y[expert_mask]
            
            logger.info(f"Training expert {i+1}/{self.n_experts} on {len(X_expert)} samples")
            
            if hasattr(expert, 'train'):
                expert.train(X_expert, y_expert, X_val, y_val, feature_names)
            else:
                expert.fit(X_expert, y_expert)
        
        # Step 3: Train gating network
        logger.info("Training gating network...")
        self.gating_network.fit(X, expert_assignments)
        
        self.is_fitted = True
        logger.info("Mixture of experts training completed")
        
        return self
    
    def _assign_experts_data_driven(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Assign samples to experts using data-driven clustering
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Expert assignments for each sample
        """
        # Use K-means clustering on combined feature-target space
        # This helps experts specialize on different regions of the input-output space
        
        # Normalize features and targets
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()
        
        # Combine features and targets for clustering
        combined_data = np.column_stack([X_scaled, y_scaled])
        
        # Cluster into expert groups
        clusterer = KMeans(n_clusters=self.n_experts, random_state=self.random_state)
        expert_assignments = clusterer.fit_predict(combined_data)
        
        return expert_assignments
    
    def _assign_experts_manual(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Assign samples to experts using manual rules
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Expert assignments for each sample
        """
        # Simple manual assignment based on target value ranges
        expert_assignments = np.zeros(len(y), dtype=int)
        
        # Sort by target values and assign to experts
        y_percentiles = np.percentile(y, np.linspace(0, 100, self.n_experts + 1))
        
        for i in range(self.n_experts):
            if i == self.n_experts - 1:
                # Last expert gets remaining samples
                mask = y >= y_percentiles[i]
            else:
                mask = (y >= y_percentiles[i]) & (y < y_percentiles[i + 1])
            
            expert_assignments[mask] = i
        
        return expert_assignments
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using mixture of experts
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get expert predictions
        expert_predictions = np.zeros((X.shape[0], self.n_experts))
        
        for i, expert in enumerate(self.experts):
            expert_predictions[:, i] = expert.predict(X)
        
        # Get gating weights
        gating_weights = self.gating_network.predict_proba(X)
        
        # Weighted combination of expert predictions
        final_predictions = np.sum(expert_predictions * gating_weights, axis=1)
        
        return final_predictions
    
    def get_expert_weights(self, X: np.ndarray) -> np.ndarray:
        """
        Get gating weights for each expert
        
        Args:
            X: Input features
            
        Returns:
            Expert weights for each sample
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting expert weights")
        
        return self.gating_network.predict_proba(X)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the mixture of experts"""
        return {
            "model_type": "mixture_of_experts",
            "n_experts": self.n_experts,
            "expert_types": [type(expert).__name__ for expert in self.experts],
            "expert_specialization": self.expert_specialization,
            "gating_network_type": type(self.gating_network).__name__,
            "is_fitted": self.is_fitted
        }