"""
Automated Hyperparameter Optimization for RUL Prediction System

This module implements Optuna-based hyperparameter optimization for all models
in the RUL prediction system, including multi-objective optimization for
balancing FPR and accuracy, Bayesian optimization for ensemble weights,
and automated model selection pipeline.

Requirements: 1.1, 2.1, 5.2
"""

import logging
import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from typing import Dict, Any, List, Tuple, Optional, Callable, Union
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import joblib
import json
from pathlib import Path
import time

from .rul_regression_model import RULRegressionModel
from .ensemble_anomaly_detector import EnsembleAnomalyDetector
from .gradient_boosting_predictor import GradientBoostingRULPredictor
from .random_forest_predictor import RandomForestRULPredictor
from .elastic_net_predictor import ElasticNetRULPredictor
from .hybrid_ensemble_predictor import HybridEnsembleRULPredictor

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """
    Automated hyperparameter optimization using Optuna
    
    This class provides comprehensive hyperparameter optimization for all models
    in the RUL prediction system, including:
    - Single-objective optimization for individual models
    - Multi-objective optimization balancing FPR vs accuracy
    - Bayesian optimization for ensemble weights
    - Automated model selection pipeline
    
    Attributes:
        study_name: Name of the optimization study
        storage_url: URL for study storage (optional)
        n_trials: Number of optimization trials
        timeout: Maximum optimization time in seconds
        sampler: Optuna sampler for hyperparameter selection
        pruner: Optuna pruner for early stopping
        best_params: Best parameters found during optimization
        best_score: Best score achieved during optimization
        optimization_history: History of all trials
    """
    
    def __init__(self,
                 study_name: str = "rul_hyperopt",
                 storage_url: Optional[str] = None,
                 n_trials: int = 100,
                 timeout: Optional[int] = 3600,
                 random_state: int = 42):
        """
        Initialize hyperparameter optimizer
        
        Args:
            study_name: Name for the optimization study
            storage_url: Optional URL for persistent study storage
            n_trials: Number of optimization trials to run
            timeout: Maximum optimization time in seconds
            random_state: Random seed for reproducibility
        """
        self.study_name = study_name
        self.storage_url = storage_url
        self.n_trials = n_trials
        self.timeout = timeout
        self.random_state = random_state
        
        # Initialize Optuna components
        self.sampler = TPESampler(seed=random_state)
        self.pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        
        # Results storage
        self.best_params: Dict[str, Any] = {}
        self.best_score: float = float('inf')
        self.optimization_history: List[Dict[str, Any]] = []
        
        logger.info(f"Initialized HyperparameterOptimizer: {study_name}")
    
    def optimize_rul_model(self,
                          X_train: np.ndarray,
                          y_train: np.ndarray,
                          X_val: np.ndarray,
                          y_val: np.ndarray,
                          model_type: str = "xgboost",
                          cv_folds: int = 3,
                          scoring: str = "rmse") -> Dict[str, Any]:
        """
        Optimize hyperparameters for RUL regression model
        
        Args:
            X_train: Training features
            y_train: Training RUL labels
            X_val: Validation features
            y_val: Validation RUL labels
            model_type: Type of model to optimize
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric ("rmse", "mae", "r2")
            
        Returns:
            Dictionary with best parameters and optimization results
        """
        logger.info(f"Starting hyperparameter optimization for {model_type} RUL model")
        
        def objective(trial):
            # Get hyperparameters based on model type
            params = self._suggest_rul_params(trial, model_type)
            
            try:
                # Create and train model
                model = RULRegressionModel(model_type=model_type, **params)
                model.train(X_train, y_train, X_val, y_val)
                
                # Evaluate on validation set
                y_pred = model.predict(X_val)
                
                if scoring == "rmse":
                    score = np.sqrt(mean_squared_error(y_val, y_pred))
                elif scoring == "mae":
                    score = mean_absolute_error(y_val, y_pred)
                elif scoring == "r2":
                    score = -r2_score(y_val, y_pred)  # Negative for minimization
                else:
                    raise ValueError(f"Unknown scoring metric: {scoring}")
                
                # Report intermediate value for pruning
                trial.report(score, step=0)
                
                # Check if trial should be pruned
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                return score
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float('inf')
        
        # Create and run study
        study = optuna.create_study(
            study_name=f"{self.study_name}_{model_type}_rul",
            storage=self.storage_url,
            sampler=self.sampler,
            pruner=self.pruner,
            direction="minimize",
            load_if_exists=True
        )
        
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        # Store results
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        results = {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "n_trials": len(study.trials),
            "study": study
        }
        
        logger.info(f"Optimization completed. Best {scoring}: {study.best_value:.4f}")
        return results
    
    def optimize_anomaly_detector(self,
                                 X_normal: np.ndarray,
                                 X_val: np.ndarray,
                                 y_val: np.ndarray,
                                 target_fpr: float = 0.05) -> Dict[str, Any]:
        """
        Optimize hyperparameters for anomaly detection ensemble
        
        Args:
            X_normal: Normal training data
            X_val: Validation features
            y_val: Validation labels (0=normal, 1=anomaly)
            target_fpr: Target false positive rate
            
        Returns:
            Dictionary with best parameters and optimization results
        """
        logger.info("Starting hyperparameter optimization for anomaly detector")
        
        def objective(trial):
            # Suggest ensemble weights
            weight_if = trial.suggest_float("weight_isolation_forest", 0.1, 0.6)
            weight_ae = trial.suggest_float("weight_autoencoder", 0.1, 0.6)
            weight_ocsvm = 1.0 - weight_if - weight_ae
            
            if weight_ocsvm < 0.1:
                # Ensure minimum weight for OCSVM
                return float('inf')
            
            weights = [weight_if, weight_ae, weight_ocsvm]
            
            # Suggest individual detector parameters
            if_params = {
                "contamination": trial.suggest_float("if_contamination", 0.01, 0.1)
            }
            
            ae_params = {
                "encoding_dim": trial.suggest_int("ae_encoding_dim", 8, 32),
                "hidden_layers": trial.suggest_int("ae_hidden_layers", 1, 3),
                "dropout_rate": trial.suggest_float("ae_dropout_rate", 0.0, 0.3)
            }
            
            ocsvm_params = {
                "nu": trial.suggest_float("ocsvm_nu", 0.01, 0.1),
                "gamma": trial.suggest_categorical("ocsvm_gamma", ["scale", "auto"]),
                "kernel": trial.suggest_categorical("ocsvm_kernel", ["rbf", "poly", "sigmoid"])
            }
            
            try:
                # Create and train ensemble
                detector = EnsembleAnomalyDetector(
                    weights=weights,
                    isolation_forest_params=if_params,
                    autoencoder_params=ae_params,
                    ocsvm_params=ocsvm_params
                )
                
                detector.fit(X_normal, validation_data=X_val, 
                           validation_labels=y_val, target_fpr=target_fpr)
                
                # Evaluate on validation set
                y_pred, scores, info = detector.predict(X_val)
                
                # Calculate metrics
                fpr = np.mean(y_pred[y_val == 0])  # False positive rate
                tpr = np.mean(y_pred[y_val == 1])  # True positive rate (recall)
                
                # Multi-objective: minimize FPR deviation and maximize TPR
                fpr_penalty = abs(fpr - target_fpr) * 10  # Heavy penalty for FPR deviation
                tpr_reward = -tpr  # Negative for maximization
                
                score = fpr_penalty + tpr_reward
                
                # Report intermediate values
                trial.report(score, step=0)
                
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                return score
                
            except Exception as e:
                logger.warning(f"Trial failed: {e}")
                return float('inf')
        
        # Create and run study
        study = optuna.create_study(
            study_name=f"{self.study_name}_anomaly_detector",
            storage=self.storage_url,
            sampler=self.sampler,
            pruner=self.pruner,
            direction="minimize",
            load_if_exists=True
        )
        
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        results = {
            "best_params": study.best_params,
            "best_score": study.best_value,
            "n_trials": len(study.trials),
            "study": study
        }
        
        logger.info(f"Anomaly detector optimization completed. Best score: {study.best_value:.4f}")
        return results
    
    def multi_objective_optimization(self,
                                   X_train: np.ndarray,
                                   y_train: np.ndarray,
                                   X_val: np.ndarray,
                                   y_val: np.ndarray,
                                   X_normal: np.ndarray,
                                   y_anomaly: np.ndarray,
                                   model_type: str = "ensemble") -> Dict[str, Any]:
        """
        Multi-objective optimization balancing RUL accuracy and anomaly detection FPR
        
        Args:
            X_train: RUL training features
            y_train: RUL training labels
            X_val: RUL validation features
            y_val: RUL validation labels
            X_normal: Normal data for anomaly detection
            y_anomaly: Anomaly labels for validation
            model_type: Type of RUL model to optimize
            
        Returns:
            Dictionary with Pareto-optimal solutions
        """
        logger.info("Starting multi-objective optimization (RUL accuracy vs FPR)")
        
        def objective(trial):
            # Suggest RUL model parameters
            rul_params = self._suggest_rul_params(trial, model_type)
            
            # Suggest anomaly detector parameters
            ad_weights = [
                trial.suggest_float("ad_weight_if", 0.1, 0.6),
                trial.suggest_float("ad_weight_ae", 0.1, 0.6),
            ]
            ad_weights.append(1.0 - sum(ad_weights))
            
            if ad_weights[2] < 0.1:
                return float('inf'), float('inf')
            
            try:
                # Train RUL model
                rul_model = RULRegressionModel(model_type=model_type, **rul_params)
                rul_model.train(X_train, y_train, X_val, y_val)
                
                # Evaluate RUL performance
                y_rul_pred = rul_model.predict(X_val)
                rul_rmse = np.sqrt(mean_squared_error(y_val, y_rul_pred))
                
                # Train anomaly detector
                ad_model = EnsembleAnomalyDetector(weights=ad_weights)
                ad_model.fit(X_normal)
                
                # Evaluate anomaly detection performance
                y_ad_pred, _, _ = ad_model.predict(X_val)
                fpr = np.mean(y_ad_pred[y_anomaly == 0])
                
                return rul_rmse, fpr
                
            except Exception as e:
                logger.warning(f"Multi-objective trial failed: {e}")
                return float('inf'), float('inf')
        
        # Create multi-objective study
        study = optuna.create_study(
            study_name=f"{self.study_name}_multi_objective",
            storage=self.storage_url,
            sampler=self.sampler,
            directions=["minimize", "minimize"],  # Minimize both RMSE and FPR
            load_if_exists=True
        )
        
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        # Extract Pareto-optimal solutions
        pareto_solutions = []
        for trial in study.best_trials:
            pareto_solutions.append({
                "params": trial.params,
                "rul_rmse": trial.values[0],
                "fpr": trial.values[1]
            })
        
        results = {
            "pareto_solutions": pareto_solutions,
            "n_trials": len(study.trials),
            "study": study
        }
        
        logger.info(f"Multi-objective optimization completed. Found {len(pareto_solutions)} Pareto-optimal solutions")
        return results
    
    def optimize_ensemble_weights(self,
                                 models: List[Any],
                                 X_val: np.ndarray,
                                 y_val: np.ndarray,
                                 scoring: str = "rmse") -> Dict[str, Any]:
        """
        Bayesian optimization for ensemble weights
        
        Args:
            models: List of trained models to ensemble
            X_val: Validation features
            y_val: Validation labels
            scoring: Scoring metric for optimization
            
        Returns:
            Dictionary with optimal weights and results
        """
        logger.info(f"Optimizing ensemble weights for {len(models)} models")
        
        def objective(trial):
            # Suggest weights for each model (except last one)
            weights = []
            remaining_weight = 1.0
            
            for i in range(len(models) - 1):
                weight = trial.suggest_float(f"weight_{i}", 0.0, remaining_weight)
                weights.append(weight)
                remaining_weight -= weight
            
            weights.append(remaining_weight)  # Last weight is remainder
            
            # Ensure all weights are non-negative
            if any(w < 0 for w in weights):
                return float('inf')
            
            try:
                # Compute ensemble predictions
                predictions = []
                for model in models:
                    if hasattr(model, 'predict'):
                        pred = model.predict(X_val)
                    else:
                        pred = model(X_val)  # For callable models
                    predictions.append(pred)
                
                # Weighted ensemble prediction
                ensemble_pred = np.zeros_like(predictions[0])
                for pred, weight in zip(predictions, weights):
                    ensemble_pred += weight * pred
                
                # Calculate score
                if scoring == "rmse":
                    score = np.sqrt(mean_squared_error(y_val, ensemble_pred))
                elif scoring == "mae":
                    score = mean_absolute_error(y_val, ensemble_pred)
                elif scoring == "r2":
                    score = -r2_score(y_val, ensemble_pred)
                else:
                    raise ValueError(f"Unknown scoring metric: {scoring}")
                
                return score
                
            except Exception as e:
                logger.warning(f"Ensemble weight optimization trial failed: {e}")
                return float('inf')
        
        # Create study for weight optimization
        study = optuna.create_study(
            study_name=f"{self.study_name}_ensemble_weights",
            storage=self.storage_url,
            sampler=self.sampler,
            direction="minimize",
            load_if_exists=True
        )
        
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)
        
        # Extract optimal weights
        optimal_weights = []
        remaining_weight = 1.0
        
        for i in range(len(models) - 1):
            weight = study.best_params[f"weight_{i}"]
            optimal_weights.append(weight)
            remaining_weight -= weight
        
        optimal_weights.append(remaining_weight)
        
        results = {
            "optimal_weights": optimal_weights,
            "best_score": study.best_value,
            "n_trials": len(study.trials),
            "study": study
        }
        
        logger.info(f"Ensemble weight optimization completed. Best {scoring}: {study.best_value:.4f}")
        return results
    
    def automated_model_selection(self,
                                 X_train: np.ndarray,
                                 y_train: np.ndarray,
                                 X_val: np.ndarray,
                                 y_val: np.ndarray,
                                 model_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Automated model selection pipeline with hyperparameter optimization
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            model_types: List of model types to consider
            
        Returns:
            Dictionary with best model and comprehensive results
        """
        if model_types is None:
            model_types = ["xgboost", "lightgbm", "random_forest", "elastic_net", "ensemble"]
        
        logger.info(f"Starting automated model selection for {len(model_types)} model types")
        
        results = {}
        best_model_type = None
        best_score = float('inf')
        
        for model_type in model_types:
            logger.info(f"Optimizing {model_type}...")
            
            try:
                # Optimize hyperparameters for this model type
                model_results = self.optimize_rul_model(
                    X_train, y_train, X_val, y_val,
                    model_type=model_type,
                    scoring="rmse"
                )
                
                results[model_type] = model_results
                
                # Track best model
                if model_results["best_score"] < best_score:
                    best_score = model_results["best_score"]
                    best_model_type = model_type
                
            except Exception as e:
                logger.error(f"Failed to optimize {model_type}: {e}")
                results[model_type] = {"error": str(e)}
        
        # Train best model with optimal parameters
        if best_model_type:
            best_params = results[best_model_type]["best_params"]
            best_model = RULRegressionModel(model_type=best_model_type, **best_params)
            best_model.train(X_train, y_train, X_val, y_val)
        else:
            best_model = None
        
        final_results = {
            "best_model_type": best_model_type,
            "best_model": best_model,
            "best_score": best_score,
            "all_results": results,
            "model_ranking": sorted(
                [(k, v.get("best_score", float('inf'))) for k, v in results.items()],
                key=lambda x: x[1]
            )
        }
        
        logger.info(f"Model selection completed. Best model: {best_model_type} (RMSE: {best_score:.4f})")
        return final_results
    
    def _suggest_rul_params(self, trial, model_type: str) -> Dict[str, Any]:
        """
        Suggest hyperparameters for RUL models based on model type
        
        Args:
            trial: Optuna trial object
            model_type: Type of model
            
        Returns:
            Dictionary of suggested parameters
        """
        if model_type == "xgboost":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0)
            }
        
        elif model_type == "lightgbm":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
                "num_leaves": trial.suggest_int("num_leaves", 10, 300)
            }
        
        elif model_type == "random_forest":
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 5, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
                "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", None])
            }
        
        elif model_type == "elastic_net":
            return {
                "alpha": trial.suggest_float("alpha", 0.001, 10.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
                "degree": trial.suggest_int("degree", 1, 3),
                "max_iter": trial.suggest_int("max_iter", 1000, 10000)
            }
        
        elif model_type == "ensemble":
            return {
                "xgb_weight": trial.suggest_float("xgb_weight", 0.2, 0.6),
                "lgb_weight": trial.suggest_float("lgb_weight", 0.2, 0.6),
                # RF weight will be 1 - xgb_weight - lgb_weight
            }
        
        else:
            return {}
    
    def save_results(self, filepath: str) -> None:
        """
        Save optimization results to file
        
        Args:
            filepath: Path to save results
        """
        results = {
            "study_name": self.study_name,
            "best_params": self.best_params,
            "best_score": self.best_score,
            "optimization_history": self.optimization_history,
            "timestamp": time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Optimization results saved to {filepath}")
    
    def load_results(self, filepath: str) -> None:
        """
        Load optimization results from file
        
        Args:
            filepath: Path to load results from
        """
        with open(filepath, 'r') as f:
            results = json.load(f)
        
        self.best_params = results.get("best_params", {})
        self.best_score = results.get("best_score", float('inf'))
        self.optimization_history = results.get("optimization_history", [])
        
        logger.info(f"Optimization results loaded from {filepath}")


class BayesianEnsembleOptimizer:
    """
    Specialized Bayesian optimizer for ensemble weights
    
    This class implements advanced Bayesian optimization techniques
    specifically for optimizing ensemble model weights using
    Gaussian Process regression and acquisition functions.
    """
    
    def __init__(self, 
                 acquisition_function: str = "expected_improvement",
                 n_initial_points: int = 10,
                 random_state: int = 42):
        """
        Initialize Bayesian ensemble optimizer
        
        Args:
            acquisition_function: Acquisition function for Bayesian optimization
            n_initial_points: Number of initial random points
            random_state: Random seed
        """
        self.acquisition_function = acquisition_function
        self.n_initial_points = n_initial_points
        self.random_state = random_state
        
        logger.info("Initialized BayesianEnsembleOptimizer")
    
    def optimize_weights(self,
                        models: List[Any],
                        X_val: np.ndarray,
                        y_val: np.ndarray,
                        n_calls: int = 50) -> Dict[str, Any]:
        """
        Optimize ensemble weights using Bayesian optimization
        
        Args:
            models: List of trained models
            X_val: Validation features
            y_val: Validation labels
            n_calls: Number of optimization calls
            
        Returns:
            Dictionary with optimal weights and optimization history
        """
        try:
            from skopt import gp_minimize
            from skopt.space import Real
            from skopt.utils import use_named_args
        except ImportError:
            logger.error("scikit-optimize not installed. Using fallback optimization.")
            return self._fallback_optimization(models, X_val, y_val)
        
        # Define search space (weights must sum to 1)
        dimensions = [Real(0.0, 1.0, name=f'weight_{i}') for i in range(len(models) - 1)]
        
        @use_named_args(dimensions)
        def objective(**params):
            # Extract weights and compute last weight
            weights = [params[f'weight_{i}'] for i in range(len(models) - 1)]
            weights.append(1.0 - sum(weights))
            
            # Ensure all weights are non-negative
            if any(w < 0 for w in weights):
                return 1e6  # Large penalty
            
            # Compute ensemble prediction
            ensemble_pred = np.zeros_like(y_val, dtype=float)
            for model, weight in zip(models, weights):
                pred = model.predict(X_val)
                ensemble_pred += weight * pred
            
            # Return RMSE
            rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
            return rmse
        
        # Run Bayesian optimization
        result = gp_minimize(
            func=objective,
            dimensions=dimensions,
            n_calls=n_calls,
            n_initial_points=self.n_initial_points,
            acq_func=self.acquisition_function,
            random_state=self.random_state
        )
        
        # Extract optimal weights
        optimal_weights = list(result.x) + [1.0 - sum(result.x)]
        
        return {
            "optimal_weights": optimal_weights,
            "best_score": result.fun,
            "n_calls": len(result.func_vals),
            "optimization_history": result.func_vals
        }
    
    def _fallback_optimization(self,
                              models: List[Any],
                              X_val: np.ndarray,
                              y_val: np.ndarray) -> Dict[str, Any]:
        """
        Fallback optimization using grid search when scikit-optimize is not available
        
        Args:
            models: List of trained models
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary with optimal weights
        """
        logger.info("Using fallback grid search for ensemble weight optimization")
        
        best_weights = None
        best_score = float('inf')
        
        # Simple grid search over weight combinations
        n_models = len(models)
        grid_points = 11  # 0.0, 0.1, 0.2, ..., 1.0
        
        for i in range(grid_points ** (n_models - 1)):
            # Convert index to weight combination
            weights = []
            remaining = 1.0
            temp_i = i
            
            for j in range(n_models - 1):
                weight = (temp_i % grid_points) / (grid_points - 1) * remaining
                weights.append(weight)
                remaining -= weight
                temp_i //= grid_points
            
            weights.append(remaining)
            
            # Skip if any weight is negative
            if any(w < 0 for w in weights):
                continue
            
            # Compute ensemble prediction
            ensemble_pred = np.zeros_like(y_val, dtype=float)
            for model, weight in zip(models, weights):
                pred = model.predict(X_val)
                ensemble_pred += weight * pred
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
            
            if rmse < best_score:
                best_score = rmse
                best_weights = weights
        
        return {
            "optimal_weights": best_weights,
            "best_score": best_score,
            "n_calls": grid_points ** (n_models - 1),
            "optimization_history": []
        }