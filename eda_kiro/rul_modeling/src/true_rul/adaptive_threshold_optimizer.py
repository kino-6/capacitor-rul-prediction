"""
Adaptive Threshold Optimizer

This module implements dynamic threshold adjustment for anomaly detection
using Bayesian optimization, cross-validation, and online learning techniques.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import logging
from dataclasses import dataclass, field
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import precision_recall_curve, roc_curve, f1_score
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import joblib
from pathlib import Path
import json
from collections import deque
import time

logger = logging.getLogger(__name__)


@dataclass
class ThresholdOptimizationConfig:
    """Configuration for adaptive threshold optimization."""
    # Bayesian optimization settings
    n_trials: int = 100
    optimization_timeout: int = 300  # seconds
    sampler_seed: int = 42
    
    # Cross-validation settings
    cv_folds: int = 5
    cv_stratify: bool = True
    cv_random_state: int = 42
    
    # Optimization objectives
    primary_metric: str = "f1_score"  # "f1_score", "precision", "recall", "fpr"
    target_fpr: float = 0.05  # Target false positive rate
    min_precision: float = 0.8  # Minimum precision constraint
    min_recall: float = 0.8  # Minimum recall constraint
    
    # Online learning settings
    online_learning: bool = True
    history_window: int = 1000  # Number of recent predictions to consider
    adaptation_rate: float = 0.1  # Learning rate for threshold adaptation
    min_samples_for_update: int = 50  # Minimum samples before updating threshold
    
    # Threshold bounds
    min_threshold: float = 0.01
    max_threshold: float = 0.99
    
    # Performance tracking
    track_performance: bool = True
    performance_window: int = 100  # Window for performance tracking


@dataclass
class ThresholdPerformance:
    """Performance metrics for a specific threshold."""
    threshold: float
    fpr: float
    tpr: float
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class OptimizationResult:
    """Result of threshold optimization."""
    optimal_threshold: float
    best_score: float
    optimization_history: List[Dict[str, Any]]
    cv_scores: Dict[str, float]
    performance_metrics: ThresholdPerformance
    optimization_time: float


class AdaptiveThresholdOptimizer:
    """
    Adaptive threshold optimizer for anomaly detection.
    
    This class implements dynamic threshold adjustment using:
    1. Bayesian optimization for initial threshold selection
    2. Cross-validation for robust threshold evaluation
    3. Online learning for continuous threshold adaptation
    4. Performance monitoring and drift detection
    """
    
    def __init__(self, config: ThresholdOptimizationConfig):
        self.config = config
        self.current_threshold: Optional[float] = None
        self.optimization_history: List[OptimizationResult] = []
        self.performance_history: deque = deque(maxlen=config.performance_window)
        self.online_predictions: deque = deque(maxlen=config.history_window)
        self.online_labels: deque = deque(maxlen=config.history_window)
        self.is_fitted = False
        
        # Performance tracking
        self.performance_tracker: Dict[str, List[float]] = {
            'fpr': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        
        logger.info("Adaptive threshold optimizer initialized")
    
    def optimize_threshold(
        self,
        anomaly_scores: np.ndarray,
        true_labels: np.ndarray,
        detector_predict_fn: Optional[Callable] = None
    ) -> OptimizationResult:
        """
        Optimize threshold using Bayesian optimization and cross-validation.
        
        Args:
            anomaly_scores: Anomaly scores from detector
            true_labels: True binary labels (1 = anomaly, 0 = normal)
            detector_predict_fn: Optional detector prediction function for CV
            
        Returns:
            Optimization result with optimal threshold and metrics
        """
        logger.info("Starting Bayesian threshold optimization...")
        start_time = time.time()
        
        # Create Optuna study
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=self.config.sampler_seed),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3)  # Less aggressive pruning
        )
        
        # Define objective function
        def objective(trial):
            threshold = trial.suggest_float(
                "threshold",
                self.config.min_threshold,
                self.config.max_threshold
            )
            
            # Compute performance metrics
            performance = self._evaluate_threshold(
                threshold, anomaly_scores, true_labels
            )
            
            # Multi-objective optimization with constraints
            score = self._compute_objective_score(performance, trial)
            
            # Log intermediate results
            trial.set_user_attr("fpr", performance.fpr)
            trial.set_user_attr("precision", performance.precision)
            trial.set_user_attr("recall", performance.recall)
            trial.set_user_attr("f1_score", performance.f1_score)
            
            return score
        
        # Run optimization
        try:
            study.optimize(
                objective,
                n_trials=self.config.n_trials,
                timeout=self.config.optimization_timeout,
                show_progress_bar=True
            )
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            # Fallback to simple threshold selection
            return self._fallback_threshold_selection(anomaly_scores, true_labels)
        
        # Extract best results
        best_trial = study.best_trial
        optimal_threshold = best_trial.params["threshold"]
        
        # Evaluate optimal threshold
        optimal_performance = self._evaluate_threshold(
            optimal_threshold, anomaly_scores, true_labels
        )
        
        # Cross-validation evaluation
        cv_scores = {}
        if detector_predict_fn is not None:
            cv_scores = self._cross_validate_threshold(
                optimal_threshold, detector_predict_fn, anomaly_scores, true_labels
            )
        
        # Create optimization result
        optimization_time = time.time() - start_time
        result = OptimizationResult(
            optimal_threshold=optimal_threshold,
            best_score=study.best_value,
            optimization_history=[
                {
                    "trial": trial.number,
                    "threshold": trial.params["threshold"],
                    "score": trial.value,
                    "fpr": trial.user_attrs.get("fpr", 0),
                    "precision": trial.user_attrs.get("precision", 0),
                    "recall": trial.user_attrs.get("recall", 0),
                    "f1_score": trial.user_attrs.get("f1_score", 0)
                }
                for trial in study.trials
            ],
            cv_scores=cv_scores,
            performance_metrics=optimal_performance,
            optimization_time=optimization_time
        )
        
        # Update current threshold and history
        self.current_threshold = optimal_threshold
        self.optimization_history.append(result)
        self.is_fitted = True
        
        logger.info(f"Optimization completed in {optimization_time:.2f}s")
        logger.info(f"Optimal threshold: {optimal_threshold:.4f}")
        logger.info(f"Best score: {study.best_value:.4f}")
        logger.info(f"FPR: {optimal_performance.fpr:.4f}")
        logger.info(f"F1 Score: {optimal_performance.f1_score:.4f}")
        
        return result
    
    def _evaluate_threshold(
        self,
        threshold: float,
        anomaly_scores: np.ndarray,
        true_labels: np.ndarray
    ) -> ThresholdPerformance:
        """Evaluate threshold performance."""
        predictions = (anomaly_scores > threshold).astype(int)
        
        # Compute confusion matrix components
        tp = np.sum((predictions == 1) & (true_labels == 1))
        fp = np.sum((predictions == 1) & (true_labels == 0))
        tn = np.sum((predictions == 0) & (true_labels == 0))
        fn = np.sum((predictions == 0) & (true_labels == 1))
        
        # Compute metrics
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tpr
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / len(true_labels) if len(true_labels) > 0 else 0.0
        
        return ThresholdPerformance(
            threshold=threshold,
            fpr=fpr,
            tpr=tpr,
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy
        )
    
    def _compute_objective_score(
        self,
        performance: ThresholdPerformance,
        trial: optuna.Trial
    ) -> float:
        """Compute objective score with constraints."""
        # Primary metric
        if self.config.primary_metric == "f1_score":
            primary_score = performance.f1_score
        elif self.config.primary_metric == "precision":
            primary_score = performance.precision
        elif self.config.primary_metric == "recall":
            primary_score = performance.recall
        elif self.config.primary_metric == "fpr":
            primary_score = 1.0 - performance.fpr  # Minimize FPR
        else:
            primary_score = performance.f1_score
        
        # Apply constraints
        penalty = 0.0
        
        # FPR constraint (more lenient for small datasets)
        if performance.fpr > self.config.target_fpr * 2.0:  # More lenient
            penalty += (performance.fpr - self.config.target_fpr * 2.0) * 5  # Reduced penalty
        
        # Precision constraint (more lenient)
        if performance.precision < self.config.min_precision * 0.7:  # More lenient
            penalty += (self.config.min_precision * 0.7 - performance.precision) * 3  # Reduced penalty
        
        # Recall constraint (more lenient)
        if performance.recall < self.config.min_recall * 0.7:  # More lenient
            penalty += (self.config.min_recall * 0.7 - performance.recall) * 3  # Reduced penalty
        
        # Final score with penalty
        final_score = primary_score - penalty
        
        # Prune unpromising trials (more lenient)
        if penalty > 2.0:  # Increased threshold
            raise optuna.TrialPruned()
        
        return final_score
    
    def _cross_validate_threshold(
        self,
        threshold: float,
        detector_predict_fn: Callable,
        X: np.ndarray,
        y: np.ndarray
    ) -> Dict[str, float]:
        """Cross-validate threshold performance."""
        logger.info("Performing cross-validation...")
        
        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.cv_random_state
        )
        
        cv_metrics = {
            'fpr': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            try:
                # Get validation data
                X_val = X[val_idx]
                y_val = y[val_idx]
                
                # Get validation predictions - handle both 1D and 2D inputs
                if X_val.ndim == 1:
                    X_val = X_val.reshape(1, -1)
                
                val_scores = detector_predict_fn(X_val)
                
                # Evaluate threshold
                performance = self._evaluate_threshold(threshold, val_scores, y_val)
                
                cv_metrics['fpr'].append(performance.fpr)
                cv_metrics['precision'].append(performance.precision)
                cv_metrics['recall'].append(performance.recall)
                cv_metrics['f1_score'].append(performance.f1_score)
                
            except Exception as e:
                logger.warning(f"CV fold {fold} failed: {e}")
        
        # Compute mean and std
        cv_results = {}
        for metric, values in cv_metrics.items():
            if values:
                cv_results[f"{metric}_mean"] = np.mean(values)
                cv_results[f"{metric}_std"] = np.std(values)
            else:
                cv_results[f"{metric}_mean"] = 0.0
                cv_results[f"{metric}_std"] = 0.0
        
        return cv_results
    
    def _fallback_threshold_selection(
        self,
        anomaly_scores: np.ndarray,
        true_labels: np.ndarray
    ) -> OptimizationResult:
        """Fallback threshold selection using precision-recall curve."""
        logger.info("Using fallback threshold selection...")
        
        precision, recall, thresholds = precision_recall_curve(true_labels, anomaly_scores)
        
        # Find threshold that maximizes F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        
        if best_idx < len(thresholds):
            optimal_threshold = thresholds[best_idx]
        else:
            optimal_threshold = 0.5
        
        # Evaluate performance
        performance = self._evaluate_threshold(optimal_threshold, anomaly_scores, true_labels)
        
        return OptimizationResult(
            optimal_threshold=optimal_threshold,
            best_score=performance.f1_score,
            optimization_history=[],
            cv_scores={},
            performance_metrics=performance,
            optimization_time=0.0
        )
    
    def update_threshold_online(
        self,
        new_scores: np.ndarray,
        new_labels: np.ndarray
    ) -> Optional[float]:
        """
        Update threshold using online learning.
        
        Args:
            new_scores: New anomaly scores
            new_labels: New true labels
            
        Returns:
            Updated threshold if changed, None otherwise
        """
        if not self.config.online_learning or not self.is_fitted:
            return None
        
        # Add new data to history
        self.online_predictions.extend(new_scores)
        self.online_labels.extend(new_labels)
        
        # Check if we have enough samples for update
        if len(self.online_predictions) < self.config.min_samples_for_update:
            return None
        
        # Convert to arrays
        recent_scores = np.array(list(self.online_predictions))
        recent_labels = np.array(list(self.online_labels))
        
        # Evaluate current threshold performance
        current_performance = self._evaluate_threshold(
            self.current_threshold, recent_scores, recent_labels
        )
        
        # Check if adaptation is needed
        if self._should_adapt_threshold(current_performance):
            # Find better threshold using recent data
            new_threshold = self._adapt_threshold(recent_scores, recent_labels)
            
            if new_threshold != self.current_threshold:
                logger.info(f"Threshold adapted: {self.current_threshold:.4f} -> {new_threshold:.4f}")
                self.current_threshold = new_threshold
                return new_threshold
        
        return None
    
    def _should_adapt_threshold(self, performance: ThresholdPerformance) -> bool:
        """Determine if threshold should be adapted."""
        # Adapt if FPR exceeds target
        if performance.fpr > self.config.target_fpr * 1.2:
            return True
        
        # Adapt if F1 score is significantly below recent average
        if self.performance_history:
            recent_f1 = np.mean([p.f1_score for p in self.performance_history])
            if performance.f1_score < recent_f1 * 0.9:
                return True
        
        return False
    
    def _adapt_threshold(
        self,
        scores: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Adapt threshold using gradient-based approach."""
        current_threshold = self.current_threshold
        
        # Compute gradient of F1 score with respect to threshold
        epsilon = 0.01
        
        # Current performance
        current_perf = self._evaluate_threshold(current_threshold, scores, labels)
        
        # Performance with small increase
        upper_perf = self._evaluate_threshold(current_threshold + epsilon, scores, labels)
        
        # Performance with small decrease
        lower_perf = self._evaluate_threshold(current_threshold - epsilon, scores, labels)
        
        # Compute gradient
        gradient = (upper_perf.f1_score - lower_perf.f1_score) / (2 * epsilon)
        
        # Update threshold
        new_threshold = current_threshold + self.config.adaptation_rate * gradient
        
        # Clip to bounds
        new_threshold = np.clip(
            new_threshold,
            self.config.min_threshold,
            self.config.max_threshold
        )
        
        return new_threshold
    
    def get_threshold_recommendations(
        self,
        anomaly_scores: np.ndarray,
        true_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Get threshold recommendations for different objectives.
        
        Args:
            anomaly_scores: Anomaly scores
            true_labels: True labels
            
        Returns:
            Dictionary of recommended thresholds
        """
        recommendations = {}
        
        def _validate_threshold(threshold: float, fallback: float = 0.5) -> float:
            """Validate and sanitize threshold values"""
            if threshold is None:
                return fallback
            if hasattr(threshold, 'item'):  # Handle numpy scalars
                threshold = threshold.item()
            threshold = float(threshold)
            if not np.isfinite(threshold):
                return fallback
            return np.clip(threshold, 0.0, 1.0)
        
        try:
            # Precision-recall curve based recommendations
            precision, recall, pr_thresholds = precision_recall_curve(true_labels, anomaly_scores)
            
            # ROC curve based recommendations
            fpr, tpr, roc_thresholds = roc_curve(true_labels, anomaly_scores)
            
            # Maximum F1 score
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            max_f1_idx = np.argmax(f1_scores)
            if max_f1_idx < len(pr_thresholds):
                recommendations['max_f1'] = _validate_threshold(pr_thresholds[max_f1_idx])
            
            # Target FPR
            target_fpr_idx = np.argmin(np.abs(fpr - self.config.target_fpr))
            if target_fpr_idx < len(roc_thresholds):
                recommendations['target_fpr'] = _validate_threshold(roc_thresholds[target_fpr_idx])
            
            # High precision (precision >= 0.9)
            high_precision_idx = np.where(precision >= 0.9)[0]
            if len(high_precision_idx) > 0:
                recommendations['high_precision'] = _validate_threshold(pr_thresholds[high_precision_idx[0]])
            
            # High recall (recall >= 0.9)
            high_recall_idx = np.where(recall >= 0.9)[0]
            if len(high_recall_idx) > 0:
                recommendations['high_recall'] = _validate_threshold(pr_thresholds[high_recall_idx[-1]])
            
            # Balanced (precision ≈ recall)
            balance_scores = np.abs(precision - recall)
            balanced_idx = np.argmin(balance_scores)
            if balanced_idx < len(pr_thresholds):
                recommendations['balanced'] = _validate_threshold(pr_thresholds[balanced_idx])
            
            # Ensure we always have at least one recommendation
            if not recommendations:
                recommendations['fallback'] = 0.5
                
        except Exception as e:
            # If curve computation fails, provide fallback recommendations
            logger.warning(f"Failed to compute threshold recommendations: {e}")
            recommendations = {
                'fallback': 0.5,
                'conservative': 0.7,
                'aggressive': 0.3
            }
        
        return recommendations
    
    def track_performance(
        self,
        threshold: float,
        scores: np.ndarray,
        labels: np.ndarray
    ) -> None:
        """Track threshold performance over time."""
        if not self.config.track_performance:
            return
        
        performance = self._evaluate_threshold(threshold, scores, labels)
        self.performance_history.append(performance)
        
        # Update performance tracker
        self.performance_tracker['fpr'].append(performance.fpr)
        self.performance_tracker['precision'].append(performance.precision)
        self.performance_tracker['recall'].append(performance.recall)
        self.performance_tracker['f1_score'].append(performance.f1_score)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        if not self.performance_history:
            return {}
        
        summary = {}
        
        for metric in ['fpr', 'precision', 'recall', 'f1_score']:
            values = [getattr(p, metric) for p in self.performance_history]
            summary[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'current': values[-1] if values else 0.0
            }
        
        return summary
    
    def save_optimizer(self, filepath: str) -> None:
        """Save optimizer state to disk."""
        optimizer_data = {
            'config': {
                'n_trials': self.config.n_trials,
                'optimization_timeout': self.config.optimization_timeout,
                'sampler_seed': self.config.sampler_seed,
                'cv_folds': self.config.cv_folds,
                'cv_stratify': self.config.cv_stratify,
                'cv_random_state': self.config.cv_random_state,
                'primary_metric': self.config.primary_metric,
                'target_fpr': self.config.target_fpr,
                'min_precision': self.config.min_precision,
                'min_recall': self.config.min_recall,
                'online_learning': self.config.online_learning,
                'history_window': self.config.history_window,
                'adaptation_rate': self.config.adaptation_rate,
                'min_samples_for_update': self.config.min_samples_for_update,
                'min_threshold': self.config.min_threshold,
                'max_threshold': self.config.max_threshold,
                'track_performance': self.config.track_performance,
                'performance_window': self.config.performance_window
            },
            'current_threshold': self.current_threshold,
            'optimization_history': [
                {
                    'optimal_threshold': result.optimal_threshold,
                    'best_score': result.best_score,
                    'optimization_time': result.optimization_time,
                    'performance_metrics': {
                        'threshold': result.performance_metrics.threshold,
                        'fpr': result.performance_metrics.fpr,
                        'precision': result.performance_metrics.precision,
                        'recall': result.performance_metrics.recall,
                        'f1_score': result.performance_metrics.f1_score
                    }
                }
                for result in self.optimization_history
            ],
            'performance_tracker': self.performance_tracker,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'w') as f:
            json.dump(optimizer_data, f, indent=2)
        
        logger.info(f"Optimizer saved to {filepath}")
    
    def load_optimizer(self, filepath: str) -> 'AdaptiveThresholdOptimizer':
        """Load optimizer state from disk."""
        with open(filepath, 'r') as f:
            optimizer_data = json.load(f)
        
        self.config = ThresholdOptimizationConfig(**optimizer_data['config'])
        self.current_threshold = optimizer_data['current_threshold']
        self.performance_tracker = optimizer_data['performance_tracker']
        self.is_fitted = optimizer_data['is_fitted']
        
        logger.info(f"Optimizer loaded from {filepath}")
        return self


def create_adaptive_threshold_optimizer(**kwargs) -> AdaptiveThresholdOptimizer:
    """
    Factory function to create an adaptive threshold optimizer.
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        Configured adaptive threshold optimizer
    """
    config = ThresholdOptimizationConfig(**kwargs)
    return AdaptiveThresholdOptimizer(config)