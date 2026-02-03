"""
Robust Validation Framework for FPR Testing

This module provides comprehensive validation capabilities for anomaly detection
systems with focus on False Positive Rate (FPR) testing. It includes:

1. K-fold cross-validation with stratified sampling
2. Bootstrap sampling for confidence interval estimation  
3. Synthetic anomaly injection for stress testing
4. Temporal validation (time-series cross-validation)

Author: RUL Prediction System
Date: February 2026
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit, 
    cross_val_score, cross_validate
)
from sklearn.metrics import (
    precision_recall_curve, roc_curve, auc,
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.utils import resample
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import json
import time
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Container for validation metrics"""
    fpr: float
    tpr: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    auc_pr: float
    confusion_matrix: np.ndarray
    threshold: float = 0.5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'fpr': float(self.fpr),
            'tpr': float(self.tpr),
            'precision': float(self.precision),
            'recall': float(self.recall),
            'f1_score': float(self.f1_score),
            'auc_roc': float(self.auc_roc),
            'auc_pr': float(self.auc_pr),
            'confusion_matrix': self.confusion_matrix.tolist(),
            'threshold': float(self.threshold)
        }


@dataclass
class CrossValidationResult:
    """Results from cross-validation"""
    fold_metrics: List[ValidationMetrics]
    mean_metrics: ValidationMetrics
    std_metrics: Dict[str, float]
    fold_predictions: List[np.ndarray] = field(default_factory=list)
    fold_scores: List[np.ndarray] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'fold_metrics': [m.to_dict() for m in self.fold_metrics],
            'mean_metrics': self.mean_metrics.to_dict(),
            'std_metrics': self.std_metrics,
            'n_folds': len(self.fold_metrics)
        }


@dataclass
class BootstrapResult:
    """Results from bootstrap sampling"""
    bootstrap_metrics: List[ValidationMetrics]
    confidence_intervals: Dict[str, Tuple[float, float]]
    mean_metrics: ValidationMetrics
    std_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'bootstrap_metrics': [m.to_dict() for m in self.bootstrap_metrics],
            'confidence_intervals': {
                k: [float(v[0]), float(v[1])] for k, v in self.confidence_intervals.items()
            },
            'mean_metrics': self.mean_metrics.to_dict(),
            'std_metrics': self.std_metrics,
            'n_bootstrap_samples': len(self.bootstrap_metrics)
        }


@dataclass
class SyntheticAnomalyResult:
    """Results from synthetic anomaly injection"""
    injection_rates: List[float]
    metrics_by_rate: Dict[float, ValidationMetrics]
    stress_test_passed: bool
    failure_points: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'injection_rates': self.injection_rates,
            'metrics_by_rate': {
                str(k): v.to_dict() for k, v in self.metrics_by_rate.items()
            },
            'stress_test_passed': self.stress_test_passed,
            'failure_points': self.failure_points
        }


@dataclass
class TemporalValidationResult:
    """Results from temporal validation"""
    split_metrics: List[ValidationMetrics]
    temporal_stability: float
    drift_detected: bool
    performance_trend: List[float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'split_metrics': [m.to_dict() for m in self.split_metrics],
            'temporal_stability': float(self.temporal_stability),
            'drift_detected': self.drift_detected,
            'performance_trend': [float(x) for x in self.performance_trend],
            'n_splits': len(self.split_metrics)
        }


@dataclass
class ValidationConfig:
    """Configuration for robust validation framework"""
    # Cross-validation settings
    cv_folds: int = 5
    cv_stratified: bool = True
    cv_shuffle: bool = True
    cv_random_state: int = 42
    
    # Bootstrap settings
    bootstrap_samples: int = 100
    bootstrap_confidence_level: float = 0.95
    bootstrap_random_state: int = 42
    
    # Synthetic anomaly injection settings
    injection_rates: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1, 0.2, 0.3])
    anomaly_types: List[str] = field(default_factory=lambda: ['gaussian_noise', 'outliers', 'drift'])
    stress_test_fpr_threshold: float = 0.1  # Fail if FPR > 10% under stress
    
    # Temporal validation settings
    temporal_splits: int = 5
    temporal_test_size: float = 0.2
    temporal_gap: int = 0  # Gap between train and test
    
    # General settings
    n_jobs: int = -1
    verbose: bool = True
    random_state: int = 42


class RobustValidationFramework:
    """
    Comprehensive validation framework for anomaly detection systems
    with focus on False Positive Rate (FPR) testing.
    """
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize the validation framework.
        
        Args:
            config: Validation configuration. If None, uses default config.
        """
        self.config = config or ValidationConfig()
        self.results_history: List[Dict[str, Any]] = []
        
    def _compute_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        y_scores: Optional[np.ndarray] = None,
        threshold: float = 0.5
    ) -> ValidationMetrics:
        """
        Compute comprehensive validation metrics.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            y_scores: Prediction scores/probabilities (optional)
            threshold: Decision threshold
            
        Returns:
            ValidationMetrics object
        """
        # Ensure binary classification
        y_true_binary = np.array(y_true).astype(int)
        y_pred_binary = np.array(y_pred).astype(int)
        
        # Handle IsolationForest output (-1, 1) -> (1, 0) for anomaly detection
        if np.any(y_pred_binary == -1):
            y_pred_binary = (y_pred_binary == -1).astype(int)  # -1 (anomaly) -> 1, 1 (normal) -> 0
        
        # Basic metrics
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0, average='binary')
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0, average='binary')
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0, average='binary')
        
        # Confusion matrix
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        
        # FPR and TPR
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        else:
            fpr = 0.0
            tpr = 0.0
        
        # AUC metrics (if scores available)
        auc_roc = 0.0
        auc_pr = 0.0
        if y_scores is not None and len(np.unique(y_true_binary)) > 1:
            try:
                fpr_curve, tpr_curve, _ = roc_curve(y_true_binary, y_scores)
                auc_roc = auc(fpr_curve, tpr_curve)
                
                precision_curve, recall_curve, _ = precision_recall_curve(y_true_binary, y_scores)
                auc_pr = auc(recall_curve, precision_curve)
            except Exception as e:
                logger.warning(f"Could not compute AUC metrics: {e}")
        
        return ValidationMetrics(
            fpr=fpr,
            tpr=tpr,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            confusion_matrix=cm,
            threshold=threshold
        )
    
    def k_fold_cross_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Any,
        scoring_func: Optional[Callable] = None
    ) -> CrossValidationResult:
        """
        Perform k-fold cross-validation with stratified sampling.
        
        Args:
            X: Feature matrix
            y: Target labels
            model: Anomaly detection model with fit/predict methods
            scoring_func: Custom scoring function (optional)
            
        Returns:
            CrossValidationResult object
        """
        logger.info(f"Starting {self.config.cv_folds}-fold cross-validation...")
        
        # Choose cross-validation strategy
        if self.config.cv_stratified and len(np.unique(y)) > 1:
            cv = StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=self.config.cv_shuffle,
                random_state=self.config.cv_random_state
            )
        else:
            cv = KFold(
                n_splits=self.config.cv_folds,
                shuffle=self.config.cv_shuffle,
                random_state=self.config.cv_random_state
            )
        
        fold_metrics = []
        fold_predictions = []
        fold_scores = []
        
        # Perform cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            if self.config.verbose:
                logger.info(f"Processing fold {fold_idx + 1}/{self.config.cv_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_val)
                
                # Get scores if available
                y_scores = None
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_val)
                    if proba is not None and proba.shape[1] > 1:
                        y_scores = proba[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_scores = model.decision_function(X_val)
                elif hasattr(model, 'score_samples'):
                    y_scores = -model.score_samples(X_val)  # Negative for anomaly scores
                
                # Compute metrics
                metrics = self._compute_metrics(y_val, y_pred, y_scores)
                fold_metrics.append(metrics)
                fold_predictions.append(y_pred)
                if y_scores is not None:
                    fold_scores.append(y_scores)
                
            except Exception as e:
                logger.error(f"Error in fold {fold_idx + 1}: {e}")
                # Create dummy metrics for failed fold
                dummy_metrics = ValidationMetrics(
                    fpr=1.0, tpr=0.0, precision=0.0, recall=0.0,
                    f1_score=0.0, auc_roc=0.0, auc_pr=0.0,
                    confusion_matrix=np.array([[0, 1], [1, 0]])
                )
                fold_metrics.append(dummy_metrics)
        
        # Compute mean and std metrics
        metric_names = ['fpr', 'tpr', 'precision', 'recall', 'f1_score', 'auc_roc', 'auc_pr']
        mean_values = {}
        std_values = {}
        
        for metric_name in metric_names:
            values = [getattr(m, metric_name) for m in fold_metrics]
            mean_values[metric_name] = np.mean(values)
            std_values[metric_name] = np.std(values)
        
        # Create mean metrics object
        mean_cm = np.mean([m.confusion_matrix for m in fold_metrics], axis=0)
        mean_metrics = ValidationMetrics(
            fpr=mean_values['fpr'],
            tpr=mean_values['tpr'],
            precision=mean_values['precision'],
            recall=mean_values['recall'],
            f1_score=mean_values['f1_score'],
            auc_roc=mean_values['auc_roc'],
            auc_pr=mean_values['auc_pr'],
            confusion_matrix=mean_cm
        )
        
        logger.info(f"Cross-validation completed. Mean FPR: {mean_values['fpr']:.4f} ± {std_values['fpr']:.4f}")
        
        return CrossValidationResult(
            fold_metrics=fold_metrics,
            mean_metrics=mean_metrics,
            std_metrics=std_values,
            fold_predictions=fold_predictions,
            fold_scores=fold_scores
        )
    
    def bootstrap_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Any,
        n_samples: Optional[int] = None
    ) -> BootstrapResult:
        """
        Perform bootstrap sampling for confidence interval estimation.
        
        Args:
            X: Feature matrix
            y: Target labels
            model: Trained anomaly detection model
            n_samples: Number of bootstrap samples (uses config if None)
            
        Returns:
            BootstrapResult object
        """
        n_samples = n_samples or self.config.bootstrap_samples
        logger.info(f"Starting bootstrap validation with {n_samples} samples...")
        
        bootstrap_metrics = []
        
        # Perform bootstrap sampling
        for i in tqdm(range(n_samples), desc="Bootstrap sampling", disable=not self.config.verbose):
            try:
                # Bootstrap sample
                X_boot, y_boot = resample(
                    X, y, 
                    random_state=self.config.bootstrap_random_state + i,
                    stratify=y if len(np.unique(y)) > 1 else None
                )
                
                # Make predictions
                y_pred = model.predict(X_boot)
                
                # Get scores if available
                y_scores = None
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_boot)
                    if proba is not None and proba.shape[1] > 1:
                        y_scores = proba[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_scores = model.decision_function(X_boot)
                elif hasattr(model, 'score_samples'):
                    y_scores = -model.score_samples(X_boot)
                
                # Compute metrics
                metrics = self._compute_metrics(y_boot, y_pred, y_scores)
                bootstrap_metrics.append(metrics)
                
            except Exception as e:
                logger.warning(f"Error in bootstrap sample {i + 1}: {e}")
                continue
        
        if not bootstrap_metrics:
            raise ValueError("All bootstrap samples failed")
        
        # Compute confidence intervals
        confidence_level = self.config.bootstrap_confidence_level
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        metric_names = ['fpr', 'tpr', 'precision', 'recall', 'f1_score', 'auc_roc', 'auc_pr']
        confidence_intervals = {}
        mean_values = {}
        std_values = {}
        
        for metric_name in metric_names:
            values = [getattr(m, metric_name) for m in bootstrap_metrics]
            mean_values[metric_name] = np.mean(values)
            std_values[metric_name] = np.std(values)
            
            lower_bound = np.percentile(values, lower_percentile)
            upper_bound = np.percentile(values, upper_percentile)
            confidence_intervals[metric_name] = (lower_bound, upper_bound)
        
        # Create mean metrics object
        mean_cm = np.mean([m.confusion_matrix for m in bootstrap_metrics], axis=0)
        mean_metrics = ValidationMetrics(
            fpr=mean_values['fpr'],
            tpr=mean_values['tpr'],
            precision=mean_values['precision'],
            recall=mean_values['recall'],
            f1_score=mean_values['f1_score'],
            auc_roc=mean_values['auc_roc'],
            auc_pr=mean_values['auc_pr'],
            confusion_matrix=mean_cm
        )
        
        logger.info(f"Bootstrap validation completed. FPR CI: {confidence_intervals['fpr']}")
        
        return BootstrapResult(
            bootstrap_metrics=bootstrap_metrics,
            confidence_intervals=confidence_intervals,
            mean_metrics=mean_metrics,
            std_metrics=std_values
        )
    
    def _inject_synthetic_anomalies(
        self,
        X: np.ndarray,
        injection_rate: float,
        anomaly_type: str = 'gaussian_noise'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject synthetic anomalies into the dataset.
        
        Args:
            X: Original feature matrix
            injection_rate: Fraction of samples to make anomalous
            anomaly_type: Type of anomaly to inject
            
        Returns:
            Tuple of (modified_X, anomaly_labels)
        """
        n_samples = len(X)
        n_anomalies = int(n_samples * injection_rate)
        
        # Create copy of data
        X_modified = X.copy()
        anomaly_labels = np.zeros(n_samples, dtype=int)
        
        if n_anomalies == 0:
            return X_modified, anomaly_labels
        
        # Select random samples to make anomalous
        anomaly_indices = np.random.choice(
            n_samples, size=n_anomalies, replace=False
        )
        anomaly_labels[anomaly_indices] = 1
        
        # Inject anomalies based on type
        if anomaly_type == 'gaussian_noise':
            # Add Gaussian noise
            noise_scale = np.std(X, axis=0) * 2.0  # 2x standard deviation
            noise = np.random.normal(0, noise_scale, size=(n_anomalies, X.shape[1]))
            X_modified[anomaly_indices] += noise
            
        elif anomaly_type == 'outliers':
            # Create outliers by scaling values
            scale_factors = np.random.uniform(3.0, 5.0, size=(n_anomalies, X.shape[1]))
            X_modified[anomaly_indices] *= scale_factors
            
        elif anomaly_type == 'drift':
            # Add systematic drift
            drift_direction = np.random.choice([-1, 1], size=X.shape[1])
            drift_magnitude = np.std(X, axis=0) * 3.0
            drift = drift_direction * drift_magnitude
            X_modified[anomaly_indices] += drift
            
        else:
            raise ValueError(f"Unknown anomaly type: {anomaly_type}")
        
        return X_modified, anomaly_labels
    
    def synthetic_anomaly_injection(
        self,
        X: np.ndarray,
        model: Any,
        injection_rates: Optional[List[float]] = None,
        anomaly_types: Optional[List[str]] = None
    ) -> SyntheticAnomalyResult:
        """
        Perform synthetic anomaly injection for stress testing.
        
        Args:
            X: Clean feature matrix (assumed to be normal)
            model: Trained anomaly detection model
            injection_rates: List of injection rates to test
            anomaly_types: List of anomaly types to test
            
        Returns:
            SyntheticAnomalyResult object
        """
        injection_rates = injection_rates or self.config.injection_rates
        anomaly_types = anomaly_types or self.config.anomaly_types
        
        logger.info(f"Starting synthetic anomaly injection test...")
        logger.info(f"Injection rates: {injection_rates}")
        logger.info(f"Anomaly types: {anomaly_types}")
        
        metrics_by_rate = {}
        failure_points = []
        
        for rate in injection_rates:
            rate_metrics = []
            
            # Test each anomaly type
            for anomaly_type in anomaly_types:
                try:
                    # Inject synthetic anomalies
                    X_modified, y_anomaly = self._inject_synthetic_anomalies(
                        X, rate, anomaly_type
                    )
                    
                    # Make predictions
                    y_pred = model.predict(X_modified)
                    
                    # Get scores if available
                    y_scores = None
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_modified)
                        if proba is not None and proba.shape[1] > 1:
                            y_scores = proba[:, 1]
                    elif hasattr(model, 'decision_function'):
                        y_scores = model.decision_function(X_modified)
                    elif hasattr(model, 'score_samples'):
                        y_scores = -model.score_samples(X_modified)
                    
                    # Compute metrics
                    metrics = self._compute_metrics(y_anomaly, y_pred, y_scores)
                    rate_metrics.append(metrics)
                    
                except Exception as e:
                    logger.warning(f"Error with rate {rate}, type {anomaly_type}: {e}")
                    continue
            
            if rate_metrics:
                # Average metrics across anomaly types
                avg_fpr = np.mean([m.fpr for m in rate_metrics])
                avg_tpr = np.mean([m.tpr for m in rate_metrics])
                avg_precision = np.mean([m.precision for m in rate_metrics])
                avg_recall = np.mean([m.recall for m in rate_metrics])
                avg_f1 = np.mean([m.f1_score for m in rate_metrics])
                avg_auc_roc = np.mean([m.auc_roc for m in rate_metrics])
                avg_auc_pr = np.mean([m.auc_pr for m in rate_metrics])
                avg_cm = np.mean([m.confusion_matrix for m in rate_metrics], axis=0)
                
                avg_metrics = ValidationMetrics(
                    fpr=avg_fpr, tpr=avg_tpr, precision=avg_precision,
                    recall=avg_recall, f1_score=avg_f1, auc_roc=avg_auc_roc,
                    auc_pr=avg_auc_pr, confusion_matrix=avg_cm
                )
                
                metrics_by_rate[rate] = avg_metrics
                
                # Check if FPR exceeds stress test threshold
                if avg_fpr > self.config.stress_test_fpr_threshold:
                    failure_points.append(rate)
                    logger.warning(f"Stress test failure at injection rate {rate}: FPR = {avg_fpr:.4f}")
        
        # Determine if stress test passed
        stress_test_passed = len(failure_points) == 0
        
        logger.info(f"Synthetic anomaly injection completed. Stress test passed: {stress_test_passed}")
        
        return SyntheticAnomalyResult(
            injection_rates=injection_rates,
            metrics_by_rate=metrics_by_rate,
            stress_test_passed=stress_test_passed,
            failure_points=failure_points
        )
    
    def temporal_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Any,
        time_index: Optional[np.ndarray] = None
    ) -> TemporalValidationResult:
        """
        Perform temporal validation (time-series cross-validation).
        
        Args:
            X: Feature matrix
            y: Target labels
            model: Anomaly detection model
            time_index: Time index for samples (optional)
            
        Returns:
            TemporalValidationResult object
        """
        logger.info(f"Starting temporal validation with {self.config.temporal_splits} splits...")
        
        # Use TimeSeriesSplit for temporal validation
        tscv = TimeSeriesSplit(
            n_splits=self.config.temporal_splits,
            test_size=int(len(X) * self.config.temporal_test_size),
            gap=self.config.temporal_gap
        )
        
        split_metrics = []
        performance_trend = []
        
        for split_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            if self.config.verbose:
                logger.info(f"Processing temporal split {split_idx + 1}/{self.config.temporal_splits}")
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            try:
                # Train model on historical data
                model.fit(X_train, y_train)
                
                # Test on future data
                y_pred = model.predict(X_test)
                
                # Get scores if available
                y_scores = None
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_test)
                    if proba is not None and proba.shape[1] > 1:
                        y_scores = proba[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_scores = model.decision_function(X_test)
                elif hasattr(model, 'score_samples'):
                    y_scores = -model.score_samples(X_test)
                
                # Compute metrics
                metrics = self._compute_metrics(y_test, y_pred, y_scores)
                split_metrics.append(metrics)
                performance_trend.append(metrics.f1_score)
                
            except Exception as e:
                logger.error(f"Error in temporal split {split_idx + 1}: {e}")
                # Add dummy metrics for failed split
                dummy_metrics = ValidationMetrics(
                    fpr=1.0, tpr=0.0, precision=0.0, recall=0.0,
                    f1_score=0.0, auc_roc=0.0, auc_pr=0.0,
                    confusion_matrix=np.array([[0, 1], [1, 0]])
                )
                split_metrics.append(dummy_metrics)
                performance_trend.append(0.0)
        
        # Compute temporal stability (coefficient of variation of F1 scores)
        if len(performance_trend) > 1:
            temporal_stability = 1.0 - (np.std(performance_trend) / np.mean(performance_trend))
        else:
            temporal_stability = 1.0
        
        # Detect performance drift (significant downward trend)
        drift_detected = False
        if len(performance_trend) >= 3:
            # Simple linear trend detection
            x = np.arange(len(performance_trend))
            slope = np.polyfit(x, performance_trend, 1)[0]
            drift_detected = slope < -0.05  # Significant negative slope
        
        logger.info(f"Temporal validation completed. Stability: {temporal_stability:.4f}, Drift detected: {drift_detected}")
        
        return TemporalValidationResult(
            split_metrics=split_metrics,
            temporal_stability=temporal_stability,
            drift_detected=drift_detected,
            performance_trend=performance_trend
        )
    
    def comprehensive_validation(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model: Any,
        save_results: bool = True,
        results_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive validation including all validation methods.
        
        Args:
            X: Feature matrix
            y: Target labels
            model: Anomaly detection model
            save_results: Whether to save results to file
            results_path: Path to save results (optional)
            
        Returns:
            Dictionary containing all validation results
        """
        logger.info("Starting comprehensive validation...")
        start_time = time.time()
        
        results = {
            'timestamp': time.time(),
            'config': self.config.__dict__,
            'data_info': {
                'n_samples': len(X),
                'n_features': X.shape[1],
                'n_anomalies': int(np.sum(y)),
                'anomaly_rate': float(np.mean(y))
            }
        }
        
        try:
            # 1. K-fold cross-validation
            logger.info("1/4: Running k-fold cross-validation...")
            cv_result = self.k_fold_cross_validation(X, y, model)
            results['cross_validation'] = cv_result.to_dict()
            
            # 2. Bootstrap validation
            logger.info("2/4: Running bootstrap validation...")
            bootstrap_result = self.bootstrap_validation(X, y, model)
            results['bootstrap'] = bootstrap_result.to_dict()
            
            # 3. Synthetic anomaly injection
            logger.info("3/4: Running synthetic anomaly injection...")
            # Use only normal samples for injection
            normal_indices = y == 0
            if np.any(normal_indices):
                X_normal = X[normal_indices]
                synthetic_result = self.synthetic_anomaly_injection(X_normal, model)
                results['synthetic_anomaly'] = synthetic_result.to_dict()
            else:
                logger.warning("No normal samples found for synthetic anomaly injection")
                results['synthetic_anomaly'] = None
            
            # 4. Temporal validation
            logger.info("4/4: Running temporal validation...")
            temporal_result = self.temporal_validation(X, y, model)
            results['temporal'] = temporal_result.to_dict()
            
        except Exception as e:
            logger.error(f"Error during comprehensive validation: {e}")
            results['error'] = str(e)
        
        # Add timing information
        results['validation_time'] = time.time() - start_time
        
        # Save results if requested
        if save_results:
            if results_path is None:
                results_path = f"validation_results_{int(time.time())}.json"
            
            results_path = Path(results_path)
            results_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Results saved to {results_path}")
        
        # Add to history
        self.results_history.append(results)
        
        logger.info(f"Comprehensive validation completed in {results['validation_time']:.2f} seconds")
        
        return results
    
    def generate_validation_report(
        self,
        results: Dict[str, Any],
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate a human-readable validation report.
        
        Args:
            results: Validation results from comprehensive_validation
            output_path: Path to save report (optional)
            
        Returns:
            Report as string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ROBUST VALIDATION FRAMEWORK - COMPREHENSIVE REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Data information
        if 'data_info' in results:
            info = results['data_info']
            report_lines.append("📊 DATASET INFORMATION")
            report_lines.append("-" * 40)
            report_lines.append(f"Samples: {info['n_samples']:,}")
            report_lines.append(f"Features: {info['n_features']:,}")
            report_lines.append(f"Anomalies: {info['n_anomalies']:,} ({info['anomaly_rate']:.2%})")
            report_lines.append("")
        
        # Cross-validation results
        if 'cross_validation' in results and results['cross_validation']:
            cv = results['cross_validation']
            mean_metrics = cv['mean_metrics']
            std_metrics = cv['std_metrics']
            
            report_lines.append("🔄 K-FOLD CROSS-VALIDATION RESULTS")
            report_lines.append("-" * 40)
            report_lines.append(f"Folds: {cv['n_folds']}")
            report_lines.append(f"FPR: {mean_metrics['fpr']:.4f} ± {std_metrics['fpr']:.4f}")
            report_lines.append(f"TPR: {mean_metrics['tpr']:.4f} ± {std_metrics['tpr']:.4f}")
            report_lines.append(f"Precision: {mean_metrics['precision']:.4f} ± {std_metrics['precision']:.4f}")
            report_lines.append(f"Recall: {mean_metrics['recall']:.4f} ± {std_metrics['recall']:.4f}")
            report_lines.append(f"F1 Score: {mean_metrics['f1_score']:.4f} ± {std_metrics['f1_score']:.4f}")
            report_lines.append(f"AUC-ROC: {mean_metrics['auc_roc']:.4f} ± {std_metrics['auc_roc']:.4f}")
            report_lines.append("")
        
        # Bootstrap results
        if 'bootstrap' in results and results['bootstrap']:
            bootstrap = results['bootstrap']
            mean_metrics = bootstrap['mean_metrics']
            ci = bootstrap['confidence_intervals']
            
            report_lines.append("🎯 BOOTSTRAP VALIDATION RESULTS")
            report_lines.append("-" * 40)
            report_lines.append(f"Bootstrap samples: {bootstrap['n_bootstrap_samples']}")
            report_lines.append(f"FPR: {mean_metrics['fpr']:.4f} [{ci['fpr'][0]:.4f}, {ci['fpr'][1]:.4f}]")
            report_lines.append(f"F1 Score: {mean_metrics['f1_score']:.4f} [{ci['f1_score'][0]:.4f}, {ci['f1_score'][1]:.4f}]")
            report_lines.append("")
        
        # Synthetic anomaly results
        if 'synthetic_anomaly' in results and results['synthetic_anomaly']:
            synthetic = results['synthetic_anomaly']
            
            report_lines.append("🧪 SYNTHETIC ANOMALY INJECTION RESULTS")
            report_lines.append("-" * 40)
            report_lines.append(f"Stress test passed: {'✅ YES' if synthetic['stress_test_passed'] else '❌ NO'}")
            
            if synthetic['failure_points']:
                report_lines.append(f"Failure points: {synthetic['failure_points']}")
            
            report_lines.append("Performance by injection rate:")
            for rate_str, metrics in synthetic['metrics_by_rate'].items():
                rate = float(rate_str)
                report_lines.append(f"  {rate:.1%}: FPR={metrics['fpr']:.4f}, F1={metrics['f1_score']:.4f}")
            report_lines.append("")
        
        # Temporal validation results
        if 'temporal' in results and results['temporal']:
            temporal = results['temporal']
            
            report_lines.append("⏰ TEMPORAL VALIDATION RESULTS")
            report_lines.append("-" * 40)
            report_lines.append(f"Temporal splits: {temporal['n_splits']}")
            report_lines.append(f"Temporal stability: {temporal['temporal_stability']:.4f}")
            report_lines.append(f"Drift detected: {'⚠️  YES' if temporal['drift_detected'] else '✅ NO'}")
            
            if temporal['performance_trend']:
                trend = temporal['performance_trend']
                report_lines.append(f"Performance trend: {trend[0]:.4f} → {trend[-1]:.4f}")
            report_lines.append("")
        
        # Summary and recommendations
        report_lines.append("📋 SUMMARY AND RECOMMENDATIONS")
        report_lines.append("-" * 40)
        
        # Extract key metrics for summary
        fpr_values = []
        if 'cross_validation' in results and results['cross_validation']:
            fpr_values.append(results['cross_validation']['mean_metrics']['fpr'])
        if 'bootstrap' in results and results['bootstrap']:
            fpr_values.append(results['bootstrap']['mean_metrics']['fpr'])
        
        if fpr_values:
            avg_fpr = np.mean(fpr_values)
            if avg_fpr < 0.05:
                report_lines.append("✅ FPR Performance: EXCELLENT (< 5%)")
            elif avg_fpr < 0.10:
                report_lines.append("⚠️  FPR Performance: ACCEPTABLE (5-10%)")
            else:
                report_lines.append("❌ FPR Performance: NEEDS IMPROVEMENT (> 10%)")
        
        # Stress test summary
        if 'synthetic_anomaly' in results and results['synthetic_anomaly']:
            if results['synthetic_anomaly']['stress_test_passed']:
                report_lines.append("✅ Stress Test: PASSED")
            else:
                report_lines.append("❌ Stress Test: FAILED")
        
        # Temporal stability summary
        if 'temporal' in results and results['temporal']:
            stability = results['temporal']['temporal_stability']
            if stability > 0.8:
                report_lines.append("✅ Temporal Stability: HIGH")
            elif stability > 0.6:
                report_lines.append("⚠️  Temporal Stability: MODERATE")
            else:
                report_lines.append("❌ Temporal Stability: LOW")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        # Save report if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                f.write(report_text)
            
            logger.info(f"Validation report saved to {output_path}")
        
        return report_text


# Example usage and testing functions
def create_sample_data(n_samples: int = 1000, n_features: int = 10, anomaly_rate: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """Create sample data for testing"""
    np.random.seed(42)
    
    # Generate normal samples
    n_normal = int(n_samples * (1 - anomaly_rate))
    n_anomalies = n_samples - n_normal
    
    # Normal samples from multivariate normal
    X_normal = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features),
        size=n_normal
    )
    
    # Anomalous samples (shifted and scaled)
    X_anomaly = np.random.multivariate_normal(
        mean=np.ones(n_features) * 3,
        cov=np.eye(n_features) * 2,
        size=n_anomalies
    )
    
    # Combine data
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import IsolationForest
    
    # Create sample data
    X, y = create_sample_data(n_samples=1000, n_features=10, anomaly_rate=0.1)
    
    # Create model
    model = IsolationForest(contamination=0.1, random_state=42)
    
    # Create validation framework
    config = ValidationConfig(
        cv_folds=5,
        bootstrap_samples=50,  # Reduced for faster testing
        injection_rates=[0.05, 0.1, 0.2],
        temporal_splits=3,
        verbose=True
    )
    
    validator = RobustValidationFramework(config)
    
    # Run comprehensive validation
    results = validator.comprehensive_validation(X, y, model)
    
    # Generate report
    report = validator.generate_validation_report(results)
    print(report)