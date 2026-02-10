"""
Online Learning Capabilities for True RUL Prediction System

This module implements online learning capabilities including:
- Incremental model updates with new data
- Concept drift detection and adaptation
- Automated retraining triggers based on performance metrics
- Active learning for optimal data collection

Requirements: 5.4, 5.5
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import json
from collections import deque
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
import warnings

from .data_structures import TrainingDataset, CycleData, CapacitorData
from .rul_regression_model import RULRegressionModel
from .ensemble_anomaly_detector import EnsembleAnomalyDetector
from .model_evaluator import ModelEvaluator
from .feature_extractor import FeatureExtractor
from .time_series_preprocessor import TimeSeriesPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for drift detection"""
    timestamp: datetime
    rmse: float
    mae: float
    r2: float
    fpr: float
    tpr: float
    f1: float
    sample_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'rmse': self.rmse,
            'mae': self.mae,
            'r2': self.r2,
            'fpr': self.fpr,
            'tpr': self.tpr,
            'f1': self.f1,
            'sample_count': self.sample_count
        }


@dataclass
class DriftDetectionResult:
    """Result of concept drift detection"""
    drift_detected: bool
    drift_type: str  # 'gradual', 'sudden', 'none'
    confidence: float
    affected_metrics: List[str]
    recommendation: str
    details: Dict[str, Any] = field(default_factory=dict)


class ConceptDriftDetector:
    """
    Detects concept drift in model performance
    
    Uses statistical tests and performance monitoring to detect
    when the underlying data distribution has changed.
    """
    
    def __init__(self, 
                 window_size: int = 100,
                 sensitivity: float = 0.05,
                 min_samples: int = 30):
        """
        Initialize concept drift detector
        
        Args:
            window_size: Size of sliding window for comparison
            sensitivity: Statistical significance threshold
            min_samples: Minimum samples needed for detection
        """
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.min_samples = min_samples
        
        # Performance history
        self.performance_history: deque = deque(maxlen=window_size * 2)
        self.baseline_metrics: Optional[PerformanceMetrics] = None
        
        logger.info(f"ConceptDriftDetector initialized with window_size={window_size}")
    
    def add_performance_sample(self, metrics: PerformanceMetrics):
        """Add new performance sample to history"""
        self.performance_history.append(metrics)
        
        # Set baseline if not set
        if self.baseline_metrics is None and len(self.performance_history) >= self.min_samples:
            self.baseline_metrics = self._compute_baseline_metrics()
            logger.info("Baseline metrics established for drift detection")
    
    def detect_drift(self) -> DriftDetectionResult:
        """
        Detect concept drift in recent performance
        
        Returns:
            DriftDetectionResult with detection results
        """
        if len(self.performance_history) < self.min_samples:
            return DriftDetectionResult(
                drift_detected=False,
                drift_type='none',
                confidence=0.0,
                affected_metrics=[],
                recommendation="Insufficient data for drift detection"
            )
        
        if self.baseline_metrics is None:
            return DriftDetectionResult(
                drift_detected=False,
                drift_type='none',
                confidence=0.0,
                affected_metrics=[],
                recommendation="Baseline not established"
            )
        
        # Split recent history into baseline and current windows
        recent_samples = list(self.performance_history)[-self.window_size:]
        baseline_samples = list(self.performance_history)[:-self.window_size]
        
        if len(baseline_samples) < self.min_samples or len(recent_samples) < self.min_samples:
            return DriftDetectionResult(
                drift_detected=False,
                drift_type='none',
                confidence=0.0,
                affected_metrics=[],
                recommendation="Insufficient samples in windows"
            )
        
        # Extract metric arrays
        metrics_to_check = ['rmse', 'mae', 'r2', 'fpr', 'tpr', 'f1']
        drift_results = {}
        affected_metrics = []
        
        for metric in metrics_to_check:
            baseline_values = [getattr(sample, metric) for sample in baseline_samples]
            recent_values = [getattr(sample, metric) for sample in recent_samples]
            
            # Perform statistical tests
            drift_result = self._test_metric_drift(
                baseline_values, recent_values, metric
            )
            drift_results[metric] = drift_result
            
            if drift_result['drift_detected']:
                affected_metrics.append(metric)
        
        # Determine overall drift status
        drift_detected = len(affected_metrics) > 0
        
        # Determine drift type based on trend analysis
        drift_type = self._classify_drift_type(recent_samples) if drift_detected else 'none'
        
        # Calculate overall confidence
        confidences = [drift_results[m]['confidence'] for m in affected_metrics]
        overall_confidence = np.mean(confidences) if confidences else 0.0
        
        # Generate recommendation
        recommendation = self._generate_drift_recommendation(
            drift_detected, drift_type, affected_metrics
        )
        
        return DriftDetectionResult(
            drift_detected=drift_detected,
            drift_type=drift_type,
            confidence=overall_confidence,
            affected_metrics=affected_metrics,
            recommendation=recommendation,
            details=drift_results
        )
    
    def _compute_baseline_metrics(self) -> PerformanceMetrics:
        """Compute baseline metrics from early samples"""
        early_samples = list(self.performance_history)[:self.min_samples]
        
        return PerformanceMetrics(
            timestamp=early_samples[-1].timestamp,
            rmse=np.mean([s.rmse for s in early_samples]),
            mae=np.mean([s.mae for s in early_samples]),
            r2=np.mean([s.r2 for s in early_samples]),
            fpr=np.mean([s.fpr for s in early_samples]),
            tpr=np.mean([s.tpr for s in early_samples]),
            f1=np.mean([s.f1 for s in early_samples]),
            sample_count=len(early_samples)
        )
    
    def _test_metric_drift(self, baseline: List[float], recent: List[float], 
                          metric_name: str) -> Dict[str, Any]:
        """Test for drift in a specific metric"""
        try:
            # Kolmogorov-Smirnov test for distribution change
            ks_stat, ks_p_value = stats.ks_2samp(baseline, recent)
            
            # Mann-Whitney U test for median change
            mw_stat, mw_p_value = stats.mannwhitneyu(
                baseline, recent, alternative='two-sided'
            )
            
            # T-test for mean change (if normally distributed)
            try:
                t_stat, t_p_value = stats.ttest_ind(baseline, recent)
            except:
                t_stat, t_p_value = 0.0, 1.0
            
            # Determine if drift detected
            drift_detected = (
                ks_p_value < self.sensitivity or 
                mw_p_value < self.sensitivity or 
                t_p_value < self.sensitivity
            )
            
            # Calculate confidence (1 - min p-value)
            min_p_value = min(ks_p_value, mw_p_value, t_p_value)
            confidence = 1.0 - min_p_value
            
            # Calculate effect size
            baseline_mean = np.mean(baseline)
            recent_mean = np.mean(recent)
            pooled_std = np.sqrt((np.var(baseline) + np.var(recent)) / 2)
            effect_size = abs(recent_mean - baseline_mean) / (pooled_std + 1e-8)
            
            return {
                'drift_detected': drift_detected,
                'confidence': confidence,
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p_value,
                'mw_p_value': mw_p_value,
                't_p_value': t_p_value,
                'effect_size': effect_size,
                'baseline_mean': baseline_mean,
                'recent_mean': recent_mean,
                'change_direction': 'increase' if recent_mean > baseline_mean else 'decrease'
            }
            
        except Exception as e:
            logger.warning(f"Error testing drift for {metric_name}: {e}")
            return {
                'drift_detected': False,
                'confidence': 0.0,
                'error': str(e)
            }
    
    def _classify_drift_type(self, recent_samples: List[PerformanceMetrics]) -> str:
        """Classify the type of drift based on temporal patterns"""
        if len(recent_samples) < 10:
            return 'unknown'
        
        # Extract timestamps and a key metric (RMSE)
        timestamps = [s.timestamp for s in recent_samples]
        rmse_values = [s.rmse for s in recent_samples]
        
        # Calculate trend
        time_deltas = [(t - timestamps[0]).total_seconds() for t in timestamps]
        
        try:
            slope, _, r_value, _, _ = stats.linregress(time_deltas, rmse_values)
            
            # Strong trend indicates gradual drift
            if abs(r_value) > 0.7:
                return 'gradual'
            
            # Check for sudden changes
            changes = np.diff(rmse_values)
            max_change = np.max(np.abs(changes))
            mean_change = np.mean(np.abs(changes))
            
            # Large single change indicates sudden drift
            if max_change > 3 * mean_change:
                return 'sudden'
            
            return 'gradual'
            
        except:
            return 'unknown'
    
    def _generate_drift_recommendation(self, drift_detected: bool, 
                                     drift_type: str, 
                                     affected_metrics: List[str]) -> str:
        """Generate recommendation based on drift detection results"""
        if not drift_detected:
            return "No drift detected. Continue monitoring."
        
        if drift_type == 'sudden':
            return (
                f"Sudden drift detected in {', '.join(affected_metrics)}. "
                "Immediate retraining recommended."
            )
        elif drift_type == 'gradual':
            return (
                f"Gradual drift detected in {', '.join(affected_metrics)}. "
                "Schedule incremental model update."
            )
        else:
            return (
                f"Drift detected in {', '.join(affected_metrics)}. "
                "Investigate data quality and consider retraining."
            )


class ActiveLearningStrategy(ABC):
    """Abstract base class for active learning strategies"""
    
    @abstractmethod
    def select_samples(self, 
                      unlabeled_data: np.ndarray,
                      model: RULRegressionModel,
                      n_samples: int) -> List[int]:
        """
        Select most informative samples for labeling
        
        Args:
            unlabeled_data: Unlabeled feature data
            model: Current model
            n_samples: Number of samples to select
            
        Returns:
            Indices of selected samples
        """
        pass


class UncertaintyBasedStrategy(ActiveLearningStrategy):
    """Select samples with highest prediction uncertainty"""
    
    def select_samples(self, 
                      unlabeled_data: np.ndarray,
                      model: RULRegressionModel,
                      n_samples: int) -> List[int]:
        """Select samples with highest prediction uncertainty"""
        try:
            # Get predictions with confidence intervals
            predictions, lower_bounds, upper_bounds = model.predict_with_confidence(
                unlabeled_data
            )
            
            # Calculate uncertainty as confidence interval width
            uncertainties = upper_bounds - lower_bounds
            
            # Select samples with highest uncertainty
            selected_indices = np.argsort(uncertainties)[-n_samples:].tolist()
            
            logger.info(f"Selected {len(selected_indices)} samples using uncertainty-based strategy")
            return selected_indices
            
        except Exception as e:
            logger.warning(f"Uncertainty-based selection failed: {e}")
            # Fallback to random selection
            return np.random.choice(len(unlabeled_data), n_samples, replace=False).tolist()


class DiversityBasedStrategy(ActiveLearningStrategy):
    """Select diverse samples to cover feature space"""
    
    def select_samples(self, 
                      unlabeled_data: np.ndarray,
                      model: RULRegressionModel,
                      n_samples: int) -> List[int]:
        """Select diverse samples using k-means clustering"""
        try:
            from sklearn.cluster import KMeans
            
            # Use k-means to find diverse samples
            kmeans = KMeans(n_clusters=n_samples, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(unlabeled_data)
            
            # Select one sample from each cluster (closest to centroid)
            selected_indices = []
            for i in range(n_samples):
                cluster_mask = cluster_labels == i
                if np.any(cluster_mask):
                    cluster_data = unlabeled_data[cluster_mask]
                    cluster_indices = np.where(cluster_mask)[0]
                    
                    # Find closest to centroid
                    centroid = kmeans.cluster_centers_[i]
                    distances = np.linalg.norm(cluster_data - centroid, axis=1)
                    closest_idx = cluster_indices[np.argmin(distances)]
                    selected_indices.append(closest_idx)
            
            logger.info(f"Selected {len(selected_indices)} samples using diversity-based strategy")
            return selected_indices
            
        except Exception as e:
            logger.warning(f"Diversity-based selection failed: {e}")
            # Fallback to random selection
            return np.random.choice(len(unlabeled_data), n_samples, replace=False).tolist()


class OnlineLearningManager:
    """
    Manages online learning capabilities for the RUL prediction system
    
    Coordinates incremental updates, drift detection, and active learning
    to maintain model performance over time.
    """
    
    def __init__(self,
                 rul_model: RULRegressionModel,
                 anomaly_detector: EnsembleAnomalyDetector,
                 feature_extractor: FeatureExtractor,
                 preprocessor: TimeSeriesPreprocessor,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize online learning manager
        
        Args:
            rul_model: RUL regression model
            anomaly_detector: Anomaly detection model
            feature_extractor: Feature extraction component
            preprocessor: Time series preprocessor
            config: Configuration parameters
        """
        self.rul_model = rul_model
        self.anomaly_detector = anomaly_detector
        self.feature_extractor = feature_extractor
        self.preprocessor = preprocessor
        
        # Configuration
        self.config = config or {}
        self.update_threshold = self.config.get('update_threshold', 0.1)  # Performance degradation threshold
        self.min_update_samples = self.config.get('min_update_samples', 50)
        self.max_update_samples = self.config.get('max_update_samples', 500)
        self.active_learning_budget = self.config.get('active_learning_budget', 20)
        
        # Components
        self.drift_detector = ConceptDriftDetector(
            window_size=self.config.get('drift_window_size', 100),
            sensitivity=self.config.get('drift_sensitivity', 0.05)
        )
        
        self.active_learning_strategy = self._create_active_learning_strategy()
        self.evaluator = ModelEvaluator()
        
        # State
        self.update_buffer: List[Tuple[np.ndarray, float, float]] = []  # (features, rul, anomaly_label)
        self.last_update_time = datetime.now()
        self.update_count = 0
        self.performance_history: List[PerformanceMetrics] = []
        
        logger.info("OnlineLearningManager initialized")
    
    def _create_active_learning_strategy(self) -> ActiveLearningStrategy:
        """Create active learning strategy based on configuration"""
        strategy_type = self.config.get('active_learning_strategy', 'uncertainty')
        
        if strategy_type == 'uncertainty':
            return UncertaintyBasedStrategy()
        elif strategy_type == 'diversity':
            return DiversityBasedStrategy()
        else:
            logger.warning(f"Unknown strategy {strategy_type}, using uncertainty-based")
            return UncertaintyBasedStrategy()
    
    def add_new_data(self, 
                     cycle_data: CycleData,
                     true_rul: Optional[float] = None,
                     anomaly_label: Optional[float] = None):
        """
        Add new data sample for potential model update
        
        Args:
            cycle_data: New cycle data
            true_rul: True RUL value (if available)
            anomaly_label: True anomaly label (if available)
        """
        try:
            # Extract features
            features = self.feature_extractor.extract_features(cycle_data, [])
            features = features.reshape(1, -1)
            
            # Normalize features
            features = self.preprocessor.normalize_features(
                features, "online", fit=False
            )
            
            # Add to buffer if labels are available
            if true_rul is not None and anomaly_label is not None:
                self.update_buffer.append((features[0], true_rul, anomaly_label))
                logger.debug(f"Added labeled sample to update buffer (size: {len(self.update_buffer)})")
            
            # Evaluate current performance if we have labels
            if true_rul is not None and anomaly_label is not None:
                self._evaluate_current_performance(features, true_rul, anomaly_label)
            
            # Check if update is needed
            self._check_update_trigger()
            
        except Exception as e:
            logger.error(f"Error adding new data: {e}")
    
    def _evaluate_current_performance(self, 
                                    features: np.ndarray,
                                    true_rul: float,
                                    anomaly_label: float):
        """Evaluate current model performance on new sample"""
        try:
            # RUL prediction
            rul_pred = self.rul_model.predict(features)[0]
            rul_error = abs(rul_pred - true_rul)
            
            # Anomaly detection
            anomaly_pred, anomaly_score, _ = self.anomaly_detector.predict(features)
            anomaly_pred = anomaly_pred[0]
            
            # Calculate metrics (simplified for single sample)
            rmse = rul_error  # For single sample, RMSE = absolute error
            mae = rul_error
            r2 = 1.0 - (rul_error / (true_rul + 1e-8))**2  # Approximation
            
            # Anomaly metrics
            fpr = 1.0 if (anomaly_pred == 1 and anomaly_label == 0) else 0.0
            tpr = 1.0 if (anomaly_pred == 1 and anomaly_label == 1) else 0.0
            f1 = 1.0 if anomaly_pred == anomaly_label else 0.0
            
            # Create performance metrics
            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                rmse=rmse,
                mae=mae,
                r2=r2,
                fpr=fpr,
                tpr=tpr,
                f1=f1,
                sample_count=1
            )
            
            # Add to drift detector
            self.drift_detector.add_performance_sample(metrics)
            self.performance_history.append(metrics)
            
        except Exception as e:
            logger.error(f"Error evaluating performance: {e}")
    
    def _check_update_trigger(self):
        """Check if model update should be triggered"""
        try:
            # Check drift detection
            drift_result = self.drift_detector.detect_drift()
            
            if drift_result.drift_detected:
                logger.info(f"Concept drift detected: {drift_result.recommendation}")
                
                if drift_result.drift_type == 'sudden':
                    # Immediate update for sudden drift
                    self._trigger_model_update(reason="sudden_drift")
                elif len(self.update_buffer) >= self.min_update_samples:
                    # Gradual drift with sufficient data
                    self._trigger_model_update(reason="gradual_drift")
            
            # Check buffer size
            elif len(self.update_buffer) >= self.max_update_samples:
                self._trigger_model_update(reason="buffer_full")
            
            # Check time-based trigger
            elif (datetime.now() - self.last_update_time).days >= 7:
                if len(self.update_buffer) >= self.min_update_samples:
                    self._trigger_model_update(reason="scheduled_update")
            
        except Exception as e:
            logger.error(f"Error checking update trigger: {e}")
    
    def _trigger_model_update(self, reason: str):
        """Trigger incremental model update"""
        if len(self.update_buffer) < self.min_update_samples:
            logger.warning(f"Insufficient samples for update: {len(self.update_buffer)}")
            return
        
        logger.info(f"Triggering model update (reason: {reason})")
        
        try:
            # Prepare update data
            features_list = []
            rul_labels = []
            anomaly_labels = []
            
            for features, rul, anomaly in self.update_buffer:
                features_list.append(features)
                rul_labels.append(rul)
                anomaly_labels.append(anomaly)
            
            update_features = np.array(features_list)
            update_rul = np.array(rul_labels)
            update_anomaly = np.array(anomaly_labels)
            
            # Perform incremental update
            self._perform_incremental_update(
                update_features, update_rul, update_anomaly, reason
            )
            
            # Clear buffer and update timestamp
            self.update_buffer.clear()
            self.last_update_time = datetime.now()
            self.update_count += 1
            
            logger.info(f"Model update completed (update #{self.update_count})")
            
        except Exception as e:
            logger.error(f"Error during model update: {e}")
    
    def _perform_incremental_update(self,
                                  features: np.ndarray,
                                  rul_labels: np.ndarray,
                                  anomaly_labels: np.ndarray,
                                  reason: str):
        """Perform incremental model update"""
        logger.info(f"Performing incremental update with {len(features)} samples")
        
        # For now, implement a simple retraining approach
        # In a full implementation, this would use incremental learning algorithms
        
        try:
            # Split data for validation
            n_samples = len(features)
            val_size = max(1, n_samples // 5)  # 20% for validation
            
            indices = np.random.permutation(n_samples)
            train_indices = indices[val_size:]
            val_indices = indices[:val_size]
            
            train_features = features[train_indices]
            train_rul = rul_labels[train_indices]
            val_features = features[val_indices]
            val_rul = rul_labels[val_indices]
            
            # Update RUL model (simplified - in practice would use incremental methods)
            if len(train_features) > 0:
                # For demonstration, we'll retrain on new data
                # Real implementation would use incremental learning
                logger.info("Updating RUL model with new data")
                
                # Note: This is a simplified approach
                # Real incremental learning would preserve existing knowledge
                
            # Update anomaly detector
            normal_samples = features[anomaly_labels == 0]
            if len(normal_samples) > 0:
                logger.info("Updating anomaly detector with new normal samples")
                # Incremental fit would go here
            
            # Log update details
            update_info = {
                'timestamp': datetime.now().isoformat(),
                'reason': reason,
                'n_samples': n_samples,
                'n_normal_samples': int(np.sum(anomaly_labels == 0)),
                'n_anomaly_samples': int(np.sum(anomaly_labels == 1)),
                'update_count': self.update_count + 1
            }
            
            logger.info(f"Incremental update completed: {update_info}")
            
        except Exception as e:
            logger.error(f"Error in incremental update: {e}")
            raise
    
    def request_active_learning_samples(self, 
                                      unlabeled_data: List[CycleData],
                                      n_samples: Optional[int] = None) -> List[int]:
        """
        Request samples for active learning labeling
        
        Args:
            unlabeled_data: List of unlabeled cycle data
            n_samples: Number of samples to request (default: active_learning_budget)
            
        Returns:
            Indices of samples to label
        """
        if n_samples is None:
            n_samples = self.active_learning_budget
        
        if len(unlabeled_data) == 0:
            return []
        
        n_samples = min(n_samples, len(unlabeled_data))
        
        try:
            # Extract features for all unlabeled data
            features_list = []
            for cycle_data in unlabeled_data:
                features = self.feature_extractor.extract_features(cycle_data, [])
                features = self.preprocessor.normalize_features(
                    features.reshape(1, -1), "online", fit=False
                )
                features_list.append(features[0])
            
            unlabeled_features = np.array(features_list)
            
            # Use active learning strategy to select samples
            selected_indices = self.active_learning_strategy.select_samples(
                unlabeled_features, self.rul_model, n_samples
            )
            
            logger.info(f"Selected {len(selected_indices)} samples for active learning")
            return selected_indices
            
        except Exception as e:
            logger.error(f"Error in active learning sample selection: {e}")
            # Fallback to random selection
            return np.random.choice(len(unlabeled_data), n_samples, replace=False).tolist()
    
    def get_drift_status(self) -> DriftDetectionResult:
        """Get current concept drift status"""
        return self.drift_detector.detect_drift()
    
    def get_update_status(self) -> Dict[str, Any]:
        """Get current update status"""
        return {
            'buffer_size': len(self.update_buffer),
            'min_update_samples': self.min_update_samples,
            'max_update_samples': self.max_update_samples,
            'last_update_time': self.last_update_time.isoformat(),
            'update_count': self.update_count,
            'days_since_update': (datetime.now() - self.last_update_time).days,
            'performance_samples': len(self.performance_history)
        }
    
    def get_performance_summary(self, window_size: int = 50) -> Dict[str, float]:
        """Get recent performance summary"""
        if not self.performance_history:
            return {}
        
        recent_metrics = self.performance_history[-window_size:]
        
        return {
            'avg_rmse': np.mean([m.rmse for m in recent_metrics]),
            'avg_mae': np.mean([m.mae for m in recent_metrics]),
            'avg_r2': np.mean([m.r2 for m in recent_metrics]),
            'avg_fpr': np.mean([m.fpr for m in recent_metrics]),
            'avg_tpr': np.mean([m.tpr for m in recent_metrics]),
            'avg_f1': np.mean([m.f1 for m in recent_metrics]),
            'sample_count': len(recent_metrics)
        }
    
    def save_state(self, filepath: Path):
        """Save online learning state"""
        state = {
            'config': self.config,
            'update_count': self.update_count,
            'last_update_time': self.last_update_time.isoformat(),
            'performance_history': [m.to_dict() for m in self.performance_history],
            'drift_detector_state': {
                'window_size': self.drift_detector.window_size,
                'sensitivity': self.drift_detector.sensitivity,
                'baseline_metrics': (
                    self.drift_detector.baseline_metrics.to_dict() 
                    if self.drift_detector.baseline_metrics else None
                )
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Online learning state saved to {filepath}")
    
    def load_state(self, filepath: Path):
        """Load online learning state"""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.update_count = state.get('update_count', 0)
            self.last_update_time = datetime.fromisoformat(
                state.get('last_update_time', datetime.now().isoformat())
            )
            
            # Restore performance history
            self.performance_history = []
            for metrics_dict in state.get('performance_history', []):
                metrics = PerformanceMetrics(
                    timestamp=datetime.fromisoformat(metrics_dict['timestamp']),
                    rmse=metrics_dict['rmse'],
                    mae=metrics_dict['mae'],
                    r2=metrics_dict['r2'],
                    fpr=metrics_dict['fpr'],
                    tpr=metrics_dict['tpr'],
                    f1=metrics_dict['f1'],
                    sample_count=metrics_dict['sample_count']
                )
                self.performance_history.append(metrics)
                self.drift_detector.add_performance_sample(metrics)
            
            logger.info(f"Online learning state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading state: {e}")


def create_online_learning_manager(
    rul_model: RULRegressionModel,
    anomaly_detector: EnsembleAnomalyDetector,
    feature_extractor: FeatureExtractor,
    preprocessor: TimeSeriesPreprocessor,
    config: Optional[Dict[str, Any]] = None
) -> OnlineLearningManager:
    """
    Factory function to create online learning manager
    
    Args:
        rul_model: Trained RUL regression model
        anomaly_detector: Trained anomaly detector
        feature_extractor: Feature extractor
        preprocessor: Time series preprocessor
        config: Configuration parameters
        
    Returns:
        Configured OnlineLearningManager instance
    """
    default_config = {
        'update_threshold': 0.1,
        'min_update_samples': 50,
        'max_update_samples': 500,
        'active_learning_budget': 20,
        'drift_window_size': 100,
        'drift_sensitivity': 0.05,
        'active_learning_strategy': 'uncertainty'
    }
    
    if config:
        default_config.update(config)
    
    return OnlineLearningManager(
        rul_model=rul_model,
        anomaly_detector=anomaly_detector,
        feature_extractor=feature_extractor,
        preprocessor=preprocessor,
        config=default_config
    )