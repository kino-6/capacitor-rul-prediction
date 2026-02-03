#!/usr/bin/env python3
"""
Test script for Adaptive Threshold Optimizer

This script tests the adaptive threshold optimization functionality
including Bayesian optimization, cross-validation, and online learning.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import logging
import time
from typing import Tuple, Dict, Any

from true_rul.adaptive_threshold_optimizer import (
    AdaptiveThresholdOptimizer,
    ThresholdOptimizationConfig,
    create_adaptive_threshold_optimizer
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_anomaly_data(
    n_samples: int = 1000,
    n_features: int = 20,
    contamination: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic data with anomalies."""
    logger.info(f"Generating synthetic data: {n_samples} samples, {n_features} features, {contamination:.1%} contamination")
    
    # Generate normal data
    X_normal, _ = make_classification(
        n_samples=int(n_samples * (1 - contamination)),
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=random_state
    )
    
    # Generate anomalous data (shifted distribution)
    X_anomaly, _ = make_classification(
        n_samples=int(n_samples * contamination),
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=random_state + 1
    )
    
    # Shift anomalies to make them more distinct
    X_anomaly = X_anomaly + np.random.normal(3, 1, X_anomaly.shape)
    
    # Combine data
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_anomaly))])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    # Train anomaly detector
    detector = IsolationForest(contamination=contamination, random_state=random_state)
    detector.fit(X_normal)  # Train only on normal data
    
    # Get anomaly scores
    anomaly_scores = -detector.decision_function(X)  # Negative for higher = more anomalous
    
    logger.info(f"Data generated: {len(X)} samples, {np.sum(y)} anomalies ({np.mean(y):.1%})")
    
    return X, y, anomaly_scores


def test_basic_threshold_optimization():
    """Test basic threshold optimization functionality."""
    logger.info("=== Testing Basic Threshold Optimization ===")
    
    # Generate test data
    X, y, scores = generate_synthetic_anomaly_data(n_samples=500, contamination=0.15)
    
    # Create optimizer with basic configuration
    config = ThresholdOptimizationConfig(
        n_trials=50,
        optimization_timeout=60,
        primary_metric="f1_score",
        target_fpr=0.05
    )
    optimizer = AdaptiveThresholdOptimizer(config)
    
    # Optimize threshold
    result = optimizer.optimize_threshold(scores, y)
    
    # Validate results
    assert result.optimal_threshold is not None
    assert 0 < result.optimal_threshold < 1
    assert result.best_score > 0
    assert result.performance_metrics.fpr <= 0.1  # Should be reasonable
    
    logger.info(f"✓ Optimal threshold: {result.optimal_threshold:.4f}")
    logger.info(f"✓ Best F1 score: {result.best_score:.4f}")
    logger.info(f"✓ FPR: {result.performance_metrics.fpr:.4f}")
    logger.info(f"✓ Precision: {result.performance_metrics.precision:.4f}")
    logger.info(f"✓ Recall: {result.performance_metrics.recall:.4f}")
    
    return optimizer, result


def test_cross_validation_optimization():
    """Test threshold optimization with cross-validation."""
    logger.info("=== Testing Cross-Validation Optimization ===")
    
    # Generate test data
    X, y, scores = generate_synthetic_anomaly_data(n_samples=800, contamination=0.12)
    
    # Create detector for CV
    detector = IsolationForest(contamination=0.12, random_state=42)
    detector.fit(X[y == 0])  # Train on normal samples
    
    def detector_predict_fn(X_test):
        return -detector.decision_function(X_test)
    
    # Create optimizer with CV
    config = ThresholdOptimizationConfig(
        n_trials=30,
        cv_folds=3,
        primary_metric="f1_score",
        target_fpr=0.08
    )
    optimizer = AdaptiveThresholdOptimizer(config)
    
    # Optimize with CV
    result = optimizer.optimize_threshold(scores, y, detector_predict_fn)
    
    # Validate CV results
    assert 'f1_score_mean' in result.cv_scores
    assert 'fpr_mean' in result.cv_scores
    assert result.cv_scores['f1_score_mean'] > 0
    
    logger.info(f"✓ CV F1 Score: {result.cv_scores['f1_score_mean']:.4f} ± {result.cv_scores['f1_score_std']:.4f}")
    logger.info(f"✓ CV FPR: {result.cv_scores['fpr_mean']:.4f} ± {result.cv_scores['fpr_std']:.4f}")
    
    return optimizer, result


def test_online_learning():
    """Test online threshold adaptation."""
    logger.info("=== Testing Online Learning ===")
    
    # Generate initial training data
    X_train, y_train, scores_train = generate_synthetic_anomaly_data(
        n_samples=600, contamination=0.1, random_state=42
    )
    
    # Create optimizer with online learning
    config = ThresholdOptimizationConfig(
        n_trials=30,
        online_learning=True,
        history_window=200,
        adaptation_rate=0.05,
        min_samples_for_update=30
    )
    optimizer = AdaptiveThresholdOptimizer(config)
    
    # Initial optimization
    initial_result = optimizer.optimize_threshold(scores_train, y_train)
    initial_threshold = initial_result.optimal_threshold
    
    logger.info(f"Initial threshold: {initial_threshold:.4f}")
    
    # Simulate online data stream with concept drift
    n_batches = 5
    batch_size = 50
    
    for batch_idx in range(n_batches):
        # Generate new batch with increasing contamination (concept drift)
        contamination = 0.1 + batch_idx * 0.02
        X_new, y_new, scores_new = generate_synthetic_anomaly_data(
            n_samples=batch_size,
            contamination=contamination,
            random_state=42 + batch_idx + 10
        )
        
        # Update threshold online
        updated_threshold = optimizer.update_threshold_online(scores_new, y_new)
        
        if updated_threshold is not None:
            logger.info(f"Batch {batch_idx + 1}: Threshold updated to {updated_threshold:.4f}")
        else:
            logger.info(f"Batch {batch_idx + 1}: No threshold update")
        
        # Track performance
        optimizer.track_performance(optimizer.current_threshold, scores_new, y_new)
    
    # Check if threshold adapted
    final_threshold = optimizer.current_threshold
    logger.info(f"Final threshold: {final_threshold:.4f}")
    
    # Get performance summary
    performance_summary = optimizer.get_performance_summary()
    if performance_summary:
        logger.info(f"✓ Average FPR: {performance_summary['fpr']['mean']:.4f}")
        logger.info(f"✓ Average F1: {performance_summary['f1_score']['mean']:.4f}")
    
    return optimizer


def test_threshold_recommendations():
    """Test threshold recommendation functionality."""
    logger.info("=== Testing Threshold Recommendations ===")
    
    # Generate test data
    X, y, scores = generate_synthetic_anomaly_data(n_samples=400, contamination=0.2)
    
    # Create optimizer
    optimizer = create_adaptive_threshold_optimizer(target_fpr=0.05)
    
    # Get recommendations
    recommendations = optimizer.get_threshold_recommendations(scores, y)
    
    logger.info("Threshold recommendations:")
    for objective, threshold in recommendations.items():
        logger.info(f"  {objective}: {threshold:.4f}")
    
    # Validate recommendations
    assert len(recommendations) > 0
    assert all(0 < t < 1 for t in recommendations.values())
    
    return recommendations


def test_optimizer_persistence():
    """Test saving and loading optimizer state."""
    logger.info("=== Testing Optimizer Persistence ===")
    
    # Generate test data
    X, y, scores = generate_synthetic_anomaly_data(n_samples=300, contamination=0.15)
    
    # Create and train optimizer
    config = ThresholdOptimizationConfig(n_trials=20)
    optimizer1 = AdaptiveThresholdOptimizer(config)
    result1 = optimizer1.optimize_threshold(scores, y)
    
    # Save optimizer
    save_path = "/tmp/test_optimizer.json"
    optimizer1.save_optimizer(save_path)
    
    # Load optimizer
    optimizer2 = AdaptiveThresholdOptimizer(config)
    optimizer2.load_optimizer(save_path)
    
    # Validate loaded state
    assert optimizer2.current_threshold == optimizer1.current_threshold
    assert optimizer2.is_fitted == optimizer1.is_fitted
    
    logger.info(f"✓ Optimizer saved and loaded successfully")
    logger.info(f"✓ Threshold preserved: {optimizer2.current_threshold:.4f}")
    
    # Clean up
    os.remove(save_path)
    
    return optimizer2


def test_multi_objective_optimization():
    """Test multi-objective optimization with different metrics."""
    logger.info("=== Testing Multi-Objective Optimization ===")
    
    # Generate test data
    X, y, scores = generate_synthetic_anomaly_data(n_samples=600, contamination=0.08)
    
    metrics = ["f1_score", "precision", "recall", "fpr"]
    results = {}
    
    for metric in metrics:
        logger.info(f"Optimizing for {metric}...")
        
        config = ThresholdOptimizationConfig(
            n_trials=25,
            primary_metric=metric,
            target_fpr=0.05,
            min_precision=0.7,
            min_recall=0.7
        )
        optimizer = AdaptiveThresholdOptimizer(config)
        result = optimizer.optimize_threshold(scores, y)
        
        results[metric] = {
            'threshold': result.optimal_threshold,
            'fpr': result.performance_metrics.fpr,
            'precision': result.performance_metrics.precision,
            'recall': result.performance_metrics.recall,
            'f1_score': result.performance_metrics.f1_score
        }
        
        logger.info(f"  Threshold: {result.optimal_threshold:.4f}")
        logger.info(f"  FPR: {result.performance_metrics.fpr:.4f}")
        logger.info(f"  F1: {result.performance_metrics.f1_score:.4f}")
    
    # Validate that different metrics produce different thresholds
    thresholds = [results[m]['threshold'] for m in metrics]
    assert len(set(thresholds)) > 1, "Different metrics should produce different thresholds"
    
    return results


def visualize_optimization_results(optimizer, result, scores, labels):
    """Visualize optimization results."""
    logger.info("Creating optimization visualizations...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Optimization history
        if result.optimization_history:
            history = result.optimization_history
            trials = [h['trial'] for h in history]
            scores_hist = [h['score'] for h in history]
            
            axes[0, 0].plot(trials, scores_hist, 'b-', alpha=0.7)
            axes[0, 0].axhline(y=result.best_score, color='r', linestyle='--', label=f'Best: {result.best_score:.3f}')
            axes[0, 0].set_xlabel('Trial')
            axes[0, 0].set_ylabel('Objective Score')
            axes[0, 0].set_title('Optimization History')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Threshold vs FPR
        if result.optimization_history:
            thresholds = [h['threshold'] for h in history]
            fprs = [h['fpr'] for h in history]
            
            axes[0, 1].scatter(thresholds, fprs, alpha=0.6, c='blue')
            axes[0, 1].axhline(y=0.05, color='r', linestyle='--', label='Target FPR (5%)')
            axes[0, 1].axvline(x=result.optimal_threshold, color='g', linestyle='--', label=f'Optimal: {result.optimal_threshold:.3f}')
            axes[0, 1].set_xlabel('Threshold')
            axes[0, 1].set_ylabel('False Positive Rate')
            axes[0, 1].set_title('Threshold vs FPR')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Score distribution
        axes[1, 0].hist(scores[labels == 0], bins=30, alpha=0.7, label='Normal', color='blue')
        axes[1, 0].hist(scores[labels == 1], bins=30, alpha=0.7, label='Anomaly', color='red')
        axes[1, 0].axvline(x=result.optimal_threshold, color='g', linestyle='--', linewidth=2, label=f'Threshold: {result.optimal_threshold:.3f}')
        axes[1, 0].set_xlabel('Anomaly Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Score Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Performance metrics
        if result.optimization_history:
            f1_scores = [h['f1_score'] for h in history]
            precisions = [h['precision'] for h in history]
            recalls = [h['recall'] for h in history]
            
            axes[1, 1].plot(trials, f1_scores, 'b-', label='F1 Score', alpha=0.7)
            axes[1, 1].plot(trials, precisions, 'g-', label='Precision', alpha=0.7)
            axes[1, 1].plot(trials, recalls, 'r-', label='Recall', alpha=0.7)
            axes[1, 1].set_xlabel('Trial')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_title('Performance Metrics')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/tmp/threshold_optimization_results.png', dpi=150, bbox_inches='tight')
        logger.info("✓ Visualization saved to /tmp/threshold_optimization_results.png")
        
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")


def run_comprehensive_test():
    """Run comprehensive test suite."""
    logger.info("🚀 Starting Adaptive Threshold Optimizer Comprehensive Test")
    logger.info("=" * 60)
    
    start_time = time.time()
    test_results = {}
    
    try:
        # Test 1: Basic optimization
        optimizer1, result1 = test_basic_threshold_optimization()
        test_results['basic_optimization'] = 'PASSED'
        
        # Test 2: Cross-validation
        optimizer2, result2 = test_cross_validation_optimization()
        test_results['cross_validation'] = 'PASSED'
        
        # Test 3: Online learning
        optimizer3 = test_online_learning()
        test_results['online_learning'] = 'PASSED'
        
        # Test 4: Threshold recommendations
        recommendations = test_threshold_recommendations()
        test_results['recommendations'] = 'PASSED'
        
        # Test 5: Persistence
        optimizer5 = test_optimizer_persistence()
        test_results['persistence'] = 'PASSED'
        
        # Test 6: Multi-objective optimization
        multi_results = test_multi_objective_optimization()
        test_results['multi_objective'] = 'PASSED'
        
        # Generate visualization
        X, y, scores = generate_synthetic_anomaly_data(n_samples=500, contamination=0.15)
        config = ThresholdOptimizationConfig(n_trials=30)
        viz_optimizer = AdaptiveThresholdOptimizer(config)
        viz_result = viz_optimizer.optimize_threshold(scores, y)
        visualize_optimization_results(viz_optimizer, viz_result, scores, y)
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        test_results['error'] = str(e)
    
    # Summary
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("🎯 TEST SUMMARY")
    logger.info("=" * 60)
    
    for test_name, status in test_results.items():
        status_icon = "✅" if status == "PASSED" else "❌"
        logger.info(f"{status_icon} {test_name}: {status}")
    
    passed_tests = sum(1 for status in test_results.values() if status == "PASSED")
    total_tests = len(test_results)
    
    logger.info(f"\n📊 Results: {passed_tests}/{total_tests} tests passed")
    logger.info(f"⏱️  Total time: {total_time:.2f} seconds")
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED! Adaptive Threshold Optimizer is working correctly.")
        return True
    else:
        logger.error("❌ Some tests failed. Please check the logs above.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)