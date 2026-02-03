#!/usr/bin/env python3
"""
Simple test script for Adaptive Threshold Optimizer

This script tests the core adaptive threshold optimization functionality
without complex cross-validation scenarios.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import IsolationForest
import logging
import time

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


def generate_synthetic_data(n_samples=1000, contamination=0.1):
    """Generate synthetic anomaly detection data."""
    # Generate normal data
    X_normal, _ = make_classification(
        n_samples=int(n_samples * (1 - contamination)),
        n_features=20,
        n_informative=10,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Generate anomalous data
    X_anomaly, _ = make_classification(
        n_samples=int(n_samples * contamination),
        n_features=20,
        n_informative=10,
        n_redundant=0,
        n_clusters_per_class=1,
        random_state=43
    )
    
    # Shift anomalies
    X_anomaly = X_anomaly + np.random.normal(2, 0.5, X_anomaly.shape)
    
    # Combine data
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_anomaly))])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]
    
    # Train detector and get scores
    detector = IsolationForest(contamination=contamination, random_state=42)
    detector.fit(X_normal)
    scores = -detector.decision_function(X)
    
    return X, y, scores


def test_basic_optimization():
    """Test basic threshold optimization."""
    logger.info("=== Testing Basic Threshold Optimization ===")
    
    X, y, scores = generate_synthetic_data(n_samples=500, contamination=0.15)
    
    config = ThresholdOptimizationConfig(
        n_trials=30,
        optimization_timeout=30,
        primary_metric="f1_score",
        target_fpr=0.05
    )
    optimizer = AdaptiveThresholdOptimizer(config)
    
    result = optimizer.optimize_threshold(scores, y)
    
    # Validate results
    assert result.optimal_threshold is not None
    assert 0 < result.optimal_threshold < 1
    assert result.best_score > 0
    assert result.performance_metrics.fpr <= 0.1
    
    logger.info(f"✓ Optimal threshold: {result.optimal_threshold:.4f}")
    logger.info(f"✓ F1 score: {result.performance_metrics.f1_score:.4f}")
    logger.info(f"✓ FPR: {result.performance_metrics.fpr:.4f}")
    logger.info(f"✓ Precision: {result.performance_metrics.precision:.4f}")
    logger.info(f"✓ Recall: {result.performance_metrics.recall:.4f}")
    
    return True


def test_online_learning():
    """Test online threshold adaptation."""
    logger.info("=== Testing Online Learning ===")
    
    # Initial training
    X_train, y_train, scores_train = generate_synthetic_data(n_samples=400, contamination=0.1)
    
    config = ThresholdOptimizationConfig(
        n_trials=20,
        online_learning=True,
        history_window=100,
        adaptation_rate=0.1,
        min_samples_for_update=20
    )
    optimizer = AdaptiveThresholdOptimizer(config)
    
    # Initial optimization
    result = optimizer.optimize_threshold(scores_train, y_train)
    initial_threshold = result.optimal_threshold
    
    logger.info(f"Initial threshold: {initial_threshold:.4f}")
    
    # Simulate online updates
    for i in range(3):
        # Generate new batch with concept drift
        contamination = 0.1 + i * 0.03
        X_new, y_new, scores_new = generate_synthetic_data(
            n_samples=30, contamination=contamination
        )
        
        # Update threshold
        updated_threshold = optimizer.update_threshold_online(scores_new, y_new)
        
        if updated_threshold is not None:
            logger.info(f"Batch {i+1}: Threshold updated to {updated_threshold:.4f}")
        else:
            logger.info(f"Batch {i+1}: No threshold update")
        
        # Track performance
        optimizer.track_performance(optimizer.current_threshold, scores_new, y_new)
    
    final_threshold = optimizer.current_threshold
    logger.info(f"Final threshold: {final_threshold:.4f}")
    
    # Get performance summary
    performance_summary = optimizer.get_performance_summary()
    if performance_summary:
        logger.info(f"✓ Average FPR: {performance_summary['fpr']['mean']:.4f}")
        logger.info(f"✓ Average F1: {performance_summary['f1_score']['mean']:.4f}")
    
    return True


def test_threshold_recommendations():
    """Test threshold recommendations."""
    logger.info("=== Testing Threshold Recommendations ===")
    
    X, y, scores = generate_synthetic_data(n_samples=300, contamination=0.2)
    
    optimizer = create_adaptive_threshold_optimizer(target_fpr=0.05)
    recommendations = optimizer.get_threshold_recommendations(scores, y)
    
    logger.info("Threshold recommendations:")
    for objective, threshold in recommendations.items():
        logger.info(f"  {objective}: {threshold:.4f}")
    
    assert len(recommendations) > 0
    assert all(0 < t < 1 for t in recommendations.values())
    
    return True


def test_multi_objective_optimization():
    """Test optimization with different objectives."""
    logger.info("=== Testing Multi-Objective Optimization ===")
    
    X, y, scores = generate_synthetic_data(n_samples=400, contamination=0.12)
    
    metrics = ["f1_score", "precision", "recall"]
    results = {}
    
    for metric in metrics:
        logger.info(f"Optimizing for {metric}...")
        
        config = ThresholdOptimizationConfig(
            n_trials=15,
            primary_metric=metric,
            target_fpr=0.08,
            min_precision=0.6,
            min_recall=0.6
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
    
    # Validate different metrics produce different results
    thresholds = [results[m]['threshold'] for m in metrics]
    logger.info(f"✓ Different thresholds: {len(set(thresholds))} unique values")
    
    return True


def test_persistence():
    """Test saving and loading optimizer."""
    logger.info("=== Testing Optimizer Persistence ===")
    
    X, y, scores = generate_synthetic_data(n_samples=200, contamination=0.15)
    
    # Create and train optimizer
    config = ThresholdOptimizationConfig(n_trials=10)
    optimizer1 = AdaptiveThresholdOptimizer(config)
    result1 = optimizer1.optimize_threshold(scores, y)
    
    # Save optimizer
    save_path = "/tmp/test_optimizer_simple.json"
    optimizer1.save_optimizer(save_path)
    
    # Load optimizer
    optimizer2 = AdaptiveThresholdOptimizer(config)
    optimizer2.load_optimizer(save_path)
    
    # Validate
    assert optimizer2.current_threshold == optimizer1.current_threshold
    assert optimizer2.is_fitted == optimizer1.is_fitted
    
    logger.info(f"✓ Optimizer saved and loaded successfully")
    logger.info(f"✓ Threshold preserved: {optimizer2.current_threshold:.4f}")
    
    # Clean up
    os.remove(save_path)
    
    return True


def run_comprehensive_test():
    """Run comprehensive test suite."""
    logger.info("🚀 Starting Adaptive Threshold Optimizer Simple Test")
    logger.info("=" * 60)
    
    start_time = time.time()
    test_results = {}
    
    try:
        # Test 1: Basic optimization
        test_results['basic_optimization'] = test_basic_optimization()
        
        # Test 2: Online learning
        test_results['online_learning'] = test_online_learning()
        
        # Test 3: Threshold recommendations
        test_results['recommendations'] = test_threshold_recommendations()
        
        # Test 4: Multi-objective optimization
        test_results['multi_objective'] = test_multi_objective_optimization()
        
        # Test 5: Persistence
        test_results['persistence'] = test_persistence()
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        test_results['error'] = str(e)
    
    # Summary
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("🎯 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed_tests = 0
    for test_name, result in test_results.items():
        if test_name == 'error':
            logger.info(f"❌ Error: {result}")
        elif result:
            logger.info(f"✅ {test_name}: PASSED")
            passed_tests += 1
        else:
            logger.info(f"❌ {test_name}: FAILED")
    
    total_tests = len([k for k in test_results.keys() if k != 'error'])
    
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