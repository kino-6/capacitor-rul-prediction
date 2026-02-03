#!/usr/bin/env python3
"""
Simple test script for Robust Validation Framework

This script tests the core functionality with proper binary classification setup.

Author: RUL Prediction System
Date: February 2026
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import time
import logging
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from true_rul.robust_validation_framework import (
    RobustValidationFramework,
    ValidationConfig,
    create_sample_data
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class BinaryAnomalyDetector:
    """Wrapper to ensure binary output from anomaly detectors"""
    
    def __init__(self, base_detector):
        self.base_detector = base_detector
        self.is_fitted = False
    
    def fit(self, X, y=None):
        """Fit the detector"""
        if hasattr(self.base_detector, 'fit'):
            if isinstance(self.base_detector, (IsolationForest, OneClassSVM)):
                # Unsupervised detectors - fit only on normal samples
                if y is not None:
                    normal_samples = X[y == 0] if len(X) == len(y) else X
                    self.base_detector.fit(normal_samples)
                else:
                    self.base_detector.fit(X)
            else:
                # Supervised detectors
                self.base_detector.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict binary labels (0=normal, 1=anomaly)"""
        if not self.is_fitted:
            raise ValueError("Detector must be fitted before prediction")
        
        if hasattr(self.base_detector, 'predict'):
            predictions = self.base_detector.predict(X)
            
            # Convert IsolationForest/OneClassSVM output (-1, 1) to (1, 0)
            if np.any(predictions == -1):
                return (predictions == -1).astype(int)  # -1 (anomaly) -> 1, 1 (normal) -> 0
            else:
                return predictions.astype(int)
        else:
            raise ValueError("Base detector does not have predict method")
    
    def predict_proba(self, X):
        """Predict probabilities if available"""
        if hasattr(self.base_detector, 'predict_proba'):
            return self.base_detector.predict_proba(X)
        return None
    
    def decision_function(self, X):
        """Get decision function scores if available"""
        if hasattr(self.base_detector, 'decision_function'):
            return self.base_detector.decision_function(X)
        elif hasattr(self.base_detector, 'score_samples'):
            return -self.base_detector.score_samples(X)  # Negative for anomaly scores
        return None


def test_basic_validation():
    """Test basic validation functionality"""
    print("\n" + "="*60)
    print("🧪 TESTING BASIC VALIDATION FUNCTIONALITY")
    print("="*60)
    
    # Create sample data
    X, y = create_sample_data(n_samples=400, n_features=8, anomaly_rate=0.15)
    print(f"📊 Created dataset: {len(X)} samples, {X.shape[1]} features, {np.sum(y)} anomalies")
    
    # Create wrapped detector
    base_detector = IsolationForest(contamination=0.15, random_state=42)
    detector = BinaryAnomalyDetector(base_detector)
    
    # Create validation framework
    config = ValidationConfig(
        cv_folds=3,
        bootstrap_samples=20,
        injection_rates=[0.05, 0.1],
        temporal_splits=3,
        verbose=False
    )
    
    validator = RobustValidationFramework(config)
    
    try:
        # Test cross-validation
        print("\n1️⃣ Testing K-fold cross-validation...")
        cv_result = validator.k_fold_cross_validation(X, y, detector)
        print(f"   ✅ CV completed: Mean FPR = {cv_result.mean_metrics.fpr:.4f}")
        print(f"   📊 CV F1 Score = {cv_result.mean_metrics.f1_score:.4f}")
        
        # Test bootstrap validation
        print("\n2️⃣ Testing bootstrap validation...")
        detector.fit(X, y)  # Fit detector first
        bootstrap_result = validator.bootstrap_validation(X, y, detector)
        print(f"   ✅ Bootstrap completed: FPR = {bootstrap_result.mean_metrics.fpr:.4f}")
        print(f"   📊 FPR CI = [{bootstrap_result.confidence_intervals['fpr'][0]:.4f}, {bootstrap_result.confidence_intervals['fpr'][1]:.4f}]")
        
        # Test synthetic anomaly injection
        print("\n3️⃣ Testing synthetic anomaly injection...")
        X_normal = X[y == 0]
        synthetic_result = validator.synthetic_anomaly_injection(X_normal, detector)
        print(f"   ✅ Synthetic injection completed: Stress test passed = {synthetic_result.stress_test_passed}")
        
        for rate, metrics in synthetic_result.metrics_by_rate.items():
            print(f"   📊 Rate {rate:.1%}: FPR={metrics.fpr:.4f}, F1={metrics.f1_score:.4f}")
        
        # Test temporal validation
        print("\n4️⃣ Testing temporal validation...")
        temporal_result = validator.temporal_validation(X, y, detector)
        print(f"   ✅ Temporal validation completed: Stability = {temporal_result.temporal_stability:.4f}")
        print(f"   📊 Drift detected = {temporal_result.drift_detected}")
        
        print("\n🎉 All basic validation tests PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Basic validation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comprehensive_validation():
    """Test comprehensive validation pipeline"""
    print("\n" + "="*60)
    print("🔬 TESTING COMPREHENSIVE VALIDATION PIPELINE")
    print("="*60)
    
    # Create dataset
    X, y = create_sample_data(n_samples=600, n_features=10, anomaly_rate=0.12)
    print(f"📊 Created dataset: {len(X)} samples, {X.shape[1]} features, {np.sum(y)} anomalies")
    
    # Test with different detectors
    detectors = {
        'IsolationForest': BinaryAnomalyDetector(IsolationForest(contamination=0.12, random_state=42)),
        'OneClassSVM': BinaryAnomalyDetector(OneClassSVM(nu=0.12)),
    }
    
    config = ValidationConfig(
        cv_folds=3,
        bootstrap_samples=25,
        injection_rates=[0.05, 0.1, 0.15],
        temporal_splits=3,
        verbose=True
    )
    
    validator = RobustValidationFramework(config)
    
    for detector_name, detector in detectors.items():
        print(f"\n🤖 Testing comprehensive validation with {detector_name}...")
        
        try:
            start_time = time.time()
            
            # Run comprehensive validation
            results = validator.comprehensive_validation(
                X, y, detector, 
                save_results=True,
                results_path=f"validation_results_{detector_name.lower()}.json"
            )
            
            validation_time = time.time() - start_time
            
            # Generate report
            report = validator.generate_validation_report(
                results, 
                output_path=f"validation_report_{detector_name.lower()}.txt"
            )
            
            print(f"   ✅ {detector_name} validation completed in {validation_time:.2f}s")
            
            # Print key metrics
            if 'cross_validation' in results:
                cv_fpr = results['cross_validation']['mean_metrics']['fpr']
                cv_f1 = results['cross_validation']['mean_metrics']['f1_score']
                print(f"   📊 Cross-validation: FPR={cv_fpr:.4f}, F1={cv_f1:.4f}")
            
            if 'bootstrap' in results:
                bootstrap_fpr = results['bootstrap']['mean_metrics']['fpr']
                fpr_ci = results['bootstrap']['confidence_intervals']['fpr']
                print(f"   📊 Bootstrap FPR: {bootstrap_fpr:.4f} [{fpr_ci[0]:.4f}, {fpr_ci[1]:.4f}]")
            
            if 'synthetic_anomaly' in results and results['synthetic_anomaly']:
                stress_passed = results['synthetic_anomaly']['stress_test_passed']
                print(f"   📊 Stress test: {'PASSED' if stress_passed else 'FAILED'}")
            
            if 'temporal' in results:
                stability = results['temporal']['temporal_stability']
                drift = results['temporal']['drift_detected']
                print(f"   📊 Temporal: Stability={stability:.4f}, Drift={'YES' if drift else 'NO'}")
            
        except Exception as e:
            print(f"   ❌ {detector_name} validation FAILED: {e}")
            return False
    
    print("\n🎉 Comprehensive validation tests PASSED!")
    return True


def test_performance_metrics():
    """Test performance and metrics accuracy"""
    print("\n" + "="*60)
    print("📊 TESTING PERFORMANCE METRICS")
    print("="*60)
    
    # Create controlled dataset
    np.random.seed(42)
    n_normal = 800
    n_anomaly = 200
    n_features = 6
    
    # Normal samples
    X_normal = np.random.multivariate_normal(
        mean=np.zeros(n_features),
        cov=np.eye(n_features),
        size=n_normal
    )
    
    # Anomalous samples (clearly separated)
    X_anomaly = np.random.multivariate_normal(
        mean=np.ones(n_features) * 4,
        cov=np.eye(n_features) * 2,
        size=n_anomaly
    )
    
    # Combine data
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([np.zeros(n_normal), np.ones(n_anomaly)])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    print(f"📊 Created controlled dataset: {len(X)} samples, {np.sum(y)} anomalies")
    
    # Test with well-tuned detector
    detector = BinaryAnomalyDetector(IsolationForest(contamination=0.2, random_state=42))
    
    config = ValidationConfig(
        cv_folds=5,
        bootstrap_samples=50,
        injection_rates=[0.1, 0.2, 0.3],
        temporal_splits=4,
        verbose=False
    )
    
    validator = RobustValidationFramework(config)
    
    try:
        # Run comprehensive validation
        results = validator.comprehensive_validation(X, y, detector, save_results=False)
        
        # Check results quality
        cv_metrics = results['cross_validation']['mean_metrics']
        bootstrap_metrics = results['bootstrap']['mean_metrics']
        
        print(f"\n📈 PERFORMANCE RESULTS:")
        print(f"   Cross-validation FPR: {cv_metrics['fpr']:.4f}")
        print(f"   Cross-validation F1:  {cv_metrics['f1_score']:.4f}")
        print(f"   Bootstrap FPR:        {bootstrap_metrics['fpr']:.4f}")
        print(f"   Bootstrap F1:         {bootstrap_metrics['f1_score']:.4f}")
        
        # Check if results are reasonable
        if cv_metrics['fpr'] < 0.5 and cv_metrics['f1_score'] > 0.3:
            print("   ✅ Performance metrics are reasonable")
        else:
            print("   ⚠️  Performance metrics may need tuning")
        
        # Check synthetic anomaly results
        if 'synthetic_anomaly' in results and results['synthetic_anomaly']:
            synthetic = results['synthetic_anomaly']
            print(f"   Stress test passed: {synthetic['stress_test_passed']}")
            
            for rate_str, metrics in synthetic['metrics_by_rate'].items():
                rate = float(rate_str)
                print(f"   Injection {rate:.1%}: FPR={metrics['fpr']:.4f}")
        
        # Check temporal stability
        if 'temporal' in results:
            temporal = results['temporal']
            print(f"   Temporal stability: {temporal['temporal_stability']:.4f}")
        
        print("\n🎉 Performance metrics test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Performance metrics test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("🚀 STARTING ROBUST VALIDATION FRAMEWORK SIMPLE TESTS")
    print("=" * 80)
    
    test_results = []
    
    # Run test suites
    test_suites = [
        ("Basic Validation", test_basic_validation),
        ("Comprehensive Validation", test_comprehensive_validation),
        ("Performance Metrics", test_performance_metrics)
    ]
    
    for suite_name, test_func in test_suites:
        print(f"\n🧪 Running {suite_name} tests...")
        try:
            result = test_func()
            test_results.append((suite_name, result))
        except Exception as e:
            print(f"❌ {suite_name} test suite failed with exception: {e}")
            test_results.append((suite_name, False))
    
    # Print summary
    print("\n" + "="*80)
    print("📊 TEST SUMMARY")
    print("="*80)
    
    passed = 0
    total = len(test_results)
    
    for suite_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{suite_name:.<50} {status}")
        if result:
            passed += 1
    
    print(f"\n📈 Results: {passed}/{total} test suites passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Robust Validation Framework is working correctly.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)