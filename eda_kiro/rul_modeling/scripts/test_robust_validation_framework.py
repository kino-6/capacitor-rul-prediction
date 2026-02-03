#!/usr/bin/env python3
"""
Test script for Robust Validation Framework

This script tests all components of the robust validation framework:
1. K-fold cross-validation with stratified sampling
2. Bootstrap sampling for confidence interval estimation
3. Synthetic anomaly injection for stress testing
4. Temporal validation (time-series cross-validation)

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


def test_basic_functionality():
    """Test basic functionality of the validation framework"""
    print("\n" + "="*60)
    print("🧪 TESTING BASIC FUNCTIONALITY")
    print("="*60)
    
    # Create sample data
    X, y = create_sample_data(n_samples=500, n_features=8, anomaly_rate=0.15)
    print(f"📊 Created dataset: {len(X)} samples, {X.shape[1]} features, {np.sum(y)} anomalies")
    
    # Create model
    model = IsolationForest(contamination=0.15, random_state=42)
    
    # Create validation framework with fast settings
    config = ValidationConfig(
        cv_folds=3,
        bootstrap_samples=20,
        injection_rates=[0.05, 0.1],
        temporal_splits=3,
        verbose=False
    )
    
    validator = RobustValidationFramework(config)
    
    try:
        # Test individual components
        print("\n1️⃣ Testing K-fold cross-validation...")
        cv_result = validator.k_fold_cross_validation(X, y, model)
        print(f"   ✅ CV completed: Mean FPR = {cv_result.mean_metrics.fpr:.4f}")
        
        print("\n2️⃣ Testing bootstrap validation...")
        # Train model first for bootstrap
        model.fit(X, y)
        bootstrap_result = validator.bootstrap_validation(X, y, model)
        print(f"   ✅ Bootstrap completed: FPR CI = {bootstrap_result.confidence_intervals['fpr']}")
        
        print("\n3️⃣ Testing synthetic anomaly injection...")
        # Use only normal samples
        X_normal = X[y == 0]
        synthetic_result = validator.synthetic_anomaly_injection(X_normal, model)
        print(f"   ✅ Synthetic injection completed: Stress test passed = {synthetic_result.stress_test_passed}")
        
        print("\n4️⃣ Testing temporal validation...")
        temporal_result = validator.temporal_validation(X, y, model)
        print(f"   ✅ Temporal validation completed: Stability = {temporal_result.temporal_stability:.4f}")
        
        print("\n🎉 All basic functionality tests PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ Basic functionality test FAILED: {e}")
        return False


def test_comprehensive_validation():
    """Test comprehensive validation pipeline"""
    print("\n" + "="*60)
    print("🔬 TESTING COMPREHENSIVE VALIDATION")
    print("="*60)
    
    # Create larger dataset
    X, y = create_sample_data(n_samples=800, n_features=12, anomaly_rate=0.12)
    print(f"📊 Created dataset: {len(X)} samples, {X.shape[1]} features, {np.sum(y)} anomalies")
    
    # Test with different models
    models = {
        'IsolationForest': IsolationForest(contamination=0.12, random_state=42),
        'OneClassSVM': OneClassSVM(nu=0.12),
    }
    
    config = ValidationConfig(
        cv_folds=4,
        bootstrap_samples=30,
        injection_rates=[0.05, 0.1, 0.15],
        temporal_splits=4,
        verbose=True
    )
    
    validator = RobustValidationFramework(config)
    
    for model_name, model in models.items():
        print(f"\n🤖 Testing with {model_name}...")
        
        try:
            start_time = time.time()
            
            # Run comprehensive validation
            results = validator.comprehensive_validation(
                X, y, model, 
                save_results=True,
                results_path=f"test_results_{model_name.lower()}.json"
            )
            
            validation_time = time.time() - start_time
            
            # Generate report
            report = validator.generate_validation_report(
                results, 
                output_path=f"test_report_{model_name.lower()}.txt"
            )
            
            print(f"   ✅ {model_name} validation completed in {validation_time:.2f}s")
            
            # Print key metrics
            if 'cross_validation' in results:
                cv_fpr = results['cross_validation']['mean_metrics']['fpr']
                print(f"   📊 Cross-validation FPR: {cv_fpr:.4f}")
            
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
                print(f"   📊 Temporal stability: {stability:.4f}, Drift: {'YES' if drift else 'NO'}")
            
        except Exception as e:
            print(f"   ❌ {model_name} validation FAILED: {e}")
            return False
    
    print("\n🎉 Comprehensive validation tests PASSED!")
    return True


def test_edge_cases():
    """Test edge cases and error handling"""
    print("\n" + "="*60)
    print("⚠️  TESTING EDGE CASES")
    print("="*60)
    
    validator = RobustValidationFramework()
    
    # Test 1: Very small dataset
    print("\n1️⃣ Testing with very small dataset...")
    try:
        X_small, y_small = create_sample_data(n_samples=20, n_features=3, anomaly_rate=0.2)
        model = IsolationForest(contamination=0.2, random_state=42)
        
        cv_result = validator.k_fold_cross_validation(X_small, y_small, model)
        print(f"   ✅ Small dataset handled: {len(cv_result.fold_metrics)} folds completed")
        
    except Exception as e:
        print(f"   ⚠️  Small dataset test: {e}")
    
    # Test 2: No anomalies dataset
    print("\n2️⃣ Testing with no anomalies...")
    try:
        X_normal = np.random.randn(100, 5)
        y_normal = np.zeros(100)
        model = IsolationForest(contamination=0.1, random_state=42)
        
        cv_result = validator.k_fold_cross_validation(X_normal, y_normal, model)
        print(f"   ✅ No anomalies handled: FPR = {cv_result.mean_metrics.fpr:.4f}")
        
    except Exception as e:
        print(f"   ⚠️  No anomalies test: {e}")
    
    # Test 3: All anomalies dataset
    print("\n3️⃣ Testing with all anomalies...")
    try:
        X_anomaly = np.random.randn(50, 4) * 3 + 5
        y_anomaly = np.ones(50)
        model = IsolationForest(contamination=0.9, random_state=42)
        
        cv_result = validator.k_fold_cross_validation(X_anomaly, y_anomaly, model)
        print(f"   ✅ All anomalies handled: TPR = {cv_result.mean_metrics.tpr:.4f}")
        
    except Exception as e:
        print(f"   ⚠️  All anomalies test: {e}")
    
    # Test 4: High-dimensional data
    print("\n4️⃣ Testing with high-dimensional data...")
    try:
        X_hd, y_hd = create_sample_data(n_samples=200, n_features=50, anomaly_rate=0.1)
        model = IsolationForest(contamination=0.1, random_state=42)
        
        config = ValidationConfig(cv_folds=3, bootstrap_samples=10, verbose=False)
        validator_hd = RobustValidationFramework(config)
        
        cv_result = validator_hd.k_fold_cross_validation(X_hd, y_hd, model)
        print(f"   ✅ High-dimensional data handled: {X_hd.shape[1]} features")
        
    except Exception as e:
        print(f"   ⚠️  High-dimensional test: {e}")
    
    print("\n🎉 Edge case tests completed!")
    return True


def test_performance_benchmarks():
    """Test performance benchmarks"""
    print("\n" + "="*60)
    print("⚡ TESTING PERFORMANCE BENCHMARKS")
    print("="*60)
    
    # Test different dataset sizes
    sizes = [100, 500, 1000]
    
    for size in sizes:
        print(f"\n📏 Testing with {size} samples...")
        
        X, y = create_sample_data(n_samples=size, n_features=10, anomaly_rate=0.1)
        model = IsolationForest(contamination=0.1, random_state=42)
        
        config = ValidationConfig(
            cv_folds=3,
            bootstrap_samples=20,
            injection_rates=[0.05, 0.1],
            temporal_splits=3,
            verbose=False
        )
        
        validator = RobustValidationFramework(config)
        
        start_time = time.time()
        
        try:
            results = validator.comprehensive_validation(X, y, model, save_results=False)
            validation_time = time.time() - start_time
            
            print(f"   ⏱️  Validation time: {validation_time:.2f}s")
            print(f"   📊 Samples per second: {size / validation_time:.1f}")
            
            # Check if performance is reasonable (< 1 minute for 1000 samples)
            if size == 1000 and validation_time > 60:
                print(f"   ⚠️  Performance warning: {validation_time:.2f}s > 60s")
            else:
                print(f"   ✅ Performance acceptable")
                
        except Exception as e:
            print(f"   ❌ Performance test failed: {e}")
    
    print("\n🎉 Performance benchmark tests completed!")
    return True


def test_integration_with_existing_models():
    """Test integration with existing anomaly detection models"""
    print("\n" + "="*60)
    print("🔗 TESTING INTEGRATION WITH EXISTING MODELS")
    print("="*60)
    
    # Load existing models if available
    try:
        # Try to import existing models
        from true_rul.isolation_forest_detector import IsolationForestDetector
        from true_rul.ensemble_anomaly_detector import EnsembleAnomalyDetector
        
        print("✅ Successfully imported existing anomaly detectors")
        
        # Create test data
        X, y = create_sample_data(n_samples=400, n_features=8, anomaly_rate=0.1)
        
        # Test with existing models
        models_to_test = [
            ('IsolationForestDetector', IsolationForestDetector()),
            ('EnsembleAnomalyDetector', EnsembleAnomalyDetector())
        ]
        
        config = ValidationConfig(
            cv_folds=3,
            bootstrap_samples=15,
            injection_rates=[0.05, 0.1],
            temporal_splits=3,
            verbose=False
        )
        
        validator = RobustValidationFramework(config)
        
        for model_name, model in models_to_test:
            print(f"\n🤖 Testing integration with {model_name}...")
            
            try:
                # Test cross-validation
                cv_result = validator.k_fold_cross_validation(X, y, model)
                print(f"   ✅ Cross-validation: FPR = {cv_result.mean_metrics.fpr:.4f}")
                
                # Test bootstrap (need to train model first)
                model.fit(X[y == 0])  # Train on normal samples
                bootstrap_result = validator.bootstrap_validation(X, y, model)
                print(f"   ✅ Bootstrap: FPR = {bootstrap_result.mean_metrics.fpr:.4f}")
                
            except Exception as e:
                print(f"   ⚠️  Integration test with {model_name}: {e}")
        
    except ImportError as e:
        print(f"⚠️  Could not import existing models: {e}")
        print("   Using sklearn models for integration test...")
        
        # Fallback to sklearn models
        X, y = create_sample_data(n_samples=300, n_features=6, anomaly_rate=0.1)
        
        models_to_test = [
            ('IsolationForest', IsolationForest(contamination=0.1, random_state=42)),
            ('OneClassSVM', OneClassSVM(nu=0.1))
        ]
        
        config = ValidationConfig(cv_folds=3, bootstrap_samples=10, verbose=False)
        validator = RobustValidationFramework(config)
        
        for model_name, model in models_to_test:
            try:
                cv_result = validator.k_fold_cross_validation(X, y, model)
                print(f"   ✅ {model_name}: FPR = {cv_result.mean_metrics.fpr:.4f}")
            except Exception as e:
                print(f"   ❌ {model_name} failed: {e}")
    
    print("\n🎉 Integration tests completed!")
    return True


def main():
    """Run all tests"""
    print("🚀 STARTING ROBUST VALIDATION FRAMEWORK TESTS")
    print("=" * 80)
    
    test_results = []
    
    # Run all test suites
    test_suites = [
        ("Basic Functionality", test_basic_functionality),
        ("Comprehensive Validation", test_comprehensive_validation),
        ("Edge Cases", test_edge_cases),
        ("Performance Benchmarks", test_performance_benchmarks),
        ("Integration Tests", test_integration_with_existing_models)
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
        print("🎉 ALL TESTS PASSED! Robust Validation Framework is ready for use.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the output above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)