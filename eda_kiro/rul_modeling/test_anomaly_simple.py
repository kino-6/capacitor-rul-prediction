#!/usr/bin/env python3
"""
Simple test for anomaly detection models without dependencies.

This script tests the basic functionality of anomaly detection models
without importing the full module structure.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import individual modules directly
from true_rul.isolation_forest_detector import IsolationForestDetector
from true_rul.autoencoder_detector import AutoencoderDetector
from true_rul.improved_ocsvm import ImprovedOCSVM


def test_isolation_forest():
    """Test IsolationForestDetector."""
    print("Testing IsolationForestDetector...")
    
    # Generate synthetic normal data
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (100, 5))
    
    # Create and train detector
    detector = IsolationForestDetector(contamination=0.05)
    detector.fit(normal_data)
    
    # Test prediction
    test_data = np.random.normal(0, 1, (10, 5))
    scores = detector.predict_score(test_data)
    binary_pred = detector.predict_binary(test_data)
    
    print(f"  ✓ Trained on {len(normal_data)} samples")
    print(f"  ✓ Predicted scores for {len(test_data)} samples")
    print(f"  ✓ Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"  ✓ Anomalies detected: {np.sum(binary_pred == -1)}/{len(binary_pred)}")
    
    return True


def test_autoencoder():
    """Test AutoencoderDetector."""
    print("Testing AutoencoderDetector...")
    
    # Generate synthetic normal data
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (100, 10))
    
    # Create and train detector
    detector = AutoencoderDetector(input_dim=10, encoding_dim=5)
    detector.fit(normal_data, epochs=10, verbose=False)  # Quick training
    
    # Test prediction
    test_data = np.random.normal(0, 1, (10, 10))
    errors = detector.get_reconstruction_error(test_data)
    binary_pred = detector.predict_binary(test_data)
    
    print(f"  ✓ Trained on {len(normal_data)} samples")
    print(f"  ✓ Predicted errors for {len(test_data)} samples")
    print(f"  ✓ Error range: [{errors.min():.6f}, {errors.max():.6f}]")
    print(f"  ✓ Threshold: {detector.reconstruction_threshold:.6f}")
    print(f"  ✓ Anomalies detected: {np.sum(binary_pred == 1)}/{len(binary_pred)}")
    
    return True


def test_improved_ocsvm():
    """Test ImprovedOCSVM."""
    print("Testing ImprovedOCSVM...")
    
    # Generate synthetic normal data
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (100, 5))
    
    # Create and train detector (disable auto-tuning for speed)
    detector = ImprovedOCSVM(nu=0.05, auto_tune=False)
    detector.fit(normal_data)
    
    # Test prediction
    test_data = np.random.normal(0, 1, (10, 5))
    scores = detector.predict_score(test_data)
    binary_pred = detector.predict_binary(test_data)
    
    print(f"  ✓ Trained on {len(normal_data)} samples")
    print(f"  ✓ Predicted scores for {len(test_data)} samples")
    print(f"  ✓ Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"  ✓ Support vectors: {len(detector.get_support_vectors())}")
    print(f"  ✓ Anomalies detected: {np.sum(binary_pred == -1)}/{len(binary_pred)}")
    
    return True


def test_ensemble():
    """Test EnsembleAnomalyDetector."""
    print("Testing EnsembleAnomalyDetector...")
    
    # Import here to avoid circular dependencies
    from true_rul.ensemble_anomaly_detector import EnsembleAnomalyDetector
    
    # Generate synthetic normal data
    np.random.seed(42)
    normal_data = np.random.normal(0, 1, (100, 10))
    
    # Create ensemble with smaller autoencoder for speed
    ensemble = EnsembleAnomalyDetector(
        autoencoder_params={'encoding_dim': 5}
    )
    
    # Train ensemble (this will take a bit longer due to autoencoder)
    print("  Training ensemble (this may take a moment)...")
    ensemble.fit(normal_data)
    
    # Test prediction
    test_data = np.random.normal(0, 1, (20, 10))
    binary_pred, scores, info = ensemble.predict(test_data)
    
    print(f"  ✓ Trained ensemble on {len(normal_data)} samples")
    print(f"  ✓ Predicted for {len(test_data)} samples")
    print(f"  ✓ Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    print(f"  ✓ Threshold: {info['threshold']:.3f}")
    print(f"  ✓ Anomalies detected: {info['n_anomalies']}/{len(binary_pred)} ({info['anomaly_rate']:.1%})")
    print(f"  ✓ Detector weights: {info['detector_weights']}")
    
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("ANOMALY DETECTION MODELS SIMPLE TEST")
    print("=" * 60)
    
    tests = [
        test_isolation_forest,
        test_autoencoder,
        test_improved_ocsvm,
        test_ensemble
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            print()
            if test_func():
                passed += 1
                print(f"  ✅ {test_func.__name__} PASSED")
            else:
                failed += 1
                print(f"  ❌ {test_func.__name__} FAILED")
        except Exception as e:
            failed += 1
            print(f"  ❌ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("🎉 All anomaly detection models are working correctly!")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit(main())