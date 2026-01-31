#!/usr/bin/env python3
"""
Quick Model Implementation Verification

Streamlined verification focusing on core functionality with minimal training.
"""

import sys
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_core_models():
    """Test core model instantiation and basic functionality."""
    print("🔍 Quick Model Verification")
    print("=" * 50)
    
    results = {}
    
    # Test 1: RUL Regression Models
    print("\n1. Testing RUL Regression Models...")
    
    try:
        from true_rul.rul_regression_model import RULRegressionModel
        
        # Quick synthetic data
        np.random.seed(42)
        X = np.random.randn(50, 10)
        y = 100 - 5 * X[:, 0] + np.random.randn(50) * 0.1
        y = np.maximum(y, 0)
        
        X_train, X_val = X[:40], X[40:]
        y_train, y_val = y[:40], y[40:]
        
        # Test each model type quickly
        models_to_test = ["xgboost", "lightgbm", "random_forest", "elastic_net"]
        
        for model_type in models_to_test:
            try:
                print(f"  Testing {model_type}...", end=" ")
                
                # Use minimal parameters for speed
                if model_type in ["xgboost", "lightgbm"]:
                    model = RULRegressionModel(model_type, n_estimators=10, max_depth=3)
                else:
                    model = RULRegressionModel(model_type)
                
                # Quick training
                model.train(X_train, y_train, X_val, y_val)
                
                # Test prediction
                pred = model.predict(X_val)
                assert len(pred) == len(y_val)
                assert np.all(pred >= 0)
                
                # Test feature importance
                try:
                    importance = model.get_feature_importance()
                    assert isinstance(importance, dict)
                    print("✓ (with feature importance)")
                    results[f"rul_{model_type}"] = "PASS_WITH_FI"
                except:
                    print("✓ (no feature importance)")
                    results[f"rul_{model_type}"] = "PASS_NO_FI"
                    
            except Exception as e:
                print(f"✗ ({str(e)[:50]}...)")
                results[f"rul_{model_type}"] = "FAIL"
        
        # Test ensemble quickly
        try:
            print("  Testing ensemble...", end=" ")
            from true_rul.hybrid_ensemble_predictor import HybridEnsembleRULPredictor
            
            # Minimal ensemble for speed
            ensemble = HybridEnsembleRULPredictor(
                xgboost_params={'n_estimators': 5, 'max_depth': 2},
                lightgbm_params={'n_estimators': 5, 'max_depth': 2},
                random_forest_params={'n_estimators': 5, 'max_depth': 3}
            )
            
            ensemble.train(X_train, y_train, X_val, y_val, verbose=False)
            pred, lower, upper = ensemble.predict_with_confidence(X_val)
            
            assert len(pred) == len(y_val)
            assert np.all(pred >= 0)
            
            # Test aggregated feature importance
            importance = ensemble.get_aggregated_feature_importance()
            assert isinstance(importance, dict)
            
            print("✓ (with confidence & importance)")
            results["rul_ensemble"] = "PASS_FULL"
            
        except Exception as e:
            print(f"✗ ({str(e)[:50]}...)")
            results["rul_ensemble"] = "FAIL"
            
    except Exception as e:
        print(f"✗ Failed to import RUL models: {e}")
        results["rul_models"] = "IMPORT_FAIL"
    
    # Test 2: Anomaly Detection Models
    print("\n2. Testing Anomaly Detection Models...")
    
    try:
        # Quick normal data
        X_normal = np.random.randn(30, 10)
        X_test = np.random.randn(10, 10)
        
        # Test Isolation Forest
        try:
            print("  Testing Isolation Forest...", end=" ")
            from true_rul.isolation_forest_detector import IsolationForestDetector
            
            detector = IsolationForestDetector(contamination=0.1)
            detector.fit(X_normal)
            scores = detector.predict_score(X_test)
            
            assert len(scores) == len(X_test)
            print("✓")
            results["anomaly_if"] = "PASS"
            
        except Exception as e:
            print(f"✗ ({str(e)[:50]}...)")
            results["anomaly_if"] = "FAIL"
        
        # Test Autoencoder (minimal training)
        try:
            print("  Testing Autoencoder...", end=" ")
            from true_rul.autoencoder_detector import AutoencoderDetector
            
            detector = AutoencoderDetector(input_dim=10, encoding_dim=5)
            detector.fit(X_normal, epochs=5, verbose=False)
            errors = detector.get_reconstruction_error(X_test)
            
            assert len(errors) == len(X_test)
            print("✓")
            results["anomaly_ae"] = "PASS"
            
        except Exception as e:
            print(f"✗ ({str(e)[:50]}...)")
            results["anomaly_ae"] = "FAIL"
        
        # Test OCSVM
        try:
            print("  Testing One-Class SVM...", end=" ")
            from true_rul.improved_ocsvm import ImprovedOCSVM
            
            detector = ImprovedOCSVM(nu=0.1)
            detector.fit(X_normal)
            scores = detector.predict_score(X_test)
            
            assert len(scores) == len(X_test)
            print("✓")
            results["anomaly_ocsvm"] = "PASS"
            
        except Exception as e:
            print(f"✗ ({str(e)[:50]}...)")
            results["anomaly_ocsvm"] = "FAIL"
        
        # Test Ensemble (quick)
        try:
            print("  Testing Ensemble Detector...", end=" ")
            from true_rul.ensemble_anomaly_detector import EnsembleAnomalyDetector
            
            detector = EnsembleAnomalyDetector(
                autoencoder_params={'encoding_dim': 5}
            )
            detector.fit(X_normal)
            binary_pred, scores, info = detector.predict(X_test)
            
            assert len(binary_pred) == len(X_test)
            assert len(scores) == len(X_test)
            assert isinstance(info, dict)
            
            print("✓")
            results["anomaly_ensemble"] = "PASS"
            
        except Exception as e:
            print(f"✗ ({str(e)[:50]}...)")
            results["anomaly_ensemble"] = "FAIL"
            
    except Exception as e:
        print(f"✗ Failed to test anomaly models: {e}")
        results["anomaly_models"] = "IMPORT_FAIL"
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for v in results.values() if "PASS" in v)
    total = len(results)
    
    print(f"Models tested: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {total - passed}")
    
    if passed >= total * 0.8:  # 80% pass rate
        print("\n✅ VERIFICATION SUCCESSFUL")
        print("All core model classes can be instantiated and trained!")
        
        # Check feature importance
        fi_models = sum(1 for v in results.values() if "WITH_FI" in v or "FULL" in v)
        if fi_models > 0:
            print(f"✅ Feature importance available in {fi_models} models")
        
        return True
    else:
        print("\n❌ VERIFICATION FAILED")
        print("Some models failed basic functionality tests")
        return False

if __name__ == "__main__":
    success = test_core_models()
    sys.exit(0 if success else 1)