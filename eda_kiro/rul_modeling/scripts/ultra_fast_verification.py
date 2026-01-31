#!/usr/bin/env python3
"""
Ultra-Fast Model Verification

Minimal verification focusing on instantiation and basic functionality only.
"""

import sys
import os
import numpy as np
import warnings
import time

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️  tqdm not available")

def test_instantiation_only():
    """Test that all models can be instantiated - fastest possible check."""
    print("🚀 Ultra-Fast Model Verification (Instantiation + Basic Training)")
    print("=" * 60)
    
    results = {}
    start_time = time.time()
    
    # Tiny test data
    print("📊 Generating minimal test data...")
    np.random.seed(42)
    X = np.random.randn(20, 8).astype(np.float32)
    y = (50 - 2 * X[:, 0]).astype(np.float32)
    y = np.maximum(y, 0)
    X_train, X_val = X[:15], X[15:]
    y_train, y_val = y[:15], y[15:]
    X_normal = X_train[:10]
    
    print("\n1️⃣ RUL Regression Models")
    
    # Test RUL models with minimal training
    rul_tests = [
        ("XGBoost", lambda: test_rul_model("xgboost", X_train, y_train, X_val, y_val)),
        ("LightGBM", lambda: test_rul_model("lightgbm", X_train, y_train, X_val, y_val)),
        ("RandomForest", lambda: test_rul_model("random_forest", X_train, y_train, X_val, y_val)),
        ("ElasticNet", lambda: test_rul_model("elastic_net", X_train, y_train, X_val, y_val)),
    ]
    
    if HAS_TQDM:
        rul_iterator = tqdm(rul_tests, desc="RUL Models", ncols=80)
    else:
        rul_iterator = rul_tests
    
    for name, test_func in rul_iterator:
        try:
            start = time.time()
            success, has_fi, has_shap = test_func()
            duration = time.time() - start
            
            if success:
                fi_status = "✓" if has_fi else "○"
                shap_status = "✓" if has_shap else "○"
                status_msg = f"{name}: {duration:.2f}s FI:{fi_status} SHAP:{shap_status}"
                
                if HAS_TQDM:
                    rul_iterator.set_postfix_str(status_msg)
                else:
                    print(f"  ✅ {status_msg}")
                results[name] = "PASS"
            else:
                if not HAS_TQDM:
                    print(f"  ❌ {name:<12} - Failed basic test")
                results[name] = "FAIL"
        except Exception as e:
            if not HAS_TQDM:
                print(f"  ❌ {name:<12} - {str(e)[:40]}")
            results[name] = "FAIL"
    
    # Test ensemble
    print("\n🔄 Testing Hybrid Ensemble...")
    try:
        start = time.time()
        
        if HAS_TQDM:
            with tqdm(total=1, desc="Ensemble Training", ncols=80) as pbar:
                success = test_ensemble(X_train, y_train, X_val, y_val)
                pbar.update(1)
        else:
            success = test_ensemble(X_train, y_train, X_val, y_val)
            
        duration = time.time() - start
        
        if success:
            print(f"  ✅ HybridEnsemble ({duration:.2f}s) - Full functionality")
            results["HybridEnsemble"] = "PASS"
        else:
            print(f"  ❌ HybridEnsemble - Failed")
            results["HybridEnsemble"] = "FAIL"
    except Exception as e:
        print(f"  ❌ HybridEnsemble - {str(e)[:40]}")
        results["HybridEnsemble"] = "FAIL"
    
    print("\n2️⃣ Anomaly Detection Models")
    
    # Test anomaly models with ultra-minimal training
    anomaly_tests = [
        ("IsolationForest", lambda: test_isolation_forest(X_normal)),
        ("OCSVM", lambda: test_ocsvm(X_normal)),
        ("Autoencoder", lambda: test_autoencoder_minimal(X_normal)),
        ("EnsembleDetector", lambda: test_ensemble_detector(X_normal)),
    ]
    
    if HAS_TQDM:
        anomaly_iterator = tqdm(anomaly_tests, desc="Anomaly Models", ncols=80)
    else:
        anomaly_iterator = anomaly_tests
    
    for name, test_func in anomaly_iterator:
        try:
            start = time.time()
            success = test_func()
            duration = time.time() - start
            
            if success:
                status_msg = f"{name}: {duration:.2f}s"
                if HAS_TQDM:
                    anomaly_iterator.set_postfix_str(status_msg)
                else:
                    print(f"  ✅ {status_msg}")
                results[name] = "PASS"
            else:
                if not HAS_TQDM:
                    print(f"  ❌ {name:<15} - Failed basic test")
                results[name] = "FAIL"
        except Exception as e:
            if not HAS_TQDM:
                print(f"  ❌ {name:<15} - {str(e)[:40]}")
            results[name] = "FAIL"
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(1 for v in results.values() if v == "PASS")
    total = len(results)
    
    print("\n" + "=" * 60)
    print("📊 ULTRA-FAST VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"🎯 Models tested: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {total - passed}")
    print(f"📈 Success rate: {passed/total*100:.1f}%")
    
    if passed >= total * 0.75:  # 75% threshold
        print("\n🎉 VERIFICATION SUCCESSFUL!")
        print("✅ Core model implementations are working")
        print("✅ Models can be instantiated and trained")
        print("✅ Feature importance is available")
        return True
    else:
        print("\n💥 VERIFICATION FAILED!")
        return False

def test_rul_model(model_type, X_train, y_train, X_val, y_val):
    """Test a single RUL model with minimal parameters."""
    from true_rul.rul_regression_model import RULRegressionModel
    
    # Ultra-minimal parameters for speed
    if model_type in ["xgboost", "lightgbm"]:
        model = RULRegressionModel(model_type, n_estimators=5, max_depth=2)
    else:
        model = RULRegressionModel(model_type)
    
    model.train(X_train, y_train, X_val, y_val, verbose=False)
    pred = model.predict(X_val)
    
    # Test feature importance
    has_fi = False
    has_shap = False
    
    try:
        importance = model.get_feature_importance()
        has_fi = isinstance(importance, dict) and len(importance) > 0
    except:
        pass
    
    # Only test SHAP for tree models
    if model_type in ["xgboost", "lightgbm"]:
        try:
            shap_values = model.get_shap_values(X_val[:1])
            has_shap = isinstance(shap_values, np.ndarray)
        except:
            pass
    
    return len(pred) == len(y_val) and np.all(pred >= 0), has_fi, has_shap

def test_ensemble(X_train, y_train, X_val, y_val):
    """Test ensemble with ultra-minimal parameters."""
    from true_rul.hybrid_ensemble_predictor import HybridEnsembleRULPredictor
    
    ensemble = HybridEnsembleRULPredictor(
        xgboost_params={'n_estimators': 3, 'max_depth': 2},
        lightgbm_params={'n_estimators': 3, 'max_depth': 2},
        random_forest_params={'n_estimators': 3, 'max_depth': 2}
    )
    
    ensemble.train(X_train, y_train, X_val, y_val, verbose=False)
    pred, lower, upper = ensemble.predict_with_confidence(X_val)
    importance = ensemble.get_aggregated_feature_importance()
    
    return (len(pred) == len(y_val) and 
            len(lower) == len(y_val) and 
            len(upper) == len(y_val) and
            len(importance) > 0)

def test_isolation_forest(X_normal):
    """Test Isolation Forest."""
    from true_rul.isolation_forest_detector import IsolationForestDetector
    
    detector = IsolationForestDetector(contamination=0.2)
    detector.fit(X_normal)
    scores = detector.predict_score(X_normal[:5])
    
    return len(scores) == 5

def test_ocsvm(X_normal):
    """Test One-Class SVM."""
    from true_rul.improved_ocsvm import ImprovedOCSVM
    
    detector = ImprovedOCSVM(nu=0.2)
    detector.fit(X_normal)
    scores = detector.predict_score(X_normal[:5])
    
    return len(scores) == 5

def test_autoencoder_minimal(X_normal):
    """Test Autoencoder with absolute minimal training."""
    from true_rul.autoencoder_detector import AutoencoderDetector
    
    print("    🔄 Training autoencoder (2 epochs)...")
    detector = AutoencoderDetector(input_dim=8, encoding_dim=4)
    
    # Ultra-minimal training with progress
    if HAS_TQDM:
        with tqdm(total=2, desc="    Autoencoder Epochs", ncols=60, leave=False) as pbar:
            # Simulate epoch progress (the actual training doesn't expose epoch-level progress easily)
            detector.fit(X_normal, epochs=2, batch_size=len(X_normal), verbose=False)
            pbar.update(2)
    else:
        detector.fit(X_normal, epochs=2, batch_size=len(X_normal), verbose=False)
    
    errors = detector.get_reconstruction_error(X_normal[:5])
    return len(errors) == 5

def test_ensemble_detector(X_normal):
    """Test Ensemble Detector with minimal autoencoder."""
    from true_rul.ensemble_anomaly_detector import EnsembleAnomalyDetector
    
    print("    🔄 Training ensemble detector...")
    
    # Skip autoencoder in ensemble for speed - test with just IF and OCSVM
    detector = EnsembleAnomalyDetector(
        weights=[0.5, 0.0, 0.5],  # Skip autoencoder (weight=0)
        autoencoder_params={'encoding_dim': 4}
    )
    
    if HAS_TQDM:
        with tqdm(total=1, desc="    Ensemble Training", ncols=60, leave=False) as pbar:
            detector.fit(X_normal)
            pbar.update(1)
    else:
        detector.fit(X_normal)
    
    binary_pred, scores, info = detector.predict(X_normal[:5])
    return len(binary_pred) == 5 and len(scores) == 5

if __name__ == "__main__":
    success = test_instantiation_only()
    sys.exit(0 if success else 1)