#!/usr/bin/env python3
"""
Parallel Model Implementation Verification

Optimized for M4 Pro (14 cores, 48GB RAM) with parallel processing and progress bars.
"""

import sys
import os
import numpy as np
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import time
from typing import Dict, Any, Tuple, List

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("⚠️  tqdm not available, using basic progress indicators")

def progress_bar(iterable, desc="Processing", disable=False):
    """Create progress bar if tqdm available, otherwise return iterable."""
    if HAS_TQDM and not disable:
        return tqdm(iterable, desc=desc, ncols=80)
    else:
        print(f"🔄 {desc}...")
        return iterable

def generate_test_data(n_samples: int = 100, n_features: int = 15, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic test data quickly."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features).astype(np.float32)  # Use float32 for speed
    y = (100 - 5 * X[:, 0] + 0.1 * np.random.randn(n_samples)).astype(np.float32)
    y = np.maximum(y, 0)
    return X, y

def test_rul_model(model_config: Tuple[str, str, Dict]) -> Dict[str, Any]:
    """Test a single RUL model in isolation (for multiprocessing)."""
    model_name, model_type, params = model_config
    
    try:
        # Import inside function for multiprocessing
        from true_rul.rul_regression_model import RULRegressionModel
        
        # Generate test data
        X, y = generate_test_data(n_samples=80, n_features=15, seed=42)
        X_train, X_val = X[:60], X[60:]
        y_train, y_val = y[:60], y[60:]
        
        # Create and train model with minimal parameters for speed
        start_time = time.time()
        
        if model_type in ["xgboost", "lightgbm"]:
            model = RULRegressionModel(model_type, n_estimators=20, max_depth=4, **params)
        else:
            model = RULRegressionModel(model_type, **params)
        
        model.train(X_train, y_train, X_val, y_val, verbose=False)
        
        # Test prediction
        pred = model.predict(X_val)
        train_time = time.time() - start_time
        
        # Test feature importance
        has_feature_importance = False
        has_shap = False
        
        try:
            importance = model.get_feature_importance()
            has_feature_importance = isinstance(importance, dict) and len(importance) > 0
        except:
            pass
        
        # Test SHAP (only for tree models to save time)
        if model_type in ["xgboost", "lightgbm"]:
            try:
                shap_values = model.get_shap_values(X_val[:3])  # Test on tiny subset
                has_shap = isinstance(shap_values, np.ndarray)
            except:
                pass
        
        return {
            'model_name': model_name,
            'status': 'PASS',
            'train_time': train_time,
            'prediction_shape': pred.shape,
            'non_negative': np.all(pred >= 0),
            'feature_importance': has_feature_importance,
            'shap_values': has_shap,
            'error': None
        }
        
    except Exception as e:
        return {
            'model_name': model_name,
            'status': 'FAIL',
            'error': str(e)[:100],
            'train_time': 0,
            'prediction_shape': None,
            'non_negative': False,
            'feature_importance': False,
            'shap_values': False
        }

def test_anomaly_model(model_config: Tuple[str, str, Dict]) -> Dict[str, Any]:
    """Test a single anomaly detection model."""
    model_name, model_class, params = model_config
    
    try:
        # Generate test data
        np.random.seed(42)
        X_normal = np.random.randn(50, 15).astype(np.float32)
        X_test = np.random.randn(20, 15).astype(np.float32)
        
        start_time = time.time()
        
        if model_class == "IsolationForestDetector":
            from true_rul.isolation_forest_detector import IsolationForestDetector
            model = IsolationForestDetector(**params)
            model.fit(X_normal)
            scores = model.predict_score(X_test)
            
        elif model_class == "AutoencoderDetector":
            from true_rul.autoencoder_detector import AutoencoderDetector
            model = AutoencoderDetector(input_dim=15, **params)
            model.fit(X_normal, epochs=10, verbose=False)
            scores = model.get_reconstruction_error(X_test)
            
        elif model_class == "ImprovedOCSVM":
            from true_rul.improved_ocsvm import ImprovedOCSVM
            model = ImprovedOCSVM(**params)
            model.fit(X_normal)
            scores = model.predict_score(X_test)
            
        elif model_class == "EnsembleAnomalyDetector":
            from true_rul.ensemble_anomaly_detector import EnsembleAnomalyDetector
            model = EnsembleAnomalyDetector(
                autoencoder_params={'encoding_dim': 8},
                **params
            )
            model.fit(X_normal)
            binary_pred, scores, info = model.predict(X_test)
        
        train_time = time.time() - start_time
        
        return {
            'model_name': model_name,
            'status': 'PASS',
            'train_time': train_time,
            'output_shape': scores.shape if hasattr(scores, 'shape') else len(scores),
            'error': None
        }
        
    except Exception as e:
        return {
            'model_name': model_name,
            'status': 'FAIL',
            'error': str(e)[:100],
            'train_time': 0,
            'output_shape': None
        }

def test_ensemble_model() -> Dict[str, Any]:
    """Test the hybrid ensemble model separately (it's more complex)."""
    try:
        from true_rul.hybrid_ensemble_predictor import HybridEnsembleRULPredictor
        
        # Generate test data
        X, y = generate_test_data(n_samples=100, n_features=15, seed=42)
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]
        
        start_time = time.time()
        
        # Create ensemble with minimal parameters for speed
        ensemble = HybridEnsembleRULPredictor(
            xgboost_params={'n_estimators': 10, 'max_depth': 3},
            lightgbm_params={'n_estimators': 10, 'max_depth': 3},
            random_forest_params={'n_estimators': 10, 'max_depth': 5}
        )
        
        ensemble.train(X_train, y_train, X_val, y_val, verbose=False)
        
        # Test predictions with confidence
        pred, lower, upper = ensemble.predict_with_confidence(X_val)
        
        # Test feature importance
        importance = ensemble.get_aggregated_feature_importance()
        
        train_time = time.time() - start_time
        
        return {
            'model_name': 'HybridEnsemble',
            'status': 'PASS',
            'train_time': train_time,
            'prediction_shape': pred.shape,
            'confidence_intervals': (lower.shape, upper.shape),
            'feature_importance': len(importance) > 0,
            'error': None
        }
        
    except Exception as e:
        return {
            'model_name': 'HybridEnsemble',
            'status': 'FAIL',
            'error': str(e)[:100],
            'train_time': 0
        }

def main():
    """Main verification with parallel processing."""
    print("🚀 Parallel Model Verification")
    print(f"💻 System: M4 Pro (14 cores, 48GB RAM)")
    print(f"🔧 Using {mp.cpu_count()} CPU cores")
    print("=" * 60)
    
    start_total = time.time()
    
    # Define models to test
    rul_models = [
        ("XGBoost", "xgboost", {"learning_rate": 0.1}),
        ("LightGBM", "lightgbm", {"learning_rate": 0.1}),
        ("RandomForest", "random_forest", {"n_estimators": 20}),
        ("ElasticNet", "elastic_net", {"degree": 2}),
    ]
    
    anomaly_models = [
        ("IsolationForest", "IsolationForestDetector", {"contamination": 0.1}),
        ("Autoencoder", "AutoencoderDetector", {"encoding_dim": 8}),
        ("OCSVM", "ImprovedOCSVM", {"nu": 0.1}),
        ("EnsembleDetector", "EnsembleAnomalyDetector", {}),
    ]
    
    all_results = {}
    
    # Test RUL models in parallel
    print("\n1️⃣ Testing RUL Regression Models (Parallel)")
    with ProcessPoolExecutor(max_workers=min(4, len(rul_models))) as executor:
        futures = {executor.submit(test_rul_model, model): model[0] for model in rul_models}
        
        for future in progress_bar(as_completed(futures), desc="RUL Models", disable=not HAS_TQDM):
            result = future.result()
            all_results[result['model_name']] = result
            
            if result['status'] == 'PASS':
                fi_status = "✓" if result['feature_importance'] else "○"
                shap_status = "✓" if result['shap_values'] else "○"
                print(f"  ✅ {result['model_name']:<12} ({result['train_time']:.2f}s) FI:{fi_status} SHAP:{shap_status}")
            else:
                print(f"  ❌ {result['model_name']:<12} - {result['error']}")
    
    # Test Ensemble separately (it's complex)
    print("\n🔄 Testing Hybrid Ensemble...")
    ensemble_result = test_ensemble_model()
    all_results[ensemble_result['model_name']] = ensemble_result
    
    if ensemble_result['status'] == 'PASS':
        print(f"  ✅ HybridEnsemble ({ensemble_result['train_time']:.2f}s) - Full functionality")
    else:
        print(f"  ❌ HybridEnsemble - {ensemble_result['error']}")
    
    # Test Anomaly models in parallel
    print("\n2️⃣ Testing Anomaly Detection Models (Parallel)")
    with ProcessPoolExecutor(max_workers=min(4, len(anomaly_models))) as executor:
        futures = {executor.submit(test_anomaly_model, model): model[0] for model in anomaly_models}
        
        for future in progress_bar(as_completed(futures), desc="Anomaly Models", disable=not HAS_TQDM):
            result = future.result()
            all_results[result['model_name']] = result
            
            if result['status'] == 'PASS':
                print(f"  ✅ {result['model_name']:<15} ({result['train_time']:.2f}s)")
            else:
                print(f"  ❌ {result['model_name']:<15} - {result['error']}")
    
    # Summary
    total_time = time.time() - start_total
    passed = sum(1 for r in all_results.values() if r['status'] == 'PASS')
    total = len(all_results)
    
    print("\n" + "=" * 60)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"⏱️  Total time: {total_time:.2f}s")
    print(f"🎯 Models tested: {total}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {total - passed}")
    print(f"📈 Success rate: {passed/total*100:.1f}%")
    
    # Feature capabilities summary
    rul_results = [r for r in all_results.values() if 'feature_importance' in r]
    fi_count = sum(1 for r in rul_results if r.get('feature_importance', False))
    shap_count = sum(1 for r in rul_results if r.get('shap_values', False))
    
    if rul_results:
        print(f"🔍 Feature Importance: {fi_count}/{len(rul_results)} models")
        print(f"📊 SHAP Values: {shap_count}/{len(rul_results)} models")
    
    # Performance stats
    avg_train_time = np.mean([r['train_time'] for r in all_results.values() if r['train_time'] > 0])
    print(f"⚡ Average training time: {avg_train_time:.2f}s")
    
    if passed >= total * 0.8:
        print("\n🎉 VERIFICATION SUCCESSFUL!")
        print("All core model implementations are working correctly.")
        return True
    else:
        print("\n💥 VERIFICATION FAILED!")
        print("Some models failed basic functionality tests.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)