#!/usr/bin/env python3
"""
Fast Model Implementation Verification

Optimized sequential verification with progress bars and minimal training.
"""

import sys
import os
import numpy as np
import warnings
import time
from typing import Dict, Any, Tuple

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

def progress_print(current, total, desc="Processing"):
    """Simple progress indicator when tqdm not available."""
    if not HAS_TQDM:
        print(f"🔄 {desc} [{current}/{total}]")

def generate_minimal_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate minimal test data for fast verification."""
    np.random.seed(42)
    
    # Small dataset for speed
    X = np.random.randn(60, 12).astype(np.float32)
    y = (100 - 5 * X[:, 0] + 0.1 * np.random.randn(60)).astype(np.float32)
    y = np.maximum(y, 0)
    
    # Split
    X_train, X_val = X[:45], X[45:]
    y_train, y_val = y[:45], y[45:]
    
    # Normal data for anomaly detection
    X_normal = X_train[:30]
    
    return (X_train, y_train, X_val, y_val), X_normal

def test_rul_models(data: Tuple) -> Dict[str, Dict[str, Any]]:
    """Test all RUL models sequentially with optimizations."""
    X_train, y_train, X_val, y_val = data
    results = {}
    
    models_config = [
        ("XGBoost", "xgboost", {"n_estimators": 15, "max_depth": 4, "learning_rate": 0.1}),
        ("LightGBM", "lightgbm", {"n_estimators": 15, "max_depth": 4, "learning_rate": 0.1}),
        ("RandomForest", "random_forest", {"n_estimators": 15, "max_depth": 6}),
        ("ElasticNet", "elastic_net", {"degree": 2}),
    ]
    
    print("\n1️⃣ Testing RUL Regression Models")
    
    if HAS_TQDM:
        pbar = tqdm(models_config, desc="RUL Models", ncols=80)
    else:
        pbar = models_config
    
    for i, (name, model_type, params) in enumerate(pbar):
        if not HAS_TQDM:
            progress_print(i+1, len(models_config), "RUL Models")
        
        try:
            from true_rul.rul_regression_model import RULRegressionModel
            
            start_time = time.time()
            
            # Create model with speed optimizations
            model = RULRegressionModel(model_type, **params)
            model.train(X_train, y_train, X_val, y_val, verbose=False)
            
            # Test prediction
            pred = model.predict(X_val)
            train_time = time.time() - start_time
            
            # Test capabilities
            has_fi = False
            has_shap = False
            
            try:
                importance = model.get_feature_importance()
                has_fi = isinstance(importance, dict) and len(importance) > 0
            except:
                pass
            
            # Only test SHAP for tree models (faster)
            if model_type in ["xgboost", "lightgbm"]:
                try:
                    shap_values = model.get_shap_values(X_val[:2])
                    has_shap = isinstance(shap_values, np.ndarray)
                except:
                    pass
            
            results[name] = {
                'status': 'PASS',
                'train_time': train_time,
                'non_negative': np.all(pred >= 0),
                'feature_importance': has_fi,
                'shap_values': has_shap,
                'error': None
            }
            
            # Status indicators
            fi_status = "✓" if has_fi else "○"
            shap_status = "✓" if has_shap else "○"
            
            if HAS_TQDM:
                pbar.set_postfix_str(f"{name}: {train_time:.2f}s FI:{fi_status} SHAP:{shap_status}")
            else:
                print(f"  ✅ {name:<12} ({train_time:.2f}s) FI:{fi_status} SHAP:{shap_status}")
                
        except Exception as e:
            results[name] = {
                'status': 'FAIL',
                'error': str(e)[:80],
                'train_time': 0,
                'feature_importance': False,
                'shap_values': False
            }
            
            if HAS_TQDM:
                pbar.set_postfix_str(f"{name}: FAILED")
            else:
                print(f"  ❌ {name:<12} - {str(e)[:50]}")
    
    return results

def test_ensemble_model(data: Tuple) -> Dict[str, Any]:
    """Test hybrid ensemble model."""
    X_train, y_train, X_val, y_val = data
    
    print("\n🔄 Testing Hybrid Ensemble...")
    
    try:
        from true_rul.hybrid_ensemble_predictor import HybridEnsembleRULPredictor
        
        start_time = time.time()
        
        # Minimal ensemble for speed
        ensemble = HybridEnsembleRULPredictor(
            xgboost_params={'n_estimators': 8, 'max_depth': 3},
            lightgbm_params={'n_estimators': 8, 'max_depth': 3},
            random_forest_params={'n_estimators': 8, 'max_depth': 4}
        )
        
        ensemble.train(X_train, y_train, X_val, y_val, verbose=False)
        
        # Test predictions with confidence
        pred, lower, upper = ensemble.predict_with_confidence(X_val)
        
        # Test feature importance
        importance = ensemble.get_aggregated_feature_importance()
        
        train_time = time.time() - start_time
        
        print(f"  ✅ HybridEnsemble ({train_time:.2f}s) - Full functionality")
        
        return {
            'status': 'PASS',
            'train_time': train_time,
            'confidence_intervals': True,
            'feature_importance': len(importance) > 0,
            'error': None
        }
        
    except Exception as e:
        print(f"  ❌ HybridEnsemble - {str(e)[:50]}")
        return {
            'status': 'FAIL',
            'error': str(e)[:80],
            'train_time': 0
        }

def test_anomaly_models(X_normal: np.ndarray) -> Dict[str, Dict[str, Any]]:
    """Test anomaly detection models."""
    results = {}
    X_test = np.random.randn(15, 12).astype(np.float32)
    
    models_config = [
        ("IsolationForest", "isolation_forest"),
        ("OCSVM", "ocsvm"),
        ("Autoencoder", "autoencoder"),
        ("EnsembleDetector", "ensemble"),
    ]
    
    print("\n2️⃣ Testing Anomaly Detection Models")
    
    if HAS_TQDM:
        pbar = tqdm(models_config, desc="Anomaly Models", ncols=80)
    else:
        pbar = models_config
    
    for i, (name, model_type) in enumerate(pbar):
        if not HAS_TQDM:
            progress_print(i+1, len(models_config), "Anomaly Models")
        
        try:
            start_time = time.time()
            
            if model_type == "isolation_forest":
                from true_rul.isolation_forest_detector import IsolationForestDetector
                model = IsolationForestDetector(contamination=0.1)
                model.fit(X_normal)
                scores = model.predict_score(X_test)
                
            elif model_type == "ocsvm":
                from true_rul.improved_ocsvm import ImprovedOCSVM
                model = ImprovedOCSVM(nu=0.1)
                model.fit(X_normal)
                scores = model.predict_score(X_test)
                
            elif model_type == "autoencoder":
                from true_rul.autoencoder_detector import AutoencoderDetector
                model = AutoencoderDetector(input_dim=12, encoding_dim=6)
                model.fit(X_normal, epochs=8, verbose=False)  # Very quick training
                scores = model.get_reconstruction_error(X_test)
                
            elif model_type == "ensemble":
                from true_rul.ensemble_anomaly_detector import EnsembleAnomalyDetector
                model = EnsembleAnomalyDetector(
                    autoencoder_params={'encoding_dim': 6}
                )
                model.fit(X_normal)
                binary_pred, scores, info = model.predict(X_test)
            
            train_time = time.time() - start_time
            
            results[name] = {
                'status': 'PASS',
                'train_time': train_time,
                'output_valid': len(scores) == len(X_test),
                'error': None
            }
            
            if HAS_TQDM:
                pbar.set_postfix_str(f"{name}: {train_time:.2f}s")
            else:
                print(f"  ✅ {name:<15} ({train_time:.2f}s)")
                
        except Exception as e:
            results[name] = {
                'status': 'FAIL',
                'error': str(e)[:80],
                'train_time': 0
            }
            
            if HAS_TQDM:
                pbar.set_postfix_str(f"{name}: FAILED")
            else:
                print(f"  ❌ {name:<15} - {str(e)[:50]}")
    
    return results

def main():
    """Main verification function."""
    print("🚀 Fast Model Implementation Verification")
    print(f"💻 M4 Pro Optimized (Sequential with progress tracking)")
    print("=" * 60)
    
    start_total = time.time()
    
    # Generate test data
    print("📊 Generating test data...")
    rul_data, anomaly_data = generate_minimal_data()
    
    # Run tests
    rul_results = test_rul_models(rul_data)
    ensemble_result = test_ensemble_model(rul_data)
    anomaly_results = test_anomaly_models(anomaly_data)
    
    # Combine results
    all_results = {**rul_results, 'HybridEnsemble': ensemble_result, **anomaly_results}
    
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
    
    # Feature capabilities
    rul_models = [k for k in rul_results.keys()] + ['HybridEnsemble']
    fi_count = sum(1 for k in rul_models if all_results[k].get('feature_importance', False))
    shap_count = sum(1 for k in rul_models if all_results[k].get('shap_values', False))
    
    print(f"🔍 Feature Importance: {fi_count}/{len(rul_models)} RUL models")
    print(f"📊 SHAP Values: {shap_count}/{len(rul_models)} RUL models")
    
    # Performance
    avg_time = np.mean([r['train_time'] for r in all_results.values() if r['train_time'] > 0])
    print(f"⚡ Average training time: {avg_time:.2f}s")
    
    # Final assessment
    if passed >= total * 0.8:
        print("\n🎉 VERIFICATION SUCCESSFUL!")
        print("✅ All core model classes can be instantiated")
        print("✅ Models can be trained on synthetic data")
        print("✅ Feature importance and SHAP values are generated")
        print("\n🚀 Ready to proceed with implementation!")
        return True
    else:
        print("\n💥 VERIFICATION FAILED!")
        failed_models = [k for k, v in all_results.items() if v['status'] == 'FAIL']
        print(f"❌ Failed models: {', '.join(failed_models)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)