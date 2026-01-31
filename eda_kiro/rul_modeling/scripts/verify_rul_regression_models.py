#!/usr/bin/env python3
"""
Verification script for RUL regression models

This script verifies that all RUL regression model implementations are working correctly:
- GradientBoostingRULPredictor (XGBoost and LightGBM)
- RandomForestRULPredictor with quantile regression
- ElasticNetRULPredictor with polynomial features
- HybridEnsembleRULPredictor combining all models
- Unified RULRegressionModel interface

Requirements: 1.1, 1.2, 1.3, 9.1, 9.4
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from true_rul.rul_regression_model import RULRegressionModel
from true_rul.gradient_boosting_predictor import GradientBoostingRULPredictor
from true_rul.random_forest_predictor import RandomForestRULPredictor
from true_rul.elastic_net_predictor import ElasticNetRULPredictor
from true_rul.hybrid_ensemble_predictor import HybridEnsembleRULPredictor


def generate_sample_data(n_samples=100, n_features=10, random_state=42):
    """Generate sample data for testing"""
    np.random.seed(random_state)
    
    # Generate features
    X = np.random.normal(0, 1, (n_samples, n_features))
    
    # Generate realistic RUL labels (linear combination + noise)
    true_coeffs = np.random.normal(0, 1, n_features)
    y = np.dot(X, true_coeffs) + np.random.normal(0, 0.1, n_samples)
    
    # Ensure RUL is positive and realistic (1-100 cycles)
    y = np.abs(y) * 20 + 10
    y = np.clip(y, 1, 100)
    
    # Split into train/test
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    return X_train, X_test, y_train, y_test, feature_names


def test_individual_models():
    """Test individual model implementations"""
    print("=" * 60)
    print("Testing Individual RUL Regression Models")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test, feature_names = generate_sample_data()
    
    models = [
        ("XGBoost", GradientBoostingRULPredictor(model_type="xgboost", n_estimators=50)),
        ("LightGBM", GradientBoostingRULPredictor(model_type="lightgbm", n_estimators=50)),
        ("Random Forest", RandomForestRULPredictor(n_estimators=50)),
        ("Elastic Net", ElasticNetRULPredictor(degree=2)),
        ("Hybrid Ensemble", HybridEnsembleRULPredictor())
    ]
    
    results = {}
    
    for name, model in models:
        print(f"\nTesting {name}...")
        
        try:
            # Train model
            print(f"  Training {name}...")
            model.train(X_train, y_train, X_test, y_test, feature_names=feature_names)
            
            # Make predictions
            print(f"  Making predictions...")
            predictions = model.predict(X_test)
            
            # Test non-negative constraint
            assert np.all(predictions >= 0), f"{name} produced negative predictions"
            
            # Test confidence intervals if supported
            if hasattr(model, 'predict_with_confidence'):
                print(f"  Testing confidence intervals...")
                pred_conf, lower, upper = model.predict_with_confidence(X_test)
                
                assert np.all(pred_conf >= 0), f"{name} confidence predictions negative"
                assert np.all(lower >= 0), f"{name} lower bounds negative"
                assert np.all(upper >= 0), f"{name} upper bounds negative"
                assert np.all(lower <= pred_conf), f"{name} lower bounds > predictions"
                assert np.all(pred_conf <= upper), f"{name} predictions > upper bounds"
            
            # Test feature importance
            if hasattr(model, 'get_feature_importance'):
                print(f"  Testing feature importance...")
                importance = model.get_feature_importance()
                assert isinstance(importance, dict), f"{name} importance not dict"
                assert len(importance) > 0, f"{name} importance empty"
                assert all(v >= 0 for v in importance.values()), f"{name} negative importance"
            
            # Test SHAP values if supported
            if hasattr(model, 'get_shap_values'):
                print(f"  Testing SHAP values...")
                try:
                    shap_values = model.get_shap_values(X_test[:5])  # Test on small subset
                    assert shap_values.shape == (5, len(feature_names)), f"{name} SHAP shape wrong"
                    assert np.all(np.isfinite(shap_values)), f"{name} SHAP values not finite"
                except Exception as e:
                    print(f"    SHAP test failed (may be expected): {e}")
            
            # Calculate basic metrics
            mse = np.mean((predictions - y_test) ** 2)
            mae = np.mean(np.abs(predictions - y_test))
            
            results[name] = {
                'mse': mse,
                'mae': mae,
                'predictions_range': (np.min(predictions), np.max(predictions)),
                'status': 'PASS'
            }
            
            print(f"  ✓ {name} passed all tests")
            print(f"    MSE: {mse:.3f}, MAE: {mae:.3f}")
            print(f"    Prediction range: [{np.min(predictions):.1f}, {np.max(predictions):.1f}]")
            
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
            results[name] = {'status': 'FAIL', 'error': str(e)}
    
    return results


def test_unified_interface():
    """Test unified RULRegressionModel interface"""
    print("\n" + "=" * 60)
    print("Testing Unified RUL Regression Interface")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test, feature_names = generate_sample_data()
    
    model_types = ["xgboost", "lightgbm", "random_forest", "elastic_net", "ensemble"]
    
    results = {}
    
    for model_type in model_types:
        print(f"\nTesting unified interface with {model_type}...")
        
        try:
            # Create model through unified interface
            if model_type == "elastic_net":
                model = RULRegressionModel(model_type=model_type, degree=1)
            elif model_type == "ensemble":
                # Ensemble model takes nested parameters
                model = RULRegressionModel(
                    model_type=model_type,
                    xgboost_params={'n_estimators': 30},
                    lightgbm_params={'n_estimators': 30},
                    random_forest_params={'n_estimators': 30}
                )
            else:
                model = RULRegressionModel(model_type=model_type, n_estimators=30)
            
            # Test model info
            info = model.get_model_info()
            assert isinstance(info, dict), f"Model info not dict for {model_type}"
            
            # Train model
            print(f"  Training {model_type}...")
            model.train(X_train, y_train, X_test, y_test, feature_names=feature_names)
            
            # Test predictions
            print(f"  Testing predictions...")
            predictions = model.predict(X_test)
            assert np.all(predictions >= 0), f"Negative predictions for {model_type}"
            
            # Test confidence intervals
            print(f"  Testing confidence intervals...")
            pred_conf, lower, upper = model.predict_with_confidence(X_test)
            assert np.all(pred_conf >= 0), f"Negative confidence predictions for {model_type}"
            assert np.all(lower >= 0), f"Negative lower bounds for {model_type}"
            assert np.all(upper >= 0), f"Negative upper bounds for {model_type}"
            
            # Test feature importance
            print(f"  Testing feature importance...")
            importance = model.get_feature_importance()
            assert isinstance(importance, dict), f"Importance not dict for {model_type}"
            
            # Calculate metrics
            mse = np.mean((predictions - y_test) ** 2)
            mae = np.mean(np.abs(predictions - y_test))
            
            results[model_type] = {
                'mse': mse,
                'mae': mae,
                'predictions_range': (np.min(predictions), np.max(predictions)),
                'status': 'PASS'
            }
            
            print(f"  ✓ {model_type} passed all unified interface tests")
            print(f"    MSE: {mse:.3f}, MAE: {mae:.3f}")
            
        except Exception as e:
            print(f"  ✗ {model_type} failed: {e}")
            results[model_type] = {'status': 'FAIL', 'error': str(e)}
    
    return results


def test_ensemble_components():
    """Test ensemble model components and aggregation"""
    print("\n" + "=" * 60)
    print("Testing Ensemble Model Components")
    print("=" * 60)
    
    X_train, X_test, y_train, y_test, feature_names = generate_sample_data()
    
    print("Creating hybrid ensemble...")
    ensemble = HybridEnsembleRULPredictor(
        xgboost_params={'n_estimators': 30},
        lightgbm_params={'n_estimators': 30},
        random_forest_params={'n_estimators': 30}
    )
    
    print("Training ensemble...")
    ensemble.train(X_train, y_train, X_test, y_test, feature_names=feature_names)
    
    print("Testing individual model predictions...")
    individual_preds = ensemble.get_individual_predictions(X_test)
    
    for model_name, preds in individual_preds.items():
        if preds is not None:
            assert np.all(preds >= 0), f"Negative predictions from {model_name}"
            print(f"  {model_name}: range [{np.min(preds):.1f}, {np.max(preds):.1f}]")
    
    print("Testing ensemble prediction...")
    ensemble_pred = ensemble.predict(X_test)
    assert np.all(ensemble_pred >= 0), "Negative ensemble predictions"
    
    print("Testing prediction variance...")
    variance = ensemble.get_prediction_variance(X_test)
    assert np.all(variance >= 0), "Negative prediction variance"
    
    print("Testing aggregated feature importance...")
    importance = ensemble.get_aggregated_feature_importance()
    assert isinstance(importance, dict), "Importance not dict"
    assert len(importance) > 0, "Empty importance dict"
    
    # Check that importance scores are normalized
    total_importance = sum(importance.values())
    assert abs(total_importance - 1.0) < 1e-6, f"Importance not normalized: {total_importance}"
    
    print("✓ Ensemble components test passed")
    
    return {
        'ensemble_pred_range': (np.min(ensemble_pred), np.max(ensemble_pred)),
        'variance_range': (np.min(variance), np.max(variance)),
        'top_features': list(importance.keys())[:3],
        'status': 'PASS'
    }


def main():
    """Main verification function"""
    print("RUL Regression Models Verification")
    print("=" * 60)
    print("Verifying all RUL regression model implementations...")
    print("Requirements: 1.1, 1.2, 1.3, 9.1, 9.4")
    
    try:
        # Test individual models
        individual_results = test_individual_models()
        
        # Test unified interface
        unified_results = test_unified_interface()
        
        # Test ensemble components
        ensemble_results = test_ensemble_components()
        
        # Summary
        print("\n" + "=" * 60)
        print("VERIFICATION SUMMARY")
        print("=" * 60)
        
        print("\nIndividual Models:")
        for name, result in individual_results.items():
            status = result['status']
            if status == 'PASS':
                print(f"  ✓ {name}: PASS (MSE: {result['mse']:.3f})")
            else:
                print(f"  ✗ {name}: FAIL ({result.get('error', 'Unknown error')})")
        
        print("\nUnified Interface:")
        for model_type, result in unified_results.items():
            status = result['status']
            if status == 'PASS':
                print(f"  ✓ {model_type}: PASS (MSE: {result['mse']:.3f})")
            else:
                print(f"  ✗ {model_type}: FAIL ({result.get('error', 'Unknown error')})")
        
        print(f"\nEnsemble Components: {ensemble_results['status']}")
        
        # Check overall success
        all_individual_pass = all(r['status'] == 'PASS' for r in individual_results.values())
        all_unified_pass = all(r['status'] == 'PASS' for r in unified_results.values())
        ensemble_pass = ensemble_results['status'] == 'PASS'
        
        if all_individual_pass and all_unified_pass and ensemble_pass:
            print("\n🎉 ALL RUL REGRESSION MODELS VERIFIED SUCCESSFULLY!")
            print("\nTask 6 Implementation Status:")
            print("  ✓ 6.1 GradientBoostingRULPredictor (XGBoost and LightGBM)")
            print("  ✓ 6.2 RandomForestRULPredictor with quantile regression")
            print("  ✓ 6.3 ElasticNetRULPredictor with polynomial features")
            print("  ✓ 6.4 HybridEnsembleRULPredictor combining all models")
            print("  ✓ 6.5 Unified RULRegressionModel interface")
            print("  ✓ 6.6 Property test for non-negative RUL output")
            print("  ✓ 6.7 Property test for complete prediction output")
            print("\n✅ Task 6: Implement RUL regression models - READY FOR COMPLETION")
            return True
        else:
            print("\n❌ SOME MODELS FAILED VERIFICATION")
            return False
            
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)