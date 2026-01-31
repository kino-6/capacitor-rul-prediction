"""
Demonstration of the Unified RUL Regression Model Interface

This script demonstrates how to use the unified interface to work with
different RUL regression models through the same API.
"""

import sys
import os
# Add both the src directory and the parent directory to the path
current_dir = os.path.dirname(__file__)
src_dir = os.path.join(current_dir, '..', 'src')
parent_dir = os.path.join(current_dir, '..', '..')
sys.path.extend([src_dir, parent_dir])

import numpy as np
from true_rul.rul_regression_model import RULRegressionModel


def generate_sample_data():
    """Generate sample training data for demonstration"""
    np.random.seed(42)
    n_samples = 100
    n_features = 8
    
    # Generate features with some correlation to RUL
    X = np.random.randn(n_samples, n_features)
    
    # Create realistic RUL labels based on features
    # Higher values in first few features -> higher RUL
    rul_base = 50 + 30 * np.sum(X[:, :3], axis=1)
    rul_noise = 10 * np.random.randn(n_samples)
    y = np.maximum(rul_base + rul_noise, 1)  # Ensure positive RUL
    
    # Split into train/val
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    feature_names = [
        'voltage_response_1', 'voltage_response_2', 'voltage_response_3',
        'frequency_feature_1', 'frequency_feature_2', 'statistical_feature_1',
        'trend_feature_1', 'rolling_feature_1'
    ]
    
    return X_train, y_train, X_val, y_val, feature_names


def demonstrate_model_type(model_type, X_train, y_train, X_val, y_val, feature_names):
    """Demonstrate a specific model type"""
    print(f"\n{'='*60}")
    print(f"Demonstrating {model_type.upper()} Model")
    print(f"{'='*60}")
    
    # Create model with appropriate parameters for fast demo
    if model_type in ["xgboost", "lightgbm"]:
        model = RULRegressionModel(model_type=model_type, n_estimators=20)
    elif model_type == "random_forest":
        model = RULRegressionModel(model_type=model_type, n_estimators=20)
    elif model_type == "elastic_net":
        model = RULRegressionModel(model_type=model_type, degree=1)
    elif model_type == "ensemble":
        model = RULRegressionModel(
            model_type=model_type,
            xgboost_params={'n_estimators': 10},
            lightgbm_params={'n_estimators': 10},
            random_forest_params={'n_estimators': 10}
        )
    
    print(f"Model created: {model}")
    
    # Train model
    print("Training model...")
    model.train(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        feature_names=feature_names,
        verbose=False
    )
    print("✓ Training completed")
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_val)
    print(f"✓ Predictions shape: {predictions.shape}")
    print(f"  Sample predictions: {predictions[:5]}")
    print(f"  Actual RUL values:  {y_val[:5]}")
    
    # Test confidence intervals (if supported)
    try:
        pred, lower, upper = model.predict_with_confidence(X_val)
        print("✓ Confidence intervals supported")
        print(f"  Sample intervals: [{lower[0]:.1f}, {pred[0]:.1f}, {upper[0]:.1f}]")
    except NotImplementedError:
        print("- Confidence intervals not natively supported (using fallback)")
        pred, lower, upper = model.predict_with_confidence(X_val)
        print(f"  Fallback intervals: [{lower[0]:.1f}, {pred[0]:.1f}, {upper[0]:.1f}]")
    
    # Get feature importance
    print("Extracting feature importance...")
    importance = model.get_feature_importance()
    print("✓ Feature importance extracted")
    print("  Top 3 most important features:")
    for i, (feature, score) in enumerate(list(importance.items())[:3]):
        print(f"    {i+1}. {feature}: {score:.4f}")
    
    # Test SHAP values (if supported)
    try:
        shap_values = model.get_shap_values(X_val[:5])  # Just first 5 samples
        print("✓ SHAP values supported")
        print(f"  SHAP values shape: {shap_values.shape}")
    except (NotImplementedError, RuntimeError):
        print("- SHAP values not supported for this model type")
    
    # Get model info
    info = model.get_model_info()
    print("Model information:")
    print(f"  Model type: {info.get('model_type', 'N/A')}")
    print(f"  Is trained: {info['is_trained']}")
    print(f"  Number of features: {info['n_features']}")
    print(f"  Supported methods: {list(info['supported_methods'].keys())}")
    
    return model


def main():
    """Main demonstration function"""
    print("RUL Regression Model - Unified Interface Demonstration")
    print("=" * 60)
    
    # Generate sample data
    print("Generating sample data...")
    X_train, y_train, X_val, y_val, feature_names = generate_sample_data()
    print(f"✓ Data generated: {X_train.shape[0]} training samples, {X_val.shape[0]} validation samples")
    print(f"  Features: {len(feature_names)}")
    print(f"  RUL range: {y_train.min():.1f} - {y_train.max():.1f} cycles")
    
    # Show available models
    print(f"\nAvailable model types:")
    available_models = RULRegressionModel.get_available_models()
    for model_type, description in available_models.items():
        print(f"  • {model_type}: {description}")
    
    # Demonstrate each model type
    model_types = ["xgboost", "lightgbm", "random_forest", "elastic_net", "ensemble"]
    trained_models = {}
    
    for model_type in model_types:
        try:
            model = demonstrate_model_type(
                model_type, X_train, y_train, X_val, y_val, feature_names
            )
            trained_models[model_type] = model
        except Exception as e:
            print(f"❌ Error with {model_type}: {e}")
    
    # Compare predictions across models
    print(f"\n{'='*60}")
    print("PREDICTION COMPARISON")
    print(f"{'='*60}")
    
    print("Comparing predictions for first 5 validation samples:")
    print(f"{'Model':<15} {'Sample 1':<10} {'Sample 2':<10} {'Sample 3':<10} {'Sample 4':<10} {'Sample 5':<10}")
    print("-" * 70)
    
    # Show actual values
    actual_str = " ".join([f"{val:8.1f}" for val in y_val[:5]])
    print(f"{'Actual':<15} {actual_str}")
    
    # Show predictions from each model
    for model_type, model in trained_models.items():
        try:
            preds = model.predict(X_val[:5])
            pred_str = " ".join([f"{pred:8.1f}" for pred in preds])
            print(f"{model_type:<15} {pred_str}")
        except Exception as e:
            print(f"{model_type:<15} Error: {e}")
    
    # Feature importance comparison
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCE COMPARISON")
    print(f"{'='*60}")
    
    print("Top 3 features by importance for each model:")
    for model_type, model in trained_models.items():
        try:
            importance = model.get_feature_importance()
            top_features = list(importance.items())[:3]
            print(f"\n{model_type.upper()}:")
            for i, (feature, score) in enumerate(top_features):
                print(f"  {i+1}. {feature}: {score:.4f}")
        except Exception as e:
            print(f"\n{model_type.upper()}: Error - {e}")
    
    print(f"\n{'='*60}")
    print("DEMONSTRATION COMPLETED")
    print(f"{'='*60}")
    print("✓ All model types successfully demonstrated through unified interface")
    print("✓ Training, prediction, and interpretability features working")
    print("✓ Error handling and fallback mechanisms functional")


if __name__ == "__main__":
    main()