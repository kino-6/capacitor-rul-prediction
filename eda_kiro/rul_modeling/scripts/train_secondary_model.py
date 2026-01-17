"""Script to train the Secondary Model (RUL prediction)."""

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.secondary_model import SecondaryModel


def main():
    """Train Secondary Model on ES12 dataset."""
    print("=" * 80)
    print("SECONDARY MODEL TRAINING - RUL PREDICTION")
    print("=" * 80)
    print()
    
    # Paths
    data_dir = Path(__file__).parent.parent / "output" / "features"
    model_dir = Path(__file__).parent.parent / "output" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    print("📂 Loading datasets...")
    train_df = pd.read_csv(data_dir / "train.csv")
    val_df = pd.read_csv(data_dir / "val.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    print(f"  ✓ Train: {len(train_df)} samples")
    print(f"  ✓ Val:   {len(val_df)} samples")
    print(f"  ✓ Test:  {len(test_df)} samples")
    print()
    
    # Prepare features and labels
    print("🔧 Preparing features and labels...")
    
    # Feature columns (exclude metadata)
    feature_cols = [col for col in train_df.columns 
                   if col not in ["capacitor_id", "cycle", "is_abnormal", "rul"]]
    
    X_train = train_df[feature_cols]
    y_train = train_df["rul"]
    
    X_val = val_df[feature_cols]
    y_val = val_df["rul"]
    
    X_test = test_df[feature_cols]
    y_test = test_df["rul"]
    
    print(f"  ✓ Features: {len(feature_cols)}")
    print(f"  ✓ Train RUL - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}, Range: [{y_train.min()}, {y_train.max()}]")
    print(f"  ✓ Val RUL   - Mean: {y_val.mean():.2f}, Std: {y_val.std():.2f}, Range: [{y_val.min()}, {y_val.max()}]")
    print(f"  ✓ Test RUL  - Mean: {y_test.mean():.2f}, Std: {y_test.std():.2f}, Range: [{y_test.min()}, {y_test.max()}]")
    print()
    
    # Initialize model
    print("🤖 Initializing Secondary Model...")
    model = SecondaryModel(model_type="random_forest")
    print("  ✓ Model type: Random Forest Regressor")
    print()
    
    # Train model
    print("🎓 Training model...")
    print("  Hyperparameters:")
    print("    - n_estimators: 100")
    print("    - max_depth: 15")
    print("    - min_samples_split: 5")
    print("    - min_samples_leaf: 2")
    print("    - random_state: 42")
    print()
    
    # Show progress bar during training
    with tqdm(total=100, desc="Training", unit="tree") as pbar:
        metrics = model.train(
            X_train, y_train,
            X_val, y_val,
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        pbar.update(100)
    
    print()
    print("✅ Training complete!")
    print()
    
    # Display training metrics
    print("📊 Training Metrics:")
    print("-" * 80)
    for metric, value in metrics["train"].items():
        print(f"  {metric:10s}: {value:.4f}")
    print()
    
    # Display validation metrics
    print("📊 Validation Metrics:")
    print("-" * 80)
    for metric, value in metrics["val"].items():
        print(f"  {metric:10s}: {value:.4f}")
    print()
    
    # Evaluate on test set
    print("🧪 Evaluating on test set...")
    test_metrics = model.evaluate(X_test, y_test)
    
    print("📊 Test Metrics:")
    print("-" * 80)
    print(f"  MAE:   {test_metrics['mae']:.4f}")
    print(f"  RMSE:  {test_metrics['rmse']:.4f}")
    print(f"  R²:    {test_metrics['r2']:.4f}")
    print(f"  MAPE:  {test_metrics['mape']:.2f}%")
    print()
    
    # Check if target is met
    target_mape = 20.0
    if test_metrics['mape'] <= target_mape:
        print(f"🎯 Target achieved! MAPE ({test_metrics['mape']:.2f}%) <= {target_mape}%")
    else:
        print(f"⚠️  Target not met. MAPE ({test_metrics['mape']:.2f}%) > {target_mape}%")
    print()
    
    # Save model
    print("💾 Saving model...")
    model_path = model_dir / "secondary_model.pkl"
    model.save(model_path)
    print(f"  ✓ Model saved to: {model_path}")
    print()
    
    # Display feature importance (top 10)
    print("📊 Top 10 Feature Importance:")
    print("-" * 80)
    importance_df = model.get_feature_importance()
    for idx, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:25s}: {row['importance']:.4f}")
    print()
    
    # Save feature importance
    importance_path = model_dir / "secondary_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"  ✓ Feature importance saved to: {importance_path}")
    print()
    
    # Save predictions for visualization
    print("💾 Saving predictions...")
    predictions = model.predict(X_test)
    predictions_df = pd.DataFrame({
        "capacitor_id": test_df["capacitor_id"],
        "cycle": test_df["cycle"],
        "actual_rul": y_test,
        "predicted_rul": predictions,
        "error": predictions - y_test,
        "abs_error": abs(predictions - y_test),
        "pct_error": abs((predictions - y_test) / y_test) * 100
    })
    
    predictions_path = model_dir / "secondary_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"  ✓ Predictions saved to: {predictions_path}")
    print()
    
    print("=" * 80)
    print("SECONDARY MODEL TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
