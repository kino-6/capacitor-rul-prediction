"""Script to train the Primary Model (anomaly detection)."""

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.primary_model import PrimaryModel


def main():
    """Train Primary Model on ES12 dataset."""
    print("=" * 80)
    print("PRIMARY MODEL TRAINING - ANOMALY DETECTION")
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
    y_train = train_df["is_abnormal"]
    
    X_val = val_df[feature_cols]
    y_val = val_df["is_abnormal"]
    
    X_test = test_df[feature_cols]
    y_test = test_df["is_abnormal"]
    
    print(f"  ✓ Features: {len(feature_cols)}")
    print(f"  ✓ Train labels - Normal: {(y_train == 0).sum()}, Abnormal: {(y_train == 1).sum()}")
    print(f"  ✓ Val labels   - Normal: {(y_val == 0).sum()}, Abnormal: {(y_val == 1).sum()}")
    print(f"  ✓ Test labels  - Normal: {(y_test == 0).sum()}, Abnormal: {(y_test == 1).sum()}")
    print()
    
    # Initialize model
    print("🤖 Initializing Primary Model...")
    model = PrimaryModel(model_type="random_forest")
    print("  ✓ Model type: Random Forest Classifier")
    print()
    
    # Train model
    print("🎓 Training model...")
    print("  Hyperparameters:")
    print("    - n_estimators: 100")
    print("    - max_depth: 10")
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
            max_depth=10,
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
        print(f"  {metric:15s}: {value:.4f}")
    print()
    
    # Display validation metrics
    print("📊 Validation Metrics:")
    print("-" * 80)
    for metric, value in metrics["val"].items():
        print(f"  {metric:15s}: {value:.4f}")
    print()
    
    # Evaluate on test set
    print("🧪 Evaluating on test set...")
    test_metrics = model.evaluate(X_test, y_test)
    
    print("📊 Test Metrics:")
    print("-" * 80)
    print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall:    {test_metrics['recall']:.4f}")
    print(f"  F1-Score:  {test_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    print()
    
    # Check if target is met
    target_f1 = 0.80
    if test_metrics['f1_score'] >= target_f1:
        print(f"🎯 Target achieved! F1-Score ({test_metrics['f1_score']:.4f}) >= {target_f1}")
    else:
        print(f"⚠️  Target not met. F1-Score ({test_metrics['f1_score']:.4f}) < {target_f1}")
    print()
    
    # Display confusion matrix
    print("📊 Confusion Matrix:")
    print("-" * 80)
    cm = test_metrics['confusion_matrix']
    print(f"  True Negative:  {cm[0][0]:4d}  |  False Positive: {cm[0][1]:4d}")
    print(f"  False Negative: {cm[1][0]:4d}  |  True Positive:  {cm[1][1]:4d}")
    print()
    
    # Display classification report
    print("📊 Classification Report:")
    print("-" * 80)
    print(test_metrics['classification_report'])
    
    # Save model
    print("💾 Saving model...")
    model_path = model_dir / "primary_model.pkl"
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
    importance_path = model_dir / "primary_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"  ✓ Feature importance saved to: {importance_path}")
    print()
    
    print("=" * 80)
    print("PRIMARY MODEL TRAINING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
