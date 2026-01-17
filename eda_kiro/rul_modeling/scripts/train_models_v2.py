"""
Model Training Script (Version 2)

Train Primary and Secondary models with data leakage eliminated dataset.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

# Paths
DATA_DIR = Path("output/features_v2")
MODELS_DIR = Path("output/models_v2")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_datasets():
    """Load train, val, test datasets"""
    print("Loading datasets...")
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    
    print(f"  Train: {len(train)} samples")
    print(f"  Val: {len(val)} samples")
    print(f"  Test: {len(test)} samples")
    
    return train, val, test

def prepare_data(df):
    """Prepare features and labels"""
    feature_cols = [col for col in df.columns 
                   if col not in ['capacitor_id', 'cycle', 'is_abnormal', 'rul']]
    
    X = df[feature_cols]
    y_classification = df['is_abnormal']
    y_regression = df['rul']
    
    return X, y_classification, y_regression, feature_cols

def train_primary_model(X_train, y_train, X_val, y_val):
    """Train Primary Model (Anomaly Detection)"""
    print("\n" + "="*80)
    print("TRAINING PRIMARY MODEL (ANOMALY DETECTION)")
    print("="*80)
    
    # Initialize model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("\nTraining...")
    model.fit(X_train, y_train)
    print("  ✓ Training complete")
    
    # Evaluate on train set
    print("\nTrain Set Performance:")
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)[:, 1]
    
    train_acc = accuracy_score(y_train, y_train_pred)
    train_prec = precision_score(y_train, y_train_pred, zero_division=0)
    train_rec = recall_score(y_train, y_train_pred, zero_division=0)
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    train_auc = roc_auc_score(y_train, y_train_proba)
    
    print(f"  Accuracy:  {train_acc:.4f}")
    print(f"  Precision: {train_prec:.4f}")
    print(f"  Recall:    {train_rec:.4f}")
    print(f"  F1-Score:  {train_f1:.4f}")
    print(f"  ROC-AUC:   {train_auc:.4f}")
    
    # Evaluate on validation set
    print("\nValidation Set Performance:")
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]
    
    val_acc = accuracy_score(y_val, y_val_pred)
    val_prec = precision_score(y_val, y_val_pred, zero_division=0)
    val_rec = recall_score(y_val, y_val_pred, zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
    val_auc = roc_auc_score(y_val, y_val_proba)
    
    print(f"  Accuracy:  {val_acc:.4f}")
    print(f"  Precision: {val_prec:.4f}")
    print(f"  Recall:    {val_rec:.4f}")
    print(f"  F1-Score:  {val_f1:.4f}")
    print(f"  ROC-AUC:   {val_auc:.4f}")
    
    # Check for overfitting
    f1_diff = abs(train_f1 - val_f1)
    print(f"\nOverfitting Check:")
    print(f"  Train F1-Score: {train_f1:.4f}")
    print(f"  Val F1-Score:   {val_f1:.4f}")
    print(f"  Difference:     {f1_diff:.4f}")
    
    if f1_diff < 0.05:
        print("  ✅ Good generalization (difference < 5%)")
    elif f1_diff < 0.10:
        print("  ⚠️ Acceptable generalization (difference < 10%)")
    else:
        print("  ❌ Potential overfitting (difference >= 10%)")
    
    # Save model
    model_data = {
        'model': model,
        'model_type': 'RandomForestClassifier',
        'feature_names': list(X_train.columns),
        'train_metrics': {
            'accuracy': train_acc,
            'precision': train_prec,
            'recall': train_rec,
            'f1_score': train_f1,
            'roc_auc': train_auc
        },
        'val_metrics': {
            'accuracy': val_acc,
            'precision': val_prec,
            'recall': val_rec,
            'f1_score': val_f1,
            'roc_auc': val_auc
        }
    }
    
    model_path = MODELS_DIR / "primary_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\n✓ Saved model: {model_path}")
    
    # Save feature importance
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_path = MODELS_DIR / "primary_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"✓ Saved feature importance: {importance_path}")
    
    print("\nTop 10 Important Features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return model, model_data

def train_secondary_model(X_train, y_train, X_val, y_val):
    """Train Secondary Model (RUL Prediction)"""
    print("\n" + "="*80)
    print("TRAINING SECONDARY MODEL (RUL PREDICTION)")
    print("="*80)
    
    # Initialize model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    print("\nTraining...")
    model.fit(X_train, y_train)
    print("  ✓ Training complete")
    
    # Evaluate on train set
    print("\nTrain Set Performance:")
    y_train_pred = model.predict(X_train)
    
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    print(f"  MAE:  {train_mae:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  R²:   {train_r2:.4f}")
    
    # Evaluate on validation set
    print("\nValidation Set Performance:")
    y_val_pred = model.predict(X_val)
    
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"  MAE:  {val_mae:.4f}")
    print(f"  RMSE: {val_rmse:.4f}")
    print(f"  R²:   {val_r2:.4f}")
    
    # Check for overfitting
    mae_diff_pct = abs(val_mae - train_mae) / train_mae * 100
    print(f"\nOverfitting Check:")
    print(f"  Train MAE: {train_mae:.4f}")
    print(f"  Val MAE:   {val_mae:.4f}")
    print(f"  Difference: {mae_diff_pct:.1f}%")
    
    if mae_diff_pct < 10:
        print("  ✅ Good generalization (difference < 10%)")
    elif mae_diff_pct < 20:
        print("  ⚠️ Acceptable generalization (difference < 20%)")
    else:
        print("  ❌ Potential overfitting (difference >= 20%)")
    
    # Save model
    model_data = {
        'model': model,
        'model_type': 'RandomForestRegressor',
        'feature_names': list(X_train.columns),
        'train_metrics': {
            'mae': train_mae,
            'rmse': train_rmse,
            'r2': train_r2
        },
        'val_metrics': {
            'mae': val_mae,
            'rmse': val_rmse,
            'r2': val_r2
        }
    }
    
    model_path = MODELS_DIR / "secondary_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"\n✓ Saved model: {model_path}")
    
    # Save feature importance
    importance_df = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    importance_path = MODELS_DIR / "secondary_feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"✓ Saved feature importance: {importance_path}")
    
    print("\nTop 10 Important Features:")
    for i, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return model, model_data

def evaluate_on_test(primary_model, secondary_model, test, feature_cols):
    """Quick evaluation on test set"""
    print("\n" + "="*80)
    print("TEST SET EVALUATION (QUICK CHECK)")
    print("="*80)
    
    X_test = test[feature_cols]
    y_test_class = test['is_abnormal']
    y_test_reg = test['rul']
    
    # Primary model
    print("\nPrimary Model (Anomaly Detection):")
    y_pred_class = primary_model.predict(X_test)
    y_proba_class = primary_model.predict_proba(X_test)[:, 1]
    
    test_acc = accuracy_score(y_test_class, y_pred_class)
    test_f1 = f1_score(y_test_class, y_pred_class, zero_division=0)
    test_auc = roc_auc_score(y_test_class, y_proba_class)
    
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  F1-Score:  {test_f1:.4f}")
    print(f"  ROC-AUC:   {test_auc:.4f}")
    
    # Secondary model
    print("\nSecondary Model (RUL Prediction):")
    y_pred_reg = secondary_model.predict(X_test)
    
    test_mae = mean_absolute_error(y_test_reg, y_pred_reg)
    test_rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    test_r2 = r2_score(y_test_reg, y_pred_reg)
    
    print(f"  MAE:  {test_mae:.4f}")
    print(f"  RMSE: {test_rmse:.4f}")
    print(f"  R²:   {test_r2:.4f}")
    
    # RUL range analysis
    print("\nRUL Range Analysis:")
    test_with_pred = test.copy()
    test_with_pred['pred_rul'] = y_pred_reg
    test_with_pred['abs_error'] = np.abs(y_pred_reg - y_test_reg)
    
    ranges = [(0, 50), (50, 100), (100, 150), (150, 200)]
    for start, end in ranges:
        range_data = test_with_pred[(test_with_pred['rul'] >= start) & (test_with_pred['rul'] < end)]
        if len(range_data) > 0:
            range_mae = range_data['abs_error'].mean()
            print(f"  RUL {start}-{end}: MAE = {range_mae:.2f} ({len(range_data)} samples)")

def main():
    """Main execution"""
    print("="*80)
    print("MODEL TRAINING (VERSION 2)")
    print("="*80)
    print("\nObjective: Train models with data leakage eliminated")
    print("Dataset: v2 (cycle features removed, full cycle range)")
    print("="*80)
    
    # Load datasets
    train, val, test = load_datasets()
    
    # Prepare data
    X_train, y_train_class, y_train_reg, feature_cols = prepare_data(train)
    X_val, y_val_class, y_val_reg, _ = prepare_data(val)
    
    print(f"\nFeature count: {len(feature_cols)}")
    print(f"Features: {feature_cols[:5]}... (showing first 5)")
    
    # Train Primary Model
    primary_model, primary_data = train_primary_model(X_train, y_train_class, X_val, y_val_class)
    
    # Train Secondary Model
    secondary_model, secondary_data = train_secondary_model(X_train, y_train_reg, X_val, y_val_reg)
    
    # Quick test evaluation
    evaluate_on_test(primary_model, secondary_model, test, feature_cols)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  1. {MODELS_DIR / 'primary_model.pkl'}")
    print(f"  2. {MODELS_DIR / 'secondary_model.pkl'}")
    print(f"  3. {MODELS_DIR / 'primary_feature_importance.csv'}")
    print(f"  4. {MODELS_DIR / 'secondary_feature_importance.csv'}")
    print("\nNext: Detailed evaluation and comparison (Task 6.9)")

if __name__ == "__main__":
    main()
