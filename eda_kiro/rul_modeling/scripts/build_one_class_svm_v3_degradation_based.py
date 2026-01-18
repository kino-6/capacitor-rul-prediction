"""
Build One-Class SVM v3: Degradation Score-Based Labeling

Key Improvement:
- Use degradation score (from EDA) instead of cycle number for labeling
- Train on samples with degradation_score < 0.25 (Normal stage)
- More objective and physically meaningful than cycle-based labeling

Rationale:
- Cycle number is just a time index, not a physical state indicator
- Degradation score reflects actual waveform degradation from EDA
- Eliminates train/test labeling inconsistency
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import pickle

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).parent.parent
FEATURES_PATH = BASE_DIR / "output" / "features_v3" / "es12_response_features.csv"
DEGRADATION_PATH = BASE_DIR / "output" / "degradation_prediction" / "features_with_degradation_score.csv"
OUTPUT_DIR = BASE_DIR / "output" / "anomaly_detection"
MODELS_DIR = BASE_DIR / "output" / "models_v3"

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10

# Degradation score thresholds (from EDA)
NORMAL_THRESHOLD = 0.25      # degradation_score < 0.25 = Normal
ANOMALY_THRESHOLD = 0.50     # degradation_score >= 0.50 = Anomaly (Severe+)

def load_data_with_degradation():
    """Load features with degradation scores."""
    print("\n" + "="*80)
    print("LOADING DATA WITH DEGRADATION SCORES")
    print("="*80)
    
    # Load degradation scores
    print("\nLoading degradation scores...")
    deg_df = pd.read_csv(DEGRADATION_PATH)
    print(f"  ✓ Loaded {len(deg_df)} samples with degradation scores")
    
    # Check degradation score distribution
    print(f"\nDegradation score statistics:")
    print(f"  Mean: {deg_df['degradation_score'].mean():.3f}")
    print(f"  Std:  {deg_df['degradation_score'].std():.3f}")
    print(f"  Min:  {deg_df['degradation_score'].min():.3f}")
    print(f"  Max:  {deg_df['degradation_score'].max():.3f}")
    
    # Distribution by stage
    print(f"\nSamples by degradation stage:")
    print(f"  Normal (< 0.25):     {(deg_df['degradation_score'] < 0.25).sum():4d} ({(deg_df['degradation_score'] < 0.25).sum() / len(deg_df) * 100:5.1f}%)")
    print(f"  Degrading (0.25-0.50): {((deg_df['degradation_score'] >= 0.25) & (deg_df['degradation_score'] < 0.50)).sum():4d} ({((deg_df['degradation_score'] >= 0.25) & (deg_df['degradation_score'] < 0.50)).sum() / len(deg_df) * 100:5.1f}%)")
    print(f"  Severe (0.50-0.75):  {((deg_df['degradation_score'] >= 0.50) & (deg_df['degradation_score'] < 0.75)).sum():4d} ({((deg_df['degradation_score'] >= 0.50) & (deg_df['degradation_score'] < 0.75)).sum() / len(deg_df) * 100:5.1f}%)")
    print(f"  Critical (>= 0.75):  {(deg_df['degradation_score'] >= 0.75).sum():4d} ({(deg_df['degradation_score'] >= 0.75).sum() / len(deg_df) * 100:5.1f}%)")
    
    return deg_df

def select_waveform_features():
    """Select waveform characteristics for anomaly detection."""
    features = [
        'waveform_correlation',
        'vo_variability',
        'vl_variability',
        'response_delay',
        'response_delay_normalized',
        'residual_energy_ratio',
        'vo_complexity'
    ]
    
    print("\n" + "="*80)
    print("FEATURE SELECTION")
    print("="*80)
    print(f"\nWaveform characteristics (n={len(features)}):")
    for feat in features:
        print(f"  - {feat}")
    
    return features

def prepare_degradation_based_training_data(df, features):
    """Prepare training data based on degradation score."""
    print("\n" + "="*80)
    print("DEGRADATION SCORE-BASED DATA PREPARATION")
    print("="*80)
    
    # Select normal samples (degradation_score < NORMAL_THRESHOLD)
    normal_df = df[df['degradation_score'] < NORMAL_THRESHOLD].copy()
    
    print(f"\nNormal data (training):")
    print(f"  Criterion: degradation_score < {NORMAL_THRESHOLD}")
    print(f"  Samples: {len(normal_df)}")
    print(f"  Capacitors: {normal_df['capacitor_id'].nunique()}")
    print(f"  Cycle range: {normal_df['cycle'].min():.0f}-{normal_df['cycle'].max():.0f}")
    print(f"  Degradation score range: {normal_df['degradation_score'].min():.3f}-{normal_df['degradation_score'].max():.3f}")
    
    # Extract features
    X_train = normal_df[features].copy()
    X_all = df[features].copy()
    
    # Handle missing/infinite values
    X_train = X_train.fillna(X_train.median()).replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    X_all = X_all.fillna(X_all.median()).replace([np.inf, -np.inf], np.nan).fillna(X_all.median())
    
    print(f"\n✓ Training feature matrix: {X_train.shape}")
    print(f"✓ Full feature matrix: {X_all.shape}")
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_all_scaled = scaler.transform(X_all)
    
    print("✓ Features standardized")
    
    return X_train_scaled, X_all_scaled, scaler, normal_df

def train_model(X_train, nu=0.05):
    """Train One-Class SVM."""
    print("\n" + "="*80)
    print("TRAINING ONE-CLASS SVM V3 (DEGRADATION-BASED)")
    print("="*80)
    
    print(f"\nModel parameters:")
    print(f"  kernel: 'rbf'")
    print(f"  nu: {nu}")
    print(f"  gamma: 'scale'")
    
    print(f"\nTraining approach:")
    print(f"  1. Learn normal pattern from degradation_score < {NORMAL_THRESHOLD}")
    print(f"  2. Based on EDA waveform analysis, not arbitrary cycle numbers")
    print(f"  3. Physically meaningful and objective")
    
    model = OneClassSVM(kernel='rbf', nu=nu, gamma='scale')
    model.fit(X_train)
    
    print(f"\n✓ Model trained")
    print(f"  Support vectors: {model.n_support_[0]}")
    
    return model

def predict_and_evaluate(model, X_all_scaled, df):
    """Predict anomalies and evaluate using degradation score."""
    print("\n" + "="*80)
    print("PREDICTION AND EVALUATION")
    print("="*80)
    
    # Predict
    predictions = model.predict(X_all_scaled)
    decision_scores = model.decision_function(X_all_scaled)
    
    # Add to dataframe
    results_df = df.copy()
    results_df['decision_score'] = decision_scores
    results_df['is_anomaly_pred'] = (predictions == -1).astype(int)
    
    # Ground truth based on degradation score
    results_df['is_anomaly_true'] = (results_df['degradation_score'] >= ANOMALY_THRESHOLD).astype(int)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
    
    y_true = results_df['is_anomaly_true']
    y_pred = results_df['is_anomaly_pred']
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    # False Positive Rate
    fp_rate = (cm[0][1] / (cm[0][0] + cm[0][1])) * 100 if (cm[0][0] + cm[0][1]) > 0 else 0
    tn_rate = (cm[0][0] / (cm[0][0] + cm[0][1])) * 100 if (cm[0][0] + cm[0][1]) > 0 else 0
    
    print(f"\nEvaluation Results:")
    print(f"  Ground Truth: degradation_score >= {ANOMALY_THRESHOLD} = Anomaly")
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Normal  Anomaly")
    print(f"Actual Normal   {cm[0][0]:4d}    {cm[0][1]:4d}")
    print(f"      Anomaly   {cm[1][0]:4d}    {cm[1][1]:4d}")
    
    print(f"\nMetrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"\n  False Positive Rate: {fp_rate:.1f}%")
    print(f"  True Negative Rate:  {tn_rate:.1f}%")
    
    return results_df, {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1, 
                        'confusion_matrix': cm, 'fp_rate': fp_rate, 'tn_rate': tn_rate}

def save_model(model, scaler, features, metrics):
    """Save model and results."""
    print("\n" + "="*80)
    print("SAVING MODEL")
    print("="*80)
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save model
    with open(MODELS_DIR / "one_class_svm_v3_degradation_based.pkl", 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Saved: one_class_svm_v3_degradation_based.pkl")
    
    # Save scaler
    with open(MODELS_DIR / "one_class_svm_v3_degradation_based_scaler.pkl", 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Saved: one_class_svm_v3_degradation_based_scaler.pkl")
    
    # Save features
    with open(MODELS_DIR / "one_class_svm_v3_degradation_based_features.txt", 'w') as f:
        for feat in features:
            f.write(f"{feat}\n")
    print(f"✓ Saved: one_class_svm_v3_degradation_based_features.txt")
    
    # Save metrics
    with open(MODELS_DIR / "one_class_svm_v3_degradation_based_metrics.txt", 'w') as f:
        f.write(f"Degradation Score-Based Labeling\n")
        f.write(f"Normal Threshold: < {NORMAL_THRESHOLD}\n")
        f.write(f"Anomaly Threshold: >= {ANOMALY_THRESHOLD}\n\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {metrics['f1']:.4f}\n")
        f.write(f"False Positive Rate: {metrics['fp_rate']:.1f}%\n")
        f.write(f"True Negative Rate:  {metrics['tn_rate']:.1f}%\n")
    print(f"✓ Saved: one_class_svm_v3_degradation_based_metrics.txt")

def main():
    print("="*80)
    print("ONE-CLASS SVM V3: DEGRADATION SCORE-BASED LABELING")
    print("="*80)
    print("\nKey Improvement:")
    print("  - Use degradation score from EDA, not cycle number")
    print("  - Train on degradation_score < 0.25 (Normal)")
    print("  - Evaluate on degradation_score >= 0.50 (Severe+)")
    print("  - Physically meaningful and objective")
    
    # Load data
    df = load_data_with_degradation()
    
    # Select features
    features = select_waveform_features()
    
    # Prepare data
    X_train, X_all, scaler, normal_df = prepare_degradation_based_training_data(df, features)
    
    # Train model
    model = train_model(X_train, nu=0.05)
    
    # Predict and evaluate
    results_df, metrics = predict_and_evaluate(model, X_all, df)
    
    # Save
    save_model(model, scaler, features, metrics)
    
    # Save results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(OUTPUT_DIR / "one_class_svm_v3_degradation_based_results.csv", index=False)
    print(f"✓ Saved: one_class_svm_v3_degradation_based_results.csv")
    
    print("\n" + "="*80)
    print("✅ ONE-CLASS SVM V3 COMPLETE!")
    print("="*80)
    print(f"\nImprovement over v2:")
    print(f"  v2 (Cycle-based): FP Rate = 86.5%")
    print(f"  v3 (Degradation-based): FP Rate = {metrics['fp_rate']:.1f}%")
    print(f"\n  Improvement: {86.5 - metrics['fp_rate']:.1f}% reduction in false positives!")

if __name__ == "__main__":
    main()
