"""
Build One-Class SVM model for anomaly detection.

This script builds a semi-supervised anomaly detection model using One-Class SVM.
It learns the normal pattern from initial cycles (1-10) and detects anomalies
in all cycles based on deviation from this learned pattern.

Approach rationale:
- Initial cycles (1-10) are assumed to be normal (product characteristic)
- Learn normal pattern from intrinsic features only (no temporal information)
- Detect anomalies as deviations from learned normal pattern
- Linear separability through RBF kernel
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

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Paths
BASE_DIR = Path(__file__).parent.parent
FEATURES_PATH = BASE_DIR / "output" / "features_v3" / "es12_response_features.csv"
OUTPUT_DIR = BASE_DIR / "output" / "anomaly_detection"
MODELS_DIR = BASE_DIR / "output" / "models_v3"

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10


def select_features_for_anomaly_detection(df):
    """
    Select features for anomaly detection, excluding potential data leakage.
    
    Same as Isolation Forest: use only intrinsic properties.
    """
    print("\n" + "="*80)
    print("FEATURE SELECTION FOR ANOMALY DETECTION")
    print("="*80)
    
    # Features to INCLUDE (intrinsic properties only)
    included_features = [
        'response_efficiency',
        'voltage_ratio',
        'peak_voltage_ratio',
        'rms_voltage_ratio',
        'waveform_correlation',
        'vo_variability',
        'vl_variability',
        'response_delay',
        'response_delay_normalized',
        'residual_energy_ratio',
        'vo_complexity'
    ]
    
    print("\n✅ INCLUDED Features (Intrinsic Properties):")
    print("-" * 80)
    for feat in included_features:
        print(f"  - {feat}")
    
    print("\n❌ EXCLUDED Features (Potential Data Leakage):")
    print("-" * 80)
    excluded = ['cycle', 'capacitor_id', 'efficiency_degradation_rate', 
                'voltage_ratio_deviation', 'correlation_shift', 'peak_voltage_ratio_deviation']
    for feat in excluded:
        print(f"  - {feat}")
    
    return included_features


def prepare_training_data(df, features, normal_cycle_range=(1, 10)):
    """
    Prepare training data from initial cycles only.
    
    Args:
        df: Full dataframe
        features: List of feature names
        normal_cycle_range: Cycle range to use as normal (default: 1-10)
    
    Returns:
        X_train: Training features (normal cycles only)
        X_all: All features (for prediction)
        scaler: Fitted scaler
    """
    print("\n" + "="*80)
    print("DATA PREPARATION")
    print("="*80)
    
    # Extract normal cycles for training
    normal_df = df[(df['cycle'] >= normal_cycle_range[0]) & 
                   (df['cycle'] <= normal_cycle_range[1])]
    
    print(f"\nNormal data (training):")
    print(f"  Cycle range: {normal_cycle_range[0]}-{normal_cycle_range[1]}")
    print(f"  Samples: {len(normal_df)}")
    print(f"  Capacitors: {normal_df['capacitor_id'].nunique()}")
    print(f"  Rationale: Initial cycles are assumed normal (product characteristic)")
    
    # Extract features
    X_train = normal_df[features].copy()
    X_all = df[features].copy()
    
    # Handle missing/infinite values
    X_train = X_train.fillna(X_train.median())
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    
    X_all = X_all.fillna(X_all.median())
    X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(X_all.median())
    
    print(f"\n✓ Training feature matrix: {X_train.shape}")
    print(f"✓ Full feature matrix: {X_all.shape}")
    
    # Standardize features
    print("\nStandardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_all_scaled = scaler.transform(X_all)
    
    print("✓ Features standardized (mean=0, std=1)")
    
    return X_train_scaled, X_all_scaled, scaler, normal_df


def train_one_class_svm(X_train, nu=0.1, gamma='scale'):
    """
    Train One-Class SVM model.
    
    Args:
        X_train: Training features (normal samples only)
        nu: Upper bound on fraction of outliers (default: 0.1)
        gamma: Kernel coefficient (default: 'scale')
    
    Returns:
        Trained One-Class SVM model
    """
    print("\n" + "="*80)
    print("TRAINING ONE-CLASS SVM")
    print("="*80)
    
    print(f"\nModel parameters:")
    print(f"  kernel: 'rbf' (Radial Basis Function)")
    print(f"  nu: {nu} (upper bound on fraction of training errors)")
    print(f"  gamma: '{gamma}' (kernel coefficient)")
    
    print(f"\nTraining approach:")
    print(f"  1. Learn normal pattern from initial cycles (1-10)")
    print(f"  2. Create decision boundary in feature space")
    print(f"  3. Classify deviations from normal as anomalies")
    print(f"  4. No temporal information used - only feature patterns")
    
    # Train model
    print("\nTraining model...")
    model = OneClassSVM(
        kernel='rbf',
        nu=nu,
        gamma=gamma
    )
    
    model.fit(X_train)
    
    print("✓ Model trained successfully")
    print(f"  Support vectors: {model.n_support_[0]}")
    
    return model


def predict_anomalies(model, X_all_scaled, df):
    """Predict anomalies using trained One-Class SVM."""
    print("\n" + "="*80)
    print("ANOMALY PREDICTION")
    print("="*80)
    
    # Predict anomalies (-1 for anomalies, 1 for normal)
    predictions = model.predict(X_all_scaled)
    
    # Get decision function scores (distance from decision boundary)
    # Negative values indicate anomalies
    decision_scores = model.decision_function(X_all_scaled)
    
    # Convert predictions to binary (0=normal, 1=anomaly)
    is_anomaly = (predictions == -1).astype(int)
    
    # Add to dataframe
    results_df = df.copy()
    results_df['decision_score'] = decision_scores
    results_df['is_anomaly'] = is_anomaly
    results_df['prediction'] = predictions
    
    # Statistics
    n_anomalies = is_anomaly.sum()
    anomaly_pct = n_anomalies / len(df) * 100
    
    print(f"\nPrediction results:")
    print(f"  Total samples: {len(df)}")
    print(f"  Normal samples: {len(df) - n_anomalies} ({100 - anomaly_pct:.1f}%)")
    print(f"  Anomalous samples: {n_anomalies} ({anomaly_pct:.1f}%)")
    
    print(f"\nDecision score statistics:")
    print(f"  Mean: {decision_scores.mean():.4f}")
    print(f"  Std:  {decision_scores.std():.4f}")
    print(f"  Min:  {decision_scores.min():.4f} (most anomalous)")
    print(f"  Max:  {decision_scores.max():.4f} (most normal)")
    
    # Anomalies by capacitor
    print(f"\nAnomalies by capacitor:")
    print("-" * 80)
    for cap_id in sorted(results_df['capacitor_id'].unique()):
        cap_data = results_df[results_df['capacitor_id'] == cap_id]
        cap_anomalies = cap_data['is_anomaly'].sum()
        cap_pct = cap_anomalies / len(cap_data) * 100
        print(f"  {cap_id}: {cap_anomalies}/{len(cap_data)} ({cap_pct:.1f}%)")
    
    # Anomalies by cycle range
    print(f"\nAnomalies by cycle range:")
    print("-" * 80)
    cycle_ranges = [(1, 50), (51, 100), (101, 150), (151, 200)]
    for start, end in cycle_ranges:
        range_data = results_df[(results_df['cycle'] >= start) & (results_df['cycle'] <= end)]
        range_anomalies = range_data['is_anomaly'].sum()
        range_pct = range_anomalies / len(range_data) * 100
        print(f"  Cycles {start:3d}-{end:3d}: {range_anomalies:3d}/{len(range_data):3d} ({range_pct:5.1f}%)")
    
    return results_df


def analyze_anomaly_characteristics(results_df, features):
    """Analyze characteristics of detected anomalies."""
    print("\n" + "="*80)
    print("ANOMALY CHARACTERISTICS ANALYSIS")
    print("="*80)
    
    normal_df = results_df[results_df['is_anomaly'] == 0]
    anomaly_df = results_df[results_df['is_anomaly'] == 1]
    
    print("\nFeature comparison (Normal vs Anomaly):")
    print("-" * 80)
    print(f"{'Feature':<30} {'Normal Mean':<15} {'Anomaly Mean':<15} {'Difference':<15}")
    print("-" * 80)
    
    for feat in features:
        normal_mean = normal_df[feat].mean()
        anomaly_mean = anomaly_df[feat].mean()
        diff = anomaly_mean - normal_mean
        diff_pct = (diff / normal_mean * 100) if normal_mean != 0 else 0
        
        print(f"{feat:<30} {normal_mean:>14.4f} {anomaly_mean:>14.4f} {diff_pct:>13.1f}%")
    
    # Cycle distribution
    print("\n\nCycle distribution:")
    print("-" * 80)
    print(f"Normal samples:")
    print(f"  Mean cycle: {normal_df['cycle'].mean():.1f}")
    print(f"  Cycle range: {normal_df['cycle'].min():.0f} - {normal_df['cycle'].max():.0f}")
    
    print(f"\nAnomalous samples:")
    print(f"  Mean cycle: {anomaly_df['cycle'].mean():.1f}")
    print(f"  Cycle range: {anomaly_df['cycle'].min():.0f} - {anomaly_df['cycle'].max():.0f}")


def visualize_one_class_svm_results(results_df, features):
    """Create comprehensive One-Class SVM visualizations."""
    print("\n" + "="*80)
    print("CREATING ONE-CLASS SVM VISUALIZATIONS")
    print("="*80)
    
    # Create figure
    fig = plt.figure(figsize=(20, 14))
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
    
    capacitors = sorted(results_df['capacitor_id'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(capacitors)))
    
    # 1. Decision score over time
    ax1 = fig.add_subplot(gs[0, :])
    for i, cap_id in enumerate(capacitors):
        cap_data = results_df[results_df['capacitor_id'] == cap_id]
        ax1.plot(cap_data['cycle'], cap_data['decision_score'], 
                label=cap_id, color=colors[i], alpha=0.7, linewidth=1.5)
    
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2, label='Decision Boundary')
    ax1.axvspan(1, 10, alpha=0.2, color='green', label='Training Data (Normal)')
    ax1.set_xlabel('Cycle', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Decision Score', fontsize=12, fontweight='bold')
    ax1.set_title('One-Class SVM Decision Score Over Time\n(Negative = Anomaly, Positive = Normal)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='lower left', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    
    # 2. Anomaly detection results
    ax2 = fig.add_subplot(gs[1, 0])
    for i, cap_id in enumerate(capacitors):
        cap_data = results_df[results_df['capacitor_id'] == cap_id]
        normal = cap_data[cap_data['is_anomaly'] == 0]
        anomaly = cap_data[cap_data['is_anomaly'] == 1]
        
        ax2.scatter(normal['cycle'], [i] * len(normal), 
                   color='green', alpha=0.6, s=20, marker='o')
        ax2.scatter(anomaly['cycle'], [i] * len(anomaly), 
                   color='red', alpha=0.8, s=30, marker='x')
    
    ax2.axvspan(1, 10, alpha=0.2, color='lightgreen')
    ax2.set_xlabel('Cycle', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Capacitor', fontsize=11, fontweight='bold')
    ax2.set_yticks(range(len(capacitors)))
    ax2.set_yticklabels(capacitors)
    ax2.set_title('One-Class SVM Detection Results\n(Green=Normal, Red=Anomaly)', 
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # 3. Response Efficiency with anomalies
    ax3 = fig.add_subplot(gs[1, 1])
    for i, cap_id in enumerate(capacitors):
        cap_data = results_df[results_df['capacitor_id'] == cap_id]
        normal = cap_data[cap_data['is_anomaly'] == 0]
        anomaly = cap_data[cap_data['is_anomaly'] == 1]
        
        ax3.plot(cap_data['cycle'], cap_data['response_efficiency'], 
                color=colors[i], alpha=0.3, linewidth=1)
        ax3.scatter(anomaly['cycle'], anomaly['response_efficiency'], 
                   color='red', alpha=0.8, s=30, marker='x')
    
    ax3.axvspan(1, 10, alpha=0.2, color='lightgreen')
    ax3.set_xlabel('Cycle', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Response Efficiency', fontsize=11, fontweight='bold')
    ax3.set_title('Response Efficiency\n(Red X = Detected Anomaly)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)
    
    # 4. Waveform Correlation with anomalies
    ax4 = fig.add_subplot(gs[1, 2])
    for i, cap_id in enumerate(capacitors):
        cap_data = results_df[results_df['capacitor_id'] == cap_id]
        normal = cap_data[cap_data['is_anomaly'] == 0]
        anomaly = cap_data[cap_data['is_anomaly'] == 1]
        
        ax4.plot(cap_data['cycle'], cap_data['waveform_correlation'], 
                color=colors[i], alpha=0.3, linewidth=1)
        ax4.scatter(anomaly['cycle'], anomaly['waveform_correlation'], 
                   color='red', alpha=0.8, s=30, marker='x')
    
    ax4.axvspan(1, 10, alpha=0.2, color='lightgreen')
    ax4.set_xlabel('Cycle', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Waveform Correlation', fontsize=11, fontweight='bold')
    ax4.set_title('Waveform Correlation\n(Red X = Detected Anomaly)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Decision score distribution
    ax5 = fig.add_subplot(gs[2, 0])
    normal_scores = results_df[results_df['is_anomaly'] == 0]['decision_score']
    anomaly_scores = results_df[results_df['is_anomaly'] == 1]['decision_score']
    
    ax5.hist(normal_scores, bins=50, alpha=0.6, label=f'Normal (n={len(normal_scores)})', color='green')
    ax5.hist(anomaly_scores, bins=50, alpha=0.6, label=f'Anomaly (n={len(anomaly_scores)})', color='red')
    ax5.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Decision Boundary')
    
    ax5.set_xlabel('Decision Score', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax5.set_title('Decision Score Distribution', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper left', fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Feature comparison - Response Efficiency
    ax6 = fig.add_subplot(gs[2, 1])
    normal_eff = results_df[results_df['is_anomaly'] == 0]['response_efficiency']
    anomaly_eff = results_df[results_df['is_anomaly'] == 1]['response_efficiency']
    
    bp = ax6.boxplot([normal_eff, anomaly_eff], 
                     tick_labels=['Normal', 'Anomaly'], 
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    ax6.set_ylabel('Response Efficiency', fontsize=11, fontweight='bold')
    ax6.set_title('Response Efficiency: Normal vs Anomaly', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_yscale('log')
    
    # 7. Feature comparison - Waveform Correlation
    ax7 = fig.add_subplot(gs[2, 2])
    normal_corr = results_df[results_df['is_anomaly'] == 0]['waveform_correlation']
    anomaly_corr = results_df[results_df['is_anomaly'] == 1]['waveform_correlation']
    
    bp = ax7.boxplot([normal_corr, anomaly_corr], 
                     tick_labels=['Normal', 'Anomaly'], 
                     patch_artist=True)
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    ax7.set_ylabel('Waveform Correlation', fontsize=11, fontweight='bold')
    ax7.set_title('Waveform Correlation: Normal vs Anomaly', fontsize=12, fontweight='bold')
    ax7.grid(True, alpha=0.3, axis='y')
    
    # 8. Cycle distribution
    ax8 = fig.add_subplot(gs[3, 0])
    normal_cycles = results_df[results_df['is_anomaly'] == 0]['cycle']
    anomaly_cycles = results_df[results_df['is_anomaly'] == 1]['cycle']
    
    ax8.hist(normal_cycles, bins=20, alpha=0.6, label=f'Normal (n={len(normal_cycles)})', color='green')
    ax8.hist(anomaly_cycles, bins=20, alpha=0.6, label=f'Anomaly (n={len(anomaly_cycles)})', color='red')
    ax8.axvspan(1, 10, alpha=0.2, color='lightgreen')
    
    ax8.set_xlabel('Cycle', fontsize=11, fontweight='bold')
    ax8.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax8.set_title('Cycle Distribution: Normal vs Anomaly', fontsize=12, fontweight='bold')
    ax8.legend(loc='upper right', fontsize=9)
    ax8.grid(True, alpha=0.3, axis='y')
    
    # 9. Anomaly rate by cycle range
    ax9 = fig.add_subplot(gs[3, 1:])
    cycle_ranges = [(1, 50), (51, 100), (101, 150), (151, 200)]
    range_labels = ['1-50\n(Training)', '51-100', '101-150', '151-200']
    anomaly_rates = []
    
    for start, end in cycle_ranges:
        range_data = results_df[(results_df['cycle'] >= start) & (results_df['cycle'] <= end)]
        rate = range_data['is_anomaly'].mean() * 100
        anomaly_rates.append(rate)
    
    bars = ax9.bar(range_labels, anomaly_rates, color=['lightgreen', 'yellow', 'orange', 'red'], alpha=0.7)
    ax9.set_xlabel('Cycle Range', fontsize=11, fontweight='bold')
    ax9.set_ylabel('Anomaly Rate (%)', fontsize=11, fontweight='bold')
    ax9.set_title('Anomaly Detection Rate by Cycle Range', fontsize=12, fontweight='bold')
    ax9.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, rate in zip(bars, anomaly_rates):
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('One-Class SVM Anomaly Detection Results\n(Trained on Initial Cycles 1-10)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    output_path = OUTPUT_DIR / "one_class_svm_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()


def save_model_and_results(model, scaler, results_df, features):
    """Save trained model, scaler, and results."""
    print("\n" + "="*80)
    print("SAVING MODEL AND RESULTS")
    print("="*80)
    
    # Save model
    model_path = MODELS_DIR / "one_class_svm.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Saved model: {model_path}")
    
    # Save scaler
    scaler_path = MODELS_DIR / "one_class_svm_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Saved scaler: {scaler_path}")
    
    # Save feature list
    features_path = MODELS_DIR / "one_class_svm_features.txt"
    with open(features_path, 'w') as f:
        for feat in features:
            f.write(f"{feat}\n")
    print(f"✓ Saved feature list: {features_path}")
    
    # Save results
    results_path = OUTPUT_DIR / "one_class_svm_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"✓ Saved results: {results_path}")


def main():
    """Main execution."""
    print("="*80)
    print("ONE-CLASS SVM ANOMALY DETECTION")
    print("="*80)
    print("\nApproach: Learn normal pattern from initial cycles (1-10)")
    print("Rationale: Initial cycles are assumed normal (product characteristic)")
    print("Method: Detect anomalies as deviations from learned normal pattern")
    
    # Load features
    print("\nLoading features...")
    df = pd.read_csv(FEATURES_PATH)
    print(f"  ✓ Loaded {len(df)} samples")
    
    # Select features (exclude data leakage)
    features = select_features_for_anomaly_detection(df)
    
    # Prepare data (train on initial cycles only)
    X_train, X_all, scaler, normal_df = prepare_training_data(df, features, normal_cycle_range=(1, 10))
    
    # Train model
    model = train_one_class_svm(X_train, nu=0.1, gamma='scale')
    
    # Predict anomalies
    results_df = predict_anomalies(model, X_all, df)
    
    # Analyze anomaly characteristics
    analyze_anomaly_characteristics(results_df, features)
    
    # Create visualizations
    visualize_one_class_svm_results(results_df, features)
    
    # Save model and results
    save_model_and_results(model, scaler, results_df, features)
    
    print("\n" + "="*80)
    print("ONE-CLASS SVM ANOMALY DETECTION COMPLETE!")
    print("="*80)
    print(f"\nOutput directories:")
    print(f"  Models: {MODELS_DIR}")
    print(f"  Results: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  1. one_class_svm.pkl - Trained model")
    print("  2. one_class_svm_scaler.pkl - Feature scaler")
    print("  3. one_class_svm_features.txt - Feature list")
    print("  4. one_class_svm_results.csv - Detection results")
    print("  5. one_class_svm_results.png - Visualization")
    print("\n✅ One-Class SVM anomaly detection complete!")


if __name__ == "__main__":
    main()
