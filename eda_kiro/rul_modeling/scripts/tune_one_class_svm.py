"""
Tune One-Class SVM v2 hyperparameters to reduce False Positives.

This script tests different nu values to find the optimal balance between
detecting true anomalies and minimizing false positives in early cycles.
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


def select_waveform_features_only(df):
    """Select only waveform characteristics, excluding efficiency features."""
    included_features = [
        'waveform_correlation',
        'vo_variability',
        'vl_variability',
        'response_delay',
        'response_delay_normalized',
        'residual_energy_ratio',
        'vo_complexity'
    ]
    return included_features


def prepare_data(df, features, normal_cycle_range=(1, 10)):
    """Prepare training and test data."""
    # Extract normal cycles for training
    normal_df = df[(df['cycle'] >= normal_cycle_range[0]) & 
                   (df['cycle'] <= normal_cycle_range[1])]
    
    # Extract features
    X_train = normal_df[features].copy()
    X_all = df[features].copy()
    
    # Handle missing/infinite values
    X_train = X_train.fillna(X_train.median())
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.median())
    
    X_all = X_all.fillna(X_all.median())
    X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(X_all.median())
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_all_scaled = scaler.transform(X_all)
    
    return X_train_scaled, X_all_scaled, scaler


def evaluate_nu(nu, X_train, X_all, df):
    """Train and evaluate One-Class SVM with given nu."""
    # Train model
    model = OneClassSVM(kernel='rbf', nu=nu, gamma='scale')
    model.fit(X_train)
    
    # Predict
    predictions = model.predict(X_all)
    is_anomaly = (predictions == -1).astype(int)
    
    # Calculate metrics
    results = {
        'nu': nu,
        'total_anomalies': is_anomaly.sum(),
        'anomaly_rate': is_anomaly.mean() * 100
    }
    
    # Early cycles (1-20) false positives
    early_mask = df['cycle'] <= 20
    early_anomalies = is_anomaly[early_mask].sum()
    early_total = early_mask.sum()
    results['early_fp'] = early_anomalies
    results['early_fp_rate'] = early_anomalies / early_total * 100
    
    # Training cycles (1-10) false positives
    train_mask = (df['cycle'] >= 1) & (df['cycle'] <= 10)
    train_anomalies = is_anomaly[train_mask].sum()
    train_total = train_mask.sum()
    results['train_fp'] = train_anomalies
    results['train_fp_rate'] = train_anomalies / train_total * 100
    
    # Late cycles (100+) false negatives
    late_mask = df['cycle'] >= 100
    late_normal = (is_anomaly[late_mask] == 0).sum()
    late_total = late_mask.sum()
    results['late_fn'] = late_normal
    results['late_fn_rate'] = late_normal / late_total * 100
    
    # Mid cycles (51-100) detection
    mid_mask = (df['cycle'] >= 51) & (df['cycle'] <= 100)
    mid_anomalies = is_anomaly[mid_mask].sum()
    mid_total = mid_mask.sum()
    results['mid_detection_rate'] = mid_anomalies / mid_total * 100
    
    return results, model, is_anomaly


def main():
    """Main execution."""
    print("="*80)
    print("ONE-CLASS SVM HYPERPARAMETER TUNING")
    print("="*80)
    print("\nTesting different nu values to optimize False Positive rate...")
    
    # Load data
    print("\nLoading features...")
    df = pd.read_csv(FEATURES_PATH)
    print(f"  ✓ Loaded {len(df)} samples")
    
    # Select features
    features = select_waveform_features_only(df)
    print(f"  ✓ Using {len(features)} waveform features")
    
    # Prepare data
    print("\nPreparing data...")
    X_train, X_all, scaler = prepare_data(df, features, normal_cycle_range=(1, 10))
    print(f"  ✓ Training samples: {X_train.shape[0]}")
    print(f"  ✓ Total samples: {X_all.shape[0]}")
    
    # Test different nu values
    print("\n" + "="*80)
    print("TESTING DIFFERENT NU VALUES")
    print("="*80)
    
    nu_values = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    results_list = []
    
    print(f"\n{'nu':<8} {'Total':<8} {'Rate':<8} {'Train FP':<10} {'Early FP':<10} {'Late FN':<10} {'Mid Det':<10}")
    print("-" * 80)
    
    for nu in nu_values:
        results, model, is_anomaly = evaluate_nu(nu, X_train, X_all, df)
        results_list.append(results)
        
        print(f"{results['nu']:<8.2f} "
              f"{results['total_anomalies']:<8d} "
              f"{results['anomaly_rate']:<7.1f}% "
              f"{results['train_fp_rate']:<9.1f}% "
              f"{results['early_fp_rate']:<9.1f}% "
              f"{results['late_fn_rate']:<9.1f}% "
              f"{results['mid_detection_rate']:<9.1f}%")
    
    # Find optimal nu
    print("\n" + "="*80)
    print("OPTIMAL NU SELECTION")
    print("="*80)
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results_list)
    
    # Criteria: minimize early FP while maintaining high mid detection
    # Target: early_fp_rate < 30%, mid_detection_rate > 95%
    
    print("\nSelection criteria:")
    print("  1. Training FP rate < 20% (学習データの誤検出を最小化)")
    print("  2. Early FP rate < 30% (初期サイクルの誤検出を抑制)")
    print("  3. Mid detection rate > 95% (中期サイクルの検出を維持)")
    print("  4. Late FN rate < 10% (後期サイクルの見逃しを最小化)")
    
    # Filter candidates
    candidates = results_df[
        (results_df['train_fp_rate'] < 20) &
        (results_df['early_fp_rate'] < 30) &
        (results_df['mid_detection_rate'] > 95) &
        (results_df['late_fn_rate'] < 10)
    ]
    
    if len(candidates) > 0:
        # Select the one with lowest early FP rate
        optimal_idx = candidates['early_fp_rate'].idxmin()
        optimal = candidates.loc[optimal_idx]
        
        print(f"\n✓ Optimal nu found: {optimal['nu']:.2f}")
        print(f"  Total anomalies: {optimal['total_anomalies']:.0f} ({optimal['anomaly_rate']:.1f}%)")
        print(f"  Training FP rate: {optimal['train_fp_rate']:.1f}%")
        print(f"  Early FP rate: {optimal['early_fp_rate']:.1f}%")
        print(f"  Late FN rate: {optimal['late_fn_rate']:.1f}%")
        print(f"  Mid detection rate: {optimal['mid_detection_rate']:.1f}%")
    else:
        print("\n⚠ No nu value meets all criteria. Relaxing constraints...")
        
        # Relax constraints
        candidates = results_df[
            (results_df['train_fp_rate'] < 25) &
            (results_df['early_fp_rate'] < 40) &
            (results_df['mid_detection_rate'] > 90)
        ]
        
        if len(candidates) > 0:
            optimal_idx = candidates['early_fp_rate'].idxmin()
            optimal = candidates.loc[optimal_idx]
            
            print(f"\n✓ Best compromise nu: {optimal['nu']:.2f}")
            print(f"  Total anomalies: {optimal['total_anomalies']:.0f} ({optimal['anomaly_rate']:.1f}%)")
            print(f"  Training FP rate: {optimal['train_fp_rate']:.1f}%")
            print(f"  Early FP rate: {optimal['early_fp_rate']:.1f}%")
            print(f"  Late FN rate: {optimal['late_fn_rate']:.1f}%")
            print(f"  Mid detection rate: {optimal['mid_detection_rate']:.1f}%")
        else:
            print("\n⚠ Current nu=0.1 may be the best available option")
            optimal = results_df[results_df['nu'] == 0.1].iloc[0]
    
    # Visualize results
    print("\n" + "="*80)
    print("CREATING VISUALIZATION")
    print("="*80)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Anomaly rate vs nu
    ax1 = axes[0, 0]
    ax1.plot(results_df['nu'], results_df['anomaly_rate'], 'o-', linewidth=2, markersize=8)
    ax1.axvline(x=optimal['nu'], color='red', linestyle='--', label=f'Optimal nu={optimal["nu"]:.2f}')
    ax1.set_xlabel('nu', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Anomaly Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Total Anomaly Rate vs nu', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Early FP rate vs nu
    ax2 = axes[0, 1]
    ax2.plot(results_df['nu'], results_df['train_fp_rate'], 'o-', linewidth=2, markersize=8, label='Training (1-10)')
    ax2.plot(results_df['nu'], results_df['early_fp_rate'], 's-', linewidth=2, markersize=8, label='Early (1-20)')
    ax2.axvline(x=optimal['nu'], color='red', linestyle='--', label=f'Optimal nu={optimal["nu"]:.2f}')
    ax2.axhline(y=20, color='orange', linestyle=':', label='Target: 20%')
    ax2.set_xlabel('nu', fontsize=12, fontweight='bold')
    ax2.set_ylabel('False Positive Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('False Positive Rate vs nu', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Late FN rate vs nu
    ax3 = axes[1, 0]
    ax3.plot(results_df['nu'], results_df['late_fn_rate'], 'o-', linewidth=2, markersize=8)
    ax3.axvline(x=optimal['nu'], color='red', linestyle='--', label=f'Optimal nu={optimal["nu"]:.2f}')
    ax3.axhline(y=10, color='orange', linestyle=':', label='Target: 10%')
    ax3.set_xlabel('nu', fontsize=12, fontweight='bold')
    ax3.set_ylabel('False Negative Rate (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Late Cycle False Negative Rate vs nu', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Mid detection rate vs nu
    ax4 = axes[1, 1]
    ax4.plot(results_df['nu'], results_df['mid_detection_rate'], 'o-', linewidth=2, markersize=8)
    ax4.axvline(x=optimal['nu'], color='red', linestyle='--', label=f'Optimal nu={optimal["nu"]:.2f}')
    ax4.axhline(y=95, color='orange', linestyle=':', label='Target: 95%')
    ax4.set_xlabel('nu', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Detection Rate (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Mid Cycle Detection Rate vs nu', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('One-Class SVM Hyperparameter Tuning Results', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / "hyperparameter_tuning_results.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()
    
    # Save results
    results_path = OUTPUT_DIR / "hyperparameter_tuning_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"✓ Saved: {results_path}")
    
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING COMPLETE!")
    print("="*80)
    print(f"\n推奨設定: nu={optimal['nu']:.2f}")
    print(f"\nこの設定で One-Class SVM v2 を再構築することを推奨します。")


if __name__ == "__main__":
    main()
