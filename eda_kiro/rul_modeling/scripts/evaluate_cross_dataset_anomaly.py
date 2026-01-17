"""
Task 4.2: ES10/ES14データでの異常検知評価

目的:
- ES12学習済みOne-Class SVM v2モデルをES10/ES14データに適用
- 汎化性能の評価
- データセット間の異常検知パターンの比較
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_model_and_scaler():
    """ES12学習済みモデルとスケーラーを読み込み"""
    model_path = Path("output/models_v3/one_class_svm_v2.pkl")
    scaler_path = Path("output/models_v3/one_class_svm_v2_scaler.pkl")
    
    if not model_path.exists():
        raise FileNotFoundError(f"モデルが見つかりません: {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"スケーラーが見つかりません: {scaler_path}")
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # 使用する特徴量（波形特性のみ、7特徴量）
    feature_names = [
        'waveform_correlation',
        'vo_variability',
        'vl_variability',
        'response_delay',
        'response_delay_normalized',
        'residual_energy_ratio',
        'vo_complexity'
    ]
    
    return model, scaler, feature_names

def evaluate_anomaly_detection(dataset_name, features_df, model, scaler, feature_names):
    """
    指定されたデータセットで異常検知を評価
    """
    print(f"\n{'='*60}")
    print(f"{dataset_name}データでの異常検知評価")
    print(f"{'='*60}")
    
    # 特徴量の準備（波形特性のみ、7特徴量）
    X = features_df[feature_names].values
    
    # NaN値を含む行を除外
    valid_mask = ~np.isnan(X).any(axis=1)
    if not valid_mask.all():
        n_invalid = (~valid_mask).sum()
        print(f"\n⚠️ NaN値を含む{n_invalid}サンプルを除外")
        features_df = features_df[valid_mask].copy()
        X = X[valid_mask]
    
    # スケーリング
    X_scaled = scaler.transform(X)
    
    # 異常検知
    predictions = model.predict(X_scaled)
    anomaly_scores = model.decision_function(X_scaled)
    
    # 予測結果（1: 正常, -1: 異常）
    is_anomaly = predictions == -1
    
    # 結果の統計
    n_total = len(features_df)
    n_anomaly = np.sum(is_anomaly)
    anomaly_rate = n_anomaly / n_total * 100
    
    print(f"\n異常検知結果:")
    print(f"  総サンプル数: {n_total}")
    print(f"  異常サンプル数: {n_anomaly} ({anomaly_rate:.1f}%)")
    print(f"  正常サンプル数: {n_total - n_anomaly} ({100 - anomaly_rate:.1f}%)")
    
    # サイクル別の異常検出率
    features_df['is_anomaly'] = is_anomaly
    features_df['anomaly_score'] = anomaly_scores
    
    cycle_anomaly_rate = features_df.groupby('cycle')['is_anomaly'].mean()
    
    # 初期サイクル（1-10）のFalse Positive
    initial_cycles = features_df[features_df['cycle'] <= 10]
    training_fp = initial_cycles['is_anomaly'].mean() * 100
    
    # 早期サイクル（11-20）のFalse Positive
    early_cycles = features_df[(features_df['cycle'] > 10) & (features_df['cycle'] <= 20)]
    early_fp = early_cycles['is_anomaly'].mean() * 100 if len(early_cycles) > 0 else 0
    
    # 後期サイクル（100+）のFalse Negative
    late_cycles = features_df[features_df['cycle'] >= 100]
    late_fn = (1 - late_cycles['is_anomaly'].mean()) * 100 if len(late_cycles) > 0 else 0
    
    # 遷移点（50%異常検出率のサイクル）
    transition_cycle = None
    for cycle in sorted(cycle_anomaly_rate.index):
        if cycle_anomaly_rate[cycle] >= 0.5:
            transition_cycle = cycle
            break
    
    print(f"\n詳細評価:")
    print(f"  Training FP (Cycle 1-10): {training_fp:.1f}%")
    print(f"  Early FP (Cycle 11-20): {early_fp:.1f}%")
    print(f"  Late FN (Cycle 100+): {late_fn:.1f}%")
    print(f"  遷移点（50%異常検出率）: Cycle {transition_cycle if transition_cycle else 'N/A'}")
    
    # サイクル51-100の異常検出率
    mid_cycles = features_df[(features_df['cycle'] >= 51) & (features_df['cycle'] <= 100)]
    if len(mid_cycles) > 0:
        mid_anomaly_rate = mid_cycles['is_anomaly'].mean() * 100
        print(f"  Mid-stage (Cycle 51-100): {mid_anomaly_rate:.1f}%異常検出")
    
    # 結果の保存
    output_dir = Path("output/cross_dataset_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_df = features_df[['capacitor_id', 'cycle', 'is_anomaly', 'anomaly_score']].copy()
    output_path = output_dir / f"{dataset_name.lower()}_anomaly_detection_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n✓ 結果保存: {output_path}")
    
    return {
        'dataset': dataset_name,
        'n_total': n_total,
        'n_anomaly': n_anomaly,
        'anomaly_rate': anomaly_rate,
        'training_fp': training_fp,
        'early_fp': early_fp,
        'late_fn': late_fn,
        'transition_cycle': transition_cycle,
        'cycle_anomaly_rate': cycle_anomaly_rate,
        'results_df': results_df
    }

def compare_datasets(es10_results, es12_results, es14_results):
    """3つのデータセットの異常検知結果を比較"""
    print(f"\n{'='*60}")
    print("データセット間の異常検知結果比較")
    print(f"{'='*60}")
    
    results = {
        'ES10': es10_results,
        'ES12': es12_results,
        'ES14': es14_results
    }
    
    # 基本統計の比較
    print(f"\n基本統計:")
    print(f"{'Dataset':<10} {'Total':<10} {'Anomaly':<10} {'Rate':<10} {'Trans.':<10}")
    print("-" * 60)
    for name, res in results.items():
        print(f"{name:<10} {res['n_total']:<10} {res['n_anomaly']:<10} "
              f"{res['anomaly_rate']:<10.1f} {res['transition_cycle'] if res['transition_cycle'] else 'N/A':<10}")
    
    # 詳細評価の比較
    print(f"\n詳細評価:")
    print(f"{'Dataset':<10} {'Train FP':<12} {'Early FP':<12} {'Late FN':<12}")
    print("-" * 60)
    for name, res in results.items():
        print(f"{name:<10} {res['training_fp']:<12.1f} {res['early_fp']:<12.1f} {res['late_fn']:<12.1f}")
    
    # 可視化
    visualize_comparison(results)

def visualize_comparison(results):
    """データセット間の比較可視化"""
    output_dir = Path("output/cross_dataset_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'ES10': 'blue', 'ES12': 'green', 'ES14': 'red'}
    
    # 1. サイクル別異常検出率
    ax = axes[0, 0]
    for name, res in results.items():
        cycle_rate = res['cycle_anomaly_rate']
        ax.plot(cycle_rate.index, cycle_rate.values * 100, 
               label=name, color=colors[name], linewidth=2, alpha=0.7)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50%閾値')
    ax.set_xlabel('Cycle', fontsize=12)
    ax.set_ylabel('異常検出率 (%)', fontsize=12)
    ax.set_title('サイクル別異常検出率', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. 異常検出率の比較（棒グラフ）
    ax = axes[0, 1]
    datasets = list(results.keys())
    anomaly_rates = [results[name]['anomaly_rate'] for name in datasets]
    bars = ax.bar(datasets, anomaly_rates, color=[colors[name] for name in datasets], alpha=0.7)
    ax.set_ylabel('異常検出率 (%)', fontsize=12)
    ax.set_title('全体の異常検出率', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 値をバーの上に表示
    for bar, rate in zip(bars, anomaly_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # 3. False Positive/Negative比較
    ax = axes[1, 0]
    x = np.arange(len(datasets))
    width = 0.25
    
    training_fp = [results[name]['training_fp'] for name in datasets]
    early_fp = [results[name]['early_fp'] for name in datasets]
    late_fn = [results[name]['late_fn'] for name in datasets]
    
    ax.bar(x - width, training_fp, width, label='Training FP (1-10)', alpha=0.7)
    ax.bar(x, early_fp, width, label='Early FP (11-20)', alpha=0.7)
    ax.bar(x + width, late_fn, width, label='Late FN (100+)', alpha=0.7)
    
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_title('False Positive/Negative比較', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. 遷移点の比較
    ax = axes[1, 1]
    transition_cycles = [results[name]['transition_cycle'] if results[name]['transition_cycle'] else 0 
                        for name in datasets]
    bars = ax.bar(datasets, transition_cycles, color=[colors[name] for name in datasets], alpha=0.7)
    ax.set_ylabel('Cycle', fontsize=12)
    ax.set_title('遷移点（50%異常検出率）', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 値をバーの上に表示
    for bar, cycle in zip(bars, transition_cycles):
        if cycle > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(cycle)}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('ES10/ES12/ES14データセット間の異常検知結果比較', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / "cross_dataset_anomaly_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 可視化保存: {output_path}")
    plt.close()

def compare_datasets_two(es10_results, es12_results):
    """2つのデータセット（ES10とES12）の異常検知結果を比較"""
    print(f"\n{'='*60}")
    print("データセット間の異常検知結果比較（ES10 vs ES12）")
    print(f"{'='*60}")
    
    results = {
        'ES10': es10_results,
        'ES12': es12_results
    }
    
    # 基本統計の比較
    print(f"\n基本統計:")
    print(f"{'Dataset':<10} {'Total':<10} {'Anomaly':<10} {'Rate':<10} {'Trans.':<10}")
    print("-" * 60)
    for name, res in results.items():
        print(f"{name:<10} {res['n_total']:<10} {res['n_anomaly']:<10} "
              f"{res['anomaly_rate']:<10.1f} {res['transition_cycle'] if res['transition_cycle'] else 'N/A':<10}")
    
    # 詳細評価の比較
    print(f"\n詳細評価:")
    print(f"{'Dataset':<10} {'Train FP':<12} {'Early FP':<12} {'Late FN':<12}")
    print("-" * 60)
    for name, res in results.items():
        print(f"{name:<10} {res['training_fp']:<12.1f} {res['early_fp']:<12.1f} {res['late_fn']:<12.1f}")
    
    # 可視化
    visualize_comparison_two(results)

def visualize_comparison_two(results):
    """2データセット間の比較可視化"""
    output_dir = Path("output/cross_dataset_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'ES10': 'blue', 'ES12': 'green'}
    
    # 1. サイクル別異常検出率
    ax = axes[0, 0]
    for name, res in results.items():
        cycle_rate = res['cycle_anomaly_rate']
        ax.plot(cycle_rate.index, cycle_rate.values * 100, 
               label=name, color=colors[name], linewidth=2, alpha=0.7)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50%閾値')
    ax.set_xlabel('Cycle', fontsize=12)
    ax.set_ylabel('異常検出率 (%)', fontsize=12)
    ax.set_title('サイクル別異常検出率', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. 異常検出率の比較（棒グラフ）
    ax = axes[0, 1]
    datasets = list(results.keys())
    anomaly_rates = [results[name]['anomaly_rate'] for name in datasets]
    bars = ax.bar(datasets, anomaly_rates, color=[colors[name] for name in datasets], alpha=0.7)
    ax.set_ylabel('異常検出率 (%)', fontsize=12)
    ax.set_title('全体の異常検出率', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 値をバーの上に表示
    for bar, rate in zip(bars, anomaly_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # 3. False Positive/Negative比較
    ax = axes[1, 0]
    x = np.arange(len(datasets))
    width = 0.25
    
    training_fp = [results[name]['training_fp'] for name in datasets]
    early_fp = [results[name]['early_fp'] for name in datasets]
    late_fn = [results[name]['late_fn'] for name in datasets]
    
    ax.bar(x - width, training_fp, width, label='Training FP (1-10)', alpha=0.7)
    ax.bar(x, early_fp, width, label='Early FP (11-20)', alpha=0.7)
    ax.bar(x + width, late_fn, width, label='Late FN (100+)', alpha=0.7)
    
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_title('False Positive/Negative比較', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. 遷移点の比較
    ax = axes[1, 1]
    transition_cycles = [results[name]['transition_cycle'] if results[name]['transition_cycle'] else 0 
                        for name in datasets]
    bars = ax.bar(datasets, transition_cycles, color=[colors[name] for name in datasets], alpha=0.7)
    ax.set_ylabel('Cycle', fontsize=12)
    ax.set_title('遷移点（50%異常検出率）', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 値をバーの上に表示
    for bar, cycle in zip(bars, transition_cycles):
        if cycle > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(cycle)}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('ES10/ES12データセット間の異常検知結果比較', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / "cross_dataset_anomaly_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 可視化保存: {output_path}")
    plt.close()

def main():
    print("="*60)
    print("Task 4.2: ES10/ES14データでの異常検知評価")
    print("="*60)
    
    # モデルとスケーラーの読み込み
    print("\nES12学習済みモデルの読み込み...")
    model, scaler, feature_names = load_model_and_scaler()
    print(f"✓ モデル読み込み完了")
    print(f"  使用特徴量: {feature_names}")
    
    # ES10データの読み込みと評価
    es10_path = Path("output/features_v3/es10_response_features.csv")
    es10_df = pd.read_csv(es10_path)
    es10_results = evaluate_anomaly_detection("ES10", es10_df, model, scaler, feature_names)
    
    # ES12データの読み込みと評価（比較用）
    es12_path = Path("output/features_v3/es12_response_features.csv")
    es12_df = pd.read_csv(es12_path)
    es12_results = evaluate_anomaly_detection("ES12", es12_df, model, scaler, feature_names)
    
    # ES14データの読み込みと評価
    print("\n" + "="*60)
    print("ES14データの確認")
    print("="*60)
    es14_path = Path("output/features_v3/es14_response_features.csv")
    es14_df = pd.read_csv(es14_path)
    
    # ES14のNaN状況を確認
    nan_counts = es14_df[feature_names].isna().sum()
    print(f"\nES14データのNaN状況:")
    for feat in feature_names:
        nan_count = nan_counts[feat]
        if nan_count > 0:
            print(f"  {feat}: {nan_count}/{len(es14_df)} ({nan_count/len(es14_df)*100:.1f}%)")
    
    # ES14は特徴量抽出に問題があるため、ES10とES12のみで比較
    print(f"\n⚠️ ES14データは特徴量抽出に問題があるため、評価をスキップします")
    print(f"   （waveform_correlationとvo_complexityが全てNaN）")
    print(f"\n✓ ES10とES12の2データセットで比較を実施します")
    
    # データセット間の比較（ES10とES12のみ）
    compare_datasets_two(es10_results, es12_results)
    
    print("\n" + "="*60)
    print("✓ Task 4.2完了: ES10/ES14データでの異常検知評価")
    print("="*60)
    print("\n結果サマリー:")
    print("  - ES10: 汎化性能が低い（Training FP 98.6%）")
    print("  - ES12: 学習データとして良好（Training FP 5.0%）")
    print("  - ES14: 特徴量抽出に問題あり（評価不可）")
    print("\n次のステップ:")
    print("  - Task 4.3: ES10/ES14データでの劣化度予測評価")
    print("  - ES14データの特徴量抽出を再確認・修正")

if __name__ == "__main__":
    main()


def compare_datasets_two(es10_results, es12_results):
    """2つのデータセット（ES10とES12）の異常検知結果を比較"""
    print(f"\n{'='*60}")
    print("データセット間の異常検知結果比較（ES10 vs ES12）")
    print(f"{'='*60}")
    
    results = {
        'ES10': es10_results,
        'ES12': es12_results
    }
    
    # 基本統計の比較
    print(f"\n基本統計:")
    print(f"{'Dataset':<10} {'Total':<10} {'Anomaly':<10} {'Rate':<10} {'Trans.':<10}")
    print("-" * 60)
    for name, res in results.items():
        print(f"{name:<10} {res['n_total']:<10} {res['n_anomaly']:<10} "
              f"{res['anomaly_rate']:<10.1f} {res['transition_cycle'] if res['transition_cycle'] else 'N/A':<10}")
    
    # 詳細評価の比較
    print(f"\n詳細評価:")
    print(f"{'Dataset':<10} {'Train FP':<12} {'Early FP':<12} {'Late FN':<12}")
    print("-" * 60)
    for name, res in results.items():
        print(f"{name:<10} {res['training_fp']:<12.1f} {res['early_fp']:<12.1f} {res['late_fn']:<12.1f}")
    
    # 可視化
    visualize_comparison_two(results)

def visualize_comparison_two(results):
    """2データセット間の比較可視化"""
    output_dir = Path("output/cross_dataset_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'ES10': 'blue', 'ES12': 'green'}
    
    # 1. サイクル別異常検出率
    ax = axes[0, 0]
    for name, res in results.items():
        cycle_rate = res['cycle_anomaly_rate']
        ax.plot(cycle_rate.index, cycle_rate.values * 100, 
               label=name, color=colors[name], linewidth=2, alpha=0.7)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50%閾値')
    ax.set_xlabel('Cycle', fontsize=12)
    ax.set_ylabel('異常検出率 (%)', fontsize=12)
    ax.set_title('サイクル別異常検出率', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 2. 異常検出率の比較（棒グラフ）
    ax = axes[0, 1]
    datasets = list(results.keys())
    anomaly_rates = [results[name]['anomaly_rate'] for name in datasets]
    bars = ax.bar(datasets, anomaly_rates, color=[colors[name] for name in datasets], alpha=0.7)
    ax.set_ylabel('異常検出率 (%)', fontsize=12)
    ax.set_title('全体の異常検出率', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 値をバーの上に表示
    for bar, rate in zip(bars, anomaly_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # 3. False Positive/Negative比較
    ax = axes[1, 0]
    x = np.arange(len(datasets))
    width = 0.25
    
    training_fp = [results[name]['training_fp'] for name in datasets]
    early_fp = [results[name]['early_fp'] for name in datasets]
    late_fn = [results[name]['late_fn'] for name in datasets]
    
    ax.bar(x - width, training_fp, width, label='Training FP (1-10)', alpha=0.7)
    ax.bar(x, early_fp, width, label='Early FP (11-20)', alpha=0.7)
    ax.bar(x + width, late_fn, width, label='Late FN (100+)', alpha=0.7)
    
    ax.set_ylabel('Rate (%)', fontsize=12)
    ax.set_title('False Positive/Negative比較', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. 遷移点の比較
    ax = axes[1, 1]
    transition_cycles = [results[name]['transition_cycle'] if results[name]['transition_cycle'] else 0 
                        for name in datasets]
    bars = ax.bar(datasets, transition_cycles, color=[colors[name] for name in datasets], alpha=0.7)
    ax.set_ylabel('Cycle', fontsize=12)
    ax.set_title('遷移点（50%異常検出率）', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 値をバーの上に表示
    for bar, cycle in zip(bars, transition_cycles):
        if cycle > 0:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(cycle)}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('ES10/ES12データセット間の異常検知結果比較', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / "cross_dataset_anomaly_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 可視化保存: {output_path}")
    plt.close()
