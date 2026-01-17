"""
Phase 0: 探索的特徴量分析

モデル実装前に、特徴量とRULの相関を確認し、有効な特徴量を特定する。
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
rul_modeling_root = Path(__file__).parent.parent
eda_root = rul_modeling_root.parent

# パスを追加
sys.path.insert(0, str(rul_modeling_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# RUL modeling のコードを使用
from src.feature_extraction import CycleFeatureExtractor
from src.utils import load_es12_cycle_data


def step1_test_extraction(es12_path: str, cap_id: str = "ES12C1", n_cycles: int = 10):
    """
    Step 1: 特徴量抽出の動作確認（少数サイクル）
    """
    print("\n" + "=" * 80)
    print(f"[Step 1] {cap_id}の最初の{n_cycles}サイクルで特徴量抽出テスト")
    print("=" * 80)
    
    extractor = CycleFeatureExtractor()
    features_list = []
    
    for cycle in range(1, n_cycles + 1):
        vl, vo = load_es12_cycle_data(es12_path, cap_id, cycle)
        features = extractor.extract_all_features(vl, vo, cycle)
        features['capacitor_id'] = cap_id
        features_list.append(features)
    
    df = pd.DataFrame(features_list)
    
    print(f"\n✅ 抽出した特徴量: {len(df.columns) - 1}個（capacitor_id除く）")
    print(f"✅ データ形状: {df.shape}")
    print(f"\n最初の3サイクル:")
    print(df[['cycle_number', 'vl_mean', 'vo_mean', 'voltage_ratio']].head(3))
    
    # 欠損値チェック
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"\n⚠️  欠損値あり:")
        print(null_counts[null_counts > 0])
    else:
        print(f"\n✅ 欠損値なし")
    
    return df


def step2_correlation_analysis(es12_path: str, cap_id: str = "ES12C1"):
    """
    Step 2: 1つのコンデンサで相関分析（全サイクル）
    """
    print("\n" + "=" * 80)
    print(f"[Step 2] {cap_id}の全サイクルで相関分析")
    print("=" * 80)
    
    extractor = CycleFeatureExtractor()
    features_list = []
    
    # 履歴特徴量なしで高速化
    print("  履歴特徴量なしで抽出中...")
    for cycle in range(1, 201):
        if cycle % 50 == 0:
            print(f"    処理中: {cycle}/200 サイクル")
        vl, vo = load_es12_cycle_data(es12_path, cap_id, cycle)
        features = extractor.extract_all_features(vl, vo, cycle, history_df=None)
        features['capacitor_id'] = cap_id
        features_list.append(features)
    
    df = pd.DataFrame(features_list)
    
    # RULを計算
    df['rul'] = 200 - df['cycle_number']
    
    print(f"\n✅ データセット形状: {df.shape}")
    print(f"\n統計情報:")
    print(df[['cycle_number', 'voltage_ratio', 'rul']].describe())
    
    # 相関分析
    print(f"\n" + "-" * 80)
    print("相関分析")
    print("-" * 80)
    
    # 数値特徴量のみ選択
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('rul')
    if 'capacitor_id' in numeric_cols:
        numeric_cols.remove('capacitor_id')
    
    # 相関係数を計算
    correlations = []
    for feature in numeric_cols:
        corr, p_value = pearsonr(df[feature], df['rul'])
        correlations.append({
            'feature': feature,
            'correlation': corr,
            'abs_correlation': abs(corr),
            'p_value': p_value
        })
    
    corr_df = pd.DataFrame(correlations).sort_values('abs_correlation', ascending=False)
    
    print(f"\n特徴量とRULの相関係数（絶対値の降順）:")
    print(corr_df.head(15).to_string(index=False))
    
    # 高相関特徴量
    high_corr = corr_df[corr_df['abs_correlation'] > 0.5]
    print(f"\n✅ 高相関特徴量（|r| > 0.5）: {len(high_corr)}個")
    print(high_corr['feature'].tolist())
    
    # 低相関特徴量
    low_corr = corr_df[corr_df['abs_correlation'] < 0.1]
    print(f"\n⚠️  低相関特徴量（|r| < 0.1）: {len(low_corr)}個")
    print(low_corr['feature'].tolist())
    
    return df, corr_df


def step3_visualize_trends(df: pd.DataFrame, corr_df: pd.DataFrame, output_dir: Path):
    """
    Step 3: 特徴量のトレンドを可視化
    """
    print("\n" + "=" * 80)
    print("[Step 3] 特徴量のトレンドを可視化")
    print("=" * 80)
    
    # Top 6 features by correlation
    top_features = corr_df.head(6)['feature'].tolist()
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        ax.plot(df['cycle_number'], df[feature], marker='o', markersize=2, alpha=0.6)
        ax.set_xlabel('Cycle Number', fontsize=10)
        ax.set_ylabel(feature, fontsize=10)
        
        # 相関係数を表示
        corr_value = corr_df[corr_df['feature'] == feature]['correlation'].values[0]
        ax.set_title(f'{feature}\n(r = {corr_value:.3f})', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / "feature_trends.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 保存: {output_path}")
    plt.close()


def step4_multiple_capacitors(es12_path: str, cap_ids: list):
    """
    Step 4: 複数コンデンサでの一貫性確認
    """
    print("\n" + "=" * 80)
    print(f"[Step 4] 複数コンデンサ（{len(cap_ids)}個）での一貫性確認")
    print("=" * 80)
    
    extractor = CycleFeatureExtractor()
    all_correlations = {}
    
    for cap_id in cap_ids:
        print(f"  処理中: {cap_id}...", end=" ", flush=True)
        
        features_list = []
        
        # 履歴特徴量なしで高速化
        for cycle in range(1, 201):
            vl, vo = load_es12_cycle_data(es12_path, cap_id, cycle)
            features = extractor.extract_all_features(vl, vo, cycle, history_df=None)
            features['capacitor_id'] = cap_id
            features_list.append(features)
        
        df = pd.DataFrame(features_list)
        df['rul'] = 200 - df['cycle_number']
        
        # 相関係数を計算
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('rul')
        if 'capacitor_id' in numeric_cols:
            numeric_cols.remove('capacitor_id')
        
        correlations = {}
        for feature in numeric_cols:
            corr, _ = pearsonr(df[feature], df['rul'])
            correlations[feature] = corr
        
        all_correlations[cap_id] = correlations
        print("✓")
    
    # DataFrameに変換
    corr_matrix = pd.DataFrame(all_correlations).T
    
    print(f"\n各コンデンサでの相関係数（上位5特徴量）:")
    mean_corr = corr_matrix.mean().sort_values(ascending=False, key=abs)
    top_5_features = mean_corr.head(5).index.tolist()
    print(corr_matrix[top_5_features].to_string())
    
    # 平均相関係数
    print(f"\n平均相関係数（絶対値の降順、上位10）:")
    print(mean_corr.head(10).to_string())
    
    # 一貫性の高い特徴量
    consistent_features = []
    for feature in corr_matrix.columns:
        if (corr_matrix[feature].abs() > 0.3).all():
            consistent_features.append(feature)
    
    print(f"\n✅ 一貫性の高い特徴量（全コンデンサで|r| > 0.3）: {len(consistent_features)}個")
    if consistent_features:
        print(consistent_features)
    else:
        print("  なし")
    
    return corr_matrix, consistent_features


def main():
    """Main execution"""
    print("=" * 80)
    print("Phase 0: 探索的特徴量分析")
    print("=" * 80)
    
    # データ読み込み
    print("\n[データ読み込み]")
    es12_path = eda_root / "data" / "raw" / "ES12.mat"
    
    if not es12_path.exists():
        print(f"❌ エラー: {es12_path} が見つかりません")
        return
    
    print(f"✅ ES12データ: {es12_path}")
    es12_path_str = str(es12_path)
    
    # 出力ディレクトリ
    output_dir = Path(__file__).parent.parent / "output" / "phase0_analysis"
    
    # Step 1: 動作確認
    df_test = step1_test_extraction(es12_path_str, "ES12C1", 10)
    
    # Step 2: 相関分析
    df_full, corr_df = step2_correlation_analysis(es12_path_str, "ES12C1")
    
    # Step 3: 可視化
    step3_visualize_trends(df_full, corr_df, output_dir)
    
    # Step 4: 複数コンデンサ
    cap_ids = [f"ES12C{i}" for i in range(1, 9)]
    corr_matrix, consistent_features = step4_multiple_capacitors(es12_path_str, cap_ids)
    
    # 結果サマリー
    print("\n" + "=" * 80)
    print("Phase 0 完了サマリー")
    print("=" * 80)
    
    high_corr = corr_df[corr_df['abs_correlation'] > 0.5]
    print(f"\n✅ 高相関特徴量（|r| > 0.5）: {len(high_corr)}個")
    print(f"✅ 一貫性の高い特徴量（全コンデンサで|r| > 0.3）: {len(consistent_features)}個")
    
    if len(high_corr) >= 5:
        print(f"\n🎉 成功基準達成！相関の高い特徴量が十分にあります")
        print(f"   → Phase 1（データセット構築）に進めます")
    else:
        print(f"\n⚠️  高相関特徴量が少ないです（目標: 5個以上）")
        print(f"   → 新規特徴量の追加を検討してください")
    
    print(f"\n出力ディレクトリ: {output_dir}")


if __name__ == "__main__":
    main()
