"""
Phase 0: 探索的特徴量分析（高速版）

1つのコンデンサのみで相関分析を実施
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
rul_modeling_root = Path(__file__).parent.parent
eda_root = rul_modeling_root.parent

sys.path.insert(0, str(rul_modeling_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

from src.feature_extraction import CycleFeatureExtractor
from src.utils import load_es12_cycle_data


def main():
    print("=" * 80)
    print("Phase 0: 探索的特徴量分析（高速版）")
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
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ES12C1の全サイクルで特徴量抽出
    print("\n" + "=" * 80)
    print("[Step 1] ES12C1の全サイクルで特徴量抽出")
    print("=" * 80)
    
    cap_id = "ES12C1"
    extractor = CycleFeatureExtractor()
    features_list = []
    
    print(f"  処理中: {cap_id}")
    for cycle in range(1, 201):
        if cycle % 20 == 0:  # より頻繁に進捗表示
            print(f"    {cycle}/200 サイクル完了 ({cycle/200*100:.0f}%)", flush=True)
        
        vl, vo = load_es12_cycle_data(es12_path_str, cap_id, cycle)
        
        # 履歴特徴量なしで抽出（高速化）
        features = extractor.extract_all_features(vl, vo, cycle, history_df=None)
        features['capacitor_id'] = cap_id
        features_list.append(features)
    
    df = pd.DataFrame(features_list)
    df['rul'] = 200 - df['cycle_number']
    
    print(f"\n✅ データセット形状: {df.shape}")
    print(f"✅ 特徴量数: {len(df.columns) - 2}個（capacitor_id, rul除く）")
    
    # 相関分析
    print("\n" + "=" * 80)
    print("[Step 2] 相関分析")
    print("=" * 80)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols.remove('rul')
    if 'capacitor_id' in numeric_cols:
        numeric_cols.remove('capacitor_id')
    
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
    
    print(f"\n特徴量とRULの相関係数（上位15）:")
    print(corr_df.head(15)[['feature', 'correlation', 'p_value']].to_string(index=False))
    
    # 高相関特徴量
    high_corr = corr_df[corr_df['abs_correlation'] > 0.5]
    print(f"\n✅ 高相関特徴量（|r| > 0.5）: {len(high_corr)}個")
    for idx, row in high_corr.iterrows():
        print(f"  - {row['feature']}: r = {row['correlation']:.3f}")
    
    # 低相関特徴量
    low_corr = corr_df[corr_df['abs_correlation'] < 0.1]
    print(f"\n⚠️  低相関特徴量（|r| < 0.1）: {len(low_corr)}個")
    
    # 可視化
    print("\n" + "=" * 80)
    print("[Step 3] 可視化")
    print("=" * 80)
    
    top_features = corr_df.head(6)['feature'].tolist()
    
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        ax.plot(df['cycle_number'], df[feature], marker='o', markersize=2, alpha=0.6)
        ax.set_xlabel('Cycle Number', fontsize=10)
        ax.set_ylabel(feature, fontsize=10)
        
        corr_value = corr_df[corr_df['feature'] == feature]['correlation'].values[0]
        ax.set_title(f'{feature}\n(r = {corr_value:.3f})', fontsize=11)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = output_dir / "feature_trends_quick.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 保存: {output_path}")
    plt.close()
    
    # CSVに保存
    csv_path = output_dir / "es12c1_features.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ 保存: {csv_path}")
    
    corr_csv_path = output_dir / "feature_correlations.csv"
    corr_df.to_csv(corr_csv_path, index=False)
    print(f"✅ 保存: {corr_csv_path}")
    
    # 結果サマリー
    print("\n" + "=" * 80)
    print("Phase 0 完了サマリー")
    print("=" * 80)
    
    print(f"\n✅ 高相関特徴量（|r| > 0.5）: {len(high_corr)}個")
    
    if len(high_corr) >= 5:
        print(f"\n🎉 成功基準達成！相関の高い特徴量が十分にあります")
        print(f"   → Phase 1（データセット構築）に進めます")
    else:
        print(f"\n⚠️  高相関特徴量が少ないです（目標: 5個以上）")
        print(f"   → 新規特徴量の追加を検討してください")
    
    print(f"\n出力ディレクトリ: {output_dir}")


if __name__ == "__main__":
    main()
