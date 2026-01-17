"""
Phase 0の結果を可視化するスクリプト
"""

import sys
from pathlib import Path

rul_modeling_root = Path(__file__).parent.parent
sys.path.insert(0, str(rul_modeling_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def main():
    print("=" * 80)
    print("Phase 0 可視化生成")
    print("=" * 80)
    
    # データ読み込み
    output_dir = rul_modeling_root / "output" / "phase0_analysis"
    features_df = pd.read_csv(output_dir / "es12c1_features.csv")
    corr_df = pd.read_csv(output_dir / "feature_correlations.csv")
    
    # 履歴特徴量（NaN）を除外
    corr_df = corr_df.dropna(subset=['correlation'])
    
    print(f"\n✅ データ読み込み完了")
    print(f"  - 特徴量データ: {features_df.shape}")
    print(f"  - 相関データ: {corr_df.shape}")
    
    # 1. 相関係数の棒グラフ
    print("\n[1] 相関係数の棒グラフを作成中...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 上位15特徴量
    top_corr = corr_df.head(15).copy()
    
    colors = ['red' if x < 0 else 'blue' for x in top_corr['correlation']]
    bars = ax.barh(range(len(top_corr)), top_corr['correlation'], color=colors, alpha=0.7)
    
    ax.set_yticks(range(len(top_corr)))
    ax.set_yticklabels(top_corr['feature'])
    ax.set_xlabel('Correlation with RUL', fontsize=12)
    ax.set_title('Top 15 Features: Correlation with RUL', fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.axvline(x=0.5, color='green', linestyle='--', linewidth=0.8, alpha=0.5, label='|r| = 0.5')
    ax.axvline(x=-0.5, color='green', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend()
    
    # 値をバーに表示
    for i, (bar, val) in enumerate(zip(bars, top_corr['correlation'])):
        ax.text(val + 0.02 if val > 0 else val - 0.02, i, f'{val:.3f}', 
                va='center', ha='left' if val > 0 else 'right', fontsize=9)
    
    plt.tight_layout()
    output_path = output_dir / "correlation_barplot.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ 保存: {output_path}")
    plt.close()
    
    # 2. 相関係数のヒートマップ（カテゴリ別）
    print("\n[2] カテゴリ別相関ヒートマップを作成中...")
    
    # 特徴量をカテゴリ分け
    vl_features = [f for f in corr_df['feature'] if f.startswith('vl_')]
    vo_features = [f for f in corr_df['feature'] if f.startswith('vo_')]
    other_features = [f for f in corr_df['feature'] if not (f.startswith('vl_') or f.startswith('vo_'))]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, features, title in zip(axes, 
                                    [vl_features, vo_features, other_features],
                                    ['VL Features', 'VO Features', 'Other Features']):
        subset = corr_df[corr_df['feature'].isin(features)].sort_values('abs_correlation', ascending=False)
        
        if len(subset) > 0:
            # ヒートマップ用のデータ
            data = subset[['correlation']].T
            data.columns = subset['feature']
            
            sns.heatmap(data, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                       vmin=-1, vmax=1, cbar_kws={'label': 'Correlation'},
                       ax=ax, linewidths=0.5)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylabel('')
            ax.set_yticklabels(['RUL'], rotation=0)
    
    plt.tight_layout()
    output_path = output_dir / "correlation_heatmap_by_category.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ 保存: {output_path}")
    plt.close()
    
    # 3. 散布図マトリックス（上位4特徴量 vs RUL）
    print("\n[3] 散布図マトリックスを作成中...")
    
    # cycle_number/normalized以外の上位4特徴量
    top_features = corr_df[~corr_df['feature'].isin(['cycle_number', 'cycle_normalized'])].head(4)['feature'].tolist()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        
        # 散布図
        ax.scatter(features_df[feature], features_df['rul'], alpha=0.5, s=20)
        
        # 回帰直線
        z = np.polyfit(features_df[feature], features_df['rul'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(features_df[feature].min(), features_df[feature].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
        
        # 相関係数を表示
        corr_val = corr_df[corr_df['feature'] == feature]['correlation'].values[0]
        ax.text(0.05, 0.95, f'r = {corr_val:.3f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel(feature, fontsize=11)
        ax.set_ylabel('RUL (Remaining Useful Life)', fontsize=11)
        ax.set_title(f'{feature} vs RUL', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "scatter_matrix_top4.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ 保存: {output_path}")
    plt.close()
    
    # 4. 相関係数の分布
    print("\n[4] 相関係数の分布を作成中...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(corr_df['correlation'], bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='|r| = 0.5 (High correlation)')
    ax.axvline(x=-0.5, color='green', linestyle='--', linewidth=2)
    ax.axvline(x=0, color='red', linestyle='-', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Correlation Coefficient', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Feature Correlations with RUL', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 統計情報を表示
    stats_text = f'Mean: {corr_df["correlation"].mean():.3f}\n'
    stats_text += f'Median: {corr_df["correlation"].median():.3f}\n'
    stats_text += f'High corr (|r|>0.5): {len(corr_df[corr_df["abs_correlation"] > 0.5])}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    output_path = output_dir / "correlation_distribution.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  ✅ 保存: {output_path}")
    plt.close()
    
    print("\n" + "=" * 80)
    print("✅ 可視化生成完了")
    print("=" * 80)
    print(f"\n生成された画像:")
    print(f"  1. correlation_barplot.png - 相関係数の棒グラフ")
    print(f"  2. correlation_heatmap_by_category.png - カテゴリ別ヒートマップ")
    print(f"  3. scatter_matrix_top4.png - 散布図マトリックス")
    print(f"  4. correlation_distribution.png - 相関係数の分布")
    print(f"  5. feature_trends_quick.png - 特徴量のトレンド（既存）")


if __name__ == "__main__":
    main()
