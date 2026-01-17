"""
最終レポート用の統合可視化作成

全Phase の成果を1つの図にまとめる
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def create_comprehensive_visualization():
    """包括的な可視化を作成"""
    
    # データ読み込み
    features = pd.read_csv("output/features_v3/es12_response_features.csv")
    anomaly = pd.read_csv("output/anomaly_detection/one_class_svm_v2_results.csv")
    degradation = pd.read_csv("output/degradation_prediction/features_with_degradation_score.csv")
    
    # 統合
    data = features.merge(anomaly[['capacitor_id', 'cycle', 'is_anomaly']], 
                         on=['capacitor_id', 'cycle'])
    data = data.merge(degradation[['capacitor_id', 'cycle', 'degradation_score']], 
                     on=['capacitor_id', 'cycle'])
    
    # 図の作成
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Response Efficiency の時系列
    ax1 = fig.add_subplot(gs[0, 0])
    for cap in sorted(data['capacitor_id'].unique()):
        cap_data = data[data['capacitor_id'] == cap]
        ax1.plot(cap_data['cycle'], cap_data['response_efficiency'], 
                alpha=0.6, linewidth=1.5)
    ax1.set_xlabel('Cycle', fontsize=11)
    ax1.set_ylabel('Response Efficiency', fontsize=11)
    ax1.set_title('(a) Response Efficiency推移', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 150)
    
    # 2. Waveform Correlation の時系列
    ax2 = fig.add_subplot(gs[0, 1])
    for cap in sorted(data['capacitor_id'].unique()):
        cap_data = data[data['capacitor_id'] == cap]
        ax2.plot(cap_data['cycle'], cap_data['waveform_correlation'], 
                alpha=0.6, linewidth=1.5)
    ax2.set_xlabel('Cycle', fontsize=11)
    ax2.set_ylabel('Waveform Correlation', fontsize=11)
    ax2.set_title('(b) Waveform Correlation推移', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. 異常検出率（サイクル別）
    ax3 = fig.add_subplot(gs[0, 2])
    cycle_anomaly = data.groupby('cycle')['is_anomaly'].mean() * 100
    ax3.plot(cycle_anomaly.index, cycle_anomaly.values, 
            color='red', linewidth=2.5)
    ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(x=13, color='green', linestyle='--', alpha=0.5, label='遷移点 (Cycle 13)')
    ax3.set_xlabel('Cycle', fontsize=11)
    ax3.set_ylabel('異常検出率 (%)', fontsize=11)
    ax3.set_title('(c) サイクル別異常検出率', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # 4. 劣化度スコアの時系列
    ax4 = fig.add_subplot(gs[1, 0])
    for cap in sorted(data['capacitor_id'].unique()):
        cap_data = data[data['capacitor_id'] == cap]
        ax4.plot(cap_data['cycle'], cap_data['degradation_score'], 
                alpha=0.6, linewidth=1.5)
    ax4.axhline(y=0.25, color='orange', linestyle='--', alpha=0.5, label='Degrading')
    ax4.axhline(y=0.50, color='red', linestyle='--', alpha=0.5, label='Severe')
    ax4.set_xlabel('Cycle', fontsize=11)
    ax4.set_ylabel('Degradation Score', fontsize=11)
    ax4.set_title('(d) 劣化度スコア推移', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # 5. 劣化ステージ分布
    ax5 = fig.add_subplot(gs[1, 1])
    stage_counts = []
    stage_labels = []
    if (data['degradation_score'] < 0.25).sum() > 0:
        stage_counts.append((data['degradation_score'] < 0.25).sum())
        stage_labels.append('Normal\n(0-0.25)')
    if ((data['degradation_score'] >= 0.25) & (data['degradation_score'] < 0.50)).sum() > 0:
        stage_counts.append(((data['degradation_score'] >= 0.25) & (data['degradation_score'] < 0.50)).sum())
        stage_labels.append('Degrading\n(0.25-0.50)')
    if ((data['degradation_score'] >= 0.50) & (data['degradation_score'] < 0.75)).sum() > 0:
        stage_counts.append(((data['degradation_score'] >= 0.50) & (data['degradation_score'] < 0.75)).sum())
        stage_labels.append('Severe\n(0.50-0.75)')
    if (data['degradation_score'] >= 0.75).sum() > 0:
        stage_counts.append((data['degradation_score'] >= 0.75).sum())
        stage_labels.append('Critical\n(0.75-1.0)')
    
    colors_stage = ['green', 'orange', 'red', 'darkred'][:len(stage_counts)]
    bars = ax5.bar(range(len(stage_counts)), stage_counts, color=colors_stage, alpha=0.7)
    ax5.set_xticks(range(len(stage_labels)))
    ax5.set_xticklabels(stage_labels, fontsize=10)
    ax5.set_ylabel('サンプル数', fontsize=11)
    ax5.set_title('(e) 劣化ステージ分布', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 値を表示
    for bar, count in zip(bars, stage_counts):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/len(data)*100:.1f}%)',
                ha='center', va='bottom', fontsize=9)
    
    # 6. 異常 vs 正常の特徴量比較
    ax6 = fig.add_subplot(gs[1, 2])
    normal_data = data[data['is_anomaly'] == 0]
    anomaly_data = data[data['is_anomaly'] == 1]
    
    features_to_compare = ['waveform_correlation', 'vo_variability', 'vl_variability']
    x_pos = np.arange(len(features_to_compare))
    width = 0.35
    
    normal_means = [normal_data[f].mean() for f in features_to_compare]
    anomaly_means = [anomaly_data[f].mean() for f in features_to_compare]
    
    ax6.bar(x_pos - width/2, normal_means, width, label='Normal', alpha=0.7, color='blue')
    ax6.bar(x_pos + width/2, anomaly_means, width, label='Anomaly', alpha=0.7, color='red')
    
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(['Correlation', 'VO Var.', 'VL Var.'], fontsize=10)
    ax6.set_ylabel('平均値', fontsize=11)
    ax6.set_title('(f) Normal vs Anomaly特徴量比較', fontsize=12, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 7. モデル性能サマリー（テキスト）
    ax7 = fig.add_subplot(gs[2, :])
    ax7.axis('off')
    
    summary_text = f"""
    【プロジェクト成果サマリー】
    
    ■ データセット: ES12（8コンデンサ × 200サイクル = 1,600サンプル）
    
    ■ Phase 1: VL-VO関係性分析
      • 15個の応答性特徴量を設計・抽出
      • Response Efficiency: 77-117% → 1.1-1.2%（98.5%減少）
      • Waveform Correlation: 0.83 → 0.9998（波形単純化）
    
    ■ Phase 2: 異常検知モデル（One-Class SVM v2, nu=0.05）
      • 異常検出率: 90.8%（1,452/1,600サンプル）
      • Training False Positive: 5.0%（優秀）
      • Late False Negative: 5.2%（優秀）
      • 遷移点: Cycle 13（50%異常検出率）
      • 使用特徴量: 波形特性のみ（7特徴量）
    
    ■ Phase 3: 劣化度予測モデル（Random Forest）
      • Test MAE: 0.0036（目標0.1を大幅達成）
      • Test R²: 0.9996（極めて高精度）
      • 劣化度を0-1スケールで定量化
      • 最重要特徴量: waveform_correlation (93.3%)
    
    ■ Phase 4: モデル汎化性能検証
      • ES10/ES14データには汎化せず
      • データセットごとにモデル学習が必要
    
    ■ 実用化の可能性
      • 同一条件下のコンデンサ劣化監視に適用可能
      • リアルタイム異常検知システムへの統合が可能
      • 劣化度の定量的評価により、予防保全が可能
    """
    
    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('コンデンサ劣化予測プロジェクト - 統合レポート', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # 保存
    output_dir = Path("output/final_report")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "comprehensive_visualization.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 統合可視化保存: {output_path}")
    plt.close()

if __name__ == "__main__":
    print("="*60)
    print("最終レポート用統合可視化の作成")
    print("="*60)
    create_comprehensive_visualization()
    print("\n✓ 完了")
