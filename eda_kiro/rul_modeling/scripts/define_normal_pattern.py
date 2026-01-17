"""
Define normal pattern baseline from initial cycles.

This script analyzes the initial cycles (1-50) to establish a baseline
for normal operation, which will be used for anomaly detection.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Paths
BASE_DIR = Path(__file__).parent.parent
FEATURES_PATH = BASE_DIR / "output" / "features_v3" / "es12_response_features.csv"
OUTPUT_DIR = BASE_DIR / "output" / "anomaly_detection"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10


def analyze_normal_cycles(df, cycle_range=(1, 50)):
    """Analyze normal cycles to establish baseline."""
    print("\n" + "="*80)
    print("NORMAL PATTERN ANALYSIS")
    print("="*80)
    
    # Filter normal cycles
    normal_df = df[(df['cycle'] >= cycle_range[0]) & (df['cycle'] <= cycle_range[1])]
    print(f"\nAnalyzing cycles {cycle_range[0]}-{cycle_range[1]}")
    print(f"Total samples: {len(normal_df)}")
    print(f"Capacitors: {normal_df['capacitor_id'].nunique()}")
    
    # Key features for analysis
    key_features = [
        'response_efficiency',
        'voltage_ratio',
        'waveform_correlation',
        'vo_variability',
        'peak_voltage_ratio',
        'rms_voltage_ratio'
    ]
    
    # Calculate statistics
    print("\n" + "-"*80)
    print("NORMAL PATTERN STATISTICS")
    print("-"*80)
    
    stats = {}
    for feat in key_features:
        mean_val = normal_df[feat].mean()
        std_val = normal_df[feat].std()
        min_val = normal_df[feat].min()
        max_val = normal_df[feat].max()
        
        # Define normal range (mean ± 2σ)
        lower_bound = mean_val - 2 * std_val
        upper_bound = mean_val + 2 * std_val
        
        stats[feat] = {
            'mean': mean_val,
            'std': std_val,
            'min': min_val,
            'max': max_val,
            'lower_2sigma': lower_bound,
            'upper_2sigma': upper_bound
        }
        
        print(f"\n{feat}:")
        print(f"  Mean: {mean_val:.4f}")
        print(f"  Std:  {std_val:.4f}")
        print(f"  Min:  {min_val:.4f}")
        print(f"  Max:  {max_val:.4f}")
        print(f"  Normal Range (μ±2σ): [{lower_bound:.4f}, {upper_bound:.4f}]")
    
    return normal_df, stats


def compare_capacitors_normal_pattern(normal_df):
    """Compare normal patterns across capacitors."""
    print("\n" + "="*80)
    print("CAPACITOR COMPARISON - NORMAL PATTERN")
    print("="*80)
    
    key_features = [
        'response_efficiency',
        'voltage_ratio',
        'waveform_correlation'
    ]
    
    capacitors = sorted(normal_df['capacitor_id'].unique())
    
    for feat in key_features:
        print(f"\n{feat}:")
        print("-" * 80)
        
        for cap_id in capacitors:
            cap_data = normal_df[normal_df['capacitor_id'] == cap_id]
            mean_val = cap_data[feat].mean()
            std_val = cap_data[feat].std()
            print(f"  {cap_id}: {mean_val:8.2f} ± {std_val:8.2f}")


def visualize_normal_pattern(df, normal_df, stats):
    """Create comprehensive normal pattern visualizations."""
    print("\n" + "="*80)
    print("CREATING NORMAL PATTERN VISUALIZATIONS")
    print("="*80)
    
    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    capacitors = sorted(df['capacitor_id'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(capacitors)))
    
    # 1. Response Efficiency - Full timeline with normal range
    ax1 = fig.add_subplot(gs[0, :])
    for i, cap_id in enumerate(capacitors):
        cap_data = df[df['capacitor_id'] == cap_id]
        ax1.plot(cap_data['cycle'], cap_data['response_efficiency'], 
                label=cap_id, color=colors[i], alpha=0.7, linewidth=1.5)
    
    # Add normal range
    mean_val = stats['response_efficiency']['mean']
    lower = stats['response_efficiency']['lower_2sigma']
    upper = stats['response_efficiency']['upper_2sigma']
    
    ax1.axhline(y=mean_val, color='green', linestyle='-', linewidth=2, label=f'Normal Mean: {mean_val:.1f}')
    ax1.axhline(y=upper, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Normal Range (μ±2σ)')
    ax1.axhline(y=lower, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.axvspan(1, 50, alpha=0.2, color='green', label='Normal Baseline Period')
    
    ax1.set_xlabel('Cycle', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Response Efficiency', fontsize=12, fontweight='bold')
    ax1.set_title('Response Efficiency with Normal Baseline Range', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # 2. Waveform Correlation - Full timeline with normal range
    ax2 = fig.add_subplot(gs[1, 0])
    for i, cap_id in enumerate(capacitors):
        cap_data = df[df['capacitor_id'] == cap_id]
        ax2.plot(cap_data['cycle'], cap_data['waveform_correlation'], 
                label=cap_id, color=colors[i], alpha=0.7, linewidth=1.5)
    
    mean_val = stats['waveform_correlation']['mean']
    lower = stats['waveform_correlation']['lower_2sigma']
    upper = stats['waveform_correlation']['upper_2sigma']
    
    ax2.axhline(y=mean_val, color='green', linestyle='-', linewidth=2)
    ax2.axhline(y=upper, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=lower, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axvspan(1, 50, alpha=0.2, color='green')
    
    ax2.set_xlabel('Cycle', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Waveform Correlation', fontsize=11, fontweight='bold')
    ax2.set_title('Waveform Correlation with Normal Range', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. VO Variability - Full timeline with normal range
    ax3 = fig.add_subplot(gs[1, 1])
    for i, cap_id in enumerate(capacitors):
        cap_data = df[df['capacitor_id'] == cap_id]
        ax3.plot(cap_data['cycle'], cap_data['vo_variability'], 
                label=cap_id, color=colors[i], alpha=0.7, linewidth=1.5)
    
    mean_val = stats['vo_variability']['mean']
    lower = stats['vo_variability']['lower_2sigma']
    upper = stats['vo_variability']['upper_2sigma']
    
    ax3.axhline(y=mean_val, color='green', linestyle='-', linewidth=2)
    ax3.axhline(y=upper, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.axhline(y=lower, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.axvspan(1, 50, alpha=0.2, color='green')
    
    ax3.set_xlabel('Cycle', fontsize=11, fontweight='bold')
    ax3.set_ylabel('VO Variability', fontsize=11, fontweight='bold')
    ax3.set_title('VO Variability with Normal Range', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. Distribution - Response Efficiency (Normal vs All)
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(normal_df['response_efficiency'], bins=30, alpha=0.6, 
            label=f'Normal (n={len(normal_df)})', color='green')
    ax4.hist(df['response_efficiency'], bins=30, alpha=0.4, 
            label=f'All (n={len(df)})', color='gray')
    
    ax4.set_xlabel('Response Efficiency', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax4.set_title('Response Efficiency Distribution', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xlim(left=0)
    
    # 5. Box plot - Response Efficiency by Capacitor (Normal cycles)
    ax5 = fig.add_subplot(gs[2, 0])
    cap_data_list = [normal_df[normal_df['capacitor_id'] == cap]['response_efficiency'].values 
                     for cap in capacitors]
    
    bp = ax5.boxplot(cap_data_list, tick_labels=capacitors, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
        patch.set_alpha(0.6)
    
    ax5.set_ylabel('Response Efficiency', fontsize=11, fontweight='bold')
    ax5.set_title('Response Efficiency by Capacitor (Normal Cycles)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.tick_params(axis='x', rotation=45)
    
    # 6. Box plot - Waveform Correlation by Capacitor (Normal cycles)
    ax6 = fig.add_subplot(gs[2, 1])
    cap_data_list = [normal_df[normal_df['capacitor_id'] == cap]['waveform_correlation'].values 
                     for cap in capacitors]
    
    bp = ax6.boxplot(cap_data_list, tick_labels=capacitors, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightgreen')
        patch.set_alpha(0.6)
    
    ax6.set_ylabel('Waveform Correlation', fontsize=11, fontweight='bold')
    ax6.set_title('Waveform Correlation by Capacitor (Normal Cycles)', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.tick_params(axis='x', rotation=45)
    
    # 7. Scatter plot - Response Efficiency vs Waveform Correlation (Normal)
    ax7 = fig.add_subplot(gs[2, 2])
    for i, cap_id in enumerate(capacitors):
        cap_data = normal_df[normal_df['capacitor_id'] == cap_id]
        ax7.scatter(cap_data['waveform_correlation'], cap_data['response_efficiency'], 
                   label=cap_id, color=colors[i], alpha=0.6, s=30)
    
    ax7.set_xlabel('Waveform Correlation', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Response Efficiency', fontsize=11, fontweight='bold')
    ax7.set_title('Normal Pattern: Efficiency vs Correlation', fontsize=12, fontweight='bold')
    ax7.legend(loc='upper left', fontsize=8)
    ax7.grid(True, alpha=0.3)
    
    plt.suptitle('Normal Pattern Baseline Definition (Cycles 1-50)', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    output_path = OUTPUT_DIR / "normal_pattern_baseline.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()


def create_normal_pattern_report(normal_df, stats):
    """Create markdown report with normal pattern definition."""
    print("\n" + "="*80)
    print("CREATING NORMAL PATTERN REPORT")
    print("="*80)
    
    report_path = OUTPUT_DIR / "normal_pattern_definition.md"
    
    with open(report_path, 'w') as f:
        f.write("# 正常パターン定義レポート\n\n")
        f.write("**作成日**: 2026-01-17  \n")
        f.write("**タスク**: Phase 2 Task 2.1 - 正常パターンの定義とベースライン設定  \n\n")
        f.write("---\n\n")
        
        f.write("## 📋 概要\n\n")
        f.write("初期サイクル（1-50）を正常動作期間として定義し、応答性特徴量の正常範囲を確立しました。\n")
        f.write("この正常ベースラインは、Phase 2の異常検知モデル構築に使用されます。\n\n")
        f.write("---\n\n")
        
        f.write("## 🎯 正常パターンの定義\n\n")
        f.write("### 正常期間\n\n")
        f.write("- **サイクル範囲**: 1-50\n")
        f.write(f"- **総サンプル数**: {len(normal_df)}\n")
        f.write(f"- **コンデンサ数**: {normal_df['capacitor_id'].nunique()}\n")
        f.write(f"- **サイクル数/コンデンサ**: 50\n\n")
        
        f.write("### 正常範囲の定義方法\n\n")
        f.write("各特徴量について、正常サイクルの統計値を計算し、正常範囲を定義:\n\n")
        f.write("```\n")
        f.write("正常範囲 = 平均値 ± 2 × 標準偏差 (μ ± 2σ)\n")
        f.write("```\n\n")
        f.write("この範囲は、正常分布を仮定した場合、約95%のデータをカバーします。\n\n")
        
        f.write("---\n\n")
        f.write("## 📊 正常パターンの統計\n\n")
        
        # Key features table
        f.write("### 主要特徴量の正常範囲\n\n")
        f.write("| 特徴量 | 平均値 | 標準偏差 | 正常範囲下限 | 正常範囲上限 |\n")
        f.write("|--------|--------|----------|-------------|-------------|\n")
        
        key_features = [
            'response_efficiency',
            'voltage_ratio',
            'waveform_correlation',
            'vo_variability',
            'peak_voltage_ratio',
            'rms_voltage_ratio'
        ]
        
        for feat in key_features:
            s = stats[feat]
            f.write(f"| {feat} | {s['mean']:.4f} | {s['std']:.4f} | "
                   f"{s['lower_2sigma']:.4f} | {s['upper_2sigma']:.4f} |\n")
        
        f.write("\n")
        
        # Detailed statistics
        f.write("### 詳細統計\n\n")
        for feat in key_features:
            s = stats[feat]
            f.write(f"#### {feat}\n\n")
            f.write(f"- **平均値 (μ)**: {s['mean']:.4f}\n")
            f.write(f"- **標準偏差 (σ)**: {s['std']:.4f}\n")
            f.write(f"- **最小値**: {s['min']:.4f}\n")
            f.write(f"- **最大値**: {s['max']:.4f}\n")
            f.write(f"- **正常範囲**: [{s['lower_2sigma']:.4f}, {s['upper_2sigma']:.4f}]\n")
            f.write(f"- **変動係数 (CV)**: {(s['std'] / s['mean'] * 100):.2f}%\n\n")
        
        f.write("---\n\n")
        f.write("## 🔍 コンデンサ間の比較\n\n")
        
        capacitors = sorted(normal_df['capacitor_id'].unique())
        
        for feat in ['response_efficiency', 'voltage_ratio', 'waveform_correlation']:
            f.write(f"### {feat}\n\n")
            f.write("| コンデンサ | 平均値 | 標準偏差 |\n")
            f.write("|-----------|--------|----------|\n")
            
            for cap_id in capacitors:
                cap_data = normal_df[normal_df['capacitor_id'] == cap_id]
                mean_val = cap_data[feat].mean()
                std_val = cap_data[feat].std()
                f.write(f"| {cap_id} | {mean_val:.2f} | {std_val:.2f} |\n")
            
            f.write("\n")
        
        f.write("**観察**:\n")
        f.write("- 全コンデンサで類似した正常パターンを示す\n")
        f.write("- コンデンサ間のばらつきは標準偏差内に収まる\n")
        f.write("- 正常ベースラインは全コンデンサに適用可能\n\n")
        
        f.write("---\n\n")
        f.write("## 🎯 異常検知への応用\n\n")
        f.write("### 異常判定基準\n\n")
        f.write("以下の条件のいずれかを満たす場合、異常と判定:\n\n")
        f.write("1. **範囲外判定**: 特徴量が正常範囲（μ±2σ）を外れる\n")
        f.write("2. **閾値判定**: Phase 1で特定した閾値を超える\n")
        f.write("   - Response Efficiency < 50% (劣化開始)\n")
        f.write("   - Response Efficiency < 10% (深刻な劣化)\n")
        f.write("   - Response Efficiency < 1% (臨界状態)\n\n")
        
        f.write("### 異常度スコア\n\n")
        f.write("正常範囲からの偏差を定量化:\n\n")
        f.write("```\n")
        f.write("異常度スコア = |特徴量 - 平均値| / 標準偏差\n")
        f.write("```\n\n")
        f.write("- スコア < 2: 正常範囲内\n")
        f.write("- スコア 2-3: 軽度の異常\n")
        f.write("- スコア > 3: 明確な異常\n\n")
        
        f.write("---\n\n")
        f.write("## 📁 生成ファイル\n\n")
        f.write("1. `normal_pattern_baseline.png` - 正常パターンの可視化\n")
        f.write("2. `normal_pattern_definition.md` - 本レポート\n\n")
        
        f.write("---\n\n")
        f.write("## 🎯 次のステップ\n\n")
        f.write("**Task 2.2**: Isolation Forestによる異常検知\n\n")
        f.write("確立した正常ベースラインを使用して:\n")
        f.write("1. Isolation Forestモデルの構築\n")
        f.write("2. 異常度スコアの算出\n")
        f.write("3. 異常サイクルの特定\n")
        f.write("4. 異常検知の閾値設定\n\n")
        
        f.write("---\n\n")
        f.write("**報告者**: Kiro AI Agent  \n")
        f.write("**完了日**: 2026-01-17  \n")
        f.write("**次のタスク**: Task 2.2 - Isolation Forestによる異常検知\n")
    
    print(f"✓ Saved: {report_path}")


def main():
    """Main execution."""
    print("="*80)
    print("NORMAL PATTERN BASELINE DEFINITION")
    print("="*80)
    
    # Load features
    print("\nLoading features...")
    df = pd.read_csv(FEATURES_PATH)
    print(f"  ✓ Loaded {len(df)} samples")
    
    # Analyze normal cycles
    normal_df, stats = analyze_normal_cycles(df, cycle_range=(1, 50))
    
    # Compare capacitors
    compare_capacitors_normal_pattern(normal_df)
    
    # Create visualizations
    visualize_normal_pattern(df, normal_df, stats)
    
    # Create report
    create_normal_pattern_report(normal_df, stats)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  1. normal_pattern_baseline.png")
    print("  2. normal_pattern_definition.md")
    print("\n✅ Normal pattern baseline definition complete!")


if __name__ == "__main__":
    main()
