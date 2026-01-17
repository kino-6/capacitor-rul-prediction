"""
Analyze degradation patterns and identify failure threshold candidates.

This script performs detailed analysis of response features to:
1. Visualize degradation patterns over time
2. Identify threshold candidates for failure detection
3. Define degradation stages (Normal, Degrading, Severe, Critical)
4. Analyze feature distributions across stages
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
OUTPUT_DIR = BASE_DIR / "output" / "features_v3"

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10


def analyze_threshold_candidates(df):
    """Analyze threshold candidates for failure detection."""
    print("\n" + "="*80)
    print("THRESHOLD CANDIDATE ANALYSIS")
    print("="*80)
    
    # Define threshold candidates
    thresholds = {
        'response_efficiency': [50, 10, 5, 2, 1],
        'efficiency_degradation_rate': [0.5, 0.8, 0.9, 0.95, 0.98],
        'waveform_correlation': [0.90, 0.95, 0.98, 0.99, 0.995]
    }
    
    results = []
    
    for feature, threshold_list in thresholds.items():
        print(f"\n{feature}:")
        print("-" * 80)
        
        for threshold in threshold_list:
            # Count samples below/above threshold
            if feature == 'response_efficiency':
                # Lower is worse
                failing = df[df[feature] < threshold]
                condition = f"< {threshold}"
            elif feature == 'efficiency_degradation_rate':
                # Higher is worse (more degraded)
                # Skip NaN values
                valid_df = df[df[feature].notna()]
                failing = valid_df[valid_df[feature] > threshold]
                condition = f"> {threshold}"
            else:  # waveform_correlation
                # Higher is worse (more simplified)
                failing = df[df[feature] > threshold]
                condition = f"> {threshold}"
            
            failing_pct = len(failing) / len(df) * 100
            
            # Find first cycle where each capacitor crosses threshold
            first_cycles = []
            for cap_id in df['capacitor_id'].unique():
                cap_data = df[df['capacitor_id'] == cap_id].sort_values('cycle')
                
                if feature == 'response_efficiency':
                    crossing = cap_data[cap_data[feature] < threshold]
                elif feature == 'efficiency_degradation_rate':
                    cap_data_valid = cap_data[cap_data[feature].notna()]
                    crossing = cap_data_valid[cap_data_valid[feature] > threshold]
                else:
                    crossing = cap_data[cap_data[feature] > threshold]
                
                if len(crossing) > 0:
                    first_cycles.append(crossing.iloc[0]['cycle'])
            
            avg_first_cycle = np.mean(first_cycles) if first_cycles else None
            
            print(f"  {condition}: {len(failing)} samples ({failing_pct:.1f}%)")
            if avg_first_cycle:
                print(f"    Average first detection: Cycle {avg_first_cycle:.1f}")
            
            results.append({
                'feature': feature,
                'threshold': threshold,
                'condition': condition,
                'failing_samples': len(failing),
                'failing_pct': failing_pct,
                'avg_first_cycle': avg_first_cycle
            })
    
    return pd.DataFrame(results)


def define_degradation_stages(df):
    """Define degradation stages based on response efficiency."""
    print("\n" + "="*80)
    print("DEGRADATION STAGE DEFINITION")
    print("="*80)
    
    # Define stages based on response efficiency
    def classify_stage(eff):
        if eff > 50:
            return 'Normal'
        elif eff > 10:
            return 'Degrading'
        elif eff > 1:
            return 'Severe'
        else:
            return 'Critical'
    
    df['degradation_stage'] = df['response_efficiency'].apply(classify_stage)
    
    # Count samples per stage
    stage_counts = df['degradation_stage'].value_counts()
    print("\nSample Distribution by Stage:")
    print("-" * 80)
    for stage in ['Normal', 'Degrading', 'Severe', 'Critical']:
        if stage in stage_counts:
            count = stage_counts[stage]
            pct = count / len(df) * 100
            print(f"  {stage:12s}: {count:4d} samples ({pct:5.1f}%)")
    
    # Analyze feature ranges per stage
    print("\nFeature Ranges by Stage:")
    print("-" * 80)
    
    key_features = [
        'response_efficiency',
        'voltage_ratio',
        'waveform_correlation',
        'efficiency_degradation_rate'
    ]
    
    for stage in ['Normal', 'Degrading', 'Severe', 'Critical']:
        stage_data = df[df['degradation_stage'] == stage]
        if len(stage_data) == 0:
            continue
        
        print(f"\n{stage}:")
        for feat in key_features:
            if feat == 'efficiency_degradation_rate':
                # Skip NaN values
                valid_data = stage_data[stage_data[feat].notna()]
                if len(valid_data) == 0:
                    continue
                mean_val = valid_data[feat].mean()
                std_val = valid_data[feat].std()
            else:
                mean_val = stage_data[feat].mean()
                std_val = stage_data[feat].std()
            
            print(f"  {feat:30s}: {mean_val:8.2f} ± {std_val:8.2f}")
    
    return df


def visualize_degradation_patterns(df):
    """Create comprehensive degradation pattern visualizations."""
    print("\n" + "="*80)
    print("CREATING DEGRADATION PATTERN VISUALIZATIONS")
    print("="*80)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    capacitors = sorted(df['capacitor_id'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(capacitors)))
    
    # 1. Response Efficiency with threshold lines
    ax1 = fig.add_subplot(gs[0, :])
    for i, cap_id in enumerate(capacitors):
        cap_data = df[df['capacitor_id'] == cap_id]
        ax1.plot(cap_data['cycle'], cap_data['response_efficiency'], 
                label=cap_id, color=colors[i], alpha=0.7, linewidth=1.5)
    
    # Add threshold lines
    ax1.axhline(y=50, color='orange', linestyle='--', linewidth=2, label='Threshold: 50% (Normal/Degrading)')
    ax1.axhline(y=10, color='red', linestyle='--', linewidth=2, label='Threshold: 10% (Degrading/Severe)')
    ax1.axhline(y=1, color='darkred', linestyle='--', linewidth=2, label='Threshold: 1% (Severe/Critical)')
    
    ax1.set_xlabel('Cycle', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Response Efficiency', fontsize=12, fontweight='bold')
    ax1.set_title('Response Efficiency Over Time with Degradation Thresholds', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)
    
    # 2. Efficiency Degradation Rate
    ax2 = fig.add_subplot(gs[1, 0])
    for i, cap_id in enumerate(capacitors):
        cap_data = df[df['capacitor_id'] == cap_id]
        cap_data_valid = cap_data[cap_data['efficiency_degradation_rate'].notna()]
        ax2.plot(cap_data_valid['cycle'], cap_data_valid['efficiency_degradation_rate'], 
                label=cap_id, color=colors[i], alpha=0.7, linewidth=1.5)
    
    ax2.axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=0.9, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.set_xlabel('Cycle', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Efficiency Degradation Rate', fontsize=11, fontweight='bold')
    ax2.set_title('Efficiency Degradation Rate', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # 3. Waveform Correlation
    ax3 = fig.add_subplot(gs[1, 1])
    for i, cap_id in enumerate(capacitors):
        cap_data = df[df['capacitor_id'] == cap_id]
        ax3.plot(cap_data['cycle'], cap_data['waveform_correlation'], 
                label=cap_id, color=colors[i], alpha=0.7, linewidth=1.5)
    
    ax3.axhline(y=0.95, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.axhline(y=0.99, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax3.set_xlabel('Cycle', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Waveform Correlation', fontsize=11, fontweight='bold')
    ax3.set_title('Waveform Correlation (Simplification)', fontsize=12, fontweight='bold')
    ax3.legend(loc='lower right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # 4. VO Variability
    ax4 = fig.add_subplot(gs[1, 2])
    for i, cap_id in enumerate(capacitors):
        cap_data = df[df['capacitor_id'] == cap_id]
        ax4.plot(cap_data['cycle'], cap_data['vo_variability'], 
                label=cap_id, color=colors[i], alpha=0.7, linewidth=1.5)
    
    ax4.set_xlabel('Cycle', fontsize=11, fontweight='bold')
    ax4.set_ylabel('VO Variability', fontsize=11, fontweight='bold')
    ax4.set_title('VO Variability Over Time', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # 5. Distribution by Stage - Response Efficiency
    ax5 = fig.add_subplot(gs[2, 0])
    stage_order = ['Normal', 'Degrading', 'Severe', 'Critical']
    stage_colors = {'Normal': 'green', 'Degrading': 'orange', 'Severe': 'red', 'Critical': 'darkred'}
    
    for stage in stage_order:
        stage_data = df[df['degradation_stage'] == stage]
        if len(stage_data) > 0:
            ax5.hist(stage_data['response_efficiency'], bins=30, alpha=0.6, 
                    label=f"{stage} (n={len(stage_data)})", color=stage_colors[stage])
    
    ax5.set_xlabel('Response Efficiency', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax5.set_title('Response Efficiency Distribution by Stage', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper right', fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.set_xlim(left=0)
    
    # 6. Box plot - Response Efficiency by Stage
    ax6 = fig.add_subplot(gs[2, 1])
    stage_data_list = [df[df['degradation_stage'] == stage]['response_efficiency'].values 
                       for stage in stage_order if len(df[df['degradation_stage'] == stage]) > 0]
    stage_labels = [stage for stage in stage_order if len(df[df['degradation_stage'] == stage]) > 0]
    
    bp = ax6.boxplot(stage_data_list, labels=stage_labels, patch_artist=True)
    for patch, stage in zip(bp['boxes'], stage_labels):
        patch.set_facecolor(stage_colors[stage])
        patch.set_alpha(0.6)
    
    ax6.set_ylabel('Response Efficiency', fontsize=11, fontweight='bold')
    ax6.set_title('Response Efficiency by Degradation Stage', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_yscale('log')
    
    # 7. Cycle distribution by stage
    ax7 = fig.add_subplot(gs[2, 2])
    for stage in stage_order:
        stage_data = df[df['degradation_stage'] == stage]
        if len(stage_data) > 0:
            ax7.hist(stage_data['cycle'], bins=20, alpha=0.6, 
                    label=f"{stage} (n={len(stage_data)})", color=stage_colors[stage])
    
    ax7.set_xlabel('Cycle', fontsize=11, fontweight='bold')
    ax7.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax7.set_title('Cycle Distribution by Degradation Stage', fontsize=12, fontweight='bold')
    ax7.legend(loc='upper right', fontsize=9)
    ax7.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Degradation Pattern Analysis with Threshold Identification', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    output_path = OUTPUT_DIR / "degradation_patterns_detailed.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()


def create_threshold_analysis_report(df, threshold_results):
    """Create markdown report with threshold analysis."""
    print("\n" + "="*80)
    print("CREATING THRESHOLD ANALYSIS REPORT")
    print("="*80)
    
    report_path = OUTPUT_DIR / "degradation_stages_definition.md"
    
    with open(report_path, 'w') as f:
        f.write("# 劣化ステージ定義と閾値分析レポート\n\n")
        f.write("**作成日**: 2026-01-17  \n")
        f.write("**タスク**: Phase 1 Task 1.4 - 劣化パターンの詳細可視化と閾値探索  \n\n")
        f.write("---\n\n")
        
        f.write("## 📋 概要\n\n")
        f.write("応答性特徴量の時系列分析に基づき、故障兆候を示す閾値候補を特定し、\n")
        f.write("劣化ステージ（Normal, Degrading, Severe, Critical）を定義しました。\n\n")
        f.write("---\n\n")
        
        f.write("## 🎯 劣化ステージの定義\n\n")
        f.write("Response Efficiencyを基準として、4つの劣化ステージを定義:\n\n")
        f.write("| ステージ | Response Efficiency | 説明 |\n")
        f.write("|---------|---------------------|------|\n")
        f.write("| **Normal** | > 50% | 正常動作範囲。VL-VO応答性が良好 |\n")
        f.write("| **Degrading** | 10% - 50% | 劣化進行中。応答性が低下し始める |\n")
        f.write("| **Severe** | 1% - 10% | 深刻な劣化。応答性が大幅に低下 |\n")
        f.write("| **Critical** | < 1% | 臨界状態。ほぼ応答なし |\n\n")
        
        # Sample distribution
        stage_counts = df['degradation_stage'].value_counts()
        f.write("### サンプル分布\n\n")
        f.write("```\n")
        for stage in ['Normal', 'Degrading', 'Severe', 'Critical']:
            if stage in stage_counts:
                count = stage_counts[stage]
                pct = count / len(df) * 100
                f.write(f"{stage:12s}: {count:4d} samples ({pct:5.1f}%)\n")
        f.write("```\n\n")
        
        f.write("---\n\n")
        f.write("## 📊 閾値候補の分析\n\n")
        
        # Response Efficiency thresholds
        f.write("### 1. Response Efficiency 閾値\n\n")
        f.write("| 閾値 | 条件 | 該当サンプル数 | 割合 | 平均検出サイクル |\n")
        f.write("|------|------|---------------|------|----------------|\n")
        
        eff_results = threshold_results[threshold_results['feature'] == 'response_efficiency']
        for _, row in eff_results.iterrows():
            avg_cycle = f"{row['avg_first_cycle']:.1f}" if pd.notna(row['avg_first_cycle']) else "N/A"
            f.write(f"| {row['threshold']}% | {row['condition']} | {row['failing_samples']} | "
                   f"{row['failing_pct']:.1f}% | Cycle {avg_cycle} |\n")
        
        f.write("\n**推奨閾値**: 50% (Normal/Degrading境界), 10% (Degrading/Severe境界), 1% (Severe/Critical境界)\n\n")
        
        # Efficiency Degradation Rate thresholds
        f.write("### 2. Efficiency Degradation Rate 閾値\n\n")
        f.write("| 閾値 | 条件 | 該当サンプル数 | 割合 | 平均検出サイクル |\n")
        f.write("|------|------|---------------|------|----------------|\n")
        
        deg_results = threshold_results[threshold_results['feature'] == 'efficiency_degradation_rate']
        for _, row in deg_results.iterrows():
            avg_cycle = f"{row['avg_first_cycle']:.1f}" if pd.notna(row['avg_first_cycle']) else "N/A"
            f.write(f"| {row['threshold']} | {row['condition']} | {row['failing_samples']} | "
                   f"{row['failing_pct']:.1f}% | Cycle {avg_cycle} |\n")
        
        f.write("\n**推奨閾値**: 0.5 (50%劣化), 0.9 (90%劣化)\n\n")
        
        # Waveform Correlation thresholds
        f.write("### 3. Waveform Correlation 閾値\n\n")
        f.write("| 閾値 | 条件 | 該当サンプル数 | 割合 | 平均検出サイクル |\n")
        f.write("|------|------|---------------|------|----------------|\n")
        
        corr_results = threshold_results[threshold_results['feature'] == 'waveform_correlation']
        for _, row in corr_results.iterrows():
            avg_cycle = f"{row['avg_first_cycle']:.1f}" if pd.notna(row['avg_first_cycle']) else "N/A"
            f.write(f"| {row['threshold']} | {row['condition']} | {row['failing_samples']} | "
                   f"{row['failing_pct']:.1f}% | Cycle {avg_cycle} |\n")
        
        f.write("\n**推奨閾値**: 0.95 (波形単純化開始), 0.99 (深刻な単純化)\n\n")
        
        f.write("---\n\n")
        f.write("## 📈 ステージ別特徴量範囲\n\n")
        
        key_features = [
            'response_efficiency',
            'voltage_ratio',
            'waveform_correlation',
            'efficiency_degradation_rate'
        ]
        
        for stage in ['Normal', 'Degrading', 'Severe', 'Critical']:
            stage_data = df[df['degradation_stage'] == stage]
            if len(stage_data) == 0:
                continue
            
            f.write(f"### {stage} ステージ\n\n")
            f.write(f"**サンプル数**: {len(stage_data)}\n\n")
            f.write("| 特徴量 | 平均値 | 標準偏差 | 最小値 | 最大値 |\n")
            f.write("|--------|--------|----------|--------|--------|\n")
            
            for feat in key_features:
                if feat == 'efficiency_degradation_rate':
                    valid_data = stage_data[stage_data[feat].notna()]
                    if len(valid_data) == 0:
                        continue
                    mean_val = valid_data[feat].mean()
                    std_val = valid_data[feat].std()
                    min_val = valid_data[feat].min()
                    max_val = valid_data[feat].max()
                else:
                    mean_val = stage_data[feat].mean()
                    std_val = stage_data[feat].std()
                    min_val = stage_data[feat].min()
                    max_val = stage_data[feat].max()
                
                f.write(f"| {feat} | {mean_val:.2f} | {std_val:.2f} | {min_val:.2f} | {max_val:.2f} |\n")
            
            f.write("\n")
        
        f.write("---\n\n")
        f.write("## 🎯 故障兆候検出の推奨アプローチ\n\n")
        f.write("### 早期警告（Early Warning）\n")
        f.write("- **Response Efficiency < 50%**: 劣化開始の兆候\n")
        f.write("- **Efficiency Degradation Rate > 0.5**: 初期効率から50%以上低下\n")
        f.write("- **Waveform Correlation > 0.95**: 波形単純化の開始\n\n")
        
        f.write("### 深刻な劣化（Severe Degradation）\n")
        f.write("- **Response Efficiency < 10%**: 深刻な応答性低下\n")
        f.write("- **Efficiency Degradation Rate > 0.9**: 初期効率から90%以上低下\n")
        f.write("- **Waveform Correlation > 0.99**: 深刻な波形単純化\n\n")
        
        f.write("### 臨界状態（Critical State）\n")
        f.write("- **Response Efficiency < 1%**: ほぼ応答なし\n")
        f.write("- **Efficiency Degradation Rate > 0.98**: 初期効率から98%以上低下\n")
        f.write("- **Waveform Correlation > 0.995**: 完全な波形単純化\n\n")
        
        f.write("---\n\n")
        f.write("## 📁 生成ファイル\n\n")
        f.write("1. `degradation_patterns_detailed.png` - 劣化パターンの詳細可視化\n")
        f.write("2. `degradation_stages_definition.md` - 本レポート\n\n")
        
        f.write("---\n\n")
        f.write("**報告者**: Kiro AI Agent  \n")
        f.write("**完了日**: 2026-01-17  \n")
        f.write("**次のステップ**: Phase 2 - 異常検知モデル構築\n")
    
    print(f"✓ Saved: {report_path}")


def main():
    """Main execution."""
    print("="*80)
    print("DEGRADATION THRESHOLD ANALYSIS")
    print("="*80)
    
    # Load features
    print("\nLoading features...")
    df = pd.read_csv(FEATURES_PATH)
    print(f"  ✓ Loaded {len(df)} samples")
    
    # Analyze threshold candidates
    threshold_results = analyze_threshold_candidates(df)
    
    # Define degradation stages
    df = define_degradation_stages(df)
    
    # Create visualizations
    visualize_degradation_patterns(df)
    
    # Create report
    create_threshold_analysis_report(df, threshold_results)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  1. degradation_patterns_detailed.png")
    print("  2. degradation_stages_definition.md")
    print("\n✅ Degradation threshold analysis complete!")


if __name__ == "__main__":
    main()
