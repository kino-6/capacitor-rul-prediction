#!/usr/bin/env python3
"""
Visualize Best Nearly-Identical VL Pairs

Based on actual data analysis, visualize the best pairs found
at different time gaps.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
import warnings
from scipy.stats import pearsonr

# Suppress warnings
warnings.filterwarnings('ignore')

def main():
    print("🎨 Visualizing Best Nearly-Identical VL Pairs")
    print("=" * 70)
    
    data_path = Path("data/raw/ES12.mat")
    output_dir = Path("output/best_identical_vl")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plt.rcParams['font.family'] = ['DejaVu Sans']
    
    # Best pairs found from analysis
    best_pairs = [
        (88, 98, 10),   # 10 cycle gap, corr=0.9939
        (80, 100, 20),  # 20 cycle gap, corr=0.9802
        (70, 100, 30),  # 30 cycle gap, corr=0.9539
        (46, 96, 50),   # 50 cycle gap, corr=0.8941
    ]
    
    with h5py.File(data_path, 'r') as f:
        cap_group = f['ES12']['Transient_Data']['ES12C4']
        vl_data = cap_group['VL'][:]
        vo_data = cap_group['VO'][:]
        
        for cycle1, cycle2, gap in best_pairs:
            print(f"\n📊 Processing Cycles {cycle1} vs {cycle2} (gap: {gap})")
            
            # Extract data
            vl1 = vl_data[:, cycle1-1]
            vo1 = vo_data[:, cycle1-1]
            vl2 = vl_data[:, cycle2-1]
            vo2 = vo_data[:, cycle2-1]
            
            # Remove NaN
            valid1 = ~np.isnan(vl1) & ~np.isnan(vo1)
            valid2 = ~np.isnan(vl2) & ~np.isnan(vo2)
            
            vl1 = vl1[valid1][:3000]
            vo1 = vo1[valid1][:3000]
            vl2 = vl2[valid2][:3000]
            vo2 = vo2[valid2][:3000]
            
            # Calculate metrics
            corr, _ = pearsonr(vl1, vl2)
            mean_diff = abs(np.mean(vl1) - np.mean(vl2))
            std_diff = abs(np.std(vl1) - np.std(vl2))
            
            ratio1 = np.mean(vo1) / np.mean(vl1) if np.mean(vl1) != 0 else np.nan
            ratio2 = np.mean(vo2) / np.mean(vl2) if np.mean(vl2) != 0 else np.nan
            degradation = abs((ratio2 - ratio1) / ratio1) * 100 if ratio1 != 0 else 0
            
            print(f"   Correlation: {corr:.4f}")
            print(f"   Mean diff: {mean_diff:.4f}V")
            print(f"   Std diff: {std_diff:.4f}V")
            print(f"   Degradation: {degradation:.1f}%")
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(16, 10))
            fig.suptitle(f'ES12C4: Cycle {cycle1} vs Cycle {cycle2} - Nearly Identical VL\n'
                        f'Time Gap: {gap} cycles, Correlation: {corr:.4f}, Degradation: {degradation:.1f}%',
                        fontsize=14, fontweight='bold')
            
            time = np.arange(len(vl1))
            
            # VL Full waveform
            axes[0, 0].plot(time, vl1, 'b-', label=f'Cycle {cycle1}', alpha=0.7, linewidth=0.5)
            axes[0, 0].plot(time, vl2, 'r-', label=f'Cycle {cycle2}', alpha=0.7, linewidth=0.5)
            axes[0, 0].set_title('VL Input - Full Waveform (3000 points)', fontsize=12, fontweight='bold')
            axes[0, 0].set_xlabel('Time Points')
            axes[0, 0].set_ylabel('VL Voltage (V)')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            axes[0, 0].text(0.02, 0.98,
                           f'Correlation: {corr:.4f}\n'
                           f'Mean Diff: {mean_diff:.4f}V\n'
                           f'Std Diff: {std_diff:.4f}V\n'
                           f'VL{cycle1}: {np.mean(vl1):.3f}±{np.std(vl1):.3f}V\n'
                           f'VL{cycle2}: {np.mean(vl2):.3f}±{np.std(vl2):.3f}V',
                           transform=axes[0, 0].transAxes,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                           fontsize=9)
            
            # VL Zoomed
            axes[0, 1].plot(time[:500], vl1[:500], 'b-', label=f'Cycle {cycle1}', alpha=0.8, linewidth=1)
            axes[0, 1].plot(time[:500], vl2[:500], 'r-', label=f'Cycle {cycle2}', alpha=0.8, linewidth=1)
            axes[0, 1].set_title('VL Input - Zoomed (First 500 points)', fontsize=12, fontweight='bold')
            axes[0, 1].set_xlabel('Time Points')
            axes[0, 1].set_ylabel('VL Voltage (V)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # VO Full waveform
            axes[1, 0].plot(time, vo1, 'b-', label=f'Cycle {cycle1}', alpha=0.7, linewidth=0.5)
            axes[1, 0].plot(time, vo2, 'r-', label=f'Cycle {cycle2}', alpha=0.7, linewidth=0.5)
            axes[1, 0].set_title('VO Output - Full Waveform (3000 points)', fontsize=12, fontweight='bold')
            axes[1, 0].set_xlabel('Time Points')
            axes[1, 0].set_ylabel('VO Voltage (V)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            axes[1, 0].text(0.02, 0.98,
                           f'Degradation: {degradation:.1f}%\n'
                           f'Ratio {cycle1}: {ratio1:.2f}\n'
                           f'Ratio {cycle2}: {ratio2:.2f}\n'
                           f'Time Gap: {gap} cycles',
                           transform=axes[1, 0].transAxes,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
                           fontsize=9)
            
            # VO Zoomed
            axes[1, 1].plot(time[:500], vo1[:500], 'b-', label=f'Cycle {cycle1}', alpha=0.8, linewidth=1)
            axes[1, 1].plot(time[:500], vo2[:500], 'r-', label=f'Cycle {cycle2}', alpha=0.8, linewidth=1)
            axes[1, 1].set_title('VO Output - Zoomed (First 500 points)', fontsize=12, fontweight='bold')
            axes[1, 1].set_xlabel('Time Points')
            axes[1, 1].set_ylabel('VO Voltage (V)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plot_path = output_dir / f'ES12C4_cycles_{cycle1}_{cycle2}_gap{gap}.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"   ✅ Saved: {plot_path.name}")
    
    # Generate report
    report_path = output_dir / 'ES12C4_best_identical_vl_report.md'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# ES12C4 ほぼ同一VL入力サイクル - 正直な分析レポート\n\n")
        
        f.write("## 📊 ES12データの現実\n\n")
        f.write("### ❌ 存在しないもの\n")
        f.write("- **Sin波のような周期的波形**: FFT分析で周期性比率0.003-0.004（ほぼゼロ）\n")
        f.write("- **制御された動的入力**: 実運用環境の不規則な変動データ\n")
        f.write("- **理想的な実験条件**: ランダムノイズを含む実データ\n\n")
        
        f.write("### ✅ 実際に存在するもの\n")
        f.write("- **ほぼ一定値 ± ノイズ**: 大部分のサイクルがこのパターン\n")
        f.write("- **高い類似性のペア**: 短い時間差（10-30サイクル）で見つかる\n")
        f.write("- **劣化の観測**: 同一入力に対する出力応答の変化\n\n")
        
        f.write("## 🔍 発見されたほぼ同一VLペア\n\n")
        
        f.write("### ペア1: サイクル88 vs 98（時間差10サイクル）\n\n")
        f.write("![Cycles 88-98](ES12C4_cycles_88_98_gap10.png)\n\n")
        f.write("- **相関係数**: 0.9939（ほぼ完璧）\n")
        f.write("- **平均値差**: 0.0050V（非常に小さい）\n")
        f.write("- **標準偏差差**: 0.0007V（非常に小さい）\n")
        f.write("- **時間差**: 10サイクル\n")
        f.write("- **劣化**: 観測可能\n\n")
        f.write("**評価**: VL入力が視覚的にほぼ同一。10サイクルの時間差で劣化を観測。\n\n")
        f.write("---\n\n")
        
        f.write("### ペア2: サイクル80 vs 100（時間差20サイクル）\n\n")
        f.write("![Cycles 80-100](ES12C4_cycles_80_100_gap20.png)\n\n")
        f.write("- **相関係数**: 0.9802（非常に高い）\n")
        f.write("- **平均値差**: 0.0127V（小さい）\n")
        f.write("- **標準偏差差**: 0.0012V（非常に小さい）\n")
        f.write("- **時間差**: 20サイクル\n")
        f.write("- **劣化**: より明確に観測可能\n\n")
        f.write("**評価**: VL入力が高い類似性。20サイクルの時間差でより明確な劣化。\n\n")
        f.write("---\n\n")
        
        f.write("### ペア3: サイクル70 vs 100（時間差30サイクル）\n\n")
        f.write("![Cycles 70-100](ES12C4_cycles_70_100_gap30.png)\n\n")
        f.write("- **相関係数**: 0.9539（高い）\n")
        f.write("- **平均値差**: 0.0341V（やや大きい）\n")
        f.write("- **標準偏差差**: 0.0037V（小さい）\n")
        f.write("- **時間差**: 30サイクル\n")
        f.write("- **劣化**: 明確に観測可能\n\n")
        f.write("**評価**: VL入力が良好な類似性。30サイクルの時間差で劣化が明確。\n\n")
        f.write("---\n\n")
        
        f.write("### ペア4: サイクル46 vs 96（時間差50サイクル）\n\n")
        f.write("![Cycles 46-96](ES12C4_cycles_46_96_gap50.png)\n\n")
        f.write("- **相関係数**: 0.8941（良好）\n")
        f.write("- **平均値差**: 0.6974V（大きい）\n")
        f.write("- **標準偏差差**: 0.0027V（小さい）\n")
        f.write("- **時間差**: 50サイクル\n")
        f.write("- **劣化**: 非常に明確\n\n")
        f.write("**評価**: 波形形状は類似だが、オフセットが大きく異なる。これは以前ユーザーが指摘した問題。\n\n")
        f.write("---\n\n")
        
        f.write("## 💡 重要な洞察\n\n")
        
        f.write("### 時間差と類似性のトレードオフ\n\n")
        f.write("| 時間差 | 最高相関 | 平均値差 | 評価 |\n")
        f.write("|--------|----------|----------|------|\n")
        f.write("| 10サイクル | 0.9939 | 0.0050V | ほぼ完璧な類似性 |\n")
        f.write("| 20サイクル | 0.9802 | 0.0127V | 非常に高い類似性 |\n")
        f.write("| 30サイクル | 0.9539 | 0.0341V | 高い類似性 |\n")
        f.write("| 50サイクル | 0.8941 | 0.6974V | 形状類似、オフセット大 |\n\n")
        
        f.write("### ユーザー要求への回答\n\n")
        f.write("1. **「VLがほぼ同じCycleをリストアップ」**: ✅ 完了\n")
        f.write("   - サイクル88-98: 相関0.9939（ほぼ完璧）\n")
        f.write("   - サイクル80-100: 相関0.9802（非常に高い）\n\n")
        
        f.write("2. **「Sin波のような波形」**: ❌ 存在しない\n")
        f.write("   - ES12データには周期的Sin波パターンなし\n")
        f.write("   - 実運用データの制約\n\n")
        
        f.write("3. **「時間差が小さい問題」**: ⚠️ トレードオフ\n")
        f.write("   - 高い類似性 → 短い時間差（10-30サイクル）\n")
        f.write("   - 長い時間差（50サイクル以上）→ 類似性低下\n\n")
        
        f.write("## 📝 結論\n\n")
        f.write("### 正直な評価\n\n")
        f.write("ES12データセットは：\n")
        f.write("- 実運用環境の実データ\n")
        f.write("- 制御された実験データではない\n")
        f.write("- Sin波のような理想的入力は含まれない\n")
        f.write("- ほぼ一定値±ノイズのパターンが主\n\n")
        
        f.write("### 実用的な推奨\n\n")
        f.write("**短期劣化分析**（10-30サイクル）:\n")
        f.write("- サイクル88-98: 最高の類似性\n")
        f.write("- サイクル80-100: 良好な類似性と時間差のバランス\n\n")
        
        f.write("**長期劣化分析**（50サイクル以上）:\n")
        f.write("- 類似性は低下するが、劣化は明確\n")
        f.write("- オフセット差を考慮した分析が必要\n\n")
        
        f.write("---\n")
        f.write(f"**レポート生成**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\n✅ Report generated: {report_path.name}")
    print(f"📍 Output Directory: {output_dir}")
    print("\n" + "=" * 70)
    print("✅ Analysis Complete!")

if __name__ == "__main__":
    main()
