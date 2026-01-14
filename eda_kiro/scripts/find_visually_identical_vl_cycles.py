#!/usr/bin/env python3
"""
Find Visually Identical VL Input Cycles

Simple, honest approach: Find cycles where VL waveforms look nearly identical
when plotted, with large time gaps for degradation observation.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from typing import Dict, List
import warnings
from scipy.stats import pearsonr

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

def main():
    """Main execution function"""
    print("🔍 Finding Visually Identical VL Input Cycles")
    print("=" * 70)
    print("Goal: Find cycles where VL waveforms look nearly identical")
    print("=" * 70)
    
    # Data path configuration
    data_path = Path("data/raw/ES12.mat")
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    # Output directory
    output_dir = Path("output/identical_vl_cycles")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set font to avoid rendering issues
    plt.rcParams['font.family'] = ['DejaVu Sans']
    
    try:
        target_capacitor = "ES12C4"
        
        print(f"🎯 Analysis Target: {target_capacitor}")
        
        # Load data
        with h5py.File(data_path, 'r') as f:
            cap_group = f['ES12']['Transient_Data'][target_capacitor]
            vl_data = cap_group['VL'][:]
            vo_data = cap_group['VO'][:]
            
            print(f"✅ Raw data loaded: VL {vl_data.shape}, VO {vo_data.shape}")
            
            # Process all cycles
            target_length = 3000
            cycles_data = {}
            valid_cycles = []
            vl_matrix = []
            
            print(f"📊 Processing cycles...")
            
            for cycle_idx in range(min(400, vl_data.shape[1])):
                cycle_num = cycle_idx + 1
                
                vl_cycle = vl_data[:, cycle_idx]
                vo_cycle = vo_data[:, cycle_idx]
                
                # Remove NaN
                valid_mask = ~np.isnan(vl_cycle) & ~np.isnan(vo_cycle)
                
                if np.sum(valid_mask) < target_length:
                    continue
                
                vl = vl_cycle[valid_mask][:target_length]
                vo = vo_cycle[valid_mask][:target_length]
                
                cycles_data[cycle_num] = {
                    'vl': vl,
                    'vo': vo,
                    'vl_mean': np.mean(vl),
                    'vl_std': np.std(vl),
                    'vl_range': np.max(vl) - np.min(vl),
                    'vo_mean': np.mean(vo),
                    'vo_std': np.std(vo),
                    'voltage_ratio': np.mean(vo) / np.mean(vl) if np.mean(vl) != 0 else np.nan
                }
                
                vl_matrix.append(vl)
                valid_cycles.append(cycle_num)
            
            vl_matrix = np.array(vl_matrix)
            print(f"✅ Processed {len(valid_cycles)} valid cycles")
        
        # Find pairs with nearly identical VL
        print(f"\n🔍 Finding pairs with nearly identical VL waveforms...")
        print(f"   Criteria:")
        print(f"   - Correlation ≥ 0.95 (very high shape similarity)")
        print(f"   - Mean difference < 0.1V (similar offset)")
        print(f"   - Std difference < 0.02V (similar amplitude)")
        print(f"   - Time gap ≥ 50 cycles (observe significant degradation)")
        
        identical_pairs = []
        
        for i, cycle1 in enumerate(valid_cycles):
            for j, cycle2 in enumerate(valid_cycles):
                if i >= j:
                    continue
                
                time_gap = cycle2 - cycle1
                
                # Require large time gap
                if time_gap < 50:
                    continue
                
                # Get data
                data1 = cycles_data[cycle1]
                data2 = cycles_data[cycle2]
                
                # Calculate correlation
                try:
                    corr, _ = pearsonr(vl_matrix[i], vl_matrix[j])
                    if np.isnan(corr):
                        corr = 0.0
                except:
                    corr = 0.0
                
                # Check strict similarity criteria
                mean_diff = abs(data1['vl_mean'] - data2['vl_mean'])
                std_diff = abs(data1['vl_std'] - data2['vl_std'])
                
                if (corr >= 0.95 and 
                    mean_diff < 0.1 and 
                    std_diff < 0.02):
                    
                    # Calculate degradation
                    ratio1 = data1['voltage_ratio']
                    ratio2 = data2['voltage_ratio']
                    
                    if not np.isnan(ratio1) and not np.isnan(ratio2) and ratio1 != 0:
                        degradation_pct = abs((ratio2 - ratio1) / ratio1) * 100
                    else:
                        degradation_pct = 0
                    
                    identical_pairs.append({
                        'cycle1': cycle1,
                        'cycle2': cycle2,
                        'time_gap': time_gap,
                        'correlation': corr,
                        'mean_diff': mean_diff,
                        'std_diff': std_diff,
                        'degradation_pct': degradation_pct,
                        'ratio1': ratio1,
                        'ratio2': ratio2,
                        'vl1_mean': data1['vl_mean'],
                        'vl2_mean': data2['vl_mean'],
                        'vl1_std': data1['vl_std'],
                        'vl2_std': data2['vl_std']
                    })
        
        # Sort by correlation (highest first)
        identical_pairs.sort(key=lambda x: x['correlation'], reverse=True)
        
        print(f"\n✅ Found {len(identical_pairs)} pairs with nearly identical VL")
        
        if identical_pairs:
            print(f"\n📊 Top 10 Pairs with Most Identical VL:")
            for i, pair in enumerate(identical_pairs[:10], 1):
                print(f"   #{i}: Cycles {pair['cycle1']}-{pair['cycle2']} "
                      f"(corr:{pair['correlation']:.4f}, "
                      f"mean_diff:{pair['mean_diff']:.4f}V, "
                      f"std_diff:{pair['std_diff']:.4f}V, "
                      f"gap:{pair['time_gap']}, "
                      f"deg:{pair['degradation_pct']:.1f}%)")
            
            # Visualize top 5 pairs
            print(f"\n📊 Creating visualizations for top 5 pairs...")
            
            for i, pair in enumerate(identical_pairs[:5], 1):
                cycle1, cycle2 = pair['cycle1'], pair['cycle2']
                
                data1 = cycles_data[cycle1]
                data2 = cycles_data[cycle2]
                
                fig, axes = plt.subplots(2, 2, figsize=(16, 10))
                fig.suptitle(f'ES12C4: Cycle {cycle1} vs Cycle {cycle2} - Nearly Identical VL Input\n'
                            f'Time Gap: {pair["time_gap"]} cycles, '
                            f'Correlation: {pair["correlation"]:.4f}, '
                            f'Degradation: {pair["degradation_pct"]:.1f}%',
                            fontsize=14, fontweight='bold')
                
                time = np.arange(len(data1['vl']))
                
                # VL comparison - FULL WAVEFORM
                axes[0, 0].plot(time, data1['vl'], 'b-', label=f'Cycle {cycle1}', alpha=0.7, linewidth=0.5)
                axes[0, 0].plot(time, data2['vl'], 'r-', label=f'Cycle {cycle2}', alpha=0.7, linewidth=0.5)
                axes[0, 0].set_title('VL Input Comparison (Full Waveform)', fontsize=12, fontweight='bold')
                axes[0, 0].set_xlabel('Time Points')
                axes[0, 0].set_ylabel('VL Voltage (V)')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                # Add similarity metrics
                axes[0, 0].text(0.02, 0.98,
                               f'Correlation: {pair["correlation"]:.4f}\n'
                               f'Mean Diff: {pair["mean_diff"]:.4f}V\n'
                               f'Std Diff: {pair["std_diff"]:.4f}V\n'
                               f'VL1: {pair["vl1_mean"]:.3f}±{pair["vl1_std"]:.3f}V\n'
                               f'VL2: {pair["vl2_mean"]:.3f}±{pair["vl2_std"]:.3f}V',
                               transform=axes[0, 0].transAxes,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                               fontsize=9)
                
                # VL comparison - ZOOMED (first 500 points)
                axes[0, 1].plot(time[:500], data1['vl'][:500], 'b-', label=f'Cycle {cycle1}', alpha=0.8, linewidth=1)
                axes[0, 1].plot(time[:500], data2['vl'][:500], 'r-', label=f'Cycle {cycle2}', alpha=0.8, linewidth=1)
                axes[0, 1].set_title('VL Input Comparison (Zoomed: First 500 Points)', fontsize=12, fontweight='bold')
                axes[0, 1].set_xlabel('Time Points')
                axes[0, 1].set_ylabel('VL Voltage (V)')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
                
                # VO comparison - FULL WAVEFORM
                axes[1, 0].plot(time, data1['vo'], 'b-', label=f'Cycle {cycle1}', alpha=0.7, linewidth=0.5)
                axes[1, 0].plot(time, data2['vo'], 'r-', label=f'Cycle {cycle2}', alpha=0.7, linewidth=0.5)
                axes[1, 0].set_title('VO Output Comparison (Full Waveform)', fontsize=12, fontweight='bold')
                axes[1, 0].set_xlabel('Time Points')
                axes[1, 0].set_ylabel('VO Voltage (V)')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
                
                # Add degradation info
                axes[1, 0].text(0.02, 0.98,
                               f'Degradation: {pair["degradation_pct"]:.1f}%\n'
                               f'Ratio 1: {pair["ratio1"]:.2f}\n'
                               f'Ratio 2: {pair["ratio2"]:.2f}\n'
                               f'Time Gap: {pair["time_gap"]} cycles',
                               transform=axes[1, 0].transAxes,
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8),
                               fontsize=9)
                
                # VO comparison - ZOOMED (first 500 points)
                axes[1, 1].plot(time[:500], data1['vo'][:500], 'b-', label=f'Cycle {cycle1}', alpha=0.8, linewidth=1)
                axes[1, 1].plot(time[:500], data2['vo'][:500], 'r-', label=f'Cycle {cycle2}', alpha=0.8, linewidth=1)
                axes[1, 1].set_title('VO Output Comparison (Zoomed: First 500 Points)', fontsize=12, fontweight='bold')
                axes[1, 1].set_xlabel('Time Points')
                axes[1, 1].set_ylabel('VO Voltage (V)')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                plot_path = output_dir / f'ES12C4_identical_vl_cycles_{cycle1}_{cycle2}.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"   ✅ Saved: {plot_path.name}")
            
            # Generate report
            report_path = output_dir / 'ES12C4_identical_vl_cycles_report.md'
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("# ES12C4 Nearly Identical VL Input Cycles Report\n\n")
                
                f.write("## Honest Assessment\n\n")
                f.write("**ES12データの現実**:\n")
                f.write("- Sin波のような周期的波形は存在しません\n")
                f.write("- ほぼ一定値 ± ランダムノイズのパターンです\n")
                f.write("- 実運用環境の不規則な変動データです\n\n")
                
                f.write("## Analysis Approach\n\n")
                f.write("VL波形が視覚的にほぼ同じサイクルペアを探しました：\n\n")
                f.write("### Strict Similarity Criteria\n")
                f.write("- **Correlation ≥ 0.95**: Very high shape similarity\n")
                f.write("- **Mean difference < 0.1V**: Similar DC offset\n")
                f.write("- **Std difference < 0.02V**: Similar amplitude variation\n")
                f.write("- **Time gap ≥ 50 cycles**: Observe significant degradation\n\n")
                
                f.write("## Results\n\n")
                f.write(f"**Total pairs found**: {len(identical_pairs)}\n\n")
                
                if identical_pairs:
                    f.write("### Top 10 Pairs with Most Identical VL\n\n")
                    f.write("| Rank | Cycle Pair | Correlation | Mean Diff (V) | Std Diff (V) | Time Gap | Degradation |\n")
                    f.write("|------|------------|-------------|---------------|--------------|----------|-------------|\n")
                    
                    for i, pair in enumerate(identical_pairs[:10], 1):
                        f.write(f"| {i} | {pair['cycle1']}-{pair['cycle2']} | "
                               f"{pair['correlation']:.4f} | {pair['mean_diff']:.4f} | "
                               f"{pair['std_diff']:.4f} | {pair['time_gap']} | "
                               f"{pair['degradation_pct']:.1f}% |\n")
                    
                    f.write("\n### Detailed Analysis: Top 5 Pairs\n\n")
                    
                    for i, pair in enumerate(identical_pairs[:5], 1):
                        f.write(f"#### Pair {i}: Cycle {pair['cycle1']} vs Cycle {pair['cycle2']}\n\n")
                        f.write(f"![Comparison](ES12C4_identical_vl_cycles_{pair['cycle1']}_{pair['cycle2']}.png)\n\n")
                        f.write(f"**VL Input Similarity**:\n")
                        f.write(f"- Correlation: {pair['correlation']:.4f} (nearly perfect)\n")
                        f.write(f"- Mean difference: {pair['mean_diff']:.4f}V (very small)\n")
                        f.write(f"- Std difference: {pair['std_diff']:.4f}V (very small)\n")
                        f.write(f"- Cycle {pair['cycle1']} VL: {pair['vl1_mean']:.3f}±{pair['vl1_std']:.3f}V\n")
                        f.write(f"- Cycle {pair['cycle2']} VL: {pair['vl2_mean']:.3f}±{pair['vl2_std']:.3f}V\n\n")
                        f.write(f"**VO Output Degradation**:\n")
                        f.write(f"- Time gap: {pair['time_gap']} cycles\n")
                        f.write(f"- Degradation: {pair['degradation_pct']:.1f}%\n")
                        f.write(f"- Early ratio: {pair['ratio1']:.2f}\n")
                        f.write(f"- Late ratio: {pair['ratio2']:.2f}\n\n")
                        f.write("---\n\n")
                else:
                    f.write("No pairs found meeting the strict criteria.\n\n")
                
                f.write("## Conclusion\n\n")
                f.write("This analysis provides the most honest assessment of ES12 data:\n")
                f.write("- No ideal Sin wave inputs exist\n")
                f.write("- Data consists of nearly-constant values with noise\n")
                f.write("- Found pairs with visually identical VL waveforms\n")
                f.write("- Large time gaps allow degradation observation\n\n")
                
                f.write(f"Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            print(f"\n✅ Report generated: {report_path.name}")
        
        else:
            print(f"\n⚠️  No pairs found meeting the strict criteria")
            print(f"   ES12データには、50サイクル以上離れた視覚的に同一のVLペアが存在しません")
        
        print(f"\n" + "=" * 70)
        print("✅ Analysis Complete!")
        print(f"📍 Output Directory: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
