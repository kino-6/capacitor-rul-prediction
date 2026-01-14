#!/usr/bin/env python3
"""
Analyze Cycle 1 (Most Dynamic) and Find Similar Cycles

Focus on Cycle 1 which has the highest dynamism score and find
cycles with similar dynamic patterns for degradation comparison.
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
    print("🚀 Analyzing Cycle 1 and Finding Similar Dynamic Cycles")
    print("=" * 70)
    
    # Data path configuration
    data_path = Path("data/raw/ES12.mat")
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    # Output directory
    output_dir = Path("output/cycle1_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set font to avoid rendering issues
    plt.rcParams['font.family'] = ['DejaVu Sans']
    
    try:
        target_capacitor = "ES12C4"
        
        print(f"🎯 Analysis Target: {target_capacitor}, Cycle 1")
        
        # Load data
        with h5py.File(data_path, 'r') as f:
            cap_group = f['ES12']['Transient_Data'][target_capacitor]
            vl_data = cap_group['VL'][:]
            vo_data = cap_group['VO'][:]
            
            print(f"✅ Raw data loaded: VL {vl_data.shape}, VO {vo_data.shape}")
            
            # Extract Cycle 1
            vl_cycle1 = vl_data[:, 0]  # Cycle 1 (index 0)
            vo_cycle1 = vo_data[:, 0]
            
            # Remove NaN
            valid_mask1 = ~np.isnan(vl_cycle1) & ~np.isnan(vo_cycle1)
            vl1 = vl_cycle1[valid_mask1][:3000]
            vo1 = vo_cycle1[valid_mask1][:3000]
            
            print(f"✅ Cycle 1: VL length={len(vl1)}, VO length={len(vo1)}")
            print(f"   VL: mean={np.mean(vl1):.4f}V, std={np.std(vl1):.4f}V, range={np.max(vl1)-np.min(vl1):.4f}V")
            print(f"   VO: mean={np.mean(vo1):.4f}V, std={np.std(vo1):.4f}V, range={np.max(vo1)-np.min(vo1):.4f}V")
            
            # Visualize Cycle 1
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            
            time = np.arange(len(vl1))
            
            # VL plot
            axes[0].plot(time, vl1, 'b-', linewidth=0.5, alpha=0.8)
            axes[0].set_title('Cycle 1: VL Input (Most Dynamic Cycle)', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Time Points')
            axes[0].set_ylabel('VL Voltage (V)')
            axes[0].grid(True, alpha=0.3)
            axes[0].axhline(y=np.mean(vl1), color='r', linestyle='--', label=f'Mean: {np.mean(vl1):.4f}V')
            axes[0].legend()
            
            # VO plot
            axes[1].plot(time, vo1, 'g-', linewidth=0.5, alpha=0.8)
            axes[1].set_title('Cycle 1: VO Output', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Time Points')
            axes[1].set_ylabel('VO Voltage (V)')
            axes[1].grid(True, alpha=0.3)
            axes[1].axhline(y=np.mean(vo1), color='r', linestyle='--', label=f'Mean: {np.mean(vo1):.4f}V')
            axes[1].legend()
            
            plt.tight_layout()
            plot1_path = output_dir / 'ES12C4_cycle1_waveform.png'
            plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ Saved Cycle 1 waveform plot")
            
            # Now find similar cycles to Cycle 1
            print(f"\n🔍 Finding cycles similar to Cycle 1...")
            
            similar_cycles = []
            
            for cycle_idx in range(1, min(400, vl_data.shape[1])):  # Skip cycle 0 (already analyzed)
                cycle_num = cycle_idx + 1
                
                vl_cycle = vl_data[:, cycle_idx]
                vo_cycle = vo_data[:, cycle_idx]
                
                valid_mask = ~np.isnan(vl_cycle) & ~np.isnan(vo_cycle)
                
                if np.sum(valid_mask) < 3000:
                    continue
                
                vl = vl_cycle[valid_mask][:3000]
                vo = vo_cycle[valid_mask][:3000]
                
                # Calculate similarity to Cycle 1
                try:
                    corr, _ = pearsonr(vl1, vl)
                    if np.isnan(corr):
                        corr = 0.0
                except:
                    corr = 0.0
                
                # Calculate statistics
                vl_mean = np.mean(vl)
                vo_mean = np.mean(vo)
                vl_std = np.std(vl)
                vo_std = np.std(vo)
                vl_range = np.max(vl) - np.min(vl)
                vo_range = np.max(vo) - np.min(vo)
                
                # Voltage ratio
                ratio1 = np.mean(vo1) / np.mean(vl1) if np.mean(vl1) != 0 else np.nan
                ratio = vo_mean / vl_mean if vl_mean != 0 else np.nan
                
                if not np.isnan(ratio1) and not np.isnan(ratio) and ratio1 != 0:
                    degradation_pct = abs((ratio - ratio1) / ratio1) * 100
                else:
                    degradation_pct = 0
                
                # Amplitude similarity
                std_sim = 1 - abs(vl_std - np.std(vl1)) / max(vl_std, np.std(vl1)) if max(vl_std, np.std(vl1)) > 0 else 0
                range_sim = 1 - abs(vl_range - (np.max(vl1)-np.min(vl1))) / max(vl_range, np.max(vl1)-np.min(vl1)) if max(vl_range, np.max(vl1)-np.min(vl1)) > 0 else 0
                amp_sim = (std_sim + range_sim) / 2
                
                # Offset similarity
                mean_diff = abs(vl_mean - np.mean(vl1))
                max_mean = max(abs(vl_mean), abs(np.mean(vl1)))
                offset_sim = 1 - mean_diff / max_mean if max_mean > 0 else 1.0
                
                # Composite similarity
                composite_sim = corr * 0.5 + amp_sim * 0.3 + offset_sim * 0.2
                
                # Store if reasonably similar
                if composite_sim >= 0.5:  # Relaxed threshold
                    similar_cycles.append({
                        'cycle': cycle_num,
                        'correlation': corr,
                        'composite_similarity': composite_sim,
                        'amplitude_similarity': amp_sim,
                        'offset_similarity': offset_sim,
                        'degradation_pct': degradation_pct,
                        'time_gap': cycle_num - 1,
                        'ratio1': ratio1,
                        'ratio': ratio,
                        'vl_mean': vl_mean,
                        'vl_std': vl_std,
                        'vl_range': vl_range,
                        'vo_mean': vo_mean,
                        'vo_std': vo_std,
                        'vo_range': vo_range,
                        'vl_raw': vl,
                        'vo_raw': vo
                    })
            
            # Sort by composite similarity
            similar_cycles.sort(key=lambda x: x['composite_similarity'], reverse=True)
            
            print(f"✅ Found {len(similar_cycles)} cycles with similarity ≥ 0.5")
            
            if similar_cycles:
                print(f"\n📊 Top 10 Most Similar Cycles to Cycle 1:")
                for i, cyc in enumerate(similar_cycles[:10], 1):
                    print(f"   #{i}: Cycle {cyc['cycle']} "
                          f"(sim:{cyc['composite_similarity']:.3f}, "
                          f"corr:{cyc['correlation']:.3f}, "
                          f"gap:{cyc['time_gap']}, "
                          f"deg:{cyc['degradation_pct']:.1f}%)")
                
                # Visualize top 3 comparisons
                print(f"\n📊 Creating comparison visualizations...")
                
                for i, cyc in enumerate(similar_cycles[:3], 1):
                    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                    fig.suptitle(f'ES12C4: Cycle 1 vs Cycle {cyc["cycle"]}\n'
                                f'Time Gap: {cyc["time_gap"]} cycles, '
                                f'Similarity: {cyc["composite_similarity"]:.3f}, '
                                f'Degradation: {cyc["degradation_pct"]:.1f}%',
                                fontsize=14, fontweight='bold')
                    
                    time1 = np.arange(len(vl1))
                    time2 = np.arange(len(cyc['vl_raw']))
                    
                    # VL comparison
                    axes[0, 0].plot(time1, vl1, 'b-', label='Cycle 1', alpha=0.7, linewidth=0.5)
                    axes[0, 0].plot(time2, cyc['vl_raw'], 'r-', label=f'Cycle {cyc["cycle"]}', alpha=0.7, linewidth=0.5)
                    axes[0, 0].set_title('VL Input Comparison')
                    axes[0, 0].set_xlabel('Time Points')
                    axes[0, 0].set_ylabel('VL Voltage (V)')
                    axes[0, 0].legend()
                    axes[0, 0].grid(True, alpha=0.3)
                    
                    # Add similarity info
                    axes[0, 0].text(0.02, 0.98,
                                   f'Correlation: {cyc["correlation"]:.3f}\n'
                                   f'Amp Sim: {cyc["amplitude_similarity"]:.3f}\n'
                                   f'Offset Sim: {cyc["offset_similarity"]:.3f}',
                                   transform=axes[0, 0].transAxes,
                                   verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                    
                    # VO comparison
                    axes[0, 1].plot(time1, vo1, 'b-', label='Cycle 1', alpha=0.7, linewidth=0.5)
                    axes[0, 1].plot(time2, cyc['vo_raw'], 'r-', label=f'Cycle {cyc["cycle"]}', alpha=0.7, linewidth=0.5)
                    axes[0, 1].set_title('VO Output Comparison')
                    axes[0, 1].set_xlabel('Time Points')
                    axes[0, 1].set_ylabel('VO Voltage (V)')
                    axes[0, 1].legend()
                    axes[0, 1].grid(True, alpha=0.3)
                    
                    # Add degradation info
                    axes[0, 1].text(0.02, 0.98,
                                   f'Degradation: {cyc["degradation_pct"]:.1f}%\n'
                                   f'Ratio 1: {cyc["ratio1"]:.2f}\n'
                                   f'Ratio {cyc["cycle"]}: {cyc["ratio"]:.2f}',
                                   transform=axes[0, 1].transAxes,
                                   verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
                    
                    # VL statistics
                    stats = ['mean', 'std', 'range']
                    vl1_vals = [np.mean(vl1), np.std(vl1), np.max(vl1)-np.min(vl1)]
                    vl2_vals = [cyc['vl_mean'], cyc['vl_std'], cyc['vl_range']]
                    
                    x_pos = np.arange(len(stats))
                    width = 0.35
                    
                    axes[1, 0].bar(x_pos - width/2, vl1_vals, width, label='Cycle 1', alpha=0.8)
                    axes[1, 0].bar(x_pos + width/2, vl2_vals, width, label=f'Cycle {cyc["cycle"]}', alpha=0.8)
                    axes[1, 0].set_title('VL Input Statistics')
                    axes[1, 0].set_xlabel('Statistics')
                    axes[1, 0].set_ylabel('Value')
                    axes[1, 0].set_xticks(x_pos)
                    axes[1, 0].set_xticklabels(stats)
                    axes[1, 0].legend()
                    axes[1, 0].grid(True, alpha=0.3)
                    
                    # VO statistics
                    vo1_vals = [np.mean(vo1), np.std(vo1), np.max(vo1)-np.min(vo1)]
                    vo2_vals = [cyc['vo_mean'], cyc['vo_std'], cyc['vo_range']]
                    
                    axes[1, 1].bar(x_pos - width/2, vo1_vals, width, label='Cycle 1', alpha=0.8)
                    axes[1, 1].bar(x_pos + width/2, vo2_vals, width, label=f'Cycle {cyc["cycle"]}', alpha=0.8)
                    axes[1, 1].set_title('VO Output Statistics')
                    axes[1, 1].set_xlabel('Statistics')
                    axes[1, 1].set_ylabel('Value')
                    axes[1, 1].set_xticks(x_pos)
                    axes[1, 1].set_xticklabels(stats)
                    axes[1, 1].legend()
                    axes[1, 1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    
                    plot_path = output_dir / f'ES12C4_cycle1_vs_{cyc["cycle"]}_comparison.png'
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"   ✅ Saved comparison: Cycle 1 vs {cyc['cycle']}")
                
                # Generate report
                report_path = output_dir / 'ES12C4_cycle1_analysis_report.md'
                
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write("# ES12C4 Cycle 1 Analysis Report\n\n")
                    
                    f.write("## Overview\n\n")
                    f.write("Cycle 1 is the most dynamic cycle in the ES12C4 dataset with a dynamism score of 30.4. ")
                    f.write("This analysis identifies cycles similar to Cycle 1 for degradation comparison.\n\n")
                    
                    f.write("## Cycle 1 Characteristics\n\n")
                    f.write(f"- **VL Mean**: {np.mean(vl1):.4f}V\n")
                    f.write(f"- **VL Std**: {np.std(vl1):.4f}V\n")
                    f.write(f"- **VL Range**: {np.max(vl1)-np.min(vl1):.4f}V\n")
                    f.write(f"- **VO Mean**: {np.mean(vo1):.4f}V\n")
                    f.write(f"- **VO Std**: {np.std(vo1):.4f}V\n")
                    f.write(f"- **VO Range**: {np.max(vo1)-np.min(vo1):.4f}V\n\n")
                    
                    f.write("![Cycle 1 Waveform](ES12C4_cycle1_waveform.png)\n\n")
                    
                    f.write("## Similar Cycles Found\n\n")
                    f.write(f"Total cycles with similarity ≥ 0.5: {len(similar_cycles)}\n\n")
                    
                    f.write("### Top 10 Most Similar Cycles\n\n")
                    f.write("| Rank | Cycle | Similarity | Correlation | Time Gap | Degradation |\n")
                    f.write("|------|-------|------------|-------------|----------|-------------|\n")
                    
                    for i, cyc in enumerate(similar_cycles[:10], 1):
                        f.write(f"| {i} | {cyc['cycle']} | {cyc['composite_similarity']:.3f} | "
                               f"{cyc['correlation']:.3f} | {cyc['time_gap']} | {cyc['degradation_pct']:.1f}% |\n")
                    
                    f.write("\n## Detailed Comparison: Top 3 Pairs\n\n")
                    
                    for i, cyc in enumerate(similar_cycles[:3], 1):
                        f.write(f"### Pair {i}: Cycle 1 vs Cycle {cyc['cycle']}\n\n")
                        f.write(f"![Comparison](ES12C4_cycle1_vs_{cyc['cycle']}_comparison.png)\n\n")
                        f.write(f"- **Time Gap**: {cyc['time_gap']} cycles\n")
                        f.write(f"- **Composite Similarity**: {cyc['composite_similarity']:.3f}\n")
                        f.write(f"- **Correlation**: {cyc['correlation']:.3f}\n")
                        f.write(f"- **Amplitude Similarity**: {cyc['amplitude_similarity']:.3f}\n")
                        f.write(f"- **Offset Similarity**: {cyc['offset_similarity']:.3f}\n")
                        f.write(f"- **Degradation**: {cyc['degradation_pct']:.1f}%\n\n")
                    
                    f.write("## Conclusion\n\n")
                    f.write("Cycle 1 shows the most dynamic input pattern in the dataset. ")
                    f.write(f"Found {len(similar_cycles)} cycles with reasonable similarity for degradation analysis.\n\n")
                    
                    f.write(f"Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                
                print(f"\n✅ Report generated: {report_path.name}")
            
            else:
                print(f"\n⚠️  No similar cycles found")
        
        print(f"\n" + "=" * 70)
        print("✅ Cycle 1 Analysis Complete!")
        print(f"📍 Output Directory: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
