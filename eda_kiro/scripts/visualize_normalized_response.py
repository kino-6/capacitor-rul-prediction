#!/usr/bin/env python3
"""
Normalized Response Visualization Script

Compares capacitor responses by normalizing input signals to enable
proper comparison of "same input, different output" degradation patterns.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
import h5py

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# 警告を抑制
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class NormalizedResponseVisualizer:
    """Normalized response visualization class"""
    
    def __init__(self, output_dir: Path = Path("output/normalized_response")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set font to avoid rendering issues
        plt.rcParams['font.family'] = ['DejaVu Sans']
        
        # Color palette
        self.colors = plt.cm.viridis(np.linspace(0, 1, 10))
        
    def load_multiple_cycles(self, data_path: Path, capacitor_id: str, cycle_numbers: List[int]) -> Dict:
        """Load waveform data for multiple cycles"""
        print(f"📊 Loading multiple cycles for {capacitor_id}: {cycle_numbers}")
        
        try:
            with h5py.File(data_path, 'r') as f:
                # Navigate to transient data
                transient_group = f['ES12']['Transient_Data']
                cap_group = transient_group[capacitor_id]
                vl_data = cap_group['VL'][:]  # Shape: (time_points, cycles)
                vo_data = cap_group['VO'][:]  # Shape: (time_points, cycles)
                
                print(f"✅ Raw data loaded: VL shape {vl_data.shape}, VO shape {vo_data.shape}")
                
                cycles_data = {}
                
                for cycle_num in cycle_numbers:
                    if cycle_num > vl_data.shape[1]:
                        print(f"⚠️  Cycle {cycle_num} not available (max: {vl_data.shape[1]})")
                        continue
                    
                    # Extract cycle data (0-based indexing)
                    cycle_idx = cycle_num - 1
                    vl_cycle = vl_data[:, cycle_idx]
                    vo_cycle = vo_data[:, cycle_idx]
                    
                    # Remove NaN values
                    valid_mask = ~np.isnan(vl_cycle) & ~np.isnan(vo_cycle)
                    valid_indices = np.where(valid_mask)[0]
                    
                    if len(valid_indices) == 0:
                        print(f"⚠️  No valid data for cycle {cycle_num}")
                        continue
                    
                    vl_clean = vl_cycle[valid_mask]
                    vo_clean = vo_cycle[valid_mask]
                    
                    # Create time axis
                    time_points = np.arange(len(vl_clean))
                    
                    cycles_data[cycle_num] = {
                        'vl_raw': vl_clean,
                        'vo_raw': vo_clean,
                        'time_points': time_points,
                        'valid_points': len(vl_clean),
                        'total_points': len(vl_cycle)
                    }
                    
                    print(f"   Cycle {cycle_num}: {len(vl_clean)} valid points ({len(vl_clean)/len(vl_cycle)*100:.1f}%)")
                
                return {
                    'capacitor_id': capacitor_id,
                    'cycles_data': cycles_data
                }
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def normalize_responses(self, multi_cycle_data: Dict) -> Dict:
        """Normalize responses for proper comparison"""
        print(f"🔧 Normalizing responses for comparison...")
        
        cycles_data = multi_cycle_data['cycles_data']
        normalized_data = {}
        
        # Find reference cycle (typically cycle 1) for normalization
        reference_cycle = min(cycles_data.keys())
        ref_data = cycles_data[reference_cycle]
        
        # Calculate reference statistics
        ref_vl_std = np.std(ref_data['vl_raw'])
        ref_vl_mean = np.mean(ref_data['vl_raw'])
        
        print(f"📏 Using Cycle {reference_cycle} as reference:")
        print(f"   Reference VL mean: {ref_vl_mean:.6f}")
        print(f"   Reference VL std: {ref_vl_std:.6f}")
        
        for cycle_num, cycle_data in cycles_data.items():
            vl_raw = cycle_data['vl_raw']
            vo_raw = cycle_data['vo_raw']
            
            # Method 1: Z-score normalization (zero mean, unit variance)
            vl_zscore = (vl_raw - np.mean(vl_raw)) / np.std(vl_raw)
            vo_zscore = (vo_raw - np.mean(vo_raw)) / np.std(vo_raw)
            
            # Method 2: Scale to reference cycle's range
            current_vl_std = np.std(vl_raw)
            current_vl_mean = np.mean(vl_raw)
            
            if current_vl_std > 0:
                vl_scaled = (vl_raw - current_vl_mean) / current_vl_std * ref_vl_std + ref_vl_mean
            else:
                vl_scaled = vl_raw
            
            # Method 3: Min-Max normalization to [-1, 1]
            vl_min, vl_max = np.min(vl_raw), np.max(vl_raw)
            vo_min, vo_max = np.min(vo_raw), np.max(vo_raw)
            
            if vl_max != vl_min:
                vl_minmax = 2 * (vl_raw - vl_min) / (vl_max - vl_min) - 1
            else:
                vl_minmax = np.zeros_like(vl_raw)
                
            if vo_max != vo_min:
                vo_minmax = 2 * (vo_raw - vo_min) / (vo_max - vo_min) - 1
            else:
                vo_minmax = np.zeros_like(vo_raw)
            
            # Calculate response characteristics
            raw_ratio = np.mean(vo_raw) / np.mean(vl_raw) if np.mean(vl_raw) != 0 else np.nan
            correlation = np.corrcoef(vl_raw, vo_raw)[0, 1]
            
            # Calculate gain (slope of linear fit)
            try:
                coeffs = np.polyfit(vl_raw, vo_raw, 1)
                gain = coeffs[0]
            except:
                gain = np.nan
            
            normalized_data[cycle_num] = {
                'vl_raw': vl_raw,
                'vo_raw': vo_raw,
                'vl_zscore': vl_zscore,
                'vo_zscore': vo_zscore,
                'vl_scaled': vl_scaled,
                'vl_minmax': vl_minmax,
                'vo_minmax': vo_minmax,
                'time_points': cycle_data['time_points'],
                'raw_ratio': raw_ratio,
                'correlation': correlation,
                'gain': gain,
                'vl_stats': {
                    'mean': np.mean(vl_raw),
                    'std': np.std(vl_raw),
                    'min': np.min(vl_raw),
                    'max': np.max(vl_raw)
                },
                'vo_stats': {
                    'mean': np.mean(vo_raw),
                    'std': np.std(vo_raw),
                    'min': np.min(vo_raw),
                    'max': np.max(vo_raw)
                }
            }
            
            print(f"   Cycle {cycle_num}: Ratio={raw_ratio:.3f}, Gain={gain:.3f}, Corr={correlation:.3f}")
        
        return {
            'capacitor_id': multi_cycle_data['capacitor_id'],
            'reference_cycle': reference_cycle,
            'normalized_data': normalized_data
        }
    
    def visualize_normalized_comparison(self, normalized_data: Dict) -> Path:
        """Visualize normalized response comparison"""
        capacitor_id = normalized_data['capacitor_id']
        cycles_data = normalized_data['normalized_data']
        
        print(f"📈 Creating normalized response comparison for {capacitor_id}...")
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(f'{capacitor_id} Normalized Response Comparison - Same Input Analysis', 
                     fontsize=16, fontweight='bold')
        
        cycle_numbers = sorted(cycles_data.keys())
        colors = plt.cm.tab10(np.linspace(0, 1, len(cycle_numbers)))
        
        # 1. Raw waveforms (same scale)
        ax1 = axes[0, 0]
        
        # Find global scale for raw data
        all_vl = np.concatenate([data['vl_raw'] for data in cycles_data.values()])
        all_vo = np.concatenate([data['vo_raw'] for data in cycles_data.values()])
        vl_global_range = [np.min(all_vl), np.max(all_vl)]
        vo_global_range = [np.min(all_vo), np.max(all_vo)]
        
        for i, cycle_num in enumerate(cycle_numbers):
            data = cycles_data[cycle_num]
            # Subsample for visualization
            subsample = max(1, len(data['vl_raw']) // 1000)
            time_sub = data['time_points'][::subsample]
            vl_sub = data['vl_raw'][::subsample]
            vo_sub = data['vo_raw'][::subsample]
            
            ax1.plot(time_sub, vl_sub, color=colors[i], alpha=0.7, linewidth=1, 
                    label=f'Cycle {cycle_num} VL')
            ax1.plot(time_sub, vo_sub, color=colors[i], alpha=0.7, linewidth=1, 
                    linestyle='--', label=f'Cycle {cycle_num} VO')
        
        ax1.set_title('Raw Waveforms (Same Scale)', fontweight='bold')
        ax1.set_xlabel('Time Points')
        ax1.set_ylabel('Voltage (V)')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 2. Normalized waveforms (Z-score)
        ax2 = axes[0, 1]
        
        for i, cycle_num in enumerate(cycle_numbers):
            data = cycles_data[cycle_num]
            subsample = max(1, len(data['vl_zscore']) // 1000)
            time_sub = data['time_points'][::subsample]
            vl_sub = data['vl_zscore'][::subsample]
            vo_sub = data['vo_zscore'][::subsample]
            
            ax2.plot(time_sub, vl_sub, color=colors[i], alpha=0.7, linewidth=1, 
                    label=f'Cycle {cycle_num} VL')
            ax2.plot(time_sub, vo_sub, color=colors[i], alpha=0.7, linewidth=1, 
                    linestyle='--', label=f'Cycle {cycle_num} VO')
        
        ax2.set_title('Z-Score Normalized Waveforms', fontweight='bold')
        ax2.set_xlabel('Time Points')
        ax2.set_ylabel('Z-Score')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        
        # 3. Input-Output relationships
        ax3 = axes[1, 0]
        
        for i, cycle_num in enumerate(cycle_numbers):
            data = cycles_data[cycle_num]
            subsample = max(1, len(data['vl_raw']) // 2000)
            vl_sub = data['vl_raw'][::subsample]
            vo_sub = data['vo_raw'][::subsample]
            
            ax3.scatter(vl_sub, vo_sub, color=colors[i], alpha=0.6, s=1, 
                       label=f'Cycle {cycle_num} (Gain: {data["gain"]:.2f})')
        
        ax3.set_title('Input-Output Relationships', fontweight='bold')
        ax3.set_xlabel('VL (Input Voltage)')
        ax3.set_ylabel('VO (Output Voltage)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Response characteristics comparison
        ax4 = axes[1, 1]
        
        ratios = [cycles_data[c]['raw_ratio'] for c in cycle_numbers]
        gains = [cycles_data[c]['gain'] for c in cycle_numbers]
        correlations = [cycles_data[c]['correlation'] for c in cycle_numbers]
        
        x_pos = np.arange(len(cycle_numbers))
        width = 0.25
        
        ax4.bar(x_pos - width, ratios, width, label='Voltage Ratio', alpha=0.8)
        ax4.bar(x_pos, gains, width, label='Gain (Slope)', alpha=0.8)
        ax4.bar(x_pos + width, [c*10 for c in correlations], width, label='Correlation×10', alpha=0.8)
        
        ax4.set_title('Response Characteristics Comparison', fontweight='bold')
        ax4.set_xlabel('Cycle')
        ax4.set_ylabel('Value')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f'Cycle {c}' for c in cycle_numbers])
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Min-Max normalized comparison
        ax5 = axes[2, 0]
        
        for i, cycle_num in enumerate(cycle_numbers):
            data = cycles_data[cycle_num]
            subsample = max(1, len(data['vl_minmax']) // 1000)
            time_sub = data['time_points'][::subsample]
            vl_sub = data['vl_minmax'][::subsample]
            vo_sub = data['vo_minmax'][::subsample]
            
            ax5.plot(time_sub, vl_sub, color=colors[i], alpha=0.7, linewidth=1, 
                    label=f'Cycle {cycle_num} VL')
            ax5.plot(time_sub, vo_sub, color=colors[i], alpha=0.7, linewidth=1, 
                    linestyle='--', label=f'Cycle {cycle_num} VO')
        
        ax5.set_title('Min-Max Normalized [-1,1]', fontweight='bold')
        ax5.set_xlabel('Time Points')
        ax5.set_ylabel('Normalized Value')
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax5.grid(True, alpha=0.3)
        ax5.set_ylim(-1.1, 1.1)
        
        # 6. Statistics summary
        ax6 = axes[2, 1]
        
        # Create summary table
        summary_data = []
        for cycle_num in cycle_numbers:
            data = cycles_data[cycle_num]
            summary_data.append([
                f'Cycle {cycle_num}',
                f'{data["vl_stats"]["mean"]:.4f}',
                f'{data["vo_stats"]["mean"]:.4f}',
                f'{data["raw_ratio"]:.3f}',
                f'{data["correlation"]:.3f}'
            ])
        
        table = ax6.table(cellText=summary_data,
                         colLabels=['Cycle', 'VL Mean', 'VO Mean', 'Ratio', 'Correlation'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        ax6.set_title('Summary Statistics', fontweight='bold')
        ax6.axis('off')
        
        plt.tight_layout()
        
        plot_path = self.output_dir / f'{capacitor_id}_normalized_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def generate_normalized_report(self, normalized_data: Dict) -> Path:
        """Generate normalized response analysis report"""
        capacitor_id = normalized_data['capacitor_id']
        cycles_data = normalized_data['normalized_data']
        
        print(f"📄 Generating normalized response report...")
        
        report_path = self.output_dir / f'{capacitor_id}_normalized_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# {capacitor_id} Normalized Response Analysis Report\n\n")
            
            f.write("## Overview\n\n")
            f.write(f"Analysis of {capacitor_id} response characteristics across multiple cycles.\n")
            f.write("This report normalizes input signals to enable proper comparison of degradation patterns.\n\n")
            
            f.write("## Key Issue Identified\n\n")
            f.write("⚠️ **Input Signal Variation**: The input signals (VL) vary significantly between cycles,\n")
            f.write("making direct comparison of 'same input, different output' impossible without normalization.\n\n")
            
            f.write("## Input Signal Variation Analysis\n\n")
            
            cycle_numbers = sorted(cycles_data.keys())
            for cycle_num in cycle_numbers:
                data = cycles_data[cycle_num]
                vl_stats = data['vl_stats']
                f.write(f"### Cycle {cycle_num}\n")
                f.write(f"- **VL Mean**: {vl_stats['mean']:.6f} V\n")
                f.write(f"- **VL Std**: {vl_stats['std']:.6f} V\n")
                f.write(f"- **VL Range**: {vl_stats['min']:.6f} to {vl_stats['max']:.6f} V\n\n")
            
            f.write("## Response Characteristics Comparison\n\n")
            
            f.write("| Cycle | VL Mean (V) | VO Mean (V) | Voltage Ratio | Gain | Correlation |\n")
            f.write("|-------|-------------|-------------|---------------|------|-------------|\n")
            
            for cycle_num in cycle_numbers:
                data = cycles_data[cycle_num]
                f.write(f"| {cycle_num} | {data['vl_stats']['mean']:.6f} | {data['vo_stats']['mean']:.6f} | ")
                f.write(f"{data['raw_ratio']:.3f} | {data['gain']:.3f} | {data['correlation']:.3f} |\n")
            
            f.write("\n## Degradation Analysis\n\n")
            
            # Compare to reference cycle
            ref_cycle = min(cycle_numbers)
            ref_data = cycles_data[ref_cycle]
            
            f.write(f"Using Cycle {ref_cycle} as reference:\n\n")
            
            for cycle_num in cycle_numbers:
                if cycle_num == ref_cycle:
                    continue
                    
                data = cycles_data[cycle_num]
                ratio_change = ((data['raw_ratio'] - ref_data['raw_ratio']) / abs(ref_data['raw_ratio'])) * 100
                gain_change = ((data['gain'] - ref_data['gain']) / abs(ref_data['gain'])) * 100
                
                f.write(f"### Cycle {cycle_num} vs Cycle {ref_cycle}\n")
                f.write(f"- **Voltage Ratio Change**: {ratio_change:+.1f}%\n")
                f.write(f"- **Gain Change**: {gain_change:+.1f}%\n")
                f.write(f"- **Correlation**: {data['correlation']:.3f}\n")
                
                if abs(ratio_change) > 100:
                    f.write("- 🔴 **Severe degradation detected**\n")
                elif abs(ratio_change) > 50:
                    f.write("- 🟡 **Moderate degradation detected**\n")
                else:
                    f.write("- 🟢 **Stable response**\n")
                f.write("\n")
            
            f.write("## Conclusions\n\n")
            f.write("1. **Input Variability**: Input signals vary significantly between cycles\n")
            f.write("2. **Normalization Required**: Direct comparison requires signal normalization\n")
            f.write("3. **Response Changes**: Significant changes in voltage ratio and gain observed\n")
            f.write("4. **Correlation Maintained**: High correlation suggests linear relationship preserved\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("- Use normalized signals for degradation analysis\n")
            f.write("- Focus on gain and correlation changes rather than absolute values\n")
            f.write("- Consider input signal stabilization in future experiments\n\n")
            
            f.write(f"---\n")
            f.write(f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        return report_path

def main():
    """Main execution function"""
    print("🚀 Starting Normalized Response Analysis")
    print("=" * 60)
    
    # Data path configuration
    data_path = Path("data/raw/ES12.mat")
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    # Initialize visualizer
    visualizer = NormalizedResponseVisualizer()
    
    try:
        # Configuration
        target_capacitor = "ES12C4"
        cycles_to_analyze = [1, 100, 200, 300]
        
        print(f"🎯 Analysis Target: {target_capacitor}")
        print(f"🔄 Cycles to analyze: {cycles_to_analyze}")
        
        # Load multiple cycles
        multi_cycle_data = visualizer.load_multiple_cycles(data_path, target_capacitor, cycles_to_analyze)
        
        if multi_cycle_data is None:
            print("❌ Failed to load data")
            return
        
        # Normalize responses
        normalized_data = visualizer.normalize_responses(multi_cycle_data)
        
        # Generate visualization
        plot_path = visualizer.visualize_normalized_comparison(normalized_data)
        
        # Generate report
        report_path = visualizer.generate_normalized_report(normalized_data)
        
        # Summary
        print(f"\n" + "=" * 60)
        print("✅ Normalized Response Analysis Complete!")
        
        print(f"\n📊 Key Findings:")
        cycles_data = normalized_data['normalized_data']
        for cycle_num in sorted(cycles_data.keys()):
            data = cycles_data[cycle_num]
            print(f"   Cycle {cycle_num}: VL_mean={data['vl_stats']['mean']:.6f}V, Ratio={data['raw_ratio']:.3f}")
        
        print(f"\n📁 Generated Files:")
        print(f"   - {plot_path.name}")
        print(f"   - {report_path.name}")
        
        print(f"\n📍 Output Directory: {visualizer.output_dir}")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()