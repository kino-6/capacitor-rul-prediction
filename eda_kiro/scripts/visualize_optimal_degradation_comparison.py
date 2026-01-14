#!/usr/bin/env python3
"""
Optimal Degradation Comparison Visualization

This script finds the optimal pairs of cycles for degradation analysis by balancing:
1. Input similarity (similar VL patterns)
2. Temporal separation (sufficient time gap to observe degradation)
3. Degradation significance (meaningful response changes)

The goal is to find cycles that are similar enough in input but far enough apart 
in time to clearly demonstrate degradation effects.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import itertools

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class OptimalDegradationVisualizer:
    """Find and visualize optimal cycle pairs for degradation analysis"""
    
    def __init__(self, output_dir: Path = Path("output/optimal_degradation")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set font to avoid rendering issues
        plt.rcParams['font.family'] = ['DejaVu Sans']
        
        # Color palette
        self.colors = plt.cm.viridis(np.linspace(0, 1, 10))
        
    def load_and_analyze_cycles(self, data_path: Path, capacitor_id: str, max_cycles: int = 100) -> Dict:
        """Load cycles and calculate similarity/degradation metrics"""
        print(f"📊 Loading and analyzing cycles for {capacitor_id}...")
        
        with h5py.File(data_path, 'r') as f:
            cap_group = f['ES12']['Transient_Data'][capacitor_id]
            vl_data = cap_group['VL'][:]
            vo_data = cap_group['VO'][:]
            
            print(f"✅ Raw data loaded: VL {vl_data.shape}, VO {vo_data.shape}")
            
            # Process cycles
            target_length = 3000
            cycles_data = {}
            valid_cycles = []
            vl_matrix = []
            
            for cycle_idx in range(min(max_cycles, vl_data.shape[1])):
                cycle_num = cycle_idx + 1
                
                vl_cycle = vl_data[:, cycle_idx]
                vo_cycle = vo_data[:, cycle_idx]
                
                # Remove NaN values
                valid_mask = ~np.isnan(vl_cycle) & ~np.isnan(vo_cycle)
                
                if np.sum(valid_mask) < target_length:
                    continue
                
                vl_clean = vl_cycle[valid_mask][:target_length]
                vo_clean = vo_cycle[valid_mask][:target_length]
                
                cycles_data[cycle_num] = {
                    'vl_raw': vl_clean,
                    'vo_raw': vo_clean,
                    'vl_mean': np.mean(vl_clean),
                    'vo_mean': np.mean(vo_clean),
                    'vl_std': np.std(vl_clean),
                    'vo_std': np.std(vo_clean),
                    'voltage_ratio': np.mean(vo_clean) / np.mean(vl_clean) if np.mean(vl_clean) != 0 else np.nan,
                    'vo_amplitude': np.max(vo_clean) - np.min(vo_clean)
                }
                
                vl_matrix.append(vl_clean)
                valid_cycles.append(cycle_num)
            
            print(f"✅ Processed {len(valid_cycles)} valid cycles")
            
            # Calculate pairwise similarities and degradation metrics
            vl_matrix = np.array(vl_matrix)
            similarity_matrix = np.corrcoef(vl_matrix)
            
            return {
                'capacitor_id': capacitor_id,
                'cycles_data': cycles_data,
                'valid_cycles': valid_cycles,
                'similarity_matrix': similarity_matrix,
                'target_length': target_length
            }
    
    def find_optimal_pairs(self, analysis_data: Dict, 
                          min_similarity: float = 0.7,
                          min_time_gap: int = 10,
                          min_degradation: float = 20.0) -> List[Tuple]:
        """Find optimal cycle pairs balancing similarity, time gap, and degradation"""
        print(f"🔍 Finding optimal cycle pairs...")
        print(f"   Criteria: similarity≥{min_similarity}, time_gap≥{min_time_gap}, degradation≥{min_degradation}%")
        
        cycles_data = analysis_data['cycles_data']
        valid_cycles = analysis_data['valid_cycles']
        similarity_matrix = analysis_data['similarity_matrix']
        
        optimal_pairs = []
        
        # Evaluate all possible pairs
        for i, cycle1 in enumerate(valid_cycles):
            for j, cycle2 in enumerate(valid_cycles):
                if i >= j:  # Only consider unique pairs where cycle1 < cycle2
                    continue
                
                # Calculate metrics
                time_gap = cycle2 - cycle1
                input_similarity = similarity_matrix[i, j]
                
                # Calculate degradation
                ratio1 = cycles_data[cycle1]['voltage_ratio']
                ratio2 = cycles_data[cycle2]['voltage_ratio']
                
                if not np.isnan(ratio1) and not np.isnan(ratio2) and ratio1 != 0:
                    degradation_pct = abs((ratio2 - ratio1) / ratio1) * 100
                else:
                    degradation_pct = 0
                
                # Check if pair meets criteria
                if (input_similarity >= min_similarity and 
                    time_gap >= min_time_gap and 
                    degradation_pct >= min_degradation):
                    
                    # Calculate composite score (higher is better)
                    # Balance similarity, time gap, and degradation significance
                    similarity_score = input_similarity
                    time_score = min(time_gap / 50.0, 1.0)  # Normalize to [0,1], cap at 50 cycles
                    degradation_score = min(degradation_pct / 100.0, 1.0)  # Normalize to [0,1], cap at 100%
                    
                    composite_score = (similarity_score * 0.4 + 
                                     time_score * 0.3 + 
                                     degradation_score * 0.3)
                    
                    optimal_pairs.append({
                        'cycle1': cycle1,
                        'cycle2': cycle2,
                        'time_gap': time_gap,
                        'input_similarity': input_similarity,
                        'degradation_pct': degradation_pct,
                        'composite_score': composite_score,
                        'ratio1': ratio1,
                        'ratio2': ratio2
                    })
        
        # Sort by composite score (best pairs first)
        optimal_pairs.sort(key=lambda x: x['composite_score'], reverse=True)
        
        print(f"✅ Found {len(optimal_pairs)} optimal pairs")
        
        # Show top pairs
        for i, pair in enumerate(optimal_pairs[:5]):
            print(f"   #{i+1}: Cycles {pair['cycle1']}-{pair['cycle2']} "
                  f"(gap:{pair['time_gap']}, sim:{pair['input_similarity']:.3f}, "
                  f"deg:{pair['degradation_pct']:.1f}%, score:{pair['composite_score']:.3f})")
        
        return optimal_pairs
    
    def visualize_optimal_pairs(self, analysis_data: Dict, optimal_pairs: List[Tuple], 
                               top_n: int = 6) -> Path:
        """Visualize the top optimal cycle pairs"""
        capacitor_id = analysis_data['capacitor_id']
        cycles_data = analysis_data['cycles_data']
        
        print(f"📈 Creating optimal pairs visualization for {capacitor_id}...")
        
        # Select top pairs
        top_pairs = optimal_pairs[:top_n]
        
        # Create visualization
        n_pairs = len(top_pairs)
        fig, axes = plt.subplots(n_pairs, 3, figsize=(18, 6*n_pairs))
        if n_pairs == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'{capacitor_id} Optimal Degradation Comparison Pairs\n'
                     f'Balanced: Input Similarity + Time Gap + Degradation Significance', 
                     fontsize=16, fontweight='bold')
        
        for pair_idx, pair in enumerate(top_pairs):
            cycle1, cycle2 = pair['cycle1'], pair['cycle2']
            
            data1 = cycles_data[cycle1]
            data2 = cycles_data[cycle2]
            
            # Find common scale for both cycles
            all_vl = np.concatenate([data1['vl_raw'], data2['vl_raw']])
            all_vo = np.concatenate([data1['vo_raw'], data2['vo_raw']])
            
            vl_min, vl_max = np.min(all_vl), np.max(all_vl)
            vo_min, vo_max = np.min(all_vo), np.max(all_vo)
            
            # Add padding
            vl_padding = (vl_max - vl_min) * 0.05
            vo_padding = (vo_max - vo_min) * 0.05
            
            vl_ylim = [vl_min - vl_padding, vl_max + vl_padding]
            vo_ylim = [vo_min - vo_padding, vo_max + vo_padding]
            
            # 1. Input comparison (VL)
            ax1 = axes[pair_idx, 0]
            
            # Subsample for visualization
            subsample = max(1, len(data1['vl_raw']) // 2000)
            time_points = np.arange(0, len(data1['vl_raw']), subsample)
            
            vl1_sub = data1['vl_raw'][::subsample]
            vl2_sub = data2['vl_raw'][::subsample]
            
            ax1.plot(time_points, vl1_sub, color='blue', linewidth=2, alpha=0.8,
                    label=f'Cycle {cycle1} (Early)')
            ax1.plot(time_points, vl2_sub, color='red', linewidth=2, alpha=0.8,
                    label=f'Cycle {cycle2} (Late)')
            
            ax1.set_title(f'Pair #{pair_idx+1}: Input Comparison (VL)\n'
                         f'Similarity: {pair["input_similarity"]:.3f}', 
                         fontweight='bold')
            ax1.set_xlabel('Time Points')
            ax1.set_ylabel('VL (Input Voltage)')
            ax1.set_ylim(vl_ylim)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Output comparison (VO)
            ax2 = axes[pair_idx, 1]
            
            vo1_sub = data1['vo_raw'][::subsample]
            vo2_sub = data2['vo_raw'][::subsample]
            
            ax2.plot(time_points, vo1_sub, color='blue', linewidth=2, alpha=0.8,
                    label=f'Cycle {cycle1} Response')
            ax2.plot(time_points, vo2_sub, color='red', linewidth=2, alpha=0.8,
                    label=f'Cycle {cycle2} Response')
            
            ax2.set_title(f'Output Response Comparison (VO)\n'
                         f'Degradation: {pair["degradation_pct"]:.1f}%', 
                         fontweight='bold')
            ax2.set_xlabel('Time Points')
            ax2.set_ylabel('VO (Output Voltage)')
            ax2.set_ylim(vo_ylim)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # 3. Metrics comparison
            ax3 = axes[pair_idx, 2]
            
            # Calculate additional metrics
            try:
                corr1, _ = pearsonr(data1['vl_raw'], data1['vo_raw'])
                corr2, _ = pearsonr(data2['vl_raw'], data2['vo_raw'])
            except:
                corr1, corr2 = np.nan, np.nan
            
            amp1 = data1['vo_amplitude']
            amp2 = data2['vo_amplitude']
            
            # Create comparison bars
            metrics = ['Voltage\nRatio', 'VL-VO\nCorrelation', 'Output\nAmplitude']
            early_values = [pair['ratio1'], corr1, amp1]
            late_values = [pair['ratio2'], corr2, amp2]
            
            x_pos = np.arange(len(metrics))
            width = 0.35
            
            bars1 = ax3.bar(x_pos - width/2, early_values, width, 
                           label=f'Cycle {cycle1}', color='blue', alpha=0.7)
            bars2 = ax3.bar(x_pos + width/2, late_values, width, 
                           label=f'Cycle {cycle2}', color='red', alpha=0.7)
            
            ax3.set_title(f'Metrics Comparison\n'
                         f'Time Gap: {pair["time_gap"]} cycles', 
                         fontweight='bold')
            ax3.set_xlabel('Metrics')
            ax3.set_ylabel('Value')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(metrics)
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars1, early_values):
                if not np.isnan(value):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{value:.2f}', ha='center', va='bottom', fontsize=9)
            
            for bar, value in zip(bars2, late_values):
                if not np.isnan(value):
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{value:.2f}', ha='center', va='bottom', fontsize=9)
            
            # Add composite score
            ax3.text(0.98, 0.98, f'Score: {pair["composite_score"]:.3f}', 
                    transform=ax3.transAxes, verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
        
        plt.tight_layout()
        
        plot_path = self.output_dir / f'{capacitor_id}_optimal_pairs.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def create_selection_criteria_analysis(self, analysis_data: Dict, optimal_pairs: List[Tuple]) -> Path:
        """Analyze and visualize the selection criteria trade-offs"""
        capacitor_id = analysis_data['capacitor_id']
        
        print(f"📊 Creating selection criteria analysis...")
        
        if not optimal_pairs:
            print("⚠️  No optimal pairs found for criteria analysis")
            return None
        
        # Extract data for analysis
        similarities = [pair['input_similarity'] for pair in optimal_pairs]
        time_gaps = [pair['time_gap'] for pair in optimal_pairs]
        degradations = [pair['degradation_pct'] for pair in optimal_pairs]
        scores = [pair['composite_score'] for pair in optimal_pairs]
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{capacitor_id} Selection Criteria Analysis\n'
                     f'Trade-offs between Similarity, Time Gap, and Degradation', 
                     fontsize=16, fontweight='bold')
        
        # 1. Similarity vs Time Gap
        ax1 = axes[0, 0]
        scatter = ax1.scatter(similarities, time_gaps, c=scores, cmap='viridis', 
                             s=60, alpha=0.7, edgecolors='black')
        ax1.set_xlabel('Input Similarity')
        ax1.set_ylabel('Time Gap (cycles)')
        ax1.set_title('Similarity vs Time Gap\n(Color = Composite Score)', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Composite Score')
        
        # 2. Similarity vs Degradation
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(similarities, degradations, c=time_gaps, cmap='plasma', 
                              s=60, alpha=0.7, edgecolors='black')
        ax2.set_xlabel('Input Similarity')
        ax2.set_ylabel('Degradation (%)')
        ax2.set_title('Similarity vs Degradation\n(Color = Time Gap)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=ax2, label='Time Gap (cycles)')
        
        # 3. Time Gap vs Degradation
        ax3 = axes[1, 0]
        scatter3 = ax3.scatter(time_gaps, degradations, c=similarities, cmap='coolwarm', 
                              s=60, alpha=0.7, edgecolors='black')
        ax3.set_xlabel('Time Gap (cycles)')
        ax3.set_ylabel('Degradation (%)')
        ax3.set_title('Time Gap vs Degradation\n(Color = Similarity)', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=ax3, label='Input Similarity')
        
        # 4. Score distribution
        ax4 = axes[1, 1]
        
        # Create score histogram
        ax4.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.axvline(np.mean(scores), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(scores):.3f}')
        ax4.axvline(np.median(scores), color='green', linestyle='--', linewidth=2, 
                   label=f'Median: {np.median(scores):.3f}')
        
        ax4.set_xlabel('Composite Score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Composite Score Distribution', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f'Total Pairs: {len(optimal_pairs)}\n'
        stats_text += f'Score Range: {min(scores):.3f} - {max(scores):.3f}\n'
        stats_text += f'Avg Similarity: {np.mean(similarities):.3f}\n'
        stats_text += f'Avg Time Gap: {np.mean(time_gaps):.1f} cycles\n'
        stats_text += f'Avg Degradation: {np.mean(degradations):.1f}%'
        
        ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        plot_path = self.output_dir / f'{capacitor_id}_criteria_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def generate_optimal_pairs_report(self, analysis_data: Dict, optimal_pairs: List[Tuple]) -> Path:
        """Generate detailed report on optimal cycle pairs"""
        capacitor_id = analysis_data['capacitor_id']
        
        print(f"📄 Generating optimal pairs report...")
        
        report_path = self.output_dir / f'{capacitor_id}_optimal_pairs_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# {capacitor_id} Optimal Degradation Comparison Pairs Report\n\n")
            
            f.write("## Overview\n\n")
            f.write("This report identifies optimal cycle pairs for degradation analysis by balancing:\n")
            f.write("1. **Input Similarity**: Cycles with similar VL patterns for fair comparison\n")
            f.write("2. **Temporal Separation**: Sufficient time gap to observe meaningful degradation\n")
            f.write("3. **Degradation Significance**: Measurable changes in response characteristics\n\n")
            
            f.write("## Selection Criteria\n\n")
            f.write("- **Minimum Input Similarity**: ≥0.7 (correlation coefficient)\n")
            f.write("- **Minimum Time Gap**: ≥10 cycles\n")
            f.write("- **Minimum Degradation**: ≥20% voltage ratio change\n")
            f.write("- **Composite Score**: Weighted combination (40% similarity + 30% time + 30% degradation)\n\n")
            
            f.write(f"## Results Summary\n\n")
            f.write(f"- **Total Optimal Pairs Found**: {len(optimal_pairs)}\n")
            
            if optimal_pairs:
                similarities = [pair['input_similarity'] for pair in optimal_pairs]
                time_gaps = [pair['time_gap'] for pair in optimal_pairs]
                degradations = [pair['degradation_pct'] for pair in optimal_pairs]
                scores = [pair['composite_score'] for pair in optimal_pairs]
                
                f.write(f"- **Average Input Similarity**: {np.mean(similarities):.3f}\n")
                f.write(f"- **Average Time Gap**: {np.mean(time_gaps):.1f} cycles\n")
                f.write(f"- **Average Degradation**: {np.mean(degradations):.1f}%\n")
                f.write(f"- **Score Range**: {min(scores):.3f} - {max(scores):.3f}\n\n")
                
                f.write("## Top 10 Optimal Pairs\n\n")
                f.write("| Rank | Cycle Pair | Time Gap | Similarity | Degradation | Score | Early Ratio | Late Ratio |\n")
                f.write("|------|------------|----------|------------|-------------|-------|-------------|------------|\n")
                
                for i, pair in enumerate(optimal_pairs[:10]):
                    f.write(f"| {i+1} | {pair['cycle1']}-{pair['cycle2']} | {pair['time_gap']} | ")
                    f.write(f"{pair['input_similarity']:.3f} | {pair['degradation_pct']:.1f}% | ")
                    f.write(f"{pair['composite_score']:.3f} | {pair['ratio1']:.2f} | {pair['ratio2']:.2f} |\n")
                
                f.write("\n## Detailed Analysis of Top 5 Pairs\n\n")
                
                for i, pair in enumerate(optimal_pairs[:5]):
                    f.write(f"### Pair #{i+1}: Cycles {pair['cycle1']} → {pair['cycle2']}\n\n")
                    
                    f.write("#### Characteristics\n")
                    f.write(f"- **Time Separation**: {pair['time_gap']} cycles\n")
                    f.write(f"- **Input Similarity**: {pair['input_similarity']:.3f} (correlation)\n")
                    f.write(f"- **Voltage Ratio Change**: {pair['degradation_pct']:.1f}%\n")
                    f.write(f"- **Composite Score**: {pair['composite_score']:.3f}\n\n")
                    
                    f.write("#### Response Changes\n")
                    f.write(f"- **Early Cycle ({pair['cycle1']}) Ratio**: {pair['ratio1']:.3f}\n")
                    f.write(f"- **Late Cycle ({pair['cycle2']}) Ratio**: {pair['ratio2']:.3f}\n")
                    
                    ratio_change = ((pair['ratio2'] - pair['ratio1']) / pair['ratio1']) * 100
                    f.write(f"- **Change Direction**: {'+' if ratio_change > 0 else ''}{ratio_change:.1f}%\n")
                    
                    # Severity assessment
                    if abs(ratio_change) > 100:
                        f.write("- **Degradation Severity**: 🔴 **Severe** (>100% change)\n")
                    elif abs(ratio_change) > 50:
                        f.write("- **Degradation Severity**: 🟡 **Moderate** (50-100% change)\n")
                    else:
                        f.write("- **Degradation Severity**: 🟠 **Mild** (20-50% change)\n")
                    
                    f.write("\n#### Why This Pair is Optimal\n")
                    f.write(f"- **Similar Inputs**: High correlation ({pair['input_similarity']:.3f}) ensures fair comparison\n")
                    f.write(f"- **Sufficient Time Gap**: {pair['time_gap']} cycles allows degradation to manifest\n")
                    f.write(f"- **Clear Degradation**: {pair['degradation_pct']:.1f}% change is easily observable\n")
                    f.write(f"- **Balanced Score**: Optimal trade-off between all criteria\n\n")
            
            else:
                f.write("No pairs met the selection criteria. Consider relaxing the thresholds.\n\n")
            
            f.write("## Methodology Benefits\n\n")
            f.write("1. **Avoids Trivial Comparisons**: Minimum time gap prevents comparing adjacent cycles\n")
            f.write("2. **Ensures Fair Comparison**: Similarity threshold maintains input consistency\n")
            f.write("3. **Focuses on Significant Changes**: Degradation threshold filters meaningful differences\n")
            f.write("4. **Balanced Selection**: Composite scoring prevents over-optimization of single criteria\n\n")
            
            f.write("## Recommendations\n\n")
            f.write("- Use top-ranked pairs for detailed degradation analysis\n")
            f.write("- Focus on pairs with high similarity (>0.8) for most reliable comparisons\n")
            f.write("- Consider pairs with larger time gaps for long-term degradation studies\n")
            f.write("- Validate findings across multiple optimal pairs to ensure robustness\n\n")
            
            f.write(f"---\n")
            f.write(f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        return report_path

def main():
    """Main execution function"""
    print("🚀 Starting Optimal Degradation Comparison Analysis")
    print("=" * 60)
    
    # Data path configuration
    data_path = Path("data/raw/ES12.mat")
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    # Initialize visualizer
    visualizer = OptimalDegradationVisualizer()
    
    try:
        # Configuration
        target_capacitor = "ES12C4"
        max_cycles = 100
        
        print(f"🎯 Analysis Target: {target_capacitor}")
        print(f"🔄 Max cycles to analyze: {max_cycles}")
        
        # Load and analyze cycles
        analysis_data = visualizer.load_and_analyze_cycles(data_path, target_capacitor, max_cycles)
        
        # Find optimal pairs
        optimal_pairs = visualizer.find_optimal_pairs(
            analysis_data,
            min_similarity=0.7,    # Minimum input similarity
            min_time_gap=10,       # Minimum cycles apart
            min_degradation=20.0   # Minimum degradation percentage
        )
        
        if not optimal_pairs:
            print("❌ No optimal pairs found with current criteria")
            print("💡 Try relaxing the criteria (lower similarity, smaller time gap, or less degradation)")
            return
        
        # Generate visualizations
        pairs_plot = visualizer.visualize_optimal_pairs(analysis_data, optimal_pairs, top_n=6)
        criteria_plot = visualizer.create_selection_criteria_analysis(analysis_data, optimal_pairs)
        
        # Generate report
        report_path = visualizer.generate_optimal_pairs_report(analysis_data, optimal_pairs)
        
        # Summary
        print(f"\n" + "=" * 60)
        print("✅ Optimal Degradation Comparison Analysis Complete!")
        
        print(f"\n📊 Key Results:")
        print(f"   Found {len(optimal_pairs)} optimal pairs")
        
        if optimal_pairs:
            top_pair = optimal_pairs[0]
            print(f"   Best pair: Cycles {top_pair['cycle1']}-{top_pair['cycle2']}")
            print(f"   - Time gap: {top_pair['time_gap']} cycles")
            print(f"   - Similarity: {top_pair['input_similarity']:.3f}")
            print(f"   - Degradation: {top_pair['degradation_pct']:.1f}%")
            print(f"   - Score: {top_pair['composite_score']:.3f}")
        
        print(f"\n📁 Generated Files:")
        print(f"   - {pairs_plot.name}")
        if criteria_plot:
            print(f"   - {criteria_plot.name}")
        print(f"   - {report_path.name}")
        
        print(f"\n📍 Output Directory: {visualizer.output_dir}")
        
        print(f"\n💡 Analysis Benefits:")
        print(f"   - Balances similarity, time separation, and degradation significance")
        print(f"   - Avoids trivial comparisons between adjacent cycles")
        print(f"   - Focuses on meaningful degradation patterns")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()