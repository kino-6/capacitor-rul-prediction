#!/usr/bin/env python3
"""
Find Cycles with Dynamic Input Patterns

This script searches for cycles with more dynamic VL input patterns
(larger variations, more interesting waveforms) rather than nearly-constant inputs.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
from scipy.stats import pearsonr
from scipy.fft import fft, fftfreq

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

def analyze_waveform_dynamics(data: np.ndarray) -> Dict:
    """Analyze how dynamic/interesting a waveform is"""
    
    # 1. Variation metrics
    std = np.std(data)
    range_val = np.max(data) - np.min(data)
    mean = np.mean(data)
    
    # Coefficient of variation (normalized variability)
    cv = std / abs(mean) if mean != 0 else 0
    
    # 2. Rate of change (how much it changes over time)
    diff = np.diff(data)
    mean_abs_change = np.mean(np.abs(diff))
    max_change = np.max(np.abs(diff))
    
    # 3. Number of direction changes (peaks and valleys)
    sign_changes = np.sum(np.diff(np.sign(diff)) != 0)
    
    # 4. FFT analysis for periodicity
    n = len(data)
    fft_vals = fft(data)
    fft_mag = np.abs(fft_vals[:n//2])
    
    # Exclude DC component
    if len(fft_mag) > 1:
        fft_mag[0] = 0
        max_freq_power = np.max(fft_mag)
        total_power = np.sum(fft_mag)
        periodicity_ratio = max_freq_power / total_power if total_power > 0 else 0
    else:
        periodicity_ratio = 0
    
    # 5. Composite "dynamism" score
    # Higher score = more dynamic/interesting waveform
    dynamism_score = (
        cv * 0.3 +                              # Normalized variation
        (mean_abs_change / abs(mean)) * 0.3 +   # Rate of change
        (sign_changes / n) * 0.2 +              # Complexity
        periodicity_ratio * 0.2                 # Periodicity
    )
    
    return {
        'std': std,
        'range': range_val,
        'mean': mean,
        'cv': cv,
        'mean_abs_change': mean_abs_change,
        'max_change': max_change,
        'sign_changes': sign_changes,
        'periodicity_ratio': periodicity_ratio,
        'dynamism_score': dynamism_score
    }

def calculate_comprehensive_similarity(data1: np.ndarray, data2: np.ndarray, 
                                     stats1: Dict, stats2: Dict) -> Dict:
    """Calculate comprehensive similarity including shape, amplitude, and offset"""
    
    # 1. Shape similarity (correlation)
    try:
        shape_similarity, _ = pearsonr(data1, data2)
        if np.isnan(shape_similarity):
            shape_similarity = 0.0
    except:
        shape_similarity = 0.0
    
    # 2. Amplitude similarity
    std1, std2 = stats1['vl_std'], stats2['vl_std']
    range1, range2 = stats1['vl_range'], stats2['vl_range']
    
    if std1 > 0 and std2 > 0:
        std_similarity = 1 - abs(std1 - std2) / max(std1, std2)
    else:
        std_similarity = 0.0
        
    if range1 > 0 and range2 > 0:
        range_similarity = 1 - abs(range1 - range2) / max(range1, range2)
    else:
        range_similarity = 0.0
        
    amplitude_similarity = (std_similarity + range_similarity) / 2
    
    # 3. Offset similarity
    mean1, mean2 = stats1['vl_mean'], stats2['vl_mean']
    max_mean = max(abs(mean1), abs(mean2))
    if max_mean > 0:
        mean_similarity = 1 - abs(mean1 - mean2) / max_mean
    else:
        mean_similarity = 1.0
        
    offset_similarity = mean_similarity
    
    # 4. Composite similarity
    composite_similarity = (
        shape_similarity * 0.5 +
        amplitude_similarity * 0.3 +
        offset_similarity * 0.2
    )
    
    return {
        'shape_similarity': shape_similarity,
        'amplitude_similarity': amplitude_similarity,
        'offset_similarity': offset_similarity,
        'composite_similarity': composite_similarity,
        'mean_diff': abs(mean1 - mean2),
        'std_diff': abs(std1 - std2),
        'range_diff': abs(range1 - range2)
    }

def main():
    """Main execution function"""
    print("🚀 Finding Cycles with Dynamic Input Patterns")
    print("=" * 70)
    
    # Data path configuration
    data_path = Path("data/raw/ES12.mat")
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    # Output directory
    output_dir = Path("output/dynamic_input_cycles")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set font to avoid rendering issues
    plt.rcParams['font.family'] = ['DejaVu Sans']
    
    try:
        # Configuration
        target_capacitor = "ES12C4"
        max_cycles = 400  # Analyze all available cycles
        
        print(f"🎯 Analysis Target: {target_capacitor}")
        print(f"🔄 Max cycles to analyze: {max_cycles}")
        
        # Load and analyze cycles
        print(f"📊 Loading and analyzing cycles...")
        
        with h5py.File(data_path, 'r') as f:
            cap_group = f['ES12']['Transient_Data'][target_capacitor]
            vl_data = cap_group['VL'][:]
            vo_data = cap_group['VO'][:]
            
            print(f"✅ Raw data loaded: VL {vl_data.shape}, VO {vo_data.shape}")
            
            # Process cycles
            target_length = 3000
            cycles_data = {}
            valid_cycles = []
            vl_matrix = []
            dynamics_list = []
            
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
                
                # Analyze waveform dynamics
                dynamics = analyze_waveform_dynamics(vl_clean)
                
                cycles_data[cycle_num] = {
                    'vl_raw': vl_clean,
                    'vo_raw': vo_clean,
                    'vl_mean': np.mean(vl_clean),
                    'vo_mean': np.mean(vo_clean),
                    'vl_std': np.std(vl_clean),
                    'vo_std': np.std(vo_clean),
                    'vl_range': np.max(vl_clean) - np.min(vl_clean),
                    'vo_range': np.max(vo_clean) - np.min(vo_clean),
                    'voltage_ratio': np.mean(vo_clean) / np.mean(vl_clean) if np.mean(vl_clean) != 0 else np.nan,
                    'dynamics': dynamics
                }
                
                vl_matrix.append(vl_clean)
                valid_cycles.append(cycle_num)
                dynamics_list.append({
                    'cycle': cycle_num,
                    **dynamics
                })
            
            vl_matrix = np.array(vl_matrix)
            print(f"✅ Processed {len(valid_cycles)} valid cycles")
        
        # Sort cycles by dynamism score
        dynamics_df = pd.DataFrame(dynamics_list)
        dynamics_df = dynamics_df.sort_values('dynamism_score', ascending=False)
        
        print(f"\n📊 Top 10 Most Dynamic Cycles:")
        print(dynamics_df.head(10).to_string(index=False))
        
        # Find pairs with dynamic inputs and large time gaps
        print(f"\n🔍 Finding dynamic cycle pairs with large time gaps...")
        
        # Get top 20 most dynamic cycles
        top_dynamic_cycles = dynamics_df.head(20)['cycle'].tolist()
        
        print(f"   Top dynamic cycles: {top_dynamic_cycles[:10]}...")
        
        # Find pairs among dynamic cycles
        dynamic_pairs = []
        
        for i, cycle1 in enumerate(top_dynamic_cycles):
            for j, cycle2 in enumerate(top_dynamic_cycles):
                if i >= j:
                    continue
                
                time_gap = abs(cycle2 - cycle1)
                
                # Require large time gap (at least 50 cycles)
                if time_gap < 50:
                    continue
                
                # Get cycle data
                data1 = cycles_data[cycle1]
                data2 = cycles_data[cycle2]
                
                # Calculate similarity
                idx1 = valid_cycles.index(cycle1)
                idx2 = valid_cycles.index(cycle2)
                
                similarity_metrics = calculate_comprehensive_similarity(
                    vl_matrix[idx1], vl_matrix[idx2], data1, data2
                )
                
                # Calculate degradation
                ratio1 = data1['voltage_ratio']
                ratio2 = data2['voltage_ratio']
                
                if not np.isnan(ratio1) and not np.isnan(ratio2) and ratio1 != 0:
                    degradation_pct = abs((ratio2 - ratio1) / ratio1) * 100
                else:
                    degradation_pct = 0
                
                # Require reasonable similarity and significant degradation
                if (similarity_metrics['composite_similarity'] >= 0.7 and
                    degradation_pct >= 50.0):
                    
                    avg_dynamism = (data1['dynamics']['dynamism_score'] + 
                                  data2['dynamics']['dynamism_score']) / 2
                    
                    # Score: prioritize dynamism and time gap
                    score = (
                        avg_dynamism * 0.4 +
                        min(time_gap / 100.0, 1.0) * 0.3 +
                        similarity_metrics['composite_similarity'] * 0.2 +
                        min(degradation_pct / 200.0, 1.0) * 0.1
                    )
                    
                    dynamic_pairs.append({
                        'cycle1': cycle1,
                        'cycle2': cycle2,
                        'time_gap': time_gap,
                        'similarity': similarity_metrics['composite_similarity'],
                        'degradation_pct': degradation_pct,
                        'avg_dynamism': avg_dynamism,
                        'dynamism1': data1['dynamics']['dynamism_score'],
                        'dynamism2': data2['dynamics']['dynamism_score'],
                        'score': score,
                        'ratio1': ratio1,
                        'ratio2': ratio2
                    })
        
        # Sort by score
        dynamic_pairs.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"\n✅ Found {len(dynamic_pairs)} dynamic cycle pairs")
        
        if dynamic_pairs:
            print(f"\n📊 Top 10 Dynamic Pairs:")
            for i, pair in enumerate(dynamic_pairs[:10], 1):
                print(f"   #{i}: Cycles {pair['cycle1']}-{pair['cycle2']} "
                      f"(gap:{pair['time_gap']}, "
                      f"sim:{pair['similarity']:.3f}, "
                      f"dyn:{pair['avg_dynamism']:.3f}, "
                      f"deg:{pair['degradation_pct']:.1f}%)")
        
        # Generate report
        report_path = output_dir / 'ES12C4_dynamic_cycles_analysis.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# ES12C4 Dynamic Input Cycles Analysis\n\n")
            
            f.write("## Objective\n\n")
            f.write("Find cycles with more dynamic VL input patterns (not nearly-constant) ")
            f.write("to better observe input-output response relationships and degradation.\n\n")
            
            f.write("## Dynamism Metrics\n\n")
            f.write("- **Coefficient of Variation**: Normalized variability\n")
            f.write("- **Rate of Change**: How quickly the signal changes\n")
            f.write("- **Sign Changes**: Number of peaks and valleys (complexity)\n")
            f.write("- **Periodicity Ratio**: Presence of periodic patterns\n")
            f.write("- **Dynamism Score**: Weighted combination of above metrics\n\n")
            
            f.write("## Top 20 Most Dynamic Cycles\n\n")
            f.write("| Rank | Cycle | Dynamism | CV | Mean Change | Sign Changes | Periodicity |\n")
            f.write("|------|-------|----------|----|--------------|--------------|--------------|\n")
            
            for i, row in dynamics_df.head(20).iterrows():
                f.write(f"| {i+1} | {int(row['cycle'])} | {row['dynamism_score']:.4f} | "
                       f"{row['cv']:.4f} | {row['mean_abs_change']:.4f} | "
                       f"{int(row['sign_changes'])} | {row['periodicity_ratio']:.4f} |\n")
            
            f.write("\n## Dynamic Cycle Pairs for Degradation Analysis\n\n")
            
            if dynamic_pairs:
                f.write("### Selection Criteria\n")
                f.write("- Both cycles have high dynamism scores\n")
                f.write("- Time gap ≥ 50 cycles (to observe significant degradation)\n")
                f.write("- Composite similarity ≥ 0.7\n")
                f.write("- Degradation ≥ 50%\n\n")
                
                f.write("### Top 10 Recommended Pairs\n\n")
                f.write("| Rank | Cycle Pair | Time Gap | Similarity | Avg Dynamism | Degradation |\n")
                f.write("|------|------------|----------|------------|--------------|-------------|\n")
                
                for i, pair in enumerate(dynamic_pairs[:10], 1):
                    f.write(f"| {i} | {pair['cycle1']}-{pair['cycle2']} | "
                           f"{pair['time_gap']} | {pair['similarity']:.3f} | "
                           f"{pair['avg_dynamism']:.3f} | {pair['degradation_pct']:.1f}% |\n")
                
                f.write("\n### Detailed Analysis of Top 3 Pairs\n\n")
                
                for i, pair in enumerate(dynamic_pairs[:3], 1):
                    f.write(f"#### Pair {i}: Cycles {pair['cycle1']} → {pair['cycle2']}\n\n")
                    f.write(f"- **Time Gap**: {pair['time_gap']} cycles\n")
                    f.write(f"- **Input Similarity**: {pair['similarity']:.3f}\n")
                    f.write(f"- **Cycle {pair['cycle1']} Dynamism**: {pair['dynamism1']:.3f}\n")
                    f.write(f"- **Cycle {pair['cycle2']} Dynamism**: {pair['dynamism2']:.3f}\n")
                    f.write(f"- **Degradation**: {pair['degradation_pct']:.1f}%\n")
                    f.write(f"- **Early Ratio**: {pair['ratio1']:.2f}\n")
                    f.write(f"- **Late Ratio**: {pair['ratio2']:.2f}\n\n")
            else:
                f.write("No pairs found meeting the criteria.\n\n")
            
            f.write("## Conclusion\n\n")
            
            if dynamic_pairs:
                f.write("Successfully identified cycle pairs with dynamic input patterns ")
                f.write("that show clear input variations (not nearly-constant) and ")
                f.write("significant degradation over large time gaps.\n\n")
            else:
                f.write("The ES12 dataset appears to contain primarily steady-state ")
                f.write("or nearly-constant input patterns. Dynamic input-output response ")
                f.write("analysis may be limited with this dataset.\n\n")
            
            f.write(f"Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Summary
        print(f"\n" + "=" * 70)
        print("✅ Dynamic Input Cycles Analysis Complete!")
        
        print(f"\n📊 Key Results:")
        print(f"   Total cycles analyzed: {len(valid_cycles)}")
        print(f"   Dynamic pairs found: {len(dynamic_pairs)}")
        
        if dynamic_pairs:
            print(f"\n🎯 Top Recommendation:")
            top_pair = dynamic_pairs[0]
            print(f"   Cycles {top_pair['cycle1']} → {top_pair['cycle2']}")
            print(f"   Time gap: {top_pair['time_gap']} cycles")
            print(f"   Dynamism: {top_pair['avg_dynamism']:.3f}")
            print(f"   Degradation: {top_pair['degradation_pct']:.1f}%")
        
        print(f"\n📁 Generated Files:")
        print(f"   - {report_path.name}")
        
        print(f"\n📍 Output Directory: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
