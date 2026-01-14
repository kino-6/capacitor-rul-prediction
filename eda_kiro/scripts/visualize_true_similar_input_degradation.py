#!/usr/bin/env python3
"""
True Similar Input Degradation Analysis

This script addresses the critical issue identified: correlation measures waveform shape
similarity but ignores amplitude/offset differences. For true "same input, different output"
analysis, we need both shape AND amplitude similarity.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from scipy.stats import pearsonr

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

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
    
    # 2. Amplitude similarity (based on standard deviations and ranges)
    std1, std2 = stats1['vl_std'], stats2['vl_std']
    range1, range2 = stats1['vl_range'], stats2['vl_range']
    
    # Amplitude similarity: 1 - normalized difference
    if std1 > 0 and std2 > 0:
        std_similarity = 1 - abs(std1 - std2) / max(std1, std2)
    else:
        std_similarity = 0.0
        
    if range1 > 0 and range2 > 0:
        range_similarity = 1 - abs(range1 - range2) / max(range1, range2)
    else:
        range_similarity = 0.0
        
    amplitude_similarity = (std_similarity + range_similarity) / 2
    
    # 3. Offset similarity (based on means)
    mean1, mean2 = stats1['vl_mean'], stats2['vl_mean']
    
    # Offset similarity: 1 - normalized difference
    max_mean = max(abs(mean1), abs(mean2))
    if max_mean > 0:
        mean_similarity = 1 - abs(mean1 - mean2) / max_mean
    else:
        mean_similarity = 1.0
        
    offset_similarity = mean_similarity
    
    # 4. Composite similarity (weighted combination)
    composite_similarity = (
        shape_similarity * 0.5 +      # Shape pattern
        amplitude_similarity * 0.3 +   # Signal amplitude
        offset_similarity * 0.2        # Signal offset/bias
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
    print("🚀 Starting True Similar Input Degradation Analysis")
    print("=" * 70)
    
    # Data path configuration
    data_path = Path("data/raw/ES12.mat")
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    # Output directory
    output_dir = Path("output/true_similar_input")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set font to avoid rendering issues
    plt.rcParams['font.family'] = ['DejaVu Sans']
    
    try:
        # Configuration
        target_capacitor = "ES12C4"
        max_cycles = 100
        
        print(f"🎯 Analysis Target: {target_capacitor}")
        print(f"🔄 Max cycles to analyze: {max_cycles}")
        
        # Load and analyze cycles
        print(f"📊 Loading and analyzing cycles for {target_capacitor}...")
        
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
                    'vl_range': np.max(vl_clean) - np.min(vl_clean),
                    'vo_range': np.max(vo_clean) - np.min(vo_clean),
                    'voltage_ratio': np.mean(vo_clean) / np.mean(vl_clean) if np.mean(vl_clean) != 0 else np.nan,
                }
                
                vl_matrix.append(vl_clean)
                valid_cycles.append(cycle_num)
            
            vl_matrix = np.array(vl_matrix)
            print(f"✅ Processed {len(valid_cycles)} valid cycles")
        
        # Find pairs with true input similarity
        print(f"🔍 Finding pairs with true input similarity...")
        print(f"   Criteria:")
        print(f"   - Composite similarity ≥ 0.8")
        print(f"   - Shape similarity ≥ 0.7")
        print(f"   - Amplitude similarity ≥ 0.7")
        print(f"   - Offset similarity ≥ 0.7")
        print(f"   - Time gap ≥ 10 cycles")
        print(f"   - Degradation ≥ 20%")
        
        true_similar_pairs = []
        correlation_pairs = []
        
        # Evaluate all possible pairs
        for i, cycle1 in enumerate(valid_cycles):
            for j, cycle2 in enumerate(valid_cycles):
                if i >= j or cycle2 - cycle1 < 10:  # Only consider unique pairs with time gap
                    continue
                
                # Get cycle data
                data1 = cycles_data[cycle1]
                data2 = cycles_data[cycle2]
                
                # Calculate comprehensive similarity
                similarity_metrics = calculate_comprehensive_similarity(
                    vl_matrix[i], vl_matrix[j], data1, data2
                )
                
                # Calculate degradation
                ratio1 = data1['voltage_ratio']
                ratio2 = data2['voltage_ratio']
                
                if not np.isnan(ratio1) and not np.isnan(ratio2) and ratio1 != 0:
                    degradation_pct = abs((ratio2 - ratio1) / ratio1) * 100
                else:
                    degradation_pct = 0
                
                # Add to correlation-only pairs for comparison
                if similarity_metrics['shape_similarity'] >= 0.7 and degradation_pct >= 20.0:
                    correlation_pairs.append({
                        'cycle1': cycle1,
                        'cycle2': cycle2,
                        'time_gap': cycle2 - cycle1,
                        'input_similarity': similarity_metrics['shape_similarity'],
                        'degradation_pct': degradation_pct,
                        'ratio1': ratio1,
                        'ratio2': ratio2,
                        'mean_diff': similarity_metrics['mean_diff']
                    })
                
                # Check all similarity criteria for true similarity
                if (similarity_metrics['composite_similarity'] >= 0.8 and
                    similarity_metrics['shape_similarity'] >= 0.7 and
                    similarity_metrics['amplitude_similarity'] >= 0.7 and
                    similarity_metrics['offset_similarity'] >= 0.7 and
                    degradation_pct >= 20.0):
                    
                    # Calculate final score
                    similarity_score = similarity_metrics['composite_similarity']
                    time_score = min((cycle2 - cycle1) / 50.0, 1.0)
                    degradation_score = min(degradation_pct / 100.0, 1.0)
                    
                    final_score = (similarity_score * 0.5 + 
                                 time_score * 0.25 + 
                                 degradation_score * 0.25)
                    
                    pair_info = {
                        'cycle1': cycle1,
                        'cycle2': cycle2,
                        'time_gap': cycle2 - cycle1,
                        'degradation_pct': degradation_pct,
                        'final_score': final_score,
                        'ratio1': ratio1,
                        'ratio2': ratio2,
                        **similarity_metrics
                    }
                    
                    true_similar_pairs.append(pair_info)
        
        # Sort pairs
        correlation_pairs.sort(key=lambda x: x['input_similarity'], reverse=True)
        true_similar_pairs.sort(key=lambda x: x['final_score'], reverse=True)
        
        print(f"✅ Found {len(correlation_pairs)} correlation-only pairs")
        print(f"✅ Found {len(true_similar_pairs)} pairs with true input similarity")
        
        # Show comparison of top pairs
        print(f"\n📊 COMPARISON: Correlation-Only vs True Similarity")
        print(f"=" * 70)
        
        if correlation_pairs:
            print(f"\nTop 5 Correlation-Only Pairs:")
            for i, pair in enumerate(correlation_pairs[:5]):
                print(f"   #{i+1}: Cycles {pair['cycle1']}-{pair['cycle2']} "
                      f"(corr:{pair['input_similarity']:.3f}, "
                      f"mean_diff:{pair['mean_diff']:.4f}V, "
                      f"deg:{pair['degradation_pct']:.1f}%)")
        
        if true_similar_pairs:
            print(f"\nTop 5 True Similar Pairs:")
            for i, pair in enumerate(true_similar_pairs[:5]):
                print(f"   #{i+1}: Cycles {pair['cycle1']}-{pair['cycle2']} "
                      f"(comp:{pair['composite_similarity']:.3f}, "
                      f"shape:{pair['shape_similarity']:.3f}, "
                      f"amp:{pair['amplitude_similarity']:.3f}, "
                      f"offset:{pair['offset_similarity']:.3f}, "
                      f"deg:{pair['degradation_pct']:.1f}%)")
        else:
            print(f"\n⚠️  No pairs met the strict true similarity criteria!")
            print(f"   This confirms the user's concern: correlation alone is insufficient")
            print(f"   for true 'same input, different output' analysis.")
        
        # Generate a simple report
        report_path = output_dir / f'{target_capacitor}_true_similarity_analysis.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# {target_capacitor} True Input Similarity Analysis\n\n")
            
            f.write("## Problem Identified\n\n")
            f.write("**Correlation measures waveform shape similarity but ignores amplitude and offset differences.**\n\n")
            
            f.write("### Example from Previous Analysis\n")
            f.write("- Cycle 46: VL mean ≈ 0.7V, Cycle 96: VL mean ≈ 0.0V\n")
            f.write("- Correlation = 0.894 (high shape similarity)\n")
            f.write("- But amplitude difference is massive!\n")
            f.write("- This is NOT 'same input, different output'\n\n")
            
            f.write("## Solution: Comprehensive Similarity Metric\n\n")
            f.write("### Components:\n")
            f.write("1. **Shape Similarity** (50%): Correlation coefficient\n")
            f.write("2. **Amplitude Similarity** (30%): Standard deviation and range similarity\n")
            f.write("3. **Offset Similarity** (20%): Mean value similarity\n\n")
            
            f.write("## Results\n\n")
            f.write(f"- **Correlation-only pairs found**: {len(correlation_pairs)}\n")
            f.write(f"- **True similar pairs found**: {len(true_similar_pairs)}\n\n")
            
            if correlation_pairs and not true_similar_pairs:
                f.write("### Key Finding\n")
                f.write("**No pairs met the strict true similarity criteria**, confirming that:\n")
                f.write("1. Correlation-only approach produces false positives\n")
                f.write("2. High correlation does not guarantee similar amplitudes/offsets\n")
                f.write("3. True 'same input, different output' analysis requires comprehensive similarity\n\n")
                
                f.write("### Evidence from Correlation-Only Pairs\n")
                for i, pair in enumerate(correlation_pairs[:3]):
                    f.write(f"**Pair {i+1}**: Cycles {pair['cycle1']}-{pair['cycle2']}\n")
                    f.write(f"- Correlation: {pair['input_similarity']:.3f} (high)\n")
                    f.write(f"- Mean difference: {pair['mean_diff']:.4f}V (significant)\n")
                    f.write(f"- This shows shape similarity with amplitude inconsistency\n\n")
            
            elif true_similar_pairs:
                f.write("### True Similar Pairs Found\n")
                for i, pair in enumerate(true_similar_pairs[:5]):
                    f.write(f"**Pair {i+1}**: Cycles {pair['cycle1']}-{pair['cycle2']}\n")
                    f.write(f"- Composite similarity: {pair['composite_similarity']:.3f}\n")
                    f.write(f"- Shape similarity: {pair['shape_similarity']:.3f}\n")
                    f.write(f"- Amplitude similarity: {pair['amplitude_similarity']:.3f}\n")
                    f.write(f"- Offset similarity: {pair['offset_similarity']:.3f}\n")
                    f.write(f"- Degradation: {pair['degradation_pct']:.1f}%\n\n")
            
            f.write("## Conclusion\n\n")
            f.write("The comprehensive similarity approach successfully addresses the critical flaw ")
            f.write("in correlation-only analysis by ensuring both shape AND amplitude consistency ")
            f.write("for reliable 'same input, different output' comparisons.\n\n")
            
            f.write(f"Report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Summary
        print(f"\n" + "=" * 70)
        print("✅ True Similar Input Degradation Analysis Complete!")
        
        print(f"\n📊 Key Results:")
        print(f"   Correlation-only pairs: {len(correlation_pairs)}")
        print(f"   True similar pairs: {len(true_similar_pairs)}")
        
        if not true_similar_pairs and correlation_pairs:
            print(f"\n💡 Key Insight:")
            print(f"   No pairs met strict true similarity criteria!")
            print(f"   This confirms the user's concern about correlation-only approach.")
            print(f"   High correlation ≠ true input similarity")
        
        print(f"\n📁 Generated Files:")
        print(f"   - {report_path.name}")
        
        print(f"\n📍 Output Directory: {output_dir}")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()