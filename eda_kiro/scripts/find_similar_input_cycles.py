#!/usr/bin/env python3
"""
Find Similar Input Cycles for Degradation Analysis

This script calculates similarity between input waveforms (VL) across different cycles
to identify groups of cycles with similar input patterns. This enables proper
"similar input, different output" degradation analysis without data modification.

Approach:
1. Load raw VL (input) waveforms for all cycles
2. Calculate similarity metrics between cycles (correlation, DTW, etc.)
3. Identify clusters of cycles with similar input patterns
4. Compare VO (output) responses within similar input clusters
5. Visualize degradation patterns for similar inputs across time
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
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import itertools

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class SimilarInputCycleFinder:
    """Find cycles with similar input patterns for degradation analysis"""
    
    def __init__(self, output_dir: Path = Path("output/similar_input_cycles")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set font to avoid rendering issues
        plt.rcParams['font.family'] = ['DejaVu Sans']
        
        # Color palette
        self.colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    def load_all_cycles(self, data_path: Path, capacitor_id: str, max_cycles: int = 100) -> Dict:
        """Load VL and VO data for all cycles of a capacitor"""
        print(f"📊 Loading all cycles for {capacitor_id} (max: {max_cycles})")
        
        try:
            with h5py.File(data_path, 'r') as f:
                # Navigate to transient data
                transient_group = f['ES12']['Transient_Data']
                cap_group = transient_group[capacitor_id]
                vl_data = cap_group['VL'][:]  # Shape: (time_points, cycles)
                vo_data = cap_group['VO'][:]  # Shape: (time_points, cycles)
                
                print(f"✅ Raw data loaded: VL shape {vl_data.shape}, VO shape {vo_data.shape}")
                
                # Limit number of cycles for computational efficiency
                n_cycles = min(max_cycles, vl_data.shape[1])
                vl_data = vl_data[:, :n_cycles]
                vo_data = vo_data[:, :n_cycles]
                
                cycles_data = {}
                valid_cycles = []
                
                for cycle_idx in range(n_cycles):
                    cycle_num = cycle_idx + 1  # 1-based indexing
                    
                    vl_cycle = vl_data[:, cycle_idx]
                    vo_cycle = vo_data[:, cycle_idx]
                    
                    # Remove NaN values
                    valid_mask = ~np.isnan(vl_cycle) & ~np.isnan(vo_cycle)
                    
                    if np.sum(valid_mask) < 1000:  # Need sufficient data points
                        continue
                    
                    vl_clean = vl_cycle[valid_mask]
                    vo_clean = vo_cycle[valid_mask]
                    
                    # Standardize length by taking first N points or padding
                    target_length = 5000  # Standard length for comparison
                    if len(vl_clean) >= target_length:
                        vl_std = vl_clean[:target_length]
                        vo_std = vo_clean[:target_length]
                    else:
                        # Skip cycles that are too short
                        continue
                    
                    cycles_data[cycle_num] = {
                        'vl_raw': vl_std,
                        'vo_raw': vo_std,
                        'vl_stats': {
                            'mean': np.mean(vl_std),
                            'std': np.std(vl_std),
                            'min': np.min(vl_std),
                            'max': np.max(vl_std)
                        },
                        'vo_stats': {
                            'mean': np.mean(vo_std),
                            'std': np.std(vo_std),
                            'min': np.min(vo_std),
                            'max': np.max(vo_std)
                        }
                    }
                    
                    valid_cycles.append(cycle_num)
                
                print(f"✅ Processed {len(valid_cycles)} valid cycles: {valid_cycles[:10]}{'...' if len(valid_cycles) > 10 else ''}")
                
                return {
                    'capacitor_id': capacitor_id,
                    'cycles_data': cycles_data,
                    'valid_cycles': valid_cycles,
                    'target_length': target_length
                }
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def calculate_input_similarity(self, all_cycles_data: Dict) -> Dict:
        """Calculate similarity metrics between input waveforms"""
        print(f"🔧 Calculating input waveform similarity...")
        
        cycles_data = all_cycles_data['cycles_data']
        valid_cycles = all_cycles_data['valid_cycles']
        
        # Prepare data matrices
        n_cycles = len(valid_cycles)
        vl_matrix = np.zeros((n_cycles, all_cycles_data['target_length']))
        
        cycle_to_idx = {cycle: idx for idx, cycle in enumerate(valid_cycles)}
        
        for idx, cycle_num in enumerate(valid_cycles):
            vl_matrix[idx, :] = cycles_data[cycle_num]['vl_raw']
        
        print(f"📊 Similarity matrix shape: {vl_matrix.shape}")
        
        # Calculate different similarity metrics
        similarity_metrics = {}
        
        # 1. Pearson correlation
        print("   Computing Pearson correlations...")
        corr_matrix = np.corrcoef(vl_matrix)
        similarity_metrics['correlation'] = corr_matrix
        
        # 2. Cosine similarity
        print("   Computing cosine similarities...")
        cosine_matrix = cosine_similarity(vl_matrix)
        similarity_metrics['cosine'] = cosine_matrix
        
        # 3. Normalized Euclidean distance (converted to similarity)
        print("   Computing Euclidean distances...")
        # Normalize each waveform
        scaler = StandardScaler()
        vl_normalized = scaler.fit_transform(vl_matrix.T).T
        
        # Calculate pairwise distances
        distances = pdist(vl_normalized, metric='euclidean')
        distance_matrix = squareform(distances)
        
        # Convert distance to similarity (higher = more similar)
        max_distance = np.max(distance_matrix)
        euclidean_similarity = 1 - (distance_matrix / max_distance)
        similarity_metrics['euclidean'] = euclidean_similarity
        
        # 4. Statistical similarity (based on mean and std)
        print("   Computing statistical similarities...")
        stat_similarity = np.zeros((n_cycles, n_cycles))
        
        for i, cycle_i in enumerate(valid_cycles):
            for j, cycle_j in enumerate(valid_cycles):
                if i == j:
                    stat_similarity[i, j] = 1.0
                else:
                    stats_i = cycles_data[cycle_i]['vl_stats']
                    stats_j = cycles_data[cycle_j]['vl_stats']
                    
                    # Similarity based on mean and std differences
                    mean_diff = abs(stats_i['mean'] - stats_j['mean'])
                    std_diff = abs(stats_i['std'] - stats_j['std'])
                    
                    # Normalize by the larger value to get relative difference
                    mean_sim = 1 - mean_diff / (max(abs(stats_i['mean']), abs(stats_j['mean'])) + 1e-10)
                    std_sim = 1 - std_diff / (max(stats_i['std'], stats_j['std']) + 1e-10)
                    
                    # Combined similarity
                    stat_similarity[i, j] = (mean_sim + std_sim) / 2
        
        similarity_metrics['statistical'] = stat_similarity
        
        return {
            'similarity_metrics': similarity_metrics,
            'valid_cycles': valid_cycles,
            'cycle_to_idx': cycle_to_idx,
            'vl_matrix': vl_matrix
        }
    
    def find_similar_clusters(self, similarity_data: Dict, similarity_threshold: float = 0.8) -> Dict:
        """Find clusters of cycles with similar input patterns"""
        print(f"🔍 Finding similar input clusters (threshold: {similarity_threshold})")
        
        similarity_metrics = similarity_data['similarity_metrics']
        valid_cycles = similarity_data['valid_cycles']
        
        clusters_by_method = {}
        
        for method_name, similarity_matrix in similarity_metrics.items():
            print(f"   Processing {method_name} similarity...")
            
            # Convert similarity to distance for clustering
            distance_matrix = 1 - similarity_matrix
            
            # Perform hierarchical clustering
            # Convert to condensed distance matrix for linkage
            condensed_distances = squareform(distance_matrix, checks=False)
            
            # Perform linkage
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # Form clusters based on distance threshold
            distance_threshold = 1 - similarity_threshold
            cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')
            
            # Group cycles by cluster
            clusters = {}
            for cycle_idx, cluster_id in enumerate(cluster_labels):
                cycle_num = valid_cycles[cycle_idx]
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(cycle_num)
            
            # Filter clusters with at least 3 cycles
            filtered_clusters = {cid: cycles for cid, cycles in clusters.items() if len(cycles) >= 3}
            
            clusters_by_method[method_name] = {
                'clusters': filtered_clusters,
                'linkage_matrix': linkage_matrix,
                'cluster_labels': cluster_labels,
                'n_clusters': len(filtered_clusters)
            }
            
            print(f"      Found {len(filtered_clusters)} clusters with ≥3 cycles")
            for cid, cycles in filtered_clusters.items():
                print(f"         Cluster {cid}: {len(cycles)} cycles {cycles[:5]}{'...' if len(cycles) > 5 else ''}")
        
        return clusters_by_method
    
    def analyze_degradation_within_clusters(self, all_cycles_data: Dict, clusters_data: Dict) -> Dict:
        """Analyze degradation patterns within similar input clusters"""
        print(f"📈 Analyzing degradation within similar input clusters...")
        
        cycles_data = all_cycles_data['cycles_data']
        degradation_analysis = {}
        
        # Use correlation-based clusters as primary method
        primary_method = 'correlation'
        if primary_method not in clusters_data:
            primary_method = list(clusters_data.keys())[0]
        
        clusters = clusters_data[primary_method]['clusters']
        
        for cluster_id, cycle_list in clusters.items():
            print(f"   Analyzing Cluster {cluster_id} ({len(cycle_list)} cycles)...")
            
            # Sort cycles chronologically
            sorted_cycles = sorted(cycle_list)
            
            # Calculate degradation metrics
            cluster_analysis = {
                'cycles': sorted_cycles,
                'n_cycles': len(sorted_cycles),
                'time_span': sorted_cycles[-1] - sorted_cycles[0],
                'degradation_metrics': {}
            }
            
            # Reference cycle (earliest in cluster)
            ref_cycle = sorted_cycles[0]
            ref_data = cycles_data[ref_cycle]
            
            # Calculate degradation for each cycle relative to reference
            for cycle_num in sorted_cycles:
                cycle_data = cycles_data[cycle_num]
                
                # Voltage ratio change
                ref_ratio = ref_data['vo_stats']['mean'] / ref_data['vl_stats']['mean'] if ref_data['vl_stats']['mean'] != 0 else np.nan
                curr_ratio = cycle_data['vo_stats']['mean'] / cycle_data['vl_stats']['mean'] if cycle_data['vl_stats']['mean'] != 0 else np.nan
                
                ratio_change = ((curr_ratio - ref_ratio) / abs(ref_ratio)) * 100 if not np.isnan(ref_ratio) and ref_ratio != 0 else np.nan
                
                # Output amplitude change
                ref_vo_range = ref_data['vo_stats']['max'] - ref_data['vo_stats']['min']
                curr_vo_range = cycle_data['vo_stats']['max'] - cycle_data['vo_stats']['min']
                
                amplitude_change = ((curr_vo_range - ref_vo_range) / ref_vo_range) * 100 if ref_vo_range != 0 else np.nan
                
                # Response correlation (between VL and VO within cycle)
                try:
                    vl_cycle = cycle_data['vl_raw']
                    vo_cycle = cycle_data['vo_raw']
                    correlation, _ = pearsonr(vl_cycle, vo_cycle)
                except:
                    correlation = np.nan
                
                cluster_analysis['degradation_metrics'][cycle_num] = {
                    'ratio_change_pct': ratio_change,
                    'amplitude_change_pct': amplitude_change,
                    'vl_vo_correlation': correlation,
                    'vo_mean': cycle_data['vo_stats']['mean'],
                    'vl_mean': cycle_data['vl_stats']['mean'],
                    'voltage_ratio': curr_ratio
                }
            
            # Calculate cluster-level degradation trend
            ratio_changes = [cluster_analysis['degradation_metrics'][c]['ratio_change_pct'] 
                           for c in sorted_cycles if not np.isnan(cluster_analysis['degradation_metrics'][c]['ratio_change_pct'])]
            
            if len(ratio_changes) > 1:
                # Linear trend in degradation
                x = np.arange(len(ratio_changes))
                coeffs = np.polyfit(x, ratio_changes, 1)
                degradation_rate = coeffs[0]  # slope
                
                cluster_analysis['degradation_rate_pct_per_cycle'] = degradation_rate
                cluster_analysis['total_degradation_pct'] = ratio_changes[-1] - ratio_changes[0] if ratio_changes else 0
            else:
                cluster_analysis['degradation_rate_pct_per_cycle'] = np.nan
                cluster_analysis['total_degradation_pct'] = np.nan
            
            degradation_analysis[cluster_id] = cluster_analysis
        
        return degradation_analysis
    
    def visualize_similar_clusters(self, all_cycles_data: Dict, similarity_data: Dict, clusters_data: Dict) -> Path:
        """Visualize similar input clusters and their characteristics"""
        capacitor_id = all_cycles_data['capacitor_id']
        
        print(f"📊 Creating similar clusters visualization for {capacitor_id}...")
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'{capacitor_id} Similar Input Cycles Analysis', fontsize=16, fontweight='bold')
        
        # Use correlation method for primary analysis
        primary_method = 'correlation'
        if primary_method not in clusters_data:
            primary_method = list(clusters_data.keys())[0]
        
        similarity_matrix = similarity_data['similarity_metrics'][primary_method]
        clusters = clusters_data[primary_method]['clusters']
        valid_cycles = similarity_data['valid_cycles']
        
        # 1. Similarity heatmap
        ax1 = axes[0, 0]
        im = ax1.imshow(similarity_matrix, cmap='viridis', aspect='auto')
        ax1.set_title(f'Input Similarity Matrix ({primary_method.title()})', fontweight='bold')
        ax1.set_xlabel('Cycle Index')
        ax1.set_ylabel('Cycle Index')
        
        # Add cycle numbers as ticks (sample every 10th)
        tick_indices = range(0, len(valid_cycles), max(1, len(valid_cycles)//10))
        tick_labels = [str(valid_cycles[i]) for i in tick_indices]
        ax1.set_xticks(tick_indices)
        ax1.set_xticklabels(tick_labels, rotation=45)
        ax1.set_yticks(tick_indices)
        ax1.set_yticklabels(tick_labels)
        
        plt.colorbar(im, ax=ax1, label='Similarity')
        
        # 2. Cluster distribution
        ax2 = axes[0, 1]
        
        cluster_sizes = [len(cycles) for cycles in clusters.values()]
        cluster_ids = list(clusters.keys())
        
        bars = ax2.bar(range(len(cluster_ids)), cluster_sizes, color=self.colors[:len(cluster_ids)])
        ax2.set_title('Cluster Sizes', fontweight='bold')
        ax2.set_xlabel('Cluster ID')
        ax2.set_ylabel('Number of Cycles')
        ax2.set_xticks(range(len(cluster_ids)))
        ax2.set_xticklabels([f'C{cid}' for cid in cluster_ids])
        
        # Add value labels on bars
        for bar, size in zip(bars, cluster_sizes):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(size), ha='center', va='bottom')
        
        # 3. Sample waveforms from largest cluster
        ax3 = axes[0, 2]
        
        if clusters:
            # Find largest cluster
            largest_cluster_id = max(clusters.keys(), key=lambda x: len(clusters[x]))
            largest_cluster_cycles = clusters[largest_cluster_id][:5]  # Show first 5 cycles
            
            cycles_data = all_cycles_data['cycles_data']
            
            for i, cycle_num in enumerate(largest_cluster_cycles):
                vl_data = cycles_data[cycle_num]['vl_raw']
                # Subsample for visualization
                subsample = max(1, len(vl_data) // 1000)
                time_points = np.arange(0, len(vl_data), subsample)
                vl_sub = vl_data[::subsample]
                
                ax3.plot(time_points, vl_sub, alpha=0.7, linewidth=1, 
                        label=f'Cycle {cycle_num}', color=self.colors[i])
            
            ax3.set_title(f'Sample Input Waveforms (Cluster {largest_cluster_id})', fontweight='bold')
            ax3.set_xlabel('Time Points')
            ax3.set_ylabel('VL (Input Voltage)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Similarity distribution
        ax4 = axes[1, 0]
        
        # Get upper triangle of similarity matrix (excluding diagonal)
        upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
        
        ax4.hist(upper_triangle, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.axvline(0.8, color='red', linestyle='--', label='Similarity Threshold (0.8)')
        ax4.set_title('Similarity Score Distribution', fontweight='bold')
        ax4.set_xlabel('Similarity Score')
        ax4.set_ylabel('Frequency')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Cluster timeline
        ax5 = axes[1, 1]
        
        colors_cycle = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
        
        for i, (cluster_id, cycle_list) in enumerate(clusters.items()):
            y_pos = [i] * len(cycle_list)
            ax5.scatter(cycle_list, y_pos, color=colors_cycle[i], s=50, alpha=0.7, 
                       label=f'Cluster {cluster_id}')
        
        ax5.set_title('Cluster Timeline', fontweight='bold')
        ax5.set_xlabel('Cycle Number')
        ax5.set_ylabel('Cluster ID')
        ax5.set_yticks(range(len(clusters)))
        ax5.set_yticklabels([f'C{cid}' for cid in clusters.keys()])
        ax5.grid(True, alpha=0.3)
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 6. Summary statistics table
        ax6 = axes[1, 2]
        
        # Create summary table
        summary_data = []
        for cluster_id, cycle_list in clusters.items():
            summary_data.append([
                f'Cluster {cluster_id}',
                len(cycle_list),
                min(cycle_list),
                max(cycle_list),
                max(cycle_list) - min(cycle_list)
            ])
        
        if summary_data:
            table = ax6.table(cellText=summary_data,
                             colLabels=['Cluster', 'Size', 'Min Cycle', 'Max Cycle', 'Span'],
                             cellLoc='center',
                             loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
        
        ax6.set_title('Cluster Summary', fontweight='bold')
        ax6.axis('off')
        
        plt.tight_layout()
        
        plot_path = self.output_dir / f'{capacitor_id}_similar_clusters.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def visualize_degradation_analysis(self, all_cycles_data: Dict, degradation_analysis: Dict) -> Path:
        """Visualize degradation patterns within similar input clusters"""
        capacitor_id = all_cycles_data['capacitor_id']
        
        print(f"📈 Creating degradation analysis visualization for {capacitor_id}...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{capacitor_id} Degradation Analysis - Similar Input Clusters', 
                     fontsize=16, fontweight='bold')
        
        # 1. Voltage ratio degradation by cluster
        ax1 = axes[0, 0]
        
        colors_cycle = plt.cm.tab10(np.linspace(0, 1, len(degradation_analysis)))
        
        for i, (cluster_id, cluster_data) in enumerate(degradation_analysis.items()):
            cycles = cluster_data['cycles']
            ratio_changes = [cluster_data['degradation_metrics'][c]['ratio_change_pct'] 
                           for c in cycles if not np.isnan(cluster_data['degradation_metrics'][c]['ratio_change_pct'])]
            
            if ratio_changes:
                valid_cycles = [c for c in cycles if not np.isnan(cluster_data['degradation_metrics'][c]['ratio_change_pct'])]
                ax1.plot(valid_cycles, ratio_changes, 'o-', color=colors_cycle[i], 
                        label=f'Cluster {cluster_id} ({len(cycles)} cycles)', linewidth=2, markersize=6)
        
        ax1.set_title('Voltage Ratio Degradation by Cluster', fontweight='bold')
        ax1.set_xlabel('Cycle Number')
        ax1.set_ylabel('Voltage Ratio Change (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(0, color='black', linestyle='-', alpha=0.5)
        
        # 2. Degradation rate comparison
        ax2 = axes[0, 1]
        
        cluster_ids = list(degradation_analysis.keys())
        degradation_rates = [degradation_analysis[cid]['degradation_rate_pct_per_cycle'] 
                           for cid in cluster_ids]
        
        # Filter out NaN values
        valid_data = [(cid, rate) for cid, rate in zip(cluster_ids, degradation_rates) if not np.isnan(rate)]
        
        if valid_data:
            valid_cluster_ids, valid_rates = zip(*valid_data)
            bars = ax2.bar(range(len(valid_cluster_ids)), valid_rates, 
                          color=colors_cycle[:len(valid_cluster_ids)])
            
            ax2.set_title('Degradation Rate by Cluster', fontweight='bold')
            ax2.set_xlabel('Cluster ID')
            ax2.set_ylabel('Degradation Rate (% per cycle)')
            ax2.set_xticks(range(len(valid_cluster_ids)))
            ax2.set_xticklabels([f'C{cid}' for cid in valid_cluster_ids])
            ax2.grid(True, alpha=0.3)
            ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
            
            # Add value labels on bars
            for bar, rate in zip(bars, valid_rates):
                ax2.text(bar.get_x() + bar.get_width()/2, 
                        bar.get_height() + (0.01 if rate >= 0 else -0.01), 
                        f'{rate:.3f}', ha='center', va='bottom' if rate >= 0 else 'top')
        
        # 3. Input-Output correlation by cluster
        ax3 = axes[1, 0]
        
        for i, (cluster_id, cluster_data) in enumerate(degradation_analysis.items()):
            cycles = cluster_data['cycles']
            correlations = [cluster_data['degradation_metrics'][c]['vl_vo_correlation'] 
                          for c in cycles if not np.isnan(cluster_data['degradation_metrics'][c]['vl_vo_correlation'])]
            
            if correlations:
                valid_cycles = [c for c in cycles if not np.isnan(cluster_data['degradation_metrics'][c]['vl_vo_correlation'])]
                ax3.plot(valid_cycles, correlations, 'o-', color=colors_cycle[i], 
                        label=f'Cluster {cluster_id}', linewidth=2, markersize=6)
        
        ax3.set_title('Input-Output Correlation by Cluster', fontweight='bold')
        ax3.set_xlabel('Cycle Number')
        ax3.set_ylabel('VL-VO Correlation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1.1)
        
        # 4. Summary statistics table
        ax4 = axes[1, 1]
        
        # Create summary table
        summary_data = []
        for cluster_id, cluster_data in degradation_analysis.items():
            n_cycles = cluster_data['n_cycles']
            time_span = cluster_data['time_span']
            degradation_rate = cluster_data['degradation_rate_pct_per_cycle']
            total_degradation = cluster_data['total_degradation_pct']
            
            summary_data.append([
                f'Cluster {cluster_id}',
                n_cycles,
                time_span,
                f'{degradation_rate:.3f}' if not np.isnan(degradation_rate) else 'N/A',
                f'{total_degradation:.1f}%' if not np.isnan(total_degradation) else 'N/A'
            ])
        
        if summary_data:
            table = ax4.table(cellText=summary_data,
                             colLabels=['Cluster', 'Cycles', 'Span', 'Rate (%/cycle)', 'Total (%)'],
                             cellLoc='center',
                             loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
        
        ax4.set_title('Degradation Summary by Cluster', fontweight='bold')
        ax4.axis('off')
        
        plt.tight_layout()
        
        plot_path = self.output_dir / f'{capacitor_id}_degradation_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def generate_similarity_report(self, all_cycles_data: Dict, similarity_data: Dict, 
                                 clusters_data: Dict, degradation_analysis: Dict) -> Path:
        """Generate comprehensive similarity analysis report"""
        capacitor_id = all_cycles_data['capacitor_id']
        
        print(f"📄 Generating similarity analysis report...")
        
        report_path = self.output_dir / f'{capacitor_id}_similarity_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# {capacitor_id} Similar Input Cycles Analysis Report\n\n")
            
            f.write("## Overview\n\n")
            f.write(f"Analysis of {capacitor_id} to identify cycles with similar input patterns (VL)\n")
            f.write("and analyze degradation within these similar input groups.\n\n")
            
            f.write("## Methodology\n\n")
            f.write("1. **Input Similarity Calculation**: Computed multiple similarity metrics between VL waveforms\n")
            f.write("2. **Clustering**: Grouped cycles with similar input patterns using hierarchical clustering\n")
            f.write("3. **Degradation Analysis**: Analyzed output (VO) changes within similar input clusters\n")
            f.write("4. **Temporal Analysis**: Tracked degradation patterns over time\n\n")
            
            # Data summary
            valid_cycles = similarity_data['valid_cycles']
            f.write("## Data Summary\n\n")
            f.write(f"- **Total Valid Cycles**: {len(valid_cycles)}\n")
            f.write(f"- **Cycle Range**: {min(valid_cycles)} to {max(valid_cycles)}\n")
            f.write(f"- **Waveform Length**: {all_cycles_data['target_length']} points per cycle\n\n")
            
            # Similarity metrics summary
            f.write("## Similarity Metrics\n\n")
            similarity_metrics = similarity_data['similarity_metrics']
            
            for method_name, similarity_matrix in similarity_metrics.items():
                upper_triangle = similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]
                mean_sim = np.mean(upper_triangle)
                std_sim = np.std(upper_triangle)
                
                f.write(f"### {method_name.title()} Similarity\n\n")
                f.write(f"- **Mean Similarity**: {mean_sim:.3f}\n")
                f.write(f"- **Std Deviation**: {std_sim:.3f}\n")
                f.write(f"- **Range**: {np.min(upper_triangle):.3f} to {np.max(upper_triangle):.3f}\n\n")
            
            # Clustering results
            f.write("## Clustering Results\n\n")
            
            primary_method = 'correlation'
            if primary_method not in clusters_data:
                primary_method = list(clusters_data.keys())[0]
            
            clusters = clusters_data[primary_method]['clusters']
            
            f.write(f"Using **{primary_method}** similarity for primary analysis:\n\n")
            f.write(f"- **Number of Clusters**: {len(clusters)}\n")
            f.write(f"- **Similarity Threshold**: 0.8\n")
            f.write(f"- **Minimum Cluster Size**: 3 cycles\n\n")
            
            f.write("### Cluster Details\n\n")
            f.write("| Cluster ID | Size | Cycles | Time Span |\n")
            f.write("|------------|------|--------|----------|\n")
            
            for cluster_id, cycle_list in clusters.items():
                cycles_str = ', '.join(map(str, sorted(cycle_list)[:5]))
                if len(cycle_list) > 5:
                    cycles_str += f", ... (+{len(cycle_list)-5} more)"
                time_span = max(cycle_list) - min(cycle_list)
                
                f.write(f"| {cluster_id} | {len(cycle_list)} | {cycles_str} | {time_span} |\n")
            
            f.write("\n")
            
            # Degradation analysis results
            f.write("## Degradation Analysis Results\n\n")
            
            f.write("Analysis of output voltage (VO) changes within similar input clusters:\n\n")
            
            for cluster_id, cluster_data in degradation_analysis.items():
                f.write(f"### Cluster {cluster_id}\n\n")
                
                n_cycles = cluster_data['n_cycles']
                time_span = cluster_data['time_span']
                degradation_rate = cluster_data['degradation_rate_pct_per_cycle']
                total_degradation = cluster_data['total_degradation_pct']
                
                f.write(f"- **Cycles**: {n_cycles} cycles over {time_span} cycle span\n")
                f.write(f"- **Degradation Rate**: {degradation_rate:.3f}% per cycle\n" if not np.isnan(degradation_rate) else "- **Degradation Rate**: N/A\n")
                f.write(f"- **Total Degradation**: {total_degradation:.1f}%\n" if not np.isnan(total_degradation) else "- **Total Degradation**: N/A\n")
                
                # Severity assessment
                if not np.isnan(total_degradation):
                    if abs(total_degradation) > 50:
                        f.write("- 🔴 **Severe degradation detected**\n")
                    elif abs(total_degradation) > 20:
                        f.write("- 🟡 **Moderate degradation detected**\n")
                    else:
                        f.write("- 🟢 **Stable response**\n")
                
                f.write("\n")
            
            # Key findings
            f.write("## Key Findings\n\n")
            
            # Find most degraded cluster
            worst_cluster = None
            worst_degradation = 0
            
            for cluster_id, cluster_data in degradation_analysis.items():
                total_deg = cluster_data['total_degradation_pct']
                if not np.isnan(total_deg) and abs(total_deg) > abs(worst_degradation):
                    worst_degradation = total_deg
                    worst_cluster = cluster_id
            
            if worst_cluster is not None:
                f.write(f"1. **Most Degraded Cluster**: Cluster {worst_cluster} with {worst_degradation:.1f}% total degradation\n")
            
            # Count clusters with significant degradation
            significant_clusters = sum(1 for cluster_data in degradation_analysis.values() 
                                     if not np.isnan(cluster_data['total_degradation_pct']) and 
                                     abs(cluster_data['total_degradation_pct']) > 20)
            
            f.write(f"2. **Clusters with Significant Degradation**: {significant_clusters} out of {len(degradation_analysis)}\n")
            
            # Average degradation rate
            valid_rates = [cluster_data['degradation_rate_pct_per_cycle'] 
                          for cluster_data in degradation_analysis.values() 
                          if not np.isnan(cluster_data['degradation_rate_pct_per_cycle'])]
            
            if valid_rates:
                avg_rate = np.mean(valid_rates)
                f.write(f"3. **Average Degradation Rate**: {avg_rate:.3f}% per cycle\n")
            
            f.write("\n")
            
            # Conclusions
            f.write("## Conclusions\n\n")
            f.write("1. **Similar Input Identification**: Successfully identified clusters of cycles with similar VL patterns\n")
            f.write("2. **Degradation Quantification**: Measured output changes within similar input groups\n")
            f.write("3. **Temporal Patterns**: Observed degradation progression over time\n")
            f.write("4. **Cluster Variability**: Different clusters show varying degradation rates\n\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            f.write("- Focus degradation analysis on clusters with sufficient temporal span\n")
            f.write("- Use correlation-based similarity for robust input pattern matching\n")
            f.write("- Monitor voltage ratio changes as primary degradation indicator\n")
            f.write("- Consider cluster-specific degradation models for prediction\n\n")
            
            f.write(f"---\n")
            f.write(f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        return report_path

def main():
    """Main execution function"""
    print("🚀 Starting Similar Input Cycles Analysis")
    print("=" * 60)
    
    # Data path configuration
    data_path = Path("data/raw/ES12.mat")
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    # Initialize analyzer
    analyzer = SimilarInputCycleFinder()
    
    try:
        # Configuration
        target_capacitor = "ES12C4"  # Most degraded capacitor from previous analysis
        max_cycles = 100  # Limit for computational efficiency
        
        print(f"🎯 Analysis Target: {target_capacitor}")
        print(f"🔄 Max cycles to analyze: {max_cycles}")
        
        # Load all cycles
        all_cycles_data = analyzer.load_all_cycles(data_path, target_capacitor, max_cycles)
        
        if all_cycles_data is None:
            print("❌ Failed to load data")
            return
        
        # Calculate input similarity
        similarity_data = analyzer.calculate_input_similarity(all_cycles_data)
        
        # Find similar clusters
        clusters_data = analyzer.find_similar_clusters(similarity_data, similarity_threshold=0.8)
        
        # Analyze degradation within clusters
        degradation_analysis = analyzer.analyze_degradation_within_clusters(all_cycles_data, clusters_data)
        
        # Generate visualizations
        clusters_plot = analyzer.visualize_similar_clusters(all_cycles_data, similarity_data, clusters_data)
        degradation_plot = analyzer.visualize_degradation_analysis(all_cycles_data, degradation_analysis)
        
        # Generate report
        report_path = analyzer.generate_similarity_report(all_cycles_data, similarity_data, 
                                                        clusters_data, degradation_analysis)
        
        # Summary
        print(f"\n" + "=" * 60)
        print("✅ Similar Input Cycles Analysis Complete!")
        
        print(f"\n📊 Key Results:")
        
        # Use primary clustering method
        primary_method = 'correlation'
        if primary_method not in clusters_data:
            primary_method = list(clusters_data.keys())[0]
        
        clusters = clusters_data[primary_method]['clusters']
        print(f"   Found {len(clusters)} clusters with similar input patterns")
        
        for cluster_id, cycle_list in clusters.items():
            cluster_data = degradation_analysis[cluster_id]
            total_deg = cluster_data['total_degradation_pct']
            print(f"   Cluster {cluster_id}: {len(cycle_list)} cycles, {total_deg:.1f}% degradation" if not np.isnan(total_deg) else f"   Cluster {cluster_id}: {len(cycle_list)} cycles, N/A degradation")
        
        print(f"\n📁 Generated Files:")
        print(f"   - {clusters_plot.name}")
        print(f"   - {degradation_plot.name}")
        print(f"   - {report_path.name}")
        
        print(f"\n📍 Output Directory: {analyzer.output_dir}")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()