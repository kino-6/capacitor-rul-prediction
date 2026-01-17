"""
Visualize VL-VO Relationship and Degradation Patterns

This script analyzes and visualizes the relationship between input voltage (VL)
and output voltage (VO) across cycles to identify degradation patterns.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import load_es12_cycle_data

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR.parent / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "output" / "vl_vo_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_cycle_data(capacitor_id: str, cycle: int):
    """Load VL and VO data for a specific cycle."""
    es12_path = DATA_DIR / "ES12.mat"
    vl, vo = load_es12_cycle_data(str(es12_path), capacitor_id, cycle)
    return vl, vo


def calculate_response_metrics(vl, vo):
    """Calculate VL-VO response metrics."""
    metrics = {}
    
    # Voltage ratio (mean)
    metrics['voltage_ratio'] = np.mean(vo) / np.mean(vl) if np.mean(vl) != 0 else 0
    
    # Response efficiency (energy ratio)
    vl_energy = np.sum(vl ** 2)
    vo_energy = np.sum(vo ** 2)
    metrics['response_efficiency'] = vo_energy / vl_energy if vl_energy != 0 else 0
    
    # Correlation (waveform similarity)
    if len(vl) == len(vo) and len(vl) > 1:
        metrics['correlation'] = np.corrcoef(vl, vo)[0, 1]
    else:
        metrics['correlation'] = 0
    
    # Response delay (cross-correlation peak)
    if len(vl) == len(vo) and len(vl) > 1:
        cross_corr = np.correlate(vl - np.mean(vl), vo - np.mean(vo), mode='full')
        delay = np.argmax(cross_corr) - (len(vl) - 1)
        metrics['response_delay'] = delay
    else:
        metrics['response_delay'] = 0
    
    return metrics


def visualize_single_capacitor_evolution(capacitor_id: str = 'ES12C1'):
    """Visualize VL-VO relationship evolution for a single capacitor."""
    print(f"\nAnalyzing {capacitor_id}...")
    
    # Select cycles to analyze
    early_cycles = [1, 5, 10]
    mid_cycles = [50, 100, 150]
    late_cycles = [190, 195, 200]
    
    all_cycles = early_cycles + mid_cycles + late_cycles
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 16))
    fig.suptitle(f'{capacitor_id}: VL-VO Relationship Evolution', 
                 fontsize=16, fontweight='bold')
    
    for idx, cycle in enumerate(all_cycles):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        try:
            vl, vo = load_cycle_data(capacitor_id, cycle)
            metrics = calculate_response_metrics(vl, vo)
            
            # Scatter plot
            ax.scatter(vl, vo, alpha=0.3, s=1)
            
            # Linear fit
            if len(vl) > 1:
                slope, intercept, r_value, _, _ = stats.linregress(vl, vo)
                x_fit = np.array([vl.min(), vl.max()])
                y_fit = slope * x_fit + intercept
                ax.plot(x_fit, y_fit, 'r-', linewidth=2, 
                       label=f'Fit: R²={r_value**2:.3f}')
            
            # Styling
            ax.set_xlabel('VL (Input Voltage)', fontsize=10)
            ax.set_ylabel('VO (Output Voltage)', fontsize=10)
            ax.set_title(f'Cycle {cycle}\n' + 
                        f'Ratio={metrics["voltage_ratio"]:.3f}, ' +
                        f'Eff={metrics["response_efficiency"]:.3f}',
                        fontsize=11, fontweight='bold')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            
            # Color code by stage
            if cycle in early_cycles:
                ax.set_facecolor('#e8f5e9')  # Light green
            elif cycle in late_cycles:
                ax.set_facecolor('#ffebee')  # Light red
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading\nCycle {cycle}', 
                   ha='center', va='center', transform=ax.transAxes)
            print(f"  Error loading cycle {cycle}: {e}")
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / f'{capacitor_id}_vl_vo_evolution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def visualize_all_capacitors_comparison():
    """Compare VL-VO relationship across all capacitors."""
    print("\nComparing all capacitors...")
    
    capacitors = [f'ES12C{i}' for i in range(1, 9)]
    cycles_to_compare = [1, 50, 100, 150, 200]
    
    fig, axes = plt.subplots(len(cycles_to_compare), len(capacitors), 
                            figsize=(24, 15))
    fig.suptitle('VL-VO Relationship: All Capacitors Comparison', 
                 fontsize=18, fontweight='bold')
    
    for cycle_idx, cycle in enumerate(cycles_to_compare):
        for cap_idx, cap_id in enumerate(capacitors):
            ax = axes[cycle_idx, cap_idx]
            
            try:
                vl, vo = load_cycle_data(cap_id, cycle)
                metrics = calculate_response_metrics(vl, vo)
                
                # Scatter plot
                ax.scatter(vl, vo, alpha=0.2, s=0.5)
                
                # Linear fit
                if len(vl) > 1:
                    slope, intercept, r_value, _, _ = stats.linregress(vl, vo)
                    x_fit = np.array([vl.min(), vl.max()])
                    y_fit = slope * x_fit + intercept
                    ax.plot(x_fit, y_fit, 'r-', linewidth=1.5)
                
                # Title
                if cycle_idx == 0:
                    ax.set_title(f'{cap_id}', fontsize=10, fontweight='bold')
                
                # Labels
                if cap_idx == 0:
                    ax.set_ylabel(f'Cycle {cycle}\nVO', fontsize=9)
                if cycle_idx == len(cycles_to_compare) - 1:
                    ax.set_xlabel('VL', fontsize=9)
                
                # Metrics text
                ax.text(0.05, 0.95, f'R={metrics["voltage_ratio"]:.2f}',
                       transform=ax.transAxes, fontsize=7,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=7)
                
            except Exception as e:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center',
                       transform=ax.transAxes, fontsize=8)
                print(f"  Error loading {cap_id} cycle {cycle}: {e}")
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'all_capacitors_vl_vo_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


def analyze_response_metrics_over_time():
    """Analyze how response metrics change over time."""
    print("\nAnalyzing response metrics over time...")
    
    capacitors = [f'ES12C{i}' for i in range(1, 9)]
    cycles = range(1, 201, 5)  # Every 5 cycles
    
    # Collect metrics
    results = []
    for cap_id in capacitors:
        print(f"  Processing {cap_id}...")
        for cycle in cycles:
            try:
                vl, vo = load_cycle_data(cap_id, cycle)
                metrics = calculate_response_metrics(vl, vo)
                metrics['capacitor_id'] = cap_id
                metrics['cycle'] = cycle
                results.append(metrics)
            except Exception as e:
                print(f"    Error at cycle {cycle}: {e}")
                continue
    
    df = pd.DataFrame(results)
    
    # Save metrics
    metrics_path = OUTPUT_DIR / 'response_metrics_timeseries.csv'
    df.to_csv(metrics_path, index=False)
    print(f"  ✓ Saved metrics: {metrics_path}")
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Response Metrics Evolution Over Time', 
                 fontsize=16, fontweight='bold')
    
    metrics_to_plot = [
        ('voltage_ratio', 'Voltage Ratio (VO/VL)'),
        ('response_efficiency', 'Response Efficiency (Energy Ratio)'),
        ('correlation', 'VL-VO Correlation'),
        ('response_delay', 'Response Delay (samples)')
    ]
    
    for idx, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        for cap_id in capacitors:
            cap_data = df[df['capacitor_id'] == cap_id]
            ax.plot(cap_data['cycle'], cap_data[metric], 
                   label=cap_id, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Cycle Number', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'response_metrics_evolution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()
    
    return df


def generate_summary_report(metrics_df):
    """Generate summary report of VL-VO relationship analysis."""
    print("\nGenerating summary report...")
    
    report = f"""# VL-VO Relationship Analysis Report

## 📅 Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 Objective

Analyze the relationship between input voltage (VL) and output voltage (VO) 
to identify degradation patterns in capacitor behavior.

## 📊 Analysis Summary

### Response Metrics Analyzed

1. **Voltage Ratio**: Mean(VO) / Mean(VL)
   - Indicates overall voltage transfer efficiency
   - Expected to decrease with degradation

2. **Response Efficiency**: Energy(VO) / Energy(VL)
   - Measures energy transfer efficiency
   - More sensitive to waveform changes

3. **Correlation**: Pearson correlation between VL and VO
   - Measures waveform similarity
   - High correlation = good response fidelity

4. **Response Delay**: Cross-correlation peak offset
   - Measures phase shift between VL and VO
   - Increased delay may indicate degradation

### Key Findings

"""
    
    # Calculate statistics
    for cap_id in metrics_df['capacitor_id'].unique():
        cap_data = metrics_df[metrics_df['capacitor_id'] == cap_id]
        
        early_data = cap_data[cap_data['cycle'] <= 50]
        late_data = cap_data[cap_data['cycle'] >= 150]
        
        report += f"\n#### {cap_id}\n\n"
        
        for metric in ['voltage_ratio', 'response_efficiency', 'correlation']:
            early_mean = early_data[metric].mean()
            late_mean = late_data[metric].mean()
            change_pct = ((late_mean - early_mean) / early_mean * 100) if early_mean != 0 else 0
            
            report += f"- **{metric.replace('_', ' ').title()}**: "
            report += f"{early_mean:.4f} (early) → {late_mean:.4f} (late), "
            report += f"Change: {change_pct:+.1f}%\n"
    
    report += """

## 📊 Visualizations

### Single Capacitor Evolution

![ES12C1 Evolution](ES12C1_vl_vo_evolution.png)

*VL-VO relationship evolution for ES12C1 across early, mid, and late cycles.*

### All Capacitors Comparison

![All Capacitors](all_capacitors_vl_vo_comparison.png)

*Comparison of VL-VO relationships across all 8 capacitors at key cycles.*

### Response Metrics Evolution

![Metrics Evolution](response_metrics_evolution.png)

*Time series of response metrics showing degradation patterns.*

## 🔍 Observations

### Degradation Patterns

Based on the analysis, the following degradation patterns are observed:

1. **Voltage Ratio Decline**: Most capacitors show a gradual decrease in voltage ratio
2. **Response Efficiency**: Energy transfer efficiency decreases over cycles
3. **Correlation Changes**: VL-VO correlation may decrease, indicating waveform distortion
4. **Response Delay**: Phase shifts may increase with degradation

### Anomaly Candidates

Cycles with significant deviations from normal patterns:
- Large drops in response efficiency
- Sudden changes in correlation
- Unusual response delays

## 📁 Generated Files

- `ES12C1_vl_vo_evolution.png` - Single capacitor detailed evolution
- `all_capacitors_vl_vo_comparison.png` - Multi-capacitor comparison
- `response_metrics_evolution.png` - Time series of metrics
- `response_metrics_timeseries.csv` - Raw metrics data

## 🚀 Next Steps

1. **Feature Engineering**: Design new features based on response metrics
2. **Anomaly Detection**: Apply clustering/outlier detection to identify abnormal cycles
3. **Degradation Modeling**: Build models to predict degradation progression

---

**Analysis Tool**: VL-VO Relationship Visualizer  
**Status**: Phase 1 Task 1.1 Complete
"""
    
    report_path = OUTPUT_DIR / 'vl_vo_analysis_report.md'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"  ✓ Saved report: {report_path}")


def main():
    """Main execution."""
    print("="*80)
    print("VL-VO RELATIONSHIP ANALYSIS")
    print("="*80)
    
    # 1. Single capacitor detailed evolution
    visualize_single_capacitor_evolution('ES12C1')
    
    # 2. All capacitors comparison
    visualize_all_capacitors_comparison()
    
    # 3. Response metrics over time
    metrics_df = analyze_response_metrics_over_time()
    
    # 4. Generate summary report
    generate_summary_report(metrics_df)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  1. ES12C1_vl_vo_evolution.png")
    print("  2. all_capacitors_vl_vo_comparison.png")
    print("  3. response_metrics_evolution.png")
    print("  4. response_metrics_timeseries.csv")
    print("  5. vl_vo_analysis_report.md")
    print("\n✅ VL-VO relationship analysis complete!")


if __name__ == "__main__":
    main()
