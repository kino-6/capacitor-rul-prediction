#!/usr/bin/env python3
"""
Actual Waveform Visualization Script

Visualizes the real input-output waveforms from ES12 dataset,
showing the actual time-series response of capacitors.
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

class ActualWaveformVisualizer:
    """Actual waveform visualization class"""
    
    def __init__(self, output_dir: Path = Path("output/actual_waveforms")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set font to avoid rendering issues
        plt.rcParams['font.family'] = ['DejaVu Sans']
        
        # Color palette
        self.colors = plt.cm.viridis(np.linspace(0, 1, 10))
        
    def load_actual_waveform_data(self, data_path: Path, capacitor_id: str, cycle_number: int) -> Dict:
        """Load actual waveform data for specific capacitor and cycle"""
        print(f"📊 Loading actual waveform data for {capacitor_id}, Cycle {cycle_number}...")
        
        try:
            with h5py.File(data_path, 'r') as f:
                # Navigate to transient data
                transient_group = f['ES12']['Transient_Data']
                
                # Get capacitor data
                cap_group = transient_group[capacitor_id]
                vl_data = cap_group['VL'][:]  # Shape: (time_points, cycles)
                vo_data = cap_group['VO'][:]  # Shape: (time_points, cycles)
                
                print(f"✅ Raw data loaded:")
                print(f"   - VL data shape: {vl_data.shape}")
                print(f"   - VO data shape: {vo_data.shape}")
                
                # Check if requested cycle exists
                if cycle_number > vl_data.shape[1]:
                    available_cycles = vl_data.shape[1]
                    print(f"⚠️  Requested cycle {cycle_number} not available. Max cycles: {available_cycles}")
                    cycle_number = min(cycle_number, available_cycles)
                    print(f"   Using cycle {cycle_number} instead.")
                
                # Extract specific cycle (0-based indexing)
                cycle_idx = cycle_number - 1
                vl_cycle = vl_data[:, cycle_idx]
                vo_cycle = vo_data[:, cycle_idx]
                
                # Remove NaN values and create corresponding time indices
                vl_valid_mask = ~np.isnan(vl_cycle)
                vo_valid_mask = ~np.isnan(vo_cycle)
                
                # Use intersection of valid points
                valid_mask = vl_valid_mask & vo_valid_mask
                valid_indices = np.where(valid_mask)[0]
                
                if len(valid_indices) == 0:
                    print(f"❌ No valid data points found for cycle {cycle_number}")
                    return None
                
                # Extract valid data
                vl_clean = vl_cycle[valid_mask]
                vo_clean = vo_cycle[valid_mask]
                
                # Create time axis (assuming uniform sampling within valid data)
                # Since we don't have actual time stamps for each point, create relative time
                time_points = np.arange(len(vl_clean))
                
                # Estimate sampling rate (this is approximate)
                # Assuming the measurement covers a reasonable time period
                estimated_duration = 1.0  # Assume 1 second per cycle (this is an estimate)
                sampling_rate = len(vl_clean) / estimated_duration
                time_seconds = time_points / sampling_rate
                
                print(f"✅ Processed data:")
                print(f"   - Valid data points: {len(vl_clean)}")
                print(f"   - Estimated sampling rate: {sampling_rate:.1f} Hz")
                print(f"   - Estimated duration: {estimated_duration:.3f} seconds")
                
                return {
                    'capacitor_id': capacitor_id,
                    'cycle_number': cycle_number,
                    'time_seconds': time_seconds,
                    'vl_waveform': vl_clean,
                    'vo_waveform': vo_clean,
                    'sampling_rate': sampling_rate,
                    'duration_seconds': estimated_duration,
                    'valid_points': len(vl_clean),
                    'total_points': len(vl_cycle)
                }
                
        except Exception as e:
            print(f"❌ Error loading actual data: {e}")
            return None
    
    def analyze_actual_waveform(self, waveform_data: Dict) -> Dict:
        """Analyze characteristics of the actual waveforms"""
        print(f"🔬 Analyzing actual waveform characteristics...")
        
        time_seconds = waveform_data['time_seconds']
        vl_waveform = waveform_data['vl_waveform']
        vo_waveform = waveform_data['vo_waveform']
        
        # Basic statistics
        vl_stats = {
            'mean': np.mean(vl_waveform),
            'std': np.std(vl_waveform),
            'min': np.min(vl_waveform),
            'max': np.max(vl_waveform),
            'peak_to_peak': np.max(vl_waveform) - np.min(vl_waveform)
        }
        
        vo_stats = {
            'mean': np.mean(vo_waveform),
            'std': np.std(vo_waveform),
            'min': np.min(vo_waveform),
            'max': np.max(vo_waveform),
            'peak_to_peak': np.max(vo_waveform) - np.min(vo_waveform)
        }
        
        # Calculate voltage ratio
        voltage_ratio = vo_stats['mean'] / vl_stats['mean'] if vl_stats['mean'] != 0 else np.nan
        
        # Simple correlation analysis
        correlation = np.corrcoef(vl_waveform, vo_waveform)[0, 1]
        
        # Find peaks and valleys for pattern analysis
        vl_peaks = []
        vo_peaks = []
        
        # Simple peak detection (local maxima)
        for i in range(1, len(vl_waveform) - 1):
            if vl_waveform[i] > vl_waveform[i-1] and vl_waveform[i] > vl_waveform[i+1]:
                vl_peaks.append(i)
            if vo_waveform[i] > vo_waveform[i-1] and vo_waveform[i] > vo_waveform[i+1]:
                vo_peaks.append(i)
        
        return {
            'vl_stats': vl_stats,
            'vo_stats': vo_stats,
            'voltage_ratio': voltage_ratio,
            'correlation': correlation,
            'vl_peaks': vl_peaks,
            'vo_peaks': vo_peaks,
            'num_vl_peaks': len(vl_peaks),
            'num_vo_peaks': len(vo_peaks)
        }
    
    def visualize_actual_waveform(self, waveform_data: Dict, analysis: Dict) -> Path:
        """Visualize actual waveform with detailed analysis"""
        capacitor_id = waveform_data['capacitor_id']
        cycle_number = waveform_data['cycle_number']
        
        print(f"📈 Visualizing actual waveform for {capacitor_id}, Cycle {cycle_number}...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{capacitor_id} - Cycle {cycle_number} Actual Input-Output Waveforms', 
                     fontsize=16, fontweight='bold')
        
        time_seconds = waveform_data['time_seconds']
        vl_waveform = waveform_data['vl_waveform']
        vo_waveform = waveform_data['vo_waveform']
        
        # 1. Full waveforms
        ax1 = axes[0, 0]
        ax1.plot(time_seconds, vl_waveform, 'b-', linewidth=1, alpha=0.8, label='VL (Input)')
        ax1.plot(time_seconds, vo_waveform, 'r-', linewidth=1, alpha=0.8, label='VO (Output)')
        
        ax1.set_title('Complete Waveforms', fontweight='bold')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Voltage (V)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add statistics text
        vl_stats = analysis['vl_stats']
        vo_stats = analysis['vo_stats']
        stats_text = f"VL: μ={vl_stats['mean']:.4f}V, σ={vl_stats['std']:.4f}V\n"
        stats_text += f"VO: μ={vo_stats['mean']:.4f}V, σ={vo_stats['std']:.4f}V\n"
        stats_text += f"Ratio: {analysis['voltage_ratio']:.3f}\n"
        stats_text += f"Correlation: {analysis['correlation']:.3f}"
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 2. Zoomed view (first 1000 points or 10% of data)
        ax2 = axes[0, 1]
        zoom_end = min(1000, len(time_seconds) // 10)
        if zoom_end < 100:
            zoom_end = min(100, len(time_seconds))
        
        time_zoom = time_seconds[:zoom_end]
        vl_zoom = vl_waveform[:zoom_end]
        vo_zoom = vo_waveform[:zoom_end]
        
        ax2.plot(time_zoom, vl_zoom, 'b-', linewidth=1.5, alpha=0.8, label='VL (Input)')
        ax2.plot(time_zoom, vo_zoom, 'r-', linewidth=1.5, alpha=0.8, label='VO (Output)')
        
        # Mark peaks in zoomed view
        vl_peaks = analysis['vl_peaks']
        vo_peaks = analysis['vo_peaks']
        
        for peak in vl_peaks:
            if peak < zoom_end:
                ax2.plot(time_zoom[peak], vl_zoom[peak], 'bo', markersize=4, alpha=0.7)
        
        for peak in vo_peaks:
            if peak < zoom_end:
                ax2.plot(time_zoom[peak], vo_zoom[peak], 'ro', markersize=4, alpha=0.7)
        
        ax2.set_title(f'Waveform Detail (First {zoom_end} points)', fontweight='bold')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Voltage (V)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Histogram of voltage values
        ax3 = axes[1, 0]
        
        ax3.hist(vl_waveform, bins=50, alpha=0.6, color='blue', label='VL Distribution', density=True)
        ax3.hist(vo_waveform, bins=50, alpha=0.6, color='red', label='VO Distribution', density=True)
        
        ax3.set_title('Voltage Distribution', fontweight='bold')
        ax3.set_xlabel('Voltage (V)')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Input-Output scatter plot
        ax4 = axes[1, 1]
        
        # Subsample for clearer visualization if too many points
        subsample = max(1, len(vl_waveform) // 2000)
        vl_sub = vl_waveform[::subsample]
        vo_sub = vo_waveform[::subsample]
        
        ax4.scatter(vl_sub, vo_sub, alpha=0.6, s=1, c=self.colors[0])
        
        # Add linear fit
        try:
            coeffs = np.polyfit(vl_sub, vo_sub, 1)
            vl_fit = np.linspace(np.min(vl_sub), np.max(vl_sub), 100)
            vo_fit = np.polyval(coeffs, vl_fit)
            ax4.plot(vl_fit, vo_fit, 'r-', linewidth=2, alpha=0.8, 
                    label=f'Linear Fit (slope={coeffs[0]:.3f})')
            ax4.legend()
        except Exception as e:
            print(f"⚠️  Linear fit failed: {e}")
        
        # Add diagonal line for reference
        min_val = min(np.min(vl_sub), np.min(vo_sub))
        max_val = max(np.max(vl_sub), np.max(vo_sub))
        ax4.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='y=x')
        
        ax4.set_title('Input-Output Relationship', fontweight='bold')
        ax4.set_xlabel('VL (Input Voltage)')
        ax4.set_ylabel('VO (Output Voltage)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / f'{capacitor_id}_cycle_{cycle_number}_actual_waveform.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def generate_actual_waveform_report(self, waveform_data: Dict, analysis: Dict) -> Path:
        """Generate detailed actual waveform analysis report"""
        capacitor_id = waveform_data['capacitor_id']
        cycle_number = waveform_data['cycle_number']
        
        print(f"📄 Generating actual waveform analysis report...")
        
        report_path = self.output_dir / f'{capacitor_id}_cycle_{cycle_number}_actual_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# {capacitor_id} Cycle {cycle_number} Actual Waveform Analysis Report\n\n")
            
            f.write("## Overview\n\n")
            f.write(f"Analysis of actual measured input-output waveforms for {capacitor_id} during cycle {cycle_number}.\n")
            f.write("This report analyzes the real voltage measurements from the ES12 dataset.\n\n")
            
            f.write("## Measurement Details\n\n")
            f.write(f"- **Capacitor**: {capacitor_id}\n")
            f.write(f"- **Cycle Number**: {cycle_number}\n")
            f.write(f"- **Valid Data Points**: {waveform_data['valid_points']}\n")
            f.write(f"- **Total Data Points**: {waveform_data['total_points']}\n")
            f.write(f"- **Data Completeness**: {waveform_data['valid_points']/waveform_data['total_points']*100:.1f}%\n")
            f.write(f"- **Estimated Duration**: {waveform_data['duration_seconds']:.3f} seconds\n")
            f.write(f"- **Estimated Sampling Rate**: {waveform_data['sampling_rate']:.1f} Hz\n\n")
            
            f.write("## Input Signal (VL) Characteristics\n\n")
            vl_stats = analysis['vl_stats']
            f.write(f"- **Mean Voltage**: {vl_stats['mean']:.6f} V\n")
            f.write(f"- **Standard Deviation**: {vl_stats['std']:.6f} V\n")
            f.write(f"- **Minimum**: {vl_stats['min']:.6f} V\n")
            f.write(f"- **Maximum**: {vl_stats['max']:.6f} V\n")
            f.write(f"- **Peak-to-Peak**: {vl_stats['peak_to_peak']:.6f} V\n")
            f.write(f"- **Number of Peaks**: {analysis['num_vl_peaks']}\n\n")
            
            f.write("## Output Signal (VO) Characteristics\n\n")
            vo_stats = analysis['vo_stats']
            f.write(f"- **Mean Voltage**: {vo_stats['mean']:.6f} V\n")
            f.write(f"- **Standard Deviation**: {vo_stats['std']:.6f} V\n")
            f.write(f"- **Minimum**: {vo_stats['min']:.6f} V\n")
            f.write(f"- **Maximum**: {vo_stats['max']:.6f} V\n")
            f.write(f"- **Peak-to-Peak**: {vo_stats['peak_to_peak']:.6f} V\n")
            f.write(f"- **Number of Peaks**: {analysis['num_vo_peaks']}\n\n")
            
            f.write("## Input-Output Relationship\n\n")
            f.write(f"- **Voltage Ratio (VO/VL)**: {analysis['voltage_ratio']:.6f}\n")
            f.write(f"- **Correlation Coefficient**: {analysis['correlation']:.6f}\n")
            
            # Interpret correlation
            if analysis['correlation'] > 0.8:
                correlation_desc = "Strong positive correlation"
            elif analysis['correlation'] > 0.5:
                correlation_desc = "Moderate positive correlation"
            elif analysis['correlation'] > -0.5:
                correlation_desc = "Weak correlation"
            elif analysis['correlation'] > -0.8:
                correlation_desc = "Moderate negative correlation"
            else:
                correlation_desc = "Strong negative correlation"
            
            f.write(f"- **Correlation Description**: {correlation_desc}\n\n")
            
            f.write("## Signal Quality Assessment\n\n")
            
            # Signal-to-noise ratio estimation
            vl_snr = abs(vl_stats['mean']) / vl_stats['std'] if vl_stats['std'] > 0 else float('inf')
            vo_snr = abs(vo_stats['mean']) / vo_stats['std'] if vo_stats['std'] > 0 else float('inf')
            
            f.write(f"- **VL Signal-to-Noise Ratio**: {vl_snr:.2f}\n")
            f.write(f"- **VO Signal-to-Noise Ratio**: {vo_snr:.2f}\n")
            
            # Efficiency assessment
            efficiency = abs(analysis['voltage_ratio']) * 100
            f.write(f"- **Apparent Efficiency**: {efficiency:.2f}%\n\n")
            
            f.write("## Interpretation\n\n")
            
            if abs(analysis['voltage_ratio']) > 1.0:
                f.write("- 🔍 **Amplification**: Output voltage is higher than input voltage\n")
            elif abs(analysis['voltage_ratio']) > 0.8:
                f.write("- 🟢 **High Efficiency**: Output voltage is close to input voltage\n")
            elif abs(analysis['voltage_ratio']) > 0.5:
                f.write("- 🟡 **Moderate Efficiency**: Some voltage drop observed\n")
            else:
                f.write("- 🔴 **Low Efficiency**: Significant voltage drop indicates degradation\n")
            
            if analysis['correlation'] > 0.8:
                f.write("- 🟢 **Good Linearity**: Strong positive correlation between input and output\n")
            elif analysis['correlation'] < -0.5:
                f.write("- 🔴 **Phase Inversion**: Negative correlation may indicate phase issues\n")
            else:
                f.write("- 🟡 **Non-linear Response**: Moderate correlation indicates complex behavior\n")
            
            f.write(f"\n---\n")
            f.write(f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        return report_path

def main():
    """Main execution function"""
    print("🚀 Starting Actual Waveform Analysis")
    print("=" * 60)
    
    # Data path configuration
    data_path = Path("data/raw/ES12.mat")
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    # Initialize visualizer
    visualizer = ActualWaveformVisualizer()
    
    try:
        # Configuration - analyze multiple cycles for comparison
        target_capacitor = "ES12C4"
        cycles_to_analyze = [1, 100, 200, 300]  # Different degradation stages
        
        print(f"🎯 Analysis Target: {target_capacitor}")
        print(f"🔄 Cycles to analyze: {cycles_to_analyze}")
        
        generated_files = []
        
        for cycle_num in cycles_to_analyze:
            print(f"\n{'='*40}")
            print(f"Analyzing Cycle {cycle_num}")
            print(f"{'='*40}")
            
            # Load actual waveform data
            waveform_data = visualizer.load_actual_waveform_data(data_path, target_capacitor, cycle_num)
            
            if waveform_data is None:
                print(f"⚠️  Skipping cycle {cycle_num} due to loading error")
                continue
            
            # Analyze waveform characteristics
            analysis = visualizer.analyze_actual_waveform(waveform_data)
            
            # Generate visualization
            plot_path = visualizer.visualize_actual_waveform(waveform_data, analysis)
            generated_files.append(plot_path)
            
            # Generate report
            report_path = visualizer.generate_actual_waveform_report(waveform_data, analysis)
            generated_files.append(report_path)
            
            # Print summary
            print(f"📊 Cycle {cycle_num} Summary:")
            print(f"   - Valid data points: {waveform_data['valid_points']}")
            print(f"   - Data completeness: {waveform_data['valid_points']/waveform_data['total_points']*100:.1f}%")
            print(f"   - VL mean: {analysis['vl_stats']['mean']:.6f} V")
            print(f"   - VO mean: {analysis['vo_stats']['mean']:.6f} V")
            print(f"   - Voltage ratio: {analysis['voltage_ratio']:.6f}")
            print(f"   - Correlation: {analysis['correlation']:.6f}")
        
        # Final summary
        print(f"\n" + "=" * 60)
        print("✅ Actual Waveform Analysis Complete!")
        
        print(f"\n📁 Generated Files ({len(generated_files)}):")
        for file_path in generated_files:
            print(f"   - {file_path.name}")
        
        print(f"\n📍 Output Directory: {visualizer.output_dir}")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()