#!/usr/bin/env python3
"""
Single Cycle Waveform Visualization Script

Visualizes the actual input-output waveforms for a single measurement cycle
from the ES12 dataset, showing the detailed time-series response of a capacitor
to the applied voltage signal.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import signal
import h5py

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from nasa_pcoe_eda.data.es12_loader import ES12DataLoader

# 警告を抑制
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class SingleCycleWaveformVisualizer:
    """Single cycle waveform visualization class"""
    
    def __init__(self, output_dir: Path = Path("output/single_cycle_waveforms")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set font to avoid rendering issues
        plt.rcParams['font.family'] = ['DejaVu Sans']
        
        # Color palette
        self.colors = plt.cm.viridis(np.linspace(0, 1, 10))
        
    def load_raw_waveform_data(self, data_path: Path, capacitor_id: str, cycle_number: int) -> Dict:
        """Load raw waveform data for specific capacitor and cycle"""
        print(f"📊 Loading raw waveform data for {capacitor_id}, Cycle {cycle_number}...")
        
        try:
            with h5py.File(data_path, 'r') as f:
                # Navigate to transient data
                transient_group = f['ES12']['Transient_Data']
                
                # Get time information
                serial_dates = transient_group['Serial_Date'][:]
                
                # Get capacitor data
                cap_group = transient_group[capacitor_id]
                vl_data = cap_group['VL'][:]  # Shape: (time_points, cycles)
                vo_data = cap_group['VO'][:]  # Shape: (time_points, cycles)
                
                print(f"✅ Raw data loaded:")
                print(f"   - VL data shape: {vl_data.shape}")
                print(f"   - VO data shape: {vo_data.shape}")
                print(f"   - Time points: {len(serial_dates)}")
                
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
                
                # Create time axis (assuming uniform sampling)
                # Convert MATLAB serial dates to relative time in seconds
                time_seconds = (serial_dates - serial_dates[0]) * 24 * 3600  # Convert days to seconds
                
                # Handle potential size mismatch
                min_length = min(len(time_seconds), len(vl_cycle), len(vo_cycle))
                time_seconds = time_seconds[:min_length]
                vl_cycle = vl_cycle[:min_length]
                vo_cycle = vo_cycle[:min_length]
                
                return {
                    'capacitor_id': capacitor_id,
                    'cycle_number': cycle_number,
                    'time_seconds': time_seconds,
                    'vl_waveform': vl_cycle,
                    'vo_waveform': vo_cycle,
                    'sampling_rate': len(time_seconds) / (time_seconds[-1] - time_seconds[0]) if len(time_seconds) > 1 else 1,
                    'duration_seconds': time_seconds[-1] - time_seconds[0] if len(time_seconds) > 1 else 0
                }
                
        except Exception as e:
            print(f"❌ Error loading raw data: {e}")
            return None
    
    def analyze_waveform_characteristics(self, waveform_data: Dict) -> Dict:
        """Analyze characteristics of the waveforms"""
        print(f"🔬 Analyzing waveform characteristics...")
        
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
        
        # Frequency analysis
        try:
            # Compute power spectral density
            sampling_rate = waveform_data['sampling_rate']
            
            vl_freqs, vl_psd = signal.welch(vl_waveform, fs=sampling_rate, nperseg=min(1024, len(vl_waveform)//4))
            vo_freqs, vo_psd = signal.welch(vo_waveform, fs=sampling_rate, nperseg=min(1024, len(vo_waveform)//4))
            
            # Find dominant frequencies
            vl_dominant_freq = vl_freqs[np.argmax(vl_psd[1:])] if len(vl_psd) > 1 else 0
            vo_dominant_freq = vo_freqs[np.argmax(vo_psd[1:])] if len(vo_psd) > 1 else 0
            
        except Exception as e:
            print(f"⚠️  Frequency analysis failed: {e}")
            vl_freqs = vo_freqs = np.array([0])
            vl_psd = vo_psd = np.array([0])
            vl_dominant_freq = vo_dominant_freq = 0
        
        # Phase relationship (cross-correlation)
        try:
            correlation = np.correlate(vl_waveform - np.mean(vl_waveform), 
                                     vo_waveform - np.mean(vo_waveform), mode='full')
            max_corr_idx = np.argmax(np.abs(correlation))
            phase_delay_samples = max_corr_idx - (len(correlation) // 2)
            phase_delay_seconds = phase_delay_samples / sampling_rate if sampling_rate > 0 else 0
        except Exception as e:
            print(f"⚠️  Phase analysis failed: {e}")
            phase_delay_seconds = 0
        
        return {
            'vl_stats': vl_stats,
            'vo_stats': vo_stats,
            'vl_freqs': vl_freqs,
            'vl_psd': vl_psd,
            'vo_freqs': vo_freqs,
            'vo_psd': vo_psd,
            'vl_dominant_freq': vl_dominant_freq,
            'vo_dominant_freq': vo_dominant_freq,
            'phase_delay_seconds': phase_delay_seconds,
            'voltage_ratio': vo_stats['mean'] / vl_stats['mean'] if vl_stats['mean'] != 0 else 0
        }
    
    def visualize_single_cycle_waveform(self, waveform_data: Dict, analysis: Dict) -> Path:
        """Visualize single cycle waveform with detailed analysis"""
        capacitor_id = waveform_data['capacitor_id']
        cycle_number = waveform_data['cycle_number']
        
        print(f"📈 Visualizing single cycle waveform for {capacitor_id}, Cycle {cycle_number}...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{capacitor_id} - Cycle {cycle_number} Input-Output Waveform Analysis', 
                     fontsize=16, fontweight='bold')
        
        time_seconds = waveform_data['time_seconds']
        vl_waveform = waveform_data['vl_waveform']
        vo_waveform = waveform_data['vo_waveform']
        
        # 1. Time domain waveforms
        ax1 = axes[0, 0]
        ax1.plot(time_seconds, vl_waveform, 'b-', linewidth=1.5, alpha=0.8, label='VL (Input)')
        ax1.plot(time_seconds, vo_waveform, 'r-', linewidth=1.5, alpha=0.8, label='VO (Output)')
        
        ax1.set_title('Input-Output Waveforms', fontweight='bold')
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Voltage')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add statistics text
        vl_stats = analysis['vl_stats']
        vo_stats = analysis['vo_stats']
        stats_text = f"VL: μ={vl_stats['mean']:.3f}, σ={vl_stats['std']:.3f}\n"
        stats_text += f"VO: μ={vo_stats['mean']:.3f}, σ={vo_stats['std']:.3f}\n"
        stats_text += f"Ratio: {analysis['voltage_ratio']:.3f}"
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # 2. Zoomed view (first 10% of signal)
        ax2 = axes[0, 1]
        zoom_end = max(1, len(time_seconds) // 10)
        time_zoom = time_seconds[:zoom_end]
        vl_zoom = vl_waveform[:zoom_end]
        vo_zoom = vo_waveform[:zoom_end]
        
        ax2.plot(time_zoom, vl_zoom, 'b-', linewidth=2, alpha=0.8, label='VL (Input)')
        ax2.plot(time_zoom, vo_zoom, 'r-', linewidth=2, alpha=0.8, label='VO (Output)')
        
        ax2.set_title('Waveform Detail (First 10%)', fontweight='bold')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Voltage')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add phase delay information
        phase_delay = float(analysis['phase_delay_seconds'])
        phase_text = f"Phase Delay: {phase_delay:.6f} s"
        ax2.text(0.02, 0.98, phase_text, transform=ax2.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 3. Frequency domain analysis
        ax3 = axes[1, 0]
        vl_freqs = analysis['vl_freqs']
        vl_psd = analysis['vl_psd']
        vo_freqs = analysis['vo_freqs']
        vo_psd = analysis['vo_psd']
        
        if len(vl_freqs) > 1 and len(vo_freqs) > 1:
            ax3.semilogy(vl_freqs, vl_psd, 'b-', alpha=0.8, label='VL (Input)')
            ax3.semilogy(vo_freqs, vo_psd, 'r-', alpha=0.8, label='VO (Output)')
            
            ax3.set_title('Power Spectral Density', fontweight='bold')
            ax3.set_xlabel('Frequency (Hz)')
            ax3.set_ylabel('Power Spectral Density')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Mark dominant frequencies
            ax3.axvline(analysis['vl_dominant_freq'], color='blue', linestyle='--', alpha=0.7)
            ax3.axvline(analysis['vo_dominant_freq'], color='red', linestyle='--', alpha=0.7)
            
            freq_text = f"VL Dominant: {float(analysis['vl_dominant_freq']):.2f} Hz\n"
            freq_text += f"VO Dominant: {float(analysis['vo_dominant_freq']):.2f} Hz"
            ax3.text(0.02, 0.98, freq_text, transform=ax3.transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        else:
            ax3.text(0.5, 0.5, 'Frequency analysis\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Power Spectral Density', fontweight='bold')
        
        # 4. Input-Output relationship (X-Y plot)
        ax4 = axes[1, 1]
        
        # Subsample for clearer visualization
        subsample = max(1, len(vl_waveform) // 5000)
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
        except Exception as e:
            print(f"⚠️  Linear fit failed: {e}")
        
        ax4.set_title('Input-Output Relationship', fontweight='bold')
        ax4.set_xlabel('VL (Input Voltage)')
        ax4.set_ylabel('VO (Output Voltage)')
        ax4.grid(True, alpha=0.3)
        if 'coeffs' in locals():
            ax4.legend()
        
        plt.tight_layout()
        
        plot_path = self.output_dir / f'{capacitor_id}_cycle_{cycle_number}_waveform.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def generate_waveform_report(self, waveform_data: Dict, analysis: Dict) -> Path:
        """Generate detailed waveform analysis report"""
        capacitor_id = waveform_data['capacitor_id']
        cycle_number = waveform_data['cycle_number']
        
        print(f"📄 Generating waveform analysis report...")
        
        report_path = self.output_dir / f'{capacitor_id}_cycle_{cycle_number}_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# {capacitor_id} Cycle {cycle_number} Waveform Analysis Report\n\n")
            
            f.write("## Overview\n\n")
            f.write(f"Detailed analysis of input-output waveforms for {capacitor_id} during measurement cycle {cycle_number}.\n")
            f.write("This report provides time-domain and frequency-domain characteristics of the capacitor response.\n\n")
            
            f.write("## Measurement Details\n\n")
            f.write(f"- **Capacitor**: {capacitor_id}\n")
            f.write(f"- **Cycle Number**: {cycle_number}\n")
            f.write(f"- **Duration**: {float(waveform_data['duration_seconds']):.3f} seconds\n")
            f.write(f"- **Sampling Rate**: {float(waveform_data['sampling_rate']):.1f} Hz\n")
            f.write(f"- **Data Points**: {len(waveform_data['time_seconds'])}\n\n")
            
            f.write("## Input Signal (VL) Characteristics\n\n")
            vl_stats = analysis['vl_stats']
            f.write(f"- **Mean Voltage**: {vl_stats['mean']:.6f} V\n")
            f.write(f"- **Standard Deviation**: {vl_stats['std']:.6f} V\n")
            f.write(f"- **Minimum**: {vl_stats['min']:.6f} V\n")
            f.write(f"- **Maximum**: {vl_stats['max']:.6f} V\n")
            f.write(f"- **Peak-to-Peak**: {vl_stats['peak_to_peak']:.6f} V\n")
            f.write(f"- **Dominant Frequency**: {float(analysis['vl_dominant_freq']):.2f} Hz\n\n")
            
            f.write("## Output Signal (VO) Characteristics\n\n")
            vo_stats = analysis['vo_stats']
            f.write(f"- **Mean Voltage**: {vo_stats['mean']:.6f} V\n")
            f.write(f"- **Standard Deviation**: {vo_stats['std']:.6f} V\n")
            f.write(f"- **Minimum**: {vo_stats['min']:.6f} V\n")
            f.write(f"- **Maximum**: {vo_stats['max']:.6f} V\n")
            f.write(f"- **Peak-to-Peak**: {vo_stats['peak_to_peak']:.6f} V\n")
            f.write(f"- **Dominant Frequency**: {float(analysis['vo_dominant_freq']):.2f} Hz\n\n")
            
            f.write("## Input-Output Relationship\n\n")
            f.write(f"- **Voltage Ratio (VO/VL)**: {analysis['voltage_ratio']:.6f}\n")
            f.write(f"- **Phase Delay**: {float(analysis['phase_delay_seconds']):.6f} seconds\n")
            
            # Efficiency analysis
            efficiency = analysis['voltage_ratio'] * 100
            f.write(f"- **Apparent Efficiency**: {efficiency:.2f}%\n\n")
            
            f.write("## Signal Quality Assessment\n\n")
            
            # Signal-to-noise ratio estimation
            vl_snr = vl_stats['mean'] / vl_stats['std'] if vl_stats['std'] > 0 else float('inf')
            vo_snr = vo_stats['mean'] / vo_stats['std'] if vo_stats['std'] > 0 else float('inf')
            
            f.write(f"- **VL Signal-to-Noise Ratio**: {vl_snr:.2f}\n")
            f.write(f"- **VO Signal-to-Noise Ratio**: {vo_snr:.2f}\n")
            
            # Linearity assessment
            phase_delay = float(analysis['phase_delay_seconds'])
            if abs(phase_delay) < 0.001:
                linearity = "Good (minimal phase delay)"
            elif abs(phase_delay) < 0.01:
                linearity = "Moderate (small phase delay)"
            else:
                linearity = "Poor (significant phase delay)"
            
            f.write(f"- **Linearity Assessment**: {linearity}\n\n")
            
            f.write("## Interpretation\n\n")
            
            if analysis['voltage_ratio'] > 0.8:
                f.write("- 🟢 **High Efficiency**: Output voltage is close to input voltage\n")
            elif analysis['voltage_ratio'] > 0.5:
                f.write("- 🟡 **Moderate Efficiency**: Some voltage drop observed\n")
            else:
                f.write("- 🔴 **Low Efficiency**: Significant voltage drop indicates degradation\n")
            
            if abs(phase_delay) < 0.001:
                f.write("- 🟢 **Good Response**: Minimal phase delay between input and output\n")
            else:
                f.write("- 🟡 **Delayed Response**: Phase delay may indicate capacitive/resistive changes\n")
            
            f.write(f"\n---\n")
            f.write(f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        return report_path

def main():
    """Main execution function"""
    print("🚀 Starting Single Cycle Waveform Analysis")
    print("=" * 60)
    
    # Data path configuration
    data_path = Path("data/raw/ES12.mat")
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    # Initialize visualizer
    visualizer = SingleCycleWaveformVisualizer()
    
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
            
            # Load raw waveform data
            waveform_data = visualizer.load_raw_waveform_data(data_path, target_capacitor, cycle_num)
            
            if waveform_data is None:
                print(f"⚠️  Skipping cycle {cycle_num} due to loading error")
                continue
            
            # Analyze waveform characteristics
            analysis = visualizer.analyze_waveform_characteristics(waveform_data)
            
            # Generate visualization
            plot_path = visualizer.visualize_single_cycle_waveform(waveform_data, analysis)
            generated_files.append(plot_path)
            
            # Generate report
            report_path = visualizer.generate_waveform_report(waveform_data, analysis)
            generated_files.append(report_path)
            
            # Print summary
            print(f"📊 Cycle {cycle_num} Summary:")
            print(f"   - Duration: {float(waveform_data['duration_seconds']):.3f} seconds")
            print(f"   - Data points: {len(waveform_data['time_seconds'])}")
            print(f"   - Voltage ratio: {analysis['voltage_ratio']:.6f}")
            print(f"   - Phase delay: {float(analysis['phase_delay_seconds']):.6f} seconds")
            print(f"   - VL dominant freq: {float(analysis['vl_dominant_freq']):.2f} Hz")
            print(f"   - VO dominant freq: {float(analysis['vo_dominant_freq']):.2f} Hz")
        
        # Final summary
        print(f"\n" + "=" * 60)
        print("✅ Single Cycle Waveform Analysis Complete!")
        
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