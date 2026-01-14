#!/usr/bin/env python3
"""
Explore ES12 data for periodic waveform patterns like sine waves
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy.fft import fft, fftfreq

def explore_waveform_patterns():
    """Explore ES12 data for periodic patterns"""
    data_path = Path("data/raw/ES12.mat")
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    with h5py.File(data_path, 'r') as f:
        print("🔍 Exploring ES12 Waveform Patterns")
        print("=" * 50)
        
        # Check ES12C4 data
        transient_group = f['ES12']['Transient_Data']
        cap_group = transient_group['ES12C4']
        
        vl_data = cap_group['VL'][:]
        vo_data = cap_group['VO'][:]
        
        print(f"📊 Data shape: VL {vl_data.shape}, VO {vo_data.shape}")
        
        # Analyze first few cycles for patterns
        for cycle_idx in [0, 1, 2]:  # Cycles 1, 2, 3
            print(f"\n🔄 Analyzing Cycle {cycle_idx + 1}")
            
            vl_cycle = vl_data[:, cycle_idx]
            vo_cycle = vo_data[:, cycle_idx]
            
            # Remove NaN values
            valid_mask = ~np.isnan(vl_cycle) & ~np.isnan(vo_cycle)
            vl_clean = vl_cycle[valid_mask]
            vo_clean = vo_cycle[valid_mask]
            
            if len(vl_clean) < 100:
                print(f"   ⚠️  Too few valid points: {len(vl_clean)}")
                continue
            
            print(f"   📈 Valid points: {len(vl_clean)}")
            print(f"   📊 VL range: {np.min(vl_clean):.4f} to {np.max(vl_clean):.4f}")
            print(f"   📊 VO range: {np.min(vo_clean):.4f} to {np.max(vo_clean):.4f}")
            
            # Look for periodic patterns
            # 1. Check for sine-like patterns by analyzing frequency content
            if len(vl_clean) > 1000:
                # Take a subset for analysis
                subset_size = min(5000, len(vl_clean))
                vl_subset = vl_clean[:subset_size]
                vo_subset = vo_clean[:subset_size]
                
                # FFT analysis
                vl_fft = np.abs(fft(vl_subset - np.mean(vl_subset)))
                vo_fft = np.abs(fft(vo_subset - np.mean(vo_subset)))
                freqs = fftfreq(len(vl_subset))
                
                # Find dominant frequencies (excluding DC)
                vl_dominant_idx = np.argmax(vl_fft[1:len(vl_fft)//2]) + 1
                vo_dominant_idx = np.argmax(vo_fft[1:len(vo_fft)//2]) + 1
                
                vl_dominant_freq = freqs[vl_dominant_idx]
                vo_dominant_freq = freqs[vo_dominant_idx]
                
                print(f"   🎵 VL dominant frequency: {vl_dominant_freq:.6f}")
                print(f"   🎵 VO dominant frequency: {vo_dominant_freq:.6f}")
                
                # Check if there's a clear periodic pattern
                vl_peak_power = vl_fft[vl_dominant_idx]
                vl_total_power = np.sum(vl_fft[1:len(vl_fft)//2])
                vl_periodicity = vl_peak_power / vl_total_power if vl_total_power > 0 else 0
                
                print(f"   📊 VL periodicity ratio: {vl_periodicity:.3f}")
                
                # Look for phase relationship
                if vl_periodicity > 0.1:  # If there's some periodicity
                    # Cross-correlation to find phase delay
                    correlation = np.correlate(vo_subset - np.mean(vo_subset), 
                                             vl_subset - np.mean(vl_subset), mode='full')
                    max_corr_idx = np.argmax(np.abs(correlation))
                    phase_delay_samples = max_corr_idx - (len(correlation) // 2)
                    
                    print(f"   ⏱️  Phase delay: {phase_delay_samples} samples")
                    
                    # Check if this looks like a sine wave response
                    if vl_periodicity > 0.3:
                        print(f"   ✅ Potential periodic pattern detected!")
                        
                        # Create a detailed plot for this cycle
                        create_detailed_waveform_plot(vl_subset, vo_subset, cycle_idx + 1)
                    else:
                        print(f"   ❌ No clear periodic pattern")
                else:
                    print(f"   ❌ No significant periodicity")
            
            # Sample some values to see the pattern
            sample_size = min(20, len(vl_clean))
            print(f"   📋 First {sample_size} VL values: {vl_clean[:sample_size]}")
            print(f"   📋 First {sample_size} VO values: {vo_clean[:sample_size]}")

def create_detailed_waveform_plot(vl_data, vo_data, cycle_num):
    """Create detailed waveform plot for cycles with periodic patterns"""
    print(f"📈 Creating detailed plot for Cycle {cycle_num}")
    
    output_dir = Path("output/waveform_exploration")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create time axis
    time_points = np.arange(len(vl_data))
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    fig.suptitle(f'ES12C4 Cycle {cycle_num} - Detailed Waveform Analysis', 
                 fontsize=16, fontweight='bold')
    
    # 1. Full waveforms
    ax1 = axes[0]
    ax1.plot(time_points, vl_data, 'b-', linewidth=1, alpha=0.8, label='VL (Input)')
    ax1.plot(time_points, vo_data, 'r-', linewidth=1, alpha=0.8, label='VO (Output)')
    ax1.set_title('Complete Waveforms', fontweight='bold')
    ax1.set_xlabel('Sample Points')
    ax1.set_ylabel('Voltage (V)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Zoomed view (first 1000 points)
    ax2 = axes[1]
    zoom_end = min(1000, len(vl_data))
    time_zoom = time_points[:zoom_end]
    vl_zoom = vl_data[:zoom_end]
    vo_zoom = vo_data[:zoom_end]
    
    ax2.plot(time_zoom, vl_zoom, 'b-', linewidth=1.5, alpha=0.8, label='VL (Input)')
    ax2.plot(time_zoom, vo_zoom, 'r-', linewidth=1.5, alpha=0.8, label='VO (Output)')
    ax2.set_title(f'Zoomed View (First {zoom_end} points)', fontweight='bold')
    ax2.set_xlabel('Sample Points')
    ax2.set_ylabel('Voltage (V)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Phase relationship plot
    ax3 = axes[2]
    # Plot VL vs VO to see phase relationship
    ax3.plot(vl_data[::10], vo_data[::10], 'g.', alpha=0.6, markersize=1)
    ax3.set_title('Input-Output Phase Relationship', fontweight='bold')
    ax3.set_xlabel('VL (Input Voltage)')
    ax3.set_ylabel('VO (Output Voltage)')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = output_dir / f'ES12C4_cycle_{cycle_num}_detailed_waveform.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   💾 Saved plot: {plot_path}")

def check_all_capacitors_for_patterns():
    """Check all capacitors for periodic patterns"""
    data_path = Path("data/raw/ES12.mat")
    
    with h5py.File(data_path, 'r') as f:
        transient_group = f['ES12']['Transient_Data']
        
        print(f"\n🔍 Checking all capacitors for periodic patterns")
        print("=" * 60)
        
        for cap_name in ['ES12C1', 'ES12C2', 'ES12C3', 'ES12C4', 'ES12C5', 'ES12C6', 'ES12C7', 'ES12C8']:
            if cap_name in transient_group:
                print(f"\n📋 {cap_name}:")
                
                cap_group = transient_group[cap_name]
                vl_data = cap_group['VL'][:]
                vo_data = cap_group['VO'][:]
                
                # Check first cycle
                vl_cycle = vl_data[:, 0]
                vo_cycle = vo_data[:, 0]
                
                valid_mask = ~np.isnan(vl_cycle) & ~np.isnan(vo_cycle)
                vl_clean = vl_cycle[valid_mask]
                vo_clean = vo_cycle[valid_mask]
                
                if len(vl_clean) > 1000:
                    # Quick FFT analysis
                    subset_size = min(2000, len(vl_clean))
                    vl_subset = vl_clean[:subset_size]
                    
                    vl_fft = np.abs(fft(vl_subset - np.mean(vl_subset)))
                    vl_peak_power = np.max(vl_fft[1:len(vl_fft)//2])
                    vl_total_power = np.sum(vl_fft[1:len(vl_fft)//2])
                    vl_periodicity = vl_peak_power / vl_total_power if vl_total_power > 0 else 0
                    
                    print(f"   Valid points: {len(vl_clean)}, Periodicity: {vl_periodicity:.3f}")
                    
                    if vl_periodicity > 0.3:
                        print(f"   ✅ {cap_name} has potential periodic patterns!")
                    else:
                        print(f"   ❌ {cap_name} no clear periodic pattern")
                else:
                    print(f"   ⚠️  Insufficient data: {len(vl_clean)} points")

if __name__ == "__main__":
    explore_waveform_patterns()
    check_all_capacitors_for_patterns()