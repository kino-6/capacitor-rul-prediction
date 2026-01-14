#!/usr/bin/env python3
"""
Same Input Response Change Visualization Script

Detailed analysis and visualization of output response changes when repeatedly 
applying the same input (voltage signal) to a specific capacitor unit.

Analysis Items:
- Response delay changes (time axis shifts)
- Response amplitude changes (output level changes)  
- Response shape changes (waveform distortion)
- Response speed changes (rise time, etc.)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import stats, signal
from scipy.optimize import curve_fit
# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from nasa_pcoe_eda.data.es12_loader import ES12DataLoader

# 警告を抑制
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class ResponseDegradationAnalyzer:
    """Response degradation analysis for same input signals"""
    
    def __init__(self, output_dir: Path = Path("output/response_degradation")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set font to avoid rendering issues
        plt.rcParams['font.family'] = ['DejaVu Sans']
        
        # カラーパレット設定
        self.colors = plt.cm.plasma(np.linspace(0, 1, 10))
        
    def load_capacitor_data(self, data_path: Path, capacitor_id: str) -> pd.DataFrame:
        """Load data for specific capacitor"""
        print(f"📊 Loading data for {capacitor_id}...")
        
        loader = ES12DataLoader()
        df = loader.load_dataset(data_path)
        
        # Extract data for specific capacitor
        cap_data = df[df['capacitor'] == capacitor_id].copy()
        cap_data = cap_data.sort_values('cycle').reset_index(drop=True)
        
        print(f"✅ {capacitor_id} data loaded successfully:")
        print(f"   - Measurement cycles: {len(cap_data)}")
        print(f"   - Cycle range: {cap_data['cycle'].min()} - {cap_data['cycle'].max()}")
        
        return cap_data
    
    def analyze_response_characteristics(self, cap_data: pd.DataFrame, capacitor_id: str) -> Dict:
        """Detailed analysis of response characteristics"""
        print(f"🔬 Analyzing response characteristics for {capacitor_id}...")
        
        cycles = cap_data['cycle'].values
        vl_mean = cap_data['vl_mean'].values
        vo_mean = cap_data['vo_mean'].values
        vl_std = cap_data['vl_std'].values
        vo_std = cap_data['vo_std'].values
        voltage_ratio = cap_data['voltage_ratio'].values
        
        # 1. 応答遅延の分析（電圧比の変化から推定）
        def calculate_response_delay(ratio_values, cycles):
            """Estimate response delay from voltage ratio changes"""
            # Delay rate based on initial value
            if len(ratio_values) < 2 or ratio_values[0] == 0:
                return np.zeros_like(ratio_values)
            
            # Interpret voltage ratio changes as delay
            # Lower ratio = slower response
            initial_ratio = ratio_values[0]
            delay_percentage = ((initial_ratio - ratio_values) / initial_ratio) * 100
            return np.maximum(delay_percentage, 0)  # Clip negative values to 0
        
        response_delays = calculate_response_delay(voltage_ratio, cycles)
        
        # 2. Response amplitude change analysis
        def calculate_amplitude_change(values):
            """Calculate amplitude change rate"""
            if len(values) < 2 or values[0] == 0:
                return np.zeros_like(values)
            
            initial_value = values[0]
            amplitude_change = ((values - initial_value) / abs(initial_value)) * 100
            return amplitude_change
        
        vl_amplitude_change = calculate_amplitude_change(vl_mean)
        vo_amplitude_change = calculate_amplitude_change(vo_mean)
        
        # 3. Response stability analysis (standard deviation changes)
        vl_stability_change = calculate_amplitude_change(vl_std)
        vo_stability_change = calculate_amplitude_change(vo_std)
        
        # 4. Response speed analysis (rate of change)
        def calculate_response_speed_change(values, window=5):
            """Calculate response speed changes"""
            if len(values) < window:
                return np.zeros_like(values)
            
            # Smoothing with moving average
            smoothed = np.convolve(values, np.ones(window)/window, mode='same')
            
            # Calculate gradients (change speed)
            gradients = np.gradient(smoothed)
            
            # Change rate based on initial gradient
            if abs(gradients[0]) > 1e-10:
                speed_change = ((gradients - gradients[0]) / abs(gradients[0])) * 100
            else:
                speed_change = np.zeros_like(gradients)
            
            return speed_change
        
        vl_speed_change = calculate_response_speed_change(vl_mean)
        vo_speed_change = calculate_response_speed_change(vo_mean)
        
        # 5. Specific cycle comparison analysis
        def get_cycle_comparison_data(cycles, values, comparison_cycles=[1, 50, 100, 200, 300]):
            """Extract values at specific cycles"""
            comparison_data = {}
            for target_cycle in comparison_cycles:
                # Find closest cycle
                if target_cycle <= cycles.max():
                    closest_idx = np.argmin(np.abs(cycles - target_cycle))
                    actual_cycle = cycles[closest_idx]
                    value = values[closest_idx]
                    comparison_data[target_cycle] = {
                        'actual_cycle': actual_cycle,
                        'value': value,
                        'index': closest_idx
                    }
            return comparison_data
        
        delay_comparison = get_cycle_comparison_data(cycles, response_delays)
        vl_comparison = get_cycle_comparison_data(cycles, vl_mean)
        vo_comparison = get_cycle_comparison_data(cycles, vo_mean)
        
        analysis_result = {
            'capacitor_id': capacitor_id,
            'cycles': cycles,
            'vl_mean': vl_mean,
            'vo_mean': vo_mean,
            'voltage_ratio': voltage_ratio,
            'vl_std': vl_std,
            'vo_std': vo_std,
            
            # Response characteristic changes
            'response_delays': response_delays,
            'vl_amplitude_change': vl_amplitude_change,
            'vo_amplitude_change': vo_amplitude_change,
            'vl_stability_change': vl_stability_change,
            'vo_stability_change': vo_stability_change,
            'vl_speed_change': vl_speed_change,
            'vo_speed_change': vo_speed_change,
            
            # Specific cycle comparisons
            'delay_comparison': delay_comparison,
            'vl_comparison': vl_comparison,
            'vo_comparison': vo_comparison
        }
        
        return analysis_result
    
    def visualize_response_degradation_timeline(self, analysis_result: Dict) -> Path:
        """Visualize response degradation timeline"""
        capacitor_id = analysis_result['capacitor_id']
        print(f"📈 Visualizing response degradation timeline for {capacitor_id}...")
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle(f'{capacitor_id} Same Input Response Change Analysis - Degradation Timeline', 
                     fontsize=16, fontweight='bold')
        
        cycles = analysis_result['cycles']
        
        # 1. Response delay changes
        ax1 = axes[0, 0]
        delays = analysis_result['response_delays']
        ax1.plot(cycles, delays, 'o-', color=self.colors[0], linewidth=2, markersize=3, alpha=0.8)
        ax1.fill_between(cycles, 0, delays, alpha=0.3, color=self.colors[0])
        
        # Highlight specific cycles
        delay_comp = analysis_result['delay_comparison']
        for cycle_num, data in delay_comp.items():
            if cycle_num in [1, 100, 200, 300]:
                idx = data['index']
                ax1.scatter(cycles[idx], delays[idx], s=100, color='red', 
                           marker='o', edgecolor='white', linewidth=2, zorder=5)
                ax1.annotate(f'Cycle {cycle_num}\nDelay: {delays[idx]:.1f}%', 
                           xy=(cycles[idx], delays[idx]), 
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                           fontsize=8)
        
        ax1.set_title('Response Delay Changes', fontweight='bold')
        ax1.set_xlabel('Measurement Cycle')
        ax1.set_ylabel('Delay Rate (%)')
        ax1.grid(True, alpha=0.3)
        
        # 2. VL response amplitude changes
        ax2 = axes[0, 1]
        vl_amp_change = analysis_result['vl_amplitude_change']
        ax2.plot(cycles, vl_amp_change, 'o-', color=self.colors[1], linewidth=2, markersize=3, alpha=0.8)
        
        # Highlight specific cycles
        vl_comp = analysis_result['vl_comparison']
        for cycle_num, data in vl_comp.items():
            if cycle_num in [1, 100, 200, 300]:
                idx = data['index']
                ax2.scatter(cycles[idx], vl_amp_change[idx], s=100, color='red', 
                           marker='s', edgecolor='white', linewidth=2, zorder=5)
                ax2.annotate(f'Cycle {cycle_num}\nChange: {vl_amp_change[idx]:.1f}%', 
                           xy=(cycles[idx], vl_amp_change[idx]), 
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                           fontsize=8)
        
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.set_title('VL Response Amplitude Changes', fontweight='bold')
        ax2.set_xlabel('Measurement Cycle')
        ax2.set_ylabel('Amplitude Change Rate (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. VO response amplitude changes
        ax3 = axes[1, 0]
        vo_amp_change = analysis_result['vo_amplitude_change']
        ax3.plot(cycles, vo_amp_change, 'o-', color=self.colors[2], linewidth=2, markersize=3, alpha=0.8)
        
        # Highlight specific cycles
        vo_comp = analysis_result['vo_comparison']
        for cycle_num, data in vo_comp.items():
            if cycle_num in [1, 100, 200, 300]:
                idx = data['index']
                ax3.scatter(cycles[idx], vo_amp_change[idx], s=100, color='red', 
                           marker='^', edgecolor='white', linewidth=2, zorder=5)
                ax3.annotate(f'Cycle {cycle_num}\nChange: {vo_amp_change[idx]:.1f}%', 
                           xy=(cycles[idx], vo_amp_change[idx]), 
                           xytext=(10, 10), textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                           fontsize=8)
        
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.set_title('VO Response Amplitude Changes', fontweight='bold')
        ax3.set_xlabel('Measurement Cycle')
        ax3.set_ylabel('Amplitude Change Rate (%)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Response stability changes (VL standard deviation)
        ax4 = axes[1, 1]
        vl_stability = analysis_result['vl_stability_change']
        ax4.plot(cycles, vl_stability, 'o-', color=self.colors[3], linewidth=2, markersize=3, alpha=0.8)
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.set_title('VL Response Stability Changes', fontweight='bold')
        ax4.set_xlabel('Measurement Cycle')
        ax4.set_ylabel('Stability Change Rate (%)')
        ax4.grid(True, alpha=0.3)
        
        # 5. Response speed changes (VL)
        ax5 = axes[2, 0]
        vl_speed = analysis_result['vl_speed_change']
        ax5.plot(cycles, vl_speed, 'o-', color=self.colors[4], linewidth=2, markersize=3, alpha=0.8)
        ax5.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax5.set_title('VL Response Speed Changes', fontweight='bold')
        ax5.set_xlabel('Measurement Cycle')
        ax5.set_ylabel('Speed Change Rate (%)')
        ax5.grid(True, alpha=0.3)
        
        # 6. Composite degradation index
        ax6 = axes[2, 1]
        
        # Normalize and integrate multiple indicators
        normalized_delay = delays / (np.max(delays) + 1e-10)
        normalized_vl_change = np.abs(vl_amp_change) / (np.max(np.abs(vl_amp_change)) + 1e-10)
        normalized_vo_change = np.abs(vo_amp_change) / (np.max(np.abs(vo_amp_change)) + 1e-10)
        
        # Composite degradation index (weighted average)
        composite_degradation = (0.4 * normalized_delay + 
                               0.3 * normalized_vl_change + 
                               0.3 * normalized_vo_change) * 100
        
        ax6.plot(cycles, composite_degradation, 'o-', color='red', linewidth=3, markersize=4, alpha=0.8)
        ax6.fill_between(cycles, 0, composite_degradation, alpha=0.3, color='red')
        ax6.set_title('Composite Degradation Index', fontweight='bold')
        ax6.set_xlabel('Measurement Cycle')
        ax6.set_ylabel('Degradation Index (%)')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / f'{capacitor_id}_response_timeline.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def visualize_cycle_comparison(self, analysis_result: Dict) -> Path:
        """Visualize response comparison between specific cycles"""
        capacitor_id = analysis_result['capacitor_id']
        print(f"📊 Visualizing cycle comparison for {capacitor_id}...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{capacitor_id} Same Input Response Comparison - Changes Between Specific Cycles', 
                     fontsize=16, fontweight='bold')
        
        # Target comparison cycles
        comparison_cycles = [1, 50, 100, 200, 300]
        colors_cycle = plt.cm.viridis(np.linspace(0, 1, len(comparison_cycles)))
        
        # 1. Response delay comparison
        ax1 = axes[0, 0]
        delay_comp = analysis_result['delay_comparison']
        
        cycles_list = []
        delays_list = []
        for cycle_num in comparison_cycles:
            if cycle_num in delay_comp:
                cycles_list.append(cycle_num)
                delays_list.append(delay_comp[cycle_num]['value'])
        
        bars1 = ax1.bar(range(len(cycles_list)), delays_list, 
                        color=colors_cycle[:len(cycles_list)], alpha=0.8)
        ax1.set_title('Response Delay by Cycle', fontweight='bold')
        ax1.set_xlabel('Measurement Cycle')
        ax1.set_ylabel('Delay Rate (%)')
        ax1.set_xticks(range(len(cycles_list)))
        ax1.set_xticklabels([f'Cycle {c}' for c in cycles_list])
        ax1.grid(True, alpha=0.3)
        
        # Display values on bars
        for i, (bar, delay) in enumerate(zip(bars1, delays_list)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{delay:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        # 2. VL response value comparison
        ax2 = axes[0, 1]
        vl_comp = analysis_result['vl_comparison']
        
        vl_values = []
        for cycle_num in comparison_cycles:
            if cycle_num in vl_comp:
                vl_values.append(vl_comp[cycle_num]['value'])
        
        bars2 = ax2.bar(range(len(cycles_list)), vl_values, 
                        color=colors_cycle[:len(cycles_list)], alpha=0.8)
        ax2.set_title('VL Response Value by Cycle', fontweight='bold')
        ax2.set_xlabel('Measurement Cycle')
        ax2.set_ylabel('VL Response Value')
        ax2.set_xticks(range(len(cycles_list)))
        ax2.set_xticklabels([f'Cycle {c}' for c in cycles_list])
        ax2.grid(True, alpha=0.3)
        
        # Display values on bars
        for i, (bar, vl_val) in enumerate(zip(bars2, vl_values)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vl_values)*0.01,
                    f'{vl_val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. VO response value comparison
        ax3 = axes[1, 0]
        vo_comp = analysis_result['vo_comparison']
        
        vo_values = []
        for cycle_num in comparison_cycles:
            if cycle_num in vo_comp:
                vo_values.append(vo_comp[cycle_num]['value'])
        
        bars3 = ax3.bar(range(len(cycles_list)), vo_values, 
                        color=colors_cycle[:len(cycles_list)], alpha=0.8)
        ax3.set_title('VO Response Value by Cycle', fontweight='bold')
        ax3.set_xlabel('Measurement Cycle')
        ax3.set_ylabel('VO Response Value')
        ax3.set_xticks(range(len(cycles_list)))
        ax3.set_xticklabels([f'Cycle {c}' for c in cycles_list])
        ax3.grid(True, alpha=0.3)
        
        # Display values on bars
        for i, (bar, vo_val) in enumerate(zip(bars3, vo_values)):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(vo_values)*0.01,
                    f'{vo_val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Response change rate progression
        ax4 = axes[1, 1]
        
        # Calculate change rates from initial values
        if len(vl_values) > 0 and len(vo_values) > 0:
            vl_initial = vl_values[0]
            vo_initial = vo_values[0]
            
            vl_change_rates = [(v - vl_initial) / abs(vl_initial) * 100 for v in vl_values]
            vo_change_rates = [(v - vo_initial) / abs(vo_initial) * 100 for v in vo_values]
            
            x_pos = np.arange(len(cycles_list))
            width = 0.35
            
            bars4a = ax4.bar(x_pos - width/2, vl_change_rates, width, 
                            label='VL Response Change Rate', color=self.colors[1], alpha=0.8)
            bars4b = ax4.bar(x_pos + width/2, vo_change_rates, width, 
                            label='VO Response Change Rate', color=self.colors[2], alpha=0.8)
            
            ax4.set_title('Response Change Rate Progression', fontweight='bold')
            ax4.set_xlabel('Measurement Cycle')
            ax4.set_ylabel('Change Rate (%)')
            ax4.set_xticks(x_pos)
            ax4.set_xticklabels([f'Cycle {c}' for c in cycles_list])
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / f'{capacitor_id}_cycle_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def visualize_response_waveform_evolution(self, analysis_result: Dict) -> Path:
        """Visualize response waveform evolution (simulated)"""
        capacitor_id = analysis_result['capacitor_id']
        print(f"🌊 Visualizing response waveform evolution for {capacitor_id}...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{capacitor_id} Response Waveform Evolution for Same Input', 
                     fontsize=16, fontweight='bold')
        
        # Time axis (simulated)
        t = np.linspace(0, 1, 1000)  # 1 second response
        
        # Specific cycles for response characteristics
        target_cycles = [1, 50, 100, 200, 300, 390]
        
        for i, cycle_num in enumerate(target_cycles):
            if i >= 6:  # Maximum 6 plots
                break
                
            row = i // 3
            col = i % 3
            ax = axes[row, col]
            
            # Get response characteristics from real data
            cycles = analysis_result['cycles']
            delays = analysis_result['response_delays']
            vl_values = analysis_result['vl_mean']
            vo_values = analysis_result['vo_mean']
            
            # Find closest cycle index
            if cycle_num <= cycles.max():
                closest_idx = np.argmin(np.abs(cycles - cycle_num))
                actual_cycle = cycles[closest_idx]
                delay_percent = delays[closest_idx]
                vl_amplitude = vl_values[closest_idx]
                vo_amplitude = vo_values[closest_idx]
                
                # Input signal (same for all)
                input_signal = np.sin(2 * np.pi * 5 * t)  # 5Hz sine wave
                
                # Output signal (reflecting degradation)
                delay_samples = int(delay_percent / 100 * 50)  # Convert delay to sample count
                
                # VL response (reflecting delay and amplitude changes)
                vl_response = np.zeros_like(t)
                if delay_samples < len(t):
                    vl_response[delay_samples:] = vl_amplitude * input_signal[:-delay_samples if delay_samples > 0 else len(input_signal)]
                
                # VO response (reflecting amplitude changes)
                vo_response = vo_amplitude * input_signal
                
                # Add noise (expressing instability due to degradation)
                noise_level = delay_percent / 100 * 0.1
                vl_response += np.random.normal(0, noise_level, len(vl_response))
                vo_response += np.random.normal(0, noise_level, len(vo_response))
                
                # Plot
                ax.plot(t, input_signal, 'k--', linewidth=2, alpha=0.7, label='Input Signal')
                ax.plot(t, vl_response, color=self.colors[1], linewidth=2, label=f'VL Response')
                ax.plot(t, vo_response, color=self.colors[2], linewidth=2, label=f'VO Response')
                
                ax.set_title(f'Cycle {actual_cycle}\nDelay: {delay_percent:.1f}%', fontweight='bold')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Response Value')
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)
                
                # Visually show delay
                if delay_percent > 1:
                    ax.axvline(x=delay_samples/1000, color='red', linestyle=':', 
                              alpha=0.7, label=f'Delay: {delay_percent:.1f}%')
        
        plt.tight_layout()
        
        plot_path = self.output_dir / f'{capacitor_id}_waveform_evolution.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def generate_response_report(self, analysis_result: Dict) -> Path:
        """Generate response degradation report"""
        capacitor_id = analysis_result['capacitor_id']
        print(f"📄 Generating response degradation report for {capacitor_id}...")
        
        report_path = self.output_dir / f'{capacitor_id}_response_report.md'
        
        # Statistical calculations
        delays = analysis_result['response_delays']
        vl_changes = analysis_result['vl_amplitude_change']
        vo_changes = analysis_result['vo_amplitude_change']
        
        max_delay = np.max(delays)
        final_delay = delays[-1]
        max_vl_change = np.max(np.abs(vl_changes))
        max_vo_change = np.max(np.abs(vo_changes))
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# {capacitor_id} Same Input Response Change Analysis Report\n\n")
            f.write("## Overview\n\n")
            f.write(f"Detailed analysis of output response changes when repeatedly applying the same input to {capacitor_id}.\n")
            f.write("Quantified degradation indicators including response delay, amplitude changes, and stability changes.\n\n")
            
            f.write("## Key Findings\n\n")
            f.write(f"### Response Delay Changes\n\n")
            f.write(f"- **Maximum Delay Rate**: {max_delay:.2f}%\n")
            f.write(f"- **Final Delay Rate**: {final_delay:.2f}%\n")
            f.write(f"- **Delay Progression**: {'Progressive' if final_delay > max_delay * 0.8 else 'Temporary'}\n\n")
            
            f.write(f"### Response Amplitude Changes\n\n")
            f.write(f"- **VL Maximum Change Rate**: {max_vl_change:.2f}%\n")
            f.write(f"- **VO Maximum Change Rate**: {max_vo_change:.2f}%\n")
            f.write(f"- **Primary Change**: {'VL Response' if max_vl_change > max_vo_change else 'VO Response'}\n\n")
            
            # Specific cycle details
            f.write("### Response Changes at Specific Cycles\n\n")
            delay_comp = analysis_result['delay_comparison']
            vl_comp = analysis_result['vl_comparison']
            
            for cycle_num in [1, 100, 200, 300]:
                if cycle_num in delay_comp and cycle_num in vl_comp:
                    delay_val = delay_comp[cycle_num]['value']
                    vl_val = vl_comp[cycle_num]['value']
                    
                    f.write(f"#### Cycle {cycle_num} Measurement\n\n")
                    f.write(f"- **Response Delay**: {delay_val:.2f}%\n")
                    f.write(f"- **VL Response Value**: {vl_val:.4f}\n")
                    
                    if cycle_num == 1:
                        f.write("- **Status**: Initial State (Baseline)\n")
                    elif delay_val > 10:
                        f.write("- **Status**: 🔴 Severe Degradation\n")
                    elif delay_val > 5:
                        f.write("- **Status**: 🟡 Moderate Degradation\n")
                    else:
                        f.write("- **Status**: 🟢 Mild Degradation\n")
                    f.write("\n")
            
            f.write("## Estimated Degradation Mechanisms\n\n")
            if max_delay > 20:
                f.write("1. **Response Delay Increase**: Response speed decreased due to increased internal resistance\n")
            if max_vl_change > 50:
                f.write("2. **VL Response Changes**: Voltage response changes due to capacitive component degradation\n")
            if max_vo_change > 10:
                f.write("3. **VO Response Changes**: Output characteristic degradation\n")
            
            f.write("\n## Recommendations\n\n")
            if max_delay > 15 or max_vl_change > 100:
                f.write("- 🔴 **Immediate Replacement Recommended**: Significant degradation in response characteristics confirmed\n")
            elif max_delay > 5 or max_vl_change > 50:
                f.write("- 🟡 **Enhanced Monitoring Recommended**: Carefully monitor degradation progression\n")
            else:
                f.write("- 🟢 **Normal Range**: Current degradation level is within acceptable range\n")
            
            f.write(f"\n---\n")
            f.write(f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        return report_path

def main():
    """Main execution function"""
    print("🚀 Starting Same Input Response Change Analysis")
    print("=" * 60)
    
    # Data path configuration
    data_path = Path("data/raw/ES12.mat")
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    # Initialize analyzer
    analyzer = ResponseDegradationAnalyzer()
    
    try:
        # Select most degraded capacitor (from previous analysis results)
        target_capacitor = "ES12C4"  # Most severe degradation (-126.8%)
        
        print(f"🎯 Analysis Target: {target_capacitor} (Most severely degraded unit)")
        
        # Load data
        cap_data = analyzer.load_capacitor_data(data_path, target_capacitor)
        
        # Analyze response characteristics
        analysis_result = analyzer.analyze_response_characteristics(cap_data, target_capacitor)
        
        # Generate visualizations
        print(f"\n📈 Generating visualizations...")
        generated_plots = []
        
        # 1. Response degradation timeline
        timeline_plot = analyzer.visualize_response_degradation_timeline(analysis_result)
        generated_plots.append(timeline_plot)
        
        # 2. Cycle comparison
        comparison_plot = analyzer.visualize_cycle_comparison(analysis_result)
        generated_plots.append(comparison_plot)
        
        # 3. Response waveform evolution
        waveform_plot = analyzer.visualize_response_waveform_evolution(analysis_result)
        generated_plots.append(waveform_plot)
        
        # Generate report
        report_path = analyzer.generate_response_report(analysis_result)
        
        # Results summary
        print(f"\n" + "=" * 60)
        print("✅ Same Input Response Change Analysis Complete!")
        
        delays = analysis_result['response_delays']
        vl_changes = analysis_result['vl_amplitude_change']
        
        print(f"\n📊 {target_capacitor} Response Change Summary:")
        print(f"   - Maximum Response Delay: {np.max(delays):.1f}%")
        print(f"   - Final Response Delay: {delays[-1]:.1f}%")
        print(f"   - Maximum VL Change Rate: {np.max(np.abs(vl_changes)):.1f}%")
        print(f"   - Measurement Cycles: {len(analysis_result['cycles'])}")
        
        print(f"\n📈 Response Changes at Specific Cycles:")
        delay_comp = analysis_result['delay_comparison']
        for cycle_num in [1, 100, 200, 300]:
            if cycle_num in delay_comp:
                delay_val = delay_comp[cycle_num]['value']
                print(f"   - Cycle {cycle_num}: Delay {delay_val:.1f}%")
        
        print(f"\n📁 Generated Files:")
        for plot_path in generated_plots:
            print(f"   - {plot_path.name}")
        print(f"   - {report_path.name}")
        
        print(f"\n💡 Degradation Status:")
        max_delay = np.max(delays)
        if max_delay > 20:
            print(f"   - 🔴 Severe Degradation: Response delay reached {max_delay:.1f}%")
            print(f"   - Immediate replacement recommended")
        elif max_delay > 10:
            print(f"   - 🟡 Moderate Degradation: Response delay is {max_delay:.1f}%")
            print(f"   - Enhanced monitoring recommended")
        else:
            print(f"   - 🟢 Mild Degradation: Response delay is {max_delay:.1f}%")
        
        print(f"\n📍 Output Directory: {analyzer.output_dir}")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()