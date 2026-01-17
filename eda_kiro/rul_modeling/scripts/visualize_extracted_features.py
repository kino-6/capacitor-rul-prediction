"""
Visualize extracted response features to verify degradation patterns.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Paths
BASE_DIR = Path(__file__).parent.parent
FEATURES_PATH = BASE_DIR / "output" / "features_v3" / "es12_response_features.csv"
OUTPUT_DIR = BASE_DIR / "output" / "features_v3"

def visualize_key_features():
    """Visualize key response features over time."""
    print("="*80)
    print("RESPONSE FEATURES VISUALIZATION")
    print("="*80)
    
    # Load features
    print("\nLoading features...")
    df = pd.read_csv(FEATURES_PATH)
    print(f"  ✓ Loaded {len(df)} samples")
    
    # Create figure
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('Response Features Evolution - All Capacitors', fontsize=16, fontweight='bold')
    
    capacitors = sorted(df['capacitor_id'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(capacitors)))
    
    # 1. Response Efficiency
    ax = axes[0, 0]
    for i, cap_id in enumerate(capacitors):
        cap_data = df[df['capacitor_id'] == cap_id]
        ax.plot(cap_data['cycle'], cap_data['response_efficiency'], 
                label=cap_id, color=colors[i], alpha=0.7)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Response Efficiency')
    ax.set_title('Response Efficiency Over Time')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. Voltage Ratio
    ax = axes[0, 1]
    for i, cap_id in enumerate(capacitors):
        cap_data = df[df['capacitor_id'] == cap_id]
        ax.plot(cap_data['cycle'], cap_data['voltage_ratio'], 
                label=cap_id, color=colors[i], alpha=0.7)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Voltage Ratio')
    ax.set_title('Voltage Ratio Over Time')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. Waveform Correlation
    ax = axes[1, 0]
    for i, cap_id in enumerate(capacitors):
        cap_data = df[df['capacitor_id'] == cap_id]
        ax.plot(cap_data['cycle'], cap_data['waveform_correlation'], 
                label=cap_id, color=colors[i], alpha=0.7)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Waveform Correlation')
    ax.set_title('Waveform Correlation Over Time')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. Efficiency Degradation Rate
    ax = axes[1, 1]
    for i, cap_id in enumerate(capacitors):
        cap_data = df[df['capacitor_id'] == cap_id]
        # Skip first 10 cycles (NaN values)
        cap_data_valid = cap_data[cap_data['cycle'] > 10]
        ax.plot(cap_data_valid['cycle'], cap_data_valid['efficiency_degradation_rate'], 
                label=cap_id, color=colors[i], alpha=0.7)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('Efficiency Degradation Rate')
    ax.set_title('Efficiency Degradation Rate Over Time')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 5. VO Variability
    ax = axes[2, 0]
    for i, cap_id in enumerate(capacitors):
        cap_data = df[df['capacitor_id'] == cap_id]
        ax.plot(cap_data['cycle'], cap_data['vo_variability'], 
                label=cap_id, color=colors[i], alpha=0.7)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('VO Variability')
    ax.set_title('VO Variability Over Time')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 6. VO Complexity
    ax = axes[2, 1]
    for i, cap_id in enumerate(capacitors):
        cap_data = df[df['capacitor_id'] == cap_id]
        ax.plot(cap_data['cycle'], cap_data['vo_complexity'], 
                label=cap_id, color=colors[i], alpha=0.7)
    ax.set_xlabel('Cycle')
    ax.set_ylabel('VO Complexity')
    ax.set_title('VO Complexity Over Time')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = OUTPUT_DIR / "response_features_evolution.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {output_path}")
    
    plt.close()
    
    # Create summary statistics
    print("\n" + "="*80)
    print("FEATURE STATISTICS BY CYCLE RANGE")
    print("="*80)
    
    for cap_id in capacitors:
        cap_data = df[df['capacitor_id'] == cap_id]
        
        early = cap_data[cap_data['cycle'] <= 50]
        late = cap_data[cap_data['cycle'] >= 150]
        
        print(f"\n{cap_id}:")
        print(f"  Response Efficiency:")
        print(f"    Early (1-50):   {early['response_efficiency'].mean():.2f} ± {early['response_efficiency'].std():.2f}")
        print(f"    Late (150-200): {late['response_efficiency'].mean():.2f} ± {late['response_efficiency'].std():.2f}")
        print(f"  Waveform Correlation:")
        print(f"    Early (1-50):   {early['waveform_correlation'].mean():.4f} ± {early['waveform_correlation'].std():.4f}")
        print(f"    Late (150-200): {late['waveform_correlation'].mean():.4f} ± {late['waveform_correlation'].std():.4f}")


def main():
    """Main execution."""
    visualize_key_features()
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print(f"\nOutput: {OUTPUT_DIR / 'response_features_evolution.png'}")


if __name__ == "__main__":
    main()
