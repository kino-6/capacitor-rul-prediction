#!/usr/bin/env python3
"""
Demonstration of RULFeatureAnalyzer functionality.

This script shows how to use the RULFeatureAnalyzer to:
1. Identify features with degradation trends
2. Compute degradation rates
3. Rank features for RUL prediction
4. Visualize degradation patterns
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nasa_pcoe_eda.analysis.rul_features import RULFeatureAnalyzer

def create_sample_data():
    """Create sample degradation data for demonstration."""
    np.random.seed(42)  # For reproducible results
    
    # Time points (e.g., cycles, days, etc.)
    time_points = np.arange(100)
    
    # Create different types of features
    # 1. Strongly degrading feature (capacity)
    capacity = 100 - 0.8 * time_points + np.random.normal(0, 2, 100)
    
    # 2. Moderately degrading feature (resistance)
    resistance = 10 + 0.3 * time_points + np.random.normal(0, 1, 100)
    
    # 3. Stable feature (temperature)
    temperature = np.random.normal(25, 2, 100)
    
    # 4. RUL values (decreasing over time)
    rul = 100 - time_points + np.random.normal(0, 1, 100)
    rul = np.maximum(rul, 0)  # RUL cannot be negative
    
    # Create DataFrame
    df = pd.DataFrame({
        'cycle': time_points,
        'capacity': capacity,
        'resistance': resistance,
        'temperature': temperature,
        'rul': rul
    })
    
    return df

def main():
    """Main demonstration function."""
    print("RULFeatureAnalyzer Demonstration")
    print("=" * 40)
    
    # Create sample data
    print("1. Creating sample degradation data...")
    df = create_sample_data()
    print(f"   Created dataset with {len(df)} samples and {len(df.columns)} features")
    print(f"   Features: {list(df.columns)}")
    
    # Initialize analyzer
    analyzer = RULFeatureAnalyzer()
    
    # 1. Identify degradation features
    print("\n2. Identifying features with degradation trends...")
    degradation_features = analyzer.identify_degradation_features(df)
    print(f"   Degradation features found: {degradation_features}")
    
    # 2. Compute degradation rates
    print("\n3. Computing degradation rates...")
    feature_list = ['capacity', 'resistance', 'temperature']
    rates = analyzer.compute_degradation_rates(df, feature_list)
    print("   Degradation rates:")
    for feature, rate in rates.items():
        direction = "decreasing" if rate < 0 else "increasing"
        print(f"     {feature}: {rate:.4f} units/cycle ({direction})")
    
    # 3. Rank features for RUL prediction
    print("\n4. Ranking features for RUL prediction...")
    ranking = analyzer.rank_features_for_rul(df, 'rul')
    print("   Feature ranking (by correlation with RUL):")
    for i, (feature, correlation) in enumerate(ranking, 1):
        print(f"     {i}. {feature}: correlation = {correlation:.4f}")
    
    # 4. Visualize degradation patterns
    print("\n5. Generating degradation pattern visualizations...")
    figures = analyzer.visualize_degradation_patterns(df, degradation_features)
    print(f"   Generated {len(figures)} visualization figures")
    
    # Save visualizations
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, fig in enumerate(figures):
        filename = f"{output_dir}/degradation_pattern_{i+1}.png"
        fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"   Saved: {filename}")
    
    plt.close('all')  # Close all figures to free memory
    
    print("\n6. Summary:")
    print(f"   - Found {len(degradation_features)} features with degradation trends")
    print(f"   - Computed degradation rates for {len(rates)} features")
    print(f"   - Ranked {len(ranking)} features for RUL prediction")
    print(f"   - Generated {len(figures)} visualization plots")
    print("\nDemonstration completed successfully!")

if __name__ == "__main__":
    main()