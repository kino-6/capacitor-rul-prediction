"""
Extract response features from all capacitors and cycles.

This script extracts VL-VO response features from all ES12 capacitors
across all cycles using parallel processing.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.data_loader import load_es12_cycle_data, get_available_capacitors
from src.feature_extraction.response_extractor import ResponseFeatureExtractor

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR.parent / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "output" / "features_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ES12_PATH = DATA_DIR / "ES12.mat"


def extract_features_for_capacitor(cap_id: str) -> pd.DataFrame:
    """Extract features for all cycles of a single capacitor."""
    extractor = ResponseFeatureExtractor()
    results = []
    
    print(f"\n  Processing {cap_id}...")
    
    for cycle in tqdm(range(1, 201), desc=f"  {cap_id}", leave=False):
        try:
            # Load cycle data
            vl, vo = load_es12_cycle_data(str(ES12_PATH), cap_id, cycle)
            
            # Extract features
            features = extractor.extract_features(
                vl, vo, cap_id, cycle, include_advanced=True
            )
            
            # Add metadata
            features['capacitor_id'] = cap_id
            features['cycle'] = cycle
            
            results.append(features)
            
        except Exception as e:
            print(f"    Error at {cap_id} cycle {cycle}: {e}")
            continue
    
    return pd.DataFrame(results)


def extract_all_features_parallel():
    """Extract features from all capacitors using parallel processing."""
    print("="*80)
    print("RESPONSE FEATURE EXTRACTION")
    print("="*80)
    
    capacitors = get_available_capacitors()
    print(f"\nCapacitors to process: {len(capacitors)}")
    print(f"Cycles per capacitor: 200")
    print(f"Total samples: {len(capacitors) * 200}")
    print(f"CPU cores available: {cpu_count()}")
    
    # Extract features in parallel
    print("\nExtracting features (parallel processing)...")
    
    with Pool(processes=min(len(capacitors), cpu_count())) as pool:
        results = pool.map(extract_features_for_capacitor, capacitors)
    
    # Combine results
    print("\nCombining results...")
    all_features_df = pd.concat(results, ignore_index=True)
    
    print(f"  ✓ Extracted {len(all_features_df)} samples")
    print(f"  ✓ Features: {len(all_features_df.columns) - 2} (+ 2 metadata)")
    
    return all_features_df


def save_features(df: pd.DataFrame):
    """Save extracted features to CSV."""
    output_path = OUTPUT_DIR / "es12_response_features.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved features: {output_path}")
    
    # Generate summary
    summary_path = OUTPUT_DIR / "feature_extraction_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("Response Feature Extraction Summary\n")
        f.write("="*80 + "\n\n")
        f.write(f"Total Samples: {len(df)}\n")
        f.write(f"Capacitors: {df['capacitor_id'].nunique()}\n")
        f.write(f"Cycles per Capacitor: {df.groupby('capacitor_id')['cycle'].count().mean():.0f}\n")
        f.write(f"Total Features: {len(df.columns) - 2}\n\n")
        
        f.write("Feature List:\n")
        f.write("-" * 80 + "\n")
        for col in df.columns:
            if col not in ['capacitor_id', 'cycle']:
                f.write(f"  - {col}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("Feature Statistics:\n")
        f.write("-" * 80 + "\n")
        f.write(df.describe().to_string())
    
    print(f"✓ Saved summary: {summary_path}")


def analyze_feature_quality(df: pd.DataFrame):
    """Analyze quality of extracted features."""
    print("\n" + "="*80)
    print("FEATURE QUALITY ANALYSIS")
    print("="*80)
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("\n⚠️ Missing values detected:")
        print(missing[missing > 0])
    else:
        print("\n✅ No missing values")
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = {}
    for col in numeric_cols:
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            inf_counts[col] = inf_count
    
    if inf_counts:
        print("\n⚠️ Infinite values detected:")
        for col, count in inf_counts.items():
            print(f"  {col}: {count}")
    else:
        print("✅ No infinite values")
    
    # Check feature ranges
    print("\n📊 Feature Ranges:")
    print("-" * 80)
    
    key_features = [
        'response_efficiency',
        'voltage_ratio',
        'waveform_correlation',
        'efficiency_degradation_rate',
        'response_delay'
    ]
    
    for feat in key_features:
        if feat in df.columns:
            print(f"\n{feat}:")
            print(f"  Min: {df[feat].min():.4f}")
            print(f"  Max: {df[feat].max():.4f}")
            print(f"  Mean: {df[feat].mean():.4f}")
            print(f"  Std: {df[feat].std():.4f}")
    
    # Check degradation patterns
    print("\n📈 Degradation Pattern Check:")
    print("-" * 80)
    
    for cap_id in df['capacitor_id'].unique():
        cap_data = df[df['capacitor_id'] == cap_id]
        
        # Check if efficiency decreases over time
        early_eff = cap_data[cap_data['cycle'] <= 50]['response_efficiency'].mean()
        late_eff = cap_data[cap_data['cycle'] >= 150]['response_efficiency'].mean()
        
        if early_eff > late_eff:
            status = "✅"
        else:
            status = "⚠️"
        
        print(f"{status} {cap_id}: {early_eff:.2f}% → {late_eff:.2f}% "
              f"(Change: {((late_eff - early_eff) / early_eff * 100):+.1f}%)")


def main():
    """Main execution."""
    # Extract features
    df = extract_all_features_parallel()
    
    # Save features
    save_features(df)
    
    # Analyze quality
    analyze_feature_quality(df)
    
    print("\n" + "="*80)
    print("EXTRACTION COMPLETE!")
    print("="*80)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nGenerated files:")
    print("  1. es12_response_features.csv")
    print("  2. feature_extraction_summary.txt")
    print("\n✅ Response feature extraction complete!")


if __name__ == "__main__":
    main()
