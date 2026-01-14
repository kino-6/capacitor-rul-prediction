#!/usr/bin/env python3
"""
Integration test for ES12 data loading with the existing EDA system.

This script tests that the ES12 loader integrates properly with the
existing analysis components and can be used for degradation analysis.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from nasa_pcoe_eda.data.loader import DataLoader
from nasa_pcoe_eda.analysis.statistics import StatisticsAnalyzer
from nasa_pcoe_eda.analysis.correlation import CorrelationAnalyzer
from nasa_pcoe_eda.analysis.outliers import OutlierDetector
from nasa_pcoe_eda.analysis.rul_features import RULFeatureAnalyzer

def test_es12_with_statistics_analyzer():
    """Test ES12 data with StatisticsAnalyzer."""
    print("=== Testing ES12 with StatisticsAnalyzer ===")
    
    file_path = Path("data/raw/ES12.mat")
    if not file_path.exists():
        print(f"ES12 data file not found: {file_path}")
        return False
    
    try:
        # Load ES12 data
        loader = DataLoader()
        df = loader.load_dataset(file_path)
        print(f"Loaded ES12 data: {df.shape}")
        
        # Test statistics analyzer
        stats_analyzer = StatisticsAnalyzer()
        
        # Compute descriptive statistics
        stats = stats_analyzer.compute_descriptive_stats(df)
        print(f"Computed statistics for {len(stats)} features")
        
        # Check some key statistics
        if 'vl_mean' in stats:
            vl_stats = stats['vl_mean']
            print(f"VL mean statistics: mean={vl_stats.mean:.3f}, std={vl_stats.std:.3f}")
        
        # Analyze missing values
        missing_report = stats_analyzer.analyze_missing_values(df)
        print(f"Missing values analysis: {missing_report.total_missing} total missing")
        
        # Identify data types
        data_types = stats_analyzer.identify_data_types(df)
        print(f"Identified {len(data_types)} data types")
        
        return True
        
    except Exception as e:
        print(f"Statistics analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_es12_with_correlation_analyzer():
    """Test ES12 data with CorrelationAnalyzer."""
    print("\n=== Testing ES12 with CorrelationAnalyzer ===")
    
    file_path = Path("data/raw/ES12.mat")
    if not file_path.exists():
        print(f"ES12 data file not found: {file_path}")
        return False
    
    try:
        # Load ES12 data
        loader = DataLoader()
        df = loader.load_dataset(file_path)
        
        # Test correlation analyzer
        corr_analyzer = CorrelationAnalyzer()
        
        # Compute correlation matrix
        corr_matrix = corr_analyzer.compute_correlation_matrix(df)
        print(f"Computed correlation matrix: {corr_matrix.shape}")
        
        # Identify high correlations
        high_corrs = corr_analyzer.identify_high_correlations(corr_matrix, threshold=0.7)
        print(f"Found {len(high_corrs)} high correlations (>0.7)")
        
        # Show some high correlations
        for i, (feat1, feat2, corr) in enumerate(high_corrs[:5]):
            print(f"  {feat1} <-> {feat2}: {corr:.3f}")
        
        return True
        
    except Exception as e:
        print(f"Correlation analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_es12_with_outlier_detector():
    """Test ES12 data with OutlierDetector."""
    print("\n=== Testing ES12 with OutlierDetector ===")
    
    file_path = Path("data/raw/ES12.mat")
    if not file_path.exists():
        print(f"ES12 data file not found: {file_path}")
        return False
    
    try:
        # Load ES12 data
        loader = DataLoader()
        df = loader.load_dataset(file_path)
        
        # Test outlier detector
        outlier_detector = OutlierDetector()
        
        # Detect outliers using IQR method
        outliers_iqr = outlier_detector.detect_outliers_iqr(df)
        print(f"IQR outlier detection completed for {len(outliers_iqr)} features")
        
        # Detect outliers using Z-score method
        outliers_zscore = outlier_detector.detect_outliers_zscore(df)
        print(f"Z-score outlier detection completed for {len(outliers_zscore)} features")
        
        # Summarize outliers
        outlier_summary = outlier_detector.summarize_outliers(outliers_iqr)
        print(f"Outlier summary: {outlier_summary.outlier_counts}")
        
        return True
        
    except Exception as e:
        print(f"Outlier detector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_es12_with_rul_analyzer():
    """Test ES12 data with RULFeatureAnalyzer."""
    print("\n=== Testing ES12 with RULFeatureAnalyzer ===")
    
    file_path = Path("data/raw/ES12.mat")
    if not file_path.exists():
        print(f"ES12 data file not found: {file_path}")
        return False
    
    try:
        # Load ES12 data
        loader = DataLoader()
        df = loader.load_dataset(file_path)
        
        # Test RUL feature analyzer
        rul_analyzer = RULFeatureAnalyzer()
        
        # Identify degradation features
        degradation_features = rul_analyzer.identify_degradation_features(df)
        print(f"Identified {len(degradation_features)} degradation features")
        print(f"Degradation features: {degradation_features[:5]}")  # Show first 5
        
        # Compute degradation rates
        if degradation_features:
            degradation_rates = rul_analyzer.compute_degradation_rates(df, degradation_features[:5])
            print(f"Computed degradation rates for {len(degradation_rates)} features")
            
            for feat, rate in list(degradation_rates.items())[:3]:
                print(f"  {feat}: {rate:.6f}")
        
        # Use cycle as a proxy for RUL (higher cycle = lower RUL)
        if 'cycle' in df.columns:
            # Create a synthetic RUL column (max_cycle - current_cycle)
            max_cycle = df['cycle'].max()
            df['synthetic_rul'] = max_cycle - df['cycle']
            
            # Rank features for RUL prediction
            rul_features = rul_analyzer.rank_features_for_rul(df, 'synthetic_rul')
            print(f"Ranked {len(rul_features)} features for RUL prediction")
            
            # Show top features
            for i, (feat, score) in enumerate(rul_features[:5]):
                print(f"  {i+1}. {feat}: {score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"RUL analyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_degradation_analysis():
    """Test degradation analysis specific to capacitor data."""
    print("\n=== Testing Degradation Analysis ===")
    
    file_path = Path("data/raw/ES12.mat")
    if not file_path.exists():
        print(f"ES12 data file not found: {file_path}")
        return False
    
    try:
        # Load ES12 data
        loader = DataLoader()
        df = loader.load_dataset(file_path)
        
        # Analyze degradation patterns per capacitor
        print("Degradation analysis per capacitor:")
        
        for cap in sorted(df['capacitor'].unique()):
            cap_data = df[df['capacitor'] == cap].copy()
            cap_data = cap_data.sort_values('cycle')
            
            # Calculate degradation metrics
            if len(cap_data) > 10:  # Need sufficient data
                # Voltage ratio degradation
                if 'voltage_ratio' in cap_data.columns:
                    initial_ratio = cap_data['voltage_ratio'].iloc[0]
                    final_ratio = cap_data['voltage_ratio'].iloc[-1]
                    ratio_change = (final_ratio - initial_ratio) / initial_ratio * 100
                    
                    print(f"  {cap}: Voltage ratio change: {ratio_change:.2f}%")
                
                # VL mean degradation
                if 'vl_mean' in cap_data.columns:
                    initial_vl = cap_data['vl_mean'].iloc[0]
                    final_vl = cap_data['vl_mean'].iloc[-1]
                    vl_change = (final_vl - initial_vl) / initial_vl * 100
                    
                    print(f"  {cap}: VL mean change: {vl_change:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"Degradation analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all integration tests."""
    print("Testing ES12 integration with EDA system...")
    
    tests = [
        test_es12_with_statistics_analyzer,
        test_es12_with_correlation_analyzer,
        test_es12_with_outlier_detector,
        test_es12_with_rul_analyzer,
        test_degradation_analysis,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n=== Integration Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All integration tests passed!")
        return 0
    else:
        print("✗ Some integration tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())