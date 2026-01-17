"""
Quality checker for extracted features.

This module provides functionality to check data quality including:
- Missing values detection
- Outlier detection
- Statistical summary generation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats


class FeatureQualityChecker:
    """Check quality of extracted features."""
    
    def __init__(self, features_path: str):
        """
        Initialize quality checker.
        
        Args:
            features_path: Path to features CSV file
        """
        self.features_path = Path(features_path)
        self.df = None
        self.report = []
        
    def load_data(self) -> None:
        """Load features data."""
        print(f"Loading data from {self.features_path}...")
        self.df = pd.read_csv(self.features_path)
        print(f"Loaded {len(self.df)} rows × {len(self.df.columns)} columns")
        
    def check_missing_values(self) -> Dict[str, int]:
        """
        Check for missing values in the dataset.
        
        Returns:
            Dictionary mapping column names to missing value counts
        """
        print("\n" + "="*80)
        print("MISSING VALUES CHECK")
        print("="*80)
        
        missing = self.df.isnull().sum()
        missing_dict = missing[missing > 0].to_dict()
        
        if len(missing_dict) == 0:
            msg = "✓ No missing values found in any column"
            print(msg)
            self.report.append(msg)
        else:
            msg = f"✗ Found missing values in {len(missing_dict)} columns:"
            print(msg)
            self.report.append(msg)
            for col, count in missing_dict.items():
                pct = (count / len(self.df)) * 100
                msg = f"  - {col}: {count} ({pct:.2f}%)"
                print(msg)
                self.report.append(msg)
                
        return missing_dict
    
    def detect_outliers_iqr(self, column: str, multiplier: float = 1.5) -> Tuple[List[int], float, float]:
        """
        Detect outliers using IQR method.
        
        Args:
            column: Column name to check
            multiplier: IQR multiplier (default: 1.5)
            
        Returns:
            Tuple of (outlier_indices, lower_bound, upper_bound)
        """
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = self.df[
            (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        ].index.tolist()
        
        return outliers, lower_bound, upper_bound
    
    def detect_outliers_zscore(self, column: str, threshold: float = 3.0) -> List[int]:
        """
        Detect outliers using Z-score method.
        
        Args:
            column: Column name to check
            threshold: Z-score threshold (default: 3.0)
            
        Returns:
            List of outlier indices
        """
        z_scores = np.abs(stats.zscore(self.df[column]))
        outliers = self.df[z_scores > threshold].index.tolist()
        return outliers
    
    def check_outliers(self) -> Dict[str, Dict]:
        """
        Check for outliers in all numeric columns.
        
        Returns:
            Dictionary mapping column names to outlier information
        """
        print("\n" + "="*80)
        print("OUTLIER DETECTION")
        print("="*80)
        
        # Get numeric columns (exclude identifiers)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['cycle', 'cycle_number', 'cycle_normalized']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        outlier_summary = {}
        
        for col in feature_cols:
            # Use IQR method
            outliers_iqr, lower, upper = self.detect_outliers_iqr(col)
            
            # Use Z-score method
            outliers_zscore = self.detect_outliers_zscore(col)
            
            # Store results
            outlier_summary[col] = {
                'iqr_outliers': len(outliers_iqr),
                'iqr_bounds': (lower, upper),
                'zscore_outliers': len(outliers_zscore)
            }
        
        # Report columns with significant outliers (>5% by IQR)
        significant_outliers = {
            col: info for col, info in outlier_summary.items()
            if info['iqr_outliers'] > len(self.df) * 0.05
        }
        
        if len(significant_outliers) == 0:
            msg = "✓ No significant outliers detected (>5% threshold)"
            print(msg)
            self.report.append(msg)
        else:
            msg = f"⚠ Found significant outliers in {len(significant_outliers)} columns:"
            print(msg)
            self.report.append(msg)
            for col, info in significant_outliers.items():
                pct = (info['iqr_outliers'] / len(self.df)) * 100
                msg = f"  - {col}: {info['iqr_outliers']} outliers ({pct:.2f}%)"
                print(msg)
                self.report.append(msg)
        
        # Summary statistics
        total_iqr = sum(info['iqr_outliers'] for info in outlier_summary.values())
        total_zscore = sum(info['zscore_outliers'] for info in outlier_summary.values())
        
        msg = f"\nOutlier Summary:"
        print(msg)
        self.report.append(msg)
        msg = f"  - Total outliers (IQR method): {total_iqr}"
        print(msg)
        self.report.append(msg)
        msg = f"  - Total outliers (Z-score method): {total_zscore}"
        print(msg)
        self.report.append(msg)
        
        return outlier_summary
    
    def generate_statistical_summary(self) -> pd.DataFrame:
        """
        Generate statistical summary for all features.
        
        Returns:
            DataFrame with statistical summary
        """
        print("\n" + "="*80)
        print("STATISTICAL SUMMARY")
        print("="*80)
        
        # Get numeric columns (exclude identifiers)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['cycle', 'cycle_number', 'cycle_normalized']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Calculate statistics
        summary = self.df[feature_cols].describe().T
        
        # Add additional statistics
        summary['skewness'] = self.df[feature_cols].skew()
        summary['kurtosis'] = self.df[feature_cols].kurtosis()
        summary['cv'] = (summary['std'] / summary['mean']).abs()  # Coefficient of variation
        
        # Round for readability
        summary = summary.round(4)
        
        print("\nBasic Statistics (first 10 features):")
        print(summary.head(10).to_string())
        
        msg = f"\n✓ Generated statistical summary for {len(feature_cols)} features"
        print(msg)
        self.report.append(msg)
        
        return summary
    
    def check_data_distribution(self) -> Dict[str, Dict]:
        """
        Check data distribution by capacitor.
        
        Returns:
            Dictionary with distribution information
        """
        print("\n" + "="*80)
        print("DATA DISTRIBUTION")
        print("="*80)
        
        # Check samples per capacitor
        cap_counts = self.df['capacitor_id'].value_counts().sort_index()
        
        msg = "\nSamples per capacitor:"
        print(msg)
        self.report.append(msg)
        
        for cap_id, count in cap_counts.items():
            msg = f"  - {cap_id}: {count} cycles"
            print(msg)
            self.report.append(msg)
        
        # Check cycle range per capacitor
        msg = "\nCycle range per capacitor:"
        print(msg)
        self.report.append(msg)
        
        for cap_id in sorted(self.df['capacitor_id'].unique()):
            cap_data = self.df[self.df['capacitor_id'] == cap_id]
            min_cycle = cap_data['cycle'].min()
            max_cycle = cap_data['cycle'].max()
            msg = f"  - {cap_id}: cycles {min_cycle} to {max_cycle}"
            print(msg)
            self.report.append(msg)
        
        distribution_info = {
            'capacitor_counts': cap_counts.to_dict(),
            'total_samples': len(self.df),
            'total_capacitors': len(cap_counts)
        }
        
        return distribution_info
    
    def check_feature_ranges(self) -> Dict[str, Tuple[float, float]]:
        """
        Check feature value ranges.
        
        Returns:
            Dictionary mapping feature names to (min, max) tuples
        """
        print("\n" + "="*80)
        print("FEATURE RANGES")
        print("="*80)
        
        # Get numeric columns (exclude identifiers)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_cols = ['cycle', 'cycle_number', 'cycle_normalized']
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        ranges = {}
        
        msg = "\nFeature value ranges (first 10 features):"
        print(msg)
        self.report.append(msg)
        
        for i, col in enumerate(feature_cols[:10]):
            min_val = self.df[col].min()
            max_val = self.df[col].max()
            ranges[col] = (min_val, max_val)
            msg = f"  - {col}: [{min_val:.4f}, {max_val:.4f}]"
            print(msg)
            self.report.append(msg)
        
        # Get all ranges
        for col in feature_cols[10:]:
            min_val = self.df[col].min()
            max_val = self.df[col].max()
            ranges[col] = (min_val, max_val)
        
        return ranges
    
    def run_all_checks(self) -> None:
        """Run all quality checks."""
        self.load_data()
        
        # Add header to report
        self.report.append("="*80)
        self.report.append("FEATURE QUALITY REPORT")
        self.report.append("="*80)
        self.report.append(f"Dataset: {self.features_path}")
        self.report.append(f"Total samples: {len(self.df)}")
        self.report.append(f"Total features: {len(self.df.columns)}")
        self.report.append("")
        
        # Run checks
        missing = self.check_missing_values()
        outliers = self.check_outliers()
        summary = self.generate_statistical_summary()
        distribution = self.check_data_distribution()
        ranges = self.check_feature_ranges()
        
        # Overall assessment
        print("\n" + "="*80)
        print("OVERALL ASSESSMENT")
        print("="*80)
        
        issues = []
        if len(missing) > 0:
            issues.append(f"Missing values in {len(missing)} columns")
        
        significant_outliers = sum(
            1 for info in outliers.values()
            if info['iqr_outliers'] > len(self.df) * 0.05
        )
        if significant_outliers > 0:
            issues.append(f"Significant outliers in {significant_outliers} columns")
        
        if len(issues) == 0:
            msg = "✓ Data quality is GOOD - No critical issues found"
            print(msg)
            self.report.append("")
            self.report.append(msg)
        else:
            msg = f"⚠ Data quality issues found:"
            print(msg)
            self.report.append("")
            self.report.append(msg)
            for issue in issues:
                msg = f"  - {issue}"
                print(msg)
                self.report.append(msg)
        
        print("\n" + "="*80)
        
    def save_report(self, output_path: str) -> None:
        """
        Save quality report to file.
        
        Args:
            output_path: Path to save report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.report))
        
        print(f"\n✓ Quality report saved to: {output_path}")


def main():
    """Main function to run quality checks."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Check feature quality')
    parser.add_argument(
        '--input',
        type=str,
        default='output/features/es12_features.csv',
        help='Input features CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/features/es12_quality_report.txt',
        help='Output quality report file'
    )
    
    args = parser.parse_args()
    
    # Run quality checks
    checker = FeatureQualityChecker(args.input)
    checker.run_all_checks()
    checker.save_report(args.output)


if __name__ == '__main__':
    main()
