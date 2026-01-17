"""
Dataset Summary Generator for RUL Prediction Model

This module generates comprehensive summaries of the train/val/test datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List


class DatasetSummaryGenerator:
    """Generate comprehensive dataset summaries."""
    
    def __init__(self):
        """Initialize the summary generator."""
        self.metadata_columns = ['capacitor_id', 'cycle', 'is_abnormal', 'rul']
        
    def generate_summary(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_path: str
    ) -> str:
        """
        Generate a comprehensive dataset summary.
        
        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            output_path: Path to save the summary
            
        Returns:
            Summary text
        """
        summary_lines = []
        
        # Header
        summary_lines.append("=" * 80)
        summary_lines.append("RUL PREDICTION MODEL - DATASET SUMMARY")
        summary_lines.append("=" * 80)
        summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_lines.append("")
        
        # Overall statistics
        summary_lines.append("=" * 80)
        summary_lines.append("1. DATASET OVERVIEW")
        summary_lines.append("=" * 80)
        summary_lines.append("")
        
        total_samples = len(train_df) + len(val_df) + len(test_df)
        summary_lines.append(f"Total Samples:       {total_samples:,}")
        summary_lines.append(f"  - Training:        {len(train_df):,} ({len(train_df)/total_samples*100:.1f}%)")
        summary_lines.append(f"  - Validation:      {len(val_df):,} ({len(val_df)/total_samples*100:.1f}%)")
        summary_lines.append(f"  - Test:            {len(test_df):,} ({len(test_df)/total_samples*100:.1f}%)")
        summary_lines.append("")
        
        summary_lines.append(f"Total Features:      {train_df.shape[1]}")
        feature_cols = [col for col in train_df.columns if col not in self.metadata_columns]
        summary_lines.append(f"  - Feature columns: {len(feature_cols)}")
        summary_lines.append(f"  - Metadata:        {len(self.metadata_columns)}")
        summary_lines.append("")
        
        # Capacitor distribution
        summary_lines.append("=" * 80)
        summary_lines.append("2. CAPACITOR DISTRIBUTION")
        summary_lines.append("=" * 80)
        summary_lines.append("")
        
        for name, df in [("Training", train_df), ("Validation", val_df), ("Test", test_df)]:
            cap_counts = df['capacitor_id'].value_counts().sort_index()
            summary_lines.append(f"{name} Set:")
            for cap_id, count in cap_counts.items():
                summary_lines.append(f"  {cap_id}: {count:3d} samples")
            summary_lines.append("")
        
        # Cycle range
        summary_lines.append("=" * 80)
        summary_lines.append("3. CYCLE RANGE")
        summary_lines.append("=" * 80)
        summary_lines.append("")
        
        for name, df in [("Training", train_df), ("Validation", val_df), ("Test", test_df)]:
            cycle_min = df['cycle'].min()
            cycle_max = df['cycle'].max()
            summary_lines.append(f"{name} Set: Cycles {cycle_min}-{cycle_max}")
        summary_lines.append("")
        
        # Label distribution
        summary_lines.append("=" * 80)
        summary_lines.append("4. LABEL DISTRIBUTION")
        summary_lines.append("=" * 80)
        summary_lines.append("")
        
        for name, df in [("Training", train_df), ("Validation", val_df), ("Test", test_df)]:
            normal_count = (df['is_abnormal'] == 0).sum()
            abnormal_count = (df['is_abnormal'] == 1).sum()
            summary_lines.append(f"{name} Set:")
            summary_lines.append(f"  Normal (0):   {normal_count:3d} ({normal_count/len(df)*100:.1f}%)")
            summary_lines.append(f"  Abnormal (1): {abnormal_count:3d} ({abnormal_count/len(df)*100:.1f}%)")
            summary_lines.append("")
        
        # RUL statistics
        summary_lines.append("=" * 80)
        summary_lines.append("5. RUL STATISTICS")
        summary_lines.append("=" * 80)
        summary_lines.append("")
        
        for name, df in [("Training", train_df), ("Validation", val_df), ("Test", test_df)]:
            rul_stats = df['rul'].describe()
            summary_lines.append(f"{name} Set:")
            summary_lines.append(f"  Mean:   {rul_stats['mean']:.2f}")
            summary_lines.append(f"  Std:    {rul_stats['std']:.2f}")
            summary_lines.append(f"  Min:    {rul_stats['min']:.0f}")
            summary_lines.append(f"  25%:    {rul_stats['25%']:.0f}")
            summary_lines.append(f"  Median: {rul_stats['50%']:.0f}")
            summary_lines.append(f"  75%:    {rul_stats['75%']:.0f}")
            summary_lines.append(f"  Max:    {rul_stats['max']:.0f}")
            summary_lines.append("")
        
        # Feature statistics
        summary_lines.append("=" * 80)
        summary_lines.append("6. FEATURE STATISTICS (Training Set)")
        summary_lines.append("=" * 80)
        summary_lines.append("")
        
        feature_stats = train_df[feature_cols].describe()
        summary_lines.append(f"{'Feature':<25} {'Mean':>12} {'Std':>12} {'Min':>12} {'Max':>12}")
        summary_lines.append("-" * 80)
        
        for col in feature_cols:
            stats = feature_stats[col]
            summary_lines.append(
                f"{col:<25} {stats['mean']:>12.4f} {stats['std']:>12.4f} "
                f"{stats['min']:>12.4f} {stats['max']:>12.4f}"
            )
        summary_lines.append("")
        
        # Missing values check
        summary_lines.append("=" * 80)
        summary_lines.append("7. DATA QUALITY")
        summary_lines.append("=" * 80)
        summary_lines.append("")
        
        for name, df in [("Training", train_df), ("Validation", val_df), ("Test", test_df)]:
            missing_count = df.isnull().sum().sum()
            summary_lines.append(f"{name} Set:")
            summary_lines.append(f"  Missing values: {missing_count}")
            if missing_count > 0:
                missing_cols = df.columns[df.isnull().any()].tolist()
                summary_lines.append(f"  Columns with missing values: {missing_cols}")
            summary_lines.append("")
        
        # Data split strategy
        summary_lines.append("=" * 80)
        summary_lines.append("8. DATA SPLIT STRATEGY")
        summary_lines.append("=" * 80)
        summary_lines.append("")
        summary_lines.append("Hybrid Split Strategy:")
        summary_lines.append("  - Training:   C1-C5, Cycles 1-150 (750 samples)")
        summary_lines.append("  - Validation: C6, Cycles 1-150 (150 samples)")
        summary_lines.append("  - Test:       C7-C8, Cycles 1-200 (400 samples)")
        summary_lines.append("")
        summary_lines.append("Rationale:")
        summary_lines.append("  - Considers both capacitor and cycle dimensions")
        summary_lines.append("  - Evaluates generalization to unseen capacitors")
        summary_lines.append("  - Ensures sufficient data for training")
        summary_lines.append("")
        
        # Feature scaling info
        summary_lines.append("=" * 80)
        summary_lines.append("9. FEATURE SCALING")
        summary_lines.append("=" * 80)
        summary_lines.append("")
        summary_lines.append("Scaling Method: StandardScaler")
        summary_lines.append(f"  - Scaled features: {len(feature_cols)}")
        summary_lines.append(f"  - Excluded metadata: {', '.join(self.metadata_columns)}")
        summary_lines.append("  - Fitted on: Training set")
        summary_lines.append("  - Applied to: Training, Validation, Test sets")
        summary_lines.append("  - Scaler saved to: output/models/scaler.pkl")
        summary_lines.append("")
        
        # Footer
        summary_lines.append("=" * 80)
        summary_lines.append("END OF SUMMARY")
        summary_lines.append("=" * 80)
        
        # Join all lines
        summary_text = "\n".join(summary_lines)
        
        # Save to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(summary_text)
        
        print(f"✓ Dataset summary saved to: {output_path}")
        
        return summary_text


def generate_and_save_summary(
    train_path: str,
    val_path: str,
    test_path: str,
    output_path: str
) -> str:
    """
    Generate and save dataset summary.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data
        test_path: Path to test data
        output_path: Path to save the summary
        
    Returns:
        Summary text
    """
    print("=" * 80)
    print("Dataset Summary Generation")
    print("=" * 80)
    print("")
    
    # Load datasets
    print("Loading datasets...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    print(f"  ✓ Train: {train_df.shape}")
    print(f"  ✓ Val:   {val_df.shape}")
    print(f"  ✓ Test:  {test_df.shape}")
    print("")
    
    # Generate summary
    print("Generating summary...")
    generator = DatasetSummaryGenerator()
    summary_text = generator.generate_summary(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        output_path=output_path
    )
    
    print("")
    print("=" * 80)
    print("Summary Generation Complete!")
    print("=" * 80)
    
    return summary_text


if __name__ == "__main__":
    # Use scaled datasets
    train_path = "output/features/train_scaled.csv"
    val_path = "output/features/val_scaled.csv"
    test_path = "output/features/test_scaled.csv"
    output_path = "output/features/dataset_summary.txt"
    
    # Generate summary
    summary = generate_and_save_summary(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        output_path=output_path
    )
    
    # Print summary to console
    print("\n" + summary)
