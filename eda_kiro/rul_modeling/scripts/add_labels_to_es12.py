"""
Script to add labels to ES12 features dataset.

This script demonstrates the usage of LabelGenerator to add
anomaly detection labels and RUL values to the extracted features.
"""

from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_preparation.label_generator import add_labels_to_features


def main():
    """Add labels to ES12 features."""
    # Define paths
    features_path = 'output/features/es12_features.csv'
    output_path = 'output/features/es12_features_with_labels.csv'
    
    print("="*70)
    print("Adding Labels to ES12 Features")
    print("="*70)
    print(f"Input: {features_path}")
    print(f"Output: {output_path}")
    print(f"Strategy: cycle_based")
    print(f"Total cycles: 200")
    print(f"Normal cycles: 1-100 (50%)")
    print(f"Abnormal cycles: 101-200 (50%)")
    print("="*70)
    print()
    
    # Add labels
    labeled_df = add_labels_to_features(
        features_path=features_path,
        output_path=output_path,
        total_cycles=200,
        strategy='cycle_based'
    )
    
    print("\n" + "="*70)
    print("Label Generation Complete!")
    print("="*70)
    print(f"Total samples: {len(labeled_df)}")
    print(f"Total features: {len(labeled_df.columns)}")
    print(f"Normal samples: {(labeled_df['is_abnormal'] == 0).sum()}")
    print(f"Abnormal samples: {(labeled_df['is_abnormal'] == 1).sum()}")
    print("="*70)
    
    # Show sample data
    print("\nSample data (first 5 rows):")
    print(labeled_df[['capacitor_id', 'cycle', 'is_abnormal', 'rul']].head())
    
    print("\nSample data (around boundary - cycles 98-103):")
    boundary_df = labeled_df[
        (labeled_df['capacitor_id'] == 'ES12C1') &
        (labeled_df['cycle'] >= 98) &
        (labeled_df['cycle'] <= 103)
    ]
    print(boundary_df[['capacitor_id', 'cycle', 'is_abnormal', 'rul']])
    
    print("\nSample data (last 5 rows):")
    print(labeled_df[['capacitor_id', 'cycle', 'is_abnormal', 'rul']].tail())


if __name__ == '__main__':
    main()
