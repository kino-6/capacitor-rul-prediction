"""
Dataset splitting for RUL prediction.

This module provides functionality to split the dataset into train/validation/test
sets using a hybrid splitting strategy that considers both capacitor ID and cycle range.
"""

from typing import Tuple, Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path


class DatasetSplitter:
    """Split dataset using hybrid strategy (capacitor ID + cycle range)."""
    
    def __init__(
        self,
        train_capacitors: List[str] = None,
        val_capacitors: List[str] = None,
        test_capacitors: List[str] = None,
        train_cycle_range: Tuple[int, int] = (1, 150),
        val_cycle_range: Tuple[int, int] = (1, 150),
        test_cycle_range: Tuple[int, int] = (1, 200)
    ):
        """
        Initialize the dataset splitter with hybrid splitting strategy.
        
        Args:
            train_capacitors: List of capacitor IDs for training (default: C1-C5)
            val_capacitors: List of capacitor IDs for validation (default: C6)
            test_capacitors: List of capacitor IDs for testing (default: C7-C8)
            train_cycle_range: Cycle range for training (start, end) inclusive
            val_cycle_range: Cycle range for validation (start, end) inclusive
            test_cycle_range: Cycle range for testing (start, end) inclusive
        """
        # Default capacitor assignments
        self.train_capacitors = train_capacitors or ['ES12C1', 'ES12C2', 'ES12C3', 'ES12C4', 'ES12C5']
        self.val_capacitors = val_capacitors or ['ES12C6']
        self.test_capacitors = test_capacitors or ['ES12C7', 'ES12C8']
        
        # Cycle ranges
        self.train_cycle_range = train_cycle_range
        self.val_cycle_range = val_cycle_range
        self.test_cycle_range = test_cycle_range
    
    def split(
        self,
        df: pd.DataFrame,
        capacitor_col: str = 'capacitor_id',
        cycle_col: str = 'cycle'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split the dataset into train/validation/test sets.
        
        The hybrid strategy splits by:
        1. Capacitor ID: Different capacitors for train/val/test
        2. Cycle range: Different cycle ranges for each split
        
        This ensures:
        - Temporal consistency (no future data leakage within capacitors)
        - Generalization testing (test on unseen capacitors)
        
        Args:
            df: DataFrame with features and labels
            capacitor_col: Name of the capacitor ID column
            cycle_col: Name of the cycle column
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Validate input
        if capacitor_col not in df.columns:
            raise ValueError(f"Column '{capacitor_col}' not found in DataFrame")
        if cycle_col not in df.columns:
            raise ValueError(f"Column '{cycle_col}' not found in DataFrame")
        
        # Split by capacitor and cycle range
        train_df = self._filter_by_capacitor_and_cycle(
            df, self.train_capacitors, self.train_cycle_range, capacitor_col, cycle_col
        )
        
        val_df = self._filter_by_capacitor_and_cycle(
            df, self.val_capacitors, self.val_cycle_range, capacitor_col, cycle_col
        )
        
        test_df = self._filter_by_capacitor_and_cycle(
            df, self.test_capacitors, self.test_cycle_range, capacitor_col, cycle_col
        )
        
        return train_df, val_df, test_df
    
    def _filter_by_capacitor_and_cycle(
        self,
        df: pd.DataFrame,
        capacitors: List[str],
        cycle_range: Tuple[int, int],
        capacitor_col: str,
        cycle_col: str
    ) -> pd.DataFrame:
        """
        Filter DataFrame by capacitor IDs and cycle range.
        
        Args:
            df: DataFrame to filter
            capacitors: List of capacitor IDs to include
            cycle_range: Tuple of (start_cycle, end_cycle) inclusive
            capacitor_col: Name of the capacitor ID column
            cycle_col: Name of the cycle column
        
        Returns:
            Filtered DataFrame
        """
        start_cycle, end_cycle = cycle_range
        
        # Filter by capacitor ID and cycle range
        mask = (
            df[capacitor_col].isin(capacitors) &
            (df[cycle_col] >= start_cycle) &
            (df[cycle_col] <= end_cycle)
        )
        
        return df[mask].copy()
    
    def get_split_statistics(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        capacitor_col: str = 'capacitor_id',
        cycle_col: str = 'cycle'
    ) -> Dict[str, Dict]:
        """
        Get statistics about the dataset splits.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            capacitor_col: Name of the capacitor ID column
            cycle_col: Name of the cycle column
        
        Returns:
            Dictionary with statistics for each split
        """
        def get_stats(df: pd.DataFrame, split_name: str) -> Dict:
            """Get statistics for a single split."""
            if len(df) == 0:
                return {
                    'split': split_name,
                    'n_samples': 0,
                    'n_capacitors': 0,
                    'capacitors': [],
                    'cycle_range': (0, 0),
                    'n_normal': 0,
                    'n_abnormal': 0,
                    'normal_ratio': 0.0,
                    'mean_rul': 0.0
                }
            
            capacitors = df[capacitor_col].unique().tolist()
            cycle_min = df[cycle_col].min()
            cycle_max = df[cycle_col].max()
            
            stats = {
                'split': split_name,
                'n_samples': len(df),
                'n_capacitors': len(capacitors),
                'capacitors': sorted(capacitors),
                'cycle_range': (int(cycle_min), int(cycle_max)),
            }
            
            # Add label statistics if available
            if 'is_abnormal' in df.columns:
                stats['n_normal'] = (df['is_abnormal'] == 0).sum()
                stats['n_abnormal'] = (df['is_abnormal'] == 1).sum()
                stats['normal_ratio'] = (df['is_abnormal'] == 0).mean()
            
            if 'rul' in df.columns:
                stats['mean_rul'] = df['rul'].mean()
            
            return stats
        
        return {
            'train': get_stats(train_df, 'train'),
            'val': get_stats(val_df, 'val'),
            'test': get_stats(test_df, 'test')
        }
    
    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: str,
        train_filename: str = 'train.csv',
        val_filename: str = 'val.csv',
        test_filename: str = 'test.csv'
    ) -> Dict[str, str]:
        """
        Save the split datasets to CSV files.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            output_dir: Directory to save the files
            train_filename: Filename for training set
            val_filename: Filename for validation set
            test_filename: Filename for test set
        
        Returns:
            Dictionary with paths to saved files
        """
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save files
        train_path = output_path / train_filename
        val_path = output_path / val_filename
        test_path = output_path / test_filename
        
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        return {
            'train': str(train_path),
            'val': str(val_path),
            'test': str(test_path)
        }
    
    def print_split_summary(
        self,
        stats: Dict[str, Dict]
    ) -> None:
        """
        Print a formatted summary of the dataset splits.
        
        Args:
            stats: Statistics dictionary from get_split_statistics()
        """
        print("\n" + "="*70)
        print("Dataset Split Summary")
        print("="*70)
        
        for split_name in ['train', 'val', 'test']:
            split_stats = stats[split_name]
            print(f"\n{split_name.upper()} SET:")
            print(f"  Samples: {split_stats['n_samples']}")
            print(f"  Capacitors: {split_stats['n_capacitors']} {split_stats['capacitors']}")
            print(f"  Cycle Range: {split_stats['cycle_range'][0]}-{split_stats['cycle_range'][1]}")
            
            if 'n_normal' in split_stats:
                print(f"  Normal: {split_stats['n_normal']} ({split_stats['normal_ratio']:.1%})")
                print(f"  Abnormal: {split_stats['n_abnormal']} ({1-split_stats['normal_ratio']:.1%})")
            
            if 'mean_rul' in split_stats:
                print(f"  Mean RUL: {split_stats['mean_rul']:.1f}")
        
        print("\n" + "="*70)
        print(f"Total Samples: {sum(s['n_samples'] for s in stats.values())}")
        print("="*70 + "\n")


def split_dataset(
    input_path: str,
    output_dir: str,
    train_capacitors: List[str] = None,
    val_capacitors: List[str] = None,
    test_capacitors: List[str] = None,
    train_cycle_range: Tuple[int, int] = (1, 150),
    val_cycle_range: Tuple[int, int] = (1, 150),
    test_cycle_range: Tuple[int, int] = (1, 200),
    capacitor_col: str = 'capacitor_id',
    cycle_col: str = 'cycle'
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to split a dataset from a CSV file.
    
    Args:
        input_path: Path to input CSV file with features and labels
        output_dir: Directory to save split datasets
        train_capacitors: List of capacitor IDs for training
        val_capacitors: List of capacitor IDs for validation
        test_capacitors: List of capacitor IDs for testing
        train_cycle_range: Cycle range for training (start, end) inclusive
        val_cycle_range: Cycle range for validation (start, end) inclusive
        test_cycle_range: Cycle range for testing (start, end) inclusive
        capacitor_col: Name of the capacitor ID column
        cycle_col: Name of the cycle column
    
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Load data
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} samples with {len(df.columns)} features")
    
    # Create splitter
    splitter = DatasetSplitter(
        train_capacitors=train_capacitors,
        val_capacitors=val_capacitors,
        test_capacitors=test_capacitors,
        train_cycle_range=train_cycle_range,
        val_cycle_range=val_cycle_range,
        test_cycle_range=test_cycle_range
    )
    
    # Split dataset
    print("\nSplitting dataset...")
    train_df, val_df, test_df = splitter.split(df, capacitor_col, cycle_col)
    
    # Get and print statistics
    stats = splitter.get_split_statistics(train_df, val_df, test_df, capacitor_col, cycle_col)
    splitter.print_split_summary(stats)
    
    # Save splits
    print(f"Saving splits to: {output_dir}")
    paths = splitter.save_splits(train_df, val_df, test_df, output_dir)
    
    for split_name, path in paths.items():
        print(f"  {split_name}: {path}")
    
    return train_df, val_df, test_df
