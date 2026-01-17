"""
Label generation for RUL prediction.

This module provides functionality to generate labels for anomaly detection
and RUL prediction based on different labeling strategies.
"""

from typing import Literal, Optional
import pandas as pd
import numpy as np


class LabelGenerator:
    """Generate labels for anomaly detection and RUL prediction."""
    
    def __init__(
        self,
        total_cycles: int = 200,
        strategy: Literal['cycle_based', 'threshold_based'] = 'cycle_based'
    ):
        """
        Initialize the label generator.
        
        Args:
            total_cycles: Total number of cycles per capacitor (default: 200)
            strategy: Labeling strategy to use (default: 'cycle_based')
                - 'cycle_based': First 50% cycles are Normal, last 50% are Abnormal
                - 'threshold_based': Based on degradation threshold
        """
        self.total_cycles = total_cycles
        self.strategy = strategy
    
    def generate_cycle_based_labels(
        self,
        df: pd.DataFrame,
        cycle_col: str = 'cycle',
        normal_ratio: float = 0.5
    ) -> pd.DataFrame:
        """
        Generate labels using cycle-based strategy.
        
        First 50% of cycles are labeled as Normal (0), last 50% as Abnormal (1).
        RUL is calculated as: total_cycles - cycle_number
        
        Args:
            df: DataFrame with features
            cycle_col: Name of the cycle column
            normal_ratio: Ratio of cycles to label as normal (default: 0.5)
        
        Returns:
            DataFrame with added 'is_abnormal' and 'rul' columns
        """
        df = df.copy()
        
        # Calculate threshold cycle
        threshold_cycle = self.total_cycles * normal_ratio
        
        # Generate is_abnormal label
        # Cycles 1-100: Normal (0), Cycles 101-200: Abnormal (1)
        df['is_abnormal'] = (df[cycle_col] > threshold_cycle).astype(int)
        
        # Calculate RUL (Remaining Useful Life)
        # RUL = total_cycles - current_cycle
        df['rul'] = self.total_cycles - df[cycle_col]
        
        return df
    
    def generate_threshold_based_labels(
        self,
        df: pd.DataFrame,
        feature_col: str = 'voltage_ratio',
        cycle_col: str = 'cycle',
        capacitor_col: str = 'capacitor_id',
        threshold_pct: float = 0.2,
        baseline_cycles: int = 10
    ) -> pd.DataFrame:
        """
        Generate labels using threshold-based strategy.
        
        Labels as Abnormal when the feature deviates more than threshold_pct
        from the initial baseline value.
        
        Args:
            df: DataFrame with features
            feature_col: Feature column to use for threshold detection
            cycle_col: Name of the cycle column
            capacitor_col: Name of the capacitor ID column
            threshold_pct: Percentage deviation threshold (default: 0.2 = 20%)
            baseline_cycles: Number of initial cycles to use as baseline (default: 10)
        
        Returns:
            DataFrame with added 'is_abnormal' and 'rul' columns
        """
        df = df.copy()
        
        # Initialize labels
        df['is_abnormal'] = 0
        df['rul'] = self.total_cycles - df[cycle_col]
        
        # Calculate baseline for each capacitor
        for cap_id in df[capacitor_col].unique():
            cap_mask = df[capacitor_col] == cap_id
            
            # Get baseline value (mean of first N cycles)
            baseline_mask = cap_mask & (df[cycle_col] <= baseline_cycles)
            baseline_value = df.loc[baseline_mask, feature_col].mean()
            
            # Calculate deviation from baseline
            cap_values = df.loc[cap_mask, feature_col]
            deviation = np.abs(cap_values - baseline_value) / np.abs(baseline_value)
            
            # Label as abnormal if deviation exceeds threshold
            abnormal_mask = cap_mask & (deviation > threshold_pct)
            df.loc[abnormal_mask, 'is_abnormal'] = 1
        
        return df
    
    def generate_labels(
        self,
        df: pd.DataFrame,
        cycle_col: str = 'cycle',
        capacitor_col: str = 'capacitor_id',
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate labels using the configured strategy.
        
        Args:
            df: DataFrame with features
            cycle_col: Name of the cycle column
            capacitor_col: Name of the capacitor ID column
            **kwargs: Additional arguments for specific strategies
        
        Returns:
            DataFrame with added 'is_abnormal' and 'rul' columns
        """
        if self.strategy == 'cycle_based':
            return self.generate_cycle_based_labels(
                df,
                cycle_col=cycle_col,
                **kwargs
            )
        elif self.strategy == 'threshold_based':
            return self.generate_threshold_based_labels(
                df,
                cycle_col=cycle_col,
                capacitor_col=capacitor_col,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def get_label_statistics(
        self,
        df: pd.DataFrame,
        capacitor_col: str = 'capacitor_id'
    ) -> pd.DataFrame:
        """
        Get statistics about the generated labels.
        
        Args:
            df: DataFrame with labels
            capacitor_col: Name of the capacitor ID column
        
        Returns:
            DataFrame with label statistics per capacitor
        """
        stats = []
        
        for cap_id in df[capacitor_col].unique():
            cap_df = df[df[capacitor_col] == cap_id]
            
            stats.append({
                'capacitor_id': cap_id,
                'total_cycles': len(cap_df),
                'normal_cycles': (cap_df['is_abnormal'] == 0).sum(),
                'abnormal_cycles': (cap_df['is_abnormal'] == 1).sum(),
                'normal_ratio': (cap_df['is_abnormal'] == 0).mean(),
                'abnormal_ratio': (cap_df['is_abnormal'] == 1).mean(),
                'mean_rul': cap_df['rul'].mean(),
                'min_rul': cap_df['rul'].min(),
                'max_rul': cap_df['rul'].max()
            })
        
        return pd.DataFrame(stats)


def add_labels_to_features(
    features_path: str,
    output_path: str,
    total_cycles: int = 200,
    strategy: Literal['cycle_based', 'threshold_based'] = 'cycle_based',
    **kwargs
) -> pd.DataFrame:
    """
    Convenience function to add labels to a features CSV file.
    
    Args:
        features_path: Path to features CSV file
        output_path: Path to save labeled features CSV
        total_cycles: Total number of cycles per capacitor
        strategy: Labeling strategy to use
        **kwargs: Additional arguments for the labeling strategy
    
    Returns:
        DataFrame with features and labels
    """
    # Load features
    features_df = pd.read_csv(features_path)
    
    # Generate labels
    label_gen = LabelGenerator(total_cycles=total_cycles, strategy=strategy)
    labeled_df = label_gen.generate_labels(features_df, **kwargs)
    
    # Save labeled features
    labeled_df.to_csv(output_path, index=False)
    print(f"Labeled features saved to: {output_path}")
    print(f"Shape: {labeled_df.shape}")
    
    # Print statistics
    stats = label_gen.get_label_statistics(labeled_df)
    print("\nLabel Statistics:")
    print(stats.to_string(index=False))
    
    return labeled_df
