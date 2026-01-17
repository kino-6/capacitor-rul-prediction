"""
Tests for the LabelGenerator class.

This module tests label generation functionality for both cycle-based
and threshold-based strategies.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from src.data_preparation.label_generator import (
    LabelGenerator,
    add_labels_to_features
)


class TestLabelGenerator:
    """Test suite for LabelGenerator class."""
    
    @pytest.fixture
    def sample_features_df(self):
        """Create a sample features DataFrame for testing."""
        data = []
        for cap_id in ['ES12C1', 'ES12C2']:
            for cycle in range(1, 201):  # 200 cycles
                data.append({
                    'capacitor_id': cap_id,
                    'cycle': cycle,
                    'vl_mean': 2.0 + np.random.randn() * 0.1,
                    'vo_mean': 1.8 + np.random.randn() * 0.1,
                    'voltage_ratio': 0.9 - (cycle / 200) * 0.2,  # Degrading
                    'vl_cv': 0.05,
                    'vo_cv': 0.06
                })
        return pd.DataFrame(data)
    
    def test_initialization(self):
        """Test LabelGenerator initialization."""
        # Default initialization
        label_gen = LabelGenerator()
        assert label_gen.total_cycles == 200
        assert label_gen.strategy == 'cycle_based'
        
        # Custom initialization
        label_gen = LabelGenerator(total_cycles=150, strategy='threshold_based')
        assert label_gen.total_cycles == 150
        assert label_gen.strategy == 'threshold_based'
    
    def test_cycle_based_labels_basic(self, sample_features_df):
        """Test basic cycle-based label generation."""
        label_gen = LabelGenerator(total_cycles=200, strategy='cycle_based')
        labeled_df = label_gen.generate_cycle_based_labels(sample_features_df)
        
        # Check that labels were added
        assert 'is_abnormal' in labeled_df.columns
        assert 'rul' in labeled_df.columns
        
        # Check label types
        assert labeled_df['is_abnormal'].dtype == np.int64
        assert labeled_df['rul'].dtype == np.int64
    
    def test_cycle_based_labels_50_50_split(self, sample_features_df):
        """Test that cycle-based labels create 50/50 split."""
        label_gen = LabelGenerator(total_cycles=200, strategy='cycle_based')
        labeled_df = label_gen.generate_cycle_based_labels(sample_features_df)
        
        # For each capacitor, check 50/50 split
        for cap_id in ['ES12C1', 'ES12C2']:
            cap_df = labeled_df[labeled_df['capacitor_id'] == cap_id]
            
            # First 100 cycles should be Normal (0)
            first_half = cap_df[cap_df['cycle'] <= 100]
            assert (first_half['is_abnormal'] == 0).all()
            
            # Last 100 cycles should be Abnormal (1)
            second_half = cap_df[cap_df['cycle'] > 100]
            assert (second_half['is_abnormal'] == 1).all()
    
    def test_cycle_based_labels_rul_calculation(self, sample_features_df):
        """Test RUL calculation in cycle-based strategy."""
        label_gen = LabelGenerator(total_cycles=200, strategy='cycle_based')
        labeled_df = label_gen.generate_cycle_based_labels(sample_features_df)
        
        # Check RUL calculation: RUL = 200 - cycle
        for _, row in labeled_df.iterrows():
            expected_rul = 200 - row['cycle']
            assert row['rul'] == expected_rul
        
        # Check RUL range
        assert labeled_df['rul'].min() == 0  # Last cycle
        assert labeled_df['rul'].max() == 199  # First cycle
    
    def test_cycle_based_labels_custom_ratio(self, sample_features_df):
        """Test cycle-based labels with custom normal ratio."""
        label_gen = LabelGenerator(total_cycles=200, strategy='cycle_based')
        
        # 70% normal, 30% abnormal
        labeled_df = label_gen.generate_cycle_based_labels(
            sample_features_df,
            normal_ratio=0.7
        )
        
        # Check that first 140 cycles are normal
        for cap_id in ['ES12C1', 'ES12C2']:
            cap_df = labeled_df[labeled_df['capacitor_id'] == cap_id]
            
            first_70pct = cap_df[cap_df['cycle'] <= 140]
            assert (first_70pct['is_abnormal'] == 0).all()
            
            last_30pct = cap_df[cap_df['cycle'] > 140]
            assert (last_30pct['is_abnormal'] == 1).all()
    
    def test_threshold_based_labels_basic(self, sample_features_df):
        """Test basic threshold-based label generation."""
        label_gen = LabelGenerator(total_cycles=200, strategy='threshold_based')
        labeled_df = label_gen.generate_threshold_based_labels(sample_features_df)
        
        # Check that labels were added
        assert 'is_abnormal' in labeled_df.columns
        assert 'rul' in labeled_df.columns
        
        # Check that some cycles are labeled as abnormal
        # (since voltage_ratio is degrading in sample data)
        assert labeled_df['is_abnormal'].sum() > 0
    
    def test_threshold_based_labels_baseline(self, sample_features_df):
        """Test that threshold-based labels use baseline correctly."""
        label_gen = LabelGenerator(total_cycles=200, strategy='threshold_based')
        labeled_df = label_gen.generate_threshold_based_labels(
            sample_features_df,
            baseline_cycles=10,
            threshold_pct=0.1  # 10% threshold
        )
        
        # First 10 cycles should mostly be normal (baseline)
        for cap_id in ['ES12C1', 'ES12C2']:
            cap_df = labeled_df[labeled_df['capacitor_id'] == cap_id]
            baseline = cap_df[cap_df['cycle'] <= 10]
            
            # Most baseline cycles should be normal
            normal_ratio = (baseline['is_abnormal'] == 0).mean()
            assert normal_ratio >= 0.5  # At least 50% normal in baseline
    
    def test_threshold_based_labels_custom_feature(self):
        """Test threshold-based labels with custom feature."""
        # Create data with clear degradation in a custom feature
        data = []
        for cycle in range(1, 201):
            data.append({
                'capacitor_id': 'ES12C1',
                'cycle': cycle,
                'custom_feature': 1.0 if cycle <= 50 else 0.5,  # Clear drop
                'voltage_ratio': 0.9
            })
        df = pd.DataFrame(data)
        
        label_gen = LabelGenerator(total_cycles=200, strategy='threshold_based')
        labeled_df = label_gen.generate_threshold_based_labels(
            df,
            feature_col='custom_feature',
            threshold_pct=0.2
        )
        
        # Cycles after 50 should be abnormal (50% drop > 20% threshold)
        after_50 = labeled_df[labeled_df['cycle'] > 50]
        assert (after_50['is_abnormal'] == 1).all()
    
    def test_generate_labels_cycle_based(self, sample_features_df):
        """Test generate_labels with cycle_based strategy."""
        label_gen = LabelGenerator(total_cycles=200, strategy='cycle_based')
        labeled_df = label_gen.generate_labels(sample_features_df)
        
        # Should produce same result as generate_cycle_based_labels
        assert 'is_abnormal' in labeled_df.columns
        assert 'rul' in labeled_df.columns
        
        # Check 50/50 split
        for cap_id in ['ES12C1', 'ES12C2']:
            cap_df = labeled_df[labeled_df['capacitor_id'] == cap_id]
            normal_count = (cap_df['is_abnormal'] == 0).sum()
            abnormal_count = (cap_df['is_abnormal'] == 1).sum()
            assert normal_count == 100
            assert abnormal_count == 100
    
    def test_generate_labels_threshold_based(self, sample_features_df):
        """Test generate_labels with threshold_based strategy."""
        label_gen = LabelGenerator(total_cycles=200, strategy='threshold_based')
        labeled_df = label_gen.generate_labels(sample_features_df)
        
        # Should produce same result as generate_threshold_based_labels
        assert 'is_abnormal' in labeled_df.columns
        assert 'rul' in labeled_df.columns
    
    def test_generate_labels_invalid_strategy(self, sample_features_df):
        """Test that invalid strategy raises error."""
        label_gen = LabelGenerator(total_cycles=200, strategy='cycle_based')
        label_gen.strategy = 'invalid_strategy'
        
        with pytest.raises(ValueError, match="Unknown strategy"):
            label_gen.generate_labels(sample_features_df)
    
    def test_get_label_statistics(self, sample_features_df):
        """Test label statistics generation."""
        label_gen = LabelGenerator(total_cycles=200, strategy='cycle_based')
        labeled_df = label_gen.generate_labels(sample_features_df)
        
        stats = label_gen.get_label_statistics(labeled_df)
        
        # Check statistics structure
        assert len(stats) == 2  # Two capacitors
        assert 'capacitor_id' in stats.columns
        assert 'total_cycles' in stats.columns
        assert 'normal_cycles' in stats.columns
        assert 'abnormal_cycles' in stats.columns
        assert 'normal_ratio' in stats.columns
        assert 'abnormal_ratio' in stats.columns
        assert 'mean_rul' in stats.columns
        assert 'min_rul' in stats.columns
        assert 'max_rul' in stats.columns
        
        # Check values for cycle-based strategy
        for _, row in stats.iterrows():
            assert row['total_cycles'] == 200
            assert row['normal_cycles'] == 100
            assert row['abnormal_cycles'] == 100
            assert row['normal_ratio'] == 0.5
            assert row['abnormal_ratio'] == 0.5
            assert row['mean_rul'] == 99.5  # Average of 0-199
            assert row['min_rul'] == 0
            assert row['max_rul'] == 199
    
    def test_label_statistics_different_capacitors(self):
        """Test label statistics with different cycle counts."""
        # Create data with different cycle counts
        data = []
        for cycle in range(1, 151):  # 150 cycles for C1
            data.append({
                'capacitor_id': 'ES12C1',
                'cycle': cycle,
                'voltage_ratio': 0.9
            })
        for cycle in range(1, 201):  # 200 cycles for C2
            data.append({
                'capacitor_id': 'ES12C2',
                'cycle': cycle,
                'voltage_ratio': 0.9
            })
        df = pd.DataFrame(data)
        
        label_gen = LabelGenerator(total_cycles=200, strategy='cycle_based')
        labeled_df = label_gen.generate_labels(df)
        stats = label_gen.get_label_statistics(labeled_df)
        
        # Check C1 stats (150 cycles)
        c1_stats = stats[stats['capacitor_id'] == 'ES12C1'].iloc[0]
        assert c1_stats['total_cycles'] == 150
        
        # Check C2 stats (200 cycles)
        c2_stats = stats[stats['capacitor_id'] == 'ES12C2'].iloc[0]
        assert c2_stats['total_cycles'] == 200
    
    def test_add_labels_to_features_function(self, sample_features_df):
        """Test the convenience function add_labels_to_features."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save sample features
            features_path = Path(tmpdir) / 'features.csv'
            sample_features_df.to_csv(features_path, index=False)
            
            # Add labels
            output_path = Path(tmpdir) / 'features_with_labels.csv'
            labeled_df = add_labels_to_features(
                str(features_path),
                str(output_path),
                total_cycles=200,
                strategy='cycle_based'
            )
            
            # Check that file was created
            assert output_path.exists()
            
            # Check that labels were added
            assert 'is_abnormal' in labeled_df.columns
            assert 'rul' in labeled_df.columns
            
            # Load and verify saved file
            loaded_df = pd.read_csv(output_path)
            assert 'is_abnormal' in loaded_df.columns
            assert 'rul' in loaded_df.columns
            assert len(loaded_df) == len(sample_features_df)
    
    def test_labels_preserve_original_columns(self, sample_features_df):
        """Test that label generation preserves original columns."""
        original_cols = set(sample_features_df.columns)
        
        label_gen = LabelGenerator(total_cycles=200, strategy='cycle_based')
        labeled_df = label_gen.generate_labels(sample_features_df)
        
        # All original columns should be present
        for col in original_cols:
            assert col in labeled_df.columns
        
        # New columns should be added
        assert 'is_abnormal' in labeled_df.columns
        assert 'rul' in labeled_df.columns
    
    def test_labels_do_not_modify_original_df(self, sample_features_df):
        """Test that label generation doesn't modify the original DataFrame."""
        original_cols = list(sample_features_df.columns)
        original_shape = sample_features_df.shape
        
        label_gen = LabelGenerator(total_cycles=200, strategy='cycle_based')
        labeled_df = label_gen.generate_labels(sample_features_df)
        
        # Original DataFrame should be unchanged
        assert list(sample_features_df.columns) == original_cols
        assert sample_features_df.shape == original_shape
        assert 'is_abnormal' not in sample_features_df.columns
        assert 'rul' not in sample_features_df.columns


class TestLabelGeneratorEdgeCases:
    """Test edge cases for LabelGenerator."""
    
    def test_single_cycle(self):
        """Test with single cycle."""
        df = pd.DataFrame([{
            'capacitor_id': 'ES12C1',
            'cycle': 1,
            'voltage_ratio': 0.9
        }])
        
        label_gen = LabelGenerator(total_cycles=200, strategy='cycle_based')
        labeled_df = label_gen.generate_labels(df)
        
        assert labeled_df['is_abnormal'].iloc[0] == 0  # First cycle is normal
        assert labeled_df['rul'].iloc[0] == 199
    
    def test_last_cycle(self):
        """Test with last cycle."""
        df = pd.DataFrame([{
            'capacitor_id': 'ES12C1',
            'cycle': 200,
            'voltage_ratio': 0.9
        }])
        
        label_gen = LabelGenerator(total_cycles=200, strategy='cycle_based')
        labeled_df = label_gen.generate_labels(df)
        
        assert labeled_df['is_abnormal'].iloc[0] == 1  # Last cycle is abnormal
        assert labeled_df['rul'].iloc[0] == 0
    
    def test_boundary_cycle_100(self):
        """Test boundary at cycle 100."""
        df = pd.DataFrame([
            {'capacitor_id': 'ES12C1', 'cycle': 100, 'voltage_ratio': 0.9},
            {'capacitor_id': 'ES12C1', 'cycle': 101, 'voltage_ratio': 0.9}
        ])
        
        label_gen = LabelGenerator(total_cycles=200, strategy='cycle_based')
        labeled_df = label_gen.generate_labels(df)
        
        # Cycle 100 should be normal (≤ 100)
        assert labeled_df[labeled_df['cycle'] == 100]['is_abnormal'].iloc[0] == 0
        # Cycle 101 should be abnormal (> 100)
        assert labeled_df[labeled_df['cycle'] == 101]['is_abnormal'].iloc[0] == 1
    
    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame(columns=['capacitor_id', 'cycle', 'voltage_ratio'])
        
        label_gen = LabelGenerator(total_cycles=200, strategy='cycle_based')
        labeled_df = label_gen.generate_labels(df)
        
        assert len(labeled_df) == 0
        assert 'is_abnormal' in labeled_df.columns
        assert 'rul' in labeled_df.columns
    
    def test_missing_voltage_ratio_threshold_based(self):
        """Test threshold-based with missing voltage_ratio column."""
        df = pd.DataFrame([{
            'capacitor_id': 'ES12C1',
            'cycle': 1,
            'other_feature': 0.9
        }])
        
        label_gen = LabelGenerator(total_cycles=200, strategy='threshold_based')
        
        # Should raise KeyError when voltage_ratio is missing
        with pytest.raises(KeyError):
            label_gen.generate_labels(df)


class TestLabelGeneratorIntegration:
    """Integration tests with realistic data."""
    
    def test_realistic_es12_data(self):
        """Test with realistic ES12-like data."""
        # Create realistic data for 8 capacitors, 200 cycles each
        data = []
        for cap_num in range(1, 9):
            cap_id = f'ES12C{cap_num}'
            for cycle in range(1, 201):
                # Simulate degradation
                degradation_factor = 1 - (cycle / 200) * 0.3
                data.append({
                    'capacitor_id': cap_id,
                    'cycle': cycle,
                    'vl_mean': 2.0 + np.random.randn() * 0.05,
                    'vo_mean': 1.8 * degradation_factor + np.random.randn() * 0.05,
                    'voltage_ratio': 0.9 * degradation_factor,
                    'vl_cv': 0.05,
                    'vo_cv': 0.06
                })
        df = pd.DataFrame(data)
        
        label_gen = LabelGenerator(total_cycles=200, strategy='cycle_based')
        labeled_df = label_gen.generate_labels(df)
        
        # Check total samples
        assert len(labeled_df) == 8 * 200  # 1600 samples
        
        # Check label distribution
        total_normal = (labeled_df['is_abnormal'] == 0).sum()
        total_abnormal = (labeled_df['is_abnormal'] == 1).sum()
        assert total_normal == 8 * 100  # 800 normal
        assert total_abnormal == 8 * 100  # 800 abnormal
        
        # Check statistics
        stats = label_gen.get_label_statistics(labeled_df)
        assert len(stats) == 8  # 8 capacitors
        assert (stats['total_cycles'] == 200).all()
        assert (stats['normal_cycles'] == 100).all()
        assert (stats['abnormal_cycles'] == 100).all()
