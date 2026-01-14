"""
Property-based tests for ES12 real data loading functionality.

This module contains property-based tests that verify the completeness
and correctness of ES12 real data loading operations.
"""

import pytest
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume, HealthCheck
import pandas as pd
import numpy as np

from nasa_pcoe_eda.data.es12_loader import ES12DataLoader
from nasa_pcoe_eda.data.loader import DataLoader


class TestES12DataLoadingProperties:
    """Property-based tests for ES12 data loading completeness."""

    @pytest.fixture
    def es12_file_path(self):
        """Fixture providing path to ES12.mat file."""
        path = Path("data/raw/ES12.mat")
        if not path.exists():
            pytest.skip("ES12.mat file not found")
        return path

    @pytest.fixture
    def es12_loader(self):
        """Fixture providing ES12DataLoader instance."""
        return ES12DataLoader()

    @pytest.fixture
    def generic_loader(self):
        """Fixture providing generic DataLoader instance."""
        return DataLoader()

    # Feature: nasa-pcoe-eda, Property 25: 実データ読み込みの完全性
    @given(st.just(True))  # Dummy strategy since we're testing with real file
    @settings(max_examples=1, deadline=60000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_real_data_loading_completeness(self, es12_file_path, es12_loader, _dummy):
        """
        Property 25: 実データ読み込みの完全性
        
        For any valid ES12.mat file, loading should result in complete data
        for all capacitors, all cycles, and all frequency data being accurately loaded.
        
        **Validates: Requirements 2.1, 2.3**
        """
        # Load the data
        df = es12_loader.load_dataset(es12_file_path)
        
        # Verify completeness: All 8 capacitors should be present
        expected_capacitors = {f'ES12C{i}' for i in range(1, 9)}
        actual_capacitors = set(df['capacitor'].unique())
        assert actual_capacitors == expected_capacitors, \
            f"Missing capacitors: {expected_capacitors - actual_capacitors}"
        
        # Verify each capacitor has data
        for cap in expected_capacitors:
            cap_data = df[df['capacitor'] == cap]
            assert len(cap_data) > 0, f"No data found for capacitor {cap}"
            
            # Verify cycle data completeness
            assert cap_data['cycle'].min() >= 1, f"Invalid minimum cycle for {cap}"
            assert cap_data['cycle'].max() > cap_data['cycle'].min(), \
                f"No cycle progression for {cap}"
            
            # Verify voltage data completeness (should have valid measurements)
            voltage_cols = ['vl_mean', 'vl_std', 'vo_mean', 'vo_std']
            for col in voltage_cols:
                valid_count = cap_data[col].notna().sum()
                assert valid_count > 0, f"No valid {col} data for {cap}"
        
        # Verify raw data access completeness
        for cap in expected_capacitors:
            raw_data = es12_loader.get_raw_transient_data(cap)
            assert raw_data is not None, f"No raw data for {cap}"
            assert 'VL' in raw_data, f"Missing VL data for {cap}"
            assert 'VO' in raw_data, f"Missing VO data for {cap}"
            
            # Verify raw data dimensions
            vl_shape = raw_data['VL'].shape
            vo_shape = raw_data['VO'].shape
            assert len(vl_shape) == 2, f"Invalid VL data shape for {cap}: {vl_shape}"
            assert len(vo_shape) == 2, f"Invalid VO data shape for {cap}: {vo_shape}"
            assert vl_shape == vo_shape, f"VL and VO shape mismatch for {cap}"
            
            # Verify reasonable data dimensions
            n_timepoints, n_cycles = vl_shape
            assert n_timepoints > 1000, f"Too few time points for {cap}: {n_timepoints}"
            assert n_cycles > 100, f"Too few cycles for {cap}: {n_cycles}"

    @given(st.just(True))  # Dummy strategy since we're testing with real file
    @settings(max_examples=1, deadline=60000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_es12_auto_detection_completeness(self, es12_file_path, generic_loader, _dummy):
        """
        Property: ES12 auto-detection completeness
        
        For any ES12.mat file, the generic loader should auto-detect ES12 format
        and produce the same complete results as the specialized loader.
        
        **Validates: Requirements 2.1, 2.3**
        """
        # Load with generic loader (should auto-detect ES12)
        df_generic = generic_loader.load_dataset(es12_file_path)
        
        # Load with specialized loader
        es12_loader = ES12DataLoader()
        df_specialized = es12_loader.load_dataset(es12_file_path)
        
        # Verify both produce equivalent results
        assert df_generic.shape == df_specialized.shape, \
            "Generic and specialized loaders produce different shapes"
        
        assert set(df_generic.columns) == set(df_specialized.columns), \
            "Generic and specialized loaders produce different columns"
        
        assert set(df_generic['capacitor'].unique()) == set(df_specialized['capacitor'].unique()), \
            "Generic and specialized loaders produce different capacitor sets"

    @given(st.just(True))  # Dummy strategy since we're testing with real file
    @settings(max_examples=1, deadline=60000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_data_validation_completeness(self, es12_file_path, es12_loader, _dummy):
        """
        Property: Data validation completeness
        
        For any loaded ES12 data, validation should be comprehensive and
        identify all data quality issues accurately.
        
        **Validates: Requirements 2.1, 2.3**
        """
        df = es12_loader.load_dataset(es12_file_path)
        validation = es12_loader.validate_data(df)
        
        # Validation should complete without errors
        assert validation is not None, "Validation should return a result"
        assert hasattr(validation, 'is_valid'), "Validation should have is_valid attribute"
        assert hasattr(validation, 'errors'), "Validation should have errors attribute"
        assert hasattr(validation, 'warnings'), "Validation should have warnings attribute"
        
        # For real ES12 data, validation should generally pass
        if not validation.is_valid:
            # If validation fails, errors should be specific and actionable
            assert len(validation.errors) > 0, "Invalid data should have specific errors"
            for error in validation.errors:
                assert isinstance(error, str), "Errors should be strings"
                assert len(error) > 0, "Errors should not be empty"

    @given(st.just(True))  # Dummy strategy since we're testing with real file
    @settings(max_examples=1, deadline=60000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_metadata_extraction_completeness(self, es12_file_path, es12_loader, _dummy):
        """
        Property: Metadata extraction completeness
        
        For any loaded ES12 data, metadata extraction should be complete
        and accurate, providing all essential dataset information.
        
        **Validates: Requirements 2.1, 2.3**
        """
        df = es12_loader.load_dataset(es12_file_path)
        metadata = es12_loader.get_metadata(df)
        
        # Verify metadata completeness
        assert metadata.n_records == len(df), "Record count should match DataFrame length"
        assert metadata.n_features == len(df.columns), "Feature count should match DataFrame columns"
        assert len(metadata.feature_names) == len(df.columns), "Feature names should match columns"
        assert metadata.memory_usage > 0, "Memory usage should be positive"
        
        # Verify data types are captured
        assert len(metadata.data_types) == len(df.columns), "All column types should be captured"
        for col in df.columns:
            assert str(col) in metadata.data_types, f"Column {col} type not captured"
        
        # For ES12 data, we expect temporal information
        if 'timestamp' in df.columns and df['timestamp'].notna().any():
            assert metadata.date_range is not None, "Date range should be extracted from timestamps"
            start_date, end_date = metadata.date_range
            assert start_date <= end_date, "Date range should be valid"

    @given(capacitor_name=st.sampled_from(['ES12C1', 'ES12C2', 'ES12C3', 'ES12C4', 
                                          'ES12C5', 'ES12C6', 'ES12C7', 'ES12C8']))
    @settings(max_examples=8, deadline=60000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_capacitor_specific_data_completeness(self, es12_file_path, es12_loader, capacitor_name):
        """
        Property: Capacitor-specific data completeness
        
        For any valid capacitor name, the loader should provide complete
        data access for that specific capacitor.
        
        **Validates: Requirements 2.1, 2.3**
        """
        # Load the full dataset
        df = es12_loader.load_dataset(es12_file_path)
        
        # Get capacitor-specific data
        cap_data = es12_loader.get_capacitor_data(capacitor_name)
        
        # Verify completeness
        assert cap_data is not None, f"Should return data for {capacitor_name}"
        assert len(cap_data) > 0, f"Should have records for {capacitor_name}"
        assert (cap_data['capacitor'] == capacitor_name).all(), \
            f"All records should be for {capacitor_name}"
        
        # Verify it matches the full dataset
        expected_data = df[df['capacitor'] == capacitor_name]
        assert len(cap_data) == len(expected_data), \
            f"Capacitor data length should match filtered data for {capacitor_name}"
        
        # Get raw transient data
        raw_data = es12_loader.get_raw_transient_data(capacitor_name)
        assert raw_data is not None, f"Should return raw data for {capacitor_name}"
        assert 'VL' in raw_data and 'VO' in raw_data, \
            f"Raw data should contain VL and VO for {capacitor_name}"

    @given(st.just(True))  # Dummy strategy since we're testing with real file
    @settings(max_examples=1, deadline=60000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_data_consistency_across_loads(self, es12_file_path, _dummy):
        """
        Property: Data consistency across multiple loads
        
        For any ES12 file, loading the same file multiple times should
        produce identical results (deterministic loading).
        
        **Validates: Requirements 2.1, 2.3**
        """
        loader1 = ES12DataLoader()
        loader2 = ES12DataLoader()
        
        # Load the same file twice
        df1 = loader1.load_dataset(es12_file_path)
        df2 = loader2.load_dataset(es12_file_path)
        
        # Results should be identical
        assert df1.shape == df2.shape, "Multiple loads should produce same shape"
        assert list(df1.columns) == list(df2.columns), "Multiple loads should produce same columns"
        
        # Compare data values (allowing for floating point precision)
        for col in df1.columns:
            if df1[col].dtype in ['float64', 'float32']:
                # For floating point columns, use approximate equality
                pd.testing.assert_series_equal(df1[col], df2[col], check_exact=False, rtol=1e-10)
            else:
                # For other columns, use exact equality
                pd.testing.assert_series_equal(df1[col], df2[col], check_exact=True)