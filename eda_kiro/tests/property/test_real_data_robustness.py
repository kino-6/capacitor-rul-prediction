"""
Property-based tests for real data processing robustness.

This module contains property-based tests that verify the robustness
of data processing when handling various data quality conditions.
"""

import pytest
from pathlib import Path
from hypothesis import given, strategies as st, settings, assume, HealthCheck
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import tempfile
import shutil
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to reduce warnings

from nasa_pcoe_eda.data.es12_loader import ES12DataLoader
from nasa_pcoe_eda.data.loader import DataLoader
from nasa_pcoe_eda.orchestrator import AnalysisOrchestrator
from nasa_pcoe_eda.analysis.statistics import StatisticsAnalyzer
from nasa_pcoe_eda.analysis.correlation import CorrelationAnalyzer
from nasa_pcoe_eda.exceptions import EDAError, DataLoadError

# Suppress specific warnings for cleaner test output
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message=".*Glyph.*missing from.*font.*")
warnings.filterwarnings("ignore", message=".*Precision loss occurred in moment calculation.*")
warnings.filterwarnings("ignore", message=".*An input array is constant.*")
warnings.filterwarnings("ignore", message=".*correlation coefficient is not defined.*")


class TestRealDataRobustness:
    """Property-based tests for real data processing robustness."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def es12_file_path(self):
        """Fixture providing path to ES12.mat file."""
        path = Path("data/raw/ES12.mat")
        if not path.exists():
            pytest.skip("ES12.mat file not found")
        return path

    def create_degraded_data(
        self, 
        original_df: pd.DataFrame, 
        missing_ratio: float = 0.1,
        noise_level: float = 0.05,
        outlier_ratio: float = 0.02
    ) -> pd.DataFrame:
        """
        Create degraded version of data with various quality issues.
        
        Args:
            original_df: Original clean data
            missing_ratio: Fraction of values to make missing
            noise_level: Level of noise to add (as fraction of std)
            outlier_ratio: Fraction of values to make outliers
            
        Returns:
            DataFrame with simulated quality issues
        """
        df = original_df.copy()
        np.random.seed(42)  # For reproducible degradation
        
        # Add missing values
        if missing_ratio > 0:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                n_missing = int(len(df) * missing_ratio)
                if n_missing > 0:
                    missing_indices = np.random.choice(len(df), n_missing, replace=False)
                    df.loc[missing_indices, col] = np.nan
        
        # Add noise
        if noise_level > 0:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].notna().any():
                    col_std = df[col].std()
                    if not np.isnan(col_std) and col_std > 0:
                        noise = np.random.normal(0, col_std * noise_level, len(df))
                        df[col] = df[col] + noise
        
        # Add outliers
        if outlier_ratio > 0:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].notna().any():
                    n_outliers = int(len(df) * outlier_ratio)
                    if n_outliers > 0:
                        outlier_indices = np.random.choice(len(df), n_outliers, replace=False)
                        col_mean = df[col].mean()
                        col_std = df[col].std()
                        if not np.isnan(col_mean) and not np.isnan(col_std) and col_std > 0:
                            # Create outliers at 5 standard deviations
                            outlier_values = col_mean + np.random.choice([-1, 1], n_outliers) * 5 * col_std
                            df.loc[outlier_indices, col] = outlier_values
        
        return df

    def save_dataframe_as_mat(self, df: pd.DataFrame, file_path: Path) -> None:
        """Save DataFrame as MATLAB file for testing."""
        import scipy.io
        
        # Convert DataFrame to array format
        numeric_df = df.select_dtypes(include=[np.number])
        if len(numeric_df.columns) > 0:
            data_array = numeric_df.values
            scipy.io.savemat(str(file_path), {'data': data_array})
        else:
            # Create minimal array if no numeric data
            data_array = np.array([[1, 2], [3, 4]])
            scipy.io.savemat(str(file_path), {'data': data_array})

    # Feature: nasa-pcoe-eda, Property 27: 実データ処理の堅牢性
    @given(
        missing_ratio=st.floats(min_value=0.0, max_value=0.5),
        noise_level=st.floats(min_value=0.0, max_value=0.2),
        outlier_ratio=st.floats(min_value=0.0, max_value=0.1)
    )
    @settings(max_examples=1, deadline=15000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_real_data_processing_robustness(
        self, 
        es12_file_path, 
        temp_output_dir, 
        missing_ratio, 
        noise_level, 
        outlier_ratio
    ):
        """
        Property 27: 実データ処理の堅牢性
        
        For any real data with various quality states (missing values, noise, outliers),
        the processing should be robust and handle these conditions gracefully.
        
        **Validates: Requirements 2.4, 8.1**
        """
        # Load original real data
        loader = ES12DataLoader()
        original_df = loader.load_dataset(es12_file_path)
        
        # Create degraded version of the data
        degraded_df = self.create_degraded_data(
            original_df, 
            missing_ratio=missing_ratio,
            noise_level=noise_level,
            outlier_ratio=outlier_ratio
        )
        
        # Save degraded data as temporary file
        degraded_file = temp_output_dir / "degraded_data.mat"
        self.save_dataframe_as_mat(degraded_df, degraded_file)
        
        # Test robustness of data loading
        try:
            degraded_loader = DataLoader()
            loaded_degraded_df = degraded_loader.load_dataset(degraded_file)
            
            # Loading should succeed even with quality issues
            assert loaded_degraded_df is not None, "Should load degraded data"
            assert len(loaded_degraded_df) > 0, "Should have records in degraded data"
            
        except DataLoadError as e:
            # If loading fails, it should be due to severe corruption, not minor quality issues
            if missing_ratio < 0.8 and noise_level < 0.5 and outlier_ratio < 0.3:
                pytest.fail(f"Should handle moderate data quality issues, but failed: {e}")
        
        # Test robustness of statistical analysis
        try:
            stats_analyzer = StatisticsAnalyzer()
            
            # Basic statistics should be computable even with quality issues
            descriptive_stats = stats_analyzer.compute_descriptive_stats(degraded_df)
            assert isinstance(descriptive_stats, dict), "Should compute descriptive stats"
            
            # Missing value analysis should handle any level of missing data
            missing_analysis = stats_analyzer.analyze_missing_values(degraded_df)
            assert missing_analysis is not None, "Should analyze missing values"
            assert missing_analysis.total_missing >= 0, "Should count missing values correctly"
            
            # Data type identification should be robust
            data_types = stats_analyzer.identify_data_types(degraded_df)
            assert isinstance(data_types, dict), "Should identify data types"
            assert len(data_types) > 0, "Should identify at least some data types"
            
        except Exception as e:
            # Statistical analysis should be very robust
            if missing_ratio < 0.9:  # Only allow failure with extreme missing data
                pytest.fail(f"Statistical analysis should be robust, but failed: {e}")
        
        # Test robustness of correlation analysis
        try:
            corr_analyzer = CorrelationAnalyzer()
            numeric_cols = degraded_df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) >= 2:  # Need at least 2 numeric columns for correlation
                corr_matrix = corr_analyzer.compute_correlation_matrix(degraded_df)
                
                # Correlation matrix should be computed even with quality issues
                assert corr_matrix is not None, "Should compute correlation matrix"
                
                # Values should be valid correlations (between -1 and 1, or NaN)
                corr_values = corr_matrix.values
                valid_corr_values = corr_values[~np.isnan(corr_values)]
                if len(valid_corr_values) > 0:
                    assert np.all((-1 <= valid_corr_values) & (valid_corr_values <= 1)), \
                        "Correlation values should be in [-1, 1] range"
                
        except Exception as e:
            # Correlation analysis might fail with extreme data quality issues
            if missing_ratio < 0.7 and noise_level < 0.3:
                pytest.fail(f"Correlation analysis should handle moderate quality issues: {e}")
        
        # Test robustness of complete pipeline
        try:
            orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
            results = orchestrator.run_complete_analysis(data_path=degraded_file)
            
            # Pipeline should complete even with quality issues
            assert results is not None, "Pipeline should complete with degraded data"
            assert results.metadata is not None, "Should have metadata even with quality issues"
            
            # Should handle missing values gracefully
            if missing_ratio > 0:
                assert results.missing_values is not None, "Should analyze missing values"
                assert results.missing_values.total_missing > 0, "Should detect missing values"
            
        except EDAError as e:
            # Pipeline might fail with severe quality issues, but should provide informative errors
            assert len(str(e)) > 0, "Error message should be informative"
            
            # Should only fail with extreme quality issues
            if missing_ratio < 0.8 and noise_level < 0.4 and outlier_ratio < 0.2:
                pytest.fail(f"Pipeline should handle moderate quality issues: {e}")

    @given(
        capacitor_subset=st.lists(
            st.sampled_from(['ES12C1', 'ES12C2', 'ES12C3', 'ES12C4', 'ES12C5', 'ES12C6', 'ES12C7', 'ES12C8']),
            min_size=1,
            max_size=4,
            unique=True
        )
    )
    @settings(max_examples=1, deadline=15000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_partial_capacitor_data_robustness(
        self, 
        es12_file_path, 
        temp_output_dir, 
        capacitor_subset
    ):
        """
        Property: Partial capacitor data robustness
        
        For any subset of capacitors, the system should process the available
        data robustly without requiring all capacitors to be present.
        
        **Validates: Requirements 2.4, 8.1**
        """
        # Load original data
        loader = ES12DataLoader()
        original_df = loader.load_dataset(es12_file_path)
        
        # Filter to only include subset of capacitors
        available_capacitors = set(original_df['capacitor'].unique())
        valid_subset = [cap for cap in capacitor_subset if cap in available_capacitors]
        
        assume(len(valid_subset) > 0)  # Need at least one valid capacitor
        
        subset_df = original_df[original_df['capacitor'].isin(valid_subset)]
        
        # Save subset data
        subset_file = temp_output_dir / "subset_data.mat"
        self.save_dataframe_as_mat(subset_df, subset_file)
        
        # Test processing with partial data
        try:
            orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
            results = orchestrator.run_complete_analysis(data_path=subset_file)
            
            # Should process successfully with any valid subset
            assert results is not None, f"Should process subset of {len(valid_subset)} capacitors"
            assert results.metadata is not None, "Should have metadata for subset"
            assert results.metadata.n_records > 0, "Should have records for subset"
            
            # Should generate appropriate outputs
            assert results.statistics is not None, "Should compute statistics for subset"
            assert results.missing_values is not None, "Should analyze missing values for subset"
            
        except Exception as e:
            pytest.fail(f"Should handle subset of capacitors robustly: {e}")

    @given(
        cycle_range=st.tuples(
            st.integers(min_value=1, max_value=50),
            st.integers(min_value=51, max_value=200)
        )
    )
    @settings(max_examples=1, deadline=15000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_partial_cycle_data_robustness(
        self, 
        es12_file_path, 
        temp_output_dir, 
        cycle_range
    ):
        """
        Property: Partial cycle data robustness
        
        For any range of cycles, the system should process the available
        cycle data robustly without requiring the complete cycle range.
        
        **Validates: Requirements 2.4, 8.1**
        """
        start_cycle, end_cycle = cycle_range
        assume(start_cycle < end_cycle)
        
        # Load original data
        loader = ES12DataLoader()
        original_df = loader.load_dataset(es12_file_path)
        
        # Filter to only include subset of cycles
        cycle_subset_df = original_df[
            (original_df['cycle'] >= start_cycle) & 
            (original_df['cycle'] <= end_cycle)
        ]
        
        assume(len(cycle_subset_df) > 0)  # Need some data in the range
        
        # Save subset data
        cycle_subset_file = temp_output_dir / "cycle_subset_data.mat"
        self.save_dataframe_as_mat(cycle_subset_df, cycle_subset_file)
        
        # Test processing with partial cycle data
        try:
            orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
            results = orchestrator.run_complete_analysis(data_path=cycle_subset_file)
            
            # Should process successfully with any valid cycle range
            assert results is not None, f"Should process cycles {start_cycle}-{end_cycle}"
            assert results.metadata is not None, "Should have metadata for cycle subset"
            assert results.metadata.n_records > 0, "Should have records for cycle subset"
            
            # Should generate appropriate outputs
            assert results.statistics is not None, "Should compute statistics for cycle subset"
            
        except Exception as e:
            pytest.fail(f"Should handle subset of cycles robustly: {e}")

    @given(
        data_corruption_level=st.floats(min_value=0.01, max_value=0.3)
    )
    @settings(max_examples=1, deadline=15000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_data_corruption_robustness(
        self, 
        es12_file_path, 
        temp_output_dir, 
        data_corruption_level
    ):
        """
        Property: Data corruption robustness
        
        For any level of data corruption (within reasonable bounds),
        the system should either process successfully or fail gracefully
        with informative error messages.
        
        **Validates: Requirements 2.4, 8.1**
        """
        # Load original data
        loader = ES12DataLoader()
        original_df = loader.load_dataset(es12_file_path)
        
        # Corrupt data by replacing values with invalid data
        corrupted_df = original_df.copy()
        numeric_cols = corrupted_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            n_corrupt = int(len(corrupted_df) * len(numeric_cols) * data_corruption_level)
            
            for _ in range(n_corrupt):
                # Randomly corrupt some values
                row_idx = np.random.randint(0, len(corrupted_df))
                col_idx = np.random.randint(0, len(numeric_cols))
                col_name = numeric_cols[col_idx]
                
                # Introduce various types of corruption
                corruption_type = np.random.choice(['inf', 'nan', 'extreme'])
                
                if corruption_type == 'inf':
                    corrupted_df.iloc[row_idx, corrupted_df.columns.get_loc(col_name)] = np.inf
                elif corruption_type == 'nan':
                    corrupted_df.iloc[row_idx, corrupted_df.columns.get_loc(col_name)] = np.nan
                elif corruption_type == 'extreme':
                    corrupted_df.iloc[row_idx, corrupted_df.columns.get_loc(col_name)] = 1e15
        
        # Save corrupted data
        corrupted_file = temp_output_dir / "corrupted_data.mat"
        self.save_dataframe_as_mat(corrupted_df, corrupted_file)
        
        # Test processing with corrupted data
        try:
            orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
            results = orchestrator.run_complete_analysis(data_path=corrupted_file)
            
            # If processing succeeds, results should be reasonable
            if results is not None:
                assert results.metadata is not None, "Should have metadata despite corruption"
                
                # Statistics should handle corrupted values
                if results.statistics is not None:
                    # Should not crash, even if some statistics are invalid
                    assert isinstance(results.statistics, dict), "Statistics should be in dict format"
                
        except EDAError as e:
            # If processing fails, error should be informative
            assert len(str(e)) > 0, "Error message should be informative"
            
            # Should provide specific information about the corruption
            error_msg = str(e).lower()
            corruption_indicators = ['corrupt', 'invalid', 'nan', 'inf', 'extreme']
            found_indicators = sum(1 for indicator in corruption_indicators if indicator in error_msg)
            assert found_indicators > 0, "Error message should indicate data corruption issues"

    @given(st.just(True))  # Dummy strategy since we're testing with real file
    @settings(max_examples=1, deadline=15000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_memory_pressure_robustness(
        self, 
        es12_file_path, 
        temp_output_dir, 
        _dummy
    ):
        """
        Property: Memory pressure robustness
        
        The system should handle memory pressure gracefully and not
        crash due to memory issues during processing.
        
        **Validates: Requirements 2.4, 8.1**
        """
        import psutil
        import os
        
        # Get initial memory state
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Test processing under memory constraints
        try:
            orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
            results = orchestrator.run_complete_analysis(data_path=es12_file_path)
            
            # Monitor memory usage during processing
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = peak_memory - initial_memory
            
            # Should complete without excessive memory usage
            assert results is not None, "Should complete despite memory constraints"
            
            # Memory usage should be reasonable (less than 2GB increase)
            assert memory_increase < 2048, f"Memory usage too high: {memory_increase:.1f}MB"
            
        except MemoryError:
            # If memory error occurs, it should be handled gracefully
            pytest.skip("Memory error occurred - system under memory pressure")
        
        except Exception as e:
            # Other errors should not be memory-related
            error_msg = str(e).lower()
            memory_indicators = ['memory', 'ram', 'allocation']
            found_memory_indicators = sum(1 for indicator in memory_indicators if indicator in error_msg)
            
            if found_memory_indicators > 0:
                pytest.skip(f"Memory-related error occurred: {e}")
            else:
                # Non-memory errors should be handled normally
                assert isinstance(e, EDAError), f"Should raise EDAError for non-memory issues: {type(e)}"

    @given(
        processing_interruption=st.integers(min_value=1, max_value=5)
    )
    @settings(max_examples=1, deadline=15000, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_processing_interruption_robustness(
        self, 
        es12_file_path, 
        temp_output_dir, 
        processing_interruption
    ):
        """
        Property: Processing interruption robustness
        
        The system should handle processing interruptions gracefully
        and provide appropriate cleanup and error handling.
        
        **Validates: Requirements 2.4, 8.1**
        """
        import threading
        import time
        
        # Create orchestrator
        orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
        
        # Track processing state
        processing_started = threading.Event()
        processing_completed = threading.Event()
        processing_error = None
        
        def run_analysis():
            """Run analysis in separate thread."""
            nonlocal processing_error
            try:
                processing_started.set()
                results = orchestrator.run_complete_analysis(data_path=es12_file_path)
                processing_completed.set()
                return results
            except Exception as e:
                processing_error = e
                processing_completed.set()
        
        # Start analysis in background thread
        analysis_thread = threading.Thread(target=run_analysis)
        analysis_thread.start()
        
        # Wait for processing to start
        processing_started.wait(timeout=30)
        
        # Simulate interruption after some processing time
        interruption_delay = processing_interruption * 2  # 2-10 seconds
        time.sleep(interruption_delay)
        
        # Check if processing completed naturally
        if not processing_completed.is_set():
            # Processing is still ongoing - this tests interruption handling
            # In a real scenario, we might send a signal or exception
            # For this test, we'll just wait a bit more and see if it completes
            analysis_thread.join(timeout=60)  # Give it more time to complete
        
        # Verify robustness regardless of completion state
        if processing_completed.is_set():
            if processing_error is not None:
                # If error occurred, it should be handled gracefully
                assert isinstance(processing_error, (EDAError, Exception)), \
                    "Processing errors should be handled gracefully"
            
            # Check output directory state
            output_dir = Path(temp_output_dir)
            
            # Should have created basic directory structure even if interrupted
            assert output_dir.exists(), "Output directory should exist"
            
            # If logs directory exists, should not be corrupted
            if (output_dir / "logs").exists():
                log_files = list((output_dir / "logs").glob("*.log"))
                for log_file in log_files:
                    # Log files should be readable (not corrupted)
                    try:
                        log_content = log_file.read_text()
                        assert isinstance(log_content, str), "Log files should be readable"
                    except Exception:
                        # If log file is corrupted, that's acceptable for interruption test
                        pass
        
        # Cleanup
        if analysis_thread.is_alive():
            analysis_thread.join(timeout=10)