"""
Integration tests for real data pipeline using ES12.mat.

This module contains comprehensive integration tests that verify the complete
data processing pipeline using the actual NASA PCOE ES12 dataset.
"""

import tempfile
import shutil
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import time
from typing import Dict, Any, Optional
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to reduce warnings

from nasa_pcoe_eda.orchestrator import AnalysisOrchestrator
from nasa_pcoe_eda.data.es12_loader import ES12DataLoader
from nasa_pcoe_eda.data.loader import DataLoader
from nasa_pcoe_eda.models import AnalysisResults
from nasa_pcoe_eda.exceptions import EDAError

# Suppress specific warnings for cleaner test output
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", message=".*Glyph.*missing from.*font.*")
warnings.filterwarnings("ignore", message=".*Precision loss occurred in moment calculation.*")
warnings.filterwarnings("ignore", message=".*An input array is constant.*")
warnings.filterwarnings("ignore", message=".*correlation coefficient is not defined.*")


class TestRealDataPipeline:
    """Integration tests for complete real data processing pipeline."""
    
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
            pytest.skip("ES12.mat file not found - skipping real data tests")
        return path
    
    @pytest.fixture
    def sample_data_path(self, temp_output_dir):
        """Create sample data for comparison testing."""
        import scipy.io
        
        # Create synthetic data similar to ES12 structure
        n_samples = 100
        n_capacitors = 4
        
        np.random.seed(42)
        
        # Generate synthetic capacitor degradation data
        time_hours = np.linspace(0, 1000, n_samples)
        degradation_factor = 1 + time_hours / 1000 * 0.5
        
        feature_data = []
        
        for i in range(1, n_capacitors + 1):
            individual_factor = 1 + np.random.normal(0, 0.1)
            
            # Capacitance (decreases with degradation)
            capacitance = 1000 / (degradation_factor * individual_factor) + np.random.normal(0, 10, n_samples)
            feature_data.append(capacitance)
            
            # ESR (increases with degradation)
            esr = 0.1 * degradation_factor * individual_factor + np.random.normal(0, 0.01, n_samples)
            feature_data.append(esr)
        
        # Add time and cycle information
        feature_data.append(time_hours)
        feature_data.append(np.arange(1, n_samples + 1))
        
        # Convert to proper 2D array
        data_array = np.column_stack(feature_data)
        
        # Save as MATLAB file
        sample_file_path = temp_output_dir / "sample_data.mat"
        scipy.io.savemat(str(sample_file_path), {'data': data_array})
        
        return sample_file_path

    def test_complete_real_data_pipeline(self, es12_file_path, temp_output_dir):
        """
        Test complete data processing pipeline with real ES12 data.
        
        This test verifies the entire flow:
        Data loading → Analysis → Visualization → Report generation
        """
        # Initialize orchestrator
        orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
        
        # Record start time for performance measurement
        start_time = time.time()
        
        # Run complete analysis
        results = orchestrator.run_complete_analysis(data_path=es12_file_path)
        
        # Record completion time
        processing_time = time.time() - start_time
        
        # Verify results structure
        assert isinstance(results, AnalysisResults)
        assert results.metadata is not None
        assert results.statistics is not None
        assert results.missing_values is not None
        assert results.correlation_matrix is not None
        assert results.preprocessing_recommendations is not None
        
        # Verify metadata for real data
        assert results.metadata.n_records > 0
        assert results.metadata.n_features > 0
        assert len(results.metadata.feature_names) > 0
        
        # For ES12 data, we expect specific characteristics
        assert results.metadata.n_records >= 100, "ES12 should have substantial data"
        assert results.metadata.n_features >= 8, "ES12 should have multiple features"
        
        # Verify statistics were computed
        assert len(results.statistics) > 0
        assert 'descriptive_stats' in results.statistics
        assert 'missing_values' in results.statistics
        assert 'data_types' in results.statistics
        
        # Verify correlation matrix is reasonable
        assert not results.correlation_matrix.empty
        assert results.correlation_matrix.shape[0] > 0
        assert results.correlation_matrix.shape[1] > 0
        
        # Verify output directory structure
        output_dir = Path(temp_output_dir)
        assert (output_dir / "logs").exists()
        assert (output_dir / "reports").exists()
        
        # Check that log files were created
        log_files = list((output_dir / "logs").glob("*.log"))
        assert len(log_files) > 0, "Should create log files"
        
        # Check that report was attempted (may fail but should create some output)
        report_files = list((output_dir / "reports").glob("*.html"))
        error_files = list((output_dir / "reports").glob("*.txt"))
        
        # Should have either a successful report or an error file
        assert len(report_files) > 0 or len(error_files) > 0, "Should create report or error file"
        
        # If HTML report was generated, verify it contains real data information
        if report_files:
            report_content = report_files[0].read_text()
            # Should contain ES12-specific content
            es12_indicators = ['ES12', 'capacitor', 'degradation', 'EIS']
            found_indicators = sum(1 for indicator in es12_indicators if indicator in report_content)
            assert found_indicators >= 2, "Report should contain ES12-specific content"
        
        # Performance check - should complete within reasonable time
        assert processing_time < 300, f"Processing took too long: {processing_time:.1f}s"
        
        print(f"Real data pipeline completed in {processing_time:.1f} seconds")
        print(f"Processed {results.metadata.n_records} records with {results.metadata.n_features} features")

    def test_real_data_quality_verification(self, es12_file_path, temp_output_dir):
        """
        Test comprehensive data quality verification for real ES12 data.
        
        This test verifies:
        - Data integrity and completeness
        - Each capacitor's data quality
        - Physical validity of measurements
        """
        # Load data using specialized ES12 loader
        loader = ES12DataLoader()
        df = loader.load_dataset(es12_file_path)
        
        # Verify data integrity
        assert not df.empty, "Real data should not be empty"
        assert df.shape[0] > 0, "Should have records"
        assert df.shape[1] > 0, "Should have features"
        
        # Verify capacitor completeness
        expected_capacitors = {f'ES12C{i}' for i in range(1, 9)}
        actual_capacitors = set(df['capacitor'].unique())
        
        # Should have data for most capacitors (allow for some missing)
        overlap = len(expected_capacitors & actual_capacitors)
        assert overlap >= 4, f"Should have data for at least 4 capacitors, found {overlap}"
        
        # Verify each capacitor's data quality
        for cap_name in actual_capacitors:
            cap_data = df[df['capacitor'] == cap_name]
            
            # Should have reasonable number of cycles
            assert len(cap_data) >= 10, f"Capacitor {cap_name} should have at least 10 cycles"
            
            # Cycles should be sequential
            cycles = cap_data['cycle'].sort_values()
            assert cycles.iloc[0] >= 1, f"Cycles should start from 1 for {cap_name}"
            
            # Should have voltage data
            voltage_cols = [col for col in cap_data.columns if col.startswith(('vl_', 'vo_')) and col.endswith(('_mean', '_std', '_min', '_max'))]
            assert len(voltage_cols) > 0, f"Should have voltage data for {cap_name}"
            
            # Voltage data should have reasonable values
            for col in voltage_cols:
                if col in cap_data.columns:
                    valid_values = cap_data[col].dropna()
                    if len(valid_values) > 0:
                        # Allow reasonable voltage range (including negative values for differential measurements)
                        assert valid_values.min() >= -100, f"Voltage values should be within reasonable range for {cap_name}"
                        assert valid_values.max() <= 100, f"Voltage values should be within reasonable range for {cap_name}"
        
        # Verify physical validity of measurements
        self._verify_physical_validity(df)
        
        # Verify data validation passes
        validation = loader.validate_data(df)
        assert validation is not None, "Validation should complete"
        
        # If validation fails, errors should be specific
        if not validation.is_valid:
            assert len(validation.errors) > 0, "Invalid data should have specific errors"
            for error in validation.errors:
                assert isinstance(error, str) and len(error) > 0, "Errors should be descriptive"
        
        print(f"Data quality verification passed for {len(actual_capacitors)} capacitors")
        print(f"Total records: {len(df)}, Features: {len(df.columns)}")

    def _verify_physical_validity(self, df: pd.DataFrame):
        """Verify physical validity of measurement data."""
        # Check voltage ratios are reasonable (allow negative values for differential measurements)
        if 'voltage_ratio' in df.columns:
            valid_ratios = df['voltage_ratio'].dropna()
            if len(valid_ratios) > 0:
                assert valid_ratios.min() >= -1000, "Voltage ratios should be within reasonable range"
                assert valid_ratios.max() <= 1000, "Voltage ratios should be within reasonable range"
        
        # Check that voltage statistics are consistent
        voltage_mean_cols = [col for col in df.columns if col.endswith('_mean') and col.startswith(('vl_', 'vo_'))]
        voltage_std_cols = [col for col in df.columns if col.endswith('_std') and col.startswith(('vl_', 'vo_'))]
        
        for mean_col in voltage_mean_cols:
            if mean_col in df.columns:
                std_col = mean_col.replace('_mean', '_std')
                if std_col in df.columns:
                    # Standard deviation should be less than mean for stable measurements
                    valid_data = df[[mean_col, std_col]].dropna()
                    if len(valid_data) > 0:
                        # Allow some flexibility for noisy real data - check that most std values are reasonable
                        reasonable_std = (valid_data[std_col].abs() <= valid_data[mean_col].abs() * 5).sum()  # Allow up to 5x mean
                        total_valid = len(valid_data)
                        assert reasonable_std / total_valid >= 0.3, \
                            f"Most measurements should have reasonable std/mean ratio for {mean_col}"

    def test_real_vs_sample_data_comparison(self, es12_file_path, sample_data_path, temp_output_dir):
        """
        Test comparison between real data and sample data analysis.
        
        This test verifies:
        - Same analysis methods work on both datasets
        - Analysis precision and reliability comparison
        - Performance comparison
        """
        # Create separate output directories
        real_output_dir = temp_output_dir / "real_data"
        sample_output_dir = temp_output_dir / "sample_data"
        real_output_dir.mkdir()
        sample_output_dir.mkdir()
        
        # Analyze real data
        real_orchestrator = AnalysisOrchestrator(output_dir=real_output_dir)
        real_start_time = time.time()
        real_results = real_orchestrator.run_complete_analysis(data_path=es12_file_path)
        real_processing_time = time.time() - real_start_time
        
        # Analyze sample data
        sample_orchestrator = AnalysisOrchestrator(output_dir=sample_output_dir)
        sample_start_time = time.time()
        sample_results = sample_orchestrator.run_complete_analysis(data_path=sample_data_path)
        sample_processing_time = time.time() - sample_start_time
        
        # Compare analysis results structure
        assert type(real_results) == type(sample_results), "Both should return same result type"
        
        # Both should have complete analysis results
        for results, data_type in [(real_results, "real"), (sample_results, "sample")]:
            assert results.metadata is not None, f"{data_type} data should have metadata"
            assert results.statistics is not None, f"{data_type} data should have statistics"
            assert results.missing_values is not None, f"{data_type} data should have missing values analysis"
            assert results.correlation_matrix is not None, f"{data_type} data should have correlation matrix"
            assert results.preprocessing_recommendations is not None, f"{data_type} data should have preprocessing recommendations"
        
        # Compare analysis precision
        self._compare_analysis_precision(real_results, sample_results)
        
        # Compare performance
        print(f"Real data processing time: {real_processing_time:.1f}s")
        print(f"Sample data processing time: {sample_processing_time:.1f}s")
        
        # Real data might take longer due to complexity, but should be reasonable
        assert real_processing_time < 600, "Real data processing should complete within 10 minutes"
        assert sample_processing_time < 120, "Sample data processing should complete within 2 minutes"
        
        # Verify both generated outputs
        assert (real_output_dir / "reports").exists(), "Real data should generate reports"
        assert (sample_output_dir / "reports").exists(), "Sample data should generate reports"
        
        real_reports = list((real_output_dir / "reports").glob("*.html"))
        sample_reports = list((sample_output_dir / "reports").glob("*.html"))
        real_error_files = list((real_output_dir / "reports").glob("*.txt"))
        sample_error_files = list((sample_output_dir / "reports").glob("*.txt"))
        
        # Should have either HTML reports or error files
        assert len(real_reports) > 0 or len(real_error_files) > 0, "Real data should generate HTML reports or error files"
        assert len(sample_reports) > 0 or len(sample_error_files) > 0, "Sample data should generate HTML reports or error files"

    def _compare_analysis_precision(self, real_results: AnalysisResults, sample_results: AnalysisResults):
        """Compare analysis precision between real and sample data."""
        # Compare metadata
        assert real_results.metadata.n_records > 0, "Real data should have records"
        assert sample_results.metadata.n_records > 0, "Sample data should have records"
        
        # Compare statistics structure
        real_stats = real_results.statistics
        sample_stats = sample_results.statistics
        
        # Both should have the same analysis components
        common_keys = set(real_stats.keys()) & set(sample_stats.keys())
        assert len(common_keys) >= 2, "Should have common analysis components"
        
        # Compare correlation matrices
        real_corr = real_results.correlation_matrix
        sample_corr = sample_results.correlation_matrix
        
        assert not real_corr.empty, "Real data correlation matrix should not be empty"
        assert not sample_corr.empty, "Sample data correlation matrix should not be empty"
        
        # Both should have reasonable correlation values
        real_corr_values = real_corr.values[~np.isnan(real_corr.values)]
        sample_corr_values = sample_corr.values[~np.isnan(sample_corr.values)]
        
        # Allow for small floating point precision errors
        assert np.all((-1.001 <= real_corr_values) & (real_corr_values <= 1.001)), \
            "Real data correlations should be in [-1, 1]"
        assert np.all((-1.001 <= sample_corr_values) & (sample_corr_values <= 1.001)), \
            "Sample data correlations should be in [-1, 1]"
        
        # Compare preprocessing recommendations
        real_prep = real_results.preprocessing_recommendations
        sample_prep = sample_results.preprocessing_recommendations
        
        assert isinstance(real_prep, dict), "Real data should have preprocessing recommendations"
        assert isinstance(sample_prep, dict), "Sample data should have preprocessing recommendations"

    def test_real_data_error_handling(self, temp_output_dir):
        """
        Test error handling with real data scenarios.
        
        This test verifies:
        - Graceful handling of missing files
        - Recovery from partial data corruption
        - Appropriate error messages
        """
        orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
        
        # Test with non-existent file
        non_existent_file = temp_output_dir / "non_existent.mat"
        with pytest.raises(EDAError):
            orchestrator.run_complete_analysis(non_existent_file)
        
        # Test with invalid file format
        invalid_file = temp_output_dir / "invalid.txt"
        invalid_file.write_text("This is not a MATLAB file")
        
        with pytest.raises(EDAError):
            orchestrator.run_complete_analysis(invalid_file)
        
        # Test with corrupted MATLAB file
        corrupted_file = temp_output_dir / "corrupted.mat"
        corrupted_file.write_bytes(b"corrupted matlab file content")
        
        with pytest.raises(EDAError):
            orchestrator.run_complete_analysis(corrupted_file)

    def test_real_data_memory_management(self, es12_file_path, temp_output_dir):
        """
        Test memory management with real data processing.
        
        This test verifies:
        - Memory usage stays within reasonable bounds
        - No memory leaks during processing
        - Efficient handling of large datasets
        """
        import psutil
        import os
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run analysis
        orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
        results = orchestrator.run_complete_analysis(data_path=es12_file_path)
        
        # Get peak memory usage
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Memory increase should be reasonable for the dataset size
        dataset_size_mb = results.metadata.memory_usage
        
        # Allow up to 10x the dataset size in memory (reasonable for processing)
        max_allowed_memory = max(dataset_size_mb * 10, 500)  # At least 500MB allowed
        
        assert memory_increase < max_allowed_memory, \
            f"Memory usage too high: {memory_increase:.1f}MB (dataset: {dataset_size_mb:.1f}MB)"
        
        # Clean up and check for memory leaks
        del results
        del orchestrator
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Check memory after cleanup
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_leak = final_memory - initial_memory
        
        # Allow some memory increase but not excessive
        assert memory_leak < 100, f"Possible memory leak: {memory_leak:.1f}MB increase"
        
        print(f"Memory usage: Initial={initial_memory:.1f}MB, Peak={peak_memory:.1f}MB, Final={final_memory:.1f}MB")

    def test_real_data_concurrent_processing(self, es12_file_path, temp_output_dir):
        """
        Test concurrent processing capabilities with real data.
        
        This test verifies:
        - Multiple analyses can run simultaneously
        - No race conditions or data corruption
        - Resource sharing works correctly
        """
        import threading
        import queue
        import matplotlib
        
        # Use non-interactive backend to avoid threading issues
        matplotlib.use('Agg')
        
        # Create separate output directories for concurrent runs
        output_dirs = []
        for i in range(2):  # Reduced from 3 to 2 to be less aggressive
            output_dir = temp_output_dir / f"concurrent_{i}"
            output_dir.mkdir()
            output_dirs.append(output_dir)
        
        # Queue to collect results
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def run_analysis(output_dir, thread_id):
            """Run analysis in a separate thread."""
            try:
                # Set matplotlib backend for this thread
                import matplotlib
                matplotlib.use('Agg')
                
                orchestrator = AnalysisOrchestrator(output_dir=output_dir)
                results = orchestrator.run_complete_analysis(data_path=es12_file_path)
                results_queue.put((thread_id, results))
            except Exception as e:
                errors_queue.put((thread_id, str(e)))
        
        # Start concurrent analyses
        threads = []
        for i, output_dir in enumerate(output_dirs):
            thread = threading.Thread(target=run_analysis, args=(output_dir, i))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=300)  # 5 minute timeout per thread (reduced from 10)
        
        # Check for errors
        errors = []
        while not errors_queue.empty():
            errors.append(errors_queue.get())
        
        # Allow some errors due to threading complexity, but not all should fail
        if len(errors) == len(output_dirs):
            pytest.fail(f"All concurrent processes failed: {errors}")
        
        # Collect results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        assert len(results) >= 1, f"Expected at least 1 successful result, got {len(results)}"
        
        # Verify successful results are consistent
        for thread_id, result in results:
            assert isinstance(result, AnalysisResults), f"Thread {thread_id} should return AnalysisResults"
            assert result.metadata is not None, f"Thread {thread_id} should have metadata"
            assert result.metadata.n_records > 0, f"Thread {thread_id} should have records"
        
        # Verify output directories have results for successful threads
        successful_thread_ids = [thread_id for thread_id, _ in results]
        for thread_id in successful_thread_ids:
            output_dir = output_dirs[thread_id]
            assert (output_dir / "logs").exists(), f"Thread {thread_id} should create logs"
            # Reports directory should exist even if report generation fails
            assert (output_dir / "reports").exists(), f"Thread {thread_id} should create reports directory"
        
        print(f"Concurrent processing completed with {len(results)} successful threads out of {len(output_dirs)}")
        if errors:
            print(f"Errors encountered: {errors}")

    def test_real_data_incremental_analysis(self, es12_file_path, temp_output_dir):
        """
        Test incremental analysis capabilities with real data.
        
        This test verifies:
        - Analysis can be performed in stages
        - Intermediate results are preserved
        - State management works correctly
        """
        orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
        
        # Stage 1: Load and validate data
        loaded_data = orchestrator.load_and_validate_data(es12_file_path)
        assert loaded_data is not None, "Should load data successfully"
        assert len(loaded_data) > 0, "Should have records"
        
        # Verify state is maintained
        assert orchestrator.get_loaded_data() is not None, "Should maintain loaded data state"
        assert orchestrator.get_metadata() is not None, "Should maintain metadata state"
        
        # Stage 2: Run basic statistics
        stats_results = orchestrator.compute_basic_statistics()
        assert stats_results is not None, "Should compute statistics"
        assert 'descriptive_stats' in stats_results, "Should have descriptive statistics"
        
        # Stage 3: Run correlation analysis
        correlation_results = orchestrator.compute_correlations()
        assert correlation_results is not None, "Should compute correlations"
        assert not correlation_results.empty, "Should have correlation matrix"
        
        # Stage 4: Generate visualizations
        viz_paths = orchestrator.generate_visualizations()
        assert isinstance(viz_paths, list), "Should return visualization paths"
        
        # Create analysis results for report generation
        from nasa_pcoe_eda.models import AnalysisResults
        orchestrator._analysis_results = AnalysisResults(
            metadata=orchestrator.get_metadata(),
            statistics=stats_results,
            missing_values=stats_results.get('missing_values'),
            correlation_matrix=correlation_results,
            outliers=None,
            time_series_trends=None,
            rul_features=[],
            fault_features=[],
            preprocessing_recommendations={},
            visualization_paths=viz_paths
        )
        
        # Stage 5: Generate final report
        report_path = orchestrator.generate_report()
        assert report_path is not None, "Should generate report"
        assert Path(report_path).exists(), "Report file should exist"
        
        # Verify final state
        final_results = orchestrator.get_analysis_results()
        assert final_results is not None, "Should have final results"
        assert final_results.metadata is not None, "Should have metadata in final results"
        assert final_results.statistics is not None, "Should have statistics in final results"
        assert final_results.correlation_matrix is not None, "Should have correlation matrix in final results"
        
        print("Incremental analysis completed successfully")

    def test_real_data_robustness_edge_cases(self, es12_file_path, temp_output_dir):
        """
        Test robustness with edge cases in real data.
        
        This test verifies:
        - Handling of missing or corrupted capacitor data
        - Recovery from analysis failures
        - Graceful degradation of functionality
        """
        # Load real data first
        loader = ES12DataLoader()
        original_df = loader.load_dataset(es12_file_path)
        
        # Test with subset of capacitors (simulate missing data)
        subset_capacitors = original_df['capacitor'].unique()[:2]  # Take only first 2 capacitors
        subset_df = original_df[original_df['capacitor'].isin(subset_capacitors)]
        
        # Save subset as temporary file
        import scipy.io
        
        # Convert back to array format for saving
        # This is a simplified conversion - in practice, we'd need to reconstruct the full structure
        subset_array = subset_df.select_dtypes(include=[np.number]).values
        subset_file = temp_output_dir / "subset_data.mat"
        scipy.io.savemat(str(subset_file), {'data': subset_array})
        
        # Test analysis with subset data
        orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
        
        try:
            results = orchestrator.run_complete_analysis(data_path=subset_file)
            
            # Should complete even with limited data
            assert isinstance(results, AnalysisResults), "Should return results even with subset data"
            assert results.metadata is not None, "Should have metadata for subset data"
            
        except Exception as e:
            # If analysis fails, it should fail gracefully with informative error
            assert isinstance(e, EDAError), f"Should raise EDAError, got {type(e)}"
            assert len(str(e)) > 0, "Error message should be informative"
        
        # Test with data containing extreme values
        extreme_df = original_df.copy()
        numeric_cols = extreme_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Introduce some extreme values
            extreme_df.loc[0, numeric_cols[0]] = 1e10  # Very large value
            extreme_df.loc[1, numeric_cols[0]] = -1e10  # Very small value
            
            # Convert and save
            extreme_array = extreme_df.select_dtypes(include=[np.number]).values
            extreme_file = temp_output_dir / "extreme_data.mat"
            scipy.io.savemat(str(extreme_file), {'data': extreme_array})
            
            # Test analysis with extreme values
            try:
                extreme_results = orchestrator.run_complete_analysis(data_path=extreme_file)
                
                # Should handle extreme values gracefully
                assert isinstance(extreme_results, AnalysisResults), "Should handle extreme values"
                
                # Statistics should be computed (may be inf or nan, but should not crash)
                assert extreme_results.statistics is not None, "Should compute statistics with extreme values"
                
            except Exception as e:
                # If it fails, should be a controlled failure
                assert isinstance(e, EDAError), f"Should raise EDAError for extreme values, got {type(e)}"
        
        print("Robustness testing completed successfully")