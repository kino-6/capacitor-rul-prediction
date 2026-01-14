"""Integration tests for RUL analysis workflow."""

import tempfile
import shutil
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import scipy.io

from nasa_pcoe_eda.orchestrator import AnalysisOrchestrator
from nasa_pcoe_eda.analysis.rul_features import RULFeatureAnalyzer
from nasa_pcoe_eda.models import AnalysisResults
from nasa_pcoe_eda.exceptions import EDAError


class TestRULWorkflow:
    """Integration tests for the RUL analysis workflow."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def rul_dataset(self, temp_output_dir):
        """Create a synthetic dataset with RUL patterns for testing."""
        # Create synthetic capacitor degradation data similar to NASA PCOE Dataset No.12
        n_samples = 150  # Sufficient samples for trend analysis
        n_capacitors = 4
        
        np.random.seed(42)  # For reproducible results
        
        # Generate time-based features
        time_hours = np.linspace(0, 1500, n_samples)  # 1500 hours of operation
        cycles = np.arange(1, n_samples + 1)
        
        # Create degradation patterns
        degradation_factor = 1 + (time_hours / 1500) * 0.6  # 60% degradation over time
        
        feature_data = []
        feature_names = []
        
        # Generate capacitor data with clear degradation patterns
        for i in range(1, n_capacitors + 1):
            cap_name = f'C{i}'
            
            # Individual variation for each capacitor
            individual_factor = 1 + np.random.normal(0, 0.05)
            
            # Capacitance (decreases with degradation) - strong RUL correlation
            base_capacitance = 1000
            capacitance = base_capacitance / (degradation_factor * individual_factor)
            capacitance += np.random.normal(0, 5, n_samples)  # Small noise
            feature_data.append(capacitance)
            feature_names.append(f'{cap_name}_Capacitance')
            
            # ESR (increases with degradation) - strong RUL correlation
            base_esr = 0.1
            esr = base_esr * degradation_factor * individual_factor
            esr += np.random.normal(0, 0.005, n_samples)  # Small noise
            feature_data.append(esr)
            feature_names.append(f'{cap_name}_ESR')
            
            # Impedance magnitude (increases with degradation) - moderate RUL correlation
            base_impedance = 0.5
            impedance = base_impedance * (degradation_factor ** 0.5) * individual_factor
            impedance += np.random.normal(0, 0.02, n_samples)
            feature_data.append(impedance)
            feature_names.append(f'{cap_name}_Impedance')
        
        # Add time-related features
        feature_data.append(time_hours)
        feature_names.append('Time_Hours')
        
        feature_data.append(cycles)
        feature_names.append('Cycle')
        
        # Add temperature (weakly correlated with degradation)
        temperature = 25 + 0.01 * time_hours + np.random.normal(0, 2, n_samples)
        feature_data.append(temperature)
        feature_names.append('Temperature')
        
        # Add a noise feature (no correlation with RUL)
        noise_feature = np.random.normal(0, 10, n_samples)
        feature_data.append(noise_feature)
        feature_names.append('Noise_Feature')
        
        # Calculate RUL (Remaining Useful Life)
        # RUL decreases as degradation increases
        max_degradation = 1.6  # Failure threshold
        current_degradation = degradation_factor
        rul = np.maximum(0, (max_degradation - current_degradation) / max_degradation * 1500)
        feature_data.append(rul)
        feature_names.append('RUL')
        
        # Convert to proper 2D array (samples x features)
        data_array = np.column_stack(feature_data)
        
        # Save as MATLAB file
        mat_file_path = temp_output_dir / "rul_dataset.mat"
        scipy.io.savemat(str(mat_file_path), {'data': data_array})
        
        return mat_file_path, feature_names
    
    @pytest.fixture
    def csv_rul_dataset(self, temp_output_dir):
        """Create a MAT dataset with RUL patterns for testing (using CSV-like structure)."""
        # Create synthetic data with clear RUL patterns
        n_samples = 100
        
        np.random.seed(42)
        
        # Generate time series
        time_values = np.arange(n_samples)
        
        # Generate features with different RUL correlation strengths
        # Strong correlation features
        feature1 = 100 - 0.8 * time_values + np.random.normal(0, 2, n_samples)  # Strong negative correlation
        feature2 = 10 + 0.6 * time_values + np.random.normal(0, 1, n_samples)   # Strong positive correlation
        
        # Moderate correlation features
        feature3 = 50 - 0.3 * time_values + np.random.normal(0, 3, n_samples)   # Moderate negative correlation
        feature4 = 20 + 0.2 * time_values + np.random.normal(0, 2, n_samples)   # Moderate positive correlation
        
        # Weak/no correlation features
        feature5 = np.random.normal(30, 5, n_samples)  # No correlation
        feature6 = 15 + 0.05 * time_values + np.random.normal(0, 4, n_samples)  # Very weak correlation
        
        # RUL values (decreasing over time)
        rul = 100 - time_values + np.random.normal(0, 1, n_samples)
        rul = np.maximum(rul, 0)  # Ensure non-negative RUL
        
        # Create data array (samples x features)
        feature_data = [
            time_values,
            feature1,
            feature2,
            feature3,
            feature4,
            feature5,
            feature6,
            rul
        ]
        
        data_array = np.column_stack(feature_data)
        
        # Save as MAT file
        mat_file_path = temp_output_dir / "rul_dataset.mat"
        scipy.io.savemat(str(mat_file_path), {'data': data_array})
        
        # Create DataFrame for reference
        df = pd.DataFrame({
            'Time': time_values,
            'Strong_Neg_Feature': feature1,
            'Strong_Pos_Feature': feature2,
            'Moderate_Neg_Feature': feature3,
            'Moderate_Pos_Feature': feature4,
            'No_Corr_Feature': feature5,
            'Weak_Corr_Feature': feature6,
            'RUL': rul
        })
        
        return mat_file_path, df
    
    def test_complete_rul_workflow_with_mat_data(self, rul_dataset, temp_output_dir):
        """Test complete RUL analysis workflow with MATLAB data."""
        mat_file_path, feature_names = rul_dataset
        
        # Initialize orchestrator
        orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
        
        # Run complete analysis with RUL column specified
        # Note: MAT files loaded by scipy.io will have numeric column names
        # The RUL column should be the last one (index depends on total columns)
        # Let's first run without RUL column to see what columns we get
        temp_results = orchestrator.run_complete_analysis(data_path=mat_file_path)
        
        # Get the loaded data to see column names
        loaded_data = orchestrator.get_loaded_data()
        column_names = list(loaded_data.columns)
        
        # The RUL column should be the last one
        rul_column_name = str(len(column_names) - 1)  # Last column (0-indexed)
        
        # Now run with proper RUL column specification
        results = orchestrator.run_complete_analysis(
            data_path=mat_file_path,
            rul_column=rul_column_name
        )
        
        # Verify results structure
        assert isinstance(results, AnalysisResults)
        assert results.metadata is not None
        assert results.rul_features is not None
        
        # Verify RUL features were identified
        assert isinstance(results.rul_features, list)
        assert len(results.rul_features) > 0, "Should identify RUL-relevant features"
        
        # Verify RUL features are properly ranked (sorted by absolute correlation)
        if len(results.rul_features) > 1:
            for i in range(len(results.rul_features) - 1):
                current_abs_corr = abs(results.rul_features[i][1])
                next_abs_corr = abs(results.rul_features[i + 1][1])
                assert current_abs_corr >= next_abs_corr, (
                    f"RUL features not properly ranked: "
                    f"{results.rul_features[i][0]}({current_abs_corr:.3f}) should be >= "
                    f"{results.rul_features[i+1][0]}({next_abs_corr:.3f})"
                )
        
        # Verify correlation values are valid
        for feature_name, correlation in results.rul_features:
            assert -1.0 <= correlation <= 1.0, (
                f"Invalid correlation {correlation} for feature {feature_name}"
            )
            assert isinstance(feature_name, str), "Feature name should be string"
            assert isinstance(correlation, (float, np.floating)), "Correlation should be numeric"
        
        # Verify that strongly correlated features are identified
        # Since MAT files have numeric column names, we can't check for specific feature names
        # Instead, verify that we have a reasonable number of features with good correlations
        strong_correlations = [corr for _, corr in results.rul_features if abs(corr) > 0.5]
        moderate_correlations = [corr for _, corr in results.rul_features if abs(corr) > 0.3]
        
        # Should have at least some moderately correlated features
        assert len(moderate_correlations) > 0, "Should identify at least some moderately correlated features"
        
        # Log the actual features found for debugging
        print(f"Found {len(results.rul_features)} RUL features:")
        for name, corr in results.rul_features[:5]:  # Show top 5
            print(f"  {name}: {corr:.3f}")
        
        # Verify preprocessing recommendations include RUL-specific suggestions
        assert isinstance(results.preprocessing_recommendations, dict)
        
        # Check that output files were created
        output_dir = Path(temp_output_dir)
        assert (output_dir / "logs").exists()
        assert (output_dir / "reports").exists()
    
    def test_complete_rul_workflow_with_csv_data(self, csv_rul_dataset, temp_output_dir):
        """Test complete RUL analysis workflow with MAT data (CSV-like structure)."""
        mat_file_path, original_df = csv_rul_dataset
        
        # Initialize orchestrator
        orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
        
        # Run complete analysis with RUL column specified
        # Note: Since we're using MAT files, column names will be numeric indices
        # We need to find which column corresponds to RUL (should be the last one)
        results = orchestrator.run_complete_analysis(
            data_path=mat_file_path,
            rul_column='7'  # RUL is the 8th column (0-indexed as '7')
        )
        
        # Verify results structure
        assert isinstance(results, AnalysisResults)
        assert results.rul_features is not None
        assert len(results.rul_features) > 0
        
        # Verify that features are ranked by correlation strength
        # Since column names will be numeric, we can't directly check feature names
        # but we can verify the ranking order
        if len(results.rul_features) > 1:
            # Check that ranking is in descending order of absolute correlation
            for i in range(len(results.rul_features) - 1):
                current_abs_corr = abs(results.rul_features[i][1])
                next_abs_corr = abs(results.rul_features[i + 1][1])
                assert current_abs_corr >= next_abs_corr, (
                    f"RUL features not properly ranked: "
                    f"{results.rul_features[i][0]}({current_abs_corr:.3f}) should be >= "
                    f"{results.rul_features[i+1][0]}({next_abs_corr:.3f})"
                )
        
        # Verify correlation values make sense
        for feature_name, correlation in results.rul_features:
            assert -1.0 <= correlation <= 1.0, (
                f"Invalid correlation {correlation} for feature {feature_name}"
            )
            # Should have some reasonably strong correlations
            if len(results.rul_features) > 0:
                max_abs_corr = max(abs(corr) for _, corr in results.rul_features)
                assert max_abs_corr > 0.3, "Should identify at least one moderately correlated feature"
    
    def test_rul_workflow_without_rul_column(self, csv_rul_dataset, temp_output_dir):
        """Test RUL workflow when no RUL column is specified."""
        mat_file_path, _ = csv_rul_dataset
        
        # Initialize orchestrator
        orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
        
        # Run analysis without specifying RUL column
        results = orchestrator.run_complete_analysis(data_path=mat_file_path)
        
        # Should still complete analysis but with degradation feature identification
        assert isinstance(results, AnalysisResults)
        assert isinstance(results.rul_features, list)
        
        # May have identified degradation features based on time trends
        # (this depends on the degradation feature identification logic)
    
    def test_rul_workflow_with_invalid_rul_column(self, csv_rul_dataset, temp_output_dir):
        """Test RUL workflow with invalid RUL column name."""
        mat_file_path, _ = csv_rul_dataset
        
        # Initialize orchestrator
        orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
        
        # Run analysis with non-existent RUL column
        results = orchestrator.run_complete_analysis(
            data_path=mat_file_path,
            rul_column='NonExistentColumn'
        )
        
        # Should handle gracefully and fall back to degradation feature identification
        assert isinstance(results, AnalysisResults)
        assert isinstance(results.rul_features, list)
    
    def test_rul_analyzer_standalone_workflow(self, csv_rul_dataset, temp_output_dir):
        """Test RUL analyzer as standalone component."""
        mat_file_path, original_df = csv_rul_dataset
        
        # Load data using the orchestrator's data loader
        from nasa_pcoe_eda.data.loader import DataLoader
        loader = DataLoader()
        df = loader.load_dataset(mat_file_path)
        
        # Initialize RUL analyzer
        rul_analyzer = RULFeatureAnalyzer()
        
        # Test degradation feature identification
        degradation_features = rul_analyzer.identify_degradation_features(df)
        assert isinstance(degradation_features, list)
        
        # Test degradation rate computation
        if degradation_features:
            degradation_rates = rul_analyzer.compute_degradation_rates(df, degradation_features)
            assert isinstance(degradation_rates, dict)
            assert len(degradation_rates) > 0
            
            # Verify rates are numeric
            for feature, rate in degradation_rates.items():
                assert isinstance(rate, (float, np.floating))
                assert not np.isnan(rate)
        
        # Test RUL feature ranking - use column '7' as RUL (last column)
        if '7' in df.columns:
            rul_ranking = rul_analyzer.rank_features_for_rul(df, '7')
            assert isinstance(rul_ranking, list)
            assert len(rul_ranking) > 0
            
            # Verify ranking structure
            for feature_name, correlation in rul_ranking:
                assert isinstance(feature_name, str)
                assert isinstance(correlation, (float, np.floating))
                assert -1.0 <= correlation <= 1.0
        
        # Test visualization generation
        if degradation_features:
            figures = rul_analyzer.visualize_degradation_patterns(df, degradation_features[:2])
            assert isinstance(figures, list)
            # Should generate at least one figure
            assert len(figures) > 0
            
            # Clean up figures to avoid memory issues
            import matplotlib.pyplot as plt
            for fig in figures:
                plt.close(fig)
    
    def test_rul_workflow_with_missing_data(self, temp_output_dir):
        """Test RUL workflow with missing data."""
        # Create dataset with missing values
        n_samples = 50
        np.random.seed(42)
        
        time_values = np.arange(n_samples)
        feature1 = 100 - 0.8 * time_values + np.random.normal(0, 2, n_samples)
        feature2 = 10 + 0.6 * time_values + np.random.normal(0, 1, n_samples)
        rul = 100 - time_values + np.random.normal(0, 1, n_samples)
        
        # Introduce missing values
        feature1[10:15] = np.nan
        feature2[20:25] = np.nan
        rul[5:8] = np.nan
        
        # Create data array
        feature_data = [time_values, feature1, feature2, rul]
        data_array = np.column_stack(feature_data)
        
        mat_file_path = temp_output_dir / "rul_missing_data.mat"
        scipy.io.savemat(str(mat_file_path), {'data': data_array})
        
        # Initialize orchestrator
        orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
        
        # Run analysis - should handle missing data gracefully
        results = orchestrator.run_complete_analysis(
            data_path=mat_file_path,
            rul_column='3'  # RUL is the 4th column (0-indexed as '3')
        )
        
        # Should complete successfully despite missing data
        assert isinstance(results, AnalysisResults)
        assert isinstance(results.rul_features, list)
        
        # Should still identify some RUL features
        # (correlation analysis should handle missing values)
    
    def test_rul_workflow_with_minimal_data(self, temp_output_dir):
        """Test RUL workflow with minimal data."""
        # Create minimal dataset
        feature_data = [
            [1, 2, 3],  # Feature1
            [3, 2, 1],  # Feature2
            [10, 5, 1]  # RUL
        ]
        data_array = np.column_stack(feature_data)
        
        mat_file_path = temp_output_dir / "rul_minimal_data.mat"
        scipy.io.savemat(str(mat_file_path), {'data': data_array})
        
        # Initialize orchestrator
        orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
        
        # Run analysis - should handle minimal data gracefully
        results = orchestrator.run_complete_analysis(
            data_path=mat_file_path,
            rul_column='2'  # RUL is the 3rd column (0-indexed as '2')
        )
        
        # Should complete without errors
        assert isinstance(results, AnalysisResults)
        assert isinstance(results.rul_features, list)
        
        # May or may not identify features depending on correlation threshold
        # but should not crash
    
    def test_rul_workflow_state_consistency(self, csv_rul_dataset, temp_output_dir):
        """Test that RUL workflow maintains consistent state."""
        mat_file_path, _ = csv_rul_dataset
        
        # Initialize orchestrator
        orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
        
        # Run analysis
        results = orchestrator.run_complete_analysis(
            data_path=mat_file_path,
            rul_column='7'  # RUL is the 8th column (0-indexed as '7')
        )
        
        # Get state from orchestrator
        loaded_data = orchestrator.get_loaded_data()
        analysis_results = orchestrator.get_analysis_results()
        
        # Verify state consistency
        assert loaded_data is not None
        # Check that RUL column exists (could be integer 7 or string '7')
        assert 7 in loaded_data.columns or '7' in loaded_data.columns, (
            f"RUL column not found in loaded data columns: {list(loaded_data.columns)}"
        )
        assert analysis_results == results
        
        # Verify RUL features are consistent with loaded data
        for feature_name, _ in results.rul_features:
            # Convert feature name to match column type
            feature_col = feature_name
            if feature_name.isdigit():
                feature_col = int(feature_name)
            assert feature_col in loaded_data.columns or feature_name in loaded_data.columns, (
                f"RUL feature {feature_name} not found in loaded data columns: {list(loaded_data.columns)}"
            )
    
    def test_rul_workflow_error_handling(self, temp_output_dir):
        """Test error handling in RUL workflow."""
        # Test with non-existent file
        non_existent_file = temp_output_dir / "non_existent.csv"
        
        orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
        
        with pytest.raises(EDAError):
            orchestrator.run_complete_analysis(
                data_path=non_existent_file,
                rul_column='RUL'
            )
    
    def test_rul_workflow_output_files(self, csv_rul_dataset, temp_output_dir):
        """Test that RUL workflow generates appropriate output files."""
        mat_file_path, _ = csv_rul_dataset
        
        # Initialize orchestrator
        orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
        
        # Run analysis
        results = orchestrator.run_complete_analysis(
            data_path=mat_file_path,
            rul_column='7'  # RUL is the 8th column (0-indexed as '7')
        )
        
        # Check output directory structure
        output_dir = Path(temp_output_dir)
        assert (output_dir / "logs").exists()
        assert (output_dir / "reports").exists()
        
        # Check that report contains RUL analysis section
        report_files = list((output_dir / "reports").glob("*.html"))
        if report_files:
            report_content = report_files[0].read_text()
            # Should contain RUL-related content
            rul_indicators = ['RUL', 'Remaining Useful Life', 'rul', '劣化']
            found_indicators = sum(1 for indicator in rul_indicators if indicator in report_content)
            assert found_indicators > 0, "Report should contain RUL-related content"
        
        # Check visualization files if generated
        if results.visualization_paths:
            for viz_path in results.visualization_paths:
                if Path(viz_path).exists():
                    assert Path(viz_path).suffix in ['.png', '.jpg', '.pdf'], (
                        f"Unexpected visualization file format: {viz_path}"
                    )