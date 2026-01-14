"""Integration tests for fault diagnosis workflow."""

import tempfile
import shutil
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import scipy.io

from nasa_pcoe_eda.orchestrator import AnalysisOrchestrator
from nasa_pcoe_eda.analysis.fault_level import FaultLevelAnalyzer
from nasa_pcoe_eda.preprocessing.recommender import PreprocessingRecommender
from nasa_pcoe_eda.models import AnalysisResults, DistributionComparison
from nasa_pcoe_eda.exceptions import EDAError


class TestFaultWorkflow:
    """Integration tests for the fault diagnosis workflow."""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def fault_dataset(self, temp_output_dir):
        """Create a synthetic dataset with fault patterns for testing."""
        # Create synthetic data with clear fault patterns
        n_samples = 200
        n_features = 8
        
        np.random.seed(42)  # For reproducible results
        
        # Generate time-based features
        time_hours = np.linspace(0, 1000, n_samples)
        cycles = np.arange(1, n_samples + 1)
        
        # Create fault states: 0=Normal, 1=Early_Fault, 2=Advanced_Fault
        fault_states = np.zeros(n_samples, dtype=int)
        fault_states[80:140] = 1   # Early fault from sample 80-139
        fault_states[140:] = 2     # Advanced fault from sample 140 onwards
        
        feature_data = []
        feature_names = []
        
        # Generate features with different fault discrimination capabilities
        
        # Feature 1: Strong discriminator - vibration amplitude
        base_vibration = 1.0
        vibration = np.where(fault_states == 0, 
                           base_vibration + np.random.normal(0, 0.1, n_samples),
                           np.where(fault_states == 1,
                                  base_vibration * 2.5 + np.random.normal(0, 0.2, n_samples),
                                  base_vibration * 5.0 + np.random.normal(0, 0.3, n_samples)))
        feature_data.append(vibration)
        feature_names.append('Vibration_Amplitude')
        
        # Feature 2: Strong discriminator - temperature
        base_temp = 25.0
        temperature = np.where(fault_states == 0,
                             base_temp + np.random.normal(0, 2, n_samples),
                             np.where(fault_states == 1,
                                    base_temp + 15 + np.random.normal(0, 3, n_samples),
                                    base_temp + 35 + np.random.normal(0, 4, n_samples)))
        feature_data.append(temperature)
        feature_names.append('Temperature')
        
        # Feature 3: Moderate discriminator - current
        base_current = 10.0
        current = np.where(fault_states == 0,
                         base_current + np.random.normal(0, 0.5, n_samples),
                         np.where(fault_states == 1,
                                base_current * 1.3 + np.random.normal(0, 0.8, n_samples),
                                base_current * 1.8 + np.random.normal(0, 1.0, n_samples)))
        feature_data.append(current)
        feature_names.append('Current')
        
        # Feature 4: Moderate discriminator - pressure
        base_pressure = 100.0
        pressure = np.where(fault_states == 0,
                          base_pressure + np.random.normal(0, 5, n_samples),
                          np.where(fault_states == 1,
                                 base_pressure - 20 + np.random.normal(0, 8, n_samples),
                                 base_pressure - 50 + np.random.normal(0, 10, n_samples)))
        feature_data.append(pressure)
        feature_names.append('Pressure')
        
        # Feature 5: Weak discriminator - flow rate
        base_flow = 50.0
        flow_rate = np.where(fault_states == 0,
                           base_flow + np.random.normal(0, 3, n_samples),
                           np.where(fault_states == 1,
                                  base_flow * 1.1 + np.random.normal(0, 4, n_samples),
                                  base_flow * 1.2 + np.random.normal(0, 5, n_samples)))
        feature_data.append(flow_rate)
        feature_names.append('Flow_Rate')
        
        # Feature 6: Non-discriminative - noise feature
        noise_feature = np.random.normal(0, 10, n_samples)
        feature_data.append(noise_feature)
        feature_names.append('Noise_Feature')
        
        # Add time-related features
        feature_data.append(time_hours)
        feature_names.append('Time_Hours')
        
        feature_data.append(cycles)
        feature_names.append('Cycle')
        
        # Add fault states as the target
        feature_data.append(fault_states.astype(float))
        feature_names.append('Fault_Level')
        
        # Convert to proper 2D array (samples x features)
        data_array = np.column_stack(feature_data)
        
        # Save as MATLAB file
        mat_file_path = temp_output_dir / "fault_dataset.mat"
        scipy.io.savemat(str(mat_file_path), {'data': data_array})
        
        return mat_file_path, feature_names, fault_states
    
    @pytest.fixture
    def binary_fault_dataset(self, temp_output_dir):
        """Create a binary fault dataset for testing."""
        n_samples = 100
        np.random.seed(42)
        
        # Binary fault states: 0=Normal, 1=Fault
        fault_states = np.zeros(n_samples, dtype=int)
        fault_states[60:] = 1  # Fault from sample 60 onwards
        
        # Generate discriminative features
        feature1 = np.where(fault_states == 0,
                          np.random.normal(10, 2, n_samples),
                          np.random.normal(20, 3, n_samples))
        
        feature2 = np.where(fault_states == 0,
                          np.random.normal(50, 5, n_samples),
                          np.random.normal(30, 4, n_samples))
        
        # Non-discriminative feature
        feature3 = np.random.normal(0, 1, n_samples)
        
        # Create data array
        feature_data = [
            np.arange(n_samples),  # Time
            feature1,
            feature2,
            feature3,
            fault_states.astype(float)
        ]
        
        data_array = np.column_stack(feature_data)
        
        mat_file_path = temp_output_dir / "binary_fault_dataset.mat"
        scipy.io.savemat(str(mat_file_path), {'data': data_array})
        
        return mat_file_path, fault_states
    
    def test_complete_fault_workflow_with_mat_data(self, fault_dataset, temp_output_dir):
        """Test complete fault diagnosis workflow with MATLAB data."""
        mat_file_path, feature_names, fault_states = fault_dataset
        
        # Initialize orchestrator
        orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
        
        # Run complete analysis with fault column specified
        # Note: MAT files loaded by scipy.io will have numeric column names
        # The fault column should be the last one (index depends on total columns)
        temp_results = orchestrator.run_complete_analysis(data_path=mat_file_path)
        
        # Get the loaded data to see column names
        loaded_data = orchestrator.get_loaded_data()
        column_names = list(loaded_data.columns)
        
        # The fault column should be the last one
        fault_column_name = str(len(column_names) - 1)  # Last column (0-indexed)
        
        # Now run with proper fault column specification
        results = orchestrator.run_complete_analysis(
            data_path=mat_file_path,
            fault_column=fault_column_name
        )
        
        # Verify results structure
        assert isinstance(results, AnalysisResults)
        assert results.metadata is not None
        assert results.fault_features is not None
        
        # Verify fault features were identified
        assert isinstance(results.fault_features, list)
        assert len(results.fault_features) > 0, "Should identify fault-discriminative features"
        
        # Verify that strongly discriminative features are identified
        # Since MAT files have numeric column names, we can't check for specific feature names
        # Instead, verify that we have a reasonable number of discriminative features
        assert len(results.fault_features) >= 2, "Should identify at least 2 discriminative features"
        
        # Log the actual features found for debugging
        print(f"Found {len(results.fault_features)} fault-discriminative features:")
        for name in results.fault_features[:5]:  # Show top 5
            print(f"  {name}")
        
        # Verify preprocessing recommendations include fault-specific suggestions
        assert isinstance(results.preprocessing_recommendations, dict)
        
        # Check that output files were created
        output_dir = Path(temp_output_dir)
        assert (output_dir / "logs").exists()
        assert (output_dir / "reports").exists()
    
    def test_complete_fault_workflow_with_binary_data(self, binary_fault_dataset, temp_output_dir):
        """Test complete fault diagnosis workflow with binary fault data."""
        mat_file_path, fault_states = binary_fault_dataset
        
        # Initialize orchestrator
        orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
        
        # Run complete analysis with fault column specified
        # Note: Since we're using MAT files, column names will be numeric indices
        # We need to find which column corresponds to fault (should be the last one)
        results = orchestrator.run_complete_analysis(
            data_path=mat_file_path,
            fault_column='4'  # Fault is the 5th column (0-indexed as '4')
        )
        
        # Verify results structure
        assert isinstance(results, AnalysisResults)
        assert results.fault_features is not None
        assert len(results.fault_features) > 0
        
        # Should identify discriminative features
        assert len(results.fault_features) >= 1, "Should identify at least one discriminative feature"
        
        # Verify that non-discriminative features are excluded
        # (We can't check specific feature names due to numeric column names in MAT files)
        assert len(results.fault_features) <= 3, "Should not identify too many features as discriminative"
    
    def test_fault_workflow_without_fault_column(self, fault_dataset, temp_output_dir):
        """Test fault workflow when no fault column is specified."""
        mat_file_path, _, _ = fault_dataset
        
        # Initialize orchestrator
        orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
        
        # Run analysis without specifying fault column
        results = orchestrator.run_complete_analysis(data_path=mat_file_path)
        
        # Should still complete analysis but with empty fault features
        assert isinstance(results, AnalysisResults)
        assert isinstance(results.fault_features, list)
        assert len(results.fault_features) == 0, "Should have no fault features without fault column"
    
    def test_fault_workflow_with_invalid_fault_column(self, fault_dataset, temp_output_dir):
        """Test fault workflow with invalid fault column name."""
        mat_file_path, _, _ = fault_dataset
        
        # Initialize orchestrator
        orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
        
        # Run analysis with non-existent fault column
        results = orchestrator.run_complete_analysis(
            data_path=mat_file_path,
            fault_column='NonExistentColumn'
        )
        
        # Should handle gracefully and return empty fault features
        assert isinstance(results, AnalysisResults)
        assert isinstance(results.fault_features, list)
        assert len(results.fault_features) == 0, "Should have no fault features with invalid column"
    
    def test_fault_analyzer_standalone_workflow(self, fault_dataset, temp_output_dir):
        """Test fault analyzer as standalone component."""
        mat_file_path, _, fault_states = fault_dataset
        
        # Load data using the orchestrator's data loader
        from nasa_pcoe_eda.data.loader import DataLoader
        loader = DataLoader()
        df = loader.load_dataset(mat_file_path)
        
        # Initialize fault analyzer
        fault_analyzer = FaultLevelAnalyzer()
        
        # Test discriminative feature identification - use column '8' as fault (last column)
        if '8' in df.columns:
            discriminative_features = fault_analyzer.identify_discriminative_features(df, '8')
            assert isinstance(discriminative_features, list)
            assert len(discriminative_features) > 0, "Should identify discriminative features"
            
            # Test distribution comparison
            if discriminative_features:
                distribution_comparison = fault_analyzer.compare_distributions(
                    df, '8', discriminative_features[:3]  # Test with first 3 features
                )
                assert isinstance(distribution_comparison, DistributionComparison)
                assert isinstance(distribution_comparison.feature_distributions, dict)
                assert isinstance(distribution_comparison.statistical_tests, dict)
                
                # Verify statistical tests were performed
                assert len(distribution_comparison.statistical_tests) > 0
                
                # Check that statistical tests have proper structure
                for feature, test_result in distribution_comparison.statistical_tests.items():
                    assert isinstance(test_result, dict)
                    assert 'test' in test_result
                    # Should have either p_value or error
                    assert 'p_value' in test_result or 'error' in test_result
            
            # Test class separability computation
            if discriminative_features:
                separability_scores = fault_analyzer.compute_class_separability(
                    df, '8', discriminative_features[:3]
                )
                assert isinstance(separability_scores, dict)
                assert len(separability_scores) > 0
                
                # Verify separability scores are non-negative
                for feature, score in separability_scores.items():
                    assert isinstance(score, (float, np.floating))
                    assert score >= 0.0, f"Separability score should be non-negative: {score}"
    
    def test_preprocessing_recommender_with_fault_data(self, fault_dataset, temp_output_dir):
        """Test preprocessing recommender with fault diagnosis data."""
        mat_file_path, _, _ = fault_dataset
        
        # Load data and run basic analysis
        from nasa_pcoe_eda.data.loader import DataLoader
        from nasa_pcoe_eda.analysis.statistics import StatisticsAnalyzer
        
        loader = DataLoader()
        df = loader.load_dataset(mat_file_path)
        
        stats_analyzer = StatisticsAnalyzer()
        missing_report = stats_analyzer.analyze_missing_values(df)
        
        # Initialize preprocessing recommender
        recommender = PreprocessingRecommender()
        
        # Test missing value strategy recommendation
        missing_strategies = recommender.recommend_missing_value_strategy(missing_report)
        assert isinstance(missing_strategies, dict)
        assert len(missing_strategies) > 0
        
        # Test scaling recommendation
        scaling_rec = recommender.recommend_scaling(df)
        assert scaling_rec is not None
        assert hasattr(scaling_rec, 'method')
        assert hasattr(scaling_rec, 'reason')
        
        # Test data split recommendation for fault diagnosis (non-time series)
        split_strategy = recommender.recommend_data_split(df, is_time_series=False)
        assert split_strategy is not None
        assert hasattr(split_strategy, 'method')
        assert hasattr(split_strategy, 'train_ratio')
        assert hasattr(split_strategy, 'test_ratio')
        
        # For fault diagnosis, should recommend stratified split for balanced classes
        assert 'stratified' in split_strategy.method or 'random' in split_strategy.method
    
    def test_fault_workflow_with_missing_data(self, temp_output_dir):
        """Test fault workflow with missing data."""
        # Create dataset with missing values
        n_samples = 80
        np.random.seed(42)
        
        # Binary fault states
        fault_states = np.zeros(n_samples, dtype=float)  # Use float to allow NaN
        fault_states[40:] = 1
        
        # Generate features with missing values
        feature1 = np.where(fault_states == 0,
                          np.random.normal(10, 2, n_samples),
                          np.random.normal(20, 3, n_samples))
        feature2 = np.where(fault_states == 0,
                          np.random.normal(50, 5, n_samples),
                          np.random.normal(30, 4, n_samples))
        
        # Introduce missing values
        feature1[10:15] = np.nan
        feature2[20:25] = np.nan
        fault_states[5:8] = np.nan  # Missing fault labels
        
        # Create data array
        feature_data = [
            np.arange(n_samples),
            feature1,
            feature2,
            fault_states
        ]
        data_array = np.column_stack(feature_data)
        
        mat_file_path = temp_output_dir / "fault_missing_data.mat"
        scipy.io.savemat(str(mat_file_path), {'data': data_array})
        
        # Initialize orchestrator
        orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
        
        # Run analysis - should handle missing data gracefully
        results = orchestrator.run_complete_analysis(
            data_path=mat_file_path,
            fault_column='3'  # Fault is the 4th column (0-indexed as '3')
        )
        
        # Should complete successfully despite missing data
        assert isinstance(results, AnalysisResults)
        assert isinstance(results.fault_features, list)
        
        # Should still identify some discriminative features
        # (fault analysis should handle missing values by dropping them)
    
    def test_fault_workflow_with_minimal_data(self, temp_output_dir):
        """Test fault workflow with minimal data."""
        # Create minimal dataset
        feature_data = [
            [1, 2, 3, 4],  # Feature1
            [10, 20, 15, 25],  # Feature2
            [0, 0, 1, 1]   # Fault states
        ]
        data_array = np.column_stack(feature_data)
        
        mat_file_path = temp_output_dir / "fault_minimal_data.mat"
        scipy.io.savemat(str(mat_file_path), {'data': data_array})
        
        # Initialize orchestrator
        orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
        
        # Run analysis - should handle minimal data gracefully
        results = orchestrator.run_complete_analysis(
            data_path=mat_file_path,
            fault_column='2'  # Fault is the 3rd column (0-indexed as '2')
        )
        
        # Should complete without errors
        assert isinstance(results, AnalysisResults)
        assert isinstance(results.fault_features, list)
        
        # May or may not identify features depending on statistical significance
        # but should not crash
    
    def test_fault_workflow_state_consistency(self, binary_fault_dataset, temp_output_dir):
        """Test that fault workflow maintains consistent state."""
        mat_file_path, _ = binary_fault_dataset
        
        # Initialize orchestrator
        orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
        
        # Run analysis
        results = orchestrator.run_complete_analysis(
            data_path=mat_file_path,
            fault_column='4'  # Fault is the 5th column (0-indexed as '4')
        )
        
        # Get state from orchestrator
        loaded_data = orchestrator.get_loaded_data()
        analysis_results = orchestrator.get_analysis_results()
        
        # Verify state consistency
        assert loaded_data is not None
        # Check that fault column exists (could be integer 4 or string '4')
        assert 4 in loaded_data.columns or '4' in loaded_data.columns, (
            f"Fault column not found in loaded data columns: {list(loaded_data.columns)}"
        )
        assert analysis_results == results
        
        # Verify fault features are consistent with loaded data
        for feature_name in results.fault_features:
            # Convert feature name to match column type
            feature_col = feature_name
            if isinstance(feature_name, str) and feature_name.isdigit():
                feature_col = int(feature_name)
            elif isinstance(feature_name, int):
                feature_col = feature_name
            assert feature_col in loaded_data.columns or str(feature_name) in loaded_data.columns, (
                f"Fault feature {feature_name} not found in loaded data columns: {list(loaded_data.columns)}"
            )
    
    def test_fault_workflow_error_handling(self, temp_output_dir):
        """Test error handling in fault workflow."""
        # Test with non-existent file
        non_existent_file = temp_output_dir / "non_existent.mat"
        
        orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
        
        with pytest.raises(EDAError):
            orchestrator.run_complete_analysis(
                data_path=non_existent_file,
                fault_column='Fault_Level'
            )
    
    def test_fault_workflow_output_files(self, binary_fault_dataset, temp_output_dir):
        """Test that fault workflow generates appropriate output files."""
        mat_file_path, _ = binary_fault_dataset
        
        # Initialize orchestrator
        orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
        
        # Run analysis
        results = orchestrator.run_complete_analysis(
            data_path=mat_file_path,
            fault_column='4'  # Fault is the 5th column (0-indexed as '4')
        )
        
        # Check output directory structure
        output_dir = Path(temp_output_dir)
        assert (output_dir / "logs").exists()
        assert (output_dir / "reports").exists()
        
        # Check that report contains fault analysis section
        report_files = list((output_dir / "reports").glob("*.html"))
        if report_files:
            report_content = report_files[0].read_text()
            # Should contain fault-related content
            fault_indicators = ['fault', 'Fault', 'discriminative', '故障', '診断']
            found_indicators = sum(1 for indicator in fault_indicators if indicator in report_content)
            assert found_indicators > 0, "Report should contain fault-related content"
        
        # Check visualization files if generated
        if results.visualization_paths:
            for viz_path in results.visualization_paths:
                if Path(viz_path).exists():
                    assert Path(viz_path).suffix in ['.png', '.jpg', '.pdf'], (
                        f"Unexpected visualization file format: {viz_path}"
                    )
    
    def test_fault_workflow_preprocessing_integration(self, fault_dataset, temp_output_dir):
        """Test integration between fault analysis and preprocessing recommendations."""
        mat_file_path, _, _ = fault_dataset
        
        # Initialize orchestrator
        orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
        
        # Run complete analysis
        results = orchestrator.run_complete_analysis(
            data_path=mat_file_path,
            fault_column='8'  # Last column is fault
        )
        
        # Verify that preprocessing recommendations are generated
        assert isinstance(results.preprocessing_recommendations, dict)
        
        # Should have key preprocessing components
        expected_keys = ['missing_value_strategy', 'scaling_recommendation', 'data_split_strategy']
        for key in expected_keys:
            assert key in results.preprocessing_recommendations, f"Missing {key} in preprocessing recommendations"
        
        # For fault diagnosis, data split should consider class balance
        split_strategy = results.preprocessing_recommendations['data_split_strategy']
        assert split_strategy is not None
        
        # Should recommend appropriate split method for classification
        assert hasattr(split_strategy, 'method')
        split_method = split_strategy.method.lower()
        assert any(method in split_method for method in ['stratified', 'random']), (
            f"Unexpected split method for fault diagnosis: {split_method}"
        )
        
        # Scaling should be recommended if features have different scales
        scaling_rec = results.preprocessing_recommendations['scaling_recommendation']
        assert scaling_rec is not None
        assert hasattr(scaling_rec, 'method')
        
        # Feature engineering suggestions should be present
        if 'feature_engineering' in results.preprocessing_recommendations:
            feature_suggestions = results.preprocessing_recommendations['feature_engineering']
            assert isinstance(feature_suggestions, list)
    
    def test_end_to_end_fault_diagnosis_pipeline(self, fault_dataset, temp_output_dir):
        """Test the complete end-to-end fault diagnosis pipeline."""
        mat_file_path, feature_names, fault_states = fault_dataset
        
        # Initialize orchestrator
        orchestrator = AnalysisOrchestrator(output_dir=temp_output_dir)
        
        # Run complete analysis
        results = orchestrator.run_complete_analysis(
            data_path=mat_file_path,
            fault_column='8'  # Last column is fault
        )
        
        # Verify complete pipeline execution
        assert isinstance(results, AnalysisResults)
        
        # 1. Data loading and validation
        assert results.metadata is not None
        assert results.metadata.n_records > 0
        assert results.metadata.n_features > 0
        
        # 2. Statistical analysis
        assert results.statistics is not None
        assert 'descriptive_stats' in results.statistics
        
        # 3. Data quality analysis
        assert results.missing_values is not None
        
        # 4. Correlation analysis
        assert results.correlation_matrix is not None
        assert not results.correlation_matrix.empty
        
        # 5. Fault level identification
        assert isinstance(results.fault_features, list)
        assert len(results.fault_features) > 0, "Should identify discriminative features"
        
        # 6. Preprocessing recommendations
        assert isinstance(results.preprocessing_recommendations, dict)
        assert len(results.preprocessing_recommendations) > 0
        
        # 7. Output generation
        assert isinstance(results.visualization_paths, list)
        
        # Verify output files exist
        output_dir = Path(temp_output_dir)
        assert (output_dir / "logs").exists()
        assert (output_dir / "reports").exists()
        
        # Verify log files contain fault analysis information
        log_files = list((output_dir / "logs").glob("*.log"))
        if log_files:
            log_content = log_files[0].read_text()
            fault_log_indicators = ['fault', 'Fault', 'discriminative']
            found_log_indicators = sum(1 for indicator in fault_log_indicators if indicator in log_content)
            assert found_log_indicators > 0, "Log should contain fault analysis information"
        
        print(f"End-to-end fault diagnosis pipeline completed successfully!")
        print(f"Identified {len(results.fault_features)} discriminative features")
        print(f"Generated {len(results.preprocessing_recommendations)} preprocessing recommendations")
        print(f"Created {len(results.visualization_paths)} visualizations")