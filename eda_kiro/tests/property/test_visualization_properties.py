"""Property-based tests for VisualizationEngine."""

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from hypothesis import given, strategies as st, settings

from nasa_pcoe_eda.visualization.engine import VisualizationEngine


class TestVisualizationEngineProperties:
    """Property-based tests for VisualizationEngine."""

    def setup_method(self):
        """Set up test fixtures."""
        self.engine = VisualizationEngine()

    # Feature: nasa-pcoe-eda, Property 10: 可視化ファイル生成
    @given(st.data())
    @settings(max_examples=20, deadline=5000)  # Reduced examples and increased deadline for visualization tests
    def test_visualization_file_generation(self, data):
        """
        任意のデータセットと出力ディレクトリに対して、可視化関数を実行すると、指定されたディレクトリに画像ファイルが生成される
        
        Property 10: Visualization File Generation
        For any dataset and output directory, executing visualization functions should generate image files in the specified directory.
        """
        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Generate a DataFrame with numeric columns
            num_columns = data.draw(st.integers(min_value=1, max_value=8))
            num_rows = data.draw(st.integers(min_value=5, max_value=50))
            
            # Generate DataFrame data manually
            df_data = {}
            feature_names = []
            for i in range(num_columns):
                col_name = f"feature_{i}"
                feature_names.append(col_name)
                # Generate numeric data for this column
                column_data = data.draw(st.lists(
                    st.floats(
                        min_value=-100, 
                        max_value=100, 
                        allow_nan=False, 
                        allow_infinity=False
                    ),
                    min_size=num_rows,
                    max_size=num_rows
                ))
                df_data[col_name] = column_data
            
            df = pd.DataFrame(df_data)
            
            # Skip if DataFrame is empty or has no numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                return
            
            # Get list of numeric features
            numeric_features = list(numeric_df.columns)
            
    # Feature: nasa-pcoe-eda, Property 10: 可視化ファイル生成
    @given(st.data())
    @settings(max_examples=20, deadline=5000)  # Reduced examples and increased deadline for visualization tests
    def test_visualization_file_generation(self, data):
        """
        任意のデータセットと出力ディレクトリに対して、可視化関数を実行すると、指定されたディレクトリに画像ファイルが生成される
        
        Property 10: Visualization File Generation
        For any dataset and output directory, executing visualization functions should generate image files in the specified directory.
        """
        # Create a temporary directory for output
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            
            # Generate a DataFrame with numeric columns
            num_columns = data.draw(st.integers(min_value=2, max_value=5))  # Reduced max columns
            num_rows = data.draw(st.integers(min_value=10, max_value=30))   # Reduced max rows
            
            # Generate DataFrame data manually
            df_data = {}
            feature_names = []
            for i in range(num_columns):
                col_name = f"feature_{i}"
                feature_names.append(col_name)
                # Generate numeric data for this column
                column_data = data.draw(st.lists(
                    st.floats(
                        min_value=-100, 
                        max_value=100, 
                        allow_nan=False, 
                        allow_infinity=False
                    ),
                    min_size=num_rows,
                    max_size=num_rows
                ))
                df_data[col_name] = column_data
            
            df = pd.DataFrame(df_data)
            
            # Skip if DataFrame is empty or has no numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                return
            
            # Get list of numeric features
            numeric_features = list(numeric_df.columns)
            
            # Test one visualization function per iteration to reduce test time
            # Randomly choose which function to test
            test_choice = data.draw(st.integers(min_value=1, max_value=4))
            
            if test_choice == 1:
                # Test plot_distributions
                try:
                    plot_paths = self.engine.plot_distributions(df, numeric_features, output_dir)
                    self._verify_generated_files(plot_paths, output_dir, "plot_distributions")
                except Exception as e:
                    self._handle_visualization_exception(e, "plot_distributions")
                    
            elif test_choice == 2:
                # Test plot_time_series
                try:
                    plot_paths = self.engine.plot_time_series(df, numeric_features, output_dir)
                    self._verify_generated_files(plot_paths, output_dir, "plot_time_series")
                except Exception as e:
                    self._handle_visualization_exception(e, "plot_time_series")
                    
            elif test_choice == 3:
                # Test plot_correlation_heatmap (if we have enough features)
                if len(numeric_features) >= 2:
                    try:
                        # Remove columns with zero variance for correlation calculation
                        filtered_df = numeric_df.copy()
                        for col in filtered_df.columns:
                            if filtered_df[col].nunique() <= 1:
                                filtered_df = filtered_df.drop(columns=[col])
                        
                        if len(filtered_df.columns) >= 2:
                            # Compute correlation matrix
                            corr_matrix = filtered_df.corr()
                            
                            if not corr_matrix.empty:
                                plot_path = self.engine.plot_correlation_heatmap(corr_matrix, output_dir)
                                self._verify_generated_files([plot_path], output_dir, "plot_correlation_heatmap")
                    except Exception as e:
                        self._handle_visualization_exception(e, "plot_correlation_heatmap")
                        
            elif test_choice == 4:
                # Test plot_scatter_matrix (if we have enough features)
                if len(numeric_features) >= 2:
                    try:
                        plot_path = self.engine.plot_scatter_matrix(df, numeric_features, output_dir)
                        self._verify_generated_files([plot_path], output_dir, "plot_scatter_matrix")
                    except Exception as e:
                        self._handle_visualization_exception(e, "plot_scatter_matrix")
            
            # Additional verification: Check that output directory was created if it didn't exist
            assert output_dir.exists(), (
                f"Output directory {output_dir} should exist after visualization functions are called. "
                f"Visualization functions must create the output directory if it doesn't exist. "
                f"This violates the file generation property."
            )
    
    def _verify_generated_files(self, plot_paths, output_dir, function_name):
        """Helper method to verify generated files meet the property requirements."""
        # Verify that files were generated
        assert len(plot_paths) > 0, (
            f"{function_name} should return at least one file path, but returned {len(plot_paths)} paths. "
            f"This violates the file generation property."
        )
        
        # Verify that all returned paths exist and are in the specified directory
        for plot_path in plot_paths:
            assert plot_path.exists(), (
                f"Generated plot file {plot_path} does not exist. "
                f"All visualization functions must generate actual files in the specified directory. "
                f"This violates the file generation property."
            )
            
            assert plot_path.parent == output_dir, (
                f"Generated plot file {plot_path} is not in the specified output directory {output_dir}. "
                f"All visualization files must be saved to the specified directory. "
                f"This violates the file generation property."
            )
            
            assert plot_path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.pdf', '.svg'], (
                f"Generated file {plot_path} does not have a valid image file extension. "
                f"Visualization functions must generate valid image files. "
                f"This violates the file generation property."
            )
            
            # Verify file is not empty
            assert plot_path.stat().st_size > 0, (
                f"Generated plot file {plot_path} is empty (0 bytes). "
                f"Visualization functions must generate non-empty image files. "
                f"This violates the file generation property."
            )
    
    def _handle_visualization_exception(self, exception, function_name):
        """Helper method to handle exceptions from visualization functions."""
        # If the function fails due to data issues, that's acceptable
        # but if it fails due to file generation issues, that's a violation
        if ("No numeric features" in str(exception) or 
            "No data" in str(exception) or 
            "empty" in str(exception).lower()):
            # This is acceptable - function correctly handles edge cases
            pass
        else:
            # Re-raise other exceptions as they indicate file generation problems
            raise AssertionError(
                f"{function_name} failed with unexpected error: {exception}. "
                f"This may indicate a file generation problem that violates the property."
            )