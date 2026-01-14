"""Unit tests for VisualizationEngine."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from nasa_pcoe_eda.visualization import VisualizationEngine
from nasa_pcoe_eda.exceptions import VisualizationError


class TestVisualizationEngine:
    """Test cases for VisualizationEngine."""
    
    @pytest.fixture
    def engine(self):
        """Create a VisualizationEngine instance."""
        return VisualizationEngine()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        return pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(5, 2, 100),
            'feature3': np.random.exponential(2, 100),
            'categorical': ['A', 'B', 'C'] * 33 + ['A'],
            'time_index': pd.date_range('2023-01-01', periods=100, freq='D')
        })
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create a temporary output directory."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_initialization(self, engine):
        """Test VisualizationEngine initialization."""
        assert isinstance(engine, VisualizationEngine)
    
    def test_plot_distributions(self, engine, sample_data, temp_output_dir):
        """Test distribution plotting."""
        features = ['feature1', 'feature2', 'feature3']
        plot_paths = engine.plot_distributions(sample_data, features, temp_output_dir)
        
        assert len(plot_paths) == 1
        assert plot_paths[0].exists()
        assert plot_paths[0].name == 'distributions.png'
    
    def test_plot_distributions_no_numeric_features(self, engine, temp_output_dir):
        """Test distribution plotting with no numeric features."""
        df = pd.DataFrame({'categorical': ['A', 'B', 'C']})
        
        with pytest.raises(VisualizationError, match="No numeric features found"):
            engine.plot_distributions(df, ['categorical'], temp_output_dir)
    
    def test_plot_time_series(self, engine, sample_data, temp_output_dir):
        """Test time series plotting."""
        features = ['feature1', 'feature2']
        plot_paths = engine.plot_time_series(sample_data, features, temp_output_dir)
        
        assert len(plot_paths) == 2
        for path in plot_paths:
            assert path.exists()
            assert 'timeseries_' in path.name
    
    def test_plot_time_series_with_time_column(self, engine, sample_data, temp_output_dir):
        """Test time series plotting with explicit time column."""
        features = ['feature1']
        plot_paths = engine.plot_time_series(
            sample_data, features, temp_output_dir, time_column='time_index'
        )
        
        assert len(plot_paths) == 1
        assert plot_paths[0].exists()
    
    def test_plot_correlation_heatmap(self, engine, sample_data, temp_output_dir):
        """Test correlation heatmap plotting."""
        # Create correlation matrix
        numeric_data = sample_data[['feature1', 'feature2', 'feature3']]
        corr_matrix = numeric_data.corr()
        
        plot_path = engine.plot_correlation_heatmap(corr_matrix, temp_output_dir)
        
        assert plot_path.exists()
        assert plot_path.name == 'correlation_heatmap.png'
    
    def test_plot_scatter_matrix(self, engine, sample_data, temp_output_dir):
        """Test scatter matrix plotting."""
        features = ['feature1', 'feature2', 'feature3']
        plot_path = engine.plot_scatter_matrix(sample_data, features, temp_output_dir)
        
        assert plot_path.exists()
        assert plot_path.name == 'scatter_matrix.png'
    
    def test_plot_scatter_matrix_no_numeric_features(self, engine, temp_output_dir):
        """Test scatter matrix with no numeric features."""
        df = pd.DataFrame({'categorical': ['A', 'B', 'C']})
        
        with pytest.raises(VisualizationError, match="No numeric features found"):
            engine.plot_scatter_matrix(df, ['categorical'], temp_output_dir)
    
    def test_plot_scatter_matrix_limits_features(self, engine, temp_output_dir):
        """Test that scatter matrix limits features to 6."""
        # Create data with many features
        data = {}
        for i in range(10):
            data[f'feature_{i}'] = np.random.normal(0, 1, 50)
        df = pd.DataFrame(data)
        
        features = list(data.keys())
        
        with pytest.warns(UserWarning, match="Limited scatter matrix to first 6 features"):
            plot_path = engine.plot_scatter_matrix(df, features, temp_output_dir)
            assert plot_path.exists()
    
    def test_output_directory_creation(self, engine, sample_data):
        """Test that output directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'new_subdir'
            assert not output_dir.exists()
            
            features = ['feature1']
            plot_paths = engine.plot_distributions(sample_data, features, output_dir)
            
            assert output_dir.exists()
            assert len(plot_paths) == 1
            assert plot_paths[0].exists()
    
    def test_capacitor_degradation_analysis_no_file(self, engine, temp_output_dir):
        """Test capacitor degradation analysis with non-existent file."""
        non_existent_file = temp_output_dir / 'non_existent.mat'
        
        with pytest.raises(VisualizationError):
            engine.plot_capacitor_degradation_analysis(non_existent_file, temp_output_dir)
    
    def test_capacitor_degradation_analysis_with_mock_data(self, engine, temp_output_dir):
        """Test capacitor degradation analysis with mock HDF5 data."""
        # Create a mock HDF5 file structure similar to ES12.mat
        mock_file = temp_output_dir / 'mock_es12.mat'
        
        try:
            import h5py
            
            with h5py.File(mock_file, 'w') as f:
                # Create EIS_Data group
                eis_group = f.create_group('EIS_Data')
                
                # Create mock capacitor data (ES12C1, ES12C2)
                for cap_num in range(1, 3):  # Just test with 2 capacitors
                    cap_name = f'ES12C{cap_num}'
                    cap_group = eis_group.create_group(cap_name)
                    cycle_group = cap_group.create_group('cycle')
                    
                    # Create mock cycle data
                    for cycle in range(1, 6):  # 5 cycles
                        cycle_str = str(cycle)
                        cycle_data = cycle_group.create_group(cycle_str)
                        
                        # Mock capacity and ESR data
                        capacity_val = 1000.0 - (cycle - 1) * 50.0  # Decreasing capacity
                        esr_val = 0.1 + (cycle - 1) * 0.02  # Increasing ESR
                        
                        cycle_data.create_dataset('capacity', data=[[capacity_val]])
                        cycle_data.create_dataset('esr', data=[[esr_val]])
            
            # Test the analysis
            plot_paths = engine.plot_capacitor_degradation_analysis(mock_file, temp_output_dir)
            
            # Should generate multiple plots
            assert len(plot_paths) >= 1
            for path in plot_paths:
                assert path.exists()
                
        except ImportError:
            pytest.skip("h5py not available for testing")