"""
Integration tests for TimeSeriesPreprocessor
"""

import pytest
import numpy as np
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from true_rul.time_series_preprocessor import TimeSeriesPreprocessor
from true_rul.data_structures import CycleData


class TestTimeSeriesPreprocessorIntegration:
    """Integration tests for TimeSeriesPreprocessor"""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a preprocessor"""
        return TimeSeriesPreprocessor(rolling_window=5, normalization="standard")
    
    @pytest.fixture
    def sample_cycles(self):
        """Create sample cycle data with realistic patterns"""
        cycles = []
        np.random.seed(42)
        
        for i in range(20):
            # Create voltage data with degradation pattern
            t = np.linspace(0, 1, 100)
            vl = 5.0 + 0.5 * np.sin(2 * np.pi * t) + np.random.randn(100) * 0.1
            vo = 4.0 + 0.4 * np.sin(2 * np.pi * t) + np.random.randn(100) * 0.1
            
            # Add degradation effect
            degradation_factor = 1 - (i / 200)  # Gradual degradation
            vo = vo * degradation_factor
            
            cycles.append(CycleData(
                cycle_number=i + 1,
                vl_series=vl,
                vo_series=vo,
                timestamp=datetime.now()
            ))
        
        return cycles
    
    def test_end_to_end_preprocessing_pipeline(self, preprocessor, sample_cycles):
        """Test complete pipeline: create temporal features -> normalize"""
        
        # Create synthetic features (simulating feature extraction)
        np.random.seed(42)
        n_cycles = 20
        n_features = 10
        
        # Simulate features with degradation trend
        features_array = np.zeros((n_cycles, n_features))
        for i in range(n_cycles):
            # Base features with slight degradation trend
            features_array[i] = np.random.randn(n_features) + i * 0.05
        
        # Step 1: Create temporal features
        temporal_features = preprocessor.create_temporal_features(
            sample_cycles, features_array
        )
        
        # Verify temporal features
        expected_temporal_shape = (20, n_features * 7)
        assert temporal_features.shape == expected_temporal_shape
        
        # Step 2: Normalize features
        normalized = preprocessor.normalize_features(
            temporal_features, "ES12C1", fit=True
        )
        
        # Verify normalization
        assert normalized.shape == temporal_features.shape
        assert np.allclose(np.mean(normalized, axis=0), 0, atol=0.1)
        assert np.allclose(np.std(normalized, axis=0), 1, atol=0.1)
    
    def test_temporal_features_capture_degradation_trend(self, preprocessor, sample_cycles):
        """Test that temporal features capture degradation trends"""
        
        # Create features with clear degradation trend
        np.random.seed(42)
        n_cycles = 20
        n_features = 5
        
        features_array = np.zeros((n_cycles, n_features))
        for i in range(n_cycles):
            # Linear degradation in all features
            features_array[i] = np.ones(n_features) * (1 - i * 0.05)
        
        # Create temporal features
        temporal_features = preprocessor.create_temporal_features(
            sample_cycles, features_array
        )
        
        # Extract trend components from last cycle
        last_cycle_features = temporal_features[-1]
        
        recent_trend = last_cycle_features[5*n_features:6*n_features]
        long_trend = last_cycle_features[6*n_features:7*n_features]
        
        # Recent trend should be negative (degradation)
        assert np.all(recent_trend < 0)
        
        # Long trend should be more negative (accumulated degradation)
        assert np.all(long_trend < recent_trend)
    
    def test_multiple_capacitors_separate_normalization(self, preprocessor):
        """Test that different capacitors get separate normalization"""
        
        # Create cycles for two capacitors
        np.random.seed(42)
        
        cycles_c1 = []
        cycles_c2 = []
        
        for i in range(10):
            vl1 = 5.0 + np.random.randn(100) * 0.1
            vo1 = 4.5 + np.random.randn(100) * 0.1
            cycles_c1.append(CycleData(i+1, vl1, vo1))
            
            vl2 = 3.0 + np.random.randn(100) * 0.1
            vo2 = 2.5 + np.random.randn(100) * 0.1
            cycles_c2.append(CycleData(i+1, vl2, vo2))
        
        # Create synthetic features with different scales
        features_c1 = np.random.randn(10, 5) * 10 + 50  # High scale
        features_c2 = np.random.randn(10, 5) * 2 + 10   # Low scale
        
        # Create temporal features
        temporal_c1 = preprocessor.create_temporal_features(cycles_c1, features_c1)
        temporal_c2 = preprocessor.create_temporal_features(cycles_c2, features_c2)
        
        # Normalize separately
        norm_c1 = preprocessor.normalize_features(temporal_c1, "ES12C1", fit=True)
        norm_c2 = preprocessor.normalize_features(temporal_c2, "ES12C2", fit=True)
        
        # Both should be normalized
        assert np.allclose(np.mean(norm_c1, axis=0), 0, atol=0.1)
        assert np.allclose(np.mean(norm_c2, axis=0), 0, atol=0.1)
        
        # Should have separate scalers
        assert preprocessor.has_scaler("ES12C1")
        assert preprocessor.has_scaler("ES12C2")
        assert preprocessor.n_scalers == 2
    
    def test_feature_names_generation(self, preprocessor):
        """Test that feature names are generated correctly"""
        
        base_names = ["feat1", "feat2", "feat3"]
        temporal_names = preprocessor.get_temporal_feature_names(base_names)
        
        # Should have 3 * 7 = 21 features
        assert len(temporal_names) == 21
        
        # Verify structure
        assert "feat1" in temporal_names
        assert "feat1_rolling_mean" in temporal_names
        assert "feat1_recent_trend" in temporal_names
        assert "feat1_long_trend" in temporal_names
    
    def test_rolling_window_effect(self, preprocessor, sample_cycles):
        """Test that rolling window size affects temporal features"""
        
        # Create features
        np.random.seed(42)
        features = np.random.randn(20, 5)
        
        # Create temporal features with default window (5)
        temporal_5 = preprocessor.create_temporal_features(sample_cycles, features)
        
        # Create preprocessor with different window
        preprocessor_3 = TimeSeriesPreprocessor(rolling_window=3)
        temporal_3 = preprocessor_3.create_temporal_features(sample_cycles, features)
        
        # Both should have same shape
        assert temporal_5.shape == temporal_3.shape
        
        # But rolling statistics should be different
        # (because they're computed over different window sizes)
        assert not np.allclose(temporal_5, temporal_3)
    
    def test_normalization_preserves_relationships(self, preprocessor):
        """Test that normalization preserves relative relationships"""
        
        # Create features where feature 1 > feature 2 for all samples
        features = np.array([
            [10, 5],
            [20, 10],
            [30, 15],
            [40, 20]
        ], dtype=float)
        
        cycles = [CycleData(i+1, np.random.randn(10), np.random.randn(10)) for i in range(4)]
        
        # Create temporal features
        temporal = preprocessor.create_temporal_features(cycles, features)
        
        # Normalize
        normalized = preprocessor.normalize_features(temporal, "C1", fit=True)
        
        # Original features are in first n_features positions
        # Check that relative ordering is preserved in original features
        orig_feat1 = temporal[:, 0]
        orig_feat2 = temporal[:, 1]
        norm_feat1 = normalized[:, 0]
        norm_feat2 = normalized[:, 1]
        
        # If original feat1 > feat2, normalized should maintain this (with tolerance)
        for i in range(len(orig_feat1)):
            if orig_feat1[i] > orig_feat2[i] + 1e-10:  # Add small tolerance
                assert norm_feat1[i] > norm_feat2[i] - 1e-10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
