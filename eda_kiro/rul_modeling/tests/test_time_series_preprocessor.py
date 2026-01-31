"""
Unit tests for TimeSeriesPreprocessor
"""

import pytest
import numpy as np
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from true_rul.time_series_preprocessor import TimeSeriesPreprocessor
from true_rul.data_structures import CycleData


class TestTimeSeriesPreprocessor:
    """Test suite for TimeSeriesPreprocessor"""
    
    @pytest.fixture
    def preprocessor(self):
        """Create a preprocessor instance"""
        return TimeSeriesPreprocessor(rolling_window=5, normalization="standard")
    
    @pytest.fixture
    def sample_cycles(self):
        """Create sample cycle data"""
        cycles = []
        for i in range(10):
            vl = np.random.randn(100) + i * 0.1  # Slight trend
            vo = np.random.randn(100) + i * 0.1
            cycles.append(CycleData(
                cycle_number=i + 1,
                vl_series=vl,
                vo_series=vo,
                timestamp=datetime.now()
            ))
        return cycles
    
    @pytest.fixture
    def sample_features(self):
        """Create sample feature array"""
        # 10 cycles, 5 features each
        return np.random.randn(10, 5)
    
    def test_initialization(self):
        """Test preprocessor initialization"""
        preprocessor = TimeSeriesPreprocessor(rolling_window=3, normalization="minmax")
        assert preprocessor.rolling_window == 3
        assert preprocessor.normalization == "minmax"
        assert len(preprocessor.scalers) == 0
    
    def test_initialization_defaults(self):
        """Test preprocessor initialization with defaults"""
        preprocessor = TimeSeriesPreprocessor()
        assert preprocessor.rolling_window == 5
        assert preprocessor.normalization == "standard"
    
    def test_create_temporal_features_shape(self, preprocessor, sample_cycles, sample_features):
        """Test that temporal features have correct shape"""
        temporal_features = preprocessor.create_temporal_features(
            sample_cycles, sample_features
        )
        
        # Expected shape: (n_cycles, n_features * 7)
        # 7 = 1 (original) + 4 (rolling stats) + 2 (trends)
        expected_shape = (10, 5 * 7)
        assert temporal_features.shape == expected_shape
    
    def test_create_temporal_features_first_cycle(self, preprocessor, sample_cycles, sample_features):
        """Test temporal features for first cycle (no history)"""
        temporal_features = preprocessor.create_temporal_features(
            sample_cycles, sample_features
        )
        
        # First cycle should have:
        # - Original features
        # - Rolling stats (only current cycle)
        # - Zero recent trend (no previous)
        # - Zero long trend (no 5 cycles ago)
        
        first_cycle_features = temporal_features[0]
        n_features = 5
        
        # Extract components
        original = first_cycle_features[:n_features]
        rolling_mean = first_cycle_features[n_features:2*n_features]
        recent_trend = first_cycle_features[5*n_features:6*n_features]
        long_trend = first_cycle_features[6*n_features:7*n_features]
        
        # Original should match input
        np.testing.assert_array_almost_equal(original, sample_features[0])
        
        # Rolling mean should equal original (only one sample)
        np.testing.assert_array_almost_equal(rolling_mean, sample_features[0])
        
        # Trends should be zero
        np.testing.assert_array_almost_equal(recent_trend, np.zeros(n_features))
        np.testing.assert_array_almost_equal(long_trend, np.zeros(n_features))
    
    def test_create_temporal_features_recent_trend(self, preprocessor, sample_cycles, sample_features):
        """Test recent trend computation"""
        temporal_features = preprocessor.create_temporal_features(
            sample_cycles, sample_features
        )
        
        # Second cycle should have recent trend = current - previous
        second_cycle_features = temporal_features[1]
        n_features = 5
        recent_trend = second_cycle_features[5*n_features:6*n_features]
        
        expected_trend = sample_features[1] - sample_features[0]
        np.testing.assert_array_almost_equal(recent_trend, expected_trend)
    
    def test_create_temporal_features_long_trend(self, preprocessor, sample_cycles, sample_features):
        """Test long-term trend computation"""
        temporal_features = preprocessor.create_temporal_features(
            sample_cycles, sample_features
        )
        
        # 6th cycle (index 5) should have long trend = current - 5 cycles ago
        sixth_cycle_features = temporal_features[5]
        n_features = 5
        long_trend = sixth_cycle_features[6*n_features:7*n_features]
        
        expected_trend = sample_features[5] - sample_features[0]
        np.testing.assert_array_almost_equal(long_trend, expected_trend)
    
    def test_create_temporal_features_rolling_stats(self, preprocessor, sample_cycles, sample_features):
        """Test rolling statistics computation"""
        temporal_features = preprocessor.create_temporal_features(
            sample_cycles, sample_features
        )
        
        # Test 7th cycle (index 6) with full window of 5
        seventh_cycle_features = temporal_features[6]
        n_features = 5
        
        # Extract rolling stats
        rolling_mean = seventh_cycle_features[n_features:2*n_features]
        rolling_std = seventh_cycle_features[2*n_features:3*n_features]
        rolling_min = seventh_cycle_features[3*n_features:4*n_features]
        rolling_max = seventh_cycle_features[4*n_features:5*n_features]
        
        # Compute expected values (window: cycles 2-6, indices 1-5)
        window = sample_features[2:7]  # 5 cycles
        expected_mean = np.mean(window, axis=0)
        expected_std = np.std(window, axis=0)
        expected_min = np.min(window, axis=0)
        expected_max = np.max(window, axis=0)
        
        np.testing.assert_array_almost_equal(rolling_mean, expected_mean)
        np.testing.assert_array_almost_equal(rolling_std, expected_std)
        np.testing.assert_array_almost_equal(rolling_min, expected_min)
        np.testing.assert_array_almost_equal(rolling_max, expected_max)
    
    def test_create_temporal_features_mismatched_lengths(self, preprocessor, sample_cycles):
        """Test error handling for mismatched lengths"""
        features = np.random.randn(5, 3)  # Only 5 cycles
        
        with pytest.raises(ValueError, match="must match features length"):
            preprocessor.create_temporal_features(sample_cycles, features)
    
    def test_create_temporal_features_empty(self, preprocessor):
        """Test error handling for empty input"""
        with pytest.raises(ValueError, match="empty cycle list"):
            preprocessor.create_temporal_features([], np.array([]))
    
    def test_normalize_features_standard(self, preprocessor):
        """Test standard normalization"""
        features = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        
        # Fit and transform
        normalized = preprocessor.normalize_features(features, "C1", fit=True)
        
        # Check that mean is close to 0 and std is close to 1
        assert np.allclose(np.mean(normalized, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(normalized, axis=0), 1, atol=1e-10)
    
    def test_normalize_features_minmax(self):
        """Test minmax normalization"""
        preprocessor = TimeSeriesPreprocessor(normalization="minmax")
        features = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        
        # Fit and transform
        normalized = preprocessor.normalize_features(features, "C1", fit=True)
        
        # Check that min is 0 and max is 1
        assert np.allclose(np.min(normalized, axis=0), 0, atol=1e-10)
        assert np.allclose(np.max(normalized, axis=0), 1, atol=1e-10)
    
    def test_normalize_features_without_fit(self, preprocessor):
        """Test error when normalizing without fitting"""
        features = np.array([[1, 2, 3], [4, 5, 6]])
        
        with pytest.raises(ValueError, match="No scaler available"):
            preprocessor.normalize_features(features, "C1", fit=False)
    
    def test_normalize_features_with_global_fallback(self, preprocessor):
        """Test fallback to global scaler"""
        features = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        
        # Fit global scaler
        preprocessor.fit_global_scaler(features)
        
        # Normalize with unknown capacitor (should use global)
        normalized = preprocessor.normalize_features(features, "unknown", fit=False)
        
        # Should succeed without error
        assert normalized.shape == features.shape
    
    def test_normalize_features_empty(self, preprocessor):
        """Test error handling for empty features"""
        with pytest.raises(ValueError, match="empty feature array"):
            preprocessor.normalize_features(np.array([]).reshape(0, 3), "C1", fit=True)
    
    def test_normalize_features_invalid_method(self):
        """Test error for invalid normalization method"""
        preprocessor = TimeSeriesPreprocessor(normalization="invalid")
        features = np.array([[1, 2, 3]])
        
        with pytest.raises(ValueError, match="Unknown normalization method"):
            preprocessor.normalize_features(features, "C1", fit=True)
    
    def test_fit_global_scaler(self, preprocessor):
        """Test fitting global scaler"""
        features = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        
        preprocessor.fit_global_scaler(features)
        
        assert preprocessor.has_scaler("global")
        assert preprocessor.n_scalers == 1
    
    def test_fit_global_scaler_empty(self, preprocessor):
        """Test error when fitting global scaler on empty data"""
        with pytest.raises(ValueError, match="empty feature array"):
            preprocessor.fit_global_scaler(np.array([]).reshape(0, 3))
    
    def test_get_scaler(self, preprocessor):
        """Test getting scaler"""
        features = np.array([[1, 2, 3], [4, 5, 6]])
        preprocessor.normalize_features(features, "C1", fit=True)
        
        scaler = preprocessor.get_scaler("C1")
        assert scaler is not None
        
        # Non-existent scaler
        assert preprocessor.get_scaler("C999") is None
    
    def test_has_scaler(self, preprocessor):
        """Test checking if scaler exists"""
        features = np.array([[1, 2, 3], [4, 5, 6]])
        preprocessor.normalize_features(features, "C1", fit=True)
        
        assert preprocessor.has_scaler("C1")
        assert not preprocessor.has_scaler("C999")
    
    def test_reset_scalers(self, preprocessor):
        """Test resetting scalers"""
        features = np.array([[1, 2, 3], [4, 5, 6]])
        preprocessor.normalize_features(features, "C1", fit=True)
        preprocessor.normalize_features(features, "C2", fit=True)
        
        assert preprocessor.n_scalers == 2
        
        preprocessor.reset_scalers()
        
        assert preprocessor.n_scalers == 0
        assert not preprocessor.has_scaler("C1")
        assert not preprocessor.has_scaler("C2")
    
    def test_get_temporal_feature_names(self, preprocessor):
        """Test getting temporal feature names"""
        base_names = ["feat1", "feat2", "feat3"]
        
        temporal_names = preprocessor.get_temporal_feature_names(base_names)
        
        # Should have 3 * 7 = 21 features
        assert len(temporal_names) == 21
        
        # Check structure
        assert "feat1" in temporal_names
        assert "feat1_rolling_mean" in temporal_names
        assert "feat1_rolling_std" in temporal_names
        assert "feat1_rolling_min" in temporal_names
        assert "feat1_rolling_max" in temporal_names
        assert "feat1_recent_trend" in temporal_names
        assert "feat1_long_trend" in temporal_names
    
    def test_n_scalers_property(self, preprocessor):
        """Test n_scalers property"""
        assert preprocessor.n_scalers == 0
        
        features = np.array([[1, 2, 3], [4, 5, 6]])
        preprocessor.normalize_features(features, "C1", fit=True)
        
        assert preprocessor.n_scalers == 1
    
    def test_repr(self, preprocessor):
        """Test string representation"""
        repr_str = repr(preprocessor)
        
        assert "TimeSeriesPreprocessor" in repr_str
        assert "rolling_window=5" in repr_str
        assert "normalization='standard'" in repr_str
        assert "n_scalers=0" in repr_str
    
    def test_multiple_capacitors(self, preprocessor):
        """Test handling multiple capacitors"""
        features1 = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        features2 = np.array([[10, 20], [30, 40], [50, 60]], dtype=float)
        
        # Fit separate scalers
        norm1 = preprocessor.normalize_features(features1, "C1", fit=True)
        norm2 = preprocessor.normalize_features(features2, "C2", fit=True)
        
        # Both should be normalized independently
        assert np.allclose(np.mean(norm1, axis=0), 0, atol=1e-10)
        assert np.allclose(np.mean(norm2, axis=0), 0, atol=1e-10)
        
        # Should have 2 scalers
        assert preprocessor.n_scalers == 2
    
    def test_temporal_features_with_single_feature(self, preprocessor, sample_cycles):
        """Test temporal features with single feature dimension"""
        features = np.random.randn(10, 1)  # Single feature
        
        temporal_features = preprocessor.create_temporal_features(
            sample_cycles, features
        )
        
        # Should have shape (10, 7)
        assert temporal_features.shape == (10, 7)
    
    def test_temporal_features_consistency(self, preprocessor, sample_cycles, sample_features):
        """Test that temporal features are consistent across multiple calls"""
        temporal1 = preprocessor.create_temporal_features(sample_cycles, sample_features)
        temporal2 = preprocessor.create_temporal_features(sample_cycles, sample_features)
        
        np.testing.assert_array_equal(temporal1, temporal2)
    
    def test_normalization_inverse_transform(self, preprocessor):
        """Test that normalization can be inverted"""
        features = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        
        # Fit and transform
        normalized = preprocessor.normalize_features(features, "C1", fit=True)
        
        # Inverse transform
        scaler = preprocessor.get_scaler("C1")
        recovered = scaler.inverse_transform(normalized)
        
        # Should recover original features
        np.testing.assert_array_almost_equal(recovered, features)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
