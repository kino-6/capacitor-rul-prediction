"""
Unit tests for Out-of-Distribution Detector
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from true_rul.ood_detector import OutOfDistributionDetector


class TestOutOfDistributionDetector:
    """Test suite for OutOfDistributionDetector"""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data"""
        np.random.seed(42)
        # Create 3 features with different distributions
        n_samples = 100
        
        feature1 = np.random.normal(0, 1, n_samples)  # Standard normal
        feature2 = np.random.normal(5, 2, n_samples)  # Mean=5, std=2
        feature3 = np.random.uniform(-1, 1, n_samples)  # Uniform [-1, 1]
        
        X = np.column_stack([feature1, feature2, feature3])
        feature_names = ["normal_feature", "shifted_normal", "uniform_feature"]
        
        return X, feature_names
    
    @pytest.fixture
    def fitted_detector(self, sample_training_data):
        """Create fitted OOD detector"""
        X, feature_names = sample_training_data
        detector = OutOfDistributionDetector(threshold_std=3.0)
        detector.fit(X, feature_names)
        return detector
    
    def test_initialization(self):
        """Test detector initialization"""
        detector = OutOfDistributionDetector(threshold_std=2.5, min_samples_for_stats=20)
        
        assert detector.threshold_std == 2.5
        assert detector.min_samples_for_stats == 20
        assert not detector.is_fitted
        assert detector.feature_stats == {}
        assert detector.feature_names == []
    
    def test_fit_basic(self, sample_training_data):
        """Test basic fitting functionality"""
        X, feature_names = sample_training_data
        detector = OutOfDistributionDetector()
        
        detector.fit(X, feature_names)
        
        assert detector.is_fitted
        assert len(detector.feature_stats) == 3
        assert detector.feature_names == feature_names
        
        # Check that statistics are computed
        for feature_name in feature_names:
            stats = detector.feature_stats[feature_name]
            assert "mean" in stats
            assert "std" in stats
            assert "min" in stats
            assert "max" in stats
            assert "q25" in stats
            assert "q75" in stats
            assert "n_valid" in stats
    
    def test_fit_without_feature_names(self, sample_training_data):
        """Test fitting without explicit feature names"""
        X, _ = sample_training_data
        detector = OutOfDistributionDetector()
        
        detector.fit(X)
        
        assert detector.is_fitted
        assert len(detector.feature_names) == 3
        assert detector.feature_names == ["feature_0", "feature_1", "feature_2"]
    
    def test_fit_insufficient_samples(self):
        """Test fitting with insufficient samples"""
        X = np.random.randn(5, 3)  # Only 5 samples
        detector = OutOfDistributionDetector(min_samples_for_stats=10)
        
        with pytest.raises(ValueError, match="Need at least 10 samples"):
            detector.fit(X)
    
    def test_is_out_of_distribution_basic(self, fitted_detector, sample_training_data):
        """Test basic OOD detection"""
        X_train, _ = sample_training_data
        
        # Test with training data (should not be OOD)
        ood_flags = fitted_detector.is_out_of_distribution(X_train[:10])
        assert isinstance(ood_flags, np.ndarray)
        assert ood_flags.shape == (10,)
        assert ood_flags.dtype == bool
        
        # Most training samples should not be OOD
        assert np.sum(ood_flags) <= 2  # Allow for some outliers
    
    def test_is_out_of_distribution_with_outliers(self, fitted_detector):
        """Test OOD detection with clear outliers"""
        # Create samples that are clearly out of distribution
        outliers = np.array([
            [10, 20, 5],    # All features way outside normal range
            [-10, -20, -5], # All features way outside normal range
            [0, 5, 0]       # Normal sample
        ])
        
        ood_flags = fitted_detector.is_out_of_distribution(outliers)
        
        assert ood_flags[0] == True   # First outlier
        assert ood_flags[1] == True   # Second outlier
        assert ood_flags[2] == False  # Normal sample
    
    def test_is_out_of_distribution_with_details(self, fitted_detector):
        """Test OOD detection with detailed output"""
        test_samples = np.array([
            [0, 5, 0],      # Normal sample
            [10, 20, 5]     # Outlier
        ])
        
        ood_flags, ood_details = fitted_detector.is_out_of_distribution(
            test_samples, return_details=True
        )
        
        assert len(ood_details) == 2
        
        # Check normal sample details
        normal_details = ood_details[0]
        assert normal_details["is_ood"] == False
        assert len(normal_details["ood_features"]) == 0
        assert "feature_deviations" in normal_details
        
        # Check outlier details
        outlier_details = ood_details[1]
        assert outlier_details["is_ood"] == True
        assert len(outlier_details["ood_features"]) > 0
        assert "feature_deviations" in outlier_details
    
    def test_get_ood_score(self, fitted_detector):
        """Test OOD score computation"""
        test_samples = np.array([
            [0, 5, 0],      # Normal sample
            [10, 20, 5],    # Strong outlier
            [2, 7, 0.5]     # Mild outlier
        ])
        
        ood_scores = fitted_detector.get_ood_score(test_samples)
        
        assert ood_scores.shape == (3,)
        assert 0 <= ood_scores[0] <= 1  # Normal sample should have low score
        assert ood_scores[1] > ood_scores[0]  # Strong outlier should have higher score
        assert ood_scores[2] >= ood_scores[0]  # Mild outlier should have >= score
    
    def test_unfitted_detector_errors(self):
        """Test that unfitted detector raises appropriate errors"""
        detector = OutOfDistributionDetector()
        test_data = np.random.randn(5, 3)
        
        with pytest.raises(RuntimeError, match="has not been fitted"):
            detector.is_out_of_distribution(test_data)
        
        with pytest.raises(RuntimeError, match="has not been fitted"):
            detector.get_ood_score(test_data)
    
    def test_dimension_mismatch_errors(self, fitted_detector):
        """Test errors for dimension mismatches"""
        # Wrong number of features
        wrong_features = np.random.randn(5, 2)  # Should be 3 features
        
        with pytest.raises(ValueError, match="must match fitted features"):
            fitted_detector.is_out_of_distribution(wrong_features)
        
        with pytest.raises(ValueError, match="must match fitted features"):
            fitted_detector.get_ood_score(wrong_features)
    
    def test_nan_handling(self, sample_training_data):
        """Test handling of NaN values"""
        X, feature_names = sample_training_data
        
        # Add some NaN values to training data
        X_with_nan = X.copy()
        X_with_nan[0, 0] = np.nan
        X_with_nan[1, 1] = np.nan
        
        detector = OutOfDistributionDetector()
        detector.fit(X_with_nan, feature_names)
        
        # Test with NaN values in test data
        test_data = np.array([[np.nan, 5, 0], [0, np.nan, 0]])
        ood_flags = detector.is_out_of_distribution(test_data)
        
        assert ood_flags.shape == (2,)
        # NaN values should be ignored, not cause errors
    
    def test_get_feature_statistics(self, fitted_detector):
        """Test feature statistics retrieval"""
        stats = fitted_detector.get_feature_statistics()
        
        assert "n_features" in stats
        assert "feature_names" in stats
        assert "feature_stats" in stats
        assert "global_stats" in stats
        assert "threshold_std" in stats
        assert "is_fitted" in stats
        
        assert stats["n_features"] == 3
        assert len(stats["feature_names"]) == 3
        assert stats["is_fitted"] == True
    
    def test_save_and_load(self, fitted_detector):
        """Test saving and loading detector"""
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            filepath = f.name
        
        try:
            # Save detector
            fitted_detector.save(filepath)
            assert Path(filepath).exists()
            
            # Load detector
            loaded_detector = OutOfDistributionDetector.load(filepath)
            
            # Check that loaded detector has same properties
            assert loaded_detector.is_fitted == fitted_detector.is_fitted
            assert loaded_detector.threshold_std == fitted_detector.threshold_std
            assert loaded_detector.feature_names == fitted_detector.feature_names
            assert len(loaded_detector.feature_stats) == len(fitted_detector.feature_stats)
            
            # Test that loaded detector works
            test_data = np.array([[0, 5, 0]])
            ood_flags_original = fitted_detector.is_out_of_distribution(test_data)
            ood_flags_loaded = loaded_detector.is_out_of_distribution(test_data)
            
            np.testing.assert_array_equal(ood_flags_original, ood_flags_loaded)
            
        finally:
            Path(filepath).unlink(missing_ok=True)
    
    def test_save_unfitted_detector_error(self):
        """Test that saving unfitted detector raises error"""
        detector = OutOfDistributionDetector()
        
        with pytest.raises(RuntimeError, match="Cannot save unfitted detector"):
            detector.save("test.pkl")
    
    def test_reset(self, fitted_detector):
        """Test detector reset functionality"""
        assert fitted_detector.is_fitted == True
        assert len(fitted_detector.feature_stats) > 0
        
        fitted_detector.reset()
        
        assert fitted_detector.is_fitted == False
        assert len(fitted_detector.feature_stats) == 0
        assert len(fitted_detector.feature_names) == 0
    
    def test_repr(self, fitted_detector):
        """Test string representation"""
        repr_str = repr(fitted_detector)
        
        assert "OutOfDistributionDetector" in repr_str
        assert "fitted=True" in repr_str
        assert "n_features=3" in repr_str
        assert "threshold_std=3.0" in repr_str


if __name__ == "__main__":
    pytest.main([__file__])