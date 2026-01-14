"""Unit tests for FaultLevelAnalyzer."""

import numpy as np
import pandas as pd
import pytest

from nasa_pcoe_eda.analysis.fault_level import FaultLevelAnalyzer
from nasa_pcoe_eda.models import DistributionComparison
from nasa_pcoe_eda.exceptions import AnalysisError


class TestFaultLevelAnalyzer:
    """Test cases for FaultLevelAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = FaultLevelAnalyzer()

    def test_identify_discriminative_features_basic(self):
        """Test basic discriminative feature identification."""
        # Create test data with discriminative features
        np.random.seed(42)
        
        # Feature 1: discriminative (different means for different fault levels)
        normal_data = np.random.normal(0, 1, 50)
        fault_data = np.random.normal(5, 1, 50)
        feature1 = np.concatenate([normal_data, fault_data])
        
        # Feature 2: non-discriminative (same distribution)
        feature2 = np.random.normal(0, 1, 100)
        
        # Feature 3: discriminative (different means, smaller difference)
        normal_data2 = np.random.normal(0, 1, 50)
        fault_data2 = np.random.normal(2, 1, 50)
        feature3 = np.concatenate([normal_data2, fault_data2])
        
        fault_labels = [0] * 50 + [1] * 50
        
        df = pd.DataFrame({
            'feature1': feature1,
            'feature2': feature2,
            'feature3': feature3,
            'fault': fault_labels
        })
        
        discriminative_features = self.analyzer.identify_discriminative_features(df, 'fault')
        
        assert isinstance(discriminative_features, list)
        assert 'feature1' in discriminative_features
        # feature2 should not be discriminative
        assert 'feature2' not in discriminative_features
        # feature3 should also be discriminative due to different means
        assert 'feature3' in discriminative_features

    def test_identify_discriminative_features_empty_dataframe(self):
        """Test discriminative feature identification with empty DataFrame."""
        df = pd.DataFrame()
        discriminative_features = self.analyzer.identify_discriminative_features(df, 'fault')
        
        assert discriminative_features == []

    def test_identify_discriminative_features_missing_fault_column(self):
        """Test discriminative feature identification with missing fault column."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        
        with pytest.raises(AnalysisError, match="Fault column 'fault' not found"):
            self.analyzer.identify_discriminative_features(df, 'fault')

    def test_identify_discriminative_features_no_numeric_columns(self):
        """Test discriminative feature identification with no numeric columns."""
        df = pd.DataFrame({
            'text': ['a', 'b', 'c', 'd'],
            'fault': [0, 0, 1, 1]
        })
        
        discriminative_features = self.analyzer.identify_discriminative_features(df, 'fault')
        
        assert discriminative_features == []

    def test_identify_discriminative_features_single_fault_level(self):
        """Test discriminative feature identification with single fault level."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10],
            'fault': [0, 0, 0, 0, 0]  # All same fault level
        })
        
        discriminative_features = self.analyzer.identify_discriminative_features(df, 'fault')
        
        assert discriminative_features == []

    def test_identify_discriminative_features_insufficient_data(self):
        """Test discriminative feature identification with insufficient data."""
        df = pd.DataFrame({
            'feature1': [1],
            'fault': [0]
        })
        
        discriminative_features = self.analyzer.identify_discriminative_features(df, 'fault')
        
        assert discriminative_features == []

    def test_identify_discriminative_features_with_missing_values(self):
        """Test discriminative feature identification with missing values."""
        df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5, 6],
            'feature2': [10, 20, 30, np.nan, 50, 60],
            'fault': [0, 0, 0, 1, 1, np.nan]  # Missing fault label
        })
        
        discriminative_features = self.analyzer.identify_discriminative_features(df, 'fault')
        
        # Should handle missing values gracefully
        assert isinstance(discriminative_features, list)

    def test_compare_distributions_basic(self):
        """Test basic distribution comparison."""
        # Create test data with different distributions
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, 50)
        fault_data = np.random.normal(3, 1, 50)
        
        df = pd.DataFrame({
            'feature1': np.concatenate([normal_data, fault_data]),
            'feature2': np.random.normal(0, 1, 100),  # Same distribution
            'fault': [0] * 50 + [1] * 50
        })
        
        comparison = self.analyzer.compare_distributions(df, 'fault', ['feature1', 'feature2'])
        
        assert isinstance(comparison, DistributionComparison)
        assert isinstance(comparison.feature_distributions, dict)
        assert isinstance(comparison.statistical_tests, dict)
        
        # Check that feature1 has distributions for both fault levels
        assert 'feature1' in comparison.feature_distributions
        feature1_dist = comparison.feature_distributions['feature1']
        assert '0' in feature1_dist
        assert '1' in feature1_dist
        
        # Check statistical test results
        assert 'feature1' in comparison.statistical_tests
        test_result = comparison.statistical_tests['feature1']
        assert 'test' in test_result
        assert 'p_value' in test_result

    def test_compare_distributions_missing_fault_column(self):
        """Test distribution comparison with missing fault column."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        
        with pytest.raises(AnalysisError, match="Fault column 'fault' not found"):
            self.analyzer.compare_distributions(df, 'fault', ['feature1'])

    def test_compare_distributions_insufficient_data(self):
        """Test distribution comparison with insufficient data."""
        df = pd.DataFrame({
            'feature1': [1],
            'fault': [0]
        })
        
        comparison = self.analyzer.compare_distributions(df, 'fault', ['feature1'])
        
        assert isinstance(comparison, DistributionComparison)
        assert comparison.feature_distributions == {}
        assert comparison.statistical_tests == {}

    def test_compare_distributions_multiclass(self):
        """Test distribution comparison with multiple classes."""
        np.random.seed(42)
        
        # Create data with 3 fault levels
        class0_data = np.random.normal(0, 1, 30)
        class1_data = np.random.normal(3, 1, 30)
        class2_data = np.random.normal(6, 1, 30)
        
        df = pd.DataFrame({
            'feature1': np.concatenate([class0_data, class1_data, class2_data]),
            'fault': [0] * 30 + [1] * 30 + [2] * 30
        })
        
        comparison = self.analyzer.compare_distributions(df, 'fault', ['feature1'])
        
        assert isinstance(comparison, DistributionComparison)
        
        # Should have distributions for all 3 classes
        feature1_dist = comparison.feature_distributions['feature1']
        assert '0' in feature1_dist
        assert '1' in feature1_dist
        assert '2' in feature1_dist
        
        # Should use ANOVA for multiclass
        test_result = comparison.statistical_tests['feature1']
        assert test_result['test'] == 'ANOVA'

    def test_compute_class_separability_basic(self):
        """Test basic class separability computation."""
        # Create test data with known separability
        np.random.seed(42)
        
        # Highly separable feature
        separable_feature = np.concatenate([
            np.random.normal(0, 0.5, 50),  # Class 0
            np.random.normal(5, 0.5, 50)   # Class 1
        ])
        
        # Less separable feature
        less_separable = np.concatenate([
            np.random.normal(0, 2, 50),    # Class 0
            np.random.normal(1, 2, 50)     # Class 1
        ])
        
        df = pd.DataFrame({
            'separable': separable_feature,
            'less_separable': less_separable,
            'fault': [0] * 50 + [1] * 50
        })
        
        separability = self.analyzer.compute_class_separability(
            df, 'fault', ['separable', 'less_separable']
        )
        
        assert isinstance(separability, dict)
        assert 'separable' in separability
        assert 'less_separable' in separability
        
        # Highly separable feature should have higher score
        assert separability['separable'] > separability['less_separable']
        
        # All scores should be non-negative
        assert all(score >= 0 for score in separability.values())

    def test_compute_class_separability_missing_fault_column(self):
        """Test class separability computation with missing fault column."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [2, 4, 6, 8, 10]
        })
        
        with pytest.raises(AnalysisError, match="Fault column 'fault' not found"):
            self.analyzer.compute_class_separability(df, 'fault', ['feature1'])

    def test_compute_class_separability_insufficient_data(self):
        """Test class separability computation with insufficient data."""
        df = pd.DataFrame({
            'feature1': [1],
            'fault': [0]
        })
        
        separability = self.analyzer.compute_class_separability(df, 'fault', ['feature1'])
        
        assert separability == {}

    def test_compute_class_separability_single_class(self):
        """Test class separability computation with single class."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'fault': [0, 0, 0, 0, 0]  # All same class
        })
        
        separability = self.analyzer.compute_class_separability(df, 'fault', ['feature1'])
        
        assert separability == {}

    def test_compute_class_separability_with_missing_values(self):
        """Test class separability computation with missing values."""
        df = pd.DataFrame({
            'feature1': [1, 2, np.nan, 4, 5, 6],
            'feature2': [10, 20, 30, np.nan, 50, 60],
            'fault': [0, 0, 0, 1, 1, 1]
        })
        
        separability = self.analyzer.compute_class_separability(
            df, 'fault', ['feature1', 'feature2']
        )
        
        # Should handle missing values gracefully
        assert isinstance(separability, dict)

    def test_fisher_score_computation(self):
        """Test Fisher score computation directly."""
        # Create data with known Fisher score characteristics
        np.random.seed(42)
        
        # Perfect separation case
        feature_values = np.array([1, 1, 1, 10, 10, 10])
        labels = np.array([0, 0, 0, 1, 1, 1])
        
        fisher_score = self.analyzer._compute_fisher_score(feature_values, labels)
        
        # Should be a high positive value for perfect separation
        assert fisher_score > 0
        assert isinstance(fisher_score, float)

    def test_fisher_score_single_class(self):
        """Test Fisher score with single class."""
        feature_values = np.array([1, 2, 3, 4, 5])
        labels = np.array([0, 0, 0, 0, 0])
        
        fisher_score = self.analyzer._compute_fisher_score(feature_values, labels)
        
        # Should be 0 for single class
        assert fisher_score == 0.0

    def test_fisher_score_perfect_separation(self):
        """Test Fisher score with perfect separation (no within-class variance)."""
        feature_values = np.array([1, 1, 5, 5])
        labels = np.array([0, 0, 1, 1])
        
        fisher_score = self.analyzer._compute_fisher_score(feature_values, labels)
        
        # Should handle perfect separation case
        assert fisher_score >= 0
        assert not np.isnan(fisher_score)

    def test_is_discriminative_feature_basic(self):
        """Test discriminative feature check."""
        # Create discriminative data
        np.random.seed(42)
        df = pd.DataFrame({
            'discriminative': np.concatenate([
                np.random.normal(0, 1, 50),
                np.random.normal(5, 1, 50)
            ]),
            'non_discriminative': np.random.normal(0, 1, 100),
            'fault': [0] * 50 + [1] * 50
        })
        
        # Test discriminative feature
        is_disc = self.analyzer._is_discriminative_feature(df, 'discriminative', 'fault')
        assert is_disc == True
        
        # Test non-discriminative feature
        is_non_disc = self.analyzer._is_discriminative_feature(df, 'non_discriminative', 'fault')
        assert is_non_disc == False

    def test_is_discriminative_feature_insufficient_data(self):
        """Test discriminative feature check with insufficient data."""
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'fault': [0, 0, 1]
        })
        
        is_disc = self.analyzer._is_discriminative_feature(df, 'feature1', 'fault')
        assert is_disc is False

    def test_is_discriminative_feature_multiclass(self):
        """Test discriminative feature check with multiple classes."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.concatenate([
                np.random.normal(0, 1, 20),
                np.random.normal(3, 1, 20),
                np.random.normal(6, 1, 20)
            ]),
            'fault': [0] * 20 + [1] * 20 + [2] * 20
        })
        
        is_disc = self.analyzer._is_discriminative_feature(df, 'feature1', 'fault')
        assert is_disc == True

    def test_class_separability_non_negative(self):
        """Test that class separability scores are non-negative."""
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'fault': np.random.choice([0, 1], 100)
        })
        
        separability = self.analyzer.compute_class_separability(
            df, 'fault', ['feature1', 'feature2']
        )
        
        # All separability scores should be non-negative
        for score in separability.values():
            assert score >= 0.0

    def test_distribution_comparison_statistical_significance(self):
        """Test that statistical tests correctly identify significance."""
        np.random.seed(42)
        
        # Create clearly different distributions
        different_dist = np.concatenate([
            np.random.normal(0, 1, 50),
            np.random.normal(10, 1, 50)  # Very different mean
        ])
        
        # Create similar distributions
        similar_dist = np.concatenate([
            np.random.normal(0, 1, 50),
            np.random.normal(0.1, 1, 50)  # Very similar mean
        ])
        
        df = pd.DataFrame({
            'different': different_dist,
            'similar': similar_dist,
            'fault': [0] * 50 + [1] * 50
        })
        
        comparison = self.analyzer.compare_distributions(
            df, 'fault', ['different', 'similar']
        )
        
        # Different distributions should be significant
        different_test = comparison.statistical_tests['different']
        assert different_test['significant'] == True
        assert different_test['p_value'] < 0.05
        
        # Similar distributions should not be significant
        similar_test = comparison.statistical_tests['similar']
        assert similar_test['significant'] == False
        assert similar_test['p_value'] >= 0.05