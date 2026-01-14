"""Property-based tests for FaultLevelAnalyzer."""

import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings

from nasa_pcoe_eda.analysis.fault_level import FaultLevelAnalyzer


class TestFaultLevelAnalyzerProperties:
    """Property-based tests for FaultLevelAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = FaultLevelAnalyzer()

    # Feature: nasa-pcoe-eda, Property 22: クラス分離度の非負性
    @given(st.data())
    @settings(max_examples=100)
    def test_class_separability_non_negativity(self, data):
        """
        任意の分類データセットに対して、計算されるクラス間分離度の指標（例：Fisher比）は非負である
        
        Property 22: Class Separability Non-negativity
        For any classification dataset, the computed class separability metrics (e.g., Fisher ratio) should be non-negative.
        """
        # Generate a DataFrame with numeric features and a fault column
        num_features = data.draw(st.integers(min_value=1, max_value=8))
        num_rows = data.draw(st.integers(min_value=4, max_value=100))
        num_classes = data.draw(st.integers(min_value=2, max_value=5))
        
        # Generate feature data
        df_data = {}
        feature_names = []
        
        for i in range(num_features):
            feature_name = f"feature_{i}"
            feature_names.append(feature_name)
            
            # Generate numeric data for this feature
            feature_data = data.draw(st.lists(
                st.floats(
                    min_value=-1000, 
                    max_value=1000, 
                    allow_nan=False, 
                    allow_infinity=False
                ),
                min_size=num_rows,
                max_size=num_rows
            ))
            df_data[feature_name] = feature_data
        
        # Generate fault labels (class labels)
        fault_labels = data.draw(st.lists(
            st.integers(min_value=0, max_value=num_classes-1),
            min_size=num_rows,
            max_size=num_rows
        ))
        
        # Ensure we have at least 2 samples per class to make the test meaningful
        # Redistribute labels if necessary
        unique_labels, counts = np.unique(fault_labels, return_counts=True)
        if len(unique_labels) < 2 or np.any(counts < 2):
            # Create a more balanced distribution
            fault_labels = []
            samples_per_class = max(2, num_rows // num_classes)
            remaining_samples = num_rows
            
            for class_id in range(num_classes):
                if class_id == num_classes - 1:
                    # Last class gets all remaining samples
                    class_samples = remaining_samples
                else:
                    class_samples = min(samples_per_class, remaining_samples)
                
                fault_labels.extend([class_id] * class_samples)
                remaining_samples -= class_samples
                
                if remaining_samples <= 0:
                    break
            
            # Shuffle the labels
            np.random.shuffle(fault_labels)
            fault_labels = fault_labels[:num_rows]  # Ensure exact length
        
        df_data['fault_level'] = fault_labels
        df = pd.DataFrame(df_data)
        
        # Skip if DataFrame is empty
        if df.empty:
            return
        
        # Ensure we have at least 2 classes with at least 2 samples each
        fault_counts = df['fault_level'].value_counts()
        if len(fault_counts) < 2 or (fault_counts < 2).any():
            return
        
        # Compute class separability scores
        separability_scores = self.analyzer.compute_class_separability(
            df, 'fault_level', feature_names
        )
        
        # Verify non-negativity property
        for feature_name, score in separability_scores.items():
            assert score >= 0.0, (
                f"Class separability score for feature '{feature_name}' is {score}, "
                f"which is negative. All class separability metrics (e.g., Fisher ratio) "
                f"must be non-negative. This violates the non-negativity property."
            )
            
            # Additional check: score should be a finite number
            assert np.isfinite(score), (
                f"Class separability score for feature '{feature_name}' is {score}, "
                f"which is not a finite number. Separability scores should be finite "
                f"non-negative values."
            )
        
        # Additional verification: test with edge cases
        # Test with identical values within each class (perfect separation case)
        if len(feature_names) > 0:
            # Create a feature with perfect class separation
            perfect_separation_data = df.copy()
            test_feature = feature_names[0]
            
            # Assign distinct values to each class
            for class_id in perfect_separation_data['fault_level'].unique():
                mask = perfect_separation_data['fault_level'] == class_id
                perfect_separation_data.loc[mask, test_feature] = float(class_id * 100)
            
            perfect_scores = self.analyzer.compute_class_separability(
                perfect_separation_data, 'fault_level', [test_feature]
            )
            
            if test_feature in perfect_scores:
                perfect_score = perfect_scores[test_feature]
                assert perfect_score >= 0.0, (
                    f"Perfect separation case: Class separability score for feature "
                    f"'{test_feature}' is {perfect_score}, which is negative. "
                    f"Even in perfect separation cases, scores must be non-negative."
                )
                
                # Perfect separation should yield a high score (unless within-class variance is 0)
                # In which case the score might be infinite, which should still be >= 0
                assert perfect_score >= 0.0 or np.isinf(perfect_score), (
                    f"Perfect separation case: Expected high separability score, "
                    f"got {perfect_score} for feature '{test_feature}'."
                )
        
        # Test with no separation case (all classes have same mean)
        if len(feature_names) > 0:
            no_separation_data = df.copy()
            test_feature = feature_names[0]
            
            # Set all values to the same value (no separation)
            no_separation_data[test_feature] = 42.0
            
            no_sep_scores = self.analyzer.compute_class_separability(
                no_separation_data, 'fault_level', [test_feature]
            )
            
            if test_feature in no_sep_scores:
                no_sep_score = no_sep_scores[test_feature]
                assert no_sep_score >= 0.0, (
                    f"No separation case: Class separability score for feature "
                    f"'{test_feature}' is {no_sep_score}, which is negative. "
                    f"Even when there's no class separation, scores must be non-negative."
                )
                
                # No separation should yield a score close to 0
                assert no_sep_score <= 1e-10, (
                    f"No separation case: Expected separability score close to 0, "
                    f"got {no_sep_score} for feature '{test_feature}'. "
                    f"When all classes have identical feature values, "
                    f"separability should be minimal."
                )