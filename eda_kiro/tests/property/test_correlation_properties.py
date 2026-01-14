"""Property-based tests for CorrelationAnalyzer."""

import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings

from nasa_pcoe_eda.analysis.correlation import CorrelationAnalyzer


class TestCorrelationAnalyzerProperties:
    """Property-based tests for CorrelationAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = CorrelationAnalyzer()

    # Feature: nasa-pcoe-eda, Property 9: 相関計算の対称性
    @given(st.data())
    @settings(max_examples=100)
    def test_correlation_calculation_symmetry(self, data):
        """
        任意の数値データセットに対して、特徴量AとBの相関係数は、
        特徴量BとAの相関係数と等しい（相関行列は対称である）
        
        Property 9: Correlation Calculation Symmetry
        For any numerical dataset, the correlation coefficient between feature A and B
        should equal the correlation coefficient between feature B and A (correlation matrix is symmetric).
        """
        # Generate a DataFrame with numeric columns
        num_columns = data.draw(st.integers(min_value=2, max_value=10))
        num_rows = data.draw(st.integers(min_value=5, max_value=100))
        
        # Generate DataFrame data manually
        df_data = {}
        for i in range(num_columns):
            col_name = f"feature_{i}"
            # Generate numeric data for this column
            column_data = data.draw(st.lists(
                st.floats(
                    min_value=-1000, 
                    max_value=1000, 
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
        if numeric_df.empty or len(numeric_df.columns) < 2:
            return
        
        # Remove columns with zero variance (constant columns)
        # as they can cause issues with correlation calculation
        for col in numeric_df.columns:
            if numeric_df[col].nunique() <= 1:
                numeric_df = numeric_df.drop(columns=[col])
        
        # Skip if we don't have at least 2 columns after filtering
        if len(numeric_df.columns) < 2:
            return
        
        # Compute correlation matrix
        corr_matrix = self.analyzer.compute_correlation_matrix(numeric_df)
        
        # Skip if correlation matrix is empty (can happen with all-NaN columns)
        if corr_matrix.empty:
            return
        
        # Verify symmetry property: corr_matrix[i,j] == corr_matrix[j,i]
        for i, feature_a in enumerate(corr_matrix.index):
            for j, feature_b in enumerate(corr_matrix.columns):
                corr_ab = corr_matrix.loc[feature_a, feature_b]
                corr_ba = corr_matrix.loc[feature_b, feature_a]
                
                # Handle NaN values - if one is NaN, both should be NaN
                if pd.isna(corr_ab) or pd.isna(corr_ba):
                    assert pd.isna(corr_ab) and pd.isna(corr_ba), (
                        f"Asymmetric NaN values: corr({feature_a}, {feature_b}) = {corr_ab}, "
                        f"corr({feature_b}, {feature_a}) = {corr_ba}. "
                        f"Both should be NaN or both should be numeric."
                    )
                else:
                    # For numeric values, they should be equal (within floating point precision)
                    assert abs(corr_ab - corr_ba) < 1e-10, (
                        f"Correlation matrix is not symmetric: "
                        f"corr({feature_a}, {feature_b}) = {corr_ab}, "
                        f"corr({feature_b}, {feature_a}) = {corr_ba}. "
                        f"Difference: {abs(corr_ab - corr_ba)}. "
                        f"This violates the correlation symmetry property."
                    )
        
        # Additional verification: matrix should equal its transpose
        # (excluding NaN comparisons which are handled above)
        corr_values = corr_matrix.values
        corr_transpose = corr_matrix.T.values
        
        # Create mask for non-NaN values
        non_nan_mask = ~(np.isnan(corr_values) | np.isnan(corr_transpose))
        
        if non_nan_mask.any():
            # Compare non-NaN values
            np.testing.assert_allclose(
                corr_values[non_nan_mask],
                corr_transpose[non_nan_mask],
                rtol=1e-10,
                atol=1e-10,
                err_msg="Correlation matrix is not symmetric when compared with its transpose"
            )
        
        # Verify diagonal elements are 1.0 (self-correlation)
        for i, feature in enumerate(corr_matrix.index):
            diagonal_value = corr_matrix.iloc[i, i]
            if not pd.isna(diagonal_value):
                assert abs(diagonal_value - 1.0) < 1e-10, (
                    f"Diagonal element for feature '{feature}' is {diagonal_value}, "
                    f"but should be 1.0 (perfect self-correlation). "
                    f"This violates the correlation matrix property."
                )

    # Feature: nasa-pcoe-eda, Property 18: 相関の範囲制約
    @given(st.data())
    @settings(max_examples=100)
    def test_correlation_range_constraints(self, data):
        """
        任意の数値データセットに対して、計算される全ての相関係数は-1から1の範囲内である
        
        Property 18: Correlation Range Constraints
        For any numerical dataset, all computed correlation coefficients should be within the range [-1, 1].
        """
        # Generate a DataFrame with numeric columns
        num_columns = data.draw(st.integers(min_value=2, max_value=10))
        num_rows = data.draw(st.integers(min_value=5, max_value=100))
        
        # Generate DataFrame data manually
        df_data = {}
        for i in range(num_columns):
            col_name = f"feature_{i}"
            # Generate numeric data for this column
            column_data = data.draw(st.lists(
                st.floats(
                    min_value=-1000, 
                    max_value=1000, 
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
        if numeric_df.empty or len(numeric_df.columns) < 2:
            return
        
        # Remove columns with zero variance (constant columns)
        # as they can cause issues with correlation calculation
        for col in numeric_df.columns:
            if numeric_df[col].nunique() <= 1:
                numeric_df = numeric_df.drop(columns=[col])
        
        # Skip if we don't have at least 2 columns after filtering
        if len(numeric_df.columns) < 2:
            return
        
        # Compute correlation matrix
        corr_matrix = self.analyzer.compute_correlation_matrix(numeric_df)
        
        # Skip if correlation matrix is empty (can happen with all-NaN columns)
        if corr_matrix.empty:
            return
        
        # Verify that all correlation coefficients are within [-1, 1] range
        for i, feature_a in enumerate(corr_matrix.index):
            for j, feature_b in enumerate(corr_matrix.columns):
                corr_value = corr_matrix.loc[feature_a, feature_b]
                
                # Skip NaN values (they are valid in correlation matrices)
                if pd.isna(corr_value):
                    continue
                
                # Check that correlation coefficient is within valid range [-1, 1]
                # Account for floating-point precision errors with a small tolerance
                tolerance = 1e-10
                assert -1.0 - tolerance <= corr_value <= 1.0 + tolerance, (
                    f"Correlation coefficient between '{feature_a}' and '{feature_b}' "
                    f"is {corr_value}, which is outside the valid range [-1, 1] "
                    f"(with tolerance {tolerance}). "
                    f"All correlation coefficients must be within this range. "
                    f"This violates the correlation range constraint property."
                )
        
        # Additional verification using numpy operations for efficiency
        corr_values = corr_matrix.values
        
        # Create mask for non-NaN values
        non_nan_mask = ~np.isnan(corr_values)
        
        if non_nan_mask.any():
            # Check all non-NaN values are within [-1, 1] with tolerance
            tolerance = 1e-10
            valid_range_mask = (corr_values >= -1.0 - tolerance) & (corr_values <= 1.0 + tolerance)
            combined_mask = non_nan_mask & valid_range_mask
            
            # All non-NaN values should satisfy the range constraint
            assert np.array_equal(non_nan_mask, combined_mask), (
                f"Some correlation coefficients are outside the valid range [-1, 1] "
                f"(with tolerance {tolerance}). "
                f"Found values outside range: "
                f"{corr_values[non_nan_mask & ~valid_range_mask].tolist()}. "
                f"This violates the correlation range constraint property."
            )

    # Feature: nasa-pcoe-eda, Property 19: 強相関識別の閾値一貫性
    @given(st.data())
    @settings(max_examples=100)
    def test_high_correlation_identification_threshold_consistency(self, data):
        """
        任意のデータセットと閾値に対して、強い相関として識別される特徴量ペアの相関係数の絶対値は、指定された閾値以上である
        
        Property 19: High Correlation Identification Threshold Consistency
        For any dataset and threshold, the absolute value of correlation coefficients of feature pairs 
        identified as having strong correlation should be greater than or equal to the specified threshold.
        """
        # Generate a DataFrame with numeric columns
        num_columns = data.draw(st.integers(min_value=2, max_value=8))
        num_rows = data.draw(st.integers(min_value=10, max_value=50))
        
        # Generate DataFrame data manually
        df_data = {}
        for i in range(num_columns):
            col_name = f"feature_{i}"
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
        if numeric_df.empty or len(numeric_df.columns) < 2:
            return
        
        # Remove columns with zero variance (constant columns)
        # as they can cause issues with correlation calculation
        for col in numeric_df.columns:
            if numeric_df[col].nunique() <= 1:
                numeric_df = numeric_df.drop(columns=[col])
        
        # Skip if we don't have at least 2 columns after filtering
        if len(numeric_df.columns) < 2:
            return
        
        # Generate a random threshold between 0.1 and 0.95
        threshold = data.draw(st.floats(min_value=0.1, max_value=0.95))
        
        # Compute correlation matrix
        corr_matrix = self.analyzer.compute_correlation_matrix(numeric_df)
        
        # Skip if correlation matrix is empty (can happen with all-NaN columns)
        if corr_matrix.empty:
            return
        
        # Identify high correlations using the threshold
        high_correlations = self.analyzer.identify_high_correlations(corr_matrix, threshold)
        
        # Verify threshold consistency property
        for feature1, feature2, correlation_value in high_correlations:
            abs_correlation = abs(correlation_value)
            
            # The absolute value of correlation should be >= threshold
            assert abs_correlation >= threshold, (
                f"High correlation identified between '{feature1}' and '{feature2}' "
                f"with correlation value {correlation_value} (absolute: {abs_correlation}), "
                f"but this is below the specified threshold {threshold}. "
                f"All identified high correlations must have absolute correlation >= threshold. "
                f"This violates the threshold consistency property."
            )
        
        # Additional verification: check that no correlations above threshold are missed
        # (This ensures the method is not only precise but also complete)
        upper_triangle_indices = np.triu_indices_from(corr_matrix, k=1)
        
        for i, j in zip(upper_triangle_indices[0], upper_triangle_indices[1]):
            feature1 = corr_matrix.index[i]
            feature2 = corr_matrix.columns[j]
            correlation_value = corr_matrix.iloc[i, j]
            
            # Skip NaN correlations
            if pd.isna(correlation_value):
                continue
            
            abs_correlation = abs(correlation_value)
            
            # If absolute correlation is >= threshold, it should be in high_correlations list
            if abs_correlation >= threshold:
                # Check if this pair is in the high_correlations list
                found = False
                for hc_feat1, hc_feat2, hc_corr in high_correlations:
                    if ((hc_feat1 == feature1 and hc_feat2 == feature2) or 
                        (hc_feat1 == feature2 and hc_feat2 == feature1)):
                        found = True
                        # Verify the correlation value matches
                        assert abs(hc_corr - correlation_value) < 1e-10, (
                            f"Correlation value mismatch for pair ({feature1}, {feature2}): "
                            f"expected {correlation_value}, got {hc_corr}. "
                            f"This indicates an inconsistency in correlation identification."
                        )
                        break
                
                assert found, (
                    f"Feature pair ({feature1}, {feature2}) with correlation {correlation_value} "
                    f"(absolute: {abs_correlation}) meets the threshold {threshold} "
                    f"but was not identified as a high correlation. "
                    f"This violates the completeness aspect of threshold consistency."
                )
        
        # Verify that the results are sorted by absolute correlation value (descending)
        if len(high_correlations) > 1:
            for i in range(len(high_correlations) - 1):
                current_abs_corr = abs(high_correlations[i][2])
                next_abs_corr = abs(high_correlations[i + 1][2])
                
                assert current_abs_corr >= next_abs_corr, (
                    f"High correlations are not sorted by absolute correlation value. "
                    f"Found {current_abs_corr} followed by {next_abs_corr}. "
                    f"Results should be sorted in descending order of absolute correlation."
                )