"""Property-based tests for OutlierDetector."""

import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings
from hypothesis.extra.pandas import data_frames, columns

from nasa_pcoe_eda.analysis.outliers import OutlierDetector


class TestOutlierDetectorProperties:
    """Property-based tests for OutlierDetector."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = OutlierDetector()

    # Feature: nasa-pcoe-eda, Property 11: 外れ値検出の閾値依存性
    @given(st.data())
    @settings(max_examples=100)
    def test_outlier_detection_threshold_dependency(self, data):
        """
        任意のデータセットに対して、より厳しい閾値（例：IQRの1.5から3.0）を使用すると、
        検出される外れ値の数は減少するか同じである
        
        Property 11: Outlier Detection Threshold Dependency
        For any dataset, when using a stricter threshold (e.g., IQR from 1.5 to 3.0),
        the number of detected outliers should decrease or stay the same.
        """
        # Generate a DataFrame with numeric columns
        num_columns = data.draw(st.integers(min_value=1, max_value=5))
        num_rows = data.draw(st.integers(min_value=10, max_value=100))
        
        # Generate DataFrame data manually
        df_data = {}
        for i in range(num_columns):
            col_name = f"feature_{i}"
            # Generate numeric data for this column
            column_data = data.draw(st.lists(
                st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                min_size=num_rows,
                max_size=num_rows
            ))
            df_data[col_name] = column_data
        
        df = pd.DataFrame(df_data)
        
        # Skip if DataFrame is empty or has no numeric columns
        if df.empty or len(df.select_dtypes(include=[np.number]).columns) == 0:
            return
        
        # Test IQR method with different thresholds
        # Stricter threshold (higher value) should detect fewer or equal outliers
        loose_threshold = data.draw(st.floats(min_value=1.0, max_value=2.0))
        strict_threshold = data.draw(st.floats(min_value=2.5, max_value=4.0))
        
        # Ensure strict_threshold > loose_threshold
        if strict_threshold <= loose_threshold:
            strict_threshold = loose_threshold + 1.0
        
        # Detect outliers with both thresholds
        loose_outliers = self.detector.detect_outliers_iqr(df, threshold=loose_threshold)
        strict_outliers = self.detector.detect_outliers_iqr(df, threshold=strict_threshold)
        
        # Verify the property: stricter threshold should detect fewer or equal outliers
        for column in loose_outliers:
            if column in strict_outliers:
                loose_count = len(loose_outliers[column])
                strict_count = len(strict_outliers[column])
                
                assert strict_count <= loose_count, (
                    f"Column '{column}': Stricter threshold ({strict_threshold}) detected "
                    f"more outliers ({strict_count}) than looser threshold ({loose_threshold}) "
                    f"which detected {loose_count} outliers. This violates the threshold dependency property."
                )
        
        # Test Z-score method with different thresholds
        loose_z_threshold = data.draw(st.floats(min_value=2.0, max_value=2.5))
        strict_z_threshold = data.draw(st.floats(min_value=3.0, max_value=5.0))
        
        # Ensure strict_z_threshold > loose_z_threshold
        if strict_z_threshold <= loose_z_threshold:
            strict_z_threshold = loose_z_threshold + 1.0
        
        # Detect outliers with both Z-score thresholds
        loose_z_outliers = self.detector.detect_outliers_zscore(df, threshold=loose_z_threshold)
        strict_z_outliers = self.detector.detect_outliers_zscore(df, threshold=strict_z_threshold)
        
        # Verify the property for Z-score method
        for column in loose_z_outliers:
            if column in strict_z_outliers:
                loose_z_count = len(loose_z_outliers[column])
                strict_z_count = len(strict_z_outliers[column])
                
                assert strict_z_count <= loose_z_count, (
                    f"Column '{column}': Stricter Z-score threshold ({strict_z_threshold}) detected "
                    f"more outliers ({strict_z_count}) than looser threshold ({loose_z_threshold}) "
                    f"which detected {loose_z_count} outliers. This violates the threshold dependency property."
                )

    # Feature: nasa-pcoe-eda, Property 12: 外れ値カウントの正確性
    @given(st.data())
    @settings(max_examples=100)
    def test_outlier_count_accuracy(self, data):
        """
        任意のデータセットと外れ値検出方法に対して、報告される外れ値の数は、
        実際に検出された外れ値インデックスの数と一致する
        
        Property 12: Outlier Count Accuracy
        For any dataset and outlier detection method, the reported number of outliers
        should match the actual number of detected outlier indices.
        """
        # Generate a DataFrame with numeric columns
        num_columns = data.draw(st.integers(min_value=1, max_value=5))
        num_rows = data.draw(st.integers(min_value=5, max_value=100))
        
        # Generate DataFrame data manually
        df_data = {}
        for i in range(num_columns):
            col_name = f"feature_{i}"
            # Generate numeric data for this column
            column_data = data.draw(st.lists(
                st.floats(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
                min_size=num_rows,
                max_size=num_rows
            ))
            df_data[col_name] = column_data
        
        df = pd.DataFrame(df_data)
        
        # Skip if DataFrame is empty or has no numeric columns
        if df.empty or len(df.select_dtypes(include=[np.number]).columns) == 0:
            return
        
        # Test IQR method
        iqr_threshold = data.draw(st.floats(min_value=1.0, max_value=3.0))
        iqr_outliers = self.detector.detect_outliers_iqr(df, threshold=iqr_threshold)
        iqr_summary = self.detector.summarize_outliers(iqr_outliers, total_records=len(df))
        
        # Verify that reported counts match actual indices counts
        for column in iqr_outliers:
            actual_count = len(iqr_outliers[column])
            reported_count = iqr_summary.outlier_counts[column]
            
            assert actual_count == reported_count, (
                f"IQR method - Column '{column}': Reported outlier count ({reported_count}) "
                f"does not match actual outlier indices count ({actual_count}). "
                f"This violates the outlier count accuracy property."
            )
            
            # Also verify that the indices are stored correctly
            assert len(iqr_summary.outlier_indices[column]) == actual_count, (
                f"IQR method - Column '{column}': Stored outlier indices count "
                f"({len(iqr_summary.outlier_indices[column])}) does not match "
                f"actual count ({actual_count})"
            )
        
        # Test Z-score method
        zscore_threshold = data.draw(st.floats(min_value=2.0, max_value=4.0))
        zscore_outliers = self.detector.detect_outliers_zscore(df, threshold=zscore_threshold)
        zscore_summary = self.detector.summarize_outliers(zscore_outliers, total_records=len(df))
        
        # Verify that reported counts match actual indices counts
        for column in zscore_outliers:
            actual_count = len(zscore_outliers[column])
            reported_count = zscore_summary.outlier_counts[column]
            
            assert actual_count == reported_count, (
                f"Z-score method - Column '{column}': Reported outlier count ({reported_count}) "
                f"does not match actual outlier indices count ({actual_count}). "
                f"This violates the outlier count accuracy property."
            )
            
            # Also verify that the indices are stored correctly
            assert len(zscore_summary.outlier_indices[column]) == actual_count, (
                f"Z-score method - Column '{column}': Stored outlier indices count "
                f"({len(zscore_summary.outlier_indices[column])}) does not match "
                f"actual count ({actual_count})"
            )
        
        # Test edge case: empty outliers
        empty_outliers = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            empty_outliers[col] = np.array([], dtype=int)
        
        empty_summary = self.detector.summarize_outliers(empty_outliers, total_records=len(df))
        
        for column in empty_outliers:
            actual_count = len(empty_outliers[column])
            reported_count = empty_summary.outlier_counts[column]
            
            assert actual_count == reported_count == 0, (
                f"Empty outliers - Column '{column}': Expected 0 outliers, "
                f"but got actual_count={actual_count}, reported_count={reported_count}"
            )