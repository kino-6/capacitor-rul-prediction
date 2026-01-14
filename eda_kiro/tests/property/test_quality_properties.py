"""Property-based tests for DataQualityAnalyzer."""

import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings

from nasa_pcoe_eda.analysis.quality import DataQualityAnalyzer


class TestDataQualityAnalyzerProperties:
    """Property-based tests for DataQualityAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = DataQualityAnalyzer()

    # Feature: nasa-pcoe-eda, Property 16: 重複検出の正確性
    @given(st.data())
    @settings(max_examples=100)
    def test_duplicate_detection_accuracy(self, data):
        """
        任意のデータセットに対して、重複として報告されるレコードは、全ての列の値が完全に一致する
        
        Property 16: Duplicate Detection Accuracy
        For any dataset, records reported as duplicates should have all column values completely matching.
        """
        # Generate a DataFrame with various data types
        num_columns = data.draw(st.integers(min_value=1, max_value=8))
        num_rows = data.draw(st.integers(min_value=1, max_value=50))
        
        # Generate DataFrame data with mixed types
        df_data = {}
        for i in range(num_columns):
            col_name = f"col_{i}"
            
            # Choose column type randomly
            col_type = data.draw(st.sampled_from(['int', 'float', 'str', 'bool']))
            
            if col_type == 'int':
                column_data = data.draw(st.lists(
                    st.integers(min_value=-100, max_value=100),
                    min_size=num_rows,
                    max_size=num_rows
                ))
            elif col_type == 'float':
                column_data = data.draw(st.lists(
                    st.floats(
                        min_value=-100.0, 
                        max_value=100.0, 
                        allow_nan=False, 
                        allow_infinity=False
                    ),
                    min_size=num_rows,
                    max_size=num_rows
                ))
            elif col_type == 'str':
                column_data = data.draw(st.lists(
                    st.text(min_size=0, max_size=10, alphabet=st.characters(min_codepoint=32, max_codepoint=126)),
                    min_size=num_rows,
                    max_size=num_rows
                ))
            else:  # bool
                column_data = data.draw(st.lists(
                    st.booleans(),
                    min_size=num_rows,
                    max_size=num_rows
                ))
            
            df_data[col_name] = column_data
        
        df = pd.DataFrame(df_data)
        
        # Skip empty DataFrames
        if df.empty:
            return
        
        # Detect duplicates using our analyzer
        duplicate_records, duplicate_count = self.analyzer.detect_duplicates(df)
        
        # Property verification: All records in duplicate_records should be actual duplicates
        if not duplicate_records.empty:
            # Group duplicate records by their values to verify they are truly duplicates
            for _, group in duplicate_records.groupby(list(duplicate_records.columns)):
                # Each group should have at least 2 identical rows
                assert len(group) >= 2, (
                    f"Found a group in duplicate records with only {len(group)} row(s). "
                    f"Duplicate records should only contain rows that appear multiple times. "
                    f"Group data: {group.to_dict('records')}. "
                    f"This violates the duplicate detection accuracy property."
                )
                
                # Verify all rows in the group are identical
                first_row = group.iloc[0]
                for idx in range(1, len(group)):
                    current_row = group.iloc[idx]
                    
                    # Compare each column value
                    for col in group.columns:
                        first_val = first_row[col]
                        current_val = current_row[col]
                        
                        # Handle NaN values specially
                        if pd.isna(first_val) and pd.isna(current_val):
                            continue  # Both NaN, considered equal
                        elif pd.isna(first_val) or pd.isna(current_val):
                            assert False, (
                                f"Inconsistent NaN values in duplicate group for column '{col}': "
                                f"first_val={first_val}, current_val={current_val}. "
                                f"All rows in a duplicate group must have identical values. "
                                f"This violates the duplicate detection accuracy property."
                            )
                        else:
                            # For non-NaN values, they should be exactly equal
                            assert first_val == current_val, (
                                f"Values in duplicate group are not identical for column '{col}': "
                                f"first_val={first_val} (type: {type(first_val)}), "
                                f"current_val={current_val} (type: {type(current_val)}). "
                                f"All rows in a duplicate group must have identical values in all columns. "
                                f"This violates the duplicate detection accuracy property."
                            )
        
        # Additional verification: Check that duplicate_count matches pandas duplicated() behavior
        expected_duplicate_count = df.duplicated().sum()
        assert duplicate_count == expected_duplicate_count, (
            f"Duplicate count mismatch: analyzer reported {duplicate_count}, "
            f"but pandas duplicated() found {expected_duplicate_count}. "
            f"The duplicate count should match pandas' standard behavior. "
            f"This violates the duplicate detection accuracy property."
        )
        
        # Verify that all duplicates are captured (completeness check)
        pandas_duplicate_mask = df.duplicated(keep=False)
        expected_duplicate_records = df[pandas_duplicate_mask]
        
        # Sort both DataFrames for comparison (since order might differ)
        if not duplicate_records.empty and not expected_duplicate_records.empty:
            # Reset indices for comparison
            duplicate_records_sorted = duplicate_records.reset_index(drop=True).sort_values(
                by=list(duplicate_records.columns), na_position='last'
            ).reset_index(drop=True)
            
            expected_duplicate_records_sorted = expected_duplicate_records.reset_index(drop=True).sort_values(
                by=list(expected_duplicate_records.columns), na_position='last'
            ).reset_index(drop=True)
            
            # Compare the sorted DataFrames
            pd.testing.assert_frame_equal(
                duplicate_records_sorted,
                expected_duplicate_records_sorted,
                check_dtype=False,  # Allow for minor type differences
                check_exact=False,  # Allow for floating point precision issues
                check_names=True,
                check_like=True,
                obj="Duplicate records comparison"
            )
        elif duplicate_records.empty and expected_duplicate_records.empty:
            # Both are empty, which is correct
            pass
        else:
            # One is empty but the other is not - this is an error
            assert False, (
                f"Mismatch in duplicate detection completeness: "
                f"analyzer found {len(duplicate_records)} duplicate records, "
                f"but pandas found {len(expected_duplicate_records)} duplicate records. "
                f"All actual duplicates should be detected. "
                f"This violates the duplicate detection accuracy property."
            )
        
        # Edge case verification: If no duplicates exist, both should be empty
        if expected_duplicate_count == 0:
            assert duplicate_records.empty, (
                f"No duplicates expected (pandas found 0), but analyzer returned "
                f"{len(duplicate_records)} duplicate records. "
                f"When no duplicates exist, the result should be empty. "
                f"This violates the duplicate detection accuracy property."
            )
            assert duplicate_count == 0, (
                f"No duplicates expected, but analyzer reported duplicate_count={duplicate_count}. "
                f"When no duplicates exist, the count should be 0. "
                f"This violates the duplicate detection accuracy property."
            )

    # Feature: nasa-pcoe-eda, Property 17: データ品質問題の報告
    @given(st.data())
    @settings(max_examples=100)
    def test_data_quality_problem_reporting(self, data):
        """
        任意の品質問題を含むデータセットに対して、システムは問題の種類（欠損、重複、型不整合）と影響を受ける行/列を報告する
        
        Property 17: Data Quality Problem Reporting
        For any dataset containing quality issues, the system reports the types of problems 
        (missing values, duplicates, type inconsistencies) and the affected rows/columns.
        """
        # Generate a DataFrame with intentional quality issues
        num_columns = data.draw(st.integers(min_value=1, max_value=6))
        num_rows = data.draw(st.integers(min_value=2, max_value=20))
        
        # Create DataFrame with various quality issues
        df_data = {}
        expected_missing_columns = set()
        expected_duplicate_rows = 0
        expected_type_issue_columns = set()
        
        for i in range(num_columns):
            col_name = f"col_{i}"
            
            # Decide what type of issue to introduce (if any)
            issue_type = data.draw(st.sampled_from(['none', 'missing', 'mixed_types', 'normal']))
            
            if issue_type == 'missing':
                # Create column with missing values
                base_values = data.draw(st.lists(
                    st.integers(min_value=1, max_value=100),
                    min_size=num_rows // 2,
                    max_size=num_rows - 1
                ))
                # Add NaN values to create missing data
                missing_count = num_rows - len(base_values)
                column_data = base_values + [np.nan] * missing_count
                # Shuffle to distribute NaN values randomly
                np.random.shuffle(column_data)
                expected_missing_columns.add(col_name)
                
            elif issue_type == 'mixed_types':
                # Create column with mixed data types (in object dtype)
                mixed_values = []
                for j in range(num_rows):
                    value_type = data.draw(st.sampled_from(['int', 'str', 'float']))
                    if value_type == 'int':
                        mixed_values.append(data.draw(st.integers(min_value=1, max_value=100)))
                    elif value_type == 'str':
                        mixed_values.append(data.draw(st.text(min_size=1, max_size=5, alphabet='abcdef')))
                    else:  # float
                        mixed_values.append(data.draw(st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False)))
                
                column_data = mixed_values
                expected_type_issue_columns.add(col_name)
                
            else:  # 'none' or 'normal' - create normal column without issues
                column_data = data.draw(st.lists(
                    st.integers(min_value=1, max_value=100),
                    min_size=num_rows,
                    max_size=num_rows
                ))
            
            df_data[col_name] = column_data
        
        df = pd.DataFrame(df_data)
        
        # Optionally add some duplicate rows
        add_duplicates = data.draw(st.booleans())
        if add_duplicates and len(df) > 1:
            # Duplicate a random row
            row_to_duplicate = data.draw(st.integers(min_value=0, max_value=len(df) - 1))
            duplicate_row = df.iloc[row_to_duplicate:row_to_duplicate + 1].copy()
            df = pd.concat([df, duplicate_row], ignore_index=True)
            expected_duplicate_rows = 1
        
        # Skip if DataFrame is empty
        if df.empty:
            return
        
        # Generate quality report
        quality_report = self.analyzer.generate_quality_report(df)
        
        # Property verification: Check that all quality issues are properly reported
        
        # 1. Check missing value reporting
        completeness = quality_report['completeness']
        for col_name in expected_missing_columns:
            if col_name in completeness:
                assert completeness[col_name] < 100.0, (
                    f"Column '{col_name}' was expected to have missing values, "
                    f"but completeness is {completeness[col_name]}%. "
                    f"Missing value problems should be reported with completeness < 100%. "
                    f"This violates the data quality problem reporting property."
                )
        
        # Verify that columns without missing values have 100% completeness
        for col_name in df.columns:
            if col_name not in expected_missing_columns:
                actual_missing_count = df[col_name].isna().sum()
                if actual_missing_count == 0 and col_name in completeness:
                    assert completeness[col_name] == 100.0, (
                        f"Column '{col_name}' has no missing values, "
                        f"but completeness is {completeness[col_name]}%. "
                        f"Columns without missing values should have 100% completeness. "
                        f"This violates the data quality problem reporting property."
                    )
        
        # 2. Check duplicate reporting
        duplicates_info = quality_report['duplicates']
        actual_duplicate_count = df.duplicated().sum()
        
        assert duplicates_info['count'] == actual_duplicate_count, (
            f"Expected {actual_duplicate_count} duplicate rows, "
            f"but quality report shows {duplicates_info['count']}. "
            f"Duplicate problems should be accurately reported. "
            f"This violates the data quality problem reporting property."
        )
        
        if actual_duplicate_count > 0:
            assert not duplicates_info['records'].empty, (
                f"Found {actual_duplicate_count} duplicate rows, "
                f"but duplicate records DataFrame is empty. "
                f"When duplicates exist, the affected records should be reported. "
                f"This violates the data quality problem reporting property."
            )
        else:
            assert duplicates_info['records'].empty, (
                f"No duplicate rows expected, but duplicate records DataFrame is not empty. "
                f"When no duplicates exist, the records should be empty. "
                f"This violates the data quality problem reporting property."
            )
        
        # 3. Check type consistency reporting
        type_issues = quality_report['type_consistency']
        
        # Verify that columns with mixed types are reported
        for col_name in expected_type_issue_columns:
            if col_name in df.columns:
                # Check if this column actually has mixed types
                series = df[col_name].dropna()
                if len(series) > 0:
                    types_found = set(type(val).__name__ for val in series)
                    if len(types_found) > 1:
                        assert col_name in type_issues, (
                            f"Column '{col_name}' has mixed types {types_found}, "
                            f"but is not reported in type consistency issues. "
                            f"Type inconsistency problems should be reported. "
                            f"This violates the data quality problem reporting property."
                        )
                        
                        # Verify the issue description mentions mixed types
                        issues = type_issues[col_name]
                        assert any('Mixed data types found' in issue for issue in issues), (
                            f"Column '{col_name}' is reported in type issues, "
                            f"but the issue description doesn't mention mixed types: {issues}. "
                            f"Type inconsistency problems should be clearly described. "
                            f"This violates the data quality problem reporting property."
                        )
        
        # 4. Check that the report structure contains all required sections
        required_sections = ['completeness', 'duplicates', 'type_consistency', 'summary']
        for section in required_sections:
            assert section in quality_report, (
                f"Quality report is missing required section '{section}'. "
                f"All quality problem types should be reported. "
                f"This violates the data quality problem reporting property."
            )
        
        # 5. Check summary information
        summary = quality_report['summary']
        assert summary['total_records'] == len(df), (
            f"Summary reports {summary['total_records']} total records, "
            f"but DataFrame has {len(df)} records. "
            f"Summary information should accurately reflect the dataset. "
            f"This violates the data quality problem reporting property."
        )
        
        assert summary['total_features'] == len(df.columns), (
            f"Summary reports {summary['total_features']} total features, "
            f"but DataFrame has {len(df.columns)} columns. "
            f"Summary information should accurately reflect the dataset. "
            f"This violates the data quality problem reporting property."
        )
        
        # 6. Check that quality score reflects the presence of issues
        quality_score = summary['quality_score']
        assert 0 <= quality_score <= 100, (
            f"Quality score {quality_score} is outside valid range [0, 100]. "
            f"Quality score should be a percentage. "
            f"This violates the data quality problem reporting property."
        )
        
        # If there are any quality issues, score should be less than 100
        has_missing = any(comp < 100.0 for comp in completeness.values()) if completeness else False
        has_duplicates = duplicates_info['count'] > 0
        has_type_issues = len(type_issues) > 0
        
        if has_missing or has_duplicates or has_type_issues:
            assert quality_score < 100.0, (
                f"Dataset has quality issues (missing: {has_missing}, "
                f"duplicates: {has_duplicates}, type issues: {has_type_issues}), "
                f"but quality score is {quality_score}. "
                f"Quality score should reflect the presence of problems. "
                f"This violates the data quality problem reporting property."
            )