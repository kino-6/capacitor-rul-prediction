"""Property-based tests for ReportGenerator."""

import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings

from nasa_pcoe_eda.reporting.generator import ReportGenerator
from nasa_pcoe_eda.models import (
    AnalysisResults,
    DatasetMetadata,
    Stats,
    MissingValueReport,
    OutlierSummary,
    TrendReport
)


class TestReportGeneratorProperties:
    """Property-based tests for ReportGenerator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = ReportGenerator()

    # Feature: nasa-pcoe-eda, Property 14: レポート生成の完全性
    @given(st.data())
    @settings(max_examples=100)
    def test_report_generation_completeness(self, data):
        """
        任意の分析結果に対して、生成されるレポートには、統計、可視化、推奨事項の全てのセクションが含まれる
        
        Property 14: Report Generation Completeness
        For any analysis results, the generated report should include all sections: 
        statistics, visualizations, and recommendations.
        """
        # Generate random analysis results
        analysis_results = self._generate_analysis_results(data)
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_file:
            output_path = Path(tmp_file.name)
        
        try:
            # Generate report
            generated_path = self.generator.generate_report(
                analysis_results=analysis_results,
                output_path=output_path,
                format='html'
            )
            
            # Verify that the report file was created
            assert generated_path.exists(), (
                f"Report file was not created at {generated_path}. "
                f"The generate_report method should create a file at the specified path. "
                f"This violates the report generation completeness property."
            )
            
            # Verify that the returned path matches the input path
            assert generated_path == output_path, (
                f"Generated report path ({generated_path}) does not match expected path ({output_path}). "
                f"The method should return the exact path where the report was saved."
            )
            
            # Read the generated report content
            with open(generated_path, 'r', encoding='utf-8') as f:
                report_content = f.read()
            
            # Verify that the report is not empty
            assert len(report_content.strip()) > 0, (
                f"Generated report is empty. "
                f"The report should contain meaningful content. "
                f"This violates the report generation completeness property."
            )
            
            # Verify that all required sections are present in the report
            required_sections = [
                'summary_section',
                'statistics_section', 
                'recommendations_section'
            ]
            
            for section in required_sections:
                # Check if section content exists in the report
                # The template should include these sections
                section_found = (
                    section in report_content or
                    section.replace('_', ' ').title() in report_content or
                    self._check_section_content_exists(report_content, section)
                )
                
                assert section_found, (
                    f"Required section '{section}' is missing from the generated report. "
                    f"The report must include all essential sections: {required_sections}. "
                    f"This violates the report generation completeness property."
                )
            
            # Verify that the report contains actual data from the analysis results
            # Check for metadata information (may be HTML-escaped)
            records_found = (
                str(analysis_results.metadata.n_records) in report_content or
                f"{analysis_results.metadata.n_records:,}" in report_content or
                f"&gt;{analysis_results.metadata.n_records}&lt;" in report_content
            )
            assert records_found, (
                f"Report does not contain the number of records ({analysis_results.metadata.n_records}). "
                f"The report should include key metadata information. "
                f"This violates the report generation completeness property."
            )
            
            features_found = (
                str(analysis_results.metadata.n_features) in report_content or
                f"&gt;{analysis_results.metadata.n_features}&lt;" in report_content
            )
            assert features_found, (
                f"Report does not contain the number of features ({analysis_results.metadata.n_features}). "
                f"The report should include key metadata information. "
                f"This violates the report generation completeness property."
            )
            
            # Check for statistics information if available
            if analysis_results.statistics:
                # At least one feature name should appear in the report
                feature_found = any(
                    feature_name in report_content 
                    for feature_name in analysis_results.statistics.keys()
                )
                assert feature_found, (
                    f"Report does not contain any feature names from statistics. "
                    f"Available features: {list(analysis_results.statistics.keys())}. "
                    f"The report should include statistical information. "
                    f"This violates the report generation completeness property."
                )
            
            # Check for recommendations if available
            if analysis_results.rul_features or analysis_results.fault_features:
                # Should contain some recommendation content
                recommendation_indicators = [
                    '推奨', 'recommend', 'RUL', '故障', 'fault', '特徴量', 'feature'
                ]
                recommendation_found = any(
                    indicator in report_content 
                    for indicator in recommendation_indicators
                )
                assert recommendation_found, (
                    f"Report does not contain recommendation content. "
                    f"The report should include recommendations when RUL or fault features are available. "
                    f"This violates the report generation completeness property."
                )
            
            # Verify that the report has proper HTML structure
            assert '<html' in report_content.lower(), (
                f"Report does not contain proper HTML structure. "
                f"HTML reports should have valid HTML tags. "
                f"This violates the report generation completeness property."
            )
            
            assert '</html>' in report_content.lower(), (
                f"Report does not have proper HTML closing tag. "
                f"HTML reports should be properly structured. "
                f"This violates the report generation completeness property."
            )
            
            # Verify that the report contains generation timestamp
            # The template should include generation time
            timestamp_found = (
                '生成日時' in report_content or
                'generation_time' in report_content or
                datetime.now().strftime('%Y') in report_content
            )
            assert timestamp_found, (
                f"Report does not contain generation timestamp information. "
                f"The report should include when it was generated. "
                f"This violates the report generation completeness property."
            )
            
        finally:
            # Clean up temporary file
            if output_path.exists():
                output_path.unlink()

    def _generate_analysis_results(self, data) -> AnalysisResults:
        """Generate random but valid AnalysisResults for testing."""
        # Generate metadata
        n_features = data.draw(st.integers(min_value=1, max_value=10))
        n_records = data.draw(st.integers(min_value=10, max_value=1000))
        
        feature_names = [f"feature_{i}" for i in range(n_features)]
        data_types = {name: data.draw(st.sampled_from(['float64', 'int64', 'object'])) 
                     for name in feature_names}
        
        metadata = DatasetMetadata(
            n_records=n_records,
            n_features=n_features,
            feature_names=feature_names,
            data_types=data_types,
            memory_usage=data.draw(st.floats(min_value=1000, max_value=100000)),
            date_range=None
        )
        
        # Generate statistics
        statistics = {}
        for feature in feature_names:
            if data_types[feature] in ['float64', 'int64']:
                stats = Stats(
                    mean=data.draw(st.floats(min_value=-100, max_value=100, allow_nan=False)),
                    median=data.draw(st.floats(min_value=-100, max_value=100, allow_nan=False)),
                    std=data.draw(st.floats(min_value=0, max_value=50, allow_nan=False)),
                    min=data.draw(st.floats(min_value=-200, max_value=0, allow_nan=False)),
                    max=data.draw(st.floats(min_value=0, max_value=200, allow_nan=False)),
                    q25=data.draw(st.floats(min_value=-50, max_value=50, allow_nan=False)),
                    q75=data.draw(st.floats(min_value=-50, max_value=50, allow_nan=False))
                )
                statistics[feature] = stats
        
        # Generate missing values report
        missing_counts = {name: data.draw(st.integers(min_value=0, max_value=n_records//4)) 
                         for name in feature_names}
        missing_percentages = {name: (count / n_records) * 100 
                              for name, count in missing_counts.items()}
        total_missing = sum(missing_counts.values())
        
        missing_values = MissingValueReport(
            missing_counts=missing_counts,
            missing_percentages=missing_percentages,
            total_missing=total_missing
        )
        
        # Generate correlation matrix
        numeric_features = [name for name in feature_names if data_types[name] in ['float64', 'int64']]
        if numeric_features:
            # Create a simple correlation matrix
            corr_data = np.eye(len(numeric_features))  # Identity matrix for simplicity
            correlation_matrix = pd.DataFrame(
                corr_data, 
                index=numeric_features, 
                columns=numeric_features
            )
        else:
            correlation_matrix = pd.DataFrame()
        
        # Generate outliers summary
        outlier_counts = {name: data.draw(st.integers(min_value=0, max_value=n_records//10)) 
                         for name in feature_names if data_types[name] in ['float64', 'int64']}
        outlier_percentages = {name: (count / n_records) * 100 
                              for name, count in outlier_counts.items()}
        outlier_indices = {name: np.array(data.draw(st.lists(
            st.integers(min_value=0, max_value=n_records-1),
            min_size=0,
            max_size=min(count, 10)
        ))) for name, count in outlier_counts.items()}
        
        outliers = OutlierSummary(
            outlier_counts=outlier_counts,
            outlier_percentages=outlier_percentages,
            outlier_indices=outlier_indices
        )
        
        # Generate time series trends (optional)
        time_series_trends = None
        if data.draw(st.booleans()):
            trends = {name: {'slope': data.draw(st.floats(min_value=-1, max_value=1, allow_nan=False))} 
                     for name in feature_names[:2]}  # Only first 2 features
            trend_directions = {name: data.draw(st.sampled_from(['increasing', 'decreasing', 'stable'])) 
                               for name in feature_names[:2]}
            time_series_trends = TrendReport(
                trends=trends,
                trend_directions=trend_directions
            )
        
        # Generate RUL features
        rul_features = []
        if numeric_features:
            num_rul_features = min(len(numeric_features), data.draw(st.integers(min_value=0, max_value=5)))
            selected_features = data.draw(st.lists(
                st.sampled_from(numeric_features),
                min_size=num_rul_features,
                max_size=num_rul_features,
                unique=True
            ))
            rul_features = [(feature, data.draw(st.floats(min_value=0, max_value=1, allow_nan=False))) 
                           for feature in selected_features]
        
        # Generate fault features
        fault_features = []
        if feature_names:
            num_fault_features = min(len(feature_names), data.draw(st.integers(min_value=0, max_value=5)))
            fault_features = data.draw(st.lists(
                st.sampled_from(feature_names),
                min_size=num_fault_features,
                max_size=num_fault_features,
                unique=True
            ))
        
        # Generate preprocessing recommendations
        preprocessing_recommendations = {
            'missing_value_strategies': {name: data.draw(st.sampled_from(['mean', 'median', 'mode', 'drop', 'no_action'])) 
                                       for name in feature_names},
            'scaling_recommendation': {
                'method': data.draw(st.sampled_from(['standard', 'minmax', 'robust', 'none'])),
                'reason': 'Generated for testing'
            }
        }
        
        # Generate visualization paths
        num_visualizations = data.draw(st.integers(min_value=0, max_value=5))
        visualization_paths = [Path(f"/tmp/viz_{i}.png") for i in range(num_visualizations)]
        
        return AnalysisResults(
            metadata=metadata,
            statistics=statistics,
            missing_values=missing_values,
            correlation_matrix=correlation_matrix,
            outliers=outliers,
            time_series_trends=time_series_trends,
            rul_features=rul_features,
            fault_features=fault_features,
            preprocessing_recommendations=preprocessing_recommendations,
            visualization_paths=visualization_paths
        )

    # Feature: nasa-pcoe-eda, Property 15: レポート出力通知
    @given(st.data())
    @settings(max_examples=100)
    def test_report_output_notification(self, data):
        """
        任意のレポート生成操作に対して、システムは生成されたレポートファイルのパスを返す
        
        Property 15: Report Output Notification
        For any report generation operation, the system should return the path 
        of the generated report file.
        """
        # Generate random analysis results
        analysis_results = self._generate_analysis_results(data)
        
        # Generate random output path
        output_dir = data.draw(st.text(
            alphabet='abcdefghijklmnopqrstuvwxyz0123456789_-',
            min_size=1,
            max_size=20
        ))
        output_filename = data.draw(st.text(
            alphabet='abcdefghijklmnopqrstuvwxyz0123456789_-',
            min_size=1,
            max_size=20
        ))
        
        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_path = temp_path / output_dir / f"{output_filename}.html"
            
            # Test HTML format
            returned_path = self.generator.generate_report(
                analysis_results=analysis_results,
                output_path=output_path,
                format='html'
            )
            
            # Verify that the method returns a Path object
            assert isinstance(returned_path, Path), (
                f"generate_report should return a Path object, but returned {type(returned_path)}. "
                f"The system must notify the user of the output location by returning the file path. "
                f"This violates the report output notification property."
            )
            
            # Verify that the returned path matches the input path exactly
            assert returned_path == output_path, (
                f"Returned path ({returned_path}) does not match the expected output path ({output_path}). "
                f"The system should return the exact path where the report was saved. "
                f"This violates the report output notification property."
            )
            
            # Verify that the returned path points to an existing file
            assert returned_path.exists(), (
                f"The returned path ({returned_path}) does not point to an existing file. "
                f"The system should only return paths to successfully created files. "
                f"This violates the report output notification property."
            )
            
            # Verify that the returned path is absolute (for clarity)
            assert returned_path.is_absolute(), (
                f"The returned path ({returned_path}) is not absolute. "
                f"For clear notification, the system should return absolute paths. "
                f"This violates the report output notification property."
            )
            
            # Test with different formats
            markdown_path = temp_path / output_dir / f"{output_filename}.md"
            returned_md_path = self.generator.generate_report(
                analysis_results=analysis_results,
                output_path=markdown_path,
                format='markdown'
            )
            
            # Verify markdown format also returns correct path
            assert isinstance(returned_md_path, Path), (
                f"generate_report should return a Path object for markdown format, but returned {type(returned_md_path)}. "
                f"The system must notify the user of the output location regardless of format. "
                f"This violates the report output notification property."
            )
            
            assert returned_md_path == markdown_path, (
                f"Returned markdown path ({returned_md_path}) does not match expected path ({markdown_path}). "
                f"The system should return the exact path for all supported formats. "
                f"This violates the report output notification property."
            )
            
            assert returned_md_path.exists(), (
                f"The returned markdown path ({returned_md_path}) does not point to an existing file. "
                f"The system should only return paths to successfully created files. "
                f"This violates the report output notification property."
            )
            
            # Test with nested directory structure
            nested_path = temp_path / "level1" / "level2" / "level3" / f"{output_filename}_nested.html"
            returned_nested_path = self.generator.generate_report(
                analysis_results=analysis_results,
                output_path=nested_path,
                format='html'
            )
            
            # Verify nested directory creation and path return
            assert returned_nested_path == nested_path, (
                f"Returned nested path ({returned_nested_path}) does not match expected path ({nested_path}). "
                f"The system should handle nested directories and return correct paths. "
                f"This violates the report output notification property."
            )
            
            assert returned_nested_path.exists(), (
                f"The returned nested path ({returned_nested_path}) does not point to an existing file. "
                f"The system should create necessary directories and return valid paths. "
                f"This violates the report output notification property."
            )
            
            # Verify that the parent directories were created
            assert returned_nested_path.parent.exists(), (
                f"Parent directory ({returned_nested_path.parent}) was not created. "
                f"The system should create necessary parent directories. "
                f"This violates the report output notification property."
            )
            
            # Test with existing file (overwrite scenario)
            existing_content = "existing content"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(existing_content)
            
            returned_overwrite_path = self.generator.generate_report(
                analysis_results=analysis_results,
                output_path=output_path,
                format='html'
            )
            
            # Verify overwrite scenario
            assert returned_overwrite_path == output_path, (
                f"Returned overwrite path ({returned_overwrite_path}) does not match expected path ({output_path}). "
                f"The system should return correct paths even when overwriting existing files. "
                f"This violates the report output notification property."
            )
            
            # Verify that the file was actually overwritten (content should be different)
            with open(returned_overwrite_path, 'r', encoding='utf-8') as f:
                new_content = f.read()
            
            assert new_content != existing_content, (
                f"File content was not updated during overwrite. "
                f"The system should properly overwrite existing files and return the correct path. "
                f"This violates the report output notification property."
            )
            
            # Verify that the new content is a proper report (not empty)
            assert len(new_content.strip()) > len(existing_content), (
                f"New report content is not longer than the existing content. "
                f"The system should generate proper report content when overwriting. "
                f"This violates the report output notification property."
            )

    def _check_section_content_exists(self, report_content: str, section_name: str) -> bool:
        """Check if section-specific content exists in the report."""
        section_indicators = {
            'summary_section': ['レコード数', '特徴量数', 'records', 'features', 'メタデータ'],
            'statistics_section': ['平均値', '中央値', '標準偏差', 'mean', 'median', 'std', '統計'],
            'recommendations_section': ['推奨', 'recommend', 'RUL', '故障', 'fault', '前処理']
        }
        
        indicators = section_indicators.get(section_name, [])
        return any(indicator in report_content for indicator in indicators)