"""Unit tests for ReportGenerator."""

import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import pandas as pd
import numpy as np

from nasa_pcoe_eda.reporting.generator import ReportGenerator
from nasa_pcoe_eda.models import (
    AnalysisResults,
    DatasetMetadata,
    Stats,
    MissingValueReport,
    OutlierSummary,
    TrendReport,
    ScalingRecommendation
)
from nasa_pcoe_eda.exceptions import AnalysisError


class TestReportGenerator:
    """Test cases for ReportGenerator."""

    def setup_method(self):
        """Set up test fixtures."""
        self.generator = ReportGenerator()

    def test_init(self):
        """Test ReportGenerator initialization."""
        generator = ReportGenerator()
        assert generator.env is not None
        assert hasattr(generator.env, 'get_template')

    def test_generate_report_html_success(self):
        """Test successful HTML report generation."""
        analysis_results = self._create_mock_analysis_results()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_report.html"
            
            result_path = self.generator.generate_report(
                analysis_results, output_path, format="html"
            )
            
            assert result_path == output_path
            assert output_path.exists()
            
            # Check that file contains expected content
            content = output_path.read_text(encoding='utf-8')
            assert "NASA PCOE データセット探索的データ分析レポート" in content
            assert "データセット概要" in content
            assert "統計情報" in content
            assert "推奨事項" in content

    def test_generate_report_markdown_success(self):
        """Test successful Markdown report generation."""
        analysis_results = self._create_mock_analysis_results()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_report.md"
            
            result_path = self.generator.generate_report(
                analysis_results, output_path, format="markdown"
            )
            
            assert result_path == output_path
            assert output_path.exists()
            
            # Check that file contains expected content
            content = output_path.read_text(encoding='utf-8')
            assert "# NASA PCOE データセット探索的データ分析レポート" in content
            assert "## データセット概要" in content
            assert "## 統計情報" in content
            assert "## 推奨事項" in content

    def test_generate_report_invalid_analysis_results(self):
        """Test report generation with invalid analysis results."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_report.html"
            
            with pytest.raises(AnalysisError, match="Invalid analysis results provided"):
                self.generator.generate_report("invalid", output_path)

    def test_generate_report_unsupported_format(self):
        """Test report generation with unsupported format."""
        analysis_results = self._create_mock_analysis_results()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_report.pdf"
            
            with pytest.raises(AnalysisError, match="Unsupported format: pdf"):
                self.generator.generate_report(
                    analysis_results, output_path, format="pdf"
                )

    def test_generate_report_creates_output_directory(self):
        """Test that report generation creates output directory if it doesn't exist."""
        analysis_results = self._create_mock_analysis_results()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "subdir" / "test_report.html"
            
            result_path = self.generator.generate_report(
                analysis_results, output_path, format="html"
            )
            
            assert result_path == output_path
            assert output_path.exists()
            assert output_path.parent.exists()

    @patch('nasa_pcoe_eda.reporting.generator.open')
    def test_generate_report_file_write_error(self, mock_open):
        """Test report generation with file write error."""
        mock_open.side_effect = IOError("Permission denied")
        analysis_results = self._create_mock_analysis_results()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_report.html"
            
            with pytest.raises(AnalysisError, match="Failed to generate report"):
                self.generator.generate_report(analysis_results, output_path)

    def test_create_summary_section_valid_metadata(self):
        """Test summary section creation with valid metadata."""
        metadata = DatasetMetadata(
            n_records=1000,
            n_features=10,
            feature_names=['feature1', 'feature2', 'feature3'],
            data_types={'feature1': 'float64', 'feature2': 'int64', 'feature3': 'object'},
            memory_usage=1024 * 1024,  # 1 MB
            date_range=(datetime(2023, 1, 1), datetime(2023, 12, 31))
        )
        
        summary = self.generator.create_summary_section(metadata)
        
        assert isinstance(summary, str)
        assert "1,000" in summary  # Formatted record count
        assert "10" in summary  # Feature count
        assert "1.0 MB" in summary  # Memory usage
        assert "2023-01-01" in summary  # Start date
        assert "2023-12-31" in summary  # End date
        assert "feature1" in summary  # Feature names
        assert "float64" in summary  # Data types

    def test_create_summary_section_no_date_range(self):
        """Test summary section creation without date range."""
        metadata = DatasetMetadata(
            n_records=500,
            n_features=5,
            feature_names=['feature1', 'feature2'],
            data_types={'feature1': 'float64', 'feature2': 'int64'},
            memory_usage=512 * 1024,  # 0.5 MB
            date_range=None
        )
        
        summary = self.generator.create_summary_section(metadata)
        
        assert isinstance(summary, str)
        assert "500" in summary
        assert "5" in summary
        assert "0.5 MB" in summary
        # Should not contain date information
        assert "2023" not in summary

    def test_create_summary_section_many_features(self):
        """Test summary section creation with many features (should limit display)."""
        feature_names = [f'feature_{i}' for i in range(30)]
        data_types = {name: 'float64' for name in feature_names}
        
        metadata = DatasetMetadata(
            n_records=1000,
            n_features=30,
            feature_names=feature_names,
            data_types=data_types,
            memory_usage=1024 * 1024,
            date_range=None
        )
        
        summary = self.generator.create_summary_section(metadata)
        
        assert isinstance(summary, str)
        assert "feature_0" in summary  # First feature should be shown
        assert "feature_19" in summary  # 20th feature should be shown
        assert "他 10 個の特徴量" in summary  # Should show remaining count

    def test_create_summary_section_invalid_metadata(self):
        """Test summary section creation with invalid metadata."""
        summary = self.generator.create_summary_section("invalid")
        
        assert isinstance(summary, str)
        assert "メタデータが利用できません" in summary

    def test_create_statistics_section_valid_stats(self):
        """Test statistics section creation with valid statistics."""
        stats = {
            'feature1': Stats(1.5, 1.0, 0.5, 0.0, 3.0, 0.5, 2.0),
            'feature2': Stats(10.0, 9.5, 2.0, 5.0, 15.0, 8.0, 12.0)
        }
        
        section = self.generator.create_statistics_section(stats)
        
        assert isinstance(section, str)
        assert "feature1" in section
        assert "feature2" in section
        assert "1.5000" in section  # Mean of feature1
        assert "10.0000" in section  # Mean of feature2
        assert "stats-table" in section  # HTML table class

    def test_create_statistics_section_empty_stats(self):
        """Test statistics section creation with empty statistics."""
        section = self.generator.create_statistics_section({})
        
        assert isinstance(section, str)
        assert "統計情報が利用できません" in section

    def test_create_recommendations_section_with_rul_features(self):
        """Test recommendations section creation with RUL features."""
        rul_features = [
            ('feature1', 0.85),
            ('feature2', 0.72),
            ('feature3', 0.68)
        ]
        fault_features = []
        preprocessing_recommendations = {}
        
        section = self.generator.create_recommendations_section(
            rul_features, fault_features, preprocessing_recommendations
        )
        
        assert isinstance(section, str)
        assert "RUL予測モデル構築の推奨事項" in section
        assert "feature1" in section
        assert "0.850" in section  # Correlation score
        assert "回帰モデル" in section

    def test_create_recommendations_section_with_fault_features(self):
        """Test recommendations section creation with fault features."""
        rul_features = []
        fault_features = ['feature1', 'feature2', 'feature3']
        preprocessing_recommendations = {}
        
        section = self.generator.create_recommendations_section(
            rul_features, fault_features, preprocessing_recommendations
        )
        
        assert isinstance(section, str)
        assert "故障診断モデル構築の推奨事項" in section
        assert "feature1" in section
        assert "分類モデル" in section

    def test_create_recommendations_section_with_preprocessing(self):
        """Test recommendations section creation with preprocessing recommendations."""
        rul_features = []
        fault_features = []
        preprocessing_recommendations = {
            'missing_value_strategies': {
                'feature1': 'mean_imputation',
                'feature2': 'median_imputation'
            },
            'scaling_recommendation': {
                'method': 'standard_scaling',
                'reason': 'Large scale differences detected'
            }
        }
        
        section = self.generator.create_recommendations_section(
            rul_features, fault_features, preprocessing_recommendations
        )
        
        assert isinstance(section, str)
        assert "データ前処理の推奨事項" in section
        assert "欠損値処理" in section
        assert "mean_imputation" in section
        assert "スケーリング" in section
        assert "standard_scaling" in section

    def test_create_recommendations_section_empty(self):
        """Test recommendations section creation with empty inputs."""
        section = self.generator.create_recommendations_section([], [], {})
        
        assert isinstance(section, str)
        assert "一般的な推奨事項" in section  # Should always include general recommendations

    def test_create_quality_section_with_missing_values(self):
        """Test quality section creation with missing values."""
        missing_values = MissingValueReport(
            missing_counts={'feature1': 10, 'feature2': 5},
            missing_percentages={'feature1': 10.0, 'feature2': 5.0},
            total_missing=15
        )
        
        section = self.generator._create_quality_section(missing_values)
        
        assert isinstance(section, str)
        assert "15" in section  # Total missing
        assert "feature1" in section
        assert "10" in section  # Missing count for feature1
        assert "10.0%" in section  # Missing percentage

    def test_create_quality_section_no_missing_values(self):
        """Test quality section creation with no missing values."""
        section = self.generator._create_quality_section(None)
        
        assert isinstance(section, str)
        assert "データ品質情報が利用できません" in section

    def test_create_correlations_section_with_high_correlations(self):
        """Test correlations section creation with high correlations."""
        # Create correlation matrix with high correlations
        corr_data = {
            'feature1': [1.0, 0.85, 0.3],
            'feature2': [0.85, 1.0, 0.2],
            'feature3': [0.3, 0.2, 1.0]
        }
        correlation_matrix = pd.DataFrame(
            corr_data, 
            index=['feature1', 'feature2', 'feature3']
        )
        
        section = self.generator._create_correlations_section(correlation_matrix)
        
        assert isinstance(section, str)
        assert "1" in section  # Number of high correlation pairs
        assert "feature1" in section
        assert "feature2" in section
        assert "0.850" in section  # Correlation value

    def test_create_correlations_section_empty(self):
        """Test correlations section creation with empty correlation matrix."""
        section = self.generator._create_correlations_section(None)
        
        assert isinstance(section, str)
        assert "相関情報が利用できません" in section

    def test_create_outliers_section_with_outliers(self):
        """Test outliers section creation with outliers."""
        outliers = OutlierSummary(
            outlier_counts={'feature1': 5, 'feature2': 3},
            outlier_percentages={'feature1': 5.0, 'feature2': 3.0},
            outlier_indices={'feature1': np.array([1, 2, 3, 4, 5]), 'feature2': np.array([1, 2, 3])}
        )
        
        section = self.generator._create_outliers_section(outliers)
        
        assert isinstance(section, str)
        assert "8" in section  # Total outliers (5 + 3)
        assert "feature1" in section
        assert "5" in section  # Outlier count for feature1
        assert "5.0%" in section  # Outlier percentage

    def test_create_outliers_section_no_outliers(self):
        """Test outliers section creation with no outliers."""
        section = self.generator._create_outliers_section(None)
        
        assert isinstance(section, str)
        assert "外れ値情報が利用できません" in section

    def test_create_timeseries_section_with_trends(self):
        """Test time series section creation with trends."""
        trends = TrendReport(
            trends={'feature1': {'slope': 0.5, 'r_squared': 0.8}},
            trend_directions={'feature1': 'increasing'}
        )
        
        section = self.generator._create_timeseries_section(trends)
        
        assert isinstance(section, str)
        assert "時系列トレンド分析結果" in section
        assert "feature1" in section
        assert "increasing" in section

    def test_create_timeseries_section_no_trends(self):
        """Test time series section creation with no trends."""
        section = self.generator._create_timeseries_section(None)
        
        assert isinstance(section, str)
        assert "時系列情報が利用できません" in section

    def test_create_visualizations_section_with_paths(self):
        """Test visualizations section creation with visualization paths."""
        visualization_paths = [
            Path("output/figures/histogram.png"),
            Path("output/figures/correlation_heatmap.png")
        ]
        
        section = self.generator._create_visualizations_section(visualization_paths)
        
        assert isinstance(section, str)
        assert "2" in section  # Number of visualizations
        assert "histogram.png" in section
        assert "correlation_heatmap.png" in section

    def test_create_visualizations_section_empty(self):
        """Test visualizations section creation with no visualizations."""
        section = self.generator._create_visualizations_section([])
        
        assert isinstance(section, str)
        assert "可視化ファイルが利用できません" in section

    def test_create_rul_analysis_section_with_features(self):
        """Test RUL analysis section creation with features."""
        rul_features = [
            ('feature1', 0.85),
            ('feature2', 0.72),
            ('feature3', 0.68)
        ]
        
        section = self.generator._create_rul_analysis_section(rul_features)
        
        assert isinstance(section, str)
        assert "3" in section  # Number of features
        assert "feature1" in section
        assert "0.8500" in section  # Correlation score
        assert "上位RUL関連特徴量" in section

    def test_create_rul_analysis_section_empty(self):
        """Test RUL analysis section creation with no features."""
        section = self.generator._create_rul_analysis_section([])
        
        assert isinstance(section, str)
        assert "RUL特徴量分析結果が利用できません" in section

    def test_create_fault_analysis_section_with_features(self):
        """Test fault analysis section creation with features."""
        fault_features = ['feature1', 'feature2', 'feature3']
        
        section = self.generator._create_fault_analysis_section(fault_features)
        
        assert isinstance(section, str)
        assert "3" in section  # Number of features
        assert "feature1" in section
        assert "故障識別特徴量" in section

    def test_create_fault_analysis_section_empty(self):
        """Test fault analysis section creation with no features."""
        section = self.generator._create_fault_analysis_section([])
        
        assert isinstance(section, str)
        assert "故障レベル分析結果が利用できません" in section

    def test_html_report_contains_all_sections(self):
        """Test that HTML report contains all expected sections."""
        analysis_results = self._create_comprehensive_analysis_results()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "comprehensive_report.html"
            
            self.generator.generate_report(analysis_results, output_path, format="html")
            
            content = output_path.read_text(encoding='utf-8')
            
            # Check all sections are present
            assert "データセット概要" in content
            assert "統計情報" in content
            assert "データ品質" in content
            assert "相関分析" in content
            assert "外れ値分析" in content
            assert "時系列分析" in content
            assert "可視化" in content
            assert "RUL特徴量分析" in content
            assert "故障レベル分析" in content
            assert "推奨事項" in content

    def test_markdown_report_structure(self):
        """Test that Markdown report has correct structure."""
        analysis_results = self._create_mock_analysis_results()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "test_report.md"
            
            self.generator.generate_report(analysis_results, output_path, format="markdown")
            
            content = output_path.read_text(encoding='utf-8')
            
            # Check markdown structure
            assert content.startswith("# NASA PCOE データセット探索的データ分析レポート")
            assert "## 目次" in content
            assert "## データセット概要" in content
            assert "## 統計情報" in content
            assert "## 推奨事項" in content

    def test_report_generation_with_unicode_content(self):
        """Test report generation with Unicode content."""
        # Create analysis results with Japanese feature names
        metadata = DatasetMetadata(
            n_records=100,
            n_features=2,
            feature_names=['温度', '湿度'],
            data_types={'温度': 'float64', '湿度': 'float64'},
            memory_usage=1024.0,
            date_range=None
        )
        
        stats = {
            '温度': Stats(25.0, 24.5, 5.0, 10.0, 40.0, 20.0, 30.0),
            '湿度': Stats(60.0, 58.0, 10.0, 30.0, 90.0, 50.0, 70.0)
        }
        
        analysis_results = AnalysisResults(
            metadata=metadata,
            statistics=stats,
            missing_values=MissingValueReport({}, {}, 0),
            correlation_matrix=pd.DataFrame(),
            outliers=None,
            time_series_trends=None,
            rul_features=[],
            fault_features=[],
            preprocessing_recommendations={},
            visualization_paths=[]
        )
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "unicode_report.html"
            
            result_path = self.generator.generate_report(analysis_results, output_path)
            
            assert result_path.exists()
            content = output_path.read_text(encoding='utf-8')
            assert "温度" in content
            assert "湿度" in content

    def _create_mock_analysis_results(self):
        """Create mock AnalysisResults for testing."""
        metadata = DatasetMetadata(
            n_records=100,
            n_features=3,
            feature_names=['feature1', 'feature2', 'feature3'],
            data_types={'feature1': 'float64', 'feature2': 'float64', 'feature3': 'float64'},
            memory_usage=1024.0,
            date_range=None
        )
        
        stats = {
            'feature1': Stats(1.0, 1.0, 1.0, 0.0, 2.0, 0.5, 1.5),
            'feature2': Stats(2.0, 2.0, 1.0, 1.0, 3.0, 1.5, 2.5)
        }
        
        missing_values = MissingValueReport(
            missing_counts={'feature1': 0, 'feature2': 0},
            missing_percentages={'feature1': 0.0, 'feature2': 0.0},
            total_missing=0
        )
        
        return AnalysisResults(
            metadata=metadata,
            statistics=stats,
            missing_values=missing_values,
            correlation_matrix=pd.DataFrame(),
            outliers=None,
            time_series_trends=None,
            rul_features=[],
            fault_features=[],
            preprocessing_recommendations={},
            visualization_paths=[]
        )

    def _create_comprehensive_analysis_results(self):
        """Create comprehensive AnalysisResults with all sections populated."""
        metadata = DatasetMetadata(
            n_records=1000,
            n_features=5,
            feature_names=['feature1', 'feature2', 'feature3', 'feature4', 'feature5'],
            data_types={'feature1': 'float64', 'feature2': 'float64', 'feature3': 'float64', 
                       'feature4': 'float64', 'feature5': 'float64'},
            memory_usage=5120.0,
            date_range=(datetime(2023, 1, 1), datetime(2023, 12, 31))
        )
        
        stats = {
            'feature1': Stats(1.0, 1.0, 1.0, 0.0, 2.0, 0.5, 1.5),
            'feature2': Stats(2.0, 2.0, 1.0, 1.0, 3.0, 1.5, 2.5),
            'feature3': Stats(3.0, 3.0, 1.0, 2.0, 4.0, 2.5, 3.5)
        }
        
        missing_values = MissingValueReport(
            missing_counts={'feature1': 10, 'feature2': 5},
            missing_percentages={'feature1': 1.0, 'feature2': 0.5},
            total_missing=15
        )
        
        # Create correlation matrix with high correlations
        corr_data = {
            'feature1': [1.0, 0.85, 0.3],
            'feature2': [0.85, 1.0, 0.2],
            'feature3': [0.3, 0.2, 1.0]
        }
        correlation_matrix = pd.DataFrame(
            corr_data, 
            index=['feature1', 'feature2', 'feature3']
        )
        
        outliers = OutlierSummary(
            outlier_counts={'feature1': 5, 'feature2': 3},
            outlier_percentages={'feature1': 0.5, 'feature2': 0.3},
            outlier_indices={'feature1': np.array([1, 2, 3, 4, 5]), 'feature2': np.array([1, 2, 3])}
        )
        
        trends = TrendReport(
            trends={'feature1': {'slope': 0.5, 'r_squared': 0.8}},
            trend_directions={'feature1': 'increasing'}
        )
        
        rul_features = [
            ('feature1', 0.85),
            ('feature2', 0.72)
        ]
        
        fault_features = ['feature3', 'feature4']
        
        preprocessing_recommendations = {
            'missing_value_strategies': {
                'feature1': 'mean_imputation',
                'feature2': 'median_imputation'
            },
            'scaling_recommendation': {
                'method': 'standard_scaling',
                'reason': 'Large scale differences detected'
            }
        }
        
        visualization_paths = [
            Path("output/figures/histogram.png"),
            Path("output/figures/correlation_heatmap.png")
        ]
        
        return AnalysisResults(
            metadata=metadata,
            statistics=stats,
            missing_values=missing_values,
            correlation_matrix=correlation_matrix,
            outliers=outliers,
            time_series_trends=trends,
            rul_features=rul_features,
            fault_features=fault_features,
            preprocessing_recommendations=preprocessing_recommendations,
            visualization_paths=visualization_paths
        )