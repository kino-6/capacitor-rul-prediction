"""Report generation module for comprehensive EDA results."""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings

from jinja2 import Environment, FileSystemLoader, select_autoescape
import pandas as pd
import numpy as np

from ..models import (
    AnalysisResults,
    DatasetMetadata,
    Stats,
    MissingValueReport,
    OutlierSummary,
    TrendReport
)
from ..exceptions import AnalysisError


class ReportGenerator:
    """Generator for comprehensive EDA reports."""

    def __init__(self):
        """Initialize the report generator."""
        # Set up Jinja2 environment
        template_dir = Path(__file__).parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(['html', 'xml'])
        )

    def generate_report(
        self,
        analysis_results: AnalysisResults,
        output_path: Path,
        format: str = "html"
    ) -> Path:
        """
        Generate a comprehensive EDA report.
        
        Args:
            analysis_results: Complete analysis results
            output_path: Path where the report should be saved
            format: Report format ('html' or 'markdown')
            
        Returns:
            Path to the generated report file
            
        Raises:
            AnalysisError: If report generation fails
        """
        if not isinstance(analysis_results, AnalysisResults):
            raise AnalysisError("Invalid analysis results provided")
            
        if format not in ['html', 'markdown']:
            raise AnalysisError(f"Unsupported format: {format}")
            
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if format == 'html':
                return self._generate_html_report(analysis_results, output_path)
            else:
                return self._generate_markdown_report(analysis_results, output_path)
                
        except Exception as e:
            raise AnalysisError(f"Failed to generate report: {str(e)}")

    def _generate_html_report(
        self,
        analysis_results: AnalysisResults,
        output_path: Path
    ) -> Path:
        """Generate HTML report using Jinja2 template."""
        template = self.env.get_template('report_template.html')
        
        # Generate all sections
        summary_section = self.create_summary_section(analysis_results.metadata)
        statistics_section = self.create_statistics_section(analysis_results.statistics)
        quality_section = self._create_quality_section(analysis_results.missing_values)
        correlations_section = self._create_correlations_section(analysis_results.correlation_matrix)
        outliers_section = self._create_outliers_section(analysis_results.outliers)
        timeseries_section = self._create_timeseries_section(analysis_results.time_series_trends)
        visualizations_section = self._create_visualizations_section(analysis_results.visualization_paths)
        rul_analysis_section = self._create_rul_analysis_section(analysis_results.rul_features)
        fault_analysis_section = self._create_fault_analysis_section(analysis_results.fault_features)
        recommendations_section = self.create_recommendations_section(
            analysis_results.rul_features,
            analysis_results.fault_features,
            analysis_results.preprocessing_recommendations
        )
        
        # Render template
        html_content = template.render(
            summary_section=summary_section,
            statistics_section=statistics_section,
            quality_section=quality_section,
            correlations_section=correlations_section,
            outliers_section=outliers_section,
            timeseries_section=timeseries_section,
            visualizations_section=visualizations_section,
            rul_analysis_section=rul_analysis_section,
            fault_analysis_section=fault_analysis_section,
            recommendations_section=recommendations_section,
            generation_time=datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")
        )
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        return output_path

    def _generate_markdown_report(
        self,
        analysis_results: AnalysisResults,
        output_path: Path
    ) -> Path:
        """Generate Markdown report."""
        content = []
        
        # Title
        content.append("# NASA PCOE データセット探索的データ分析レポート\n")
        content.append(f"生成日時: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}\n")
        
        # Table of contents
        content.append("## 目次\n")
        content.append("1. [データセット概要](#データセット概要)")
        content.append("2. [統計情報](#統計情報)")
        content.append("3. [データ品質](#データ品質)")
        content.append("4. [相関分析](#相関分析)")
        content.append("5. [外れ値分析](#外れ値分析)")
        content.append("6. [時系列分析](#時系列分析)")
        content.append("7. [可視化](#可視化)")
        content.append("8. [RUL特徴量分析](#rul特徴量分析)")
        content.append("9. [故障レベル分析](#故障レベル分析)")
        content.append("10. [推奨事項](#推奨事項)\n")
        
        # Sections
        content.append("## データセット概要\n")
        content.append(self.create_summary_section(analysis_results.metadata))
        content.append("\n")
        
        content.append("## 統計情報\n")
        content.append(self.create_statistics_section(analysis_results.statistics))
        content.append("\n")
        
        content.append("## 推奨事項\n")
        content.append(self.create_recommendations_section(
            analysis_results.rul_features,
            analysis_results.fault_features,
            analysis_results.preprocessing_recommendations
        ))
        content.append("\n")
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(content))
            
        return output_path

    def create_summary_section(self, metadata: DatasetMetadata) -> str:
        """
        Create dataset summary section.
        
        Args:
            metadata: Dataset metadata
            
        Returns:
            HTML string for summary section
        """
        if not isinstance(metadata, DatasetMetadata):
            return "<p>メタデータが利用できません。</p>"
            
        summary_cards = []
        
        # Records count
        summary_cards.append(f"""
        <div class="summary-card">
            <h4>レコード数</h4>
            <div class="value">{metadata.n_records:,}</div>
        </div>
        """)
        
        # Features count
        summary_cards.append(f"""
        <div class="summary-card">
            <h4>特徴量数</h4>
            <div class="value">{metadata.n_features}</div>
        </div>
        """)
        
        # Memory usage
        memory_mb = metadata.memory_usage / (1024 * 1024)
        summary_cards.append(f"""
        <div class="summary-card">
            <h4>メモリ使用量</h4>
            <div class="value">{memory_mb:.1f} MB</div>
        </div>
        """)
        
        # Date range if available
        if metadata.date_range:
            start_date, end_date = metadata.date_range
            summary_cards.append(f"""
            <div class="summary-card">
                <h4>データ期間</h4>
                <div class="value">{start_date.strftime('%Y-%m-%d')}<br>～<br>{end_date.strftime('%Y-%m-%d')}</div>
            </div>
            """)
        
        summary_grid = f'<div class="summary-grid">{"".join(summary_cards)}</div>'
        
        # Feature list
        feature_list = "<h3>特徴量一覧</h3><ul>"
        for feature in metadata.feature_names[:20]:  # Limit to first 20
            data_type = metadata.data_types.get(feature, "unknown")
            feature_list += f"<li><strong>{feature}</strong> ({data_type})</li>"
        
        if len(metadata.feature_names) > 20:
            feature_list += f"<li><em>... 他 {len(metadata.feature_names) - 20} 個の特徴量</em></li>"
        
        feature_list += "</ul>"
        
        return summary_grid + feature_list

    def create_statistics_section(self, stats: Dict[str, Stats]) -> str:
        """
        Create statistics section.
        
        Args:
            stats: Dictionary of feature statistics
            
        Returns:
            HTML string for statistics section
        """
        if not stats:
            return "<p>統計情報が利用できません。</p>"
            
        # Create statistics table
        table_html = """
        <table class="stats-table">
            <thead>
                <tr>
                    <th>特徴量</th>
                    <th>平均値</th>
                    <th>中央値</th>
                    <th>標準偏差</th>
                    <th>最小値</th>
                    <th>最大値</th>
                    <th>第1四分位</th>
                    <th>第3四分位</th>
                </tr>
            </thead>
            <tbody>
        """
        
        for feature, stat in stats.items():
            table_html += f"""
                <tr>
                    <td><strong>{feature}</strong></td>
                    <td>{stat.mean:.4f}</td>
                    <td>{stat.median:.4f}</td>
                    <td>{stat.std:.4f}</td>
                    <td>{stat.min:.4f}</td>
                    <td>{stat.max:.4f}</td>
                    <td>{stat.q25:.4f}</td>
                    <td>{stat.q75:.4f}</td>
                </tr>
            """
        
        table_html += "</tbody></table>"
        
        return table_html

    def create_recommendations_section(
        self,
        rul_features: List[tuple],
        fault_features: List[str],
        preprocessing_recommendations: Dict[str, Any]
    ) -> str:
        """
        Create recommendations section.
        
        Args:
            rul_features: List of RUL-relevant features with scores
            fault_features: List of fault-discriminative features
            preprocessing_recommendations: Preprocessing recommendations
            
        Returns:
            HTML string for recommendations section
        """
        recommendations_html = ""
        
        # RUL prediction recommendations
        if rul_features:
            recommendations_html += """
            <div class="recommendations">
                <h4>🔧 RUL予測モデル構築の推奨事項</h4>
                <p>以下の特徴量がRUL予測に有効です：</p>
                <ul>
            """
            
            for feature, score in rul_features[:10]:  # Top 10 features
                recommendations_html += f"<li><strong>{feature}</strong> (相関スコア: {score:.3f})</li>"
            
            recommendations_html += """
                </ul>
                <p><strong>推奨アプローチ:</strong> これらの特徴量を使用して回帰モデル（Random Forest、XGBoost等）を構築してください。</p>
            </div>
            """
        
        # Fault diagnosis recommendations
        if fault_features:
            recommendations_html += """
            <div class="recommendations">
                <h4>⚠️ 故障診断モデル構築の推奨事項</h4>
                <p>以下の特徴量が故障レベルの識別に有効です：</p>
                <ul>
            """
            
            for feature in fault_features[:10]:  # Top 10 features
                recommendations_html += f"<li><strong>{feature}</strong></li>"
            
            recommendations_html += """
                </ul>
                <p><strong>推奨アプローチ:</strong> これらの特徴量を使用して分類モデル（SVM、Random Forest等）を構築してください。</p>
            </div>
            """
        
        # Preprocessing recommendations
        if preprocessing_recommendations:
            recommendations_html += """
            <div class="recommendations">
                <h4>🔄 データ前処理の推奨事項</h4>
            """
            
            if 'missing_value_strategies' in preprocessing_recommendations:
                strategies = preprocessing_recommendations['missing_value_strategies']
                if strategies:
                    recommendations_html += "<p><strong>欠損値処理:</strong></p><ul>"
                    for feature, strategy in strategies.items():
                        if strategy != 'no_action':
                            recommendations_html += f"<li>{feature}: {strategy}</li>"
                    recommendations_html += "</ul>"
            
            if 'scaling_recommendation' in preprocessing_recommendations:
                scaling = preprocessing_recommendations['scaling_recommendation']
                if scaling and scaling.get('method') != 'none':
                    recommendations_html += f"<p><strong>スケーリング:</strong> {scaling.get('method', 'unknown')} を推奨</p>"
                    recommendations_html += f"<p><strong>理由:</strong> {scaling.get('reason', '')}</p>"
            
            recommendations_html += "</div>"
        
        # General recommendations
        recommendations_html += """
        <div class="recommendations">
            <h4>📊 一般的な推奨事項</h4>
            <ul>
                <li>外れ値の詳細調査を実施し、データ品質を確認してください</li>
                <li>相関の高い特徴量ペアについて、多重共線性の影響を検討してください</li>
                <li>時系列データの場合、時間的な順序を保持したデータ分割を使用してください</li>
                <li>モデル性能評価には、ドメイン固有の評価指標を使用してください</li>
            </ul>
        </div>
        """
        
        return recommendations_html

    def _create_quality_section(self, missing_values: Optional[MissingValueReport]) -> str:
        """Create data quality section."""
        if not missing_values:
            return "<p>データ品質情報が利用できません。</p>"
            
        html = f"<p><strong>総欠損値数:</strong> {missing_values.total_missing}</p>"
        
        if missing_values.missing_counts:
            html += "<h3>特徴量別欠損値</h3>"
            html += '<table class="stats-table"><thead><tr><th>特徴量</th><th>欠損数</th><th>欠損率</th></tr></thead><tbody>'
            
            for feature, count in missing_values.missing_counts.items():
                percentage = missing_values.missing_percentages.get(feature, 0.0)
                html += f"<tr><td>{feature}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
            
            html += "</tbody></table>"
        
        return html

    def _create_correlations_section(self, correlation_matrix: Optional[pd.DataFrame]) -> str:
        """Create correlations section."""
        if correlation_matrix is None or correlation_matrix.empty:
            return "<p>相関情報が利用できません。</p>"
            
        # Find high correlations
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:  # High correlation threshold
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        corr_val
                    ))
        
        html = f"<p><strong>強い相関を持つ特徴量ペア数:</strong> {len(high_corr_pairs)}</p>"
        
        if high_corr_pairs:
            html += "<h3>強い相関を持つ特徴量ペア</h3>"
            html += '<table class="stats-table"><thead><tr><th>特徴量1</th><th>特徴量2</th><th>相関係数</th></tr></thead><tbody>'
            
            for feat1, feat2, corr in high_corr_pairs[:20]:  # Top 20
                html += f"<tr><td>{feat1}</td><td>{feat2}</td><td>{corr:.3f}</td></tr>"
            
            html += "</tbody></table>"
        
        return html

    def _create_outliers_section(self, outliers: Optional[OutlierSummary]) -> str:
        """Create outliers section."""
        if not outliers:
            return "<p>外れ値情報が利用できません。</p>"
            
        html = ""
        
        if outliers.outlier_counts:
            total_outliers = sum(outliers.outlier_counts.values())
            html += f"<p><strong>総外れ値数:</strong> {total_outliers}</p>"
            
            html += "<h3>特徴量別外れ値</h3>"
            html += '<table class="stats-table"><thead><tr><th>特徴量</th><th>外れ値数</th><th>外れ値率</th></tr></thead><tbody>'
            
            for feature, count in outliers.outlier_counts.items():
                percentage = outliers.outlier_percentages.get(feature, 0.0)
                html += f"<tr><td>{feature}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
            
            html += "</tbody></table>"
        
        return html

    def _create_timeseries_section(self, trends: Optional[TrendReport]) -> str:
        """Create time series section."""
        if not trends:
            return "<p>時系列情報が利用できません。</p>"
            
        html = "<p>時系列トレンド分析結果:</p>"
        
        if hasattr(trends, 'trend_directions') and trends.trend_directions:
            html += '<table class="stats-table"><thead><tr><th>特徴量</th><th>トレンド方向</th></tr></thead><tbody>'
            
            for feature, direction in trends.trend_directions.items():
                html += f"<tr><td>{feature}</td><td>{direction}</td></tr>"
            
            html += "</tbody></table>"
        
        return html

    def _create_visualizations_section(self, visualization_paths: List[Path]) -> str:
        """Create visualizations section."""
        if not visualization_paths:
            return "<p>可視化ファイルが利用できません。</p>"
            
        html = f"<p><strong>生成された可視化数:</strong> {len(visualization_paths)}</p>"
        html += "<h3>可視化ファイル</h3><ul>"
        
        for path in visualization_paths:
            # Convert to relative path for HTML
            rel_path = os.path.relpath(path)
            html += f'<li><a href="{rel_path}" target="_blank">{path.name}</a></li>'
        
        html += "</ul>"
        
        return html

    def _create_rul_analysis_section(self, rul_features: List[tuple]) -> str:
        """Create RUL analysis section."""
        if not rul_features:
            return "<p>RUL特徴量分析結果が利用できません。</p>"
            
        html = f"<p><strong>RUL予測に有効な特徴量数:</strong> {len(rul_features)}</p>"
        
        if rul_features:
            html += "<h3>上位RUL関連特徴量</h3>"
            html += '<table class="stats-table"><thead><tr><th>順位</th><th>特徴量</th><th>相関スコア</th></tr></thead><tbody>'
            
            for i, (feature, score) in enumerate(rul_features[:10], 1):
                html += f"<tr><td>{i}</td><td>{feature}</td><td>{score:.4f}</td></tr>"
            
            html += "</tbody></table>"
        
        return html

    def _create_fault_analysis_section(self, fault_features: List[str]) -> str:
        """Create fault analysis section."""
        if not fault_features:
            return "<p>故障レベル分析結果が利用できません。</p>"
            
        html = f"<p><strong>故障識別に有効な特徴量数:</strong> {len(fault_features)}</p>"
        
        if fault_features:
            html += "<h3>故障識別特徴量</h3><ul>"
            
            for feature in fault_features[:20]:  # Top 20
                html += f"<li><strong>{feature}</strong></li>"
            
            html += "</ul>"
        
        return html