"""
Real data analysis orchestrator.

This module provides comprehensive orchestration for real NASA PCOE ES12 data analysis,
including specialized data loading, enhanced analysis, validation, and reporting.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import warnings

import pandas as pd
import numpy as np

from .data.es12_loader import ES12DataLoader
from .analysis.statistics import StatisticsAnalyzer
from .analysis.correlation import CorrelationAnalyzer
from .analysis.outliers import OutlierDetector
from .analysis.timeseries import TimeSeriesAnalyzer
from .analysis.rul_features import RULFeatureAnalyzer
from .analysis.fault_level import FaultLevelAnalyzer
from .analysis.real_data_validator import RealDataValidator
from .preprocessing.recommender import PreprocessingRecommender
from .visualization.engine import VisualizationEngine
from .reporting.real_data_generator import RealDataReportGenerator
from .models import AnalysisResults, DatasetMetadata
from .exceptions import AnalysisError, DataLoadError


class RealDataOrchestrator:
    """
    Orchestrator for comprehensive real NASA PCOE ES12 data analysis.
    
    This orchestrator provides end-to-end analysis capabilities specifically
    designed for real capacitor degradation data, including:
    - Specialized ES12 data loading
    - Enhanced degradation pattern analysis
    - Individual capacitor comparison
    - Methodology validation
    - Real data specific reporting
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize the real data orchestrator.
        
        Args:
            output_dir: Directory for output files (default: ./output)
        """
        self.output_dir = Path(output_dir) if output_dir else Path("output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_loader = ES12DataLoader()
        self.statistics_analyzer = StatisticsAnalyzer()
        self.correlation_analyzer = CorrelationAnalyzer()
        self.outlier_detector = OutlierDetector()
        self.timeseries_analyzer = TimeSeriesAnalyzer()
        self.rul_analyzer = RULFeatureAnalyzer()
        self.fault_analyzer = FaultLevelAnalyzer()
        self.validator = RealDataValidator()
        self.preprocessing_recommender = PreprocessingRecommender()
        self.visualization_engine = VisualizationEngine()
        self.report_generator = RealDataReportGenerator()
        
        # Setup logging
        self._setup_logging()
        
        # Analysis results storage
        self.analysis_results: Optional[AnalysisResults] = None
        self.real_data_metrics: Dict[str, Any] = {}
        self.validation_results: Dict[str, Any] = {}

    def _setup_logging(self):
        """Setup logging for the orchestrator."""
        log_file = self.output_dir / "real_data_analysis.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def run_comprehensive_analysis(
        self,
        data_path: Path,
        sample_data_comparison: Optional[Dict[str, Any]] = None,
        generate_visualizations: bool = True,
        generate_report: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive real data analysis.
        
        Args:
            data_path: Path to the ES12.mat file
            sample_data_comparison: Optional comparison data from sample analysis
            generate_visualizations: Whether to generate visualizations
            generate_report: Whether to generate the final report
            
        Returns:
            Dictionary containing all analysis results and metrics
        """
        self.logger.info("Starting comprehensive real data analysis")
        
        try:
            # Step 1: Load and validate real data
            self.logger.info("Loading ES12 real data...")
            df = self._load_and_validate_data(data_path)
            
            # Step 2: Extract real data specific metrics
            self.logger.info("Extracting real data metrics...")
            self.real_data_metrics = self._extract_real_data_metrics(df)
            
            # Step 3: Run core analysis
            self.logger.info("Running core analysis...")
            self.analysis_results = self._run_core_analysis(df)
            
            # Step 4: Perform degradation-specific analysis
            self.logger.info("Performing degradation analysis...")
            degradation_analysis = self._perform_degradation_analysis(df)
            self.real_data_metrics.update(degradation_analysis)
            
            # Step 5: Individual capacitor analysis
            self.logger.info("Analyzing individual capacitors...")
            capacitor_analysis = self._analyze_individual_capacitors(df)
            self.real_data_metrics['capacitor_analysis'] = capacitor_analysis
            
            # Step 6: Validate methodology
            self.logger.info("Validating analysis methodology...")
            self.validation_results = self._validate_methodology(sample_data_comparison)
            
            # Step 7: Generate visualizations
            if generate_visualizations:
                self.logger.info("Generating visualizations...")
                visualization_paths = self._generate_enhanced_visualizations(df)
                if self.analysis_results:
                    self.analysis_results.visualization_paths.extend(visualization_paths)
            
            # Step 8: Generate comprehensive report
            if generate_report:
                self.logger.info("Generating comprehensive report...")
                report_path = self._generate_comprehensive_report()
                self.logger.info(f"Report generated: {report_path}")
            
            # Compile final results
            final_results = {
                'analysis_results': self.analysis_results,
                'real_data_metrics': self.real_data_metrics,
                'validation_results': self.validation_results,
                'data_quality_score': self._calculate_data_quality_score(),
                'methodology_reliability': self._assess_methodology_reliability(),
                'practical_recommendations': self._generate_practical_recommendations()
            }
            
            self.logger.info("Comprehensive real data analysis completed successfully")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            raise AnalysisError(f"Comprehensive analysis failed: {str(e)}")

    def _load_and_validate_data(self, data_path: Path) -> pd.DataFrame:
        """Load and validate ES12 real data."""
        try:
            # Load data using specialized ES12 loader
            df = self.data_loader.load_dataset(data_path)
            
            # Validate data
            validation_result = self.data_loader.validate_data(df)
            
            if not validation_result.is_valid:
                raise DataLoadError(f"Data validation failed: {validation_result.errors}")
            
            if validation_result.warnings:
                for warning in validation_result.warnings:
                    self.logger.warning(f"Data validation warning: {warning}")
            
            self.logger.info(f"Successfully loaded {len(df)} records with {len(df.columns)} features")
            return df
            
        except Exception as e:
            raise DataLoadError(f"Failed to load and validate data: {str(e)}")

    def _extract_real_data_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract real data specific metrics."""
        metrics = {}
        
        try:
            # Basic data characteristics
            metrics['capacitor_count'] = df['capacitor'].nunique() if 'capacitor' in df.columns else 0
            metrics['total_cycles'] = df['cycle'].max() if 'cycle' in df.columns else 0
            metrics['measurement_points'] = len(df)
            
            # Data completeness
            total_values = df.size
            missing_values = df.isnull().sum().sum()
            metrics['data_completeness'] = (total_values - missing_values) / total_values
            
            # Measurement precision (based on voltage data variability)
            voltage_cols = [col for col in df.columns if any(term in col.lower() for term in ['vl', 'vo', 'voltage'])]
            if voltage_cols:
                precisions = []
                for col in voltage_cols:
                    if df[col].std() > 0:
                        cv = df[col].std() / df[col].mean()
                        precision = 1 / (1 + cv)  # Higher precision for lower CV
                        precisions.append(precision)
                
                if precisions:
                    metrics['measurement_precision'] = np.mean(precisions)
            
            # Signal-to-noise ratio estimation
            if voltage_cols:
                snr_estimates = []
                for col in voltage_cols:
                    signal_power = np.var(df[col].dropna())
                    # Estimate noise as high-frequency component
                    if len(df[col].dropna()) > 10:
                        diff_signal = np.diff(df[col].dropna())
                        noise_power = np.var(diff_signal)
                        if noise_power > 0:
                            snr = 10 * np.log10(signal_power / noise_power)
                            snr_estimates.append(snr)
                
                if snr_estimates:
                    metrics['signal_noise_ratio'] = np.mean(snr_estimates)
            
            # Data quality score
            quality_factors = []
            if 'data_completeness' in metrics:
                quality_factors.append(metrics['data_completeness'])
            if 'measurement_precision' in metrics:
                quality_factors.append(metrics['measurement_precision'])
            
            if quality_factors:
                metrics['data_quality_score'] = np.mean(quality_factors)
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Failed to extract some real data metrics: {e}")
            return metrics

    def _run_core_analysis(self, df: pd.DataFrame) -> AnalysisResults:
        """Run core analysis components."""
        try:
            # Get metadata
            metadata = self.data_loader.get_metadata(df)
            
            # Statistical analysis
            statistics = self.statistics_analyzer.compute_descriptive_stats(df)
            missing_values = self.statistics_analyzer.analyze_missing_values(df)
            
            # Correlation analysis
            correlation_matrix = self.correlation_analyzer.compute_correlation_matrix(df)
            
            # Outlier detection
            outliers_iqr = self.outlier_detector.detect_outliers_iqr(df)
            outliers = self.outlier_detector.summarize_outliers(outliers_iqr)
            
            # Time series analysis
            time_series_trends = None
            if 'cycle' in df.columns:
                temporal_features = self.timeseries_analyzer.identify_temporal_features(df)
                if temporal_features:
                    time_series_trends = self.timeseries_analyzer.compute_trends(df, temporal_features)
            
            # RUL feature analysis
            rul_features = []
            if 'cycle' in df.columns:
                try:
                    # Use cycle as proxy for RUL (inverse relationship)
                    rul_features = self.rul_analyzer.rank_features_for_rul(df, 'cycle')
                except Exception as e:
                    self.logger.warning(f"RUL analysis failed: {e}")
            
            # Fault level analysis
            fault_features = []
            if 'capacitor' in df.columns:
                try:
                    fault_features = self.fault_analyzer.identify_discriminative_features(df, 'capacitor')
                except Exception as e:
                    self.logger.warning(f"Fault analysis failed: {e}")
            
            # Preprocessing recommendations
            preprocessing_recommendations = self.preprocessing_recommender.recommend_missing_value_strategy(missing_values)
            scaling_rec = self.preprocessing_recommender.recommend_scaling(df)
            preprocessing_recommendations['scaling_recommendation'] = scaling_rec.__dict__ if scaling_rec else {}
            
            # Create analysis results
            analysis_results = AnalysisResults(
                metadata=metadata,
                statistics=statistics,
                missing_values=missing_values,
                correlation_matrix=correlation_matrix,
                outliers=outliers,
                time_series_trends=time_series_trends,
                rul_features=rul_features,
                fault_features=fault_features,
                preprocessing_recommendations=preprocessing_recommendations,
                visualization_paths=[]
            )
            
            return analysis_results
            
        except Exception as e:
            raise AnalysisError(f"Core analysis failed: {str(e)}")

    def _perform_degradation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform degradation-specific analysis."""
        degradation_analysis = {}
        
        try:
            # Analyze degradation trends for each capacitor
            if 'capacitor' in df.columns and 'cycle' in df.columns:
                degradation_trends = {}
                
                # Voltage-related features for degradation analysis
                voltage_features = [col for col in df.columns 
                                  if any(term in col.lower() for term in ['vl', 'vo', 'voltage', 'ratio'])]
                
                for feature in voltage_features:
                    if df[feature].notna().sum() > 10:  # Sufficient data points
                        # Calculate overall degradation trend
                        feature_data = df.groupby('cycle')[feature].mean()
                        
                        if len(feature_data) > 2:
                            # Linear regression to find degradation rate
                            cycles = feature_data.index.values
                            values = feature_data.values
                            
                            # Remove NaN values
                            valid_mask = ~np.isnan(values)
                            if np.sum(valid_mask) > 2:
                                cycles_clean = cycles[valid_mask]
                                values_clean = values[valid_mask]
                                
                                # Calculate degradation rate
                                slope = np.polyfit(cycles_clean, values_clean, 1)[0]
                                
                                initial_value = values_clean[0] if len(values_clean) > 0 else np.nan
                                final_value = values_clean[-1] if len(values_clean) > 0 else np.nan
                                
                                if not np.isnan(initial_value) and initial_value != 0:
                                    change_rate = ((final_value - initial_value) / initial_value) * 100
                                    degradation_rate = f"{abs(slope):.4f} units/cycle"
                                else:
                                    change_rate = np.nan
                                    degradation_rate = "N/A"
                                
                                degradation_trends[feature] = {
                                    'initial_value': f"{initial_value:.4f}" if not np.isnan(initial_value) else "N/A",
                                    'final_value': f"{final_value:.4f}" if not np.isnan(final_value) else "N/A",
                                    'change_rate': f"{change_rate:.2f}%" if not np.isnan(change_rate) else "N/A",
                                    'degradation_rate': degradation_rate,
                                    'slope': slope
                                }
                
                degradation_analysis['degradation_trends'] = degradation_trends
            
            # Failure prediction analysis
            failure_prediction = {
                'early_warning_indicators': [
                    "電圧比の急激な変化（>5%/cycle）",
                    "標準偏差の増加傾向",
                    "測定値の不規則な変動"
                ],
                'critical_thresholds': [
                    "電圧比 < 0.8（初期値の80%）",
                    "標準偏差 > 初期値の2倍",
                    "連続3サイクルでの単調減少"
                ]
            }
            degradation_analysis['failure_prediction'] = failure_prediction
            
            return degradation_analysis
            
        except Exception as e:
            self.logger.warning(f"Degradation analysis failed: {e}")
            return degradation_analysis

    def _analyze_individual_capacitors(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze individual capacitor characteristics."""
        capacitor_analysis = {}
        
        try:
            if 'capacitor' in df.columns:
                capacitors = df['capacitor'].unique()
                
                for capacitor in capacitors:
                    cap_data = df[df['capacitor'] == capacitor]
                    
                    if len(cap_data) > 5:  # Sufficient data for analysis
                        analysis = {}
                        
                        # Calculate degradation rate
                        if 'cycle' in cap_data.columns and 'voltage_ratio' in cap_data.columns:
                            voltage_data = cap_data.groupby('cycle')['voltage_ratio'].mean()
                            if len(voltage_data) > 2:
                                cycles = voltage_data.index.values
                                ratios = voltage_data.values
                                
                                # Remove NaN values
                                valid_mask = ~np.isnan(ratios)
                                if np.sum(valid_mask) > 2:
                                    cycles_clean = cycles[valid_mask]
                                    ratios_clean = ratios[valid_mask]
                                    
                                    # Calculate degradation rate
                                    if len(ratios_clean) > 1:
                                        initial_ratio = ratios_clean[0]
                                        final_ratio = ratios_clean[-1]
                                        
                                        if initial_ratio != 0:
                                            degradation_rate = ((initial_ratio - final_ratio) / initial_ratio) * 100
                                            analysis['degradation_rate'] = degradation_rate
                        
                        # Estimate remaining useful life (simplified)
                        if 'degradation_rate' in analysis and analysis['degradation_rate'] > 0:
                            # Assume failure at 20% degradation
                            current_degradation = analysis['degradation_rate']
                            if current_degradation < 20:
                                remaining_degradation = 20 - current_degradation
                                cycles_per_percent = len(cap_data) / current_degradation if current_degradation > 0 else 0
                                estimated_rul = int(remaining_degradation * cycles_per_percent)
                                analysis['estimated_rul'] = f"{estimated_rul} cycles"
                            else:
                                analysis['estimated_rul'] = "End of life"
                        else:
                            analysis['estimated_rul'] = "Unknown"
                        
                        # Health status assessment
                        if 'degradation_rate' in analysis:
                            deg_rate = analysis['degradation_rate']
                            if deg_rate < 5:
                                analysis['health_status'] = "良好"
                            elif deg_rate < 15:
                                analysis['health_status'] = "注意"
                            else:
                                analysis['health_status'] = "警告"
                        else:
                            analysis['health_status'] = "不明"
                        
                        capacitor_analysis[capacitor] = analysis
            
            return capacitor_analysis
            
        except Exception as e:
            self.logger.warning(f"Individual capacitor analysis failed: {e}")
            return capacitor_analysis

    def _validate_methodology(self, sample_data_comparison: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate analysis methodology using real data."""
        try:
            # Prepare real data results for validation
            real_data_results = {
                'statistics': {
                    'descriptive_stats': self.analysis_results.statistics if self.analysis_results else {}
                },
                'correlations': {
                    'correlation_matrix': self.analysis_results.correlation_matrix if self.analysis_results else pd.DataFrame()
                },
                'outliers': self.analysis_results.outliers.__dict__ if self.analysis_results and self.analysis_results.outliers else {},
                'time_series': self.analysis_results.time_series_trends.__dict__ if self.analysis_results and self.analysis_results.time_series_trends else {},
                'degradation_patterns': self.real_data_metrics.get('degradation_trends', {})
            }
            
            # Run validation
            validation_results = self.validator.validate_analysis_methodology(
                real_data_results,
                sample_data_comparison
            )
            
            return validation_results
            
        except Exception as e:
            self.logger.warning(f"Methodology validation failed: {e}")
            return {}

    def _generate_enhanced_visualizations(self, df: pd.DataFrame) -> List[Path]:
        """Generate enhanced visualizations for real data."""
        visualization_paths = []
        
        try:
            viz_dir = self.output_dir / "visualizations"
            viz_dir.mkdir(exist_ok=True)
            
            # Generate standard visualizations
            if self.analysis_results:
                # Distribution plots
                numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_features:
                    dist_paths = self.visualization_engine.plot_distributions(
                        df, numeric_features[:10], viz_dir  # Limit to first 10 features
                    )
                    visualization_paths.extend(dist_paths)
                
                # Correlation heatmap
                if self.analysis_results.correlation_matrix is not None and not self.analysis_results.correlation_matrix.empty:
                    corr_path = self.visualization_engine.plot_correlation_heatmap(
                        self.analysis_results.correlation_matrix, viz_dir
                    )
                    visualization_paths.append(corr_path)
                
                # Time series plots for degradation
                if 'cycle' in df.columns:
                    voltage_features = [col for col in df.columns 
                                      if any(term in col.lower() for term in ['vl', 'vo', 'voltage', 'ratio'])]
                    if voltage_features:
                        ts_paths = self.visualization_engine.plot_time_series(
                            df, ['cycle'] + voltage_features[:5], viz_dir
                        )
                        visualization_paths.extend(ts_paths)
            
            return visualization_paths
            
        except Exception as e:
            self.logger.warning(f"Enhanced visualization generation failed: {e}")
            return visualization_paths

    def _generate_comprehensive_report(self) -> Path:
        """Generate comprehensive real data analysis report."""
        try:
            report_path = self.output_dir / "real_data_analysis_report.html"
            
            if self.analysis_results:
                generated_path = self.report_generator.generate_real_data_report(
                    self.analysis_results,
                    report_path,
                    self.real_data_metrics,
                    None,  # Sample data comparison not implemented yet
                    format="html"
                )
                return generated_path
            else:
                raise AnalysisError("No analysis results available for report generation")
                
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            raise AnalysisError(f"Failed to generate comprehensive report: {str(e)}")

    def _calculate_data_quality_score(self) -> float:
        """Calculate overall data quality score."""
        try:
            quality_factors = []
            
            # Data completeness
            if 'data_completeness' in self.real_data_metrics:
                quality_factors.append(self.real_data_metrics['data_completeness'])
            
            # Measurement precision
            if 'measurement_precision' in self.real_data_metrics:
                quality_factors.append(self.real_data_metrics['measurement_precision'])
            
            # Validation scores
            if self.validation_results and 'reliability_assessment' in self.validation_results:
                reliability = self.validation_results['reliability_assessment']
                if 'overall_score' in reliability:
                    quality_factors.append(reliability['overall_score'])
            
            return np.mean(quality_factors) if quality_factors else 0.5
            
        except Exception as e:
            self.logger.warning(f"Data quality score calculation failed: {e}")
            return 0.5

    def _assess_methodology_reliability(self) -> Dict[str, Any]:
        """Assess overall methodology reliability."""
        try:
            if self.validation_results and 'reliability_assessment' in self.validation_results:
                return self.validation_results['reliability_assessment']
            else:
                return {
                    'overall_score': 0.5,
                    'reliability_level': 'Unknown',
                    'confidence_interval': (0.0, 1.0)
                }
                
        except Exception as e:
            self.logger.warning(f"Methodology reliability assessment failed: {e}")
            return {'overall_score': 0.0, 'reliability_level': 'Error'}

    def _generate_practical_recommendations(self) -> List[str]:
        """Generate practical recommendations based on real data analysis."""
        recommendations = []
        
        try:
            # Data quality recommendations
            data_quality = self._calculate_data_quality_score()
            if data_quality > 0.9:
                recommendations.append("データ品質は優秀です。現在の分析手法を継続使用することを推奨します")
            elif data_quality > 0.7:
                recommendations.append("データ品質は良好ですが、測定精度の向上により更なる改善が期待できます")
            else:
                recommendations.append("データ品質の改善が必要です。測定条件と前処理手法の見直しを推奨します")
            
            # Degradation analysis recommendations
            if 'degradation_trends' in self.real_data_metrics:
                trends = self.real_data_metrics['degradation_trends']
                if trends:
                    recommendations.append("実測劣化パターンに基づく予測モデルの構築を推奨します")
                    recommendations.append("劣化速度の個体差を考慮した保全計画の策定を推奨します")
            
            # Capacitor-specific recommendations
            if 'capacitor_analysis' in self.real_data_metrics:
                cap_analysis = self.real_data_metrics['capacitor_analysis']
                warning_capacitors = [cap for cap, data in cap_analysis.items() 
                                    if isinstance(data, dict) and data.get('health_status') == '警告']
                
                if warning_capacitors:
                    recommendations.append(f"警告状態のコンデンサ（{', '.join(warning_capacitors)}）の優先的な監視・交換を推奨します")
            
            # Methodology recommendations
            if self.validation_results and 'recommendations' in self.validation_results:
                recommendations.extend(self.validation_results['recommendations'])
            
            # Default recommendations
            if not recommendations:
                recommendations.append("継続的なデータ収集と分析手法の改善を推奨します")
            
            return recommendations
            
        except Exception as e:
            self.logger.warning(f"Practical recommendations generation failed: {e}")
            return ["分析結果に基づく推奨事項の生成に失敗しました"]

    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get a summary of the analysis results."""
        try:
            summary = {
                'data_overview': {},
                'key_findings': {},
                'quality_assessment': {},
                'recommendations': []
            }
            
            # Data overview
            if self.analysis_results and self.analysis_results.metadata:
                metadata = self.analysis_results.metadata
                summary['data_overview'] = {
                    'records': metadata.n_records,
                    'features': metadata.n_features,
                    'capacitors': self.real_data_metrics.get('capacitor_count', 0),
                    'cycles': self.real_data_metrics.get('total_cycles', 0)
                }
            
            # Key findings
            summary['key_findings'] = {
                'data_quality_score': self._calculate_data_quality_score(),
                'degradation_patterns_detected': len(self.real_data_metrics.get('degradation_trends', {})),
                'methodology_reliability': self._assess_methodology_reliability().get('reliability_level', 'Unknown')
            }
            
            # Quality assessment
            summary['quality_assessment'] = {
                'data_completeness': self.real_data_metrics.get('data_completeness', 0.0),
                'measurement_precision': self.real_data_metrics.get('measurement_precision', 0.0),
                'signal_noise_ratio': self.real_data_metrics.get('signal_noise_ratio', 0.0)
            }
            
            # Recommendations
            summary['recommendations'] = self._generate_practical_recommendations()
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Analysis summary generation failed: {e}")
            return {'error': str(e)}