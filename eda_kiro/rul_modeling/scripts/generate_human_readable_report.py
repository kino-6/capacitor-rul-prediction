#!/usr/bin/env python3
"""
Human-Readable Comprehensive Report Generator
人間向け包括的レポート生成器

This script generates a comprehensive, human-readable report in both Japanese and English
that includes model performance, test results, and overall system evaluation.

モデル性能、テスト結果、システム全体の評価を含む、
日本語と英語の包括的で人間が読みやすいレポートを生成します。
"""

import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import json
import base64
from io import BytesIO

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from true_rul.data_loader import DataLoader
from true_rul.feature_extractor import FeatureExtractor
from true_rul.time_series_preprocessor import TimeSeriesPreprocessor
from true_rul.rul_regression_model import RULRegressionModel
from true_rul.isolation_forest_detector import IsolationForestDetector
from true_rul.improved_ocsvm import ImprovedOCSVM
from true_rul.data_structures import TrainingDataset
from true_rul.config import ES12_CONFIG, MODEL_CONFIG, setup_logging

# Import the comprehensive evaluator
from comprehensive_model_evaluation import ComprehensiveModelEvaluator

logger = logging.getLogger(__name__)

# Set up matplotlib for Japanese font support
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


class HumanReadableReportGenerator:
    """人間向けレポート生成器"""
    
    def __init__(self, output_dir: Path):
        """
        Initialize the report generator
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "html_reports").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        
        logger.info(f"Initialized human-readable report generator: {output_dir}")
    
    def plot_to_base64(self, fig) -> str:
        """Convert matplotlib figure to base64 string for HTML embedding"""
        buffer = BytesIO()
        fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        return image_base64
    
    def create_performance_summary_plot(self, rul_results: Dict[str, Any], anomaly_results: Dict[str, Any]) -> str:
        """Create a comprehensive performance summary plot"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # RUL Performance Metrics
        datasets = list(rul_results.keys())
        rmse_values = [rul_results[ds]['rmse'] for ds in datasets]
        mae_values = [rul_results[ds]['mae'] for ds in datasets]
        r2_values = [rul_results[ds]['r2'] for ds in datasets]
        
        # Plot RMSE
        axes[0, 0].bar(datasets, rmse_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0, 0].set_title('RMSE by Dataset\nデータセット別RMSE', fontweight='bold')
        axes[0, 0].set_ylabel('RMSE (cycles)')
        for i, v in enumerate(rmse_values):
            axes[0, 0].text(i, v + max(rmse_values) * 0.01, f'{v:.2f}', ha='center', va='bottom')
        
        # Plot MAE
        axes[0, 1].bar(datasets, mae_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0, 1].set_title('MAE by Dataset\nデータセット別MAE', fontweight='bold')
        axes[0, 1].set_ylabel('MAE (cycles)')
        for i, v in enumerate(mae_values):
            axes[0, 1].text(i, v + max(mae_values) * 0.01, f'{v:.2f}', ha='center', va='bottom')
        
        # Plot R²
        axes[0, 2].bar(datasets, r2_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[0, 2].set_title('R² Score by Dataset\nデータセット別R²スコア', fontweight='bold')
        axes[0, 2].set_ylabel('R² Score')
        axes[0, 2].set_ylim(0, 1)
        for i, v in enumerate(r2_values):
            axes[0, 2].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        # Anomaly Detection Metrics
        fpr_values = [anomaly_results[ds]['fpr'] for ds in datasets]
        tpr_values = [anomaly_results[ds]['tpr'] for ds in datasets]
        f1_values = [anomaly_results[ds]['f1'] for ds in datasets]
        
        # Plot FPR
        axes[1, 0].bar(datasets, fpr_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[1, 0].set_title('False Positive Rate\n偽陽性率', fontweight='bold')
        axes[1, 0].set_ylabel('FPR')
        axes[1, 0].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Target < 5%')
        axes[1, 0].legend()
        for i, v in enumerate(fpr_values):
            color = 'green' if v < 0.05 else 'red'
            axes[1, 0].text(i, v + max(fpr_values) * 0.02, f'{v:.4f}', ha='center', va='bottom', color=color, fontweight='bold')
        
        # Plot TPR
        axes[1, 1].bar(datasets, tpr_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[1, 1].set_title('True Positive Rate\n真陽性率', fontweight='bold')
        axes[1, 1].set_ylabel('TPR')
        for i, v in enumerate(tpr_values):
            axes[1, 1].text(i, v + max(tpr_values) * 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        # Plot F1 Score
        axes[1, 2].bar(datasets, f1_values, color=['skyblue', 'lightcoral', 'lightgreen'])
        axes[1, 2].set_title('F1 Score\nF1スコア', fontweight='bold')
        axes[1, 2].set_ylabel('F1 Score')
        for i, v in enumerate(f1_values):
            axes[1, 2].text(i, v + max(f1_values) * 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        return self.plot_to_base64(fig)
    
    def create_detailed_confusion_matrices(self, anomaly_results: Dict[str, Any]) -> str:
        """Create detailed confusion matrices for all datasets"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, (dataset_name, results) in enumerate(anomaly_results.items()):
            cm_data = results['confusion_matrix']
            cm = np.array([[cm_data['tn'], cm_data['fp']], 
                           [cm_data['fn'], cm_data['tp']]])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Normal', 'Anomaly'],
                       yticklabels=['Normal', 'Anomaly'],
                       ax=axes[i])
            
            axes[i].set_title(f'{dataset_name.upper()} Dataset\n{dataset_name.upper()}データセット', 
                             fontweight='bold')
            axes[i].set_xlabel('Predicted / 予測')
            axes[i].set_ylabel('Actual / 実際')
            
            # Add performance metrics
            fpr = results['fpr']
            tpr = results['tpr']
            f1 = results['f1']
            
            metrics_text = f'FPR: {fpr:.4f}\nTPR: {tpr:.4f}\nF1: {f1:.4f}'
            axes[i].text(0.02, 0.98, metrics_text, transform=axes[i].transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        return self.plot_to_base64(fig)
    
    def create_rul_performance_analysis(self, rul_results: Dict[str, Any]) -> str:
        """Create detailed RUL performance analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Prediction vs Actual scatter plots
        for i, (dataset_name, results) in enumerate(rul_results.items()):
            if i >= 3:  # Only plot first 3 datasets
                break
                
            row = i // 2
            col = i % 2
            
            true_values = np.array(results['true_values'])
            predictions = np.array(results['predictions'])
            
            axes[row, col].scatter(true_values, predictions, alpha=0.6, s=30)
            
            # Perfect prediction line
            max_val = max(np.max(true_values), np.max(predictions))
            axes[row, col].plot([0, max_val], [0, max_val], 'r--', lw=2, alpha=0.8, label='Perfect Prediction')
            
            # Calculate and plot trend line
            z = np.polyfit(true_values, predictions, 1)
            p = np.poly1d(z)
            axes[row, col].plot(true_values, p(true_values), "g--", alpha=0.8, label=f'Trend (slope={z[0]:.3f})')
            
            axes[row, col].set_xlabel('Actual RUL (cycles) / 実際のRUL')
            axes[row, col].set_ylabel('Predicted RUL (cycles) / 予測RUL')
            axes[row, col].set_title(f'{dataset_name.upper()} Dataset\nRMSE: {results["rmse"]:.2f}, R²: {results["r2"]:.3f}', 
                                   fontweight='bold')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        # Error distribution plot
        if len(rul_results) >= 3:
            all_errors = []
            labels = []
            for dataset_name, results in rul_results.items():
                true_values = np.array(results['true_values'])
                predictions = np.array(results['predictions'])
                errors = predictions - true_values
                all_errors.append(errors)
                labels.append(dataset_name.upper())
            
            axes[1, 1].boxplot(all_errors, labels=labels)
            axes[1, 1].set_title('Prediction Error Distribution\n予測誤差分布', fontweight='bold')
            axes[1, 1].set_ylabel('Prediction Error (cycles) / 予測誤差')
            axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self.plot_to_base64(fig)
    
    def create_requirements_compliance_chart(self, rul_results: Dict[str, Any], anomaly_results: Dict[str, Any]) -> str:
        """Create requirements compliance visualization"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # FPR Compliance
        datasets = list(anomaly_results.keys())
        fpr_values = [anomaly_results[ds]['fpr'] for ds in datasets]
        
        colors = ['green' if fpr < 0.05 else 'red' for fpr in fpr_values]
        bars1 = ax1.bar(datasets, fpr_values, color=colors, alpha=0.7)
        ax1.axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='Requirement: FPR < 5%')
        ax1.set_title('FPR Compliance Check\nFPR要件適合性チェック', fontweight='bold', fontsize=14)
        ax1.set_ylabel('False Positive Rate')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, fpr in zip(bars1, fpr_values):
            height = bar.get_height()
            status = '✓ PASS' if fpr < 0.05 else '✗ FAIL'
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{fpr:.4f}\n{status}', ha='center', va='bottom', fontweight='bold')
        
        # RMSE Reasonableness
        rmse_values = [rul_results[ds]['rmse'] for ds in datasets]
        rmse_threshold = 50
        
        colors = ['green' if rmse < rmse_threshold else 'orange' for rmse in rmse_values]
        bars2 = ax2.bar(datasets, rmse_values, color=colors, alpha=0.7)
        ax2.axhline(y=rmse_threshold, color='orange', linestyle='--', linewidth=2, label=f'Target: RMSE < {rmse_threshold}')
        ax2.set_title('RMSE Performance Check\nRMSE性能チェック', fontweight='bold', fontsize=14)
        ax2.set_ylabel('RMSE (cycles)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, rmse in zip(bars2, rmse_values):
            height = bar.get_height()
            status = '✓ GOOD' if rmse < rmse_threshold else '△ ACCEPTABLE'
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rmse:.1f}\n{status}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        return self.plot_to_base64(fig)
    
    def generate_html_report(self, rul_results: Dict[str, Any], anomaly_results: Dict[str, Any], 
                           dataset_info: Dict[str, Any], feature_importance: Optional[Dict[str, float]] = None) -> str:
        """Generate comprehensive HTML report"""
        
        # Generate plots
        performance_plot = self.create_performance_summary_plot(rul_results, anomaly_results)
        confusion_matrices_plot = self.create_detailed_confusion_matrices(anomaly_results)
        rul_analysis_plot = self.create_rul_performance_analysis(rul_results)
        compliance_plot = self.create_requirements_compliance_chart(rul_results, anomaly_results)
        
        # Calculate overall compliance
        val_fpr = anomaly_results.get('val', {}).get('fpr', 1.0)
        test_fpr = anomaly_results.get('test', {}).get('fpr', 1.0)
        val_rmse = rul_results.get('val', {}).get('rmse', float('inf'))
        test_rmse = rul_results.get('test', {}).get('rmse', float('inf'))
        
        fpr_compliance = val_fpr < 0.05 and test_fpr < 0.05
        rmse_reasonable = val_rmse < 50 and test_rmse < 50
        overall_status = fpr_compliance and rmse_reasonable
        
        # Generate HTML
        html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RUL Prediction System - Comprehensive Evaluation Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        .header {{
            text-align: center;
            border-bottom: 3px solid #2c3e50;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: #2c3e50;
            margin: 0;
            font-size: 2.5em;
        }}
        .header h2 {{
            color: #7f8c8d;
            margin: 10px 0 0 0;
            font-weight: normal;
        }}
        .status-badge {{
            display: inline-block;
            padding: 10px 20px;
            border-radius: 25px;
            font-weight: bold;
            font-size: 1.2em;
            margin: 20px 0;
        }}
        .status-pass {{
            background-color: #2ecc71;
            color: white;
        }}
        .status-fail {{
            background-color: #e74c3c;
            color: white;
        }}
        .section {{
            margin: 40px 0;
            padding: 20px;
            border-left: 4px solid #3498db;
            background-color: #f8f9fa;
        }}
        .section h3 {{
            color: #2c3e50;
            margin-top: 0;
            font-size: 1.8em;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            color: #7f8c8d;
            margin-top: 5px;
        }}
        .plot-container {{
            text-align: center;
            margin: 30px 0;
        }}
        .plot-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}
        .dataset-summary {{
            background: #ecf0f1;
            padding: 20px;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .requirements-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .requirements-table th, .requirements-table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        .requirements-table th {{
            background-color: #3498db;
            color: white;
        }}
        .pass {{
            color: #27ae60;
            font-weight: bold;
        }}
        .fail {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .timestamp {{
            text-align: center;
            color: #7f8c8d;
            font-style: italic;
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
        }}
        .feature-importance {{
            max-height: 400px;
            overflow-y: auto;
            background: white;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 15px;
        }}
        .feature-item {{
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #f0f0f0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>RUL Prediction System</h1>
            <h2>Comprehensive Evaluation Report<br>包括的評価レポート</h2>
            <div class="status-badge {'status-pass' if overall_status else 'status-fail'}">
                {'✓ SYSTEM PASSED' if overall_status else '✗ REQUIREMENTS NOT MET'}
            </div>
        </div>

        <div class="section">
            <h3>📊 Executive Summary / エグゼクティブサマリー</h3>
            <p><strong>English:</strong> This report presents a comprehensive evaluation of the RUL (Remaining Useful Life) prediction system using the ES12 capacitor dataset. The system combines regression models for RUL prediction with ensemble anomaly detection to achieve high accuracy while maintaining low false positive rates.</p>
            <p><strong>日本語:</strong> このレポートは、ES12コンデンサデータセットを使用したRUL（残存有用寿命）予測システムの包括的評価を示しています。システムは、RUL予測のための回帰モデルと、高精度を達成しながら低い偽陽性率を維持するアンサンブル異常検知を組み合わせています。</p>
        </div>

        <div class="section">
            <h3>📈 Dataset Information / データセット情報</h3>
            <div class="dataset-summary">
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value">{dataset_info.get('Total Samples', 'N/A')}</div>
                        <div class="metric-label">Total Samples<br>総サンプル数</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{dataset_info.get('Features', 'N/A')}</div>
                        <div class="metric-label">Features<br>特徴量数</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{dataset_info.get('Capacitors', 'N/A')}</div>
                        <div class="metric-label">Capacitors<br>コンデンサ数</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{dataset_info.get('Normal Samples', 'N/A')}</div>
                        <div class="metric-label">Normal Samples<br>正常サンプル</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{dataset_info.get('Anomalous Samples', 'N/A')}</div>
                        <div class="metric-label">Anomalous Samples<br>異常サンプル</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="section">
            <h3>🎯 Performance Overview / 性能概要</h3>
            <div class="plot-container">
                <img src="data:image/png;base64,{performance_plot}" alt="Performance Summary">
            </div>
        </div>

        <div class="section">
            <h3>🔍 Requirements Compliance / 要件適合性</h3>
            <div class="plot-container">
                <img src="data:image/png;base64,{compliance_plot}" alt="Requirements Compliance">
            </div>
            
            <table class="requirements-table">
                <thead>
                    <tr>
                        <th>Requirement / 要件</th>
                        <th>Target / 目標</th>
                        <th>Validation / 検証</th>
                        <th>Test / テスト</th>
                        <th>Status / 状態</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>False Positive Rate<br>偽陽性率</td>
                        <td>&lt; 5%</td>
                        <td class="{'pass' if val_fpr < 0.05 else 'fail'}">{val_fpr:.4f} ({val_fpr*100:.2f}%)</td>
                        <td class="{'pass' if test_fpr < 0.05 else 'fail'}">{test_fpr:.4f} ({test_fpr*100:.2f}%)</td>
                        <td class="{'pass' if fpr_compliance else 'fail'}">{'✓ PASSED' if fpr_compliance else '✗ FAILED'}</td>
                    </tr>
                    <tr>
                        <td>RMSE Performance<br>RMSE性能</td>
                        <td>&lt; 50 cycles</td>
                        <td class="{'pass' if val_rmse < 50 else 'fail'}">{val_rmse:.2f} cycles</td>
                        <td class="{'pass' if test_rmse < 50 else 'fail'}">{test_rmse:.2f} cycles</td>
                        <td class="{'pass' if rmse_reasonable else 'fail'}">{'✓ PASSED' if rmse_reasonable else '✗ FAILED'}</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="section">
            <h3>📊 RUL Regression Performance / RUL回帰性能</h3>
            <div class="metrics-grid">
        """
        
        # Add RUL metrics for each dataset
        for dataset_name, results in rul_results.items():
            html_content += f"""
                <div class="metric-card">
                    <h4>{dataset_name.upper()} Dataset</h4>
                    <div class="metric-value">{results['rmse']:.2f}</div>
                    <div class="metric-label">RMSE (cycles)</div>
                    <div style="margin-top: 10px;">
                        <small>MAE: {results['mae']:.2f} | R²: {results['r2']:.3f}</small>
                    </div>
                </div>
            """
        
        html_content += f"""
            </div>
            <div class="plot-container">
                <img src="data:image/png;base64,{rul_analysis_plot}" alt="RUL Performance Analysis">
            </div>
        </div>

        <div class="section">
            <h3>🚨 Anomaly Detection Performance / 異常検知性能</h3>
            <div class="metrics-grid">
        """
        
        # Add anomaly detection metrics for each dataset
        for dataset_name, results in anomaly_results.items():
            html_content += f"""
                <div class="metric-card">
                    <h4>{dataset_name.upper()} Dataset</h4>
                    <div class="metric-value">{results['fpr']:.4f}</div>
                    <div class="metric-label">FPR</div>
                    <div style="margin-top: 10px;">
                        <small>TPR: {results['tpr']:.3f} | F1: {results['f1']:.3f}</small>
                    </div>
                </div>
            """
        
        html_content += f"""
            </div>
            <div class="plot-container">
                <img src="data:image/png;base64,{confusion_matrices_plot}" alt="Confusion Matrices">
            </div>
        </div>
        """
        
        # Add feature importance if available
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:15]
            html_content += f"""
        <div class="section">
            <h3>🔧 Feature Importance / 特徴量重要度</h3>
            <div class="feature-importance">
            """
            for feature, importance in top_features:
                html_content += f"""
                <div class="feature-item">
                    <span>{feature}</span>
                    <span>{importance:.4f}</span>
                </div>
                """
            html_content += """
            </div>
        </div>
            """
        
        html_content += f"""
        <div class="section">
            <h3>📝 Detailed Results / 詳細結果</h3>
            <h4>RUL Regression Detailed Metrics / RUL回帰詳細指標</h4>
            <table class="requirements-table">
                <thead>
                    <tr>
                        <th>Dataset</th>
                        <th>RMSE</th>
                        <th>MAE</th>
                        <th>R²</th>
                        <th>MAPE (%)</th>
                        <th>Max Error</th>
                        <th>Samples</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for dataset_name, results in rul_results.items():
            html_content += f"""
                    <tr>
                        <td>{dataset_name.upper()}</td>
                        <td>{results['rmse']:.3f}</td>
                        <td>{results['mae']:.3f}</td>
                        <td>{results['r2']:.3f}</td>
                        <td>{results['mape']:.2f}</td>
                        <td>{results['max_error']:.3f}</td>
                        <td>{results['n_samples']}</td>
                    </tr>
            """
        
        html_content += f"""
                </tbody>
            </table>
            
            <h4>Anomaly Detection Detailed Metrics / 異常検知詳細指標</h4>
            <table class="requirements-table">
                <thead>
                    <tr>
                        <th>Dataset</th>
                        <th>FPR</th>
                        <th>TPR</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1</th>
                        <th>ROC AUC</th>
                        <th>PR AUC</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for dataset_name, results in anomaly_results.items():
            html_content += f"""
                    <tr>
                        <td>{dataset_name.upper()}</td>
                        <td>{results['fpr']:.4f}</td>
                        <td>{results['tpr']:.4f}</td>
                        <td>{results['precision']:.4f}</td>
                        <td>{results['recall']:.4f}</td>
                        <td>{results['f1']:.4f}</td>
                        <td>{results['roc_auc']:.4f}</td>
                        <td>{results['pr_auc']:.4f}</td>
                    </tr>
            """
        
        html_content += f"""
                </tbody>
            </table>
        </div>

        <div class="section">
            <h3>💡 Conclusions and Recommendations / 結論と推奨事項</h3>
            <h4>English Summary:</h4>
            <ul>
                <li><strong>Overall Performance:</strong> The system {'meets' if overall_status else 'does not meet'} the primary requirements.</li>
                <li><strong>FPR Achievement:</strong> {'Successfully achieved' if fpr_compliance else 'Failed to achieve'} the target FPR &lt; 5% across validation and test sets.</li>
                <li><strong>RUL Accuracy:</strong> {'Achieved reasonable' if rmse_reasonable else 'Did not achieve optimal'} RMSE performance for practical deployment.</li>
                <li><strong>Model Robustness:</strong> The ensemble approach provides {'good' if overall_status else 'limited'} generalization across different capacitors.</li>
            </ul>
            
            <h4>日本語要約:</h4>
            <ul>
                <li><strong>全体的な性能:</strong> システムは主要要件を{'満たしています' if overall_status else '満たしていません'}。</li>
                <li><strong>FPR達成:</strong> 検証セットとテストセットでFPR &lt; 5%の目標を{'達成しました' if fpr_compliance else '達成できませんでした'}。</li>
                <li><strong>RUL精度:</strong> 実用的な展開のための{'合理的な' if rmse_reasonable else '最適でない'}RMSE性能を{'達成しました' if rmse_reasonable else '達成できませんでした'}。</li>
                <li><strong>モデルの堅牢性:</strong> アンサンブルアプローチは異なるコンデンサ間で{'良好な' if overall_status else '限定的な'}汎化性能を提供します。</li>
            </ul>
        </div>

        <div class="timestamp">
            <p>Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            レポート生成日時: {datetime.now().strftime('%Y年%m月%d日 %H時%M分%S秒')}</p>
        </div>
    </div>
</body>
</html>
        """
        
        return html_content
    
    def run_comprehensive_evaluation_and_report(self, data_path: Path) -> bool:
        """Run comprehensive evaluation and generate human-readable report"""
        try:
            logger.info("Starting comprehensive evaluation and report generation...")
            
            # Initialize comprehensive evaluator
            comp_evaluator = ComprehensiveModelEvaluator(self.output_dir)
            
            # Load and prepare data
            logger.info("Loading and preparing ES12 data...")
            full_dataset = comp_evaluator.load_and_prepare_real_data(data_path)
            
            # Split dataset
            logger.info("Splitting dataset...")
            test_capacitors = ["ES12C7", "ES12C8"]
            train_val_dataset, test_dataset = full_dataset.split_by_capacitor(test_capacitors)
            
            val_capacitors = ["ES12C6"]
            train_dataset, val_dataset = train_val_dataset.split_by_capacitor(val_capacitors)
            
            datasets = {
                'train': train_dataset,
                'val': val_dataset,
                'test': test_dataset
            }
            
            dataset_info = {
                'Total Samples': full_dataset.n_samples,
                'Features': full_dataset.n_features,
                'Train Samples': train_dataset.n_samples,
                'Validation Samples': val_dataset.n_samples,
                'Test Samples': test_dataset.n_samples,
                'Capacitors': len(set(full_dataset.capacitor_ids)),
                'Normal Samples': int(np.sum(full_dataset.anomaly_labels == 0)),
                'Anomalous Samples': int(np.sum(full_dataset.anomaly_labels == 1))
            }
            
            # Train models
            logger.info("Training models...")
            rul_model, isolation_forest, ocsvm = comp_evaluator.train_models(train_dataset, val_dataset)
            
            # Evaluate models
            logger.info("Evaluating models...")
            rul_results = comp_evaluator.evaluate_rul_regression(rul_model, datasets)
            anomaly_results = comp_evaluator.evaluate_anomaly_detection(isolation_forest, ocsvm, datasets, train_dataset)
            
            # Get feature importance
            feature_importance = None
            try:
                feature_importance = rul_model.get_feature_importance()
            except Exception as e:
                logger.warning(f"Could not get feature importance: {e}")
            
            # Generate HTML report
            logger.info("Generating human-readable HTML report...")
            html_report = self.generate_html_report(rul_results, anomaly_results, dataset_info, feature_importance)
            
            # Save HTML report
            html_file = self.output_dir / "html_reports" / "comprehensive_evaluation_report.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_report)
            
            # Save detailed results as JSON
            results_data = {
                'dataset_info': dataset_info,
                'rul_results': rul_results,
                'anomaly_results': anomaly_results,
                'feature_importance': feature_importance,
                'timestamp': datetime.now().isoformat()
            }
            
            results_file = self.output_dir / "data" / "detailed_results.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results_data, f, indent=2, ensure_ascii=False)
            
            # Check overall success
            val_fpr = anomaly_results.get('val', {}).get('fpr', 1.0)
            test_fpr = anomaly_results.get('test', {}).get('fpr', 1.0)
            val_rmse = rul_results.get('val', {}).get('rmse', float('inf'))
            test_rmse = rul_results.get('test', {}).get('rmse', float('inf'))
            
            success = (val_fpr < 0.05 and test_fpr < 0.05 and 
                      val_rmse < 50 and test_rmse < 50)
            
            if success:
                logger.info("🎉 COMPREHENSIVE EVALUATION AND REPORT GENERATION COMPLETED SUCCESSFULLY!")
                print("\\n🎉 包括的評価とレポート生成が成功しました！")
            else:
                logger.warning("⚠️ Some requirements not met, but report generated successfully")
                print("\\n⚠️ 一部の要件が満たされていませんが、レポートは正常に生成されました")
            
            logger.info(f"HTML Report saved to: {html_file}")
            logger.info(f"Detailed results saved to: {results_file}")
            print(f"\\nHTMLレポート保存先: {html_file}")
            print(f"詳細結果保存先: {results_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            print(f"\\n❌ レポート生成が失敗しました: {e}")
            return False


def main():
    """Main function"""
    # Set up logging
    setup_logging(log_file="human_readable_report.log", level=logging.INFO)
    logger.info("Starting human-readable report generation")
    
    # Define paths
    data_path = Path("~/work/CapacitorElectricalStress/eda_kiro/data/raw/ES12.mat").expanduser()
    output_dir = Path(__file__).parent.parent / "output" / "human_readable_reports"
    
    # Initialize report generator
    report_generator = HumanReadableReportGenerator(output_dir)
    
    # Run comprehensive evaluation and generate report
    success = report_generator.run_comprehensive_evaluation_and_report(data_path)
    
    if success:
        print("\\n" + "="*80)
        print("📊 HUMAN-READABLE COMPREHENSIVE REPORT GENERATED SUCCESSFULLY!")
        print("📊 人間向け包括的レポートが正常に生成されました！")
        print("="*80)
        print(f"\\n📁 Output Directory / 出力ディレクトリ: {output_dir}")
        print(f"🌐 HTML Report / HTMLレポート: {output_dir}/html_reports/comprehensive_evaluation_report.html")
        print(f"📊 Detailed Data / 詳細データ: {output_dir}/data/detailed_results.json")
        print("\\n💡 Open the HTML file in your web browser to view the interactive report!")
        print("💡 インタラクティブなレポートを表示するには、HTMLファイルをWebブラウザで開いてください！")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)