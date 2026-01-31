#!/usr/bin/env python3
"""
Comprehensive Model Evaluation with Real ES12 Data

This script performs detailed model evaluation using the real ES12 dataset,
generating comprehensive reports including:
- Confusion Matrix
- ROC Curves
- Precision-Recall Curves
- Feature Importance Analysis
- SHAP Value Analysis
- Detailed Performance Metrics

実データ（ES12.mat）を使用した包括的なモデル評価を実行します。
"""

import sys
from pathlib import Path
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import json

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

# Import sklearn metrics for detailed evaluation
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score
)

logger = logging.getLogger(__name__)

# Set up matplotlib for Japanese font support
plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']


class ComprehensiveModelEvaluator:
    """包括的なモデル評価クラス"""
    
    def __init__(self, output_dir: Path):
        """
        Initialize the comprehensive evaluator
        
        Args:
            output_dir: Directory to save evaluation results
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different types of outputs
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        
        logger.info(f"Initialized comprehensive evaluator with output directory: {output_dir}")
    
    def load_and_prepare_real_data(self, data_path: Path) -> TrainingDataset:
        """
        Load and prepare real ES12 data for evaluation
        
        Args:
            data_path: Path to ES12.mat file
            
        Returns:
            Prepared training dataset
        """
        logger.info(f"Loading real ES12 data from {data_path}")
        
        # Initialize data loader
        data_loader = DataLoader()
        
        try:
            # Load ES12 dataset
            capacitor_data = data_loader.load_es12_dataset(data_path)
            logger.info(f"Successfully loaded {len(capacitor_data)} capacitors")
            
            # Initialize feature extraction components
            feature_extractor = FeatureExtractor(
                include_advanced=MODEL_CONFIG["feature_extraction"]["include_advanced"],
                rolling_window=MODEL_CONFIG["feature_extraction"]["rolling_window"]
            )
            
            preprocessor = TimeSeriesPreprocessor(
                rolling_window=MODEL_CONFIG["feature_extraction"]["rolling_window"],
                normalization=MODEL_CONFIG["feature_extraction"]["normalization"]
            )
            
            # Extract features for all capacitors and cycles
            all_features = []
            all_capacitor_ids = []
            all_cycle_numbers = []
            all_rul_labels = []
            all_anomaly_labels = []
            
            for cap_id, cap_data in capacitor_data.items():
                logger.info(f"Processing features for {cap_id} ({cap_data.total_cycles} cycles)")
                
                # Extract features for all cycles
                cap_features = []
                expected_feature_count = None
                
                for cycle in cap_data.cycles:
                    # Get history for rolling features
                    history_start = max(0, cycle.cycle_number - feature_extractor.rolling_window)
                    history = [c for c in cap_data.cycles 
                              if history_start <= c.cycle_number < cycle.cycle_number]
                    
                    try:
                        # Extract features
                        features_dict = feature_extractor.extract_features(cycle, cap_id, history)
                        features = np.array(list(features_dict.values()))
                        
                        # Validate feature vector length
                        if len(features) == 0:
                            logger.warning(f"Empty feature vector for {cap_id} cycle {cycle.cycle_number}")
                            continue
                        
                        # Set expected feature count from cycle with sufficient history (cycle 6+)
                        if expected_feature_count is None and cycle.cycle_number >= 6:
                            expected_feature_count = len(features)
                            logger.info(f"Expected feature count for {cap_id}: {expected_feature_count}")
                        
                        # Skip early cycles that don't have full feature set
                        if expected_feature_count is not None and len(features) != expected_feature_count:
                            logger.debug(
                                f"Skipping {cap_id} cycle {cycle.cycle_number}: "
                                f"expected {expected_feature_count}, got {len(features)}"
                            )
                            continue
                        
                        # Check for NaN or infinite values
                        if np.any(~np.isfinite(features)):
                            logger.warning(f"Non-finite features for {cap_id} cycle {cycle.cycle_number}")
                            # Replace NaN/inf with zeros
                            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        cap_features.append(features)
                        
                        # Store metadata
                        all_capacitor_ids.append(cap_id)
                        all_cycle_numbers.append(cycle.cycle_number)
                        
                        # Create RUL labels (remaining cycles)
                        rul = max(0, cap_data.total_cycles - cycle.cycle_number)
                        all_rul_labels.append(rul)
                        
                        # Create anomaly labels (cycles 1-10 are normal, rest are anomalous)
                        is_anomaly = 1 if cycle.cycle_number > ES12_CONFIG["normal_cycles"][1] else 0
                        all_anomaly_labels.append(is_anomaly)
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract features for {cap_id} cycle {cycle.cycle_number}: {e}")
                        continue
                
                if cap_features:
                    # Convert to numpy array and check shape consistency
                    try:
                        cap_features = np.array(cap_features)
                        logger.info(f"Features shape for {cap_id}: {cap_features.shape}")
                        
                        # Normalize features per capacitor
                        cap_features_normalized = preprocessor.normalize_features(
                            cap_features, cap_id, fit=True
                        )
                        all_features.extend(cap_features_normalized)
                        logger.info(f"Successfully processed {len(cap_features)} cycles for {cap_id}")
                    except ValueError as e:
                        logger.error(f"Feature array shape error for {cap_id}: {e}")
                        logger.error(f"Feature lengths: {[len(f) for f in cap_features]}")
                        # Skip this capacitor if features are inconsistent
                        # Remove the metadata we added for this capacitor
                        n_cycles_to_remove = len(cap_features)
                        all_capacitor_ids = all_capacitor_ids[:-n_cycles_to_remove]
                        all_cycle_numbers = all_cycle_numbers[:-n_cycles_to_remove]
                        all_rul_labels = all_rul_labels[:-n_cycles_to_remove]
                        all_anomaly_labels = all_anomaly_labels[:-n_cycles_to_remove]
                        continue
                else:
                    logger.warning(f"No features extracted for {cap_id}")
            
            # Convert to numpy arrays with shape validation
            if not all_features:
                raise ValueError("No features extracted from any capacitor")
            
            # Check feature consistency before creating array
            feature_lengths = [len(f) for f in all_features]
            if len(set(feature_lengths)) > 1:
                logger.error(f"Inconsistent feature lengths: {set(feature_lengths)}")
                logger.error("This indicates a bug in feature extraction")
                raise ValueError(f"Inconsistent feature vector lengths: {set(feature_lengths)}")
            
            features_array = np.array(all_features)
            rul_labels_array = np.array(all_rul_labels)
            cycle_numbers_array = np.array(all_cycle_numbers)
            anomaly_labels_array = np.array(all_anomaly_labels)
            
            # Create training dataset
            dataset = TrainingDataset(
                capacitor_ids=all_capacitor_ids,
                features=features_array,
                rul_labels=rul_labels_array,
                cycle_numbers=cycle_numbers_array,
                anomaly_labels=anomaly_labels_array
            )
            
            logger.info(f"Created real ES12 dataset:")
            logger.info(f"  - {dataset.n_samples} total samples")
            logger.info(f"  - {dataset.n_features} features per sample")
            logger.info(f"  - {len(set(all_capacitor_ids))} capacitors")
            logger.info(f"  - {np.sum(anomaly_labels_array == 0)} normal samples")
            logger.info(f"  - {np.sum(anomaly_labels_array == 1)} anomalous samples")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to load and prepare real ES12 data: {e}")
            raise
    
    def train_models(self, train_dataset: TrainingDataset, val_dataset: TrainingDataset) -> Tuple[RULRegressionModel, IsolationForestDetector, ImprovedOCSVM]:
        """
        Train RUL regression and anomaly detection models
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            
        Returns:
            Tuple of trained models
        """
        logger.info("Training models on real ES12 data...")
        
        # 1. Train RUL regression model
        logger.info("Training RUL regression model...")
        rul_model = RULRegressionModel(model_type=MODEL_CONFIG["rul_model"]["type"])
        rul_model.train(
            X_train=train_dataset.features,
            y_train=train_dataset.rul_labels,
            X_val=val_dataset.features,
            y_val=val_dataset.rul_labels
        )
        
        # 2. Train anomaly detection models
        logger.info("Training anomaly detection models...")
        
        # Get normal cycles from training data
        normal_cycles_mask = train_dataset.cycle_numbers <= ES12_CONFIG["normal_cycles"][1]
        normal_features = train_dataset.features[normal_cycles_mask]
        
        logger.info(f"Training on {len(normal_features)} normal cycles")
        
        # Train individual detectors
        isolation_forest = IsolationForestDetector(contamination=0.05)
        isolation_forest.fit(normal_features)
        
        ocsvm = ImprovedOCSVM(nu=0.05, auto_tune=False)
        ocsvm.fit(normal_features)
        
        logger.info("Model training completed")
        return rul_model, isolation_forest, ocsvm
    
    def evaluate_rul_regression(self, rul_model: RULRegressionModel, datasets: Dict[str, TrainingDataset]) -> Dict[str, Any]:
        """
        Evaluate RUL regression model with detailed metrics
        
        Args:
            rul_model: Trained RUL regression model
            datasets: Dictionary of datasets (train, val, test)
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Evaluating RUL regression model...")
        
        results = {}
        
        for dataset_name, dataset in datasets.items():
            logger.info(f"Evaluating on {dataset_name} set...")
            
            # Get predictions
            predictions = rul_model.predict(dataset.features)
            true_values = dataset.rul_labels
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(true_values, predictions))
            mae = mean_absolute_error(true_values, predictions)
            r2 = r2_score(true_values, predictions)
            
            # Calculate additional metrics
            mape = np.mean(np.abs((true_values - predictions) / np.maximum(true_values, 1))) * 100
            max_error = np.max(np.abs(true_values - predictions))
            
            results[dataset_name] = {
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'mape': float(mape),
                'max_error': float(max_error),
                'n_samples': len(true_values),
                'predictions': predictions.tolist(),
                'true_values': true_values.tolist()
            }
            
            logger.info(f"{dataset_name.upper()} - RMSE: {rmse:.3f}, MAE: {mae:.3f}, R²: {r2:.3f}")
        
        return results
    
    def evaluate_anomaly_detection(self, isolation_forest: IsolationForestDetector, ocsvm: ImprovedOCSVM, 
                                 datasets: Dict[str, TrainingDataset], train_dataset: TrainingDataset) -> Dict[str, Any]:
        """
        Evaluate anomaly detection models with detailed metrics
        
        Args:
            isolation_forest: Trained isolation forest detector
            ocsvm: Trained One-Class SVM detector
            datasets: Dictionary of datasets (train, val, test)
            train_dataset: Training dataset for threshold calculation
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("Evaluating anomaly detection models...")
        
        results = {}
        
        # Calculate threshold from training data
        normal_cycles_mask = train_dataset.cycle_numbers <= ES12_CONFIG["normal_cycles"][1]
        normal_features = train_dataset.features[normal_cycles_mask]
        
        if_train_scores = isolation_forest.predict_score(normal_features)
        ocsvm_train_scores = ocsvm.predict_score(normal_features)
        train_ensemble_scores = 0.5 * if_train_scores + 0.5 * ocsvm_train_scores
        threshold = np.percentile(train_ensemble_scores, 95)
        
        logger.info(f"Anomaly detection threshold: {threshold:.4f}")
        
        for dataset_name, dataset in datasets.items():
            logger.info(f"Evaluating anomaly detection on {dataset_name} set...")
            
            # Get predictions from both detectors
            if_scores = isolation_forest.predict_score(dataset.features)
            ocsvm_scores = ocsvm.predict_score(dataset.features)
            ensemble_scores = 0.5 * if_scores + 0.5 * ocsvm_scores
            
            # Convert to binary predictions
            binary_pred = (ensemble_scores > threshold).astype(int)
            true_labels = dataset.anomaly_labels
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(true_labels, binary_pred).ravel()
            
            # Calculate metrics
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Calculate rates
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
            tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
            
            # Calculate AUC if possible
            try:
                roc_auc = auc(*roc_curve(true_labels, ensemble_scores)[:2])
                pr_auc = average_precision_score(true_labels, ensemble_scores)
            except:
                roc_auc = 0.0
                pr_auc = 0.0
            
            results[dataset_name] = {
                'confusion_matrix': {
                    'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
                },
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'tpr': float(tpr),
                'fpr': float(fpr),
                'tnr': float(tnr),
                'fnr': float(fnr),
                'roc_auc': float(roc_auc),
                'pr_auc': float(pr_auc),
                'threshold': float(threshold),
                'n_samples': len(true_labels),
                'n_normal': int(np.sum(true_labels == 0)),
                'n_anomalous': int(np.sum(true_labels == 1)),
                'ensemble_scores': ensemble_scores.tolist(),
                'true_labels': true_labels.tolist(),
                'predictions': binary_pred.tolist()
            }
            
            logger.info(f"{dataset_name.upper()} - FPR: {fpr:.4f}, TPR: {tpr:.4f}, F1: {f1:.4f}")
        
        return results
    
    def generate_confusion_matrix_plot(self, anomaly_results: Dict[str, Any], dataset_name: str):
        """
        Generate confusion matrix visualization
        
        Args:
            anomaly_results: Anomaly detection results
            dataset_name: Name of the dataset
        """
        logger.info(f"Generating confusion matrix plot for {dataset_name}")
        
        cm_data = anomaly_results[dataset_name]['confusion_matrix']
        cm = np.array([[cm_data['tn'], cm_data['fp']], 
                       [cm_data['fn'], cm_data['tp']]])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Anomaly'],
                   yticklabels=['Normal', 'Anomaly'])
        plt.title(f'Confusion Matrix - {dataset_name.upper()} Set\n混同行列 - {dataset_name.upper()}セット', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label / 予測ラベル', fontsize=12)
        plt.ylabel('True Label / 実際のラベル', fontsize=12)
        
        # Add metrics text
        fpr = anomaly_results[dataset_name]['fpr']
        tpr = anomaly_results[dataset_name]['tpr']
        f1 = anomaly_results[dataset_name]['f1']
        
        plt.text(0.02, 0.98, f'FPR: {fpr:.4f}\nTPR: {tpr:.4f}\nF1: {f1:.4f}', 
                transform=plt.gca().transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / f"confusion_matrix_{dataset_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_roc_curve_plot(self, anomaly_results: Dict[str, Any]):
        """
        Generate ROC curve visualization
        
        Args:
            anomaly_results: Anomaly detection results for all datasets
        """
        logger.info("Generating ROC curve plot")
        
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green']
        for i, (dataset_name, results) in enumerate(anomaly_results.items()):
            true_labels = np.array(results['true_labels'])
            scores = np.array(results['ensemble_scores'])
            
            if len(np.unique(true_labels)) > 1:  # Only plot if both classes exist
                fpr, tpr, _ = roc_curve(true_labels, scores)
                roc_auc = results['roc_auc']
                
                plt.plot(fpr, tpr, color=colors[i], lw=2, 
                        label=f'{dataset_name.upper()} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR) / 偽陽性率', fontsize=12)
        plt.ylabel('True Positive Rate (TPR) / 真陽性率', fontsize=12)
        plt.title('ROC Curves - Anomaly Detection\nROC曲線 - 異常検知', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "roc_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_precision_recall_curve_plot(self, anomaly_results: Dict[str, Any]):
        """
        Generate Precision-Recall curve visualization
        
        Args:
            anomaly_results: Anomaly detection results for all datasets
        """
        logger.info("Generating Precision-Recall curve plot")
        
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'red', 'green']
        for i, (dataset_name, results) in enumerate(anomaly_results.items()):
            true_labels = np.array(results['true_labels'])
            scores = np.array(results['ensemble_scores'])
            
            if len(np.unique(true_labels)) > 1:  # Only plot if both classes exist
                precision, recall, _ = precision_recall_curve(true_labels, scores)
                pr_auc = results['pr_auc']
                
                plt.plot(recall, precision, color=colors[i], lw=2,
                        label=f'{dataset_name.upper()} (AUC = {pr_auc:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall / 再現率', fontsize=12)
        plt.ylabel('Precision / 適合率', fontsize=12)
        plt.title('Precision-Recall Curves - Anomaly Detection\n適合率-再現率曲線 - 異常検知', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "precision_recall_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_rul_prediction_plot(self, rul_results: Dict[str, Any]):
        """
        Generate RUL prediction vs actual plot
        
        Args:
            rul_results: RUL regression results for all datasets
        """
        logger.info("Generating RUL prediction plots")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, (dataset_name, results) in enumerate(rul_results.items()):
            true_values = np.array(results['true_values'])
            predictions = np.array(results['predictions'])
            
            axes[i].scatter(true_values, predictions, alpha=0.6, s=20)
            
            # Perfect prediction line
            max_val = max(np.max(true_values), np.max(predictions))
            axes[i].plot([0, max_val], [0, max_val], 'r--', lw=2, alpha=0.8)
            
            axes[i].set_xlabel('Actual RUL / 実際のRUL', fontsize=11)
            axes[i].set_ylabel('Predicted RUL / 予測RUL', fontsize=11)
            axes[i].set_title(f'{dataset_name.upper()} Set\nRMSE: {results["rmse"]:.2f}, R²: {results["r2"]:.3f}', 
                             fontsize=12, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('RUL Prediction vs Actual\nRUL予測 vs 実際の値', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "rul_predictions.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_feature_importance_plot(self, rul_model: RULRegressionModel):
        """
        Generate feature importance visualization
        
        Args:
            rul_model: Trained RUL regression model
        """
        logger.info("Generating feature importance plot")
        
        try:
            feature_importance = rul_model.get_feature_importance()
            if not feature_importance:
                logger.warning("No feature importance available")
                return
            
            # Get top 20 features
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = dict(sorted_features[:20])
            
            plt.figure(figsize=(12, 10))
            features = list(top_features.keys())
            importances = list(top_features.values())
            
            y_pos = np.arange(len(features))
            plt.barh(y_pos, importances, alpha=0.8)
            plt.yticks(y_pos, features)
            plt.xlabel('Feature Importance / 特徴量重要度', fontsize=12)
            plt.title('Top 20 Feature Importance - RUL Regression\n上位20特徴量重要度 - RUL回帰', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(True, alpha=0.3, axis='x')
            plt.tight_layout()
            plt.savefig(self.output_dir / "plots" / "feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not generate feature importance plot: {e}")
    
    def generate_comprehensive_report(self, rul_results: Dict[str, Any], anomaly_results: Dict[str, Any], 
                                    dataset_info: Dict[str, Any]) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            rul_results: RUL regression results
            anomaly_results: Anomaly detection results
            dataset_info: Dataset information
            
        Returns:
            Formatted report string
        """
        logger.info("Generating comprehensive evaluation report")
        
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("COMPREHENSIVE MODEL EVALUATION REPORT")
        report_lines.append("包括的モデル評価レポート")
        report_lines.append("=" * 100)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Dataset Information
        report_lines.append("DATASET INFORMATION / データセット情報")
        report_lines.append("-" * 50)
        for key, value in dataset_info.items():
            report_lines.append(f"{key}: {value}")
        report_lines.append("")
        
        # RUL Regression Results
        report_lines.append("RUL REGRESSION MODEL PERFORMANCE / RUL回帰モデル性能")
        report_lines.append("-" * 50)
        
        for dataset_name, results in rul_results.items():
            report_lines.append(f"{dataset_name.upper()} SET / {dataset_name.upper()}セット:")
            report_lines.append(f"  RMSE (Root Mean Square Error):     {results['rmse']:.3f} cycles")
            report_lines.append(f"  MAE (Mean Absolute Error):         {results['mae']:.3f} cycles")
            report_lines.append(f"  R² (Coefficient of Determination): {results['r2']:.3f}")
            report_lines.append(f"  MAPE (Mean Absolute Percentage Error): {results['mape']:.2f}%")
            report_lines.append(f"  Max Error:                         {results['max_error']:.3f} cycles")
            report_lines.append(f"  Number of Samples:                 {results['n_samples']}")
            report_lines.append("")
        
        # Anomaly Detection Results
        report_lines.append("ANOMALY DETECTION MODEL PERFORMANCE / 異常検知モデル性能")
        report_lines.append("-" * 50)
        
        for dataset_name, results in anomaly_results.items():
            cm = results['confusion_matrix']
            report_lines.append(f"{dataset_name.upper()} SET / {dataset_name.upper()}セット:")
            report_lines.append(f"  Confusion Matrix / 混同行列:")
            report_lines.append(f"    True Negative (TN):  {cm['tn']:4d}  |  False Positive (FP): {cm['fp']:4d}")
            report_lines.append(f"    False Negative (FN): {cm['fn']:4d}  |  True Positive (TP):  {cm['tp']:4d}")
            report_lines.append("")
            report_lines.append(f"  Performance Metrics / 性能指標:")
            report_lines.append(f"    Accuracy (精度):           {results['accuracy']:.4f}")
            report_lines.append(f"    Precision (適合率):        {results['precision']:.4f}")
            report_lines.append(f"    Recall (再現率):           {results['recall']:.4f}")
            report_lines.append(f"    F1-Score:                  {results['f1']:.4f}")
            report_lines.append(f"    True Positive Rate (TPR):  {results['tpr']:.4f}")
            report_lines.append(f"    False Positive Rate (FPR): {results['fpr']:.4f}")
            report_lines.append(f"    True Negative Rate (TNR):  {results['tnr']:.4f}")
            report_lines.append(f"    False Negative Rate (FNR): {results['fnr']:.4f}")
            report_lines.append(f"    ROC AUC:                   {results['roc_auc']:.4f}")
            report_lines.append(f"    PR AUC:                    {results['pr_auc']:.4f}")
            report_lines.append("")
        
        # Requirements Compliance
        report_lines.append("REQUIREMENTS COMPLIANCE / 要件適合性")
        report_lines.append("-" * 50)
        
        # Check FPR < 5% requirement
        val_fpr = anomaly_results.get('val', {}).get('fpr', 1.0)
        test_fpr = anomaly_results.get('test', {}).get('fpr', 1.0)
        
        fpr_val_passed = val_fpr < 0.05
        fpr_test_passed = test_fpr < 0.05
        
        report_lines.append(f"FPR < 5% (Validation): {'✓ PASSED' if fpr_val_passed else '✗ FAILED'} (Actual: {val_fpr:.4f})")
        report_lines.append(f"FPR < 5% (Test):       {'✓ PASSED' if fpr_test_passed else '✗ FAILED'} (Actual: {test_fpr:.4f})")
        
        # Check RMSE reasonableness
        val_rmse = rul_results.get('val', {}).get('rmse', float('inf'))
        test_rmse = rul_results.get('test', {}).get('rmse', float('inf'))
        rmse_threshold = 50  # Reasonable threshold for RUL prediction
        
        rmse_val_passed = val_rmse < rmse_threshold
        rmse_test_passed = test_rmse < rmse_threshold
        
        report_lines.append(f"RMSE Reasonable (< {rmse_threshold}) (Validation): {'✓ PASSED' if rmse_val_passed else '✗ FAILED'} (Actual: {val_rmse:.3f})")
        report_lines.append(f"RMSE Reasonable (< {rmse_threshold}) (Test):       {'✓ PASSED' if rmse_test_passed else '✗ FAILED'} (Actual: {test_rmse:.3f})")
        
        report_lines.append("")
        
        # Summary
        overall_passed = fpr_val_passed and fpr_test_passed and rmse_val_passed and rmse_test_passed
        report_lines.append("OVERALL ASSESSMENT / 総合評価")
        report_lines.append("-" * 50)
        report_lines.append(f"Overall Status: {'✓ PASSED' if overall_passed else '✗ FAILED'}")
        report_lines.append("")
        
        report_lines.append("Generated Files / 生成ファイル:")
        report_lines.append("- plots/confusion_matrix_*.png: Confusion matrices")
        report_lines.append("- plots/roc_curves.png: ROC curves")
        report_lines.append("- plots/precision_recall_curves.png: Precision-Recall curves")
        report_lines.append("- plots/rul_predictions.png: RUL prediction plots")
        report_lines.append("- plots/feature_importance.png: Feature importance plot")
        report_lines.append("- data/evaluation_results.json: Detailed results in JSON format")
        
        report_lines.append("")
        report_lines.append("=" * 100)
        
        return "\n".join(report_lines)


def main():
    """Main evaluation function"""
    # Set up logging
    setup_logging(log_file="comprehensive_evaluation.log", level=logging.INFO)
    logger.info("Starting comprehensive model evaluation with real ES12 data")
    
    # Define paths
    data_path = Path("~/work/CapacitorElectricalStress/eda_kiro/data/raw/ES12.mat").expanduser()
    output_dir = Path(__file__).parent.parent / "output" / "comprehensive_evaluation"
    
    try:
        # Initialize evaluator
        evaluator = ComprehensiveModelEvaluator(output_dir)
        
        # Load and prepare real data
        logger.info("Loading and preparing real ES12 data...")
        full_dataset = evaluator.load_and_prepare_real_data(data_path)
        
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
        
        logger.info(f"Dataset splits: Train={train_dataset.n_samples}, Val={val_dataset.n_samples}, Test={test_dataset.n_samples}")
        
        # Train models
        logger.info("Training models...")
        rul_model, isolation_forest, ocsvm = evaluator.train_models(train_dataset, val_dataset)
        
        # Evaluate models
        logger.info("Evaluating models...")
        rul_results = evaluator.evaluate_rul_regression(rul_model, datasets)
        anomaly_results = evaluator.evaluate_anomaly_detection(isolation_forest, ocsvm, datasets, train_dataset)
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        
        # Confusion matrices
        for dataset_name in datasets.keys():
            evaluator.generate_confusion_matrix_plot(anomaly_results, dataset_name)
        
        # ROC and PR curves
        evaluator.generate_roc_curve_plot(anomaly_results)
        evaluator.generate_precision_recall_curve_plot(anomaly_results)
        
        # RUL prediction plots
        evaluator.generate_rul_prediction_plot(rul_results)
        
        # Feature importance
        evaluator.generate_feature_importance_plot(rul_model)
        
        # Generate comprehensive report
        logger.info("Generating comprehensive report...")
        report = evaluator.generate_comprehensive_report(rul_results, anomaly_results, dataset_info)
        
        # Save report
        report_file = output_dir / "reports" / "comprehensive_evaluation_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Save detailed results as JSON
        results_data = {
            'dataset_info': dataset_info,
            'rul_results': rul_results,
            'anomaly_results': anomaly_results,
            'timestamp': datetime.now().isoformat()
        }
        
        results_file = output_dir / "data" / "evaluation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        # Print report
        print(report)
        
        # Check overall success
        val_fpr = anomaly_results.get('val', {}).get('fpr', 1.0)
        test_fpr = anomaly_results.get('test', {}).get('fpr', 1.0)
        val_rmse = rul_results.get('val', {}).get('rmse', float('inf'))
        test_rmse = rul_results.get('test', {}).get('rmse', float('inf'))
        
        success = (val_fpr < 0.05 and test_fpr < 0.05 and 
                  val_rmse < 50 and test_rmse < 50)
        
        if success:
            logger.info("🎉 COMPREHENSIVE MODEL EVALUATION PASSED!")
            print("\n🎉 包括的モデル評価が成功しました！")
        else:
            logger.warning("⚠️ Some requirements not met in comprehensive evaluation")
            print("\n⚠️ 一部の要件が満たされていません")
        
        logger.info(f"Results saved to: {output_dir}")
        print(f"\n結果は以下に保存されました: {output_dir}")
        
        return success
        
    except Exception as e:
        logger.error(f"Comprehensive evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\n❌ 評価が失敗しました: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)