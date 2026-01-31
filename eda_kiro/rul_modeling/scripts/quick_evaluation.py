#!/usr/bin/env python3
"""
Quick Model Evaluation with Real ES12 Data

This script performs a quick evaluation using a subset of the real ES12 dataset
to verify that the feature extraction fixes work and the models can be trained.
"""

import sys
from pathlib import Path
import logging
import numpy as np
from datetime import datetime

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

from sklearn.metrics import (
    confusion_matrix, mean_squared_error, mean_absolute_error, r2_score
)

logger = logging.getLogger(__name__)


def main():
    """Quick evaluation function"""
    # Set up logging
    setup_logging(log_file="quick_evaluation.log", level=logging.INFO)
    logger.info("Starting quick model evaluation with real ES12 data")
    
    # Define paths
    data_path = Path("~/work/CapacitorElectricalStress/eda_kiro/data/raw/ES12.mat").expanduser()
    output_dir = Path(__file__).parent.parent / "output" / "quick_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize components
        data_loader = DataLoader()
        feature_extractor = FeatureExtractor(include_advanced=True, rolling_window=5)
        preprocessor = TimeSeriesPreprocessor(rolling_window=5, normalization="standard")
        
        # Load ES12 dataset
        logger.info("Loading ES12 dataset...")
        capacitor_data = data_loader.load_es12_dataset(data_path)
        logger.info(f"Successfully loaded {len(capacitor_data)} capacitors")
        
        # Process limited data (first 2 capacitors, first 30 cycles each)
        all_features = []
        all_capacitor_ids = []
        all_cycle_numbers = []
        all_rul_labels = []
        all_anomaly_labels = []
        
        cap_ids = list(capacitor_data.keys())[:2]  # First 2 capacitors
        max_cycles = 30  # First 30 cycles
        
        for cap_id in cap_ids:
            cap_data = capacitor_data[cap_id]
            logger.info(f"Processing {cap_id}: {min(max_cycles, cap_data.total_cycles)} cycles")
            
            cap_features = []
            for i in range(min(max_cycles, cap_data.total_cycles)):
                cycle = cap_data.cycles[i]
                
                # Get history for rolling features
                history_start = max(0, cycle.cycle_number - feature_extractor.rolling_window)
                history = [c for c in cap_data.cycles 
                          if history_start <= c.cycle_number < cycle.cycle_number]
                
                try:
                    # Extract features
                    features_dict = feature_extractor.extract_features(cycle, cap_id, history)
                    features = np.array(list(features_dict.values()))
                    
                    # Handle non-finite values
                    if np.any(~np.isfinite(features)):
                        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    cap_features.append(features)
                    
                    # Store metadata
                    all_capacitor_ids.append(cap_id)
                    all_cycle_numbers.append(cycle.cycle_number)
                    
                    # Create RUL labels
                    rul = max(0, cap_data.total_cycles - cycle.cycle_number)
                    all_rul_labels.append(rul)
                    
                    # Create anomaly labels
                    is_anomaly = 1 if cycle.cycle_number > ES12_CONFIG["normal_cycles"][1] else 0
                    all_anomaly_labels.append(is_anomaly)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract features for {cap_id} cycle {cycle.cycle_number}: {e}")
                    continue
            
            if cap_features:
                cap_features = np.array(cap_features)
                logger.info(f"Features shape for {cap_id}: {cap_features.shape}")
                
                # Normalize features
                cap_features_normalized = preprocessor.normalize_features(
                    cap_features, cap_id, fit=True
                )
                all_features.extend(cap_features_normalized)
                logger.info(f"Successfully processed {len(cap_features)} cycles for {cap_id}")
        
        # Create dataset
        features_array = np.array(all_features)
        rul_labels_array = np.array(all_rul_labels)
        cycle_numbers_array = np.array(all_cycle_numbers)
        anomaly_labels_array = np.array(all_anomaly_labels)
        
        dataset = TrainingDataset(
            capacitor_ids=all_capacitor_ids,
            features=features_array,
            rul_labels=rul_labels_array,
            cycle_numbers=cycle_numbers_array,
            anomaly_labels=anomaly_labels_array
        )
        
        logger.info(f"Created dataset with {dataset.n_samples} samples and {dataset.n_features} features")
        
        # Split dataset (simple split)
        n_train = int(0.6 * dataset.n_samples)
        n_val = int(0.2 * dataset.n_samples)
        
        train_features = dataset.features[:n_train]
        train_rul = dataset.rul_labels[:n_train]
        train_cycles = dataset.cycle_numbers[:n_train]
        train_anomaly = dataset.anomaly_labels[:n_train]
        
        val_features = dataset.features[n_train:n_train+n_val]
        val_rul = dataset.rul_labels[n_train:n_train+n_val]
        val_anomaly = dataset.anomaly_labels[n_train:n_train+n_val]
        
        test_features = dataset.features[n_train+n_val:]
        test_rul = dataset.rul_labels[n_train+n_val:]
        test_anomaly = dataset.anomaly_labels[n_train+n_val:]
        
        logger.info(f"Dataset splits: Train={len(train_features)}, Val={len(val_features)}, Test={len(test_features)}")
        
        # Train RUL model
        logger.info("Training RUL regression model...")
        rul_model = RULRegressionModel(model_type="xgboost")
        rul_model.train(
            X_train=train_features,
            y_train=train_rul,
            X_val=val_features,
            y_val=val_rul
        )
        
        # Train anomaly detection models
        logger.info("Training anomaly detection models...")
        normal_mask = train_cycles <= ES12_CONFIG["normal_cycles"][1]
        normal_features = train_features[normal_mask]
        
        logger.info(f"Training on {len(normal_features)} normal cycles")
        
        isolation_forest = IsolationForestDetector(contamination=0.05)
        isolation_forest.fit(normal_features)
        
        ocsvm = ImprovedOCSVM(nu=0.05, auto_tune=False)
        ocsvm.fit(normal_features)
        
        # Evaluate models
        logger.info("Evaluating models...")
        
        # RUL evaluation
        test_rul_pred = rul_model.predict(test_features)
        rmse = np.sqrt(mean_squared_error(test_rul, test_rul_pred))
        mae = mean_absolute_error(test_rul, test_rul_pred)
        r2 = r2_score(test_rul, test_rul_pred)
        
        # Anomaly evaluation
        if_train_scores = isolation_forest.predict_score(normal_features)
        ocsvm_train_scores = ocsvm.predict_score(normal_features)
        train_ensemble_scores = 0.5 * if_train_scores + 0.5 * ocsvm_train_scores
        threshold = np.percentile(train_ensemble_scores, 95)
        
        if_test_scores = isolation_forest.predict_score(test_features)
        ocsvm_test_scores = ocsvm.predict_score(test_features)
        test_ensemble_scores = 0.5 * if_test_scores + 0.5 * ocsvm_test_scores
        
        test_anomaly_pred = (test_ensemble_scores > threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(test_anomaly, test_anomaly_pred).ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # Generate report
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("QUICK MODEL EVALUATION REPORT")
        report_lines.append("クイックモデル評価レポート")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        report_lines.append("DATASET INFORMATION / データセット情報")
        report_lines.append("-" * 30)
        report_lines.append(f"Total Samples: {dataset.n_samples}")
        report_lines.append(f"Features: {dataset.n_features}")
        report_lines.append(f"Capacitors: {len(set(all_capacitor_ids))}")
        report_lines.append(f"Normal Samples: {np.sum(anomaly_labels_array == 0)}")
        report_lines.append(f"Anomalous Samples: {np.sum(anomaly_labels_array == 1)}")
        report_lines.append("")
        
        report_lines.append("RUL REGRESSION RESULTS / RUL回帰結果")
        report_lines.append("-" * 30)
        report_lines.append(f"RMSE: {rmse:.3f} cycles")
        report_lines.append(f"MAE:  {mae:.3f} cycles")
        report_lines.append(f"R²:   {r2:.3f}")
        report_lines.append("")
        
        report_lines.append("ANOMALY DETECTION RESULTS / 異常検知結果")
        report_lines.append("-" * 30)
        report_lines.append(f"True Positive Rate (TPR): {tpr:.4f}")
        report_lines.append(f"False Positive Rate (FPR): {fpr:.4f}")
        report_lines.append(f"Confusion Matrix:")
        report_lines.append(f"  TN: {tn:3d}  FP: {fp:3d}")
        report_lines.append(f"  FN: {fn:3d}  TP: {tp:3d}")
        report_lines.append("")
        
        report_lines.append("REQUIREMENTS CHECK / 要件チェック")
        report_lines.append("-" * 30)
        fpr_passed = fpr < 0.05
        report_lines.append(f"FPR < 5%: {'✓ PASSED' if fpr_passed else '✗ FAILED'} (Actual: {fpr:.4f})")
        rmse_passed = rmse < 50
        report_lines.append(f"RMSE < 50: {'✓ PASSED' if rmse_passed else '✗ FAILED'} (Actual: {rmse:.3f})")
        
        overall_passed = fpr_passed and rmse_passed
        report_lines.append("")
        report_lines.append(f"OVERALL: {'✓ PASSED' if overall_passed else '✗ FAILED'}")
        report_lines.append("")
        report_lines.append("=" * 60)
        
        report = "\n".join(report_lines)
        
        # Save and print report
        report_file = output_dir / "quick_evaluation_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        
        if overall_passed:
            logger.info("🎉 QUICK EVALUATION PASSED!")
            print("\n🎉 クイック評価が成功しました！")
        else:
            logger.warning("⚠️ Some requirements not met")
            print("\n⚠️ 一部の要件が満たされていません")
        
        logger.info(f"Results saved to: {output_dir}")
        print(f"\n結果は以下に保存されました: {output_dir}")
        
        return overall_passed
        
    except Exception as e:
        logger.error(f"Quick evaluation failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        print(f"\n❌ 評価が失敗しました: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)