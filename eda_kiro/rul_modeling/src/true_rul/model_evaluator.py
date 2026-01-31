"""
Model Evaluator for True RUL Prediction System

This module implements comprehensive model evaluation including:
- RMSE, MAE, R² for RUL predictions
- FPR, TPR, precision, recall for anomaly detection
- Generation of evaluation reports with all metrics

Requirements: 2.1, 5.3
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve,
    confusion_matrix, classification_report
)
import json
from datetime import datetime
from pathlib import Path

from .data_structures import TrainingDataset, PredictionResult
from .rul_regression_model import RULRegressionModel
from .ensemble_anomaly_detector import EnsembleAnomalyDetector

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation for RUL prediction system
    
    This class provides evaluation metrics for both RUL regression and
    anomaly detection models, generating detailed reports for analysis.
    
    Attributes:
        rul_metrics: Dictionary of RUL regression metrics
        anomaly_metrics: Dictionary of anomaly detection metrics
        evaluation_results: Complete evaluation results
    """
    
    def __init__(self):
        """Initialize the model evaluator"""
        self.rul_metrics: Dict[str, float] = {}
        self.anomaly_metrics: Dict[str, Any] = {}
        self.evaluation_results: Dict[str, Any] = {}
        
        logger.info("ModelEvaluator initialized")
    
    def evaluate_rul_model(
        self,
        model: RULRegressionModel,
        test_dataset: TrainingDataset,
        dataset_name: str = "test"
    ) -> Dict[str, float]:
        """
        Evaluate RUL regression model performance
        
        Args:
            model: Trained RUL regression model
            test_dataset: Test dataset for evaluation
            dataset_name: Name of the dataset being evaluated
            
        Returns:
            Dictionary with RUL regression metrics
            
        Raises:
            ValueError: If model is not trained or dataset is empty
        """
        if not model.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        if test_dataset.n_samples == 0:
            raise ValueError("Test dataset cannot be empty")
        
        logger.info(f"Evaluating RUL model on {dataset_name} set ({test_dataset.n_samples} samples)")
        
        # Get predictions
        try:
            predictions = model.predict(test_dataset.features)
            
            # Get confidence intervals if available
            if hasattr(model.model, 'predict_with_confidence'):
                pred_with_conf, lower_bounds, upper_bounds = model.predict_with_confidence(
                    test_dataset.features
                )
            else:
                pred_with_conf = predictions
                lower_bounds = predictions * 0.9  # Simple fallback
                upper_bounds = predictions * 1.1
                
        except Exception as e:
            logger.error(f"Failed to get predictions: {e}")
            raise RuntimeError(f"Failed to get predictions: {e}")
        
        # Ensure predictions are non-negative
        predictions = np.maximum(predictions, 0)
        
        # Calculate regression metrics
        true_values = test_dataset.rul_labels
        
        # Root Mean Square Error
        rmse = np.sqrt(mean_squared_error(true_values, predictions))
        
        # Mean Absolute Error
        mae = mean_absolute_error(true_values, predictions)
        
        # R² Score (coefficient of determination)
        r2 = r2_score(true_values, predictions)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((true_values - predictions) / (true_values + 1e-8))) * 100
        
        # Maximum Error
        max_error = np.max(np.abs(true_values - predictions))
        
        # Median Absolute Error
        median_ae = np.median(np.abs(true_values - predictions))
        
        # Confidence interval metrics
        # Coverage: percentage of true values within confidence intervals
        within_ci = np.sum(
            (true_values >= lower_bounds) & (true_values <= upper_bounds)
        ) / len(true_values)
        
        # Average confidence interval width
        avg_ci_width = np.mean(upper_bounds - lower_bounds)
        
        # Compile metrics
        rul_metrics = {
            f"{dataset_name}_rmse": float(rmse),
            f"{dataset_name}_mae": float(mae),
            f"{dataset_name}_r2": float(r2),
            f"{dataset_name}_mape": float(mape),
            f"{dataset_name}_max_error": float(max_error),
            f"{dataset_name}_median_ae": float(median_ae),
            f"{dataset_name}_ci_coverage": float(within_ci),
            f"{dataset_name}_avg_ci_width": float(avg_ci_width),
            f"{dataset_name}_n_samples": test_dataset.n_samples,
        }
        
        # Store detailed results for analysis
        detailed_results = {
            "true_values": true_values.tolist(),
            "predictions": predictions.tolist(),
            "lower_bounds": lower_bounds.tolist(),
            "upper_bounds": upper_bounds.tolist(),
            "residuals": (true_values - predictions).tolist(),
            "capacitor_ids": test_dataset.capacitor_ids,
            "cycle_numbers": test_dataset.cycle_numbers.tolist(),
        }
        
        self.rul_metrics.update(rul_metrics)
        self.evaluation_results[f"rul_{dataset_name}_detailed"] = detailed_results
        
        logger.info(
            f"RUL evaluation completed - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.3f}"
        )
        
        return rul_metrics
    
    def evaluate_anomaly_detector(
        self,
        detector: EnsembleAnomalyDetector,
        test_dataset: TrainingDataset,
        dataset_name: str = "test"
    ) -> Dict[str, Any]:
        """
        Evaluate anomaly detection model performance
        
        Args:
            detector: Trained anomaly detector
            test_dataset: Test dataset for evaluation
            dataset_name: Name of the dataset being evaluated
            
        Returns:
            Dictionary with anomaly detection metrics
            
        Raises:
            ValueError: If detector is not fitted or dataset is empty
        """
        if not detector.is_fitted:
            raise ValueError("Detector must be fitted before evaluation")
        
        if test_dataset.n_samples == 0:
            raise ValueError("Test dataset cannot be empty")
        
        if test_dataset.anomaly_labels is None:
            raise ValueError("Test dataset must have anomaly labels for evaluation")
        
        logger.info(f"Evaluating anomaly detector on {dataset_name} set ({test_dataset.n_samples} samples)")
        
        # Get predictions
        try:
            binary_predictions, anomaly_scores, info = detector.predict(test_dataset.features)
        except Exception as e:
            logger.error(f"Failed to get anomaly predictions: {e}")
            raise RuntimeError(f"Failed to get anomaly predictions: {e}")
        
        true_labels = test_dataset.anomaly_labels
        
        # Calculate classification metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, binary_predictions, average='binary', zero_division=0
        )
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(true_labels, binary_predictions).ravel()
        
        # Calculate rates
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Sensitivity)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate (Specificity)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
        
        # ROC AUC
        try:
            roc_auc = roc_auc_score(true_labels, anomaly_scores)
            fpr_curve, tpr_curve, thresholds = roc_curve(true_labels, anomaly_scores)
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {e}")
            roc_auc = 0.0
            fpr_curve = np.array([])
            tpr_curve = np.array([])
            thresholds = np.array([])
        
        # Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        
        # Compile metrics
        anomaly_metrics = {
            f"{dataset_name}_precision": float(precision),
            f"{dataset_name}_recall": float(recall),
            f"{dataset_name}_f1": float(f1),
            f"{dataset_name}_accuracy": float(accuracy),
            f"{dataset_name}_tpr": float(tpr),
            f"{dataset_name}_fpr": float(fpr),
            f"{dataset_name}_tnr": float(tnr),
            f"{dataset_name}_fnr": float(fnr),
            f"{dataset_name}_roc_auc": float(roc_auc),
            f"{dataset_name}_tp": int(tp),
            f"{dataset_name}_fp": int(fp),
            f"{dataset_name}_tn": int(tn),
            f"{dataset_name}_fn": int(fn),
            f"{dataset_name}_n_samples": test_dataset.n_samples,
            f"{dataset_name}_n_anomalies": int(np.sum(true_labels)),
            f"{dataset_name}_n_normal": int(np.sum(1 - true_labels)),
        }
        
        # Store detailed results for analysis
        detailed_results = {
            "true_labels": true_labels.tolist(),
            "binary_predictions": binary_predictions.tolist(),
            "anomaly_scores": anomaly_scores.tolist(),
            "capacitor_ids": test_dataset.capacitor_ids,
            "cycle_numbers": test_dataset.cycle_numbers.tolist(),
            "confusion_matrix": {
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
            },
            "roc_curve": {
                "fpr": fpr_curve.tolist(),
                "tpr": tpr_curve.tolist(),
                "thresholds": thresholds.tolist()
            },
            "detector_info": info,
        }
        
        self.anomaly_metrics.update(anomaly_metrics)
        self.evaluation_results[f"anomaly_{dataset_name}_detailed"] = detailed_results
        
        logger.info(
            f"Anomaly detection evaluation completed - "
            f"FPR: {fpr:.3f}, TPR: {tpr:.3f}, F1: {f1:.3f}, AUC: {roc_auc:.3f}"
        )
        
        return anomaly_metrics
    
    def evaluate_complete_system(
        self,
        rul_model: RULRegressionModel,
        anomaly_detector: EnsembleAnomalyDetector,
        train_dataset: TrainingDataset,
        val_dataset: TrainingDataset,
        test_dataset: TrainingDataset
    ) -> Dict[str, Any]:
        """
        Evaluate the complete RUL prediction system
        
        Args:
            rul_model: Trained RUL regression model
            anomaly_detector: Trained anomaly detector
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Test dataset
            
        Returns:
            Complete evaluation results
        """
        logger.info("Starting complete system evaluation")
        
        # Evaluate RUL model on all datasets
        train_rul_metrics = self.evaluate_rul_model(rul_model, train_dataset, "train")
        val_rul_metrics = self.evaluate_rul_model(rul_model, val_dataset, "val")
        test_rul_metrics = self.evaluate_rul_model(rul_model, test_dataset, "test")
        
        # Evaluate anomaly detector on all datasets
        train_anomaly_metrics = self.evaluate_anomaly_detector(anomaly_detector, train_dataset, "train")
        val_anomaly_metrics = self.evaluate_anomaly_detector(anomaly_detector, val_dataset, "val")
        test_anomaly_metrics = self.evaluate_anomaly_detector(anomaly_detector, test_dataset, "test")
        
        # Compile complete results
        complete_results = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "rul_metrics": {
                **train_rul_metrics,
                **val_rul_metrics,
                **test_rul_metrics
            },
            "anomaly_metrics": {
                **train_anomaly_metrics,
                **val_anomaly_metrics,
                **test_anomaly_metrics
            },
            "model_info": {
                "rul_model": rul_model.get_model_info(),
                "anomaly_detector": anomaly_detector.get_model_info()
            },
            "dataset_info": {
                "train_samples": train_dataset.n_samples,
                "val_samples": val_dataset.n_samples,
                "test_samples": test_dataset.n_samples,
                "n_features": train_dataset.n_features,
            }
        }
        
        # Add detailed results
        complete_results.update(self.evaluation_results)
        
        # Check if FPR requirement is met
        test_fpr = test_anomaly_metrics.get("test_fpr", 1.0)
        fpr_requirement_met = test_fpr < 0.05
        
        complete_results["requirements_check"] = {
            "fpr_less_than_5_percent": fpr_requirement_met,
            "actual_test_fpr": test_fpr,
            "fpr_requirement": 0.05
        }
        
        self.evaluation_results = complete_results
        
        logger.info(
            f"Complete system evaluation finished - "
            f"Test FPR: {test_fpr:.3f} ({'✓' if fpr_requirement_met else '✗'} < 5%)"
        )
        
        return complete_results
    
    def generate_evaluation_report(
        self,
        save_path: Optional[Path] = None,
        include_detailed: bool = True
    ) -> str:
        """
        Generate a comprehensive evaluation report
        
        Args:
            save_path: Optional path to save the report
            include_detailed: Whether to include detailed results
            
        Returns:
            Formatted evaluation report as string
        """
        if not self.evaluation_results:
            raise ValueError("No evaluation results available. Run evaluation first.")
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("TRUE RUL PREDICTION SYSTEM - EVALUATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Requirements check
        if "requirements_check" in self.evaluation_results:
            req_check = self.evaluation_results["requirements_check"]
            report_lines.append("REQUIREMENTS COMPLIANCE")
            report_lines.append("-" * 40)
            fpr_status = "✓ PASSED" if req_check["fpr_less_than_5_percent"] else "✗ FAILED"
            report_lines.append(f"FPR < 5%: {fpr_status} (Actual: {req_check['actual_test_fpr']:.3f})")
            report_lines.append("")
        
        # RUL Model Performance
        if "rul_metrics" in self.evaluation_results:
            rul_metrics = self.evaluation_results["rul_metrics"]
            report_lines.append("RUL REGRESSION MODEL PERFORMANCE")
            report_lines.append("-" * 40)
            
            for dataset in ["train", "val", "test"]:
                if f"{dataset}_rmse" in rul_metrics:
                    report_lines.append(f"{dataset.upper()} SET:")
                    report_lines.append(f"  RMSE: {rul_metrics[f'{dataset}_rmse']:.2f}")
                    report_lines.append(f"  MAE:  {rul_metrics[f'{dataset}_mae']:.2f}")
                    report_lines.append(f"  R²:   {rul_metrics[f'{dataset}_r2']:.3f}")
                    report_lines.append(f"  MAPE: {rul_metrics[f'{dataset}_mape']:.1f}%")
                    report_lines.append(f"  CI Coverage: {rul_metrics[f'{dataset}_ci_coverage']:.1%}")
                    report_lines.append("")
        
        # Anomaly Detection Performance
        if "anomaly_metrics" in self.evaluation_results:
            anomaly_metrics = self.evaluation_results["anomaly_metrics"]
            report_lines.append("ANOMALY DETECTION MODEL PERFORMANCE")
            report_lines.append("-" * 40)
            
            for dataset in ["train", "val", "test"]:
                if f"{dataset}_fpr" in anomaly_metrics:
                    report_lines.append(f"{dataset.upper()} SET:")
                    report_lines.append(f"  FPR:       {anomaly_metrics[f'{dataset}_fpr']:.3f}")
                    report_lines.append(f"  TPR:       {anomaly_metrics[f'{dataset}_tpr']:.3f}")
                    report_lines.append(f"  Precision: {anomaly_metrics[f'{dataset}_precision']:.3f}")
                    report_lines.append(f"  Recall:    {anomaly_metrics[f'{dataset}_recall']:.3f}")
                    report_lines.append(f"  F1:        {anomaly_metrics[f'{dataset}_f1']:.3f}")
                    report_lines.append(f"  ROC AUC:   {anomaly_metrics[f'{dataset}_roc_auc']:.3f}")
                    report_lines.append("")
        
        # Dataset Information
        if "dataset_info" in self.evaluation_results:
            dataset_info = self.evaluation_results["dataset_info"]
            report_lines.append("DATASET INFORMATION")
            report_lines.append("-" * 40)
            report_lines.append(f"Training samples:   {dataset_info['train_samples']}")
            report_lines.append(f"Validation samples: {dataset_info['val_samples']}")
            report_lines.append(f"Test samples:       {dataset_info['test_samples']}")
            report_lines.append(f"Number of features: {dataset_info['n_features']}")
            report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Save report if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                f.write(report_text)
            
            # Also save detailed results as JSON
            if include_detailed:
                json_path = save_path.with_suffix('.json')
                with open(json_path, 'w') as f:
                    json.dump(self.evaluation_results, f, indent=2, default=str)
                
                logger.info(f"Evaluation report saved to {save_path}")
                logger.info(f"Detailed results saved to {json_path}")
        
        return report_text
    
    def get_summary_metrics(self) -> Dict[str, float]:
        """
        Get summary metrics for quick assessment
        
        Returns:
            Dictionary with key performance metrics
        """
        if not self.evaluation_results:
            return {}
        
        summary = {}
        
        # RUL metrics
        if "rul_metrics" in self.evaluation_results:
            rul_metrics = self.evaluation_results["rul_metrics"]
            summary.update({
                "test_rmse": rul_metrics.get("test_rmse", 0.0),
                "test_mae": rul_metrics.get("test_mae", 0.0),
                "test_r2": rul_metrics.get("test_r2", 0.0),
            })
        
        # Anomaly metrics
        if "anomaly_metrics" in self.evaluation_results:
            anomaly_metrics = self.evaluation_results["anomaly_metrics"]
            summary.update({
                "test_fpr": anomaly_metrics.get("test_fpr", 1.0),
                "test_tpr": anomaly_metrics.get("test_tpr", 0.0),
                "test_f1": anomaly_metrics.get("test_f1", 0.0),
                "test_roc_auc": anomaly_metrics.get("test_roc_auc", 0.0),
            })
        
        # Requirements compliance
        if "requirements_check" in self.evaluation_results:
            req_check = self.evaluation_results["requirements_check"]
            summary["fpr_requirement_met"] = req_check["fpr_less_than_5_percent"]
        
        return summary
    
    def reset(self):
        """Reset all evaluation results"""
        self.rul_metrics.clear()
        self.anomaly_metrics.clear()
        self.evaluation_results.clear()
        logger.info("Evaluation results reset")


def evaluate_models(
    rul_model: RULRegressionModel,
    anomaly_detector: EnsembleAnomalyDetector,
    train_dataset: TrainingDataset,
    val_dataset: TrainingDataset,
    test_dataset: TrainingDataset,
    save_report: bool = True,
    report_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Convenience function to evaluate models and generate report
    
    Args:
        rul_model: Trained RUL regression model
        anomaly_detector: Trained anomaly detector
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        save_report: Whether to save evaluation report
        report_path: Path to save report (optional)
        
    Returns:
        Complete evaluation results
    """
    evaluator = ModelEvaluator()
    
    results = evaluator.evaluate_complete_system(
        rul_model=rul_model,
        anomaly_detector=anomaly_detector,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset
    )
    
    if save_report:
        if report_path is None:
            report_path = Path("evaluation_report.txt")
        
        report_text = evaluator.generate_evaluation_report(
            save_path=report_path,
            include_detailed=True
        )
        
        print("Evaluation Report:")
        print("=" * 80)
        print(report_text)
    
    return results