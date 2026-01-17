"""Model evaluator for comprehensive model evaluation and reporting."""

from pathlib import Path
from typing import Dict, Optional, Union, TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

if TYPE_CHECKING:
    from ..models.primary_model import PrimaryModel
    from ..models.secondary_model import SecondaryModel


class ModelEvaluator:
    """
    Comprehensive model evaluator for Primary and Secondary models.
    
    Provides evaluation metrics, visualizations, and report generation.
    """
    
    def __init__(self):
        """Initialize ModelEvaluator."""
        sns.set_style("whitegrid")
    
    def evaluate_primary_model(
        self,
        model,  # PrimaryModel
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict:
        """
        Evaluate Primary Model (anomaly detection).
        
        Args:
            model: Trained Primary Model
            X_test: Test features
            y_test: Test labels
        
        Returns:
            Dictionary containing evaluation metrics
        """
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "y_true": y_test.values,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
        
        return metrics
    
    def evaluate_secondary_model(
        self,
        model,  # SecondaryModel
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict:
        """
        Evaluate Secondary Model (RUL prediction).
        
        Args:
            model: Trained Secondary Model
            X_test: Test features
            y_test: Test RUL values
        
        Returns:
            Dictionary containing evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        metrics = {
            "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2": r2_score(y_test, y_pred),
            "mape": self._calculate_mape(y_test.values, y_pred),
            "y_true": y_test.values,
            "y_pred": y_pred,
        }
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Union[str, Path],
        title: str = "Confusion Matrix"
    ) -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
            title: Plot title
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Normal', 'Abnormal'],
            yticklabels=['Normal', 'Abnormal'],
            ax=ax
        )
        
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save_path: Union[str, Path],
        title: str = "ROC Curve"
    ) -> None:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            save_path: Path to save the plot
            title: Plot title
        """
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
               label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_prediction_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: Union[str, Path],
        title: str = "Predicted vs Actual RUL"
    ) -> None:
        """
        Plot predicted vs actual values.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            save_path: Path to save the plot
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        ax.scatter(y_true, y_pred, alpha=0.5, s=30, color='steelblue')
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel('Actual RUL', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted RUL', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(
        self,
        primary_metrics: Dict,
        secondary_metrics: Dict,
        output_path: Union[str, Path],
        dataset_info: Optional[Dict] = None
    ) -> None:
        """
        Generate comprehensive evaluation report.
        
        Args:
            primary_metrics: Primary model metrics
            secondary_metrics: Secondary model metrics
            output_path: Path to save the report
            dataset_info: Optional dataset information
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("# RUL Prediction Model - Baseline Evaluation Report\n\n")
            f.write("## 📅 Report Information\n\n")
            f.write(f"- **Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- **Model Type**: Random Forest (Baseline)\n")
            f.write(f"- **Dataset**: ES12\n\n")
            
            # Dataset information
            if dataset_info:
                f.write("## 📊 Dataset Information\n\n")
                for key, value in dataset_info.items():
                    f.write(f"- **{key}**: {value}\n")
                f.write("\n")
            
            # Primary Model Results
            f.write("## 🎯 Primary Model (Anomaly Detection)\n\n")
            f.write("### Performance Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| Accuracy | {primary_metrics['accuracy']:.4f} |\n")
            f.write(f"| Precision | {primary_metrics['precision']:.4f} |\n")
            f.write(f"| Recall | {primary_metrics['recall']:.4f} |\n")
            f.write(f"| F1-Score | {primary_metrics['f1_score']:.4f} |\n")
            f.write(f"| ROC-AUC | {primary_metrics['roc_auc']:.4f} |\n\n")
            
            # Target achievement
            target_f1 = 0.80
            if primary_metrics['f1_score'] >= target_f1:
                f.write(f"✅ **Target Achieved**: F1-Score ({primary_metrics['f1_score']:.4f}) >= {target_f1}\n\n")
            else:
                f.write(f"⚠️ **Target Not Met**: F1-Score ({primary_metrics['f1_score']:.4f}) < {target_f1}\n\n")
            
            # Confusion Matrix
            cm = primary_metrics['confusion_matrix']
            f.write("### Confusion Matrix\n\n")
            f.write("```\n")
            f.write(f"True Negative:  {cm[0][0]:4d}  |  False Positive: {cm[0][1]:4d}\n")
            f.write(f"False Negative: {cm[1][0]:4d}  |  True Positive:  {cm[1][1]:4d}\n")
            f.write("```\n\n")
            
            # Secondary Model Results
            f.write("## 📈 Secondary Model (RUL Prediction)\n\n")
            f.write("### Performance Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            f.write(f"| MAE | {secondary_metrics['mae']:.4f} |\n")
            f.write(f"| RMSE | {secondary_metrics['rmse']:.4f} |\n")
            f.write(f"| R² | {secondary_metrics['r2']:.4f} |\n")
            f.write(f"| MAPE | {secondary_metrics['mape']:.2f}% |\n\n")
            
            # Target achievement
            target_mape = 20.0
            if secondary_metrics['mape'] <= target_mape:
                f.write(f"✅ **Target Achieved**: MAPE ({secondary_metrics['mape']:.2f}%) <= {target_mape}%\n\n")
            else:
                f.write(f"⚠️ **Target Not Met**: MAPE ({secondary_metrics['mape']:.2f}%) > {target_mape}%\n\n")
            
            # Summary
            f.write("## 📝 Summary\n\n")
            f.write("### Strengths\n\n")
            
            if primary_metrics['f1_score'] >= target_f1:
                f.write("- ✅ Primary Model achieves excellent anomaly detection performance\n")
            if secondary_metrics['r2'] >= 0.9:
                f.write("- ✅ Secondary Model shows strong R² score\n")
            
            f.write("\n### Areas for Improvement\n\n")
            
            if secondary_metrics['mape'] > target_mape:
                f.write("- ⚠️ MAPE exceeds target, particularly for low RUL values\n")
                f.write("- 💡 Consider: Better handling of end-of-life predictions\n")
            
            f.write("\n### Recommendations\n\n")
            f.write("1. Investigate prediction errors for RUL < 50 cycles\n")
            f.write("2. Consider alternative MAPE calculation (exclude RUL=0)\n")
            f.write("3. Explore hyperparameter tuning for improved performance\n")
            f.write("4. Add more training data from ES10 and ES14 datasets\n\n")
            
            f.write("---\n\n")
            f.write("**End of Report**\n")
    
    @staticmethod
    def _calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        mask = y_true != 0
        if not mask.any():
            return 0.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
