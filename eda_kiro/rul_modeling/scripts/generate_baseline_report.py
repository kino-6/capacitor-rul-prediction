"""Script to generate comprehensive baseline evaluation report."""

import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.primary_model import PrimaryModel
from models.secondary_model import SecondaryModel
from evaluation.evaluator import ModelEvaluator


def main():
    """Generate baseline evaluation report."""
    print("=" * 80)
    print("BASELINE MODEL EVALUATION REPORT GENERATION")
    print("=" * 80)
    print()
    
    # Paths
    data_dir = Path(__file__).parent.parent / "output" / "features"
    model_dir = Path(__file__).parent.parent / "output" / "models"
    eval_dir = Path(__file__).parent.parent / "output" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    print("📂 Loading test data...")
    test_df = pd.read_csv(data_dir / "test.csv")
    
    feature_cols = [col for col in test_df.columns 
                   if col not in ["capacitor_id", "cycle", "is_abnormal", "rul"]]
    
    X_test = test_df[feature_cols]
    y_test_primary = test_df["is_abnormal"]
    y_test_secondary = test_df["rul"]
    
    print(f"  ✓ Test samples: {len(test_df)}")
    print(f"  ✓ Features: {len(feature_cols)}")
    print()
    
    # Load models
    print("📂 Loading trained models...")
    primary_model = PrimaryModel()
    primary_model.load(model_dir / "primary_model.pkl")
    print("  ✓ Primary Model loaded")
    
    secondary_model = SecondaryModel()
    secondary_model.load(model_dir / "secondary_model.pkl")
    print("  ✓ Secondary Model loaded")
    print()
    
    # Initialize evaluator
    print("🔧 Initializing evaluator...")
    evaluator = ModelEvaluator()
    print("  ✓ Evaluator ready")
    print()
    
    # Evaluate Primary Model
    print("📊 Evaluating Primary Model...")
    primary_metrics = evaluator.evaluate_primary_model(
        primary_model, X_test, y_test_primary
    )
    print("  ✓ Evaluation complete")
    print(f"    - Accuracy:  {primary_metrics['accuracy']:.4f}")
    print(f"    - F1-Score:  {primary_metrics['f1_score']:.4f}")
    print(f"    - ROC-AUC:   {primary_metrics['roc_auc']:.4f}")
    print()
    
    # Evaluate Secondary Model
    print("📊 Evaluating Secondary Model...")
    secondary_metrics = evaluator.evaluate_secondary_model(
        secondary_model, X_test, y_test_secondary
    )
    print("  ✓ Evaluation complete")
    print(f"    - MAE:   {secondary_metrics['mae']:.4f}")
    print(f"    - RMSE:  {secondary_metrics['rmse']:.4f}")
    print(f"    - R²:    {secondary_metrics['r2']:.4f}")
    print(f"    - MAPE:  {secondary_metrics['mape']:.2f}%")
    print()
    
    # Generate visualizations
    print("📊 Generating visualizations...")
    
    # Confusion Matrix
    evaluator.plot_confusion_matrix(
        primary_metrics['y_true'],
        primary_metrics['y_pred'],
        eval_dir / "confusion_matrix.png",
        title="Primary Model - Confusion Matrix"
    )
    print("  ✓ Confusion matrix saved")
    
    # ROC Curve
    evaluator.plot_roc_curve(
        primary_metrics['y_true'],
        primary_metrics['y_proba'],
        eval_dir / "roc_curve.png",
        title="Primary Model - ROC Curve"
    )
    print("  ✓ ROC curve saved")
    
    # Prediction vs Actual
    evaluator.plot_prediction_vs_actual(
        secondary_metrics['y_true'],
        secondary_metrics['y_pred'],
        eval_dir / "rul_prediction_scatter.png",
        title="Secondary Model - Predicted vs Actual RUL"
    )
    print("  ✓ Prediction scatter plot saved")
    print()
    
    # Generate report
    print("📝 Generating evaluation report...")
    
    dataset_info = {
        "Total Samples": len(test_df),
        "Features": len(feature_cols),
        "Capacitors": ", ".join(test_df["capacitor_id"].unique()),
        "Cycle Range": f"{test_df['cycle'].min()}-{test_df['cycle'].max()}",
        "Normal Samples": (y_test_primary == 0).sum(),
        "Abnormal Samples": (y_test_primary == 1).sum(),
    }
    
    evaluator.generate_report(
        primary_metrics,
        secondary_metrics,
        eval_dir / "baseline_report.md",
        dataset_info=dataset_info
    )
    print(f"  ✓ Report saved to: {eval_dir / 'baseline_report.md'}")
    print()
    
    # Summary
    print("=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print()
    print("Primary Model (Anomaly Detection):")
    print(f"  F1-Score: {primary_metrics['f1_score']:.4f} (Target: ≥ 0.80)")
    if primary_metrics['f1_score'] >= 0.80:
        print("  ✅ Target achieved!")
    else:
        print("  ⚠️  Target not met")
    print()
    
    print("Secondary Model (RUL Prediction):")
    print(f"  MAPE: {secondary_metrics['mape']:.2f}% (Target: ≤ 20%)")
    if secondary_metrics['mape'] <= 20.0:
        print("  ✅ Target achieved!")
    else:
        print("  ⚠️  Target not met")
    print()
    
    print("Generated Files:")
    print(f"  - {eval_dir / 'baseline_report.md'}")
    print(f"  - {eval_dir / 'confusion_matrix.png'}")
    print(f"  - {eval_dir / 'roc_curve.png'}")
    print(f"  - {eval_dir / 'rul_prediction_scatter.png'}")
    print()
    
    print("=" * 80)
    print("REPORT GENERATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
