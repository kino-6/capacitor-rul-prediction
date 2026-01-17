"""Script to test trained models on ES10 and ES14 datasets (unseen data)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

from rul_modeling.src.models.primary_model import PrimaryModel
from rul_modeling.src.models.secondary_model import SecondaryModel
from rul_modeling.src.data_preparation.parallel_extractor import ParallelFeatureExtractor
from rul_modeling.src.data_preparation.label_generator import LabelGenerator
from rul_modeling.src.data_preparation.feature_scaler import FeatureScaler


def main():
    """Test models on ES10 and ES14 datasets."""
    print("=" * 80)
    print("TESTING ON ES10 AND ES14 DATASETS (UNSEEN DATA)")
    print("=" * 80)
    print()
    
    # Paths
    data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    model_dir = Path(__file__).parent.parent / "output" / "models"
    output_dir = Path(__file__).parent.parent / "output" / "external_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if ES10 and ES14 exist
    es10_path = data_dir / "ES10.mat"
    es14_path = data_dir / "ES14.mat"
    
    datasets_to_test = []
    if es10_path.exists():
        datasets_to_test.append(("ES10", es10_path))
        print(f"✓ Found ES10 dataset: {es10_path}")
    else:
        print(f"⚠️  ES10 dataset not found: {es10_path}")
    
    if es14_path.exists():
        datasets_to_test.append(("ES14", es14_path))
        print(f"✓ Found ES14 dataset: {es14_path}")
    else:
        print(f"⚠️  ES14 dataset not found: {es14_path}")
    
    if not datasets_to_test:
        print("\n❌ No external datasets found. Exiting.")
        return
    
    print()
    
    # Load trained models
    print("📂 Loading trained models...")
    primary_model = PrimaryModel()
    primary_model.load(model_dir / "primary_model.pkl")
    print("  ✓ Primary Model loaded")
    
    secondary_model = SecondaryModel()
    secondary_model.load(model_dir / "secondary_model.pkl")
    print("  ✓ Secondary Model loaded")
    
    # Load scaler
    scaler = FeatureScaler()
    scaler.load(model_dir / "scaler.pkl")
    print("  ✓ Feature Scaler loaded")
    print()
    
    # Test on each dataset
    all_results = []
    
    for dataset_name, dataset_path in datasets_to_test:
        print("=" * 80)
        print(f"TESTING ON {dataset_name}")
        print("=" * 80)
        print()
        
        # Extract features
        print(f"📊 Extracting features from {dataset_name}...")
        extractor = ParallelFeatureExtractor(
            str(dataset_path),
            n_processes=8,
            include_history=False
        )
        features_df = extractor.extract_all_capacitors(
            max_cycles=200,
            progress_interval=20
        )
        print(f"  ✓ Extracted {len(features_df)} samples")
        print()
        
        # Add labels
        print("🏷️  Adding labels...")
        label_gen = LabelGenerator()
        labeled_df = label_gen.generate_labels(features_df, strategy="cycle_based")
        print(f"  ✓ Labels added")
        print(f"    - Normal: {(labeled_df['is_abnormal'] == 0).sum()}")
        print(f"    - Abnormal: {(labeled_df['is_abnormal'] == 1).sum()}")
        print()
        
        # Prepare features
        feature_cols = [col for col in labeled_df.columns 
                       if col not in ["capacitor_id", "cycle", "is_abnormal", "rul"]]
        
        X = labeled_df[feature_cols]
        y_primary = labeled_df["is_abnormal"]
        y_secondary = labeled_df["rul"]
        
        # Scale features
        print("🔧 Scaling features...")
        X_scaled = scaler.transform(X)
        print("  ✓ Features scaled")
        print()
        
        # Evaluate Primary Model
        print("📊 Evaluating Primary Model...")
        primary_metrics = primary_model.evaluate(X_scaled, y_primary)
        
        print("  Results:")
        print(f"    - Accuracy:  {primary_metrics['accuracy']:.4f}")
        print(f"    - Precision: {primary_metrics['precision']:.4f}")
        print(f"    - Recall:    {primary_metrics['recall']:.4f}")
        print(f"    - F1-Score:  {primary_metrics['f1_score']:.4f}")
        print(f"    - ROC-AUC:   {primary_metrics['roc_auc']:.4f}")
        print()
        
        # Evaluate Secondary Model
        print("📊 Evaluating Secondary Model...")
        secondary_metrics = secondary_model.evaluate(X_scaled, y_secondary)
        
        print("  Results:")
        print(f"    - MAE:   {secondary_metrics['mae']:.4f}")
        print(f"    - RMSE:  {secondary_metrics['rmse']:.4f}")
        print(f"    - R²:    {secondary_metrics['r2']:.4f}")
        print(f"    - MAPE:  {secondary_metrics['mape']:.2f}%")
        print()
        
        # Analyze by RUL range
        print("📊 Error Analysis by RUL Range:")
        print("-" * 80)
        
        y_pred = secondary_model.predict(X_scaled)
        predictions_df = pd.DataFrame({
            "actual_rul": y_secondary,
            "predicted_rul": y_pred,
            "abs_error": abs(y_pred - y_secondary)
        })
        
        rul_ranges = [
            (0, 50, "Very Low (0-50)"),
            (50, 100, "Low (50-100)"),
            (100, 150, "Medium (100-150)"),
            (150, 200, "High (150-200)")
        ]
        
        for min_rul, max_rul, label in rul_ranges:
            mask = (predictions_df['actual_rul'] >= min_rul) & (predictions_df['actual_rul'] < max_rul)
            if mask.sum() > 0:
                range_data = predictions_df[mask]
                mae = range_data['abs_error'].mean()
                # Calculate MAPE safely
                mask_nonzero = range_data['actual_rul'] != 0
                if mask_nonzero.sum() > 0:
                    mape = (abs((range_data.loc[mask_nonzero, 'predicted_rul'] - 
                                range_data.loc[mask_nonzero, 'actual_rul']) / 
                               range_data.loc[mask_nonzero, 'actual_rul']) * 100).mean()
                else:
                    mape = float('inf')
                
                print(f"  {label:20s}: n={mask.sum():4d}, MAE={mae:7.2f}, MAPE={mape:7.2f}%")
        print()
        
        # Store results
        result = {
            "dataset": dataset_name,
            "samples": len(labeled_df),
            "primary_f1": primary_metrics['f1_score'],
            "primary_accuracy": primary_metrics['accuracy'],
            "primary_roc_auc": primary_metrics['roc_auc'],
            "secondary_mae": secondary_metrics['mae'],
            "secondary_rmse": secondary_metrics['rmse'],
            "secondary_r2": secondary_metrics['r2'],
            "secondary_mape": secondary_metrics['mape'],
        }
        all_results.append(result)
        
        # Save predictions
        predictions_full = pd.DataFrame({
            "capacitor_id": labeled_df["capacitor_id"],
            "cycle": labeled_df["cycle"],
            "actual_abnormal": y_primary,
            "predicted_abnormal": primary_model.predict(X_scaled),
            "actual_rul": y_secondary,
            "predicted_rul": y_pred,
            "error": y_pred - y_secondary,
            "abs_error": abs(y_pred - y_secondary),
        })
        
        pred_path = output_dir / f"{dataset_name.lower()}_predictions.csv"
        predictions_full.to_csv(pred_path, index=False)
        print(f"💾 Predictions saved to: {pred_path}")
        print()
    
    # Summary comparison
    print("=" * 80)
    print("SUMMARY: ES12 (Training) vs ES10/ES14 (Unseen)")
    print("=" * 80)
    print()
    
    # ES12 test results (from training)
    print("ES12 Test Set (Seen Data):")
    print("  Primary Model:  F1-Score = 1.0000")
    print("  Secondary Model: MAPE = 89.78%, R² = 0.9330")
    print()
    
    # External test results
    for result in all_results:
        print(f"{result['dataset']} (Unseen Data):")
        print(f"  Primary Model:  F1-Score = {result['primary_f1']:.4f}")
        print(f"  Secondary Model: MAPE = {result['secondary_mape']:.2f}%, R² = {result['secondary_r2']:.4f}")
        print()
    
    # Analysis
    print("📊 Analysis:")
    print("-" * 80)
    
    avg_f1_external = sum(r['primary_f1'] for r in all_results) / len(all_results)
    avg_mape_external = sum(r['secondary_mape'] for r in all_results) / len(all_results)
    avg_r2_external = sum(r['secondary_r2'] for r in all_results) / len(all_results)
    
    print(f"Average External Performance:")
    print(f"  Primary Model:  F1-Score = {avg_f1_external:.4f}")
    print(f"  Secondary Model: MAPE = {avg_mape_external:.2f}%, R² = {avg_r2_external:.4f}")
    print()
    
    # Overfitting check
    print("🔍 Overfitting Analysis:")
    print("-" * 80)
    
    if avg_f1_external < 0.95:
        print("  ⚠️  Primary Model: Significant performance drop on unseen data")
        print("     → Possible overfitting detected")
    else:
        print("  ✅ Primary Model: Good generalization")
    
    if avg_r2_external < 0.85:
        print("  ⚠️  Secondary Model: Significant performance drop on unseen data")
        print("     → Possible overfitting detected")
    else:
        print("  ✅ Secondary Model: Good generalization")
    
    print()
    
    # Save summary
    summary_df = pd.DataFrame(all_results)
    summary_path = output_dir / "external_test_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"💾 Summary saved to: {summary_path}")
    print()
    
    print("=" * 80)
    print("EXTERNAL TESTING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
