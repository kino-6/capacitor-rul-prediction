"""Script to visualize Secondary Model predictions and performance."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")


def main():
    """Visualize Secondary Model predictions."""
    print("=" * 80)
    print("SECONDARY MODEL - PREDICTION VISUALIZATION")
    print("=" * 80)
    print()
    
    # Paths
    model_dir = Path(__file__).parent.parent / "output" / "models"
    eval_dir = Path(__file__).parent.parent / "output" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictions
    print("📂 Loading predictions...")
    predictions_path = model_dir / "secondary_predictions.csv"
    pred_df = pd.read_csv(predictions_path)
    print(f"  ✓ Loaded {len(pred_df)} predictions")
    print()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Actual vs Predicted scatter plot
    ax1 = plt.subplot(2, 3, 1)
    ax1.scatter(pred_df['actual_rul'], pred_df['predicted_rul'], 
               alpha=0.5, s=30, color='steelblue')
    
    # Add perfect prediction line
    min_val = min(pred_df['actual_rul'].min(), pred_df['predicted_rul'].min())
    max_val = max(pred_df['actual_rul'].max(), pred_df['predicted_rul'].max())
    ax1.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Perfect Prediction')
    
    ax1.set_xlabel('Actual RUL', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Predicted RUL', fontsize=11, fontweight='bold')
    ax1.set_title('Actual vs Predicted RUL', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Residual plot
    ax2 = plt.subplot(2, 3, 2)
    ax2.scatter(pred_df['predicted_rul'], pred_df['error'], 
               alpha=0.5, s=30, color='coral')
    ax2.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax2.set_xlabel('Predicted RUL', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Residual (Predicted - Actual)', fontsize=11, fontweight='bold')
    ax2.set_title('Residual Plot', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # 3. Error distribution
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(pred_df['error'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax3.set_xlabel('Prediction Error', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # 4. Absolute error by actual RUL
    ax4 = plt.subplot(2, 3, 4)
    ax4.scatter(pred_df['actual_rul'], pred_df['abs_error'], 
               alpha=0.5, s=30, color='purple')
    ax4.set_xlabel('Actual RUL', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Absolute Error', fontsize=11, fontweight='bold')
    ax4.set_title('Absolute Error vs Actual RUL', fontsize=12, fontweight='bold')
    ax4.grid(alpha=0.3)
    
    # 5. Predictions by capacitor
    ax5 = plt.subplot(2, 3, 5)
    for cap_id in pred_df['capacitor_id'].unique():
        cap_data = pred_df[pred_df['capacitor_id'] == cap_id]
        ax5.plot(cap_data['cycle'], cap_data['actual_rul'], 
                'o-', label=f'{cap_id} (Actual)', alpha=0.6, markersize=4)
        ax5.plot(cap_data['cycle'], cap_data['predicted_rul'], 
                's--', label=f'{cap_id} (Pred)', alpha=0.6, markersize=3)
    
    ax5.set_xlabel('Cycle', fontsize=11, fontweight='bold')
    ax5.set_ylabel('RUL', fontsize=11, fontweight='bold')
    ax5.set_title('RUL Predictions by Capacitor', fontsize=12, fontweight='bold')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax5.grid(alpha=0.3)
    
    # 6. Percentage error distribution (excluding very high values)
    ax6 = plt.subplot(2, 3, 6)
    # Filter out extreme percentage errors for better visualization
    pct_errors = pred_df['pct_error'].copy()
    pct_errors_filtered = pct_errors[pct_errors <= 200]  # Cap at 200%
    
    ax6.hist(pct_errors_filtered, bins=30, color='gold', edgecolor='black', alpha=0.7)
    ax6.set_xlabel('Percentage Error (%)', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax6.set_title(f'Percentage Error Distribution (≤200%)\n{len(pct_errors_filtered)}/{len(pct_errors)} samples', 
                 fontsize=12, fontweight='bold')
    ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = eval_dir / "secondary_predictions.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {output_path}")
    print()
    
    # Display statistics
    print("📊 Prediction Statistics:")
    print("-" * 80)
    print(f"  Total predictions:     {len(pred_df)}")
    print(f"  MAE:                   {pred_df['abs_error'].mean():.4f}")
    print(f"  RMSE:                  {np.sqrt((pred_df['error']**2).mean()):.4f}")
    print(f"  Max absolute error:    {pred_df['abs_error'].max():.4f}")
    print(f"  Min absolute error:    {pred_df['abs_error'].min():.4f}")
    print()
    
    # Statistics by RUL range
    print("📊 Error Statistics by RUL Range:")
    print("-" * 80)
    
    rul_ranges = [
        (0, 50, "Very Low (0-50)"),
        (50, 100, "Low (50-100)"),
        (100, 150, "Medium (100-150)"),
        (150, 200, "High (150-200)")
    ]
    
    for min_rul, max_rul, label in rul_ranges:
        mask = (pred_df['actual_rul'] >= min_rul) & (pred_df['actual_rul'] < max_rul)
        if mask.sum() > 0:
            range_data = pred_df[mask]
            print(f"  {label:20s}: n={mask.sum():3d}, MAE={range_data['abs_error'].mean():6.2f}, "
                  f"MAPE={range_data['pct_error'].mean():7.2f}%")
    print()
    
    # Identify worst predictions
    print("📊 Top 5 Worst Predictions:")
    print("-" * 80)
    worst_preds = pred_df.nlargest(5, 'abs_error')
    for idx, row in worst_preds.iterrows():
        print(f"  {row['capacitor_id']} Cycle {row['cycle']:3.0f}: "
              f"Actual={row['actual_rul']:6.1f}, Predicted={row['predicted_rul']:6.1f}, "
              f"Error={row['abs_error']:6.1f}")
    print()
    
    # Feature importance
    print("📊 Feature Importance:")
    print("-" * 80)
    importance_path = model_dir / "secondary_feature_importance.csv"
    importance_df = pd.read_csv(importance_path)
    
    for idx, row in importance_df.head(5).iterrows():
        print(f"  {row['feature']:25s}: {row['importance']:.4f}")
    print()
    
    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
