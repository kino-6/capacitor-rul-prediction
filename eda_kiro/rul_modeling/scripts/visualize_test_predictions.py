"""
Test Data Prediction Visualization Script

Visualize model predictions on actual test data (ES12C7, ES12C8) cycle by cycle.
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 24)

# Paths
OUTPUT_DIR = Path("output/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_models_and_data():
    """Load trained models and test dataset"""
    print("Loading models and test data...")
    
    # Load models
    with open("output/models/primary_model.pkl", "rb") as f:
        primary_data = pickle.load(f)
        primary_model = primary_data['model']
    
    with open("output/models/secondary_model.pkl", "rb") as f:
        secondary_data = pickle.load(f)
        secondary_model = secondary_data['model']
    
    # Load test dataset
    test = pd.read_csv("output/features/test.csv")
    
    return primary_model, secondary_model, test

def visualize_test_predictions(primary_model, secondary_model, test):
    """Create comprehensive visualization of test predictions"""
    print("\nGenerating test prediction visualizations...")
    
    # Prepare features
    feature_cols = [col for col in test.columns if col not in ['capacitor_id', 'cycle', 'is_abnormal', 'rul']]
    X_test = test[feature_cols]
    
    # Get predictions
    primary_pred = primary_model.predict(X_test)
    primary_proba = primary_model.predict_proba(X_test)[:, 1]
    secondary_pred = secondary_model.predict(X_test)
    
    # Add predictions to test dataframe
    test_results = test.copy()
    test_results['pred_abnormal'] = primary_pred
    test_results['pred_abnormal_proba'] = primary_proba
    test_results['pred_rul'] = secondary_pred
    test_results['rul_error'] = secondary_pred - test['rul']
    test_results['rul_abs_error'] = np.abs(test_results['rul_error'])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(6, 2, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('ES12 Test Data - Model Predictions vs Actual Values', 
                 fontsize=20, fontweight='bold', y=0.995)
    
    # ========== PRIMARY MODEL VISUALIZATIONS ==========
    
    # 1. Anomaly Detection: Prediction Probability by Cycle (Both Capacitors)
    ax1 = fig.add_subplot(gs[0, :])
    for cap in test_results['capacitor_id'].unique():
        cap_data = test_results[test_results['capacitor_id'] == cap].sort_values('cycle')
        
        # Plot actual labels as background
        normal_cycles = cap_data[cap_data['is_abnormal'] == 0]
        abnormal_cycles = cap_data[cap_data['is_abnormal'] == 1]
        
        ax1.scatter(normal_cycles['cycle'], normal_cycles['pred_abnormal_proba'], 
                   marker='o', s=50, alpha=0.6, label=f'{cap} - Normal (Actual)')
        ax1.scatter(abnormal_cycles['cycle'], abnormal_cycles['pred_abnormal_proba'], 
                   marker='s', s=50, alpha=0.6, label=f'{cap} - Abnormal (Actual)')
        
        # Connect with line
        ax1.plot(cap_data['cycle'], cap_data['pred_abnormal_proba'], 
                alpha=0.3, linewidth=2)
    
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold (0.5)')
    ax1.axhline(y=0.0, color='blue', linestyle=':', linewidth=1, alpha=0.5, label='Normal (0.0)')
    ax1.axhline(y=1.0, color='orange', linestyle=':', linewidth=1, alpha=0.5, label='Abnormal (1.0)')
    ax1.set_xlabel('Cycle Number', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Predicted Abnormal Probability', fontsize=14, fontweight='bold')
    ax1.set_title('Primary Model: Anomaly Detection Probability Over Time', 
                  fontsize=16, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-0.05, 1.05])
    
    # 2. Anomaly Detection: Confusion by Capacitor
    ax2 = fig.add_subplot(gs[1, 0])
    confusion_data = []
    for cap in test_results['capacitor_id'].unique():
        cap_data = test_results[test_results['capacitor_id'] == cap]
        correct = (cap_data['is_abnormal'] == cap_data['pred_abnormal']).sum()
        incorrect = (cap_data['is_abnormal'] != cap_data['pred_abnormal']).sum()
        accuracy = correct / len(cap_data) * 100
        confusion_data.append({
            'Capacitor': cap,
            'Correct': correct,
            'Incorrect': incorrect,
            'Accuracy': accuracy
        })
    
    conf_df = pd.DataFrame(confusion_data)
    x_pos = np.arange(len(conf_df))
    ax2.bar(x_pos - 0.2, conf_df['Correct'], width=0.4, label='Correct', color='green', alpha=0.7)
    ax2.bar(x_pos + 0.2, conf_df['Incorrect'], width=0.4, label='Incorrect', color='red', alpha=0.7)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(conf_df['Capacitor'])
    ax2.set_ylabel('Number of Samples', fontsize=12, fontweight='bold')
    ax2.set_title('Primary Model: Prediction Accuracy by Capacitor', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add accuracy text
    for i, row in conf_df.iterrows():
        ax2.text(i, row['Correct'] + row['Incorrect'] + 5, f"{row['Accuracy']:.1f}%", 
                ha='center', fontsize=11, fontweight='bold')
    
    # 3. Anomaly Detection: Prediction Distribution
    ax3 = fig.add_subplot(gs[1, 1])
    for cap in test_results['capacitor_id'].unique():
        cap_data = test_results[test_results['capacitor_id'] == cap]
        ax3.hist(cap_data['pred_abnormal_proba'], bins=30, alpha=0.5, label=cap, edgecolor='black')
    
    ax3.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Threshold')
    ax3.set_xlabel('Predicted Abnormal Probability', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax3.set_title('Primary Model: Prediction Probability Distribution', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ========== SECONDARY MODEL VISUALIZATIONS ==========
    
    # 4. RUL Prediction: Actual vs Predicted by Cycle (Both Capacitors)
    ax4 = fig.add_subplot(gs[2, :])
    for cap in test_results['capacitor_id'].unique():
        cap_data = test_results[test_results['capacitor_id'] == cap].sort_values('cycle')
        
        # Plot actual RUL
        ax4.plot(cap_data['cycle'], cap_data['rul'], 
                marker='o', markersize=6, linewidth=2, alpha=0.8, 
                label=f'{cap} - Actual RUL', linestyle='-')
        
        # Plot predicted RUL
        ax4.plot(cap_data['cycle'], cap_data['pred_rul'], 
                marker='s', markersize=6, linewidth=2, alpha=0.8, 
                label=f'{cap} - Predicted RUL', linestyle='--')
    
    ax4.set_xlabel('Cycle Number', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Remaining Useful Life (RUL)', fontsize=14, fontweight='bold')
    ax4.set_title('Secondary Model: RUL Prediction vs Actual Over Time', 
                  fontsize=16, fontweight='bold')
    ax4.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    # 5. RUL Prediction: Error by Cycle (Both Capacitors)
    ax5 = fig.add_subplot(gs[3, :])
    for cap in test_results['capacitor_id'].unique():
        cap_data = test_results[test_results['capacitor_id'] == cap].sort_values('cycle')
        
        # Plot error
        ax5.plot(cap_data['cycle'], cap_data['rul_error'], 
                marker='o', markersize=5, linewidth=2, alpha=0.7, label=cap)
        
        # Fill area for positive/negative errors
        ax5.fill_between(cap_data['cycle'], 0, cap_data['rul_error'], 
                        alpha=0.2)
    
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax5.set_xlabel('Cycle Number', fontsize=14, fontweight='bold')
    ax5.set_ylabel('Prediction Error (Predicted - Actual)', fontsize=14, fontweight='bold')
    ax5.set_title('Secondary Model: RUL Prediction Error Over Time', 
                  fontsize=16, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. RUL Prediction: Absolute Error by Cycle
    ax6 = fig.add_subplot(gs[4, 0])
    for cap in test_results['capacitor_id'].unique():
        cap_data = test_results[test_results['capacitor_id'] == cap].sort_values('cycle')
        ax6.plot(cap_data['cycle'], cap_data['rul_abs_error'], 
                marker='o', markersize=5, linewidth=2, alpha=0.7, label=cap)
    
    ax6.set_xlabel('Cycle Number', fontsize=12, fontweight='bold')
    ax6.set_ylabel('Absolute Error', fontsize=12, fontweight='bold')
    ax6.set_title('Secondary Model: Absolute Error Over Time', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. RUL Prediction: Scatter Plot (Actual vs Predicted)
    ax7 = fig.add_subplot(gs[4, 1])
    for cap in test_results['capacitor_id'].unique():
        cap_data = test_results[test_results['capacitor_id'] == cap]
        ax7.scatter(cap_data['rul'], cap_data['pred_rul'], 
                   alpha=0.6, s=50, label=cap, edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    max_rul = max(test_results['rul'].max(), test_results['pred_rul'].max())
    ax7.plot([0, max_rul], [0, max_rul], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax7.set_xlabel('Actual RUL', fontsize=12, fontweight='bold')
    ax7.set_ylabel('Predicted RUL', fontsize=12, fontweight='bold')
    ax7.set_title('Secondary Model: Actual vs Predicted RUL', fontsize=14, fontweight='bold')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_aspect('equal', adjustable='box')
    
    # 8. Performance Metrics Summary Table
    ax8 = fig.add_subplot(gs[5, :])
    ax8.axis('off')
    
    # Calculate metrics for each capacitor
    metrics_data = []
    for cap in test_results['capacitor_id'].unique():
        cap_data = test_results[test_results['capacitor_id'] == cap]
        
        # Primary model metrics
        primary_acc = (cap_data['is_abnormal'] == cap_data['pred_abnormal']).mean() * 100
        
        # Secondary model metrics
        mae = cap_data['rul_abs_error'].mean()
        rmse = np.sqrt((cap_data['rul_error']**2).mean())
        max_error = cap_data['rul_abs_error'].max()
        
        metrics_data.append([
            cap,
            f"{primary_acc:.2f}%",
            f"{mae:.2f}",
            f"{rmse:.2f}",
            f"{max_error:.2f}"
        ])
    
    # Add overall metrics
    overall_primary_acc = (test_results['is_abnormal'] == test_results['pred_abnormal']).mean() * 100
    overall_mae = test_results['rul_abs_error'].mean()
    overall_rmse = np.sqrt((test_results['rul_error']**2).mean())
    overall_max_error = test_results['rul_abs_error'].max()
    
    metrics_data.append([
        'Overall',
        f"{overall_primary_acc:.2f}%",
        f"{overall_mae:.2f}",
        f"{overall_rmse:.2f}",
        f"{overall_max_error:.2f}"
    ])
    
    # Create table
    table = ax8.table(cellText=metrics_data,
                     colLabels=['Capacitor', 'Anomaly\nAccuracy', 'RUL\nMAE', 'RUL\nRMSE', 'RUL\nMax Error'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0.1, 0.2, 0.8, 0.6])
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style overall row
    for i in range(5):
        table[(len(metrics_data), i)].set_facecolor('#FFC107')
        table[(len(metrics_data), i)].set_text_props(weight='bold')
    
    ax8.set_title('Performance Metrics Summary', fontsize=16, fontweight='bold', pad=20)
    
    # Save figure
    plt.savefig(OUTPUT_DIR / "test_predictions_detailed.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'test_predictions_detailed.png'}")
    plt.close()
    
    # Save predictions to CSV
    test_results.to_csv(OUTPUT_DIR / "test_predictions_detailed.csv", index=False)
    print(f"✓ Saved: {OUTPUT_DIR / 'test_predictions_detailed.csv'}")
    
    return test_results

def generate_performance_report(test_results):
    """Generate detailed performance report"""
    print("\nGenerating performance report...")
    
    report = f"""# Test Data Performance Report

## 📅 Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Test Dataset Information

- **Capacitors**: {', '.join(test_results['capacitor_id'].unique())}
- **Total Samples**: {len(test_results)}
- **Cycle Range**: {test_results['cycle'].min()} - {test_results['cycle'].max()}

## 🎯 Primary Model (Anomaly Detection) Performance

### Overall Performance

"""
    
    # Primary model overall metrics
    overall_acc = (test_results['is_abnormal'] == test_results['pred_abnormal']).mean() * 100
    overall_precision = ((test_results['is_abnormal'] == 1) & (test_results['pred_abnormal'] == 1)).sum() / (test_results['pred_abnormal'] == 1).sum() * 100 if (test_results['pred_abnormal'] == 1).sum() > 0 else 0
    overall_recall = ((test_results['is_abnormal'] == 1) & (test_results['pred_abnormal'] == 1)).sum() / (test_results['is_abnormal'] == 1).sum() * 100 if (test_results['is_abnormal'] == 1).sum() > 0 else 0
    
    report += f"""
- **Accuracy**: {overall_acc:.2f}%
- **Precision**: {overall_precision:.2f}%
- **Recall**: {overall_recall:.2f}%

### Performance by Capacitor

"""
    
    for cap in test_results['capacitor_id'].unique():
        cap_data = test_results[test_results['capacitor_id'] == cap]
        cap_acc = (cap_data['is_abnormal'] == cap_data['pred_abnormal']).mean() * 100
        correct = (cap_data['is_abnormal'] == cap_data['pred_abnormal']).sum()
        total = len(cap_data)
        
        report += f"""
#### {cap}
- Accuracy: {cap_acc:.2f}% ({correct}/{total} correct)
- Samples: {total}
"""
    
    report += """

## 📈 Secondary Model (RUL Prediction) Performance

### Overall Performance

"""
    
    # Secondary model overall metrics
    overall_mae = test_results['rul_abs_error'].mean()
    overall_rmse = np.sqrt((test_results['rul_error']**2).mean())
    overall_max_error = test_results['rul_abs_error'].max()
    overall_r2 = 1 - (test_results['rul_error']**2).sum() / ((test_results['rul'] - test_results['rul'].mean())**2).sum()
    
    report += f"""
- **MAE**: {overall_mae:.2f} cycles
- **RMSE**: {overall_rmse:.2f} cycles
- **Max Error**: {overall_max_error:.2f} cycles
- **R²**: {overall_r2:.4f}

### Performance by Capacitor

"""
    
    for cap in test_results['capacitor_id'].unique():
        cap_data = test_results[test_results['capacitor_id'] == cap]
        cap_mae = cap_data['rul_abs_error'].mean()
        cap_rmse = np.sqrt((cap_data['rul_error']**2).mean())
        cap_max_error = cap_data['rul_abs_error'].max()
        cap_r2 = 1 - (cap_data['rul_error']**2).sum() / ((cap_data['rul'] - cap_data['rul'].mean())**2).sum()
        
        report += f"""
#### {cap}
- MAE: {cap_mae:.2f} cycles
- RMSE: {cap_rmse:.2f} cycles
- Max Error: {cap_max_error:.2f} cycles
- R²: {cap_r2:.4f}
"""
    
    report += """

### Performance by RUL Range

"""
    
    rul_ranges = [(0, 50), (50, 100), (100, 150), (150, 200)]
    for start, end in rul_ranges:
        range_data = test_results[(test_results['rul'] >= start) & (test_results['rul'] < end)]
        if len(range_data) > 0:
            range_mae = range_data['rul_abs_error'].mean()
            range_rmse = np.sqrt((range_data['rul_error']**2).mean())
            
            report += f"""
#### RUL {start}-{end}
- Samples: {len(range_data)}
- MAE: {range_mae:.2f} cycles
- RMSE: {range_rmse:.2f} cycles
"""
    
    report += """

## 🔍 Key Observations

### Primary Model (Anomaly Detection)

"""
    
    if overall_acc >= 99:
        report += """
✅ **Excellent Performance**: The model achieves near-perfect anomaly detection accuracy.

⚠️ **Potential Concern**: Perfect or near-perfect performance may indicate overfitting or data leakage. 
External validation on ES10/ES14 is recommended.
"""
    else:
        report += f"""
📊 **Good Performance**: The model achieves {overall_acc:.2f}% accuracy on test data.
"""
    
    report += """

### Secondary Model (RUL Prediction)

"""
    
    # Check end-of-life performance
    eol_data = test_results[test_results['rul'] < 50]
    if len(eol_data) > 0:
        eol_mae = eol_data['rul_abs_error'].mean()
        if eol_mae > 20:
            report += f"""
⚠️ **End-of-Life Challenge**: The model struggles with RUL < 50 (MAE = {eol_mae:.2f} cycles).

**Cause**: Training data only includes RUL 50-199, so the model cannot extrapolate to lower values.

**Recommendation**: Include end-of-life data in training or use alternative labeling strategy.
"""
    
    mid_life_data = test_results[(test_results['rul'] >= 50) & (test_results['rul'] < 150)]
    if len(mid_life_data) > 0:
        mid_mae = mid_life_data['rul_abs_error'].mean()
        if mid_mae < 1:
            report += f"""
✅ **Excellent Mid-Life Performance**: The model achieves MAE = {mid_mae:.2f} cycles for RUL 50-150.
"""
    
    report += """

## 📁 Generated Files

- `test_predictions_detailed.png` - Comprehensive visualization of predictions
- `test_predictions_detailed.csv` - Detailed predictions for all test samples

---

**Report Generated by**: Kiro AI Agent  
**Status**: Test Data Performance Analysis Complete
"""
    
    # Save report
    report_path = OUTPUT_DIR / "test_performance_report.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✓ Saved: {report_path}")
    return report

def main():
    """Main execution function"""
    print("="*80)
    print("TEST DATA PREDICTION VISUALIZATION")
    print("="*80)
    
    # Load models and data
    primary_model, secondary_model, test = load_models_and_data()
    
    # Visualize predictions
    test_results = visualize_test_predictions(primary_model, secondary_model, test)
    
    # Generate performance report
    report = generate_performance_report(test_results)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  1. {OUTPUT_DIR / 'test_predictions_detailed.png'}")
    print(f"  2. {OUTPUT_DIR / 'test_predictions_detailed.csv'}")
    print(f"  3. {OUTPUT_DIR / 'test_performance_report.md'}")

if __name__ == "__main__":
    main()
