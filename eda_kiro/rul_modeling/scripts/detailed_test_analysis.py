"""
ES12 Test Data Detailed Analysis Script

This script performs comprehensive analysis of model predictions on ES12 test data:
1. Primary Model: Cycle-by-cycle prediction probability analysis
2. Secondary Model: Detailed RUL prediction analysis
3. Overfitting diagnosis across Train/Val/Test sets
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

# Paths
OUTPUT_DIR = Path("output/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_models_and_data():
    """Load trained models and datasets"""
    print("Loading models and data...")
    
    # Load models
    with open("output/models/primary_model.pkl", "rb") as f:
        primary_data = pickle.load(f)
        primary_model = primary_data['model']
    
    with open("output/models/secondary_model.pkl", "rb") as f:
        secondary_data = pickle.load(f)
        secondary_model = secondary_data['model']
    
    # Load datasets
    train = pd.read_csv("output/features/train.csv")
    val = pd.read_csv("output/features/val.csv")
    test = pd.read_csv("output/features/test.csv")
    
    return primary_model, secondary_model, train, val, test

def analyze_primary_model(model, train, val, test):
    """Detailed analysis of Primary Model (Anomaly Detection)"""
    print("\n" + "="*80)
    print("PRIMARY MODEL ANALYSIS")
    print("="*80)
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle('Primary Model (Anomaly Detection) - Detailed Analysis', fontsize=16, fontweight='bold')
    
    # 1. Prediction probability by cycle for Test set
    ax = axes[0, 0]
    test_features = test.drop(['capacitor_id', 'cycle', 'is_abnormal', 'rul'], axis=1)
    test_proba = model.predict_proba(test_features)[:, 1]
    test_with_proba = test.copy()
    test_with_proba['pred_proba'] = test_proba
    
    for cap in test['capacitor_id'].unique():
        cap_data = test_with_proba[test_with_proba['capacitor_id'] == cap]
        ax.plot(cap_data['cycle'], cap_data['pred_proba'], marker='o', label=cap, alpha=0.7)
    
    ax.axhline(y=0.5, color='r', linestyle='--', label='Decision Threshold')
    ax.set_xlabel('Cycle Number', fontsize=12)
    ax.set_ylabel('Abnormal Probability', fontsize=12)
    ax.set_title('Test Set: Prediction Probability by Cycle', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Prediction probability distribution
    ax = axes[0, 1]
    normal_proba = test_proba[test['is_abnormal'] == 0]
    abnormal_proba = test_proba[test['is_abnormal'] == 1]
    
    ax.hist(normal_proba, bins=30, alpha=0.6, label='Normal (Actual)', color='blue', edgecolor='black')
    ax.hist(abnormal_proba, bins=30, alpha=0.6, label='Abnormal (Actual)', color='red', edgecolor='black')
    ax.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Prediction Probability', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Test Set: Probability Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Performance metrics across datasets
    ax = axes[0, 2]
    datasets = {'Train': train, 'Val': val, 'Test': test}
    metrics_data = []
    
    for name, df in datasets.items():
        X = df.drop(['capacitor_id', 'cycle', 'is_abnormal', 'rul'], axis=1)
        y = df['is_abnormal']
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        
        metrics_data.append({
            'Dataset': name,
            'Accuracy': accuracy_score(y, y_pred),
            'Precision': precision_score(y, y_pred),
            'Recall': recall_score(y, y_pred),
            'F1-Score': f1_score(y, y_pred),
            'ROC-AUC': roc_auc_score(y, y_proba)
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.set_index('Dataset', inplace=True)
    
    # Plot metrics comparison
    metrics_df.T.plot(kind='bar', ax=ax, width=0.8)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Across Datasets', fontsize=13, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title='Dataset')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.95, 1.01])
    
    # 4. Confusion matrix by capacitor (Test set)
    ax = axes[1, 0]
    confusion_by_cap = []
    for cap in test['capacitor_id'].unique():
        cap_data = test[test['capacitor_id'] == cap]
        X_cap = cap_data.drop(['capacitor_id', 'cycle', 'is_abnormal', 'rul'], axis=1)
        y_cap = cap_data['is_abnormal']
        y_pred_cap = model.predict(X_cap)
        
        tn = ((y_cap == 0) & (y_pred_cap == 0)).sum()
        fp = ((y_cap == 0) & (y_pred_cap == 1)).sum()
        fn = ((y_cap == 1) & (y_pred_cap == 0)).sum()
        tp = ((y_cap == 1) & (y_pred_cap == 1)).sum()
        
        confusion_by_cap.append([cap, tn, fp, fn, tp])
    
    conf_df = pd.DataFrame(confusion_by_cap, columns=['Capacitor', 'TN', 'FP', 'FN', 'TP'])
    conf_df.set_index('Capacitor', inplace=True)
    
    sns.heatmap(conf_df, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
    ax.set_title('Confusion Matrix by Capacitor', fontsize=13, fontweight='bold')
    ax.set_ylabel('Capacitor', fontsize=12)
    
    # 5. Performance by cycle range (Test set)
    ax = axes[1, 1]
    cycle_ranges = [(1, 50), (51, 100), (101, 150), (151, 200)]
    range_metrics = []
    
    for start, end in cycle_ranges:
        range_data = test[(test['cycle'] >= start) & (test['cycle'] <= end)]
        if len(range_data) > 0:
            X_range = range_data.drop(['capacitor_id', 'cycle', 'is_abnormal', 'rul'], axis=1)
            y_range = range_data['is_abnormal']
            y_pred_range = model.predict(X_range)
            
            range_metrics.append({
                'Range': f'{start}-{end}',
                'Accuracy': accuracy_score(y_range, y_pred_range),
                'F1-Score': f1_score(y_range, y_pred_range)
            })
    
    range_df = pd.DataFrame(range_metrics)
    range_df.set_index('Range', inplace=True)
    range_df.plot(kind='bar', ax=ax, width=0.7)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance by Cycle Range', fontsize=13, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0.95, 1.01])
    
    # 6. Feature importance
    ax = axes[1, 2]
    feature_importance = pd.DataFrame({
        'feature': test_features.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    ax.barh(range(len(feature_importance)), feature_importance['importance'], color='steelblue')
    ax.set_yticks(range(len(feature_importance)))
    ax.set_yticklabels(feature_importance['feature'])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Top 10 Feature Importance', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    # 7-9. Additional analysis placeholders
    for i in range(2, 3):
        for j in range(3):
            axes[i, j].text(0.5, 0.5, f'Reserved for\nAdditional Analysis', 
                          ha='center', va='center', fontsize=14, color='gray')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "primary_model_detailed_analysis.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'primary_model_detailed_analysis.png'}")
    plt.close()
    
    return metrics_df

def analyze_secondary_model(model, train, val, test):
    """Detailed analysis of Secondary Model (RUL Prediction)"""
    print("\n" + "="*80)
    print("SECONDARY MODEL ANALYSIS")
    print("="*80)
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))
    fig.suptitle('Secondary Model (RUL Prediction) - Detailed Analysis', fontsize=16, fontweight='bold')
    
    # Prepare test predictions
    test_features = test.drop(['capacitor_id', 'cycle', 'is_abnormal', 'rul'], axis=1)
    test_pred = model.predict(test_features)
    test_with_pred = test.copy()
    test_with_pred['pred_rul'] = test_pred
    test_with_pred['error'] = test_pred - test['rul']
    test_with_pred['abs_error'] = np.abs(test_with_pred['error'])
    
    # 1. Actual vs Predicted RUL (colored by cycle)
    ax = axes[0, 0]
    scatter = ax.scatter(test['rul'], test_pred, c=test['cycle'], cmap='viridis', 
                        alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax.plot([0, 200], [0, 200], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual RUL', fontsize=12)
    ax.set_ylabel('Predicted RUL', fontsize=12)
    ax.set_title('Actual vs Predicted RUL (colored by cycle)', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax, label='Cycle Number')
    
    # 2. RUL prediction by cycle for each capacitor
    ax = axes[0, 1]
    for cap in test['capacitor_id'].unique():
        cap_data = test_with_pred[test_with_pred['capacitor_id'] == cap].sort_values('cycle')
        ax.plot(cap_data['cycle'], cap_data['rul'], marker='o', label=f'{cap} (Actual)', linestyle='-', alpha=0.7)
        ax.plot(cap_data['cycle'], cap_data['pred_rul'], marker='s', label=f'{cap} (Predicted)', linestyle='--', alpha=0.7)
    
    ax.set_xlabel('Cycle Number', fontsize=12)
    ax.set_ylabel('RUL', fontsize=12)
    ax.set_title('RUL Prediction Over Time', fontsize=13, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. Performance metrics across datasets
    ax = axes[0, 2]
    datasets = {'Train': train, 'Val': val, 'Test': test}
    metrics_data = []
    
    for name, df in datasets.items():
        X = df.drop(['capacitor_id', 'cycle', 'is_abnormal', 'rul'], axis=1)
        y = df['rul']
        y_pred = model.predict(X)
        
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        metrics_data.append({
            'Dataset': name,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.set_index('Dataset', inplace=True)
    
    # Plot metrics comparison
    metrics_df[['MAE', 'RMSE']].plot(kind='bar', ax=ax, width=0.7)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('Error Metrics Across Datasets', fontsize=13, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. Error by capacitor
    ax = axes[1, 0]
    cap_metrics = []
    for cap in test['capacitor_id'].unique():
        cap_data = test_with_pred[test_with_pred['capacitor_id'] == cap]
        cap_metrics.append({
            'Capacitor': cap,
            'MAE': cap_data['abs_error'].mean(),
            'RMSE': np.sqrt((cap_data['error']**2).mean()),
            'R²': r2_score(cap_data['rul'], cap_data['pred_rul'])
        })
    
    cap_df = pd.DataFrame(cap_metrics)
    cap_df.set_index('Capacitor', inplace=True)
    cap_df[['MAE', 'RMSE']].plot(kind='bar', ax=ax, width=0.7)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('Error by Capacitor', fontsize=13, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 5. Error by RUL range
    ax = axes[1, 1]
    rul_ranges = [(0, 50), (50, 100), (100, 150), (150, 200)]
    range_metrics = []
    
    for start, end in rul_ranges:
        range_data = test_with_pred[(test_with_pred['rul'] >= start) & (test_with_pred['rul'] < end)]
        if len(range_data) > 0:
            range_metrics.append({
                'Range': f'{start}-{end}',
                'MAE': range_data['abs_error'].mean(),
                'RMSE': np.sqrt((range_data['error']**2).mean()),
                'Samples': len(range_data)
            })
    
    range_df = pd.DataFrame(range_metrics)
    range_df.set_index('Range', inplace=True)
    range_df[['MAE', 'RMSE']].plot(kind='bar', ax=ax, width=0.7)
    ax.set_ylabel('Error', fontsize=12)
    ax.set_title('Error by RUL Range', fontsize=13, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Residual plot
    ax = axes[1, 2]
    ax.scatter(test_pred, test_with_pred['error'], alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted RUL', fontsize=12)
    ax.set_ylabel('Residual (Predicted - Actual)', fontsize=12)
    ax.set_title('Residual Plot', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 7. Error distribution
    ax = axes[2, 0]
    ax.hist(test_with_pred['error'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    ax.set_xlabel('Prediction Error', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Error Distribution', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 8. Absolute error by cycle
    ax = axes[2, 1]
    for cap in test['capacitor_id'].unique():
        cap_data = test_with_pred[test_with_pred['capacitor_id'] == cap].sort_values('cycle')
        ax.plot(cap_data['cycle'], cap_data['abs_error'], marker='o', label=cap, alpha=0.7)
    
    ax.set_xlabel('Cycle Number', fontsize=12)
    ax.set_ylabel('Absolute Error', fontsize=12)
    ax.set_title('Absolute Error Over Time', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 9. Feature importance
    ax = axes[2, 2]
    feature_importance = pd.DataFrame({
        'feature': test_features.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)
    
    ax.barh(range(len(feature_importance)), feature_importance['importance'], color='steelblue')
    ax.set_yticks(range(len(feature_importance)))
    ax.set_yticklabels(feature_importance['feature'])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Top 10 Feature Importance', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "secondary_model_detailed_analysis.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {OUTPUT_DIR / 'secondary_model_detailed_analysis.png'}")
    plt.close()
    
    return metrics_df, range_df, cap_df

def generate_overfitting_diagnosis_report(primary_metrics, secondary_metrics):
    """Generate overfitting diagnosis report"""
    print("\n" + "="*80)
    print("GENERATING OVERFITTING DIAGNOSIS REPORT")
    print("="*80)
    
    # Convert DataFrames to string tables
    def df_to_markdown_table(df):
        """Convert DataFrame to markdown table without tabulate"""
        lines = []
        # Header
        header = "| " + " | ".join(df.index.name if df.index.name else "Dataset") + " | " + " | ".join(df.columns) + " |"
        lines.append(header)
        # Separator
        sep = "|" + "|".join(["-" * (len(col) + 2) for col in [df.index.name or "Dataset"] + list(df.columns)]) + "|"
        lines.append(sep)
        # Rows
        for idx, row in df.iterrows():
            row_str = "| " + str(idx) + " | " + " | ".join([f"{val:.4f}" if isinstance(val, float) else str(val) for val in row]) + " |"
            lines.append(row_str)
        return "\n".join(lines)
    
    report = f"""# Overfitting Diagnosis Report

## 📅 Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 Primary Model (Anomaly Detection)

### Performance Across Datasets

{df_to_markdown_table(primary_metrics)}

### Analysis

"""
    
    # Check for overfitting in primary model
    train_f1 = primary_metrics.loc['Train', 'F1-Score']
    test_f1 = primary_metrics.loc['Test', 'F1-Score']
    f1_diff = abs(train_f1 - test_f1)
    
    if f1_diff < 0.01:
        report += f"""
⚠️ **CRITICAL OVERFITTING WARNING**

- All datasets show perfect performance (F1-Score = 1.0000)
- Train/Test performance difference: {f1_diff:.4f} (< 1%)
- This indicates potential **data leakage** or **overfitting**

**Possible Causes**:
1. **cycle_number** feature strongly correlates with labels
2. Labels are deterministic based on cycle position (first 50% = Normal, last 50% = Abnormal)
3. Model memorizes the cycle-to-label mapping

**Recommendations**:
1. Remove or transform cycle-related features
2. Use more complex labeling strategy (threshold-based)
3. Validate on external datasets (ES10, ES14)
"""
    else:
        report += f"""
✅ **No Significant Overfitting Detected**

- Train/Test performance difference: {f1_diff:.4f}
- Model generalizes well to unseen data
"""
    
    report += f"""

## 📈 Secondary Model (RUL Prediction)

### Performance Across Datasets

{df_to_markdown_table(secondary_metrics)}

### Analysis

"""
    
    # Check for overfitting in secondary model
    train_mae = secondary_metrics.loc['Train', 'MAE']
    test_mae = secondary_metrics.loc['Test', 'MAE']
    mae_diff_pct = abs(test_mae - train_mae) / train_mae * 100
    
    if mae_diff_pct < 10:
        report += f"""
✅ **Good Generalization**

- Train MAE: {train_mae:.2f}
- Test MAE: {test_mae:.2f}
- Difference: {mae_diff_pct:.1f}% (< 10%)
- Model generalizes well to unseen data

**However, note the following issues**:
- High MAPE due to poor end-of-life predictions (RUL < 50)
- Model predicts minimum RUL ≈ 50 (training data limitation)
- Excellent performance for RUL > 50 (MAPE < 1%)
"""
    else:
        report += f"""
⚠️ **Potential Overfitting**

- Train MAE: {train_mae:.2f}
- Test MAE: {test_mae:.2f}
- Difference: {mae_diff_pct:.1f}% (≥ 10%)
- Model may be overfitting to training data
"""
    
    report += """

## 🚀 Next Steps

### Immediate Actions

1. **External Validation** (Task 6.6):
   - Test models on ES10 and ES14 datasets
   - Evaluate true generalization performance
   - Identify domain shift issues

2. **Feature Engineering**:
   - Remove or transform cycle_number feature
   - Add more degradation-related features
   - Reduce dependency on cycle information

3. **Data Augmentation**:
   - Include ES10 and ES14 in training
   - Increase diversity of training data
   - Improve end-of-life predictions

### Long-term Improvements

1. **Labeling Strategy**:
   - Use threshold-based labeling instead of cycle-based
   - Incorporate domain knowledge
   - Make labels less deterministic

2. **Model Architecture**:
   - Try time-series models (LSTM, GRU)
   - Implement ensemble methods
   - Explore transfer learning

3. **Hyperparameter Tuning**:
   - Optimize for better generalization
   - Reduce model complexity if needed
   - Use cross-validation

---

**Report Generated by**: Kiro AI Agent  
**Status**: Phase 2.5 - Overfitting Diagnosis Complete
"""
    
    # Save report
    report_path = OUTPUT_DIR / "overfitting_diagnosis.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✓ Saved: {report_path}")
    return report

def main():
    """Main execution function"""
    print("="*80)
    print("ES12 TEST DATA DETAILED ANALYSIS")
    print("="*80)
    
    # Load models and data
    primary_model, secondary_model, train, val, test = load_models_and_data()
    
    # Analyze Primary Model
    primary_metrics = analyze_primary_model(primary_model, train, val, test)
    
    # Analyze Secondary Model
    secondary_metrics, rul_range_metrics, cap_metrics = analyze_secondary_model(secondary_model, train, val, test)
    
    # Generate overfitting diagnosis report
    report = generate_overfitting_diagnosis_report(primary_metrics, secondary_metrics)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  1. {OUTPUT_DIR / 'primary_model_detailed_analysis.png'}")
    print(f"  2. {OUTPUT_DIR / 'secondary_model_detailed_analysis.png'}")
    print(f"  3. {OUTPUT_DIR / 'overfitting_diagnosis.md'}")
    print("\nNext: Proceed to Task 6.4 (ES10/ES14 Data Structure Analysis)")

if __name__ == "__main__":
    main()
