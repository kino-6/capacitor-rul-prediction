"""Script to visualize Primary Model feature importance."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


def main():
    """Visualize Primary Model feature importance."""
    print("=" * 80)
    print("PRIMARY MODEL - FEATURE IMPORTANCE VISUALIZATION")
    print("=" * 80)
    print()
    
    # Paths
    model_dir = Path(__file__).parent.parent / "output" / "models"
    eval_dir = Path(__file__).parent.parent / "output" / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    # Load feature importance
    print("📂 Loading feature importance...")
    importance_path = model_dir / "primary_feature_importance.csv"
    importance_df = pd.read_csv(importance_path)
    print(f"  ✓ Loaded {len(importance_df)} features")
    print()
    
    # Create visualization
    print("📊 Creating visualization...")
    
    # Plot top 15 features
    top_n = 15
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create horizontal bar plot
    bars = ax.barh(
        range(len(top_features)),
        top_features['importance'],
        color='steelblue',
        alpha=0.8
    )
    
    # Customize plot
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Primary Model - Top {top_n} Feature Importance\n'
        'Random Forest Classifier (Anomaly Detection)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax.text(
            row['importance'] + 0.002,
            i,
            f"{row['importance']:.4f}",
            va='center',
            fontsize=10
        )
    
    # Invert y-axis to show highest importance at top
    ax.invert_yaxis()
    
    # Add grid
    ax.grid(axis='x', alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    output_path = eval_dir / "primary_feature_importance.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved to: {output_path}")
    print()
    
    # Display summary statistics
    print("📊 Feature Importance Summary:")
    print("-" * 80)
    print(f"  Total features:        {len(importance_df)}")
    print(f"  Max importance:        {importance_df['importance'].max():.4f} ({importance_df.iloc[0]['feature']})")
    print(f"  Min importance:        {importance_df['importance'].min():.4f} ({importance_df.iloc[-1]['feature']})")
    print(f"  Mean importance:       {importance_df['importance'].mean():.4f}")
    print(f"  Std importance:        {importance_df['importance'].std():.4f}")
    print()
    
    # Top 5 features
    print("📊 Top 5 Most Important Features:")
    print("-" * 80)
    for idx, row in importance_df.head(5).iterrows():
        print(f"  {idx+1}. {row['feature']:25s}: {row['importance']:.4f}")
    print()
    
    # Feature categories
    print("📊 Feature Importance by Category:")
    print("-" * 80)
    
    # Categorize features
    categories = {
        'Cycle Info': ['cycle_number', 'cycle_normalized'],
        'VL (Input)': [f for f in importance_df['feature'] if f.startswith('vl_')],
        'VO (Output)': [f for f in importance_df['feature'] if f.startswith('vo_')],
        'Degradation': ['voltage_ratio', 'voltage_ratio_std', 'response_efficiency', 'signal_attenuation']
    }
    
    for category, features in categories.items():
        cat_importance = importance_df[importance_df['feature'].isin(features)]['importance'].sum()
        print(f"  {category:20s}: {cat_importance:.4f} ({cat_importance/importance_df['importance'].sum()*100:.1f}%)")
    print()
    
    print("=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
