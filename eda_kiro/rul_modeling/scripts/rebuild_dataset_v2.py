"""
Dataset Rebuild Script (Version 2)

Purpose: Eliminate data leakage by removing cycle-related features
and use full cycle range (1-200) for training data.

Changes from v1:
1. Remove cycle_number and cycle_normalized features
2. Use full cycle range (1-200) for Train/Val sets
3. RUL range: 0-199 (previously 50-199)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle

# Paths
INPUT_DIR = Path("output/features")
OUTPUT_DIR = Path("output/features_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS_DIR = Path("output/models_v2")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_original_features():
    """Load original features with labels"""
    print("Loading original features...")
    df = pd.read_csv(INPUT_DIR / "es12_features_with_labels.csv")
    print(f"  Loaded: {len(df)} samples, {len(df.columns)} columns")
    return df

def remove_cycle_features(df):
    """Remove cycle-related features to eliminate data leakage"""
    print("\nRemoving cycle-related features...")
    
    # Features to remove
    features_to_remove = ['cycle_number', 'cycle_normalized']
    
    # Check if features exist
    existing_features = [f for f in features_to_remove if f in df.columns]
    print(f"  Features to remove: {existing_features}")
    
    # Remove features
    df_clean = df.drop(columns=existing_features, errors='ignore')
    
    print(f"  Before: {len(df.columns)} columns")
    print(f"  After: {len(df_clean.columns)} columns")
    print(f"  Removed: {len(existing_features)} features")
    
    return df_clean

def split_dataset_v2(df):
    """
    Split dataset with full cycle range
    
    v1 (old):
    - Train: C1-C5, cycles 1-150 (RUL: 50-199)
    - Val: C6, cycles 1-150 (RUL: 50-199)
    - Test: C7-C8, cycles 1-200 (RUL: 0-199)
    
    v2 (new):
    - Train: C1-C5, cycles 1-200 (RUL: 0-199) ← Full range
    - Val: C6, cycles 1-200 (RUL: 0-199) ← Full range
    - Test: C7-C8, cycles 1-200 (RUL: 0-199) ← No change
    """
    print("\nSplitting dataset (v2 strategy)...")
    
    # Train: C1-C5, all cycles
    train = df[df['capacitor_id'].isin(['ES12C1', 'ES12C2', 'ES12C3', 'ES12C4', 'ES12C5'])].copy()
    
    # Val: C6, all cycles
    val = df[df['capacitor_id'] == 'ES12C6'].copy()
    
    # Test: C7-C8, all cycles
    test = df[df['capacitor_id'].isin(['ES12C7', 'ES12C8'])].copy()
    
    print(f"  Train: {len(train)} samples (C1-C5, cycles 1-200)")
    print(f"    RUL range: {train['rul'].min()}-{train['rul'].max()}")
    print(f"    Normal: {(train['is_abnormal']==0).sum()}, Abnormal: {(train['is_abnormal']==1).sum()}")
    
    print(f"  Val: {len(val)} samples (C6, cycles 1-200)")
    print(f"    RUL range: {val['rul'].min()}-{val['rul'].max()}")
    print(f"    Normal: {(val['is_abnormal']==0).sum()}, Abnormal: {(val['is_abnormal']==1).sum()}")
    
    print(f"  Test: {len(test)} samples (C7-C8, cycles 1-200)")
    print(f"    RUL range: {test['rul'].min()}-{test['rul'].max()}")
    print(f"    Normal: {(test['is_abnormal']==0).sum()}, Abnormal: {(test['is_abnormal']==1).sum()}")
    
    return train, val, test

def scale_features(train, val, test):
    """Scale features using StandardScaler"""
    print("\nScaling features...")
    
    # Identify feature columns (exclude metadata and labels)
    feature_cols = [col for col in train.columns 
                   if col not in ['capacitor_id', 'cycle', 'is_abnormal', 'rul']]
    
    print(f"  Feature columns: {len(feature_cols)}")
    print(f"  Features: {feature_cols[:5]}... (showing first 5)")
    
    # Fit scaler on training data
    scaler = StandardScaler()
    scaler.fit(train[feature_cols])
    
    # Transform all datasets
    train_scaled = train.copy()
    val_scaled = val.copy()
    test_scaled = test.copy()
    
    train_scaled[feature_cols] = scaler.transform(train[feature_cols])
    val_scaled[feature_cols] = scaler.transform(val[feature_cols])
    test_scaled[feature_cols] = scaler.transform(test[feature_cols])
    
    # Save scaler
    scaler_path = MODELS_DIR / "scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  ✓ Saved scaler: {scaler_path}")
    
    return train_scaled, val_scaled, test_scaled, feature_cols

def save_datasets(train, val, test):
    """Save datasets to CSV"""
    print("\nSaving datasets...")
    
    train.to_csv(OUTPUT_DIR / "train.csv", index=False)
    print(f"  ✓ Saved: {OUTPUT_DIR / 'train.csv'}")
    
    val.to_csv(OUTPUT_DIR / "val.csv", index=False)
    print(f"  ✓ Saved: {OUTPUT_DIR / 'val.csv'}")
    
    test.to_csv(OUTPUT_DIR / "test.csv", index=False)
    print(f"  ✓ Saved: {OUTPUT_DIR / 'test.csv'}")

def generate_summary(train, val, test, feature_cols):
    """Generate dataset summary"""
    print("\nGenerating summary...")
    
    summary = f"""# Dataset Summary (Version 2)

## 📅 Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🔧 Changes from Version 1

### Data Leakage Elimination
- **Removed features**: cycle_number, cycle_normalized
- **Reason**: These features directly correlate with labels (data leakage)
- **Impact**: Model must learn from actual degradation patterns (VL/VO features)

### Full Cycle Range Usage
- **Train**: C1-C5, cycles 1-200 (previously 1-150)
- **Val**: C6, cycles 1-200 (previously 1-150)
- **Test**: C7-C8, cycles 1-200 (no change)
- **RUL range**: 0-199 (previously 50-199)
- **Impact**: Model can now learn end-of-life patterns (RUL < 50)

## 📊 Dataset Statistics

### Sample Counts

| Dataset | Samples | Capacitors | Cycles | RUL Range |
|---------|---------|------------|--------|-----------|
| Train   | {len(train)} | C1-C5 | 1-200 | {train['rul'].min()}-{train['rul'].max()} |
| Val     | {len(val)} | C6 | 1-200 | {val['rul'].min()}-{val['rul'].max()} |
| Test    | {len(test)} | C7-C8 | 1-200 | {test['rul'].min()}-{test['rul'].max()} |
| **Total** | **{len(train) + len(val) + len(test)}** | **8** | **1-200** | **0-199** |

### Label Distribution

#### Train Set
- Normal (is_abnormal=0): {(train['is_abnormal']==0).sum()} samples ({(train['is_abnormal']==0).sum()/len(train)*100:.1f}%)
- Abnormal (is_abnormal=1): {(train['is_abnormal']==1).sum()} samples ({(train['is_abnormal']==1).sum()/len(train)*100:.1f}%)

#### Val Set
- Normal (is_abnormal=0): {(val['is_abnormal']==0).sum()} samples ({(val['is_abnormal']==0).sum()/len(val)*100:.1f}%)
- Abnormal (is_abnormal=1): {(val['is_abnormal']==1).sum()} samples ({(val['is_abnormal']==1).sum()/len(val)*100:.1f}%)

#### Test Set
- Normal (is_abnormal=0): {(test['is_abnormal']==0).sum()} samples ({(test['is_abnormal']==0).sum()/len(test)*100:.1f}%)
- Abnormal (is_abnormal=1): {(test['is_abnormal']==1).sum()} samples ({(test['is_abnormal']==1).sum()/len(test)*100:.1f}%)

## 🔍 Feature Information

### Total Features: {len(feature_cols)}

**Feature Categories**:
- VL (Input Voltage): {len([f for f in feature_cols if f.startswith('vl_')])} features
- VO (Output Voltage): {len([f for f in feature_cols if f.startswith('vo_')])} features
- Degradation Indicators: {len([f for f in feature_cols if 'ratio' in f])} features

**Feature List**:
"""
    
    for i, feat in enumerate(feature_cols, 1):
        summary += f"{i}. {feat}\n"
    
    summary += f"""

## 📈 RUL Distribution

### Train Set
- Min RUL: {train['rul'].min()}
- Max RUL: {train['rul'].max()}
- Mean RUL: {train['rul'].mean():.2f}
- Median RUL: {train['rul'].median():.2f}

### Val Set
- Min RUL: {val['rul'].min()}
- Max RUL: {val['rul'].max()}
- Mean RUL: {val['rul'].mean():.2f}
- Median RUL: {val['rul'].median():.2f}

### Test Set
- Min RUL: {test['rul'].min()}
- Max RUL: {test['rul'].max()}
- Mean RUL: {test['rul'].mean():.2f}
- Median RUL: {test['rul'].median():.2f}

## 🎯 Expected Improvements

### Primary Model (Anomaly Detection)
- **v1 Performance**: F1-Score = 1.0000 (data leakage)
- **v2 Expected**: F1-Score = 0.75-0.85 (true performance)
- **Reason**: Model must learn from actual voltage patterns, not cycle numbers

### Secondary Model (RUL Prediction)
- **v1 Performance**: MAE (RUL 0-50) = 26.04 cycles
- **v2 Expected**: MAE (RUL 0-50) = 3-5 cycles
- **Reason**: Training data now includes RUL 0-50 range

## 📁 Generated Files

- `train.csv` - Training dataset ({len(train)} samples)
- `val.csv` - Validation dataset ({len(val)} samples)
- `test.csv` - Test dataset ({len(test)} samples)
- `scaler.pkl` - StandardScaler fitted on training data

---

**Generated by**: Kiro AI Agent  
**Status**: Dataset v2 Ready for Training
"""
    
    summary_path = OUTPUT_DIR / "dataset_summary_v2.md"
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    print(f"  ✓ Saved summary: {summary_path}")

def main():
    """Main execution"""
    print("="*80)
    print("DATASET REBUILD (VERSION 2)")
    print("="*80)
    print("\nObjective: Eliminate data leakage and use full cycle range")
    print("Changes:")
    print("  1. Remove cycle_number and cycle_normalized features")
    print("  2. Use full cycle range (1-200) for Train/Val")
    print("  3. RUL range: 0-199 (previously 50-199)")
    print("="*80)
    
    # Load original features
    df = load_original_features()
    
    # Remove cycle features
    df_clean = remove_cycle_features(df)
    
    # Split dataset
    train, val, test = split_dataset_v2(df_clean)
    
    # Scale features
    train_scaled, val_scaled, test_scaled, feature_cols = scale_features(train, val, test)
    
    # Save datasets
    save_datasets(train_scaled, val_scaled, test_scaled)
    
    # Generate summary
    generate_summary(train_scaled, val_scaled, test_scaled, feature_cols)
    
    print("\n" + "="*80)
    print("DATASET REBUILD COMPLETE!")
    print("="*80)
    print("\nGenerated files:")
    print(f"  1. {OUTPUT_DIR / 'train.csv'} ({len(train_scaled)} samples)")
    print(f"  2. {OUTPUT_DIR / 'val.csv'} ({len(val_scaled)} samples)")
    print(f"  3. {OUTPUT_DIR / 'test.csv'} ({len(test_scaled)} samples)")
    print(f"  4. {MODELS_DIR / 'scaler.pkl'}")
    print(f"  5. {OUTPUT_DIR / 'dataset_summary_v2.md'}")
    print("\nNext: Train models with new dataset (Task 6.8)")

if __name__ == "__main__":
    main()
