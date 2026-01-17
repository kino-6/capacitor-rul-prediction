"""
Feature Scaler for RUL Prediction Model

This module provides functionality to scale features using StandardScaler.
Metadata columns (capacitor_id, cycle, is_abnormal, rul) are excluded from scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path
from typing import Tuple, List
from tqdm import tqdm


class FeatureScaler:
    """
    Scale features using StandardScaler.
    
    Metadata columns are excluded from scaling:
    - capacitor_id
    - cycle
    - is_abnormal
    - rul
    """
    
    def __init__(self):
        """Initialize the FeatureScaler."""
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.metadata_columns = ['capacitor_id', 'cycle', 'is_abnormal', 'rul']
        
    def fit(self, train_df: pd.DataFrame) -> 'FeatureScaler':
        """
        Fit the scaler on training data.
        
        Args:
            train_df: Training dataframe
            
        Returns:
            self: Fitted scaler
        """
        # Identify feature columns (exclude metadata)
        self.feature_columns = [
            col for col in train_df.columns 
            if col not in self.metadata_columns
        ]
        
        print(f"Fitting scaler on {len(self.feature_columns)} feature columns...")
        print(f"Excluded metadata columns: {self.metadata_columns}")
        
        # Fit scaler on feature columns only
        self.scaler.fit(train_df[self.feature_columns])
        
        print("✓ Scaler fitted successfully")
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features using the fitted scaler.
        
        Args:
            df: Dataframe to transform
            
        Returns:
            Transformed dataframe with scaled features and original metadata
        """
        if self.feature_columns is None:
            raise ValueError("Scaler must be fitted before transform. Call fit() first.")
        
        # Create a copy to avoid modifying original
        df_scaled = df.copy()
        
        # Scale feature columns
        df_scaled[self.feature_columns] = self.scaler.transform(df[self.feature_columns])
        
        return df_scaled
    
    def fit_transform(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the scaler and transform training data.
        
        Args:
            train_df: Training dataframe
            
        Returns:
            Transformed training dataframe
        """
        self.fit(train_df)
        return self.transform(train_df)
    
    def save(self, path: str) -> None:
        """
        Save the scaler to a pickle file.
        
        Args:
            path: Path to save the scaler
        """
        # Create directory if it doesn't exist
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"✓ Scaler saved to: {path}")
    
    def load(self, path: str) -> 'FeatureScaler':
        """
        Load a scaler from a pickle file.
        
        Args:
            path: Path to load the scaler from
            
        Returns:
            self: Loaded scaler
        """
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        print(f"✓ Scaler loaded from: {path}")
        return self
    
    def get_feature_stats(self) -> pd.DataFrame:
        """
        Get statistics about the fitted scaler.
        
        Returns:
            DataFrame with mean and std for each feature
        """
        if self.feature_columns is None:
            raise ValueError("Scaler must be fitted first.")
        
        stats_df = pd.DataFrame({
            'feature': self.feature_columns,
            'mean': self.scaler.mean_,
            'std': self.scaler.scale_
        })
        
        return stats_df


def scale_and_save_datasets(
    train_path: str,
    val_path: str,
    test_path: str,
    output_dir: str,
    scaler_path: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Scale train/val/test datasets and save them.
    
    Args:
        train_path: Path to training data
        val_path: Path to validation data
        test_path: Path to test data
        output_dir: Directory to save scaled datasets
        scaler_path: Path to save the scaler
        
    Returns:
        Tuple of (train_scaled, val_scaled, test_scaled)
    """
    print("=" * 60)
    print("Feature Scaling Pipeline")
    print("=" * 60)
    
    # Load datasets
    print("\n[1/5] Loading datasets...")
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    print(f"  Train: {train_df.shape}")
    print(f"  Val:   {val_df.shape}")
    print(f"  Test:  {test_df.shape}")
    
    # Initialize and fit scaler
    print("\n[2/5] Fitting scaler on training data...")
    scaler = FeatureScaler()
    train_scaled = scaler.fit_transform(train_df)
    
    # Transform val and test
    print("\n[3/5] Transforming validation and test data...")
    val_scaled = scaler.transform(val_df)
    test_scaled = scaler.transform(test_df)
    
    # Save scaler
    print("\n[4/5] Saving scaler...")
    scaler.save(scaler_path)
    
    # Save scaled datasets
    print("\n[5/5] Saving scaled datasets...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_output = output_path / "train_scaled.csv"
    val_output = output_path / "val_scaled.csv"
    test_output = output_path / "test_scaled.csv"
    
    train_scaled.to_csv(train_output, index=False)
    val_scaled.to_csv(val_output, index=False)
    test_scaled.to_csv(test_output, index=False)
    
    print(f"  ✓ Train: {train_output}")
    print(f"  ✓ Val:   {val_output}")
    print(f"  ✓ Test:  {test_output}")
    
    # Print scaler statistics
    print("\n" + "=" * 60)
    print("Scaler Statistics (first 10 features)")
    print("=" * 60)
    stats_df = scaler.get_feature_stats()
    print(stats_df.head(10).to_string(index=False))
    
    print("\n" + "=" * 60)
    print("Feature Scaling Complete!")
    print("=" * 60)
    
    return train_scaled, val_scaled, test_scaled


if __name__ == "__main__":
    # Paths
    train_path = "output/features/train.csv"
    val_path = "output/features/val.csv"
    test_path = "output/features/test.csv"
    output_dir = "output/features"
    scaler_path = "output/models/scaler.pkl"
    
    # Run scaling pipeline
    train_scaled, val_scaled, test_scaled = scale_and_save_datasets(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        output_dir=output_dir,
        scaler_path=scaler_path
    )
