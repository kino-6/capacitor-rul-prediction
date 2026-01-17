"""
Phase 0: 探索的特徴量分析

モデル実装前に、特徴量とRULの相関を確認し、有効な特徴量を特定する。
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import h5py

# EDAのコードを再利用
from src.es12_loader import ES12Loader


def extract_basic_features(vl: np.ndarray, vo: np.ndarray, cycle_num: int) -> dict:
    """
    1サイクルから基本的な特徴量を抽出
    
    Args:
        vl: VL時系列データ
        vo: VO時系列データ
        cycle_num: サイクル番号
    
    Returns:
        特徴量辞書
    """
    features = {}
    
    # 基本統計量 - VL
    features['vl_mean'] = np.mean(vl)
    features['vl_std'] = np.std(vl)
    features['vl_min'] = np.min(vl)
    features['vl_max'] = np.max(vl)
    features['vl_range'] = features['vl_max'] - features['vl_min']
    features['vl_median'] = np.median(vl)
    
    # 基本統計量 - VO
    features['vo_mean'] = np.mean(vo)
    features['vo_std'] = np.std(vo)
    features['vo_min'] = np.min(vo)
    features['vo_max'] = np.max(vo)
    features['vo_range'] = features['vo_max'] - features['vo_min']
    features['vo_median'] = np.median(vo)
    
    # 劣化指標
    features['voltage_ratio'] = features['vo_mean'] / features['vl_mean'] if features['vl_mean'] != 0 else 0
    features['response_efficiency'] = features['vo_range'] / features['vl_range'] if features['vl_range'] != 0 else 0
    
    # サイクル情報
    features['cycle_number'] = cycle_num
    features['cycle_normalized'] = cycle_num / 200.0
    
    return features


def main():
    print("=" * 80)
    print("Phase 0: 探索的特徴量分析")
    print("=" * 80)
    
    # データ読み込み
    print("\n[Step 1] ES12データの読み込み...")
    es12_path = project_root / "data" / "raw" / "ES12.mat"
    
    if not es12_path.exists():
        print(f"エラー: {es12_path} が見つかりません")
        return
    
    loader = ES12Loader(str(es12_path))
    
    # ES12C1の最初の10サイクルで試す
    print("\n[Step 2] ES12C1の最初の10サイクルで特徴量抽出...")
    cap_id = "ES12C1"
    features_list = []
    
    for cycle in range(1, 11):
        vl, vo = loader.get_cycle_data(cap_id, cycle)
        features = extract_basic_features(vl, vo, cycle)
        features['capacitor_id'] = cap_id
        features_list.append(features)
    
    df_sample = pd.DataFrame(features_list)
    print(f"\n抽出した特徴量: {len(df_sample.columns)}個")
    print(df_sample.head())
    
    # 全サイクルで特徴量抽出
    print(f"\n[Step 3] {cap_id}の全サイクル（200サイクル）で特徴量抽出...")
    features_list = []
    
    for cycle in range(1, 201):
        vl, vo = loader.get_cycle_data(cap_id, cycle)
        features = extract_basic_features(vl, vo, cycle)
        features['capacitor_id'] = cap_id
        features_list.append(features)
    
    df_full = pd.DataFrame(features_list)
    
    # RULを計算
    df_full['rul'] = 200 - df_full['cycle_number']
    
    print(f"データセット形状: {df_full.shape}")
    print(f"\n統計情報:")
    print(df_full[['cycle_number', 'voltage_ratio', 'rul']].describe())
