"""
最終プロジェクトレポート生成スクリプト

ES12データでの成果を統合し、可視化を含む包括的なレポートを作成
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

def load_all_results():
    """全ての結果データを読み込み"""
    print("結果データの読み込み中...")
    
    results = {}
    
    # 特徴量データ
    features_path = Path("output/features_v3/es12_response_features.csv")
    results['features'] = pd.read_csv(features_path)
    print(f"  ✓ 特徴量: {len(results['features'])}サンプル")
    
    # 異常検知結果
    anomaly_path = Path("output/anomaly_detection/one_class_svm_v2_results.csv")
    results['anomaly'] = pd.read_csv(anomaly_path)
    print(f"  ✓ 異常検知結果: {len(results['anomaly'])}サンプル")
    
    # 劣化度予測結果
    degradation_path = Path("output/degradation_prediction/features_with_degradation_score.csv")
    results['degradation'] = pd.read_csv(degradation_path)
    print(f"  ✓ 劣化度スコア: {len(results['degradation'])}サンプル")
    
    # モデル読み込み
    model_path = Path("output/models_v3/one_class_svm_v2.pkl")
    with open(model_path, 'rb') as f:
        results['anomaly_model'] = pickle.load(f)
    
    degradation_model_path = Path("output/models_v3/degradation_predictor.pkl")
    with open(degradation_model_path, 'rb') as f:
        results['degradation_model'] = pickle.load(f)
    
    print("  ✓ モデル読み込み完了\n")
    
    return results

def create_overview_visualization(results):
    """プロジェクト概要の可視化"""
