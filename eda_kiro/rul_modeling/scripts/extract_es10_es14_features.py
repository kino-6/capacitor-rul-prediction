"""
Task 4.1: ES10/ES14データの応答性特徴量抽出

目的:
- ES10/ES14データから同じ特徴量を抽出
- ES12と同じResponseFeatureExtractorを使用
- データ構造の確認と比較
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.feature_extraction.response_extractor import ResponseFeatureExtractor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 日本語フォント設定
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Hiragino Sans']
plt.rcParams['axes.unicode_minus'] = False

def extract_dataset_features(dataset_name):
    """
    指定されたデータセット（ES10 or ES14）の特徴量を抽出
    """
    print(f"\n{'='*60}")
    print(f"{dataset_name}データの特徴量抽出")
    print(f"{'='*60}")
    
    # データパス
    data_path = Path(f"../data/raw/{dataset_name}.mat")
    
    if not data_path.exists():
        print(f"❌ エラー: {data_path}が見つかりません")
        return None
    
    print(f"✓ データファイル: {data_path}")
    
    # MATファイルの読み込み（MATLAB v7.3形式）
    import h5py
    
    with h5py.File(str(data_path), 'r') as f:
        print(f"\nデータ構造:")
        for key in f.keys():
            if not key.startswith('#'):
                print(f"  - {key}: {type(f[key])}")
        
        # データセット名を確認
        if dataset_name not in f:
            print(f"❌ エラー: {dataset_name}キーが見つかりません")
            return None
        
        data_group = f[dataset_name]
        
        # Transient_Dataにアクセス
        if 'Transient_Data' not in data_group:
            print(f"❌ エラー: Transient_Dataが見つかりません")
            return None
        
        transient_data = data_group['Transient_Data']
        print(f"\nTransient_Dataのキー: {list(transient_data.keys())}")
        
        # ResponseFeatureExtractorの初期化
        extractor = ResponseFeatureExtractor()
        
        # 全特徴量の抽出
        print(f"\n特徴量抽出中...")
        
        all_features = []
        
        # コンデンサごとに処理
        capacitor_keys = [k for k in transient_data.keys() if k.startswith(dataset_name)]
        n_capacitors = len(capacitor_keys)
        
        print(f"  コンデンサ数: {n_capacitors}")
        
        for cap_key in sorted(capacitor_keys):
            cap_data = transient_data[cap_key]
            
            # VLとVOを取得（2D配列: samples × cycles）
            if 'VL' not in cap_data or 'VO' not in cap_data:
                print(f"  ⚠️ {cap_key}: VLまたはVOが見つかりません")
                continue
            
            vl_array = cap_data['VL'][:]  # (samples, cycles)
            vo_array = cap_data['VO'][:]  # (samples, cycles)
            
            n_samples, n_cycles = vl_array.shape
            
            print(f"  {cap_key}: {n_cycles}サイクル × {n_samples}サンプル")
            
            # 各サイクルを処理
            for cycle_idx in range(n_cycles):
                vl = vl_array[:, cycle_idx]
                vo = vo_array[:, cycle_idx]
                
                # NaN値を除去（ES10/ES14にはNaNが含まれる）
                valid_mask = ~(np.isnan(vl) | np.isnan(vo))
                vl = vl[valid_mask]
                vo = vo[valid_mask]
                
                # 有効なデータが少なすぎる場合はスキップ
                if len(vl) < 100:
                    print(f"    ⚠️ {cap_key} Cycle {cycle_idx + 1}: 有効データ不足 ({len(vl)}サンプル)")
                    continue
                
                # ES10/ES14は非常に大きな配列（75,826サンプル）なので、
                # 計算効率のためにダウンサンプリング
                # ES12と同程度のサンプル数（~1000）にする
                if len(vl) > 5000:
                    # 均等にサンプリング
                    downsample_factor = len(vl) // 1000
                    vl = vl[::downsample_factor]
                    vo = vo[::downsample_factor]
                
                # 特徴量抽出
                features = extractor.extract_features(
                    vl=vl,
                    vo=vo,
                    capacitor_id=cap_key,
                    cycle=cycle_idx + 1,
                    include_advanced=True
                )
                
                features['capacitor_id'] = cap_key
                features['cycle'] = cycle_idx + 1
                all_features.append(features)
                
                # 進捗表示（50サイクルごと）
                if (cycle_idx + 1) % 50 == 0:
                    print(f"    {cap_key}: {cycle_idx + 1}/{n_cycles}サイクル完了")
    
    features_df = pd.DataFrame(all_features)
    
    print(f"\n✓ 特徴量抽出完了: {len(features_df)}サンプル")
    print(f"  - コンデンサ数: {features_df['capacitor_id'].nunique()}")
    print(f"  - サイクル範囲: {features_df['cycle'].min()} - {features_df['cycle'].max()}")
    print(f"  - 特徴量数: {len(features_df.columns) - 2}個（capacitor_id, cycleを除く）")
    
    # 基本統計
    print(f"\n基本統計:")
    print(f"  Response Efficiency: {features_df['response_efficiency'].mean():.2f} ± {features_df['response_efficiency'].std():.2f}")
    print(f"  Waveform Correlation: {features_df['waveform_correlation'].mean():.3f} ± {features_df['waveform_correlation'].std():.3f}")
    print(f"  VO Variability: {features_df['vo_variability'].mean():.3f} ± {features_df['vo_variability'].std():.3f}")
    
    # 保存
    output_dir = Path("output/features_v3")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{dataset_name.lower()}_response_features.csv"
    features_df.to_csv(output_path, index=False)
    print(f"\n✓ 特徴量保存: {output_path}")
    
    return features_df


def compare_datasets(es10_df, es12_df, es14_df):
    """3つのデータセットの比較"""
    print(f"\n{'='*60}")
    print("データセット間の比較")
    print(f"{'='*60}")
    
    datasets = {
        'ES10': es10_df,
        'ES12': es12_df,
        'ES14': es14_df
    }
    
    # 基本統計の比較
    print(f"\n基本統計の比較:")
    print(f"{'Dataset':<10} {'Samples':<10} {'Capacitors':<12} {'Cycles':<10}")
    print("-" * 50)
    for name, df in datasets.items():
        n_samples = len(df)
        n_caps = df['capacitor_id'].nunique()
        n_cycles = df['cycle'].nunique()
        print(f"{name:<10} {n_samples:<10} {n_caps:<12} {n_cycles:<10}")
    
    # 主要特徴量の比較
    print(f"\n主要特徴量の平均値比較:")
    features_to_compare = [
        'response_efficiency',
        'waveform_correlation',
        'vo_variability',
        'vl_variability',
        'residual_energy_ratio'
    ]
    
    print(f"{'Feature':<30} {'ES10':<15} {'ES12':<15} {'ES14':<15}")
    print("-" * 75)
    for feature in features_to_compare:
        es10_mean = datasets['ES10'][feature].mean()
        es12_mean = datasets['ES12'][feature].mean()
        es14_mean = datasets['ES14'][feature].mean()
        print(f"{feature:<30} {es10_mean:<15.4f} {es12_mean:<15.4f} {es14_mean:<15.4f}")
    
    # 可視化
    visualize_dataset_comparison(datasets, features_to_compare)

def visualize_dataset_comparison(datasets, features):
    """データセット比較の可視化"""
    output_dir = Path("output/cross_dataset_validation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_features = len(features)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    colors = {'ES10': 'blue', 'ES12': 'green', 'ES14': 'red'}
    
    for idx, feature in enumerate(features):
        ax = axes[idx]
        
        for name, df in datasets.items():
            # サイクル平均
            cycle_avg = df.groupby('cycle')[feature].mean()
            ax.plot(cycle_avg.index, cycle_avg.values, 
                   label=name, color=colors[name], linewidth=2, alpha=0.7)
        
        ax.set_xlabel('Cycle', fontsize=12)
        ax.set_ylabel(feature.replace('_', ' ').title(), fontsize=12)
        ax.set_title(f'{feature.replace("_", " ").title()}\n（全コンデンサ平均）', 
                    fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # 最後のサブプロットを使用してサマリー
    ax = axes[-1]
    ax.axis('off')
    
    summary_text = """
    データセット比較サマリー:
    
    - ES10: 8コンデンサ × 200サイクル
    - ES12: 8コンデンサ × 200サイクル
    - ES14: 8コンデンサ × 200サイクル
    
    主な観察:
    - 劣化パターンの類似性
    - 初期状態の違い
    - 劣化速度の違い
    """
    
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='center', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('ES10/ES12/ES14データセット比較', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / "dataset_features_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 可視化保存: {output_path}")
    plt.close()

def main():
    print("="*60)
    print("Task 4.1: ES10/ES14データの応答性特徴量抽出")
    print("="*60)
    
    # ES10データの特徴量抽出
    es10_df = extract_dataset_features("ES10")
    
    if es10_df is None:
        print("❌ ES10データの抽出に失敗しました")
        return
    
    # ES14データの特徴量抽出
    es14_df = extract_dataset_features("ES14")
    
    if es14_df is None:
        print("❌ ES14データの抽出に失敗しました")
        return
    
    # ES12データの読み込み（比較用）
    es12_path = Path("output/features_v3/es12_response_features.csv")
    if es12_path.exists():
        es12_df = pd.read_csv(es12_path)
        print(f"\n✓ ES12データ読み込み: {len(es12_df)}サンプル")
        
        # 3つのデータセットを比較
        compare_datasets(es10_df, es12_df, es14_df)
    else:
        print(f"\n⚠️ ES12データが見つかりません: {es12_path}")
    
    print("\n" + "="*60)
    print("✓ Task 4.1完了: ES10/ES14データの特徴量抽出")
    print("="*60)
    print("\n次のステップ:")
    print("  - Task 4.2: ES10/ES14データでの異常検知評価")
    print("  - Task 4.3: ES10/ES14データでの劣化度予測評価")

if __name__ == "__main__":
    main()
