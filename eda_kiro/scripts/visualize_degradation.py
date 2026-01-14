#!/usr/bin/env python3
"""
NASA PCOE Dataset No.12 劣化可視化スクリプト

コンデンサの充電・放電サイクルにおける劣化パターンを可視化します。
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from nasa_pcoe_eda.analysis.statistics import StatisticsAnalyzer
from nasa_pcoe_eda.data.loader import DataLoader

# 日本語フォント設定
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Hiragino Sans", "Yu Gothic", "Meiryo"]
plt.rcParams["axes.unicode_minus"] = False


def setup_output_directory():
    """出力ディレクトリを作成"""
    output_dir = Path("output/figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_data(data_path: str = "../12.CapacitorElectricalStress/ES12.mat"):
    """データを読み込む"""
    loader = DataLoader()
    df = loader.load_dataset(Path(data_path))
    metadata = loader.get_metadata(df)
    
    print(f"データ読み込み完了:")
    print(f"  レコード数: {metadata.n_records}")
    print(f"  特徴量数: {metadata.n_features}")
    print(f"  メモリ使用量: {metadata.memory_usage:.2f} MB")
    print(f"\n特徴量名（最初の10個）:")
    for i, name in enumerate(metadata.feature_names[:10], 1):
        print(f"  {i}. {name}")
    if len(metadata.feature_names) > 10:
        print(f"  ... 他 {len(metadata.feature_names) - 10} 個")
    
    return df, metadata


def analyze_statistics(df: pd.DataFrame):
    """基本統計量を計算（サンプルのみ）"""
    # 特徴量が多すぎる場合は最初の10個だけ分析
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    sample_cols = numeric_cols[:10]
    df_sample = df[sample_cols]
    
    analyzer = StatisticsAnalyzer()
    stats = analyzer.compute_descriptive_stats(df_sample)
    missing = analyzer.analyze_missing_values(df_sample)
    
    print(f"\n基本統計量（最初の5特徴量）:")
    for feature, stat in list(stats.items())[:5]:
        print(f"\n{feature}:")
        print(f"  平均: {stat.mean:.4f}")
        print(f"  中央値: {stat.median:.4f}")
        print(f"  標準偏差: {stat.std:.4f}")
        print(f"  範囲: [{stat.min:.4f}, {stat.max:.4f}]")
    
    print(f"\n欠損値:")
    print(f"  総欠損値数: {missing.total_missing}")
    
    return stats, missing


def plot_time_series_overview(df: pd.DataFrame, output_dir: Path):
    """時系列データの概要をプロット（サイクルごとの変化）"""
    # 数値列のみを取得
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        print("数値データが見つかりません")
        return
    
    # 変動が大きい特徴量を選択（劣化を示す可能性が高い）
    # 標準偏差が大きい特徴量を選ぶ
    std_values = df[numeric_cols].std()
    # ゼロでない標準偏差を持つ特徴量を選択
    non_zero_std = std_values[std_values > 0].sort_values(ascending=False)
    
    if len(non_zero_std) == 0:
        print("変動のある特徴量が見つかりません")
        return
    
    # 上位6つの変動が大きい特徴量を選択
    selected_features = non_zero_std.head(6).index.tolist()
    
    n_features = len(selected_features)
    fig, axes = plt.subplots(n_features, 1, figsize=(14, 3 * n_features))
    
    if n_features == 1:
        axes = [axes]
    
    for i, col in enumerate(selected_features):
        # サイクル番号（行インデックス）をX軸に使用
        cycles = range(len(df))
        values = df[col].values
        
        axes[i].plot(cycles, values, marker='o', linewidth=2, markersize=6, alpha=0.8)
        axes[i].set_ylabel(col, fontsize=10)
        axes[i].set_xlabel("サイクル番号（時間経過）", fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_title(f"{col} の劣化推移（サイクルごと）", fontsize=12, fontweight='bold')
        
        # トレンドラインを追加
        z = np.polyfit(cycles, values, 1)
        p = np.poly1d(z)
        axes[i].plot(cycles, p(cycles), "--", linewidth=2, alpha=0.7, 
                    label=f"トレンド (傾き: {z[0]:.2e})")
        axes[i].legend(loc='best')
    
    plt.tight_layout()
    output_path = output_dir / "time_series_overview.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n時系列概要プロットを保存: {output_path}")
    plt.close()


def plot_degradation_patterns(df: pd.DataFrame, output_dir: Path):
    """劣化パターンを可視化（サイクルごとの変化率も表示）"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        return
    
    # 変動が大きい特徴量を選択
    std_values = df[numeric_cols].std()
    non_zero_std = std_values[std_values > 0].sort_values(ascending=False)
    
    if len(non_zero_std) < 4:
        print("十分な変動のある特徴量が見つかりません")
        return
    
    selected_features = non_zero_std.head(4).index.tolist()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    cycles = range(len(df))
    
    for i, col in enumerate(selected_features):
        data = df[col].values
        
        # 正規化（初期値を100%として）
        if data[0] != 0:
            normalized_data = (data / data[0]) * 100
        else:
            normalized_data = data
        
        # プロット
        axes[i].plot(cycles, normalized_data, marker='o', linewidth=2.5, 
                    markersize=8, alpha=0.8, color='blue', label='測定値')
        
        # トレンドラインを追加
        z = np.polyfit(cycles, normalized_data, 1)
        p = np.poly1d(z)
        axes[i].plot(cycles, p(cycles), "--", linewidth=2.5, alpha=0.7, 
                    color='red', label=f"トレンド (変化率: {z[0]:.2f}%/サイクル)")
        
        # 初期値の参照線
        axes[i].axhline(y=100, color='green', linestyle=':', linewidth=2, 
                       alpha=0.5, label='初期値 (100%)')
        
        axes[i].set_xlabel("サイクル番号（時間経過）", fontsize=11, fontweight='bold')
        axes[i].set_ylabel("正規化値 (%)", fontsize=11, fontweight='bold')
        axes[i].set_title(f"{col}\nの劣化推移（初期値=100%）", fontsize=12, fontweight='bold')
        axes[i].legend(loc="best", fontsize=9)
        axes[i].grid(True, alpha=0.3, linestyle='--')
        
        # 劣化率を計算して表示
        if len(data) > 1:
            total_change = ((data[-1] - data[0]) / data[0] * 100) if data[0] != 0 else 0
            axes[i].text(0.02, 0.98, f'総変化率: {total_change:.2f}%', 
                        transform=axes[i].transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle='round', 
                        facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = output_dir / "degradation_patterns.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"劣化パターンプロットを保存: {output_path}")
    plt.close()


def plot_distribution_changes(df: pd.DataFrame, output_dir: Path):
    """サイクルごとの主要指標の変化を可視化"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols or len(df) < 3:
        return
    
    # 変動が大きい特徴量を選択
    std_values = df[numeric_cols].std()
    non_zero_std = std_values[std_values > 0].sort_values(ascending=False)
    
    if len(non_zero_std) < 4:
        return
    
    selected_features = non_zero_std.head(4).index.tolist()
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    cycles = range(len(df))
    
    # 初期、中期、後期のサイクルを定義
    n = len(df)
    early_idx = n // 3
    late_idx = 2 * n // 3
    
    for i, col in enumerate(selected_features):
        data = df[col].values
        
        # プロット
        axes[i].plot(cycles, data, linewidth=2.5, alpha=0.8, color='navy')
        axes[i].scatter(cycles, data, s=100, alpha=0.6, c=cycles, cmap='RdYlGn_r', 
                       edgecolors='black', linewidth=1.5)
        
        # 期間を色分けして背景に表示
        axes[i].axvspan(0, early_idx, alpha=0.1, color='green', label='初期')
        axes[i].axvspan(early_idx, late_idx, alpha=0.1, color='orange', label='中期')
        axes[i].axvspan(late_idx, n, alpha=0.1, color='red', label='後期')
        
        # 各期間の平均値を表示
        early_mean = data[:early_idx].mean()
        middle_mean = data[early_idx:late_idx].mean()
        late_mean = data[late_idx:].mean()
        
        axes[i].axhline(y=early_mean, xmin=0, xmax=early_idx/n, 
                       color='green', linestyle='--', linewidth=2, alpha=0.7)
        axes[i].axhline(y=middle_mean, xmin=early_idx/n, xmax=late_idx/n, 
                       color='orange', linestyle='--', linewidth=2, alpha=0.7)
        axes[i].axhline(y=late_mean, xmin=late_idx/n, xmax=1, 
                       color='red', linestyle='--', linewidth=2, alpha=0.7)
        
        axes[i].set_xlabel("サイクル番号", fontsize=11, fontweight='bold')
        axes[i].set_ylabel(col, fontsize=11, fontweight='bold')
        axes[i].set_title(f"{col}\nサイクルごとの変化（色：緑→黄→赤 = 劣化進行）", 
                         fontsize=12, fontweight='bold')
        axes[i].legend(loc="best", fontsize=9)
        axes[i].grid(True, alpha=0.3, linestyle='--')
        
        # 統計情報を表示
        change_pct = ((late_mean - early_mean) / early_mean * 100) if early_mean != 0 else 0
        info_text = f'初期平均: {early_mean:.2e}\n後期平均: {late_mean:.2e}\n変化: {change_pct:.1f}%'
        axes[i].text(0.02, 0.98, info_text, transform=axes[i].transAxes, 
                    fontsize=9, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    output_path = output_dir / "cycle_by_cycle_changes.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"サイクルごとの変化プロットを保存: {output_path}")
    plt.close()


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Path):
    """相関ヒートマップを作成（サンプル特徴量のみ）"""
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.shape[1] < 2:
        return
    
    # 特徴量が多すぎる場合は最初の20個だけ使用
    max_features = 20
    if numeric_df.shape[1] > max_features:
        print(f"\n注: 相関ヒートマップは最初の{max_features}特徴量のみを表示します")
        numeric_df = numeric_df.iloc[:, :max_features]
    
    # 相関行列を計算
    corr_matrix = numeric_df.corr()
    
    # ヒートマップをプロット
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
    
    # カラーバーを追加
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("相関係数", rotation=270, labelpad=20)
    
    # 軸ラベルを設定
    ax.set_xticks(range(len(corr_matrix.columns)))
    ax.set_yticks(range(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr_matrix.columns, fontsize=8)
    
    # 相関係数の値を表示（小さいフォントで）
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            text = ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                          ha="center", va="center", color="black", fontsize=6)
    
    ax.set_title("特徴量間の相関ヒートマップ（サンプル）")
    plt.tight_layout()
    
    output_path = output_dir / "correlation_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"相関ヒートマップを保存: {output_path}")
    plt.close()


def main():
    """メイン処理"""
    print("=" * 60)
    print("NASA PCOE Dataset No.12 劣化可視化")
    print("=" * 60)
    
    # 出力ディレクトリを作成
    output_dir = setup_output_directory()
    
    # データを読み込む
    try:
        df, metadata = load_data()
    except Exception as e:
        print(f"\nエラー: データの読み込みに失敗しました: {e}")
        print("\nデータファイルが存在することを確認してください:")
        print("  ../12.CapacitorElectricalStress/ES12.mat")
        return
    
    # 統計分析
    stats, missing = analyze_statistics(df)
    
    # 可視化
    print("\n" + "=" * 60)
    print("可視化を生成中...")
    print("=" * 60)
    
    plot_time_series_overview(df, output_dir)
    plot_degradation_patterns(df, output_dir)
    plot_distribution_changes(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    
    print("\n" + "=" * 60)
    print("完了！")
    print(f"全ての図は {output_dir} に保存されました")
    print("=" * 60)


if __name__ == "__main__":
    main()
