#!/usr/bin/env python3
"""
同一個体劣化可視化スクリプト

NASA PCOE ES12データから特定のコンデンサの劣化プロセスを詳細に分析・可視化します。
同一入力（周波数）に対する出力応答（インピーダンス、位相）の時間変化を追跡し、
劣化パターンを定量化します。
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import stats
from scipy.signal import savgol_filter
import japanize_matplotlib

# プロジェクトルートをパスに追加
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from nasa_pcoe_eda.data.es12_loader import ES12DataLoader
from nasa_pcoe_eda.visualization.engine import VisualizationEngine

# 警告を抑制
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

class IndividualDegradationAnalyzer:
    """同一個体の劣化分析・可視化クラス"""
    
    def __init__(self, output_dir: Path = Path("output/individual_degradation")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 日本語フォント設定
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
        
        # カラーパレット設定
        self.colors = plt.cm.viridis(np.linspace(0, 1, 10))
        
    def load_and_prepare_data(self, data_path: Path) -> pd.DataFrame:
        """ES12データを読み込み、分析用に準備"""
        print("📊 ES12データを読み込み中...")
        
        loader = ES12DataLoader()
        df = loader.load_dataset(data_path)
        
        print(f"✅ データ読み込み完了: {df.shape[0]}レコード, {df.shape[1]}特徴量")
        print(f"📋 利用可能なコンデンサ: {sorted(df['capacitor'].unique())}")
        print(f"📅 測定期間: {df['cycle'].min()} - {df['cycle'].max()}サイクル")
        
        return df
    
    def extract_capacitor_data(self, df: pd.DataFrame, capacitor_id: str) -> pd.DataFrame:
        """特定のコンデンサのデータを抽出"""
        cap_data = df[df['capacitor'] == capacitor_id].copy()
        
        if cap_data.empty:
            raise ValueError(f"コンデンサ {capacitor_id} のデータが見つかりません")
        
        # サイクル順にソート
        cap_data = cap_data.sort_values('cycle').reset_index(drop=True)
        
        print(f"🔍 {capacitor_id} データ抽出完了:")
        print(f"   - 測定サイクル数: {len(cap_data)}")
        print(f"   - サイクル範囲: {cap_data['cycle'].min()} - {cap_data['cycle'].max()}")
        
        return cap_data
    
    def analyze_response_degradation(self, cap_data: pd.DataFrame, capacitor_id: str) -> Dict:
        """応答劣化の詳細分析"""
        print(f"🔬 {capacitor_id} の応答劣化を分析中...")
        
        # 基本統計
        cycles = cap_data['cycle'].values
        n_cycles = len(cycles)
        
        # 電圧応答の変化分析
        vl_mean_values = cap_data['vl_mean'].values
        vo_mean_values = cap_data['vo_mean'].values
        voltage_ratio = cap_data['voltage_ratio'].values
        
        # 劣化率計算（線形回帰）
        def calculate_degradation_rate(x, y):
            """劣化率を線形回帰で計算"""
            if len(x) < 2 or np.all(np.isnan(y)) or np.var(y) == 0:
                return 0.0, 0.0, 0.0
            
            # 有効なデータポイントのみ使用
            valid_mask = ~(np.isnan(x) | np.isnan(y))
            if np.sum(valid_mask) < 2:
                return 0.0, 0.0, 0.0
            
            x_valid, y_valid = x[valid_mask], y[valid_mask]
            
            try:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)
                return slope, r_value**2, p_value
            except:
                return 0.0, 0.0, 1.0
        
        # 各パラメータの劣化率計算
        vl_slope, vl_r2, vl_p = calculate_degradation_rate(cycles, vl_mean_values)
        vo_slope, vo_r2, vo_p = calculate_degradation_rate(cycles, vo_mean_values)
        ratio_slope, ratio_r2, ratio_p = calculate_degradation_rate(cycles, voltage_ratio)
        
        # 変化率計算（初期値からの変化）
        def calculate_change_rate(values):
            """初期値からの変化率を計算"""
            if len(values) < 2 or values[0] == 0:
                return 0.0
            return ((values[-1] - values[0]) / abs(values[0])) * 100
        
        vl_change = calculate_change_rate(vl_mean_values)
        vo_change = calculate_change_rate(vo_mean_values)
        ratio_change = calculate_change_rate(voltage_ratio)
        
        # 劣化加速度検出（変化点分析）
        def detect_acceleration_points(x, y, window=5):
            """劣化加速点を検出"""
            if len(y) < window * 2:
                return []
            
            # 移動平均の勾配を計算
            smoothed = savgol_filter(y, min(window, len(y)//2*2-1), 1)
            gradients = np.gradient(smoothed)
            
            # 勾配の変化点を検出
            gradient_changes = np.abs(np.gradient(gradients))
            threshold = np.percentile(gradient_changes, 75)
            
            acceleration_points = []
            for i in range(window, len(x) - window):
                if gradient_changes[i] > threshold:
                    acceleration_points.append((x[i], y[i], gradient_changes[i]))
            
            return acceleration_points
        
        vl_accelerations = detect_acceleration_points(cycles, vl_mean_values)
        ratio_accelerations = detect_acceleration_points(cycles, voltage_ratio)
        
        analysis_result = {
            'capacitor_id': capacitor_id,
            'n_cycles': n_cycles,
            'cycle_range': (cycles[0], cycles[-1]),
            
            # 劣化率（サイクルあたりの変化）
            'vl_degradation_rate': vl_slope,
            'vo_degradation_rate': vo_slope,
            'ratio_degradation_rate': ratio_slope,
            
            # 決定係数（トレンドの信頼性）
            'vl_r_squared': vl_r2,
            'vo_r_squared': vo_r2,
            'ratio_r_squared': ratio_r2,
            
            # 総変化率（%）
            'vl_total_change': vl_change,
            'vo_total_change': vo_change,
            'ratio_total_change': ratio_change,
            
            # 劣化加速点
            'vl_acceleration_points': vl_accelerations,
            'ratio_acceleration_points': ratio_accelerations,
            
            # 生データ
            'cycles': cycles,
            'vl_mean': vl_mean_values,
            'vo_mean': vo_mean_values,
            'voltage_ratio': voltage_ratio
        }
        
        return analysis_result
    
    def visualize_individual_degradation(self, analysis_result: Dict) -> List[Path]:
        """個別劣化パターンの可視化"""
        capacitor_id = analysis_result['capacitor_id']
        print(f"📈 {capacitor_id} の劣化パターンを可視化中...")
        
        generated_plots = []
        
        # 1. 総合劣化トレンド
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{capacitor_id} 劣化応答分析 - 同一入力に対する出力変化', fontsize=16, fontweight='bold')
        
        cycles = analysis_result['cycles']
        vl_mean = analysis_result['vl_mean']
        vo_mean = analysis_result['vo_mean']
        voltage_ratio = analysis_result['voltage_ratio']
        
        # VL平均値の変化
        ax1 = axes[0, 0]
        ax1.plot(cycles, vl_mean, 'o-', color=self.colors[0], linewidth=2, markersize=4, alpha=0.8)
        
        # トレンドライン
        if analysis_result['vl_r_squared'] > 0.1:
            z = np.polyfit(cycles, vl_mean, 1)
            p = np.poly1d(z)
            ax1.plot(cycles, p(cycles), '--', color='red', alpha=0.7, linewidth=2)
            ax1.text(0.05, 0.95, f'変化率: {analysis_result["vl_total_change"]:.1f}%\nR²: {analysis_result["vl_r_squared"]:.3f}', 
                    transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.set_title('VL平均値の劣化トレンド', fontweight='bold')
        ax1.set_xlabel('測定サイクル')
        ax1.set_ylabel('VL平均値')
        ax1.grid(True, alpha=0.3)
        
        # VO平均値の変化
        ax2 = axes[0, 1]
        ax2.plot(cycles, vo_mean, 'o-', color=self.colors[1], linewidth=2, markersize=4, alpha=0.8)
        
        if analysis_result['vo_r_squared'] > 0.1:
            z = np.polyfit(cycles, vo_mean, 1)
            p = np.poly1d(z)
            ax2.plot(cycles, p(cycles), '--', color='red', alpha=0.7, linewidth=2)
            ax2.text(0.05, 0.95, f'変化率: {analysis_result["vo_total_change"]:.1f}%\nR²: {analysis_result["vo_r_squared"]:.3f}', 
                    transform=ax2.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.set_title('VO平均値の劣化トレンド', fontweight='bold')
        ax2.set_xlabel('測定サイクル')
        ax2.set_ylabel('VO平均値')
        ax2.grid(True, alpha=0.3)
        
        # 電圧比の変化
        ax3 = axes[1, 0]
        ax3.plot(cycles, voltage_ratio, 'o-', color=self.colors[2], linewidth=2, markersize=4, alpha=0.8)
        
        if analysis_result['ratio_r_squared'] > 0.1:
            z = np.polyfit(cycles, voltage_ratio, 1)
            p = np.poly1d(z)
            ax3.plot(cycles, p(cycles), '--', color='red', alpha=0.7, linewidth=2)
            ax3.text(0.05, 0.95, f'変化率: {analysis_result["ratio_total_change"]:.1f}%\nR²: {analysis_result["ratio_r_squared"]:.3f}', 
                    transform=ax3.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 劣化加速点をマーク
        if analysis_result['ratio_acceleration_points']:
            acc_cycles, acc_values, _ = zip(*analysis_result['ratio_acceleration_points'])
            ax3.scatter(acc_cycles, acc_values, color='red', s=100, marker='x', linewidth=3, 
                       label=f'劣化加速点 ({len(acc_cycles)}箇所)')
            ax3.legend()
        
        ax3.set_title('電圧比の劣化トレンド（応答性指標）', fontweight='bold')
        ax3.set_xlabel('測定サイクル')
        ax3.set_ylabel('電圧比 (VO/VL)')
        ax3.grid(True, alpha=0.3)
        
        # 劣化速度の変化
        ax4 = axes[1, 1]
        
        # 移動平均による劣化速度計算
        window = max(3, len(cycles) // 10)
        if len(voltage_ratio) >= window:
            smoothed_ratio = savgol_filter(voltage_ratio, min(window, len(voltage_ratio)//2*2-1), 1)
            degradation_speed = np.abs(np.gradient(smoothed_ratio))
            
            ax4.plot(cycles[1:], degradation_speed[1:], 'o-', color=self.colors[3], linewidth=2, markersize=3, alpha=0.8)
            ax4.axhline(y=np.mean(degradation_speed), color='red', linestyle='--', alpha=0.7, 
                       label=f'平均劣化速度: {np.mean(degradation_speed):.4f}')
            ax4.legend()
        
        ax4.set_title('劣化速度の変化', fontweight='bold')
        ax4.set_xlabel('測定サイクル')
        ax4.set_ylabel('劣化速度 (|d(電圧比)/dサイクル|)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / f'{capacitor_id}_degradation_overview.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_plots.append(plot_path)
        
        # 2. 詳細応答変化分析
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        fig.suptitle(f'{capacitor_id} 詳細応答変化分析 - 同一入力への応答劣化', fontsize=16, fontweight='bold')
        
        # 正規化された応答変化
        ax1 = axes[0]
        
        # 初期値で正規化
        vl_normalized = (vl_mean / vl_mean[0]) * 100 if vl_mean[0] != 0 else vl_mean
        vo_normalized = (vo_mean / vo_mean[0]) * 100 if vo_mean[0] != 0 else vo_mean
        ratio_normalized = (voltage_ratio / voltage_ratio[0]) * 100 if voltage_ratio[0] != 0 else voltage_ratio
        
        ax1.plot(cycles, vl_normalized, 'o-', label='VL応答 (正規化)', color=self.colors[0], linewidth=2, markersize=4)
        ax1.plot(cycles, vo_normalized, 's-', label='VO応答 (正規化)', color=self.colors[1], linewidth=2, markersize=4)
        ax1.plot(cycles, ratio_normalized, '^-', label='電圧比 (正規化)', color=self.colors[2], linewidth=2, markersize=4)
        
        ax1.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='初期値 (100%)')
        ax1.set_title('正規化応答変化 (初期値=100%)', fontweight='bold')
        ax1.set_xlabel('測定サイクル')
        ax1.set_ylabel('正規化応答値 (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 累積劣化量
        ax2 = axes[1]
        
        # 初期値からの累積変化量
        vl_cumulative = np.abs(vl_mean - vl_mean[0])
        vo_cumulative = np.abs(vo_mean - vo_mean[0])
        ratio_cumulative = np.abs(voltage_ratio - voltage_ratio[0])
        
        ax2.fill_between(cycles, 0, vl_cumulative, alpha=0.3, color=self.colors[0], label='VL累積変化')
        ax2.fill_between(cycles, 0, vo_cumulative, alpha=0.3, color=self.colors[1], label='VO累積変化')
        ax2.plot(cycles, ratio_cumulative, 'o-', color=self.colors[2], linewidth=3, markersize=4, label='電圧比累積変化')
        
        ax2.set_title('累積劣化量 (初期値からの絶対変化)', fontweight='bold')
        ax2.set_xlabel('測定サイクル')
        ax2.set_ylabel('累積変化量')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.output_dir / f'{capacitor_id}_detailed_response.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        generated_plots.append(plot_path)
        
        return generated_plots
    
    def compare_multiple_capacitors(self, df: pd.DataFrame, capacitor_ids: List[str]) -> Path:
        """複数コンデンサの劣化比較"""
        print(f"🔄 {len(capacitor_ids)}個のコンデンサを比較中...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('複数コンデンサの劣化パターン比較 - 個体差分析', fontsize=16, fontweight='bold')
        
        comparison_data = []
        
        for i, cap_id in enumerate(capacitor_ids):
            cap_data = self.extract_capacitor_data(df, cap_id)
            analysis = self.analyze_response_degradation(cap_data, cap_id)
            comparison_data.append(analysis)
            
            color = self.colors[i % len(self.colors)]
            
            # 電圧比の比較
            ax1 = axes[0, 0]
            ax1.plot(analysis['cycles'], analysis['voltage_ratio'], 'o-', 
                    label=f'{cap_id} (変化率: {analysis["ratio_total_change"]:.1f}%)', 
                    color=color, linewidth=2, markersize=3, alpha=0.8)
            
            # VL応答の比較
            ax2 = axes[0, 1]
            vl_normalized = (analysis['vl_mean'] / analysis['vl_mean'][0]) * 100
            ax2.plot(analysis['cycles'], vl_normalized, 'o-', 
                    label=f'{cap_id}', color=color, linewidth=2, markersize=3, alpha=0.8)
            
            # 劣化速度の比較
            ax3 = axes[1, 0]
            if len(analysis['voltage_ratio']) > 3:
                degradation_speed = np.abs(np.gradient(analysis['voltage_ratio']))
                ax3.plot(analysis['cycles'][1:], degradation_speed[1:], 'o-', 
                        label=f'{cap_id}', color=color, linewidth=2, markersize=3, alpha=0.8)
        
        # 電圧比比較
        ax1.set_title('電圧比劣化の個体差', fontweight='bold')
        ax1.set_xlabel('測定サイクル')
        ax1.set_ylabel('電圧比 (VO/VL)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # VL応答比較
        ax2.set_title('VL応答劣化の個体差 (正規化)', fontweight='bold')
        ax2.set_xlabel('測定サイクル')
        ax2.set_ylabel('正規化VL応答 (%)')
        ax2.axhline(y=100, color='black', linestyle='--', alpha=0.5)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 劣化速度比較
        ax3.set_title('劣化速度の個体差', fontweight='bold')
        ax3.set_xlabel('測定サイクル')
        ax3.set_ylabel('劣化速度')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 劣化統計サマリー
        ax4 = axes[1, 1]
        
        # 各コンデンサの劣化統計をバープロット
        cap_names = [data['capacitor_id'] for data in comparison_data]
        total_changes = [data['ratio_total_change'] for data in comparison_data]
        r_squared_values = [data['ratio_r_squared'] for data in comparison_data]
        
        x_pos = np.arange(len(cap_names))
        bars1 = ax4.bar(x_pos - 0.2, total_changes, 0.4, label='総変化率 (%)', alpha=0.8, color=self.colors[0])
        
        # 右軸でR²値を表示
        ax4_twin = ax4.twinx()
        bars2 = ax4_twin.bar(x_pos + 0.2, r_squared_values, 0.4, label='R² (信頼性)', alpha=0.8, color=self.colors[1])
        
        ax4.set_title('劣化統計サマリー', fontweight='bold')
        ax4.set_xlabel('コンデンサID')
        ax4.set_ylabel('総変化率 (%)', color=self.colors[0])
        ax4_twin.set_ylabel('R² 値', color=self.colors[1])
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(cap_names, rotation=45)
        ax4.grid(True, alpha=0.3)
        
        # 凡例を統合
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_twin.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        plt.tight_layout()
        
        plot_path = self.output_dir / 'capacitor_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def generate_degradation_report(self, analysis_results: List[Dict]) -> Path:
        """劣化分析レポートの生成"""
        print("📄 劣化分析レポートを生成中...")
        
        report_path = self.output_dir / 'degradation_analysis_report.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# 同一個体劣化分析レポート\n\n")
            f.write("## 概要\n\n")
            f.write("NASA PCOE ES12データセットから抽出した個別コンデンサの劣化パターン分析結果です。\n")
            f.write("同一入力に対する出力応答の変化を追跡し、劣化プロセスを定量化しました。\n\n")
            
            f.write("## 分析対象\n\n")
            f.write(f"- 分析コンデンサ数: {len(analysis_results)}\n")
            f.write(f"- 分析期間: {min(r['cycle_range'][0] for r in analysis_results)} - {max(r['cycle_range'][1] for r in analysis_results)} サイクル\n\n")
            
            f.write("## 個別分析結果\n\n")
            
            for result in analysis_results:
                cap_id = result['capacitor_id']
                f.write(f"### {cap_id}\n\n")
                f.write(f"- **測定サイクル数**: {result['n_cycles']}\n")
                f.write(f"- **電圧比総変化率**: {result['ratio_total_change']:.2f}%\n")
                f.write(f"- **VL応答総変化率**: {result['vl_total_change']:.2f}%\n")
                f.write(f"- **VO応答総変化率**: {result['vo_total_change']:.2f}%\n")
                f.write(f"- **電圧比トレンド信頼性 (R²)**: {result['ratio_r_squared']:.3f}\n")
                f.write(f"- **劣化加速点数**: {len(result['ratio_acceleration_points'])}\n\n")
                
                # 劣化評価
                if abs(result['ratio_total_change']) > 50:
                    status = "🔴 重度劣化"
                elif abs(result['ratio_total_change']) > 20:
                    status = "🟡 中度劣化"
                else:
                    status = "🟢 軽度劣化"
                
                f.write(f"**劣化状態**: {status}\n\n")
            
            f.write("## 劣化パターンの特徴\n\n")
            
            # 統計サマリー
            total_changes = [abs(r['ratio_total_change']) for r in analysis_results]
            f.write(f"- **平均劣化率**: {np.mean(total_changes):.2f}%\n")
            f.write(f"- **劣化率標準偏差**: {np.std(total_changes):.2f}%\n")
            f.write(f"- **最大劣化率**: {np.max(total_changes):.2f}%\n")
            f.write(f"- **最小劣化率**: {np.min(total_changes):.2f}%\n\n")
            
            # 劣化加速点の統計
            total_accelerations = sum(len(r['ratio_acceleration_points']) for r in analysis_results)
            f.write(f"- **総劣化加速点数**: {total_accelerations}\n")
            f.write(f"- **平均加速点数/個体**: {total_accelerations / len(analysis_results):.1f}\n\n")
            
            f.write("## 推奨事項\n\n")
            
            # 重度劣化個体の特定
            severe_degradation = [r for r in analysis_results if abs(r['ratio_total_change']) > 50]
            if severe_degradation:
                f.write("### 優先監視対象\n\n")
                for r in severe_degradation:
                    f.write(f"- **{r['capacitor_id']}**: 劣化率 {r['ratio_total_change']:.1f}% - 即座の交換を推奨\n")
                f.write("\n")
            
            f.write("### 保全戦略\n\n")
            f.write("1. **予防保全**: 劣化率20%を超えた個体の定期監視強化\n")
            f.write("2. **予測保全**: 劣化加速点検出時の詳細診断実施\n")
            f.write("3. **状態基準保全**: 個体差を考慮した個別保全計画策定\n\n")
            
            f.write("---\n")
            f.write(f"レポート生成日時: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        return report_path

def main():
    """メイン実行関数"""
    print("🚀 同一個体劣化可視化分析を開始します")
    print("=" * 60)
    
    # データパス設定
    data_path = Path("data/raw/ES12.mat")
    
    if not data_path.exists():
        print(f"❌ データファイルが見つかりません: {data_path}")
        return
    
    # 分析器初期化
    analyzer = IndividualDegradationAnalyzer()
    
    try:
        # データ読み込み
        df = analyzer.load_and_prepare_data(data_path)
        
        # 利用可能なコンデンサIDを取得
        available_capacitors = sorted(df['capacitor'].unique())
        print(f"\n📋 利用可能なコンデンサ: {available_capacitors}")
        
        # 分析対象を選択（最初の4個を例として）
        target_capacitors = available_capacitors[:4]
        print(f"🎯 分析対象: {target_capacitors}")
        
        analysis_results = []
        generated_plots = []
        
        # 各コンデンサの個別分析
        print(f"\n🔍 個別劣化分析を実行中...")
        for cap_id in target_capacitors:
            print(f"\n--- {cap_id} 分析開始 ---")
            
            # データ抽出
            cap_data = analyzer.extract_capacitor_data(df, cap_id)
            
            # 劣化分析
            analysis_result = analyzer.analyze_response_degradation(cap_data, cap_id)
            analysis_results.append(analysis_result)
            
            # 個別可視化
            plots = analyzer.visualize_individual_degradation(analysis_result)
            generated_plots.extend(plots)
            
            print(f"✅ {cap_id} 分析完了")
            print(f"   - 電圧比変化率: {analysis_result['ratio_total_change']:.2f}%")
            print(f"   - 劣化加速点: {len(analysis_result['ratio_acceleration_points'])}箇所")
        
        # 比較分析
        print(f"\n🔄 複数コンデンサ比較分析...")
        comparison_plot = analyzer.compare_multiple_capacitors(df, target_capacitors)
        generated_plots.append(comparison_plot)
        
        # レポート生成
        report_path = analyzer.generate_degradation_report(analysis_results)
        
        # 結果サマリー
        print(f"\n" + "=" * 60)
        print("✅ 同一個体劣化分析が完了しました！")
        print(f"\n📊 分析結果:")
        print(f"   - 分析対象: {len(target_capacitors)}個のコンデンサ")
        print(f"   - 生成された可視化: {len(generated_plots)}個")
        print(f"   - 出力ディレクトリ: {analyzer.output_dir}")
        
        print(f"\n📈 劣化サマリー:")
        for result in analysis_results:
            status = "🔴重度" if abs(result['ratio_total_change']) > 50 else "🟡中度" if abs(result['ratio_total_change']) > 20 else "🟢軽度"
            print(f"   - {result['capacitor_id']}: {status} (変化率: {result['ratio_total_change']:.1f}%)")
        
        print(f"\n📁 生成ファイル:")
        for plot_path in generated_plots:
            print(f"   - {plot_path.name}")
        print(f"   - {report_path.name}")
        
        print(f"\n💡 推奨事項:")
        severe_count = sum(1 for r in analysis_results if abs(r['ratio_total_change']) > 50)
        if severe_count > 0:
            print(f"   - {severe_count}個のコンデンサが重度劣化状態です")
            print(f"   - 優先的な監視・交換を推奨します")
        else:
            print(f"   - 全コンデンサが正常範囲内です")
            print(f"   - 定期監視を継続してください")
        
    except Exception as e:
        print(f"❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()