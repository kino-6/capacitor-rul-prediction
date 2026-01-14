# 実データ分析ガイド

NASA PCOE ES12データセットの実測データを用いた高精度分析機能のガイドです。

## 概要

実データ分析機能は、NASA PCOE Dataset No.12の実際のコンデンサ劣化試験データを用いて、以下の高度な分析を提供します：

- **実測データ専用の読み込み・処理機能**
- **劣化パターンの詳細分析**
- **個別コンデンサの特性比較**
- **分析手法の妥当性検証**
- **実用的な故障予測・保全指針の提供**

## 主要機能

### 1. 実データ専用データローダー (ES12DataLoader)

```python
from nasa_pcoe_eda.data.es12_loader import ES12DataLoader

loader = ES12DataLoader()
df = loader.load_dataset("data/raw/ES12.mat")
```

**特徴:**
- MATLAB v7.3 (HDF5) 形式の正確な読み込み
- 8個のコンデンサ（ES12C1～ES12C8）データの統合処理
- EIS（電気化学インピーダンス分光）データの抽出
- 過渡応答データの統計的要約
- 実測データの品質検証

### 2. 実データ分析オーケストレーター (RealDataOrchestrator)

```python
from nasa_pcoe_eda.real_data_orchestrator import RealDataOrchestrator

orchestrator = RealDataOrchestrator(output_dir="output/real_analysis")
results = orchestrator.run_comprehensive_analysis(
    data_path="data/raw/ES12.mat",
    generate_visualizations=True,
    generate_report=True
)
```

**提供する分析:**
- 実データ特有のメトリクス抽出
- 劣化パターンの定量的分析
- 個別コンデンサの健全性評価
- 分析手法の妥当性検証
- 包括的レポート生成

### 3. 分析手法妥当性検証 (RealDataValidator)

```python
from nasa_pcoe_eda.analysis.real_data_validator import RealDataValidator

validator = RealDataValidator()
validation_results = validator.validate_analysis_methodology(
    real_data_results,
    sample_data_results,
    theoretical_values
)
```

**検証項目:**
- 統計分析手法の精度
- 相関分析の数学的妥当性
- 外れ値検出の適切性
- 時系列分析の一貫性
- 劣化パターン分析の物理的整合性

### 4. 実データ専用レポート生成 (RealDataReportGenerator)

```python
from nasa_pcoe_eda.reporting.real_data_generator import RealDataReportGenerator

generator = RealDataReportGenerator()
report_path = generator.generate_real_data_report(
    analysis_results,
    output_path,
    real_data_metrics,
    sample_data_comparison
)
```

**レポート内容:**
- 実データ概要と信頼性評価
- 劣化パターン詳細分析
- 個体差・劣化特性比較
- 分析手法妥当性検証結果
- 実データ vs 理論値比較
- 実用的な故障予測・保全指針

## 使用方法

### 基本的な使用例

```python
#!/usr/bin/env python3
import sys
from pathlib import Path

# プロジェクトのsrcディレクトリをパスに追加
sys.path.append(str(Path.cwd() / "src"))

from nasa_pcoe_eda.real_data_orchestrator import RealDataOrchestrator

def main():
    # データパスの設定
    data_path = Path("data/raw/ES12.mat")
    output_dir = Path("output/real_analysis")
    
    # オーケストレーターの初期化
    orchestrator = RealDataOrchestrator(output_dir=output_dir)
    
    # 包括的分析の実行
    results = orchestrator.run_comprehensive_analysis(
        data_path=data_path,
        generate_visualizations=True,
        generate_report=True
    )
    
    # 結果の確認
    summary = orchestrator.get_analysis_summary()
    print(f"データ品質スコア: {summary['key_findings']['data_quality_score']:.1%}")
    print(f"検出された劣化パターン数: {summary['key_findings']['degradation_patterns_detected']}")

if __name__ == "__main__":
    main()
```

### デモスクリプトの実行

```bash
# デモスクリプトの実行
python scripts/demo_real_data_analysis.py
```

## 出力ファイル

### 1. 包括的HTMLレポート
- **ファイル名**: `real_data_analysis_report.html`
- **内容**: 実データ分析の全結果を含む詳細レポート

### 2. 可視化ファイル
- **ディレクトリ**: `output/visualizations/`
- **内容**: 
  - 分布プロット
  - 相関ヒートマップ
  - 時系列劣化パターン
  - 個別コンデンサ比較

### 3. 分析ログ
- **ファイル名**: `real_data_analysis.log`
- **内容**: 分析プロセスの詳細ログ

## 実データ特有の分析項目

### 1. 劣化パターン分析

**分析内容:**
- 各特徴量の初期値・最終値・変化率
- 劣化速度の定量化（単位/サイクル）
- 線形・非線形劣化パターンの識別
- 故障予兆の早期検出

**出力例:**
```
voltage_ratio:
  初期値: 0.9850
  最終値: 0.8234
  変化率: -16.4%
  劣化速度: 0.0032 units/cycle
```

### 2. 個別コンデンサ分析

**分析内容:**
- 各コンデンサの劣化率計算
- 推定残存耐用寿命（RUL）
- 健全性ステータス評価
- 個体差の定量化

**出力例:**
```
ES12C1:
  劣化率: 12.3%
  推定RUL: 45 cycles
  健全性: 注意
```

### 3. データ信頼性評価

**評価項目:**
- データ完全性（欠損値率）
- 測定精度（変動係数）
- 信号対雑音比（S/N比）
- 物理的妥当性

### 4. 分析手法妥当性検証

**検証項目:**
- 統計分析の数学的正確性
- 相関分析の対称性・範囲妥当性
- 外れ値検出の適切性
- 時系列分析の一貫性

## 実用的な活用例

### 1. 予防保全計画の策定

```python
# 個別コンデンサの健全性評価
capacitor_analysis = results['real_data_metrics']['capacitor_analysis']

for capacitor, data in capacitor_analysis.items():
    if data['health_status'] == '警告':
        print(f"{capacitor}: 優先的な交換が必要")
    elif data['health_status'] == '注意':
        print(f"{capacitor}: 監視強化を推奨")
```

### 2. 故障予測モデルの構築

```python
# RUL予測に有効な特徴量の特定
rul_features = results['analysis_results'].rul_features

print("RUL予測推奨特徴量:")
for feature, score in rul_features[:5]:
    print(f"  {feature}: {score:.3f}")
```

### 3. 品質管理指標の設定

```python
# データ品質スコアに基づく判定
quality_score = results['data_quality_score']

if quality_score > 0.9:
    print("データ品質: 優秀 - 高精度分析が可能")
elif quality_score > 0.7:
    print("データ品質: 良好 - 通常分析に適用可能")
else:
    print("データ品質: 要改善 - データ収集条件の見直しが必要")
```

## トラブルシューティング

### よくある問題と解決方法

1. **ES12.matファイルが読み込めない**
   - ファイルがMATLAB v7.3 (HDF5) 形式であることを確認
   - ファイルパスが正しいことを確認
   - ファイルの破損がないことを確認

2. **メモリ不足エラー**
   - 大規模データの場合、チャンク処理を使用
   - 不要な可視化生成を無効化
   - システムメモリの増設を検討

3. **分析結果が期待と異なる**
   - データ品質スコアを確認
   - 検証結果を確認
   - ログファイルでエラー・警告を確認

### ログレベルの調整

```python
import logging

# デバッグ情報を表示
logging.getLogger('nasa_pcoe_eda').setLevel(logging.DEBUG)

# 警告のみ表示
logging.getLogger('nasa_pcoe_eda').setLevel(logging.WARNING)
```

## 今後の拡張予定

1. **深層学習モデルの統合**
   - LSTM/GRUによる時系列劣化予測
   - オートエンコーダーによる異常検知

2. **リアルタイム分析機能**
   - ストリーミングデータ対応
   - リアルタイム劣化監視

3. **多種類データセット対応**
   - 他のNASA PCOEデータセット
   - 異なる種類のコンデンサデータ

4. **クラウド連携機能**
   - AWS/Azure連携
   - 分散処理対応

## 参考資料

- [NASA PCOE Dataset Repository](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)
- [ES12データセット詳細](docs/es12_data_structure_guide.md)
- [基本的なEDA機能](README.md)