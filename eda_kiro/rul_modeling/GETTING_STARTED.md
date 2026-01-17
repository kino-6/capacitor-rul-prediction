# Getting Started - RUL予測モデル開発

## 📋 概要

このドキュメントは、RUL予測モデル開発プロジェクトを開始するためのガイドです。

## 🎯 プロジェクトの目的

ES12データセット（およびES10, ES14）を用いて、コンデンサの劣化状態を検知し、残存耐用寿命（RUL）を予測する機械学習モデルを構築します。

## 🏗️ プロジェクト構造

```
rul_modeling/
├── README.md                    # プロジェクト概要
├── GETTING_STARTED.md          # このファイル
├── pyproject.toml              # Python依存関係
├── .gitignore                  # Git除外設定
│
├── .kiro/                      # Kiro設定
│   └── specs/                  # Spec-driven development
│       └── rul_model_spec/     # RULモデル開発のSpec
│           ├── requirements.md # 要件定義
│           └── design.md       # 設計書
│
├── src/                        # ソースコード
│   └── __init__.py
│
├── tests/                      # テスト
│
├── notebooks/                  # 実験用ノートブック
│
├── output/                     # モデル出力・結果
│   └── .gitkeep
│
└── docs/                       # ドキュメント
    └── rul_model_design.md     # 詳細設計書
```

## 🚀 セットアップ

### 1. 依存関係のインストール

```bash
cd rul_modeling
uv sync
```

### 2. データの確認

ES12データは親ディレクトリの `../data/raw/ES12.mat` にあります。

```bash
ls -lh ../data/raw/ES12.mat
```

### 3. EDA結果の確認

劣化パターンの分析結果を確認：

```bash
cat ../output/large_gap_similar_vl_dissimilar_vo/SUMMARY.md
```

## 📖 開発フロー

### Phase 1: データ準備

1. **特徴量抽出スクリプトの作成**
   - `src/feature_extraction/extractor.py`
   - ES12データから各サイクルの特徴量を抽出

2. **ラベル生成**
   - `src/data_preparation/label_generator.py`
   - Primary Model用の異常ラベル
   - Secondary Model用のRUL

3. **データセット構築**
   - `src/data_preparation/dataset_builder.py`
   - Train/Val/Test分割

### Phase 2: ベースラインモデル

1. **Primary Model（異常検知）**
   - `src/models/primary_model.py`
   - Random Forest Classifier

2. **Secondary Model（RUL予測）**
   - `src/models/secondary_model.py`
   - Random Forest Regressor

3. **評価**
   - `src/evaluation/evaluator.py`
   - 性能評価とレポート生成

### Phase 3: モデル改善

1. ハイパーパラメータチューニング
2. 複数アルゴリズムの比較（XGBoost, LightGBM, LSTM）
3. 特徴量エンジニアリング
4. アンサンブル手法

## 📚 重要なドキュメント

### Spec Files（開発ガイド）

- **[requirements.md](.kiro/specs/rul_model_spec/requirements.md)** - 要件定義
  - ユーザーストーリー
  - 受け入れ基準
  - 成功基準

- **[design.md](.kiro/specs/rul_model_spec/design.md)** - 設計書
  - システムアーキテクチャ
  - モジュール設計
  - データフロー

### Technical Docs

- **[rul_model_design.md](docs/rul_model_design.md)** - 詳細設計
  - モデル構成
  - 特徴量設計
  - 実装例

## 🎯 最初のタスク

### Task 1: 特徴量抽出スクリプトの作成

**目的**: ES12データから各サイクルの特徴量を抽出する

**実装内容**:
1. `src/feature_extraction/extractor.py` を作成
2. `CycleFeatureExtractor` クラスを実装
3. 基本統計量、劣化指標、時系列特徴、履歴特徴を抽出
4. 全コンデンサから特徴量を抽出し、CSV保存

**参考**:
- [design.md](.kiro/specs/rul_model_spec/design.md) - モジュール設計
- [rul_model_design.md](docs/rul_model_design.md) - 実装例

**期待される出力**:
```
output/features/features.csv
┌──────────┬───────┬──────────┬──────────┬─────────────────┬─────┐
│ cap_id   │ cycle │ vl_mean  │ vo_mean  │ voltage_ratio   │ ... │
├──────────┼───────┼──────────┼──────────┼─────────────────┼─────┤
│ ES12C1   │ 1     │ 5.234    │ 4.123    │ 0.787           │ ... │
│ ES12C1   │ 2     │ 5.241    │ 4.098    │ 0.782           │ ... │
│ ...      │ ...   │ ...      │ ...      │ ...             │ ... │
└──────────┴───────┴──────────┴──────────┴─────────────────┴─────┘
```

## 🧪 テスト

```bash
# 全テストの実行
uv run pytest

# カバレッジ付き
uv run pytest --cov=src --cov-report=html
```

## 📊 評価指標

### Primary Model（異常検知）
- Accuracy: 全体的な精度
- Precision / Recall: 異常検知の精度
- F1-Score: バランス指標
- ROC-AUC: 閾値に依存しない評価

### Secondary Model（RUL予測）
- MAE: 平均絶対誤差
- RMSE: 二乗平均平方根誤差
- R²: 決定係数
- MAPE: 平均絶対パーセント誤差

## 🎯 成功基準

### Phase 1: データ準備
- [ ] 全コンデンサから特徴量を抽出完了
- [ ] ラベル生成完了
- [ ] Train/Val/Test分割完了

### Phase 2: ベースラインモデル
- [ ] Primary Model: F1-Score ≥ 0.80
- [ ] Secondary Model: MAPE ≤ 20%
- [ ] 評価レポート作成完了

### Phase 3: モデル改善
- [ ] ハイパーパラメータチューニング完了
- [ ] 複数アルゴリズムの比較完了
- [ ] 最終モデルの選定完了

## 🔗 関連リソース

- **EDAプロジェクト**: `../` - ES12データセットの探索的データ分析
- **GitHub**: https://github.com/kino-6/capacitor-rul-prediction
- **NASA PCOE Dataset**: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

## 💡 Tips

1. **Spec-driven development**: `.kiro/specs/` のドキュメントを参照しながら開発
2. **EDA結果の活用**: `../output/large_gap_similar_vl_dissimilar_vo/` の分析結果を参考に
3. **段階的な実装**: Phase 1 → Phase 2 → Phase 3 の順に進める
4. **テスト駆動**: 各モジュールのテストを書きながら実装

---

**作成日**: 2026-01-15

準備完了！特徴量抽出スクリプトの作成から始めましょう 🚀
