# RUL Modeling Project Setup - 完了サマリー

## ✅ セットアップ完了

RUL予測モデル開発プロジェクトの初期構造を作成しました。

## 📁 作成されたファイル・ディレクトリ

```
rul_modeling/
├── README.md                           # プロジェクト概要
├── GETTING_STARTED.md                  # 開発開始ガイド
├── PROJECT_SETUP_SUMMARY.md           # このファイル
├── pyproject.toml                      # Python依存関係
├── .gitignore                          # Git除外設定
│
├── .kiro/                              # Kiro設定
│   └── specs/                          # Spec-driven development
│       └── rul_model_spec/             # RULモデル開発のSpec
│           ├── requirements.md         # ✅ 要件定義（7つのユーザーストーリー）
│           └── design.md               # ✅ 設計書（アーキテクチャ、モジュール設計）
│
├── src/                                # ソースコード（準備完了）
│   └── __init__.py
│
├── tests/                              # テスト（準備完了）
│
├── notebooks/                          # 実験用ノートブック（準備完了）
│
├── output/                             # モデル出力・結果（準備完了）
│   └── .gitkeep
│
└── docs/                               # ドキュメント
    └── rul_model_design.md             # ✅ 詳細設計書（EDAから移動）
```

## 📋 Spec Files の内容

### 1. requirements.md（要件定義）

**7つのユーザーストーリー**:
- US-1: 特徴量抽出
- US-2: ラベル生成
- US-3: データ分割
- US-4: Primary Model（異常検知）
- US-5: Secondary Model（RUL予測）
- US-6: モデル評価
- US-7: モデル改善

**各ストーリーに含まれる内容**:
- 明確な受け入れ基準（チェックリスト形式）
- 技術要件
- 性能要件
- 成功基準

### 2. design.md（設計書）

**含まれる内容**:
- システムアーキテクチャ図
- モジュール設計（4つの主要モジュール）
  - Feature Extraction Module
  - Data Preparation Module
  - Model Module（Primary & Secondary）
  - Evaluation Module
- データフロー図
- 特徴量詳細設計（26特徴量）
- モデルハイパーパラメータ
- ファイル構成
- 実装順序

## 🎯 2段階モデルアーキテクチャ

```
Input Features → [Primary: Anomaly Classifier] → [Secondary: RUL Predictor] → RUL Output
                         ↓
                   Normal / Abnormal
```

**Primary Model**:
- 目的: 異常検知（Normal/Abnormal分類）
- アルゴリズム: Random Forest Classifier（ベースライン）
- 評価指標: Accuracy, Precision, Recall, F1-Score, ROC-AUC

**Secondary Model**:
- 目的: RUL予測（残りサイクル数）
- アルゴリズム: Random Forest Regressor（ベースライン）
- 評価指標: MAE, RMSE, R², MAPE

## 📊 データ戦略

### 特徴量（26特徴量/サイクル）

1. **基本統計量**（14特徴量）: VL/VOの平均、標準偏差、範囲など
2. **劣化指標**（4特徴量）: 電圧比、応答効率、信号減衰
3. **時系列特徴**（2特徴量）: トレンド
4. **サイクル情報**（2特徴量）: サイクル番号、正規化サイクル番号
5. **履歴特徴**（4特徴量）: 過去Nサイクルの統計、劣化率

### データ分割（ハイブリッド戦略）

```
Train: C1-C5 の サイクル 1-150  (750サンプル)
Val:   C6 の サイクル 1-150     (150サンプル)
Test:  C7-C8 の サイクル 1-200  (400サンプル)
```

## 🚀 次のステップ

### Phase 1: データ準備（Week 1）

**Task 1: 特徴量抽出スクリプトの作成**
```bash
# 実装ファイル
src/feature_extraction/extractor.py

# 期待される出力
output/features/features.csv
```

**Task 2: ラベル生成**
```bash
# 実装ファイル
src/data_preparation/label_generator.py

# 期待される出力
output/features/features_with_labels.csv
```

**Task 3: データセット構築**
```bash
# 実装ファイル
src/data_preparation/dataset_builder.py

# 期待される出力
output/features/train.csv
output/features/val.csv
output/features/test.csv
```

### Phase 2: ベースラインモデル（Week 2）

**Task 4: Primary Model**
```bash
# 実装ファイル
src/models/primary_model.py

# 期待される出力
output/models/primary_model.pkl
output/evaluation/primary_model_report.md
```

**Task 5: Secondary Model**
```bash
# 実装ファイル
src/models/secondary_model.py

# 期待される出力
output/models/secondary_model.pkl
output/evaluation/secondary_model_report.md
```

### Phase 3: モデル改善（Week 3-4）

- ハイパーパラメータチューニング
- 複数アルゴリズムの比較（XGBoost, LightGBM, LSTM）
- 特徴量エンジニアリング
- アンサンブル手法

## 📖 ドキュメント参照順序

開発を始める際は、以下の順序でドキュメントを参照してください：

1. **[GETTING_STARTED.md](GETTING_STARTED.md)** - 開発開始ガイド
2. **[.kiro/specs/rul_model_spec/requirements.md](.kiro/specs/rul_model_spec/requirements.md)** - 要件定義
3. **[.kiro/specs/rul_model_spec/design.md](.kiro/specs/rul_model_spec/design.md)** - 設計書
4. **[docs/rul_model_design.md](docs/rul_model_design.md)** - 詳細設計・実装例

## 🔗 関連リソース

- **EDAプロジェクト**: `../` - ES12データセットの探索的データ分析
- **劣化分析結果**: `../output/large_gap_similar_vl_dissimilar_vo/`
- **GitHub**: https://github.com/kino-6/capacitor-rul-prediction

## ✨ Spec-driven Development の利点

1. **明確な要件**: ユーザーストーリーと受け入れ基準で何を作るかが明確
2. **体系的な設計**: アーキテクチャとモジュール設計が事前に定義済み
3. **段階的な実装**: Phase 1 → Phase 2 → Phase 3 の順に進められる
4. **品質保証**: 各ストーリーの受け入れ基準でチェック可能

---

**作成日**: 2026-01-15

準備完了！特徴量抽出スクリプトの作成から始めましょう 🚀
