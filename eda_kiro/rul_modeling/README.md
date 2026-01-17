# RUL Prediction Model Development

コンデンサの残存耐用寿命（RUL: Remaining Useful Life）予測モデルの開発プロジェクト

## 🎯 プロジェクト概要

ES12データセット（およびES10, ES14）を用いて、コンデンサの劣化状態を検知し、残存耐用寿命を予測する機械学習モデルを構築します。

## 🏗️ モデルアーキテクチャ

### 2段階アプローチ

```
Input Features → [Primary: Anomaly Classifier] → [Secondary: RUL Predictor] → RUL Output
                         ↓
                   Normal / Abnormal
```

**Primary Model (異常検知)**:
- 目的: コンデンサの状態が正常か異常かを分類
- Output: Binary (0: Normal, 1: Abnormal)

**Secondary Model (RUL予測)**:
- 目的: 残存耐用寿命（RUL）を予測
- Input: Primary Modelで異常と判定されたデータ
- Output: Continuous (残りサイクル数)

## 📊 データソース

- **EDA結果**: `../output/large_gap_similar_vl_dissimilar_vo/` - 劣化パターンの分析結果
- **生データ**: `../data/raw/ES12.mat` - 8個のコンデンサ、各約200サイクル
- **特徴量**: サイクルレベルの統計量（VL, VO, 電圧比など）

## 🗂️ プロジェクト構造

```
rul_modeling/
├── README.md                    # このファイル
├── .kiro/
│   └── specs/                   # Spec-driven development
│       └── rul_model_spec/      # RULモデル開発のSpec
├── src/                         # ソースコード
│   ├── feature_extraction/      # 特徴量抽出
│   ├── models/                  # モデル定義
│   └── evaluation/              # 評価スクリプト
├── tests/                       # テスト
├── notebooks/                   # 実験用ノートブック
├── output/                      # モデル出力・結果
└── docs/                        # ドキュメント
```

## 🚀 開発フロー

### Phase 1: データ準備
1. 特徴量抽出スクリプトの作成
2. ラベル生成（Normal/Abnormal, RUL）
3. Train/Val/Test分割

### Phase 2: ベースラインモデル
1. Primary Model（Random Forest Classifier）
2. Secondary Model（Random Forest Regressor）
3. 性能評価

### Phase 3: モデル改善
1. ハイパーパラメータチューニング
2. 特徴量エンジニアリング
3. アンサンブル手法

## 📖 ドキュメント

- [RULモデル設計書](docs/rul_model_design.md) - 詳細な設計ドキュメント
- [特徴量設計](docs/feature_engineering.md) - 特徴量の定義と抽出方法
- [評価指標](docs/evaluation_metrics.md) - モデル評価の基準

## 🔗 関連プロジェクト

- **EDA**: `../` - ES12データセットの探索的データ分析
- **GitHub**: https://github.com/kino-6/capacitor-rul-prediction

---

**作成日**: 2026-01-15
