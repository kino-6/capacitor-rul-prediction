# Next Steps - 次にやること

## 🎯 現在の状況

RUL予測モデル開発プロジェクトの初期構造が完成しました。

## 📋 今すぐできること

### Option 1: 特徴量抽出スクリプトの作成（推奨）

**最初のタスク**: ES12データから特徴量を抽出するスクリプトを作成

```bash
# 作成するファイル
src/feature_extraction/__init__.py
src/feature_extraction/extractor.py

# 実装内容
- CycleFeatureExtractor クラス
- 基本統計量の抽出（VL/VO）
- 劣化指標の計算（電圧比など）
- 履歴特徴量の生成
```

**参考ドキュメント**:
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

### Option 2: Specファイルのレビュー

作成されたSpecファイルを確認して、要件や設計を理解する：

```bash
# 要件定義を確認
cat .kiro/specs/rul_model_spec/requirements.md

# 設計書を確認
cat .kiro/specs/rul_model_spec/design.md

# 詳細設計を確認
cat docs/rul_model_design.md
```

### Option 3: 実験用ノートブックの作成

Jupyter Notebookで探索的に実装を試す：

```bash
# ノートブックを作成
notebooks/01_feature_extraction_exploration.ipynb

# 内容
- ES12データの読み込み
- 1サイクルからの特徴量抽出を試す
- 可視化して確認
```

## 🚀 推奨される開発フロー

### Step 1: 特徴量抽出（今週）

1. **実装**:
   ```bash
   # ファイル作成
   src/feature_extraction/__init__.py
   src/feature_extraction/extractor.py
   ```

2. **テスト**:
   ```bash
   # テスト作成
   tests/test_feature_extraction.py
   
   # テスト実行
   uv run pytest tests/test_feature_extraction.py
   ```

3. **実行**:
   ```bash
   # スクリプト作成
   scripts/extract_features.py
   
   # 実行
   uv run python scripts/extract_features.py
   ```

4. **確認**:
   ```bash
   # 出力確認
   head output/features/features.csv
   wc -l output/features/features.csv  # 約1600行（8コンデンサ × 200サイクル）
   ```

### Step 2: ラベル生成（今週）

1. **実装**:
   ```bash
   src/data_preparation/__init__.py
   src/data_preparation/label_generator.py
   ```

2. **実行**:
   ```bash
   scripts/generate_labels.py
   ```

3. **確認**:
   ```bash
   head output/features/features_with_labels.csv
   ```

### Step 3: データ分割（今週）

1. **実装**:
   ```bash
   src/data_preparation/dataset_builder.py
   ```

2. **実行**:
   ```bash
   scripts/build_dataset.py
   ```

3. **確認**:
   ```bash
   wc -l output/features/train.csv  # 750行
   wc -l output/features/val.csv    # 150行
   wc -l output/features/test.csv   # 400行
   ```

### Step 4: ベースラインモデル（来週）

1. **Primary Model**:
   ```bash
   src/models/primary_model.py
   scripts/train_primary_model.py
   ```

2. **Secondary Model**:
   ```bash
   src/models/secondary_model.py
   scripts/train_secondary_model.py
   ```

3. **評価**:
   ```bash
   src/evaluation/evaluator.py
   scripts/evaluate_models.py
   ```

## 📖 重要なドキュメント

開発を始める前に、以下のドキュメントを読むことを推奨します：

1. **[GETTING_STARTED.md](GETTING_STARTED.md)** - 開発開始ガイド（5分）
2. **[requirements.md](.kiro/specs/rul_model_spec/requirements.md)** - 要件定義（10分）
3. **[design.md](.kiro/specs/rul_model_spec/design.md)** - 設計書（15分）

## 💡 開発のヒント

### 1. Spec-driven Development

- 各タスクの受け入れ基準を確認しながら実装
- チェックリストを埋めていく感覚で進める

### 2. EDA結果の活用

- `../output/large_gap_similar_vl_dissimilar_vo/` の分析結果を参考に
- 劣化パターンの理解が特徴量設計に役立つ

### 3. 段階的な実装

- 一度に全部作らない
- 小さく作って、テストして、確認する

### 4. テスト駆動

- 実装前にテストを書く（TDD）
- または実装後すぐにテストを書く

## 🎯 成功基準（Phase 1）

Phase 1（データ準備）の完了条件：

- [ ] 全コンデンサから特徴量を抽出完了
- [ ] ラベル生成完了
- [ ] Train/Val/Test分割完了
- [ ] データセットの品質確認完了

## 🔗 関連リソース

- **EDAプロジェクト**: `../` - ES12データセットの探索的データ分析
- **GitHub**: https://github.com/kino-6/capacitor-rul-prediction
- **NASA PCOE Dataset**: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/

## ❓ 質問がある場合

- Specファイル（requirements.md, design.md）を確認
- 詳細設計書（rul_model_design.md）を確認
- EDAの分析結果を確認

---

**作成日**: 2026-01-15

準備完了！特徴量抽出スクリプトの作成から始めましょう 🚀

**最初のコマンド**:
```bash
cd rul_modeling
mkdir -p src/feature_extraction
touch src/feature_extraction/__init__.py
touch src/feature_extraction/extractor.py
```
