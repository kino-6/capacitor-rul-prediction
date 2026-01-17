# RUL予測モデル開発 - 実装タスク

## 📋 概要

Phase 0の探索的特徴量分析の結果を踏まえ、RUL予測モデルの実装を進める。

**Phase 0の成果**:

- ✅ 高相関特徴量: 10個特定（目標: 5個以上）
- ✅ VO（出力電圧）関連の特徴量が最も有効
- ✅ データセット構築の準備完了
- ✅ CycleFeatureExtractor実装完了
- ✅ データローダー実装完了

---

## Phase 1: データセット構築

### タスク1: 全コンデンサから特徴量を抽出

- [x] 1.1 並列処理機能の実装
  - `src/data_preparation/parallel_extractor.py`を作成
  - multiprocessingを使用して複数コンデンサを並列処理
  - 進捗表示機能の実装（20サイクルごと + パーセンテージ）
  - ES12の8コンデンサを並列処理（M4 Pro 14コア活用）
  - _Requirements: US-1_

- [x] 1.2 ES12データセットから特徴量を抽出
  - ES12C1～ES12C8の全200サイクルから特徴量を抽出
  - 並列処理機能を使用して高速化
  - 履歴特徴量なしで抽出（処理時間を考慮）
  - 出力: `output/features/es12_features.csv`（1600行 × 30列）
  - _Requirements: US-1_

- [x] 1.3 特徴量の品質確認
  - 欠損値のチェック
  - 外れ値の検出
  - 統計サマリーの生成
  - 出力: `output/features/es12_quality_report.txt`
  - _Requirements: US-1_

### タスク2: ラベル生成

- [x] 2.1 LabelGeneratorクラスの実装
  - `src/data_preparation/label_generator.py`を作成
  - Cycle-based Strategy: 前半50%をNormal、後半50%をAbnormal
  - RUL計算: 200 - cycle_number
  - _Requirements: US-2_

- [x] 2.2 ラベルの追加と保存
  - 特徴量データにラベル（is_abnormal, rul）を追加
  - 出力: `output/features/es12_features_with_labels.csv`
  - _Requirements: US-2_

### タスク3: データ分割

- [ ] 3.1 DatasetSplitterクラスの実装
  - `src/data_preparation/dataset_splitter.py`を作成
  - ハイブリッド分割戦略の実装
  - Train: C1-C5のサイクル1-150（750サンプル）
  - Val: C6のサイクル1-150（150サンプル）
  - Test: C7-C8のサイクル1-200（400サンプル）
  - _Requirements: US-3_

- [ ] 3.2 特徴量スケーリングの実装
  - StandardScalerの適用
  - Trainセットで学習、Val/Testに適用
  - スケーラーの保存: `output/models/scaler.pkl`
  - _Requirements: US-3_

- [ ] 3.3 分割データの保存
  - 出力: `output/features/train.csv`, `val.csv`, `test.csv`
  - データセットサマリーの生成: `output/features/dataset_summary.txt`
  - _Requirements: US-3_

### チェックポイント1: データセット構築完了

- [ ] CP1: データセットの品質確認
  - 全ファイルが正しく生成されているか確認
  - サンプル数（Train: 750, Val: 150, Test: 400）
  - 特徴量の分布を確認
  - ユーザーに確認を求める

---

## Phase 2: ベースラインモデル構築

### タスク4: Primary Model（異常検知）

- [ ] 4.1 PrimaryModelクラスの実装
  - `src/models/primary_model.py`を作成
  - Random Forest Classifierをベースライン
  - 学習・予測・保存・読み込み機能
  - predict_proba機能（ROC-AUC計算用）
  - _Requirements: US-4_

- [ ] 4.2 Primary Modelの学習
  - Trainセットで学習
  - Valセットで検証
  - ハイパーパラメータ: n_estimators=100, max_depth=10, random_state=42
  - モデル保存: `output/models/primary_model.pkl`
  - _Requirements: US-4_

- [ ] 4.3 Primary Modelの評価
  - Testセットで評価
  - 評価指標: Accuracy, Precision, Recall, F1-Score, ROC-AUC
  - 目標: F1-Score ≥ 0.80
  - _Requirements: US-4, US-6_

- [ ] 4.4 特徴量重要度の分析
  - 特徴量重要度の可視化
  - Phase 0の相関分析結果と比較
  - 出力: `output/evaluation/primary_feature_importance.png`
  - _Requirements: US-4, US-6_

### タスク5: Secondary Model（RUL予測）

- [ ] 5.1 SecondaryModelクラスの実装
  - `src/models/secondary_model.py`を作成
  - Random Forest Regressorをベースライン
  - 学習・予測・保存・読み込み機能
  - _Requirements: US-5_

- [ ] 5.2 Secondary Modelの学習
  - Trainセットで学習
  - Valセットで検証
  - ハイパーパラメータ: n_estimators=100, max_depth=15, random_state=42
  - モデル保存: `output/models/secondary_model.pkl`
  - _Requirements: US-5_

- [ ] 5.3 Secondary Modelの評価
  - Testセットで評価
  - 評価指標: MAE, RMSE, R², MAPE
  - 目標: MAPE ≤ 20%
  - _Requirements: US-5, US-6_

- [ ] 5.4 予測結果の可視化
  - 実測値 vs 予測値の散布図
  - 残差プロット
  - 予測誤差の分布
  - 出力: `output/evaluation/secondary_predictions.png`
  - _Requirements: US-5, US-6_

### タスク6: モデル評価とレポート生成

- [ ] 6.1 ModelEvaluatorクラスの実装
  - `src/evaluation/evaluator.py`を作成
  - Primary/Secondary Modelの評価機能
  - 可視化機能（混同行列、ROC曲線、散布図など）
  - _Requirements: US-6_

- [ ] 6.2 評価レポートの自動生成
  - Markdown形式のレポート生成
  - 画像を含む詳細レポート
  - 出力: `output/evaluation/baseline_report.md`
  - _Requirements: US-6_

### チェックポイント2: ベースラインモデル完了

- [ ] CP2: ベースライン性能の確認
  - Primary Model: F1-Score ≥ 0.80を達成したか
  - Secondary Model: MAPE ≤ 20%を達成したか
  - 評価レポートを確認
  - ユーザーに確認を求める

---

## Phase 3: モデル改善（オプション）

### タスク7: ハイパーパラメータチューニング

- [ ] 7.1 Grid Searchの実装
  - `src/models/tuner.py`を作成
  - Primary Modelのチューニング
  - Secondary Modelのチューニング
  - 最適パラメータの保存
  - _Requirements: US-7_

- [ ] 7.2 チューニング結果の評価
  - ベースラインとの比較
  - 性能向上の確認
  - 出力: `output/evaluation/tuning_report.md`
  - _Requirements: US-7_

### タスク8: 複数アルゴリズムの比較（オプション）

- [ ]* 8.1 XGBoost/LightGBMの実装
  - Primary Model用
  - Secondary Model用
  - _Requirements: US-7_

- [ ]* 8.2 アルゴリズム比較
  - Random Forest vs XGBoost vs LightGBM
  - 性能・処理時間の比較
  - 最適アルゴリズムの選定
  - _Requirements: US-7_

### タスク9: 特徴量エンジニアリング（オプション）

- [ ]* 9.1 履歴特徴量の追加
  - voltage_ratio_mean_last_5
  - voltage_ratio_std_last_5
  - voltage_ratio_trend_last_10
  - degradation_rate
  - _Requirements: US-7_

- [ ]* 9.2 新規特徴量の効果検証
  - 履歴特徴量ありなしでの性能比較
  - 特徴量重要度の再分析
  - _Requirements: US-7_

### タスク10: 最終評価とレポート（オプション）

- [ ]* 10.1 最終モデルの選定
  - 全実験結果の比較
  - 最適モデルの選定
  - _Requirements: US-7_

- [ ]* 10.2 最終評価レポートの作成
  - 全実験結果のまとめ
  - 最終モデルの性能評価
  - 出力: `output/evaluation/final_report.md`
  - _Requirements: US-7_

---

## 📝 注意事項

### 実装の焦点

- **ES12データセットのみを使用** - 要件と設計はES12に焦点を当てている
- **8コンデンサ（ES12C1～ES12C8）** - 各約200サイクル
- **履歴特徴量なし** - Phase 1では処理時間を考慮して履歴特徴量を省略
- **Phase 3で履歴特徴量を追加** - 効果を検証

### 並列処理について

- **M4 Pro（14コア）を活用** - 8コンデンサを並列処理
- 進捗表示を頻繁に行う（20サイクルごと + パーセンテージ）
- 期待処理時間: 約3-4分（並列処理）

### Phase 0の結果の活用

- 高相関特徴量（vo_cv, vo_mean, vo_max, vl_cvなど）を優先
- 低相関特徴量（vo_trend, vl_rangeなど）は削除を検討
- voltage_ratioは相関は低いが、非線形モデルで効果を期待

### 成功基準

- **Phase 1**: データセット構築完了（1600サンプル）
- **Phase 2**:
  - Primary Model: F1-Score ≥ 0.80
  - Secondary Model: MAPE ≤ 20%
- **Phase 3**: ベースラインからの性能向上

### 推奨される実装順序

1. **Phase 1**: データセット構築（タスク1-3）
2. **Phase 2**: ベースラインモデル（タスク4-6）
3. **Phase 3**: モデル改善（タスク7-10、オプション）

---

**作成日**: 2026-01-16
**Phase 0完了日**: 2026-01-16
**最終更新日**: 2026-01-16
**次のタスク**: 1.1 並列処理機能の実装
