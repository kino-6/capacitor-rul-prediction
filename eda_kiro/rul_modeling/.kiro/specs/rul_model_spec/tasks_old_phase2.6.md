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

- [x] 3.1 DatasetSplitterクラスの実装
  - `src/data_preparation/dataset_splitter.py`を作成
  - ハイブリッド分割戦略の実装
  - Train: C1-C5のサイクル1-150（750サンプル）
  - Val: C6のサイクル1-150（150サンプル）
  - Test: C7-C8のサイクル1-200（400サンプル）
  - _Requirements: US-3_

- [x] 3.2 特徴量スケーリングの実装
  - StandardScalerの適用
  - Trainセットで学習、Val/Testに適用
  - スケーラーの保存: `output/models/scaler.pkl`
  - _Requirements: US-3_

- [x] 3.3 分割データの保存
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

- [x] 4.1 PrimaryModelクラスの実装
  - `src/models/primary_model.py`を作成
  - Random Forest Classifierをベースライン
  - 学習・予測・保存・読み込み機能
  - predict_proba機能（ROC-AUC計算用）
  - _Requirements: US-4_

- [x] 4.2 Primary Modelの学習
  - Trainセットで学習
  - Valセットで検証
  - ハイパーパラメータ: n_estimators=100, max_depth=10, random_state=42
  - モデル保存: `output/models/primary_model.pkl`
  - _Requirements: US-4_

- [x] 4.3 Primary Modelの評価
  - Testセットで評価
  - 評価指標: Accuracy, Precision, Recall, F1-Score, ROC-AUC
  - 目標: F1-Score ≥ 0.80
  - _Requirements: US-4, US-6_

- [x] 4.4 特徴量重要度の分析
  - 特徴量重要度の可視化
  - Phase 0の相関分析結果と比較
  - 出力: `output/evaluation/primary_feature_importance.png`
  - _Requirements: US-4, US-6_

### タスク5: Secondary Model（RUL予測）

- [x] 5.1 SecondaryModelクラスの実装
  - `src/models/secondary_model.py`を作成
  - Random Forest Regressorをベースライン
  - 学習・予測・保存・読み込み機能
  - _Requirements: US-5_

- [x] 5.2 Secondary Modelの学習
  - Trainセットで学習
  - Valセットで検証
  - ハイパーパラメータ: n_estimators=100, max_depth=15, random_state=42
  - モデル保存: `output/models/secondary_model.pkl`
  - _Requirements: US-5_

- [x] 5.3 Secondary Modelの評価
  - Testセットで評価
  - 評価指標: MAE, RMSE, R², MAPE
  - 目標: MAPE ≤ 20%
  - _Requirements: US-5, US-6_

- [x] 5.4 予測結果の可視化
  - 実測値 vs 予測値の散布図
  - 残差プロット
  - 予測誤差の分布
  - 出力: `output/evaluation/secondary_predictions.png`
  - _Requirements: US-5, US-6_

### タスク6: モデル評価とレポート生成

- [x] 6.1 ModelEvaluatorクラスの実装
  - `src/evaluation/evaluator.py`を作成
  - Primary/Secondary Modelの評価機能
  - 可視化機能（混同行列、ROC曲線、散布図など）
  - _Requirements: US-6_

- [x] 6.2 評価レポートの自動生成
  - Markdown形式のレポート生成
  - 画像を含む詳細レポート
  - 出力: `output/evaluation/baseline_report.md`
  - _Requirements: US-6_

### チェックポイント2: ベースラインモデル完了

- [x] CP2: ベースライン性能の確認
  - Primary Model: F1-Score ≥ 0.80を達成したか
  - Secondary Model: MAPE ≤ 20%を達成したか
  - 評価レポートを確認
  - ユーザーに確認を求める

---

## Phase 2.5: モデル検証とOverfitting診断

### タスク6.3: ES12 Test Dataでの詳細評価と可視化

- [x] 6.3 ES12 Test Dataでの予測結果詳細可視化
  - **Primary Model（異常検知）の詳細評価**：
    - Test Data（C7-C8の全サイクル）での予測結果を可視化
    - サイクルごとの予測確率（predict_proba）の推移をプロット
    - 誤分類サンプルの詳細分析（どのサイクル・どのコンデンサで誤分類が発生したか）
    - 混同行列の詳細版（コンデンサ別、サイクル範囲別）
  - **Secondary Model（RUL予測）の詳細評価**：
    - Test Data全サンプルでの「正解RUL vs 予測RUL」の散布図（サイクル番号で色分け）
    - コンデンサ別（C7, C8）の予測精度比較
    - サイクル範囲別（初期/中期/末期）の予測誤差分析
    - 予測誤差が大きいサンプルの特定と特徴量分析
  - **Overfitting診断**：
    - Train/Val/Testでの性能比較表を作成
    - 学習曲線（Training/Validation Loss）の可視化
    - Test性能がTrain性能と大きく乖離していないか確認
  - 出力: `output/evaluation/es12_test_detailed_analysis.png`, `output/evaluation/overfitting_diagnosis.md`
  - _Requirements: US-6, モデルの汎化性能検証_

### タスク6.4: ES10/ES14データの構造解析と整備

- [x] 6.4 ES10/ES14データセットの構造解析
  - **データ構造の詳細調査**：
    - ES10.mat, ES14.matの内部構造をh5pyで解析
    - ES12との構造の違いを特定（階層構造、データ形式、測定項目）
    - コンデンサ数、サイクル数、測定周波数の確認
    - 欠損データや異常値の有無を確認
  - **データ構造比較レポート作成**：
    - ES10/ES12/ES14の構造比較表を作成
    - 各データセットの特徴（ストレス条件、劣化速度など）を整理
    - ES12と同じ解釈が可能な共通特徴量を特定
  - 出力: `docs/es10_es14_structure_analysis.md`
  - _Requirements: US-1, クロスデータセット対応_

- [ ] 6.5 ES10/ES14データの統一フォーマット変換
  - **DataLoaderの拡張**：
    - `src/data_loading/multi_dataset_loader.py`を作成
    - ES10/ES14データをES12と同じフォーマットに変換する機能を実装
    - データセット間の差異を吸収する正規化処理を実装
    - 各データセットのメタデータ（ストレス条件、測定条件）を保持
  - **特徴量抽出の統一**：
    - ES10/ES14から同じ特徴量セット（30特徴量）を抽出
    - 並列処理機能を使用してES10/ES14の全コンデンサから特徴量を抽出
    - ES12と同じラベル生成戦略を適用
  - **統一データセットの作成**：
    - ES10/ES12/ES14を統合したデータセットを作成（オプション）
    - データセット識別子（dataset_id）を追加
    - 出力: `output/features/es10_features.csv`, `output/features/es14_features.csv`
  - _Requirements: US-1, US-2, クロスデータセット対応_

- [ ] 6.6 ES10/ES14での外部検証（External Validation）
  - **ES12モデルのES10/ES14への適用**：
    - ES12で学習したPrimary/Secondary Modelを読み込み
    - ES10/ES14データに対して予測を実行（ES12のスケーラーを使用）
    - 予測結果を可視化・評価
  - **汎化性能の評価**：
    - ES10/ES14での性能指標を計算（F1-Score, MAPE, R²など）
    - ES12 Test性能との比較
    - データセット間の性能差を分析（ドメインシフトの影響）
  - **ドメイン適応の必要性評価**：
    - ES10/ES14で性能が大きく低下する場合、原因を分析
    - 特徴量分布の違いを可視化（ES12 vs ES10/ES14）
    - ドメイン適応手法（Transfer Learning, Fine-tuningなど）の必要性を判断
  - 出力: `output/evaluation/external_validation_report.md`, `output/evaluation/cross_dataset_performance.png`
  - _Requirements: US-6, モデルの汎化性能検証_

### チェックポイント2.5: Overfitting診断と外部検証完了

- [ ] CP2.5: モデルの汎化性能確認
  - ES12 Testでの詳細分析が完了したか
  - Overfittingの兆候が見られるか（Train/Test性能差）
  - ES10/ES14データが正しく整備されたか
  - ES10/ES14での外部検証結果を確認
  - 次のアクション（Phase 3での改善 or ドメイン適応）を決定
  - ユーザーに確認を求める

---

## Phase 2.6: データリーケージ解消とモデル改善（Minimum Start）

**目的**: Phase 2の分析で発見されたデータリーケージを解消し、真の汎化性能を測定する

**背景**: 
- Primary Model: cycle_numberがラベルと直接相関（データリーケージ）
- Secondary Model: 特徴量重要度の90%がサイクル情報に依存
- 訓練データ: RUL 50-199のみ（RUL < 50が欠如）

**参照**: `docs/phase2_analysis_and_recommendations.md`

### タスク6.7: データリーケージ解消

- [x] 6.7 cycle関連特徴量の除外とデータ再構築
  - **特徴量の除外**：
    - cycle_number, cycle_normalizedを特徴量から除外
    - 残す特徴量: VL/VO関連（26特徴量 → 24特徴量）
  - **データ分割の変更**：
    - Train: C1-C5のサイクル1-200（全サイクル使用）
    - Val: C6のサイクル1-200（全サイクル使用）
    - Test: C7-C8のサイクル1-200（変更なし）
    - 新しいRUL範囲: 0-199（従来: 50-199）
  - **データセット再構築**：
    - 特徴量抽出（cycle関連除外）
    - ラベル生成（変更なし）
    - データ分割（全サイクル使用）
    - スケーリング（StandardScaler）
  - 出力: `output/features_v2/train.csv`, `val.csv`, `test.csv`
  - _Requirements: データリーケージ解消_

### タスク6.8: モデル再訓練（Baseline v2）

- [x] 6.8 データリーケージ解消後のモデル訓練
  - **Primary Model再訓練**：
    - 新しい訓練データで学習
    - cycle情報なしで劣化パターンを学習
    - 期待性能: F1-Score 0.75-0.85（真の性能）
  - **Secondary Model再訓練**：
    - 新しい訓練データで学習（RUL 0-199をカバー）
    - RUL < 50の予測精度改善を期待
    - 期待性能: MAE (RUL 0-50) < 5.0
  - モデル保存: `output/models_v2/primary_model.pkl`, `secondary_model.pkl`
  - _Requirements: 真の汎化性能測定_

### タスク6.9: 改善効果の評価

- [x] 6.9 Baseline v1 vs v2の比較分析
  - **性能比較**：
    - Primary Model: v1 (F1=1.0) vs v2 (F1=0.7-0.85)
    - Secondary Model: v1 (MAE 0-50=26.04) vs v2 (MAE 0-50=?)
    - Train/Val/Test性能差の確認（Overfitting診断）
  - **特徴量重要度の変化**：
    - v1: cycle情報90% → v2: VL/VO特徴量が主要に
    - 劣化指標の寄与度を確認
  - **詳細可視化**：
    - Test予測結果の可視化（v1 vs v2比較）
    - RUL範囲別の性能比較
    - コンデンサ別の性能比較
  - 出力: `output/evaluation_v2/comparison_report.md`, `comparison_visualizations.png`
  - _Requirements: 改善効果の定量評価_

### チェックポイント2.6: データリーケージ解消完了

- [ ] CP2.6: 改善効果の確認
  - データリーケージが解消されたか
  - Primary Modelが実際の劣化パターンを学習しているか
  - Secondary ModelのRUL < 50予測が改善したか
  - Train/Val/Test性能差が妥当な範囲か（< 10%）
  - 次のステップ（ES10/ES14追加 or 特徴量エンジニアリング）を決定
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

- **ES12データセットを中心に使用** - Phase 2まではES12に焦点
- **Phase 2.5でES10/ES14を追加** - 外部検証とクロスデータセット対応
- **8コンデンサ（ES12C1～ES12C8）** - 各約200サイクル
- **履歴特徴量なし** - Phase 1-2では処理時間を考慮して履歴特徴量を省略
- **Phase 3で履歴特徴量を追加** - 効果を検証
- **Overfitting対策** - Phase 2.5で詳細診断と外部検証を実施

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
- **Phase 2.5**: 
  - ES12 Testでの詳細分析完了
  - Overfitting診断完了（Train/Val/Test性能差 < 10%が理想）
  - ES10/ES14データ整備完了
  - ES10/ES14での外部検証完了（性能低下 < 20%が理想）
- **Phase 3**: ベースラインからの性能向上

### 推奨される実装順序

1. **Phase 1**: データセット構築（タスク1-3）
2. **Phase 2**: ベースラインモデル（タスク4-6）
3. **Phase 2.5**: Overfitting診断と外部検証（タスク6.3-6.6）
4. **Phase 2.6**: データリーケージ解消（タスク6.7-6.9） ← **現在ここ**
5. **Phase 3**: モデル改善（タスク7-10、オプション）

---

## 📝 進捗メモ

### 2026-01-17 更新（Phase 2.6完了）

- ✅ Task 1.1-1.3: 特徴量抽出と品質確認完了
- ✅ Task 2.1-2.2: ラベル生成完了
- ✅ Task 3.1-3.3: データ分割、特徴量スケーリング、サマリー生成完了
- ✅ Task 4.1-4.4: Primary Model（異常検知）完了 - F1-Score = 1.0000 🎯
- ✅ Task 5.1-5.4: Secondary Model（RUL予測）完了 - R² = 0.9330, MAPE = 89.78% ⚠️
- ✅ Task 6.1-6.2: モデル評価とレポート生成完了
- ✅ Task 6.3: ES12 Test詳細可視化完了
- ✅ Task 6.4: ES10/ES14構造解析完了
- ✅ CP2: ベースライン性能確認完了
- 📊 Phase 2結果分析:
  - **データリーケージ発見**: cycle_numberがラベルと直接相関
  - **Overfitting確認**: Train MAE=0.07 vs Test MAE=6.79（100倍差）
  - **訓練データ不足**: RUL < 50のデータが訓練セットに存在しない
- ✅ **Phase 2.6完了**: データリーケージ解消とモデル改善（Minimum Start）
  - ✅ Task 6.7: cycle関連特徴量の除外とデータ再構築完了
  - ✅ Task 6.8: モデル再訓練（Baseline v2）完了
  - ✅ Task 6.9: v1 vs v2比較分析完了
- 🎉 **Phase 2.6成果**:
  - **データリーケージ解消**: cycle特徴量を除外（30→28列、24特徴量）
  - **Primary Model v2**: F1=0.9975（真の性能、汎化性能良好）
  - **Secondary Model v2**: Test MAE=1.95（71%改善）、RUL 0-50 MAE=2.05（92%改善！）
  - **特徴量重要度**: VL特徴量が主要に（vl_q25, vl_mean, vl_median）
  - **完全RULカバレッジ**: RUL 0-199全範囲で予測可能

---

**作成日**: 2026-01-16
**Phase 0完了日**: 2026-01-16
**Phase 1完了日**: 2026-01-17
**Phase 2完了日**: 2026-01-17
**Phase 2.6完了日**: 2026-01-17
**最終更新日**: 2026-01-17
**次のタスク**: CP2.6 改善効果の確認（ユーザー確認待ち）
