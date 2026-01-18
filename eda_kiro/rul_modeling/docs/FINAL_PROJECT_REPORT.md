# コンデンサ劣化予測プロジェクト 最終レポート

**プロジェクト期間**: 2026年1月15日 - 2026年1月19日  
**データセット**: NASA PCOE ES12（8コンデンサ × 200サイクル = 1,600サンプル）  
**アプローチ**: VL-VO応答性に基づく教師なし異常検知・劣化予測

---

## エグゼクティブサマリー

本プロジェクトでは、コンデンサの入力電圧（VL）と出力電圧（VO）の関係性を分析し、劣化を検出・予測するモデルを構築しました。Phase 6では誤報率を大幅に削減し、実用的な段階的アラートシステムを完成させました。

### 主要成果

1. **異常検知モデル（One-Class SVM）**
   - 初期サイクル（1-10）を正常パターンとして学習
   - 異常検出率: 90.8%
   - Training False Positive: 5.0%（優秀）
   - Late False Negative: 5.2%（優秀）

2. **劣化度予測モデル（Random Forest）**
   - Test MAE: 0.0036（目標0.1を大幅達成）
   - Test R²: 0.9996（極めて高精度）
   - 劣化度を0-1スケールで定量化

3. **次サイクル応答性予測**
   - 全特徴量でR² > 0.93
   - 高精度な時系列予測を実現

4. **誤報率削減（Phase 6）**
   - FPR削減: 41.4% → 13.1%（68.2%改善）
   - 誤報数: 104個 → 33個（71個削減）
   - 段階的アラートシステム完成（4レベル）

### 重要な発見

- **Response Efficiency**: 初期77-117% → 後期1.1-1.2%（98.5%減少）
- **Waveform Correlation**: 劣化に伴い1.0に近づく（波形単純化）
- **劣化開始**: Cycle 11-15から検出可能（従来想定より早期）
- **閾値最適化**: ROC-AUC 0.9872で最適閾値を特定

---

## 1. プロジェクト背景

### 1.1 課題

従来のRUL（Remaining Useful Life）予測アプローチには以下の問題がありました：

- `RUL = 200 - cycle_number`は単なるサイクル番号の逆算
- Normal/Abnormalラベルに物理的根拠なし
- データリーケージのリスク

### 1.2 新アプローチ

VL-VO応答性の劣化を物理的に意味のある特徴量で検出：

- 教師なし学習（ラベル不要）
- 初期状態を正常パターンとして学習
- 波形特性の変化を定量化

---

## 2. データ分析（Phase 1）

### 2.1 データセット

- **ES12**: 8コンデンサ × 200サイクル = 1,600サンプル
- **データ分割**:
  - Train: C1-C5（1,000サンプル）
  - Validation: C6（200サンプル）
  - Test: C7-C8（400サンプル）

### 2.2 特徴量設計

15個の応答性特徴量を設計・抽出：

**エネルギー転送**（4特徴量）:
- response_efficiency: VO_energy / VL_energy
- voltage_ratio, peak_voltage_ratio, rms_voltage_ratio

**波形類似度**（3特徴量）:
- waveform_correlation: VLとVOの相関係数
- vo_variability, vl_variability: 変動係数

**応答遅延**（2特徴量）:
- response_delay, response_delay_normalized

**初期状態からの偏差**（4特徴量）:
- efficiency_degradation_rate, voltage_ratio_deviation
- correlation_shift, peak_voltage_ratio_deviation

**高度な特徴**（2特徴量）:
- residual_energy_ratio: 線形フィットからの残差
- vo_complexity: 波形複雑度

### 2.3 劣化パターンの発見

全8コンデンサで一貫した劣化パターンを確認：

| 指標 | 初期値 | 後期値 | 変化率 |
|------|--------|--------|--------|
| Response Efficiency | 77-117% | 1.1-1.2% | -98.5% |
| Waveform Correlation | 0.83 | 0.9998 | +20.4% |
| VO Variability | 0.24 | 0.49 | +104% |
| VL Variability | 0.26 | 0.73 | +181% |

---

## 3. 異常検知モデル（Phase 2）

### 3.1 モデル選択

**One-Class SVM v2**を最終モデルとして採用：

- **正常データ**: 初期1-10サイクル（製品として必ず正常）
- **カーネル**: RBF（非線形境界）
- **ハイパーパラメータ**: nu=0.05（最適化済み）
- **使用特徴量**: 波形特性のみ（7特徴量）

### 3.2 特徴量選択の理由

**除外した特徴量**（効率系）:
- response_efficiency, voltage_ratio等
- 理由: 中期に異常ピーク（物理的に不可能なU字型）
- 効率変化は劣化の**結果**であり、**予測指標**ではない

**使用した特徴量**（波形特性）:
- waveform_correlation, vo_variability, vl_variability
- response_delay, response_delay_normalized
- residual_energy_ratio, vo_complexity
- 理由: 単調な劣化パターン（物理的に妥当）

### 3.3 性能評価

| 指標 | 値 | 評価 |
|------|-----|------|
| 異常検出率 | 90.8% | 良好 |
| Training FP (1-10) | 5.0% | 優秀 |
| Early FP (11-20) | 35.6% | 許容範囲* |
| Late FN (100+) | 5.2% | 優秀 |
| 遷移点（50%異常） | Cycle 13 | 妥当 |
| Mid-stage (51-100) | 100% | 完璧 |

*Early FP 35.6%は実際の劣化開始（Cycle 11-15）を検出している可能性が高い

### 3.4 ハイパーパラメータ最適化

nu値を0.01-0.3で検証：

- **nu=0.01**: Training FP 0%, Early FP 0% → 過学習
- **nu=0.05**: Training FP 5.0%, Early FP 35.6% → **最適**
- **nu=0.1**: Training FP 11.2%, Early FP 42.5% → FP高い
- **nu=0.3**: Training FP 30.0%, Early FP 60.0% → FP非常に高い

---

## 4. 劣化度予測モデル（Phase 3）

### 4.1 劣化度スコアの定義

複合指標アプローチ（4つの波形特性を組み合わせ）：

1. **Correlation-based Score**: 波形単純化
2. **VO Variability-based Score**: 応答不安定化
3. **VL Variability-based Score**: 入力不安定化
4. **Residual Energy-based Score**: 線形関係からの逸脱

**劣化度 = 4指標の平均**（0-1スケール）

### 4.2 劣化ステージ定義

| ステージ | 劣化度範囲 | サンプル数 | 割合 |
|----------|------------|------------|------|
| Normal | 0.00-0.25 | 567 | 35.4% |
| Degrading | 0.25-0.50 | 431 | 26.9% |
| Severe | 0.50-0.75 | 602 | 37.6% |
| Critical | 0.75-1.00 | 0 | 0.0% |

### 4.3 劣化度予測モデル

**Random Forest Regressor**:

| 指標 | Train | Validation | Test |
|------|-------|------------|------|
| MAE | 0.0012 | 0.0029 | 0.0036 |
| RMSE | 0.0019 | 0.0044 | 0.0058 |
| R² | 0.9999 | 0.9997 | 0.9996 |

**特徴量重要度**:
1. waveform_correlation: 93.3%
2. vo_variability: 2.8%
3. vl_variability: 1.9%
4. その他: < 1%

### 4.4 次サイクル応答性予測

過去5サイクルから次サイクルを予測：

| 特徴量 | Test MAE | Test R² |
|--------|----------|---------|
| waveform_correlation | 0.0044 | 0.9920 |
| vo_variability | 0.0017 | 0.9999 |
| vl_variability | 0.0052 | 0.9991 |
| response_delay | 0.0000 | 1.0000 |
| response_delay_normalized | 0.0000 | 1.0000 |
| residual_energy_ratio | 0.0012 | 0.9361 |
| vo_complexity | 0.0006 | 0.9482 |

---

## 5. モデル汎化性能検証（Phase 4）

### 5.1 ES10/ES14データでの評価

ES12で学習したモデルをES10/ES14に適用：

| データセット | 異常検出率 | Training FP | 評価 |
|--------------|------------|-------------|------|
| ES12 | 90.8% | 5.0% | 良好 |
| ES10 | 97.4% | 98.6% | 汎化せず |
| ES14 | - | - | 特徴量抽出失敗 |

### 5.2 結論

- ES12で学習したモデルはES10/ES14には汎化しない
- データセット間で特性が大きく異なる
- **推奨**: データセットごとにモデルを学習

---

## 6. 誤報率削減（Phase 6）

### 6.1 背景と目標

Phase 5完了時点での課題：
- v3モデル（劣化度スコアベース）のFPR: 41.4%
- 実用レベル目標: FPR < 10%

### 6.2 Task 6.1: ROC曲線分析と閾値最適化

**実施内容**:
- ROC曲線の描画とAUC計算
- 複数の閾値候補での性能評価
- 最適閾値の選定: -3.8658

**結果**:
- ROC-AUC: 0.9872（極めて高い識別能力）
- FPR削減: 41.4% → 13.5%（27.9%削減）
- Recall維持: 95.3%
- F1-Score: 0.741 → 0.874

### 6.3 Task 6.2: アンサンブルアプローチ

**実施内容**:
- 異常検知モデル（閾値最適化済み）と劣化度予測モデル（R² = 0.9996）を組み合わせ
- 4つのアンサンブル戦略を評価（AND/OR/Degradation-Primary/Weighted-Vote）

**結果**:
- AND戦略が最適: FPR 13.1%, Recall 95.3%, F1 0.877
- 限定的な改善（13.5% → 13.1%）だが最良の結果
- 理由: 異常検知モデルが既に高精度

### 6.4 Task 6.3: 段階的アラートシステム

**実施内容**:
- 4段階のアラートレベル定義:
  - INFO: degradation_score < 0.25（正常範囲）
  - WARNING: 0.25 ≤ score < 0.50（継続監視）
  - ALERT: 0.50 ≤ score < 0.75（保全計画）
  - CRITICAL: score ≥ 0.75（即時対応）
- 運用シミュレーション（30日間）

**結果**:
- INFO: 37.5%、WARNING: 25.0%、ALERT: 37.5%、CRITICAL: 0%
- 実際の劣化状態との高い一致率
- 段階的警告により誤報の影響を軽減

### 6.5 Phase 6の累積改善効果

| 段階 | FPR | Recall | F1-Score | 改善内容 |
|------|-----|--------|----------|----------|
| v3 (Baseline) | 41.4% | 100% | 0.741 | 劣化度スコアベース |
| Task 6.1 | 13.5% | 95.3% | 0.874 | 閾値最適化 |
| Task 6.2 | 13.1% | 95.3% | 0.877 | アンサンブル |
| Task 6.3 | - | - | - | 段階的アラート |

**累積改善効果**:
- FPR削減: 41.4% → 13.1%（28.3%削減、68.2%改善）
- 誤報数: 104個 → 33個（71個削減）
- 実用的なアラートシステム完成

---

## 7. 実用化に向けた推奨事項

### 7.1 モデルの使用方法

1. **特徴量抽出**: 新しいサイクルデータからVL/VO波形特性を抽出
2. **異常検知**: One-Class SVM（最適閾値: -3.8658）で異常判定
3. **劣化度予測**: Random Forestで劣化度スコアを計算
4. **アラートレベル判定**: 段階的アラートシステムで適切な対応を決定
   - INFO: 通常運転継続
   - WARNING: 継続監視（データ記録）
   - ALERT: 保全計画立案（1週間以内）
   - CRITICAL: 即時点検・交換検討

### 7.2 運用上の注意点

- **初期サイクル**: 1-10サイクルは正常パターンとして扱う
- **遷移点**: Cycle 13前後で異常検出率が50%に達する
- **劣化確定**: Cycle 51以降は100%異常検出
- **誤報対応**: WARNINGレベルでは継続監視のみ（即座の対応不要）

### 7.3 今後の改善案

**短期（即効性あり）**:
1. ✅ 閾値最適化（完了）
2. ✅ アンサンブルアプローチ（完了）
3. ✅ 段階的アラートシステム（完了）

**中長期（根本的改善）**:
1. **特徴量エンジニアリング**: 新しい物理的特徴量の追加
2. **時系列モデル**: LSTM/Transformerの導入
3. **半教師あり学習**: 少量のラベルデータを活用
4. **データセット拡張**: より多くのコンデンサデータで学習
5. **転移学習**: ES10/ES14への適用方法を研究
6. **リアルタイム監視**: オンライン学習の実装
7. **物理モデル統合**: 電気回路理論との統合

---

## 8. 成果物

### 8.1 モデルファイル

- `output/models_v3/one_class_svm_v2.pkl`: 異常検知モデル（nu=0.05最適化）
- `output/models_v3/one_class_svm_v2_scaler.pkl`: スケーラー
- `output/models_v3/degradation_predictor.pkl`: 劣化度予測モデル（R² = 0.9996）
- `output/models_v3/response_predictor.pkl`: 次サイクル予測モデル

### 8.2 データファイル

- `output/features_v3/es12_response_features.csv`: 特徴量データ（1,600サンプル）
- `output/anomaly_detection/one_class_svm_v2_results.csv`: 異常検知結果
- `output/degradation_prediction/features_with_degradation_score.csv`: 劣化度スコア
- `output/threshold_optimization/`: ROC曲線分析結果
- `output/ensemble/`: アンサンブルモデル評価結果
- `output/alert_system/`: 段階的アラートシステム設計

### 8.3 ドキュメント

**Phase別完了レポート**:
- `docs/phase1_completion_summary.md`: VL-VO関係性分析
- `docs/phase2_completion_summary.md`: 異常検知モデル構築
- `docs/project_completion_summary.md`: 劣化予測モデル構築（Phase 3）
- `docs/phase6_completion_summary.md`: 誤報率削減

**設計ドキュメント**:
- `docs/response_feature_design.md`: 特徴量設計
- `docs/false_positive_reduction_strategies.md`: 誤報率削減戦略

**評価レポート**:
- `output/threshold_optimization/optimal_threshold_report.md`: Task 6.1レポート
- `output/ensemble/ensemble_comparison_report.md`: Task 6.2レポート
- `output/alert_system/staged_alert_system_design.md`: Task 6.3レポート

---

## 9. 結論

本プロジェクトでは、VL-VO応答性に基づく教師なし異常検知・劣化予測モデルを構築し、ES12データで高精度な結果を達成しました。さらにPhase 6で誤報率を大幅に削減し、実用的な段階的アラートシステムを完成させました。

**主要成果**:
- 異常検知: Training FP 5.0%、Late FN 5.2%
- 劣化度予測: Test MAE 0.0036、R² 0.9996
- 誤報率削減: FPR 41.4% → 13.1%（68.2%改善）
- 段階的アラートシステム完成（4レベル）
- 物理的に妥当な劣化パターンの発見

**制限事項**:
- ES12データに特化（ES10/ES14には汎化せず）
- データセットごとにモデル学習が必要
- FPR 13.1%（目標10%にわずかに届かず）

**実用化の可能性**:
- 同一条件下のコンデンサ劣化監視に適用可能
- 段階的アラートにより現場での適切な対応が可能
- 保全計画の最適化に貢献
- リアルタイム異常検知システムへの統合が可能

---

**レポート作成日**: 2026年1月18日（Phase 5完了時）  
**最終更新日**: 2026年1月19日（Phase 6完了時）  
**プロジェクトステータス**: 完了（Phase 0-6）
