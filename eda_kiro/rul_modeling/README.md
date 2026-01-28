# コンデンサ劣化予測プロジェクト

VL-VO応答性に基づく教師なし異常検知・劣化予測モデル

**プロジェクト期間**: 2026年1月15日 - 2026年1月19日  
**ステータス**: ✅ 完了（Phase 0-6）

---

## 🎯 プロジェクト概要

コンデンサの入力電圧（VL）と出力電圧（VO）の関係性を分析し、劣化を検出・予測するモデルを構築しました。従来のRUL予測アプローチの問題点を解決し、物理的に意味のある特徴量で高精度な異常検知と劣化度予測を実現しました。

### 主要成果

| 指標 | 結果 |
|------|------|
| **異常検知精度** | Training FP 5.0%, Late FN 5.2% |
| **劣化度予測精度** | Test MAE 0.0036, R² 0.9996 |
| **誤報率削減** | FPR 41.4% → 13.1%（68.2%改善） |
| **アラートシステム** | 4段階（INFO/WARNING/ALERT/CRITICAL） |

---

## 📖 まず読むべきドキュメント

### 1. 最終レポート（推奨）⭐
**[`docs/FINAL_PROJECT_REPORT.md`](docs/FINAL_PROJECT_REPORT.md)**

プロジェクト全体の成果を統合した最終レポート。以下の内容を含みます：
- エグゼクティブサマリー
- Phase 1-6の詳細結果
- モデル性能評価
- 実用化に向けた推奨事項

### 2. エンドツーエンドデモ
**[`output/demo/end_to_end_demo_report.md`](output/demo/end_to_end_demo_report.md)**

実用化イメージを提示するデモレポート。新しいVL/VOデータから段階的アラートまでの一貫した処理フローを実演。

**デモスクリプト**: [`scripts/end_to_end_inference_demo.py`](scripts/end_to_end_inference_demo.py)

### 3. Phase別完了サマリー

各Phaseの詳細な成果をまとめたドキュメント：

- **Phase 1**: [`docs/phase1_completion_summary.md`](docs/phase1_completion_summary.md) - VL-VO関係性分析
- **Phase 2**: [`docs/phase2_completion_summary.md`](docs/phase2_completion_summary.md) - 異常検知モデル構築
- **Phase 3**: [`docs/project_completion_summary.md`](docs/project_completion_summary.md) - 劣化予測モデル構築
- **Phase 6**: [`docs/phase6_completion_summary.md`](docs/phase6_completion_summary.md) - 誤報率削減

---

## 🚀 クイックスタート

### デモの実行

エンドツーエンドの推論デモを実行：

```bash
cd rul_modeling
python scripts/end_to_end_inference_demo.py
```

出力ファイル：
- `output/demo/end_to_end_demo_visualization.png` - 9パネルの統合可視化
- `output/demo/end_to_end_demo_report.md` - 詳細レポート
- `output/demo/end_to_end_demo_results.csv` - 結果データ

---

## 📊 主要な成果物

### モデルファイル

| ファイル | 説明 | 性能 |
|---------|------|------|
| `output/models_v3/one_class_svm_v2.pkl` | 異常検知モデル | Training FP 5.0% |
| `output/models_v3/degradation_predictor.pkl` | 劣化度予測モデル | R² 0.9996 |
| `output/models_v3/response_predictor.pkl` | 次サイクル予測モデル | R² > 0.93 |

### 可視化・レポート

**Phase 6（誤報率削減）**:
- [`output/threshold_optimization/optimal_threshold_report.md`](output/threshold_optimization/optimal_threshold_report.md) - ROC曲線分析
- [`output/ensemble/ensemble_comparison_report.md`](output/ensemble/ensemble_comparison_report.md) - アンサンブル評価
- [`output/alert_system/staged_alert_system_design.md`](output/alert_system/staged_alert_system_design.md) - アラートシステム設計

**Phase 1-3**:
- `output/vl_vo_analysis/` - VL-VO関係性分析結果
- `output/anomaly_detection/` - 異常検知モデル評価結果
- `output/degradation_prediction/` - 劣化度予測結果

---

## 🏗️ プロジェクト構造

```
rul_modeling/
├── README.md                           # このファイル
├── docs/
│   ├── FINAL_PROJECT_REPORT.md        # ⭐ 最終レポート
│   ├── phase1_completion_summary.md   # Phase 1サマリー
│   ├── phase2_completion_summary.md   # Phase 2サマリー
│   ├── project_completion_summary.md  # Phase 3サマリー
│   ├── phase6_completion_summary.md   # Phase 6サマリー
│   └── response_feature_design.md     # 特徴量設計
├── scripts/
│   ├── end_to_end_inference_demo.py   # ⭐ エンドツーエンドデモ
│   ├── optimize_threshold_roc.py      # ROC曲線分析
│   ├── build_ensemble_model.py        # アンサンブルモデル
│   └── design_staged_alert_system.py  # アラートシステム設計
├── output/
│   ├── demo/                          # ⭐ デモ結果
│   ├── models_v3/                     # 学習済みモデル
│   ├── threshold_optimization/        # 閾値最適化結果
│   ├── ensemble/                      # アンサンブル評価結果
│   └── alert_system/                  # アラートシステム設計
├── src/
│   └── feature_extraction/
│       └── response_extractor.py      # 特徴量抽出クラス
└── .kiro/specs/rul_model_spec/
    ├── tasks.md                       # プロジェクトタスク
    └── tasks_archive.md               # 完了タスクアーカイブ
```

---

## 🔬 技術的詳細

### アプローチ

従来のRUL予測の問題点（`RUL = 200 - cycle_number`は単なる逆算、物理的根拠なし）を解決し、VL-VO応答性の劣化を物理的に意味のある特徴量で検出。

### 処理フロー

```
VL/VOデータ
    ↓
特徴量抽出（7特徴量）
    ↓
異常検知（One-Class SVM）
    ↓
劣化度予測（Random Forest）
    ↓
段階的アラート判定
    ↓
推奨アクション
```

### 使用した特徴量（7個）

波形特性のみを使用：
- `waveform_correlation` - VLとVOの相関係数
- `vo_variability` - VO変動係数
- `vl_variability` - VL変動係数
- `response_delay` - 応答遅延
- `response_delay_normalized` - 正規化応答遅延
- `residual_energy_ratio` - 残差エネルギー比
- `vo_complexity` - 波形複雑度

### データ分割

```
Train: C1-C5（1,000サンプル）
Val:   C6（200サンプル）
Test:  C7-C8（400サンプル）
```

---

## 📈 Phase別の成果

### Phase 0: 探索的特徴量分析
- VL/VO特徴量の相関分析
- 高相関特徴量10個特定

### Phase 1: VL-VO関係性分析
- 劣化パターンの発見（Response Efficiency 98.5%減少）
- 15個の応答性特徴量を設計・抽出
- 劣化ステージの定義

### Phase 2: 異常検知モデル構築
- One-Class SVM v2（波形特性のみ、nu=0.05最適化）
- Training FP 5.0%, Late FN 5.2%
- 物理的に妥当な劣化パターンを検出

### Phase 3: 劣化予測モデル構築
- 劣化度スコアの定義（0-1スケール）
- Random Forest Regressor（Test MAE 0.0036, R² 0.9996）
- 次サイクル応答性予測（全特徴量でR² > 0.93）

### Phase 4: モデル汎化性能検証
- ES10/ES14データでの評価
- 結論: ES12モデルはES10/ES14に汎化せず
- データセットごとにモデル学習が必要

### Phase 5: 最終レポート作成
- プロジェクト全体の成果を統合
- 実用化に向けた推奨事項を提供

### Phase 6: 誤報率削減
- **Task 6.1**: ROC曲線分析と閾値最適化（FPR 41.4% → 13.5%）
- **Task 6.2**: アンサンブルアプローチ（FPR 13.5% → 13.1%）
- **Task 6.3**: 段階的アラートシステム設計（4レベル）

---

## 🎯 実用化に向けて

### 段階的アラートシステム

| レベル | 条件 | 推奨アクション | 頻度 |
|--------|------|---------------|------|
| INFO | degradation_score < 0.25 | 通常運転継続 | 37.5% |
| WARNING | 0.25 ≤ score < 0.50 | 継続監視（データ記録） | 25.0% |
| ALERT | 0.50 ≤ score < 0.75 | 保全計画立案（1週間以内） | 37.5% |
| CRITICAL | score ≥ 0.75 | 即時点検・交換検討 | 0% |

### 実用化のメリット

✅ 段階的な警告により適切な対応が可能  
✅ 誤報の影響を軽減（WARNINGレベルでは継続監視のみ）  
✅ 保全計画の最適化（計画的な部品交換）  
✅ 高精度な劣化度予測（R² = 0.9996）

---

## 📚 過去の経緯・詳細ドキュメント

プロジェクトの詳細な経緯や中間成果物は以下を参照：

### 設計ドキュメント
- [`docs/response_feature_design.md`](docs/response_feature_design.md) - 特徴量設計の詳細
- [`docs/false_positive_reduction_strategies.md`](docs/false_positive_reduction_strategies.md) - 誤報率削減戦略
- [`docs/phase_restructure_plan.md`](docs/phase_restructure_plan.md) - Phase再構築計画

### 評価レポート
- `output/anomaly_detection/anomaly_detection_comparison.md` - 異常検知モデル比較
- `output/anomaly_detection/anomaly_validation_report.md` - 異常検知結果の検証
- `output/degradation_prediction/degradation_score_definition.md` - 劣化度スコア定義

### タスク管理
- [`.kiro/specs/rul_model_spec/tasks.md`](.kiro/specs/rul_model_spec/tasks.md) - 現在のタスク状況
- [`.kiro/specs/rul_model_spec/tasks_archive.md`](.kiro/specs/rul_model_spec/tasks_archive.md) - 完了タスクの詳細アーカイブ

---

## 🔗 関連リンク

- **GitHub**: https://github.com/kino-6/capacitor-rul-prediction
- **データセット**: NASA PCOE ES12

---

**作成日**: 2026-01-15  
**最終更新日**: 2026-01-19  
**プロジェクトステータス**: Phase 0-6完了、実用化準備完了
