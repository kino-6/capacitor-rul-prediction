# 劣化度スコアの定義

**作成日**: 2026-01-18
**Task**: 3.1 劣化度スコアの定義

---

## 概要

劣化度スコアは、コンデンサの劣化状態を0（正常）から1（完全劣化）までの連続値で表現する指標です。

## 計算方法

### 複合指標アプローチ

4つの波形特性指標を組み合わせた複合指標を使用:

```
degradation_score = (
    degradation_score_corr +
    degradation_score_vo_var +
    degradation_score_vl_var +
    degradation_score_residual
) / 4.0
```

### 各指標の定義

1. **Correlation-based Score**:
   ```
   degradation_score_corr = (correlation - initial_correlation) / (1.0 - initial_correlation)
   ```
   - 劣化でWaveform Correlationが1.0に近づく（波形単純化）

2. **VO Variability-based Score**:
   ```
   degradation_score_vo_var = (vo_variability - initial_vo_var) / (max_vo_var - initial_vo_var)
   ```
   - 劣化でVO Variabilityが増加（応答不安定化）

3. **VL Variability-based Score**:
   ```
   degradation_score_vl_var = (vl_variability - initial_vl_var) / (max_vl_var - initial_vl_var)
   ```
   - 劣化でVL Variabilityが増加

4. **Residual Energy-based Score**:
   ```
   degradation_score_residual = (residual_energy - initial_residual) / (max_residual - initial_residual)
   ```
   - 劣化でResidual Energy Ratioが増加（線形関係からの逸脱）

## 劣化ステージの定義

| Stage | Score Range | 特徴 |
|-------|-------------|------|
| Normal | 0.0 - 0.25 | 初期サイクル、正常状態 |
| Degrading | 0.25 - 0.50 | 劣化開始、応答性低下 |
| Severe | 0.50 - 0.75 | 深刻な劣化、大幅な性能低下 |
| Critical | 0.75 - 1.0 | 故障寸前、極めて不安定 |

## 物理的妥当性

- ✅ 単調増加: 劣化度スコアはサイクル数と正の相関
- ✅ 回復なし: コンデンサは劣化から回復しない
- ✅ 初期正常: 初期サイクル（1-10）は0に近い
- ✅ 後期劣化: 後期サイクル（190-200）は1に近い

## 統計情報

- 全サンプル数: 1600
- Degradation Score範囲: 0.000 - 0.731
- Degradation Score平均: 0.401 ± 0.281

### ステージ別サンプル数

- Normal: 567サンプル (35.4%)
- Degrading: 431サンプル (26.9%)
- Severe: 602サンプル (37.6%)
- Critical: 0サンプル (0.0%)

---

## 出力ファイル

- `features_with_degradation_score.csv`: 劣化度スコア付き特徴量データ
- `degradation_score_visualization.png`: 劣化度スコアの可視化
- `degradation_score_definition.md`: 本ドキュメント
