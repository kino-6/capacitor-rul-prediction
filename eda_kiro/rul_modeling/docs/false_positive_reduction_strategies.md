# 誤報率削減のための改善戦略

**作成日**: 2026-01-19  
**現状**: False Positive Rate = 41.4%（251個中104個を誤報）  
**目標**: False Positive Rate < 10%（実用レベル）

---

## 現状分析

### v3の問題点

```
混同行列（v3 Degradation-based）:
                予測
              Normal  Anomaly
実際 Normal    147     104     ← 104個の誤報（41.4%）
    Anomaly      0     149     ← 見逃しゼロ（良い）
```

**問題**:
- 正常サンプル251個のうち104個を異常と誤判定
- 誤報率41.4%は実用上許容できない
- モデルが過敏に反応している

**良い点**:
- 見逃しゼロ（Recall = 100%）
- 異常検出能力は高い

---

## 改善戦略

### 戦略1: 閾値の最適化（ROC曲線分析）

**概要**:
- 現在の閾値（decision_score < 0）を調整
- ROC曲線を描画してFalse Positive RateとTrue Positive Rateのトレードオフを分析
- 最適な閾値を選択

**実装方法**:
```python
from sklearn.metrics import roc_curve, roc_auc_score

# ROC曲線の計算
fpr, tpr, thresholds = roc_curve(y_true, decision_scores)
roc_auc = roc_auc_score(y_true, decision_scores)

# 最適閾値の選択（例：FPR < 10%となる閾値）
optimal_threshold = thresholds[fpr < 0.10][0]
```

**期待される効果**:
- FPR: 41.4% → 10-20%程度
- Recall: 100% → 90-95%程度（若干の見逃しは許容）

**メリット**:
- 実装が簡単
- 既存モデルをそのまま使用可能
- すぐに効果が出る

**デメリット**:
- 見逃しが増える可能性
- 根本的な解決ではない

---

### 戦略2: アンサンブルアプローチ

**概要**:
- 異常検知モデルと劣化度予測モデルの両方を使用
- 両方が「異常」と判定した場合のみアラート

**実装方法**:
```python
# 条件1: 異常検知モデルが異常と判定
is_anomaly_detection = (anomaly_score < threshold)

# 条件2: 劣化度予測モデルが深刻な劣化と判定
is_severe_degradation = (predicted_degradation >= 0.50)

# 両方の条件を満たす場合のみアラート
final_alert = is_anomaly_detection AND is_severe_degradation
```

**期待される効果**:
- FPR: 41.4% → 5-15%程度
- 劣化度予測モデル（R² = 0.9996）の高精度を活用

**メリット**:
- 誤報を大幅に削減
- 2つのモデルの相互補完
- 信頼性が高い

**デメリット**:
- 両方のモデルが必要
- 計算コストが増加

---

### 戦略3: 段階的アラートシステム

**概要**:
- 劣化度スコアに基づいて4段階のアラートレベルを設定
- 異常検知モデルは補助的に使用

**実装方法**:
```python
def get_alert_level(degradation_score, anomaly_score):
    if degradation_score < 0.25:
        # Normal: 正常範囲
        return "INFO"  # 情報レベル（アラートなし）
    
    elif degradation_score < 0.50:
        # Degrading: 劣化開始
        if anomaly_score < -0.5:
            return "WARNING"  # 警告レベル（継続監視）
        else:
            return "INFO"
    
    elif degradation_score < 0.75:
        # Severe: 深刻な劣化
        return "ALERT"  # アラートレベル（保全計画立案）
    
    else:
        # Critical: 故障寸前
        return "CRITICAL"  # 緊急レベル（即時対応）
```

**期待される効果**:
- 誤報の定義が変わる（段階的な警告）
- 現場が適切に対応可能

**メリット**:
- 実用的
- 劣化度予測モデルの高精度を主軸に使用
- 段階的な対応が可能

**デメリット**:
- アラートレベルの定義が必要
- 運用ルールの策定が必要

---

### 戦略4: 特徴量エンジニアリング

**概要**:
- 現在の7つの特徴量を見直し
- より識別力の高い特徴量を追加・選択

**実装方法**:
```python
# 現在の特徴量
current_features = [
    'waveform_correlation',
    'vo_variability',
    'vl_variability',
    'response_delay',
    'response_delay_normalized',
    'residual_energy_ratio',
    'vo_complexity'
]

# 追加候補の特徴量
additional_features = [
    'degradation_score',  # 劣化度スコア自体を特徴量に
    'cycle_normalized',   # 正規化されたサイクル番号
    'vo_vl_ratio',        # VO/VL比率
    'response_efficiency', # 応答効率
    # ... その他のドメイン知識に基づく特徴量
]

# 特徴量選択（例：相互情報量、SHAP値）
from sklearn.feature_selection import mutual_info_classif
mi_scores = mutual_info_classif(X, y)
selected_features = features[mi_scores > threshold]
```

**期待される効果**:
- FPR: 41.4% → 20-30%程度
- モデルの識別力向上

**メリット**:
- 根本的な改善
- モデルの性能向上

**デメリット**:
- 時間がかかる
- ドメイン知識が必要
- 再学習が必要

---

### 戦略5: ハイパーパラメータチューニング

**概要**:
- One-Class SVMのハイパーパラメータ（nu, gamma）を最適化
- グリッドサーチやベイズ最適化を使用

**実装方法**:
```python
from sklearn.model_selection import GridSearchCV

# パラメータグリッド
param_grid = {
    'nu': [0.01, 0.03, 0.05, 0.07, 0.10],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
}

# グリッドサーチ（カスタムスコアリング関数が必要）
# 目標: FPRを最小化しつつRecallを維持
```

**期待される効果**:
- FPR: 41.4% → 30-35%程度
- 現在のnu=0.05が最適とは限らない

**メリット**:
- 既存の枠組みで改善
- 実装が比較的簡単

**デメリット**:
- 改善幅は限定的
- 計算コストが高い

---

### 戦略6: 学習データの拡張

**概要**:
- 現在の学習データ（degradation_score < 0.25, 567サンプル）を見直し
- より多様な正常パターンを学習

**実装方法**:
```python
# 現在
normal_df = df[df['degradation_score'] < 0.25]  # 567サンプル

# 改善案1: 閾値を上げる
normal_df = df[df['degradation_score'] < 0.30]  # より多くのサンプル

# 改善案2: Degradingステージの一部を含める
normal_df = df[df['degradation_score'] < 0.35]

# 改善案3: データ拡張（SMOTE等）
from imblearn.over_sampling import SMOTE
X_resampled, y_resampled = SMOTE().fit_resample(X, y)
```

**期待される効果**:
- FPR: 41.4% → 25-35%程度
- より多様な正常パターンを学習

**メリット**:
- モデルの汎化性能向上
- 正常範囲の定義を拡張

**デメリット**:
- 正常/異常の境界が曖昧になる
- 見逃しが増える可能性

---

### 戦略7: 異なるアルゴリズムの検討

**概要**:
- One-Class SVM以外のアルゴリズムを試す
- Isolation Forest, Local Outlier Factor, Autoencoderなど

**実装方法**:
```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_forest.fit(X_train)

# Local Outlier Factor
lof = LocalOutlierFactor(novelty=True, contamination=0.1)
lof.fit(X_train)

# Autoencoder（深層学習）
# 正常データで学習し、再構成誤差で異常検知
```

**期待される効果**:
- FPR: 41.4% → 15-30%程度（アルゴリズム依存）
- One-Class SVMより適したアルゴリズムがある可能性

**メリット**:
- 新しいアプローチ
- 性能向上の可能性

**デメリット**:
- 実装・評価に時間がかかる
- 必ずしも改善するとは限らない

---

## 推奨される実装順序

### フェーズ1: 即効性のある改善（1-2日）

1. **戦略1: 閾値の最適化**
   - ROC曲線分析
   - 最適閾値の選択
   - 期待効果: FPR 41.4% → 10-20%

2. **戦略2: アンサンブルアプローチ**
   - 異常検知 + 劣化度予測の組み合わせ
   - 期待効果: FPR 41.4% → 5-15%

### フェーズ2: 実用化に向けた改善（3-5日）

3. **戦略3: 段階的アラートシステム**
   - 4段階のアラートレベル定義
   - 運用ルールの策定
   - 期待効果: 実用的な警告システム

4. **戦略5: ハイパーパラメータチューニング**
   - グリッドサーチ
   - 期待効果: FPR 30-35%

### フェーズ3: 根本的な改善（1-2週間）

5. **戦略4: 特徴量エンジニアリング**
   - 新しい特徴量の追加
   - 特徴量選択
   - 期待効果: FPR 20-30%

6. **戦略6: 学習データの拡張**
   - 正常範囲の再定義
   - 期待効果: FPR 25-35%

### フェーズ4: 代替アプローチ（必要に応じて）

7. **戦略7: 異なるアルゴリズムの検討**
   - Isolation Forest, LOF, Autoencoder
   - 期待効果: FPR 15-30%

---

## 評価指標

### 目標値

| 指標 | 現状 | 目標 | 許容範囲 |
|------|------|------|----------|
| False Positive Rate | 41.4% | < 10% | < 15% |
| Recall | 100% | > 90% | > 85% |
| F1-Score | 0.741 | > 0.90 | > 0.85 |

### 評価方法

1. **ROC-AUC**: 全体的な性能評価
2. **Precision-Recall曲線**: 不均衡データでの評価
3. **実用シミュレーション**: 工場での運用を想定したコスト試算（改善後）

---

## まとめ

### 最優先で実装すべき戦略

1. **戦略1: 閾値の最適化** - 即効性あり、簡単
2. **戦略2: アンサンブルアプローチ** - 効果大、実装容易
3. **戦略3: 段階的アラートシステム** - 実用的

### 期待される最終結果

- **戦略1+2の組み合わせ**: FPR 5-15%程度
- **戦略3の追加**: 実用的な警告システム
- **実用化の可能性**: 高い

### 次のステップ

1. ROC曲線分析の実装
2. 最適閾値の選択
3. アンサンブルアプローチの実装
4. 性能評価とレポート作成

---

**作成者**: Kiro AI Agent  
**作成日**: 2026-01-19  
**関連ファイル**:
- `scripts/build_one_class_svm_v3_degradation_based.py` (現在のモデル)
- `scripts/enhanced_inference_demo_v3_degradation_based.py` (現在の評価)
