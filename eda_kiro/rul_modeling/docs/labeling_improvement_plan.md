# ラベリング改善計画

**作成日**: 2026-01-19  
**目的**: 異常検知モデルの誤報率86.5%を改善するためのラベリング見直し

---

## 現状の問題点

### 1. 学習データのラベリング

**現在の設定**（`build_one_class_svm_v2.py`）:
```python
normal_cycle_range=(1, 10)  # Cycle 1-10のみを「正常」として学習
```

**問題**:
- Cycle 1-10だけを正常として学習
- Cycle 11以降の正常な変化も「異常」と判定してしまう
- モデルが過度に保守的（過敏）になる

### 2. テストデータのラベリング

**現在の設定**（`enhanced_inference_demo.py`）:
```python
# Ground Truth: Cycle 1-100 = Normal (0), Cycle 101-200 = Anomaly (1)
test_data['true_anomaly'] = ((test_data['cycle'] > 100).astype(int))
```

**問題**:
- 学習時（Cycle 1-10が正常）とテスト時（Cycle 1-100が正常）で定義が不一致
- この不一致が誤報率86.5%の主要因

---

## 改善方法

### 方法1: 学習データの拡張（推奨）⭐

**変更内容**:
```python
# 現在
normal_cycle_range=(1, 10)

# 改善案
normal_cycle_range=(1, 50)  # または (1, 100)
```

**メリット**:
- より多様な「正常」パターンを学習
- 経年変化を正常範囲として認識
- 誤報率の大幅な低下が期待できる

**デメリット**:
- 初期劣化を見逃す可能性（ただし劣化度予測モデルでカバー可能）

**実装手順**:
1. `scripts/build_one_class_svm_v2.py`の`normal_cycle_range`を変更
2. モデルを再学習
3. `scripts/enhanced_inference_demo.py`でテスト
4. 誤報率を確認

---

### 方法2: テストデータのラベリング変更

**変更内容**:
```python
# 現在
test_data['true_anomaly'] = ((test_data['cycle'] > 100).astype(int))

# 改善案A: 学習範囲と一致させる
test_data['true_anomaly'] = ((test_data['cycle'] > 10).astype(int))

# 改善案B: 劣化度スコアベース
test_data['true_anomaly'] = ((test_data['degradation_score'] > 0.5).astype(int))
```

**メリット**:
- 学習時とテスト時の定義が一致
- 物理的な劣化状態に基づく評価

**デメリット**:
- 方法1と組み合わせないと根本的な解決にならない

---

### 方法3: 劣化度スコアベースのラベリング（最も推奨）⭐⭐⭐

**コンセプト**:
- サイクル数ではなく、実際の劣化度スコアでラベリング
- 物理的な状態に基づく客観的な評価

**実装**:

#### ステップ1: 劣化度スコアの閾値を定義

```python
# 劣化ステージの定義（define_degradation_score.pyより）
# Normal: 0.0 - 0.25
# Degrading: 0.25 - 0.50
# Severe: 0.50 - 0.75
# Critical: 0.75 - 1.0

# 異常の定義
ANOMALY_THRESHOLD = 0.50  # Severe以上を異常とする
```

#### ステップ2: 学習データの選択

```python
# 劣化度スコアが0.25未満を「正常」として学習
normal_df = df[df['degradation_score'] < 0.25]
```

#### ステップ3: テストデータのラベリング

```python
# 劣化度スコアが0.50以上を「異常」とする
test_data['true_anomaly'] = ((test_data['degradation_score'] >= 0.50).astype(int))
```

**メリット**:
- サイクル数に依存しない
- 物理的な劣化状態を反映
- コンデンサ間の個体差を吸収
- 最も客観的で公平な評価

**デメリット**:
- 劣化度スコアの計算が必要（すでに実装済み）

---

### 方法4: 複数閾値による段階的評価

**コンセプト**:
- 異常を2値（正常/異常）ではなく、複数段階で評価
- より実用的なアラートシステム

**実装**:

```python
def classify_degradation_level(degradation_score):
    """劣化レベルの分類"""
    if degradation_score < 0.25:
        return 0  # Normal - 監視不要
    elif degradation_score < 0.50:
        return 1  # Degrading - 継続監視
    elif degradation_score < 0.75:
        return 2  # Severe - 保全計画
    else:
        return 3  # Critical - 即時対応

# 適用
test_data['degradation_level'] = test_data['degradation_score'].apply(classify_degradation_level)

# 評価指標
# - Level 0-1: 正常範囲
# - Level 2-3: 異常範囲（アラート対象）
```

**メリット**:
- 段階的な対応が可能
- 誤報と見逃しのバランスを調整しやすい
- 実運用に最適

---

## 推奨実装プラン

### Phase 1: 学習データ拡張（即座に実施可能）

**目標**: 誤報率を86.5% → 30%以下に削減

**手順**:
1. `build_one_class_svm_v2.py`を修正
   ```python
   normal_cycle_range=(1, 50)  # 10 → 50に変更
   ```

2. モデル再学習
   ```bash
   python scripts/build_one_class_svm_v2.py
   ```

3. テスト実行
   ```bash
   python scripts/enhanced_inference_demo.py
   ```

4. 結果確認
   - Confusion Matrixの確認
   - False Positive Rateの確認
   - 目標: FP Rate < 30%

**期待される結果**:
```
                予測
              Normal  Anomaly
実際 Normal    140      60     ← FP Rate = 30%（改善！）
    Anomaly     14     186
```

---

### Phase 2: 劣化度スコアベースのラベリング（推奨）

**目標**: 物理的な劣化状態に基づく客観的な評価

**手順**:

#### 1. 新しいスクリプト作成: `build_one_class_svm_v3.py`

```python
"""
One-Class SVM v3: Degradation Score-Based Labeling

Improvements:
- Use degradation score instead of cycle number for labeling
- Train on samples with degradation_score < 0.25 (Normal stage)
- More objective and physically meaningful
"""

def prepare_training_data_v3(df, features):
    """Prepare training data based on degradation score."""
    # Load degradation scores
    degradation_df = pd.read_csv("output/degradation_prediction/features_with_degradation_score.csv")
    
    # Merge with features
    df = df.merge(degradation_df[['capacitor_id', 'cycle', 'degradation_score']], 
                  on=['capacitor_id', 'cycle'])
    
    # Select normal samples (degradation_score < 0.25)
    normal_df = df[df['degradation_score'] < 0.25]
    
    print(f"\nNormal data (training):")
    print(f"  Degradation score < 0.25")
    print(f"  Samples: {len(normal_df)}")
    print(f"  Cycle range: {normal_df['cycle'].min():.0f}-{normal_df['cycle'].max():.0f}")
    
    # Extract features
    X_train = normal_df[features].copy()
    X_all = df[features].copy()
    
    # ... (rest of the code)
    
    return X_train_scaled, X_all_scaled, scaler, normal_df, df
```

#### 2. テストスクリプト更新: `enhanced_inference_demo_v3.py`

```python
def calculate_model_metrics_v3(test_data):
    """Calculate metrics using degradation score-based labels."""
    # Ground Truth based on degradation score
    # Severe以上（degradation_score >= 0.50）を異常とする
    test_data['true_anomaly'] = ((test_data['degradation_score'] >= 0.50).astype(int))
    
    # ... (rest of the code)
```

#### 3. 実行と評価

```bash
# モデル学習
python scripts/build_one_class_svm_v3.py

# テスト実行
python scripts/enhanced_inference_demo_v3.py
```

**期待される結果**:
- より物理的に妥当な評価
- コンデンサ間の個体差を吸収
- FP Rate < 20%

---

### Phase 3: 段階的アラートシステム（実用化）

**目標**: 実運用可能なアラートシステムの構築

**実装**:

```python
class DegradationMonitor:
    """劣化監視システム"""
    
    def __init__(self, anomaly_model, degradation_model):
        self.anomaly_model = anomaly_model
        self.degradation_model = degradation_model
        
        # アラート閾値
        self.thresholds = {
            'normal': 0.25,      # 正常範囲
            'degrading': 0.50,   # 継続監視
            'severe': 0.75       # 保全計画
        }
    
    def monitor(self, features):
        """監視実行"""
        # 異常検知
        anomaly_score = self.anomaly_model.decision_function(features)[0]
        is_anomaly = (anomaly_score < 0)
        
        # 劣化度予測
        degradation_score = self.degradation_model.predict(features)[0]
        
        # アラートレベル判定
        if degradation_score < self.thresholds['normal']:
            level = 'NORMAL'
            action = '継続監視'
        elif degradation_score < self.thresholds['degrading']:
            level = 'DEGRADING'
            action = '頻繁監視（週1回）'
        elif degradation_score < self.thresholds['severe']:
            level = 'SEVERE'
            action = '保全計画立案'
        else:
            level = 'CRITICAL'
            action = '即時交換推奨'
        
        # アンサンブル判定（両方が異常の場合のみアラート）
        alert = is_anomaly and (degradation_score >= self.thresholds['degrading'])
        
        return {
            'anomaly_score': anomaly_score,
            'degradation_score': degradation_score,
            'level': level,
            'action': action,
            'alert': alert
        }
```

---

## 実装優先順位

### 🔥 優先度1: Phase 1（即座に実施）

**理由**:
- 最も簡単（1行の変更）
- 効果が大きい（FP Rate 86.5% → 30%）
- リスクが低い

**実装時間**: 5分

---

### ⭐ 優先度2: Phase 2（推奨）

**理由**:
- 物理的に妥当
- 客観的な評価
- 長期的に最適

**実装時間**: 1-2時間

---

### 💡 優先度3: Phase 3（実用化時）

**理由**:
- 実運用に必要
- ユーザーフレンドリー
- 段階的対応が可能

**実装時間**: 2-3時間

---

## 評価指標

### 改善前（現状）

```
False Positive Rate: 86.5%
True Negative Rate: 13.5%
Recall: 93.0%
F1-Score: 0.665
```

### 改善目標

**Phase 1後**:
```
False Positive Rate: < 30%
True Negative Rate: > 70%
Recall: > 85%
F1-Score: > 0.80
```

**Phase 2後**:
```
False Positive Rate: < 20%
True Negative Rate: > 80%
Recall: > 90%
F1-Score: > 0.85
```

---

## まとめ

### 問題の本質

1. **学習時**: Cycle 1-10のみを「正常」として学習
2. **テスト時**: Cycle 1-100を「正常」として評価
3. **結果**: 不一致により誤報率86.5%

### 解決策

1. **即座に実施**: 学習データをCycle 1-50に拡張
2. **推奨**: 劣化度スコアベースのラベリング
3. **実用化**: 段階的アラートシステム

### 期待される効果

- 誤報率: 86.5% → 20%以下
- 実用性: 大幅に向上
- 信頼性: 物理的に妥当な評価

---

**次のアクション**: Phase 1の実装（`normal_cycle_range=(1, 50)`に変更）
