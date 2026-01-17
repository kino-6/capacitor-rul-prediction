# RUL予測モデル設計

## 🎯 モデル構成

### 2段階アプローチ

```
Input Features → [Primary: Anomaly Classifier] → [Secondary: RUL Predictor] → RUL Output
                         ↓
                   Normal / Abnormal
```

**Primary Model (異常検知)**:
- 目的: コンデンサの状態が正常か異常かを分類
- Output: Binary (0: Normal, 1: Abnormal)
- 用途: 異常検知後にRUL予測を実施

**Secondary Model (RUL予測)**:
- 目的: 残存耐用寿命（RUL）を予測
- Input: Primary Modelで異常と判定されたデータ
- Output: Continuous (残りサイクル数)

## 📊 ES12データの特徴

### 利用可能なデータ

**ES12データセット構造**:
- 8個のコンデンサ (ES12C1 ~ ES12C8)
- 各コンデンサ約200サイクル
- 各サイクルに時系列データ（VL, VO）

**時系列データ**:
- VL (Input Voltage): 入力電圧波形（約3000ポイント/サイクル）
- VO (Output Voltage): 出力電圧波形（約3000ポイント/サイクル）

### EDAからの知見

1. **劣化パターン**:
   - VL入力が類似していても、VOは時間経過で変化
   - 電圧比（VO/VL）が劣化指標として有効
   - 50サイクル以上の時間差で明確な劣化を観測

2. **データの特性**:
   - Sin波のような周期的パターンは存在しない
   - ほぼ一定値 ± ノイズのパターン
   - 実運用環境の実データ

## 🔧 特徴量設計

### サイクルレベル特徴量

各サイクルから以下の統計量を抽出：

#### 1. 基本統計量
```python
# VL (Input)
- vl_mean: 平均値
- vl_std: 標準偏差
- vl_min: 最小値
- vl_max: 最大値
- vl_range: 範囲 (max - min)
- vl_median: 中央値
- vl_q25, vl_q75: 四分位数

# VO (Output)
- vo_mean: 平均値
- vo_std: 標準偏差
- vo_min: 最小値
- vo_max: 最大値
- vo_range: 範囲
- vo_median: 中央値
- vo_q25, vo_q75: 四分位数
```

#### 2. 劣化指標
```python
# 電圧比
- voltage_ratio: vo_mean / vl_mean
- voltage_ratio_std: std(VO/VL)

# 応答特性
- response_efficiency: vo_range / vl_range
- signal_attenuation: 1 - (vo_std / vl_std)
```

#### 3. 時系列特徴
```python
# トレンド
- vl_trend: 線形回帰の傾き
- vo_trend: 線形回帰の傾き

# 変動性
- vl_cv: 変動係数 (std / mean)
- vo_cv: 変動係数
```

#### 4. 履歴特徴（ウィンドウベース）
```python
# 過去Nサイクルの統計
- voltage_ratio_mean_last_5: 過去5サイクルの平均電圧比
- voltage_ratio_std_last_5: 過去5サイクルの電圧比の標準偏差
- voltage_ratio_trend_last_10: 過去10サイクルの電圧比のトレンド

# 劣化率
- degradation_rate: (current_ratio - initial_ratio) / cycle_number
```

### 特徴量の次元

**サイクルごと**: 約20-30特徴量
**コンデンサごと**: 約200サイクル × 20-30特徴量

## 📈 データセット構築

### Primary Model (異常検知)

#### Input
```python
X_primary = [
    # サイクルレベル特徴量 (20-30次元)
    vl_mean, vl_std, vl_range, ...,
    vo_mean, vo_std, vo_range, ...,
    voltage_ratio, response_efficiency, ...,
    # 履歴特徴量
    voltage_ratio_mean_last_5, ...
]
```

#### Output (Label)
```python
y_primary = {
    0: "Normal",    # 初期サイクル（例: 1-50サイクル）
    1: "Abnormal"   # 劣化サイクル（例: 150-200サイクル）
}
```

**ラベリング戦略**:
- Option 1: 閾値ベース（電圧比が初期値から X% 変化）
- Option 2: サイクル番号ベース（前半50%: Normal, 後半50%: Abnormal）
- Option 3: 専門家ラベル（ドメイン知識に基づく）

### Secondary Model (RUL予測)

#### Input
```python
X_secondary = [
    # Primary Modelと同じ特徴量
    # + Primary Modelの予測確率
    anomaly_probability,
    ...
]
```

#### Output (Target)
```python
y_secondary = remaining_cycles  # 残りサイクル数
# 例: サイクル150の場合、RUL = 200 - 150 = 50
```

## 🗂️ データ分割戦略

### 時系列データの特性を考慮

#### Option 1: コンデンサ単位分割
```python
# 8個のコンデンサを分割
Train: ES12C1, C2, C3, C4, C5, C6  (6個)
Val:   ES12C7                       (1個)
Test:  ES12C8                       (1個)
```

**利点**: 未知のコンデンサに対する汎化性能を評価
**欠点**: データ量が少ない

#### Option 2: サイクル単位分割（時系列考慮）
```python
# 各コンデンサの前半を学習、後半をテスト
Train: サイクル 1-120  (全コンデンサ)
Val:   サイクル 121-160 (全コンデンサ)
Test:  サイクル 161-200 (全コンデンサ)
```

**利点**: データ量が多い
**欠点**: 時系列リークの可能性

#### Option 3: ハイブリッド（推奨）
```python
# コンデンサとサイクルの両方を考慮
Train: C1-C5 の サイクル 1-150
Val:   C6 の サイクル 1-150
Test:  C7-C8 の サイクル 1-200
```

## 🧮 実装例

### 特徴量抽出
```python
def extract_cycle_features(vl, vo, cycle_num, history_df=None):
    """
    1サイクルから特徴量を抽出
    
    Args:
        vl: VL時系列データ (array)
        vo: VO時系列データ (array)
        cycle_num: サイクル番号
        history_df: 過去のサイクル特徴量 (DataFrame)
    
    Returns:
        features: 特徴量辞書
    """
    features = {}
    
    # 基本統計量
    features['vl_mean'] = np.mean(vl)
    features['vl_std'] = np.std(vl)
    features['vl_range'] = np.max(vl) - np.min(vl)
    
    features['vo_mean'] = np.mean(vo)
    features['vo_std'] = np.std(vo)
    features['vo_range'] = np.max(vo) - np.min(vo)
    
    # 劣化指標
    features['voltage_ratio'] = features['vo_mean'] / features['vl_mean']
    features['response_efficiency'] = features['vo_range'] / features['vl_range']
    
    # サイクル情報
    features['cycle_number'] = cycle_num
    features['cycle_normalized'] = cycle_num / 200  # 正規化
    
    # 履歴特徴量（過去データがある場合）
    if history_df is not None and len(history_df) >= 5:
        features['voltage_ratio_mean_last_5'] = history_df['voltage_ratio'].tail(5).mean()
        features['voltage_ratio_std_last_5'] = history_df['voltage_ratio'].tail(5).std()
    
    return features
```

### ラベル生成
```python
def generate_labels(capacitor_id, total_cycles=200):
    """
    Primary Model用のラベルとSecondary Model用のRULを生成
    
    Args:
        capacitor_id: コンデンサID
        total_cycles: 総サイクル数
    
    Returns:
        labels_df: ラベルDataFrame
    """
    labels = []
    
    for cycle in range(1, total_cycles + 1):
        # Primary: 異常検知ラベル
        # 前半50%を正常、後半50%を異常とする
        is_abnormal = 1 if cycle > total_cycles * 0.5 else 0
        
        # Secondary: RUL
        rul = total_cycles - cycle
        
        labels.append({
            'capacitor_id': capacitor_id,
            'cycle': cycle,
            'is_abnormal': is_abnormal,
            'rul': rul
        })
    
    return pd.DataFrame(labels)
```

## 📊 モデル選択

### Primary Model (異常検知)

**候補アルゴリズム**:
1. **Random Forest Classifier**
   - 特徴量重要度が分かる
   - 非線形関係を捉えられる
   - ハイパーパラメータ調整が比較的容易

2. **XGBoost / LightGBM**
   - 高精度
   - 特徴量重要度
   - 不均衡データに対応可能

3. **Logistic Regression**
   - ベースライン
   - 解釈性が高い

### Secondary Model (RUL予測)

**候補アルゴリズム**:
1. **Random Forest Regressor**
   - 非線形関係
   - ロバスト

2. **XGBoost / LightGBM Regressor**
   - 高精度
   - 特徴量重要度

3. **LSTM / GRU**
   - 時系列パターンを学習
   - 長期依存関係を捉えられる

4. **Linear Regression**
   - ベースライン
   - 解釈性

## 🎯 評価指標

### Primary Model
- **Accuracy**: 全体的な精度
- **Precision / Recall**: 異常検知の精度
- **F1-Score**: バランス指標
- **ROC-AUC**: 閾値に依存しない評価

### Secondary Model
- **MAE (Mean Absolute Error)**: 平均絶対誤差
- **RMSE (Root Mean Squared Error)**: 二乗平均平方根誤差
- **R² Score**: 決定係数
- **MAPE (Mean Absolute Percentage Error)**: 平均絶対パーセント誤差

## 🚀 次のステップ

1. **特徴量抽出スクリプト作成**
   - 全コンデンサ・全サイクルから特徴量を抽出
   - CSV形式で保存

2. **ラベル生成**
   - Primary Model用の異常ラベル
   - Secondary Model用のRUL

3. **データセット構築**
   - Train/Val/Test分割
   - 特徴量スケーリング

4. **ベースラインモデル構築**
   - Random Forest で初期モデル
   - 性能評価

5. **モデル改善**
   - ハイパーパラメータチューニング
   - 特徴量エンジニアリング
   - アンサンブル

---

**作成日**: 2026-01-15
