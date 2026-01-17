# RUL予測モデル - 設計書

## 🏗️ システムアーキテクチャ

### 全体構成

```
┌─────────────────────────────────────────────────────────────┐
│                     RUL Prediction System                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │   ES12 Data  │───▶│   Feature    │───▶│   Dataset    │  │
│  │   (HDF5)     │    │  Extraction  │    │   (CSV)      │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                   │           │
│                                                   ▼           │
│                            ┌──────────────────────────────┐  │
│                            │    Data Splitting            │  │
│                            │  (Train/Val/Test)            │  │
│                            └──────────────────────────────┘  │
│                                      │                        │
│                    ┌─────────────────┼─────────────────┐     │
│                    ▼                 ▼                 ▼     │
│              ┌─────────┐       ┌─────────┐      ┌─────────┐ │
│              │  Train  │       │   Val   │      │  Test   │ │
│              └─────────┘       └─────────┘      └─────────┘ │
│                    │                 │                 │     │
│                    ▼                 ▼                 │     │
│         ┌──────────────────────────────────┐          │     │
│         │      Primary Model               │          │     │
│         │  (Anomaly Classifier)            │          │     │
│         │  - Random Forest                 │          │     │
│         │  - XGBoost / LightGBM            │          │     │
│         └──────────────────────────────────┘          │     │
│                    │                                   │     │
│                    ▼                                   │     │
│         ┌──────────────────────────────────┐          │     │
│         │     Secondary Model              │          │     │
│         │   (RUL Predictor)                │          │     │
│         │  - Random Forest Regressor       │          │     │
│         │  - XGBoost / LightGBM            │          │     │
│         │  - LSTM / GRU                    │          │     │
│         └──────────────────────────────────┘          │     │
│                    │                                   │     │
│                    └───────────────────────────────────┘     │
│                                    ▼                          │
│                         ┌──────────────────┐                 │
│                         │    Evaluation    │                 │
│                         │    & Report      │                 │
│                         └──────────────────┘                 │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## 📦 モジュール設計

### 1. Feature Extraction Module

**責務**: ES12データから特徴量を抽出

```python
# src/feature_extraction/extractor.py

class CycleFeatureExtractor:
    """サイクルレベルの特徴量を抽出"""
    
    def extract_basic_stats(self, vl: np.ndarray, vo: np.ndarray) -> dict:
        """基本統計量を抽出"""
        pass
    
    def extract_degradation_indicators(self, vl: np.ndarray, vo: np.ndarray) -> dict:
        """劣化指標を抽出"""
        pass
    
    def extract_time_series_features(self, vl: np.ndarray, vo: np.ndarray) -> dict:
        """時系列特徴を抽出"""
        pass
    
    def extract_historical_features(self, history_df: pd.DataFrame, window: int = 5) -> dict:
        """履歴特徴量を抽出"""
        pass
    
    def extract_all_features(self, vl: np.ndarray, vo: np.ndarray, 
                            cycle_num: int, history_df: pd.DataFrame = None) -> dict:
        """全特徴量を抽出"""
        pass
```

**入力**:
- VL時系列データ（numpy array, shape: (n_points,)）
- VO時系列データ（numpy array, shape: (n_points,)）
- サイクル番号（int）
- 履歴データ（pandas DataFrame, optional）

**出力**:
- 特徴量辞書（dict）: 20-30個の特徴量

### 2. Data Preparation Module

**責務**: データセットの構築とラベル生成

```python
# src/data_preparation/dataset_builder.py

class DatasetBuilder:
    """データセットを構築"""
    
    def __init__(self, es12_path: str):
        self.es12_path = es12_path
        self.feature_extractor = CycleFeatureExtractor()
    
    def build_feature_dataset(self, capacitor_ids: list) -> pd.DataFrame:
        """特徴量データセットを構築"""
        pass
    
    def generate_labels(self, df: pd.DataFrame, strategy: str = 'cycle_based') -> pd.DataFrame:
        """ラベルを生成"""
        pass
    
    def split_dataset(self, df: pd.DataFrame, 
                     train_caps: list, val_caps: list, test_caps: list,
                     train_cycles: tuple, val_cycles: tuple, test_cycles: tuple) -> tuple:
        """データセットを分割"""
        pass
```

**ラベリング戦略**:

1. **Cycle-based Strategy**（推奨）:
   ```python
   # 前半50%を正常、後半50%を異常
   is_abnormal = 1 if cycle > total_cycles * 0.5 else 0
   ```

2. **Threshold-based Strategy**:
   ```python
   # 電圧比が初期値から20%以上変化したら異常
   initial_ratio = df[df['cycle'] <= 10]['voltage_ratio'].mean()
   is_abnormal = 1 if abs(voltage_ratio - initial_ratio) / initial_ratio > 0.2 else 0
   ```

### 3. Model Module

**責務**: モデルの定義と学習

```python
# src/models/primary_model.py

class PrimaryModel:
    """Primary Model: 異常検知"""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
             X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """モデルを学習"""
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """予測を実行"""
        pass
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """予測確率を取得"""
        pass
    
    def save(self, path: str):
        """モデルを保存"""
        pass
    
    def load(self, path: str):
        """モデルを読み込み"""
        pass
```

```python
# src/models/secondary_model.py

class SecondaryModel:
    """Secondary Model: RUL予測"""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
             X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """モデルを学習"""
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """RULを予測"""
        pass
    
    def save(self, path: str):
        """モデルを保存"""
        pass
    
    def load(self, path: str):
        """モデルを読み込み"""
        pass
```

### 4. Evaluation Module

**責務**: モデルの評価とレポート生成

```python
# src/evaluation/evaluator.py

class ModelEvaluator:
    """モデル評価"""
    
    def evaluate_primary_model(self, model: PrimaryModel, 
                              X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Primary Modelを評価"""
        # Accuracy, Precision, Recall, F1-Score, ROC-AUC
        pass
    
    def evaluate_secondary_model(self, model: SecondaryModel,
                                X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """Secondary Modelを評価"""
        # MAE, RMSE, R², MAPE
        pass
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             save_path: str):
        """混同行列を可視化"""
        pass
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                      save_path: str):
        """ROC曲線を可視化"""
        pass
    
    def plot_prediction_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                 save_path: str):
        """予測値 vs 実測値を可視化"""
        pass
    
    def generate_report(self, primary_metrics: dict, secondary_metrics: dict,
                       output_path: str):
        """評価レポートを生成"""
        pass
```

## 🔧 データフロー

### Phase 1: Feature Extraction

```
ES12.mat (HDF5)
    │
    ├─ ES12C1 (200 cycles)
    │   ├─ Cycle 1: VL[3000], VO[3000] → Features[20-30]
    │   ├─ Cycle 2: VL[3000], VO[3000] → Features[20-30]
    │   └─ ...
    │
    ├─ ES12C2 (200 cycles)
    └─ ...
    
    ↓
    
features.csv
┌──────────┬───────┬──────────┬──────────┬─────────────────┬─────┐
│ cap_id   │ cycle │ vl_mean  │ vo_mean  │ voltage_ratio   │ ... │
├──────────┼───────┼──────────┼──────────┼─────────────────┼─────┤
│ ES12C1   │ 1     │ 5.234    │ 4.123    │ 0.787           │ ... │
│ ES12C1   │ 2     │ 5.241    │ 4.098    │ 0.782           │ ... │
│ ...      │ ...   │ ...      │ ...      │ ...             │ ... │
└──────────┴───────┴──────────┴──────────┴─────────────────┴─────┘
```

### Phase 2: Label Generation

```
features.csv + Labeling Strategy
    ↓
features_with_labels.csv
┌──────────┬───────┬─────────────┬─────┬─────┐
│ cap_id   │ cycle │ voltage_... │ ... │ RUL │
├──────────┼───────┼─────────────┼─────┼─────┤
│ ES12C1   │ 1     │ 0.787       │ ... │ 199 │
│ ES12C1   │ 2     │ 0.782       │ ... │ 198 │
│ ...      │ ...   │ ...         │ ... │ ... │
└──────────┴───────┴─────────────┴─────┴─────┘
```

### Phase 3: Data Splitting

```
features_with_labels.csv
    │
    ├─ Train: C1-C5, Cycles 1-150 (750 samples)
    ├─ Val:   C6, Cycles 1-150 (150 samples)
    └─ Test:  C7-C8, Cycles 1-200 (400 samples)
```

### Phase 4: Model Training & Evaluation

```
Train Data → Primary Model → Predictions → Evaluation Metrics
                                              ├─ Accuracy
                                              ├─ F1-Score
                                              └─ ROC-AUC

Train Data → Secondary Model → RUL Predictions → Evaluation Metrics
                                                   ├─ MAE
                                                   ├─ RMSE
                                                   └─ R²
```

## 📊 特徴量詳細設計

### 特徴量リスト（合計26特徴量）

#### 基本統計量（14特徴量）
```python
# VL (Input) - 7特徴量
'vl_mean', 'vl_std', 'vl_min', 'vl_max', 'vl_range', 'vl_median', 'vl_cv'

# VO (Output) - 7特徴量
'vo_mean', 'vo_std', 'vo_min', 'vo_max', 'vo_range', 'vo_median', 'vo_cv'
```

#### 劣化指標（4特徴量）
```python
'voltage_ratio',           # vo_mean / vl_mean
'voltage_ratio_std',       # std(VO/VL)
'response_efficiency',     # vo_range / vl_range
'signal_attenuation'       # 1 - (vo_std / vl_std)
```

#### 時系列特徴（2特徴量）
```python
'vl_trend',  # 線形回帰の傾き
'vo_trend'   # 線形回帰の傾き
```

#### サイクル情報（2特徴量）
```python
'cycle_number',       # サイクル番号
'cycle_normalized'    # サイクル番号 / 200
```

#### 履歴特徴（4特徴量）
```python
'voltage_ratio_mean_last_5',   # 過去5サイクルの平均電圧比
'voltage_ratio_std_last_5',    # 過去5サイクルの電圧比の標準偏差
'voltage_ratio_trend_last_10', # 過去10サイクルの電圧比のトレンド
'degradation_rate'             # (current_ratio - initial_ratio) / cycle_number
```

## 🎯 モデルハイパーパラメータ

### Random Forest Classifier (Primary Model)

```python
{
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42
}
```

### Random Forest Regressor (Secondary Model)

```python
{
    'n_estimators': 100,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42
}
```

## 📁 ファイル構成

```
rul_modeling/
├── src/
│   ├── __init__.py
│   ├── feature_extraction/
│   │   ├── __init__.py
│   │   └── extractor.py          # CycleFeatureExtractor
│   ├── data_preparation/
│   │   ├── __init__.py
│   │   ├── dataset_builder.py    # DatasetBuilder
│   │   └── label_generator.py    # LabelGenerator
│   ├── models/
│   │   ├── __init__.py
│   │   ├── primary_model.py      # PrimaryModel
│   │   └── secondary_model.py    # SecondaryModel
│   └── evaluation/
│       ├── __init__.py
│       └── evaluator.py          # ModelEvaluator
├── tests/
│   ├── test_feature_extraction.py
│   ├── test_data_preparation.py
│   ├── test_models.py
│   └── test_evaluation.py
├── notebooks/
│   ├── 01_feature_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_model_evaluation.ipynb
└── output/
    ├── features/
    │   ├── features.csv
    │   └── features_with_labels.csv
    ├── models/
    │   ├── primary_model.pkl
    │   └── secondary_model.pkl
    └── evaluation/
        ├── primary_model_report.md
        ├── secondary_model_report.md
        └── figures/
```

## 🔄 実装順序

### Step 1: Feature Extraction
1. `CycleFeatureExtractor`の実装
2. 単一サイクルからの特徴量抽出テスト
3. 全コンデンサからの特徴量抽出

### Step 2: Data Preparation
1. `DatasetBuilder`の実装
2. ラベル生成ロジックの実装
3. データ分割の実装

### Step 3: Primary Model
1. `PrimaryModel`の実装
2. Random Forestでの学習
3. 評価指標の計算

### Step 4: Secondary Model
1. `SecondaryModel`の実装
2. Random Forestでの学習
3. 評価指標の計算

### Step 5: Evaluation
1. `ModelEvaluator`の実装
2. 可視化機能の実装
3. レポート生成機能の実装

---

**作成日**: 2026-01-15
**最終更新**: 2026-01-15
