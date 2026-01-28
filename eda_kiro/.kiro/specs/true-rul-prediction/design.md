# Design Document: True RUL Prediction System

## Overview

This design document describes a comprehensive system for predicting the Remaining Useful Life (RUL) of capacitors using the NASA PCOE ES12 dataset. The system addresses three critical improvements over the existing approach:

1. **True RUL Prediction**: Implements regression models to predict remaining cycle counts (not just binary classification)
2. **High-Precision Anomaly Detection**: Reduces FPR from 13.1% to below 5% using ensemble methods and time-series pattern recognition
3. **Staged Degradation Prediction**: Provides continuous degradation progression with confidence intervals

The system uses a hybrid architecture combining:
- Deep learning time-series models (LSTM/GRU/Transformer) for temporal pattern recognition
- Advanced feature engineering from voltage time-series data
- Ensemble methods for robust predictions
- Semi-supervised learning leveraging early cycles as normal baseline

## Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                     RUL Prediction System                    │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐                     │
│  │ Data Loader  │─────▶│   Feature    │                     │
│  │              │      │  Extractor   │                     │
│  └──────────────┘      └──────┬───────┘                     │
│                               │                              │
│                               ▼                              │
│                    ┌──────────────────┐                     │
│                    │  Time-Series     │                     │
│                    │  Preprocessor    │                     │
│                    └────────┬─────────┘                     │
│                             │                               │
│              ┌──────────────┴──────────────┐               │
│              ▼                              ▼               │
│    ┌─────────────────┐          ┌─────────────────┐       │
│    │  RUL Regression │          │    Anomaly      │       │
│    │     Model       │          │   Detection     │       │
│    │  (LSTM/GRU/     │          │   Model         │       │
│    │  Transformer)   │          │  (Ensemble)     │       │
│    └────────┬────────┘          └────────┬────────┘       │
│             │                            │                 │
│             └──────────┬─────────────────┘                 │
│                        ▼                                    │
│              ┌──────────────────┐                          │
│              │   Prediction     │                          │
│              │   Aggregator     │                          │
│              └────────┬─────────┘                          │
│                       │                                     │
│                       ▼                                     │
│              ┌──────────────────┐                          │
│              │  Confidence      │                          │
│              │  Estimator       │                          │
│              └────────┬─────────┘                          │
│                       │                                     │
│                       ▼                                     │
│              ┌──────────────────┐                          │
│              │  Output          │                          │
│              │  Formatter       │                          │
│              └──────────────────┘                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input**: Raw voltage time-series data (VL, VO) for each cycle
2. **Feature Extraction**: Extract 15 existing responsiveness features + new time-series features
3. **Preprocessing**: Normalize, create sequences, handle variable lengths
4. **Parallel Processing**:
   - RUL Regression Model predicts remaining cycles
   - Anomaly Detection Model identifies abnormal patterns
5. **Aggregation**: Combine predictions from multiple models
6. **Confidence Estimation**: Calculate prediction intervals using ensemble variance
7. **Output**: Structured prediction with RUL, confidence interval, degradation score, anomaly flag

## Components and Interfaces

### 1. Data Loader

**Purpose**: Load and parse ES12 dataset files

**Interface**:
```python
class DataLoader:
    def load_es12_dataset(self, data_path: str) -> Dict[str, CapacitorData]:
        """
        Load ES12 dataset from disk
        
        Args:
            data_path: Path to ES12 dataset directory
            
        Returns:
            Dictionary mapping capacitor_id to CapacitorData object
        """
        
    def get_capacitor_cycles(self, capacitor_id: str) -> List[CycleData]:
        """
        Get all cycles for a specific capacitor
        
        Args:
            capacitor_id: Identifier for the capacitor (e.g., "C1", "C2", ...)
            
        Returns:
            List of CycleData objects ordered by cycle number
        """
```

**Data Structures**:
```python
@dataclass
class CycleData:
    cycle_number: int
    vl_series: np.ndarray  # Input voltage time-series
    vo_series: np.ndarray  # Output voltage time-series
    timestamp: float
    
@dataclass
class CapacitorData:
    capacitor_id: str
    cycles: List[CycleData]
    total_cycles: int
```

### 2. Feature Extractor

**Purpose**: Extract features from raw voltage time-series data

**Interface**:
```python
class FeatureExtractor:
    def extract_responsiveness_features(self, cycle: CycleData) -> np.ndarray:
        """
        Extract 15 existing responsiveness features
        
        Args:
            cycle: CycleData object containing voltage time-series
            
        Returns:
            Array of 15 responsiveness features
        """
        
    def extract_statistical_features(self, cycle: CycleData) -> np.ndarray:
        """
        Extract statistical features (mean, std, skewness, kurtosis, etc.)
        
        Args:
            cycle: CycleData object
            
        Returns:
            Array of statistical features
        """
        
    def extract_frequency_features(self, cycle: CycleData) -> np.ndarray:
        """
        Extract frequency domain features using FFT
        
        Args:
            cycle: CycleData object
            
        Returns:
            Array of frequency domain features
        """
        
    def extract_rolling_features(self, cycles: List[CycleData], window: int) -> np.ndarray:
        """
        Extract rolling window statistics across multiple cycles
        
        Args:
            cycles: List of consecutive CycleData objects
            window: Rolling window size
            
        Returns:
            Array of rolling statistics features
        """
        
    def extract_all_features(self, cycle: CycleData, history: List[CycleData]) -> np.ndarray:
        """
        Extract all features for a single cycle
        
        Args:
            cycle: Current cycle data
            history: Previous cycles for rolling features
            
        Returns:
            Concatenated feature vector
        """
```

**Feature Categories**:
- **Responsiveness Features (15)**: Existing features from voltage response characteristics
- **Statistical Features (12)**: Mean, std, min, max, median, skewness, kurtosis for VL and VO
- **Frequency Features (10)**: FFT-based features, dominant frequencies, spectral energy
- **Trend Features (8)**: Linear trend slope, acceleration, deceleration indicators
- **Rolling Features (10)**: Rolling mean, std, min, max over last N cycles

**Total Features**: ~55 features per cycle

### 3. Time-Series Preprocessor

**Purpose**: Prepare time-series data and create temporal features

**Interface**:
```python
class TimeSeriesPreprocessor:
    def __init__(self, rolling_window: int = 5, normalization: str = "standard"):
        self.rolling_window = rolling_window
        self.normalization = normalization
        self.scalers = {}
        
    def create_temporal_features(self, cycles: List[CycleData], features: np.ndarray) -> np.ndarray:
        """
        Create temporal features from cycle history
        
        Args:
            cycles: List of CycleData objects (ordered by cycle number)
            features: Extracted features for each cycle (n_cycles, n_features)
            
        Returns:
            Enhanced feature array with temporal features
        """
        temporal_features = []
        
        for i in range(len(cycles)):
            # Get rolling window of previous cycles
            window_start = max(0, i - self.rolling_window + 1)
            window_features = features[window_start:i+1]
            
            # Compute rolling statistics
            rolling_mean = np.mean(window_features, axis=0)
            rolling_std = np.std(window_features, axis=0)
            rolling_min = np.min(window_features, axis=0)
            rolling_max = np.max(window_features, axis=0)
            
            # Compute trend features (if enough history)
            if i >= 2:
                recent_trend = features[i] - features[i-1]
                long_trend = features[i] - features[max(0, i-5)]
            else:
                recent_trend = np.zeros_like(features[i])
                long_trend = np.zeros_like(features[i])
            
            # Concatenate all temporal features
            temp_feat = np.concatenate([
                rolling_mean, rolling_std, rolling_min, rolling_max,
                recent_trend, long_trend
            ])
            temporal_features.append(temp_feat)
            
        return np.array(temporal_features)
        
    def normalize_features(self, features: np.ndarray, capacitor_id: str, fit: bool = False) -> np.ndarray:
        """
        Normalize features using capacitor-specific statistics
        
        Args:
            features: Feature array to normalize
            capacitor_id: Capacitor identifier for scaler lookup
            fit: Whether to fit the scaler (training) or use existing (inference)
            
        Returns:
            Normalized feature array
        """
        if fit:
            if self.normalization == "standard":
                scaler = StandardScaler()
            elif self.normalization == "minmax":
                scaler = MinMaxScaler()
            else:
                raise ValueError(f"Unknown normalization: {self.normalization}")
            
            self.scalers[capacitor_id] = scaler.fit(features)
            
        if capacitor_id not in self.scalers:
            # Fallback to global scaler
            capacitor_id = "global"
            if capacitor_id not in self.scalers:
                raise ValueError("No scaler available for normalization")
        
        return self.scalers[capacitor_id].transform(features)
```

### 4. RUL Regression Model

**Purpose**: Predict remaining cycle count using interpretable machine learning

**Architecture Options**:

**Option A: Gradient Boosting (XGBoost/LightGBM) - RECOMMENDED**
```python
class GradientBoostingRULPredictor:
    def __init__(self, model_type: str = "xgboost"):
        """
        Gradient boosting model with high interpretability
        
        Advantages:
        - Native feature importance (gain, split, cover)
        - SHAP values for detailed explanations
        - Handles non-linear relationships well
        - Fast training and inference
        - No need for extensive feature scaling
        """
        if model_type == "xgboost":
            self.model = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='reg:squarederror',
                random_state=42
            )
        elif model_type == "lightgbm":
            self.model = lgb.LGBMRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        self.shap_explainer = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray):
        """Train with early stopping"""
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        # Initialize SHAP explainer after training
        self.shap_explainer = shap.TreeExplainer(self.model)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict RUL"""
        return self.model.predict(X)
        
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))
        
    def get_shap_values(self, X: np.ndarray) -> np.ndarray:
        """Get SHAP values for interpretability"""
        return self.shap_explainer.shap_values(X)
```

**Option B: Random Forest with Quantile Regression**
```python
class RandomForestRULPredictor:
    def __init__(self, n_estimators: int = 500):
        """
        Random Forest for RUL prediction with confidence intervals
        
        Advantages:
        - Built-in feature importance
        - Quantile regression for confidence intervals
        - Robust to outliers
        - Easy to interpret individual trees
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.quantile_models = {
            'lower': RandomForestQuantileRegressor(q=0.025),
            'upper': RandomForestQuantileRegressor(q=0.975)
        }
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train main model and quantile models"""
        self.model.fit(X_train, y_train)
        self.quantile_models['lower'].fit(X_train, y_train)
        self.quantile_models['upper'].fit(X_train, y_train)
        
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict with confidence intervals"""
        pred = self.model.predict(X)
        lower = self.quantile_models['lower'].predict(X)
        upper = self.quantile_models['upper'].predict(X)
        return pred, lower, upper
```

**Option C: Elastic Net with Polynomial Features**
```python
class ElasticNetRULPredictor:
    def __init__(self, degree: int = 2):
        """
        Linear model with polynomial features for interpretability
        
        Advantages:
        - Fully interpretable coefficients
        - Fast training and inference
        - Regularization prevents overfitting
        - Easy to understand feature contributions
        """
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.scaler = StandardScaler()
        self.model = ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train with polynomial features"""
        X_poly = self.poly.fit_transform(X_train)
        X_scaled = self.scaler.fit_transform(X_poly)
        self.model.fit(X_scaled, y_train)
        
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict RUL"""
        X_poly = self.poly.transform(X)
        X_scaled = self.scaler.transform(X_poly)
        return self.model.predict(X_scaled)
        
    def get_feature_coefficients(self) -> Dict[str, float]:
        """Get model coefficients for interpretation"""
        feature_names = self.poly.get_feature_names_out()
        return dict(zip(feature_names, self.model.coef_))
```

**Option D: Hybrid Ensemble (Recommended for Best Performance)**
```python
class HybridEnsembleRULPredictor:
    def __init__(self):
        """
        Ensemble combining interpretable models
        
        Advantages:
        - Combines strengths of multiple models
        - Ensemble variance for confidence intervals
        - Multiple interpretability methods
        - Robust predictions
        """
        self.models = {
            'xgboost': GradientBoostingRULPredictor('xgboost'),
            'lightgbm': GradientBoostingRULPredictor('lightgbm'),
            'rf': RandomForestRULPredictor(n_estimators=300)
        }
        self.weights = {'xgboost': 0.4, 'lightgbm': 0.4, 'rf': 0.2}
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray, y_val: np.ndarray):
        """Train all models"""
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.train(X_train, y_train, X_val, y_val)
            
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[float, float, float]:
        """Predict with ensemble confidence"""
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred * self.weights[name])
            
        ensemble_pred = np.sum(predictions, axis=0)
        ensemble_std = np.std([m.predict(X) for m in self.models.values()], axis=0)
        
        lower = ensemble_pred - 1.96 * ensemble_std
        upper = ensemble_pred + 1.96 * ensemble_std
        
        return ensemble_pred, lower, upper
        
    def get_aggregated_feature_importance(self, X: np.ndarray) -> Dict[str, float]:
        """Get feature importance aggregated across models"""
        importance_scores = {}
        for name, model in self.models.items():
            if hasattr(model, 'get_feature_importance'):
                scores = model.get_feature_importance()
                for feature, score in scores.items():
                    importance_scores[feature] = importance_scores.get(feature, 0) + score * self.weights[name]
        return importance_scores
```

**Unified Interface**:
```python
class RULRegressionModel:
    def __init__(self, model_type: str = "xgboost", **kwargs):
        """
        Initialize RUL regression model
        
        Args:
            model_type: One of "xgboost", "lightgbm", "random_forest", "elastic_net", "ensemble"
        """
        self.model_type = model_type
        self.model = self._build_model(model_type, **kwargs)
        
    def _build_model(self, model_type: str, **kwargs):
        """Build the specified model type"""
        if model_type == "xgboost":
            return GradientBoostingRULPredictor("xgboost")
        elif model_type == "lightgbm":
            return GradientBoostingRULPredictor("lightgbm")
        elif model_type == "random_forest":
            return RandomForestRULPredictor(**kwargs)
        elif model_type == "elastic_net":
            return ElasticNetRULPredictor(**kwargs)
        elif model_type == "ensemble":
            return HybridEnsembleRULPredictor()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray):
        """
        Train the RUL regression model
        
        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training RUL labels (n_samples,)
            X_val: Validation features
            y_val: Validation RUL labels
        """
        self.model.train(X_train, y_train, X_val, y_val)
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict RUL with confidence intervals
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        if hasattr(self.model, 'predict_with_confidence'):
            return self.model.predict_with_confidence(X)
        else:
            pred = self.model.predict(X)
            # Fallback: use simple std-based confidence
            uncertainty = np.std(pred) * 1.96
            return pred, pred - uncertainty, pred + uncertainty
        
    def get_feature_importance(self, X: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Get feature importance for interpretability
        
        Args:
            X: Optional input for SHAP-based importance
            
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if hasattr(self.model, 'get_aggregated_feature_importance'):
            return self.model.get_aggregated_feature_importance(X)
        elif hasattr(self.model, 'get_feature_importance'):
            return self.model.get_feature_importance()
        else:
            return {}
            
    def get_shap_values(self, X: np.ndarray) -> np.ndarray:
        """
        Get SHAP values for detailed explanations
        
        Args:
            X: Input features
            
        Returns:
            SHAP values array
        """
        if hasattr(self.model, 'get_shap_values'):
            return self.model.get_shap_values(X)
        else:
            raise NotImplementedError(f"SHAP values not available for {self.model_type}")
```

**Recommended Configuration**:
For maximum interpretability with high performance, use the Hybrid Ensemble approach:
- XGBoost (40% weight): Best overall performance, native feature importance
- LightGBM (40% weight): Fast training, handles large feature sets well
- Random Forest (20% weight): Provides quantile-based confidence intervals

This combination provides:
1. **Multiple interpretability methods**: Feature importance, SHAP values, partial dependence plots
2. **Robust predictions**: Ensemble reduces variance and overfitting
3. **Confidence intervals**: From ensemble variance and quantile regression
4. **Fast inference**: All models are tree-based with O(log n) prediction time

### 5. Anomaly Detection Model

**Purpose**: Detect anomalous behavior with FPR < 5%

**Architecture**: Ensemble of multiple detectors

**Components**:

**A. Isolation Forest**
```python
class IsolationForestDetector:
    def __init__(self, contamination: float = 0.05):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        
    def fit(self, normal_data: np.ndarray):
        """Fit on normal cycles (1-10)"""
        
    def predict_score(self, x: np.ndarray) -> np.ndarray:
        """Return anomaly scores"""
```

**B. Autoencoder**
```python
class AutoencoderDetector(nn.Module):
    def __init__(self, input_dim: int, encoding_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, encoding_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
        
    def get_reconstruction_error(self, x: np.ndarray) -> np.ndarray:
        """Return reconstruction error as anomaly score"""
```

**C. One-Class SVM (Improved)**
```python
class ImprovedOCSVM:
    def __init__(self, kernel: str = "rbf", nu: float = 0.05):
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma="auto")
        
    def fit(self, normal_data: np.ndarray):
        """Fit on normal cycles"""
        
    def predict_score(self, x: np.ndarray) -> np.ndarray:
        """Return decision function scores"""
```

**Ensemble Interface**:
```python
class EnsembleAnomalyDetector:
    def __init__(self):
        self.detectors = [
            IsolationForestDetector(contamination=0.05),
            AutoencoderDetector(input_dim=55, encoding_dim=16),
            ImprovedOCSVM(nu=0.05)
        ]
        self.weights = [0.35, 0.40, 0.25]  # Tuned weights
        
    def fit(self, normal_data: np.ndarray):
        """Fit all detectors on normal cycles"""
        for detector in self.detectors:
            detector.fit(normal_data)
            
    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Predict anomaly status
        
        Args:
            x: Input features
            
        Returns:
            Tuple of (binary_predictions, anomaly_scores, feature_importance)
        """
        scores = []
        for detector, weight in zip(self.detectors, self.weights):
            score = detector.predict_score(x)
            scores.append(score * weight)
            
        ensemble_score = np.sum(scores, axis=0)
        binary_pred = (ensemble_score > self.threshold).astype(int)
        
        return binary_pred, ensemble_score, self._compute_feature_importance(x)
```

### 6. Prediction Aggregator

**Purpose**: Combine RUL predictions and anomaly detection results

**Interface**:
```python
class PredictionAggregator:
    def aggregate(self, rul_pred: float, rul_uncertainty: float,
                  anomaly_flag: int, anomaly_score: float,
                  degradation_history: List[float]) -> PredictionResult:
        """
        Aggregate predictions from multiple models
        
        Args:
            rul_pred: RUL prediction from regression model
            rul_uncertainty: Uncertainty estimate
            anomaly_flag: Binary anomaly flag
            anomaly_score: Continuous anomaly score
            degradation_history: Historical degradation scores
            
        Returns:
            PredictionResult object with aggregated information
        """
        
    def compute_degradation_stage(self, rul: float, anomaly_score: float) -> str:
        """
        Compute degradation stage based on RUL and anomaly score
        
        Returns:
            One of: "healthy", "early_degradation", "advanced_degradation", "critical"
        """
```

**Data Structure**:
```python
@dataclass
class PredictionResult:
    rul_cycles: int
    rul_confidence_lower: int
    rul_confidence_upper: int
    degradation_score: float
    degradation_stage: str
    anomaly_flag: bool
    anomaly_score: float
    feature_importance: Dict[str, float]
    attention_weights: Optional[np.ndarray]
    timestamp: float
    model_version: str
```

### 7. Confidence Estimator

**Purpose**: Estimate prediction confidence intervals

**Methods**:

**A. Ensemble Variance**
```python
def estimate_confidence_ensemble(predictions: List[float]) -> Tuple[float, float]:
    """
    Estimate confidence interval from ensemble predictions
    
    Args:
        predictions: List of predictions from different models/bootstraps
        
    Returns:
        Tuple of (lower_bound, upper_bound) for 95% confidence interval
    """
    mean = np.mean(predictions)
    std = np.std(predictions)
    lower = mean - 1.96 * std
    upper = mean + 1.96 * std
    return lower, upper
```

**B. Monte Carlo Dropout**
```python
def estimate_confidence_mcdropout(model: nn.Module, x: torch.Tensor, 
                                  n_samples: int = 100) -> Tuple[float, float]:
    """
    Estimate confidence using Monte Carlo Dropout
    
    Args:
        model: Neural network model with dropout layers
        x: Input tensor
        n_samples: Number of forward passes
        
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    model.train()  # Enable dropout
    predictions = []
    for _ in range(n_samples):
        pred = model(x).item()
        predictions.append(pred)
    model.eval()
    
    return estimate_confidence_ensemble(predictions)
```

**Interface**:
```python
class ConfidenceEstimator:
    def __init__(self, method: str = "ensemble"):
        self.method = method
        
    def estimate(self, model: Any, x: np.ndarray, 
                 n_samples: int = 100) -> Tuple[float, float]:
        """
        Estimate confidence interval for prediction
        
        Args:
            model: Trained model
            x: Input data
            n_samples: Number of samples for estimation
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
```

## Data Models

### Training Data Structure

```python
@dataclass
class TrainingDataset:
    """Dataset for model training"""
    capacitor_ids: List[str]
    sequences: np.ndarray  # Shape: (n_samples, sequence_length, n_features)
    rul_labels: np.ndarray  # Shape: (n_samples,)
    cycle_numbers: np.ndarray  # Shape: (n_samples,)
    anomaly_labels: Optional[np.ndarray]  # Shape: (n_samples,) - for validation only
    
    def split_by_capacitor(self, test_capacitors: List[str]) -> Tuple['TrainingDataset', 'TrainingDataset']:
        """Split dataset by capacitor for cross-validation"""
        
    def get_normal_cycles(self, max_cycle: int = 10) -> np.ndarray:
        """Get features from early cycles assumed to be normal"""
```

### Model Configuration

```python
@dataclass
class ModelConfig:
    """Configuration for RUL prediction models"""
    # RUL Model
    rul_model_type: str = "lstm"  # "lstm", "gru", "transformer"
    sequence_length: int = 20
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.2
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    
    # Anomaly Detection
    ensemble_weights: List[float] = field(default_factory=lambda: [0.35, 0.40, 0.25])
    anomaly_threshold: float = 0.5
    contamination: float = 0.05
    
    # Feature Extraction
    n_responsiveness_features: int = 15
    n_statistical_features: int = 12
    n_frequency_features: int = 10
    n_trend_features: int = 8
    n_rolling_features: int = 10
    rolling_window: int = 5
    
    # Training
    validation_split: float = 0.2
    early_stopping_patience: int = 15
    normal_cycle_range: Tuple[int, int] = (1, 10)
```

### Prediction Output

```python
@dataclass
class RULPredictionOutput:
    """Complete output from RUL prediction system"""
    capacitor_id: str
    current_cycle: int
    
    # RUL Prediction
    predicted_rul: int
    rul_confidence_interval: Tuple[int, int]
    rul_uncertainty: float
    
    # Degradation Assessment
    degradation_score: float  # 0-1 scale
    degradation_stage: str  # "healthy", "early_degradation", "advanced_degradation", "critical"
    degradation_trend: float  # Rate of degradation change
    
    # Anomaly Detection
    is_anomalous: bool
    anomaly_score: float
    anomaly_confidence: float
    
    # Interpretability
    feature_importance: Dict[str, float]
    attention_weights: Optional[np.ndarray]
    contributing_features: List[str]
    
    # Metadata
    timestamp: datetime
    model_version: str
    prediction_latency_ms: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        
    def to_json(self) -> str:
        """Convert to JSON string"""
```

## Correctness Properties


*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property Reflection

After analyzing all acceptance criteria, I identified the following redundancies:
- Properties 1.3 and 7.2 both check output format completeness - can be combined
- Properties 2.2 and 7.2 overlap on output structure - will consolidate
- Properties 9.1, 9.2, 9.3 all check interpretability outputs - can be combined into comprehensive property
- Properties 7.5 and 10.3 both check metadata/logging - will merge

### Core Properties

**Property 1: Non-negative RUL Output**
*For any* valid voltage time-series input, the RUL_Predictor should output a remaining cycle count that is a non-negative integer.
**Validates: Requirements 1.1**

**Property 2: Complete Prediction Output Structure**
*For any* prediction request, the RUL_Predictor should return a structured output containing: RUL value, confidence interval (lower <= prediction <= upper), degradation score (0-1 range), anomaly flag (boolean), and all required metadata (timestamp, model version).
**Validates: Requirements 1.3, 2.2, 7.2, 7.5**

**Property 3: Feature Importance for Anomalies**
*For any* sample classified as anomalous, the Anomaly_Detector should provide feature importance information identifying which features contributed to the classification.
**Validates: Requirements 2.5**

**Property 4: Responsiveness Feature Count**
*For any* cycle data input, the Feature_Extractor should extract exactly 15 responsiveness features.
**Validates: Requirements 3.1**

**Property 5: Multi-Category Feature Extraction**
*For any* cycle data input, the Feature_Extractor should generate features from all required categories: statistical moments, trend indicators, frequency domain features, and rolling window statistics.
**Validates: Requirements 3.2, 3.3**

**Property 6: Feature Normalization Consistency**
*For any* set of extracted features after normalization, the features should have consistent statistical properties appropriate to the normalization method (e.g., for standard normalization: mean ≈ 0 within tolerance, std ≈ 1 within tolerance).
**Validates: Requirements 3.4**

**Property 7: Continuous Degradation Output**
*For any* prediction, the degradation score should be a continuous float value in the range [0, 1], not a binary classification.
**Validates: Requirements 4.1**

**Property 8: Valid Degradation Stage**
*For any* prediction, the degradation stage indicator should be one of the valid stages: "healthy", "early_degradation", "advanced_degradation", or "critical".
**Validates: Requirements 4.2**

**Property 9: Degradation Monotonicity**
*For any* sequence of consecutive cycles from the same capacitor, the degradation scores should be non-decreasing (later cycles should have degradation scores >= earlier cycles).
**Validates: Requirements 4.4**

**Property 10: Complete Evaluation Metrics**
*For any* model evaluation run, the output should include all required performance metrics: RMSE, MAE, FPR, and R² score.
**Validates: Requirements 5.3**

**Property 11: Variable-Length Sequence Handling**
*For any* set of input sequences with different lengths, the Time_Series_Model should successfully process all sequences and produce valid outputs.
**Validates: Requirements 6.2**

**Property 12: Temporal Order Preservation**
*For any* input sequence with timestamps or ordered indices, the Time_Series_Model should preserve the temporal ordering in its internal representation (no reordering of time steps).
**Validates: Requirements 6.3**

**Property 13: Real-Time Prediction Latency**
*For any* single prediction request, the RUL_Predictor should return results within 1000 milliseconds.
**Validates: Requirements 7.1**

**Property 14: Low Confidence Flagging**
*For any* prediction where the confidence interval width exceeds a threshold (e.g., > 50 cycles), the RUL_Predictor should flag the prediction as uncertain.
**Validates: Requirements 7.3**

**Property 15: Out-of-Distribution Detection**
*For any* input that is significantly different from the ES12 training distribution (e.g., voltage values outside training range by >3 standard deviations), the RUL_Predictor should flag it as out-of-distribution.
**Validates: Requirements 8.4**

**Property 16: Comprehensive Interpretability Output**
*For any* prediction, the RUL_Predictor should provide complete interpretability information including: feature importance scores (summing to 1.0), attention weights (if applicable), and for anomalous predictions, the specific contributing features.
**Validates: Requirements 9.1, 9.2, 9.3, 9.4**

**Property 17: Deviation Diagnostic Reports**
*For any* prediction where the RUL deviates significantly from the expected range (e.g., > 2 standard deviations from historical mean), the RUL_Predictor should generate a diagnostic report.
**Validates: Requirements 9.5**

**Property 18: Prediction Logging**
*For any* prediction made, the RUL_Predictor should generate a log entry containing the input summary, prediction output, and performance metrics.
**Validates: Requirements 10.3**

**Property 19: Batch Processing Correctness**
*For any* batch of N input samples, the RUL_Predictor should produce exactly N prediction outputs, each corresponding to the correct input sample.
**Validates: Requirements 10.4**

### Example-Based Tests

**Example 1: ES12 Dataset FPR Performance**
Test that the Anomaly_Detector achieves FPR < 5% on the complete ES12 dataset with known ground truth labels.
**Validates: Requirements 2.1**

**Example 2: Degradation Stage Transition Detection**
Test that the RUL_Predictor detects transitions between degradation stages within 5 cycles using synthetic data with known transition points.
**Validates: Requirements 4.3**

**Example 3: Model Retraining Trigger**
Test that when model performance metrics degrade below thresholds (e.g., RMSE increases by >20%), the system triggers a retraining workflow.
**Validates: Requirements 5.5**

**Example 4: ES12 Voltage Range Handling**
Test that the RUL_Predictor successfully processes all voltage ranges present in the ES12 dataset without errors or warnings.
**Validates: Requirements 8.3**

**Example 5: REST API Endpoints**
Test that all required REST API endpoints exist and return proper HTTP status codes and response formats.
**Validates: Requirements 10.1**

**Example 6: Model Loading and Caching**
Test that on system startup, trained models are successfully loaded from persistent storage and cached in memory for fast inference.
**Validates: Requirements 10.2**

**Example 7: Health Check Endpoints**
Test that health check endpoints return proper status (200 OK when healthy, appropriate error codes when unhealthy) and include model readiness information.
**Validates: Requirements 10.5**

## Error Handling

### Error Categories

**1. Input Validation Errors**
- **Missing Data**: When VL or VO time-series data is missing
  - Action: Return error with code `INPUT_MISSING_DATA`
  - Message: "Required voltage time-series data (VL or VO) is missing"
  
- **Invalid Data Format**: When input data has wrong shape or type
  - Action: Return error with code `INPUT_INVALID_FORMAT`
  - Message: "Input data format is invalid. Expected shape: (sequence_length, 2)"
  
- **Out of Range Values**: When voltage values are physically impossible
  - Action: Log warning, flag as out-of-distribution, proceed with prediction
  - Message: "Input voltage values outside expected range"

**2. Model Errors**
- **Model Not Loaded**: When prediction is requested before model initialization
  - Action: Return error with code `MODEL_NOT_READY`
  - Message: "Model not loaded. Please wait for initialization to complete"
  
- **Prediction Failure**: When model inference fails
  - Action: Log error with stack trace, return error with code `PREDICTION_FAILED`
  - Message: "Prediction failed due to internal error"
  - Fallback: Attempt prediction with backup model if available

**3. Feature Extraction Errors**
- **Feature Extraction Failure**: When feature computation fails
  - Action: Log error, attempt to extract partial features
  - Message: "Feature extraction partially failed. Using available features"
  
- **Normalization Error**: When normalization fails due to missing scaler
  - Action: Use global scaler as fallback
  - Message: "Capacitor-specific scaler not found. Using global normalization"

**4. Confidence Estimation Errors**
- **Insufficient Samples**: When confidence estimation has too few samples
  - Action: Return wider confidence interval, flag as uncertain
  - Message: "Confidence estimation based on limited samples"

**5. System Errors**
- **Resource Exhaustion**: When memory or compute resources are insufficient
  - Action: Return error with code `RESOURCE_EXHAUSTED`
  - Message: "Insufficient resources for prediction. Please retry"
  
- **Timeout**: When prediction exceeds time limit
  - Action: Cancel prediction, return error with code `PREDICTION_TIMEOUT`
  - Message: "Prediction exceeded time limit of 1 second"

### Error Handling Strategy

```python
class PredictionError(Exception):
    """Base class for prediction errors"""
    def __init__(self, code: str, message: str, details: Optional[Dict] = None):
        self.code = code
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

class RULPredictor:
    def predict_with_error_handling(self, input_data: np.ndarray) -> Union[PredictionResult, PredictionError]:
        """
        Predict RUL with comprehensive error handling
        
        Returns:
            PredictionResult on success, PredictionError on failure
        """
        try:
            # Input validation
            self._validate_input(input_data)
            
            # Feature extraction with fallback
            try:
                features = self.feature_extractor.extract_all_features(input_data)
            except FeatureExtractionError as e:
                logger.warning(f"Feature extraction error: {e}")
                features = self.feature_extractor.extract_partial_features(input_data)
            
            # Prediction with timeout
            with timeout(seconds=1):
                rul_pred = self.rul_model.predict(features)
                anomaly_result = self.anomaly_detector.predict(features)
            
            # Confidence estimation with fallback
            try:
                confidence = self.confidence_estimator.estimate(self.rul_model, features)
            except InsufficientSamplesError:
                confidence = (rul_pred - 50, rul_pred + 50)  # Wide fallback interval
                
            # Aggregate results
            result = self.aggregator.aggregate(rul_pred, confidence, anomaly_result)
            
            # Log prediction
            self.logger.log_prediction(input_data, result)
            
            return result
            
        except InputValidationError as e:
            return PredictionError(e.code, e.message, {"input_shape": input_data.shape})
        except ModelNotReadyError as e:
            return PredictionError("MODEL_NOT_READY", str(e))
        except TimeoutError:
            return PredictionError("PREDICTION_TIMEOUT", "Prediction exceeded 1 second limit")
        except Exception as e:
            logger.error(f"Unexpected error in prediction: {e}", exc_info=True)
            return PredictionError("PREDICTION_FAILED", "Internal error during prediction")
```

### Graceful Degradation

When components fail, the system should degrade gracefully:

1. **Feature Extraction Failure**: Use subset of successfully extracted features
2. **Single Model Failure in Ensemble**: Use predictions from remaining models with adjusted weights
3. **Confidence Estimation Failure**: Return prediction with wide confidence interval and uncertainty flag
4. **Interpretability Failure**: Return prediction without interpretability information, log warning

## Testing Strategy

### Dual Testing Approach

The system requires both unit testing and property-based testing for comprehensive coverage:

**Unit Tests**: Focus on specific examples, edge cases, and error conditions
- Specific voltage patterns from ES12 dataset
- Edge cases: empty sequences, single-cycle inputs, extreme voltage values
- Error conditions: missing data, invalid formats, model failures
- Integration points: API endpoints, model loading, logging

**Property Tests**: Verify universal properties across all inputs
- Run minimum 100 iterations per property test
- Use randomized input generation for voltage time-series
- Test invariants that must hold for all valid inputs
- Each property test references its design document property

### Property-Based Testing Configuration

**Library Selection**: 
- Python: Use `hypothesis` library for property-based testing
- Test framework: `pytest` with `pytest-hypothesis` plugin

**Test Configuration**:
```python
from hypothesis import given, settings, strategies as st
import pytest

# Configure hypothesis for thorough testing
@settings(max_examples=100, deadline=None)
@given(
    vl_series=st.lists(st.floats(min_value=0, max_value=10), min_size=10, max_size=200),
    vo_series=st.lists(st.floats(min_value=0, max_value=10), min_size=10, max_size=200)
)
def test_property_1_non_negative_rul(vl_series, vo_series):
    """
    Feature: true-rul-prediction, Property 1: Non-negative RUL Output
    
    For any valid voltage time-series input, the RUL_Predictor should output 
    a remaining cycle count that is a non-negative integer.
    """
    # Arrange
    input_data = create_cycle_data(vl_series, vo_series)
    predictor = RULPredictor()
    
    # Act
    result = predictor.predict(input_data)
    
    # Assert
    assert isinstance(result.predicted_rul, int), "RUL should be an integer"
    assert result.predicted_rul >= 0, "RUL should be non-negative"
```

**Tag Format**: Each property test must include a docstring with:
```
Feature: true-rul-prediction, Property {number}: {property_title}
```

**Note on Interpretability Testing**:
Since we're using interpretable models (XGBoost, LightGBM, Random Forest), we can also test:
- Feature importance values sum to reasonable total
- SHAP values are available and have correct shape
- Partial dependence plots can be generated
- Individual tree paths can be extracted for explanation

### Test Coverage Requirements

**Unit Test Coverage**:
- Data loading and parsing: 90%+ coverage
- Feature extraction: 85%+ coverage
- Model interfaces: 80%+ coverage
- Error handling: 95%+ coverage
- API endpoints: 100% coverage

**Property Test Coverage**:
- All 19 core properties must have corresponding property tests
- Each property test runs minimum 100 iterations
- Property tests should cover:
  - Input validation properties (Properties 1, 11, 12, 15)
  - Output format properties (Properties 2, 7, 8, 10)
  - Behavioral properties (Properties 9, 13, 14)
  - Interpretability properties (Property 16)
  - System properties (Properties 18, 19)

**Example-Based Test Coverage**:
- All 7 example tests must be implemented
- ES12 dataset performance validation (Example 1)
- Integration tests (Examples 5, 6, 7)
- Behavioral validation (Examples 2, 3, 4)

### Test Data Strategy

**Synthetic Data Generation**:
- Generate realistic voltage time-series using signal processing
- Simulate degradation patterns with controlled noise
- Create edge cases: flat signals, spikes, missing values

**ES12 Dataset Usage**:
- Reserve 2 capacitors for final validation (never used in training)
- Use remaining 6 capacitors for cross-validation
- Ensure test data includes all degradation stages

**Property Test Generators**:
```python
import hypothesis.strategies as st

# Strategy for generating valid voltage time-series
voltage_series = st.lists(
    st.floats(min_value=0.0, max_value=10.0, allow_nan=False, allow_infinity=False),
    min_size=10,
    max_size=200
)

# Strategy for generating cycle data
@st.composite
def cycle_data_strategy(draw):
    length = draw(st.integers(min_value=10, max_value=200))
    vl = draw(st.lists(st.floats(min_value=0, max_value=10), min_size=length, max_size=length))
    vo = draw(st.lists(st.floats(min_value=0, max_value=10), min_size=length, max_size=length))
    cycle_num = draw(st.integers(min_value=1, max_value=200))
    return CycleData(cycle_number=cycle_num, vl_series=np.array(vl), vo_series=np.array(vo))

# Strategy for generating capacitor sequences
@st.composite
def capacitor_sequence_strategy(draw):
    num_cycles = draw(st.integers(min_value=10, max_value=200))
    cycles = [draw(cycle_data_strategy()) for _ in range(num_cycles)]
    # Ensure monotonic cycle numbers
    for i, cycle in enumerate(cycles):
        cycle.cycle_number = i + 1
    return cycles
```

### Continuous Testing

**Pre-commit Hooks**:
- Run fast unit tests (<5 seconds)
- Run linting and type checking

**CI/CD Pipeline**:
- Run all unit tests
- Run property tests with 100 iterations
- Run example-based tests on ES12 dataset
- Generate coverage reports
- Performance benchmarking

**Nightly Tests**:
- Extended property testing with 1000 iterations
- Full ES12 dataset validation
- Performance regression testing
- Model retraining validation

### Performance Testing

**Latency Requirements**:
- Single prediction: < 1 second (Property 13)
- Batch prediction (10 samples): < 5 seconds
- Model loading: < 10 seconds

**Load Testing**:
- Concurrent requests: Support 10 simultaneous predictions
- Throughput: Process 100 predictions per minute

**Resource Monitoring**:
- Memory usage: < 2GB for loaded models
- CPU usage: < 80% during prediction
- GPU usage (if applicable): < 90% during batch processing

## Implementation Notes

### Technology Stack

**Core Framework**:
- Python 3.9+
- NumPy, SciPy for numerical computations
- Pandas for data manipulation

**Machine Learning Libraries**:
- **XGBoost**: Primary RUL regression model (high interpretability)
- **LightGBM**: Alternative gradient boosting (fast training)
- **scikit-learn**: Random Forest, Elastic Net, preprocessing, and traditional ML models
- **Isolation Forest & One-Class SVM**: Anomaly detection components
- **SHAP**: Model interpretability and feature importance analysis
- **Matplotlib/Seaborn**: Visualization for EDA and results

**Testing**:
- pytest for test framework
- hypothesis for property-based testing
- pytest-cov for coverage reporting

**API and Deployment**:
- FastAPI for REST API
- Pydantic for data validation
- Docker for containerization
- Redis for caching (optional)

### Model Training Pipeline

1. **Data Preparation**:
   - Load ES12 dataset
   - Split by capacitor (6 for training/validation, 2 for final test)
   - Extract features for all cycles
   - Normalize features per capacitor

2. **Semi-Supervised Training**:
   - Use cycles 1-10 as labeled "normal" data
   - Train anomaly detectors on normal data
   - Generate pseudo-labels for remaining cycles based on cycle number

3. **RUL Model Training**:
   - Create sequences with sliding window
   - Train LSTM/GRU/Transformer with RUL labels
   - Use early stopping based on validation RMSE
   - Save best model checkpoint

4. **Ensemble Training**:
   - Train multiple anomaly detectors
   - Tune ensemble weights on validation set
   - Optimize for FPR < 5% while maximizing recall

5. **Validation**:
   - Evaluate on held-out capacitors
   - Verify all properties hold
   - Check FPR, RMSE, MAE, R² metrics
   - Generate performance report

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Load Balancer                        │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
┌────────▼────────┐            ┌────────▼────────┐
│  API Server 1   │            │  API Server 2   │
│  (FastAPI)      │            │  (FastAPI)      │
└────────┬────────┘            └────────┬────────┘
         │                               │
         └───────────────┬───────────────┘
                         │
                ┌────────▼────────┐
                │  Model Cache    │
                │  (Redis)        │
                └────────┬────────┘
                         │
                ┌────────▼────────┐
                │  Model Storage  │
                │  (S3/Disk)      │
                └─────────────────┘
```

### Future Enhancements

1. **Multi-Dataset Generalization**: Extend to ES10 and ES14 datasets
2. **Online Learning**: Implement incremental learning for model updates
3. **Advanced Interpretability**: 
   - Counterfactual explanations ("What would need to change for RUL to increase by 10 cycles?")
   - Individual Conditional Expectation (ICE) plots
   - Accumulated Local Effects (ALE) plots
4. **Real-Time Monitoring**: Add streaming data support for continuous monitoring
5. **Adaptive Thresholds**: Implement dynamic threshold adjustment based on operational context
6. **Causal Analysis**: Investigate causal relationships between features and degradation using causal inference methods
