# 設計書

## 概要

NASA PCOE データセットNo.12のEDAシステムは、Pythonベースのモジュラー設計を採用し、データの読み込み、分析、可視化、レポート生成を行います。システムはWindows/macOS両対応で、UVによる環境管理を使用します。

主要な設計目標：
- クロスプラットフォーム互換性（Windows/macOS）
- モジュラーで拡張可能なアーキテクチャ
- RUL予測と故障診断モデル構築への橋渡し
- 包括的な分析結果の可視化とレポート生成

## アーキテクチャ

システムは以下のレイヤーで構成されます：

```
┌─────────────────────────────────────┐
│     CLI / Notebook Interface        │
├─────────────────────────────────────┤
│        Analysis Orchestrator        │
├─────────────────────────────────────┤
│  ┌──────────┐  ┌──────────────────┐ │
│  │  Data    │  │   Analyzers      │ │
│  │  Loader  │  │  - Statistics    │ │
│  │          │  │  - Correlation   │ │
│  └──────────┘  │  - Outliers      │ │
│                │  - Time Series   │ │
│                │  - RUL Features  │ │
│                │  - Fault Level   │ │
│                └──────────────────┘ │
├─────────────────────────────────────┤
│         Visualization Engine        │
├─────────────────────────────────────┤
│          Report Generator           │
└─────────────────────────────────────┘
```

### コンポーネント間の相互作用

1. **Data Loader**: データセットファイルを読み込み、検証し、標準化されたDataFrame形式で提供
2. **Analyzers**: 各種分析機能を提供する独立したモジュール群
3. **Visualization Engine**: 分析結果を視覚化
4. **Report Generator**: 全ての分析結果を統合してレポートを生成
5. **Analysis Orchestrator**: 上記コンポーネントを調整し、分析フローを管理

## コンポーネントとインターフェース

### 1. DataLoader

**責務**: データセットの読み込み、検証、前処理

```python
class DataLoader:
    def load_dataset(self, path: Path) -> pd.DataFrame:
        """データセットを読み込む"""
        
    def validate_data(self, df: pd.DataFrame) -> ValidationResult:
        """データの整合性を検証"""
        
    def get_metadata(self, df: pd.DataFrame) -> DatasetMetadata:
        """データセットのメタデータを取得"""
```

### 2. StatisticsAnalyzer

**責務**: 基本統計量の計算

```python
class StatisticsAnalyzer:
    def compute_descriptive_stats(self, df: pd.DataFrame) -> Dict[str, Stats]:
        """記述統計を計算"""
        
    def analyze_missing_values(self, df: pd.DataFrame) -> MissingValueReport:
        """欠損値を分析"""
        
    def identify_data_types(self, df: pd.DataFrame) -> Dict[str, DataType]:
        """データ型を識別"""
```

### 3. CorrelationAnalyzer

**責務**: 特徴量間の相関分析

```python
class CorrelationAnalyzer:
    def compute_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """相関行列を計算"""
        
    def identify_high_correlations(
        self, corr_matrix: pd.DataFrame, threshold: float = 0.8
    ) -> List[Tuple[str, str, float]]:
        """強い相関を持つペアを識別"""
        
    def detect_multicollinearity(self, df: pd.DataFrame) -> MulticollinearityReport:
        """多重共線性を検出"""
```

### 4. OutlierDetector

**責務**: 外れ値と異常の検出

```python
class OutlierDetector:
    def detect_outliers_iqr(
        self, df: pd.DataFrame, threshold: float = 1.5
    ) -> Dict[str, np.ndarray]:
        """IQR法で外れ値を検出"""
        
    def detect_outliers_zscore(
        self, df: pd.DataFrame, threshold: float = 3.0
    ) -> Dict[str, np.ndarray]:
        """Zスコア法で外れ値を検出"""
        
    def summarize_outliers(self, outliers: Dict) -> OutlierSummary:
        """外れ値の統計を集計"""
```

### 5. TimeSeriesAnalyzer

**責務**: 時系列パターンの分析

```python
class TimeSeriesAnalyzer:
    def identify_temporal_features(self, df: pd.DataFrame) -> List[str]:
        """時間ベース特徴量を識別"""
        
    def compute_trends(self, df: pd.DataFrame, features: List[str]) -> TrendReport:
        """トレンドを計算"""
        
    def detect_seasonality(
        self, df: pd.DataFrame, feature: str
    ) -> SeasonalityResult:
        """季節性を検出"""
```

### 6. RULFeatureAnalyzer

**責務**: RUL予測のための特徴量分析

```python
class RULFeatureAnalyzer:
    def identify_degradation_features(self, df: pd.DataFrame) -> List[str]:
        """劣化トレンドを示す特徴量を識別"""
        
    def compute_degradation_rates(
        self, df: pd.DataFrame, features: List[str]
    ) -> Dict[str, float]:
        """劣化率を計算"""
        
    def rank_features_for_rul(
        self, df: pd.DataFrame, rul_column: str
    ) -> List[Tuple[str, float]]:
        """RUL予測に有用な特徴量をランク付け"""
        
    def visualize_degradation_patterns(
        self, df: pd.DataFrame, features: List[str]
    ) -> List[Figure]:
        """劣化パターンを可視化"""
```

### 7. FaultLevelAnalyzer

**責務**: 故障レベル同定のための分析

```python
class FaultLevelAnalyzer:
    def identify_discriminative_features(
        self, df: pd.DataFrame, fault_column: str
    ) -> List[str]:
        """故障状態を区別できる特徴量を識別"""
        
    def compare_distributions(
        self, df: pd.DataFrame, fault_column: str, features: List[str]
    ) -> DistributionComparison:
        """正常/異常状態の分布を比較"""
        
    def compute_class_separability(
        self, df: pd.DataFrame, fault_column: str, features: List[str]
    ) -> Dict[str, float]:
        """クラス間分離度を計算"""
```

### 8. VisualizationEngine

**責務**: データの可視化

```python
class VisualizationEngine:
    def __init__(self):
        # 日本語フォント設定
        self._setup_japanese_fonts()
        
    def plot_distributions(
        self, df: pd.DataFrame, features: List[str], output_dir: Path
    ) -> List[Path]:
        """分布のヒストグラムを生成"""
        
    def plot_time_series(
        self, df: pd.DataFrame, features: List[str], output_dir: Path
    ) -> List[Path]:
        """時系列プロットを生成"""
        
    def plot_correlation_heatmap(
        self, corr_matrix: pd.DataFrame, output_dir: Path
    ) -> Path:
        """相関ヒートマップを生成"""
        
    def plot_scatter_matrix(
        self, df: pd.DataFrame, features: List[str], output_dir: Path
    ) -> Path:
        """散布図行列を生成"""
```

### 9. ReportGenerator

**責務**: 包括的なEDAレポートの生成

```python
class ReportGenerator:
    def generate_report(
        self, 
        analysis_results: AnalysisResults,
        output_path: Path,
        format: str = "html"
    ) -> Path:
        """包括的なレポートを生成"""
        
    def create_summary_section(self, metadata: DatasetMetadata) -> str:
        """サマリーセクションを作成"""
        
    def create_statistics_section(self, stats: Dict) -> str:
        """統計セクションを作成"""
        
    def create_recommendations_section(
        self, rul_features: List[str], fault_features: List[str]
    ) -> str:
        """推奨事項セクションを作成"""
```

### 10. PreprocessingRecommender

**責務**: モデル構築のための前処理推奨

```python
class PreprocessingRecommender:
    def recommend_missing_value_strategy(
        self, missing_report: MissingValueReport
    ) -> Dict[str, str]:
        """欠損値処理方法を推奨"""
        
    def recommend_scaling(self, df: pd.DataFrame) -> ScalingRecommendation:
        """スケーリング方法を推奨"""
        
    def suggest_feature_engineering(
        self, df: pd.DataFrame, analysis_results: AnalysisResults
    ) -> List[FeatureSuggestion]:
        """特徴量エンジニアリングを提案"""
        
    def recommend_data_split(
        self, df: pd.DataFrame, is_time_series: bool
    ) -> DataSplitStrategy:
        """データ分割方法を推奨"""
```

## データモデル

### DatasetMetadata

```python
@dataclass
class DatasetMetadata:
    n_records: int
    n_features: int
    feature_names: List[str]
    data_types: Dict[str, str]
    memory_usage: float
    date_range: Optional[Tuple[datetime, datetime]]
```

### ValidationResult

```python
@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
```

### Stats

```python
@dataclass
class Stats:
    mean: float
    median: float
    std: float
    min: float
    max: float
    q25: float
    q75: float
```

### MissingValueReport

```python
@dataclass
class MissingValueReport:
    missing_counts: Dict[str, int]
    missing_percentages: Dict[str, float]
    total_missing: int
```

### OutlierSummary

```python
@dataclass
class OutlierSummary:
    outlier_counts: Dict[str, int]
    outlier_percentages: Dict[str, float]
    outlier_indices: Dict[str, np.ndarray]
```

### AnalysisResults

```python
@dataclass
class AnalysisResults:
    metadata: DatasetMetadata
    statistics: Dict[str, Stats]
    missing_values: MissingValueReport
    correlation_matrix: pd.DataFrame
    outliers: OutlierSummary
    time_series_trends: Optional[TrendReport]
    rul_features: List[Tuple[str, float]]
    fault_features: List[str]
    preprocessing_recommendations: Dict[str, Any]
    visualization_paths: List[Path]
```


## 正確性プロパティ

*プロパティとは、システムの全ての有効な実行において真であるべき特性や振る舞いのことです。本質的には、システムが何をすべきかについての形式的な記述です。プロパティは、人間が読める仕様と機械で検証可能な正確性保証との橋渡しとなります。*

### プロパティ1: パス処理のOS独立性

*任意の*ファイルパス文字列に対して、システムのパス処理関数は、Windows形式（バックスラッシュ）とUnix形式（スラッシュ）の両方で同じ論理的な結果を生成する
**検証: 要件 1.4**

### プロパティ2: データ読み込みの完全性

*任意の*有効なデータセットファイルに対して、読み込み後に報告されるレコード数と特徴量数は、実際のファイル内容と一致する
**検証: 要件 2.3**

### プロパティ3: データ検証の一貫性

*任意の*データファイルに対して、検証関数は同じ入力に対して常に同じ検証結果を返す
**検証: 要件 2.2**

### プロパティ4: エラーハンドリングの堅牢性

*任意の*破損または欠落したデータファイルに対して、システムは具体的なエラーメッセージを生成し、処理を継続する
**検証: 要件 2.4**

### プロパティ5: データ永続性

*任意の*データセットに対して、読み込み後にデータがメモリに保持され、後続の分析操作でアクセス可能である
**検証: 要件 2.5**

### プロパティ6: 統計計算の正確性

*任意の*数値データセットに対して、計算された平均値、中央値、標準偏差、最小値、最大値は、数学的定義に従った正しい値である
**検証: 要件 3.2**

### プロパティ7: 欠損値カウントの正確性

*任意の*データセットに対して、各特徴量について報告される欠損値数は、実際のNaN/null値の数と一致する
**検証: 要件 3.3**

### プロパティ8: データ型識別の正確性

*任意の*データセットに対して、各特徴量について識別されたデータ型は、実際のデータ型と一致する
**検証: 要件 3.4**

### プロパティ9: 相関計算の対称性

*任意の*数値データセットに対して、特徴量AとBの相関係数は、特徴量BとAの相関係数と等しい（相関行列は対称である）
**検証: 要件 3.5**

### プロパティ10: 可視化ファイル生成

*任意の*データセットと出力ディレクトリに対して、可視化関数を実行すると、指定されたディレクトリに画像ファイルが生成される
**検証: 要件 4.5**

### プロパティ11: 外れ値検出の閾値依存性

*任意の*データセットに対して、より厳しい閾値（例：IQRの1.5から3.0）を使用すると、検出される外れ値の数は減少するか同じである
**検証: 要件 5.4**

### プロパティ12: 外れ値カウントの正確性

*任意の*データセットと外れ値検出方法に対して、報告される外れ値の数は、実際に検出された外れ値インデックスの数と一致する
**検証: 要件 5.2**

### プロパティ13: 時間特徴量識別の一貫性

*任意の*データセットに対して、時間ベース特徴量として識別される列は、datetime型または時間的な命名規則を持つ
**検証: 要件 6.1**

### プロパティ14: レポート生成の完全性

*任意の*分析結果に対して、生成されるレポートには、統計、可視化、推奨事項の全てのセクションが含まれる
**検証: 要件 7.2**

### プロパティ15: レポート出力通知

*任意の*レポート生成操作に対して、システムは生成されたレポートファイルのパスを返す
**検証: 要件 7.5**

### プロパティ16: 重複検出の正確性

*任意の*データセットに対して、重複として報告されるレコードは、全ての列の値が完全に一致する
**検証: 要件 8.2**

### プロパティ17: データ品質問題の報告

*任意の*品質問題を含むデータセットに対して、システムは問題の種類（欠損、重複、型不整合）と影響を受ける行/列を報告する
**検証: 要件 8.4**

### プロパティ18: 相関の範囲制約

*任意の*数値データセットに対して、計算される全ての相関係数は-1から1の範囲内である
**検証: 要件 9.1**

### プロパティ19: 強相関識別の閾値一貫性

*任意の*データセットと閾値に対して、強い相関として識別される特徴量ペアの相関係数の絶対値は、指定された閾値以上である
**検証: 要件 9.2**

### プロパティ20: 劣化率の単調性

*任意の*劣化トレンドを示す特徴量に対して、計算される変化率は時間の経過とともに一貫した方向性（増加または減少）を示す
**検証: 要件 10.2**

### プロパティ21: RUL特徴量ランキングの順序性

*任意の*データセットに対して、RUL予測のための特徴量ランキングは、相関係数の絶対値の降順である
**検証: 要件 10.4**

### プロパティ22: クラス分離度の非負性

*任意の*分類データセットに対して、計算されるクラス間分離度の指標（例：Fisher比）は非負である
**検証: 要件 11.4**

### プロパティ23: 前処理推奨の完全性

*任意の*データセットに対して、前処理パイプラインの推奨には、欠損値処理、スケーリング、特徴量エンジニアリング、データ分割の全てのステップが含まれる
**検証: 要件 12.5**

### プロパティ24: スケーリング推奨の論理性

*任意の*データセットに対して、特徴量間のスケールの差が大きい（例：最大値/最小値の比が100以上）場合、システムは正規化を推奨する
**検証: 要件 12.2**

## エラーハンドリング

### エラーの分類

1. **データ読み込みエラー**
   - ファイルが存在しない
   - ファイル形式が不正
   - ファイルが破損している
   - メモリ不足

2. **データ検証エラー**
   - 必須列が欠落
   - データ型が不整合
   - 値の範囲が不正

3. **分析エラー**
   - 数値データが不足（統計計算不可）
   - 時系列データが不足（トレンド分析不可）
   - メモリ不足（大規模データ処理時）

4. **可視化エラー**
   - 出力ディレクトリへの書き込み権限なし
   - フォント設定エラー（日本語表示）
   - プロットライブラリのエラー

### エラーハンドリング戦略

```python
class EDAError(Exception):
    """EDAシステムの基底例外クラス"""
    pass

class DataLoadError(EDAError):
    """データ読み込みエラー"""
    pass

class DataValidationError(EDAError):
    """データ検証エラー"""
    pass

class AnalysisError(EDAError):
    """分析エラー"""
    pass

class VisualizationError(EDAError):
    """可視化エラー"""
    pass
```

### エラーハンドリングの原則

1. **早期検出**: データ読み込み時に可能な限り多くの問題を検出
2. **詳細なメッセージ**: エラーの原因と対処方法を明確に示す
3. **部分的成功**: 一部のデータや分析が失敗しても、可能な範囲で処理を継続
4. **ログ記録**: 全てのエラーと警告をログファイルに記録
5. **ユーザー通知**: エラーが発生した場合、ユーザーに分かりやすく通知

## テスト戦略

### ユニットテスト

各コンポーネントの個別機能をテストします：

1. **DataLoader**
   - 有効なファイルの読み込み
   - 無効なファイルのエラーハンドリング
   - メタデータの正確性

2. **StatisticsAnalyzer**
   - 既知の統計値を持つデータでの計算精度
   - 欠損値を含むデータの処理
   - エッジケース（全て同じ値、単一値など）

3. **CorrelationAnalyzer**
   - 完全相関（1.0）と無相関（0.0）のケース
   - 対称性の検証
   - 多重共線性検出の精度

4. **OutlierDetector**
   - 既知の外れ値を含むデータでの検出精度
   - 異なる閾値での動作
   - エッジケース（全て外れ値、外れ値なし）

5. **VisualizationEngine**
   - ファイル生成の確認
   - 日本語フォントの設定
   - 様々なデータサイズでの動作

### プロパティベーステスト

Pythonの`hypothesis`ライブラリを使用して、プロパティベーステストを実装します。

**テストライブラリ**: `hypothesis`

**設定**: 各プロパティテストは最低100回の反復を実行します。

**プロパティテストの実装要件**:
- 各プロパティテストは、設計書の正確性プロパティセクションの特定のプロパティを実装する
- 各テストには、実装するプロパティを明示的に参照するコメントを含める
- コメント形式: `# Feature: nasa-pcoe-eda, Property {番号}: {プロパティテキスト}`
- 各正確性プロパティは、単一のプロパティベーステストで実装される

**テスト例**:

```python
from hypothesis import given, strategies as st
import hypothesis

# Feature: nasa-pcoe-eda, Property 2: データ読み込みの完全性
@given(st.data())
@hypothesis.settings(max_examples=100)
def test_data_loading_completeness(data):
    """任意の有効なデータセットに対して、読み込み後の
    レコード数と特徴量数が実際の内容と一致する"""
    # テスト実装
    pass

# Feature: nasa-pcoe-eda, Property 9: 相関計算の対称性
@given(st.data())
@hypothesis.settings(max_examples=100)
def test_correlation_symmetry(data):
    """任意の数値データセットに対して、相関行列が対称である"""
    # テスト実装
    pass
```

### 統合テスト

エンドツーエンドのワークフローをテストします：

1. **完全なEDAパイプライン**
   - データ読み込みから レポート生成までの全フロー
   - 実際のNASA PCOEデータセットNo.12を使用

2. **RUL分析ワークフロー**
   - 劣化特徴量の識別
   - RUL相関分析
   - 推奨特徴量リストの生成

3. **故障診断ワークフロー**
   - 故障レベル識別
   - クラス分離度分析
   - 前処理推奨の生成

### テストデータ

1. **合成データ**
   - 既知の統計特性を持つデータ
   - 既知の外れ値を含むデータ
   - 既知の相関を持つデータ

2. **実データ**
   - NASA PCOE Dataset No.12
   - 様々なサイズのサブセット

3. **エッジケース**
   - 空のデータセット
   - 単一行/単一列のデータ
   - 全て欠損値のデータ
   - 全て同じ値のデータ

## 実装の考慮事項

### パフォーマンス

1. **メモリ管理**
   - 大規模データセットの場合、チャンク処理を使用
   - 不要なデータのコピーを避ける
   - 可視化後は大きなプロットオブジェクトを解放

2. **計算効率**
   - NumPy/Pandasのベクトル化操作を活用
   - 不要な再計算を避ける（結果のキャッシュ）
   - 並列処理が可能な部分は並列化を検討

### クロスプラットフォーム対応

1. **パス処理**
   - `pathlib.Path`を使用してOS依存性を排除
   - ファイル区切り文字のハードコーディングを避ける

2. **フォント設定**
   - OS別の日本語フォント検出ロジック
   - フォールバック機構の実装

3. **依存関係**
   - pyproject.tomlで全ての依存関係を明示
   - バージョン制約を適切に設定

### 拡張性

1. **プラグイン可能な分析器**
   - 新しい分析器を簡単に追加できる設計
   - 共通のインターフェースを定義

2. **カスタマイズ可能な可視化**
   - プロットスタイルの設定ファイル
   - カスタムカラーマップのサポート

3. **柔軟なレポート形式**
   - HTML、PDF、Markdownなど複数形式のサポート
   - テンプレートベースのレポート生成

### セキュリティ

1. **入力検証**
   - ファイルパスのサニタイゼーション
   - ファイルサイズの制限チェック

2. **安全なファイル操作**
   - パストラバーサル攻撃の防止
   - 書き込み権限の確認

## 技術スタック

- **言語**: Python 3.10+
- **パッケージ管理**: UV
- **データ処理**: pandas, numpy
- **統計分析**: scipy, statsmodels
- **可視化**: matplotlib, seaborn
- **レポート生成**: jinja2, markdown
- **テスト**: pytest, hypothesis
- **型チェック**: mypy
- **コード品質**: ruff (linter + formatter)

## プロジェクト構造

```
nasa-pcoe-eda/
├── pyproject.toml
├── README.md
├── data/
│   └── raw/              # 生データ
├── output/
│   ├── figures/          # 生成された図
│   └── reports/          # 生成されたレポート
├── src/
│   └── nasa_pcoe_eda/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   └── loader.py
│       ├── analysis/
│       │   ├── __init__.py
│       │   ├── statistics.py
│       │   ├── correlation.py
│       │   ├── outliers.py
│       │   ├── timeseries.py
│       │   ├── rul_features.py
│       │   └── fault_level.py
│       ├── visualization/
│       │   ├── __init__.py
│       │   └── engine.py
│       ├── reporting/
│       │   ├── __init__.py
│       │   ├── generator.py
│       │   └── templates/
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   └── recommender.py
│       └── orchestrator.py
├── tests/
│   ├── unit/
│   ├── property/
│   └── integration/
└── notebooks/
    └── exploratory.ipynb
```
