"""
NASA PCOE EDA System - Exploratory Data Analysis for NASA PCOE Dataset No.12

This package provides a comprehensive exploratory data analysis system specifically
designed for NASA PCOE (Prognostics Center of Excellence) datasets, with special
focus on Dataset No.12 (Capacitor Electrical Stress).

Key Features:
- Cross-platform support (Windows/macOS/Linux)
- Comprehensive statistical analysis and data quality assessment
- Advanced correlation analysis and multicollinearity detection
- Robust outlier detection using multiple methods
- Time series analysis and trend detection
- RUL (Remaining Useful Life) feature analysis for prognostics
- Fault level analysis for diagnostics applications
- Automated preprocessing recommendations
- High-quality visualizations with Japanese language support
- Automated HTML report generation
- Property-based testing for reliability
- Advanced warning management and suppression

Main Components:
- DataLoader: Robust data loading with MATLAB .mat file support
- StatisticsAnalyzer: Comprehensive descriptive statistics
- CorrelationAnalyzer: Advanced correlation and multicollinearity analysis
- OutlierDetector: Multiple outlier detection methods
- TimeSeriesAnalyzer: Time series pattern analysis
- RULFeatureAnalyzer: Degradation pattern analysis for prognostics
- FaultLevelAnalyzer: Feature analysis for fault diagnosis
- VisualizationEngine: High-quality plotting and visualization
- ReportGenerator: Automated HTML report generation
- AnalysisOrchestrator: Complete pipeline coordination
- WarningManager: Centralized warning suppression and reporting

Example usage:
    from nasa_pcoe_eda import AnalysisOrchestrator
    from pathlib import Path
    
    orchestrator = AnalysisOrchestrator()
    results = orchestrator.run_analysis(
        data_path=Path("data/ES12.mat"),
        output_dir=Path("output/")
    )
    report_path = orchestrator.generate_report(results)

For more detailed examples and documentation, see the README.md file
and the notebooks/ directory.
"""

__version__ = "0.1.0"

# Setup warning management on import
from nasa_pcoe_eda.utils.warnings_manager import setup_warning_management
setup_warning_management()

# Export exceptions
from nasa_pcoe_eda.exceptions import (
    AnalysisError,
    DataLoadError,
    DataValidationError,
    EDAError,
    VisualizationError,
)

# Export data models
from nasa_pcoe_eda.models import (
    AnalysisResults,
    DatasetMetadata,
    DataSplitStrategy,
    DistributionComparison,
    FeatureSuggestion,
    MissingValueReport,
    MulticollinearityReport,
    OutlierSummary,
    ScalingRecommendation,
    SeasonalityResult,
    Stats,
    TrendReport,
    ValidationResult,
)

# Export preprocessing components
from nasa_pcoe_eda.preprocessing import PreprocessingRecommender

# Export orchestrator
from nasa_pcoe_eda.orchestrator import AnalysisOrchestrator

# Export warning management utilities
from nasa_pcoe_eda.utils.warnings_manager import (
    get_warning_manager,
    get_warning_stats,
    generate_warning_report
)

__all__ = [
    # Exceptions
    "EDAError",
    "DataLoadError",
    "DataValidationError",
    "AnalysisError",
    "VisualizationError",
    # Data models
    "DatasetMetadata",
    "ValidationResult",
    "Stats",
    "MissingValueReport",
    "OutlierSummary",
    "TrendReport",
    "SeasonalityResult",
    "MulticollinearityReport",
    "DistributionComparison",
    "ScalingRecommendation",
    "FeatureSuggestion",
    "DataSplitStrategy",
    "AnalysisResults",
    # Preprocessing
    "PreprocessingRecommender",
    # Orchestrator
    "AnalysisOrchestrator",
    # Warning management
    "get_warning_manager",
    "get_warning_stats", 
    "generate_warning_report",
]
