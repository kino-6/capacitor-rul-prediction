#!/usr/bin/env python3
"""
NASA PCOE EDA System - Comprehensive Usage Examples

This file demonstrates various ways to use the NASA PCOE EDA system,
from basic analysis to advanced custom workflows.
"""

from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import numpy as np

# Import the main components
from nasa_pcoe_eda import (
    AnalysisOrchestrator,
    DataLoader,
    StatisticsAnalyzer,
    CorrelationAnalyzer,
    OutlierDetector,
    RULFeatureAnalyzer,
    FaultLevelAnalyzer,
    VisualizationEngine,
    PreprocessingRecommender,
    ReportGenerator,
)


def example_1_basic_analysis():
    """
    Example 1: Basic end-to-end analysis using the orchestrator.
    
    This is the simplest way to run a complete EDA analysis.
    """
    print("=== Example 1: Basic Analysis ===")
    
    # Initialize the orchestrator
    orchestrator = AnalysisOrchestrator(output_dir=Path("output/basic_analysis"))
    
    # Run complete analysis
    try:
        results = orchestrator.run_analysis(
            data_path=Path("data/raw/ES12.mat"),
            output_dir=Path("output/basic_analysis")
        )
        
        # Generate HTML report
        report_path = orchestrator.generate_report(results)
        print(f"Analysis complete! Report saved to: {report_path}")
        
        # Print basic statistics
        print(f"Dataset: {results.metadata.n_records} records, {results.metadata.n_features} features")
        print(f"Missing values: {results.missing_values.total_missing} total")
        print(f"Outliers detected in {len(results.outliers.outlier_counts)} features")
        
    except Exception as e:
        print(f"Analysis failed: {e}")


def example_2_step_by_step_analysis():
    """
    Example 2: Step-by-step analysis using individual components.
    
    This demonstrates how to use individual analyzers for custom workflows.
    """
    print("\n=== Example 2: Step-by-Step Analysis ===")
    
    # Step 1: Load data
    loader = DataLoader()
    try:
        df = loader.load_dataset(Path("data/raw/ES12.mat"))
        metadata = loader.get_metadata(df)
        validation = loader.validate_data(df)
        
        print(f"Loaded dataset: {metadata.n_records} records, {metadata.n_features} features")
        print(f"Validation: {'✓ Valid' if validation.is_valid else '✗ Invalid'}")
        
    except Exception as e:
        print(f"Data loading failed: {e}")
        return
    
    # Step 2: Basic statistics
    stats_analyzer = StatisticsAnalyzer()
    stats = stats_analyzer.compute_descriptive_stats(df)
    missing_report = stats_analyzer.analyze_missing_values(df)
    
    print(f"Computed statistics for {len(stats)} numeric features")
    print(f"Missing values: {missing_report.total_missing} total")
    
    # Step 3: Correlation analysis
    corr_analyzer = CorrelationAnalyzer()
    corr_matrix = corr_analyzer.compute_correlation_matrix(df)
    high_corrs = corr_analyzer.identify_high_correlations(corr_matrix, threshold=0.8)
    
    print(f"Found {len(high_corrs)} high correlations (>0.8)")
    
    # Step 4: Outlier detection
    outlier_detector = OutlierDetector()
    iqr_outliers = outlier_detector.detect_outliers_iqr(df, threshold=1.5)
    outlier_summary = outlier_detector.summarize_outliers(iqr_outliers, len(df))
    
    print(f"Outliers detected in {len(outlier_summary.outlier_counts)} features")
    
    # Step 5: Visualization
    viz_engine = VisualizationEngine()
    output_dir = Path("output/step_by_step")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot distributions for numeric features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()[:5]  # First 5 features
    dist_paths = viz_engine.plot_distributions(df, numeric_features, output_dir)
    print(f"Generated {len(dist_paths)} distribution plots")
    
    # Plot correlation heatmap
    if not corr_matrix.empty:
        heatmap_path = viz_engine.plot_correlation_heatmap(corr_matrix, output_dir)
        print(f"Generated correlation heatmap: {heatmap_path}")


def example_3_rul_analysis():
    """
    Example 3: RUL-focused analysis for prognostics applications.
    
    This demonstrates specialized analysis for RUL prediction.
    """
    print("\n=== Example 3: RUL Analysis ===")
    
    # Load data
    loader = DataLoader()
    try:
        df = loader.load_dataset(Path("data/raw/ES12.mat"))
    except Exception as e:
        print(f"Data loading failed: {e}")
        return
    
    # RUL feature analysis
    rul_analyzer = RULFeatureAnalyzer()
    
    # Identify degradation features
    degradation_features = rul_analyzer.identify_degradation_features(df)
    print(f"Identified {len(degradation_features)} degradation features:")
    for feature in degradation_features[:5]:  # Show first 5
        print(f"  - {feature}")
    
    # Compute degradation rates
    if degradation_features:
        degradation_rates = rul_analyzer.compute_degradation_rates(df, degradation_features)
        print(f"Computed degradation rates for {len(degradation_rates)} features")
        
        # Show top 3 fastest degrading features
        sorted_rates = sorted(degradation_rates.items(), key=lambda x: abs(x[1]), reverse=True)
        print("Top 3 fastest degrading features:")
        for feature, rate in sorted_rates[:3]:
            print(f"  - {feature}: {rate:.6f} units/time")
    
    # If RUL column exists, rank features
    if 'rul' in df.columns or 'RUL' in df.columns:
        rul_column = 'RUL' if 'RUL' in df.columns else 'rul'
        feature_ranking = rul_analyzer.rank_features_for_rul(df, rul_column)
        print(f"Ranked {len(feature_ranking)} features for RUL prediction:")
        for feature, correlation in feature_ranking[:5]:  # Top 5
            print(f"  - {feature}: correlation = {correlation:.3f}")
    
    # Visualize degradation patterns
    if degradation_features:
        viz_engine = VisualizationEngine()
        output_dir = Path("output/rul_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        viz_paths = rul_analyzer.visualize_degradation_patterns(
            df, degradation_features[:3], output_dir  # Visualize top 3
        )
        print(f"Generated {len(viz_paths)} degradation visualizations")


def example_4_fault_analysis():
    """
    Example 4: Fault diagnosis analysis.
    
    This demonstrates analysis for fault classification applications.
    """
    print("\n=== Example 4: Fault Analysis ===")
    
    # Load data
    loader = DataLoader()
    try:
        df = loader.load_dataset(Path("data/raw/ES12.mat"))
    except Exception as e:
        print(f"Data loading failed: {e}")
        return
    
    # Check if fault column exists
    fault_columns = [col for col in df.columns if 'fault' in str(col).lower()]
    if not fault_columns:
        print("No fault column found. Creating synthetic fault levels for demonstration.")
        # Create synthetic fault levels based on data quantiles
        numeric_col = df.select_dtypes(include=[np.number]).columns[0]
        df['fault_level'] = pd.cut(df[numeric_col], bins=3, labels=['Normal', 'Warning', 'Critical'])
        fault_column = 'fault_level'
    else:
        fault_column = fault_columns[0]
    
    # Fault level analysis
    fault_analyzer = FaultLevelAnalyzer()
    
    # Identify discriminative features
    discriminative_features = fault_analyzer.identify_discriminative_features(df, fault_column)
    print(f"Identified {len(discriminative_features)} discriminative features:")
    for feature in discriminative_features[:5]:  # Show first 5
        print(f"  - {feature}")
    
    # Compute class separability
    if discriminative_features:
        separability = fault_analyzer.compute_class_separability(
            df, fault_column, discriminative_features
        )
        print(f"Class separability scores:")
        for feature, score in list(separability.items())[:5]:  # Top 5
            print(f"  - {feature}: {score:.3f}")
    
    # Compare distributions between fault levels
    if discriminative_features:
        distribution_comparison = fault_analyzer.compare_distributions(
            df, fault_column, discriminative_features[:3]  # First 3 features
        )
        print("Distribution comparison completed for fault levels")


def example_5_preprocessing_recommendations():
    """
    Example 5: Preprocessing recommendations for model building.
    
    This demonstrates how to get preprocessing recommendations.
    """
    print("\n=== Example 5: Preprocessing Recommendations ===")
    
    # Load data and basic analysis
    loader = DataLoader()
    try:
        df = loader.load_dataset(Path("data/raw/ES12.mat"))
    except Exception as e:
        print(f"Data loading failed: {e}")
        return
    
    stats_analyzer = StatisticsAnalyzer()
    missing_report = stats_analyzer.analyze_missing_values(df)
    
    # Get preprocessing recommendations
    recommender = PreprocessingRecommender()
    
    # Missing value strategy
    missing_strategy = recommender.recommend_missing_value_strategy(missing_report)
    print("Missing value handling recommendations:")
    for feature, strategy in missing_strategy.items():
        print(f"  - {feature}: {strategy}")
    
    # Scaling recommendations
    scaling_rec = recommender.recommend_scaling(df)
    print(f"\nScaling recommendation: {scaling_rec.method}")
    print(f"Reason: {scaling_rec.reason}")
    print(f"Features to scale: {len(scaling_rec.features)}")
    
    # Feature engineering suggestions
    # Create mock analysis results for demonstration
    from nasa_pcoe_eda.models import AnalysisResults, DatasetMetadata, Stats, OutlierSummary
    
    metadata = DatasetMetadata(
        n_records=len(df),
        n_features=len(df.columns),
        feature_names=list(df.columns),
        data_types={str(col): str(dtype) for col, dtype in df.dtypes.items()},
        memory_usage=df.memory_usage(deep=True).sum(),
        date_range=None
    )
    
    mock_results = AnalysisResults(
        metadata=metadata,
        statistics={},
        missing_values=missing_report,
        correlation_matrix=pd.DataFrame(),
        outliers=OutlierSummary({}, {}, {}),
        time_series_trends=None,
        rul_features=[],
        fault_features=[],
        preprocessing_recommendations={},
        visualization_paths=[]
    )
    
    feature_suggestions = recommender.suggest_feature_engineering(df, mock_results)
    print(f"\nFeature engineering suggestions: {len(feature_suggestions)}")
    for suggestion in feature_suggestions[:3]:  # First 3
        print(f"  - {suggestion.feature_name}: {suggestion.operation}")
        print(f"    Rationale: {suggestion.rationale}")
    
    # Data split recommendations
    split_strategy = recommender.recommend_data_split(df, is_time_series=True)
    print(f"\nData split recommendation: {split_strategy.method}")
    print(f"Train ratio: {split_strategy.train_ratio}")
    print(f"Test ratio: {split_strategy.test_ratio}")
    print(f"Preserve temporal order: {split_strategy.preserve_temporal_order}")


def example_6_custom_visualization():
    """
    Example 6: Custom visualization workflows.
    
    This demonstrates advanced visualization capabilities.
    """
    print("\n=== Example 6: Custom Visualization ===")
    
    # Load data
    loader = DataLoader()
    try:
        df = loader.load_dataset(Path("data/raw/ES12.mat"))
    except Exception as e:
        print(f"Data loading failed: {e}")
        return
    
    # Initialize visualization engine
    viz_engine = VisualizationEngine()
    output_dir = Path("output/custom_viz")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get numeric features
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_features) >= 2:
        # Distribution plots
        dist_paths = viz_engine.plot_distributions(
            df, numeric_features[:5], output_dir
        )
        print(f"Generated {len(dist_paths)} distribution plots")
        
        # Correlation analysis and heatmap
        corr_analyzer = CorrelationAnalyzer()
        corr_matrix = corr_analyzer.compute_correlation_matrix(df)
        
        if not corr_matrix.empty:
            heatmap_path = viz_engine.plot_correlation_heatmap(corr_matrix, output_dir)
            print(f"Generated correlation heatmap: {heatmap_path}")
        
        # Scatter plot matrix for selected features
        if len(numeric_features) >= 3:
            scatter_path = viz_engine.plot_scatter_matrix(
                df, numeric_features[:4], output_dir  # First 4 features
            )
            print(f"Generated scatter plot matrix: {scatter_path}")
        
        # Time series plots (if time-based features exist)
        time_features = [col for col in df.columns if 'time' in str(col).lower() or 'cycle' in str(col).lower()]
        if time_features:
            ts_paths = viz_engine.plot_time_series(df, time_features[:3], output_dir)
            print(f"Generated {len(ts_paths)} time series plots")


def example_7_error_handling():
    """
    Example 7: Proper error handling and recovery.
    
    This demonstrates robust error handling patterns.
    """
    print("\n=== Example 7: Error Handling ===")
    
    from nasa_pcoe_eda.exceptions import DataLoadError, AnalysisError, VisualizationError
    
    # Example 1: Handle missing data file
    try:
        loader = DataLoader()
        df = loader.load_dataset(Path("data/nonexistent_file.mat"))
    except DataLoadError as e:
        print(f"✓ Caught data loading error: {e}")
    
    # Example 2: Handle analysis errors with empty data
    try:
        empty_df = pd.DataFrame()
        stats_analyzer = StatisticsAnalyzer()
        stats = stats_analyzer.compute_descriptive_stats(empty_df)
        print(f"Empty dataframe handled gracefully: {len(stats)} features analyzed")
    except AnalysisError as e:
        print(f"✓ Caught analysis error: {e}")
    
    # Example 3: Handle visualization errors
    try:
        viz_engine = VisualizationEngine()
        # Try to create visualization with invalid output directory
        invalid_output = Path("/invalid/path/that/does/not/exist")
        viz_engine.plot_distributions(pd.DataFrame(), [], invalid_output)
    except (VisualizationError, PermissionError, OSError) as e:
        print(f"✓ Caught visualization error: {type(e).__name__}: {e}")
    
    print("Error handling examples completed successfully!")


def main():
    """
    Run all examples in sequence.
    
    Note: Some examples may fail if the data file is not available.
    This is expected and demonstrates proper error handling.
    """
    print("NASA PCOE EDA System - Usage Examples")
    print("=" * 50)
    
    # Create output directories
    Path("output").mkdir(exist_ok=True)
    
    # Run examples
    examples = [
        example_1_basic_analysis,
        example_2_step_by_step_analysis,
        example_3_rul_analysis,
        example_4_fault_analysis,
        example_5_preprocessing_recommendations,
        example_6_custom_visualization,
        example_7_error_handling,
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"Example {i} failed with error: {e}")
            print("This may be expected if data files are not available.")
        
        if i < len(examples):
            print("\n" + "-" * 50)
    
    print("\n" + "=" * 50)
    print("All examples completed!")
    print("\nTo run individual examples:")
    print("python examples/usage_examples.py")
    print("\nFor more information, see the README.md file.")


if __name__ == "__main__":
    main()