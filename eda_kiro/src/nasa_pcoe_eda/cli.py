"""
Command-line interface for NASA PCOE EDA system.

This module provides a comprehensive command-line interface for running
exploratory data analysis on NASA PCOE datasets. It supports various
analysis options, output configurations, and error handling.

Example usage:
    Basic analysis:
        $ nasa-pcoe-eda data/ES12.mat
    
    Full analysis with custom output:
        $ nasa-pcoe-eda data/ES12.mat --output-dir results/ --verbose
    
    RUL-focused analysis:
        $ nasa-pcoe-eda data/ES12.mat --rul-column RUL --analysis-type rul

The CLI automatically detects dataset characteristics and applies appropriate
analysis methods. Results are saved to the specified output directory with
comprehensive HTML reports and visualizations.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import logging

from .orchestrator import AnalysisOrchestrator
from .exceptions import EDAError


def setup_argument_parser() -> argparse.ArgumentParser:
    """
    Set up command-line argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="nasa-pcoe-eda",
        description="Exploratory Data Analysis system for NASA PCOE Dataset Repository No.12",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis with default output directory
  nasa-pcoe-eda data/ES12.mat
  
  # Analysis with custom output directory
  nasa-pcoe-eda data/ES12.mat --output-dir ./results
  
  # Analysis with RUL and fault columns specified
  nasa-pcoe-eda data/ES12.mat --rul-column RUL --fault-column fault_level
  
  # Analysis with time column and verbose output
  nasa-pcoe-eda data/ES12.mat --time-column time --verbose
  
  # Quiet mode (minimal output)
  nasa-pcoe-eda data/ES12.mat --quiet
        """
    )
    
    # Required arguments
    parser.add_argument(
        "data_path",
        type=str,
        help="Path to the dataset file (e.g., ES12.mat)"
    )
    
    # Optional arguments
    parser.add_argument(
        "-o", "--output-dir",
        type=str,
        default="output",
        help="Output directory for results (default: ./output)"
    )
    
    parser.add_argument(
        "--rul-column",
        type=str,
        help="Name of the RUL (Remaining Useful Life) column in the dataset"
    )
    
    parser.add_argument(
        "--fault-column",
        type=str,
        help="Name of the fault level column in the dataset"
    )
    
    parser.add_argument(
        "--time-column",
        type=str,
        help="Name of the time column in the dataset"
    )
    
    # Verbosity options
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output (detailed progress information)"
    )
    
    verbosity_group.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Enable quiet mode (minimal output)"
    )
    
    # Additional options
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 0.1.0"
    )
    
    return parser


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """
    Set up logging configuration based on verbosity level.
    
    Args:
        verbose: Enable verbose logging
        quiet: Enable quiet logging (minimal output)
    """
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(levelname)s: %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def validate_arguments(args: argparse.Namespace) -> None:
    """
    Validate command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Raises:
        SystemExit: If validation fails
    """
    # Check if data file exists
    data_path = Path(args.data_path)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}", file=sys.stderr)
        print(f"Please check the file path and try again.", file=sys.stderr)
        sys.exit(1)
    
    if not data_path.is_file():
        print(f"Error: Path is not a file: {data_path}", file=sys.stderr)
        sys.exit(1)
    
    # Check if output directory can be created
    output_dir = Path(args.output_dir)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        print(f"Error: Cannot create output directory: {output_dir}", file=sys.stderr)
        print(f"Please check permissions and try again.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Failed to create output directory: {e}", file=sys.stderr)
        sys.exit(1)


def print_progress(message: str, verbose: bool = False, quiet: bool = False) -> None:
    """
    Print progress message based on verbosity settings.
    
    Args:
        message: Message to print
        verbose: Verbose mode enabled
        quiet: Quiet mode enabled
    """
    if not quiet:
        if verbose:
            print(f"[PROGRESS] {message}")
        else:
            print(f"• {message}")


def print_summary(orchestrator: AnalysisOrchestrator, quiet: bool = False) -> None:
    """
    Print analysis summary.
    
    Args:
        orchestrator: Analysis orchestrator instance
        quiet: Quiet mode enabled
    """
    if quiet:
        return
    
    results = orchestrator.get_analysis_results()
    if not results:
        return
    
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    
    # Dataset info
    metadata = results.metadata
    print(f"Dataset: {metadata.n_records:,} records, {metadata.n_features} features")
    
    # Memory usage
    if metadata.memory_usage:
        print(f"Memory usage: {metadata.memory_usage:.2f} MB")
    
    # Date range
    if metadata.date_range:
        start_date, end_date = metadata.date_range
        print(f"Date range: {start_date} to {end_date}")
    
    # Missing values
    if results.missing_values:
        total_missing = results.missing_values.total_missing
        if total_missing > 0:
            print(f"Missing values: {total_missing:,} total")
        else:
            print("Missing values: None")
    
    # Outliers
    if results.outliers:
        print(f"Outliers detected: Yes")
    else:
        print("Outliers detected: None")
    
    # RUL features
    if results.rul_features:
        print(f"RUL-relevant features: {len(results.rul_features)}")
        if len(results.rul_features) > 0:
            top_feature = results.rul_features[0]
            print(f"  Top feature: {top_feature[0]} (score: {top_feature[1]:.3f})")
    
    # Fault features
    if results.fault_features:
        print(f"Fault-discriminative features: {len(results.fault_features)}")
    
    # Visualizations
    if results.visualization_paths:
        print(f"Visualizations generated: {len(results.visualization_paths)}")
    
    # Output location
    output_dir = orchestrator.output_dir
    print(f"\nResults saved to: {output_dir.absolute()}")
    
    # Report location
    report_path = output_dir / "reports" / "eda_report.html"
    if report_path.exists():
        print(f"HTML report: {report_path.absolute()}")
    
    print("="*60)


def main() -> int:
    """
    Main entry point for the CLI application.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Parse command-line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose, quiet=args.quiet)
    
    # Validate arguments
    validate_arguments(args)
    
    try:
        # Convert paths
        data_path = Path(args.data_path)
        output_dir = Path(args.output_dir)
        
        # Print startup message
        if not args.quiet:
            print("NASA PCOE EDA System")
            print("="*20)
            print(f"Dataset: {data_path}")
            print(f"Output directory: {output_dir.absolute()}")
            if args.rul_column:
                print(f"RUL column: {args.rul_column}")
            if args.fault_column:
                print(f"Fault column: {args.fault_column}")
            if args.time_column:
                print(f"Time column: {args.time_column}")
            print()
        
        # Initialize orchestrator
        print_progress("Initializing analysis system...", args.verbose, args.quiet)
        orchestrator = AnalysisOrchestrator(output_dir=output_dir)
        
        # Run complete analysis
        print_progress("Starting complete analysis pipeline...", args.verbose, args.quiet)
        
        # Progress tracking for main steps
        steps = [
            "Loading and validating data",
            "Computing basic statistics", 
            "Analyzing data quality",
            "Computing correlations",
            "Detecting outliers",
            "Analyzing time series patterns",
            "Analyzing RUL features",
            "Analyzing fault levels", 
            "Generating preprocessing recommendations",
            "Generating visualizations",
            "Creating comprehensive report"
        ]
        
        if args.verbose:
            for i, step in enumerate(steps, 1):
                print_progress(f"Step {i}/{len(steps)}: {step}", args.verbose, args.quiet)
        
        # Run the analysis
        results = orchestrator.run_complete_analysis(
            data_path=data_path,
            rul_column=args.rul_column,
            fault_column=args.fault_column,
            time_column=args.time_column
        )
        
        # Print completion message
        print_progress("Analysis completed successfully!", args.verbose, args.quiet)
        
        # Print summary
        print_summary(orchestrator, args.quiet)
        
        return 0
        
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.", file=sys.stderr)
        return 1
        
    except EDAError as e:
        print(f"\nAnalysis failed: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
        
    except Exception as e:
        print(f"\nUnexpected error: {e}", file=sys.stderr)
        print("Please check your input data and try again.", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())