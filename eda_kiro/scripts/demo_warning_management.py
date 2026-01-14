#!/usr/bin/env python3
"""
Demonstration script for warning management functionality.

This script shows how the warning management system works in practice,
demonstrating the difference between suppressed and unsuppressed warnings.
"""

import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from nasa_pcoe_eda.utils.warnings_manager import (
    get_warning_manager,
    generate_warning_report,
    get_warning_stats,
    WarningCategory
)
from nasa_pcoe_eda.visualization import VisualizationEngine


def demonstrate_font_warnings():
    """Demonstrate font warning suppression."""
    print("=" * 60)
    print("FONT WARNING DEMONSTRATION")
    print("=" * 60)
    
    # Create sample data
    df = pd.DataFrame({
        'データ1': np.random.normal(0, 1, 100),
        'データ2': np.random.normal(2, 1.5, 100),
        '測定値': np.random.exponential(1, 100)
    })
    
    print("\n1. Creating plots WITHOUT warning suppression:")
    print("   (You should see Japanese font warnings)")
    
    # Temporarily disable our warning suppression
    with warnings.catch_warnings():
        warnings.resetwarnings()  # Clear all filters
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(df['データ1'], bins=20, alpha=0.7, label='データ1')
        ax.set_title('日本語タイトルのテスト')
        ax.set_xlabel('値')
        ax.set_ylabel('頻度')
        ax.legend()
        plt.close()
    
    print("\n2. Creating plots WITH warning suppression:")
    print("   (You should NOT see Japanese font warnings)")
    
    # Use our visualization engine with warning suppression
    engine = VisualizationEngine()
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        try:
            paths = engine.plot_distributions(df, list(df.columns), output_dir)
            print(f"   ✓ Successfully created {len(paths)} plots with suppressed warnings")
        except Exception as e:
            print(f"   ✗ Error creating plots: {e}")


def demonstrate_warning_statistics():
    """Demonstrate warning statistics and reporting."""
    print("\n" + "=" * 60)
    print("WARNING STATISTICS DEMONSTRATION")
    print("=" * 60)
    
    wm = get_warning_manager()
    
    print("\n1. Current warning statistics:")
    stats = get_warning_stats()
    for category, count in stats.items():
        print(f"   {category}: {count} warnings")
    
    print("\n2. Warning configuration:")
    config = wm.config
    print(f"   Font warnings suppressed: {config.suppress_font_warnings}")
    print(f"   Matplotlib warnings suppressed: {config.suppress_matplotlib_warnings}")
    print(f"   Deprecation warnings suppressed: {config.suppress_deprecation_warnings}")
    print(f"   Log suppressed warnings: {config.log_suppressed_warnings}")
    
    print("\n3. Comprehensive warning report:")
    report = generate_warning_report()
    print(report)


def demonstrate_context_managers():
    """Demonstrate warning suppression context managers."""
    print("\n" + "=" * 60)
    print("CONTEXT MANAGER DEMONSTRATION")
    print("=" * 60)
    
    wm = get_warning_manager()
    
    print("\n1. Suppressing font warnings only:")
    with wm.suppress_category(WarningCategory.FONT):
        # This would normally generate font warnings
        fig, ax = plt.subplots()
        ax.set_title('日本語のタイトル')
        plt.close()
        print("   ✓ Font warnings suppressed in this context")
    
    print("\n2. Suppressing all configured warning categories:")
    with wm.suppress_all():
        # This would normally generate various warnings
        fig, ax = plt.subplots()
        ax.set_title('Another 日本語 title')
        # Simulate other operations that might generate warnings
        plt.close()
        print("   ✓ All configured warnings suppressed in this context")


def demonstrate_environment_control():
    """Demonstrate environment variable control."""
    print("\n" + "=" * 60)
    print("ENVIRONMENT VARIABLE DEMONSTRATION")
    print("=" * 60)
    
    import os
    
    print("\n1. Current environment settings:")
    env_vars = [
        'NASA_PCOE_SUPPRESS_WARNINGS',
        'NASA_PCOE_SUPPRESS_FONT',
        'NASA_PCOE_SUPPRESS_MATPLOTLIB',
        'NASA_PCOE_LOG_WARNINGS'
    ]
    
    for var in env_vars:
        value = os.getenv(var, 'not set')
        print(f"   {var}: {value}")
    
    print("\n2. To control warnings via environment variables:")
    print("   export NASA_PCOE_SUPPRESS_WARNINGS=false  # Disable all suppression")
    print("   export NASA_PCOE_SUPPRESS_FONT=true       # Suppress font warnings only")
    print("   export NASA_PCOE_LOG_WARNINGS=true        # Log suppressed warnings")


def main():
    """Run all demonstrations."""
    print("NASA PCOE EDA Warning Management Demonstration")
    print("=" * 60)
    print("This script demonstrates the warning management capabilities")
    print("of the NASA PCOE EDA system.")
    
    try:
        demonstrate_font_warnings()
        demonstrate_warning_statistics()
        demonstrate_context_managers()
        demonstrate_environment_control()
        
        print("\n" + "=" * 60)
        print("DEMONSTRATION COMPLETE")
        print("=" * 60)
        print("The warning management system provides:")
        print("• Automatic suppression of harmless warnings")
        print("• Configurable warning control via environment variables")
        print("• Context managers for temporary suppression")
        print("• Statistics and reporting capabilities")
        print("• Clean test output while preserving important warnings")
        
    except Exception as e:
        print(f"\nError during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()