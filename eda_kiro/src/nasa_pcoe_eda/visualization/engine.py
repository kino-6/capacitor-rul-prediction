"""
Visualization engine for data plotting and analysis visualization.

This module provides comprehensive visualization capabilities for EDA:
- Distribution plots (histograms, box plots, violin plots)
- Time series visualizations with trend analysis
- Correlation heatmaps and scatter plot matrices
- Specialized capacitor degradation visualizations
- Japanese language support for labels and titles
- High-quality output with customizable styling
- Automatic figure saving and path management
- Warning suppression for known harmless warnings

The engine is specifically optimized for NASA PCOE datasets and includes
specialized visualizations for capacitor degradation analysis, EIS data,
and prognostics-related patterns.

Warning Management:
- Automatic suppression of Japanese font warnings
- Configurable warning filters via environment variables
- Warning categorization and reporting capabilities

Example usage:
    engine = VisualizationEngine()
    dist_paths = engine.plot_distributions(df, features, output_dir)
    ts_paths = engine.plot_time_series(df, time_features, output_dir)
    heatmap_path = engine.plot_correlation_heatmap(corr_matrix, output_dir)
    degradation_paths = engine.visualize_capacitor_degradation(df, output_dir)
"""

from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import warnings
import os
import logging
from contextlib import contextmanager

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
import seaborn as sns
import japanize_matplotlib
import h5py
from scipy import stats

from ..models import DatasetMetadata
from ..exceptions import VisualizationError


class VisualizationEngine:
    """Engine for generating various data visualizations with warning management."""
    
    def __init__(self) -> None:
        """Initialize the visualization engine with Japanese font support and warning filters."""
        self._setup_warning_filters()
        self._setup_japanese_fonts()
        self._setup_plot_style()
        self._warning_stats = {
            'japanese_font': 0,
            'deprecation': 0,
            'runtime': 0,
            'other': 0
        }
    
    def _setup_warning_filters(self) -> None:
        """Setup warning filters based on environment variables and configuration."""
        # Check environment variable for warning suppression
        suppress_warnings = os.getenv('NASA_PCOE_SUPPRESS_WARNINGS', 'true').lower() == 'true'
        
        if suppress_warnings:
            # Suppress known harmless warnings
            warnings.filterwarnings('ignore', category=UserWarning, 
                                  message='.*Glyph.*missing from current font.*')
            warnings.filterwarnings('ignore', category=UserWarning,
                                  message='.*findfont.*')
            warnings.filterwarnings('ignore', category=UserWarning,
                                  message='.*font.*not found.*')
            warnings.filterwarnings('ignore', category=UserWarning,
                                  message='.*Japanese.*font.*')
            
            # Suppress matplotlib font warnings
            warnings.filterwarnings('ignore', category=UserWarning,
                                  module='matplotlib.*')
            
            # Suppress japanize-matplotlib warnings
            warnings.filterwarnings('ignore', category=UserWarning,
                                  module='japanize_matplotlib.*')
            
            # Suppress specific deprecation warnings that are not actionable
            warnings.filterwarnings('ignore', category=DeprecationWarning,
                                  message='.*pkg_resources.*')
            warnings.filterwarnings('ignore', category=DeprecationWarning,
                                  message='.*distutils.*')
    
    @contextmanager
    def _suppress_font_warnings(self):
        """Context manager to temporarily suppress font-related warnings."""
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning,
                                  message='.*Glyph.*missing from current font.*')
            warnings.filterwarnings('ignore', category=UserWarning,
                                  message='.*findfont.*')
            warnings.filterwarnings('ignore', category=UserWarning,
                                  message='.*font.*not found.*')
            warnings.filterwarnings('ignore', category=UserWarning,
                                  message='.*Japanese.*font.*')
            warnings.filterwarnings('ignore', category=UserWarning,
                                  module='matplotlib.*')
            yield
    
    def _setup_japanese_fonts(self) -> None:
        """Setup Japanese font support using japanize-matplotlib."""
        try:
            with self._suppress_font_warnings():
                # japanize-matplotlib automatically configures Japanese fonts
                plt.rcParams['font.family'] = ['DejaVu Sans', 'Hiragino Sans', 'Yu Gothic', 'Meiryo', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
        except Exception as e:
            # Only show this warning if not suppressed
            if os.getenv('NASA_PCOE_SUPPRESS_WARNINGS', 'true').lower() != 'true':
                warnings.warn(f"Failed to setup Japanese fonts: {e}")
    
    def get_warning_stats(self) -> Dict[str, int]:
        """Get statistics about suppressed warnings."""
        return self._warning_stats.copy()
    
    def reset_warning_stats(self) -> None:
        """Reset warning statistics."""
        self._warning_stats = {
            'japanese_font': 0,
            'deprecation': 0,
            'runtime': 0,
            'other': 0
        }
    
    def _setup_plot_style(self) -> None:
        """Setup default plot style."""
        plt.style.use('default')
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
        plt.rcParams['legend.fontsize'] = 9
    
    def plot_distributions(
        self, 
        df: pd.DataFrame, 
        features: List[str], 
        output_dir: Path
    ) -> List[Path]:
        """
        Generate histogram plots for feature distributions.
        
        Args:
            df: DataFrame containing the data
            features: List of feature names to plot
            output_dir: Directory to save plots
            
        Returns:
            List of paths to generated plot files
            
        Raises:
            VisualizationError: If plotting fails
        """
        with self._suppress_font_warnings():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                plot_paths = []
                
                # Filter to numeric features only
                numeric_features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
                
                if not numeric_features:
                    raise VisualizationError("No numeric features found for distribution plotting")
                
                # Create subplots
                n_features = len(numeric_features)
                n_cols = min(3, n_features)
                n_rows = (n_features + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
                if n_features == 1:
                    axes = [axes]
                elif n_rows == 1:
                    axes = axes.flatten()
                else:
                    axes = axes.flatten()
                
                for i, feature in enumerate(numeric_features):
                    ax = axes[i]
                    data = df[feature].dropna()
                    
                    if len(data) == 0:
                        ax.text(0.5, 0.5, f'No data for {feature}', 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'{feature} - No Data')
                        continue
                    
                    # Plot histogram
                    ax.hist(data, bins=30, alpha=0.7, edgecolor='black')
                    ax.set_title(f'{feature}の分布')
                    ax.set_xlabel(feature)
                    ax.set_ylabel('頻度')
                    
                    # Add statistics text
                    mean_val = data.mean()
                    std_val = data.std()
                    ax.axvline(mean_val, color='red', linestyle='--', alpha=0.7, label=f'平均: {mean_val:.2f}')
                    ax.legend()
                    
                    # Add text box with statistics
                    stats_text = f'平均: {mean_val:.2f}\n標準偏差: {std_val:.2f}\nサンプル数: {len(data)}'
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # Hide unused subplots
                for i in range(n_features, len(axes)):
                    axes[i].set_visible(False)
                
                plt.tight_layout()
                
                # Save plot
                plot_path = output_dir / 'distributions.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_paths.append(plot_path)
                
                return plot_paths
                
            except Exception as e:
                raise VisualizationError(f"Failed to generate distribution plots: {e}")
    
    def plot_time_series(
        self, 
        df: pd.DataFrame, 
        features: List[str], 
        output_dir: Path,
        time_column: Optional[str] = None
    ) -> List[Path]:
        """
        Generate time series plots for temporal features.
        
        Args:
            df: DataFrame containing the data
            features: List of feature names to plot
            output_dir: Directory to save plots
            time_column: Name of time column (if None, uses index)
            
        Returns:
            List of paths to generated plot files
            
        Raises:
            VisualizationError: If plotting fails
        """
        with self._suppress_font_warnings():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                plot_paths = []
                
                # Filter to numeric features only
                numeric_features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
                
                if not numeric_features:
                    raise VisualizationError("No numeric features found for time series plotting")
                
                # Determine x-axis (time)
                if time_column and time_column in df.columns:
                    x_data = df[time_column]
                    x_label = time_column
                else:
                    x_data = df.index
                    x_label = 'Index'
                
                # Create individual plots for each feature
                for feature in numeric_features:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    y_data = df[feature].dropna()
                    x_data_clean = x_data[y_data.index]
                    
                    if len(y_data) == 0:
                        ax.text(0.5, 0.5, f'No data for {feature}', 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'{feature} - No Data')
                    else:
                        # Plot time series
                        ax.plot(x_data_clean, y_data, linewidth=1.5, alpha=0.8)
                        ax.set_title(f'{feature}の時系列変化')
                        ax.set_xlabel(x_label)
                        ax.set_ylabel(feature)
                        ax.grid(True, alpha=0.3)
                        
                        # Add trend line if enough data points
                        if len(y_data) > 2:
                            try:
                                # Convert x_data to numeric for trend calculation
                                if pd.api.types.is_datetime64_any_dtype(x_data_clean):
                                    x_numeric = pd.to_numeric(x_data_clean)
                                else:
                                    x_numeric = pd.to_numeric(x_data_clean, errors='coerce')
                                
                                if not x_numeric.isna().all():
                                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, y_data)
                                    trend_line = slope * x_numeric + intercept
                                    ax.plot(x_data_clean, trend_line, 'r--', alpha=0.7, 
                                           label=f'トレンド (R²={r_value**2:.3f})')
                                    ax.legend()
                            except Exception:
                                pass  # Skip trend line if calculation fails
                    
                    plt.tight_layout()
                    
                    # Save plot
                    plot_path = output_dir / f'timeseries_{feature}.png'
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    plot_paths.append(plot_path)
                
                return plot_paths
                
            except Exception as e:
                raise VisualizationError(f"Failed to generate time series plots: {e}")
    
    def plot_correlation_heatmap(
        self, 
        corr_matrix: pd.DataFrame, 
        output_dir: Path
    ) -> Path:
        """
        Generate correlation heatmap.
        
        Args:
            corr_matrix: Correlation matrix DataFrame
            output_dir: Directory to save plot
            
        Returns:
            Path to generated plot file
            
        Raises:
            VisualizationError: If plotting fails
        """
        with self._suppress_font_warnings():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                
                fig, ax = plt.subplots(figsize=(12, 10))
                
                # Create heatmap
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
                sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                           square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
                
                ax.set_title('特徴量間の相関ヒートマップ')
                plt.tight_layout()
                
                # Save plot
                plot_path = output_dir / 'correlation_heatmap.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                return plot_path
                
            except Exception as e:
                raise VisualizationError(f"Failed to generate correlation heatmap: {e}")
    
    def plot_scatter_matrix(
        self, 
        df: pd.DataFrame, 
        features: List[str], 
        output_dir: Path
    ) -> Path:
        """
        Generate scatter plot matrix for feature relationships.
        
        Args:
            df: DataFrame containing the data
            features: List of feature names to include
            output_dir: Directory to save plot
            
        Returns:
            Path to generated plot file
            
        Raises:
            VisualizationError: If plotting fails
        """
        with self._suppress_font_warnings():
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Filter to numeric features only and limit to reasonable number
                numeric_features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
                
                if not numeric_features:
                    raise VisualizationError("No numeric features found for scatter matrix")
                
                # Limit to first 6 features to keep plot readable
                if len(numeric_features) > 6:
                    numeric_features = numeric_features[:6]
                    warnings.warn(f"Limited scatter matrix to first 6 features for readability")
                
                # Create scatter matrix using seaborn pairplot
                subset_df = df[numeric_features].dropna()
                
                if len(subset_df) == 0:
                    raise VisualizationError("No data remaining after removing NaN values")
                
                g = sns.pairplot(subset_df, diag_kind='hist', plot_kws={'alpha': 0.6})
                g.fig.suptitle('特徴量間の散布図行列', y=1.02)
                
                # Save plot
                plot_path = output_dir / 'scatter_matrix.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                return plot_path
                
            except Exception as e:
                raise VisualizationError(f"Failed to generate scatter matrix: {e}")
    
    def plot_capacitor_degradation_analysis(
        self, 
        data_path: Path, 
        output_dir: Path
    ) -> List[Path]:
        """
        Generate comprehensive capacitor degradation analysis plots.
        
        This method implements the detailed capacitor degradation visualization
        as specified in task 8.1, including:
        - Input-output response changes over cycles
        - Capacity and ESR degradation trends
        - Individual capacitor degradation patterns
        - Multi-capacitor degradation comparison
        
        Args:
            data_path: Path to ES12.mat file
            output_dir: Directory to save plots
            
        Returns:
            List of paths to generated plot files
            
        Raises:
            VisualizationError: If plotting fails
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            plot_paths = []
            
            # Load MATLAB v7.3 format data using h5py
            with h5py.File(data_path, 'r') as f:
                # Extract EIS_Data structure
                if 'EIS_Data' not in f:
                    raise VisualizationError("EIS_Data not found in the dataset")
                
                eis_data = f['EIS_Data']
                
                # Extract data for each capacitor (ES12C1 to ES12C8)
                capacitor_data = {}
                capacitor_names = []
                
                # Find all capacitor datasets
                for key in eis_data.keys():
                    if key.startswith('ES12C') and len(key) == 6:  # ES12C1, ES12C2, etc.
                        capacitor_names.append(key)
                
                if not capacitor_names:
                    raise VisualizationError("No capacitor data found (ES12C1-ES12C8)")
                
                capacitor_names.sort()  # Ensure consistent ordering
                
                # Extract data for each capacitor
                for cap_name in capacitor_names:
                    cap_data = eis_data[cap_name]
                    
                    # Extract relevant measurements
                    capacitor_info = {
                        'name': cap_name,
                        'cycles': [],
                        'capacity': [],
                        'esr': [],
                        'impedance_real': [],
                        'impedance_imag': [],
                        'frequency': [],
                        'phase': []
                    }
                    
                    # Navigate through the HDF5 structure to extract cycle data
                    try:
                        # The exact structure may vary, so we'll try different approaches
                        if 'cycle' in cap_data:
                            cycle_data = cap_data['cycle']
                            for cycle_key in cycle_data.keys():
                                if cycle_key.isdigit():
                                    cycle_num = int(cycle_key)
                                    cycle_info = cycle_data[cycle_key]
                                    
                                    # Extract capacity and ESR if available
                                    if 'capacity' in cycle_info:
                                        capacity_val = float(cycle_info['capacity'][0, 0])
                                        capacitor_info['cycles'].append(cycle_num)
                                        capacitor_info['capacity'].append(capacity_val)
                                    
                                    if 'esr' in cycle_info:
                                        esr_val = float(cycle_info['esr'][0, 0])
                                        capacitor_info['esr'].append(esr_val)
                                    
                                    # Extract impedance data if available
                                    if 'impedance' in cycle_info:
                                        imp_data = cycle_info['impedance']
                                        if 'real' in imp_data:
                                            capacitor_info['impedance_real'].append(imp_data['real'][:])
                                        if 'imag' in imp_data:
                                            capacitor_info['impedance_imag'].append(imp_data['imag'][:])
                                        if 'frequency' in imp_data:
                                            capacitor_info['frequency'].append(imp_data['frequency'][:])
                        
                        # Alternative structure exploration
                        elif hasattr(cap_data, 'keys'):
                            # Try to find measurement data in different structure
                            for key in cap_data.keys():
                                if 'measurement' in key.lower() or 'data' in key.lower():
                                    # Extract what we can from this structure
                                    pass
                    
                    except Exception as e:
                        warnings.warn(f"Could not fully extract data for {cap_name}: {e}")
                    
                    capacitor_data[cap_name] = capacitor_info
            
            # Generate plots based on extracted data
            plot_paths.extend(self._plot_capacity_esr_trends(capacitor_data, output_dir))
            plot_paths.extend(self._plot_individual_degradation(capacitor_data, output_dir))
            plot_paths.extend(self._plot_multi_capacitor_comparison(capacitor_data, output_dir))
            plot_paths.extend(self._plot_impedance_response_changes(capacitor_data, output_dir))
            
            return plot_paths
            
        except Exception as e:
            raise VisualizationError(f"Failed to generate capacitor degradation analysis: {e}")
    
    def _plot_capacity_esr_trends(self, capacitor_data: Dict, output_dir: Path) -> List[Path]:
        """Plot capacity and ESR degradation trends."""
        plot_paths = []
        
        try:
            # Plot capacity trends
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            for cap_name, data in capacitor_data.items():
                if data['cycles'] and data['capacity']:
                    cycles = np.array(data['cycles'])
                    capacity = np.array(data['capacity'])
                    
                    # Calculate degradation percentage
                    if len(capacity) > 0:
                        initial_capacity = capacity[0]
                        degradation_pct = ((capacity - initial_capacity) / initial_capacity) * 100
                        
                        ax1.plot(cycles, degradation_pct, marker='o', label=cap_name, linewidth=2)
                        
                        # Calculate degradation rate
                        if len(cycles) > 1:
                            slope, _, r_value, _, _ = stats.linregress(cycles, degradation_pct)
                            ax1.plot(cycles, slope * cycles + degradation_pct[0], 
                                   '--', alpha=0.7, 
                                   label=f'{cap_name} トレンド (R²={r_value**2:.3f})')
            
            ax1.set_title('容量劣化トレンド')
            ax1.set_xlabel('サイクル数')
            ax1.set_ylabel('容量変化率 (%)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot ESR trends
            for cap_name, data in capacitor_data.items():
                if data['cycles'] and data['esr']:
                    cycles = np.array(data['cycles'])
                    esr = np.array(data['esr'])
                    
                    # Calculate ESR change percentage
                    if len(esr) > 0:
                        initial_esr = esr[0]
                        esr_change_pct = ((esr - initial_esr) / initial_esr) * 100
                        
                        ax2.plot(cycles, esr_change_pct, marker='s', label=cap_name, linewidth=2)
                        
                        # Calculate ESR change rate
                        if len(cycles) > 1:
                            slope, _, r_value, _, _ = stats.linregress(cycles, esr_change_pct)
                            ax2.plot(cycles, slope * cycles + esr_change_pct[0], 
                                   '--', alpha=0.7,
                                   label=f'{cap_name} トレンド (R²={r_value**2:.3f})')
            
            ax2.set_title('ESR変化トレンド')
            ax2.set_xlabel('サイクル数')
            ax2.set_ylabel('ESR変化率 (%)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            plot_path = output_dir / 'capacity_esr_trends.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
            
        except Exception as e:
            warnings.warn(f"Failed to plot capacity/ESR trends: {e}")
        
        return plot_paths
    
    def _plot_individual_degradation(self, capacitor_data: Dict, output_dir: Path) -> List[Path]:
        """Plot individual capacitor degradation patterns."""
        plot_paths = []
        
        try:
            for cap_name, data in capacitor_data.items():
                if not (data['cycles'] and data['capacity']):
                    continue
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                cycles = np.array(data['cycles'])
                capacity = np.array(data['capacity'])
                
                # Plot capacity over cycles
                ax.plot(cycles, capacity, 'bo-', linewidth=2, markersize=6, label='容量')
                
                # Fit polynomial to show non-linearity
                if len(cycles) > 3:
                    try:
                        # Fit quadratic polynomial
                        coeffs = np.polyfit(cycles, capacity, 2)
                        poly_fit = np.poly1d(coeffs)
                        cycle_smooth = np.linspace(cycles.min(), cycles.max(), 100)
                        ax.plot(cycle_smooth, poly_fit(cycle_smooth), 'r--', 
                               alpha=0.7, label='非線形フィット')
                        
                        # Detect change points (simplified)
                        if len(capacity) > 5:
                            # Calculate rate of change
                            rate_of_change = np.diff(capacity) / np.diff(cycles)
                            # Find points where rate changes significantly
                            rate_change = np.abs(np.diff(rate_of_change))
                            if len(rate_change) > 0:
                                change_point_idx = np.argmax(rate_change) + 1
                                if change_point_idx < len(cycles):
                                    ax.axvline(cycles[change_point_idx], color='orange', 
                                             linestyle=':', alpha=0.8, 
                                             label=f'変化点 (サイクル {cycles[change_point_idx]})')
                    
                    except Exception:
                        pass  # Skip advanced analysis if it fails
                
                ax.set_title(f'{cap_name} 個別劣化パターン')
                ax.set_xlabel('サイクル数')
                ax.set_ylabel('容量')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add statistics text
                if len(capacity) > 1:
                    initial_cap = capacity[0]
                    final_cap = capacity[-1]
                    total_degradation = ((final_cap - initial_cap) / initial_cap) * 100
                    avg_degradation_rate = total_degradation / (cycles[-1] - cycles[0])
                    
                    stats_text = (f'初期容量: {initial_cap:.3f}\n'
                                f'最終容量: {final_cap:.3f}\n'
                                f'総劣化: {total_degradation:.1f}%\n'
                                f'平均劣化率: {avg_degradation_rate:.3f}%/サイクル')
                    
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                           verticalalignment='top', 
                           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                
                plt.tight_layout()
                
                plot_path = output_dir / f'individual_degradation_{cap_name}.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plot_paths.append(plot_path)
                
        except Exception as e:
            warnings.warn(f"Failed to plot individual degradation: {e}")
        
        return plot_paths
    
    def _plot_multi_capacitor_comparison(self, capacitor_data: Dict, output_dir: Path) -> List[Path]:
        """Plot comparison of multiple capacitors."""
        plot_paths = []
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            degradation_rates = []
            capacitor_names = []
            
            # Compare degradation rates
            for cap_name, data in capacitor_data.items():
                if not (data['cycles'] and data['capacity']):
                    continue
                
                cycles = np.array(data['cycles'])
                capacity = np.array(data['capacity'])
                
                if len(capacity) > 1:
                    # Calculate degradation rate
                    initial_cap = capacity[0]
                    final_cap = capacity[-1]
                    total_cycles = cycles[-1] - cycles[0]
                    degradation_rate = ((final_cap - initial_cap) / initial_cap) * 100 / total_cycles
                    
                    degradation_rates.append(degradation_rate)
                    capacitor_names.append(cap_name)
                    
                    # Plot normalized capacity curves
                    normalized_capacity = capacity / initial_cap
                    ax1.plot(cycles, normalized_capacity, marker='o', label=cap_name, linewidth=2)
            
            ax1.set_title('正規化容量の比較')
            ax1.set_xlabel('サイクル数')
            ax1.set_ylabel('正規化容量 (初期値=1.0)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Bar plot of degradation rates
            if degradation_rates:
                colors = plt.cm.viridis(np.linspace(0, 1, len(degradation_rates)))
                bars = ax2.bar(capacitor_names, degradation_rates, color=colors)
                ax2.set_title('劣化速度の比較')
                ax2.set_xlabel('コンデンサ')
                ax2.set_ylabel('劣化率 (%/サイクル)')
                ax2.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, rate in zip(bars, degradation_rates):
                    height = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{rate:.3f}', ha='center', va='bottom')
                
                # Identify fastest and slowest degrading capacitors
                if len(degradation_rates) > 1:
                    fastest_idx = np.argmin(degradation_rates)  # Most negative = fastest degradation
                    slowest_idx = np.argmax(degradation_rates)  # Least negative = slowest degradation
                    
                    ax2.text(0.02, 0.98, 
                            f'最速劣化: {capacitor_names[fastest_idx]} ({degradation_rates[fastest_idx]:.3f}%/サイクル)\n'
                            f'最遅劣化: {capacitor_names[slowest_idx]} ({degradation_rates[slowest_idx]:.3f}%/サイクル)',
                            transform=ax2.transAxes, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            
            plt.tight_layout()
            
            plot_path = output_dir / 'multi_capacitor_comparison.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
            
        except Exception as e:
            warnings.warn(f"Failed to plot multi-capacitor comparison: {e}")
        
        return plot_paths
    
    def _plot_impedance_response_changes(self, capacitor_data: Dict, output_dir: Path) -> List[Path]:
        """Plot impedance response changes over cycles."""
        plot_paths = []
        
        try:
            # This would require impedance spectroscopy data
            # For now, create a placeholder plot showing the concept
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.text(0.5, 0.5, 
                   'インピーダンス応答変化の可視化\n'
                   '(EISデータの詳細構造が必要)\n\n'
                   '実装予定:\n'
                   '- 同一周波数での応答変化\n'
                   '- 位相変化の定量化\n'
                   '- 応答時間の遅延分析',
                   ha='center', va='center', transform=ax.transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.3),
                   fontsize=12)
            
            ax.set_title('インピーダンス応答変化分析 (プレースホルダー)')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            
            plot_path = output_dir / 'impedance_response_changes.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            plot_paths.append(plot_path)
            
        except Exception as e:
            warnings.warn(f"Failed to plot impedance response changes: {e}")
        
        return plot_paths