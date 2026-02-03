#!/usr/bin/env python3
"""
Test script for advanced feature extraction techniques.

This script tests the new wavelet, STFT, SPC, and change point
detection features on the ES12 dataset.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging
from pathlib import Path
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from true_rul.data_loader import DataLoader
from true_rul.feature_extractor import FeatureExtractor
from true_rul.advanced_feature_extractor import create_advanced_feature_extractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedFeatureTester:
    """Test suite for advanced feature extraction techniques."""
    
    def __init__(self, data_path: str = "data/raw"):
        self.data_path = data_path
        self.data_loader = DataLoader()
        self.basic_extractor = FeatureExtractor()
        self.advanced_extractor = create_advanced_feature_extractor()
        
    def load_and_extract_features(self) -> Dict[str, Any]:
        """Load dataset and extract both basic and advanced features."""
        logger.info("Loading ES12 dataset and extracting features...")
        
        # Load dataset
        dataset = self.data_loader.load_es12_dataset(self.data_path)
        
        # Extract features for all capacitors
        basic_features = []
        advanced_features = []
        labels = []
        capacitors = []
        cycles = []
        
        for capacitor_id, capacitor_data in dataset.items():
            logger.info(f"Processing {capacitor_id}...")
            
            for cycle_data in capacitor_data.cycles:
                # Extract basic features
                basic_feat = self.basic_extractor.extract_all_features(
                    cycle_data.vl_series,
                    cycle_data.vo_series
                )
                
                # Extract advanced features
                advanced_feat = self.advanced_extractor.extract_all_features(
                    cycle_data.vl_series,
                    cycle_data.vo_series
                )
                
                basic_features.append(basic_feat)
                advanced_features.append(advanced_feat)
                capacitors.append(capacitor_id)
                cycles.append(cycle_data.cycle_number)
                
                # Label: 0 for normal (cycles 1-10), 1 for anomalous (cycles > 10)
                label = 0 if cycle_data.cycle_number <= 10 else 1
                labels.append(label)
        
        # Convert to numpy arrays
        X_basic = np.array(basic_features)
        X_advanced = np.array(advanced_features)
        y = np.array(labels)
        capacitors = np.array(capacitors)
        cycles = np.array(cycles)
        
        # Get feature names
        basic_feature_names = self.basic_extractor.get_feature_names()
        advanced_feature_names = self.advanced_extractor.get_feature_names()
        
        logger.info(f"Basic features: {X_basic.shape[1]} features")
        logger.info(f"Advanced features: {X_advanced.shape[1]} features")
        logger.info(f"Total samples: {X_basic.shape[0]}")
        logger.info(f"Normal samples: {np.sum(y == 0)}, Anomalous samples: {np.sum(y == 1)}")
        
        return {
            'X_basic': X_basic,
            'X_advanced': X_advanced,
            'y': y,
            'capacitors': capacitors,
            'cycles': cycles,
            'basic_feature_names': basic_feature_names,
            'advanced_feature_names': advanced_feature_names
        }
    
    def analyze_feature_distributions(self, data: Dict[str, Any]) -> None:
        """Analyze distributions of advanced features."""
        logger.info("Analyzing feature distributions...")
        
        X_advanced = data['X_advanced']
        y = data['y']
        feature_names = data['advanced_feature_names']
        
        # Create output directory
        output_dir = Path("output/advanced_features_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Compute feature statistics
        normal_mask = y == 0
        anomaly_mask = y == 1
        
        X_normal = X_advanced[normal_mask]
        X_anomaly = X_advanced[anomaly_mask]
        
        # Feature statistics
        stats_data = []
        
        for i, feature_name in enumerate(feature_names):
            normal_values = X_normal[:, i]
            anomaly_values = X_anomaly[:, i]
            
            stats_data.append({
                'Feature': feature_name,
                'Normal_Mean': np.mean(normal_values),
                'Normal_Std': np.std(normal_values),
                'Normal_Min': np.min(normal_values),
                'Normal_Max': np.max(normal_values),
                'Anomaly_Mean': np.mean(anomaly_values),
                'Anomaly_Std': np.std(anomaly_values),
                'Anomaly_Min': np.min(anomaly_values),
                'Anomaly_Max': np.max(anomaly_values),
                'Mean_Difference': abs(np.mean(anomaly_values) - np.mean(normal_values)),
                'Std_Ratio': np.std(anomaly_values) / (np.std(normal_values) + 1e-8)
            })
        
        # Create DataFrame
        df_stats = pd.DataFrame(stats_data)
        
        # Sort by mean difference (most discriminative features)
        df_stats = df_stats.sort_values('Mean_Difference', ascending=False)
        
        # Save statistics
        df_stats.to_csv(output_dir / "feature_statistics.csv", index=False)
        
        # Print top discriminative features
        print("\n" + "="*80)
        print("TOP 10 MOST DISCRIMINATIVE ADVANCED FEATURES")
        print("="*80)
        print(df_stats[['Feature', 'Mean_Difference', 'Std_Ratio']].head(10).to_string(index=False))
        
        # Create visualizations for top features
        self._visualize_top_features(X_advanced, y, feature_names, df_stats, output_dir)
    
    def _visualize_top_features(self, X: np.ndarray, y: np.ndarray, 
                               feature_names: List[str], df_stats: pd.DataFrame,
                               output_dir: Path) -> None:
        """Visualize top discriminative features."""
        # Get top 12 features
        top_features = df_stats.head(12)['Feature'].tolist()
        top_indices = [feature_names.index(feat) for feat in top_features]
        
        # Create subplots
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        axes = axes.flatten()
        
        for i, (feat_idx, feat_name) in enumerate(zip(top_indices, top_features)):
            ax = axes[i]
            
            # Plot histograms
            normal_values = X[y == 0, feat_idx]
            anomaly_values = X[y == 1, feat_idx]
            
            ax.hist(normal_values, bins=30, alpha=0.7, label='Normal', density=True)
            ax.hist(anomaly_values, bins=30, alpha=0.7, label='Anomaly', density=True)
            
            ax.set_title(feat_name.replace('_', ' ').title(), fontsize=10)
            ax.set_xlabel('Feature Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "top_features_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Feature distribution visualizations saved")
    
    def compare_feature_separability(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare separability of basic vs advanced features."""
        logger.info("Comparing feature separability...")
        
        X_basic = data['X_basic']
        X_advanced = data['X_advanced']
        y = data['y']
        
        # Standardize features
        scaler_basic = StandardScaler()
        scaler_advanced = StandardScaler()
        
        X_basic_scaled = scaler_basic.fit_transform(X_basic)
        X_advanced_scaled = scaler_advanced.fit_transform(X_advanced)
        
        # Combine features
        X_combined = np.hstack([X_basic_scaled, X_advanced_scaled])
        
        results = {}
        
        # PCA analysis
        for name, X in [("Basic", X_basic_scaled), 
                       ("Advanced", X_advanced_scaled),
                       ("Combined", X_combined)]:
            
            # Fit PCA
            pca = PCA()
            X_pca = pca.fit_transform(X)
            
            # Compute explained variance
            explained_var = pca.explained_variance_ratio_
            cumsum_var = np.cumsum(explained_var)
            
            # Find components needed for 95% variance
            n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
            
            results[name] = {
                'n_features': X.shape[1],
                'explained_variance_ratio': explained_var,
                'cumulative_variance': cumsum_var,
                'n_components_95': n_components_95,
                'pca_transform': X_pca
            }
        
        # Create PCA visualization
        self._visualize_pca_comparison(results, y)
        
        return results
    
    def _visualize_pca_comparison(self, pca_results: Dict[str, Any], y: np.ndarray) -> None:
        """Visualize PCA comparison."""
        output_dir = Path("output/advanced_features_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot explained variance
        for i, (name, results) in enumerate(pca_results.items()):
            ax = axes[0, i]
            
            # Plot cumulative explained variance
            n_components = min(20, len(results['cumulative_variance']))
            ax.plot(range(1, n_components + 1), 
                   results['cumulative_variance'][:n_components], 
                   'bo-', markersize=4)
            ax.axhline(y=0.95, color='red', linestyle='--', alpha=0.7)
            ax.axvline(x=results['n_components_95'], color='red', linestyle='--', alpha=0.7)
            
            ax.set_title(f'{name} Features\n({results["n_features"]} features)')
            ax.set_xlabel('Number of Components')
            ax.set_ylabel('Cumulative Explained Variance')
            ax.grid(True, alpha=0.3)
            ax.text(0.02, 0.98, f"95% at {results['n_components_95']} components", 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2D PCA projections
        for i, (name, results) in enumerate(pca_results.items()):
            ax = axes[1, i]
            
            X_pca = results['pca_transform']
            
            # Plot first two components
            normal_mask = y == 0
            anomaly_mask = y == 1
            
            ax.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1], 
                      c='blue', alpha=0.6, label='Normal', s=20)
            ax.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1], 
                      c='red', alpha=0.6, label='Anomaly', s=20)
            
            ax.set_title(f'{name} Features - PCA Projection')
            ax.set_xlabel(f'PC1 ({results["explained_variance_ratio"][0]:.2%} var)')
            ax.set_ylabel(f'PC2 ({results["explained_variance_ratio"][1]:.2%} var)')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "pca_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("PCA comparison visualization saved")
    
    def analyze_feature_categories(self, data: Dict[str, Any]) -> None:
        """Analyze different categories of advanced features."""
        logger.info("Analyzing feature categories...")
        
        feature_names = data['advanced_feature_names']
        X_advanced = data['X_advanced']
        y = data['y']
        
        # Categorize features
        categories = {
            'Wavelet': [name for name in feature_names if 'wavelet' in name.lower()],
            'STFT': [name for name in feature_names if any(x in name.lower() for x in ['spectral', 'stft'])],
            'SPC': [name for name in feature_names if 'spc' in name.lower()],
            'Change Point': [name for name in feature_names if 'change' in name.lower()],
            'Cross Series': [name for name in feature_names if 'cross' in name.lower() or 'vl_vo' in name.lower()]
        }
        
        # Print category summary
        print("\n" + "="*60)
        print("ADVANCED FEATURE CATEGORIES")
        print("="*60)
        
        for category, features in categories.items():
            print(f"{category}: {len(features)} features")
            if features:
                print(f"  Examples: {', '.join(features[:3])}")
                if len(features) > 3:
                    print(f"  ... and {len(features) - 3} more")
            print()
        
        # Analyze discriminative power by category
        self._analyze_category_discriminative_power(X_advanced, y, feature_names, categories)
    
    def _analyze_category_discriminative_power(self, X: np.ndarray, y: np.ndarray,
                                             feature_names: List[str], 
                                             categories: Dict[str, List[str]]) -> None:
        """Analyze discriminative power of each feature category."""
        output_dir = Path("output/advanced_features_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        category_scores = {}
        
        for category, features in categories.items():
            if not features:
                continue
            
            # Get feature indices
            feature_indices = [feature_names.index(feat) for feat in features if feat in feature_names]
            
            if not feature_indices:
                continue
            
            # Extract category features
            X_category = X[:, feature_indices]
            
            # Compute discriminative scores (mean difference between classes)
            normal_mask = y == 0
            anomaly_mask = y == 1
            
            X_normal = X_category[normal_mask]
            X_anomaly = X_category[anomaly_mask]
            
            # Compute mean absolute difference for each feature
            mean_diffs = []
            for i in range(X_category.shape[1]):
                diff = abs(np.mean(X_anomaly[:, i]) - np.mean(X_normal[:, i]))
                # Normalize by standard deviation
                std_combined = np.std(X_category[:, i])
                if std_combined > 0:
                    diff = diff / std_combined
                mean_diffs.append(diff)
            
            category_scores[category] = {
                'n_features': len(feature_indices),
                'mean_discriminative_score': np.mean(mean_diffs),
                'max_discriminative_score': np.max(mean_diffs),
                'std_discriminative_score': np.std(mean_diffs),
                'feature_scores': mean_diffs,
                'feature_names': [features[i] for i in range(len(feature_indices))]
            }
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Average discriminative power by category
        categories_list = list(category_scores.keys())
        mean_scores = [category_scores[cat]['mean_discriminative_score'] for cat in categories_list]
        max_scores = [category_scores[cat]['max_discriminative_score'] for cat in categories_list]
        
        x = np.arange(len(categories_list))
        width = 0.35
        
        ax1.bar(x - width/2, mean_scores, width, label='Mean Score', alpha=0.8)
        ax1.bar(x + width/2, max_scores, width, label='Max Score', alpha=0.8)
        
        ax1.set_xlabel('Feature Category')
        ax1.set_ylabel('Discriminative Score')
        ax1.set_title('Discriminative Power by Feature Category')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories_list, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Number of features by category
        n_features = [category_scores[cat]['n_features'] for cat in categories_list]
        
        ax2.bar(categories_list, n_features, alpha=0.8, color='skyblue')
        ax2.set_xlabel('Feature Category')
        ax2.set_ylabel('Number of Features')
        ax2.set_title('Number of Features by Category')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "category_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print summary
        print("\n" + "="*60)
        print("DISCRIMINATIVE POWER BY CATEGORY")
        print("="*60)
        
        # Sort by mean discriminative score
        sorted_categories = sorted(category_scores.items(), 
                                 key=lambda x: x[1]['mean_discriminative_score'], 
                                 reverse=True)
        
        for category, scores in sorted_categories:
            print(f"{category}:")
            print(f"  Features: {scores['n_features']}")
            print(f"  Mean Score: {scores['mean_discriminative_score']:.3f}")
            print(f"  Max Score: {scores['max_discriminative_score']:.3f}")
            print(f"  Std Score: {scores['std_discriminative_score']:.3f}")
            print()
        
        logger.info("Category analysis completed")
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive analysis of advanced features."""
        logger.info("Starting comprehensive advanced feature analysis...")
        
        # Load data and extract features
        data = self.load_and_extract_features()
        
        # Analyze feature distributions
        self.analyze_feature_distributions(data)
        
        # Compare separability
        pca_results = self.compare_feature_separability(data)
        
        # Analyze feature categories
        self.analyze_feature_categories(data)
        
        # Create summary report
        self._create_summary_report(data, pca_results)
        
        return {
            'data': data,
            'pca_results': pca_results
        }
    
    def _create_summary_report(self, data: Dict[str, Any], pca_results: Dict[str, Any]) -> None:
        """Create comprehensive summary report."""
        output_dir = Path("output/advanced_features_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create summary
        summary = {
            'Basic Features': {
                'Count': data['X_basic'].shape[1],
                'PCA 95% Components': pca_results['Basic']['n_components_95']
            },
            'Advanced Features': {
                'Count': data['X_advanced'].shape[1],
                'PCA 95% Components': pca_results['Advanced']['n_components_95']
            },
            'Combined Features': {
                'Count': data['X_basic'].shape[1] + data['X_advanced'].shape[1],
                'PCA 95% Components': pca_results['Combined']['n_components_95']
            }
        }
        
        # Save summary
        with open(output_dir / "summary_report.txt", 'w') as f:
            f.write("ADVANCED FEATURE EXTRACTION ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            for feature_type, stats in summary.items():
                f.write(f"{feature_type}:\n")
                for key, value in stats.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")
            
            f.write("Key Findings:\n")
            f.write(f"- Advanced features add {data['X_advanced'].shape[1]} new dimensions\n")
            f.write(f"- Combined features require {pca_results['Combined']['n_components_95']} components for 95% variance\n")
            f.write(f"- This represents a {pca_results['Combined']['n_components_95'] / summary['Combined Features']['Count']:.1%} dimensionality reduction\n")
        
        logger.info(f"Summary report saved to {output_dir}")


def main():
    """Main function to run the advanced feature analysis."""
    # Initialize tester
    tester = AdvancedFeatureTester()
    
    # Run comprehensive analysis
    results = tester.run_comprehensive_analysis()
    
    logger.info("Advanced feature analysis completed!")


if __name__ == "__main__":
    main()