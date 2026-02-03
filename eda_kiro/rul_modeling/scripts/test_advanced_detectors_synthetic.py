#!/usr/bin/env python3
"""
Test script for advanced anomaly detection techniques using synthetic data.

This script tests the new Deep SVDD, LOF, GMM, and Advanced Ensemble
detectors on synthetic data to verify their functionality.
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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from true_rul.deep_svdd_detector import create_deep_svdd_detector
from true_rul.lof_detector import create_lof_detector
from true_rul.gmm_detector import create_gmm_detector
from true_rul.advanced_ensemble_detector import create_advanced_ensemble_detector
from true_rul.advanced_feature_extractor import create_advanced_feature_extractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SyntheticAnomalyDetectorTester:
    """Test suite for advanced anomaly detection techniques using synthetic data."""
    
    def __init__(self):
        self.results = {}
        
    def generate_synthetic_data(self, n_samples: int = 1000, n_features: int = 20) -> Dict[str, Any]:
        """Generate synthetic data for testing."""
        logger.info(f"Generating synthetic data: {n_samples} samples, {n_features} features")
        
        # Generate normal data (80% of samples)
        n_normal = int(0.8 * n_samples)
        n_anomaly = n_samples - n_normal
        
        # Normal data: multivariate normal distribution
        np.random.seed(42)
        mean_normal = np.zeros(n_features)
        cov_normal = np.eye(n_features) + 0.1 * np.random.randn(n_features, n_features)
        cov_normal = cov_normal @ cov_normal.T  # Make positive definite
        
        X_normal = np.random.multivariate_normal(mean_normal, cov_normal, n_normal)
        y_normal = np.zeros(n_normal)
        
        # Anomalous data: shifted distribution with different covariance
        mean_anomaly = 2 * np.ones(n_features)
        cov_anomaly = 2 * np.eye(n_features)
        
        X_anomaly = np.random.multivariate_normal(mean_anomaly, cov_anomaly, n_anomaly)
        y_anomaly = np.ones(n_anomaly)
        
        # Combine data
        X = np.vstack([X_normal, X_anomaly])
        y = np.hstack([y_normal, y_anomaly])
        
        # Shuffle
        indices = np.random.permutation(len(X))
        X = X[indices]
        y = y[indices]
        
        logger.info(f"Generated data: {np.sum(y == 0)} normal, {np.sum(y == 1)} anomalous samples")
        
        return {
            'X': X,
            'y': y,
            'X_normal': X_normal,  # For training anomaly detectors
            'feature_names': [f'feature_{i}' for i in range(n_features)]
        }
    
    def split_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Split data into train/validation/test sets."""
        X, y = data['X'], data['y']
        n_samples = len(X)
        
        # Split: 60% train, 20% val, 20% test
        n_train = int(0.6 * n_samples)
        n_val = int(0.2 * n_samples)
        
        X_train = X[:n_train]
        y_train = y[:n_train]
        X_val = X[n_train:n_train + n_val]
        y_val = y[n_train:n_train + n_val]
        X_test = X[n_train + n_val:]
        y_test = y[n_train + n_val:]
        
        # For training anomaly detectors, use only normal samples from training set
        normal_mask_train = y_train == 0
        X_train_normal = X_train[normal_mask_train]
        
        logger.info(f"Train: {len(X_train)} samples ({np.sum(y_train == 0)} normal, {np.sum(y_train == 1)} anomalous)")
        logger.info(f"Val: {len(X_val)} samples ({np.sum(y_val == 0)} normal, {np.sum(y_val == 1)} anomalous)")
        logger.info(f"Test: {len(X_test)} samples ({np.sum(y_test == 0)} normal, {np.sum(y_test == 1)} anomalous)")
        logger.info(f"Training normal samples: {len(X_train_normal)}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'X_train_normal': X_train_normal,
            'feature_names': data['feature_names']
        }
    
    def test_deep_svdd(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test Deep SVDD detector."""
        logger.info("Testing Deep SVDD detector...")
        
        start_time = time.time()
        
        try:
            # Create and train detector
            detector = create_deep_svdd_detector(
                input_dim=data['X_train_normal'].shape[1],
                hidden_dims=[16, 8, 4],
                epochs=20,  # Reduced for faster testing
                batch_size=32,
                learning_rate=0.001,
                device="cpu"
            )
            
            detector.fit(data['X_train_normal'])
            
            # Evaluate on validation and test sets
            val_scores = detector.predict_score(data['X_val'])
            val_predictions = detector.predict(data['X_val'])
            
            test_scores = detector.predict_score(data['X_test'])
            test_predictions = detector.predict(data['X_test'])
            
            training_time = time.time() - start_time
            
            # Compute metrics
            val_metrics = self._compute_metrics(data['y_val'], val_predictions, val_scores)
            test_metrics = self._compute_metrics(data['y_test'], test_predictions, test_scores)
            
            results = {
                'detector': detector,
                'training_time': training_time,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'val_scores': val_scores,
                'test_scores': test_scores,
                'success': True
            }
            
            logger.info(f"Deep SVDD - Val FPR: {val_metrics['fpr']:.4f}, Test FPR: {test_metrics['fpr']:.4f}")
            
        except Exception as e:
            logger.error(f"Deep SVDD test failed: {e}")
            results = {'error': str(e), 'success': False}
        
        return results
    
    def test_lof(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test LOF detector."""
        logger.info("Testing LOF detector...")
        
        start_time = time.time()
        
        try:
            # Create and train detector
            detector = create_lof_detector(
                n_neighbors=20,
                contamination=0.2,  # Match our synthetic data ratio
                novelty=True
            )
            
            detector.fit(data['X_train_normal'], data['feature_names'])
            
            # Evaluate on validation and test sets
            val_scores = detector.predict_score(data['X_val'])
            val_predictions = detector.predict(data['X_val'])
            
            test_scores = detector.predict_score(data['X_test'])
            test_predictions = detector.predict(data['X_test'])
            
            training_time = time.time() - start_time
            
            # Compute metrics
            val_metrics = self._compute_metrics(data['y_val'], val_predictions, val_scores)
            test_metrics = self._compute_metrics(data['y_test'], test_predictions, test_scores)
            
            results = {
                'detector': detector,
                'training_time': training_time,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'val_scores': val_scores,
                'test_scores': test_scores,
                'success': True
            }
            
            logger.info(f"LOF - Val FPR: {val_metrics['fpr']:.4f}, Test FPR: {test_metrics['fpr']:.4f}")
            
        except Exception as e:
            logger.error(f"LOF test failed: {e}")
            results = {'error': str(e), 'success': False}
        
        return results
    
    def test_gmm(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test GMM detector."""
        logger.info("Testing GMM detector...")
        
        start_time = time.time()
        
        try:
            # Create and train detector
            detector = create_gmm_detector(
                n_components=2,
                covariance_type="full",
                contamination=0.2
            )
            
            detector.fit(data['X_train_normal'], data['feature_names'])
            
            # Evaluate on validation and test sets
            val_scores = detector.predict_score(data['X_val'])
            val_predictions = detector.predict(data['X_val'])
            
            test_scores = detector.predict_score(data['X_test'])
            test_predictions = detector.predict(data['X_test'])
            
            training_time = time.time() - start_time
            
            # Compute metrics
            val_metrics = self._compute_metrics(data['y_val'], val_predictions, val_scores)
            test_metrics = self._compute_metrics(data['y_test'], test_predictions, test_scores)
            
            results = {
                'detector': detector,
                'training_time': training_time,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'val_scores': val_scores,
                'test_scores': test_scores,
                'success': True
            }
            
            logger.info(f"GMM - Val FPR: {val_metrics['fpr']:.4f}, Test FPR: {test_metrics['fpr']:.4f}")
            
        except Exception as e:
            logger.error(f"GMM test failed: {e}")
            results = {'error': str(e), 'success': False}
        
        return results
    
    def test_advanced_ensemble(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test Advanced Ensemble detector."""
        logger.info("Testing Advanced Ensemble detector...")
        
        start_time = time.time()
        
        try:
            # Create and train detector
            detector = create_advanced_ensemble_detector(
                use_deep_svdd=True,
                use_lof=True,
                use_gmm=True,
                use_isolation_forest=True,
                use_ocsvm=True,
                use_autoencoder=False,  # Skip for faster testing
                parallel_training=True,
                voting_strategy="confidence_weighted",
                deep_svdd_config={
                    'hidden_dims': [16, 8, 4],
                    'epochs': 15,  # Reduced for faster testing
                    'batch_size': 32,
                    'device': 'cpu'
                },
                lof_config={
                    'n_neighbors': 20,
                    'contamination': 0.2,
                    'novelty': True
                },
                gmm_config={
                    'n_components': 2,
                    'contamination': 0.2
                }
            )
            
            detector.fit(data['X_train_normal'], data['feature_names'])
            
            # Evaluate on validation and test sets
            val_scores = detector.predict_score(data['X_val'])
            val_predictions = detector.predict(data['X_val'])
            
            test_scores = detector.predict_score(data['X_test'])
            test_predictions = detector.predict(data['X_test'])
            
            training_time = time.time() - start_time
            
            # Compute metrics
            val_metrics = self._compute_metrics(data['y_val'], val_predictions, val_scores)
            test_metrics = self._compute_metrics(data['y_test'], test_predictions, test_scores)
            
            # Get detector contributions
            val_contributions = detector.get_detector_contributions(data['X_val'])
            test_contributions = detector.get_detector_contributions(data['X_test'])
            
            results = {
                'detector': detector,
                'training_time': training_time,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'val_scores': val_scores,
                'test_scores': test_scores,
                'val_contributions': val_contributions,
                'test_contributions': test_contributions,
                'success': True
            }
            
            logger.info(f"Advanced Ensemble - Val FPR: {val_metrics['fpr']:.4f}, Test FPR: {test_metrics['fpr']:.4f}")
            
        except Exception as e:
            logger.error(f"Advanced Ensemble test failed: {e}")
            results = {'error': str(e), 'success': False}
        
        return results
    
    def test_advanced_features(self, n_samples: int = 200) -> Dict[str, Any]:
        """Test advanced feature extraction."""
        logger.info("Testing advanced feature extraction...")
        
        try:
            # Create feature extractor
            extractor = create_advanced_feature_extractor(
                extract_wavelet_energy=True,
                extract_wavelet_entropy=True,
                extract_spectral_centroid=True,
                extract_spectral_bandwidth=True,
                extract_control_limits=True,
                extract_change_points=True
            )
            
            # Generate synthetic time series data
            np.random.seed(42)
            features_list = []
            
            for i in range(n_samples):
                # Generate synthetic voltage series
                t = np.linspace(0, 1, 100)
                
                # Normal series: sine wave with noise
                if i < n_samples * 0.8:  # 80% normal
                    vl_series = np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(len(t))
                    vo_series = 0.8 * np.sin(2 * np.pi * 5 * t + 0.1) + 0.1 * np.random.randn(len(t))
                else:  # 20% anomalous
                    # Anomalous: different frequency and amplitude
                    vl_series = 1.5 * np.sin(2 * np.pi * 8 * t) + 0.2 * np.random.randn(len(t))
                    vo_series = 0.5 * np.sin(2 * np.pi * 8 * t + 0.3) + 0.2 * np.random.randn(len(t))
                
                # Extract features
                features = extractor.extract_all_features(vl_series, vo_series)
                features_list.append(features)
            
            # Convert to array
            X_features = np.array(features_list)
            y_features = np.array([0] * int(n_samples * 0.8) + [1] * int(n_samples * 0.2))
            
            # Get feature names
            feature_names = extractor.get_feature_names()
            
            logger.info(f"Extracted {X_features.shape[1]} advanced features from {n_samples} time series")
            logger.info(f"Feature categories: {len([name for name in feature_names if 'wavelet' in name])} wavelet, "
                       f"{len([name for name in feature_names if 'spectral' in name])} spectral, "
                       f"{len([name for name in feature_names if 'spc' in name])} SPC, "
                       f"{len([name for name in feature_names if 'change' in name])} change point")
            
            return {
                'X_features': X_features,
                'y_features': y_features,
                'feature_names': feature_names,
                'extractor': extractor,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Advanced feature extraction test failed: {e}")
            return {'error': str(e), 'success': False}
    
    def _compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
        """Compute evaluation metrics."""
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Compute metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # AUC-ROC
        try:
            auc_roc = roc_auc_score(y_true, scores)
        except:
            auc_roc = 0.5
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fpr': fpr,
            'tpr': tpr,
            'auc_roc': auc_roc,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test of all detectors."""
        logger.info("Starting comprehensive synthetic anomaly detector testing...")
        
        # Generate synthetic data
        data = self.generate_synthetic_data(n_samples=1000, n_features=20)
        split_data = self.split_data(data)
        
        # Test all detectors
        results = {}
        
        # Test individual detectors
        results['deep_svdd'] = self.test_deep_svdd(split_data)
        results['lof'] = self.test_lof(split_data)
        results['gmm'] = self.test_gmm(split_data)
        results['advanced_ensemble'] = self.test_advanced_ensemble(split_data)
        
        # Test advanced features
        results['advanced_features'] = self.test_advanced_features()
        
        # Store data for analysis
        results['data'] = split_data
        
        return results
    
    def generate_comparison_report(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive comparison report."""
        logger.info("Generating comparison report...")
        
        # Create results summary
        summary_data = []
        
        detector_names = ['deep_svdd', 'lof', 'gmm', 'advanced_ensemble']
        
        for detector_name in detector_names:
            result = results.get(detector_name, {})
            
            if not result.get('success', False):
                continue
            
            val_metrics = result.get('val_metrics', {})
            test_metrics = result.get('test_metrics', {})
            
            summary_data.append({
                'Detector': detector_name.replace('_', ' ').title(),
                'Training Time (s)': result.get('training_time', 0),
                'Val FPR': val_metrics.get('fpr', 0),
                'Val Precision': val_metrics.get('precision', 0),
                'Val Recall': val_metrics.get('recall', 0),
                'Val F1': val_metrics.get('f1', 0),
                'Val AUC-ROC': val_metrics.get('auc_roc', 0),
                'Test FPR': test_metrics.get('fpr', 0),
                'Test Precision': test_metrics.get('precision', 0),
                'Test Recall': test_metrics.get('recall', 0),
                'Test F1': test_metrics.get('f1', 0),
                'Test AUC-ROC': test_metrics.get('auc_roc', 0)
            })
        
        # Create DataFrame
        df_summary = pd.DataFrame(summary_data)
        
        # Print summary
        print("\n" + "="*80)
        print("ADVANCED ANOMALY DETECTION SYNTHETIC TEST REPORT")
        print("="*80)
        print(df_summary.to_string(index=False, float_format='%.4f'))
        
        # Highlight best performers
        print("\n" + "-"*50)
        print("BEST PERFORMERS:")
        print("-"*50)
        
        if not df_summary.empty:
            best_fpr = df_summary.loc[df_summary['Test FPR'].idxmin()]
            best_f1 = df_summary.loc[df_summary['Test F1'].idxmax()]
            best_auc = df_summary.loc[df_summary['Test AUC-ROC'].idxmax()]
            
            print(f"Lowest Test FPR: {best_fpr['Detector']} ({best_fpr['Test FPR']:.4f})")
            print(f"Highest Test F1: {best_f1['Detector']} ({best_f1['Test F1']:.4f})")
            print(f"Highest Test AUC-ROC: {best_auc['Detector']} ({best_auc['Test AUC-ROC']:.4f})")
        
        # Advanced features summary
        if results.get('advanced_features', {}).get('success', False):
            feat_result = results['advanced_features']
            print(f"\nAdvanced Features: {feat_result['X_features'].shape[1]} features extracted")
            print(f"Feature categories detected in names: {len(feat_result['feature_names'])} total")
        
        # Create output directory
        output_dir = Path("output/synthetic_anomaly_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save results
        if not df_summary.empty:
            df_summary.to_csv(output_dir / "synthetic_detector_comparison.csv", index=False)
        
        logger.info(f"Synthetic test report saved to {output_dir}")


def main():
    """Main function to run the synthetic anomaly detector tests."""
    # Initialize tester
    tester = SyntheticAnomalyDetectorTester()
    
    # Run comprehensive test
    results = tester.run_comprehensive_test()
    
    # Generate report
    tester.generate_comparison_report(results)
    
    logger.info("Synthetic anomaly detector testing completed!")


if __name__ == "__main__":
    main()