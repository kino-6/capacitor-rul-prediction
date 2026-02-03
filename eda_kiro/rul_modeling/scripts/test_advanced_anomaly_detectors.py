#!/usr/bin/env python3
"""
Test script for advanced anomaly detection techniques.

This script tests the new Deep SVDD, LOF, GMM, and Advanced Ensemble
detectors on the ES12 dataset and compares their performance.
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
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from true_rul.data_loader import DataLoader
from true_rul.feature_extractor import FeatureExtractor
from true_rul.deep_svdd_detector import create_deep_svdd_detector
from true_rul.lof_detector import create_lof_detector
from true_rul.gmm_detector import create_gmm_detector
from true_rul.advanced_ensemble_detector import create_advanced_ensemble_detector
from true_rul.ensemble_anomaly_detector import EnsembleAnomalyDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AdvancedAnomalyDetectorTester:
    """Test suite for advanced anomaly detection techniques."""
    
    def __init__(self, data_path: str = "data/raw"):
        self.data_path = data_path
        self.data_loader = DataLoader()
        self.feature_extractor = FeatureExtractor()
        self.results = {}
        
    def load_and_prepare_data(self) -> Dict[str, Any]:
        """Load and prepare ES12 dataset for testing."""
        logger.info("Loading and preparing ES12 dataset...")
        
        # Load dataset
        dataset = self.data_loader.load_es12_dataset(self.data_path)
        
        # Extract features for all capacitors
        all_features = []
        all_labels = []
        all_capacitors = []
        all_cycles = []
        
        for capacitor_id, capacitor_data in dataset.items():
            logger.info(f"Processing {capacitor_id}...")
            
            for cycle_data in capacitor_data.cycles:
                # Extract features
                features = self.feature_extractor.extract_all_features(
                    cycle_data.vl_series,
                    cycle_data.vo_series
                )
                
                all_features.append(features)
                all_capacitors.append(capacitor_id)
                all_cycles.append(cycle_data.cycle_number)
                
                # Label: 0 for normal (cycles 1-10), 1 for anomalous (cycles > 10)
                label = 0 if cycle_data.cycle_number <= 10 else 1
                all_labels.append(label)
        
        # Convert to numpy arrays
        X = np.array(all_features)
        y = np.array(all_labels)
        capacitors = np.array(all_capacitors)
        cycles = np.array(all_cycles)
        
        # Get feature names
        feature_names = self.feature_extractor.get_feature_names()
        
        logger.info(f"Dataset prepared: {X.shape[0]} samples, {X.shape[1]} features")
        logger.info(f"Normal samples: {np.sum(y == 0)}, Anomalous samples: {np.sum(y == 1)}")
        
        return {
            'X': X,
            'y': y,
            'capacitors': capacitors,
            'cycles': cycles,
            'feature_names': feature_names
        }
    
    def split_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Split data into train/validation/test sets by capacitor."""
        X, y, capacitors = data['X'], data['y'], data['capacitors']
        
        # Split by capacitor: 6 for train/val, 2 for test
        unique_capacitors = np.unique(capacitors)
        np.random.seed(42)
        np.random.shuffle(unique_capacitors)
        
        train_val_capacitors = unique_capacitors[:6]
        test_capacitors = unique_capacitors[6:]
        
        # Further split train/val: 4 for train, 2 for val
        train_capacitors = train_val_capacitors[:4]
        val_capacitors = train_val_capacitors[4:]
        
        # Create masks
        train_mask = np.isin(capacitors, train_capacitors)
        val_mask = np.isin(capacitors, val_capacitors)
        test_mask = np.isin(capacitors, test_capacitors)
        
        # Split data
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        # For training anomaly detectors, use only normal samples
        normal_mask_train = y_train == 0
        X_train_normal = X_train[normal_mask_train]
        
        logger.info(f"Train: {X_train.shape[0]} samples ({np.sum(y_train == 0)} normal, {np.sum(y_train == 1)} anomalous)")
        logger.info(f"Val: {X_val.shape[0]} samples ({np.sum(y_val == 0)} normal, {np.sum(y_val == 1)} anomalous)")
        logger.info(f"Test: {X_test.shape[0]} samples ({np.sum(y_test == 0)} normal, {np.sum(y_test == 1)} anomalous)")
        logger.info(f"Training normal samples: {X_train_normal.shape[0]}")
        
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
        
        # Create and train detector
        detector = create_deep_svdd_detector(
            input_dim=data['X_train_normal'].shape[1],
            hidden_dims=[32, 16, 8],
            epochs=50,
            batch_size=32,
            learning_rate=0.001
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
            'test_scores': test_scores
        }
        
        logger.info(f"Deep SVDD - Val FPR: {val_metrics['fpr']:.4f}, Test FPR: {test_metrics['fpr']:.4f}")
        
        return results
    
    def test_lof(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test LOF detector."""
        logger.info("Testing LOF detector...")
        
        start_time = time.time()
        
        # Create and train detector
        detector = create_lof_detector(
            n_neighbors=20,
            contamination=0.1,
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
            'test_scores': test_scores
        }
        
        logger.info(f"LOF - Val FPR: {val_metrics['fpr']:.4f}, Test FPR: {test_metrics['fpr']:.4f}")
        
        return results
    
    def test_gmm(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test GMM detector."""
        logger.info("Testing GMM detector...")
        
        start_time = time.time()
        
        # Create and train detector
        detector = create_gmm_detector(
            n_components=3,
            covariance_type="full",
            contamination=0.1
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
            'test_scores': test_scores
        }
        
        logger.info(f"GMM - Val FPR: {val_metrics['fpr']:.4f}, Test FPR: {test_metrics['fpr']:.4f}")
        
        return results
    
    def test_advanced_ensemble(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test Advanced Ensemble detector."""
        logger.info("Testing Advanced Ensemble detector...")
        
        start_time = time.time()
        
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
                'hidden_dims': [32, 16, 8],
                'epochs': 30,  # Reduced for faster testing
                'batch_size': 32
            },
            lof_config={
                'n_neighbors': 20,
                'contamination': 0.1,
                'novelty': True
            },
            gmm_config={
                'n_components': 3,
                'contamination': 0.1
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
            'test_contributions': test_contributions
        }
        
        logger.info(f"Advanced Ensemble - Val FPR: {val_metrics['fpr']:.4f}, Test FPR: {test_metrics['fpr']:.4f}")
        
        return results
    
    def test_baseline_ensemble(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Test baseline ensemble for comparison."""
        logger.info("Testing baseline ensemble detector...")
        
        start_time = time.time()
        
        # Create and train baseline detector
        detector = EnsembleAnomalyDetector()
        detector.fit(data['X_train_normal'])
        
        # Evaluate on validation and test sets
        val_results = detector.predict(data['X_val'])
        val_predictions = val_results['anomaly_flag']
        val_scores = val_results['anomaly_score']
        
        test_results = detector.predict(data['X_test'])
        test_predictions = test_results['anomaly_flag']
        test_scores = test_results['anomaly_score']
        
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
            'test_scores': test_scores
        }
        
        logger.info(f"Baseline Ensemble - Val FPR: {val_metrics['fpr']:.4f}, Test FPR: {test_metrics['fpr']:.4f}")
        
        return results
    
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
        logger.info("Starting comprehensive anomaly detector testing...")
        
        # Load and prepare data
        data = self.load_and_prepare_data()
        split_data = self.split_data(data)
        
        # Test all detectors
        results = {}
        
        try:
            results['deep_svdd'] = self.test_deep_svdd(split_data)
        except Exception as e:
            logger.error(f"Deep SVDD test failed: {e}")
            results['deep_svdd'] = {'error': str(e)}
        
        try:
            results['lof'] = self.test_lof(split_data)
        except Exception as e:
            logger.error(f"LOF test failed: {e}")
            results['lof'] = {'error': str(e)}
        
        try:
            results['gmm'] = self.test_gmm(split_data)
        except Exception as e:
            logger.error(f"GMM test failed: {e}")
            results['gmm'] = {'error': str(e)}
        
        try:
            results['advanced_ensemble'] = self.test_advanced_ensemble(split_data)
        except Exception as e:
            logger.error(f"Advanced Ensemble test failed: {e}")
            results['advanced_ensemble'] = {'error': str(e)}
        
        try:
            results['baseline_ensemble'] = self.test_baseline_ensemble(split_data)
        except Exception as e:
            logger.error(f"Baseline Ensemble test failed: {e}")
            results['baseline_ensemble'] = {'error': str(e)}
        
        # Store data for analysis
        results['data'] = split_data
        
        return results
    
    def generate_comparison_report(self, results: Dict[str, Any]) -> None:
        """Generate comprehensive comparison report."""
        logger.info("Generating comparison report...")
        
        # Create results summary
        summary_data = []
        
        for detector_name, result in results.items():
            if detector_name == 'data' or 'error' in result:
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
        print("ADVANCED ANOMALY DETECTION COMPARISON REPORT")
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
        
        # Save results
        output_dir = Path("output/advanced_anomaly_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df_summary.to_csv(output_dir / "detector_comparison.csv", index=False)
        
        # Create visualizations
        self._create_visualizations(results, output_dir)
        
        logger.info(f"Report saved to {output_dir}")
    
    def _create_visualizations(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Create comparison visualizations."""
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        
        # 1. FPR Comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # FPR comparison
        detectors = []
        val_fprs = []
        test_fprs = []
        
        for detector_name, result in results.items():
            if detector_name == 'data' or 'error' in result:
                continue
            
            detectors.append(detector_name.replace('_', ' ').title())
            val_fprs.append(result.get('val_metrics', {}).get('fpr', 0))
            test_fprs.append(result.get('test_metrics', {}).get('fpr', 0))
        
        x = np.arange(len(detectors))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, val_fprs, width, label='Validation', alpha=0.8)
        axes[0, 0].bar(x + width/2, test_fprs, width, label='Test', alpha=0.8)
        axes[0, 0].set_xlabel('Detector')
        axes[0, 0].set_ylabel('False Positive Rate')
        axes[0, 0].set_title('FPR Comparison')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(detectors, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Target < 5%')
        
        # F1 Score comparison
        val_f1s = []
        test_f1s = []
        
        for detector_name, result in results.items():
            if detector_name == 'data' or 'error' in result:
                continue
            
            val_f1s.append(result.get('val_metrics', {}).get('f1', 0))
            test_f1s.append(result.get('test_metrics', {}).get('f1', 0))
        
        axes[0, 1].bar(x - width/2, val_f1s, width, label='Validation', alpha=0.8)
        axes[0, 1].bar(x + width/2, test_f1s, width, label='Test', alpha=0.8)
        axes[0, 1].set_xlabel('Detector')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].set_title('F1 Score Comparison')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(detectors, rotation=45)
        axes[0, 1].legend()
        
        # AUC-ROC comparison
        val_aucs = []
        test_aucs = []
        
        for detector_name, result in results.items():
            if detector_name == 'data' or 'error' in result:
                continue
            
            val_aucs.append(result.get('val_metrics', {}).get('auc_roc', 0))
            test_aucs.append(result.get('test_metrics', {}).get('auc_roc', 0))
        
        axes[1, 0].bar(x - width/2, val_aucs, width, label='Validation', alpha=0.8)
        axes[1, 0].bar(x + width/2, test_aucs, width, label='Test', alpha=0.8)
        axes[1, 0].set_xlabel('Detector')
        axes[1, 0].set_ylabel('AUC-ROC')
        axes[1, 0].set_title('AUC-ROC Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(detectors, rotation=45)
        axes[1, 0].legend()
        
        # Training time comparison
        training_times = []
        
        for detector_name, result in results.items():
            if detector_name == 'data' or 'error' in result:
                continue
            
            training_times.append(result.get('training_time', 0))
        
        axes[1, 1].bar(detectors, training_times, alpha=0.8)
        axes[1, 1].set_xlabel('Detector')
        axes[1, 1].set_ylabel('Training Time (seconds)')
        axes[1, 1].set_title('Training Time Comparison')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / "detector_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROC Curves
        plt.figure(figsize=(12, 8))
        
        for detector_name, result in results.items():
            if detector_name == 'data' or 'error' in result:
                continue
            
            y_true = results['data']['y_test']
            scores = result.get('test_scores')
            
            if scores is not None:
                try:
                    fpr, tpr, _ = roc_curve(y_true, scores)
                    auc = result.get('test_metrics', {}).get('auc_roc', 0)
                    plt.plot(fpr, tpr, label=f"{detector_name.replace('_', ' ').title()} (AUC = {auc:.3f})")
                except:
                    continue
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(output_dir / "roc_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Visualizations created successfully")


def main():
    """Main function to run the advanced anomaly detector tests."""
    # Initialize tester
    tester = AdvancedAnomalyDetectorTester()
    
    # Run comprehensive test
    results = tester.run_comprehensive_test()
    
    # Generate report
    tester.generate_comparison_report(results)
    
    logger.info("Advanced anomaly detector testing completed!")


if __name__ == "__main__":
    main()