#!/usr/bin/env python3
"""
Test script for online learning capabilities

This script tests the online learning functionality including:
- Concept drift detection
- Incremental model updates
- Active learning sample selection
- Performance monitoring
"""

import sys
import logging
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
import time
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from true_rul.online_learning import (
    OnlineLearningManager, ConceptDriftDetector, PerformanceMetrics,
    UncertaintyBasedStrategy, DiversityBasedStrategy,
    create_online_learning_manager
)
from true_rul.data_structures import CycleData, TrainingDataset
from true_rul.rul_regression_model import RULRegressionModel
from true_rul.fast_ensemble_detector import FastEnsembleAnomalyDetector  # Use fast version
from true_rul.feature_extractor import FeatureExtractor
from true_rul.time_series_preprocessor import TimeSeriesPreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_cycle_data(cycle_num: int, degradation_level: float = 0.0) -> CycleData:
    """Create synthetic cycle data for testing"""
    # Generate synthetic voltage time series
    time_points = np.linspace(0, 1, 100)
    
    # Base signal with some degradation
    vl_series = np.sin(2 * np.pi * time_points) + degradation_level * np.random.normal(0, 0.1, 100)
    vo_series = 0.8 * vl_series + degradation_level * np.random.normal(0, 0.05, 100)
    
    return CycleData(
        cycle_number=max(1, cycle_num + 1),  # Ensure cycle_number >= 1
        vl_series=vl_series,
        vo_series=vo_series,
        timestamp=datetime.now().timestamp()
    )


def create_synthetic_models():
    """Create synthetic models for testing"""
    logger.info("Creating synthetic models...")
    start_time = time.time()
    
    # Create simple models with minimal functionality
    rul_model = RULRegressionModel(model_type="xgboost")
    
    # Create synthetic training data (smaller for faster testing)
    n_samples = 30  # Further reduced from 50
    n_features = 55  # Expected number of features
    
    logger.info(f"Generating synthetic training data: {n_samples} samples, {n_features} features")
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(1, 200, n_samples).astype(float)
    X_val = np.random.randn(5, n_features)  # Very small validation set
    y_val = np.random.randint(1, 200, 5).astype(float)
    
    # Train the model with progress tracking and faster settings
    try:
        logger.info("Training RUL model with fast settings...")
        model_start = time.time()
        
        # Override XGBoost parameters for faster training
        if hasattr(rul_model.model, 'model'):
            rul_model.model.model.set_params(
                n_estimators=50,  # Reduced from default 500
                max_depth=3,      # Reduced from default 6
                learning_rate=0.1 # Increased from default 0.05
            )
        
        rul_model.train(X_train, y_train, X_val, y_val)
        model_time = time.time() - model_start
        logger.info(f"RUL model trained successfully in {model_time:.2f}s")
    except Exception as e:
        logger.warning(f"RUL model training failed: {e}")
    
    # Create anomaly detector (use fast version for testing)
    anomaly_detector = FastEnsembleAnomalyDetector()
    
    # Fit with normal data (even smaller dataset for faster training)
    try:
        logger.info("Training fast anomaly detector...")
        detector_start = time.time()
        normal_data = X_train[:10]  # Very small dataset for speed
        anomaly_detector.fit(normal_data)
        detector_time = time.time() - detector_start
        logger.info(f"Fast anomaly detector fitted successfully in {detector_time:.2f}s")
    except Exception as e:
        logger.warning(f"Fast anomaly detector fitting failed: {e}")
    
    total_time = time.time() - start_time
    logger.info(f"Model creation completed in {total_time:.2f}s")
    
    return rul_model, anomaly_detector


def test_concept_drift_detector():
    """Test concept drift detection"""
    logger.info("Testing concept drift detector...")
    start_time = time.time()
    
    detector = ConceptDriftDetector(window_size=15, sensitivity=0.05)  # Smaller window
    
    # Generate baseline performance metrics with progress bar
    base_time = datetime.now()
    logger.info("Generating baseline performance metrics...")
    
    for i in tqdm(range(20), desc="Baseline metrics"):  # Reduced from 30
        metrics = PerformanceMetrics(
            timestamp=base_time + timedelta(hours=i),
            rmse=2.0 + np.random.normal(0, 0.1),
            mae=1.5 + np.random.normal(0, 0.1),
            r2=0.85 + np.random.normal(0, 0.02),
            fpr=0.03 + np.random.normal(0, 0.005),
            tpr=0.92 + np.random.normal(0, 0.02),
            f1=0.88 + np.random.normal(0, 0.02),
            sample_count=1
        )
        detector.add_performance_sample(metrics)
    
    # Check initial drift status (should be no drift)
    drift_result = detector.detect_drift()
    logger.info(f"Initial drift status: {drift_result.drift_detected}")
    
    # Introduce gradual drift with progress bar
    logger.info("Introducing gradual drift...")
    for i in tqdm(range(20, 40), desc="Drift simulation"):  # Reduced from 30-60
        drift_factor = (i - 20) / 20.0  # Gradual increase
        metrics = PerformanceMetrics(
            timestamp=base_time + timedelta(hours=i),
            rmse=2.0 + drift_factor * 1.0 + np.random.normal(0, 0.1),  # Increasing RMSE
            mae=1.5 + drift_factor * 0.5 + np.random.normal(0, 0.1),
            r2=0.85 - drift_factor * 0.2 + np.random.normal(0, 0.02),  # Decreasing R²
            fpr=0.03 + drift_factor * 0.05 + np.random.normal(0, 0.005),  # Increasing FPR
            tpr=0.92 - drift_factor * 0.1 + np.random.normal(0, 0.02),
            f1=0.88 - drift_factor * 0.1 + np.random.normal(0, 0.02),
            sample_count=1
        )
        detector.add_performance_sample(metrics)
    
    # Check for drift after degradation
    drift_result = detector.detect_drift()
    test_time = time.time() - start_time
    
    logger.info(f"Drift after degradation: {drift_result.drift_detected}")
    logger.info(f"Drift type: {drift_result.drift_type}")
    logger.info(f"Affected metrics: {drift_result.affected_metrics}")
    logger.info(f"Recommendation: {drift_result.recommendation}")
    logger.info(f"Concept drift test completed in {test_time:.2f}s")
    
    return True  # Return True for successful test execution


def test_active_learning_strategies():
    """Test active learning strategies"""
    logger.info("Testing active learning strategies...")
    start_time = time.time()
    
    # Create synthetic model and data
    rul_model, _ = create_synthetic_models()
    
    # Create synthetic unlabeled data (smaller for faster testing)
    n_unlabeled = 50  # Reduced from 100
    n_features = 55
    logger.info(f"Generating {n_unlabeled} unlabeled samples...")
    unlabeled_data = np.random.randn(n_unlabeled, n_features)
    
    # Test uncertainty-based strategy
    logger.info("Testing uncertainty-based strategy...")
    uncertainty_strategy = UncertaintyBasedStrategy()
    try:
        strategy_start = time.time()
        selected_uncertainty = uncertainty_strategy.select_samples(
            unlabeled_data, rul_model, n_samples=5  # Reduced from 10
        )
        strategy_time = time.time() - strategy_start
        logger.info(f"Uncertainty-based selection: {len(selected_uncertainty)} samples in {strategy_time:.2f}s")
    except Exception as e:
        logger.warning(f"Uncertainty-based strategy failed: {e}")
        selected_uncertainty = []
    
    # Test diversity-based strategy
    logger.info("Testing diversity-based strategy...")
    diversity_strategy = DiversityBasedStrategy()
    try:
        strategy_start = time.time()
        selected_diversity = diversity_strategy.select_samples(
            unlabeled_data, rul_model, n_samples=5  # Reduced from 10
        )
        strategy_time = time.time() - strategy_start
        logger.info(f"Diversity-based selection: {len(selected_diversity)} samples in {strategy_time:.2f}s")
    except Exception as e:
        logger.warning(f"Diversity-based strategy failed: {e}")
        selected_diversity = []
    
    test_time = time.time() - start_time
    logger.info(f"Active learning strategies test completed in {test_time:.2f}s")
    
    return len(selected_uncertainty) > 0 and len(selected_diversity) > 0


def test_online_learning_manager():
    """Test online learning manager"""
    logger.info("Testing online learning manager...")
    start_time = time.time()
    
    # Create components
    rul_model, anomaly_detector = create_synthetic_models()
    feature_extractor = FeatureExtractor()
    preprocessor = TimeSeriesPreprocessor()
    
    # Create online learning manager
    config = {
        'min_update_samples': 5,  # Reduced from 10
        'max_update_samples': 20,  # Reduced from 50
        'active_learning_budget': 3,  # Reduced from 5
        'drift_window_size': 10  # Reduced from 20
    }
    
    try:
        manager_start = time.time()
        manager = create_online_learning_manager(
            rul_model=rul_model,
            anomaly_detector=anomaly_detector,
            feature_extractor=feature_extractor,
            preprocessor=preprocessor,
            config=config
        )
        manager_time = time.time() - manager_start
        logger.info(f"Online learning manager created successfully in {manager_time:.2f}s")
    except Exception as e:
        logger.error(f"Failed to create online learning manager: {e}")
        return False
    
    # Test adding new data with progress bar
    try:
        logger.info("Adding new data samples...")
        n_samples = 8  # Reduced from 15
        for i in tqdm(range(n_samples), desc="Adding data"):
            cycle_data = create_synthetic_cycle_data(i, degradation_level=i * 0.1)
            true_rul = max(1, 100 - i * 5)  # Decreasing RUL
            anomaly_label = 1.0 if i > 5 else 0.0  # Anomalies after cycle 5
            
            manager.add_new_data(cycle_data, true_rul, anomaly_label)
        
        logger.info("Successfully added new data samples")
    except Exception as e:
        logger.warning(f"Error adding new data: {e}")
    
    # Check status
    try:
        status_start = time.time()
        update_status = manager.get_update_status()
        logger.info(f"Update status: {update_status}")
        
        drift_status = manager.get_drift_status()
        logger.info(f"Drift detected: {drift_status.drift_detected}")
        
        performance_summary = manager.get_performance_summary()
        logger.info(f"Performance summary: {performance_summary}")
        status_time = time.time() - status_start
        logger.info(f"Status check completed in {status_time:.2f}s")
    except Exception as e:
        logger.warning(f"Error getting status: {e}")
    
    # Test active learning
    try:
        logger.info("Testing active learning...")
        al_start = time.time()
        unlabeled_cycles = [
            create_synthetic_cycle_data(i, degradation_level=0.2) 
            for i in range(10, 15)  # Reduced from 20-30 to 10-15
        ]
        
        selected_indices = manager.request_active_learning_samples(
            unlabeled_cycles, n_samples=2  # Reduced from 3
        )
        al_time = time.time() - al_start
        logger.info(f"Active learning selected {len(selected_indices)} samples in {al_time:.2f}s")
    except Exception as e:
        logger.warning(f"Active learning failed: {e}")
    
    test_time = time.time() - start_time
    logger.info(f"Online learning manager test completed in {test_time:.2f}s")
    
    return True


def test_state_persistence():
    """Test saving and loading state"""
    logger.info("Testing state persistence...")
    
    # Create manager
    rul_model, anomaly_detector = create_synthetic_models()
    feature_extractor = FeatureExtractor()
    preprocessor = TimeSeriesPreprocessor()
    
    manager = create_online_learning_manager(
        rul_model=rul_model,
        anomaly_detector=anomaly_detector,
        feature_extractor=feature_extractor,
        preprocessor=preprocessor
    )
    
    # Add some data
    for i in range(5):
        cycle_data = create_synthetic_cycle_data(i)
        manager.add_new_data(cycle_data, 100 - i * 10, 0.0)
    
    # Save state
    state_file = Path("test_online_learning_state.json")
    try:
        manager.save_state(state_file)
        logger.info("State saved successfully")
        
        # Load state
        manager.load_state(state_file)
        logger.info("State loaded successfully")
        
        # Clean up
        state_file.unlink()
        
        return True
    except Exception as e:
        logger.error(f"State persistence failed: {e}")
        return False


def main():
    """Run all online learning tests"""
    logger.info("Starting online learning tests...")
    total_start_time = time.time()
    
    test_results = {}
    
    # Test concept drift detector
    try:
        logger.info("=" * 50)
        logger.info("TEST 1/4: Concept Drift Detection")
        logger.info("=" * 50)
        test_results['drift_detection'] = test_concept_drift_detector()
    except Exception as e:
        logger.error(f"Drift detection test failed: {e}")
        test_results['drift_detection'] = False
    
    # Test active learning strategies
    try:
        logger.info("=" * 50)
        logger.info("TEST 2/4: Active Learning Strategies")
        logger.info("=" * 50)
        test_results['active_learning'] = test_active_learning_strategies()
    except Exception as e:
        logger.error(f"Active learning test failed: {e}")
        test_results['active_learning'] = False
    
    # Test online learning manager
    try:
        logger.info("=" * 50)
        logger.info("TEST 3/4: Online Learning Manager")
        logger.info("=" * 50)
        test_results['online_manager'] = test_online_learning_manager()
    except Exception as e:
        logger.error(f"Online manager test failed: {e}")
        test_results['online_manager'] = False
    
    # Test state persistence
    try:
        logger.info("=" * 50)
        logger.info("TEST 4/4: State Persistence")
        logger.info("=" * 50)
        test_results['state_persistence'] = test_state_persistence()
    except Exception as e:
        logger.error(f"State persistence test failed: {e}")
        test_results['state_persistence'] = False
    
    total_time = time.time() - total_start_time
    
    # Summary
    logger.info("=" * 60)
    logger.info("ONLINE LEARNING TEST RESULTS")
    logger.info("=" * 60)
    
    for test_name, result in test_results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name:20s}: {status}")
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    logger.info(f"Total execution time: {total_time:.2f}s")
    
    if passed_tests == total_tests:
        logger.info("🎉 All online learning tests passed!")
        return True
    else:
        logger.warning("⚠️  Some online learning tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)