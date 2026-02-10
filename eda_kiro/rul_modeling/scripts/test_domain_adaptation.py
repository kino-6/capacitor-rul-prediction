#!/usr/bin/env python3
"""
Test script for domain adaptation framework

This script tests the domain adaptation functionality including:
- Transfer learning for new capacitor types
- Domain-specific feature engineering
- Few-shot learning adaptation
- Automated model adaptation
"""

import sys
import logging
from pathlib import Path
import numpy as np
from datetime import datetime
import time
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from true_rul.domain_adaptation import (
    DomainAdaptationFramework, DomainInfo, TransferLearningAdapter,
    FewShotLearningAdapter, CapacitorTypeFeatureExtractor,
    OperatingConditionFeatureExtractor, create_domain_adaptation_framework
)
from true_rul.data_structures import CycleData, TrainingDataset
from true_rul.rul_regression_model import RULRegressionModel
from true_rul.fast_ensemble_detector import FastEnsembleAnomalyDetector

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_domain_data(domain_type: str, n_samples: int = 50) -> tuple:
    """Create synthetic data for different domains"""
    np.random.seed(42)  # For reproducibility
    
    n_features = 55
    
    if domain_type == "electrolytic":
        # Electrolytic capacitors - higher ESR, temperature sensitive
        base_features = np.random.randn(n_samples, n_features)
        # Add domain-specific characteristics
        base_features[:, 0] *= 1.5  # Higher ESR proxy
        base_features[:, 1] += 0.3  # Temperature sensitivity
        rul_labels = np.random.randint(50, 150, n_samples).astype(float)
        
    elif domain_type == "ceramic":
        # Ceramic capacitors - stable, voltage coefficient effects
        base_features = np.random.randn(n_samples, n_features) * 0.8  # More stable
        base_features[:, 2] += 0.2  # Voltage coefficient effect
        rul_labels = np.random.randint(100, 300, n_samples).astype(float)
        
    elif domain_type == "tantalum":
        # Tantalum capacitors - surge sensitive
        base_features = np.random.randn(n_samples, n_features)
        base_features[:, 3] *= 2.0  # Surge sensitivity
        rul_labels = np.random.randint(30, 120, n_samples).astype(float)
        
    elif domain_type == "high_temp":
        # High temperature operating condition
        base_features = np.random.randn(n_samples, n_features)
        base_features += 0.5  # Shift due to temperature
        rul_labels = np.random.randint(20, 80, n_samples).astype(float)  # Shorter life
        
    else:  # Default/base domain
        base_features = np.random.randn(n_samples, n_features)
        rul_labels = np.random.randint(50, 200, n_samples).astype(float)
    
    # Create cycle numbers and capacitor IDs
    cycle_numbers = np.arange(1, n_samples + 1)
    capacitor_ids = [f"{domain_type}_cap_{i//10}" for i in range(n_samples)]
    
    # Create training dataset
    dataset = TrainingDataset(
        capacitor_ids=capacitor_ids,
        features=base_features,
        rul_labels=rul_labels,
        cycle_numbers=cycle_numbers,
        anomaly_labels=np.zeros(n_samples)  # All normal for simplicity
    )
    
    return dataset, domain_type


def create_synthetic_cycle_data(domain_type: str) -> CycleData:
    """Create synthetic cycle data for feature extraction testing"""
    # Generate voltage time series based on domain type
    time_points = np.linspace(0, 1, 100)
    
    if domain_type == "electrolytic":
        # Higher ESR, more ripple
        vl_series = np.sin(2 * np.pi * time_points) + 0.2 * np.random.randn(100)
        vo_series = 0.7 * vl_series + 0.1 * np.random.randn(100)  # Higher loss
        
    elif domain_type == "ceramic":
        # Very stable response
        vl_series = np.sin(2 * np.pi * time_points) + 0.05 * np.random.randn(100)
        vo_series = 0.95 * vl_series + 0.02 * np.random.randn(100)  # Low loss
        
    elif domain_type == "tantalum":
        # Sensitive to surge
        vl_series = np.sin(2 * np.pi * time_points)
        surge_points = np.random.choice(100, 5, replace=False)
        vl_series[surge_points] *= 1.5  # Surge events
        vo_series = 0.8 * vl_series + 0.05 * np.random.randn(100)
        
    else:  # Default
        vl_series = np.sin(2 * np.pi * time_points) + 0.1 * np.random.randn(100)
        vo_series = 0.85 * vl_series + 0.05 * np.random.randn(100)
    
    return CycleData(
        cycle_number=1,
        vl_series=vl_series,
        vo_series=vo_series,
        timestamp=datetime.now().timestamp()
    )


def test_domain_specific_feature_extraction():
    """Test domain-specific feature extractors"""
    logger.info("Testing domain-specific feature extraction...")
    start_time = time.time()
    
    # Test capacitor type feature extractor
    cap_extractor = CapacitorTypeFeatureExtractor()
    
    capacitor_types = ['electrolytic', 'ceramic', 'tantalum', 'film']
    results = {}
    
    for cap_type in tqdm(capacitor_types, desc="Testing capacitor types"):
        # Create domain info
        domain_info = DomainInfo(
            domain_id=f"{cap_type}_domain",
            domain_type="capacitor_type",
            characteristics={'capacitor_type': cap_type},
            sample_count=1
        )
        
        # Create synthetic cycle data
        cycle_data = create_synthetic_cycle_data(cap_type)
        
        try:
            # Extract features
            features = cap_extractor.extract_domain_features(cycle_data, domain_info)
            results[cap_type] = {
                'success': True,
                'n_features': len(features),
                'feature_range': (float(np.min(features)), float(np.max(features)))
            }
            logger.info(f"{cap_type}: {len(features)} features extracted")
            
        except Exception as e:
            results[cap_type] = {'success': False, 'error': str(e)}
            logger.error(f"Failed to extract features for {cap_type}: {e}")
    
    # Test operating condition feature extractor
    op_extractor = OperatingConditionFeatureExtractor()
    
    conditions = [
        {'temperature': 25.0, 'humidity': 50.0, 'voltage_stress': 1.0},
        {'temperature': 85.0, 'humidity': 80.0, 'voltage_stress': 1.2},
        {'temperature': -20.0, 'humidity': 20.0, 'voltage_stress': 0.8}
    ]
    
    for i, condition in enumerate(tqdm(conditions, desc="Testing operating conditions")):
        domain_info = DomainInfo(
            domain_id=f"condition_{i}",
            domain_type="operating_condition",
            characteristics=condition,
            sample_count=1
        )
        
        cycle_data = create_synthetic_cycle_data("default")
        
        try:
            features = op_extractor.extract_domain_features(cycle_data, domain_info)
            results[f"condition_{i}"] = {
                'success': True,
                'n_features': len(features),
                'conditions': condition
            }
            logger.info(f"Condition {i}: {len(features)} features extracted")
            
        except Exception as e:
            results[f"condition_{i}"] = {'success': False, 'error': str(e)}
            logger.error(f"Failed to extract features for condition {i}: {e}")
    
    test_time = time.time() - start_time
    logger.info(f"Domain-specific feature extraction test completed in {test_time:.2f}s")
    
    # Check if all tests passed
    success_count = sum(1 for r in results.values() if r.get('success', False))
    total_tests = len(results)
    
    return success_count == total_tests


def test_transfer_learning():
    """Test transfer learning adaptation"""
    logger.info("Testing transfer learning...")
    start_time = time.time()
    
    try:
        # Create base model
        logger.info("Creating base model...")
        base_model = RULRegressionModel(model_type="xgboost")
        
        # Create source domain data
        source_data, _ = create_synthetic_domain_data("base", n_samples=30)
        
        # Train base model
        logger.info("Training base model...")
        val_size = 5
        train_size = source_data.n_samples - val_size
        
        base_model.train(
            source_data.features[:train_size],
            source_data.rul_labels[:train_size],
            source_data.features[train_size:],
            source_data.rul_labels[train_size:]
        )
        
        # Create transfer learning adapter
        transfer_adapter = TransferLearningAdapter(base_model)
        
        # Test adaptation to different domains
        target_domains = ['electrolytic', 'ceramic', 'tantalum']
        adaptation_results = {}
        
        for domain in tqdm(target_domains, desc="Testing transfer learning"):
            # Create target domain data
            target_data, _ = create_synthetic_domain_data(domain, n_samples=20)
            
            # Create domain info
            domain_info = DomainInfo(
                domain_id=f"{domain}_domain",
                domain_type="capacitor_type",
                characteristics={'capacitor_type': domain},
                sample_count=target_data.n_samples
            )
            
            # Perform adaptation
            logger.info(f"Adapting to {domain} domain...")
            result = transfer_adapter.adapt_to_domain(
                target_data, domain_info, adaptation_method="fine_tuning"
            )
            
            adaptation_results[domain] = result
            
            if result.success:
                logger.info(f"{domain} adaptation successful - RMSE: {result.performance_metrics.get('rmse', 'N/A'):.2f}")
            else:
                logger.warning(f"{domain} adaptation failed")
        
        test_time = time.time() - start_time
        logger.info(f"Transfer learning test completed in {test_time:.2f}s")
        
        # Check success rate
        successful = sum(1 for r in adaptation_results.values() if r.success)
        return successful == len(target_domains)
        
    except Exception as e:
        logger.error(f"Transfer learning test failed: {e}")
        return False


def test_few_shot_learning():
    """Test few-shot learning adaptation"""
    logger.info("Testing few-shot learning...")
    start_time = time.time()
    
    try:
        # Create base model
        base_model = RULRegressionModel(model_type="xgboost")
        
        # Create and train base model
        base_data, _ = create_synthetic_domain_data("base", n_samples=30)
        val_size = 5
        train_size = base_data.n_samples - val_size
        
        base_model.train(
            base_data.features[:train_size],
            base_data.rul_labels[:train_size],
            base_data.features[train_size:],
            base_data.rul_labels[train_size:]
        )
        
        # Create few-shot adapter
        few_shot_adapter = FewShotLearningAdapter(base_model)
        
        # Test few-shot learning on different domains
        target_domains = ['high_temp', 'ceramic']
        few_shot_results = {}
        
        for domain in tqdm(target_domains, desc="Testing few-shot learning"):
            # Create small target dataset
            target_data, _ = create_synthetic_domain_data(domain, n_samples=10)
            
            # Use first 5 samples as support set
            support_features = target_data.features[:5]
            support_labels = target_data.rul_labels[:5]
            
            few_shot_adapter.add_support_set(domain, support_features, support_labels)
            
            # Use remaining samples as query set
            query_features = target_data.features[5:]
            
            # Perform few-shot adaptation
            logger.info(f"Few-shot adaptation to {domain} domain...")
            result = few_shot_adapter.adapt_with_few_shots(domain, query_features, k_shot=3)
            
            few_shot_results[domain] = result
            
            if result.success:
                logger.info(f"{domain} few-shot adaptation successful")
            else:
                logger.warning(f"{domain} few-shot adaptation failed")
        
        test_time = time.time() - start_time
        logger.info(f"Few-shot learning test completed in {test_time:.2f}s")
        
        # Check success rate
        successful = sum(1 for r in few_shot_results.values() if r.success)
        return successful == len(target_domains)
        
    except Exception as e:
        logger.error(f"Few-shot learning test failed: {e}")
        return False


def test_domain_adaptation_framework():
    """Test complete domain adaptation framework"""
    logger.info("Testing domain adaptation framework...")
    start_time = time.time()
    
    try:
        # Create base model
        base_model = RULRegressionModel(model_type="xgboost")
        
        # Create and train base model
        base_data, _ = create_synthetic_domain_data("base", n_samples=30)
        val_size = 5
        train_size = base_data.n_samples - val_size
        
        base_model.train(
            base_data.features[:train_size],
            base_data.rul_labels[:train_size],
            base_data.features[train_size:],
            base_data.rul_labels[train_size:]
        )
        
        # Create framework
        framework = create_domain_adaptation_framework(base_model)
        
        # Register domains
        domains_to_register = [
            DomainInfo(
                domain_id="electrolytic_caps",
                domain_type="capacitor_type",
                characteristics={'capacitor_type': 'electrolytic', 'rated_voltage': 25.0},
                sample_count=0
            ),
            DomainInfo(
                domain_id="high_temp_ops",
                domain_type="operating_condition",
                characteristics={'temperature': 85.0, 'humidity': 60.0, 'voltage_stress': 1.1},
                sample_count=0
            )
        ]
        
        for domain in domains_to_register:
            framework.register_domain(domain)
        
        # Test adaptation to registered domains
        adaptation_results = {}
        
        for domain_info in tqdm(domains_to_register, desc="Testing framework adaptation"):
            # Create target data
            if domain_info.domain_type == "capacitor_type":
                target_data, _ = create_synthetic_domain_data("electrolytic", n_samples=25)
            else:
                target_data, _ = create_synthetic_domain_data("high_temp", n_samples=25)
            
            # Perform adaptation
            logger.info(f"Framework adapting to {domain_info.domain_id}...")
            result = framework.adapt_to_domain(
                domain_info.domain_id, target_data, adaptation_method="auto"
            )
            
            adaptation_results[domain_info.domain_id] = result
            
            if result.success:
                logger.info(f"{domain_info.domain_id} adaptation successful using {result.adaptation_method}")
            else:
                logger.warning(f"{domain_info.domain_id} adaptation failed")
        
        # Test domain similarity
        similarity = framework.get_domain_similarity("electrolytic_caps", "high_temp_ops")
        logger.info(f"Domain similarity: {similarity:.2f}")
        
        # Get adaptation summary
        summary = framework.get_adaptation_summary()
        logger.info(f"Adaptation summary: {summary}")
        
        test_time = time.time() - start_time
        logger.info(f"Domain adaptation framework test completed in {test_time:.2f}s")
        
        # Check success rate
        successful = sum(1 for r in adaptation_results.values() if r.success)
        return successful == len(domains_to_register)
        
    except Exception as e:
        logger.error(f"Domain adaptation framework test failed: {e}")
        return False


def test_framework_persistence():
    """Test saving and loading framework state"""
    logger.info("Testing framework persistence...")
    start_time = time.time()
    
    try:
        # Create base model
        base_model = RULRegressionModel(model_type="xgboost")
        
        # Create minimal training data
        base_data, _ = create_synthetic_domain_data("base", n_samples=20)
        base_model.train(
            base_data.features[:15],
            base_data.rul_labels[:15],
            base_data.features[15:],
            base_data.rul_labels[15:]
        )
        
        # Create framework and register domain
        framework = create_domain_adaptation_framework(base_model)
        
        domain_info = DomainInfo(
            domain_id="test_domain",
            domain_type="capacitor_type",
            characteristics={'capacitor_type': 'test'},
            sample_count=10
        )
        framework.register_domain(domain_info)
        
        # Save framework
        save_path = Path("test_domain_framework.json")
        framework.save_framework(save_path)
        logger.info("Framework saved successfully")
        
        # Create new framework and load state
        new_framework = create_domain_adaptation_framework(base_model)
        new_framework.load_framework(save_path)
        logger.info("Framework loaded successfully")
        
        # Verify loaded state
        assert "test_domain" in new_framework.domains
        assert new_framework.domains["test_domain"].domain_type == "capacitor_type"
        
        # Clean up
        save_path.unlink()
        
        test_time = time.time() - start_time
        logger.info(f"Framework persistence test completed in {test_time:.2f}s")
        
        return True
        
    except Exception as e:
        logger.error(f"Framework persistence test failed: {e}")
        return False


def main():
    """Run all domain adaptation tests"""
    logger.info("Starting domain adaptation tests...")
    total_start_time = time.time()
    
    test_results = {}
    
    # Test domain-specific feature extraction
    try:
        logger.info("=" * 50)
        logger.info("TEST 1/5: Domain-Specific Feature Extraction")
        logger.info("=" * 50)
        test_results['feature_extraction'] = test_domain_specific_feature_extraction()
    except Exception as e:
        logger.error(f"Feature extraction test failed: {e}")
        test_results['feature_extraction'] = False
    
    # Test transfer learning
    try:
        logger.info("=" * 50)
        logger.info("TEST 2/5: Transfer Learning")
        logger.info("=" * 50)
        test_results['transfer_learning'] = test_transfer_learning()
    except Exception as e:
        logger.error(f"Transfer learning test failed: {e}")
        test_results['transfer_learning'] = False
    
    # Test few-shot learning
    try:
        logger.info("=" * 50)
        logger.info("TEST 3/5: Few-Shot Learning")
        logger.info("=" * 50)
        test_results['few_shot_learning'] = test_few_shot_learning()
    except Exception as e:
        logger.error(f"Few-shot learning test failed: {e}")
        test_results['few_shot_learning'] = False
    
    # Test domain adaptation framework
    try:
        logger.info("=" * 50)
        logger.info("TEST 4/5: Domain Adaptation Framework")
        logger.info("=" * 50)
        test_results['framework'] = test_domain_adaptation_framework()
    except Exception as e:
        logger.error(f"Framework test failed: {e}")
        test_results['framework'] = False
    
    # Test framework persistence
    try:
        logger.info("=" * 50)
        logger.info("TEST 5/5: Framework Persistence")
        logger.info("=" * 50)
        test_results['persistence'] = test_framework_persistence()
    except Exception as e:
        logger.error(f"Persistence test failed: {e}")
        test_results['persistence'] = False
    
    total_time = time.time() - total_start_time
    
    # Summary
    logger.info("=" * 60)
    logger.info("DOMAIN ADAPTATION TEST RESULTS")
    logger.info("=" * 60)
    
    for test_name, result in test_results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"{test_name:20s}: {status}")
    
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    logger.info(f"Total execution time: {total_time:.2f}s")
    
    if passed_tests == total_tests:
        logger.info("🎉 All domain adaptation tests passed!")
        return True
    else:
        logger.warning("⚠️  Some domain adaptation tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)