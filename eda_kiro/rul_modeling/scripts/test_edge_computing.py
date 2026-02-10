#!/usr/bin/env python3
"""
Test script for edge computing optimization functionality

This script tests the edge computing features including:
- Lightweight model creation
- Federated learning client
- Offline prediction buffering
- Edge RUL predictor
"""

import sys
import os
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from true_rul.edge_computing import (
    EdgeDeviceConfig,
    LightweightModelFactory,
    FederatedLearningClient,
    OfflinePredictionBuffer,
    EdgeRULPredictor,
    create_edge_deployment_package,
    estimate_edge_resource_usage
)
from true_rul.data_structures import CycleData, PredictionResult
import xgboost as xgb
import torch
import torch.nn as nn


def test_lightweight_model_factory():
    """Test lightweight model factory"""
    print("Testing LightweightModelFactory...")
    
    factory = LightweightModelFactory(target_memory_mb=50)
    
    # Test lightweight XGBoost creation
    xgb_model = factory.create_lightweight_xgboost(
        xgb.XGBRegressor(),
        max_depth=3,
        n_estimators=30
    )
    assert isinstance(xgb_model, xgb.XGBRegressor)
    assert xgb_model.max_depth == 3
    assert xgb_model.n_estimators == 30
    print("✓ Lightweight XGBoost model created")
    
    # Test lightweight neural network creation
    nn_model = factory.create_lightweight_neural_network(
        input_dim=55,
        hidden_dim=32,
        num_layers=2
    )
    assert isinstance(nn_model, nn.Module)
    print("✓ Lightweight neural network created")
    
    # Test lightweight ensemble creation
    ensemble = factory.create_lightweight_ensemble(
        input_dim=55,
        use_xgboost=True,
        use_neural_net=True
    )
    assert 'xgboost' in ensemble
    assert 'neural_net' in ensemble
    print("✓ Lightweight ensemble created")
    
    # Test edge optimization
    sample_input = torch.randn(1, 55)
    optimized = factory.optimize_for_edge(
        model=nn_model,
        model_type="pytorch",
        example_input=sample_input
    )
    assert 'original_model' in optimized
    assert 'optimizations_applied' in optimized
    print("✓ Edge optimization completed")
    
    print("LightweightModelFactory tests passed!\n")


def test_federated_learning_client():
    """Test federated learning client"""
    print("Testing FederatedLearningClient...")
    
    client = FederatedLearningClient(
        device_id="edge_device_001",
        server_url="http://localhost:8000/federated",
        model_update_threshold=5
    )
    
    # Test neural network update
    model = nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 1)
    )
    
    # Generate synthetic training data
    training_data = np.random.randn(20, 10)
    training_labels = np.random.randn(20)
    
    # Test local model update
    updated_model = client.update_local_model(
        model=model,
        training_data=training_data,
        training_labels=training_labels,
        learning_rate=0.01
    )
    assert isinstance(updated_model, nn.Module)
    print("✓ Local model update completed")
    
    # Test model update preparation
    for _ in range(5):  # Trigger threshold
        client.update_local_model(
            model=updated_model,
            training_data=training_data[:5],
            training_labels=training_labels[:5]
        )
    
    model_update = client.get_model_update()
    if model_update:
        assert 'parameters' in model_update
        assert 'device_id' in model_update
        print("✓ Model update prepared for sharing")
    
    print("FederatedLearningClient tests passed!\n")


def test_offline_prediction_buffer():
    """Test offline prediction buffer"""
    print("Testing OfflinePredictionBuffer...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        buffer_path = os.path.join(temp_dir, "test_buffer.json")
        buffer = OfflinePredictionBuffer(
            buffer_size=10,
            storage_path=buffer_path
        )
        
        # Create test prediction result
        prediction_result = PredictionResult(
            rul_cycles=100,
            rul_confidence_lower=90,
            rul_confidence_upper=110,
            degradation_score=0.3,
            degradation_stage="early_degradation",
            anomaly_flag=False,
            anomaly_score=0.1,
            feature_importance={"feature_1": 0.5, "feature_2": 0.3},
            timestamp=1234567890.0,
            model_version="1.0.0"
        )
        
        input_data = {
            'cycle_number': 50,
            'capacitor_id': 'C1',
            'features': [1.0, 2.0, 3.0]
        }
        
        # Test adding predictions
        for i in range(5):
            buffer.add_prediction(prediction_result, input_data)
        
        pending = buffer.get_pending_predictions()
        assert len(pending) == 5
        print("✓ Predictions added to buffer")
        
        # Test clearing synced predictions
        buffer.clear_synced_predictions(3)
        remaining = buffer.get_pending_predictions()
        assert len(remaining) == 2
        print("✓ Synced predictions cleared")
        
        # Test buffer size limit
        for i in range(15):  # Exceed buffer size
            buffer.add_prediction(prediction_result, input_data)
        
        final_buffer = buffer.get_pending_predictions()
        assert len(final_buffer) <= 10  # Should not exceed buffer size
        print("✓ Buffer size limit enforced")
    
    print("OfflinePredictionBuffer tests passed!\n")


def test_edge_rul_predictor():
    """Test edge RUL predictor"""
    print("Testing EdgeRULPredictor...")
    
    # Create edge device configuration
    config = EdgeDeviceConfig(
        device_id="test_edge_device",
        max_memory_mb=256,
        max_model_size_mb=25,
        cpu_cores=2,
        has_gpu=False,
        network_bandwidth_mbps=5.0,
        sync_interval_hours=1,
        offline_buffer_size=100
    )
    
    # Initialize edge predictor
    predictor = EdgeRULPredictor(
        config=config,
        enable_federated_learning=True
    )
    
    # Test prediction
    cycle_data = CycleData(
        cycle_number=50,
        vl_series=np.random.randn(100),
        vo_series=np.random.randn(100),
        timestamp=1234567890.0
    )
    
    features = np.random.randn(55)  # 55 features
    
    # Test online prediction
    result = predictor.predict(
        cycle_data=cycle_data,
        features=features,
        online=True
    )
    assert isinstance(result, PredictionResult)
    assert result.rul_cycles >= 0
    print("✓ Online prediction completed")
    
    # Test offline prediction
    result_offline = predictor.predict(
        cycle_data=cycle_data,
        features=features,
        online=False
    )
    assert isinstance(result_offline, PredictionResult)
    print("✓ Offline prediction completed and buffered")
    
    # Test device status
    status = predictor.get_device_status()
    assert 'device_id' in status
    assert 'sync_status' in status
    assert 'system_metrics' in status
    print("✓ Device status retrieved")
    
    # Test memory optimization
    predictor.optimize_memory_usage()
    print("✓ Memory usage optimized")
    
    print("EdgeRULPredictor tests passed!\n")


def test_edge_deployment_package():
    """Test edge deployment package creation"""
    print("Testing edge deployment package creation...")
    
    # Create a simple model for testing
    model = xgb.XGBRegressor(n_estimators=10, max_depth=3)
    X_dummy = np.random.randn(100, 55)
    y_dummy = np.random.randn(100)
    model.fit(X_dummy, y_dummy)
    
    config = EdgeDeviceConfig(
        device_id="deployment_test_device",
        max_memory_mb=128,
        max_model_size_mb=20
    )
    
    with tempfile.TemporaryDirectory() as temp_dir:
        package_dir = create_edge_deployment_package(
            model=model,
            model_type="xgboost",
            config=config,
            output_dir=temp_dir
        )
        
        # Check if required files are created
        assert os.path.exists(os.path.join(package_dir, "edge_model.pkl"))
        assert os.path.exists(os.path.join(package_dir, "edge_config.json"))
        assert os.path.exists(os.path.join(package_dir, "deploy_edge.py"))
        print("✓ Edge deployment package created")
    
    print("Edge deployment package tests passed!\n")


def test_resource_usage_estimation():
    """Test resource usage estimation"""
    print("Testing resource usage estimation...")
    
    # Create test model
    model = xgb.XGBRegressor(n_estimators=10, max_depth=3)
    X_dummy = np.random.randn(100, 55)
    y_dummy = np.random.randn(100)
    model.fit(X_dummy, y_dummy)
    
    sample_input = np.random.randn(55)
    
    usage_stats = estimate_edge_resource_usage(
        model=model,
        model_type="xgboost",
        sample_input=sample_input
    )
    
    assert 'model_size_mb' in usage_stats
    assert 'prediction_time_ms' in usage_stats
    assert 'memory_usage_mb' in usage_stats
    assert 'estimated_throughput_per_sec' in usage_stats
    
    print(f"✓ Resource usage estimated:")
    print(f"  Model size: {usage_stats['model_size_mb']:.2f} MB")
    print(f"  Prediction time: {usage_stats['prediction_time_ms']:.2f} ms")
    print(f"  Memory usage: {usage_stats['memory_usage_mb']:.2f} MB")
    print(f"  Throughput: {usage_stats['estimated_throughput_per_sec']:.1f} predictions/sec")
    
    print("Resource usage estimation tests passed!\n")


def main():
    """Run all edge computing tests"""
    print("=" * 60)
    print("EDGE COMPUTING OPTIMIZATION TESTS")
    print("=" * 60)
    
    try:
        test_lightweight_model_factory()
        test_federated_learning_client()
        test_offline_prediction_buffer()
        test_edge_rul_predictor()
        test_edge_deployment_package()
        test_resource_usage_estimation()
        
        print("=" * 60)
        print("ALL EDGE COMPUTING TESTS PASSED! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())