#!/usr/bin/env python3
"""
Test script for multi-tenant support and scaling functionality

This script tests the multi-tenant features including:
- Tenant isolation and resource management
- Load balancing and horizontal scaling
- Tenant-specific model customization
- Resource quota enforcement
"""

import sys
import os
import numpy as np
import tempfile
import time
import asyncio
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from true_rul.multi_tenant_scaling import (
    TenantConfig,
    ResourceQuota,
    TenantMetrics,
    TenantIsolationManager,
    LoadBalancer,
    TenantModelManager,
    MultiTenantRULService,
    create_tenant_deployment_config
)
from true_rul.data_structures import CycleData, PredictionResult
from datetime import datetime, timedelta


def test_tenant_isolation_manager():
    """Test tenant isolation manager"""
    print("Testing TenantIsolationManager...")
    
    manager = TenantIsolationManager()
    
    # Test tenant registration
    tenant_config = TenantConfig(
        tenant_id="test_tenant_001",
        tenant_name="Test Company",
        max_requests_per_hour=100,
        max_concurrent_requests=5,
        max_memory_mb=512,
        max_cpu_cores=2,
        priority_level=2
    )
    
    tenant_id = manager.register_tenant(tenant_config)
    assert tenant_id == "test_tenant_001"
    print("✓ Tenant registered successfully")
    
    # Test tenant retrieval
    retrieved_tenant = manager.get_tenant(tenant_id)
    assert retrieved_tenant is not None
    assert retrieved_tenant.tenant_name == "Test Company"
    print("✓ Tenant retrieved successfully")
    
    # Test resource quota checking
    request_id = "req_001"
    can_proceed = manager.check_resource_quota(tenant_id, request_id)
    assert can_proceed is True
    print("✓ Resource quota check passed")
    
    # Test quota limits
    for i in range(5):  # Fill up concurrent request limit
        manager.check_resource_quota(tenant_id, f"req_{i+2}")
    
    # This should fail due to concurrent limit
    over_limit = manager.check_resource_quota(tenant_id, "req_over_limit")
    assert over_limit is False
    print("✓ Concurrent request limit enforced")
    
    # Test resource release
    manager.release_request_resources(tenant_id, request_id)
    print("✓ Resources released successfully")
    
    # Test metrics update
    manager.update_tenant_metrics(
        tenant_id=tenant_id,
        request_success=True,
        response_time_ms=150.0,
        prediction_count=1
    )
    
    metrics = manager.get_tenant_metrics(tenant_id)
    assert metrics is not None
    assert metrics.total_requests == 1
    assert metrics.successful_requests == 1
    print("✓ Tenant metrics updated")
    
    # Test tenant status
    status = manager.get_all_tenant_status()
    assert tenant_id in status
    assert status[tenant_id]['config'] is not None
    print("✓ Tenant status retrieved")
    
    print("TenantIsolationManager tests passed!\n")


def test_load_balancer():
    """Test load balancer"""
    print("Testing LoadBalancer...")
    
    balancer = LoadBalancer(num_workers=4, balancing_strategy="round_robin")
    
    # Test worker selection
    worker_ids = []
    for i in range(8):
        worker_id = balancer.select_worker(tenant_priority=1)
        worker_ids.append(worker_id)
    
    # Should cycle through workers 0, 1, 2, 3, 0, 1, 2, 3
    expected = [0, 1, 2, 3, 0, 1, 2, 3]
    assert worker_ids == expected
    print("✓ Round-robin worker selection working")
    
    # Test least connections strategy
    balancer_lc = LoadBalancer(num_workers=3, balancing_strategy="least_connections")
    
    # Simulate different connection loads
    balancer_lc.worker_connections = [2, 1, 3]
    worker_id = balancer_lc.select_worker()
    assert worker_id == 1  # Worker with least connections
    print("✓ Least connections strategy working")
    
    # Test task submission
    def dummy_prediction_task(x, y):
        time.sleep(0.1)  # Simulate work
        return x + y
    
    future = balancer.submit_prediction_task(
        dummy_prediction_task,
        "test_tenant",
        2,
        5, 10
    )
    
    result = future.result(timeout=5)
    assert result == 15
    print("✓ Task submission and execution working")
    
    # Test worker status
    status = balancer.get_worker_status()
    assert 'num_workers' in status
    assert 'balancing_strategy' in status
    assert 'worker_connections' in status
    print("✓ Worker status retrieved")
    
    balancer.shutdown()
    print("LoadBalancer tests passed!\n")


def test_tenant_model_manager():
    """Test tenant model manager"""
    print("Testing TenantModelManager...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        base_model_path = os.path.join(temp_dir, "base_model.pkl")
        
        # Create dummy base model file
        with open(base_model_path, 'w') as f:
            f.write("dummy_model")
        
        manager = TenantModelManager(base_model_path)
        
        # Test model customization
        customization_config = {
            'learning_rate': 0.01,
            'max_depth': 5,
            'custom_features': ['feature_1', 'feature_2']
        }
        
        job_id = manager.create_tenant_model(
            tenant_id="test_tenant_001",
            customization_config=customization_config
        )
        
        assert job_id.startswith("model_job_test_tenant_001")
        print("✓ Model customization job created")
        
        # Wait for job completion
        max_wait = 10  # seconds
        wait_time = 0
        while wait_time < max_wait:
            status = manager.get_model_status(job_id)
            if status == "ready":
                break
            elif status == "failed":
                raise Exception("Model customization failed")
            time.sleep(0.5)
            wait_time += 0.5
        
        assert manager.get_model_status(job_id) == "ready"
        print("✓ Model customization completed")
        
        # Test model retrieval
        tenant_model = manager.get_tenant_model("test_tenant_001")
        assert tenant_model is not None
        assert "customizations" in tenant_model
        print("✓ Customized model retrieved")
        
        # Test model listing
        model_list = manager.list_tenant_models()
        assert "test_tenant_001" in model_list['tenant_models']
        assert job_id in model_list['training_jobs']
        print("✓ Model listing working")
    
    print("TenantModelManager tests passed!\n")


def test_multi_tenant_rul_service():
    """Test multi-tenant RUL service"""
    print("Testing MultiTenantRULService...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        base_model_path = os.path.join(temp_dir, "base_model.pkl")
        
        # Create dummy base model file
        with open(base_model_path, 'w') as f:
            f.write("dummy_model")
        
        service = MultiTenantRULService(
            base_model_path=base_model_path,
            num_workers=2
        )
        
        # Test tenant registration
        tenant_id = service.register_tenant(
            tenant_name="Test Corporation",
            max_requests_per_hour=500,
            max_concurrent_requests=10,
            priority_level=2
        )
        
        assert tenant_id.startswith("tenant_")
        print("✓ Tenant registered in service")
        
        # Test prediction
        cycle_data = CycleData(
            cycle_number=50,
            vl_series=np.random.randn(100),
            vo_series=np.random.randn(100),
            timestamp=time.time()
        )
        
        features = np.random.randn(55)
        
        # Run async prediction test
        async def test_async_prediction():
            result = await service.predict_async(
                tenant_id=tenant_id,
                cycle_data=cycle_data,
                features=features
            )
            return result
        
        # Run the async test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(test_async_prediction())
            assert isinstance(result, PredictionResult)
            assert result.rul_cycles >= 0
            print("✓ Async prediction completed")
        finally:
            loop.close()
        
        # Test tenant model creation
        customization_config = {
            'model_type': 'xgboost',
            'n_estimators': 100,
            'max_depth': 6
        }
        
        model_job_id = service.create_tenant_model(
            tenant_id=tenant_id,
            customization_config=customization_config
        )
        
        assert model_job_id.startswith("model_job_")
        print("✓ Tenant model creation initiated")
        
        # Test tenant status
        tenant_status = service.get_tenant_status(tenant_id)
        assert 'tenant_config' in tenant_status
        assert 'resource_quota' in tenant_status
        assert 'metrics' in tenant_status
        print("✓ Tenant status retrieved")
        
        # Test system status
        system_status = service.get_system_status()
        assert 'tenant_count' in system_status
        assert 'load_balancer_status' in system_status
        assert system_status['tenant_count'] >= 1
        print("✓ System status retrieved")
        
        service.shutdown()
    
    print("MultiTenantRULService tests passed!\n")


def test_tenant_deployment_config():
    """Test tenant deployment configuration"""
    print("Testing tenant deployment configuration...")
    
    resource_limits = {
        'max_requests_per_hour': 1000,
        'max_concurrent_requests': 20,
        'max_memory_mb': 2048,
        'max_cpu_cores': 4
    }
    
    customization_options = {
        'model_type': 'ensemble',
        'enable_custom_features': True,
        'training_data_retention_days': 90
    }
    
    config = create_tenant_deployment_config(
        tenant_name="Enterprise Customer",
        resource_limits=resource_limits,
        customization_options=customization_options
    )
    
    assert config['tenant_name'] == "Enterprise Customer"
    assert config['resource_limits'] == resource_limits
    assert config['customization_options'] == customization_options
    assert 'deployment_timestamp' in config
    assert 'api_endpoints' in config
    
    # Check API endpoints
    endpoints = config['api_endpoints']
    assert '/api/v1/tenants/Enterprise Customer/predict' in endpoints['predict']
    assert '/api/v1/tenants/Enterprise Customer/status' in endpoints['status']
    
    print("✓ Tenant deployment configuration created")
    print("Tenant deployment configuration tests passed!\n")


def test_resource_quota_enforcement():
    """Test resource quota enforcement under load"""
    print("Testing resource quota enforcement...")
    
    manager = TenantIsolationManager()
    
    # Create tenant with strict limits
    tenant_config = TenantConfig(
        tenant_id="limited_tenant",
        tenant_name="Limited Tenant",
        max_requests_per_hour=10,  # Very low limit
        max_concurrent_requests=2,  # Very low limit
        max_memory_mb=100,
        max_cpu_cores=1
    )
    
    tenant_id = manager.register_tenant(tenant_config)
    
    # Test concurrent request limit
    successful_requests = 0
    failed_requests = 0
    
    for i in range(5):  # Try to make 5 concurrent requests
        if manager.check_resource_quota(tenant_id, f"req_{i}"):
            successful_requests += 1
        else:
            failed_requests += 1
    
    assert successful_requests == 2  # Should only allow 2 concurrent
    assert failed_requests == 3  # Should reject 3
    print("✓ Concurrent request limit enforced correctly")
    
    # Release some requests
    manager.release_request_resources(tenant_id, "req_0")
    manager.release_request_resources(tenant_id, "req_1")
    
    # Should be able to make new requests now
    can_proceed = manager.check_resource_quota(tenant_id, "req_new")
    assert can_proceed is True
    print("✓ Resource release working correctly")
    
    # Test hourly limit by simulating many requests
    quota = manager.quotas[tenant_id]
    quota.requests_this_hour = 9  # Set to near limit
    
    # This should succeed (request #10)
    can_proceed = manager.check_resource_quota(tenant_id, "req_10")
    assert can_proceed is True
    
    # This should fail (request #11, over limit)
    can_proceed = manager.check_resource_quota(tenant_id, "req_11")
    assert can_proceed is False
    print("✓ Hourly request limit enforced correctly")
    
    print("Resource quota enforcement tests passed!\n")


def main():
    """Run all multi-tenant scaling tests"""
    print("=" * 60)
    print("MULTI-TENANT SUPPORT AND SCALING TESTS")
    print("=" * 60)
    
    try:
        test_tenant_isolation_manager()
        test_load_balancer()
        test_tenant_model_manager()
        test_multi_tenant_rul_service()
        test_tenant_deployment_config()
        test_resource_quota_enforcement()
        
        print("=" * 60)
        print("ALL MULTI-TENANT SCALING TESTS PASSED! ✓")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())