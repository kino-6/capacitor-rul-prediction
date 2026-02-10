#!/usr/bin/env python3
"""
Test script for model versioning and A/B testing framework
"""

import asyncio
import logging
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
import pickle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from true_rul.model_versioning import (
    ModelVersionManager,
    ABTestManager,
    FileSystemModelStorage,
    ModelMetadata,
    ModelStatus,
    ABTestStatus,
    create_model_versioning_system
)
from true_rul.canary_deployment import (
    CanaryDeploymentManager,
    CanaryStatus,
    create_canary_deployment_system
)
from true_rul.data_structures import PredictionResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MockModel:
    """Mock model for testing"""
    
    def __init__(self, model_id: str, version: str, performance_multiplier: float = 1.0):
        self.model_id = model_id
        self.version = version
        self.performance_multiplier = performance_multiplier
        
    def predict(self, x):
        """Mock prediction"""
        # Simulate different performance characteristics
        latency = 100 * self.performance_multiplier
        error_rate = 0.01 * self.performance_multiplier
        
        return {
            "prediction": 50,
            "latency_ms": latency,
            "error_rate": error_rate
        }


def create_mock_prediction_result(model_id: str, cycle_number: int = 1) -> PredictionResult:
    """Create a mock prediction result"""
    rul_cycles = max(1, 100 - cycle_number)
    return PredictionResult(
        rul_cycles=rul_cycles,
        rul_confidence_lower=max(1, rul_cycles - 10),
        rul_confidence_upper=rul_cycles + 10,
        degradation_score=min(1.0, cycle_number / 100.0),
        degradation_stage="healthy" if cycle_number < 50 else "early_degradation",
        anomaly_flag=False,
        anomaly_score=0.1,
        feature_importance={"feature_1": 0.3, "feature_2": 0.7},
        timestamp=datetime.now(),
        model_version=f"{model_id}_v1.0"
    )


async def test_model_storage():
    """Test model storage functionality"""
    logger.info("Testing model storage...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = Path(temp_dir)
        storage = FileSystemModelStorage(storage_path)
        
        # Create mock models
        model_v1 = MockModel("test_model", "v1.0")
        model_v2 = MockModel("test_model", "v2.0", performance_multiplier=0.8)
        
        # Create metadata
        metadata_v1 = ModelMetadata(
            model_id="test_model",
            version="v1.0",
            name="Test Model V1",
            description="First version of test model",
            created_at=datetime.now(),
            created_by="test_user",
            model_type="rul_regression",
            framework="sklearn",
            status=ModelStatus.DEVELOPMENT,
            metrics={"rmse": 10.5, "mae": 8.2},
            hyperparameters={"n_estimators": 100, "max_depth": 10}
        )
        
        metadata_v2 = ModelMetadata(
            model_id="test_model",
            version="v2.0",
            name="Test Model V2",
            description="Improved version of test model",
            created_at=datetime.now(),
            created_by="test_user",
            model_type="rul_regression",
            framework="sklearn",
            status=ModelStatus.DEVELOPMENT,
            metrics={"rmse": 9.2, "mae": 7.1},
            hyperparameters={"n_estimators": 200, "max_depth": 15}
        )
        
        # Test saving models
        success_v1 = storage.save_model(model_v1, metadata_v1)
        success_v2 = storage.save_model(model_v2, metadata_v2)
        
        assert success_v1, "Failed to save model v1"
        assert success_v2, "Failed to save model v2"
        
        # Test loading models
        loaded_model_v1, loaded_metadata_v1 = storage.load_model("test_model", "v1.0")
        loaded_model_v2, loaded_metadata_v2 = storage.load_model("test_model", "v2.0")
        
        assert loaded_model_v1.model_id == "test_model", "Loaded model v1 ID mismatch"
        assert loaded_model_v2.version == "v2.0", "Loaded model v2 version mismatch"
        
        # Test listing models
        all_models = storage.list_models()
        assert len(all_models) == 2, f"Expected 2 models, got {len(all_models)}"
        
        dev_models = storage.list_models(ModelStatus.DEVELOPMENT)
        assert len(dev_models) == 2, f"Expected 2 development models, got {len(dev_models)}"
        
        # Test model deletion
        success_delete = storage.delete_model("test_model", "v1.0")
        assert success_delete, "Failed to delete model v1"
        
        remaining_models = storage.list_models()
        assert len(remaining_models) == 1, f"Expected 1 model after deletion, got {len(remaining_models)}"
        
        logger.info("✓ Model storage test passed")


async def test_model_version_manager():
    """Test model version manager functionality"""
    logger.info("Testing model version manager...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = Path(temp_dir)
        model_manager, _ = create_model_versioning_system(storage_path)
        
        # Register models
        model_v1 = MockModel("rul_model", "v1.0")
        model_v2 = MockModel("rul_model", "v2.0", performance_multiplier=0.9)
        
        metadata_v1 = model_manager.register_model(
            model=model_v1,
            model_id="rul_model",
            version="v1.0",
            name="RUL Model V1",
            description="Initial RUL model",
            model_type="rul_regression",
            framework="sklearn",
            created_by="test_user",
            metrics={"rmse": 12.0, "mae": 9.5}
        )
        
        metadata_v2 = model_manager.register_model(
            model=model_v2,
            model_id="rul_model",
            version="v2.0",
            name="RUL Model V2",
            description="Improved RUL model",
            model_type="rul_regression",
            framework="sklearn",
            created_by="test_user",
            metrics={"rmse": 10.8, "mae": 8.2}
        )
        
        assert metadata_v1.status == ModelStatus.DEVELOPMENT, "New model should be in development"
        assert metadata_v2.status == ModelStatus.DEVELOPMENT, "New model should be in development"
        
        # Test promotion
        success_promote = model_manager.promote_model("rul_model", "v1.0", ModelStatus.PRODUCTION)
        assert success_promote, "Failed to promote model to production"
        
        # Test getting production model
        prod_model, prod_metadata = model_manager.get_production_model("rul_regression")
        assert prod_metadata.version == "v1.0", "Production model version mismatch"
        
        # Test model comparison
        comparison = model_manager.compare_models("rul_model", "v1.0", "rul_model", "v2.0")
        
        assert "model_a" in comparison, "Comparison should include model_a"
        assert "model_b" in comparison, "Comparison should include model_b"
        assert "metric_differences" in comparison, "Comparison should include metric differences"
        
        rmse_diff = comparison["metric_differences"]["rmse"]
        assert rmse_diff["difference"] < 0, "V2 should have lower RMSE than V1"
        
        # Test rollback
        model_manager.promote_model("rul_model", "v2.0", ModelStatus.PRODUCTION)
        success_rollback = model_manager.rollback_model("rul_regression", "v1.0")
        assert success_rollback, "Failed to rollback model"
        
        # Verify rollback
        current_prod_model, current_prod_metadata = model_manager.get_production_model("rul_regression")
        assert current_prod_metadata.version == "v1.0", "Rollback should restore v1.0"
        
        logger.info("✓ Model version manager test passed")


async def test_ab_testing():
    """Test A/B testing functionality"""
    logger.info("Testing A/B testing...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = Path(temp_dir)
        model_manager, ab_test_manager = create_model_versioning_system(storage_path)
        
        # Register test models
        model_a = MockModel("model_a", "v1.0", performance_multiplier=1.0)
        model_b = MockModel("model_b", "v1.0", performance_multiplier=0.8)  # Better performance
        
        model_manager.register_model(
            model=model_a,
            model_id="model_a",
            version="v1.0",
            name="Model A",
            description="Control model",
            model_type="rul_regression",
            framework="sklearn",
            created_by="test_user"
        )
        
        model_manager.register_model(
            model=model_b,
            model_id="model_b",
            version="v1.0",
            name="Model B",
            description="Treatment model",
            model_type="rul_regression",
            framework="sklearn",
            created_by="test_user"
        )
        
        # Create A/B test
        ab_test_config = ab_test_manager.create_ab_test(
            name="Model A vs Model B",
            description="Testing improved model performance",
            model_a_id="model_a",
            model_a_version="v1.0",
            model_b_id="model_b",
            model_b_version="v1.0",
            traffic_split=0.5,
            duration_hours=1,
            created_by="test_user"
        )
        
        assert ab_test_config.status == ABTestStatus.DRAFT, "New A/B test should be in draft status"
        
        # Start A/B test
        success_start = ab_test_manager.start_ab_test(ab_test_config.test_id)
        assert success_start, "Failed to start A/B test"
        
        # Simulate predictions
        for i in range(100):
            request_id = f"request_{i}"
            selected_model = ab_test_manager.select_model_for_request(ab_test_config.test_id, request_id)
            
            # Create mock prediction result
            result = create_mock_prediction_result(selected_model.split(":")[0], i)
            
            # Simulate different performance for each model
            if "model_a" in selected_model:
                latency = 120.0
                had_error = i % 50 == 0  # 2% error rate
            else:
                latency = 95.0  # Better latency
                had_error = i % 100 == 0  # 1% error rate
                
            ab_test_manager.record_prediction_result(
                ab_test_config.test_id,
                selected_model,
                result,
                latency,
                had_error
            )
        
        # Get A/B test results
        results = ab_test_manager.get_ab_test_results(ab_test_config.test_id)
        
        assert "model_a_results" in results, "Results should include model A"
        assert "model_b_results" in results, "Results should include model B"
        assert "comparison" in results, "Results should include comparison"
        
        model_a_results = results["model_a_results"]
        model_b_results = results["model_b_results"]
        
        assert model_a_results["prediction_count"] > 0, "Model A should have predictions"
        assert model_b_results["prediction_count"] > 0, "Model B should have predictions"
        
        # Model B should have better performance
        assert model_b_results["average_latency_ms"] < model_a_results["average_latency_ms"], "Model B should have lower latency"
        
        # Stop A/B test
        success_stop = ab_test_manager.stop_ab_test(ab_test_config.test_id)
        assert success_stop, "Failed to stop A/B test"
        
        # List A/B tests
        all_tests = ab_test_manager.list_ab_tests()
        assert len(all_tests) == 1, f"Expected 1 A/B test, got {len(all_tests)}"
        
        completed_tests = ab_test_manager.list_ab_tests(ABTestStatus.COMPLETED)
        assert len(completed_tests) == 1, f"Expected 1 completed A/B test, got {len(completed_tests)}"
        
        logger.info("✓ A/B testing test passed")


async def test_canary_deployment():
    """Test canary deployment functionality"""
    logger.info("Testing canary deployment...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = Path(temp_dir)
        model_manager, ab_test_manager = create_model_versioning_system(storage_path)
        canary_manager = create_canary_deployment_system(model_manager, ab_test_manager)
        
        # Register models
        current_model = MockModel("prod_model", "v1.0", performance_multiplier=1.0)
        canary_model = MockModel("prod_model", "v2.0", performance_multiplier=0.9)
        
        model_manager.register_model(
            model=current_model,
            model_id="prod_model",
            version="v1.0",
            name="Production Model V1",
            description="Current production model",
            model_type="rul_regression",
            framework="sklearn",
            created_by="test_user"
        )
        
        model_manager.register_model(
            model=canary_model,
            model_id="prod_model",
            version="v2.0",
            name="Production Model V2",
            description="New canary model",
            model_type="rul_regression",
            framework="sklearn",
            created_by="test_user"
        )
        
        # Promote current model to production
        model_manager.promote_model("prod_model", "v1.0", ModelStatus.PRODUCTION)
        
        # Create canary deployment
        deployment = canary_manager.create_canary_deployment(
            name="V2 Canary Deployment",
            description="Testing new model version",
            current_model_id="prod_model",
            current_model_version="v1.0",
            canary_model_id="prod_model",
            canary_model_version="v2.0",
            created_by="test_user",
            config_overrides={
                "stage_durations_minutes": {
                    "stage_1": 1,  # 1 minute for testing
                    "stage_2": 1,
                    "stage_3": 1,
                    "stage_4": 1
                },
                "min_sample_size_per_stage": 5  # Lower for testing
            }
        )
        
        assert deployment.status == CanaryStatus.PREPARING, "New deployment should be preparing"
        
        # Get deployment status
        status = canary_manager.get_deployment_status(deployment.config.deployment_id)
        assert status is not None, "Should be able to get deployment status"
        assert status["status"] == "preparing", "Status should be preparing"
        
        # List deployments
        deployments = canary_manager.list_deployments()
        assert len(deployments) == 1, f"Expected 1 deployment, got {len(deployments)}"
        
        preparing_deployments = canary_manager.list_deployments(CanaryStatus.PREPARING)
        assert len(preparing_deployments) == 1, f"Expected 1 preparing deployment, got {len(preparing_deployments)}"
        
        # Note: We don't actually start the canary deployment in the test
        # because it would take several minutes to complete and involves async operations
        # In a real test environment, you would:
        # 1. Start the deployment: await canary_manager.start_canary_deployment(deployment.config.deployment_id)
        # 2. Simulate traffic and monitor progress
        # 3. Verify successful completion or rollback
        
        logger.info("✓ Canary deployment test passed")


async def test_model_comparison():
    """Test model comparison functionality"""
    logger.info("Testing model comparison...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = Path(temp_dir)
        model_manager, _ = create_model_versioning_system(storage_path)
        
        # Register models with different performance metrics
        model_old = MockModel("comparison_model", "v1.0")
        model_new = MockModel("comparison_model", "v2.0")
        
        model_manager.register_model(
            model=model_old,
            model_id="comparison_model",
            version="v1.0",
            name="Old Model",
            description="Baseline model",
            model_type="rul_regression",
            framework="sklearn",
            created_by="test_user",
            metrics={
                "rmse": 15.2,
                "mae": 12.1,
                "r2_score": 0.82,
                "fpr": 0.08
            }
        )
        
        model_manager.register_model(
            model=model_new,
            model_id="comparison_model",
            version="v2.0",
            name="New Model",
            description="Improved model",
            model_type="rul_regression",
            framework="sklearn",
            created_by="test_user",
            metrics={
                "rmse": 12.8,
                "mae": 10.3,
                "r2_score": 0.87,
                "fpr": 0.04
            }
        )
        
        # Compare models
        comparison = model_manager.compare_models(
            "comparison_model", "v1.0",
            "comparison_model", "v2.0"
        )
        
        # Verify comparison structure
        assert "model_a" in comparison, "Comparison should include model_a"
        assert "model_b" in comparison, "Comparison should include model_b"
        assert "metric_differences" in comparison, "Comparison should include metric_differences"
        
        # Check specific metric improvements
        rmse_diff = comparison["metric_differences"]["rmse"]
        assert rmse_diff["difference"] < 0, "RMSE should be lower in v2.0"
        assert rmse_diff["percent_change"] < 0, "RMSE percent change should be negative (improvement)"
        
        fpr_diff = comparison["metric_differences"]["fpr"]
        assert fpr_diff["difference"] < 0, "FPR should be lower in v2.0"
        
        r2_diff = comparison["metric_differences"]["r2_score"]
        assert r2_diff["difference"] > 0, "R2 score should be higher in v2.0"
        
        logger.info("✓ Model comparison test passed")


async def main():
    """Run all tests"""
    logger.info("Starting model versioning and A/B testing tests...")
    
    try:
        await test_model_storage()
        await test_model_version_manager()
        await test_ab_testing()
        await test_canary_deployment()
        await test_model_comparison()
        
        logger.info("🎉 All model versioning and A/B testing tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())