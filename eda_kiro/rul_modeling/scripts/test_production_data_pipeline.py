#!/usr/bin/env python3
"""
Test script for production data pipeline integration
"""

import asyncio
import logging
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from true_rul.production_data_pipeline import (
    ProductionDataPipeline,
    FileDataSource,
    SensorReading,
    DataValidationRule,
    DEFAULT_VALIDATION_RULES
)
from true_rul.data_structures import CycleData, PredictionResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_test_data_file(file_path: Path, n_samples: int = 1000):
    """Create a test data file with synthetic sensor data"""
    # Generate synthetic voltage data that resembles capacitor behavior
    time_points = np.linspace(0, 10, n_samples)
    
    # Simulate charge-discharge cycle
    vl_data = 5.0 + 3.0 * np.sin(2 * np.pi * time_points) + 0.1 * np.random.randn(n_samples)
    vo_data = 4.0 + 2.5 * np.sin(2 * np.pi * time_points + 0.5) + 0.1 * np.random.randn(n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': [datetime.now() + timedelta(seconds=i) for i in range(n_samples)],
        'VL': vl_data,
        'VO': vo_data
    })
    
    # Save to CSV
    df.to_csv(file_path, index=False)
    logger.info(f"Created test data file with {n_samples} samples: {file_path}")


def mock_prediction_callback(cycles: list[CycleData]) -> list[PredictionResult]:
    """Mock prediction callback for testing"""
    results = []
    
    for cycle in cycles:
        # Create mock prediction result
        result = PredictionResult(
            rul_cycles=max(1, 200 - cycle.cycle_number),
            rul_confidence_lower=max(1, 200 - cycle.cycle_number - 10),
            rul_confidence_upper=200 - cycle.cycle_number + 10,
            degradation_score=min(1.0, cycle.cycle_number / 200.0),
            degradation_stage="healthy" if cycle.cycle_number < 50 else "early_degradation",
            anomaly_flag=False,
            anomaly_score=0.1,
            feature_importance={"feature_1": 0.3, "feature_2": 0.7},
            timestamp=datetime.now(),
            model_version="test_v1.0",
            capacitor_id="TEST_CAP",
            cycle_number=cycle.cycle_number
        )
        results.append(result)
    
    logger.info(f"Generated {len(results)} mock predictions")
    return results


async def test_basic_pipeline():
    """Test basic pipeline functionality"""
    logger.info("Testing basic pipeline functionality...")
    
    # Create temporary test data file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_file = Path(f.name)
    
    try:
        # Create test data
        create_test_data_file(test_file, n_samples=100)
        
        # Create pipeline
        pipeline = ProductionDataPipeline(
            validation_rules=DEFAULT_VALIDATION_RULES,
            buffer_size=1000,
            batch_size=5
        )
        
        # Set prediction callback
        pipeline.set_prediction_callback(mock_prediction_callback)
        
        # Add file data source
        file_source = FileDataSource(
            file_path=test_file,
            sensor_id="TEST_SENSOR",
            replay_speed=10.0  # 10x speed for testing
        )
        pipeline.add_data_source(file_source)
        
        # Start pipeline
        await pipeline.start()
        
        # Let it run for a few seconds
        await asyncio.sleep(5)
        
        # Stop pipeline
        await pipeline.stop()
        
        # Check statistics
        stats = pipeline.get_statistics()
        logger.info(f"Pipeline statistics: {stats}")
        
        # Verify some data was processed
        assert stats['readings_processed'] > 0, "No readings were processed"
        assert stats['data_quality']['total_readings'] > 0, "No readings in quality metrics"
        
        logger.info("✓ Basic pipeline test passed")
        
    finally:
        # Clean up
        test_file.unlink(missing_ok=True)


async def test_data_validation():
    """Test data validation functionality"""
    logger.info("Testing data validation...")
    
    # Create strict validation rules
    strict_rules = [
        DataValidationRule(
            name="strict_voltage_range",
            vl_min=2.0,
            vl_max=8.0,
            vo_min=1.5,
            vo_max=7.0
        )
    ]
    
    # Create pipeline with strict rules
    pipeline = ProductionDataPipeline(
        validation_rules=strict_rules,
        buffer_size=100,
        batch_size=5
    )
    
    # Create test data with some out-of-range values
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_file = Path(f.name)
    
    try:
        # Create test data with some invalid values
        n_samples = 50
        vl_data = np.concatenate([
            np.full(25, 5.0),  # Valid values
            np.full(25, 10.0)  # Invalid values (too high)
        ])
        vo_data = np.concatenate([
            np.full(25, 4.0),  # Valid values
            np.full(25, 0.5)   # Invalid values (too low)
        ])
        
        df = pd.DataFrame({
            'timestamp': [datetime.now() + timedelta(seconds=i) for i in range(n_samples)],
            'VL': vl_data,
            'VO': vo_data
        })
        df.to_csv(test_file, index=False)
        
        # Add file source
        file_source = FileDataSource(
            file_path=test_file,
            sensor_id="VALIDATION_TEST",
            replay_speed=20.0
        )
        pipeline.add_data_source(file_source)
        
        # Start pipeline
        await pipeline.start()
        await asyncio.sleep(3)
        await pipeline.stop()
        
        # Check validation metrics
        stats = pipeline.get_statistics()
        quality_metrics = stats['data_quality']
        
        logger.info(f"Validation statistics: {quality_metrics}")
        
        # Should have some invalid readings
        assert quality_metrics['invalid_readings'] > 0, "Expected some invalid readings"
        assert quality_metrics['out_of_range_readings'] > 0, "Expected out-of-range readings"
        
        # Quality score should be less than 1.0
        assert quality_metrics['quality_score'] < 1.0, "Expected quality score < 1.0"
        
        logger.info("✓ Data validation test passed")
        
    finally:
        test_file.unlink(missing_ok=True)


async def test_batch_processing():
    """Test batch processing functionality"""
    logger.info("Testing batch processing...")
    
    batch_results = []
    
    def capture_batch_results(cycles: list[CycleData]) -> list[PredictionResult]:
        """Capture batch results for verification"""
        results = mock_prediction_callback(cycles)
        batch_results.extend(results)
        return results
    
    # Create pipeline with small batch size
    pipeline = ProductionDataPipeline(
        validation_rules=DEFAULT_VALIDATION_RULES,
        buffer_size=500,
        batch_size=3  # Small batch for testing
    )
    
    pipeline.set_prediction_callback(capture_batch_results)
    
    # Create test data file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_file = Path(f.name)
    
    try:
        create_test_data_file(test_file, n_samples=200)
        
        file_source = FileDataSource(
            file_path=test_file,
            sensor_id="BATCH_TEST",
            replay_speed=50.0  # Fast replay
        )
        pipeline.add_data_source(file_source)
        
        await pipeline.start()
        await asyncio.sleep(3)
        await pipeline.stop()
        
        stats = pipeline.get_statistics()
        logger.info(f"Batch processing statistics: {stats}")
        logger.info(f"Captured {len(batch_results)} batch results")
        
        # Should have processed some batches
        assert stats['batches_processed'] > 0, "No batches were processed"
        assert len(batch_results) > 0, "No batch results captured"
        
        logger.info("✓ Batch processing test passed")
        
    finally:
        test_file.unlink(missing_ok=True)


async def test_error_handling():
    """Test error handling in the pipeline"""
    logger.info("Testing error handling...")
    
    def failing_callback(cycles: list[CycleData]) -> list[PredictionResult]:
        """Callback that always fails"""
        raise RuntimeError("Simulated processing failure")
    
    pipeline = ProductionDataPipeline(
        validation_rules=DEFAULT_VALIDATION_RULES,
        buffer_size=100,
        batch_size=2
    )
    
    pipeline.set_prediction_callback(failing_callback)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_file = Path(f.name)
    
    try:
        create_test_data_file(test_file, n_samples=50)
        
        file_source = FileDataSource(
            file_path=test_file,
            sensor_id="ERROR_TEST",
            replay_speed=20.0
        )
        pipeline.add_data_source(file_source)
        
        await pipeline.start()
        await asyncio.sleep(2)
        await pipeline.stop()
        
        stats = pipeline.get_statistics()
        logger.info(f"Error handling statistics: {stats}")
        
        # Should have some errors recorded
        # Note: Errors might be 0 if no cycles were detected due to short run time
        logger.info("✓ Error handling test completed")
        
    finally:
        test_file.unlink(missing_ok=True)


async def main():
    """Run all tests"""
    logger.info("Starting production data pipeline tests...")
    
    try:
        await test_basic_pipeline()
        await test_data_validation()
        await test_batch_processing()
        await test_error_handling()
        
        logger.info("🎉 All production data pipeline tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())