#!/usr/bin/env python3
"""
Test script for predictive maintenance integration

This script tests the predictive maintenance functionality including:
- Maintenance scheduling optimization
- Cost-benefit analysis for replacement decisions
- CMMS integration
- Spare parts inventory optimization
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

from true_rul.predictive_maintenance import (
    PredictiveMaintenanceSystem, CostBenefitAnalyzer, MaintenanceScheduler,
    InventoryOptimizer, CMMSIntegrator, Component, SparePart, MaintenanceTask,
    MaintenanceAction, Priority, create_predictive_maintenance_system
)
from true_rul.data_structures import PredictionResult
from true_rul.rul_regression_model import RULRegressionModel
from true_rul.feature_extractor import FeatureExtractor
from true_rul.time_series_preprocessor import TimeSeriesPreprocessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_synthetic_features(num_samples: int = 100, num_features: int = 55) -> np.ndarray:
    """Create synthetic feature data for testing"""
    np.random.seed(42)
    return np.random.randn(num_samples, num_features)


def create_test_components() -> list:
    """Create test components for predictive maintenance"""
    components = []
    
    # Create different types of components with varying criticality
    component_types = [
        ("capacitor", 5000, 0.9),
        ("resistor", 1000, 0.7),
        ("inductor", 3000, 0.8),
        ("transformer", 10000, 0.95)
    ]
    
    for i, (comp_type, cost, criticality) in enumerate(component_types):
        for j in range(2):  # 2 of each type
            component = Component(
                component_id=f"{comp_type}_{i}_{j}",
                component_type=comp_type,
                location=f"Section_{i+1}",
                installation_date=datetime.now() - timedelta(days=365 * 2),
                last_maintenance_date=datetime.now() - timedelta(days=180),
                replacement_cost=cost,
                criticality_score=criticality
            )
            components.append(component)
    
    return components


def create_test_spare_parts() -> list:
    """Create test spare parts for inventory optimization"""
    spare_parts = []
    
    part_configs = [
        ("capacitor_replacement", "Replacement Capacitor", 10, 2, 20, 500, 14, ["capacitor"]),
        ("resistor_replacement", "Replacement Resistor", 25, 5, 50, 100, 7, ["resistor"]),
        ("inductor_replacement", "Replacement Inductor", 8, 1, 15, 300, 21, ["inductor"]),
        ("transformer_replacement", "Replacement Transformer", 3, 1, 6, 1000, 30, ["transformer"])
    ]
    
    for part_id, name, stock, min_stock, max_stock, cost, lead_time, compatible in part_configs:
        spare_part = SparePart(
            part_id=part_id,
            part_name=name,
            current_stock=stock,
            min_stock_level=min_stock,
            max_stock_level=max_stock,
            unit_cost=cost,
            lead_time_days=lead_time,
            supplier="Test Supplier",
            compatible_components=compatible
        )
        spare_parts.append(spare_part)
    
    return spare_parts


def create_test_predictions(components: list) -> dict:
    """Create test RUL predictions for components"""
    predictions = {}
    
    for i, component in enumerate(components):
        # Vary RUL to test different scenarios
        if i % 4 == 0:  # Critical
            rul = np.random.randint(5, 15)
            stage = "critical"
        elif i % 4 == 1:  # Advanced degradation
            rul = np.random.randint(15, 35)
            stage = "advanced_degradation"
        elif i % 4 == 2:  # Early degradation
            rul = np.random.randint(35, 65)
            stage = "early_degradation"
        else:  # Healthy
            rul = np.random.randint(65, 120)
            stage = "healthy"
        
        prediction = PredictionResult(
            rul_cycles=rul,
            rul_confidence_lower=max(1, rul - 10),
            rul_confidence_upper=rul + 10,
            degradation_score=0.8 if stage == "critical" else 0.5,
            degradation_stage=stage,
            anomaly_flag=stage == "critical",
            anomaly_score=0.8 if stage == "critical" else 0.2,
            feature_importance={},
            timestamp=datetime.now().timestamp(),
            model_version="1.0"
        )
        
        predictions[component.component_id] = prediction
    
    return predictions


def test_cost_benefit_analyzer():
    """Test cost-benefit analysis functionality"""
    logger.info("Testing cost-benefit analyzer...")
    
    analyzer = CostBenefitAnalyzer()
    
    # Create test component
    component = Component(
        component_id="test_capacitor_1",
        component_type="capacitor",
        location="Test Section",
        installation_date=datetime.now() - timedelta(days=365),
        last_maintenance_date=datetime.now() - timedelta(days=90),
        replacement_cost=5000,
        criticality_score=0.9
    )
    
    # Test different RUL scenarios
    scenarios = [
        (5, "critical"),    # Very low RUL
        (25, "advanced_degradation"),  # Medium RUL
        (80, "healthy")     # High RUL
    ]
    
    for rul, stage in tqdm(scenarios, desc="Testing cost-benefit scenarios"):
        prediction = PredictionResult(
            rul_cycles=rul,
            rul_confidence_lower=max(1, rul - 5),
            rul_confidence_upper=rul + 5,
            degradation_score=0.8 if stage == "critical" else 0.3,
            degradation_stage=stage,
            anomaly_flag=stage == "critical",
            anomaly_score=0.8 if stage == "critical" else 0.2,
            feature_importance={},
            timestamp=datetime.now().timestamp(),
            model_version="1.0"
        )
        
        analysis = analyzer.analyze_replacement_decision(component, prediction)
        
        logger.info(f"RUL {rul}: Recommended action: {analysis['recommended_action']}")
        cost_summary = [f"{k}: ${v['expected_cost']:.0f}" for k, v in analysis['scenarios'].items()]
        logger.info(f"  Expected costs: {cost_summary}")
    
    logger.info("Cost-benefit analyzer test completed successfully")


def test_maintenance_scheduler():
    """Test maintenance scheduling functionality"""
    logger.info("Testing maintenance scheduler...")
    
    scheduler = MaintenanceScheduler()
    components = create_test_components()
    predictions = create_test_predictions(components)
    
    # Test schedule optimization
    with tqdm(total=1, desc="Optimizing maintenance schedule") as pbar:
        schedule = scheduler.optimize_schedule(components, predictions, time_horizon_days=90)
        pbar.update(1)
    
    logger.info(f"Generated schedule with {len(schedule.tasks)} tasks")
    logger.info(f"Total estimated cost: ${schedule.total_cost:.2f}")
    logger.info(f"Total estimated duration: {schedule.total_duration:.1f} hours")
    
    # Test task prioritization
    priority_counts = {}
    for task in schedule.tasks:
        priority_counts[task.priority.value] = priority_counts.get(task.priority.value, 0) + 1
    
    logger.info(f"Task priorities: {priority_counts}")
    
    # Verify scheduling constraints
    daily_tasks = {}
    for task in schedule.tasks:
        date_key = task.scheduled_date.date()
        daily_tasks[date_key] = daily_tasks.get(date_key, 0) + 1
    
    max_daily_tasks = max(daily_tasks.values()) if daily_tasks else 0
    logger.info(f"Maximum tasks per day: {max_daily_tasks}")
    
    assert max_daily_tasks <= scheduler.scheduling_constraints['max_daily_tasks'], \
        "Daily task limit exceeded"
    
    logger.info("Maintenance scheduler test completed successfully")


def test_inventory_optimizer():
    """Test inventory optimization functionality"""
    logger.info("Testing inventory optimizer...")
    
    optimizer = InventoryOptimizer()
    components = create_test_components()
    spare_parts = create_test_spare_parts()
    predictions = create_test_predictions(components)
    
    # Test inventory optimization
    with tqdm(total=1, desc="Optimizing inventory levels") as pbar:
        recommendations = optimizer.optimize_inventory(
            spare_parts, components, predictions, planning_horizon_days=365
        )
        pbar.update(1)
    
    logger.info(f"Generated recommendations for {len(recommendations)} spare parts")
    
    # Analyze recommendations
    action_counts = {}
    for part_id, rec in recommendations.items():
        action = rec['action']
        action_counts[action] = action_counts.get(action, 0) + 1
        
        logger.info(f"Part {part_id}: {action} (Current: {rec['current_stock']}, "
                   f"Optimal max: {rec['optimal_levels']['optimal_max_stock']})")
    
    logger.info(f"Action distribution: {action_counts}")
    
    # Verify recommendations make sense
    for part_id, rec in recommendations.items():
        assert rec['current_stock'] >= 0, f"Invalid current stock for {part_id}"
        assert rec['optimal_levels']['reorder_point'] >= 0, f"Invalid reorder point for {part_id}"
        assert rec['optimal_levels']['economic_order_quantity'] >= 0, f"Invalid EOQ for {part_id}"
    
    logger.info("Inventory optimizer test completed successfully")


def test_cmms_integrator():
    """Test CMMS integration functionality"""
    logger.info("Testing CMMS integrator...")
    
    # Test different CMMS systems
    systems = ['generic', 'maximo', 'sap_pm']
    
    for system in tqdm(systems, desc="Testing CMMS systems"):
        cmms_config = {
            'system_type': system,
            'api_endpoint': 'http://localhost:8080/api',
            'api_key': 'test_key',
            'timeout': 30
        }
        
        integrator = CMMSIntegrator(cmms_config)
        
        # Create test schedule
        scheduler = MaintenanceScheduler()
        components = create_test_components()[:2]  # Use fewer components for faster testing
        predictions = create_test_predictions(components)
        
        schedule = scheduler.optimize_schedule(components, predictions, time_horizon_days=30)
        
        # Test export
        cmms_export = integrator.export_maintenance_schedule(schedule)
        
        logger.info(f"Exported schedule for {system}: {len(cmms_export['work_orders'])} work orders")
        
        # Verify export format
        assert 'work_orders' in cmms_export, f"Missing work_orders in {system} export"
        assert len(cmms_export['work_orders']) > 0, f"No work orders in {system} export"
        
        # Test import simulation
        test_updates = [
            {
                'work_order_id': 'test_wo_1',
                'status': 'COMPLETED',
                'actual_cost': 1000,
                'actual_duration': 4.0
            }
        ]
        
        import_result = integrator.import_work_order_status(test_updates)
        assert import_result['processed_count'] == 1, f"Failed to process update for {system}"
    
    logger.info("CMMS integrator test completed successfully")


def test_predictive_maintenance_system():
    """Test the complete predictive maintenance system"""
    logger.info("Testing complete predictive maintenance system...")
    
    # Create and train a simple RUL model for testing
    logger.info("Creating RUL model...")
    rul_model = RULRegressionModel(model_type='xgboost')
    
    # Create synthetic training data
    features = create_synthetic_features(100, 55)
    rul_labels = np.random.randint(10, 100, 100)
    
    with tqdm(total=1, desc="Training RUL model") as pbar:
        rul_model.train(features, rul_labels)
        pbar.update(1)
    
    # Create predictive maintenance system
    pm_system = create_predictive_maintenance_system(rul_model)
    
    # Add components and spare parts
    components = create_test_components()
    spare_parts = create_test_spare_parts()
    
    logger.info("Adding components and spare parts...")
    for component in tqdm(components, desc="Adding components"):
        pm_system.add_component(component)
    
    for spare_part in tqdm(spare_parts, desc="Adding spare parts"):
        pm_system.add_spare_part(spare_part)
    
    # Create prediction data
    prediction_data = {}
    for component in components:
        prediction_data[component.component_id] = create_synthetic_features(1, 55)[0]
    
    # Generate comprehensive maintenance plan
    logger.info("Generating maintenance plan...")
    with tqdm(total=1, desc="Generating maintenance plan") as pbar:
        maintenance_plan = pm_system.generate_maintenance_plan(
            prediction_data, planning_horizon_days=90
        )
        pbar.update(1)
    
    # Verify plan structure
    required_keys = [
        'plan_id', 'generated_at', 'rul_predictions', 'cost_benefit_analyses',
        'maintenance_schedule', 'inventory_recommendations', 'cmms_export', 'summary'
    ]
    
    for key in required_keys:
        assert key in maintenance_plan, f"Missing key '{key}' in maintenance plan"
    
    logger.info(f"Generated plan ID: {maintenance_plan['plan_id']}")
    logger.info(f"Total components: {maintenance_plan['summary']['total_components']}")
    logger.info(f"Components needing attention: {maintenance_plan['summary']['components_needing_attention']}")
    logger.info(f"Critical components: {maintenance_plan['summary']['critical_components']}")
    logger.info(f"Total estimated cost: ${maintenance_plan['summary']['total_estimated_cost']:.2f}")
    
    # Test system status
    status = pm_system.get_system_status()
    logger.info(f"System status: {status}")
    
    assert status['components_count'] == len(components), "Component count mismatch"
    assert status['spare_parts_count'] == len(spare_parts), "Spare parts count mismatch"
    assert status['system_health'] == 'operational', "System health check failed"
    
    logger.info("Predictive maintenance system test completed successfully")


def run_performance_test():
    """Run performance test for predictive maintenance system"""
    logger.info("Running performance test...")
    
    # Test with larger dataset
    num_components = 50
    
    # Create RUL model
    rul_model = RULRegressionModel(model_type='xgboost')
    features = create_synthetic_features(200, 55)
    rul_labels = np.random.randint(10, 100, 200)
    
    start_time = time.time()
    with tqdm(total=1, desc="Training RUL model") as pbar:
        rul_model.train(features, rul_labels)
        pbar.update(1)
    training_time = time.time() - start_time
    
    # Create system
    pm_system = create_predictive_maintenance_system(rul_model)
    
    # Add many components
    components = []
    for i in tqdm(range(num_components), desc="Creating components"):
        component = Component(
            component_id=f"component_{i}",
            component_type="capacitor",
            location=f"Section_{i//10}",
            installation_date=datetime.now() - timedelta(days=365),
            last_maintenance_date=datetime.now() - timedelta(days=180),
            replacement_cost=5000,
            criticality_score=0.8
        )
        components.append(component)
        pm_system.add_component(component)
    
    # Create prediction data
    prediction_data = {}
    for component in components:
        prediction_data[component.component_id] = create_synthetic_features(1, 55)[0]
    
    # Time the maintenance plan generation
    start_time = time.time()
    with tqdm(total=1, desc="Generating maintenance plan") as pbar:
        maintenance_plan = pm_system.generate_maintenance_plan(
            prediction_data, planning_horizon_days=90
        )
        pbar.update(1)
    plan_generation_time = time.time() - start_time
    
    logger.info(f"Performance Results:")
    logger.info(f"  Training time: {training_time:.2f}s")
    logger.info(f"  Plan generation time: {plan_generation_time:.2f}s")
    logger.info(f"  Components processed: {num_components}")
    logger.info(f"  Tasks generated: {len(maintenance_plan['maintenance_schedule']['tasks'])}")
    logger.info(f"  Time per component: {plan_generation_time/num_components:.3f}s")
    
    # Performance assertions
    assert training_time < 30, f"Training took too long: {training_time:.2f}s"
    assert plan_generation_time < 60, f"Plan generation took too long: {plan_generation_time:.2f}s"
    
    logger.info("Performance test completed successfully")


def main():
    """Run all predictive maintenance tests"""
    logger.info("Starting predictive maintenance tests...")
    
    start_time = time.time()
    
    try:
        # Run individual component tests
        test_cost_benefit_analyzer()
        test_maintenance_scheduler()
        test_inventory_optimizer()
        test_cmms_integrator()
        
        # Run integrated system test
        test_predictive_maintenance_system()
        
        # Run performance test
        run_performance_test()
        
        total_time = time.time() - start_time
        logger.info(f"All predictive maintenance tests completed successfully in {total_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()