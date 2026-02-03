"""
Simple test script for FPR Monitoring System

This script tests the core FPR monitoring functionality without
requiring external dependencies like FastAPI.

Requirements: 10.3, 5.5
"""

import sys
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from true_rul.fpr_monitor import FPRMonitor, AlertConfig
from true_rul.data_structures import PredictionResult
from true_rul.structured_logger import configure_prediction_logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_prediction(
    anomaly_probability: float = 0.1,
    fpr_rate: float = 0.02
) -> tuple:
    """
    Create synthetic prediction result with known ground truth
    
    Args:
        anomaly_probability: Probability of true anomaly
        fpr_rate: False positive rate to simulate
        
    Returns:
        Tuple of (prediction_result, true_anomaly_label)
    """
    # Generate true anomaly label
    true_anomaly = random.random() < anomaly_probability
    
    # Generate prediction with specified FPR
    if true_anomaly:
        # True positive (correctly detect anomaly)
        predicted_anomaly = random.random() < 0.9  # 90% TPR
    else:
        # False positive (incorrectly flag normal as anomaly)
        predicted_anomaly = random.random() < fpr_rate
    
    # Generate other prediction components
    rul_cycles = random.randint(10, 200) if not true_anomaly else random.randint(1, 20)
    degradation_score = random.uniform(0.1, 0.4) if not true_anomaly else random.uniform(0.6, 1.0)
    anomaly_score = random.uniform(0.0, 0.3) if not predicted_anomaly else random.uniform(0.7, 1.0)
    
    prediction_result = PredictionResult(
        rul_cycles=rul_cycles,
        rul_confidence_lower=max(1, rul_cycles - 10),
        rul_confidence_upper=rul_cycles + 10,
        degradation_score=degradation_score,
        degradation_stage="healthy" if degradation_score < 0.3 else "early_degradation",
        anomaly_flag=predicted_anomaly,
        anomaly_score=anomaly_score,
        feature_importance={"feature_1": 0.3, "feature_2": 0.2, "feature_3": 0.5},
        timestamp=datetime.now().timestamp(),
        model_version="test_v1.0"
    )
    
    return prediction_result, true_anomaly


def test_basic_fpr_monitoring():
    """Test basic FPR monitoring functionality"""
    logger.info("Testing basic FPR monitoring functionality")
    
    # Configure logging
    configure_prediction_logging(log_file="logs/fpr_test_simple.jsonl")
    
    # Create FPR monitor with test configuration
    config = AlertConfig(
        fpr_threshold=0.05,
        fpr_warning_threshold=0.03,
        min_predictions_for_alert=5,
        alert_cooldown_minutes=1,  # Short cooldown for testing
        drift_detection_window_hours=1,  # Short window for testing
        enable_email_alerts=False,
        enable_webhook_alerts=False
    )
    
    monitor = FPRMonitor(
        config=config,
        db_path="test_fpr_monitoring_simple.db",
        monitoring_window_minutes=5  # Short window for testing
    )
    
    # Track alerts
    alerts_received = []
    
    def test_alert_callback(alert):
        logger.warning(f"ALERT RECEIVED: {alert.alert_type} - {alert.message}")
        alerts_received.append(alert)
    
    monitor.add_alert_callback(test_alert_callback)
    
    try:
        # Phase 1: Generate normal predictions (low FPR)
        logger.info("Phase 1: Generating normal predictions (low FPR)")
        for i in range(20):
            prediction_result, true_anomaly = create_synthetic_prediction(fpr_rate=0.01)
            monitor.record_prediction(
                prediction_result=prediction_result,
                true_anomaly_label=true_anomaly,
                capacitor_id=f"C{i % 4 + 1}"
            )
        
        # Calculate current metrics
        current_metrics = monitor.calculate_current_fpr()
        if current_metrics:
            logger.info(f"Phase 1 - Current FPR: {current_metrics.fpr:.3f}")
            logger.info(f"Phase 1 - Total predictions: {current_metrics.total_predictions}")
            
            # Check for alerts (should be none)
            alerts = monitor.check_fpr_thresholds(current_metrics)
            logger.info(f"Phase 1 - Alerts generated: {len(alerts)}")
        
        # Phase 2: Generate high FPR predictions (should trigger alerts)
        logger.info("Phase 2: Generating high FPR predictions (should trigger alerts)")
        for i in range(30):
            prediction_result, true_anomaly = create_synthetic_prediction(fpr_rate=0.08)
            monitor.record_prediction(
                prediction_result=prediction_result,
                true_anomaly_label=true_anomaly,
                capacitor_id=f"C{i % 4 + 1}"
            )
        
        # Calculate metrics again
        current_metrics = monitor.calculate_current_fpr()
        if current_metrics:
            logger.info(f"Phase 2 - Current FPR: {current_metrics.fpr:.3f}")
            logger.info(f"Phase 2 - Total predictions: {current_metrics.total_predictions}")
            
            # Check for threshold violations
            alerts = monitor.check_fpr_thresholds(current_metrics)
            logger.info(f"Phase 2 - Alerts generated: {len(alerts)}")
            
            # Send alerts
            for alert in alerts:
                monitor.send_alert(alert)
        
        # Check database storage
        active_alerts = monitor.db.get_active_alerts()
        logger.info(f"Active alerts in database: {len(active_alerts)}")
        for alert in active_alerts:
            logger.info(f"  - {alert.alert_type}: {alert.severity} - {alert.message[:100]}...")
        
        # Generate FPR report
        logger.info("Generating FPR report...")
        report = monitor.generate_fpr_report(hours=1)
        print("\n" + "="*80)
        print("FPR MONITORING REPORT")
        print("="*80)
        print(report)
        
        # Test dashboard data generation
        logger.info("Testing dashboard data generation...")
        dashboard_data = monitor.get_dashboard_data(hours=1)
        summary = dashboard_data['summary']
        logger.info(f"Dashboard summary:")
        logger.info(f"  - Current FPR: {summary.get('current_fpr', 'N/A')}")
        logger.info(f"  - FPR Trend: {summary.get('fpr_trend', 'N/A')}")
        logger.info(f"  - Active Alerts: {summary.get('active_alert_count', 0)}")
        logger.info(f"  - Total Predictions (24h): {summary.get('total_predictions_24h', 0)}")
        
        # Test drift detection
        logger.info("Testing drift detection...")
        drift_alerts = monitor.detect_model_drift()
        logger.info(f"Drift alerts detected: {len(drift_alerts)}")
        for alert in drift_alerts:
            logger.info(f"  - {alert.message}")
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("TEST SUMMARY")
        logger.info("="*60)
        logger.info(f"Total predictions recorded: {len(monitor.prediction_buffer)}")
        logger.info(f"Alerts received via callback: {len(alerts_received)}")
        logger.info(f"Active alerts in database: {len(active_alerts)}")
        
        if current_metrics:
            fpr_requirement_met = current_metrics.fpr < 0.05
            logger.info(f"Final FPR: {current_metrics.fpr:.3f}")
            logger.info(f"FPR Requirement (<5%): {'✓ MET' if fpr_requirement_met else '✗ NOT MET'}")
        
        # Test passed if we can generate metrics and alerts
        test_passed = (
            current_metrics is not None and
            len(active_alerts) > 0 and
            dashboard_data is not None
        )
        
        if test_passed:
            logger.info("✓ FPR monitoring test PASSED")
        else:
            logger.error("✗ FPR monitoring test FAILED")
            
        return test_passed
        
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fpr_calculation_accuracy():
    """Test FPR calculation accuracy with known data"""
    logger.info("Testing FPR calculation accuracy")
    
    config = AlertConfig(
        fpr_threshold=0.05,
        min_predictions_for_alert=5
    )
    
    monitor = FPRMonitor(
        config=config,
        db_path="test_fpr_accuracy.db",
        monitoring_window_minutes=10
    )
    
    # Create predictions with known FPR
    # Expected: 2 FP, 6 TN, 1 TP, 1 FN
    # FPR = FP / (FP + TN) = 2 / (2 + 6) = 0.25
    test_cases = [
        # (predicted_anomaly, true_anomaly)
        (False, False),  # TN
        (False, False),  # TN
        (False, False),  # TN
        (False, False),  # TN
        (False, False),  # TN
        (False, False),  # TN
        (True, False),   # FP
        (True, False),   # FP
        (True, True),    # TP
        (False, True),   # FN
    ]
    
    for i, (pred_anomaly, true_anomaly) in enumerate(test_cases):
        prediction_result = PredictionResult(
            rul_cycles=100,
            rul_confidence_lower=90,
            rul_confidence_upper=110,
            degradation_score=0.3,
            degradation_stage="healthy",
            anomaly_flag=pred_anomaly,
            anomaly_score=0.5,
            feature_importance={"feature_1": 0.5},
            timestamp=datetime.now().timestamp(),
            model_version="test_v1.0"
        )
        
        monitor.record_prediction(
            prediction_result=prediction_result,
            true_anomaly_label=true_anomaly,
            capacitor_id=f"C{i % 2 + 1}"
        )
    
    # Calculate metrics
    metrics = monitor.calculate_current_fpr()
    
    if metrics:
        expected_fpr = 2 / 8  # 2 FP / (2 FP + 6 TN) = 0.25
        expected_tpr = 1 / 2  # 1 TP / (1 TP + 1 FN) = 0.5
        
        logger.info(f"Expected FPR: {expected_fpr:.3f}, Actual FPR: {metrics.fpr:.3f}")
        logger.info(f"Expected TPR: {expected_tpr:.3f}, Actual TPR: {metrics.tpr:.3f}")
        
        fpr_accurate = abs(metrics.fpr - expected_fpr) < 0.001
        tpr_accurate = abs(metrics.tpr - expected_tpr) < 0.001
        
        if fpr_accurate and tpr_accurate:
            logger.info("✓ FPR calculation accuracy test PASSED")
            return True
        else:
            logger.error("✗ FPR calculation accuracy test FAILED")
            return False
    else:
        logger.error("✗ Could not calculate metrics")
        return False


def main():
    """Main test function"""
    logger.info("Starting FPR Monitoring System Tests")
    
    # Create output directories
    Path("logs").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    
    # Run tests
    test_results = []
    
    logger.info("\n" + "="*60)
    logger.info("TEST 1: Basic FPR Monitoring")
    logger.info("="*60)
    result1 = test_basic_fpr_monitoring()
    test_results.append(("Basic FPR Monitoring", result1))
    
    logger.info("\n" + "="*60)
    logger.info("TEST 2: FPR Calculation Accuracy")
    logger.info("="*60)
    result2 = test_fpr_calculation_accuracy()
    test_results.append(("FPR Calculation Accuracy", result2))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("FINAL TEST RESULTS")
    logger.info("="*60)
    
    all_passed = True
    for test_name, passed in test_results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 All FPR monitoring tests PASSED!")
        return True
    else:
        logger.error("\n❌ Some FPR monitoring tests FAILED!")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)