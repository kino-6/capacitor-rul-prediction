"""
Test script for FPR Monitoring and Alerting System

This script demonstrates the FPR monitoring system with synthetic data
and real predictions from the RUL system.

Requirements: 10.3, 5.5
"""

import sys
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import threading
import random

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from true_rul.fpr_monitor import FPRMonitor, AlertConfig, create_fpr_monitor
from true_rul.fpr_dashboard import FPRDashboard, create_fpr_dashboard
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
) -> tuple[PredictionResult, bool]:
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
        degradation_stage="healthy" if degradation_score < 0.3 else "degrading",
        anomaly_flag=predicted_anomaly,
        anomaly_score=anomaly_score,
        feature_importance={"feature_1": 0.3, "feature_2": 0.2, "feature_3": 0.5},
        timestamp=datetime.now().timestamp(),
        model_version="test_v1.0"
    )
    
    return prediction_result, true_anomaly


def simulate_prediction_stream(
    monitor: FPRMonitor,
    duration_minutes: int = 30,
    predictions_per_minute: int = 2,
    fpr_scenarios: list = None
):
    """
    Simulate a stream of predictions with varying FPR scenarios
    
    Args:
        monitor: FPR monitor instance
        duration_minutes: Duration of simulation
        predictions_per_minute: Rate of predictions
        fpr_scenarios: List of (duration_min, fpr_rate) tuples
    """
    if fpr_scenarios is None:
        # Default scenarios: normal -> warning -> critical -> recovery
        fpr_scenarios = [
            (10, 0.01),  # Normal operation (1% FPR)
            (5, 0.035),  # Warning level (3.5% FPR)
            (5, 0.07),   # Critical level (7% FPR)
            (10, 0.015)  # Recovery (1.5% FPR)
        ]
    
    logger.info(f"Starting prediction simulation for {duration_minutes} minutes")
    logger.info(f"FPR scenarios: {fpr_scenarios}")
    
    start_time = datetime.now()
    scenario_index = 0
    scenario_start = start_time
    current_fpr = fpr_scenarios[0][1]
    
    prediction_count = 0
    
    while (datetime.now() - start_time).total_seconds() < duration_minutes * 60:
        # Check if we should move to next scenario
        if scenario_index < len(fpr_scenarios) - 1:
            scenario_duration = fpr_scenarios[scenario_index][0]
            if (datetime.now() - scenario_start).total_seconds() > scenario_duration * 60:
                scenario_index += 1
                scenario_start = datetime.now()
                current_fpr = fpr_scenarios[scenario_index][1]
                logger.info(f"Switching to scenario {scenario_index + 1}: FPR = {current_fpr:.3f}")
        
        # Generate prediction
        prediction_result, true_anomaly = create_synthetic_prediction(
            anomaly_probability=0.1,
            fpr_rate=current_fpr
        )
        
        # Record prediction in monitor
        capacitor_id = f"C{random.randint(1, 8)}"
        monitor.record_prediction(
            prediction_result=prediction_result,
            true_anomaly_label=true_anomaly,
            capacitor_id=capacitor_id
        )
        
        prediction_count += 1
        
        if prediction_count % 10 == 0:
            logger.info(f"Generated {prediction_count} predictions, current FPR scenario: {current_fpr:.3f}")
        
        # Wait before next prediction
        time.sleep(60 / predictions_per_minute)
    
    logger.info(f"Simulation completed. Generated {prediction_count} predictions")


def test_fpr_monitoring_basic():
    """Test basic FPR monitoring functionality"""
    logger.info("Testing basic FPR monitoring functionality")
    
    # Configure logging
    configure_prediction_logging(log_file="logs/fpr_test.jsonl")
    
    # Create FPR monitor with test configuration
    config = AlertConfig(
        fpr_threshold=0.05,
        fpr_warning_threshold=0.03,
        min_predictions_for_alert=5,
        alert_cooldown_minutes=2,  # Short cooldown for testing
        drift_detection_window_hours=1,  # Short window for testing
        enable_email_alerts=False,
        enable_webhook_alerts=False
    )
    
    monitor = FPRMonitor(
        config=config,
        db_path="test_fpr_monitoring.db",
        monitoring_window_minutes=5  # Short window for testing
    )
    
    # Add custom alert callback for testing
    def test_alert_callback(alert):
        logger.warning(f"CUSTOM ALERT: {alert.alert_type} - {alert.message}")
    
    monitor.add_alert_callback(test_alert_callback)
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        # Generate some normal predictions
        logger.info("Phase 1: Generating normal predictions (low FPR)")
        for i in range(20):
            prediction_result, true_anomaly = create_synthetic_prediction(fpr_rate=0.01)
            monitor.record_prediction(
                prediction_result=prediction_result,
                true_anomaly_label=true_anomaly,
                capacitor_id=f"C{i % 4 + 1}"
            )
            time.sleep(0.5)
        
        # Wait for monitoring to process
        time.sleep(10)
        
        # Check current metrics
        current_metrics = monitor.calculate_current_fpr()
        if current_metrics:
            logger.info(f"Current FPR: {current_metrics.fpr:.3f}")
            logger.info(f"Total predictions: {current_metrics.total_predictions}")
        
        # Generate high FPR predictions to trigger alerts
        logger.info("Phase 2: Generating high FPR predictions (should trigger alerts)")
        for i in range(30):
            prediction_result, true_anomaly = create_synthetic_prediction(fpr_rate=0.08)
            monitor.record_prediction(
                prediction_result=prediction_result,
                true_anomaly_label=true_anomaly,
                capacitor_id=f"C{i % 4 + 1}"
            )
            time.sleep(0.3)
        
        # Wait for alerts to be processed
        time.sleep(15)
        
        # Check metrics again
        current_metrics = monitor.calculate_current_fpr()
        if current_metrics:
            logger.info(f"Current FPR after high-FPR phase: {current_metrics.fpr:.3f}")
        
        # Get active alerts
        active_alerts = monitor.db.get_active_alerts()
        logger.info(f"Active alerts: {len(active_alerts)}")
        for alert in active_alerts:
            logger.info(f"  - {alert.alert_type}: {alert.message}")
        
        # Generate FPR report
        report = monitor.generate_fpr_report(hours=1)
        logger.info("FPR Report:")
        print(report)
        
        # Test dashboard data
        dashboard_data = monitor.get_dashboard_data(hours=1)
        logger.info(f"Dashboard data summary: {dashboard_data['summary']}")
        
    finally:
        monitor.stop_monitoring()
    
    logger.info("Basic FPR monitoring test completed")


def test_fpr_dashboard():
    """Test FPR dashboard functionality"""
    logger.info("Testing FPR dashboard")
    
    # Create FPR monitor
    monitor = create_fpr_monitor(
        fpr_threshold=0.05,
        warning_threshold=0.03,
        monitoring_window_minutes=5
    )
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Create dashboard
    dashboard = create_fpr_dashboard(
        fpr_monitor=monitor,
        host="127.0.0.1",
        port=8081,
        enable_websocket=True
    )
    
    # Start prediction simulation in background
    def run_simulation():
        simulate_prediction_stream(
            monitor=monitor,
            duration_minutes=10,
            predictions_per_minute=3,
            fpr_scenarios=[
                (3, 0.01),   # Normal
                (2, 0.04),   # Warning
                (2, 0.08),   # Critical
                (3, 0.02)    # Recovery
            ]
        )
    
    simulation_thread = threading.Thread(target=run_simulation, daemon=True)
    simulation_thread.start()
    
    logger.info("Starting FPR dashboard on http://127.0.0.1:8081")
    logger.info("Dashboard will run for 10 minutes with live data simulation")
    logger.info("Press Ctrl+C to stop")
    
    try:
        # Run dashboard (this will block)
        dashboard.run()
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    finally:
        monitor.stop_monitoring()


def test_drift_detection():
    """Test model drift detection"""
    logger.info("Testing model drift detection")
    
    # Create monitor with short drift detection window
    config = AlertConfig(
        fpr_threshold=0.05,
        fpr_warning_threshold=0.03,
        drift_detection_window_hours=0.5,  # 30 minutes for testing
        drift_threshold=0.02,
        min_predictions_for_alert=5
    )
    
    monitor = FPRMonitor(
        config=config,
        db_path="test_drift_detection.db",
        monitoring_window_minutes=3
    )
    
    monitor.start_monitoring()
    
    try:
        # Phase 1: Establish baseline (low FPR)
        logger.info("Phase 1: Establishing baseline performance")
        for i in range(50):
            prediction_result, true_anomaly = create_synthetic_prediction(fpr_rate=0.015)
            monitor.record_prediction(
                prediction_result=prediction_result,
                true_anomaly_label=true_anomaly,
                capacitor_id=f"C{i % 4 + 1}"
            )
            time.sleep(0.2)
        
        time.sleep(10)  # Let monitoring process
        
        # Phase 2: Gradual drift (increasing FPR)
        logger.info("Phase 2: Simulating gradual model drift")
        for i in range(50):
            # Gradually increase FPR to simulate drift
            drift_fpr = 0.015 + (i / 50) * 0.04  # From 1.5% to 5.5%
            prediction_result, true_anomaly = create_synthetic_prediction(fpr_rate=drift_fpr)
            monitor.record_prediction(
                prediction_result=prediction_result,
                true_anomaly_label=true_anomaly,
                capacitor_id=f"C{i % 4 + 1}"
            )
            time.sleep(0.2)
        
        time.sleep(15)  # Let drift detection run
        
        # Check for drift alerts
        active_alerts = monitor.db.get_active_alerts()
        drift_alerts = [a for a in active_alerts if a.alert_type == "drift_detected"]
        
        logger.info(f"Drift alerts detected: {len(drift_alerts)}")
        for alert in drift_alerts:
            logger.info(f"  - {alert.message}")
        
        # Generate report
        report = monitor.generate_fpr_report(hours=1)
        logger.info("Drift Detection Report:")
        print(report)
        
    finally:
        monitor.stop_monitoring()
    
    logger.info("Drift detection test completed")


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test FPR Monitoring System")
    parser.add_argument(
        "--test",
        choices=["basic", "dashboard", "drift", "all"],
        default="basic",
        help="Test to run"
    )
    
    args = parser.parse_args()
    
    # Create output directories
    Path("logs").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    
    if args.test == "basic" or args.test == "all":
        test_fpr_monitoring_basic()
    
    if args.test == "drift" or args.test == "all":
        test_drift_detection()
    
    if args.test == "dashboard":
        test_fpr_dashboard()
    
    logger.info("All tests completed successfully!")


if __name__ == "__main__":
    main()