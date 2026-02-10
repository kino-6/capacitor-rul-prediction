#!/usr/bin/env python3
"""
Test script for production monitoring and alerting system
"""

import asyncio
import logging
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from true_rul.production_monitoring import (
    ProductionMonitor,
    LogAlertHandler,
    FileAlertHandler,
    AlertSeverity,
    MetricType,
    create_production_monitor
)
from true_rul.monitoring_dashboard import (
    MonitoringDashboard,
    create_monitoring_dashboard
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_basic_monitoring():
    """Test basic monitoring functionality"""
    logger.info("Testing basic monitoring functionality...")
    
    # Create monitor with custom thresholds
    alert_thresholds = {
        "fpr_threshold": 0.03,  # Lower threshold for testing
        "prediction_latency_ms": 500,  # Lower threshold for testing
        "error_rate_threshold": 0.005
    }
    
    monitor = create_production_monitor(
        alert_thresholds=alert_thresholds,
        check_interval=1.0  # Fast checking for testing
    )
    
    # Start monitoring
    await monitor.start_monitoring()
    
    # Simulate some predictions
    logger.info("Simulating predictions...")
    
    # Good predictions
    for i in range(10):
        monitor.record_prediction(latency_ms=100 + i * 10, had_error=False)
        monitor.record_anomaly_prediction(is_false_positive=False)
        await asyncio.sleep(0.1)
    
    # Let monitoring run for a bit
    await asyncio.sleep(2)
    
    # Get dashboard data
    dashboard_data = monitor.get_dashboard_data()
    logger.info(f"Dashboard data: {dashboard_data}")
    
    # Verify basic functionality
    assert dashboard_data["system_health"]["overall_healthy"], "System should be healthy"
    assert dashboard_data["performance"]["error_rate"] == 0.0, "Error rate should be 0"
    assert dashboard_data["alerts"]["active_count"] == 0, "Should have no active alerts"
    
    # Stop monitoring
    await monitor.stop_monitoring()
    
    logger.info("✓ Basic monitoring test passed")


async def test_alert_generation():
    """Test alert generation for various conditions"""
    logger.info("Testing alert generation...")
    
    # Create temporary file for alerts
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        alert_file = Path(f.name)
    
    try:
        # Create monitor with strict thresholds
        alert_thresholds = {
            "fpr_threshold": 0.01,  # Very low threshold
            "prediction_latency_ms": 50,  # Very low threshold
            "error_rate_threshold": 0.01
        }
        
        # Add file alert handler
        file_handler = FileAlertHandler(alert_file)
        monitor = create_production_monitor(
            alert_handlers=[LogAlertHandler(), file_handler],
            alert_thresholds=alert_thresholds,
            check_interval=0.5  # Fast checking
        )
        
        await monitor.start_monitoring()
        
        # Simulate high latency predictions
        logger.info("Simulating high latency predictions...")
        for i in range(5):
            monitor.record_prediction(latency_ms=200, had_error=False)  # Above threshold
            await asyncio.sleep(0.1)
        
        # Simulate false positives
        logger.info("Simulating false positives...")
        for i in range(10):
            monitor.record_anomaly_prediction(is_false_positive=True)
            await asyncio.sleep(0.1)
        
        # Simulate errors
        logger.info("Simulating errors...")
        for i in range(3):
            monitor.record_prediction(latency_ms=100, had_error=True)
            await asyncio.sleep(0.1)
        
        # Wait for monitoring to detect issues
        await asyncio.sleep(2)
        
        # Check dashboard data
        dashboard_data = monitor.get_dashboard_data()
        logger.info(f"Alert dashboard data: {dashboard_data}")
        
        # Should have active alerts
        active_alerts = dashboard_data["alerts"]["active_count"]
        logger.info(f"Active alerts: {active_alerts}")
        
        # Verify alerts were generated
        assert active_alerts > 0, "Should have active alerts"
        
        # Check alert file
        if alert_file.exists():
            with open(alert_file, 'r') as f:
                alert_lines = f.readlines()
            logger.info(f"Found {len(alert_lines)} alerts in file")
            assert len(alert_lines) > 0, "Should have alerts in file"
        
        await monitor.stop_monitoring()
        
        logger.info("✓ Alert generation test passed")
        
    finally:
        alert_file.unlink(missing_ok=True)


async def test_dashboard_generation():
    """Test dashboard HTML generation"""
    logger.info("Testing dashboard generation...")
    
    monitor = create_production_monitor(check_interval=1.0)
    dashboard = create_monitoring_dashboard(monitor, update_interval=1.0)
    
    # Start systems
    await monitor.start_monitoring()
    await dashboard.start()
    
    # Simulate some activity
    for i in range(5):
        monitor.record_prediction(latency_ms=150 + i * 20, had_error=False)
        monitor.record_anomaly_prediction(is_false_positive=False)
        await asyncio.sleep(0.1)
    
    # Wait for dashboard to update
    await asyncio.sleep(2)
    
    # Generate HTML dashboard
    html_content = dashboard.generate_html_dashboard()
    
    # Verify HTML content
    assert "RUL Prediction System" in html_content, "Should contain title"
    assert "System Health" in html_content, "Should contain health section"
    assert "False Positive Rate" in html_content, "Should contain FPR metric"
    assert "Throughput" in html_content, "Should contain throughput metric"
    
    # Save dashboard to file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
        dashboard_file = Path(f.name)
    
    try:
        dashboard.save_dashboard_html(dashboard_file)
        
        # Verify file was created
        assert dashboard_file.exists(), "Dashboard file should be created"
        
        # Verify file content
        with open(dashboard_file, 'r') as f:
            file_content = f.read()
        
        assert len(file_content) > 1000, "Dashboard file should have substantial content"
        assert "RUL Prediction System" in file_content, "File should contain title"
        
        logger.info(f"Dashboard saved to {dashboard_file}")
        
    finally:
        dashboard_file.unlink(missing_ok=True)
    
    # Stop systems
    await dashboard.stop()
    await monitor.stop_monitoring()
    
    logger.info("✓ Dashboard generation test passed")


async def test_performance_tracking():
    """Test performance tracking functionality"""
    logger.info("Testing performance tracking...")
    
    monitor = create_production_monitor(check_interval=0.5)
    
    # Record various performance metrics
    latencies = [50, 75, 100, 125, 150, 200, 300, 400, 500, 1000]
    
    for latency in latencies:
        monitor.record_prediction(latency_ms=latency, had_error=False)
    
    # Record some errors
    for i in range(2):
        monitor.record_prediction(latency_ms=100, had_error=True)
    
    # Record anomaly predictions
    for i in range(20):
        is_fp = i < 2  # First 2 are false positives
        monitor.record_anomaly_prediction(is_false_positive=is_fp)
    
    # Get performance metrics
    perf_metrics = monitor.performance_tracker.get_performance_metrics()
    
    # Verify metrics
    assert len(perf_metrics.prediction_latency_ms) == 12, "Should have 12 latency records"
    assert perf_metrics.error_rate > 0, "Should have non-zero error rate"
    assert perf_metrics.fpr_rate > 0, "Should have non-zero FPR"
    
    # Check percentiles
    percentiles = perf_metrics.get_latency_percentiles()
    assert percentiles["p50"] > 0, "P50 should be positive"
    assert percentiles["p95"] > percentiles["p50"], "P95 should be higher than P50"
    assert percentiles["p99"] >= percentiles["p95"], "P99 should be >= P95"
    
    logger.info(f"Performance metrics: {perf_metrics}")
    logger.info(f"Latency percentiles: {percentiles}")
    
    logger.info("✓ Performance tracking test passed")


async def test_health_checks():
    """Test health check functionality"""
    logger.info("Testing health checks...")
    
    monitor = create_production_monitor(check_interval=0.5)
    
    # Add custom health check that fails
    def failing_check():
        return False
    
    def passing_check():
        return True
    
    monitor.health_checker.register_check("test_failing", failing_check)
    monitor.health_checker.register_check("test_passing", passing_check)
    
    # Start monitoring
    await monitor.start_monitoring()
    
    # Wait for health checks to run
    await asyncio.sleep(1)
    
    # Check results
    health_results = monitor.health_checker.run_checks()
    logger.info(f"Health check results: {health_results}")
    
    assert "test_failing" in health_results, "Should have failing check"
    assert "test_passing" in health_results, "Should have passing check"
    assert health_results["test_failing"] == False, "Failing check should fail"
    assert health_results["test_passing"] == True, "Passing check should pass"
    
    # Check that alerts were generated for failing check
    dashboard_data = monitor.get_dashboard_data()
    active_alerts = dashboard_data["alerts"]["active_alerts"]
    
    failing_alert_found = any(
        "test_failing" in alert.get("message", "") 
        for alert in active_alerts
    )
    
    assert failing_alert_found, "Should have alert for failing health check"
    
    await monitor.stop_monitoring()
    
    logger.info("✓ Health checks test passed")


async def main():
    """Run all tests"""
    logger.info("Starting production monitoring tests...")
    
    try:
        await test_basic_monitoring()
        await test_alert_generation()
        await test_dashboard_generation()
        await test_performance_tracking()
        await test_health_checks()
        
        logger.info("🎉 All production monitoring tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())