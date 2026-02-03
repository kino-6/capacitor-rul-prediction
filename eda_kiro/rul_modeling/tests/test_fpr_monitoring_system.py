"""
Tests for FPR Monitoring and Alerting System

This module contains comprehensive tests for the FPR monitoring system,
including unit tests and integration tests.

Requirements: 10.3, 5.5
"""

import unittest
import tempfile
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from true_rul.fpr_monitor import (
    FPRMonitor, AlertConfig, FPRMetrics, Alert, FPRDatabase
)
from true_rul.data_structures import PredictionResult
from true_rul.structured_logger import configure_prediction_logging


class TestFPRDatabase(unittest.TestCase):
    """Test FPR database functionality"""
    
    def setUp(self):
        """Set up test database"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db = FPRDatabase(self.temp_db.name)
    
    def tearDown(self):
        """Clean up test database"""
        Path(self.temp_db.name).unlink(missing_ok=True)
    
    def test_database_initialization(self):
        """Test database table creation"""
        # Check that tables exist
        with sqlite3.connect(self.temp_db.name) as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['fpr_metrics', 'alerts', 'model_performance']
        for table in expected_tables:
            self.assertIn(table, tables)
    
    def test_store_and_retrieve_metrics(self):
        """Test storing and retrieving FPR metrics"""
        # Create test metrics
        metrics = FPRMetrics(
            timestamp=datetime.now(),
            fpr=0.03,
            tpr=0.85,
            precision=0.92,
            recall=0.85,
            f1_score=0.88,
            total_predictions=100,
            false_positives=3,
            true_negatives=87,
            false_negatives=5,
            true_positives=5,
            window_duration_minutes=15,
            capacitor_ids=["C1", "C2", "C3"]
        )
        
        # Store metrics
        metrics_id = self.db.store_metrics(metrics)
        self.assertIsInstance(metrics_id, int)
        
        # Retrieve metrics
        recent_metrics = self.db.get_recent_metrics(hours=1)
        self.assertEqual(len(recent_metrics), 1)
        
        retrieved = recent_metrics[0]
        self.assertAlmostEqual(retrieved.fpr, metrics.fpr, places=3)
        self.assertEqual(retrieved.total_predictions, metrics.total_predictions)
        self.assertEqual(retrieved.capacitor_ids, metrics.capacitor_ids)
    
    def test_store_and_retrieve_alerts(self):
        """Test storing and retrieving alerts"""
        # Create test alert
        alert = Alert(
            alert_id="test_alert_001",
            alert_type="fpr_threshold",
            severity="critical",
            message="Test alert message",
            timestamp=datetime.now(),
            metrics=None
        )
        
        # Store alert
        self.db.store_alert(alert)
        
        # Retrieve alerts
        active_alerts = self.db.get_active_alerts()
        self.assertEqual(len(active_alerts), 1)
        
        retrieved = active_alerts[0]
        self.assertEqual(retrieved.alert_id, alert.alert_id)
        self.assertEqual(retrieved.alert_type, alert.alert_type)
        self.assertEqual(retrieved.severity, alert.severity)
        self.assertFalse(retrieved.acknowledged)
        self.assertFalse(retrieved.resolved)


class TestFPRMonitor(unittest.TestCase):
    """Test FPR monitor functionality"""
    
    def setUp(self):
        """Set up test monitor"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        # Configure test logging
        configure_prediction_logging(log_file=None, enable_console=False)
        
        # Create test configuration
        self.config = AlertConfig(
            fpr_threshold=0.05,
            fpr_warning_threshold=0.03,
            min_predictions_for_alert=5,
            alert_cooldown_minutes=1,  # Short cooldown for testing
            drift_detection_window_hours=1,
            enable_email_alerts=False,
            enable_webhook_alerts=False
        )
        
        self.monitor = FPRMonitor(
            config=self.config,
            db_path=self.temp_db.name,
            monitoring_window_minutes=5
        )
    
    def tearDown(self):
        """Clean up test monitor"""
        if self.monitor.is_monitoring:
            self.monitor.stop_monitoring()
        Path(self.temp_db.name).unlink(missing_ok=True)
    
    def create_test_prediction(self, anomaly_flag: bool, anomaly_score: float = 0.5):
        """Create test prediction result"""
        return PredictionResult(
            rul_cycles=100,
            rul_confidence_lower=90,
            rul_confidence_upper=110,
            degradation_score=0.3,
            degradation_stage="healthy",
            anomaly_flag=anomaly_flag,
            anomaly_score=anomaly_score,
            feature_importance={"feature_1": 0.5, "feature_2": 0.5},
            timestamp=datetime.now().timestamp(),
            model_version="test_v1.0"
        )
    
    def test_record_prediction(self):
        """Test recording predictions"""
        # Record some predictions
        for i in range(10):
            prediction = self.create_test_prediction(anomaly_flag=i % 3 == 0)
            self.monitor.record_prediction(
                prediction_result=prediction,
                true_anomaly_label=i % 4 == 0,  # Different pattern for ground truth
                capacitor_id=f"C{i % 3 + 1}"
            )
        
        # Check that predictions were recorded
        self.assertEqual(len(self.monitor.prediction_buffer), 10)
    
    def test_calculate_fpr_metrics(self):
        """Test FPR metrics calculation"""
        # Record predictions with known FPR
        predictions_data = [
            # (predicted_anomaly, true_anomaly)
            (False, False),  # TN
            (False, False),  # TN
            (False, False),  # TN
            (True, False),   # FP
            (True, True),    # TP
            (False, True),   # FN
            (False, False),  # TN
            (True, False),   # FP
        ]
        
        for i, (pred_anomaly, true_anomaly) in enumerate(predictions_data):
            prediction = self.create_test_prediction(anomaly_flag=pred_anomaly)
            self.monitor.record_prediction(
                prediction_result=prediction,
                true_anomaly_label=true_anomaly,
                capacitor_id=f"C{i % 2 + 1}"
            )
        
        # Calculate metrics
        metrics = self.monitor.calculate_current_fpr()
        self.assertIsNotNone(metrics)
        
        # Expected: TN=4, FP=2, FN=1, TP=1
        # FPR = FP / (FP + TN) = 2 / (2 + 4) = 0.333
        expected_fpr = 2 / 6
        self.assertAlmostEqual(metrics.fpr, expected_fpr, places=3)
        
        # TPR = TP / (TP + FN) = 1 / (1 + 1) = 0.5
        expected_tpr = 1 / 2
        self.assertAlmostEqual(metrics.tpr, expected_tpr, places=3)
    
    def test_fpr_threshold_alerts(self):
        """Test FPR threshold alert generation"""
        # Record predictions that will exceed threshold
        # Create high FPR scenario: 6 FP out of 10 predictions
        predictions_data = [
            (True, False),   # FP
            (True, False),   # FP
            (True, False),   # FP
            (True, False),   # FP
            (True, False),   # FP
            (True, False),   # FP
            (False, False),  # TN
            (False, False),  # TN
            (False, False),  # TN
            (False, False),  # TN
        ]
        
        for i, (pred_anomaly, true_anomaly) in enumerate(predictions_data):
            prediction = self.create_test_prediction(anomaly_flag=pred_anomaly)
            self.monitor.record_prediction(
                prediction_result=prediction,
                true_anomaly_label=true_anomaly,
                capacitor_id=f"C{i % 2 + 1}"
            )
        
        # Calculate metrics and check for alerts
        metrics = self.monitor.calculate_current_fpr()
        self.assertIsNotNone(metrics)
        
        # FPR should be 6/10 = 0.6, which exceeds threshold of 0.05
        self.assertGreater(metrics.fpr, self.config.fpr_threshold)
        
        # Check for threshold alerts
        alerts = self.monitor.check_fpr_thresholds(metrics)
        self.assertGreater(len(alerts), 0)
        
        # Should have critical alert
        critical_alerts = [a for a in alerts if a.severity == "critical"]
        self.assertGreater(len(critical_alerts), 0)
    
    def test_monitoring_loop(self):
        """Test monitoring loop functionality"""
        # Start monitoring
        self.monitor.start_monitoring()
        self.assertTrue(self.monitor.is_monitoring)
        
        # Record some predictions
        for i in range(10):
            prediction = self.create_test_prediction(anomaly_flag=i % 5 == 0)
            self.monitor.record_prediction(
                prediction_result=prediction,
                true_anomaly_label=i % 6 == 0,
                capacitor_id=f"C{i % 3 + 1}"
            )
        
        # Wait for monitoring to process
        time.sleep(2)
        
        # Stop monitoring
        self.monitor.stop_monitoring()
        self.assertFalse(self.monitor.is_monitoring)
    
    def test_alert_callbacks(self):
        """Test custom alert callbacks"""
        callback_called = []
        
        def test_callback(alert):
            callback_called.append(alert)
        
        self.monitor.add_alert_callback(test_callback)
        
        # Create high FPR scenario to trigger alert
        for i in range(10):
            prediction = self.create_test_prediction(anomaly_flag=True)  # All anomalies
            self.monitor.record_prediction(
                prediction_result=prediction,
                true_anomaly_label=False,  # All false positives
                capacitor_id=f"C{i % 2 + 1}"
            )
        
        # Calculate metrics and trigger alerts
        metrics = self.monitor.calculate_current_fpr()
        alerts = self.monitor.check_fpr_thresholds(metrics)
        
        # Send alerts (should trigger callback)
        for alert in alerts:
            self.monitor.send_alert(alert)
        
        # Check that callback was called
        self.assertGreater(len(callback_called), 0)
    
    def test_dashboard_data_generation(self):
        """Test dashboard data generation"""
        # Record some predictions
        for i in range(20):
            prediction = self.create_test_prediction(anomaly_flag=i % 4 == 0)
            self.monitor.record_prediction(
                prediction_result=prediction,
                true_anomaly_label=i % 5 == 0,
                capacitor_id=f"C{i % 3 + 1}"
            )
        
        # Generate dashboard data
        dashboard_data = self.monitor.get_dashboard_data(hours=1)
        
        # Check data structure
        self.assertIn('current_metrics', dashboard_data)
        self.assertIn('recent_metrics', dashboard_data)
        self.assertIn('active_alerts', dashboard_data)
        self.assertIn('summary', dashboard_data)
        self.assertIn('config', dashboard_data)
        
        # Check summary data
        summary = dashboard_data['summary']
        self.assertIn('current_fpr', summary)
        self.assertIn('fpr_threshold', summary)
        self.assertIn('monitoring_status', summary)
    
    def test_fpr_report_generation(self):
        """Test FPR report generation"""
        # Record some predictions and store metrics
        for i in range(15):
            prediction = self.create_test_prediction(anomaly_flag=i % 3 == 0)
            self.monitor.record_prediction(
                prediction_result=prediction,
                true_anomaly_label=i % 4 == 0,
                capacitor_id=f"C{i % 2 + 1}"
            )
        
        # Calculate and store metrics
        metrics = self.monitor.calculate_current_fpr()
        if metrics:
            self.monitor.db.store_metrics(metrics)
        
        # Generate report
        report = self.monitor.generate_fpr_report(hours=1)
        
        # Check report content
        self.assertIn("FPR MONITORING AND TREND ANALYSIS REPORT", report)
        self.assertIn("CURRENT STATUS", report)
        self.assertIn("MONITORING CONFIGURATION", report)


class TestFPRMonitoringIntegration(unittest.TestCase):
    """Integration tests for FPR monitoring system"""
    
    def setUp(self):
        """Set up integration test"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        configure_prediction_logging(log_file=None, enable_console=False)
        
        self.config = AlertConfig(
            fpr_threshold=0.05,
            fpr_warning_threshold=0.03,
            min_predictions_for_alert=5,
            alert_cooldown_minutes=0.1,  # Very short for testing
            drift_detection_window_hours=0.1,
            enable_email_alerts=False,
            enable_webhook_alerts=False
        )
        
        self.monitor = FPRMonitor(
            config=self.config,
            db_path=self.temp_db.name,
            monitoring_window_minutes=1
        )
    
    def tearDown(self):
        """Clean up integration test"""
        if self.monitor.is_monitoring:
            self.monitor.stop_monitoring()
        Path(self.temp_db.name).unlink(missing_ok=True)
    
    def test_end_to_end_monitoring(self):
        """Test end-to-end monitoring workflow"""
        alerts_received = []
        
        def alert_handler(alert):
            alerts_received.append(alert)
        
        self.monitor.add_alert_callback(alert_handler)
        self.monitor.start_monitoring()
        
        try:
            # Phase 1: Normal operation (low FPR)
            for i in range(10):
                prediction = PredictionResult(
                    rul_cycles=100,
                    rul_confidence_lower=90,
                    rul_confidence_upper=110,
                    degradation_score=0.2,
                    degradation_stage="healthy",
                    anomaly_flag=False,  # No false positives
                    anomaly_score=0.1,
                    feature_importance={"feature_1": 0.5},
                    timestamp=datetime.now().timestamp(),
                    model_version="test_v1.0"
                )
                
                self.monitor.record_prediction(
                    prediction_result=prediction,
                    true_anomaly_label=False,  # All normal
                    capacitor_id=f"C{i % 2 + 1}"
                )
                time.sleep(0.1)
            
            # Wait for processing
            time.sleep(2)
            
            # Should have no alerts yet
            current_alerts = len(alerts_received)
            
            # Phase 2: High FPR operation (should trigger alerts)
            for i in range(10):
                prediction = PredictionResult(
                    rul_cycles=100,
                    rul_confidence_lower=90,
                    rul_confidence_upper=110,
                    degradation_score=0.2,
                    degradation_stage="healthy",
                    anomaly_flag=True,  # All false positives
                    anomaly_score=0.8,
                    feature_importance={"feature_1": 0.5},
                    timestamp=datetime.now().timestamp(),
                    model_version="test_v1.0"
                )
                
                self.monitor.record_prediction(
                    prediction_result=prediction,
                    true_anomaly_label=False,  # All normal (so all FP)
                    capacitor_id=f"C{i % 2 + 1}"
                )
                time.sleep(0.1)
            
            # Wait for alert processing
            time.sleep(5)
            
            # Should have received alerts
            self.assertGreater(len(alerts_received), current_alerts)
            
            # Check alert types
            alert_types = [a.alert_type for a in alerts_received]
            self.assertIn("fpr_threshold", alert_types)
            
            # Verify database storage
            active_alerts = self.monitor.db.get_active_alerts()
            self.assertGreater(len(active_alerts), 0)
            
            # Verify metrics storage
            recent_metrics = self.monitor.db.get_recent_metrics(hours=1)
            self.assertGreater(len(recent_metrics), 0)
            
        finally:
            self.monitor.stop_monitoring()


def run_fpr_monitoring_tests():
    """Run all FPR monitoring tests"""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestFPRDatabase))
    test_suite.addTest(unittest.makeSuite(TestFPRMonitor))
    test_suite.addTest(unittest.makeSuite(TestFPRMonitoringIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_fpr_monitoring_tests()
    if success:
        print("\n✓ All FPR monitoring tests passed!")
    else:
        print("\n✗ Some FPR monitoring tests failed!")
        sys.exit(1)