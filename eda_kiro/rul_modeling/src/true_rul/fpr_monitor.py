"""
FPR Monitoring and Alerting System

This module implements real-time FPR monitoring, automated alerting,
trend analysis, and model drift detection for the RUL prediction system.

Requirements: 10.3, 5.5
"""

import logging
import json
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from pathlib import Path
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
import numpy as np
from sklearn.metrics import confusion_matrix
import sqlite3
try:
    import smtplib
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False

from .data_structures import PredictionResult
from .structured_logger import get_prediction_logger
from .model_evaluator import ModelEvaluator

logger = logging.getLogger(__name__)


@dataclass
class FPRMetrics:
    """FPR metrics for a time window"""
    timestamp: datetime
    fpr: float
    tpr: float
    precision: float
    recall: float
    f1_score: float
    total_predictions: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    true_positives: int
    window_duration_minutes: int
    capacitor_ids: List[str]


@dataclass
class AlertConfig:
    """Configuration for FPR alerts"""
    fpr_threshold: float = 0.05  # 5% FPR threshold
    fpr_warning_threshold: float = 0.03  # 3% warning threshold
    min_predictions_for_alert: int = 10  # Minimum predictions before alerting
    alert_cooldown_minutes: int = 30  # Cooldown between similar alerts
    drift_detection_window_hours: int = 24  # Window for drift detection
    drift_threshold: float = 0.02  # Significant drift threshold
    
    # Email configuration
    enable_email_alerts: bool = False
    smtp_server: str = "localhost"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    alert_recipients: List[str] = None
    
    # Webhook configuration
    enable_webhook_alerts: bool = False
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = None
    
    def __post_init__(self):
        if self.alert_recipients is None:
            self.alert_recipients = []
        if self.webhook_headers is None:
            self.webhook_headers = {}


@dataclass
class Alert:
    """Alert information"""
    alert_id: str
    alert_type: str  # "fpr_threshold", "fpr_warning", "drift_detected"
    severity: str  # "critical", "warning", "info"
    message: str
    timestamp: datetime
    metrics: FPRMetrics
    acknowledged: bool = False
    resolved: bool = False


class FPRDatabase:
    """SQLite database for storing FPR metrics and alerts"""
    
    def __init__(self, db_path: str = "fpr_monitoring.db"):
        """
        Initialize FPR database
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            # FPR metrics table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fpr_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    fpr REAL NOT NULL,
                    tpr REAL NOT NULL,
                    precision_score REAL NOT NULL,
                    recall_score REAL NOT NULL,
                    f1_score REAL NOT NULL,
                    total_predictions INTEGER NOT NULL,
                    false_positives INTEGER NOT NULL,
                    true_negatives INTEGER NOT NULL,
                    false_negatives INTEGER NOT NULL,
                    true_positives INTEGER NOT NULL,
                    window_duration_minutes INTEGER NOT NULL,
                    capacitor_ids TEXT NOT NULL
                )
            """)
            
            # Alerts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_id TEXT UNIQUE NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    metrics_id INTEGER,
                    acknowledged BOOLEAN DEFAULT FALSE,
                    resolved BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (metrics_id) REFERENCES fpr_metrics (id)
                )
            """)
            
            # Model performance history table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    fpr REAL NOT NULL,
                    tpr REAL NOT NULL,
                    precision_score REAL NOT NULL,
                    recall_score REAL NOT NULL,
                    f1_score REAL NOT NULL,
                    roc_auc REAL NOT NULL,
                    n_samples INTEGER NOT NULL
                )
            """)
            
            conn.commit()
    
    def store_metrics(self, metrics: FPRMetrics) -> int:
        """
        Store FPR metrics in database
        
        Args:
            metrics: FPR metrics to store
            
        Returns:
            Database ID of stored metrics
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO fpr_metrics (
                    timestamp, fpr, tpr, precision_score, recall_score, f1_score,
                    total_predictions, false_positives, true_negatives,
                    false_negatives, true_positives, window_duration_minutes,
                    capacitor_ids
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics.timestamp.isoformat(),
                metrics.fpr,
                metrics.tpr,
                metrics.precision,
                metrics.recall,
                metrics.f1_score,
                metrics.total_predictions,
                metrics.false_positives,
                metrics.true_negatives,
                metrics.false_negatives,
                metrics.true_positives,
                metrics.window_duration_minutes,
                json.dumps(metrics.capacitor_ids)
            ))
            return cursor.lastrowid
    
    def store_alert(self, alert: Alert, metrics_id: Optional[int] = None) -> None:
        """
        Store alert in database
        
        Args:
            alert: Alert to store
            metrics_id: Optional ID of associated metrics
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO alerts (
                    alert_id, alert_type, severity, message, timestamp,
                    metrics_id, acknowledged, resolved
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert.alert_id,
                alert.alert_type,
                alert.severity,
                alert.message,
                alert.timestamp.isoformat(),
                metrics_id,
                alert.acknowledged,
                alert.resolved
            ))
            conn.commit()
    
    def get_recent_metrics(self, hours: int = 24) -> List[FPRMetrics]:
        """
        Get recent FPR metrics
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List of recent FPR metrics
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM fpr_metrics 
                WHERE timestamp > ? 
                ORDER BY timestamp DESC
            """, (cutoff_time.isoformat(),))
            
            metrics_list = []
            for row in cursor.fetchall():
                metrics = FPRMetrics(
                    timestamp=datetime.fromisoformat(row[1]),
                    fpr=row[2],
                    tpr=row[3],
                    precision=row[4],
                    recall=row[5],
                    f1_score=row[6],
                    total_predictions=row[7],
                    false_positives=row[8],
                    true_negatives=row[9],
                    false_negatives=row[10],
                    true_positives=row[11],
                    window_duration_minutes=row[12],
                    capacitor_ids=json.loads(row[13])
                )
                metrics_list.append(metrics)
            
            return metrics_list
    
    def get_active_alerts(self) -> List[Alert]:
        """
        Get active (unresolved) alerts
        
        Returns:
            List of active alerts
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT a.alert_id, a.alert_type, a.severity, a.message, a.timestamp,
                       a.acknowledged, a.resolved, a.metrics_id
                FROM alerts a
                WHERE a.resolved = FALSE
                ORDER BY a.timestamp DESC
            """)
            
            alerts = []
            for row in cursor.fetchall():
                # Get metrics separately if metrics_id exists
                metrics = None
                if row[7]:  # metrics_id exists
                    metrics_cursor = conn.execute("""
                        SELECT timestamp, fpr, tpr, precision_score, recall_score, f1_score,
                               total_predictions, false_positives, true_negatives,
                               false_negatives, true_positives, window_duration_minutes,
                               capacitor_ids
                        FROM fpr_metrics WHERE id = ?
                    """, (row[7],))
                    
                    metrics_row = metrics_cursor.fetchone()
                    if metrics_row:
                        metrics = FPRMetrics(
                            timestamp=datetime.fromisoformat(metrics_row[0]) if metrics_row[0] else datetime.now(),
                            fpr=metrics_row[1] if metrics_row[1] is not None else 0.0,
                            tpr=metrics_row[2] if metrics_row[2] is not None else 0.0,
                            precision=metrics_row[3] if metrics_row[3] is not None else 0.0,
                            recall=metrics_row[4] if metrics_row[4] is not None else 0.0,
                            f1_score=metrics_row[5] if metrics_row[5] is not None else 0.0,
                            total_predictions=metrics_row[6] if metrics_row[6] is not None else 0,
                            false_positives=metrics_row[7] if metrics_row[7] is not None else 0,
                            true_negatives=metrics_row[8] if metrics_row[8] is not None else 0,
                            false_negatives=metrics_row[9] if metrics_row[9] is not None else 0,
                            true_positives=metrics_row[10] if metrics_row[10] is not None else 0,
                            window_duration_minutes=metrics_row[11] if metrics_row[11] is not None else 0,
                            capacitor_ids=json.loads(metrics_row[12]) if metrics_row[12] else []
                        )
                
                alert = Alert(
                    alert_id=row[0],
                    alert_type=row[1],
                    severity=row[2],
                    message=row[3],
                    timestamp=datetime.fromisoformat(row[4]) if row[4] else datetime.now(),
                    metrics=metrics,
                    acknowledged=bool(row[5]),
                    resolved=bool(row[6])
                )
                alerts.append(alert)
            
            return alerts


class FPRMonitor:
    """
    Real-time FPR monitoring system
    
    This class monitors FPR in real-time, detects threshold violations,
    performs trend analysis, and triggers alerts when necessary.
    """
    
    def __init__(
        self,
        config: AlertConfig = None,
        db_path: str = "fpr_monitoring.db",
        monitoring_window_minutes: int = 15
    ):
        """
        Initialize FPR monitor
        
        Args:
            config: Alert configuration
            db_path: Path to SQLite database
            monitoring_window_minutes: Window size for FPR calculation
        """
        self.config = config or AlertConfig()
        self.db = FPRDatabase(db_path)
        self.monitoring_window_minutes = monitoring_window_minutes
        
        # In-memory storage for real-time monitoring
        self.prediction_buffer = deque(maxlen=10000)  # Store recent predictions
        self.alert_cooldowns = {}  # Track alert cooldowns
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Callbacks for custom alert handling
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
        # Prediction logger
        self.prediction_logger = get_prediction_logger()
        
        logger.info(f"FPR Monitor initialized with {monitoring_window_minutes}min windows")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]) -> None:
        """
        Add custom alert callback
        
        Args:
            callback: Function to call when alert is triggered
        """
        self.alert_callbacks.append(callback)
    
    def record_prediction(
        self,
        prediction_result: PredictionResult,
        true_anomaly_label: Optional[bool] = None,
        capacitor_id: str = "unknown"
    ) -> None:
        """
        Record a prediction for FPR monitoring
        
        Args:
            prediction_result: Prediction result from the system
            true_anomaly_label: True anomaly label (if available)
            capacitor_id: Capacitor identifier
        """
        prediction_record = {
            "timestamp": datetime.now(),
            "predicted_anomaly": prediction_result.anomaly_flag,
            "true_anomaly": true_anomaly_label,
            "anomaly_score": prediction_result.anomaly_score,
            "capacitor_id": capacitor_id,
            "rul_cycles": prediction_result.rul_cycles,
            "degradation_score": prediction_result.degradation_score
        }
        
        self.prediction_buffer.append(prediction_record)
        
        # Log prediction for structured logging
        self.prediction_logger.log_performance_metrics(
            metrics={
                "fpr_monitor_prediction_recorded": True,
                "anomaly_flag": prediction_result.anomaly_flag,
                "anomaly_score": prediction_result.anomaly_score,
                "capacitor_id": capacitor_id
            },
            context={"component": "fpr_monitor"}
        )
    
    def calculate_current_fpr(self, window_minutes: Optional[int] = None) -> Optional[FPRMetrics]:
        """
        Calculate current FPR metrics for the specified time window
        
        Args:
            window_minutes: Time window in minutes (uses default if None)
            
        Returns:
            FPR metrics or None if insufficient data
        """
        if window_minutes is None:
            window_minutes = self.monitoring_window_minutes
        
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        
        # Filter predictions within time window
        recent_predictions = [
            p for p in self.prediction_buffer
            if p["timestamp"] > cutoff_time and p["true_anomaly"] is not None
        ]
        
        if len(recent_predictions) < self.config.min_predictions_for_alert:
            return None
        
        # Extract predictions and true labels
        predicted_labels = [p["predicted_anomaly"] for p in recent_predictions]
        true_labels = [p["true_anomaly"] for p in recent_predictions]
        capacitor_ids = list(set(p["capacitor_id"] for p in recent_predictions))
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()
        
        # Calculate metrics
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics = FPRMetrics(
            timestamp=datetime.now(),
            fpr=fpr,
            tpr=tpr,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            total_predictions=len(recent_predictions),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn),
            true_positives=int(tp),
            window_duration_minutes=window_minutes,
            capacitor_ids=capacitor_ids
        )
        
        return metrics
    
    def check_fpr_thresholds(self, metrics: FPRMetrics) -> List[Alert]:
        """
        Check if FPR exceeds configured thresholds
        
        Args:
            metrics: Current FPR metrics
            
        Returns:
            List of alerts to trigger
        """
        alerts = []
        current_time = datetime.now()
        
        # Check critical FPR threshold
        if metrics.fpr >= self.config.fpr_threshold:
            alert_key = "fpr_critical"
            
            # Check cooldown
            if (alert_key not in self.alert_cooldowns or 
                current_time - self.alert_cooldowns[alert_key] > 
                timedelta(minutes=self.config.alert_cooldown_minutes)):
                
                alert = Alert(
                    alert_id=f"fpr_critical_{int(current_time.timestamp())}",
                    alert_type="fpr_threshold",
                    severity="critical",
                    message=(
                        f"CRITICAL: FPR exceeded threshold! "
                        f"Current FPR: {metrics.fpr:.3f} "
                        f"(Threshold: {self.config.fpr_threshold:.3f}). "
                        f"Based on {metrics.total_predictions} predictions "
                        f"in last {metrics.window_duration_minutes} minutes."
                    ),
                    timestamp=current_time,
                    metrics=metrics
                )
                alerts.append(alert)
                self.alert_cooldowns[alert_key] = current_time
        
        # Check warning FPR threshold
        elif metrics.fpr >= self.config.fpr_warning_threshold:
            alert_key = "fpr_warning"
            
            # Check cooldown
            if (alert_key not in self.alert_cooldowns or 
                current_time - self.alert_cooldowns[alert_key] > 
                timedelta(minutes=self.config.alert_cooldown_minutes)):
                
                alert = Alert(
                    alert_id=f"fpr_warning_{int(current_time.timestamp())}",
                    alert_type="fpr_warning",
                    severity="warning",
                    message=(
                        f"WARNING: FPR approaching threshold. "
                        f"Current FPR: {metrics.fpr:.3f} "
                        f"(Warning: {self.config.fpr_warning_threshold:.3f}, "
                        f"Critical: {self.config.fpr_threshold:.3f}). "
                        f"Based on {metrics.total_predictions} predictions "
                        f"in last {metrics.window_duration_minutes} minutes."
                    ),
                    timestamp=current_time,
                    metrics=metrics
                )
                alerts.append(alert)
                self.alert_cooldowns[alert_key] = current_time
        
        return alerts
    
    def detect_model_drift(self) -> List[Alert]:
        """
        Detect model drift by comparing recent FPR to historical baseline
        
        Returns:
            List of drift alerts
        """
        alerts = []
        current_time = datetime.now()
        
        # Get recent metrics for drift detection
        recent_metrics = self.db.get_recent_metrics(
            hours=self.config.drift_detection_window_hours
        )
        
        if len(recent_metrics) < 10:  # Need sufficient data for drift detection
            return alerts
        
        # Calculate baseline FPR (older half of the window)
        mid_point = len(recent_metrics) // 2
        baseline_fprs = [m.fpr for m in recent_metrics[mid_point:]]
        recent_fprs = [m.fpr for m in recent_metrics[:mid_point]]
        
        if len(baseline_fprs) < 5 or len(recent_fprs) < 5:
            return alerts
        
        baseline_mean = np.mean(baseline_fprs)
        recent_mean = np.mean(recent_fprs)
        drift_magnitude = recent_mean - baseline_mean
        
        # Check for significant drift
        if abs(drift_magnitude) >= self.config.drift_threshold:
            alert_key = "model_drift"
            
            # Check cooldown
            if (alert_key not in self.alert_cooldowns or 
                current_time - self.alert_cooldowns[alert_key] > 
                timedelta(minutes=self.config.alert_cooldown_minutes * 2)):  # Longer cooldown for drift
                
                drift_direction = "increased" if drift_magnitude > 0 else "decreased"
                severity = "critical" if abs(drift_magnitude) > self.config.drift_threshold * 2 else "warning"
                
                alert = Alert(
                    alert_id=f"model_drift_{int(current_time.timestamp())}",
                    alert_type="drift_detected",
                    severity=severity,
                    message=(
                        f"MODEL DRIFT DETECTED: FPR has {drift_direction} significantly. "
                        f"Baseline FPR: {baseline_mean:.3f}, "
                        f"Recent FPR: {recent_mean:.3f}, "
                        f"Drift: {drift_magnitude:+.3f}. "
                        f"Consider model retraining."
                    ),
                    timestamp=current_time,
                    metrics=recent_metrics[0] if recent_metrics else None
                )
                alerts.append(alert)
                self.alert_cooldowns[alert_key] = current_time
        
        return alerts
    
    def send_alert(self, alert: Alert) -> None:
        """
        Send alert through configured channels
        
        Args:
            alert: Alert to send
        """
        logger.warning(f"FPR Alert: {alert.message}")
        
        # Store alert in database
        metrics_id = None
        if alert.metrics:
            metrics_id = self.db.store_metrics(alert.metrics)
        self.db.store_alert(alert, metrics_id)
        
        # Send email alert if configured
        if self.config.enable_email_alerts and self.config.alert_recipients:
            self._send_email_alert(alert)
        
        # Send webhook alert if configured
        if self.config.enable_webhook_alerts and self.config.webhook_url:
            self._send_webhook_alert(alert)
        
        # Call custom callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        # Log alert
        self.prediction_logger.log_model_event(
            event_type="alert_triggered",
            event_data={
                "alert_id": alert.alert_id,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "fpr": alert.metrics.fpr if alert.metrics else None
            }
        )
    
    def _send_email_alert(self, alert: Alert) -> None:
        """Send email alert"""
        if not EMAIL_AVAILABLE:
            logger.warning("Email functionality not available - skipping email alert")
            return
            
        try:
            msg = MimeMultipart()
            msg['From'] = self.config.smtp_username
            msg['To'] = ', '.join(self.config.alert_recipients)
            msg['Subject'] = f"RUL System Alert: {alert.severity.upper()} - {alert.alert_type}"
            
            body = f"""
            RUL Prediction System Alert
            
            Alert Type: {alert.alert_type}
            Severity: {alert.severity.upper()}
            Timestamp: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            
            Message:
            {alert.message}
            
            """
            
            if alert.metrics:
                body += f"""
            Metrics:
            - FPR: {alert.metrics.fpr:.3f}
            - TPR: {alert.metrics.tpr:.3f}
            - Precision: {alert.metrics.precision:.3f}
            - Recall: {alert.metrics.recall:.3f}
            - F1 Score: {alert.metrics.f1_score:.3f}
            - Total Predictions: {alert.metrics.total_predictions}
            - Window: {alert.metrics.window_duration_minutes} minutes
            - Capacitors: {', '.join(alert.metrics.capacitor_ids)}
            """
            
            msg.attach(MimeText(body, 'plain'))
            
            if not EMAIL_AVAILABLE:
                logger.warning("Email functionality not available")
                return
            
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            server.starttls()
            server.login(self.config.smtp_username, self.config.smtp_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent for {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _send_webhook_alert(self, alert: Alert) -> None:
        """Send webhook alert"""
        try:
            import requests
            
            payload = {
                "alert_id": alert.alert_id,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "metrics": asdict(alert.metrics) if alert.metrics else None
            }
            
            response = requests.post(
                self.config.webhook_url,
                json=payload,
                headers=self.config.webhook_headers,
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(f"Webhook alert sent for {alert.alert_id}")
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def start_monitoring(self) -> None:
        """Start real-time FPR monitoring"""
        if self.is_monitoring:
            logger.warning("FPR monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("FPR monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop real-time FPR monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("FPR monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Calculate current FPR metrics
                metrics = self.calculate_current_fpr()
                
                if metrics:
                    # Store metrics in database
                    self.db.store_metrics(metrics)
                    
                    # Check for threshold violations
                    threshold_alerts = self.check_fpr_thresholds(metrics)
                    for alert in threshold_alerts:
                        self.send_alert(alert)
                    
                    # Check for model drift (less frequently)
                    if datetime.now().minute % 15 == 0:  # Every 15 minutes
                        drift_alerts = self.detect_model_drift()
                        for alert in drift_alerts:
                            self.send_alert(alert)
                
                # Sleep for monitoring interval
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)  # Continue monitoring despite errors
    
    def get_dashboard_data(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get data for FPR monitoring dashboard
        
        Args:
            hours: Number of hours of data to retrieve
            
        Returns:
            Dashboard data dictionary
        """
        recent_metrics = self.db.get_recent_metrics(hours)
        active_alerts = self.db.get_active_alerts()
        current_metrics = self.calculate_current_fpr()
        
        # Calculate trend
        if len(recent_metrics) >= 2:
            recent_fpr = recent_metrics[0].fpr
            older_fpr = recent_metrics[-1].fpr
            fpr_trend = recent_fpr - older_fpr
        else:
            fpr_trend = 0.0
        
        # Aggregate statistics
        if recent_metrics:
            fprs = [m.fpr for m in recent_metrics]
            avg_fpr = np.mean(fprs)
            max_fpr = np.max(fprs)
            min_fpr = np.min(fprs)
        else:
            avg_fpr = max_fpr = min_fpr = 0.0
        
        dashboard_data = {
            "current_metrics": asdict(current_metrics) if current_metrics else None,
            "recent_metrics": [asdict(m) for m in recent_metrics[:100]],  # Last 100 data points
            "active_alerts": [asdict(a) for a in active_alerts],
            "summary": {
                "current_fpr": current_metrics.fpr if current_metrics else None,
                "fpr_trend": fpr_trend,
                "avg_fpr_24h": avg_fpr,
                "max_fpr_24h": max_fpr,
                "min_fpr_24h": min_fpr,
                "total_predictions_24h": sum(m.total_predictions for m in recent_metrics),
                "active_alert_count": len(active_alerts),
                "critical_alert_count": len([a for a in active_alerts if a.severity == "critical"]),
                "fpr_threshold": self.config.fpr_threshold,
                "fpr_warning_threshold": self.config.fpr_warning_threshold,
                "monitoring_status": "active" if self.is_monitoring else "inactive"
            },
            "config": asdict(self.config),
            "timestamp": datetime.now().isoformat()
        }
        
        return dashboard_data
    
    def generate_fpr_report(
        self,
        hours: int = 24,
        save_path: Optional[Path] = None
    ) -> str:
        """
        Generate FPR trend analysis report
        
        Args:
            hours: Number of hours to analyze
            save_path: Optional path to save report
            
        Returns:
            Formatted report string
        """
        recent_metrics = self.db.get_recent_metrics(hours)
        active_alerts = self.db.get_active_alerts()
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("FPR MONITORING AND TREND ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Analysis Period: Last {hours} hours")
        report_lines.append("")
        
        # Current status
        current_metrics = self.calculate_current_fpr()
        if current_metrics:
            report_lines.append("CURRENT STATUS")
            report_lines.append("-" * 40)
            report_lines.append(f"Current FPR: {current_metrics.fpr:.3f}")
            report_lines.append(f"FPR Threshold: {self.config.fpr_threshold:.3f}")
            status = "✓ NORMAL" if current_metrics.fpr < self.config.fpr_threshold else "✗ THRESHOLD EXCEEDED"
            report_lines.append(f"Status: {status}")
            report_lines.append(f"Total Predictions (current window): {current_metrics.total_predictions}")
            report_lines.append("")
        
        # Active alerts
        if active_alerts:
            report_lines.append("ACTIVE ALERTS")
            report_lines.append("-" * 40)
            for alert in active_alerts[:10]:  # Show top 10 alerts
                report_lines.append(f"[{alert.severity.upper()}] {alert.alert_type}")
                report_lines.append(f"  Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                report_lines.append(f"  Message: {alert.message}")
                report_lines.append("")
        else:
            report_lines.append("ACTIVE ALERTS: None")
            report_lines.append("")
        
        # Trend analysis
        if len(recent_metrics) >= 2:
            report_lines.append("TREND ANALYSIS")
            report_lines.append("-" * 40)
            
            fprs = [m.fpr for m in reversed(recent_metrics)]  # Chronological order
            timestamps = [m.timestamp for m in reversed(recent_metrics)]
            
            # Calculate trend statistics
            avg_fpr = np.mean(fprs)
            std_fpr = np.std(fprs)
            max_fpr = np.max(fprs)
            min_fpr = np.min(fprs)
            
            # Linear trend
            x = np.arange(len(fprs))
            trend_slope = np.polyfit(x, fprs, 1)[0]
            
            report_lines.append(f"Average FPR: {avg_fpr:.3f} ± {std_fpr:.3f}")
            report_lines.append(f"FPR Range: {min_fpr:.3f} - {max_fpr:.3f}")
            trend_direction = "increasing" if trend_slope > 0 else "decreasing"
            report_lines.append(f"Trend: {trend_direction} ({trend_slope:+.6f}/hour)")
            
            # Threshold violations
            violations = sum(1 for fpr in fprs if fpr >= self.config.fpr_threshold)
            violation_rate = violations / len(fprs) * 100
            report_lines.append(f"Threshold Violations: {violations}/{len(fprs)} ({violation_rate:.1f}%)")
            report_lines.append("")
            
            # Recent performance
            if len(fprs) >= 10:
                recent_10 = fprs[-10:]
                older_10 = fprs[:10]
                recent_avg = np.mean(recent_10)
                older_avg = np.mean(older_10)
                performance_change = recent_avg - older_avg
                
                report_lines.append("RECENT PERFORMANCE")
                report_lines.append("-" * 40)
                report_lines.append(f"Recent Average FPR (last 10 measurements): {recent_avg:.3f}")
                report_lines.append(f"Earlier Average FPR (first 10 measurements): {older_avg:.3f}")
                change_direction = "deteriorated" if performance_change > 0 else "improved"
                report_lines.append(f"Performance Change: {change_direction} ({performance_change:+.3f})")
                report_lines.append("")
        
        # Configuration summary
        report_lines.append("MONITORING CONFIGURATION")
        report_lines.append("-" * 40)
        report_lines.append(f"FPR Threshold: {self.config.fpr_threshold:.3f}")
        report_lines.append(f"Warning Threshold: {self.config.fpr_warning_threshold:.3f}")
        report_lines.append(f"Monitoring Window: {self.monitoring_window_minutes} minutes")
        report_lines.append(f"Alert Cooldown: {self.config.alert_cooldown_minutes} minutes")
        report_lines.append(f"Drift Detection Window: {self.config.drift_detection_window_hours} hours")
        report_lines.append("")
        
        report_text = "\n".join(report_lines)
        
        # Save report if path provided
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                f.write(report_text)
            
            logger.info(f"FPR report saved to {save_path}")
        
        return report_text


# Convenience functions
def create_fpr_monitor(
    fpr_threshold: float = 0.05,
    warning_threshold: float = 0.03,
    monitoring_window_minutes: int = 15,
    enable_email_alerts: bool = False,
    alert_recipients: List[str] = None
) -> FPRMonitor:
    """
    Create and configure FPR monitor
    
    Args:
        fpr_threshold: Critical FPR threshold
        warning_threshold: Warning FPR threshold
        monitoring_window_minutes: Monitoring window size
        enable_email_alerts: Whether to enable email alerts
        alert_recipients: List of email recipients
        
    Returns:
        Configured FPR monitor
    """
    config = AlertConfig(
        fpr_threshold=fpr_threshold,
        fpr_warning_threshold=warning_threshold,
        enable_email_alerts=enable_email_alerts,
        alert_recipients=alert_recipients or []
    )
    
    return FPRMonitor(
        config=config,
        monitoring_window_minutes=monitoring_window_minutes
    )


def start_fpr_monitoring(
    monitor: FPRMonitor,
    auto_start: bool = True
) -> FPRMonitor:
    """
    Start FPR monitoring with logging
    
    Args:
        monitor: FPR monitor instance
        auto_start: Whether to automatically start monitoring
        
    Returns:
        Started FPR monitor
    """
    if auto_start:
        monitor.start_monitoring()
    
    logger.info("FPR monitoring system initialized and started")
    return monitor