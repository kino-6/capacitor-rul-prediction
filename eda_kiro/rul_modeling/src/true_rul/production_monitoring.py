"""
Production Monitoring and Alerting System

This module provides comprehensive system health monitoring, performance tracking,
and automated alerting for the RUL prediction system in production environments.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
import threading
import statistics

import numpy as np

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Alert:
    """Alert message"""
    id: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "severity": self.severity.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "metadata": self.metadata,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None
        }


@dataclass
class Metric:
    """System metric"""
    name: str
    value: float
    timestamp: datetime
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp.isoformat(),
            "type": self.metric_type.value,
            "labels": self.labels
        }


@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""
    prediction_latency_ms: List[float] = field(default_factory=list)
    throughput_predictions_per_second: float = 0.0
    error_rate: float = 0.0
    fpr_rate: float = 0.0
    data_quality_score: float = 1.0
    model_accuracy: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
    def get_latency_percentiles(self) -> Dict[str, float]:
        """Get latency percentiles"""
        if not self.prediction_latency_ms:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        
        sorted_latencies = sorted(self.prediction_latency_ms)
        n = len(sorted_latencies)
        
        return {
            "p50": sorted_latencies[int(n * 0.5)],
            "p95": sorted_latencies[int(n * 0.95)],
            "p99": sorted_latencies[int(n * 0.99)]
        }


class AlertHandler(ABC):
    """Abstract base class for alert handlers"""
    
    @abstractmethod
    async def send_alert(self, alert: Alert):
        """Send an alert"""
        pass


class LogAlertHandler(AlertHandler):
    """Log-based alert handler"""
    
    def __init__(self, logger_name: str = "alerts"):
        self.logger = logging.getLogger(logger_name)
        
    async def send_alert(self, alert: Alert):
        """Send alert to logs"""
        level_map = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }
        
        level = level_map.get(alert.severity, logging.INFO)
        self.logger.log(level, f"ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.message}")


class FileAlertHandler(AlertHandler):
    """File-based alert handler"""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
    async def send_alert(self, alert: Alert):
        """Send alert to file"""
        with open(self.file_path, 'a') as f:
            f.write(json.dumps(alert.to_dict()) + '\n')


class WebhookAlertHandler(AlertHandler):
    """Webhook-based alert handler"""
    
    def __init__(self, webhook_url: str, timeout: float = 10.0):
        self.webhook_url = webhook_url
        self.timeout = timeout
        
    async def send_alert(self, alert: Alert):
        """Send alert to webhook"""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=alert.to_dict(),
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    if response.status >= 400:
                        logger.error(f"Webhook alert failed: {response.status}")
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")


class MetricsCollector:
    """Collects and stores system metrics"""
    
    def __init__(self, max_metrics: int = 10000):
        self.max_metrics = max_metrics
        self._metrics: deque = deque(maxlen=max_metrics)
        self._lock = threading.RLock()
        
    def record_metric(self, name: str, value: float, metric_type: MetricType, 
                     labels: Optional[Dict[str, str]] = None):
        """Record a metric"""
        metric = Metric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            metric_type=metric_type,
            labels=labels or {}
        )
        
        with self._lock:
            self._metrics.append(metric)
            
    def get_metrics(self, name: Optional[str] = None, 
                   since: Optional[datetime] = None) -> List[Metric]:
        """Get metrics with optional filtering"""
        with self._lock:
            metrics = list(self._metrics)
            
        if name:
            metrics = [m for m in metrics if m.name == name]
            
        if since:
            metrics = [m for m in metrics if m.timestamp >= since]
            
        return metrics
    
    def get_metric_summary(self, name: str, since: Optional[datetime] = None) -> Dict[str, float]:
        """Get summary statistics for a metric"""
        metrics = self.get_metrics(name, since)
        
        if not metrics:
            return {}
            
        values = [m.value for m in metrics]
        
        return {
            "count": len(values),
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0
        }


class PerformanceTracker:
    """Tracks system performance metrics"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._prediction_times: deque = deque(maxlen=window_size)
        self._error_count = 0
        self._total_predictions = 0
        self._fpr_count = 0
        self._total_anomaly_predictions = 0
        self._lock = threading.RLock()
        
    def record_prediction(self, latency_ms: float, had_error: bool = False):
        """Record a prediction"""
        with self._lock:
            self._prediction_times.append(latency_ms)
            self._total_predictions += 1
            
            if had_error:
                self._error_count += 1
                
    def record_anomaly_prediction(self, is_false_positive: bool = False):
        """Record an anomaly prediction"""
        with self._lock:
            self._total_anomaly_predictions += 1
            
            if is_false_positive:
                self._fpr_count += 1
                
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        with self._lock:
            latencies = list(self._prediction_times)
            
            # Calculate throughput (predictions per second over last minute)
            now = time.time()
            recent_count = len([t for t in self._prediction_times if now - t/1000 <= 60])
            throughput = recent_count / 60.0
            
            # Calculate error rate
            error_rate = self._error_count / max(1, self._total_predictions)
            
            # Calculate FPR
            fpr_rate = self._fpr_count / max(1, self._total_anomaly_predictions)
            
            return PerformanceMetrics(
                prediction_latency_ms=latencies,
                throughput_predictions_per_second=throughput,
                error_rate=error_rate,
                fpr_rate=fpr_rate
            )
    
    def reset(self):
        """Reset all counters"""
        with self._lock:
            self._prediction_times.clear()
            self._error_count = 0
            self._total_predictions = 0
            self._fpr_count = 0
            self._total_anomaly_predictions = 0


class HealthChecker:
    """Performs system health checks"""
    
    def __init__(self):
        self._checks: Dict[str, Callable[[], bool]] = {}
        
    def register_check(self, name: str, check_func: Callable[[], bool]):
        """Register a health check"""
        self._checks[name] = check_func
        
    def run_checks(self) -> Dict[str, bool]:
        """Run all health checks"""
        results = {}
        
        for name, check_func in self._checks.items():
            try:
                results[name] = check_func()
            except Exception as e:
                logger.error(f"Health check '{name}' failed: {e}")
                results[name] = False
                
        return results
    
    def is_healthy(self) -> bool:
        """Check if system is healthy"""
        results = self.run_checks()
        return all(results.values())


class AlertManager:
    """Manages alerts and notifications"""
    
    def __init__(self):
        self._handlers: List[AlertHandler] = []
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: deque = deque(maxlen=1000)
        self._lock = threading.RLock()
        
    def add_handler(self, handler: AlertHandler):
        """Add an alert handler"""
        self._handlers.append(handler)
        
    async def send_alert(self, alert_id: str, severity: AlertSeverity, 
                        title: str, message: str, source: str,
                        metadata: Optional[Dict[str, Any]] = None):
        """Send an alert"""
        alert = Alert(
            id=alert_id,
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.now(),
            source=source,
            metadata=metadata or {}
        )
        
        with self._lock:
            self._active_alerts[alert_id] = alert
            self._alert_history.append(alert)
            
        # Send to all handlers
        for handler in self._handlers:
            try:
                await handler.send_alert(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
                
    async def resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        with self._lock:
            if alert_id in self._active_alerts:
                alert = self._active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.now()
                del self._active_alerts[alert_id]
                
                # Send resolution notification
                await self.send_alert(
                    f"{alert_id}_resolved",
                    AlertSeverity.INFO,
                    f"Alert Resolved: {alert.title}",
                    f"Alert {alert_id} has been resolved",
                    "alert_manager"
                )
                
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        with self._lock:
            return list(self._active_alerts.values())
    
    def get_alert_history(self, limit: Optional[int] = None) -> List[Alert]:
        """Get alert history"""
        with self._lock:
            history = list(self._alert_history)
            
        if limit:
            history = history[-limit:]
            
        return history


class ProductionMonitor:
    """Main production monitoring system"""
    
    def __init__(self, 
                 alert_thresholds: Optional[Dict[str, float]] = None,
                 check_interval: float = 30.0):
        
        self.alert_thresholds = alert_thresholds or {
            "fpr_threshold": 0.05,
            "prediction_latency_ms": 1000,
            "data_quality_threshold": 0.95,
            "error_rate_threshold": 0.01,
            "memory_usage_mb": 1000,
            "cpu_usage_percent": 80.0
        }
        
        self.check_interval = check_interval
        
        self.metrics_collector = MetricsCollector()
        self.performance_tracker = PerformanceTracker()
        self.health_checker = HealthChecker()
        self.alert_manager = AlertManager()
        
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # Register default health checks
        self._register_default_health_checks()
        
    def _register_default_health_checks(self):
        """Register default health checks"""
        
        def check_memory_usage():
            """Check memory usage"""
            try:
                import psutil
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                threshold = self.alert_thresholds.get("memory_usage_mb", 1000)
                return memory_mb < threshold
            except ImportError:
                return True  # Skip if psutil not available
                
        def check_disk_space():
            """Check disk space"""
            try:
                import psutil
                disk_usage = psutil.disk_usage('/')
                free_percent = (disk_usage.free / disk_usage.total) * 100
                return free_percent > 10.0  # At least 10% free
            except ImportError:
                return True
                
        self.health_checker.register_check("memory_usage", check_memory_usage)
        self.health_checker.register_check("disk_space", check_disk_space)
        
    async def start_monitoring(self):
        """Start the monitoring system"""
        if self._running:
            return
            
        logger.info("Starting production monitoring system")
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        
    async def stop_monitoring(self):
        """Stop the monitoring system"""
        if not self._running:
            return
            
        logger.info("Stopping production monitoring system")
        self._running = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
                
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._running:
            try:
                await self._check_system_health()
                await self._check_performance_metrics()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(self.check_interval)
                
    async def _check_system_health(self):
        """Check system health and send alerts if needed"""
        health_results = self.health_checker.run_checks()
        
        for check_name, is_healthy in health_results.items():
            if not is_healthy:
                await self.alert_manager.send_alert(
                    f"health_check_{check_name}",
                    AlertSeverity.ERROR,
                    f"Health Check Failed: {check_name}",
                    f"System health check '{check_name}' is failing",
                    "health_checker"
                )
            else:
                # Resolve alert if it was previously active
                await self.alert_manager.resolve_alert(f"health_check_{check_name}")
                
    async def _check_performance_metrics(self):
        """Check performance metrics and send alerts if needed"""
        perf_metrics = self.performance_tracker.get_performance_metrics()
        
        # Check FPR
        if perf_metrics.fpr_rate > self.alert_thresholds["fpr_threshold"]:
            await self.alert_manager.send_alert(
                "high_fpr",
                AlertSeverity.WARNING,
                "High False Positive Rate",
                f"FPR is {perf_metrics.fpr_rate:.3f}, above threshold {self.alert_thresholds['fpr_threshold']}",
                "performance_tracker",
                {"fpr_rate": perf_metrics.fpr_rate}
            )
        else:
            await self.alert_manager.resolve_alert("high_fpr")
            
        # Check prediction latency
        latency_percentiles = perf_metrics.get_latency_percentiles()
        if latency_percentiles["p95"] > self.alert_thresholds["prediction_latency_ms"]:
            await self.alert_manager.send_alert(
                "high_latency",
                AlertSeverity.WARNING,
                "High Prediction Latency",
                f"P95 latency is {latency_percentiles['p95']:.1f}ms, above threshold {self.alert_thresholds['prediction_latency_ms']}ms",
                "performance_tracker",
                {"p95_latency_ms": latency_percentiles["p95"]}
            )
        else:
            await self.alert_manager.resolve_alert("high_latency")
            
        # Check error rate
        if perf_metrics.error_rate > self.alert_thresholds["error_rate_threshold"]:
            await self.alert_manager.send_alert(
                "high_error_rate",
                AlertSeverity.ERROR,
                "High Error Rate",
                f"Error rate is {perf_metrics.error_rate:.3f}, above threshold {self.alert_thresholds['error_rate_threshold']}",
                "performance_tracker",
                {"error_rate": perf_metrics.error_rate}
            )
        else:
            await self.alert_manager.resolve_alert("high_error_rate")
            
        # Record metrics
        self.metrics_collector.record_metric("fpr_rate", perf_metrics.fpr_rate, MetricType.GAUGE)
        self.metrics_collector.record_metric("error_rate", perf_metrics.error_rate, MetricType.GAUGE)
        self.metrics_collector.record_metric("throughput", perf_metrics.throughput_predictions_per_second, MetricType.GAUGE)
        
        if latency_percentiles["p95"] > 0:
            self.metrics_collector.record_metric("latency_p95", latency_percentiles["p95"], MetricType.GAUGE)
            
    def record_prediction(self, latency_ms: float, had_error: bool = False):
        """Record a prediction for monitoring"""
        self.performance_tracker.record_prediction(latency_ms, had_error)
        
    def record_anomaly_prediction(self, is_false_positive: bool = False):
        """Record an anomaly prediction for monitoring"""
        self.performance_tracker.record_anomaly_prediction(is_false_positive)
        
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard"""
        perf_metrics = self.performance_tracker.get_performance_metrics()
        health_results = self.health_checker.run_checks()
        active_alerts = self.alert_manager.get_active_alerts()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system_health": {
                "overall_healthy": all(health_results.values()),
                "checks": health_results
            },
            "performance": {
                "fpr_rate": perf_metrics.fpr_rate,
                "error_rate": perf_metrics.error_rate,
                "throughput": perf_metrics.throughput_predictions_per_second,
                "latency_percentiles": perf_metrics.get_latency_percentiles()
            },
            "alerts": {
                "active_count": len(active_alerts),
                "active_alerts": [alert.to_dict() for alert in active_alerts]
            },
            "thresholds": self.alert_thresholds
        }


def create_production_monitor(
    alert_handlers: Optional[List[AlertHandler]] = None,
    alert_thresholds: Optional[Dict[str, float]] = None,
    check_interval: float = 30.0
) -> ProductionMonitor:
    """
    Create a production monitor with default configuration
    
    Args:
        alert_handlers: List of alert handlers (uses log handler if None)
        alert_thresholds: Alert thresholds (uses defaults if None)
        check_interval: Monitoring check interval in seconds
        
    Returns:
        Configured ProductionMonitor
    """
    monitor = ProductionMonitor(
        alert_thresholds=alert_thresholds,
        check_interval=check_interval
    )
    
    # Add alert handlers
    if alert_handlers is None:
        alert_handlers = [LogAlertHandler()]
        
    for handler in alert_handlers:
        monitor.alert_manager.add_handler(handler)
        
    return monitor