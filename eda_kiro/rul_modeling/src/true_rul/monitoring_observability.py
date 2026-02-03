"""
Comprehensive Monitoring and Observability Module

This module implements comprehensive monitoring and observability features
including Prometheus metrics, distributed tracing with OpenTelemetry,
custom dashboards, and automated performance regression testing.

Requirements: 10.3, 10.5
"""

import logging
import time
import json
import threading
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime, timedelta
from functools import wraps
from collections import defaultdict, deque
import numpy as np

# Prometheus metrics
try:
    from prometheus_client import (
        Counter, Histogram, Gauge, Summary, Info,
        CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST,
        start_http_server
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logging.warning("Prometheus client not available. Install with: pip install prometheus-client")

# OpenTelemetry tracing
try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.logging import LoggingInstrumentor
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    logging.warning("OpenTelemetry not available. Install with: pip install opentelemetry-api opentelemetry-sdk")

# Local imports
from .data_structures import PredictionResult
from .exceptions import MonitoringError

logger = logging.getLogger(__name__)


class PrometheusMetrics:
    """
    Prometheus metrics collection for RUL prediction system
    
    Collects and exposes metrics for all system components including
    prediction latency, accuracy, cache hit rates, and system health.
    """
    
    def __init__(self, registry=None):
        """
        Initialize Prometheus metrics
        
        Args:
            registry: Custom registry, uses default if None
        """
        if not PROMETHEUS_AVAILABLE:
            raise ImportError("Prometheus client not available")
        
        from prometheus_client import CollectorRegistry
        
        self.registry = registry or CollectorRegistry()
        
        # Prediction metrics
        self.prediction_requests = Counter(
            'rul_prediction_requests_total',
            'Total number of prediction requests',
            ['model_type', 'status'],
            registry=self.registry
        )
        
        self.prediction_latency = Histogram(
            'rul_prediction_latency_seconds',
            'Prediction latency in seconds',
            ['model_type'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0],
            registry=self.registry
        )
        
        self.prediction_accuracy = Histogram(
            'rul_prediction_accuracy',
            'Prediction accuracy metrics',
            ['metric_type'],  # rmse, mae, r2
            registry=self.registry
        )
        
        # Model metrics
        self.model_load_time = Histogram(
            'rul_model_load_time_seconds',
            'Model loading time in seconds',
            ['model_type'],
            registry=self.registry
        )
        
        self.active_models = Gauge(
            'rul_active_models',
            'Number of active models in memory',
            ['model_type'],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            'rul_cache_hits_total',
            'Total cache hits',
            ['cache_type'],  # feature, prediction
            registry=self.registry
        )
        
        self.cache_misses = Counter(
            'rul_cache_misses_total',
            'Total cache misses',
            ['cache_type'],
            registry=self.registry
        )
        
        self.cache_size = Gauge(
            'rul_cache_size_bytes',
            'Cache size in bytes',
            ['cache_type'],
            registry=self.registry
        )
        
        # System metrics
        self.system_memory_usage = Gauge(
            'rul_system_memory_usage_bytes',
            'System memory usage in bytes',
            registry=self.registry
        )
        
        self.system_cpu_usage = Gauge(
            'rul_system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.batch_processing_time = Histogram(
            'rul_batch_processing_time_seconds',
            'Batch processing time in seconds',
            ['batch_size_range'],
            registry=self.registry
        )
        
        # Error metrics
        self.error_count = Counter(
            'rul_errors_total',
            'Total number of errors',
            ['error_type', 'component'],
            registry=self.registry
        )
        
        # Feature extraction metrics
        self.feature_extraction_time = Histogram(
            'rul_feature_extraction_time_seconds',
            'Feature extraction time in seconds',
            ['feature_type'],
            registry=self.registry
        )
        
        # API metrics
        self.api_requests = Counter(
            'rul_api_requests_total',
            'Total API requests',
            ['endpoint', 'method', 'status'],
            registry=self.registry
        )
        
        self.api_response_time = Histogram(
            'rul_api_response_time_seconds',
            'API response time in seconds',
            ['endpoint'],
            registry=self.registry
        )
        
        # Model performance metrics
        self.model_drift_score = Gauge(
            'rul_model_drift_score',
            'Model drift detection score',
            ['model_type'],
            registry=self.registry
        )
        
        self.false_positive_rate = Gauge(
            'rul_false_positive_rate',
            'Current false positive rate',
            ['model_type'],
            registry=self.registry
        )
        
        # System info
        self.system_info = Info(
            'rul_system_info',
            'System information',
            registry=self.registry
        )
        
        logger.info("Prometheus metrics initialized")
    
    def record_prediction_request(self, model_type: str, status: str) -> None:
        """Record a prediction request"""
        self.prediction_requests.labels(model_type=model_type, status=status).inc()
    
    def record_prediction_latency(self, model_type: str, latency: float) -> None:
        """Record prediction latency"""
        self.prediction_latency.labels(model_type=model_type).observe(latency)
    
    def record_prediction_accuracy(self, metric_type: str, value: float) -> None:
        """Record prediction accuracy metric"""
        self.prediction_accuracy.labels(metric_type=metric_type).observe(value)
    
    def record_cache_hit(self, cache_type: str) -> None:
        """Record cache hit"""
        self.cache_hits.labels(cache_type=cache_type).inc()
    
    def record_cache_miss(self, cache_type: str) -> None:
        """Record cache miss"""
        self.cache_misses.labels(cache_type=cache_type).inc()
    
    def update_cache_size(self, cache_type: str, size_bytes: int) -> None:
        """Update cache size"""
        self.cache_size.labels(cache_type=cache_type).set(size_bytes)
    
    def record_error(self, error_type: str, component: str) -> None:
        """Record an error"""
        self.error_count.labels(error_type=error_type, component=component).inc()
    
    def update_system_metrics(self, memory_bytes: int, cpu_percent: float) -> None:
        """Update system resource metrics"""
        self.system_memory_usage.set(memory_bytes)
        self.system_cpu_usage.set(cpu_percent)
    
    def update_model_drift_score(self, model_type: str, score: float) -> None:
        """Update model drift score"""
        self.model_drift_score.labels(model_type=model_type).set(score)
    
    def update_false_positive_rate(self, model_type: str, fpr: float) -> None:
        """Update false positive rate"""
        self.false_positive_rate.labels(model_type=model_type).set(fpr)
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        return generate_latest(self.registry)


class DistributedTracing:
    """
    Distributed tracing with OpenTelemetry
    
    Provides distributed tracing capabilities to track requests
    across different components of the RUL prediction system.
    """
    
    def __init__(
        self,
        service_name: str = "rul-prediction-system",
        jaeger_endpoint: Optional[str] = None,
        enable_auto_instrumentation: bool = True
    ):
        """
        Initialize distributed tracing
        
        Args:
            service_name: Name of the service
            jaeger_endpoint: Jaeger collector endpoint
            enable_auto_instrumentation: Enable automatic instrumentation
        """
        if not OPENTELEMETRY_AVAILABLE:
            raise ImportError("OpenTelemetry not available")
        
        self.service_name = service_name
        
        # Configure resource
        resource = Resource.create({
            "service.name": service_name,
            "service.version": "1.0.0"
        })
        
        # Configure tracer provider
        trace.set_tracer_provider(TracerProvider(resource=resource))
        self.tracer = trace.get_tracer(__name__)
        
        # Configure Jaeger exporter if endpoint provided
        if jaeger_endpoint:
            jaeger_exporter = JaegerExporter(
                agent_host_name="localhost",
                agent_port=14268,
                collector_endpoint=jaeger_endpoint
            )
            
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
        
        # Enable auto-instrumentation
        if enable_auto_instrumentation:
            RequestsInstrumentor().instrument()
            LoggingInstrumentor().instrument()
        
        logger.info(f"Distributed tracing initialized for service: {service_name}")
    
    def trace_prediction(self, func: Callable) -> Callable:
        """
        Decorator to trace prediction functions
        
        Args:
            func: Function to trace
        
        Returns:
            Traced function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.tracer.start_as_current_span(f"prediction.{func.__name__}") as span:
                # Add attributes
                span.set_attribute("function.name", func.__name__)
                span.set_attribute("function.module", func.__module__)
                
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("prediction.success", True)
                    
                    # Add result attributes if it's a PredictionResult
                    if isinstance(result, PredictionResult):
                        span.set_attribute("prediction.rul_cycles", result.rul_cycles)
                        span.set_attribute("prediction.anomaly_flag", result.anomaly_flag)
                        span.set_attribute("prediction.degradation_stage", result.degradation_stage)
                    
                    return result
                
                except Exception as e:
                    span.set_attribute("prediction.success", False)
                    span.set_attribute("error.type", type(e).__name__)
                    span.set_attribute("error.message", str(e))
                    span.record_exception(e)
                    raise
        
        return wrapper
    
    def trace_feature_extraction(self, func: Callable) -> Callable:
        """
        Decorator to trace feature extraction functions
        
        Args:
            func: Function to trace
        
        Returns:
            Traced function
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.tracer.start_as_current_span(f"feature_extraction.{func.__name__}") as span:
                span.set_attribute("function.name", func.__name__)
                
                try:
                    result = func(*args, **kwargs)
                    span.set_attribute("extraction.success", True)
                    
                    # Add feature information
                    if isinstance(result, np.ndarray):
                        span.set_attribute("features.shape", str(result.shape))
                        span.set_attribute("features.dtype", str(result.dtype))
                    
                    return result
                
                except Exception as e:
                    span.set_attribute("extraction.success", False)
                    span.record_exception(e)
                    raise
        
        return wrapper
    
    def create_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """
        Create a new span
        
        Args:
            name: Span name
            attributes: Optional attributes to add
        
        Returns:
            Span context manager
        """
        span = self.tracer.start_as_current_span(name)
        
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        
        return span


class PerformanceMonitor:
    """
    Performance monitoring and regression testing
    
    Monitors system performance over time and detects
    performance regressions automatically.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        regression_threshold: float = 0.2,  # 20% degradation
        min_samples: int = 10
    ):
        """
        Initialize performance monitor
        
        Args:
            window_size: Size of sliding window for metrics
            regression_threshold: Threshold for detecting regression
            min_samples: Minimum samples needed for regression detection
        """
        self.window_size = window_size
        self.regression_threshold = regression_threshold
        self.min_samples = min_samples
        
        # Metric storage
        self.metrics_history = defaultdict(lambda: deque(maxlen=window_size))
        self.baselines = {}
        self.alerts = []
        
        # Thread safety
        self.lock = threading.Lock()
        
        logger.info("Performance monitor initialized")
    
    def record_metric(
        self,
        metric_name: str,
        value: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Record a performance metric
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            timestamp: Timestamp, uses current time if None
        """
        timestamp = timestamp or datetime.now()
        
        with self.lock:
            self.metrics_history[metric_name].append({
                'value': value,
                'timestamp': timestamp
            })
            
            # Check for regression
            self._check_regression(metric_name)
    
    def set_baseline(self, metric_name: str, baseline_value: float) -> None:
        """
        Set baseline value for a metric
        
        Args:
            metric_name: Name of the metric
            baseline_value: Baseline value
        """
        with self.lock:
            self.baselines[metric_name] = baseline_value
        
        logger.info(f"Set baseline for {metric_name}: {baseline_value}")
    
    def _check_regression(self, metric_name: str) -> None:
        """
        Check for performance regression
        
        Args:
            metric_name: Name of the metric to check
        """
        history = self.metrics_history[metric_name]
        
        if len(history) < self.min_samples:
            return
        
        # Calculate recent average
        recent_values = [entry['value'] for entry in list(history)[-self.min_samples:]]
        recent_avg = np.mean(recent_values)
        
        # Compare with baseline
        baseline = self.baselines.get(metric_name)
        if baseline is None:
            # Use early values as baseline
            early_values = [entry['value'] for entry in list(history)[:self.min_samples]]
            baseline = np.mean(early_values)
            self.baselines[metric_name] = baseline
        
        # Check for regression (higher is worse for latency metrics)
        if metric_name.endswith('_latency') or metric_name.endswith('_time'):
            regression_ratio = (recent_avg - baseline) / baseline
        else:
            # For accuracy metrics, lower is worse
            regression_ratio = (baseline - recent_avg) / baseline
        
        if regression_ratio > self.regression_threshold:
            alert = {
                'metric_name': metric_name,
                'baseline': baseline,
                'recent_avg': recent_avg,
                'regression_ratio': regression_ratio,
                'timestamp': datetime.now(),
                'severity': 'high' if regression_ratio > 0.5 else 'medium'
            }
            
            self.alerts.append(alert)
            logger.warning(
                f"Performance regression detected for {metric_name}: "
                f"{regression_ratio:.2%} degradation"
            )
    
    def get_metric_summary(self, metric_name: str) -> Dict[str, Any]:
        """
        Get summary statistics for a metric
        
        Args:
            metric_name: Name of the metric
        
        Returns:
            Summary statistics
        """
        with self.lock:
            history = list(self.metrics_history[metric_name])
        
        if not history:
            return {}
        
        values = [entry['value'] for entry in history]
        
        return {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99),
            'baseline': self.baselines.get(metric_name),
            'latest': values[-1] if values else None
        }
    
    def get_active_alerts(self, max_age_hours: int = 24) -> List[Dict[str, Any]]:
        """
        Get active alerts
        
        Args:
            max_age_hours: Maximum age of alerts to return
        
        Returns:
            List of active alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        return [
            alert for alert in self.alerts
            if alert['timestamp'] > cutoff_time
        ]
    
    def clear_alerts(self) -> None:
        """Clear all alerts"""
        with self.lock:
            self.alerts.clear()
        
        logger.info("Performance alerts cleared")


class SystemHealthMonitor:
    """
    System health monitoring
    
    Monitors overall system health including resource usage,
    model availability, and service dependencies.
    """
    
    def __init__(self, check_interval: int = 60):
        """
        Initialize system health monitor
        
        Args:
            check_interval: Health check interval in seconds
        """
        self.check_interval = check_interval
        self.health_checks = {}
        self.health_status = {}
        self.is_monitoring = False
        self.monitor_thread = None
        
        logger.info("System health monitor initialized")
    
    def register_health_check(
        self,
        name: str,
        check_func: Callable[[], bool],
        critical: bool = False
    ) -> None:
        """
        Register a health check
        
        Args:
            name: Name of the health check
            check_func: Function that returns True if healthy
            critical: Whether this check is critical for system health
        """
        self.health_checks[name] = {
            'func': check_func,
            'critical': critical,
            'last_check': None,
            'last_result': None,
            'failure_count': 0
        }
        
        logger.info(f"Registered health check: {name} (critical: {critical})")
    
    def start_monitoring(self) -> None:
        """Start health monitoring"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring"""
        self.is_monitoring = False
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Health monitoring stopped")
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                self._run_health_checks()
                time.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                time.sleep(self.check_interval)
    
    def _run_health_checks(self) -> None:
        """Run all registered health checks"""
        overall_healthy = True
        
        for name, check_info in self.health_checks.items():
            try:
                result = check_info['func']()
                check_info['last_check'] = datetime.now()
                check_info['last_result'] = result
                
                if result:
                    check_info['failure_count'] = 0
                else:
                    check_info['failure_count'] += 1
                    if check_info['critical']:
                        overall_healthy = False
                    
                    logger.warning(
                        f"Health check failed: {name} "
                        f"(failures: {check_info['failure_count']})"
                    )
            
            except Exception as e:
                check_info['last_check'] = datetime.now()
                check_info['last_result'] = False
                check_info['failure_count'] += 1
                
                if check_info['critical']:
                    overall_healthy = False
                
                logger.error(f"Health check error for {name}: {e}")
        
        self.health_status = {
            'overall_healthy': overall_healthy,
            'timestamp': datetime.now(),
            'checks': {
                name: {
                    'healthy': info['last_result'],
                    'last_check': info['last_check'],
                    'failure_count': info['failure_count'],
                    'critical': info['critical']
                }
                for name, info in self.health_checks.items()
            }
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        return self.health_status.copy()
    
    def is_healthy(self) -> bool:
        """Check if system is healthy"""
        return self.health_status.get('overall_healthy', False)


class MonitoringDashboard:
    """
    Custom monitoring dashboard
    
    Provides a web-based dashboard for monitoring system metrics,
    health status, and performance trends.
    """
    
    def __init__(
        self,
        prometheus_metrics: PrometheusMetrics,
        performance_monitor: PerformanceMonitor,
        health_monitor: SystemHealthMonitor,
        port: int = 8080
    ):
        """
        Initialize monitoring dashboard
        
        Args:
            prometheus_metrics: Prometheus metrics instance
            performance_monitor: Performance monitor instance
            health_monitor: Health monitor instance
            port: Port for dashboard server
        """
        self.prometheus_metrics = prometheus_metrics
        self.performance_monitor = performance_monitor
        self.health_monitor = health_monitor
        self.port = port
        
        logger.info(f"Monitoring dashboard initialized on port {port}")
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """
        Generate dashboard data
        
        Returns:
            Dashboard data dictionary
        """
        # Get health status
        health_status = self.health_monitor.get_health_status()
        
        # Get performance alerts
        alerts = self.performance_monitor.get_active_alerts()
        
        # Get key metrics summaries
        key_metrics = [
            'prediction_latency',
            'prediction_accuracy',
            'cache_hit_rate',
            'error_rate'
        ]
        
        metrics_summaries = {}
        for metric in key_metrics:
            metrics_summaries[metric] = self.performance_monitor.get_metric_summary(metric)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'health_status': health_status,
            'alerts': alerts,
            'metrics_summaries': metrics_summaries,
            'system_info': {
                'prometheus_available': PROMETHEUS_AVAILABLE,
                'opentelemetry_available': OPENTELEMETRY_AVAILABLE
            }
        }
    
    def start_dashboard_server(self) -> None:
        """Start dashboard HTTP server"""
        if PROMETHEUS_AVAILABLE:
            # Start Prometheus metrics server
            start_http_server(self.port, registry=self.prometheus_metrics.registry)
            logger.info(f"Dashboard server started on port {self.port}")
        else:
            logger.warning("Prometheus not available, dashboard server not started")


class ObservabilityManager:
    """
    Unified observability management
    
    Coordinates all monitoring and observability components
    into a single management interface.
    """
    
    def __init__(
        self,
        service_name: str = "rul-prediction-system",
        enable_prometheus: bool = True,
        enable_tracing: bool = True,
        enable_performance_monitoring: bool = True,
        enable_health_monitoring: bool = True,
        prometheus_port: int = 8080,
        jaeger_endpoint: Optional[str] = None
    ):
        """
        Initialize observability manager
        
        Args:
            service_name: Name of the service
            enable_prometheus: Enable Prometheus metrics
            enable_tracing: Enable distributed tracing
            enable_performance_monitoring: Enable performance monitoring
            enable_health_monitoring: Enable health monitoring
            prometheus_port: Port for Prometheus metrics server
            jaeger_endpoint: Jaeger collector endpoint
        """
        self.service_name = service_name
        self.components = {}
        
        # Initialize Prometheus metrics
        if enable_prometheus and PROMETHEUS_AVAILABLE:
            self.prometheus_metrics = PrometheusMetrics()
            self.components['prometheus'] = self.prometheus_metrics
            
            # Start metrics server
            start_http_server(prometheus_port, registry=self.prometheus_metrics.registry)
            logger.info(f"Prometheus metrics server started on port {prometheus_port}")
        
        # Initialize distributed tracing
        if enable_tracing and OPENTELEMETRY_AVAILABLE:
            self.distributed_tracing = DistributedTracing(
                service_name=service_name,
                jaeger_endpoint=jaeger_endpoint
            )
            self.components['tracing'] = self.distributed_tracing
        
        # Initialize performance monitoring
        if enable_performance_monitoring:
            self.performance_monitor = PerformanceMonitor()
            self.components['performance'] = self.performance_monitor
        
        # Initialize health monitoring
        if enable_health_monitoring:
            self.health_monitor = SystemHealthMonitor()
            self.components['health'] = self.health_monitor
            
            # Register default health checks
            self._register_default_health_checks()
        
        # Initialize dashboard
        if all(comp in self.components for comp in ['prometheus', 'performance', 'health']):
            self.dashboard = MonitoringDashboard(
                self.prometheus_metrics,
                self.performance_monitor,
                self.health_monitor
            )
            self.components['dashboard'] = self.dashboard
        
        logger.info(f"Observability manager initialized with components: {list(self.components.keys())}")
    
    def _register_default_health_checks(self) -> None:
        """Register default health checks"""
        def memory_check() -> bool:
            """Check if memory usage is reasonable"""
            try:
                import psutil
                memory_percent = psutil.virtual_memory().percent
                return memory_percent < 90  # Less than 90% memory usage
            except ImportError:
                return True  # Skip check if psutil not available
        
        def disk_check() -> bool:
            """Check if disk space is available"""
            try:
                import psutil
                disk_percent = psutil.disk_usage('/').percent
                return disk_percent < 95  # Less than 95% disk usage
            except ImportError:
                return True
        
        self.health_monitor.register_health_check('memory', memory_check, critical=True)
        self.health_monitor.register_health_check('disk', disk_check, critical=True)
    
    def start_monitoring(self) -> None:
        """Start all monitoring components"""
        if 'health' in self.components:
            self.health_monitor.start_monitoring()
        
        logger.info("All monitoring components started")
    
    def stop_monitoring(self) -> None:
        """Stop all monitoring components"""
        if 'health' in self.components:
            self.health_monitor.stop_monitoring()
        
        logger.info("All monitoring components stopped")
    
    def record_prediction_metrics(
        self,
        model_type: str,
        latency: float,
        success: bool,
        accuracy_metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Record prediction metrics across all monitoring systems
        
        Args:
            model_type: Type of model used
            latency: Prediction latency in seconds
            success: Whether prediction was successful
            accuracy_metrics: Optional accuracy metrics
        """
        status = 'success' if success else 'error'
        
        # Prometheus metrics
        if 'prometheus' in self.components:
            self.prometheus_metrics.record_prediction_request(model_type, status)
            self.prometheus_metrics.record_prediction_latency(model_type, latency)
            
            if accuracy_metrics:
                for metric_name, value in accuracy_metrics.items():
                    self.prometheus_metrics.record_prediction_accuracy(metric_name, value)
        
        # Performance monitoring
        if 'performance' in self.components:
            self.performance_monitor.record_metric(f'{model_type}_latency', latency)
            
            if accuracy_metrics:
                for metric_name, value in accuracy_metrics.items():
                    self.performance_monitor.record_metric(f'{model_type}_{metric_name}', value)
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status
        
        Returns:
            System status dictionary
        """
        status = {
            'service_name': self.service_name,
            'timestamp': datetime.now().isoformat(),
            'components': list(self.components.keys())
        }
        
        # Health status
        if 'health' in self.components:
            status['health'] = self.health_monitor.get_health_status()
        
        # Performance alerts
        if 'performance' in self.components:
            status['alerts'] = self.performance_monitor.get_active_alerts()
        
        return status
    
    def create_monitoring_decorators(self) -> Dict[str, Callable]:
        """
        Create monitoring decorators for easy integration
        
        Returns:
            Dictionary of monitoring decorators
        """
        decorators = {}
        
        # Tracing decorators
        if 'tracing' in self.components:
            decorators['trace_prediction'] = self.distributed_tracing.trace_prediction
            decorators['trace_feature_extraction'] = self.distributed_tracing.trace_feature_extraction
        
        return decorators


# Global observability manager instance
_global_observability_manager = None


def get_observability_manager(**kwargs) -> ObservabilityManager:
    """Get global observability manager instance"""
    global _global_observability_manager
    if _global_observability_manager is None:
        _global_observability_manager = ObservabilityManager(**kwargs)
    return _global_observability_manager


def monitor_prediction(func: Callable) -> Callable:
    """
    Decorator to automatically monitor prediction functions
    
    Args:
        func: Function to monitor
    
    Returns:
        Monitored function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        success = False
        
        try:
            result = func(*args, **kwargs)
            success = True
            return result
        
        except Exception as e:
            # Record error
            obs_manager = get_observability_manager()
            if 'prometheus' in obs_manager.components:
                obs_manager.prometheus_metrics.record_error(
                    error_type=type(e).__name__,
                    component=func.__module__
                )
            raise
        
        finally:
            # Record metrics
            latency = time.time() - start_time
            obs_manager = get_observability_manager()
            
            # Determine model type from function name or args
            model_type = getattr(func, 'model_type', 'unknown')
            
            obs_manager.record_prediction_metrics(
                model_type=model_type,
                latency=latency,
                success=success
            )
    
    return wrapper