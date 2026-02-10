"""
Monitoring Dashboard

This module provides a web-based dashboard for visualizing system health,
performance metrics, and alerts in real-time.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from .production_monitoring import ProductionMonitor

logger = logging.getLogger(__name__)


class MonitoringDashboard:
    """Web-based monitoring dashboard"""
    
    def __init__(self, monitor: ProductionMonitor, 
                 update_interval: float = 5.0):
        self.monitor = monitor
        self.update_interval = update_interval
        self._running = False
        self._update_task: Optional[asyncio.Task] = None
        self._dashboard_data: Dict[str, Any] = {}
        
    async def start(self):
        """Start the dashboard"""
        if self._running:
            return
            
        logger.info("Starting monitoring dashboard")
        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        
    async def stop(self):
        """Stop the dashboard"""
        if not self._running:
            return
            
        logger.info("Stopping monitoring dashboard")
        self._running = False
        
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
                
    async def _update_loop(self):
        """Update dashboard data periodically"""
        while self._running:
            try:
                self._dashboard_data = self.monitor.get_dashboard_data()
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dashboard update error: {e}")
                await asyncio.sleep(self.update_interval)
                
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data"""
        return self._dashboard_data.copy()
    
    def generate_html_dashboard(self) -> str:
        """Generate HTML dashboard"""
        data = self.get_dashboard_data()
        
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RUL Prediction System - Monitoring Dashboard</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }}
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .metric-title {{
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        .metric-value.good {{ color: #28a745; }}
        .metric-value.warning {{ color: #ffc107; }}
        .metric-value.error {{ color: #dc3545; }}
        .health-status {{
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }}
        .status-indicator {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 10px;
        }}
        .status-indicator.healthy {{ background-color: #28a745; }}
        .status-indicator.unhealthy {{ background-color: #dc3545; }}
        .alerts-section {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .alert-item {{
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 5px;
            border-left: 4px solid;
        }}
        .alert-item.info {{ 
            background-color: #d1ecf1; 
            border-left-color: #17a2b8; 
        }}
        .alert-item.warning {{ 
            background-color: #fff3cd; 
            border-left-color: #ffc107; 
        }}
        .alert-item.error {{ 
            background-color: #f8d7da; 
            border-left-color: #dc3545; 
        }}
        .alert-item.critical {{ 
            background-color: #f5c6cb; 
            border-left-color: #721c24; 
        }}
        .timestamp {{
            color: #666;
            font-size: 12px;
        }}
        .refresh-info {{
            text-align: center;
            color: #666;
            margin-top: 20px;
        }}
    </style>
    <script>
        // Auto-refresh every 30 seconds
        setTimeout(function() {{
            location.reload();
        }}, 30000);
    </script>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>RUL Prediction System</h1>
            <h2>Production Monitoring Dashboard</h2>
            <p>Last Updated: {timestamp}</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-title">System Health</div>
                <div class="health-status">
                    <div class="status-indicator {health_status_class}"></div>
                    <span>{health_status_text}</span>
                </div>
                {health_checks}
            </div>
            
            <div class="metric-card">
                <div class="metric-title">False Positive Rate</div>
                <div class="metric-value {fpr_class}">{fpr_rate:.3f}</div>
                <div class="timestamp">Threshold: {fpr_threshold}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Error Rate</div>
                <div class="metric-value {error_class}">{error_rate:.3f}</div>
                <div class="timestamp">Threshold: {error_threshold}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Throughput</div>
                <div class="metric-value good">{throughput:.1f}</div>
                <div class="timestamp">Predictions/second</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Latency (P95)</div>
                <div class="metric-value {latency_class}">{latency_p95:.1f}ms</div>
                <div class="timestamp">Threshold: {latency_threshold}ms</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-title">Active Alerts</div>
                <div class="metric-value {alerts_class}">{active_alerts_count}</div>
                <div class="timestamp">Total alerts</div>
            </div>
        </div>
        
        <div class="alerts-section">
            <div class="metric-title">Active Alerts</div>
            {alerts_html}
        </div>
        
        <div class="refresh-info">
            Dashboard auto-refreshes every 30 seconds
        </div>
    </div>
</body>
</html>
        """
        
        # Process data for template
        if not data:
            return html_template.format(
                timestamp="No data available",
                health_status_class="unhealthy",
                health_status_text="Unknown",
                health_checks="",
                fpr_rate=0.0,
                fpr_class="good",
                fpr_threshold=0.05,
                error_rate=0.0,
                error_class="good",
                error_threshold=0.01,
                throughput=0.0,
                latency_p95=0.0,
                latency_class="good",
                latency_threshold=1000,
                active_alerts_count=0,
                alerts_class="good",
                alerts_html="<p>No alerts</p>"
            )
        
        # System health
        system_health = data.get("system_health", {})
        overall_healthy = system_health.get("overall_healthy", False)
        health_checks = system_health.get("checks", {})
        
        health_status_class = "healthy" if overall_healthy else "unhealthy"
        health_status_text = "Healthy" if overall_healthy else "Issues Detected"
        
        health_checks_html = ""
        for check_name, is_healthy in health_checks.items():
            status_class = "healthy" if is_healthy else "unhealthy"
            status_text = "✓" if is_healthy else "✗"
            health_checks_html += f"""
                <div class="health-status">
                    <div class="status-indicator {status_class}"></div>
                    <span>{check_name}: {status_text}</span>
                </div>
            """
        
        # Performance metrics
        performance = data.get("performance", {})
        fpr_rate = performance.get("fpr_rate", 0.0)
        error_rate = performance.get("error_rate", 0.0)
        throughput = performance.get("throughput", 0.0)
        latency_percentiles = performance.get("latency_percentiles", {})
        latency_p95 = latency_percentiles.get("p95", 0.0)
        
        # Thresholds
        thresholds = data.get("thresholds", {})
        fpr_threshold = thresholds.get("fpr_threshold", 0.05)
        error_threshold = thresholds.get("error_rate_threshold", 0.01)
        latency_threshold = thresholds.get("prediction_latency_ms", 1000)
        
        # Determine metric classes
        fpr_class = "error" if fpr_rate > fpr_threshold else "good"
        error_class = "error" if error_rate > error_threshold else "good"
        latency_class = "warning" if latency_p95 > latency_threshold else "good"
        
        # Alerts
        alerts_data = data.get("alerts", {})
        active_alerts_count = alerts_data.get("active_count", 0)
        active_alerts = alerts_data.get("active_alerts", [])
        
        alerts_class = "error" if active_alerts_count > 0 else "good"
        
        alerts_html = ""
        if active_alerts:
            for alert in active_alerts:
                severity = alert.get("severity", "info")
                title = alert.get("title", "Unknown Alert")
                message = alert.get("message", "")
                timestamp = alert.get("timestamp", "")
                
                alerts_html += f"""
                    <div class="alert-item {severity}">
                        <strong>{title}</strong><br>
                        {message}<br>
                        <div class="timestamp">{timestamp}</div>
                    </div>
                """
        else:
            alerts_html = "<p>No active alerts</p>"
        
        return html_template.format(
            timestamp=data.get("timestamp", "Unknown"),
            health_status_class=health_status_class,
            health_status_text=health_status_text,
            health_checks=health_checks_html,
            fpr_rate=fpr_rate,
            fpr_class=fpr_class,
            fpr_threshold=fpr_threshold,
            error_rate=error_rate,
            error_class=error_class,
            error_threshold=error_threshold,
            throughput=throughput,
            latency_p95=latency_p95,
            latency_class=latency_class,
            latency_threshold=latency_threshold,
            active_alerts_count=active_alerts_count,
            alerts_class=alerts_class,
            alerts_html=alerts_html
        )
    
    def save_dashboard_html(self, file_path: Path):
        """Save dashboard as HTML file"""
        html_content = self.generate_html_dashboard()
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(html_content)
            
        logger.info(f"Dashboard saved to {file_path}")


class DashboardServer:
    """Simple HTTP server for the monitoring dashboard"""
    
    def __init__(self, dashboard: MonitoringDashboard, 
                 host: str = "localhost", port: int = 8080):
        self.dashboard = dashboard
        self.host = host
        self.port = port
        self._server = None
        
    async def start_server(self):
        """Start the dashboard HTTP server"""
        try:
            from aiohttp import web, web_runner
            
            app = web.Application()
            app.router.add_get('/', self._handle_dashboard)
            app.router.add_get('/api/data', self._handle_api_data)
            
            runner = web_runner.AppRunner(app)
            await runner.setup()
            
            site = web_runner.TCPSite(runner, self.host, self.port)
            await site.start()
            
            self._server = runner
            logger.info(f"Dashboard server started at http://{self.host}:{self.port}")
            
        except ImportError:
            logger.error("aiohttp not available, cannot start dashboard server")
            
    async def stop_server(self):
        """Stop the dashboard HTTP server"""
        if self._server:
            await self._server.cleanup()
            self._server = None
            logger.info("Dashboard server stopped")
            
    async def _handle_dashboard(self, request):
        """Handle dashboard page request"""
        try:
            from aiohttp import web
            html_content = self.dashboard.generate_html_dashboard()
            return web.Response(text=html_content, content_type='text/html')
        except ImportError:
            return None
            
    async def _handle_api_data(self, request):
        """Handle API data request"""
        try:
            from aiohttp import web
            data = self.dashboard.get_dashboard_data()
            return web.json_response(data)
        except ImportError:
            return None


def create_monitoring_dashboard(
    monitor: ProductionMonitor,
    update_interval: float = 5.0,
    enable_server: bool = False,
    server_host: str = "localhost",
    server_port: int = 8080
) -> MonitoringDashboard:
    """
    Create a monitoring dashboard
    
    Args:
        monitor: ProductionMonitor instance
        update_interval: Dashboard update interval in seconds
        enable_server: Whether to enable HTTP server
        server_host: Server host
        server_port: Server port
        
    Returns:
        Configured MonitoringDashboard
    """
    dashboard = MonitoringDashboard(monitor, update_interval)
    
    if enable_server:
        dashboard.server = DashboardServer(dashboard, server_host, server_port)
    
    return dashboard