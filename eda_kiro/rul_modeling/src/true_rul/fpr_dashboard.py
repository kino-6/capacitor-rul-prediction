"""
FPR Monitoring Dashboard

This module implements a web-based dashboard for real-time FPR monitoring,
providing visualizations, alerts, and trend analysis.

Requirements: 10.3, 5.5
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json
import asyncio
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from pydantic import BaseModel

from .fpr_monitor import FPRMonitor, AlertConfig, Alert
from .structured_logger import get_prediction_logger

logger = logging.getLogger(__name__)


class DashboardConfig(BaseModel):
    """Configuration for FPR dashboard"""
    host: str = "0.0.0.0"
    port: int = 8080
    title: str = "RUL System - FPR Monitoring Dashboard"
    refresh_interval_seconds: int = 30
    max_data_points: int = 1000
    enable_websocket: bool = True


class AlertAcknowledgeRequest(BaseModel):
    """Request to acknowledge an alert"""
    alert_id: str
    acknowledged_by: str = "dashboard_user"


class AlertResolveRequest(BaseModel):
    """Request to resolve an alert"""
    alert_id: str
    resolved_by: str = "dashboard_user"
    resolution_notes: str = ""


class FPRDashboard:
    """
    Web-based FPR monitoring dashboard
    
    Provides real-time visualization of FPR metrics, alerts, and trends
    through a web interface with WebSocket updates.
    """
    
    def __init__(
        self,
        fpr_monitor: FPRMonitor,
        config: DashboardConfig = None,
        template_dir: Optional[str] = None,
        static_dir: Optional[str] = None
    ):
        """
        Initialize FPR dashboard
        
        Args:
            fpr_monitor: FPR monitor instance
            config: Dashboard configuration
            template_dir: Directory containing HTML templates
            static_dir: Directory containing static files
        """
        self.fpr_monitor = fpr_monitor
        self.config = config or DashboardConfig()
        
        # Setup FastAPI app
        self.app = FastAPI(title=self.config.title)
        
        # Setup templates and static files
        if template_dir is None:
            template_dir = str(Path(__file__).parent / "templates")
        if static_dir is None:
            static_dir = str(Path(__file__).parent / "static")
        
        # Create directories if they don't exist
        Path(template_dir).mkdir(parents=True, exist_ok=True)
        Path(static_dir).mkdir(parents=True, exist_ok=True)
        
        self.templates = Jinja2Templates(directory=template_dir)
        
        # Mount static files
        try:
            self.app.mount("/static", StaticFiles(directory=static_dir), name="static")
        except Exception as e:
            logger.warning(f"Could not mount static files: {e}")
        
        # WebSocket connections
        self.websocket_connections = set()
        
        # Setup routes
        self._setup_routes()
        
        # Prediction logger
        self.prediction_logger = get_prediction_logger()
        
        logger.info("FPR Dashboard initialized")
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home(request: Request):
            """Main dashboard page"""
            return self.templates.TemplateResponse(
                "dashboard.html",
                {
                    "request": request,
                    "title": self.config.title,
                    "refresh_interval": self.config.refresh_interval_seconds * 1000,
                    "websocket_enabled": self.config.enable_websocket
                }
            )
        
        @self.app.get("/api/dashboard-data")
        async def get_dashboard_data(hours: int = 24):
            """Get dashboard data API endpoint"""
            try:
                data = self.fpr_monitor.get_dashboard_data(hours=hours)
                return JSONResponse(content=data)
            except Exception as e:
                logger.error(f"Error getting dashboard data: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/alerts")
        async def get_alerts():
            """Get active alerts"""
            try:
                alerts = self.fpr_monitor.db.get_active_alerts()
                return JSONResponse(content=[
                    {
                        "alert_id": alert.alert_id,
                        "alert_type": alert.alert_type,
                        "severity": alert.severity,
                        "message": alert.message,
                        "timestamp": alert.timestamp.isoformat(),
                        "acknowledged": alert.acknowledged,
                        "resolved": alert.resolved,
                        "metrics": {
                            "fpr": alert.metrics.fpr,
                            "total_predictions": alert.metrics.total_predictions,
                            "window_duration_minutes": alert.metrics.window_duration_minutes
                        } if alert.metrics else None
                    }
                    for alert in alerts
                ])
            except Exception as e:
                logger.error(f"Error getting alerts: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/alerts/{alert_id}/acknowledge")
        async def acknowledge_alert(alert_id: str, request: AlertAcknowledgeRequest):
            """Acknowledge an alert"""
            try:
                # Update alert in database
                with sqlite3.connect(self.fpr_monitor.db.db_path) as conn:
                    conn.execute(
                        "UPDATE alerts SET acknowledged = TRUE WHERE alert_id = ?",
                        (alert_id,)
                    )
                    conn.commit()
                
                # Log acknowledgment
                self.prediction_logger.log_model_event(
                    event_type="alert_acknowledged",
                    event_data={
                        "alert_id": alert_id,
                        "acknowledged_by": request.acknowledged_by
                    }
                )
                
                # Broadcast update via WebSocket
                await self._broadcast_alert_update(alert_id, "acknowledged")
                
                return JSONResponse(content={"status": "acknowledged"})
                
            except Exception as e:
                logger.error(f"Error acknowledging alert {alert_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/alerts/{alert_id}/resolve")
        async def resolve_alert(alert_id: str, request: AlertResolveRequest):
            """Resolve an alert"""
            try:
                # Update alert in database
                with sqlite3.connect(self.fpr_monitor.db.db_path) as conn:
                    conn.execute(
                        "UPDATE alerts SET resolved = TRUE WHERE alert_id = ?",
                        (alert_id,)
                    )
                    conn.commit()
                
                # Log resolution
                self.prediction_logger.log_model_event(
                    event_type="alert_resolved",
                    event_data={
                        "alert_id": alert_id,
                        "resolved_by": request.resolved_by,
                        "resolution_notes": request.resolution_notes
                    }
                )
                
                # Broadcast update via WebSocket
                await self._broadcast_alert_update(alert_id, "resolved")
                
                return JSONResponse(content={"status": "resolved"})
                
            except Exception as e:
                logger.error(f"Error resolving alert {alert_id}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/report")
        async def get_fpr_report(hours: int = 24):
            """Get FPR trend analysis report"""
            try:
                report = self.fpr_monitor.generate_fpr_report(hours=hours)
                return JSONResponse(content={"report": report, "hours": hours})
            except Exception as e:
                logger.error(f"Error generating FPR report: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint"""
            return JSONResponse(content={
                "status": "healthy",
                "monitoring_active": self.fpr_monitor.is_monitoring,
                "timestamp": datetime.now().isoformat()
            })
        
        if self.config.enable_websocket:
            @self.app.websocket("/ws")
            async def websocket_endpoint(websocket: WebSocket):
                """WebSocket endpoint for real-time updates"""
                await websocket.accept()
                self.websocket_connections.add(websocket)
                
                try:
                    while True:
                        # Send periodic updates
                        data = self.fpr_monitor.get_dashboard_data(hours=1)
                        await websocket.send_json({
                            "type": "dashboard_update",
                            "data": data
                        })
                        
                        # Wait for next update
                        await asyncio.sleep(self.config.refresh_interval_seconds)
                        
                except WebSocketDisconnect:
                    self.websocket_connections.discard(websocket)
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    self.websocket_connections.discard(websocket)
    
    async def _broadcast_alert_update(self, alert_id: str, action: str):
        """Broadcast alert update to all WebSocket connections"""
        if not self.config.enable_websocket:
            return
        
        message = {
            "type": "alert_update",
            "alert_id": alert_id,
            "action": action,
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to all connected clients
        disconnected = set()
        for websocket in self.websocket_connections:
            try:
                await websocket.send_json(message)
            except Exception:
                disconnected.add(websocket)
        
        # Remove disconnected clients
        self.websocket_connections -= disconnected
    
    def create_dashboard_template(self) -> str:
        """
        Create HTML template for the dashboard
        
        Returns:
            HTML template string
        """
        template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .metric-card {
            text-align: center;
        }
        .metric-value {
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }
        .metric-label {
            color: #666;
            font-size: 0.9em;
        }
        .status-good { color: #28a745; }
        .status-warning { color: #ffc107; }
        .status-critical { color: #dc3545; }
        .alert-item {
            border-left: 4px solid;
            padding: 10px;
            margin: 10px 0;
            background: #f8f9fa;
            border-radius: 0 5px 5px 0;
        }
        .alert-critical { border-left-color: #dc3545; }
        .alert-warning { border-left-color: #ffc107; }
        .alert-info { border-left-color: #17a2b8; }
        .chart-container {
            position: relative;
            height: 300px;
            margin: 20px 0;
        }
        .btn {
            padding: 8px 16px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            margin: 2px;
        }
        .btn-primary { background: #007bff; color: white; }
        .btn-success { background: #28a745; color: white; }
        .btn-warning { background: #ffc107; color: black; }
        .connection-status {
            position: fixed;
            top: 10px;
            right: 10px;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.8em;
            font-weight: bold;
        }
        .connected { background: #28a745; color: white; }
        .disconnected { background: #dc3545; color: white; }
        .loading {
            text-align: center;
            padding: 20px;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="connection-status" id="connectionStatus">Connecting...</div>
    
    <div class="header">
        <h1>{{ title }}</h1>
        <p>Real-time FPR Monitoring and Alerting System</p>
        <p id="lastUpdate">Last updated: Loading...</p>
    </div>
    
    <div class="dashboard-grid">
        <!-- Current FPR Status -->
        <div class="card metric-card">
            <h3>Current FPR</h3>
            <div class="metric-value" id="currentFPR">--</div>
            <div class="metric-label">False Positive Rate</div>
            <div id="fprStatus" class="metric-label">Loading...</div>
        </div>
        
        <!-- FPR Trend -->
        <div class="card metric-card">
            <h3>24h Trend</h3>
            <div class="metric-value" id="fprTrend">--</div>
            <div class="metric-label">FPR Change</div>
        </div>
        
        <!-- Active Alerts -->
        <div class="card metric-card">
            <h3>Active Alerts</h3>
            <div class="metric-value" id="activeAlerts">--</div>
            <div class="metric-label">Unresolved Alerts</div>
        </div>
        
        <!-- Total Predictions -->
        <div class="card metric-card">
            <h3>24h Predictions</h3>
            <div class="metric-value" id="totalPredictions">--</div>
            <div class="metric-label">Total Processed</div>
        </div>
    </div>
    
    <!-- FPR Trend Chart -->
    <div class="card">
        <h3>FPR Trend (Last 24 Hours)</h3>
        <div class="chart-container">
            <canvas id="fprChart"></canvas>
        </div>
    </div>
    
    <!-- Alerts Panel -->
    <div class="card">
        <h3>Active Alerts</h3>
        <div id="alertsList">
            <div class="loading">Loading alerts...</div>
        </div>
    </div>
    
    <!-- Configuration -->
    <div class="card">
        <h3>Monitoring Configuration</h3>
        <div id="configInfo">
            <div class="loading">Loading configuration...</div>
        </div>
    </div>

    <script>
        let chart = null;
        let websocket = null;
        
        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initializeChart();
            loadDashboardData();
            
            {% if websocket_enabled %}
            connectWebSocket();
            {% else %}
            // Fallback to polling
            setInterval(loadDashboardData, {{ refresh_interval }});
            {% endif %}
        });
        
        function initializeChart() {
            const ctx = document.getElementById('fprChart').getContext('2d');
            chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'FPR',
                        data: [],
                        borderColor: '#007bff',
                        backgroundColor: 'rgba(0, 123, 255, 0.1)',
                        tension: 0.4
                    }, {
                        label: 'Threshold',
                        data: [],
                        borderColor: '#dc3545',
                        borderDash: [5, 5],
                        fill: false
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 0.1
                        }
                    },
                    plugins: {
                        legend: {
                            display: true
                        }
                    }
                }
            });
        }
        
        function loadDashboardData() {
            fetch('/api/dashboard-data?hours=24')
                .then(response => response.json())
                .then(data => updateDashboard(data))
                .catch(error => {
                    console.error('Error loading dashboard data:', error);
                    updateConnectionStatus(false);
                });
        }
        
        function updateDashboard(data) {
            updateConnectionStatus(true);
            
            // Update metrics
            const summary = data.summary || {};
            
            document.getElementById('currentFPR').textContent = 
                summary.current_fpr !== null ? summary.current_fpr.toFixed(3) : '--';
            
            document.getElementById('fprTrend').textContent = 
                summary.fpr_trend !== null ? (summary.fpr_trend >= 0 ? '+' : '') + summary.fpr_trend.toFixed(3) : '--';
            
            document.getElementById('activeAlerts').textContent = 
                summary.active_alert_count || 0;
            
            document.getElementById('totalPredictions').textContent = 
                summary.total_predictions_24h || 0;
            
            // Update FPR status
            const fprStatus = document.getElementById('fprStatus');
            const currentFPRElement = document.getElementById('currentFPR');
            
            if (summary.current_fpr !== null) {
                if (summary.current_fpr >= summary.fpr_threshold) {
                    fprStatus.textContent = 'THRESHOLD EXCEEDED';
                    fprStatus.className = 'metric-label status-critical';
                    currentFPRElement.className = 'metric-value status-critical';
                } else if (summary.current_fpr >= summary.fpr_warning_threshold) {
                    fprStatus.textContent = 'WARNING LEVEL';
                    fprStatus.className = 'metric-label status-warning';
                    currentFPRElement.className = 'metric-value status-warning';
                } else {
                    fprStatus.textContent = 'NORMAL';
                    fprStatus.className = 'metric-label status-good';
                    currentFPRElement.className = 'metric-value status-good';
                }
            }
            
            // Update chart
            updateChart(data.recent_metrics || [], summary.fpr_threshold || 0.05);
            
            // Update alerts
            updateAlerts(data.active_alerts || []);
            
            // Update configuration
            updateConfiguration(data.config || {});
            
            // Update timestamp
            document.getElementById('lastUpdate').textContent = 
                'Last updated: ' + new Date().toLocaleString();
        }
        
        function updateChart(metrics, threshold) {
            if (!chart || !metrics.length) return;
            
            // Sort by timestamp
            metrics.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
            
            // Limit data points
            const maxPoints = 100;
            if (metrics.length > maxPoints) {
                metrics = metrics.slice(-maxPoints);
            }
            
            const labels = metrics.map(m => new Date(m.timestamp).toLocaleTimeString());
            const fprData = metrics.map(m => m.fpr);
            const thresholdData = new Array(labels.length).fill(threshold);
            
            chart.data.labels = labels;
            chart.data.datasets[0].data = fprData;
            chart.data.datasets[1].data = thresholdData;
            chart.update();
        }
        
        function updateAlerts(alerts) {
            const alertsList = document.getElementById('alertsList');
            
            if (!alerts.length) {
                alertsList.innerHTML = '<p style="color: #28a745;">No active alerts</p>';
                return;
            }
            
            alertsList.innerHTML = alerts.map(alert => `
                <div class="alert-item alert-${alert.severity}">
                    <div style="display: flex; justify-content: between; align-items: center;">
                        <div style="flex: 1;">
                            <strong>${alert.alert_type.toUpperCase()}</strong>
                            <span style="float: right; font-size: 0.8em; color: #666;">
                                ${new Date(alert.timestamp).toLocaleString()}
                            </span>
                        </div>
                    </div>
                    <div style="margin: 5px 0;">${alert.message}</div>
                    <div style="margin-top: 10px;">
                        ${!alert.acknowledged ? `
                            <button class="btn btn-warning" onclick="acknowledgeAlert('${alert.alert_id}')">
                                Acknowledge
                            </button>
                        ` : '<span style="color: #ffc107;">✓ Acknowledged</span>'}
                        ${!alert.resolved ? `
                            <button class="btn btn-success" onclick="resolveAlert('${alert.alert_id}')">
                                Resolve
                            </button>
                        ` : '<span style="color: #28a745;">✓ Resolved</span>'}
                    </div>
                </div>
            `).join('');
        }
        
        function updateConfiguration(config) {
            const configInfo = document.getElementById('configInfo');
            configInfo.innerHTML = `
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                    <div><strong>FPR Threshold:</strong> ${(config.fpr_threshold || 0.05).toFixed(3)}</div>
                    <div><strong>Warning Threshold:</strong> ${(config.fpr_warning_threshold || 0.03).toFixed(3)}</div>
                    <div><strong>Alert Cooldown:</strong> ${config.alert_cooldown_minutes || 30} min</div>
                    <div><strong>Drift Window:</strong> ${config.drift_detection_window_hours || 24} hours</div>
                </div>
            `;
        }
        
        function acknowledgeAlert(alertId) {
            fetch(`/api/alerts/${alertId}/acknowledge`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ acknowledged_by: 'dashboard_user' })
            })
            .then(response => response.json())
            .then(() => loadDashboardData())
            .catch(error => console.error('Error acknowledging alert:', error));
        }
        
        function resolveAlert(alertId) {
            fetch(`/api/alerts/${alertId}/resolve`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    resolved_by: 'dashboard_user',
                    resolution_notes: 'Resolved via dashboard'
                })
            })
            .then(response => response.json())
            .then(() => loadDashboardData())
            .catch(error => console.error('Error resolving alert:', error));
        }
        
        {% if websocket_enabled %}
        function connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;
            
            websocket = new WebSocket(wsUrl);
            
            websocket.onopen = function() {
                updateConnectionStatus(true);
            };
            
            websocket.onmessage = function(event) {
                const message = JSON.parse(event.data);
                
                if (message.type === 'dashboard_update') {
                    updateDashboard(message.data);
                } else if (message.type === 'alert_update') {
                    loadDashboardData(); // Refresh alerts
                }
            };
            
            websocket.onclose = function() {
                updateConnectionStatus(false);
                // Attempt to reconnect after 5 seconds
                setTimeout(connectWebSocket, 5000);
            };
            
            websocket.onerror = function(error) {
                console.error('WebSocket error:', error);
                updateConnectionStatus(false);
            };
        }
        {% endif %}
        
        function updateConnectionStatus(connected) {
            const status = document.getElementById('connectionStatus');
            if (connected) {
                status.textContent = 'Connected';
                status.className = 'connection-status connected';
            } else {
                status.textContent = 'Disconnected';
                status.className = 'connection-status disconnected';
            }
        }
    </script>
</body>
</html>
        """
        return template
    
    def save_template(self, template_dir: str):
        """Save dashboard template to file"""
        template_path = Path(template_dir) / "dashboard.html"
        template_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(template_path, 'w') as f:
            f.write(self.create_dashboard_template())
        
        logger.info(f"Dashboard template saved to {template_path}")
    
    def run(self, **kwargs):
        """Run the dashboard server"""
        # Save template if it doesn't exist
        template_path = Path(self.templates.directory) / "dashboard.html"
        if not template_path.exists():
            self.save_template(self.templates.directory)
        
        # Default uvicorn configuration
        config = {
            "host": self.config.host,
            "port": self.config.port,
            "log_level": "info"
        }
        config.update(kwargs)
        
        logger.info(f"Starting FPR Dashboard on {self.config.host}:{self.config.port}")
        uvicorn.run(self.app, **config)


# Convenience functions
def create_fpr_dashboard(
    fpr_monitor: FPRMonitor,
    host: str = "0.0.0.0",
    port: int = 8080,
    enable_websocket: bool = True
) -> FPRDashboard:
    """
    Create FPR monitoring dashboard
    
    Args:
        fpr_monitor: FPR monitor instance
        host: Dashboard host
        port: Dashboard port
        enable_websocket: Whether to enable WebSocket updates
        
    Returns:
        Configured FPR dashboard
    """
    config = DashboardConfig(
        host=host,
        port=port,
        enable_websocket=enable_websocket
    )
    
    return FPRDashboard(fpr_monitor=fpr_monitor, config=config)


def run_fpr_dashboard(
    fpr_monitor: FPRMonitor,
    host: str = "0.0.0.0",
    port: int = 8080,
    auto_start_monitoring: bool = True
) -> None:
    """
    Run FPR monitoring dashboard
    
    Args:
        fpr_monitor: FPR monitor instance
        host: Dashboard host
        port: Dashboard port
        auto_start_monitoring: Whether to auto-start FPR monitoring
    """
    if auto_start_monitoring and not fpr_monitor.is_monitoring:
        fpr_monitor.start_monitoring()
    
    dashboard = create_fpr_dashboard(fpr_monitor, host=host, port=port)
    dashboard.run()