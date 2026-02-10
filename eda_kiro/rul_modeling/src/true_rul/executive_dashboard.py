"""
Executive Dashboard Generator

This module creates interactive executive dashboards with real-time KPI tracking,
trend visualization, and business intelligence insights.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path

from .advanced_analytics import AdvancedAnalytics, KPIMetrics, TrendAnalysis, CostAnalysis, EquipmentComparison

logger = logging.getLogger(__name__)

class ExecutiveDashboard:
    """Interactive executive dashboard for predictive maintenance analytics"""
    
    def __init__(self, analytics_engine: AdvancedAnalytics):
        """
        Initialize executive dashboard
        
        Args:
            analytics_engine: AdvancedAnalytics instance for data processing
        """
        self.analytics = analytics_engine
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("Executive Dashboard", className="text-center mb-4"),
                    html.H4("Predictive Maintenance Analytics", className="text-center text-muted mb-4"),
                    html.Hr()
                ])
            ]),
            
            # KPI Cards Row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Equipment Monitored", className="card-title"),
                            html.H2(id="kpi-equipment-count", className="text-primary"),
                            html.P("Total active equipment", className="card-text")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("At Risk", className="card-title"),
                            html.H2(id="kpi-at-risk", className="text-warning"),
                            html.P("Equipment requiring attention", className="card-text")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("Cost Savings", className="card-title"),
                            html.H2(id="kpi-cost-savings", className="text-success"),
                            html.P("Annual savings achieved", className="card-text")
                        ])
                    ])
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("ROI", className="card-title"),
                            html.H2(id="kpi-roi", className="text-info"),
                            html.P("Return on investment", className="card-text")
                        ])
                    ])
                ], width=3)
            ], className="mb-4"),
            
            # Charts Row 1
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Fleet Health Overview"),
                        dbc.CardBody([
                            dcc.Graph(id="fleet-health-gauge")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Maintenance Efficiency"),
                        dbc.CardBody([
                            dcc.Graph(id="maintenance-efficiency-chart")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Charts Row 2
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Cost Breakdown Analysis"),
                        dbc.CardBody([
                            dcc.Graph(id="cost-breakdown-chart")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Trend Analysis"),
                        dbc.CardBody([
                            dcc.Graph(id="trend-analysis-chart")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Equipment Comparison Table
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Equipment Type Comparison"),
                        dbc.CardBody([
                            html.Div(id="equipment-comparison-table")
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            # Refresh Controls
            dbc.Row([
                dbc.Col([
                    dbc.Button("Refresh Data", id="refresh-button", color="primary", className="me-2"),
                    html.Span(id="last-updated", className="text-muted")
                ], className="text-center")
            ]),
            
            # Hidden div to store data
            html.Div(id="dashboard-data", style={"display": "none"})
            
        ], fluid=True)
    
    def setup_callbacks(self):
        """Setup dashboard callbacks for interactivity"""
        
        @self.app.callback(
            [Output("dashboard-data", "children"),
             Output("last-updated", "children")],
            [Input("refresh-button", "n_clicks")]
        )
        def update_dashboard_data(n_clicks):
            """Update dashboard data when refresh button is clicked"""
            try:
                # Generate sample data for demonstration
                equipment_data = self._generate_sample_equipment_data()
                prediction_data = self._generate_sample_prediction_data()
                maintenance_data = self._generate_sample_maintenance_data()
                
                # Calculate analytics
                kpi_metrics = self.analytics.calculate_kpi_metrics(
                    equipment_data, prediction_data, maintenance_data
                )
                
                # Generate time series for trend analysis
                time_series = self._generate_sample_time_series()
                trend_analysis = self.analytics.perform_trend_analysis(
                    time_series, 'system_efficiency'
                )
                
                cost_analysis = self.analytics.calculate_cost_analysis(
                    maintenance_data, equipment_data
                )
                
                equipment_comparisons = self.analytics.perform_equipment_comparison(
                    equipment_data
                )
                
                # Store data as JSON
                dashboard_data = {
                    'kpi_metrics': kpi_metrics.to_dict(),
                    'trend_analysis': trend_analysis.to_dict(),
                    'cost_analysis': cost_analysis.to_dict(),
                    'equipment_comparisons': [comp.to_dict() for comp in equipment_comparisons],
                    'time_series': time_series.to_dict()
                }
                
                last_updated = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                
                return json.dumps(dashboard_data), last_updated
                
            except Exception as e:
                logger.error(f"Error updating dashboard data: {e}")
                return "{}", f"Error updating data: {e}"
        
        @self.app.callback(
            [Output("kpi-equipment-count", "children"),
             Output("kpi-at-risk", "children"),
             Output("kpi-cost-savings", "children"),
             Output("kpi-roi", "children")],
            [Input("dashboard-data", "children")]
        )
        def update_kpi_cards(dashboard_data_json):
            """Update KPI cards with latest data"""
            try:
                if not dashboard_data_json:
                    return "0", "0", "$0", "0%"
                
                data = json.loads(dashboard_data_json)
                kpi = data['kpi_metrics']
                
                equipment_count = str(kpi['total_equipment_monitored'])
                at_risk = str(kpi['equipment_at_risk'])
                cost_savings = f"${kpi['maintenance_cost_savings']:,.0f}"
                roi = f"{kpi['roi_percentage']:.1f}%"
                
                return equipment_count, at_risk, cost_savings, roi
                
            except Exception as e:
                logger.error(f"Error updating KPI cards: {e}")
                return "Error", "Error", "Error", "Error"
        
        @self.app.callback(
            Output("fleet-health-gauge", "figure"),
            [Input("dashboard-data", "children")]
        )
        def update_fleet_health_gauge(dashboard_data_json):
            """Update fleet health gauge chart"""
            try:
                if not dashboard_data_json:
                    return go.Figure()
                
                data = json.loads(dashboard_data_json)
                kpi = data['kpi_metrics']
                
                # Calculate fleet health score
                total_equipment = kpi['total_equipment_monitored']
                at_risk = kpi['equipment_at_risk']
                health_score = ((total_equipment - at_risk) / max(total_equipment, 1)) * 100
                
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=health_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Fleet Health Score (%)"},
                    delta={'reference': 90},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 80], 'color': "yellow"},
                            {'range': [80, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                
                fig.update_layout(height=300)
                return fig
                
            except Exception as e:
                logger.error(f"Error updating fleet health gauge: {e}")
                return go.Figure()
        
        @self.app.callback(
            Output("maintenance-efficiency-chart", "figure"),
            [Input("dashboard-data", "children")]
        )
        def update_maintenance_efficiency_chart(dashboard_data_json):
            """Update maintenance efficiency chart"""
            try:
                if not dashboard_data_json:
                    return go.Figure()
                
                data = json.loads(dashboard_data_json)
                kpi = data['kpi_metrics']
                
                metrics = {
                    'Prediction Accuracy': kpi['prediction_accuracy'] * 100,
                    'System Uptime': kpi['system_uptime_percentage'],
                    'Maintenance Efficiency': kpi['maintenance_efficiency_score'] * 100,
                    'FPR (inverted)': (1 - kpi['false_positive_rate']) * 100
                }
                
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(metrics.keys()),
                        y=list(metrics.values()),
                        marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
                    )
                ])
                
                fig.update_layout(
                    title="Key Performance Metrics (%)",
                    yaxis_title="Percentage",
                    height=300
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error updating maintenance efficiency chart: {e}")
                return go.Figure()
        
        @self.app.callback(
            Output("cost-breakdown-chart", "figure"),
            [Input("dashboard-data", "children")]
        )
        def update_cost_breakdown_chart(dashboard_data_json):
            """Update cost breakdown pie chart"""
            try:
                if not dashboard_data_json:
                    return go.Figure()
                
                data = json.loads(dashboard_data_json)
                cost_analysis = data['cost_analysis']
                breakdown = cost_analysis['breakdown_by_category']
                
                fig = go.Figure(data=[
                    go.Pie(
                        labels=list(breakdown.keys()),
                        values=list(breakdown.values()),
                        hole=0.3
                    )
                ])
                
                fig.update_layout(
                    title="Cost Breakdown by Category",
                    height=300
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error updating cost breakdown chart: {e}")
                return go.Figure()
        
        @self.app.callback(
            Output("trend-analysis-chart", "figure"),
            [Input("dashboard-data", "children")]
        )
        def update_trend_analysis_chart(dashboard_data_json):
            """Update trend analysis chart"""
            try:
                if not dashboard_data_json:
                    return go.Figure()
                
                data = json.loads(dashboard_data_json)
                time_series = pd.DataFrame(data['time_series'])
                
                if time_series.empty:
                    return go.Figure()
                
                fig = go.Figure()
                
                # Historical data
                fig.add_trace(go.Scatter(
                    x=time_series.index,
                    y=time_series['system_efficiency'],
                    mode='lines+markers',
                    name='Historical',
                    line=dict(color='blue')
                ))
                
                # Add trend line if available
                trend = data['trend_analysis']
                if trend['forecast_values']:
                    fig.add_trace(go.Scatter(
                        x=trend['forecast_dates'][:10],
                        y=trend['forecast_values'][:10],
                        mode='lines+markers',
                        name='Forecast',
                        line=dict(color='red', dash='dash')
                    ))
                
                fig.update_layout(
                    title="System Efficiency Trend",
                    xaxis_title="Date",
                    yaxis_title="Efficiency (%)",
                    height=300
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error updating trend analysis chart: {e}")
                return go.Figure()
        
        @self.app.callback(
            Output("equipment-comparison-table", "children"),
            [Input("dashboard-data", "children")]
        )
        def update_equipment_comparison_table(dashboard_data_json):
            """Update equipment comparison table"""
            try:
                if not dashboard_data_json:
                    return html.Div("No data available")
                
                data = json.loads(dashboard_data_json)
                comparisons = data['equipment_comparisons']
                
                if not comparisons:
                    return html.Div("No equipment comparison data available")
                
                # Create table rows
                table_header = [
                    html.Thead([
                        html.Tr([
                            html.Th("Rank"),
                            html.Th("Equipment Type"),
                            html.Th("Total Units"),
                            html.Th("Avg RUL"),
                            html.Th("Failure Rate"),
                            html.Th("Reliability Score"),
                            html.Th("Cost per Unit")
                        ])
                    ])
                ]
                
                table_rows = []
                for comp in comparisons:
                    row = html.Tr([
                        html.Td(comp['performance_ranking']),
                        html.Td(comp['equipment_type'].title()),
                        html.Td(comp['total_units']),
                        html.Td(f"{comp['average_rul']:.1f}"),
                        html.Td(f"{comp['failure_rate']:.1%}"),
                        html.Td(f"{comp['reliability_score']:.2f}"),
                        html.Td(f"${comp['cost_per_unit']:,.0f}")
                    ])
                    table_rows.append(row)
                
                table_body = [html.Tbody(table_rows)]
                
                return dbc.Table(
                    table_header + table_body,
                    bordered=True,
                    hover=True,
                    responsive=True,
                    striped=True
                )
                
            except Exception as e:
                logger.error(f"Error updating equipment comparison table: {e}")
                return html.Div(f"Error loading table: {e}")
    
    def _generate_sample_equipment_data(self) -> List[Dict[str, Any]]:
        """Generate sample equipment data for demonstration"""
        np.random.seed(42)
        equipment_types = ['capacitor', 'motor', 'pump', 'compressor']
        
        equipment_data = []
        for i in range(50):
            equipment_data.append({
                'id': f'EQ_{i:03d}',
                'type': np.random.choice(equipment_types),
                'rul_cycles': max(0, np.random.normal(100, 30)),
                'maintenance_count': np.random.poisson(2),
                'annual_maintenance_cost': np.random.normal(5000, 1500)
            })
        
        return equipment_data
    
    def _generate_sample_prediction_data(self) -> List[Dict[str, Any]]:
        """Generate sample prediction data for demonstration"""
        np.random.seed(42)
        
        prediction_data = []
        for i in range(100):
            actual_rul = np.random.normal(80, 25)
            predicted_rul = actual_rul + np.random.normal(0, 10)
            
            prediction_data.append({
                'equipment_id': f'EQ_{i % 50:03d}',
                'predicted_rul': max(0, predicted_rul),
                'actual_rul': max(0, actual_rul),
                'anomaly_flag': np.random.random() < 0.1,
                'actual_failure': np.random.random() < 0.05
            })
        
        return prediction_data
    
    def _generate_sample_maintenance_data(self) -> List[Dict[str, Any]]:
        """Generate sample maintenance data for demonstration"""
        np.random.seed(42)
        
        maintenance_data = []
        for i in range(75):
            maintenance_type = 'preventive' if np.random.random() < 0.7 else 'corrective'
            
            maintenance_data.append({
                'equipment_id': f'EQ_{i % 50:03d}',
                'type': maintenance_type,
                'downtime_hours': np.random.exponential(4 if maintenance_type == 'preventive' else 12),
                'prevented_failure': maintenance_type == 'preventive' and np.random.random() < 0.3
            })
        
        return maintenance_data
    
    def _generate_sample_time_series(self) -> pd.DataFrame:
        """Generate sample time series data for trend analysis"""
        np.random.seed(42)
        
        dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
        
        # Generate trending efficiency data
        base_efficiency = 85
        trend = np.linspace(0, 5, len(dates))  # Slight upward trend
        noise = np.random.normal(0, 2, len(dates))
        seasonal = 3 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
        
        efficiency = base_efficiency + trend + seasonal + noise
        efficiency = np.clip(efficiency, 70, 100)  # Keep within reasonable bounds
        
        return pd.DataFrame({
            'system_efficiency': efficiency
        }, index=dates)
    
    def run_server(self, host: str = '127.0.0.1', port: int = 8050, debug: bool = False):
        """
        Run the dashboard server
        
        Args:
            host: Host address
            port: Port number
            debug: Enable debug mode
        """
        logger.info(f"Starting executive dashboard server on {host}:{port}")
        self.app.run_server(host=host, port=port, debug=debug)
    
    def generate_static_report(self, output_path: str) -> str:
        """
        Generate static HTML report
        
        Args:
            output_path: Path to save the HTML report
            
        Returns:
            Path to the generated report
        """
        try:
            # Generate sample data
            equipment_data = self._generate_sample_equipment_data()
            prediction_data = self._generate_sample_prediction_data()
            maintenance_data = self._generate_sample_maintenance_data()
            
            # Calculate analytics
            kpi_metrics = self.analytics.calculate_kpi_metrics(
                equipment_data, prediction_data, maintenance_data
            )
            
            time_series = self._generate_sample_time_series()
            trend_analysis = self.analytics.perform_trend_analysis(
                time_series, 'system_efficiency'
            )
            
            cost_analysis = self.analytics.calculate_cost_analysis(
                maintenance_data, equipment_data
            )
            
            # Generate HTML report
            html_content = self.analytics.generate_executive_dashboard(
                kpi_metrics, [trend_analysis], cost_analysis
            )
            
            with open(output_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Static executive report generated: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error generating static report: {e}")
            raise