"""
Advanced Analytics and Reporting System

This module provides comprehensive analytics capabilities including:
- Executive dashboards with KPI tracking
- Trend analysis and forecasting
- Cost savings calculation and ROI analysis
- Comparative analysis across equipment types
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings

logger = logging.getLogger(__name__)

@dataclass
class KPIMetrics:
    """Key Performance Indicators for executive dashboard"""
    total_equipment_monitored: int
    equipment_at_risk: int
    predicted_failures_prevented: int
    maintenance_cost_savings: float
    system_uptime_percentage: float
    false_positive_rate: float
    prediction_accuracy: float
    average_rul_prediction: float
    maintenance_efficiency_score: float
    roi_percentage: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class TrendAnalysis:
    """Trend analysis results"""
    metric_name: str
    time_period: str
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_strength: float  # 0-1 scale
    slope: float
    r_squared: float
    forecast_values: List[float]
    forecast_dates: List[str]
    confidence_intervals: List[Tuple[float, float]]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class CostAnalysis:
    """Cost savings and ROI analysis"""
    total_maintenance_cost_baseline: float
    total_maintenance_cost_optimized: float
    cost_savings: float
    cost_savings_percentage: float
    roi_percentage: float
    payback_period_months: float
    implementation_cost: float
    operational_cost_per_month: float
    breakdown_by_category: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class EquipmentComparison:
    """Comparative analysis across equipment types"""
    equipment_type: str
    total_units: int
    average_rul: float
    failure_rate: float
    maintenance_frequency: float
    cost_per_unit: float
    reliability_score: float
    performance_ranking: int
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class AdvancedAnalytics:
    """Advanced analytics and reporting system"""
    
    def __init__(self, data_path: Optional[str] = None):
        """
        Initialize advanced analytics system
        
        Args:
            data_path: Path to historical data storage
        """
        self.data_path = Path(data_path) if data_path else Path("analytics_data")
        self.data_path.mkdir(exist_ok=True)
        
        # Cost parameters (configurable)
        self.cost_parameters = {
            'preventive_maintenance_cost': 1000.0,
            'corrective_maintenance_cost': 5000.0,
            'downtime_cost_per_hour': 2000.0,
            'replacement_cost': 15000.0,
            'labor_cost_per_hour': 100.0,
            'system_implementation_cost': 50000.0,
            'monthly_operational_cost': 2000.0
        }
        
        # Equipment type configurations
        self.equipment_types = {
            'capacitor': {'baseline_mtbf': 2000, 'maintenance_interval': 500},
            'motor': {'baseline_mtbf': 5000, 'maintenance_interval': 1000},
            'pump': {'baseline_mtbf': 3000, 'maintenance_interval': 750},
            'compressor': {'baseline_mtbf': 4000, 'maintenance_interval': 800}
        }
        
    def calculate_kpi_metrics(self, 
                            equipment_data: List[Dict[str, Any]],
                            prediction_data: List[Dict[str, Any]],
                            maintenance_data: List[Dict[str, Any]]) -> KPIMetrics:
        """
        Calculate key performance indicators for executive dashboard
        
        Args:
            equipment_data: List of equipment status records
            prediction_data: List of RUL predictions
            maintenance_data: List of maintenance records
            
        Returns:
            KPIMetrics object with calculated KPIs
        """
        try:
            # Basic counts
            total_equipment = len(equipment_data)
            equipment_at_risk = sum(1 for eq in equipment_data 
                                  if eq.get('rul_cycles', float('inf')) < 50)
            
            # Prediction accuracy metrics
            accurate_predictions = sum(1 for pred in prediction_data 
                                     if abs(pred.get('predicted_rul', 0) - pred.get('actual_rul', 0)) < 10)
            prediction_accuracy = accurate_predictions / max(len(prediction_data), 1)
            
            # False positive rate
            false_positives = sum(1 for pred in prediction_data 
                                if pred.get('anomaly_flag', False) and not pred.get('actual_failure', False))
            total_predictions = len(prediction_data)
            fpr = false_positives / max(total_predictions, 1)
            
            # Cost savings calculation
            prevented_failures = sum(1 for maint in maintenance_data 
                                   if maint.get('type') == 'preventive' and maint.get('prevented_failure', False))
            
            cost_savings = self._calculate_cost_savings(prevented_failures, maintenance_data)
            
            # System uptime
            total_downtime_hours = sum(maint.get('downtime_hours', 0) for maint in maintenance_data)
            total_operational_hours = 24 * 365 * total_equipment  # Assume 24/7 operation
            uptime_percentage = (total_operational_hours - total_downtime_hours) / total_operational_hours * 100
            
            # Average RUL
            avg_rul = np.mean([eq.get('rul_cycles', 0) for eq in equipment_data])
            
            # Maintenance efficiency score (custom metric)
            planned_maintenance = sum(1 for maint in maintenance_data if maint.get('type') == 'preventive')
            unplanned_maintenance = sum(1 for maint in maintenance_data if maint.get('type') == 'corrective')
            efficiency_score = planned_maintenance / max(planned_maintenance + unplanned_maintenance, 1)
            
            # ROI calculation
            implementation_cost = self.cost_parameters['system_implementation_cost']
            monthly_operational = self.cost_parameters['monthly_operational_cost']
            annual_operational = monthly_operational * 12
            roi_percentage = (cost_savings - annual_operational) / (implementation_cost + annual_operational) * 100
            
            return KPIMetrics(
                total_equipment_monitored=total_equipment,
                equipment_at_risk=equipment_at_risk,
                predicted_failures_prevented=prevented_failures,
                maintenance_cost_savings=cost_savings,
                system_uptime_percentage=uptime_percentage,
                false_positive_rate=fpr,
                prediction_accuracy=prediction_accuracy,
                average_rul_prediction=avg_rul,
                maintenance_efficiency_score=efficiency_score,
                roi_percentage=roi_percentage
            )
            
        except Exception as e:
            logger.error(f"Error calculating KPI metrics: {e}")
            # Return default metrics
            return KPIMetrics(
                total_equipment_monitored=0,
                equipment_at_risk=0,
                predicted_failures_prevented=0,
                maintenance_cost_savings=0.0,
                system_uptime_percentage=0.0,
                false_positive_rate=0.0,
                prediction_accuracy=0.0,
                average_rul_prediction=0.0,
                maintenance_efficiency_score=0.0,
                roi_percentage=0.0
            )
    
    def perform_trend_analysis(self, 
                             time_series_data: pd.DataFrame,
                             metric_column: str,
                             forecast_periods: int = 30) -> TrendAnalysis:
        """
        Perform trend analysis and forecasting on time series data
        
        Args:
            time_series_data: DataFrame with datetime index and metric values
            metric_column: Name of the column to analyze
            forecast_periods: Number of future periods to forecast
            
        Returns:
            TrendAnalysis object with trend and forecast results
        """
        try:
            if len(time_series_data) < 3:
                logger.warning("Insufficient data for trend analysis")
                return self._default_trend_analysis(metric_column)
            
            # Prepare data
            data = time_series_data[metric_column].dropna()
            if len(data) < 3:
                return self._default_trend_analysis(metric_column)
            
            # Convert datetime index to numeric for regression
            x = np.arange(len(data)).reshape(-1, 1)
            y = data.values
            
            # Linear trend analysis
            reg = LinearRegression()
            reg.fit(x, y)
            
            slope = reg.coef_[0]
            r_squared = reg.score(x, y)
            
            # Determine trend direction and strength
            if abs(slope) < 0.01:
                trend_direction = 'stable'
                trend_strength = 0.0
            elif slope > 0:
                trend_direction = 'increasing'
                trend_strength = min(abs(slope) / np.std(y), 1.0)
            else:
                trend_direction = 'decreasing'
                trend_strength = min(abs(slope) / np.std(y), 1.0)
            
            # Generate forecasts
            future_x = np.arange(len(data), len(data) + forecast_periods).reshape(-1, 1)
            forecast_values = reg.predict(future_x).tolist()
            
            # Generate forecast dates
            last_date = time_series_data.index[-1]
            if isinstance(last_date, str):
                last_date = pd.to_datetime(last_date)
            
            forecast_dates = []
            for i in range(1, forecast_periods + 1):
                future_date = last_date + timedelta(days=i)
                forecast_dates.append(future_date.strftime('%Y-%m-%d'))
            
            # Calculate confidence intervals (simple approach)
            residuals = y - reg.predict(x)
            mse = np.mean(residuals ** 2)
            std_error = np.sqrt(mse)
            
            confidence_intervals = []
            for forecast in forecast_values:
                lower = forecast - 1.96 * std_error
                upper = forecast + 1.96 * std_error
                confidence_intervals.append((lower, upper))
            
            return TrendAnalysis(
                metric_name=metric_column,
                time_period=f"{time_series_data.index[0]} to {time_series_data.index[-1]}",
                trend_direction=trend_direction,
                trend_strength=trend_strength,
                slope=slope,
                r_squared=r_squared,
                forecast_values=forecast_values,
                forecast_dates=forecast_dates,
                confidence_intervals=confidence_intervals
            )
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return self._default_trend_analysis(metric_column)
    
    def calculate_cost_analysis(self, 
                              maintenance_data: List[Dict[str, Any]],
                              equipment_data: List[Dict[str, Any]]) -> CostAnalysis:
        """
        Calculate comprehensive cost savings and ROI analysis
        
        Args:
            maintenance_data: Historical maintenance records
            equipment_data: Equipment status and prediction data
            
        Returns:
            CostAnalysis object with detailed cost breakdown
        """
        try:
            # Baseline costs (without predictive maintenance)
            total_equipment = len(equipment_data)
            
            # Estimate baseline failures per year
            baseline_failure_rate = 0.15  # 15% annual failure rate
            baseline_failures_per_year = total_equipment * baseline_failure_rate
            
            # Baseline maintenance costs
            baseline_corrective_cost = (baseline_failures_per_year * 
                                      self.cost_parameters['corrective_maintenance_cost'])
            baseline_downtime_cost = (baseline_failures_per_year * 8 * 
                                    self.cost_parameters['downtime_cost_per_hour'])
            baseline_replacement_cost = (baseline_failures_per_year * 0.3 * 
                                       self.cost_parameters['replacement_cost'])
            
            total_baseline_cost = (baseline_corrective_cost + 
                                 baseline_downtime_cost + 
                                 baseline_replacement_cost)
            
            # Optimized costs (with predictive maintenance)
            preventive_maintenance_count = sum(1 for maint in maintenance_data 
                                             if maint.get('type') == 'preventive')
            corrective_maintenance_count = sum(1 for maint in maintenance_data 
                                             if maint.get('type') == 'corrective')
            
            optimized_preventive_cost = (preventive_maintenance_count * 
                                       self.cost_parameters['preventive_maintenance_cost'])
            optimized_corrective_cost = (corrective_maintenance_count * 
                                       self.cost_parameters['corrective_maintenance_cost'])
            
            # Reduced downtime due to planned maintenance
            planned_downtime_hours = sum(maint.get('downtime_hours', 2) for maint in maintenance_data 
                                       if maint.get('type') == 'preventive')
            unplanned_downtime_hours = sum(maint.get('downtime_hours', 8) for maint in maintenance_data 
                                         if maint.get('type') == 'corrective')
            
            optimized_downtime_cost = ((planned_downtime_hours + unplanned_downtime_hours) * 
                                     self.cost_parameters['downtime_cost_per_hour'])
            
            # Reduced replacement costs
            prevented_failures = sum(1 for maint in maintenance_data 
                                   if maint.get('prevented_failure', False))
            optimized_replacement_cost = ((corrective_maintenance_count - prevented_failures) * 0.3 * 
                                        self.cost_parameters['replacement_cost'])
            
            total_optimized_cost = (optimized_preventive_cost + 
                                  optimized_corrective_cost + 
                                  optimized_downtime_cost + 
                                  optimized_replacement_cost)
            
            # Calculate savings and ROI
            cost_savings = total_baseline_cost - total_optimized_cost
            cost_savings_percentage = (cost_savings / total_baseline_cost * 100) if total_baseline_cost > 0 else 0
            
            implementation_cost = self.cost_parameters['system_implementation_cost']
            annual_operational_cost = self.cost_parameters['monthly_operational_cost'] * 12
            
            roi_percentage = ((cost_savings - annual_operational_cost) / 
                            (implementation_cost + annual_operational_cost) * 100)
            
            payback_period_months = (implementation_cost / 
                                   max(cost_savings / 12 - self.cost_parameters['monthly_operational_cost'], 1))
            
            # Cost breakdown by category
            breakdown = {
                'preventive_maintenance': optimized_preventive_cost,
                'corrective_maintenance': optimized_corrective_cost,
                'downtime_costs': optimized_downtime_cost,
                'replacement_costs': optimized_replacement_cost,
                'operational_costs': annual_operational_cost
            }
            
            return CostAnalysis(
                total_maintenance_cost_baseline=total_baseline_cost,
                total_maintenance_cost_optimized=total_optimized_cost,
                cost_savings=cost_savings,
                cost_savings_percentage=cost_savings_percentage,
                roi_percentage=roi_percentage,
                payback_period_months=payback_period_months,
                implementation_cost=implementation_cost,
                operational_cost_per_month=self.cost_parameters['monthly_operational_cost'],
                breakdown_by_category=breakdown
            )
            
        except Exception as e:
            logger.error(f"Error in cost analysis: {e}")
            return CostAnalysis(
                total_maintenance_cost_baseline=0.0,
                total_maintenance_cost_optimized=0.0,
                cost_savings=0.0,
                cost_savings_percentage=0.0,
                roi_percentage=0.0,
                payback_period_months=0.0,
                implementation_cost=0.0,
                operational_cost_per_month=0.0,
                breakdown_by_category={}
            )
    
    def perform_equipment_comparison(self, 
                                   equipment_data: List[Dict[str, Any]]) -> List[EquipmentComparison]:
        """
        Perform comparative analysis across different equipment types
        
        Args:
            equipment_data: List of equipment records with type and performance data
            
        Returns:
            List of EquipmentComparison objects sorted by performance ranking
        """
        try:
            # Group equipment by type
            equipment_by_type = {}
            for equipment in equipment_data:
                eq_type = equipment.get('type', 'unknown')
                if eq_type not in equipment_by_type:
                    equipment_by_type[eq_type] = []
                equipment_by_type[eq_type].append(equipment)
            
            comparisons = []
            
            for eq_type, equipment_list in equipment_by_type.items():
                if not equipment_list:
                    continue
                
                # Calculate metrics for this equipment type
                total_units = len(equipment_list)
                
                # Average RUL
                rul_values = [eq.get('rul_cycles', 0) for eq in equipment_list]
                average_rul = np.mean(rul_values) if rul_values else 0
                
                # Failure rate (equipment with RUL < 10)
                critical_equipment = sum(1 for eq in equipment_list 
                                       if eq.get('rul_cycles', float('inf')) < 10)
                failure_rate = critical_equipment / total_units if total_units > 0 else 0
                
                # Maintenance frequency (maintenance events per unit per year)
                total_maintenance_events = sum(eq.get('maintenance_count', 0) for eq in equipment_list)
                maintenance_frequency = total_maintenance_events / total_units if total_units > 0 else 0
                
                # Cost per unit (annual maintenance cost per unit)
                total_maintenance_cost = sum(eq.get('annual_maintenance_cost', 0) for eq in equipment_list)
                cost_per_unit = total_maintenance_cost / total_units if total_units > 0 else 0
                
                # Reliability score (composite metric)
                # Higher RUL, lower failure rate, lower maintenance frequency = higher reliability
                rul_score = min(average_rul / 100, 1.0)  # Normalize to 0-1
                failure_score = 1.0 - failure_rate  # Invert so lower failure rate = higher score
                maintenance_score = max(0, 1.0 - maintenance_frequency / 10)  # Normalize
                
                reliability_score = (rul_score * 0.4 + failure_score * 0.4 + maintenance_score * 0.2)
                
                comparison = EquipmentComparison(
                    equipment_type=eq_type,
                    total_units=total_units,
                    average_rul=average_rul,
                    failure_rate=failure_rate,
                    maintenance_frequency=maintenance_frequency,
                    cost_per_unit=cost_per_unit,
                    reliability_score=reliability_score,
                    performance_ranking=0  # Will be set after sorting
                )
                comparisons.append(comparison)
            
            # Sort by reliability score (descending) and assign rankings
            comparisons.sort(key=lambda x: x.reliability_score, reverse=True)
            for i, comparison in enumerate(comparisons):
                comparison.performance_ranking = i + 1
            
            return comparisons
            
        except Exception as e:
            logger.error(f"Error in equipment comparison: {e}")
            return []
    
    def generate_executive_dashboard(self, 
                                   kpi_metrics: KPIMetrics,
                                   trend_analyses: List[TrendAnalysis],
                                   cost_analysis: CostAnalysis) -> str:
        """
        Generate HTML executive dashboard with visualizations
        
        Args:
            kpi_metrics: Calculated KPI metrics
            trend_analyses: List of trend analysis results
            cost_analysis: Cost savings and ROI analysis
            
        Returns:
            HTML string for the executive dashboard
        """
        try:
            # Create dashboard with multiple subplots
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('KPI Overview', 'Cost Savings Analysis',
                              'Trend Analysis', 'ROI Breakdown',
                              'Equipment Health Status', 'Maintenance Efficiency'),
                specs=[[{"type": "indicator"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "pie"}],
                       [{"type": "indicator"}, {"type": "bar"}]]
            )
            
            # KPI indicators
            fig.add_trace(
                go.Indicator(
                    mode="number+gauge+delta",
                    value=kpi_metrics.roi_percentage,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={"text": "ROI %"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkblue"},
                           'steps': [{'range': [0, 50], 'color': "lightgray"},
                                   {'range': [50, 100], 'color': "gray"}],
                           'threshold': {'line': {'color': "red", 'width': 4},
                                       'thickness': 0.75, 'value': 90}}
                ),
                row=1, col=1
            )
            
            # Cost savings bar chart
            cost_categories = list(cost_analysis.breakdown_by_category.keys())
            cost_values = list(cost_analysis.breakdown_by_category.values())
            
            fig.add_trace(
                go.Bar(x=cost_categories, y=cost_values, name="Cost Breakdown"),
                row=1, col=2
            )
            
            # Add trend analysis if available
            if trend_analyses:
                trend = trend_analyses[0]  # Use first trend for demo
                fig.add_trace(
                    go.Scatter(
                        x=trend.forecast_dates[:10],  # Show first 10 forecasts
                        y=trend.forecast_values[:10],
                        mode='lines+markers',
                        name=f"{trend.metric_name} Forecast"
                    ),
                    row=2, col=1
                )
            
            # ROI pie chart
            roi_breakdown = {
                'Cost Savings': cost_analysis.cost_savings,
                'Implementation Cost': cost_analysis.implementation_cost,
                'Operational Cost': cost_analysis.operational_cost_per_month * 12
            }
            
            fig.add_trace(
                go.Pie(labels=list(roi_breakdown.keys()), 
                      values=list(roi_breakdown.values()),
                      name="ROI Breakdown"),
                row=2, col=2
            )
            
            # Equipment health gauge
            health_score = (100 - kpi_metrics.equipment_at_risk / 
                          max(kpi_metrics.total_equipment_monitored, 1) * 100)
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=health_score,
                    title={'text': "Fleet Health Score"},
                    gauge={'axis': {'range': [None, 100]},
                           'bar': {'color': "darkgreen"},
                           'steps': [{'range': [0, 50], 'color': "red"},
                                   {'range': [50, 80], 'color': "yellow"},
                                   {'range': [80, 100], 'color': "green"}]}
                ),
                row=3, col=1
            )
            
            # Maintenance efficiency bar
            efficiency_metrics = {
                'Prediction Accuracy': kpi_metrics.prediction_accuracy * 100,
                'Maintenance Efficiency': kpi_metrics.maintenance_efficiency_score * 100,
                'System Uptime': kpi_metrics.system_uptime_percentage
            }
            
            fig.add_trace(
                go.Bar(x=list(efficiency_metrics.keys()), 
                      y=list(efficiency_metrics.values()),
                      name="Efficiency Metrics"),
                row=3, col=2
            )
            
            # Update layout
            fig.update_layout(
                title_text="Executive Dashboard - Predictive Maintenance Analytics",
                showlegend=False,
                height=1200
            )
            
            # Generate HTML
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Executive Dashboard</title>
                <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    .kpi-summary {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                    .kpi-card {{ background: #f0f0f0; padding: 15px; border-radius: 5px; text-align: center; }}
                    .kpi-value {{ font-size: 24px; font-weight: bold; color: #2E86AB; }}
                    .kpi-label {{ font-size: 14px; color: #666; }}
                </style>
            </head>
            <body>
                <h1>Executive Dashboard - Predictive Maintenance System</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="kpi-summary">
                    <div class="kpi-card">
                        <div class="kpi-value">{kpi_metrics.total_equipment_monitored}</div>
                        <div class="kpi-label">Equipment Monitored</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value">{kpi_metrics.predicted_failures_prevented}</div>
                        <div class="kpi-label">Failures Prevented</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value">${cost_analysis.cost_savings:,.0f}</div>
                        <div class="kpi-label">Cost Savings</div>
                    </div>
                    <div class="kpi-card">
                        <div class="kpi-value">{kpi_metrics.roi_percentage:.1f}%</div>
                        <div class="kpi-label">ROI</div>
                    </div>
                </div>
                
                <div id="dashboard-plots"></div>
                
                <script>
                    var plotData = {fig.to_json()};
                    Plotly.newPlot('dashboard-plots', plotData.data, plotData.layout);
                </script>
            </body>
            </html>
            """
            
            return html_content
            
        except Exception as e:
            logger.error(f"Error generating executive dashboard: {e}")
            return f"<html><body><h1>Dashboard Error</h1><p>Error: {e}</p></body></html>"
    
    def _calculate_cost_savings(self, prevented_failures: int, maintenance_data: List[Dict[str, Any]]) -> float:
        """Calculate cost savings from prevented failures"""
        # Cost of prevented failures
        failure_cost_savings = prevented_failures * (
            self.cost_parameters['corrective_maintenance_cost'] + 
            8 * self.cost_parameters['downtime_cost_per_hour'] +
            0.3 * self.cost_parameters['replacement_cost']
        )
        
        # Additional savings from optimized maintenance scheduling
        preventive_count = sum(1 for m in maintenance_data if m.get('type') == 'preventive')
        scheduling_savings = preventive_count * 200  # Estimated savings per optimized maintenance
        
        return failure_cost_savings + scheduling_savings
    
    def _default_trend_analysis(self, metric_name: str) -> TrendAnalysis:
        """Return default trend analysis when insufficient data"""
        return TrendAnalysis(
            metric_name=metric_name,
            time_period="insufficient_data",
            trend_direction="stable",
            trend_strength=0.0,
            slope=0.0,
            r_squared=0.0,
            forecast_values=[],
            forecast_dates=[],
            confidence_intervals=[]
        )
    
    def save_analytics_report(self, 
                            kpi_metrics: KPIMetrics,
                            trend_analyses: List[TrendAnalysis],
                            cost_analysis: CostAnalysis,
                            equipment_comparisons: List[EquipmentComparison],
                            output_path: Optional[str] = None) -> str:
        """
        Save comprehensive analytics report to file
        
        Args:
            kpi_metrics: KPI metrics
            trend_analyses: Trend analysis results
            cost_analysis: Cost analysis results
            equipment_comparisons: Equipment comparison results
            output_path: Optional output file path
            
        Returns:
            Path to saved report file
        """
        try:
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = self.data_path / f"analytics_report_{timestamp}.json"
            
            report_data = {
                'generated_at': datetime.now().isoformat(),
                'kpi_metrics': kpi_metrics.to_dict(),
                'trend_analyses': [trend.to_dict() for trend in trend_analyses],
                'cost_analysis': cost_analysis.to_dict(),
                'equipment_comparisons': [comp.to_dict() for comp in equipment_comparisons]
            }
            
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Analytics report saved to {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error saving analytics report: {e}")
            raise