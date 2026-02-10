#!/usr/bin/env python3
"""
Test script for Advanced Analytics and Reporting System

This script tests the advanced analytics capabilities including:
- KPI calculation and tracking
- Trend analysis and forecasting
- Cost savings and ROI analysis
- Equipment comparison and ranking
- Executive dashboard generation
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import json

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from true_rul.advanced_analytics import AdvancedAnalytics, KPIMetrics, TrendAnalysis, CostAnalysis
from true_rul.executive_dashboard import ExecutiveDashboard

def generate_test_data():
    """Generate comprehensive test data for analytics testing"""
    np.random.seed(42)
    
    # Equipment data
    equipment_types = ['capacitor', 'motor', 'pump', 'compressor']
    equipment_data = []
    
    for i in range(100):
        eq_type = np.random.choice(equipment_types)
        base_rul = {'capacitor': 150, 'motor': 300, 'pump': 200, 'compressor': 250}[eq_type]
        
        equipment_data.append({
            'id': f'EQ_{i:03d}',
            'type': eq_type,
            'rul_cycles': max(0, np.random.normal(base_rul, base_rul * 0.3)),
            'maintenance_count': np.random.poisson(3),
            'annual_maintenance_cost': np.random.normal(8000, 2000),
            'last_maintenance': (datetime.now() - timedelta(days=np.random.randint(1, 365))).isoformat()
        })
    
    # Prediction data
    prediction_data = []
    for i in range(200):
        actual_rul = np.random.normal(120, 40)
        prediction_error = np.random.normal(0, 15)
        predicted_rul = actual_rul + prediction_error
        
        prediction_data.append({
            'equipment_id': f'EQ_{i % 100:03d}',
            'predicted_rul': max(0, predicted_rul),
            'actual_rul': max(0, actual_rul),
            'anomaly_flag': np.random.random() < 0.08,  # 8% anomaly rate
            'actual_failure': np.random.random() < 0.03,  # 3% actual failure rate
            'prediction_date': (datetime.now() - timedelta(days=np.random.randint(1, 30))).isoformat(),
            'confidence_score': np.random.uniform(0.7, 0.95)
        })
    
    # Maintenance data
    maintenance_data = []
    for i in range(150):
        maintenance_type = 'preventive' if np.random.random() < 0.75 else 'corrective'
        
        maintenance_data.append({
            'equipment_id': f'EQ_{i % 100:03d}',
            'type': maintenance_type,
            'downtime_hours': np.random.exponential(3 if maintenance_type == 'preventive' else 16),
            'cost': np.random.normal(
                1500 if maintenance_type == 'preventive' else 6000, 
                500 if maintenance_type == 'preventive' else 2000
            ),
            'prevented_failure': maintenance_type == 'preventive' and np.random.random() < 0.4,
            'maintenance_date': (datetime.now() - timedelta(days=np.random.randint(1, 180))).isoformat(),
            'technician_hours': np.random.uniform(2, 8 if maintenance_type == 'preventive' else 20)
        })
    
    return equipment_data, prediction_data, maintenance_data

def generate_time_series_data():
    """Generate time series data for trend analysis"""
    np.random.seed(42)
    
    # Generate 2 years of daily data
    dates = pd.date_range(start='2022-01-01', end='2023-12-31', freq='D')
    
    # System efficiency with trend and seasonality
    base_efficiency = 82
    trend = np.linspace(0, 8, len(dates))  # Improving trend
    seasonal = 4 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    noise = np.random.normal(0, 2.5, len(dates))
    
    system_efficiency = base_efficiency + trend + seasonal + noise
    system_efficiency = np.clip(system_efficiency, 65, 98)
    
    # Equipment health score
    base_health = 88
    health_trend = np.linspace(0, 5, len(dates))
    health_noise = np.random.normal(0, 3, len(dates))
    
    equipment_health = base_health + health_trend + health_noise
    equipment_health = np.clip(equipment_health, 70, 100)
    
    # Cost savings over time
    base_savings = 50000
    savings_trend = np.linspace(0, 30000, len(dates))
    savings_noise = np.random.normal(0, 5000, len(dates))
    
    monthly_savings = base_savings + savings_trend + savings_noise
    monthly_savings = np.clip(monthly_savings, 20000, 120000)
    
    return pd.DataFrame({
        'system_efficiency': system_efficiency,
        'equipment_health': equipment_health,
        'monthly_savings': monthly_savings
    }, index=dates)

def test_kpi_calculation():
    """Test KPI metrics calculation"""
    print("Testing KPI Calculation...")
    
    analytics = AdvancedAnalytics()
    equipment_data, prediction_data, maintenance_data = generate_test_data()
    
    kpi_metrics = analytics.calculate_kpi_metrics(
        equipment_data, prediction_data, maintenance_data
    )
    
    print(f"✓ Total Equipment Monitored: {kpi_metrics.total_equipment_monitored}")
    print(f"✓ Equipment at Risk: {kpi_metrics.equipment_at_risk}")
    print(f"✓ Predicted Failures Prevented: {kpi_metrics.predicted_failures_prevented}")
    print(f"✓ Maintenance Cost Savings: ${kpi_metrics.maintenance_cost_savings:,.2f}")
    print(f"✓ System Uptime: {kpi_metrics.system_uptime_percentage:.2f}%")
    print(f"✓ False Positive Rate: {kpi_metrics.false_positive_rate:.3f}")
    print(f"✓ Prediction Accuracy: {kpi_metrics.prediction_accuracy:.3f}")
    print(f"✓ ROI: {kpi_metrics.roi_percentage:.2f}%")
    
    # Validate KPI ranges
    assert 0 <= kpi_metrics.total_equipment_monitored <= 1000, "Invalid equipment count"
    assert 0 <= kpi_metrics.false_positive_rate <= 1, "Invalid FPR"
    assert 0 <= kpi_metrics.prediction_accuracy <= 1, "Invalid accuracy"
    assert 0 <= kpi_metrics.system_uptime_percentage <= 100, "Invalid uptime"
    
    print("✓ KPI calculation test passed!\n")
    return kpi_metrics

def test_trend_analysis():
    """Test trend analysis and forecasting"""
    print("Testing Trend Analysis...")
    
    analytics = AdvancedAnalytics()
    time_series_data = generate_time_series_data()
    
    # Test trend analysis for different metrics
    metrics_to_analyze = ['system_efficiency', 'equipment_health', 'monthly_savings']
    trend_results = []
    
    for metric in metrics_to_analyze:
        trend_analysis = analytics.perform_trend_analysis(
            time_series_data, metric, forecast_periods=30
        )
        trend_results.append(trend_analysis)
        
        print(f"✓ {metric}:")
        print(f"  - Trend Direction: {trend_analysis.trend_direction}")
        print(f"  - Trend Strength: {trend_analysis.trend_strength:.3f}")
        print(f"  - R-squared: {trend_analysis.r_squared:.3f}")
        print(f"  - Forecast Points: {len(trend_analysis.forecast_values)}")
        
        # Validate trend analysis
        assert trend_analysis.trend_direction in ['increasing', 'decreasing', 'stable'], "Invalid trend direction"
        assert 0 <= trend_analysis.trend_strength <= 1, "Invalid trend strength"
        assert 0 <= trend_analysis.r_squared <= 1, "Invalid R-squared"
        assert len(trend_analysis.forecast_values) == 30, "Invalid forecast length"
    
    print("✓ Trend analysis test passed!\n")
    return trend_results

def test_cost_analysis():
    """Test cost savings and ROI analysis"""
    print("Testing Cost Analysis...")
    
    analytics = AdvancedAnalytics()
    equipment_data, prediction_data, maintenance_data = generate_test_data()
    
    cost_analysis = analytics.calculate_cost_analysis(maintenance_data, equipment_data)
    
    print(f"✓ Baseline Maintenance Cost: ${cost_analysis.total_maintenance_cost_baseline:,.2f}")
    print(f"✓ Optimized Maintenance Cost: ${cost_analysis.total_maintenance_cost_optimized:,.2f}")
    print(f"✓ Cost Savings: ${cost_analysis.cost_savings:,.2f}")
    print(f"✓ Cost Savings Percentage: {cost_analysis.cost_savings_percentage:.2f}%")
    print(f"✓ ROI Percentage: {cost_analysis.roi_percentage:.2f}%")
    print(f"✓ Payback Period: {cost_analysis.payback_period_months:.1f} months")
    
    print("✓ Cost Breakdown:")
    for category, amount in cost_analysis.breakdown_by_category.items():
        print(f"  - {category}: ${amount:,.2f}")
    
    # Validate cost analysis
    assert cost_analysis.total_maintenance_cost_baseline >= 0, "Invalid baseline cost"
    assert cost_analysis.total_maintenance_cost_optimized >= 0, "Invalid optimized cost"
    assert cost_analysis.implementation_cost > 0, "Invalid implementation cost"
    assert cost_analysis.payback_period_months > 0, "Invalid payback period"
    
    print("✓ Cost analysis test passed!\n")
    return cost_analysis

def test_equipment_comparison():
    """Test equipment type comparison analysis"""
    print("Testing Equipment Comparison...")
    
    analytics = AdvancedAnalytics()
    equipment_data, _, _ = generate_test_data()
    
    equipment_comparisons = analytics.perform_equipment_comparison(equipment_data)
    
    print("✓ Equipment Type Rankings:")
    for i, comparison in enumerate(equipment_comparisons):
        print(f"  {comparison.performance_ranking}. {comparison.equipment_type.title()}")
        print(f"     - Total Units: {comparison.total_units}")
        print(f"     - Average RUL: {comparison.average_rul:.1f} cycles")
        print(f"     - Failure Rate: {comparison.failure_rate:.2%}")
        print(f"     - Reliability Score: {comparison.reliability_score:.3f}")
        print(f"     - Cost per Unit: ${comparison.cost_per_unit:,.0f}")
    
    # Validate equipment comparison
    assert len(equipment_comparisons) > 0, "No equipment comparisons generated"
    
    # Check ranking consistency
    for i in range(len(equipment_comparisons) - 1):
        current_score = equipment_comparisons[i].reliability_score
        next_score = equipment_comparisons[i + 1].reliability_score
        assert current_score >= next_score, "Equipment ranking is inconsistent"
    
    print("✓ Equipment comparison test passed!\n")
    return equipment_comparisons

def test_dashboard_generation():
    """Test executive dashboard generation"""
    print("Testing Dashboard Generation...")
    
    analytics = AdvancedAnalytics()
    equipment_data, prediction_data, maintenance_data = generate_test_data()
    
    # Calculate all analytics
    kpi_metrics = analytics.calculate_kpi_metrics(
        equipment_data, prediction_data, maintenance_data
    )
    
    time_series_data = generate_time_series_data()
    trend_analysis = analytics.perform_trend_analysis(
        time_series_data, 'system_efficiency'
    )
    
    cost_analysis = analytics.calculate_cost_analysis(
        maintenance_data, equipment_data
    )
    
    # Generate dashboard HTML
    dashboard_html = analytics.generate_executive_dashboard(
        kpi_metrics, [trend_analysis], cost_analysis
    )
    
    # Validate dashboard content
    assert len(dashboard_html) > 1000, "Dashboard HTML too short"
    assert "Executive Dashboard" in dashboard_html, "Missing dashboard title"
    assert "KPI" in dashboard_html or "Equipment Monitored" in dashboard_html, "Missing KPI content"
    assert "Cost Savings" in dashboard_html, "Missing cost savings content"
    
    # Save dashboard to file
    output_dir = Path("output/analytics_reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dashboard_path = output_dir / f"executive_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    with open(dashboard_path, 'w') as f:
        f.write(dashboard_html)
    
    print(f"✓ Executive dashboard generated: {dashboard_path}")
    print("✓ Dashboard generation test passed!\n")
    
    return dashboard_path

def test_analytics_report_saving():
    """Test comprehensive analytics report saving"""
    print("Testing Analytics Report Saving...")
    
    analytics = AdvancedAnalytics()
    equipment_data, prediction_data, maintenance_data = generate_test_data()
    
    # Calculate all analytics
    kpi_metrics = analytics.calculate_kpi_metrics(
        equipment_data, prediction_data, maintenance_data
    )
    
    time_series_data = generate_time_series_data()
    trend_analyses = []
    for metric in ['system_efficiency', 'equipment_health']:
        trend_analysis = analytics.perform_trend_analysis(time_series_data, metric)
        trend_analyses.append(trend_analysis)
    
    cost_analysis = analytics.calculate_cost_analysis(maintenance_data, equipment_data)
    equipment_comparisons = analytics.perform_equipment_comparison(equipment_data)
    
    # Save comprehensive report
    report_path = analytics.save_analytics_report(
        kpi_metrics, trend_analyses, cost_analysis, equipment_comparisons
    )
    
    # Validate saved report
    assert os.path.exists(report_path), "Report file not created"
    
    with open(report_path, 'r') as f:
        report_data = json.load(f)
    
    assert 'generated_at' in report_data, "Missing generation timestamp"
    assert 'kpi_metrics' in report_data, "Missing KPI metrics"
    assert 'trend_analyses' in report_data, "Missing trend analyses"
    assert 'cost_analysis' in report_data, "Missing cost analysis"
    assert 'equipment_comparisons' in report_data, "Missing equipment comparisons"
    
    print(f"✓ Analytics report saved: {report_path}")
    print("✓ Report saving test passed!\n")
    
    return report_path

def test_interactive_dashboard():
    """Test interactive dashboard creation (without running server)"""
    print("Testing Interactive Dashboard Setup...")
    
    analytics = AdvancedAnalytics()
    dashboard = ExecutiveDashboard(analytics)
    
    # Validate dashboard components
    assert dashboard.app is not None, "Dashboard app not initialized"
    assert dashboard.analytics is not None, "Analytics engine not set"
    
    # Test static report generation
    output_dir = Path("output/analytics_reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    static_report_path = output_dir / f"static_executive_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    
    generated_path = dashboard.generate_static_report(str(static_report_path))
    
    assert os.path.exists(generated_path), "Static report not generated"
    
    with open(generated_path, 'r') as f:
        content = f.read()
        assert len(content) > 1000, "Static report content too short"
        assert "Executive Dashboard" in content, "Missing dashboard title in static report"
    
    print(f"✓ Static dashboard report generated: {generated_path}")
    print("✓ Interactive dashboard test passed!\n")
    
    return generated_path

def run_comprehensive_test():
    """Run comprehensive test of all advanced analytics features"""
    print("=" * 60)
    print("ADVANCED ANALYTICS AND REPORTING SYSTEM TEST")
    print("=" * 60)
    print()
    
    try:
        # Test individual components
        kpi_metrics = test_kpi_calculation()
        trend_results = test_trend_analysis()
        cost_analysis = test_cost_analysis()
        equipment_comparisons = test_equipment_comparison()
        
        # Test dashboard generation
        dashboard_path = test_dashboard_generation()
        report_path = test_analytics_report_saving()
        static_dashboard_path = test_interactive_dashboard()
        
        # Summary
        print("=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print("✓ All advanced analytics tests passed successfully!")
        print()
        print("Generated Files:")
        print(f"  - Executive Dashboard: {dashboard_path}")
        print(f"  - Analytics Report: {report_path}")
        print(f"  - Static Dashboard: {static_dashboard_path}")
        print()
        print("Key Metrics Achieved:")
        print(f"  - Equipment Monitored: {kpi_metrics.total_equipment_monitored}")
        print(f"  - Cost Savings: ${kpi_metrics.maintenance_cost_savings:,.0f}")
        print(f"  - ROI: {kpi_metrics.roi_percentage:.1f}%")
        print(f"  - System Uptime: {kpi_metrics.system_uptime_percentage:.1f}%")
        print(f"  - Prediction Accuracy: {kpi_metrics.prediction_accuracy:.1%}")
        print()
        print("Advanced analytics system is ready for production deployment!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)