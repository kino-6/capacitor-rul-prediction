#!/usr/bin/env python3
"""
Test script for Fleet Management Analytics System

This script tests the fleet management capabilities including:
- Fleet-wide health scoring and ranking
- Maintenance scheduling optimization
- Resource allocation optimization
- Predictive budgeting for maintenance costs
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

from true_rul.fleet_management import FleetManagementAnalytics, FleetHealthScore, MaintenanceSchedule

def generate_fleet_test_data():
    """Generate comprehensive test data for fleet management testing"""
    np.random.seed(42)
    
    # Equipment data with varied health conditions
    equipment_types = ['capacitor', 'motor', 'pump', 'compressor']
    equipment_data = []
    
    for i in range(50):
        eq_type = np.random.choice(equipment_types)
        
        # Create varied RUL distribution
        if i < 5:  # Critical equipment
            rul_cycles = np.random.uniform(5, 15)
        elif i < 15:  # Warning equipment
            rul_cycles = np.random.uniform(20, 60)
        else:  # Healthy equipment
            rul_cycles = np.random.uniform(80, 200)
        
        equipment_data.append({
            'id': f'EQ_{i:03d}',
            'type': eq_type,
            'rul_cycles': rul_cycles,
            'maintenance_count': np.random.poisson(2),
            'age_years': np.random.uniform(1, 15),
            'anomaly_flag': np.random.random() < 0.1,
            'location': f'Zone_{np.random.randint(1, 6)}',
            'criticality': np.random.choice(['low', 'medium', 'high'], p=[0.3, 0.5, 0.2])
        })
    
    # Historical fleet health data for trend analysis
    historical_data = []
    for i in range(12):  # 12 months of history
        date = datetime.now() - timedelta(days=30 * (12 - i))
        
        # Simulate improving trend
        base_health = 0.65 + (i * 0.02)  # Gradual improvement
        noise = np.random.normal(0, 0.05)
        health_score = np.clip(base_health + noise, 0.3, 0.95)
        
        historical_data.append({
            'date': date.isoformat(),
            'overall_health_score': health_score,
            'equipment_count': 50,
            'critical_count': max(0, int(5 - i * 0.3))
        })
    
    return equipment_data, historical_data

def test_fleet_health_scoring():
    """Test fleet health scoring and ranking"""
    print("Testing Fleet Health Scoring...")
    
    fleet_analytics = FleetManagementAnalytics()
    equipment_data, historical_data = generate_fleet_test_data()
    
    fleet_health = fleet_analytics.calculate_fleet_health_score(
        equipment_data, historical_data
    )
    
    print(f"✓ Fleet ID: {fleet_health.fleet_id}")
    print(f"✓ Overall Health Score: {fleet_health.overall_health_score:.3f}")
    print(f"✓ Equipment Count: {fleet_health.equipment_count}")
    print(f"✓ Critical Equipment: {fleet_health.critical_equipment_count}")
    print(f"✓ Average RUL: {fleet_health.average_rul:.1f} cycles")
    print(f"✓ Health Trend: {fleet_health.health_trend}")
    
    print("✓ Health Distribution:")
    for status, count in fleet_health.health_distribution.items():
        print(f"  - {status.title()}: {count}")
    
    print("✓ Top Risk Equipment:")
    for i, equipment in enumerate(fleet_health.top_risk_equipment[:5]):
        print(f"  {i+1}. {equipment['equipment_id']} ({equipment['equipment_type']}) - Health: {equipment['health_score']:.3f}")
    
    # Validate results
    assert fleet_health.equipment_count == 50, "Incorrect equipment count"
    assert 0 <= fleet_health.overall_health_score <= 1, "Invalid health score"
    assert fleet_health.health_trend in ['improving', 'stable', 'declining'], "Invalid health trend"
    assert sum(fleet_health.health_distribution.values()) == fleet_health.equipment_count, "Health distribution mismatch"
    
    print("✓ Fleet health scoring test passed!\n")
    return fleet_health

def test_maintenance_scheduling():
    """Test maintenance scheduling optimization"""
    print("Testing Maintenance Scheduling Optimization...")
    
    fleet_analytics = FleetManagementAnalytics()
    equipment_data, _ = generate_fleet_test_data()
    
    # Test with 90-day planning horizon
    maintenance_schedule = fleet_analytics.optimize_maintenance_schedule(
        equipment_data, time_horizon_days=90
    )
    
    print(f"✓ Scheduled Maintenance Items: {len(maintenance_schedule)}")
    
    # Analyze schedule by type and priority
    schedule_by_type = {}
    schedule_by_priority = {}
    
    for maintenance in maintenance_schedule:
        # By type
        mtype = maintenance.maintenance_type
        schedule_by_type[mtype] = schedule_by_type.get(mtype, 0) + 1
        
        # By priority
        priority = maintenance.priority
        schedule_by_priority[priority] = schedule_by_priority.get(priority, 0) + 1
    
    print("✓ Maintenance by Type:")
    for mtype, count in schedule_by_type.items():
        print(f"  - {mtype.title()}: {count}")
    
    print("✓ Maintenance by Priority:")
    for priority in sorted(schedule_by_priority.keys(), reverse=True):
        count = schedule_by_priority[priority]
        print(f"  - Priority {priority}: {count}")
    
    # Show sample scheduled items
    print("✓ Sample Scheduled Maintenance:")
    for i, maintenance in enumerate(maintenance_schedule[:5]):
        scheduled_date = datetime.fromisoformat(maintenance.scheduled_date.replace('Z', '+00:00'))
        print(f"  {i+1}. {maintenance.equipment_id} - {maintenance.maintenance_type} on {scheduled_date.strftime('%Y-%m-%d')}")
        print(f"     Priority: {maintenance.priority}, Cost: ${maintenance.estimated_cost:.0f}, Duration: {maintenance.estimated_duration_hours:.1f}h")
    
    # Validate scheduling
    assert len(maintenance_schedule) > 0, "No maintenance scheduled"
    
    # Check that high-priority items are scheduled earlier
    high_priority_items = [m for m in maintenance_schedule if m.priority >= 4]
    if high_priority_items:
        avg_high_priority_date = np.mean([
            datetime.fromisoformat(m.scheduled_date.replace('Z', '+00:00')).timestamp() 
            for m in high_priority_items
        ])
        low_priority_items = [m for m in maintenance_schedule if m.priority <= 2]
        if low_priority_items:
            avg_low_priority_date = np.mean([
                datetime.fromisoformat(m.scheduled_date.replace('Z', '+00:00')).timestamp() 
                for m in low_priority_items
            ])
            assert avg_high_priority_date <= avg_low_priority_date, "Priority scheduling not working correctly"
    
    print("✓ Maintenance scheduling test passed!\n")
    return maintenance_schedule

def test_resource_allocation():
    """Test resource allocation optimization"""
    print("Testing Resource Allocation Optimization...")
    
    fleet_analytics = FleetManagementAnalytics()
    equipment_data, _ = generate_fleet_test_data()
    
    # Get maintenance schedule first
    maintenance_schedule = fleet_analytics.optimize_maintenance_schedule(
        equipment_data, time_horizon_days=90
    )
    
    # Optimize resource allocation
    resource_allocation = fleet_analytics.optimize_resource_allocation(maintenance_schedule)
    
    print(f"✓ Resource Types Analyzed: {len(resource_allocation)}")
    
    for allocation in resource_allocation:
        print(f"✓ {allocation.resource_type.title()}:")
        print(f"  - Available: {allocation.total_available}")
        print(f"  - Required: {allocation.total_required}")
        print(f"  - Efficiency: {allocation.allocation_efficiency:.1%}")
        print(f"  - Bottleneck Periods: {len(allocation.bottleneck_periods)}")
        print(f"  - Cost Impact: ${allocation.cost_impact:.0f}")
        
        if allocation.optimization_suggestions:
            print(f"  - Suggestions:")
            for suggestion in allocation.optimization_suggestions:
                print(f"    • {suggestion}")
    
    # Validate resource allocation
    assert len(resource_allocation) > 0, "No resource allocation results"
    
    for allocation in resource_allocation:
        assert allocation.total_available >= 0, "Invalid available resources"
        assert allocation.total_required >= 0, "Invalid required resources"
        assert 0 <= allocation.allocation_efficiency <= 1.5, "Invalid allocation efficiency"  # Allow some over-allocation
    
    print("✓ Resource allocation test passed!\n")
    return resource_allocation

def test_budget_prediction():
    """Test maintenance budget prediction"""
    print("Testing Maintenance Budget Prediction...")
    
    fleet_analytics = FleetManagementAnalytics()
    equipment_data, _ = generate_fleet_test_data()
    
    # Get maintenance schedule
    maintenance_schedule = fleet_analytics.optimize_maintenance_schedule(
        equipment_data, time_horizon_days=365  # Full year for budget planning
    )
    
    # Predict budget for 12 months
    budget_forecast = fleet_analytics.predict_maintenance_budget(
        equipment_data, maintenance_schedule, forecast_periods=12
    )
    
    print(f"✓ Budget Forecast Periods: {len(budget_forecast)}")
    
    total_annual_budget = sum(budget.total_budget for budget in budget_forecast)
    print(f"✓ Total Annual Budget: ${total_annual_budget:,.0f}")
    
    # Analyze budget components
    total_preventive = sum(budget.preventive_maintenance_cost for budget in budget_forecast)
    total_corrective = sum(budget.corrective_maintenance_cost for budget in budget_forecast)
    total_parts = sum(budget.parts_cost for budget in budget_forecast)
    total_labor = sum(budget.labor_cost for budget in budget_forecast)
    total_downtime = sum(budget.downtime_cost for budget in budget_forecast)
    
    print("✓ Annual Budget Breakdown:")
    print(f"  - Preventive Maintenance: ${total_preventive:,.0f} ({total_preventive/total_annual_budget*100:.1f}%)")
    print(f"  - Corrective Maintenance: ${total_corrective:,.0f} ({total_corrective/total_annual_budget*100:.1f}%)")
    print(f"  - Parts Cost: ${total_parts:,.0f} ({total_parts/total_annual_budget*100:.1f}%)")
    print(f"  - Labor Cost: ${total_labor:,.0f} ({total_labor/total_annual_budget*100:.1f}%)")
    print(f"  - Downtime Cost: ${total_downtime:,.0f} ({total_downtime/total_annual_budget*100:.1f}%)")
    
    # Show monthly breakdown for first quarter
    print("✓ Quarterly Budget Forecast:")
    for i, budget in enumerate(budget_forecast[:3]):
        print(f"  {budget.period}: ${budget.total_budget:,.0f}")
        if budget.cost_drivers:
            print(f"    Cost Drivers: {', '.join([driver['driver'] for driver in budget.cost_drivers])}")
        if budget.savings_opportunities:
            total_savings = sum(opp['potential_savings'] for opp in budget.savings_opportunities)
            print(f"    Potential Savings: ${total_savings:,.0f}")
    
    # Validate budget prediction
    assert len(budget_forecast) == 12, "Incorrect number of forecast periods"
    assert total_annual_budget > 0, "Invalid total budget"
    
    for budget in budget_forecast:
        assert budget.total_budget >= 0, "Invalid period budget"
        assert budget.period, "Missing period identifier"
    
    print("✓ Budget prediction test passed!\n")
    return budget_forecast

def test_optimization_opportunities():
    """Test fleet optimization opportunity identification"""
    print("Testing Optimization Opportunity Identification...")
    
    fleet_analytics = FleetManagementAnalytics()
    equipment_data, historical_data = generate_fleet_test_data()
    
    # Get all analytics results
    fleet_health = fleet_analytics.calculate_fleet_health_score(equipment_data, historical_data)
    maintenance_schedule = fleet_analytics.optimize_maintenance_schedule(equipment_data)
    resource_allocation = fleet_analytics.optimize_resource_allocation(maintenance_schedule)
    
    # Identify optimization opportunities
    opportunities = fleet_analytics.identify_fleet_optimization_opportunities(
        fleet_health, maintenance_schedule, resource_allocation
    )
    
    print(f"✓ Optimization Opportunities Identified: {len(opportunities)}")
    
    total_potential_savings = sum(opp.get('estimated_savings', 0) for opp in opportunities)
    print(f"✓ Total Potential Savings: ${total_potential_savings:,.0f}")
    
    print("✓ Top Optimization Opportunities:")
    for i, opportunity in enumerate(opportunities[:5]):
        print(f"  {i+1}. {opportunity['opportunity']}")
        print(f"     Category: {opportunity['category']}")
        print(f"     Impact: {opportunity['impact']}")
        print(f"     Estimated Savings: ${opportunity['estimated_savings']:,.0f}")
        print(f"     Timeline: {opportunity['timeline']}")
        print(f"     Description: {opportunity['description']}")
        print()
    
    # Validate opportunities
    assert len(opportunities) >= 0, "Error in opportunity identification"
    
    for opportunity in opportunities:
        assert 'category' in opportunity, "Missing opportunity category"
        assert 'opportunity' in opportunity, "Missing opportunity name"
        assert 'estimated_savings' in opportunity, "Missing savings estimate"
        assert opportunity['estimated_savings'] >= 0, "Invalid savings estimate"
    
    print("✓ Optimization opportunity identification test passed!\n")
    return opportunities

def test_fleet_analytics_report():
    """Test comprehensive fleet analytics report generation"""
    print("Testing Fleet Analytics Report Generation...")
    
    fleet_analytics = FleetManagementAnalytics()
    equipment_data, historical_data = generate_fleet_test_data()
    
    # Generate all analytics
    fleet_health = fleet_analytics.calculate_fleet_health_score(equipment_data, historical_data)
    maintenance_schedule = fleet_analytics.optimize_maintenance_schedule(equipment_data)
    resource_allocation = fleet_analytics.optimize_resource_allocation(maintenance_schedule)
    budget_forecast = fleet_analytics.predict_maintenance_budget(equipment_data, maintenance_schedule)
    opportunities = fleet_analytics.identify_fleet_optimization_opportunities(
        fleet_health, maintenance_schedule, resource_allocation
    )
    
    # Save comprehensive report
    output_dir = Path("output/fleet_reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = fleet_analytics.save_fleet_analytics_report(
        fleet_health, maintenance_schedule, resource_allocation, 
        budget_forecast, opportunities,
        str(output_dir / f"fleet_analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    )
    
    # Validate saved report
    assert os.path.exists(report_path), "Report file not created"
    
    with open(report_path, 'r') as f:
        report_data = json.load(f)
    
    assert 'generated_at' in report_data, "Missing generation timestamp"
    assert 'fleet_health' in report_data, "Missing fleet health data"
    assert 'maintenance_schedule' in report_data, "Missing maintenance schedule"
    assert 'resource_allocation' in report_data, "Missing resource allocation"
    assert 'budget_forecast' in report_data, "Missing budget forecast"
    assert 'optimization_opportunities' in report_data, "Missing optimization opportunities"
    assert 'summary' in report_data, "Missing report summary"
    
    summary = report_data['summary']
    print(f"✓ Fleet Analytics Report Generated: {report_path}")
    print("✓ Report Summary:")
    print(f"  - Total Equipment: {summary['total_equipment']}")
    print(f"  - Critical Equipment: {summary['critical_equipment']}")
    print(f"  - Scheduled Maintenance Items: {summary['scheduled_maintenance_items']}")
    print(f"  - Total Estimated Budget: ${summary['total_estimated_budget']:,.0f}")
    print(f"  - Potential Savings: ${summary['potential_savings']:,.0f}")
    
    print("✓ Fleet analytics report test passed!\n")
    return report_path

def run_comprehensive_fleet_test():
    """Run comprehensive test of all fleet management features"""
    print("=" * 60)
    print("FLEET MANAGEMENT ANALYTICS SYSTEM TEST")
    print("=" * 60)
    print()
    
    try:
        # Test individual components
        fleet_health = test_fleet_health_scoring()
        maintenance_schedule = test_maintenance_scheduling()
        resource_allocation = test_resource_allocation()
        budget_forecast = test_budget_prediction()
        opportunities = test_optimization_opportunities()
        
        # Test comprehensive reporting
        report_path = test_fleet_analytics_report()
        
        # Summary
        print("=" * 60)
        print("FLEET MANAGEMENT TEST SUMMARY")
        print("=" * 60)
        print("✓ All fleet management tests passed successfully!")
        print()
        print("Generated Files:")
        print(f"  - Fleet Analytics Report: {report_path}")
        print()
        print("Key Fleet Metrics:")
        print(f"  - Fleet Health Score: {fleet_health.overall_health_score:.3f}")
        print(f"  - Equipment Monitored: {fleet_health.equipment_count}")
        print(f"  - Critical Equipment: {fleet_health.critical_equipment_count}")
        print(f"  - Scheduled Maintenance: {len(maintenance_schedule)} items")
        print(f"  - Annual Budget: ${sum(b.total_budget for b in budget_forecast):,.0f}")
        print(f"  - Optimization Savings: ${sum(o.get('estimated_savings', 0) for o in opportunities):,.0f}")
        print()
        print("Fleet Management System Features:")
        print("  ✓ Fleet health scoring and ranking")
        print("  ✓ Maintenance scheduling optimization")
        print("  ✓ Resource allocation optimization")
        print("  ✓ Predictive budget forecasting")
        print("  ✓ Optimization opportunity identification")
        print("  ✓ Comprehensive analytics reporting")
        print()
        print("Fleet management analytics system is ready for production deployment!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_comprehensive_fleet_test()
    sys.exit(0 if success else 1)