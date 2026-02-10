"""
Fleet Management Analytics System

This module provides predictive analytics for fleet-wide management including:
- Fleet-wide health scoring and ranking
- Optimization algorithms for maintenance scheduling
- Resource allocation optimization
- Predictive budgeting for maintenance costs
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
from scipy.optimize import minimize, linprog
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

logger = logging.getLogger(__name__)

@dataclass
class FleetHealthScore:
    """Fleet-wide health scoring results"""
    fleet_id: str
    overall_health_score: float
    equipment_count: int
    critical_equipment_count: int
    average_rul: float
    health_distribution: Dict[str, int]  # healthy, warning, critical counts
    top_risk_equipment: List[Dict[str, Any]]
    health_trend: str  # improving, stable, declining
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class MaintenanceSchedule:
    """Optimized maintenance schedule"""
    equipment_id: str
    equipment_type: str
    scheduled_date: str
    maintenance_type: str  # preventive, corrective, inspection
    priority: int  # 1-5 scale
    estimated_duration_hours: float
    estimated_cost: float
    required_resources: List[str]
    rul_at_schedule: float
    risk_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ResourceAllocation:
    """Resource allocation optimization results"""
    resource_type: str  # technician, parts, tools
    total_available: int
    total_required: int
    allocation_efficiency: float
    bottleneck_periods: List[str]
    optimization_suggestions: List[str]
    cost_impact: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class MaintenanceBudget:
    """Predictive maintenance budget forecast"""
    period: str  # monthly, quarterly, annual
    total_budget: float
    preventive_maintenance_cost: float
    corrective_maintenance_cost: float
    parts_cost: float
    labor_cost: float
    downtime_cost: float
    budget_variance: float  # vs previous period
    cost_drivers: List[Dict[str, Any]]
    savings_opportunities: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class FleetManagementAnalytics:
    """Predictive analytics for fleet management"""
    
    def __init__(self, cost_parameters: Optional[Dict[str, float]] = None):
        """
        Initialize fleet management analytics
        
        Args:
            cost_parameters: Dictionary of cost parameters for calculations
        """
        self.cost_parameters = cost_parameters or {
            'preventive_maintenance_base_cost': 1200.0,
            'corrective_maintenance_base_cost': 4500.0,
            'inspection_cost': 300.0,
            'technician_hourly_rate': 85.0,
            'downtime_cost_per_hour': 1800.0,
            'parts_markup': 1.3,
            'emergency_multiplier': 2.0,
            'weekend_multiplier': 1.5
        }
        
        # Resource constraints
        self.resource_constraints = {
            'technicians': 8,
            'maintenance_bays': 4,
            'specialized_tools': 2,
            'working_hours_per_day': 8,
            'working_days_per_week': 5
        }
        
        # Equipment type parameters
        self.equipment_parameters = {
            'capacitor': {
                'maintenance_duration': 2.5,
                'parts_cost_avg': 450.0,
                'criticality_weight': 0.8
            },
            'motor': {
                'maintenance_duration': 4.0,
                'parts_cost_avg': 850.0,
                'criticality_weight': 0.9
            },
            'pump': {
                'maintenance_duration': 3.5,
                'parts_cost_avg': 650.0,
                'criticality_weight': 0.85
            },
            'compressor': {
                'maintenance_duration': 5.0,
                'parts_cost_avg': 1200.0,
                'criticality_weight': 0.95
            }
        }
    
    def calculate_fleet_health_score(self, 
                                   equipment_data: List[Dict[str, Any]],
                                   historical_data: Optional[List[Dict[str, Any]]] = None) -> FleetHealthScore:
        """
        Calculate comprehensive fleet health score and ranking
        
        Args:
            equipment_data: List of equipment status records
            historical_data: Optional historical health data for trend analysis
            
        Returns:
            FleetHealthScore object with comprehensive fleet health metrics
        """
        try:
            if not equipment_data:
                logger.warning("No equipment data provided for fleet health calculation")
                return self._default_fleet_health_score()
            
            # Calculate individual equipment health scores
            equipment_scores = []
            for equipment in equipment_data:
                health_score = self._calculate_equipment_health_score(equipment)
                equipment_scores.append({
                    'equipment_id': equipment.get('id', 'unknown'),
                    'equipment_type': equipment.get('type', 'unknown'),
                    'health_score': health_score,
                    'rul_cycles': equipment.get('rul_cycles', 0),
                    'risk_factors': self._identify_risk_factors(equipment)
                })
            
            # Calculate fleet-wide metrics
            health_scores = [eq['health_score'] for eq in equipment_scores]
            overall_health_score = np.mean(health_scores)
            
            equipment_count = len(equipment_data)
            critical_equipment_count = sum(1 for eq in equipment_scores if eq['health_score'] < 0.3)
            average_rul = np.mean([eq.get('rul_cycles', 0) for eq in equipment_data])
            
            # Health distribution
            health_distribution = {
                'healthy': sum(1 for score in health_scores if score >= 0.7),
                'warning': sum(1 for score in health_scores if 0.3 <= score < 0.7),
                'critical': sum(1 for score in health_scores if score < 0.3)
            }
            
            # Identify top risk equipment
            equipment_scores.sort(key=lambda x: x['health_score'])
            top_risk_equipment = equipment_scores[:min(10, len(equipment_scores))]
            
            # Determine health trend
            health_trend = self._calculate_health_trend(historical_data) if historical_data else 'stable'
            
            return FleetHealthScore(
                fleet_id="main_fleet",
                overall_health_score=overall_health_score,
                equipment_count=equipment_count,
                critical_equipment_count=critical_equipment_count,
                average_rul=average_rul,
                health_distribution=health_distribution,
                top_risk_equipment=top_risk_equipment,
                health_trend=health_trend
            )
            
        except Exception as e:
            logger.error(f"Error calculating fleet health score: {e}")
            return self._default_fleet_health_score()
    
    def optimize_maintenance_schedule(self, 
                                    equipment_data: List[Dict[str, Any]],
                                    time_horizon_days: int = 90,
                                    resource_constraints: Optional[Dict[str, Any]] = None) -> List[MaintenanceSchedule]:
        """
        Optimize maintenance scheduling using constraint optimization
        
        Args:
            equipment_data: List of equipment with RUL predictions
            time_horizon_days: Planning horizon in days
            resource_constraints: Optional resource constraint overrides
            
        Returns:
            List of optimized maintenance schedules
        """
        try:
            if not equipment_data:
                logger.warning("No equipment data provided for maintenance scheduling")
                return []
            
            # Use provided constraints or defaults
            constraints = resource_constraints or self.resource_constraints
            
            # Calculate maintenance priorities and requirements
            maintenance_requirements = []
            for equipment in equipment_data:
                req = self._calculate_maintenance_requirement(equipment, time_horizon_days)
                if req:
                    maintenance_requirements.append(req)
            
            # Sort by priority (risk score * criticality)
            maintenance_requirements.sort(key=lambda x: x['priority_score'], reverse=True)
            
            # Optimize scheduling with resource constraints
            optimized_schedule = self._optimize_schedule_with_constraints(
                maintenance_requirements, time_horizon_days, constraints
            )
            
            return optimized_schedule
            
        except Exception as e:
            logger.error(f"Error optimizing maintenance schedule: {e}")
            return []
    
    def optimize_resource_allocation(self, 
                                   maintenance_schedule: List[MaintenanceSchedule],
                                   resource_constraints: Optional[Dict[str, Any]] = None) -> List[ResourceAllocation]:
        """
        Optimize resource allocation for maintenance activities
        
        Args:
            maintenance_schedule: List of scheduled maintenance activities
            resource_constraints: Optional resource constraint overrides
            
        Returns:
            List of resource allocation optimization results
        """
        try:
            if not maintenance_schedule:
                logger.warning("No maintenance schedule provided for resource allocation")
                return []
            
            constraints = resource_constraints or self.resource_constraints
            
            # Analyze resource requirements by type
            resource_types = ['technicians', 'maintenance_bays', 'specialized_tools']
            allocations = []
            
            for resource_type in resource_types:
                allocation = self._optimize_single_resource_allocation(
                    maintenance_schedule, resource_type, constraints
                )
                allocations.append(allocation)
            
            return allocations
            
        except Exception as e:
            logger.error(f"Error optimizing resource allocation: {e}")
            return []
    
    def predict_maintenance_budget(self, 
                                 equipment_data: List[Dict[str, Any]],
                                 maintenance_schedule: List[MaintenanceSchedule],
                                 forecast_periods: int = 12) -> List[MaintenanceBudget]:
        """
        Predict maintenance budget for future periods
        
        Args:
            equipment_data: Current equipment status
            maintenance_schedule: Planned maintenance activities
            forecast_periods: Number of months to forecast
            
        Returns:
            List of maintenance budget forecasts by period
        """
        try:
            if not equipment_data or not maintenance_schedule:
                logger.warning("Insufficient data for budget prediction")
                return []
            
            budget_forecasts = []
            
            for period in range(1, forecast_periods + 1):
                # Calculate period start and end dates
                period_start = datetime.now() + timedelta(days=(period - 1) * 30)
                period_end = period_start + timedelta(days=30)
                
                # Filter maintenance activities for this period
                period_maintenance = [
                    m for m in maintenance_schedule
                    if period_start <= datetime.fromisoformat(m.scheduled_date.replace('Z', '+00:00')) < period_end
                ]
                
                # Calculate budget components
                budget = self._calculate_period_budget(period_maintenance, equipment_data)
                budget.period = f"{period_start.strftime('%Y-%m')}"
                
                budget_forecasts.append(budget)
            
            return budget_forecasts
            
        except Exception as e:
            logger.error(f"Error predicting maintenance budget: {e}")
            return []
    
    def identify_fleet_optimization_opportunities(self, 
                                                fleet_health: FleetHealthScore,
                                                maintenance_schedule: List[MaintenanceSchedule],
                                                resource_allocation: List[ResourceAllocation]) -> List[Dict[str, Any]]:
        """
        Identify optimization opportunities across the fleet
        
        Args:
            fleet_health: Fleet health scoring results
            maintenance_schedule: Current maintenance schedule
            resource_allocation: Resource allocation results
            
        Returns:
            List of optimization opportunities with impact estimates
        """
        try:
            opportunities = []
            
            # Fleet health optimization opportunities
            if fleet_health.critical_equipment_count > 0:
                opportunities.append({
                    'category': 'fleet_health',
                    'opportunity': 'Critical Equipment Intervention',
                    'description': f'Immediate attention needed for {fleet_health.critical_equipment_count} critical equipment',
                    'impact': 'High',
                    'estimated_savings': fleet_health.critical_equipment_count * 15000,
                    'implementation_effort': 'Medium',
                    'timeline': '1-2 weeks'
                })
            
            # Maintenance scheduling optimization
            high_priority_count = sum(1 for m in maintenance_schedule if m.priority >= 4)
            if high_priority_count > len(maintenance_schedule) * 0.3:
                opportunities.append({
                    'category': 'maintenance_scheduling',
                    'opportunity': 'Preventive Maintenance Optimization',
                    'description': 'Rebalance maintenance schedule to reduce high-priority reactive maintenance',
                    'impact': 'Medium',
                    'estimated_savings': high_priority_count * 3000,
                    'implementation_effort': 'Low',
                    'timeline': '2-4 weeks'
                })
            
            # Resource allocation optimization
            for allocation in resource_allocation:
                if allocation.allocation_efficiency < 0.8:
                    opportunities.append({
                        'category': 'resource_allocation',
                        'opportunity': f'{allocation.resource_type.title()} Efficiency Improvement',
                        'description': f'Optimize {allocation.resource_type} allocation to improve efficiency from {allocation.allocation_efficiency:.1%} to 85%+',
                        'impact': 'Medium',
                        'estimated_savings': abs(allocation.cost_impact) * 0.5,
                        'implementation_effort': 'Medium',
                        'timeline': '4-6 weeks'
                    })
            
            # Fleet-wide patterns
            if fleet_health.overall_health_score < 0.7:
                opportunities.append({
                    'category': 'fleet_strategy',
                    'opportunity': 'Fleet Modernization Assessment',
                    'description': 'Consider equipment replacement or major overhaul for aging fleet',
                    'impact': 'High',
                    'estimated_savings': fleet_health.equipment_count * 8000,
                    'implementation_effort': 'High',
                    'timeline': '3-6 months'
                })
            
            # Sort by estimated savings (descending)
            opportunities.sort(key=lambda x: x.get('estimated_savings', 0), reverse=True)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error identifying optimization opportunities: {e}")
            return []
    
    def _calculate_equipment_health_score(self, equipment: Dict[str, Any]) -> float:
        """Calculate health score for individual equipment"""
        try:
            rul_cycles = equipment.get('rul_cycles', 0)
            equipment_type = equipment.get('type', 'unknown')
            
            # Base health score from RUL
            if rul_cycles <= 0:
                rul_score = 0.0
            elif rul_cycles < 20:
                rul_score = 0.2
            elif rul_cycles < 50:
                rul_score = 0.5
            elif rul_cycles < 100:
                rul_score = 0.7
            else:
                rul_score = 0.9
            
            # Adjust for maintenance history
            maintenance_count = equipment.get('maintenance_count', 0)
            if maintenance_count > 5:
                maintenance_penalty = 0.1
            elif maintenance_count > 3:
                maintenance_penalty = 0.05
            else:
                maintenance_penalty = 0.0
            
            # Adjust for equipment age/usage
            age_factor = equipment.get('age_years', 1)
            age_penalty = min(age_factor * 0.02, 0.2)  # Max 20% penalty
            
            # Calculate final health score
            health_score = max(0.0, rul_score - maintenance_penalty - age_penalty)
            
            return health_score
            
        except Exception as e:
            logger.error(f"Error calculating equipment health score: {e}")
            return 0.5  # Default moderate health score
    
    def _identify_risk_factors(self, equipment: Dict[str, Any]) -> List[str]:
        """Identify risk factors for equipment"""
        risk_factors = []
        
        rul_cycles = equipment.get('rul_cycles', 0)
        if rul_cycles < 20:
            risk_factors.append('Very Low RUL')
        elif rul_cycles < 50:
            risk_factors.append('Low RUL')
        
        maintenance_count = equipment.get('maintenance_count', 0)
        if maintenance_count > 5:
            risk_factors.append('High Maintenance Frequency')
        
        if equipment.get('anomaly_flag', False):
            risk_factors.append('Anomaly Detected')
        
        age_years = equipment.get('age_years', 0)
        if age_years > 10:
            risk_factors.append('Aging Equipment')
        
        return risk_factors
    
    def _calculate_health_trend(self, historical_data: List[Dict[str, Any]]) -> str:
        """Calculate fleet health trend from historical data"""
        try:
            if len(historical_data) < 2:
                return 'stable'
            
            # Extract health scores over time
            health_scores = [data.get('overall_health_score', 0.5) for data in historical_data]
            
            # Simple trend analysis
            recent_avg = np.mean(health_scores[-3:])  # Last 3 periods
            older_avg = np.mean(health_scores[:-3])   # Earlier periods
            
            if recent_avg > older_avg + 0.05:
                return 'improving'
            elif recent_avg < older_avg - 0.05:
                return 'declining'
            else:
                return 'stable'
                
        except Exception as e:
            logger.error(f"Error calculating health trend: {e}")
            return 'stable'
    
    def _calculate_maintenance_requirement(self, equipment: Dict[str, Any], time_horizon_days: int) -> Optional[Dict[str, Any]]:
        """Calculate maintenance requirement for equipment"""
        try:
            rul_cycles = equipment.get('rul_cycles', 0)
            equipment_type = equipment.get('type', 'unknown')
            equipment_id = equipment.get('id', 'unknown')
            
            # Determine if maintenance is needed within time horizon
            # Assume 1 cycle = 1 day for simplicity
            if rul_cycles > time_horizon_days:
                return None  # No maintenance needed in this period
            
            # Calculate priority score
            urgency_score = max(0, (time_horizon_days - rul_cycles) / time_horizon_days)
            criticality_score = self.equipment_parameters.get(equipment_type, {}).get('criticality_weight', 0.8)
            priority_score = urgency_score * criticality_score
            
            # Determine maintenance type
            if rul_cycles < 10:
                maintenance_type = 'corrective'
                priority = 5
            elif rul_cycles < 30:
                maintenance_type = 'preventive'
                priority = 4
            else:
                maintenance_type = 'inspection'
                priority = 2
            
            # Calculate estimated cost and duration
            base_cost = self.cost_parameters.get(f'{maintenance_type}_maintenance_base_cost', 1000)
            parts_cost = self.equipment_parameters.get(equipment_type, {}).get('parts_cost_avg', 500)
            duration = self.equipment_parameters.get(equipment_type, {}).get('maintenance_duration', 3)
            
            total_cost = base_cost + parts_cost
            
            # Schedule date (based on RUL)
            schedule_date = datetime.now() + timedelta(days=max(1, rul_cycles - 5))
            
            return {
                'equipment_id': equipment_id,
                'equipment_type': equipment_type,
                'maintenance_type': maintenance_type,
                'priority': priority,
                'priority_score': priority_score,
                'estimated_cost': total_cost,
                'estimated_duration': duration,
                'schedule_date': schedule_date,
                'rul_at_schedule': rul_cycles
            }
            
        except Exception as e:
            logger.error(f"Error calculating maintenance requirement: {e}")
            return None
    
    def _optimize_schedule_with_constraints(self, 
                                          maintenance_requirements: List[Dict[str, Any]],
                                          time_horizon_days: int,
                                          constraints: Dict[str, Any]) -> List[MaintenanceSchedule]:
        """Optimize maintenance schedule with resource constraints"""
        try:
            optimized_schedule = []
            
            # Available capacity per day
            daily_technician_hours = constraints['technicians'] * constraints['working_hours_per_day']
            daily_bay_hours = constraints['maintenance_bays'] * constraints['working_hours_per_day']
            
            # Track resource usage by day
            resource_usage = {}
            
            for req in maintenance_requirements:
                # Find optimal scheduling date
                earliest_date = req['schedule_date']
                duration = req['estimated_duration']
                
                # Find first available slot
                scheduled_date = self._find_available_slot(
                    earliest_date, duration, resource_usage, 
                    daily_technician_hours, daily_bay_hours, time_horizon_days
                )
                
                if scheduled_date:
                    # Create maintenance schedule entry
                    schedule_entry = MaintenanceSchedule(
                        equipment_id=req['equipment_id'],
                        equipment_type=req['equipment_type'],
                        scheduled_date=scheduled_date.isoformat(),
                        maintenance_type=req['maintenance_type'],
                        priority=req['priority'],
                        estimated_duration_hours=duration,
                        estimated_cost=req['estimated_cost'],
                        required_resources=['technician', 'maintenance_bay'],
                        rul_at_schedule=req['rul_at_schedule'],
                        risk_score=req['priority_score']
                    )
                    
                    optimized_schedule.append(schedule_entry)
                    
                    # Update resource usage
                    date_key = scheduled_date.strftime('%Y-%m-%d')
                    if date_key not in resource_usage:
                        resource_usage[date_key] = {'technician_hours': 0, 'bay_hours': 0}
                    
                    resource_usage[date_key]['technician_hours'] += duration
                    resource_usage[date_key]['bay_hours'] += duration
            
            return optimized_schedule
            
        except Exception as e:
            logger.error(f"Error optimizing schedule with constraints: {e}")
            return []
    
    def _find_available_slot(self, 
                           earliest_date: datetime,
                           duration: float,
                           resource_usage: Dict[str, Dict[str, float]],
                           daily_technician_hours: float,
                           daily_bay_hours: float,
                           time_horizon_days: int) -> Optional[datetime]:
        """Find available time slot for maintenance"""
        try:
            current_date = earliest_date
            end_date = datetime.now() + timedelta(days=time_horizon_days)
            
            while current_date <= end_date:
                date_key = current_date.strftime('%Y-%m-%d')
                
                # Check if it's a working day (Monday-Friday)
                if current_date.weekday() < 5:  # 0-4 are Monday-Friday
                    current_usage = resource_usage.get(date_key, {'technician_hours': 0, 'bay_hours': 0})
                    
                    # Check if resources are available
                    if (current_usage['technician_hours'] + duration <= daily_technician_hours and
                        current_usage['bay_hours'] + duration <= daily_bay_hours):
                        return current_date
                
                current_date += timedelta(days=1)
            
            return None  # No available slot found
            
        except Exception as e:
            logger.error(f"Error finding available slot: {e}")
            return None
    
    def _optimize_single_resource_allocation(self, 
                                           maintenance_schedule: List[MaintenanceSchedule],
                                           resource_type: str,
                                           constraints: Dict[str, Any]) -> ResourceAllocation:
        """Optimize allocation for a single resource type"""
        try:
            # Calculate resource requirements
            total_required = 0
            daily_requirements = {}
            
            for maintenance in maintenance_schedule:
                date_key = datetime.fromisoformat(maintenance.scheduled_date.replace('Z', '+00:00')).strftime('%Y-%m-%d')
                
                if resource_type == 'technicians':
                    required = maintenance.estimated_duration_hours / constraints['working_hours_per_day']
                elif resource_type == 'maintenance_bays':
                    required = maintenance.estimated_duration_hours / constraints['working_hours_per_day']
                else:
                    required = 1  # Specialized tools
                
                total_required += required
                daily_requirements[date_key] = daily_requirements.get(date_key, 0) + required
            
            # Calculate metrics
            total_available = constraints.get(resource_type, 1)
            max_daily_available = total_available
            
            # Find bottleneck periods
            bottleneck_periods = []
            for date, required in daily_requirements.items():
                if required > max_daily_available:
                    bottleneck_periods.append(date)
            
            # Calculate efficiency
            if total_available > 0:
                allocation_efficiency = min(1.0, total_required / (total_available * len(daily_requirements)))
            else:
                allocation_efficiency = 0.0
            
            # Generate optimization suggestions
            suggestions = []
            if bottleneck_periods:
                suggestions.append(f"Reschedule maintenance during {len(bottleneck_periods)} bottleneck periods")
            
            if allocation_efficiency < 0.7:
                suggestions.append(f"Consider increasing {resource_type} capacity")
            elif allocation_efficiency > 0.95:
                suggestions.append(f"Optimize {resource_type} utilization to reduce idle time")
            
            # Estimate cost impact
            cost_impact = 0.0
            if bottleneck_periods:
                cost_impact = len(bottleneck_periods) * 2000  # Estimated delay cost per day
            
            return ResourceAllocation(
                resource_type=resource_type,
                total_available=int(total_available),
                total_required=int(total_required),
                allocation_efficiency=allocation_efficiency,
                bottleneck_periods=bottleneck_periods,
                optimization_suggestions=suggestions,
                cost_impact=cost_impact
            )
            
        except Exception as e:
            logger.error(f"Error optimizing {resource_type} allocation: {e}")
            return ResourceAllocation(
                resource_type=resource_type,
                total_available=0,
                total_required=0,
                allocation_efficiency=0.0,
                bottleneck_periods=[],
                optimization_suggestions=[],
                cost_impact=0.0
            )
    
    def _calculate_period_budget(self, 
                               period_maintenance: List[MaintenanceSchedule],
                               equipment_data: List[Dict[str, Any]]) -> MaintenanceBudget:
        """Calculate budget for a specific period"""
        try:
            # Calculate cost components
            preventive_cost = sum(m.estimated_cost for m in period_maintenance if m.maintenance_type == 'preventive')
            corrective_cost = sum(m.estimated_cost for m in period_maintenance if m.maintenance_type == 'corrective')
            inspection_cost = sum(m.estimated_cost for m in period_maintenance if m.maintenance_type == 'inspection')
            
            # Estimate parts and labor breakdown
            total_maintenance_cost = preventive_cost + corrective_cost + inspection_cost
            parts_cost = total_maintenance_cost * 0.4  # Assume 40% parts
            labor_cost = total_maintenance_cost * 0.6  # Assume 60% labor
            
            # Estimate downtime costs
            total_downtime_hours = sum(m.estimated_duration_hours for m in period_maintenance)
            downtime_cost = total_downtime_hours * self.cost_parameters['downtime_cost_per_hour']
            
            total_budget = total_maintenance_cost + downtime_cost
            
            # Identify cost drivers
            cost_drivers = []
            if corrective_cost > preventive_cost:
                cost_drivers.append({
                    'driver': 'High Corrective Maintenance',
                    'impact': corrective_cost,
                    'percentage': corrective_cost / total_budget * 100
                })
            
            high_priority_maintenance = [m for m in period_maintenance if m.priority >= 4]
            if len(high_priority_maintenance) > len(period_maintenance) * 0.3:
                cost_drivers.append({
                    'driver': 'Emergency Maintenance',
                    'impact': len(high_priority_maintenance) * 1500,
                    'percentage': 15.0
                })
            
            # Identify savings opportunities
            savings_opportunities = []
            if corrective_cost > preventive_cost * 2:
                savings_opportunities.append({
                    'opportunity': 'Increase Preventive Maintenance',
                    'potential_savings': corrective_cost * 0.3,
                    'description': 'Shift from reactive to proactive maintenance'
                })
            
            return MaintenanceBudget(
                period="",  # Will be set by caller
                total_budget=total_budget,
                preventive_maintenance_cost=preventive_cost,
                corrective_maintenance_cost=corrective_cost,
                parts_cost=parts_cost,
                labor_cost=labor_cost,
                downtime_cost=downtime_cost,
                budget_variance=0.0,  # Would need historical data
                cost_drivers=cost_drivers,
                savings_opportunities=savings_opportunities
            )
            
        except Exception as e:
            logger.error(f"Error calculating period budget: {e}")
            return MaintenanceBudget(
                period="",
                total_budget=0.0,
                preventive_maintenance_cost=0.0,
                corrective_maintenance_cost=0.0,
                parts_cost=0.0,
                labor_cost=0.0,
                downtime_cost=0.0,
                budget_variance=0.0,
                cost_drivers=[],
                savings_opportunities=[]
            )
    
    def _default_fleet_health_score(self) -> FleetHealthScore:
        """Return default fleet health score when calculation fails"""
        return FleetHealthScore(
            fleet_id="main_fleet",
            overall_health_score=0.5,
            equipment_count=0,
            critical_equipment_count=0,
            average_rul=0.0,
            health_distribution={'healthy': 0, 'warning': 0, 'critical': 0},
            top_risk_equipment=[],
            health_trend='stable'
        )
    
    def save_fleet_analytics_report(self, 
                                  fleet_health: FleetHealthScore,
                                  maintenance_schedule: List[MaintenanceSchedule],
                                  resource_allocation: List[ResourceAllocation],
                                  budget_forecast: List[MaintenanceBudget],
                                  optimization_opportunities: List[Dict[str, Any]],
                                  output_path: Optional[str] = None) -> str:
        """
        Save comprehensive fleet analytics report
        
        Args:
            fleet_health: Fleet health scoring results
            maintenance_schedule: Optimized maintenance schedule
            resource_allocation: Resource allocation results
            budget_forecast: Budget forecast results
            optimization_opportunities: Identified optimization opportunities
            output_path: Optional output file path
            
        Returns:
            Path to saved report file
        """
        try:
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"fleet_analytics_report_{timestamp}.json"
            
            report_data = {
                'generated_at': datetime.now().isoformat(),
                'fleet_health': fleet_health.to_dict(),
                'maintenance_schedule': [schedule.to_dict() for schedule in maintenance_schedule],
                'resource_allocation': [allocation.to_dict() for allocation in resource_allocation],
                'budget_forecast': [budget.to_dict() for budget in budget_forecast],
                'optimization_opportunities': optimization_opportunities,
                'summary': {
                    'total_equipment': fleet_health.equipment_count,
                    'critical_equipment': fleet_health.critical_equipment_count,
                    'scheduled_maintenance_items': len(maintenance_schedule),
                    'total_estimated_budget': sum(budget.total_budget for budget in budget_forecast),
                    'potential_savings': sum(opp.get('estimated_savings', 0) for opp in optimization_opportunities)
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"Fleet analytics report saved to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error saving fleet analytics report: {e}")
            raise