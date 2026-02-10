"""
Predictive Maintenance Integration for True RUL Prediction System

This module implements predictive maintenance capabilities including:
- Maintenance scheduling optimization
- Cost-benefit analysis for replacement decisions
- Integration with CMMS (Computerized Maintenance Management Systems)
- Spare parts inventory optimization based on RUL predictions

Requirements: 4.1, 4.2, 4.5
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import warnings

from .data_structures import PredictionResult
from .rul_regression_model import RULRegressionModel

logger = logging.getLogger(__name__)


class MaintenanceAction(Enum):
    """Types of maintenance actions"""
    INSPECT = "inspect"
    REPLACE = "replace"
    REPAIR = "repair"
    MONITOR = "monitor"
    DEFER = "defer"


class Priority(Enum):
    """Maintenance priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class MaintenanceTask:
    """Represents a maintenance task"""
    task_id: str
    component_id: str
    action: MaintenanceAction
    priority: Priority
    scheduled_date: datetime
    estimated_duration: float  # hours
    estimated_cost: float
    required_parts: List[str]
    required_skills: List[str]
    description: str
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    actual_cost: Optional[float] = None
    actual_duration: Optional[float] = None


@dataclass
class Component:
    """Represents a component in the system"""
    component_id: str
    component_type: str
    location: str
    installation_date: datetime
    last_maintenance_date: Optional[datetime]
    replacement_cost: float
    criticality_score: float  # 0-1, higher = more critical
    current_rul: Optional[float] = None
    rul_confidence: Optional[float] = None
    degradation_stage: Optional[str] = None
    maintenance_history: List[MaintenanceTask] = field(default_factory=list)


@dataclass
class SparePart:
    """Represents a spare part"""
    part_id: str
    part_name: str
    current_stock: int
    min_stock_level: int
    max_stock_level: int
    unit_cost: float
    lead_time_days: int
    supplier: str
    compatible_components: List[str]
    last_order_date: Optional[datetime] = None
    pending_orders: int = 0


@dataclass
class MaintenanceSchedule:
    """Represents a maintenance schedule"""
    schedule_id: str
    tasks: List[MaintenanceTask]
    total_cost: float
    total_duration: float
    resource_requirements: Dict[str, int]
    optimization_objective: str
    created_at: datetime = field(default_factory=datetime.now)


class CostBenefitAnalyzer:
    """Analyzes cost-benefit of maintenance decisions"""
    
    def __init__(self):
        self.cost_factors = {
            'replacement_cost': 1.0,
            'downtime_cost_per_hour': 1000.0,
            'labor_cost_per_hour': 100.0,
            'inspection_cost': 200.0,
            'emergency_multiplier': 3.0,
            'preventive_discount': 0.8
        }
        
        logger.info("CostBenefitAnalyzer initialized")
    
    def analyze_replacement_decision(self, 
                                   component: Component,
                                   prediction: PredictionResult) -> Dict[str, Any]:
        """
        Analyze whether to replace a component now or later
        
        Args:
            component: Component information
            prediction: RUL prediction result
            
        Returns:
            Analysis results with recommendations
        """
        current_rul = prediction.rul_cycles
        confidence = (prediction.rul_confidence_upper - prediction.rul_confidence_lower) / 2
        
        # Calculate costs for different scenarios
        scenarios = {
            'replace_now': self._calculate_immediate_replacement_cost(component),
            'replace_at_failure': self._calculate_failure_replacement_cost(component, current_rul),
            'replace_preventive': self._calculate_preventive_replacement_cost(component, current_rul)
        }
        
        # Calculate expected values considering uncertainty
        for scenario, costs in scenarios.items():
            costs['expected_cost'] = self._calculate_expected_cost(costs, confidence)
        
        # Determine best option
        best_scenario = min(scenarios.keys(), key=lambda k: scenarios[k]['expected_cost'])
        
        # Generate recommendation
        recommendation = self._generate_replacement_recommendation(
            component, prediction, scenarios, best_scenario
        )
        
        return {
            'component_id': component.component_id,
            'current_rul': current_rul,
            'confidence': confidence,
            'scenarios': scenarios,
            'recommended_action': best_scenario,
            'recommendation': recommendation,
            'analysis_date': datetime.now().isoformat()
        }
    
    def _calculate_immediate_replacement_cost(self, component: Component) -> Dict[str, float]:
        """Calculate cost of immediate replacement"""
        replacement_cost = component.replacement_cost
        labor_cost = self.cost_factors['labor_cost_per_hour'] * 4  # Assume 4 hours
        downtime_cost = self.cost_factors['downtime_cost_per_hour'] * 2  # Assume 2 hours downtime
        
        # Preventive maintenance discount
        total_cost = (replacement_cost + labor_cost + downtime_cost) * self.cost_factors['preventive_discount']
        
        return {
            'replacement_cost': replacement_cost,
            'labor_cost': labor_cost,
            'downtime_cost': downtime_cost,
            'total_cost': total_cost,
            'timing': 'immediate'
        }
    
    def _calculate_failure_replacement_cost(self, component: Component, rul: float) -> Dict[str, float]:
        """Calculate cost of replacement at failure"""
        replacement_cost = component.replacement_cost
        
        # Emergency replacement costs more
        labor_cost = self.cost_factors['labor_cost_per_hour'] * 6 * self.cost_factors['emergency_multiplier']
        downtime_cost = self.cost_factors['downtime_cost_per_hour'] * 8 * component.criticality_score
        
        # Additional costs due to emergency
        emergency_premium = replacement_cost * 0.2  # 20% premium for emergency parts
        
        total_cost = replacement_cost + labor_cost + downtime_cost + emergency_premium
        
        return {
            'replacement_cost': replacement_cost + emergency_premium,
            'labor_cost': labor_cost,
            'downtime_cost': downtime_cost,
            'total_cost': total_cost,
            'timing': f'at_failure_in_{rul}_cycles'
        }
    
    def _calculate_preventive_replacement_cost(self, component: Component, rul: float) -> Dict[str, float]:
        """Calculate cost of preventive replacement"""
        # Replace when RUL reaches 20% of current value
        preventive_rul = max(1, rul * 0.2)
        
        replacement_cost = component.replacement_cost
        labor_cost = self.cost_factors['labor_cost_per_hour'] * 4
        downtime_cost = self.cost_factors['downtime_cost_per_hour'] * 2
        
        # Preventive discount
        total_cost = (replacement_cost + labor_cost + downtime_cost) * self.cost_factors['preventive_discount']
        
        return {
            'replacement_cost': replacement_cost,
            'labor_cost': labor_cost,
            'downtime_cost': downtime_cost,
            'total_cost': total_cost,
            'timing': f'preventive_at_{preventive_rul}_cycles'
        }
    
    def _calculate_expected_cost(self, costs: Dict[str, float], confidence: float) -> float:
        """Calculate expected cost considering uncertainty"""
        base_cost = costs['total_cost']
        
        # Adjust for confidence - lower confidence increases expected cost due to risk
        uncertainty_factor = 1.0 + (1.0 - confidence) * 0.5
        
        return base_cost * uncertainty_factor
    
    def _generate_replacement_recommendation(self, 
                                           component: Component,
                                           prediction: PredictionResult,
                                           scenarios: Dict[str, Dict],
                                           best_scenario: str) -> str:
        """Generate human-readable recommendation"""
        rul = prediction.rul_cycles
        confidence = (prediction.rul_confidence_upper - prediction.rul_confidence_lower) / 2
        
        if best_scenario == 'replace_now':
            return (f"Recommend immediate replacement. Current RUL ({rul} cycles) is low "
                   f"and immediate replacement is most cost-effective.")
        
        elif best_scenario == 'replace_preventive':
            preventive_rul = max(1, rul * 0.2)
            return (f"Recommend preventive replacement when RUL reaches {preventive_rul:.0f} cycles. "
                   f"This balances cost savings with reliability.")
        
        else:  # replace_at_failure
            if confidence < 0.7:
                return (f"Monitor closely and replace at failure. However, confidence is low ({confidence:.2f}), "
                       f"consider more frequent monitoring.")
            else:
                return (f"Safe to operate until failure. RUL prediction confidence is high ({confidence:.2f}).")


class MaintenanceScheduler:
    """Optimizes maintenance scheduling"""
    
    def __init__(self):
        self.scheduling_constraints = {
            'max_daily_tasks': 5,
            'max_weekly_downtime_hours': 40,
            'min_lead_time_days': 7,
            'resource_availability': {
                'technician': 2,
                'specialist': 1,
                'crane_operator': 1
            }
        }
        
        logger.info("MaintenanceScheduler initialized")
    
    def optimize_schedule(self, 
                         components: List[Component],
                         predictions: Dict[str, PredictionResult],
                         time_horizon_days: int = 90) -> MaintenanceSchedule:
        """
        Optimize maintenance schedule for multiple components
        
        Args:
            components: List of components to schedule
            predictions: RUL predictions for each component
            time_horizon_days: Planning horizon in days
            
        Returns:
            Optimized maintenance schedule
        """
        logger.info(f"Optimizing maintenance schedule for {len(components)} components")
        
        # Generate candidate tasks
        candidate_tasks = self._generate_candidate_tasks(components, predictions)
        
        # Prioritize tasks
        prioritized_tasks = self._prioritize_tasks(candidate_tasks, predictions)
        
        # Schedule tasks considering constraints
        scheduled_tasks = self._schedule_tasks(prioritized_tasks, time_horizon_days)
        
        # Calculate total cost and duration
        total_cost = sum(task.estimated_cost for task in scheduled_tasks)
        total_duration = sum(task.estimated_duration for task in scheduled_tasks)
        
        # Calculate resource requirements
        resource_requirements = self._calculate_resource_requirements(scheduled_tasks)
        
        schedule = MaintenanceSchedule(
            schedule_id=f"schedule_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            tasks=scheduled_tasks,
            total_cost=total_cost,
            total_duration=total_duration,
            resource_requirements=resource_requirements,
            optimization_objective="minimize_cost_maximize_reliability"
        )
        
        logger.info(f"Generated schedule with {len(scheduled_tasks)} tasks, "
                   f"total cost: ${total_cost:.2f}, total duration: {total_duration:.1f}h")
        
        return schedule
    
    def _generate_candidate_tasks(self, 
                                components: List[Component],
                                predictions: Dict[str, PredictionResult]) -> List[MaintenanceTask]:
        """Generate candidate maintenance tasks"""
        tasks = []
        
        for component in components:
            if component.component_id not in predictions:
                continue
            
            prediction = predictions[component.component_id]
            rul = prediction.rul_cycles
            
            # Generate different types of tasks based on RUL and degradation stage
            if rul <= 10 or prediction.degradation_stage == "critical":
                # Critical - immediate replacement
                task = MaintenanceTask(
                    task_id=f"{component.component_id}_replace_critical",
                    component_id=component.component_id,
                    action=MaintenanceAction.REPLACE,
                    priority=Priority.CRITICAL,
                    scheduled_date=datetime.now() + timedelta(days=1),
                    estimated_duration=6.0,
                    estimated_cost=component.replacement_cost + 600,  # Labor cost
                    required_parts=[f"{component.component_type}_replacement"],
                    required_skills=["technician", "specialist"],
                    description=f"Critical replacement for {component.component_id} (RUL: {rul})"
                )
                tasks.append(task)
                
            elif rul <= 30 or prediction.degradation_stage == "advanced_degradation":
                # High priority - preventive replacement
                task = MaintenanceTask(
                    task_id=f"{component.component_id}_replace_preventive",
                    component_id=component.component_id,
                    action=MaintenanceAction.REPLACE,
                    priority=Priority.HIGH,
                    scheduled_date=datetime.now() + timedelta(days=14),
                    estimated_duration=4.0,
                    estimated_cost=component.replacement_cost + 400,
                    required_parts=[f"{component.component_type}_replacement"],
                    required_skills=["technician"],
                    description=f"Preventive replacement for {component.component_id} (RUL: {rul})"
                )
                tasks.append(task)
                
            elif rul <= 60 or prediction.degradation_stage == "early_degradation":
                # Medium priority - inspection
                task = MaintenanceTask(
                    task_id=f"{component.component_id}_inspect",
                    component_id=component.component_id,
                    action=MaintenanceAction.INSPECT,
                    priority=Priority.MEDIUM,
                    scheduled_date=datetime.now() + timedelta(days=30),
                    estimated_duration=1.0,
                    estimated_cost=200,
                    required_parts=[],
                    required_skills=["technician"],
                    description=f"Detailed inspection for {component.component_id} (RUL: {rul})"
                )
                tasks.append(task)
                
            else:
                # Low priority - monitoring
                task = MaintenanceTask(
                    task_id=f"{component.component_id}_monitor",
                    component_id=component.component_id,
                    action=MaintenanceAction.MONITOR,
                    priority=Priority.LOW,
                    scheduled_date=datetime.now() + timedelta(days=60),
                    estimated_duration=0.5,
                    estimated_cost=50,
                    required_parts=[],
                    required_skills=["technician"],
                    description=f"Routine monitoring for {component.component_id} (RUL: {rul})"
                )
                tasks.append(task)
        
        return tasks
    
    def _prioritize_tasks(self, 
                         tasks: List[MaintenanceTask],
                         predictions: Dict[str, PredictionResult]) -> List[MaintenanceTask]:
        """Prioritize tasks based on multiple criteria"""
        def priority_score(task: MaintenanceTask) -> float:
            # Base priority score
            priority_scores = {
                Priority.CRITICAL: 1000,
                Priority.HIGH: 100,
                Priority.MEDIUM: 10,
                Priority.LOW: 1
            }
            
            score = priority_scores[task.priority]
            
            # Adjust based on RUL if available
            if task.component_id in predictions:
                rul = predictions[task.component_id].rul_cycles
                # Lower RUL = higher priority
                score *= (1000 / max(1, rul))
            
            # Adjust based on action type
            action_multipliers = {
                MaintenanceAction.REPLACE: 1.0,
                MaintenanceAction.REPAIR: 0.8,
                MaintenanceAction.INSPECT: 0.5,
                MaintenanceAction.MONITOR: 0.2
            }
            score *= action_multipliers.get(task.action, 1.0)
            
            return score
        
        # Sort by priority score (descending)
        return sorted(tasks, key=priority_score, reverse=True)
    
    def _schedule_tasks(self, 
                       prioritized_tasks: List[MaintenanceTask],
                       time_horizon_days: int) -> List[MaintenanceTask]:
        """Schedule tasks considering constraints"""
        scheduled_tasks = []
        daily_task_count = {}
        weekly_downtime = {}
        resource_usage = {}
        
        for task in prioritized_tasks:
            # Find earliest feasible date
            feasible_date = self._find_feasible_date(
                task, daily_task_count, weekly_downtime, resource_usage, time_horizon_days
            )
            
            if feasible_date:
                task.scheduled_date = feasible_date
                scheduled_tasks.append(task)
                
                # Update constraints tracking
                date_key = feasible_date.date()
                week_key = feasible_date.isocalendar()[:2]  # (year, week)
                
                daily_task_count[date_key] = daily_task_count.get(date_key, 0) + 1
                weekly_downtime[week_key] = weekly_downtime.get(week_key, 0) + task.estimated_duration
                
                # Update resource usage
                for skill in task.required_skills:
                    if skill not in resource_usage:
                        resource_usage[skill] = {}
                    resource_usage[skill][date_key] = resource_usage[skill].get(date_key, 0) + 1
            else:
                logger.warning(f"Could not schedule task {task.task_id} within time horizon")
        
        return scheduled_tasks
    
    def _find_feasible_date(self, 
                           task: MaintenanceTask,
                           daily_task_count: Dict,
                           weekly_downtime: Dict,
                           resource_usage: Dict,
                           time_horizon_days: int) -> Optional[datetime]:
        """Find the earliest feasible date for a task"""
        start_date = max(task.scheduled_date, datetime.now() + timedelta(days=1))
        end_date = datetime.now() + timedelta(days=time_horizon_days)
        
        current_date = start_date
        while current_date <= end_date:
            date_key = current_date.date()
            week_key = current_date.isocalendar()[:2]
            
            # Check daily task limit
            if daily_task_count.get(date_key, 0) >= self.scheduling_constraints['max_daily_tasks']:
                current_date += timedelta(days=1)
                continue
            
            # Check weekly downtime limit
            if weekly_downtime.get(week_key, 0) + task.estimated_duration > self.scheduling_constraints['max_weekly_downtime_hours']:
                # Skip to next week
                current_date += timedelta(days=7 - current_date.weekday())
                continue
            
            # Check resource availability
            resource_available = True
            for skill in task.required_skills:
                max_available = self.scheduling_constraints['resource_availability'].get(skill, 1)
                current_usage = resource_usage.get(skill, {}).get(date_key, 0)
                if current_usage >= max_available:
                    resource_available = False
                    break
            
            if resource_available:
                return current_date
            
            current_date += timedelta(days=1)
        
        return None
    
    def _calculate_resource_requirements(self, tasks: List[MaintenanceTask]) -> Dict[str, int]:
        """Calculate total resource requirements"""
        requirements = {}
        
        for task in tasks:
            for skill in task.required_skills:
                requirements[skill] = requirements.get(skill, 0) + 1
        
        return requirements


class InventoryOptimizer:
    """Optimizes spare parts inventory based on RUL predictions"""
    
    def __init__(self):
        self.optimization_params = {
            'service_level': 0.95,  # 95% service level
            'holding_cost_rate': 0.2,  # 20% annual holding cost
            'ordering_cost': 100,  # Fixed cost per order
            'safety_stock_factor': 1.65  # For 95% service level (z-score)
        }
        
        logger.info("InventoryOptimizer initialized")
    
    def optimize_inventory(self, 
                          spare_parts: List[SparePart],
                          components: List[Component],
                          predictions: Dict[str, PredictionResult],
                          planning_horizon_days: int = 365) -> Dict[str, Dict[str, Any]]:
        """
        Optimize inventory levels based on RUL predictions
        
        Args:
            spare_parts: List of spare parts
            components: List of components
            predictions: RUL predictions
            planning_horizon_days: Planning horizon
            
        Returns:
            Optimization recommendations for each part
        """
        logger.info(f"Optimizing inventory for {len(spare_parts)} spare parts")
        
        recommendations = {}
        
        for part in spare_parts:
            # Calculate demand forecast based on RUL predictions
            demand_forecast = self._forecast_demand(part, components, predictions, planning_horizon_days)
            
            # Calculate optimal inventory levels
            optimal_levels = self._calculate_optimal_levels(part, demand_forecast)
            
            # Generate recommendations
            recommendation = self._generate_inventory_recommendation(part, optimal_levels, demand_forecast)
            
            recommendations[part.part_id] = recommendation
        
        return recommendations
    
    def _forecast_demand(self, 
                        part: SparePart,
                        components: List[Component],
                        predictions: Dict[str, PredictionResult],
                        planning_horizon_days: int) -> Dict[str, Any]:
        """Forecast demand for a spare part based on RUL predictions"""
        # Find components that use this part
        compatible_components = [c for c in components if c.component_type in part.compatible_components]
        
        # Calculate expected failures within planning horizon
        expected_failures = 0
        failure_timeline = []
        
        for component in compatible_components:
            if component.component_id in predictions:
                prediction = predictions[component.component_id]
                rul_cycles = prediction.rul_cycles
                
                # Convert cycles to days (assume 1 cycle = 1 day for simplicity)
                rul_days = rul_cycles
                
                if rul_days <= planning_horizon_days:
                    expected_failures += 1
                    failure_timeline.append({
                        'component_id': component.component_id,
                        'expected_failure_date': datetime.now() + timedelta(days=rul_days),
                        'confidence': (prediction.rul_confidence_upper - prediction.rul_confidence_lower) / 2
                    })
        
        # Calculate demand statistics
        annual_demand = expected_failures * (365 / planning_horizon_days)
        demand_variance = annual_demand * 0.2  # Assume 20% coefficient of variation
        
        return {
            'expected_failures': expected_failures,
            'annual_demand': annual_demand,
            'demand_variance': demand_variance,
            'failure_timeline': failure_timeline,
            'planning_horizon_days': planning_horizon_days
        }
    
    def _calculate_optimal_levels(self, 
                                 part: SparePart,
                                 demand_forecast: Dict[str, Any]) -> Dict[str, float]:
        """Calculate optimal inventory levels using EOQ and safety stock"""
        annual_demand = demand_forecast['annual_demand']
        demand_variance = demand_forecast['demand_variance']
        
        if annual_demand <= 0:
            return {
                'economic_order_quantity': 0,
                'reorder_point': part.min_stock_level,
                'safety_stock': 0,
                'optimal_max_stock': part.max_stock_level
            }
        
        # Economic Order Quantity (EOQ)
        holding_cost = part.unit_cost * self.optimization_params['holding_cost_rate']
        ordering_cost = self.optimization_params['ordering_cost']
        
        eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost) if holding_cost > 0 else 1
        
        # Safety stock calculation
        lead_time_demand = annual_demand * (part.lead_time_days / 365)
        lead_time_variance = demand_variance * (part.lead_time_days / 365)
        
        safety_stock = self.optimization_params['safety_stock_factor'] * np.sqrt(lead_time_variance)
        
        # Reorder point
        reorder_point = lead_time_demand + safety_stock
        
        # Optimal maximum stock
        optimal_max_stock = reorder_point + eoq
        
        return {
            'economic_order_quantity': max(1, int(eoq)),
            'reorder_point': max(part.min_stock_level, int(reorder_point)),
            'safety_stock': max(0, int(safety_stock)),
            'optimal_max_stock': max(part.max_stock_level, int(optimal_max_stock))
        }
    
    def _generate_inventory_recommendation(self, 
                                         part: SparePart,
                                         optimal_levels: Dict[str, float],
                                         demand_forecast: Dict[str, Any]) -> Dict[str, Any]:
        """Generate inventory recommendation"""
        current_stock = part.current_stock
        reorder_point = optimal_levels['reorder_point']
        eoq = optimal_levels['economic_order_quantity']
        
        # Determine action needed
        if current_stock <= reorder_point:
            action = "ORDER_NOW"
            order_quantity = eoq
            urgency = "HIGH" if current_stock < part.min_stock_level else "MEDIUM"
        elif current_stock > optimal_levels['optimal_max_stock']:
            action = "REDUCE_STOCK"
            order_quantity = 0
            urgency = "LOW"
        else:
            action = "MAINTAIN"
            order_quantity = 0
            urgency = "LOW"
        
        # Calculate cost implications
        holding_cost_current = current_stock * part.unit_cost * self.optimization_params['holding_cost_rate']
        holding_cost_optimal = optimal_levels['optimal_max_stock'] * part.unit_cost * self.optimization_params['holding_cost_rate']
        
        return {
            'part_id': part.part_id,
            'current_stock': current_stock,
            'optimal_levels': optimal_levels,
            'action': action,
            'order_quantity': order_quantity,
            'urgency': urgency,
            'expected_annual_demand': demand_forecast['annual_demand'],
            'cost_analysis': {
                'current_holding_cost': holding_cost_current,
                'optimal_holding_cost': holding_cost_optimal,
                'potential_savings': holding_cost_current - holding_cost_optimal
            },
            'failure_timeline': demand_forecast['failure_timeline']
        }


class CMMSIntegrator:
    """Integrates with Computerized Maintenance Management Systems"""
    
    def __init__(self, cmms_config: Optional[Dict[str, Any]] = None):
        """
        Initialize CMMS integrator
        
        Args:
            cmms_config: Configuration for CMMS integration
        """
        self.cmms_config = cmms_config or {
            'system_type': 'generic',
            'api_endpoint': 'http://localhost:8080/api',
            'api_key': 'demo_key',
            'timeout': 30
        }
        
        self.supported_systems = ['maximo', 'sap_pm', 'maintenance_connection', 'generic']
        
        logger.info(f"CMMSIntegrator initialized for {self.cmms_config['system_type']} system")
    
    def export_maintenance_schedule(self, schedule: MaintenanceSchedule) -> Dict[str, Any]:
        """
        Export maintenance schedule to CMMS format
        
        Args:
            schedule: Maintenance schedule to export
            
        Returns:
            CMMS-formatted schedule data
        """
        logger.info(f"Exporting schedule {schedule.schedule_id} to CMMS format")
        
        # Convert to CMMS format
        cmms_data = {
            'schedule_id': schedule.schedule_id,
            'created_date': schedule.created_at.isoformat(),
            'total_estimated_cost': schedule.total_cost,
            'total_estimated_hours': schedule.total_duration,
            'work_orders': []
        }
        
        for task in schedule.tasks:
            work_order = self._convert_task_to_work_order(task)
            cmms_data['work_orders'].append(work_order)
        
        # Add system-specific formatting
        if self.cmms_config['system_type'] == 'maximo':
            cmms_data = self._format_for_maximo(cmms_data)
        elif self.cmms_config['system_type'] == 'sap_pm':
            cmms_data = self._format_for_sap(cmms_data)
        
        return cmms_data
    
    def _convert_task_to_work_order(self, task: MaintenanceTask) -> Dict[str, Any]:
        """Convert maintenance task to CMMS work order format"""
        return {
            'work_order_id': task.task_id,
            'asset_id': task.component_id,
            'work_type': task.action.value,
            'priority': task.priority.value,
            'scheduled_start': task.scheduled_date.isoformat(),
            'estimated_duration': task.estimated_duration,
            'estimated_cost': task.estimated_cost,
            'description': task.description,
            'required_parts': task.required_parts,
            'required_skills': task.required_skills,
            'status': 'PLANNED'
        }
    
    def _format_for_maximo(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format data for IBM Maximo"""
        # Maximo-specific field mappings
        for work_order in data['work_orders']:
            work_order['WONUM'] = work_order.pop('work_order_id')
            work_order['ASSETNUM'] = work_order.pop('asset_id')
            work_order['WORKTYPE'] = work_order.pop('work_type').upper()
            work_order['WOPRIORITY'] = work_order.pop('priority').upper()
            work_order['SCHEDSTART'] = work_order.pop('scheduled_start')
            work_order['ESTDUR'] = work_order.pop('estimated_duration')
            work_order['ESTLABCOST'] = work_order.pop('estimated_cost')
            work_order['DESCRIPTION'] = work_order.pop('description')
            work_order['STATUS'] = 'WPLAN'
        
        return data
    
    def _format_for_sap(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format data for SAP Plant Maintenance"""
        # SAP PM-specific field mappings
        for work_order in data['work_orders']:
            work_order['OrderNumber'] = work_order.pop('work_order_id')
            work_order['Equipment'] = work_order.pop('asset_id')
            work_order['OrderType'] = work_order.pop('work_type').upper()
            work_order['Priority'] = work_order.pop('priority').upper()
            work_order['BasicStartDate'] = work_order.pop('scheduled_start')
            work_order['EstimatedWork'] = work_order.pop('estimated_duration')
            work_order['EstimatedCosts'] = work_order.pop('estimated_cost')
            work_order['ShortText'] = work_order.pop('description')
            work_order['SystemStatus'] = 'CRTD'
        
        return data
    
    def import_work_order_status(self, work_order_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Import work order status updates from CMMS
        
        Args:
            work_order_updates: List of work order status updates
            
        Returns:
            Import results
        """
        logger.info(f"Importing {len(work_order_updates)} work order updates from CMMS")
        
        processed_updates = []
        errors = []
        
        for update in work_order_updates:
            try:
                processed_update = self._process_work_order_update(update)
                processed_updates.append(processed_update)
            except Exception as e:
                errors.append({
                    'work_order_id': update.get('work_order_id', 'unknown'),
                    'error': str(e)
                })
        
        return {
            'processed_count': len(processed_updates),
            'error_count': len(errors),
            'processed_updates': processed_updates,
            'errors': errors
        }
    
    def _process_work_order_update(self, update: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single work order update"""
        # Normalize field names based on CMMS system
        if self.cmms_config['system_type'] == 'maximo':
            normalized_update = {
                'work_order_id': update.get('WONUM'),
                'status': update.get('STATUS'),
                'actual_start': update.get('ACTSTART'),
                'actual_finish': update.get('ACTFINISH'),
                'actual_cost': update.get('ACTLABCOST'),
                'actual_duration': update.get('ACTDUR')
            }
        elif self.cmms_config['system_type'] == 'sap_pm':
            normalized_update = {
                'work_order_id': update.get('OrderNumber'),
                'status': update.get('SystemStatus'),
                'actual_start': update.get('ActualStartDate'),
                'actual_finish': update.get('ActualFinishDate'),
                'actual_cost': update.get('ActualCosts'),
                'actual_duration': update.get('ActualWork')
            }
        else:
            normalized_update = update
        
        return normalized_update


class PredictiveMaintenanceSystem:
    """
    Main system that coordinates all predictive maintenance components
    """
    
    def __init__(self, 
                 rul_model: RULRegressionModel,
                 cmms_config: Optional[Dict[str, Any]] = None):
        """
        Initialize predictive maintenance system
        
        Args:
            rul_model: Trained RUL prediction model
            cmms_config: CMMS integration configuration
        """
        self.rul_model = rul_model
        
        # Initialize components
        self.cost_analyzer = CostBenefitAnalyzer()
        self.scheduler = MaintenanceScheduler()
        self.inventory_optimizer = InventoryOptimizer()
        self.cmms_integrator = CMMSIntegrator(cmms_config)
        
        # System state
        self.components: Dict[str, Component] = {}
        self.spare_parts: Dict[str, SparePart] = {}
        self.maintenance_history: List[MaintenanceTask] = []
        
        logger.info("PredictiveMaintenanceSystem initialized")
    
    def add_component(self, component: Component):
        """Add a component to the system"""
        self.components[component.component_id] = component
        logger.info(f"Added component: {component.component_id}")
    
    def add_spare_part(self, spare_part: SparePart):
        """Add a spare part to the system"""
        self.spare_parts[spare_part.part_id] = spare_part
        logger.info(f"Added spare part: {spare_part.part_id}")
    
    def generate_maintenance_plan(self, 
                                 prediction_data: Dict[str, np.ndarray],
                                 planning_horizon_days: int = 90) -> Dict[str, Any]:
        """
        Generate comprehensive maintenance plan
        
        Args:
            prediction_data: Feature data for RUL predictions
            planning_horizon_days: Planning horizon in days
            
        Returns:
            Comprehensive maintenance plan
        """
        logger.info("Generating comprehensive maintenance plan")
        
        # Get RUL predictions for all components
        predictions = {}
        for component_id, features in prediction_data.items():
            if component_id in self.components:
                try:
                    rul_pred = self.rul_model.predict(features.reshape(1, -1))[0]
                    confidence_pred, lower, upper = self.rul_model.predict_with_confidence(
                        features.reshape(1, -1)
                    )
                    
                    prediction = PredictionResult(
                        rul_cycles=int(rul_pred),
                        rul_confidence_lower=int(lower[0]),
                        rul_confidence_upper=int(upper[0]),
                        degradation_score=0.5,  # Simplified
                        degradation_stage="early_degradation",  # Simplified
                        anomaly_flag=False,
                        anomaly_score=0.1,
                        feature_importance={},
                        timestamp=datetime.now().timestamp(),
                        model_version="1.0"
                    )
                    
                    predictions[component_id] = prediction
                    
                    # Update component with current RUL
                    self.components[component_id].current_rul = rul_pred
                    self.components[component_id].rul_confidence = (upper[0] - lower[0]) / 2
                    
                except Exception as e:
                    logger.error(f"Failed to predict RUL for {component_id}: {e}")
        
        # Perform cost-benefit analysis
        cost_analyses = {}
        for component_id, prediction in predictions.items():
            if component_id in self.components:
                analysis = self.cost_analyzer.analyze_replacement_decision(
                    self.components[component_id], prediction
                )
                cost_analyses[component_id] = analysis
        
        # Generate optimized maintenance schedule
        components_list = list(self.components.values())
        maintenance_schedule = self.scheduler.optimize_schedule(
            components_list, predictions, planning_horizon_days
        )
        
        # Optimize inventory
        spare_parts_list = list(self.spare_parts.values())
        inventory_recommendations = self.inventory_optimizer.optimize_inventory(
            spare_parts_list, components_list, predictions, planning_horizon_days
        )
        
        # Export to CMMS format
        cmms_export = self.cmms_integrator.export_maintenance_schedule(maintenance_schedule)
        
        # Compile comprehensive plan
        maintenance_plan = {
            'plan_id': f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'generated_at': datetime.now().isoformat(),
            'planning_horizon_days': planning_horizon_days,
            'rul_predictions': {k: {
                'rul_cycles': v.rul_cycles,
                'confidence_range': (v.rul_confidence_lower, v.rul_confidence_upper),
                'degradation_stage': v.degradation_stage
            } for k, v in predictions.items()},
            'cost_benefit_analyses': cost_analyses,
            'maintenance_schedule': {
                'schedule_id': maintenance_schedule.schedule_id,
                'total_tasks': len(maintenance_schedule.tasks),
                'total_cost': maintenance_schedule.total_cost,
                'total_duration': maintenance_schedule.total_duration,
                'tasks': [{
                    'task_id': task.task_id,
                    'component_id': task.component_id,
                    'action': task.action.value,
                    'priority': task.priority.value,
                    'scheduled_date': task.scheduled_date.isoformat(),
                    'estimated_cost': task.estimated_cost,
                    'estimated_duration': task.estimated_duration
                } for task in maintenance_schedule.tasks]
            },
            'inventory_recommendations': inventory_recommendations,
            'cmms_export': cmms_export,
            'summary': {
                'total_components': len(self.components),
                'components_needing_attention': len([p for p in predictions.values() if p.rul_cycles <= 60]),
                'critical_components': len([p for p in predictions.values() if p.rul_cycles <= 10]),
                'total_estimated_cost': maintenance_schedule.total_cost,
                'inventory_actions_needed': len([r for r in inventory_recommendations.values() if r['action'] != 'MAINTAIN'])
            }
        }
        
        logger.info(f"Generated maintenance plan with {len(maintenance_schedule.tasks)} tasks")
        
        return maintenance_plan
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            'components_count': len(self.components),
            'spare_parts_count': len(self.spare_parts),
            'maintenance_history_count': len(self.maintenance_history),
            'components_with_rul': len([c for c in self.components.values() if c.current_rul is not None]),
            'critical_components': len([c for c in self.components.values() 
                                     if c.current_rul is not None and c.current_rul <= 10]),
            'system_health': 'operational'
        }


def create_predictive_maintenance_system(
    rul_model: RULRegressionModel,
    cmms_config: Optional[Dict[str, Any]] = None
) -> PredictiveMaintenanceSystem:
    """
    Factory function to create predictive maintenance system
    
    Args:
        rul_model: Trained RUL prediction model
        cmms_config: CMMS integration configuration
        
    Returns:
        Configured PredictiveMaintenanceSystem
    """
    return PredictiveMaintenanceSystem(rul_model, cmms_config)