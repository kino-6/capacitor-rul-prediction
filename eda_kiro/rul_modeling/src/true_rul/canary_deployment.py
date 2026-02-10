"""
Canary Deployment System

This module provides canary deployment capabilities for gradual model rollouts
with automated monitoring and rollback functionality.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
import threading

import numpy as np

from .model_versioning import ModelVersionManager, ABTestManager, ABTestConfig, ABTestStatus, ModelStatus
from .production_monitoring import ProductionMonitor, AlertSeverity

logger = logging.getLogger(__name__)


class CanaryStatus(Enum):
    """Canary deployment status"""
    PREPARING = "preparing"
    STARTING = "starting"
    RUNNING = "running"
    PROMOTING = "promoting"
    COMPLETED = "completed"
    ROLLING_BACK = "rolling_back"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CanaryStage(Enum):
    """Canary deployment stages"""
    STAGE_1 = "stage_1"  # 5% traffic
    STAGE_2 = "stage_2"  # 25% traffic
    STAGE_3 = "stage_3"  # 50% traffic
    STAGE_4 = "stage_4"  # 100% traffic


@dataclass
class CanaryConfig:
    """Canary deployment configuration"""
    deployment_id: str
    name: str
    description: str
    current_model_id: str
    current_model_version: str
    canary_model_id: str
    canary_model_version: str
    created_by: str
    
    # Stage configuration
    stage_durations_minutes: Dict[CanaryStage, int] = field(default_factory=lambda: {
        CanaryStage.STAGE_1: 30,   # 5% for 30 minutes
        CanaryStage.STAGE_2: 60,   # 25% for 1 hour
        CanaryStage.STAGE_3: 120,  # 50% for 2 hours
        CanaryStage.STAGE_4: 60    # 100% for 1 hour before completion
    })
    
    stage_traffic_splits: Dict[CanaryStage, float] = field(default_factory=lambda: {
        CanaryStage.STAGE_1: 0.05,
        CanaryStage.STAGE_2: 0.25,
        CanaryStage.STAGE_3: 0.50,
        CanaryStage.STAGE_4: 1.00
    })
    
    # Success criteria
    max_error_rate_increase: float = 0.01  # 1% increase
    max_fpr_increase: float = 0.02  # 2% increase
    max_latency_increase_percent: float = 20.0  # 20% increase
    min_sample_size_per_stage: int = 50
    
    # Rollback criteria
    auto_rollback_enabled: bool = True
    rollback_on_alert: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        # Handle stage durations - keys might be strings or enums
        stage_durations = {}
        for k, v in self.stage_durations_minutes.items():
            key = k.value if hasattr(k, 'value') else k
            stage_durations[key] = v
            
        # Handle stage traffic splits - keys might be strings or enums  
        stage_splits = {}
        for k, v in self.stage_traffic_splits.items():
            key = k.value if hasattr(k, 'value') else k
            stage_splits[key] = v
            
        return {
            "deployment_id": self.deployment_id,
            "name": self.name,
            "description": self.description,
            "current_model_id": self.current_model_id,
            "current_model_version": self.current_model_version,
            "canary_model_id": self.canary_model_id,
            "canary_model_version": self.canary_model_version,
            "created_by": self.created_by,
            "stage_durations_minutes": stage_durations,
            "stage_traffic_splits": stage_splits,
            "max_error_rate_increase": self.max_error_rate_increase,
            "max_fpr_increase": self.max_fpr_increase,
            "max_latency_increase_percent": self.max_latency_increase_percent,
            "min_sample_size_per_stage": self.min_sample_size_per_stage,
            "auto_rollback_enabled": self.auto_rollback_enabled,
            "rollback_on_alert": self.rollback_on_alert
        }


@dataclass
class CanaryDeployment:
    """Canary deployment state"""
    config: CanaryConfig
    status: CanaryStatus = CanaryStatus.PREPARING
    current_stage: Optional[CanaryStage] = None
    stage_start_time: Optional[datetime] = None
    ab_test_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "config": self.config.to_dict(),
            "status": self.status.value,
            "current_stage": self.current_stage.value if self.current_stage else None,
            "stage_start_time": self.stage_start_time.isoformat() if self.stage_start_time else None,
            "ab_test_id": self.ab_test_id,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message
        }


class CanaryDeploymentManager:
    """Manages canary deployments"""
    
    def __init__(self, 
                 model_manager: ModelVersionManager,
                 ab_test_manager: ABTestManager,
                 monitor: Optional[ProductionMonitor] = None):
        self.model_manager = model_manager
        self.ab_test_manager = ab_test_manager
        self.monitor = monitor
        
        self._active_deployments: Dict[str, CanaryDeployment] = {}
        self._deployment_tasks: Dict[str, asyncio.Task] = {}
        self._lock = threading.RLock()
        
    def create_canary_deployment(self,
                                name: str,
                                description: str,
                                current_model_id: str,
                                current_model_version: str,
                                canary_model_id: str,
                                canary_model_version: str,
                                created_by: str,
                                config_overrides: Optional[Dict[str, Any]] = None) -> CanaryDeployment:
        """Create a new canary deployment"""
        
        deployment_id = f"canary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create configuration
        config = CanaryConfig(
            deployment_id=deployment_id,
            name=name,
            description=description,
            current_model_id=current_model_id,
            current_model_version=current_model_version,
            canary_model_id=canary_model_id,
            canary_model_version=canary_model_version,
            created_by=created_by
        )
        
        # Apply configuration overrides
        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        deployment = CanaryDeployment(config=config)
        
        with self._lock:
            self._active_deployments[deployment_id] = deployment
            
        logger.info(f"Created canary deployment {deployment_id}: {name}")
        return deployment
    
    async def start_canary_deployment(self, deployment_id: str) -> bool:
        """Start a canary deployment"""
        with self._lock:
            if deployment_id not in self._active_deployments:
                logger.error(f"Canary deployment {deployment_id} not found")
                return False
                
            deployment = self._active_deployments[deployment_id]
            
            if deployment.status != CanaryStatus.PREPARING:
                logger.error(f"Canary deployment {deployment_id} is not in preparing state")
                return False
                
            deployment.status = CanaryStatus.STARTING
            
            # Start deployment task
            task = asyncio.create_task(self._run_canary_deployment(deployment))
            self._deployment_tasks[deployment_id] = task
            
        logger.info(f"Started canary deployment {deployment_id}")
        return True
    
    async def cancel_canary_deployment(self, deployment_id: str) -> bool:
        """Cancel a running canary deployment"""
        with self._lock:
            if deployment_id not in self._active_deployments:
                return False
                
            deployment = self._active_deployments[deployment_id]
            deployment.status = CanaryStatus.CANCELLED
            
            # Cancel task
            if deployment_id in self._deployment_tasks:
                task = self._deployment_tasks[deployment_id]
                task.cancel()
                
        logger.info(f"Cancelled canary deployment {deployment_id}")
        return True
    
    async def _run_canary_deployment(self, deployment: CanaryDeployment):
        """Run the canary deployment process"""
        try:
            deployment.status = CanaryStatus.RUNNING
            
            # Create A/B test for the deployment
            ab_test_config = self.ab_test_manager.create_ab_test(
                name=f"Canary: {deployment.config.name}",
                description=f"Canary deployment A/B test: {deployment.config.description}",
                model_a_id=deployment.config.current_model_id,
                model_a_version=deployment.config.current_model_version,
                model_b_id=deployment.config.canary_model_id,
                model_b_version=deployment.config.canary_model_version,
                traffic_split=0.05,  # Start with 5%
                duration_hours=24,  # Long duration, will be managed by canary
                created_by=deployment.config.created_by,
                success_metrics=["fpr_rate", "error_rate", "latency_p95"]
            )
            
            deployment.ab_test_id = ab_test_config.test_id
            self.ab_test_manager.start_ab_test(ab_test_config.test_id)
            
            # Run through canary stages
            stages = [CanaryStage.STAGE_1, CanaryStage.STAGE_2, CanaryStage.STAGE_3, CanaryStage.STAGE_4]
            
            for stage in stages:
                if deployment.status == CanaryStatus.CANCELLED:
                    break
                    
                success = await self._run_canary_stage(deployment, stage)
                
                if not success:
                    await self._rollback_canary_deployment(deployment, f"Stage {stage.value} failed")
                    return
                    
            # If we made it through all stages, promote the canary
            await self._complete_canary_deployment(deployment)
            
        except asyncio.CancelledError:
            deployment.status = CanaryStatus.CANCELLED
            logger.info(f"Canary deployment {deployment.config.deployment_id} was cancelled")
            
        except Exception as e:
            logger.error(f"Canary deployment {deployment.config.deployment_id} failed: {e}")
            deployment.error_message = str(e)
            await self._rollback_canary_deployment(deployment, f"Deployment failed: {e}")
    
    async def _run_canary_stage(self, deployment: CanaryDeployment, stage: CanaryStage) -> bool:
        """Run a single canary stage"""
        logger.info(f"Starting canary stage {stage.value} for deployment {deployment.config.deployment_id}")
        
        deployment.current_stage = stage
        deployment.stage_start_time = datetime.now()
        
        # Update A/B test traffic split
        if deployment.ab_test_id:
            with self.ab_test_manager._lock:
                if deployment.ab_test_id in self.ab_test_manager._active_tests:
                    ab_config = self.ab_test_manager._active_tests[deployment.ab_test_id]
                    ab_config.traffic_split = deployment.config.stage_traffic_splits[stage]
        
        # Wait for stage duration
        stage_duration = deployment.config.stage_durations_minutes[stage]
        
        # Check metrics periodically during the stage
        check_interval_minutes = min(5, stage_duration // 4)  # Check 4 times per stage, max every 5 minutes
        checks_per_stage = stage_duration // check_interval_minutes
        
        for check_num in range(checks_per_stage):
            if deployment.status == CanaryStatus.CANCELLED:
                return False
                
            await asyncio.sleep(check_interval_minutes * 60)  # Convert to seconds
            
            # Check if we should rollback
            should_rollback, reason = await self._should_rollback_canary(deployment)
            
            if should_rollback:
                logger.warning(f"Canary rollback triggered: {reason}")
                return False
                
        logger.info(f"Completed canary stage {stage.value} for deployment {deployment.config.deployment_id}")
        return True
    
    async def _should_rollback_canary(self, deployment: CanaryDeployment) -> tuple[bool, Optional[str]]:
        """Check if canary deployment should be rolled back"""
        
        if not deployment.ab_test_id:
            return False, None
            
        try:
            # Get A/B test results
            results = self.ab_test_manager.get_ab_test_results(deployment.ab_test_id)
            
            model_a_results = results["model_a_results"]
            model_b_results = results["model_b_results"]
            
            # Check minimum sample size
            if model_b_results["prediction_count"] < deployment.config.min_sample_size_per_stage:
                return False, None  # Not enough data yet
                
            # Check error rate increase
            error_rate_increase = model_b_results["error_rate"] - model_a_results["error_rate"]
            if error_rate_increase > deployment.config.max_error_rate_increase:
                return True, f"Error rate increased by {error_rate_increase:.3f}"
                
            # Check FPR increase
            if model_a_results["anomaly_prediction_count"] > 0 and model_b_results["anomaly_prediction_count"] > 0:
                fpr_increase = model_b_results["fpr_rate"] - model_a_results["fpr_rate"]
                if fpr_increase > deployment.config.max_fpr_increase:
                    return True, f"FPR increased by {fpr_increase:.3f}"
                    
            # Check latency increase
            if model_a_results["average_latency_ms"] > 0:
                latency_increase_percent = ((model_b_results["average_latency_ms"] - model_a_results["average_latency_ms"]) 
                                          / model_a_results["average_latency_ms"]) * 100
                if latency_increase_percent > deployment.config.max_latency_increase_percent:
                    return True, f"Latency increased by {latency_increase_percent:.1f}%"
                    
            # Check for alerts if monitoring is enabled
            if self.monitor and deployment.config.rollback_on_alert:
                active_alerts = self.monitor.alert_manager.get_active_alerts()
                critical_alerts = [a for a in active_alerts if a.severity == AlertSeverity.CRITICAL]
                
                if critical_alerts:
                    return True, f"Critical alerts detected: {len(critical_alerts)}"
                    
            return False, None
            
        except Exception as e:
            logger.error(f"Error checking canary rollback conditions: {e}")
            return False, None
    
    async def _rollback_canary_deployment(self, deployment: CanaryDeployment, reason: str):
        """Rollback a canary deployment"""
        logger.warning(f"Rolling back canary deployment {deployment.config.deployment_id}: {reason}")
        
        deployment.status = CanaryStatus.ROLLING_BACK
        deployment.error_message = reason
        
        try:
            # Stop A/B test
            if deployment.ab_test_id:
                self.ab_test_manager.stop_ab_test(deployment.ab_test_id)
                
            # Ensure current model is still in production
            # (In a real system, this might involve updating load balancer configuration)
            
            deployment.status = CanaryStatus.FAILED
            deployment.completed_at = datetime.now()
            
            logger.info(f"Canary deployment {deployment.config.deployment_id} rolled back successfully")
            
        except Exception as e:
            logger.error(f"Failed to rollback canary deployment {deployment.config.deployment_id}: {e}")
            deployment.error_message = f"{reason}; Rollback failed: {e}"
    
    async def _complete_canary_deployment(self, deployment: CanaryDeployment):
        """Complete a successful canary deployment"""
        logger.info(f"Completing canary deployment {deployment.config.deployment_id}")
        
        deployment.status = CanaryStatus.PROMOTING
        
        try:
            # Promote canary model to production
            success = self.model_manager.promote_model(
                deployment.config.canary_model_id,
                deployment.config.canary_model_version,
                ModelStatus.PRODUCTION
            )
            
            if not success:
                raise RuntimeError("Failed to promote canary model to production")
                
            # Demote old model
            self.model_manager.promote_model(
                deployment.config.current_model_id,
                deployment.config.current_model_version,
                ModelStatus.DEPRECATED
            )
            
            # Stop A/B test
            if deployment.ab_test_id:
                self.ab_test_manager.stop_ab_test(deployment.ab_test_id)
                
            deployment.status = CanaryStatus.COMPLETED
            deployment.completed_at = datetime.now()
            
            logger.info(f"Canary deployment {deployment.config.deployment_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Failed to complete canary deployment {deployment.config.deployment_id}: {e}")
            deployment.error_message = f"Promotion failed: {e}"
            await self._rollback_canary_deployment(deployment, f"Promotion failed: {e}")
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a canary deployment"""
        with self._lock:
            if deployment_id not in self._active_deployments:
                return None
                
            deployment = self._active_deployments[deployment_id]
            status = deployment.to_dict()
            
            # Add A/B test results if available
            if deployment.ab_test_id:
                try:
                    ab_results = self.ab_test_manager.get_ab_test_results(deployment.ab_test_id)
                    status["ab_test_results"] = ab_results
                except Exception as e:
                    logger.error(f"Failed to get A/B test results: {e}")
                    
            return status
    
    def list_deployments(self, status: Optional[CanaryStatus] = None) -> List[Dict[str, Any]]:
        """List canary deployments"""
        with self._lock:
            deployments = list(self._active_deployments.values())
            
            if status is not None:
                deployments = [d for d in deployments if d.status == status]
                
            return [d.to_dict() for d in sorted(deployments, key=lambda d: d.created_at, reverse=True)]


def create_canary_deployment_system(
    model_manager: ModelVersionManager,
    ab_test_manager: ABTestManager,
    monitor: Optional[ProductionMonitor] = None
) -> CanaryDeploymentManager:
    """
    Create a canary deployment system
    
    Args:
        model_manager: Model version manager
        ab_test_manager: A/B test manager
        monitor: Optional production monitor for rollback decisions
        
    Returns:
        Configured CanaryDeploymentManager
    """
    return CanaryDeploymentManager(model_manager, ab_test_manager, monitor)