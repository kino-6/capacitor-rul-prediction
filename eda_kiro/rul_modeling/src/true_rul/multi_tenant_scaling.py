"""
Multi-Tenant Support and Scaling Module

This module implements multi-tenant support and horizontal scaling features including:
- Tenant isolation for multi-customer deployments
- Horizontal scaling with load balancing
- Resource allocation and quota management
- Tenant-specific model customization

Requirements: 10.1, 10.4
"""

import logging
import os
import json
import time
import threading
import hashlib
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from collections import defaultdict
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Core libraries
import numpy as np
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None
    
try:
    from fastapi import HTTPException
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    # Create a simple HTTPException replacement
    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            self.status_code = status_code
            self.detail = detail
            super().__init__(f"HTTP {status_code}: {detail}")

import psutil

# Local imports
from .data_structures import PredictionResult, CycleData
from .exceptions import ModelCompressionError
from .rul_predictor import RULPredictor

logger = logging.getLogger(__name__)


@dataclass
class TenantConfig:
    """Configuration for a tenant"""
    tenant_id: str
    tenant_name: str
    max_requests_per_hour: int = 1000
    max_concurrent_requests: int = 10
    max_memory_mb: int = 1024
    max_cpu_cores: int = 4
    model_customization_enabled: bool = True
    data_retention_days: int = 30
    priority_level: int = 1  # 1=low, 2=medium, 3=high
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    is_active: bool = True


@dataclass
class ResourceQuota:
    """Resource quota for a tenant"""
    tenant_id: str
    cpu_limit_cores: float = 2.0
    memory_limit_mb: int = 1024
    storage_limit_gb: int = 10
    requests_per_hour: int = 1000
    concurrent_requests: int = 10
    model_training_hours_per_month: int = 10
    
    # Current usage
    cpu_usage_cores: float = 0.0
    memory_usage_mb: int = 0
    storage_usage_gb: float = 0.0
    requests_this_hour: int = 0
    active_requests: int = 0
    training_hours_this_month: float = 0.0
    
    # Timestamps
    last_reset_time: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class TenantMetrics:
    """Metrics for tenant monitoring"""
    tenant_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time_ms: float = 0.0
    total_predictions: int = 0
    total_training_time_hours: float = 0.0
    last_request_time: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)


class TenantIsolationManager:
    """
    Manages tenant isolation and resource allocation
    
    Ensures that tenants are properly isolated and resources
    are allocated according to their quotas.
    """
    
    def __init__(self, redis_client: Optional[Any] = None):
        """
        Initialize tenant isolation manager
        
        Args:
            redis_client: Redis client for distributed state management
        """
        self.redis_client = redis_client
        self.tenants: Dict[str, TenantConfig] = {}
        self.quotas: Dict[str, ResourceQuota] = {}
        self.metrics: Dict[str, TenantMetrics] = {}
        self.tenant_models: Dict[str, Any] = {}  # Tenant-specific models
        self.active_requests: Dict[str, List[str]] = defaultdict(list)  # Track active requests per tenant
        self.lock = threading.RLock()
        
        # Start background quota reset thread
        self.quota_reset_thread = threading.Thread(target=self._quota_reset_worker, daemon=True)
        self.quota_reset_thread.start()
        
        logger.info("TenantIsolationManager initialized")
    
    def register_tenant(
        self,
        tenant_config: TenantConfig,
        resource_quota: Optional[ResourceQuota] = None
    ) -> str:
        """
        Register a new tenant
        
        Args:
            tenant_config: Tenant configuration
            resource_quota: Resource quota (optional, defaults will be used)
        
        Returns:
            Tenant ID
        """
        with self.lock:
            tenant_id = tenant_config.tenant_id
            
            # Store tenant configuration
            self.tenants[tenant_id] = tenant_config
            
            # Set up resource quota
            if resource_quota is None:
                resource_quota = ResourceQuota(
                    tenant_id=tenant_id,
                    cpu_limit_cores=tenant_config.max_cpu_cores,
                    memory_limit_mb=tenant_config.max_memory_mb,
                    requests_per_hour=tenant_config.max_requests_per_hour,
                    concurrent_requests=tenant_config.max_concurrent_requests
                )
            
            self.quotas[tenant_id] = resource_quota
            
            # Initialize metrics
            self.metrics[tenant_id] = TenantMetrics(tenant_id=tenant_id)
            
            # Store in Redis if available
            if self.redis_client:
                self.redis_client.hset(
                    "tenants",
                    tenant_id,
                    json.dumps(asdict(tenant_config), default=str)
                )
                self.redis_client.hset(
                    "quotas",
                    tenant_id,
                    json.dumps(asdict(resource_quota), default=str)
                )
            
            logger.info(f"Registered tenant: {tenant_id} ({tenant_config.tenant_name})")
            return tenant_id
    
    def get_tenant(self, tenant_id: str) -> Optional[TenantConfig]:
        """Get tenant configuration"""
        with self.lock:
            if tenant_id in self.tenants:
                return self.tenants[tenant_id]
            
            # Try to load from Redis
            if self.redis_client:
                tenant_data = self.redis_client.hget("tenants", tenant_id)
                if tenant_data:
                    tenant_dict = json.loads(tenant_data)
                    # Convert datetime strings back to datetime objects
                    for field in ['created_at', 'last_active']:
                        if field in tenant_dict:
                            tenant_dict[field] = datetime.fromisoformat(tenant_dict[field])
                    
                    tenant_config = TenantConfig(**tenant_dict)
                    self.tenants[tenant_id] = tenant_config
                    return tenant_config
            
            return None
    
    def check_resource_quota(self, tenant_id: str, request_id: str) -> bool:
        """
        Check if tenant can make a request within quota limits
        
        Args:
            tenant_id: Tenant identifier
            request_id: Unique request identifier
        
        Returns:
            True if request is allowed, False otherwise
        """
        with self.lock:
            if tenant_id not in self.quotas:
                logger.warning(f"No quota found for tenant: {tenant_id}")
                return False
            
            quota = self.quotas[tenant_id]
            
            # Check concurrent requests limit
            if quota.active_requests >= quota.concurrent_requests:
                logger.warning(f"Tenant {tenant_id} exceeded concurrent requests limit")
                return False
            
            # Check hourly requests limit
            if quota.requests_this_hour >= quota.requests_per_hour:
                logger.warning(f"Tenant {tenant_id} exceeded hourly requests limit")
                return False
            
            # Check memory usage
            if quota.memory_usage_mb >= quota.memory_limit_mb:
                logger.warning(f"Tenant {tenant_id} exceeded memory limit")
                return False
            
            # Check CPU usage
            if quota.cpu_usage_cores >= quota.cpu_limit_cores:
                logger.warning(f"Tenant {tenant_id} exceeded CPU limit")
                return False
            
            # Reserve resources for this request
            quota.active_requests += 1
            quota.requests_this_hour += 1
            quota.last_updated = datetime.now()
            
            # Track active request
            self.active_requests[tenant_id].append(request_id)
            
            return True
    
    def release_request_resources(self, tenant_id: str, request_id: str):
        """
        Release resources after request completion
        
        Args:
            tenant_id: Tenant identifier
            request_id: Request identifier to release
        """
        with self.lock:
            if tenant_id in self.quotas:
                quota = self.quotas[tenant_id]
                quota.active_requests = max(0, quota.active_requests - 1)
                quota.last_updated = datetime.now()
            
            # Remove from active requests
            if tenant_id in self.active_requests:
                if request_id in self.active_requests[tenant_id]:
                    self.active_requests[tenant_id].remove(request_id)
    
    def update_resource_usage(
        self,
        tenant_id: str,
        cpu_usage: float = 0.0,
        memory_usage: int = 0,
        storage_usage: float = 0.0
    ):
        """
        Update resource usage for a tenant
        
        Args:
            tenant_id: Tenant identifier
            cpu_usage: CPU usage in cores
            memory_usage: Memory usage in MB
            storage_usage: Storage usage in GB
        """
        with self.lock:
            if tenant_id in self.quotas:
                quota = self.quotas[tenant_id]
                quota.cpu_usage_cores = cpu_usage
                quota.memory_usage_mb = memory_usage
                quota.storage_usage_gb = storage_usage
                quota.last_updated = datetime.now()
    
    def get_tenant_metrics(self, tenant_id: str) -> Optional[TenantMetrics]:
        """Get metrics for a tenant"""
        with self.lock:
            return self.metrics.get(tenant_id)
    
    def update_tenant_metrics(
        self,
        tenant_id: str,
        request_success: bool,
        response_time_ms: float,
        prediction_count: int = 1
    ):
        """
        Update tenant metrics
        
        Args:
            tenant_id: Tenant identifier
            request_success: Whether the request was successful
            response_time_ms: Response time in milliseconds
            prediction_count: Number of predictions made
        """
        with self.lock:
            if tenant_id not in self.metrics:
                self.metrics[tenant_id] = TenantMetrics(tenant_id=tenant_id)
            
            metrics = self.metrics[tenant_id]
            metrics.total_requests += 1
            
            if request_success:
                metrics.successful_requests += 1
            else:
                metrics.failed_requests += 1
            
            # Update average response time
            total_time = metrics.average_response_time_ms * (metrics.total_requests - 1)
            metrics.average_response_time_ms = (total_time + response_time_ms) / metrics.total_requests
            
            metrics.total_predictions += prediction_count
            metrics.last_request_time = datetime.now()
    
    def _quota_reset_worker(self):
        """Background worker to reset hourly quotas"""
        while True:
            try:
                current_time = datetime.now()
                
                with self.lock:
                    for tenant_id, quota in self.quotas.items():
                        # Reset hourly quota if an hour has passed
                        if current_time - quota.last_reset_time >= timedelta(hours=1):
                            quota.requests_this_hour = 0
                            quota.last_reset_time = current_time
                            logger.debug(f"Reset hourly quota for tenant: {tenant_id}")
                
                # Sleep for 5 minutes before next check
                time.sleep(300)
                
            except Exception as e:
                logger.error(f"Quota reset worker error: {e}")
                time.sleep(60)
    
    def get_all_tenant_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all tenants"""
        with self.lock:
            status = {}
            
            for tenant_id in self.tenants.keys():
                tenant_config = self.tenants.get(tenant_id)
                quota = self.quotas.get(tenant_id)
                metrics = self.metrics.get(tenant_id)
                
                status[tenant_id] = {
                    'config': asdict(tenant_config) if tenant_config else None,
                    'quota': asdict(quota) if quota else None,
                    'metrics': asdict(metrics) if metrics else None,
                    'active_requests': len(self.active_requests.get(tenant_id, []))
                }
            
            return status


class LoadBalancer:
    """
    Load balancer for horizontal scaling
    
    Distributes requests across multiple worker instances
    based on tenant priority and resource availability.
    """
    
    def __init__(
        self,
        num_workers: int = None,
        balancing_strategy: str = "round_robin"
    ):
        """
        Initialize load balancer
        
        Args:
            num_workers: Number of worker processes (defaults to CPU count)
            balancing_strategy: Load balancing strategy ("round_robin", "least_connections", "weighted")
        """
        self.num_workers = num_workers or mp.cpu_count()
        self.balancing_strategy = balancing_strategy
        self.worker_pool = ProcessPoolExecutor(max_workers=self.num_workers)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.num_workers * 2)
        
        # Worker state tracking
        self.worker_connections = [0] * self.num_workers
        self.worker_last_used = [time.time()] * self.num_workers
        self.current_worker = 0
        self.lock = threading.Lock()
        
        logger.info(f"LoadBalancer initialized with {self.num_workers} workers")
    
    def select_worker(self, tenant_priority: int = 1) -> int:
        """
        Select worker based on balancing strategy
        
        Args:
            tenant_priority: Priority level of the tenant (1=low, 2=medium, 3=high)
        
        Returns:
            Worker index
        """
        with self.lock:
            if self.balancing_strategy == "round_robin":
                worker_id = self.current_worker
                self.current_worker = (self.current_worker + 1) % self.num_workers
                return worker_id
            
            elif self.balancing_strategy == "least_connections":
                # Select worker with least active connections
                worker_id = min(range(self.num_workers), key=lambda i: self.worker_connections[i])
                return worker_id
            
            elif self.balancing_strategy == "weighted":
                # Weight selection by tenant priority and worker load
                weights = []
                for i in range(self.num_workers):
                    # Higher priority tenants get preference for less loaded workers
                    load_factor = 1.0 / (1.0 + self.worker_connections[i])
                    priority_factor = tenant_priority / 3.0  # Normalize to 0-1
                    weight = load_factor * priority_factor
                    weights.append(weight)
                
                # Select worker with highest weight
                worker_id = max(range(self.num_workers), key=lambda i: weights[i])
                return worker_id
            
            else:
                # Default to round robin
                return self.current_worker
    
    def submit_prediction_task(
        self,
        predictor_func,
        tenant_id: str,
        tenant_priority: int,
        *args,
        **kwargs
    ):
        """
        Submit prediction task to worker pool
        
        Args:
            predictor_func: Prediction function to execute
            tenant_id: Tenant identifier
            tenant_priority: Tenant priority level
            *args: Arguments for prediction function
            **kwargs: Keyword arguments for prediction function
        
        Returns:
            Future object for the prediction task
        """
        worker_id = self.select_worker(tenant_priority)
        
        with self.lock:
            self.worker_connections[worker_id] += 1
            self.worker_last_used[worker_id] = time.time()
        
        # Submit to thread pool for I/O bound tasks
        future = self.thread_pool.submit(
            self._execute_prediction_task,
            worker_id,
            predictor_func,
            tenant_id,
            *args,
            **kwargs
        )
        
        return future
    
    def _execute_prediction_task(
        self,
        worker_id: int,
        predictor_func,
        tenant_id: str,
        *args,
        **kwargs
    ):
        """Execute prediction task and update worker state"""
        try:
            # Execute the prediction
            result = predictor_func(*args, **kwargs)
            return result
        
        finally:
            # Update worker connection count
            with self.lock:
                self.worker_connections[worker_id] = max(0, self.worker_connections[worker_id] - 1)
    
    def get_worker_status(self) -> Dict[str, Any]:
        """Get status of all workers"""
        with self.lock:
            return {
                'num_workers': self.num_workers,
                'balancing_strategy': self.balancing_strategy,
                'worker_connections': self.worker_connections.copy(),
                'worker_last_used': self.worker_last_used.copy(),
                'total_active_connections': sum(self.worker_connections)
            }
    
    def shutdown(self):
        """Shutdown worker pools"""
        self.worker_pool.shutdown(wait=True)
        self.thread_pool.shutdown(wait=True)
        logger.info("LoadBalancer shutdown completed")


class TenantModelManager:
    """
    Manages tenant-specific model customization
    
    Allows tenants to have customized models while maintaining
    isolation and resource limits.
    """
    
    def __init__(self, base_model_path: str):
        """
        Initialize tenant model manager
        
        Args:
            base_model_path: Path to base model for customization
        """
        self.base_model_path = base_model_path
        self.tenant_models: Dict[str, Any] = {}
        self.model_training_status: Dict[str, str] = {}  # "training", "ready", "failed"
        self.lock = threading.RLock()
        
        logger.info("TenantModelManager initialized")
    
    def create_tenant_model(
        self,
        tenant_id: str,
        customization_config: Dict[str, Any]
    ) -> str:
        """
        Create customized model for tenant
        
        Args:
            tenant_id: Tenant identifier
            customization_config: Model customization configuration
        
        Returns:
            Model creation job ID
        """
        job_id = f"model_job_{tenant_id}_{int(time.time())}"
        
        with self.lock:
            self.model_training_status[job_id] = "training"
        
        # Start model customization in background
        threading.Thread(
            target=self._customize_model_worker,
            args=(job_id, tenant_id, customization_config),
            daemon=True
        ).start()
        
        logger.info(f"Started model customization for tenant {tenant_id}, job: {job_id}")
        return job_id
    
    def _customize_model_worker(
        self,
        job_id: str,
        tenant_id: str,
        config: Dict[str, Any]
    ):
        """Background worker for model customization"""
        try:
            # Load base model
            base_model = self._load_base_model()
            
            # Apply customizations
            customized_model = self._apply_customizations(base_model, config)
            
            # Store customized model
            with self.lock:
                self.tenant_models[tenant_id] = customized_model
                self.model_training_status[job_id] = "ready"
            
            logger.info(f"Model customization completed for tenant {tenant_id}")
            
        except Exception as e:
            logger.error(f"Model customization failed for tenant {tenant_id}: {e}")
            with self.lock:
                self.model_training_status[job_id] = "failed"
    
    def _load_base_model(self):
        """Load base model for customization"""
        # Placeholder - in real implementation, load actual model
        return {"type": "base_model", "parameters": {}}
    
    def _apply_customizations(self, base_model: Any, config: Dict[str, Any]) -> Any:
        """Apply customizations to base model"""
        # Placeholder - in real implementation, apply actual customizations
        customized_model = base_model.copy()
        customized_model["customizations"] = config
        
        # Simulate training time
        time.sleep(2)
        
        return customized_model
    
    def get_tenant_model(self, tenant_id: str) -> Optional[Any]:
        """Get customized model for tenant"""
        with self.lock:
            return self.tenant_models.get(tenant_id)
    
    def get_model_status(self, job_id: str) -> Optional[str]:
        """Get model training status"""
        with self.lock:
            return self.model_training_status.get(job_id)
    
    def list_tenant_models(self) -> Dict[str, Any]:
        """List all tenant models and their status"""
        with self.lock:
            return {
                'tenant_models': list(self.tenant_models.keys()),
                'training_jobs': self.model_training_status.copy()
            }


class MultiTenantRULService:
    """
    Multi-tenant RUL prediction service
    
    Combines tenant isolation, load balancing, and model management
    into a unified service for multi-tenant deployments.
    """
    
    def __init__(
        self,
        base_model_path: str,
        redis_url: Optional[str] = None,
        num_workers: int = None
    ):
        """
        Initialize multi-tenant RUL service
        
        Args:
            base_model_path: Path to base RUL model
            redis_url: Redis URL for distributed state (optional)
            num_workers: Number of worker processes
        """
        # Initialize Redis client if URL provided
        self.redis_client = None
        if redis_url and REDIS_AVAILABLE:
            try:
                self.redis_client = redis.from_url(redis_url)
                self.redis_client.ping()  # Test connection
                logger.info(f"Connected to Redis: {redis_url}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
        elif redis_url and not REDIS_AVAILABLE:
            logger.warning("Redis URL provided but redis module not available")
        
        # Initialize components
        self.tenant_manager = TenantIsolationManager(self.redis_client)
        self.load_balancer = LoadBalancer(num_workers=num_workers)
        self.model_manager = TenantModelManager(base_model_path)
        
        # Base RUL predictor
        self.base_predictor = RULPredictor()
        
        logger.info("MultiTenantRULService initialized")
    
    def register_tenant(
        self,
        tenant_name: str,
        max_requests_per_hour: int = 1000,
        max_concurrent_requests: int = 10,
        max_memory_mb: int = 1024,
        max_cpu_cores: int = 4,
        priority_level: int = 1
    ) -> str:
        """
        Register a new tenant
        
        Args:
            tenant_name: Name of the tenant
            max_requests_per_hour: Maximum requests per hour
            max_concurrent_requests: Maximum concurrent requests
            max_memory_mb: Maximum memory usage in MB
            max_cpu_cores: Maximum CPU cores
            priority_level: Priority level (1=low, 2=medium, 3=high)
        
        Returns:
            Tenant ID
        """
        tenant_id = f"tenant_{hashlib.md5(tenant_name.encode()).hexdigest()[:8]}"
        
        tenant_config = TenantConfig(
            tenant_id=tenant_id,
            tenant_name=tenant_name,
            max_requests_per_hour=max_requests_per_hour,
            max_concurrent_requests=max_concurrent_requests,
            max_memory_mb=max_memory_mb,
            max_cpu_cores=max_cpu_cores,
            priority_level=priority_level
        )
        
        return self.tenant_manager.register_tenant(tenant_config)
    
    async def predict_async(
        self,
        tenant_id: str,
        cycle_data: CycleData,
        features: np.ndarray
    ) -> PredictionResult:
        """
        Make async RUL prediction for tenant
        
        Args:
            tenant_id: Tenant identifier
            cycle_data: Input cycle data
            features: Extracted features
        
        Returns:
            Prediction result
        """
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Check tenant exists and is active
            tenant_config = self.tenant_manager.get_tenant(tenant_id)
            if not tenant_config or not tenant_config.is_active:
                raise HTTPException(status_code=404, detail="Tenant not found or inactive")
            
            # Check resource quota
            if not self.tenant_manager.check_resource_quota(tenant_id, request_id):
                raise HTTPException(status_code=429, detail="Resource quota exceeded")
            
            try:
                # Get tenant-specific model or use base model
                tenant_model = self.model_manager.get_tenant_model(tenant_id)
                predictor = tenant_model if tenant_model else self.base_predictor
                
                # Submit prediction task to load balancer
                future = self.load_balancer.submit_prediction_task(
                    self._make_prediction,
                    tenant_id,
                    tenant_config.priority_level,
                    predictor,
                    cycle_data,
                    features
                )
                
                # Wait for result
                result = future.result(timeout=30)  # 30 second timeout
                
                # Update metrics
                response_time = (time.time() - start_time) * 1000
                self.tenant_manager.update_tenant_metrics(
                    tenant_id=tenant_id,
                    request_success=True,
                    response_time_ms=response_time
                )
                
                return result
                
            finally:
                # Always release resources
                self.tenant_manager.release_request_resources(tenant_id, request_id)
        
        except Exception as e:
            # Update failure metrics
            response_time = (time.time() - start_time) * 1000
            self.tenant_manager.update_tenant_metrics(
                tenant_id=tenant_id,
                request_success=False,
                response_time_ms=response_time
            )
            raise e
    
    def _make_prediction(
        self,
        predictor: Any,
        cycle_data: CycleData,
        features: np.ndarray
    ) -> PredictionResult:
        """Make prediction using specified predictor"""
        if hasattr(predictor, 'predict'):
            return predictor.predict(cycle_data, features)
        else:
            # Fallback prediction for custom models
            return PredictionResult(
                rul_cycles=100,
                rul_confidence_lower=90,
                rul_confidence_upper=110,
                degradation_score=0.5,
                degradation_stage="healthy",  # Use valid stage
                anomaly_flag=False,
                anomaly_score=0.0,
                feature_importance={},
                timestamp=time.time(),
                model_version="custom"
            )
    
    def create_tenant_model(
        self,
        tenant_id: str,
        customization_config: Dict[str, Any]
    ) -> str:
        """Create customized model for tenant"""
        tenant_config = self.tenant_manager.get_tenant(tenant_id)
        if not tenant_config:
            raise HTTPException(status_code=404, detail="Tenant not found")
        
        if not tenant_config.model_customization_enabled:
            raise HTTPException(status_code=403, detail="Model customization not enabled for tenant")
        
        return self.model_manager.create_tenant_model(tenant_id, customization_config)
    
    def get_tenant_status(self, tenant_id: str) -> Dict[str, Any]:
        """Get comprehensive status for a tenant"""
        tenant_config = self.tenant_manager.get_tenant(tenant_id)
        if not tenant_config:
            raise HTTPException(status_code=404, detail="Tenant not found")
        
        quota = self.tenant_manager.quotas.get(tenant_id)
        metrics = self.tenant_manager.get_tenant_metrics(tenant_id)
        tenant_model = self.model_manager.get_tenant_model(tenant_id)
        
        return {
            'tenant_config': asdict(tenant_config),
            'resource_quota': asdict(quota) if quota else None,
            'metrics': asdict(metrics) if metrics else None,
            'has_custom_model': tenant_model is not None,
            'load_balancer_status': self.load_balancer.get_worker_status()
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        return {
            'tenant_count': len(self.tenant_manager.tenants),
            'active_tenants': sum(1 for t in self.tenant_manager.tenants.values() if t.is_active),
            'total_active_requests': sum(len(reqs) for reqs in self.tenant_manager.active_requests.values()),
            'load_balancer_status': self.load_balancer.get_worker_status(),
            'model_manager_status': self.model_manager.list_tenant_models(),
            'redis_connected': self.redis_client is not None
        }
    
    def shutdown(self):
        """Shutdown the multi-tenant service"""
        self.load_balancer.shutdown()
        logger.info("MultiTenantRULService shutdown completed")


# Utility functions for multi-tenant deployment
def create_tenant_deployment_config(
    tenant_name: str,
    resource_limits: Dict[str, Any],
    customization_options: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create deployment configuration for a tenant
    
    Args:
        tenant_name: Name of the tenant
        resource_limits: Resource limits configuration
        customization_options: Model customization options
    
    Returns:
        Complete deployment configuration
    """
    return {
        'tenant_name': tenant_name,
        'resource_limits': resource_limits,
        'customization_options': customization_options,
        'deployment_timestamp': datetime.now().isoformat(),
        'api_endpoints': {
            'predict': f'/api/v1/tenants/{tenant_name}/predict',
            'batch_predict': f'/api/v1/tenants/{tenant_name}/batch_predict',
            'status': f'/api/v1/tenants/{tenant_name}/status',
            'models': f'/api/v1/tenants/{tenant_name}/models'
        }
    }


def monitor_tenant_resources(
    tenant_manager: TenantIsolationManager,
    interval_seconds: int = 60
):
    """
    Monitor tenant resource usage
    
    Args:
        tenant_manager: Tenant isolation manager
        interval_seconds: Monitoring interval in seconds
    """
    def monitor_worker():
        while True:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                
                # Update resource usage for all tenants
                for tenant_id in tenant_manager.tenants.keys():
                    # In a real implementation, you would track per-tenant usage
                    # For now, we'll distribute system usage across active tenants
                    active_tenants = len([t for t in tenant_manager.tenants.values() if t.is_active])
                    if active_tenants > 0:
                        tenant_cpu = cpu_percent / active_tenants
                        tenant_memory = (memory.used / (1024 * 1024)) / active_tenants
                        
                        tenant_manager.update_resource_usage(
                            tenant_id=tenant_id,
                            cpu_usage=tenant_cpu / 100.0,  # Convert to cores
                            memory_usage=int(tenant_memory)
                        )
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                time.sleep(interval_seconds)
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_worker, daemon=True)
    monitor_thread.start()
    logger.info(f"Started tenant resource monitoring (interval: {interval_seconds}s)")