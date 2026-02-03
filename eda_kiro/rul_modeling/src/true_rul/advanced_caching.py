"""
Advanced Caching and Optimization Module

This module implements advanced caching strategies and optimization techniques
for production deployment including Redis-based feature caching, model prediction
caching with TTL, batch processing optimization, and asynchronous processing.

Requirements: 10.2, 10.4
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import wraps, lru_cache
import numpy as np

# Redis for distributed caching
try:
    import redis
    import redis.asyncio as aioredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available. Install with: pip install redis")

# Async support
import asyncio
from asyncio import Queue

# Local imports
from .data_structures import PredictionResult, CycleData
from .exceptions import CachingError

logger = logging.getLogger(__name__)


class FeatureCache:
    """
    Redis-based feature caching system
    
    Caches extracted features to avoid recomputation for repeated
    inputs or similar voltage patterns.
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        default_ttl: int = 3600,  # 1 hour
        max_memory_cache: int = 1000
    ):
        """
        Initialize feature cache
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            default_ttl: Default TTL for cached features (seconds)
            max_memory_cache: Maximum items in memory cache
        """
        self.default_ttl = default_ttl
        self.max_memory_cache = max_memory_cache
        
        # Memory cache as fallback
        self.memory_cache = {}
        self.cache_access_times = {}
        
        # Redis connection
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    decode_responses=False  # Keep binary for numpy arrays
                )
                # Test connection
                self.redis_client.ping()
                self.redis_available = True
                logger.info(f"Connected to Redis at {redis_host}:{redis_port}")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Using memory cache only.")
                self.redis_available = False
        else:
            self.redis_available = False
    
    def _generate_cache_key(self, cycle_data: CycleData, feature_config: Dict[str, Any]) -> str:
        """
        Generate cache key for cycle data and feature configuration
        
        Args:
            cycle_data: Input cycle data
            feature_config: Feature extraction configuration
        
        Returns:
            Cache key string
        """
        # Create hash from voltage data and config
        vl_hash = hashlib.md5(cycle_data.vl_series.tobytes()).hexdigest()[:8]
        vo_hash = hashlib.md5(cycle_data.vo_series.tobytes()).hexdigest()[:8]
        config_hash = hashlib.md5(
            json.dumps(feature_config, sort_keys=True).encode()
        ).hexdigest()[:8]
        
        return f"features:{vl_hash}:{vo_hash}:{config_hash}"
    
    def get_features(
        self,
        cycle_data: CycleData,
        feature_config: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """
        Get cached features for cycle data
        
        Args:
            cycle_data: Input cycle data
            feature_config: Feature extraction configuration
        
        Returns:
            Cached features or None if not found
        """
        cache_key = self._generate_cache_key(cycle_data, feature_config)
        
        # Try Redis first
        if self.redis_available:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    features = pickle.loads(cached_data)
                    logger.debug(f"Features cache hit (Redis): {cache_key}")
                    return features
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")
        
        # Try memory cache
        if cache_key in self.memory_cache:
            self.cache_access_times[cache_key] = time.time()
            logger.debug(f"Features cache hit (memory): {cache_key}")
            return self.memory_cache[cache_key]
        
        logger.debug(f"Features cache miss: {cache_key}")
        return None
    
    def set_features(
        self,
        cycle_data: CycleData,
        feature_config: Dict[str, Any],
        features: np.ndarray,
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache features for cycle data
        
        Args:
            cycle_data: Input cycle data
            feature_config: Feature extraction configuration
            features: Extracted features to cache
            ttl: Time to live (seconds), uses default if None
        """
        cache_key = self._generate_cache_key(cycle_data, feature_config)
        ttl = ttl or self.default_ttl
        
        # Store in Redis
        if self.redis_available:
            try:
                serialized_features = pickle.dumps(features)
                self.redis_client.setex(cache_key, ttl, serialized_features)
                logger.debug(f"Features cached in Redis: {cache_key}")
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")
        
        # Store in memory cache
        self._manage_memory_cache_size()
        self.memory_cache[cache_key] = features
        self.cache_access_times[cache_key] = time.time()
        logger.debug(f"Features cached in memory: {cache_key}")
    
    def _manage_memory_cache_size(self) -> None:
        """Manage memory cache size by removing least recently used items"""
        if len(self.memory_cache) >= self.max_memory_cache:
            # Remove 20% of least recently used items
            items_to_remove = int(self.max_memory_cache * 0.2)
            sorted_items = sorted(
                self.cache_access_times.items(),
                key=lambda x: x[1]
            )
            
            for cache_key, _ in sorted_items[:items_to_remove]:
                self.memory_cache.pop(cache_key, None)
                self.cache_access_times.pop(cache_key, None)
    
    def clear_cache(self) -> None:
        """Clear all cached features"""
        if self.redis_available:
            try:
                # Clear only feature keys
                keys = self.redis_client.keys("features:*")
                if keys:
                    self.redis_client.delete(*keys)
                logger.info("Cleared Redis feature cache")
            except Exception as e:
                logger.warning(f"Redis clear failed: {e}")
        
        self.memory_cache.clear()
        self.cache_access_times.clear()
        logger.info("Cleared memory feature cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        stats = {
            "memory_cache_size": len(self.memory_cache),
            "max_memory_cache": self.max_memory_cache,
            "redis_available": self.redis_available
        }
        
        if self.redis_available:
            try:
                info = self.redis_client.info()
                stats.update({
                    "redis_used_memory": info.get("used_memory_human", "unknown"),
                    "redis_connected_clients": info.get("connected_clients", 0)
                })
            except Exception as e:
                logger.warning(f"Failed to get Redis stats: {e}")
        
        return stats


class PredictionCache:
    """
    Model prediction caching with TTL
    
    Caches model predictions to avoid recomputation for identical
    or very similar inputs.
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 1,
        default_ttl: int = 1800,  # 30 minutes
        similarity_threshold: float = 0.95
    ):
        """
        Initialize prediction cache
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            default_ttl: Default TTL for cached predictions (seconds)
            similarity_threshold: Threshold for considering inputs similar
        """
        self.default_ttl = default_ttl
        self.similarity_threshold = similarity_threshold
        
        # Redis connection
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    decode_responses=False
                )
                self.redis_client.ping()
                self.redis_available = True
                logger.info(f"Prediction cache connected to Redis at {redis_host}:{redis_port}")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. Prediction caching disabled.")
                self.redis_available = False
        else:
            self.redis_available = False
    
    def _generate_prediction_key(
        self,
        features: np.ndarray,
        model_version: str,
        model_config: Dict[str, Any]
    ) -> str:
        """
        Generate cache key for prediction
        
        Args:
            features: Input features
            model_version: Model version identifier
            model_config: Model configuration
        
        Returns:
            Cache key string
        """
        features_hash = hashlib.md5(features.tobytes()).hexdigest()[:12]
        config_hash = hashlib.md5(
            json.dumps(model_config, sort_keys=True).encode()
        ).hexdigest()[:8]
        
        return f"prediction:{model_version}:{features_hash}:{config_hash}"
    
    def get_prediction(
        self,
        features: np.ndarray,
        model_version: str,
        model_config: Dict[str, Any]
    ) -> Optional[PredictionResult]:
        """
        Get cached prediction
        
        Args:
            features: Input features
            model_version: Model version identifier
            model_config: Model configuration
        
        Returns:
            Cached prediction result or None if not found
        """
        if not self.redis_available:
            return None
        
        cache_key = self._generate_prediction_key(features, model_version, model_config)
        
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                prediction_result = pickle.loads(cached_data)
                logger.debug(f"Prediction cache hit: {cache_key}")
                return prediction_result
        except Exception as e:
            logger.warning(f"Prediction cache get failed: {e}")
        
        logger.debug(f"Prediction cache miss: {cache_key}")
        return None
    
    def set_prediction(
        self,
        features: np.ndarray,
        model_version: str,
        model_config: Dict[str, Any],
        prediction_result: PredictionResult,
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache prediction result
        
        Args:
            features: Input features
            model_version: Model version identifier
            model_config: Model configuration
            prediction_result: Prediction result to cache
            ttl: Time to live (seconds), uses default if None
        """
        if not self.redis_available:
            return
        
        cache_key = self._generate_prediction_key(features, model_version, model_config)
        ttl = ttl or self.default_ttl
        
        try:
            serialized_result = pickle.dumps(prediction_result)
            self.redis_client.setex(cache_key, ttl, serialized_result)
            logger.debug(f"Prediction cached: {cache_key}")
        except Exception as e:
            logger.warning(f"Prediction cache set failed: {e}")
    
    def clear_predictions(self, model_version: Optional[str] = None) -> None:
        """
        Clear cached predictions
        
        Args:
            model_version: Clear only predictions for specific model version,
                          or all if None
        """
        if not self.redis_available:
            return
        
        try:
            if model_version:
                pattern = f"prediction:{model_version}:*"
            else:
                pattern = "prediction:*"
            
            keys = self.redis_client.keys(pattern)
            if keys:
                self.redis_client.delete(*keys)
            
            logger.info(f"Cleared prediction cache for pattern: {pattern}")
        except Exception as e:
            logger.warning(f"Prediction cache clear failed: {e}")


class BatchProcessor:
    """
    Optimized batch processing for multiple predictions
    
    Implements efficient batching strategies, parallel processing,
    and memory management for high-throughput scenarios.
    """
    
    def __init__(
        self,
        max_batch_size: int = 64,
        max_workers: int = 4,
        use_multiprocessing: bool = False,
        memory_limit_mb: int = 1024
    ):
        """
        Initialize batch processor
        
        Args:
            max_batch_size: Maximum batch size for processing
            max_workers: Maximum number of worker threads/processes
            use_multiprocessing: Use multiprocessing instead of threading
            memory_limit_mb: Memory limit in MB for batch processing
        """
        self.max_batch_size = max_batch_size
        self.max_workers = max_workers
        self.use_multiprocessing = use_multiprocessing
        self.memory_limit_bytes = memory_limit_mb * 1024 * 1024
        
        # Initialize executor
        if use_multiprocessing:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        logger.info(
            f"Batch processor initialized: batch_size={max_batch_size}, "
            f"workers={max_workers}, multiprocessing={use_multiprocessing}"
        )
    
    def process_batch(
        self,
        prediction_func: Callable,
        inputs: List[Any],
        batch_size: Optional[int] = None
    ) -> List[Any]:
        """
        Process inputs in batches
        
        Args:
            prediction_func: Function to apply to each batch
            inputs: List of inputs to process
            batch_size: Batch size override
        
        Returns:
            List of results
        """
        batch_size = batch_size or self.max_batch_size
        results = []
        
        # Process in batches
        for i in range(0, len(inputs), batch_size):
            batch = inputs[i:i + batch_size]
            
            # Check memory usage
            batch_memory = self._estimate_batch_memory(batch)
            if batch_memory > self.memory_limit_bytes:
                # Reduce batch size
                reduced_batch_size = max(1, len(batch) // 2)
                logger.warning(
                    f"Batch too large ({batch_memory / 1024 / 1024:.1f}MB), "
                    f"reducing size to {reduced_batch_size}"
                )
                # Recursively process smaller batches
                sub_results = self.process_batch(
                    prediction_func, batch, reduced_batch_size
                )
                results.extend(sub_results)
            else:
                # Process batch
                batch_results = prediction_func(batch)
                if isinstance(batch_results, list):
                    results.extend(batch_results)
                else:
                    results.append(batch_results)
        
        return results
    
    def process_parallel(
        self,
        prediction_func: Callable,
        inputs: List[Any],
        batch_size: Optional[int] = None
    ) -> List[Any]:
        """
        Process inputs in parallel batches
        
        Args:
            prediction_func: Function to apply to each batch
            inputs: List of inputs to process
            batch_size: Batch size override
        
        Returns:
            List of results
        """
        batch_size = batch_size or self.max_batch_size
        
        # Create batches
        batches = [
            inputs[i:i + batch_size]
            for i in range(0, len(inputs), batch_size)
        ]
        
        # Submit all batches to executor
        futures = [
            self.executor.submit(prediction_func, batch)
            for batch in batches
        ]
        
        # Collect results
        results = []
        for future in futures:
            try:
                batch_results = future.result(timeout=300)  # 5 minute timeout
                if isinstance(batch_results, list):
                    results.extend(batch_results)
                else:
                    results.append(batch_results)
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Add None for failed batch
                results.extend([None] * len(batches[futures.index(future)]))
        
        return results
    
    def _estimate_batch_memory(self, batch: List[Any]) -> int:
        """
        Estimate memory usage of a batch
        
        Args:
            batch: Batch of inputs
        
        Returns:
            Estimated memory usage in bytes
        """
        if not batch:
            return 0
        
        # Simple estimation based on first item
        first_item = batch[0]
        if isinstance(first_item, np.ndarray):
            item_size = first_item.nbytes
        elif hasattr(first_item, '__sizeof__'):
            item_size = first_item.__sizeof__()
        else:
            # Rough estimate
            item_size = 1024  # 1KB default
        
        return len(batch) * item_size * 2  # Factor of 2 for processing overhead
    
    def __del__(self):
        """Cleanup executor on deletion"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


class AsyncProcessor:
    """
    Asynchronous processing for non-critical tasks
    
    Handles background tasks like logging, monitoring, and
    non-critical computations without blocking main prediction flow.
    """
    
    def __init__(self, max_queue_size: int = 1000):
        """
        Initialize async processor
        
        Args:
            max_queue_size: Maximum size of task queue
        """
        self.max_queue_size = max_queue_size
        self.task_queue = Queue(maxsize=max_queue_size)
        self.is_running = False
        self.worker_task = None
        
        logger.info(f"Async processor initialized with queue size: {max_queue_size}")
    
    async def start(self) -> None:
        """Start the async processor"""
        if self.is_running:
            return
        
        self.is_running = True
        self.worker_task = asyncio.create_task(self._worker())
        logger.info("Async processor started")
    
    async def stop(self) -> None:
        """Stop the async processor"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Add sentinel to stop worker
        await self.task_queue.put(None)
        
        if self.worker_task:
            await self.worker_task
        
        logger.info("Async processor stopped")
    
    async def submit_task(
        self,
        func: Callable,
        *args,
        priority: int = 0,
        **kwargs
    ) -> bool:
        """
        Submit a task for async processing
        
        Args:
            func: Function to execute
            *args: Function arguments
            priority: Task priority (higher = more important)
            **kwargs: Function keyword arguments
        
        Returns:
            True if task was queued, False if queue is full
        """
        task = {
            'func': func,
            'args': args,
            'kwargs': kwargs,
            'priority': priority,
            'timestamp': time.time()
        }
        
        try:
            await self.task_queue.put(task)
            return True
        except asyncio.QueueFull:
            logger.warning("Async task queue is full, dropping task")
            return False
    
    async def _worker(self) -> None:
        """Worker coroutine that processes tasks from the queue"""
        while self.is_running:
            try:
                # Get task from queue
                task = await self.task_queue.get()
                
                # Check for sentinel (stop signal)
                if task is None:
                    break
                
                # Execute task
                try:
                    func = task['func']
                    args = task['args']
                    kwargs = task['kwargs']
                    
                    if asyncio.iscoroutinefunction(func):
                        await func(*args, **kwargs)
                    else:
                        # Run in thread pool for blocking functions
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(None, func, *args, **kwargs)
                
                except Exception as e:
                    logger.error(f"Async task failed: {e}")
                
                finally:
                    self.task_queue.task_done()
            
            except Exception as e:
                logger.error(f"Async worker error: {e}")
    
    def submit_sync(self, func: Callable, *args, **kwargs) -> bool:
        """
        Submit task from synchronous code
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
        
        Returns:
            True if task was queued, False otherwise
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create task in running loop
                asyncio.create_task(self.submit_task(func, *args, **kwargs))
                return True
            else:
                # Run in new event loop
                return asyncio.run(self.submit_task(func, *args, **kwargs))
        except Exception as e:
            logger.error(f"Failed to submit sync task: {e}")
            return False


class CacheManager:
    """
    Unified cache management interface
    
    Coordinates all caching systems and provides a single interface
    for cache operations.
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        enable_feature_cache: bool = True,
        enable_prediction_cache: bool = True,
        feature_cache_ttl: int = 3600,
        prediction_cache_ttl: int = 1800
    ):
        """
        Initialize cache manager
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            enable_feature_cache: Enable feature caching
            enable_prediction_cache: Enable prediction caching
            feature_cache_ttl: TTL for feature cache
            prediction_cache_ttl: TTL for prediction cache
        """
        self.enable_feature_cache = enable_feature_cache
        self.enable_prediction_cache = enable_prediction_cache
        
        # Initialize caches
        if enable_feature_cache:
            self.feature_cache = FeatureCache(
                redis_host=redis_host,
                redis_port=redis_port,
                redis_db=0,
                default_ttl=feature_cache_ttl
            )
        
        if enable_prediction_cache:
            self.prediction_cache = PredictionCache(
                redis_host=redis_host,
                redis_port=redis_port,
                redis_db=1,
                default_ttl=prediction_cache_ttl
            )
        
        logger.info("Cache manager initialized")
    
    def get_cached_features(
        self,
        cycle_data: CycleData,
        feature_config: Dict[str, Any]
    ) -> Optional[np.ndarray]:
        """Get cached features"""
        if not self.enable_feature_cache:
            return None
        
        return self.feature_cache.get_features(cycle_data, feature_config)
    
    def cache_features(
        self,
        cycle_data: CycleData,
        feature_config: Dict[str, Any],
        features: np.ndarray
    ) -> None:
        """Cache extracted features"""
        if not self.enable_feature_cache:
            return
        
        self.feature_cache.set_features(cycle_data, feature_config, features)
    
    def get_cached_prediction(
        self,
        features: np.ndarray,
        model_version: str,
        model_config: Dict[str, Any]
    ) -> Optional[PredictionResult]:
        """Get cached prediction"""
        if not self.enable_prediction_cache:
            return None
        
        return self.prediction_cache.get_prediction(features, model_version, model_config)
    
    def cache_prediction(
        self,
        features: np.ndarray,
        model_version: str,
        model_config: Dict[str, Any],
        prediction_result: PredictionResult
    ) -> None:
        """Cache prediction result"""
        if not self.enable_prediction_cache:
            return
        
        self.prediction_cache.set_prediction(
            features, model_version, model_config, prediction_result
        )
    
    def clear_all_caches(self) -> None:
        """Clear all caches"""
        if self.enable_feature_cache:
            self.feature_cache.clear_cache()
        
        if self.enable_prediction_cache:
            self.prediction_cache.clear_predictions()
        
        logger.info("All caches cleared")
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get statistics for all caches"""
        stats = {}
        
        if self.enable_feature_cache:
            stats['feature_cache'] = self.feature_cache.get_cache_stats()
        
        if self.enable_prediction_cache:
            stats['prediction_cache'] = {
                'redis_available': self.prediction_cache.redis_available
            }
        
        return stats


# Decorator for automatic caching
def cached_prediction(
    cache_manager: CacheManager,
    model_version: str,
    model_config: Dict[str, Any],
    ttl: Optional[int] = None
):
    """
    Decorator for automatic prediction caching
    
    Args:
        cache_manager: Cache manager instance
        model_version: Model version identifier
        model_config: Model configuration
        ttl: Cache TTL override
    
    Returns:
        Decorated function with caching
    """
    def decorator(func):
        @wraps(func)
        def wrapper(features: np.ndarray, *args, **kwargs):
            # Try to get from cache
            cached_result = cache_manager.get_cached_prediction(
                features, model_version, model_config
            )
            
            if cached_result is not None:
                return cached_result
            
            # Compute result
            result = func(features, *args, **kwargs)
            
            # Cache result
            if isinstance(result, PredictionResult):
                cache_manager.cache_prediction(
                    features, model_version, model_config, result
                )
            
            return result
        
        return wrapper
    return decorator


# Global instances for easy access
_global_cache_manager = None
_global_batch_processor = None
_global_async_processor = None


def get_cache_manager(**kwargs) -> CacheManager:
    """Get global cache manager instance"""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager(**kwargs)
    return _global_cache_manager


def get_batch_processor(**kwargs) -> BatchProcessor:
    """Get global batch processor instance"""
    global _global_batch_processor
    if _global_batch_processor is None:
        _global_batch_processor = BatchProcessor(**kwargs)
    return _global_batch_processor


def get_async_processor(**kwargs) -> AsyncProcessor:
    """Get global async processor instance"""
    global _global_async_processor
    if _global_async_processor is None:
        _global_async_processor = AsyncProcessor(**kwargs)
    return _global_async_processor