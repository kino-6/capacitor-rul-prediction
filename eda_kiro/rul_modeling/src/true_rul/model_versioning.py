"""
Model Versioning and A/B Testing Framework

This module provides comprehensive model version management, A/B testing capabilities,
and automated performance comparison for the RUL prediction system.
"""

import asyncio
import json
import logging
import pickle
import hashlib
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
import threading
import uuid

import numpy as np

from .data_structures import PredictionResult, CycleData

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model deployment status"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class ABTestStatus(Enum):
    """A/B test status"""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass
class ModelMetadata:
    """Model metadata and information"""
    model_id: str
    version: str
    name: str
    description: str
    created_at: datetime
    created_by: str
    model_type: str  # "rul_regression", "anomaly_detection", "ensemble"
    framework: str  # "xgboost", "lightgbm", "sklearn", etc.
    status: ModelStatus
    tags: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    training_data_hash: Optional[str] = None
    file_path: Optional[Path] = None
    file_size_bytes: Optional[int] = None
    checksum: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model_id": self.model_id,
            "version": self.version,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "created_by": self.created_by,
            "model_type": self.model_type,
            "framework": self.framework,
            "status": self.status.value,
            "tags": self.tags,
            "metrics": self.metrics,
            "hyperparameters": self.hyperparameters,
            "training_data_hash": self.training_data_hash,
            "file_path": str(self.file_path) if self.file_path else None,
            "file_size_bytes": self.file_size_bytes,
            "checksum": self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary"""
        return cls(
            model_id=data["model_id"],
            version=data["version"],
            name=data["name"],
            description=data["description"],
            created_at=datetime.fromisoformat(data["created_at"]),
            created_by=data["created_by"],
            model_type=data["model_type"],
            framework=data["framework"],
            status=ModelStatus(data["status"]),
            tags=data.get("tags", []),
            metrics=data.get("metrics", {}),
            hyperparameters=data.get("hyperparameters", {}),
            training_data_hash=data.get("training_data_hash"),
            file_path=Path(data["file_path"]) if data.get("file_path") else None,
            file_size_bytes=data.get("file_size_bytes"),
            checksum=data.get("checksum")
        )


@dataclass
class ABTestConfig:
    """A/B test configuration"""
    test_id: str
    name: str
    description: str
    model_a_id: str  # Control model
    model_b_id: str  # Treatment model
    traffic_split: float  # Percentage of traffic to model B (0.0 to 1.0)
    start_time: datetime
    end_time: datetime
    created_by: str
    status: ABTestStatus = ABTestStatus.DRAFT
    success_metrics: List[str] = field(default_factory=lambda: ["fpr_rate", "error_rate", "latency_p95"])
    minimum_sample_size: int = 100
    statistical_significance_threshold: float = 0.05
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "test_id": self.test_id,
            "name": self.name,
            "description": self.description,
            "model_a_id": self.model_a_id,
            "model_b_id": self.model_b_id,
            "traffic_split": self.traffic_split,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "created_by": self.created_by,
            "status": self.status.value,
            "success_metrics": self.success_metrics,
            "minimum_sample_size": self.minimum_sample_size,
            "statistical_significance_threshold": self.statistical_significance_threshold
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ABTestConfig":
        """Create from dictionary"""
        return cls(
            test_id=data["test_id"],
            name=data["name"],
            description=data["description"],
            model_a_id=data["model_a_id"],
            model_b_id=data["model_b_id"],
            traffic_split=data["traffic_split"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]),
            created_by=data["created_by"],
            status=ABTestStatus(data.get("status", "draft")),
            success_metrics=data.get("success_metrics", ["fpr_rate", "error_rate", "latency_p95"]),
            minimum_sample_size=data.get("minimum_sample_size", 100),
            statistical_significance_threshold=data.get("statistical_significance_threshold", 0.05)
        )


@dataclass
class ABTestResult:
    """A/B test result data"""
    test_id: str
    model_id: str
    model_version: str
    prediction_count: int = 0
    total_latency_ms: float = 0.0
    error_count: int = 0
    false_positive_count: int = 0
    anomaly_prediction_count: int = 0
    metrics: Dict[str, List[float]] = field(default_factory=dict)
    
    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency"""
        return self.total_latency_ms / max(1, self.prediction_count)
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate"""
        return self.error_count / max(1, self.prediction_count)
    
    @property
    def fpr_rate(self) -> float:
        """Calculate false positive rate"""
        return self.false_positive_count / max(1, self.anomaly_prediction_count)
    
    def add_prediction_result(self, result: PredictionResult, latency_ms: float, had_error: bool = False):
        """Add a prediction result to the test data"""
        self.prediction_count += 1
        self.total_latency_ms += latency_ms
        
        if had_error:
            self.error_count += 1
            
        if result.anomaly_flag:
            self.anomaly_prediction_count += 1
            # Note: We can't determine false positives without ground truth
            # This would need to be provided externally
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "test_id": self.test_id,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "prediction_count": self.prediction_count,
            "total_latency_ms": self.total_latency_ms,
            "error_count": self.error_count,
            "false_positive_count": self.false_positive_count,
            "anomaly_prediction_count": self.anomaly_prediction_count,
            "average_latency_ms": self.average_latency_ms,
            "error_rate": self.error_rate,
            "fpr_rate": self.fpr_rate,
            "metrics": self.metrics
        }


class ModelStorage(ABC):
    """Abstract base class for model storage"""
    
    @abstractmethod
    def save_model(self, model: Any, metadata: ModelMetadata) -> bool:
        """Save a model with metadata"""
        pass
    
    @abstractmethod
    def load_model(self, model_id: str, version: str) -> Tuple[Any, ModelMetadata]:
        """Load a model by ID and version"""
        pass
    
    @abstractmethod
    def list_models(self, status: Optional[ModelStatus] = None) -> List[ModelMetadata]:
        """List all models, optionally filtered by status"""
        pass
    
    @abstractmethod
    def delete_model(self, model_id: str, version: str) -> bool:
        """Delete a model"""
        pass


class FileSystemModelStorage(ModelStorage):
    """File system-based model storage"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.base_path / "models_metadata.json"
        self._metadata_cache: Dict[str, ModelMetadata] = {}
        self._load_metadata_cache()
        
    def _load_metadata_cache(self):
        """Load metadata cache from file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    
                for key, metadata_dict in data.items():
                    self._metadata_cache[key] = ModelMetadata.from_dict(metadata_dict)
                    
            except Exception as e:
                logger.error(f"Failed to load metadata cache: {e}")
                
    def _save_metadata_cache(self):
        """Save metadata cache to file"""
        try:
            data = {
                key: metadata.to_dict() 
                for key, metadata in self._metadata_cache.items()
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save metadata cache: {e}")
            
    def _get_model_key(self, model_id: str, version: str) -> str:
        """Get cache key for model"""
        return f"{model_id}:{version}"
    
    def _get_model_path(self, model_id: str, version: str) -> Path:
        """Get file path for model"""
        return self.base_path / f"{model_id}_{version}.pkl"
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def save_model(self, model: Any, metadata: ModelMetadata) -> bool:
        """Save a model with metadata"""
        try:
            model_path = self._get_model_path(metadata.model_id, metadata.version)
            
            # Save model using pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
                
            # Update metadata with file information
            metadata.file_path = model_path
            metadata.file_size_bytes = model_path.stat().st_size
            metadata.checksum = self._calculate_checksum(model_path)
            
            # Save metadata
            key = self._get_model_key(metadata.model_id, metadata.version)
            self._metadata_cache[key] = metadata
            self._save_metadata_cache()
            
            logger.info(f"Saved model {metadata.model_id}:{metadata.version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model {metadata.model_id}:{metadata.version}: {e}")
            return False
    
    def load_model(self, model_id: str, version: str) -> Tuple[Any, ModelMetadata]:
        """Load a model by ID and version"""
        key = self._get_model_key(model_id, version)
        
        if key not in self._metadata_cache:
            raise ValueError(f"Model {model_id}:{version} not found")
            
        metadata = self._metadata_cache[key]
        
        if not metadata.file_path or not metadata.file_path.exists():
            raise FileNotFoundError(f"Model file not found: {metadata.file_path}")
            
        try:
            with open(metadata.file_path, 'rb') as f:
                model = pickle.load(f)
                
            logger.info(f"Loaded model {model_id}:{version}")
            return model, metadata
            
        except Exception as e:
            logger.error(f"Failed to load model {model_id}:{version}: {e}")
            raise
    
    def list_models(self, status: Optional[ModelStatus] = None) -> List[ModelMetadata]:
        """List all models, optionally filtered by status"""
        models = list(self._metadata_cache.values())
        
        if status is not None:
            models = [m for m in models if m.status == status]
            
        return sorted(models, key=lambda m: m.created_at, reverse=True)
    
    def delete_model(self, model_id: str, version: str) -> bool:
        """Delete a model"""
        try:
            key = self._get_model_key(model_id, version)
            
            if key not in self._metadata_cache:
                return False
                
            metadata = self._metadata_cache[key]
            
            # Delete model file
            if metadata.file_path and metadata.file_path.exists():
                metadata.file_path.unlink()
                
            # Remove from cache
            del self._metadata_cache[key]
            self._save_metadata_cache()
            
            logger.info(f"Deleted model {model_id}:{version}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}:{version}: {e}")
            return False


class ModelVersionManager:
    """Manages model versions and deployments"""
    
    def __init__(self, storage: ModelStorage):
        self.storage = storage
        self._production_models: Dict[str, str] = {}  # model_type -> version
        self._staging_models: Dict[str, str] = {}
        
    def register_model(self, 
                      model: Any,
                      model_id: str,
                      version: str,
                      name: str,
                      description: str,
                      model_type: str,
                      framework: str,
                      created_by: str,
                      metrics: Optional[Dict[str, float]] = None,
                      hyperparameters: Optional[Dict[str, Any]] = None,
                      tags: Optional[List[str]] = None) -> ModelMetadata:
        """Register a new model version"""
        
        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            name=name,
            description=description,
            created_at=datetime.now(),
            created_by=created_by,
            model_type=model_type,
            framework=framework,
            status=ModelStatus.DEVELOPMENT,
            metrics=metrics or {},
            hyperparameters=hyperparameters or {},
            tags=tags or []
        )
        
        success = self.storage.save_model(model, metadata)
        
        if not success:
            raise RuntimeError(f"Failed to save model {model_id}:{version}")
            
        return metadata
    
    def promote_model(self, model_id: str, version: str, target_status: ModelStatus) -> bool:
        """Promote a model to a different status"""
        try:
            model, metadata = self.storage.load_model(model_id, version)
            
            # Update status
            old_status = metadata.status
            metadata.status = target_status
            
            # Save updated metadata
            success = self.storage.save_model(model, metadata)
            
            if success:
                # Update tracking
                if target_status == ModelStatus.PRODUCTION:
                    self._production_models[metadata.model_type] = f"{model_id}:{version}"
                elif target_status == ModelStatus.STAGING:
                    self._staging_models[metadata.model_type] = f"{model_id}:{version}"
                    
                logger.info(f"Promoted model {model_id}:{version} from {old_status.value} to {target_status.value}")
                
            return success
            
        except Exception as e:
            logger.error(f"Failed to promote model {model_id}:{version}: {e}")
            return False
    
    def rollback_model(self, model_type: str, target_version: Optional[str] = None) -> bool:
        """Rollback to a previous model version"""
        try:
            # Get current production model
            current_key = self._production_models.get(model_type)
            if not current_key:
                logger.error(f"No production model found for type {model_type}")
                return False
                
            current_model_id, current_version = current_key.split(":", 1)
            
            # Find target version
            if target_version is None:
                # Find previous production version
                models = self.storage.list_models(ModelStatus.PRODUCTION)
                type_models = [m for m in models if m.model_type == model_type and m.version != current_version]
                
                if not type_models:
                    logger.error(f"No previous version found for rollback")
                    return False
                    
                target_metadata = type_models[0]  # Most recent
                target_model_id = target_metadata.model_id
                target_version = target_metadata.version
            else:
                target_model_id = current_model_id  # Assume same model ID
                
            # Demote current model
            self.promote_model(current_model_id, current_version, ModelStatus.DEPRECATED)
            
            # Promote target model
            success = self.promote_model(target_model_id, target_version, ModelStatus.PRODUCTION)
            
            if success:
                logger.info(f"Rolled back {model_type} from {current_version} to {target_version}")
                
            return success
            
        except Exception as e:
            logger.error(f"Failed to rollback model {model_type}: {e}")
            return False
    
    def get_production_model(self, model_type: str) -> Tuple[Any, ModelMetadata]:
        """Get the current production model for a type"""
        key = self._production_models.get(model_type)
        if not key:
            raise ValueError(f"No production model found for type {model_type}")
            
        model_id, version = key.split(":", 1)
        return self.storage.load_model(model_id, version)
    
    def list_model_versions(self, model_id: str) -> List[ModelMetadata]:
        """List all versions of a specific model"""
        all_models = self.storage.list_models()
        return [m for m in all_models if m.model_id == model_id]
    
    def compare_models(self, model_a_id: str, model_a_version: str,
                      model_b_id: str, model_b_version: str) -> Dict[str, Any]:
        """Compare two model versions"""
        try:
            _, metadata_a = self.storage.load_model(model_a_id, model_a_version)
            _, metadata_b = self.storage.load_model(model_b_id, model_b_version)
            
            comparison = {
                "model_a": {
                    "id": model_a_id,
                    "version": model_a_version,
                    "metrics": metadata_a.metrics,
                    "created_at": metadata_a.created_at.isoformat(),
                    "status": metadata_a.status.value
                },
                "model_b": {
                    "id": model_b_id,
                    "version": model_b_version,
                    "metrics": metadata_b.metrics,
                    "created_at": metadata_b.created_at.isoformat(),
                    "status": metadata_b.status.value
                },
                "metric_differences": {}
            }
            
            # Compare metrics
            all_metrics = set(metadata_a.metrics.keys()) | set(metadata_b.metrics.keys())
            
            for metric in all_metrics:
                value_a = metadata_a.metrics.get(metric, 0.0)
                value_b = metadata_b.metrics.get(metric, 0.0)
                
                comparison["metric_differences"][metric] = {
                    "model_a": value_a,
                    "model_b": value_b,
                    "difference": value_b - value_a,
                    "percent_change": ((value_b - value_a) / max(abs(value_a), 1e-10)) * 100
                }
                
            return comparison
            
        except Exception as e:
            logger.error(f"Failed to compare models: {e}")
            raise


class ABTestManager:
    """Manages A/B tests for model comparison"""
    
    def __init__(self, model_manager: ModelVersionManager):
        self.model_manager = model_manager
        self._active_tests: Dict[str, ABTestConfig] = {}
        self._test_results: Dict[str, Dict[str, ABTestResult]] = {}  # test_id -> model_id -> results
        self._lock = threading.RLock()
        
    def create_ab_test(self, 
                      name: str,
                      description: str,
                      model_a_id: str,
                      model_a_version: str,
                      model_b_id: str,
                      model_b_version: str,
                      traffic_split: float,
                      duration_hours: int,
                      created_by: str,
                      success_metrics: Optional[List[str]] = None) -> ABTestConfig:
        """Create a new A/B test"""
        
        test_id = str(uuid.uuid4())
        
        config = ABTestConfig(
            test_id=test_id,
            name=name,
            description=description,
            model_a_id=f"{model_a_id}:{model_a_version}",
            model_b_id=f"{model_b_id}:{model_b_version}",
            traffic_split=traffic_split,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(hours=duration_hours),
            created_by=created_by,
            success_metrics=success_metrics or ["fpr_rate", "error_rate", "latency_p95"]
        )
        
        with self._lock:
            self._active_tests[test_id] = config
            self._test_results[test_id] = {
                config.model_a_id: ABTestResult(test_id, model_a_id, model_a_version),
                config.model_b_id: ABTestResult(test_id, model_b_id, model_b_version)
            }
            
        logger.info(f"Created A/B test {test_id}: {name}")
        return config
    
    def start_ab_test(self, test_id: str) -> bool:
        """Start an A/B test"""
        with self._lock:
            if test_id not in self._active_tests:
                return False
                
            config = self._active_tests[test_id]
            config.status = ABTestStatus.RUNNING
            
            logger.info(f"Started A/B test {test_id}")
            return True
    
    def stop_ab_test(self, test_id: str) -> bool:
        """Stop an A/B test"""
        with self._lock:
            if test_id not in self._active_tests:
                return False
                
            config = self._active_tests[test_id]
            config.status = ABTestStatus.COMPLETED
            
            logger.info(f"Stopped A/B test {test_id}")
            return True
    
    def select_model_for_request(self, test_id: str, request_id: Optional[str] = None) -> str:
        """Select which model to use for a request in an A/B test"""
        with self._lock:
            if test_id not in self._active_tests:
                raise ValueError(f"A/B test {test_id} not found")
                
            config = self._active_tests[test_id]
            
            if config.status != ABTestStatus.RUNNING:
                # Default to model A if test is not running
                return config.model_a_id
                
            # Use hash of request_id for consistent assignment
            if request_id:
                hash_value = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
                assignment_value = (hash_value % 100) / 100.0
            else:
                # Random assignment
                assignment_value = np.random.random()
                
            if assignment_value < config.traffic_split:
                return config.model_b_id
            else:
                return config.model_a_id
    
    def record_prediction_result(self, test_id: str, model_key: str, 
                               result: PredictionResult, latency_ms: float, 
                               had_error: bool = False, is_false_positive: bool = False):
        """Record a prediction result for A/B test analysis"""
        with self._lock:
            if test_id not in self._test_results:
                return
                
            if model_key not in self._test_results[test_id]:
                return
                
            test_result = self._test_results[test_id][model_key]
            test_result.add_prediction_result(result, latency_ms, had_error)
            
            if is_false_positive:
                test_result.false_positive_count += 1
    
    def get_ab_test_results(self, test_id: str) -> Dict[str, Any]:
        """Get A/B test results and analysis"""
        with self._lock:
            if test_id not in self._active_tests or test_id not in self._test_results:
                raise ValueError(f"A/B test {test_id} not found")
                
            config = self._active_tests[test_id]
            results = self._test_results[test_id]
            
            model_a_results = results[config.model_a_id]
            model_b_results = results[config.model_b_id]
            
            # Calculate statistical significance (simplified)
            def calculate_significance(metric_a: float, metric_b: float, 
                                     count_a: int, count_b: int) -> Dict[str, Any]:
                """Calculate statistical significance (simplified z-test)"""
                if count_a < 30 or count_b < 30:
                    return {"significant": False, "reason": "insufficient_sample_size"}
                    
                # Simplified z-test for proportions
                pooled_rate = (metric_a * count_a + metric_b * count_b) / (count_a + count_b)
                pooled_se = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/count_a + 1/count_b))
                
                if pooled_se == 0:
                    return {"significant": False, "reason": "zero_variance"}
                    
                z_score = abs(metric_b - metric_a) / pooled_se
                p_value = 2 * (1 - 0.5 * (1 + np.sign(z_score) * np.sqrt(1 - np.exp(-2 * z_score**2 / np.pi))))
                
                return {
                    "significant": p_value < config.statistical_significance_threshold,
                    "p_value": p_value,
                    "z_score": z_score
                }
            
            analysis = {
                "test_config": config.to_dict(),
                "model_a_results": model_a_results.to_dict(),
                "model_b_results": model_b_results.to_dict(),
                "comparison": {},
                "recommendation": "continue"  # Default
            }
            
            # Compare key metrics
            metrics_comparison = {}
            
            # Error rate comparison
            if model_a_results.prediction_count > 0 and model_b_results.prediction_count > 0:
                error_sig = calculate_significance(
                    model_a_results.error_rate, model_b_results.error_rate,
                    model_a_results.prediction_count, model_b_results.prediction_count
                )
                
                metrics_comparison["error_rate"] = {
                    "model_a": model_a_results.error_rate,
                    "model_b": model_b_results.error_rate,
                    "improvement": model_a_results.error_rate - model_b_results.error_rate,
                    "significance": error_sig
                }
                
                # FPR comparison
                if model_a_results.anomaly_prediction_count > 0 and model_b_results.anomaly_prediction_count > 0:
                    fpr_sig = calculate_significance(
                        model_a_results.fpr_rate, model_b_results.fpr_rate,
                        model_a_results.anomaly_prediction_count, model_b_results.anomaly_prediction_count
                    )
                    
                    metrics_comparison["fpr_rate"] = {
                        "model_a": model_a_results.fpr_rate,
                        "model_b": model_b_results.fpr_rate,
                        "improvement": model_a_results.fpr_rate - model_b_results.fpr_rate,
                        "significance": fpr_sig
                    }
                
                # Latency comparison
                latency_improvement = model_a_results.average_latency_ms - model_b_results.average_latency_ms
                metrics_comparison["latency"] = {
                    "model_a": model_a_results.average_latency_ms,
                    "model_b": model_b_results.average_latency_ms,
                    "improvement": latency_improvement
                }
            
            analysis["comparison"] = metrics_comparison
            
            # Generate recommendation
            if any(m.get("significance", {}).get("significant", False) for m in metrics_comparison.values()):
                # Check if model B is better
                error_better = metrics_comparison.get("error_rate", {}).get("improvement", 0) > 0
                fpr_better = metrics_comparison.get("fpr_rate", {}).get("improvement", 0) > 0
                latency_better = metrics_comparison.get("latency", {}).get("improvement", 0) > 0
                
                if error_better and fpr_better:
                    analysis["recommendation"] = "promote_model_b"
                elif not error_better and not fpr_better:
                    analysis["recommendation"] = "keep_model_a"
                else:
                    analysis["recommendation"] = "mixed_results"
            
            return analysis
    
    def list_ab_tests(self, status: Optional[ABTestStatus] = None) -> List[ABTestConfig]:
        """List A/B tests, optionally filtered by status"""
        with self._lock:
            tests = list(self._active_tests.values())
            
            if status is not None:
                tests = [t for t in tests if t.status == status]
                
            return sorted(tests, key=lambda t: t.start_time, reverse=True)


def create_model_versioning_system(storage_path: Path) -> Tuple[ModelVersionManager, ABTestManager]:
    """
    Create a complete model versioning system
    
    Args:
        storage_path: Path for model storage
        
    Returns:
        Tuple of (ModelVersionManager, ABTestManager)
    """
    storage = FileSystemModelStorage(storage_path)
    model_manager = ModelVersionManager(storage)
    ab_test_manager = ABTestManager(model_manager)
    
    return model_manager, ab_test_manager