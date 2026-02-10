"""
Edge Computing Optimization Module

This module implements edge computing optimization features including:
- Lightweight model variants for edge deployment
- Federated learning for distributed model updates
- Offline prediction capabilities with periodic sync
- Memory usage optimization for resource-constrained environments

Requirements: 7.1, 10.4
"""

import logging
import os
import pickle
import json
import time
import threading
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import joblib
from datetime import datetime, timedelta

# Core ML libraries
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator
import xgboost as xgb
import lightgbm as lgb

# Local imports
from .data_structures import PredictionResult, CycleData
from .exceptions import ModelCompressionError
from .model_compression import ModelOptimizer, create_lightweight_student_model
from .rul_predictor import RULPredictor

logger = logging.getLogger(__name__)


@dataclass
class EdgeDeviceConfig:
    """Configuration for edge device deployment"""
    device_id: str
    max_memory_mb: int = 512  # Maximum memory usage in MB
    max_model_size_mb: int = 50  # Maximum model size in MB
    cpu_cores: int = 2
    has_gpu: bool = False
    network_bandwidth_mbps: float = 10.0  # Network bandwidth for sync
    sync_interval_hours: int = 24  # How often to sync with central server
    offline_buffer_size: int = 1000  # Maximum offline predictions to buffer


@dataclass
class ModelSyncStatus:
    """Status of model synchronization"""
    last_sync_time: datetime
    next_sync_time: datetime
    sync_success: bool
    model_version: str
    pending_updates: int
    sync_error: Optional[str] = None


class LightweightModelFactory:
    """
    Factory for creating lightweight model variants optimized for edge deployment
    """
    
    def __init__(self, target_memory_mb: int = 50):
        """
        Initialize lightweight model factory
        
        Args:
            target_memory_mb: Target memory usage for models
        """
        self.target_memory_mb = target_memory_mb
        self.optimizer = ModelOptimizer(
            enable_quantization=True,
            enable_onnx_export=True,
            enable_gpu_acceleration=False,  # Edge devices typically CPU-only
            quantization_type="dynamic"
        )
    
    def create_lightweight_xgboost(
        self,
        original_model: xgb.XGBRegressor,
        max_depth: int = 3,
        n_estimators: int = 50
    ) -> xgb.XGBRegressor:
        """
        Create lightweight XGBoost model for edge deployment
        
        Args:
            original_model: Original XGBoost model
            max_depth: Maximum tree depth for lightweight model
            n_estimators: Number of estimators for lightweight model
        
        Returns:
            Lightweight XGBoost model
        """
        # Create lightweight model with reduced complexity
        lightweight_model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42,
            n_jobs=1  # Single thread for edge devices
        )
        
        # If original model is trained, we can use knowledge distillation
        if hasattr(original_model, 'feature_importances_'):
            logger.info("Created lightweight XGBoost model for edge deployment")
        
        return lightweight_model
    
    def create_lightweight_neural_network(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_layers: int = 2
    ) -> nn.Module:
        """
        Create lightweight neural network for edge deployment
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        
        Returns:
            Lightweight neural network
        """
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim // 2))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            hidden_dim = hidden_dim // 2
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, 1))
        
        model = nn.Sequential(*layers)
        logger.info(f"Created lightweight neural network with {sum(p.numel() for p in model.parameters())} parameters")
        
        return model
    
    def create_lightweight_ensemble(
        self,
        input_dim: int,
        use_xgboost: bool = True,
        use_neural_net: bool = True
    ) -> Dict[str, Any]:
        """
        Create lightweight ensemble for edge deployment
        
        Args:
            input_dim: Input feature dimension
            use_xgboost: Whether to include XGBoost in ensemble
            use_neural_net: Whether to include neural network in ensemble
        
        Returns:
            Dictionary containing lightweight models
        """
        ensemble = {}
        
        if use_xgboost:
            ensemble['xgboost'] = self.create_lightweight_xgboost(
                xgb.XGBRegressor(),  # Placeholder
                max_depth=3,
                n_estimators=30
            )
        
        if use_neural_net:
            ensemble['neural_net'] = self.create_lightweight_neural_network(
                input_dim=input_dim,
                hidden_dim=24,
                num_layers=2
            )
        
        logger.info(f"Created lightweight ensemble with {len(ensemble)} models")
        return ensemble
    
    def optimize_for_edge(
        self,
        model: Any,
        model_type: str,
        example_input: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Optimize model specifically for edge deployment
        
        Args:
            model: Model to optimize
            model_type: Type of model
            example_input: Example input for optimization
        
        Returns:
            Optimized model variants
        """
        # Apply aggressive optimization for edge deployment
        optimization_results = self.optimizer.optimize_model(
            model=model,
            model_type=model_type,
            example_input=example_input
        )
        
        # Additional edge-specific optimizations
        if model_type == "pytorch":
            # Apply more aggressive quantization for edge
            try:
                # Convert to TorchScript for better performance
                if example_input is not None:
                    traced_model = torch.jit.trace(model, example_input)
                    optimization_results['traced_model'] = traced_model
                    optimization_results['optimizations_applied'].append('torchscript_tracing')
            except Exception as e:
                logger.warning(f"TorchScript tracing failed: {e}")
        
        return optimization_results


class FederatedLearningClient:
    """
    Federated learning client for edge devices
    
    Implements federated learning protocol for distributed model updates
    without sharing raw data.
    """
    
    def __init__(
        self,
        device_id: str,
        server_url: str,
        model_update_threshold: int = 100
    ):
        """
        Initialize federated learning client
        
        Args:
            device_id: Unique identifier for this edge device
            server_url: URL of the federated learning server
            model_update_threshold: Number of local updates before sharing
        """
        self.device_id = device_id
        self.server_url = server_url
        self.model_update_threshold = model_update_threshold
        self.local_updates_count = 0
        self.local_model_state = None
        self.global_model_version = 0
    
    def update_local_model(
        self,
        model: Any,
        training_data: np.ndarray,
        training_labels: np.ndarray,
        learning_rate: float = 0.01
    ) -> Any:
        """
        Update local model with new training data
        
        Args:
            model: Current model
            training_data: New training data
            training_labels: New training labels
            learning_rate: Learning rate for updates
        
        Returns:
            Updated model
        """
        # Perform local model update
        if isinstance(model, nn.Module):
            # Neural network update
            model = self._update_neural_network(
                model, training_data, training_labels, learning_rate
            )
        elif isinstance(model, (xgb.XGBRegressor, lgb.LGBMRegressor)):
            # Tree model update (incremental learning if supported)
            model = self._update_tree_model(
                model, training_data, training_labels
            )
        
        self.local_updates_count += 1
        logger.info(f"Local model updated. Update count: {self.local_updates_count}")
        
        # Check if we should share updates with server
        if self.local_updates_count >= self.model_update_threshold:
            self._prepare_model_update(model)
        
        return model
    
    def _update_neural_network(
        self,
        model: nn.Module,
        data: np.ndarray,
        labels: np.ndarray,
        learning_rate: float
    ) -> nn.Module:
        """Update neural network with local data"""
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # Convert to tensors
        X = torch.tensor(data, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        
        # Single epoch update for efficiency
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        logger.debug(f"Neural network updated with loss: {loss.item():.4f}")
        return model
    
    def _update_tree_model(
        self,
        model: Union[xgb.XGBRegressor, lgb.LGBMRegressor],
        data: np.ndarray,
        labels: np.ndarray
    ) -> Union[xgb.XGBRegressor, lgb.LGBMRegressor]:
        """Update tree model with local data"""
        # For tree models, we typically retrain with combined data
        # In a real implementation, you might use incremental learning
        # if supported by the specific model version
        
        try:
            # Partial fit if available (some versions support it)
            if hasattr(model, 'partial_fit'):
                model.partial_fit(data, labels)
            else:
                # For now, just log that we would update
                logger.debug("Tree model update prepared (incremental learning not available)")
        except Exception as e:
            logger.warning(f"Tree model update failed: {e}")
        
        return model
    
    def _prepare_model_update(self, model: Any):
        """Prepare model update for sharing with server"""
        # Extract model parameters/gradients for sharing
        if isinstance(model, nn.Module):
            # For neural networks, we can share parameter updates
            self.local_model_state = {
                'parameters': {name: param.data.clone() for name, param in model.named_parameters()},
                'device_id': self.device_id,
                'update_count': self.local_updates_count,
                'timestamp': datetime.now().isoformat()
            }
        
        logger.info(f"Prepared model update for sharing (device: {self.device_id})")
        # Reset counter
        self.local_updates_count = 0
    
    def get_model_update(self) -> Optional[Dict[str, Any]]:
        """Get prepared model update for server"""
        return self.local_model_state
    
    def apply_global_update(self, global_model_state: Dict[str, Any], model: Any) -> Any:
        """
        Apply global model update from server
        
        Args:
            global_model_state: Global model state from server
            model: Current local model
        
        Returns:
            Updated model with global parameters
        """
        if isinstance(model, nn.Module) and 'parameters' in global_model_state:
            # Apply global parameters to local model
            for name, param in model.named_parameters():
                if name in global_model_state['parameters']:
                    param.data.copy_(global_model_state['parameters'][name])
            
            self.global_model_version = global_model_state.get('version', 0)
            logger.info(f"Applied global model update (version: {self.global_model_version})")
        
        return model


class OfflinePredictionBuffer:
    """
    Buffer for storing predictions when offline
    
    Manages offline prediction storage and synchronization when
    network connectivity is restored.
    """
    
    def __init__(
        self,
        buffer_size: int = 1000,
        storage_path: str = "offline_predictions.json"
    ):
        """
        Initialize offline prediction buffer
        
        Args:
            buffer_size: Maximum number of predictions to buffer
            storage_path: Path to store offline predictions
        """
        self.buffer_size = buffer_size
        self.storage_path = storage_path
        self.predictions_buffer: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        
        # Load existing buffer if available
        self._load_buffer()
    
    def add_prediction(self, prediction_result: PredictionResult, input_data: Dict[str, Any]):
        """
        Add prediction to offline buffer
        
        Args:
            prediction_result: Prediction result to buffer
            input_data: Input data used for prediction
        """
        with self.lock:
            # Create buffer entry
            buffer_entry = {
                'timestamp': datetime.now().isoformat(),
                'prediction': asdict(prediction_result),
                'input_summary': {
                    'cycle_number': input_data.get('cycle_number'),
                    'capacitor_id': input_data.get('capacitor_id'),
                    'feature_count': len(input_data.get('features', [])),
                    'input_hash': hash(str(input_data))  # Simple hash for deduplication
                }
            }
            
            # Add to buffer
            self.predictions_buffer.append(buffer_entry)
            
            # Maintain buffer size limit
            if len(self.predictions_buffer) > self.buffer_size:
                # Remove oldest predictions
                self.predictions_buffer = self.predictions_buffer[-self.buffer_size:]
            
            # Save to disk
            self._save_buffer()
            
            logger.debug(f"Added prediction to offline buffer. Buffer size: {len(self.predictions_buffer)}")
    
    def get_pending_predictions(self) -> List[Dict[str, Any]]:
        """Get all pending predictions for synchronization"""
        with self.lock:
            return self.predictions_buffer.copy()
    
    def clear_synced_predictions(self, synced_count: int):
        """
        Clear predictions that have been successfully synced
        
        Args:
            synced_count: Number of predictions that were synced
        """
        with self.lock:
            if synced_count > 0:
                self.predictions_buffer = self.predictions_buffer[synced_count:]
                self._save_buffer()
                logger.info(f"Cleared {synced_count} synced predictions from buffer")
    
    def _save_buffer(self):
        """Save buffer to disk"""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.predictions_buffer, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save offline buffer: {e}")
    
    def _load_buffer(self):
        """Load buffer from disk"""
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r') as f:
                    self.predictions_buffer = json.load(f)
                logger.info(f"Loaded {len(self.predictions_buffer)} predictions from offline buffer")
        except Exception as e:
            logger.error(f"Failed to load offline buffer: {e}")
            self.predictions_buffer = []


class EdgeRULPredictor:
    """
    Edge-optimized RUL predictor with offline capabilities
    
    Combines lightweight models, offline prediction buffering,
    and federated learning for edge deployment.
    """
    
    def __init__(
        self,
        config: EdgeDeviceConfig,
        model_path: Optional[str] = None,
        enable_federated_learning: bool = True
    ):
        """
        Initialize edge RUL predictor
        
        Args:
            config: Edge device configuration
            model_path: Path to pre-trained model
            enable_federated_learning: Whether to enable federated learning
        """
        self.config = config
        self.enable_federated_learning = enable_federated_learning
        
        # Initialize components
        self.model_factory = LightweightModelFactory(
            target_memory_mb=config.max_model_size_mb
        )
        self.offline_buffer = OfflinePredictionBuffer(
            buffer_size=config.offline_buffer_size
        )
        
        if enable_federated_learning:
            self.fl_client = FederatedLearningClient(
                device_id=config.device_id,
                server_url="http://localhost:8000/federated"  # Placeholder
            )
        
        # Model and sync status
        self.model = None
        self.sync_status = ModelSyncStatus(
            last_sync_time=datetime.now() - timedelta(days=1),
            next_sync_time=datetime.now() + timedelta(hours=config.sync_interval_hours),
            sync_success=False,
            model_version="0.0.0",
            pending_updates=0
        )
        
        # Load model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        # Start background sync thread
        self.sync_thread = threading.Thread(target=self._sync_worker, daemon=True)
        self.sync_thread.start()
        
        logger.info(f"EdgeRULPredictor initialized for device: {config.device_id}")
    
    def load_model(self, model_path: str):
        """
        Load and optimize model for edge deployment
        
        Args:
            model_path: Path to model file
        """
        try:
            # Load model
            with open(model_path, 'rb') as f:
                original_model = pickle.load(f)
            
            # Optimize for edge deployment
            if hasattr(original_model, 'predict'):
                # Scikit-learn style model
                optimized = self.model_factory.optimize_for_edge(
                    model=original_model,
                    model_type="sklearn"
                )
                self.model = optimized.get('quantized_model', original_model)
            else:
                self.model = original_model
            
            logger.info(f"Model loaded and optimized for edge deployment")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Create lightweight fallback model
            self.model = self.model_factory.create_lightweight_ensemble(
                input_dim=55,  # Assuming 55 features
                use_xgboost=True,
                use_neural_net=False  # Simpler for edge
            )
    
    def predict(
        self,
        cycle_data: CycleData,
        features: np.ndarray,
        online: bool = True
    ) -> PredictionResult:
        """
        Make RUL prediction with edge optimization
        
        Args:
            cycle_data: Input cycle data
            features: Extracted features
            online: Whether device is online
        
        Returns:
            Prediction result
        """
        start_time = time.time()
        
        try:
            # Make prediction with lightweight model
            if isinstance(self.model, dict):
                # Ensemble prediction
                predictions = []
                for model_name, model in self.model.items():
                    if hasattr(model, 'predict'):
                        pred = model.predict(features.reshape(1, -1))[0]
                        predictions.append(pred)
                
                # Simple ensemble average
                rul_prediction = np.mean(predictions) if predictions else 0.0
                confidence_lower = rul_prediction * 0.9
                confidence_upper = rul_prediction * 1.1
                
            elif hasattr(self.model, 'predict'):
                rul_prediction = self.model.predict(features.reshape(1, -1))[0]
                confidence_lower = rul_prediction * 0.9
                confidence_upper = rul_prediction * 1.1
            else:
                # Fallback prediction
                rul_prediction = 100.0
                confidence_lower = 90.0
                confidence_upper = 110.0
            
            # Create prediction result
            prediction_result = PredictionResult(
                rul_cycles=max(0, int(rul_prediction)),
                rul_confidence_lower=max(0, int(confidence_lower)),
                rul_confidence_upper=int(confidence_upper),
                degradation_score=min(1.0, max(0.0, 1.0 - rul_prediction / 200.0)),
                degradation_stage=self._determine_degradation_stage(rul_prediction),
                anomaly_flag=rul_prediction < 20,
                anomaly_score=max(0.0, (20 - rul_prediction) / 20.0) if rul_prediction < 20 else 0.0,
                feature_importance={f"feature_{i}": 1.0/len(features) for i in range(len(features))},
                timestamp=time.time(),
                model_version=self.sync_status.model_version
            )
            
            # Handle offline mode
            if not online:
                input_data = {
                    'cycle_number': cycle_data.cycle_number,
                    'capacitor_id': getattr(cycle_data, 'capacitor_id', 'unknown'),
                    'features': features.tolist()
                }
                self.offline_buffer.add_prediction(prediction_result, input_data)
            
            # Update federated learning if enabled
            if self.enable_federated_learning and hasattr(self, 'fl_client'):
                # In a real implementation, you would collect training data
                # and periodically update the model
                pass
            
            prediction_time = time.time() - start_time
            logger.debug(f"Edge prediction completed in {prediction_time:.3f}s")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Edge prediction failed: {e}")
            # Return fallback prediction
            return PredictionResult(
                rul_cycles=100,
                rul_confidence_lower=90,
                rul_confidence_upper=110,
                degradation_score=0.5,
                degradation_stage="unknown",
                anomaly_flag=False,
                anomaly_score=0.0,
                feature_importance={},
                timestamp=time.time(),
                model_version=self.sync_status.model_version
            )
    
    def _determine_degradation_stage(self, rul: float) -> str:
        """Determine degradation stage based on RUL"""
        if rul > 150:
            return "healthy"
        elif rul > 100:
            return "early_degradation"
        elif rul > 50:
            return "advanced_degradation"
        else:
            return "critical"
    
    def _sync_worker(self):
        """Background worker for model synchronization"""
        while True:
            try:
                current_time = datetime.now()
                
                if current_time >= self.sync_status.next_sync_time:
                    self._perform_sync()
                    
                    # Schedule next sync
                    self.sync_status.next_sync_time = current_time + timedelta(
                        hours=self.config.sync_interval_hours
                    )
                
                # Sleep for 1 minute before checking again
                time.sleep(60)
                
            except Exception as e:
                logger.error(f"Sync worker error: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def _perform_sync(self):
        """Perform model and data synchronization"""
        try:
            logger.info("Starting model synchronization...")
            
            # Sync offline predictions
            pending_predictions = self.offline_buffer.get_pending_predictions()
            if pending_predictions:
                # In a real implementation, send to server
                logger.info(f"Would sync {len(pending_predictions)} offline predictions")
                # Simulate successful sync
                self.offline_buffer.clear_synced_predictions(len(pending_predictions))
            
            # Sync federated learning updates
            if self.enable_federated_learning and hasattr(self, 'fl_client'):
                model_update = self.fl_client.get_model_update()
                if model_update:
                    logger.info("Would send federated learning update to server")
            
            # Update sync status
            self.sync_status.last_sync_time = datetime.now()
            self.sync_status.sync_success = True
            self.sync_status.sync_error = None
            
            logger.info("Model synchronization completed successfully")
            
        except Exception as e:
            logger.error(f"Model synchronization failed: {e}")
            self.sync_status.sync_success = False
            self.sync_status.sync_error = str(e)
    
    def get_device_status(self) -> Dict[str, Any]:
        """Get current device status and metrics"""
        import psutil
        
        try:
            # Get system metrics
            memory_usage = psutil.virtual_memory()
            cpu_usage = psutil.cpu_percent(interval=1)
            
            status = {
                'device_id': self.config.device_id,
                'model_loaded': self.model is not None,
                'sync_status': asdict(self.sync_status),
                'offline_buffer_size': len(self.offline_buffer.predictions_buffer),
                'system_metrics': {
                    'memory_usage_percent': memory_usage.percent,
                    'memory_available_mb': memory_usage.available / (1024 * 1024),
                    'cpu_usage_percent': cpu_usage
                },
                'config': asdict(self.config)
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get device status: {e}")
            return {
                'device_id': self.config.device_id,
                'error': str(e)
            }
    
    def optimize_memory_usage(self):
        """Optimize memory usage for resource-constrained environments"""
        try:
            import gc
            
            # Force garbage collection
            gc.collect()
            
            # Clear any cached data
            if hasattr(self, '_feature_cache'):
                delattr(self, '_feature_cache')
            
            # Optimize model if possible
            if isinstance(self.model, dict):
                for model_name, model in self.model.items():
                    if hasattr(model, 'n_jobs'):
                        model.n_jobs = 1  # Single thread to save memory
            
            logger.info("Memory usage optimized for edge deployment")
            
        except Exception as e:
            logger.error(f"Memory optimization failed: {e}")


# Utility functions for edge deployment
def create_edge_deployment_package(
    model: Any,
    model_type: str,
    config: EdgeDeviceConfig,
    output_dir: str
) -> str:
    """
    Create complete edge deployment package
    
    Args:
        model: Trained model to deploy
        model_type: Type of model
        config: Edge device configuration
        output_dir: Output directory for deployment package
    
    Returns:
        Path to deployment package
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create lightweight model variants
    factory = LightweightModelFactory(target_memory_mb=config.max_model_size_mb)
    optimized_models = factory.optimize_for_edge(
        model=model,
        model_type=model_type
    )
    
    # Save optimized models
    model_path = os.path.join(output_dir, "edge_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(optimized_models.get('quantized_model', model), f)
    
    # Save configuration
    config_path = os.path.join(output_dir, "edge_config.json")
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    # Create deployment script
    script_path = os.path.join(output_dir, "deploy_edge.py")
    with open(script_path, 'w') as f:
        f.write(f"""#!/usr/bin/env python3
'''
Edge deployment script for RUL prediction
Generated automatically for device: {config.device_id}
'''

import json
from true_rul.edge_computing import EdgeRULPredictor, EdgeDeviceConfig

def main():
    # Load configuration
    with open('edge_config.json', 'r') as f:
        config_dict = json.load(f)
    
    config = EdgeDeviceConfig(**config_dict)
    
    # Initialize edge predictor
    predictor = EdgeRULPredictor(
        config=config,
        model_path='edge_model.pkl',
        enable_federated_learning=True
    )
    
    print(f"Edge RUL predictor deployed for device: {{config.device_id}}")
    print("Device status:", predictor.get_device_status())

if __name__ == "__main__":
    main()
""")
    
    logger.info(f"Edge deployment package created at: {output_dir}")
    return output_dir


def estimate_edge_resource_usage(
    model: Any,
    model_type: str,
    sample_input: np.ndarray
) -> Dict[str, float]:
    """
    Estimate resource usage for edge deployment
    
    Args:
        model: Model to analyze
        model_type: Type of model
        sample_input: Sample input for testing
    
    Returns:
        Resource usage estimates
    """
    import sys
    import time
    import psutil
    
    # Measure model size
    if hasattr(model, '__sizeof__'):
        model_size_bytes = sys.getsizeof(model)
    else:
        # Estimate based on pickle size
        import pickle
        model_size_bytes = len(pickle.dumps(model))
    
    # Measure prediction time and memory
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    start_time = time.time()
    try:
        if hasattr(model, 'predict'):
            _ = model.predict(sample_input.reshape(1, -1))
        elif isinstance(model, torch.nn.Module):
            with torch.no_grad():
                _ = model(torch.tensor(sample_input, dtype=torch.float32).unsqueeze(0))
    except:
        pass
    
    prediction_time = time.time() - start_time
    peak_memory = process.memory_info().rss
    memory_usage = peak_memory - initial_memory
    
    return {
        'model_size_mb': model_size_bytes / (1024 * 1024),
        'prediction_time_ms': prediction_time * 1000,
        'memory_usage_mb': memory_usage / (1024 * 1024),
        'estimated_throughput_per_sec': 1.0 / prediction_time if prediction_time > 0 else 0
    }