"""
Configuration settings for True RUL Prediction System
"""

from pathlib import Path
from typing import Dict, List, Any
import os

# ES12 Dataset Configuration
ES12_CONFIG = {
    "capacitor_ids": ["ES12C1", "ES12C2", "ES12C3", "ES12C4", 
                      "ES12C5", "ES12C6", "ES12C7", "ES12C8"],
    "max_cycles": 200,
    "normal_cycles_range": (1, 10),
    "voltage_range": {
        "vl_min": 0.0,
        "vl_max": 10.0,
        "vo_min": 0.0,
        "vo_max": 10.0
    }
}

# Model Configuration
MODEL_CONFIG = {
    "rul_model_type": "ensemble",  # "xgboost", "lightgbm", "random_forest", "elastic_net", "ensemble"
    "anomaly_model_type": "ensemble",  # "isolation_forest", "autoencoder", "ocsvm", "ensemble"
    "feature_normalization": "standard",  # "standard", "minmax"
    "confidence_method": "ensemble",  # "ensemble", "quantile"
    "random_state": 42
}

# Feature Extraction Configuration
FEATURE_CONFIG = {
    "responsiveness_features": True,
    "statistical_features": True,
    "frequency_features": True,
    "trend_features": True,
    "rolling_features": True,
    "rolling_window": 5,
    "fft_components": 10
}

# Training Configuration
TRAINING_CONFIG = {
    "test_capacitors": ["ES12C7", "ES12C8"],
    "validation_split": 0.2,
    "early_stopping_rounds": 50,
    "max_epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001
}

# Production Pipeline Configuration
PRODUCTION_CONFIG = {
    "buffer_size": 10000,
    "batch_size": 10,
    "max_wait_time": 30.0,
    "max_workers": 4,
    "cycle_length_threshold": 1000,
    "voltage_change_threshold": 0.1,
    "data_quality_threshold": 0.95
}

# API Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 4,
    "timeout": 60,
    "max_request_size": 10 * 1024 * 1024,  # 10MB
    "cors_origins": ["*"],
    "api_key_required": False
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_path": "logs/rul_prediction.log",
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
    "structured_logging": True
}

# Monitoring Configuration
MONITORING_CONFIG = {
    "metrics_enabled": True,
    "prometheus_port": 9090,
    "health_check_interval": 30,
    "performance_tracking": True,
    "alert_thresholds": {
        "fpr_threshold": 0.05,
        "prediction_latency_ms": 1000,
        "data_quality_threshold": 0.95,
        "error_rate_threshold": 0.01
    }
}

# Paths Configuration
def get_paths_config(base_path: Path = None) -> Dict[str, Path]:
    """Get paths configuration"""
    if base_path is None:
        base_path = Path(__file__).parent.parent.parent
    
    return {
        "base_path": base_path,
        "data_path": base_path / "data",
        "models_path": base_path / "models",
        "logs_path": base_path / "logs",
        "output_path": base_path / "output",
        "notebooks_path": base_path / "notebooks",
        "tests_path": base_path / "tests",
        "es12_data_path": base_path / "data" / "ES12.mat"
    }

# Environment-specific overrides
def load_config_from_env() -> Dict[str, Any]:
    """Load configuration overrides from environment variables"""
    config_overrides = {}
    
    # API configuration
    if os.getenv("RUL_API_HOST"):
        config_overrides["api_host"] = os.getenv("RUL_API_HOST")
    if os.getenv("RUL_API_PORT"):
        config_overrides["api_port"] = int(os.getenv("RUL_API_PORT"))
    
    # Model configuration
    if os.getenv("RUL_MODEL_TYPE"):
        config_overrides["rul_model_type"] = os.getenv("RUL_MODEL_TYPE")
    if os.getenv("ANOMALY_MODEL_TYPE"):
        config_overrides["anomaly_model_type"] = os.getenv("ANOMALY_MODEL_TYPE")
    
    # Production configuration
    if os.getenv("BATCH_SIZE"):
        config_overrides["batch_size"] = int(os.getenv("BATCH_SIZE"))
    if os.getenv("BUFFER_SIZE"):
        config_overrides["buffer_size"] = int(os.getenv("BUFFER_SIZE"))
    
    # Logging configuration
    if os.getenv("LOG_LEVEL"):
        config_overrides["log_level"] = os.getenv("LOG_LEVEL")
    
    return config_overrides

# Combined configuration
def get_config() -> Dict[str, Any]:
    """Get complete configuration with environment overrides"""
    config = {
        "es12": ES12_CONFIG,
        "model": MODEL_CONFIG,
        "features": FEATURE_CONFIG,
        "training": TRAINING_CONFIG,
        "production": PRODUCTION_CONFIG,
        "api": API_CONFIG,
        "logging": LOGGING_CONFIG,
        "monitoring": MONITORING_CONFIG,
        "paths": get_paths_config()
    }
    
    # Apply environment overrides
    env_overrides = load_config_from_env()
    for key, value in env_overrides.items():
        if key.startswith("api_"):
            config["api"][key[4:]] = value
        elif key in ["rul_model_type", "anomaly_model_type"]:
            config["model"][key] = value
        elif key in ["batch_size", "buffer_size"]:
            config["production"][key] = value
        elif key == "log_level":
            config["logging"]["level"] = value
    
    return config