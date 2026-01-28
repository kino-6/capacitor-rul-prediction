"""
Configuration settings for True RUL Prediction System
"""

import logging
from pathlib import Path
from typing import Dict, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_LEVEL = logging.INFO

def setup_logging(log_file: str = "true_rul.log", level: int = LOG_LEVEL) -> None:
    """
    Set up logging configuration
    
    Args:
        log_file: Name of the log file
        level: Logging level
    """
    log_path = LOGS_DIR / log_file
    
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )

# Model configuration
MODEL_CONFIG: Dict[str, Any] = {
    "rul_model": {
        "type": "ensemble",  # "xgboost", "lightgbm", "random_forest", "ensemble"
        "xgboost": {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        },
        "lightgbm": {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        },
        "random_forest": {
            "n_estimators": 300,
            "max_depth": 15,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
        },
        "ensemble_weights": {
            "xgboost": 0.4,
            "lightgbm": 0.4,
            "random_forest": 0.2,
        },
    },
    "anomaly_detection": {
        "isolation_forest": {
            "contamination": 0.05,
            "random_state": 42,
        },
        "ocsvm": {
            "kernel": "rbf",
            "nu": 0.05,
            "gamma": "auto",
        },
        "ensemble_weights": {
            "isolation_forest": 0.35,
            "autoencoder": 0.40,
            "ocsvm": 0.25,
        },
    },
    "feature_extraction": {
        "include_advanced": True,
        "rolling_window": 5,
        "normalization": "standard",  # "standard" or "minmax"
    },
    "training": {
        "test_size": 0.2,
        "val_size": 0.1,
        "random_state": 42,
        "early_stopping_rounds": 50,
    },
}

# ES12 dataset configuration
ES12_CONFIG: Dict[str, Any] = {
    "capacitor_ids": [f"ES12C{i}" for i in range(1, 9)],
    "normal_cycles": (1, 10),  # Cycles assumed to be normal
    "total_cycles": 200,
}

# API configuration
API_CONFIG: Dict[str, Any] = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": False,
    "workers": 4,
    "timeout": 60,
}
