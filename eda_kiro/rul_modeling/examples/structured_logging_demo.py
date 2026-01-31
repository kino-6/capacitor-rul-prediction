"""
Structured Logging Demo for RUL Prediction System

This script demonstrates the structured logging capabilities of the RUL prediction system.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from datetime import datetime
import json

from true_rul.structured_logger import (
    configure_prediction_logging, log_batch_prediction_start,
    log_batch_prediction_complete, log_model_loading
)
from true_rul.data_structures import CycleData
from true_rul.rul_predictor import RULPredictor


def main():
    """Demonstrate structured logging functionality"""
    
    # Configure structured logging
    log_file = "logs/demo_predictions.jsonl"
    logger = configure_prediction_logging(
        log_file=log_file,
        enable_console=True
    )
    
    print(f"Structured logging configured. Logs will be written to: {log_file}")
    
    # Log model loading events
    log_model_loading(
        model_type="xgboost",
        model_path="models/rul_model.pkl",
        load_time=2.5,
        success=True
    )
    
    # Create RUL predictor with structured logging
    predictor = RULPredictor(structured_logger=logger)
    
    # Demonstrate batch prediction logging
    capacitor_ids = ["ES12C1", "ES12C2", "ES12C1", "ES12C3"]
    log_batch_prediction_start(
        batch_size=len(capacitor_ids),
        capacitor_ids=capacitor_ids,
        context={"demo": "structured_logging"}
    )
    
    # Simulate some predictions with logging
    success_count = 0
    error_count = 0
    
    for i, cap_id in enumerate(capacitor_ids):
        try:
            # Create sample cycle data
            cycle_data = CycleData(
                cycle_number=i + 1,
                vl_series=np.random.randn(100),
                vo_series=np.random.randn(100),
                timestamp=datetime.now()
            )
            
            # This will fail due to no trained models, but will demonstrate error logging
            predictor.predict_with_error_handling(
                cycle_data=cycle_data,
                capacitor_id=cap_id,
                cycle_history=None
            )
            success_count += 1
            
        except Exception as e:
            print(f"Prediction failed for {cap_id}: {e}")
            error_count += 1
    
    # Log batch completion
    log_batch_prediction_complete(
        batch_size=len(capacitor_ids),
        success_count=success_count,
        error_count=error_count,
        total_time=1.5
    )
    
    print(f"\nDemo completed. Check {log_file} for structured JSON logs.")
    print("Each log entry contains:")
    print("- Timestamp and event type")
    print("- Request/response correlation IDs")
    print("- Performance metrics")
    print("- Error details with stack traces")
    print("- Model metadata and context")


if __name__ == "__main__":
    main()