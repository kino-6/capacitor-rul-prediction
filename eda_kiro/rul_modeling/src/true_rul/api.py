"""
FastAPI REST API for RUL Prediction System
High-performance API with parallel processing and caching
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
import time

from .data_structures import CycleData, PredictionResult
from .rul_predictor import RULPredictor
from .structured_logger import configure_prediction_logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model caching
_rul_predictor: Optional[RULPredictor] = None
_model_cache: Dict[str, Any] = {}
_executor = ThreadPoolExecutor(max_workers=8)  # Use 8 threads for parallel processing

# Pydantic models
class VoltageData(BaseModel):
    vl_series: List[float] = Field(..., description="Input voltage time series")
    vo_series: List[float] = Field(..., description="Output voltage time series")

class PredictionRequest(BaseModel):
    capacitor_id: str = Field(..., description="Capacitor identifier")
    cycle_number: int = Field(..., ge=1, description="Cycle number")
    voltage_data: VoltageData = Field(..., description="Voltage measurements")
    include_interpretability: bool = Field(default=True, description="Include SHAP values")

class PredictionResponse(BaseModel):
    rul_cycles: int
    rul_confidence_lower: int
    rul_confidence_upper: int
    degradation_score: float
    degradation_stage: str
    anomaly_flag: bool
    anomaly_score: float
    feature_importance: Dict[str, float]
    timestamp: str
    model_version: str
    processing_time_ms: float

class BatchPredictionRequest(BaseModel):
    predictions: List[PredictionRequest] = Field(..., max_items=100)

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total_processing_time_ms: float
    success_count: int
    error_count: int

class HealthCheckResponse(BaseModel):
    status: str
    model_ready: bool
    uptime_seconds: float
    version: str

class ModelInfoResponse(BaseModel):
    model_version: str
    model_types: List[str]
    feature_count: int
    last_trained: Optional[str]
    performance_metrics: Dict[str, float]

# FastAPI app
app = FastAPI(
    title="RUL Prediction API",
    description="High-performance API for Remaining Useful Life prediction",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup time for uptime calculation
_startup_time = time.time()

@app.on_event("startup")
async def startup_event():
    """Initialize models and logging on startup"""
    global _rul_predictor
    
    logger.info("🚀 Starting RUL Prediction API...")
    
    # Configure structured logging
    configure_prediction_logging(
        log_file="logs/api_predictions.jsonl",
        enable_console=True
    )
    
    # Initialize predictor (models will be loaded lazily)
    _rul_predictor = RULPredictor()
    
    logger.info("✅ API startup complete")

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint"""
    uptime = time.time() - _startup_time
    
    return HealthCheckResponse(
        status="healthy",
        model_ready=_rul_predictor.is_ready if _rul_predictor else False,
        uptime_seconds=uptime,
        version="1.0.0"
    )

@app.get("/model_info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model information and metadata"""
    if not _rul_predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    model_status = _rul_predictor.get_model_status()
    
    return ModelInfoResponse(
        model_version="1.0.0",
        model_types=["xgboost", "lightgbm", "random_forest", "ensemble"],
        feature_count=55,  # Approximate feature count
        last_trained=None,  # Would be set from model metadata
        performance_metrics={}  # Would be loaded from model evaluation
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_rul(request: PredictionRequest):
    """Single RUL prediction endpoint"""
    if not _rul_predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    start_time = time.time()
    
    try:
        # Convert request to CycleData
        cycle_data = CycleData(
            cycle_number=request.cycle_number,
            vl_series=np.array(request.voltage_data.vl_series),
            vo_series=np.array(request.voltage_data.vo_series),
            timestamp=datetime.now()
        )
        
        # Run prediction in thread pool for non-blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _executor,
            _rul_predictor.predict_with_error_handling,
            cycle_data,
            request.capacitor_id,
            None
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return PredictionResponse(
            rul_cycles=result.rul_cycles,
            rul_confidence_lower=result.rul_confidence_lower,
            rul_confidence_upper=result.rul_confidence_upper,
            degradation_score=result.degradation_score,
            degradation_stage=result.degradation_stage,
            anomaly_flag=result.anomaly_flag,
            anomaly_score=result.anomaly_score,
            feature_importance=result.feature_importance,
            timestamp=result.timestamp.isoformat() if hasattr(result.timestamp, 'isoformat') else str(result.timestamp),
            model_version=result.model_version,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict_rul(request: BatchPredictionRequest):
    """Batch RUL prediction endpoint with parallel processing"""
    if not _rul_predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    start_time = time.time()
    
    # Process predictions in parallel
    async def process_single_prediction(pred_request: PredictionRequest) -> Optional[PredictionResponse]:
        try:
            cycle_data = CycleData(
                cycle_number=pred_request.cycle_number,
                vl_series=np.array(pred_request.voltage_data.vl_series),
                vo_series=np.array(pred_request.voltage_data.vo_series),
                timestamp=datetime.now()
            )
            
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                _executor,
                _rul_predictor.predict_with_error_handling,
                cycle_data,
                pred_request.capacitor_id,
                None
            )
            
            return PredictionResponse(
                rul_cycles=result.rul_cycles,
                rul_confidence_lower=result.rul_confidence_lower,
                rul_confidence_upper=result.rul_confidence_upper,
                degradation_score=result.degradation_score,
                degradation_stage=result.degradation_stage,
                anomaly_flag=result.anomaly_flag,
                anomaly_score=result.anomaly_score,
                feature_importance=result.feature_importance,
                timestamp=result.timestamp.isoformat() if hasattr(result.timestamp, 'isoformat') else str(result.timestamp),
                model_version=result.model_version,
                processing_time_ms=0  # Will be set at batch level
            )
            
        except Exception as e:
            logger.error(f"Batch prediction item failed: {e}")
            return None
    
    # Process all predictions concurrently
    tasks = [process_single_prediction(pred_req) for pred_req in request.predictions]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Filter successful results
    successful_results = [r for r in results if isinstance(r, PredictionResponse)]
    error_count = len(results) - len(successful_results)
    
    total_time = (time.time() - start_time) * 1000
    
    return BatchPredictionResponse(
        results=successful_results,
        total_processing_time_ms=total_time,
        success_count=len(successful_results),
        error_count=error_count
    )

# Additional utility endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "RUL Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)