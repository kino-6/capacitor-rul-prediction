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
from .help_system import get_help_system, HelpCategory, TutorialStep
from .knowledge_management import get_knowledge_system, ContentType
from .collaboration_platform import get_collaboration_platform, DiscussionType, Priority as CollabPriority

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model caching
_rul_predictor: Optional[RULPredictor] = None
_model_cache: Dict[str, Any] = {}
_executor = ThreadPoolExecutor(max_workers=8)  # Use 8 threads for parallel processing
_help_system = get_help_system()  # Initialize help system
_knowledge_system = get_knowledge_system()  # Initialize knowledge management
_collaboration_platform = get_collaboration_platform()  # Initialize collaboration platform

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

class HelpArticleResponse(BaseModel):
    id: str
    title: str
    category: str
    content: str
    tags: List[str]
    related_articles: List[str]
    difficulty: str
    estimated_time: str

class TutorialStepResponse(BaseModel):
    step: str
    title: str
    description: str
    instructions: List[str]
    code_example: Optional[str]
    validation_criteria: List[str]
    next_step: Optional[str]
    help_links: List[str]

class HelpSearchRequest(BaseModel):
    query: str
    category: Optional[str] = None

class TutorialProgressRequest(BaseModel):
    completed_steps: List[str]

class KnowledgeSearchRequest(BaseModel):
    query: str
    content_type: Optional[str] = None
    limit: int = 20

class CreateDiscussionRequest(BaseModel):
    title: str
    description: str
    discussion_type: str
    priority: str
    tags: List[str] = []

class AddCommentRequest(BaseModel):
    content: str
    parent_comment_id: Optional[str] = None

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
        "health": "/health",
        "help": "/help"
    }

# Help System Endpoints
@app.get("/help/article/{article_id}", response_model=HelpArticleResponse)
async def get_help_article(article_id: str):
    """Get a specific help article"""
    article = _help_system.get_article(article_id)
    if not article:
        raise HTTPException(status_code=404, detail=f"Help article '{article_id}' not found")
    
    return HelpArticleResponse(
        id=article.id,
        title=article.title,
        category=article.category.value,
        content=article.content,
        tags=article.tags,
        related_articles=article.related_articles,
        difficulty=article.difficulty,
        estimated_time=article.estimated_time
    )

@app.post("/help/search")
async def search_help_articles(request: HelpSearchRequest):
    """Search help articles"""
    category = None
    if request.category:
        try:
            category = HelpCategory(request.category)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid category: {request.category}")
    
    articles = _help_system.search_articles(request.query, category)
    
    return {
        "query": request.query,
        "category": request.category,
        "results": [
            {
                "id": article.id,
                "title": article.title,
                "category": article.category.value,
                "tags": article.tags,
                "difficulty": article.difficulty,
                "estimated_time": article.estimated_time
            }
            for article in articles
        ]
    }

@app.get("/help/category/{category}")
async def get_help_by_category(category: str):
    """Get all help articles in a category"""
    try:
        help_category = HelpCategory(category)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
    
    articles = _help_system.get_articles_by_category(help_category)
    
    return {
        "category": category,
        "articles": [
            {
                "id": article.id,
                "title": article.title,
                "tags": article.tags,
                "difficulty": article.difficulty,
                "estimated_time": article.estimated_time
            }
            for article in articles
        ]
    }

@app.get("/help/tutorial/{step}", response_model=TutorialStepResponse)
async def get_tutorial_step(step: str):
    """Get a specific tutorial step"""
    try:
        tutorial_step = TutorialStep(step)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid tutorial step: {step}")
    
    step_data = _help_system.get_tutorial_step(tutorial_step)
    if not step_data:
        raise HTTPException(status_code=404, detail=f"Tutorial step '{step}' not found")
    
    return TutorialStepResponse(
        step=step_data.step.value,
        title=step_data.title,
        description=step_data.description,
        instructions=step_data.instructions,
        code_example=step_data.code_example,
        validation_criteria=step_data.validation_criteria,
        next_step=step_data.next_step.value if step_data.next_step else None,
        help_links=step_data.help_links
    )

@app.post("/help/tutorial/progress")
async def get_tutorial_progress(request: TutorialProgressRequest):
    """Get tutorial progress information"""
    try:
        completed_steps = [TutorialStep(step) for step in request.completed_steps]
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid tutorial step: {e}")
    
    progress = _help_system.get_tutorial_progress(completed_steps)
    return progress

@app.get("/help/contextual/{context}")
async def get_contextual_help(context: str):
    """Get contextual help based on user context"""
    articles = _help_system.get_contextual_help(context)
    
    return {
        "context": context,
        "help_articles": [
            {
                "id": article.id,
                "title": article.title,
                "category": article.category.value,
                "difficulty": article.difficulty,
                "estimated_time": article.estimated_time
            }
            for article in articles
        ]
    }

@app.get("/help")
async def get_help_overview():
    """Get help system overview"""
    return {
        "message": "RUL Prediction System Help",
        "available_endpoints": {
            "article": "/help/article/{article_id}",
            "search": "/help/search (POST)",
            "category": "/help/category/{category}",
            "tutorial": "/help/tutorial/{step}",
            "progress": "/help/tutorial/progress (POST)",
            "contextual": "/help/contextual/{context}"
        },
        "categories": [cat.value for cat in HelpCategory],
        "tutorial_steps": [step.value for step in TutorialStep],
        "quick_start": "/help/article/quick_start",
        "interactive_tutorial": "/help/tutorial/introduction"
    }

# Knowledge Management Endpoints
@app.post("/knowledge/search")
async def search_knowledge_base(request: KnowledgeSearchRequest):
    """Search knowledge base"""
    content_type = None
    if request.content_type:
        try:
            content_type = ContentType(request.content_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid content type: {request.content_type}")
    
    results = _knowledge_system.search_knowledge_base(
        request.query, content_type, request.limit
    )
    
    return {
        "query": request.query,
        "content_type": request.content_type,
        "results": results
    }

@app.get("/knowledge/article/{article_id}")
async def get_knowledge_article(article_id: str):
    """Get knowledge article by ID"""
    article = _knowledge_system.get_knowledge_item(article_id)
    if not article:
        raise HTTPException(status_code=404, detail=f"Article '{article_id}' not found")
    
    # Increment view count
    _knowledge_system.increment_views(article_id)
    
    return {
        "id": article.id,
        "title": article.title,
        "content_type": article.content_type.value,
        "content": article.content,
        "tags": article.tags,
        "author": article.author,
        "created_date": article.created_date.isoformat(),
        "last_modified": article.last_modified.isoformat(),
        "views": article.views + 1,
        "rating": article.rating,
        "rating_count": article.rating_count,
        "related_items": article.related_items
    }

@app.get("/knowledge/popular")
async def get_popular_knowledge():
    """Get most popular knowledge items"""
    popular_items = _knowledge_system.get_popular_content(limit=10)
    return {"popular_items": popular_items}

@app.get("/knowledge/recent")
async def get_recent_knowledge():
    """Get recently added knowledge items"""
    recent_items = _knowledge_system.get_recent_content(days=30, limit=10)
    return {"recent_items": recent_items}

@app.post("/knowledge/rate/{article_id}")
async def rate_knowledge_article(article_id: str, rating: int, user_id: str, comment: str = ""):
    """Rate a knowledge article"""
    if not (1 <= rating <= 5):
        raise HTTPException(status_code=400, detail="Rating must be between 1 and 5")
    
    success = _knowledge_system.rate_item(user_id, article_id, rating, comment)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to rate article")
    
    return {"message": "Rating submitted successfully"}

# Collaboration Platform Endpoints
@app.get("/collaboration/discussions")
async def get_discussions(
    status: Optional[str] = None,
    discussion_type: Optional[str] = None,
    limit: int = 20,
    offset: int = 0
):
    """Get discussions with filtering"""
    status_enum = None
    if status:
        try:
            from .collaboration_platform import Status
            status_enum = Status(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")
    
    type_enum = None
    if discussion_type:
        try:
            type_enum = DiscussionType(discussion_type)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid discussion type: {discussion_type}")
    
    discussions = _collaboration_platform.get_discussions(
        status_enum, type_enum, limit, offset
    )
    
    return {"discussions": discussions}

@app.post("/collaboration/discussions")
async def create_discussion(request: CreateDiscussionRequest, user_id: str):
    """Create a new discussion"""
    try:
        discussion_type = DiscussionType(request.discussion_type)
        priority = CollabPriority(request.priority)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    discussion_id = _collaboration_platform.create_discussion(
        request.title, request.description, discussion_type,
        priority, user_id, request.tags
    )
    
    return {"discussion_id": discussion_id, "message": "Discussion created successfully"}

@app.post("/collaboration/discussions/{discussion_id}/comments")
async def add_comment(discussion_id: str, request: AddCommentRequest, user_id: str):
    """Add comment to discussion"""
    comment_id = _collaboration_platform.add_comment(
        discussion_id, user_id, request.content, request.parent_comment_id
    )
    
    return {"comment_id": comment_id, "message": "Comment added successfully"}

@app.post("/collaboration/vote/{target_type}/{target_id}")
async def vote_on_content(target_type: str, target_id: str, vote_type: str, user_id: str):
    """Vote on discussion or comment"""
    if target_type not in ["discussion", "comment"]:
        raise HTTPException(status_code=400, detail="Invalid target type")
    
    if vote_type not in ["upvote", "downvote"]:
        raise HTTPException(status_code=400, detail="Invalid vote type")
    
    success = _collaboration_platform.vote_on_content(user_id, target_type, target_id, vote_type)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to record vote")
    
    return {"message": "Vote recorded successfully"}

@app.get("/collaboration/notifications/{user_id}")
async def get_user_notifications(user_id: str, unread_only: bool = False):
    """Get notifications for user"""
    notifications = _collaboration_platform.get_user_notifications(user_id, unread_only)
    return {"notifications": notifications}

@app.get("/collaboration/report")
async def get_collaboration_report():
    """Get collaboration platform activity report"""
    report = _collaboration_platform.generate_collaboration_report()
    return report

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)