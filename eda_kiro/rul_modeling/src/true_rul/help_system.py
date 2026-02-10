"""
In-System Help and Guided Tutorial System
Provides contextual help and interactive tutorials for the RUL Prediction System
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path

class HelpCategory(Enum):
    GETTING_STARTED = "getting_started"
    API_USAGE = "api_usage"
    TROUBLESHOOTING = "troubleshooting"
    INTERPRETATION = "interpretation"
    MAINTENANCE = "maintenance"
    ADVANCED = "advanced"

class TutorialStep(Enum):
    INTRODUCTION = "introduction"
    SETUP_VERIFICATION = "setup_verification"
    FIRST_PREDICTION = "first_prediction"
    RESULT_INTERPRETATION = "result_interpretation"
    BATCH_PROCESSING = "batch_processing"
    MONITORING_SETUP = "monitoring_setup"
    COMPLETION = "completion"

@dataclass
class HelpArticle:
    id: str
    title: str
    category: HelpCategory
    content: str
    tags: List[str]
    related_articles: List[str]
    difficulty: str  # "beginner", "intermediate", "advanced"
    estimated_time: str  # "2 minutes", "10 minutes", etc.

@dataclass
class TutorialStepData:
    step: TutorialStep
    title: str
    description: str
    instructions: List[str]
    code_example: Optional[str]
    validation_criteria: List[str]
    next_step: Optional[TutorialStep]
    help_links: List[str]

class HelpSystem:
    """In-system help and tutorial system"""
    
    def __init__(self, help_data_path: str = "docs/help_data"):
        self.help_data_path = Path(help_data_path)
        self.articles: Dict[str, HelpArticle] = {}
        self.tutorials: Dict[TutorialStep, TutorialStepData] = {}
        self._load_help_data()
        self._initialize_tutorials()
    
    def _load_help_data(self):
        """Load help articles from documentation"""
        self._create_help_articles()
    
    def _create_help_articles(self):
        """Create help articles from documentation content"""
        
        # Getting Started Articles
        self.articles["quick_start"] = HelpArticle(
            id="quick_start",
            title="Quick Start Guide",
            category=HelpCategory.GETTING_STARTED,
            content="""
# Quick Start Guide

Welcome to the RUL Prediction System! This guide will help you make your first prediction in under 5 minutes.

## Prerequisites
- System is running (check with `curl http://localhost:8000/health`)
- You have voltage data from a capacitor cycle

## Step 1: Verify System Health
```bash
curl http://localhost:8000/health
```
You should see: `{"status": "healthy", "model_ready": true}`

## Step 2: Prepare Your Data
Your voltage data should be in this format:
```json
{
  "capacitor_id": "C1",
  "cycle_number": 50,
  "voltage_data": {
    "vl_series": [1.0, 1.05, 1.1, ...],
    "vo_series": [0.9, 0.95, 1.0, ...]
  }
}
```

## Step 3: Make Your First Prediction
```bash
curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d @your_data.json
```

## Step 4: Interpret Results
- `rul_cycles`: Remaining operational cycles
- `degradation_stage`: Current health status
- `confidence_lower/upper`: Uncertainty range

Need help? Use the interactive tutorial: `/help/tutorial/start`
            """,
            tags=["beginner", "first-time", "setup"],
            related_articles=["api_basics", "result_interpretation"],
            difficulty="beginner",
            estimated_time="5 minutes"
        )
        
        self.articles["api_basics"] = HelpArticle(
            id="api_basics",
            title="API Basics",
            category=HelpCategory.API_USAGE,
            content="""
# API Basics

The RUL Prediction API provides RESTful endpoints for equipment health monitoring.

## Base URL
`http://localhost:8000`

## Key Endpoints

### Health Check
```bash
GET /health
```
Returns system status and model readiness.

### Single Prediction
```bash
POST /predict
```
Predict RUL for one capacitor cycle.

### Batch Prediction
```bash
POST /batch_predict
```
Process multiple predictions (up to 100).

### Model Information
```bash
GET /model_info
```
Get model version and performance metrics.

## Request Format
All prediction requests use this structure:
```json
{
  "capacitor_id": "string",
  "cycle_number": integer,
  "voltage_data": {
    "vl_series": [float, ...],
    "vo_series": [float, ...]
  },
  "include_interpretability": boolean
}
```

## Response Format
```json
{
  "rul_cycles": integer,
  "rul_confidence_lower": integer,
  "rul_confidence_upper": integer,
  "degradation_score": float,
  "degradation_stage": string,
  "anomaly_flag": boolean,
  "anomaly_score": float,
  "feature_importance": object,
  "timestamp": string,
  "model_version": string
}
```

## Error Handling
- 400: Bad request (check input format)
- 422: Validation error (check field types)
- 500: Internal error (check logs)
- 503: Service unavailable (models not loaded)

For detailed examples, see: `/help/article/api_examples`
            """,
            tags=["api", "endpoints", "requests"],
            related_articles=["quick_start", "api_examples", "error_handling"],
            difficulty="beginner",
            estimated_time="10 minutes"
        )
        
        self.articles["result_interpretation"] = HelpArticle(
            id="result_interpretation",
            title="Understanding Prediction Results",
            category=HelpCategory.INTERPRETATION,
            content="""
# Understanding Prediction Results

Learn how to interpret RUL predictions and make informed maintenance decisions.

## Key Metrics Explained

### RUL Cycles
- **Definition**: Predicted remaining operational cycles before failure
- **Range**: 0 to ~200 cycles (depends on equipment)
- **Usage**: Primary metric for maintenance planning

### Confidence Intervals
- **Lower Bound**: Conservative estimate (95% confidence)
- **Upper Bound**: Optimistic estimate (95% confidence)
- **Width**: Indicates prediction uncertainty

**Example**:
```json
{
  "rul_cycles": 45,
  "rul_confidence_lower": 38,
  "rul_confidence_upper": 52
}
```
**Interpretation**: Best estimate is 45 cycles, but could be anywhere from 38-52 cycles.

### Degradation Stages

1. **Healthy** (score: 0.0-0.3)
   - Normal operation
   - Routine monitoring sufficient
   - No immediate action needed

2. **Early Degradation** (score: 0.3-0.6)
   - Initial signs of wear
   - Increase monitoring frequency
   - Plan future maintenance

3. **Advanced Degradation** (score: 0.6-0.8)
   - Significant degradation
   - Schedule maintenance soon
   - Prepare replacement parts

4. **Critical** (score: 0.8-1.0)
   - Failure imminent
   - Immediate action required
   - Consider emergency shutdown

### Anomaly Detection
- **Anomaly Flag**: true/false for unusual behavior
- **Anomaly Score**: 0.0 (normal) to 1.0 (highly anomalous)

**Action Guidelines**:
- Score < 0.3: Normal operation
- Score 0.3-0.7: Monitor closely
- Score > 0.7: Investigate immediately

### Feature Importance
Shows which measurements influenced the prediction:
```json
{
  "feature_importance": {
    "responsiveness_feature_1": 0.25,
    "voltage_std": 0.18,
    "frequency_peak": 0.15
  }
}
```

Higher values indicate more influential features.

## Decision Making Framework

### Maintenance Planning
1. **RUL > 50 cycles**: Normal operation
2. **RUL 25-50 cycles**: Plan maintenance
3. **RUL 10-25 cycles**: Schedule maintenance
4. **RUL < 10 cycles**: Immediate maintenance

### Confidence Considerations
- **Narrow interval (±5 cycles)**: High confidence, trust prediction
- **Wide interval (±15+ cycles)**: Low confidence, increase monitoring

### Anomaly Response
1. Verify data quality
2. Check equipment visually
3. Review maintenance history
4. Consider environmental factors

For advanced interpretation techniques, see: `/help/article/advanced_interpretation`
            """,
            tags=["interpretation", "decision-making", "maintenance"],
            related_articles=["maintenance_planning", "anomaly_handling"],
            difficulty="intermediate",
            estimated_time="15 minutes"
        )
        
        self.articles["troubleshooting_common"] = HelpArticle(
            id="troubleshooting_common",
            title="Common Issues and Solutions",
            category=HelpCategory.TROUBLESHOOTING,
            content="""
# Common Issues and Solutions

Quick solutions for the most frequently encountered problems.

## Issue 1: API Returns 503 Service Unavailable

**Symptoms**: All requests return 503, health check shows `model_ready: false`

**Quick Fix**:
```bash
# Restart the service
docker-compose restart rul-api

# Wait 2-3 minutes for models to load
sleep 180

# Verify recovery
curl http://localhost:8000/health
```

**Root Causes**:
- Insufficient memory for model loading
- Corrupted model files
- Startup timeout

## Issue 2: Predictions Taking Too Long

**Symptoms**: Requests timeout or take >2 seconds

**Quick Fixes**:
1. **Reduce data size**:
   ```python
   # Limit voltage series length
   max_length = 500
   vl_series = vl_series[:max_length]
   vo_series = vo_series[:max_length]
   ```

2. **Use batch processing**:
   ```bash
   # Instead of multiple single requests
   curl -X POST "/batch_predict" -d '{"predictions": [...]}'
   ```

3. **Enable caching** (if making repeated requests)

## Issue 3: High False Positive Rate

**Symptoms**: Too many anomaly alerts for healthy equipment

**Quick Fix**:
```python
# Adjust anomaly threshold in configuration
ANOMALY_THRESHOLD = 0.7  # Increase from default 0.5
```

**Long-term Solution**: Retrain models with recent data

## Issue 4: Inconsistent Predictions

**Symptoms**: Large variations in consecutive predictions

**Quick Fixes**:
1. **Apply data smoothing**:
   ```python
   from scipy.signal import savgol_filter
   vl_smooth = savgol_filter(vl_series, 5, 2)
   ```

2. **Use ensemble mode** (already default)

3. **Check data quality**:
   - Consistent sampling rate
   - No missing values
   - Proper voltage ranges

## Issue 5: Memory Usage Growing

**Symptoms**: System becomes slow over time, high memory usage

**Quick Fix**:
```bash
# Restart service to clear memory
docker-compose restart rul-api
```

**Prevention**: Set up automated daily restarts

## Diagnostic Commands

### System Health
```bash
# Check API status
curl http://localhost:8000/health

# Check system resources
docker stats rul-api

# Check recent logs
docker-compose logs --tail=50 rul-api
```

### Test Prediction
```bash
# Test with sample data
curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "capacitor_id": "TEST",
    "cycle_number": 1,
    "voltage_data": {
      "vl_series": [1.0, 1.1, 1.2, 1.1, 1.0],
      "vo_series": [0.9, 1.0, 1.1, 1.0, 0.9]
    }
  }'
```

## When to Contact Support

Contact technical support if:
- Issues persist after trying quick fixes
- System shows repeated crashes
- Data corruption suspected
- Performance severely degraded

**Support**: support@rul-system.com
**Emergency**: +1-800-RUL-HELP

For detailed troubleshooting, see: `/help/article/advanced_troubleshooting`
            """,
            tags=["troubleshooting", "errors", "performance"],
            related_articles=["advanced_troubleshooting", "performance_optimization"],
            difficulty="intermediate",
            estimated_time="10 minutes"
        )
    
    def _initialize_tutorials(self):
        """Initialize guided tutorial steps"""
        
        self.tutorials[TutorialStep.INTRODUCTION] = TutorialStepData(
            step=TutorialStep.INTRODUCTION,
            title="Welcome to RUL Prediction System",
            description="Learn how to predict equipment remaining useful life with machine learning",
            instructions=[
                "This tutorial will guide you through making your first RUL prediction",
                "You'll learn to interpret results and set up monitoring",
                "The tutorial takes about 15 minutes to complete",
                "You can exit and resume at any time"
            ],
            code_example=None,
            validation_criteria=[],
            next_step=TutorialStep.SETUP_VERIFICATION,
            help_links=["quick_start", "api_basics"]
        )
        
        self.tutorials[TutorialStep.SETUP_VERIFICATION] = TutorialStepData(
            step=TutorialStep.SETUP_VERIFICATION,
            title="Verify System Setup",
            description="Check that the RUL prediction system is running correctly",
            instructions=[
                "First, let's verify that the system is healthy and ready",
                "Run the health check command below",
                "You should see 'status: healthy' and 'model_ready: true'",
                "If not, check the troubleshooting guide"
            ],
            code_example="""
# Check system health
curl http://localhost:8000/health

# Expected response:
{
  "status": "healthy",
  "model_ready": true,
  "uptime_seconds": 3600.5,
  "version": "1.0.0"
}
            """,
            validation_criteria=[
                "Health endpoint returns 200 status",
                "Response shows 'status': 'healthy'",
                "Response shows 'model_ready': true"
            ],
            next_step=TutorialStep.FIRST_PREDICTION,
            help_links=["troubleshooting_common"]
        )
        
        self.tutorials[TutorialStep.FIRST_PREDICTION] = TutorialStepData(
            step=TutorialStep.FIRST_PREDICTION,
            title="Make Your First Prediction",
            description="Submit voltage data and get a RUL prediction",
            instructions=[
                "Now let's make a prediction using sample voltage data",
                "Copy the command below and run it in your terminal",
                "This simulates voltage measurements from a capacitor cycle",
                "The system will return a prediction with confidence intervals"
            ],
            code_example="""
# Make a prediction with sample data
curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "capacitor_id": "TUTORIAL_C1",
    "cycle_number": 50,
    "voltage_data": {
      "vl_series": [1.0, 1.05, 1.1, 1.15, 1.2, 1.18, 1.15, 1.1, 1.05, 1.0],
      "vo_series": [0.9, 0.95, 1.0, 1.05, 1.1, 1.08, 1.05, 1.0, 0.95, 0.9]
    },
    "include_interpretability": true
  }'
            """,
            validation_criteria=[
                "Request returns 200 status code",
                "Response contains 'rul_cycles' field",
                "Response contains 'degradation_stage' field",
                "Response contains 'feature_importance' object"
            ],
            next_step=TutorialStep.RESULT_INTERPRETATION,
            help_links=["api_basics", "result_interpretation"]
        )
        
        self.tutorials[TutorialStep.RESULT_INTERPRETATION] = TutorialStepData(
            step=TutorialStep.RESULT_INTERPRETATION,
            title="Interpret Prediction Results",
            description="Learn how to understand and act on RUL predictions",
            instructions=[
                "Let's examine the prediction result from the previous step",
                "Look for these key fields in the response:",
                "• rul_cycles: Predicted remaining operational cycles",
                "• degradation_stage: Current health status",
                "• confidence_lower/upper: Uncertainty range",
                "• anomaly_flag: Whether behavior is unusual"
            ],
            code_example="""
# Example prediction result:
{
  "rul_cycles": 45,                    # 45 cycles remaining
  "rul_confidence_lower": 38,          # Conservative estimate
  "rul_confidence_upper": 52,          # Optimistic estimate
  "degradation_score": 0.35,           # 35% degraded
  "degradation_stage": "early_degradation",  # Health status
  "anomaly_flag": false,               # Normal behavior
  "anomaly_score": 0.12,               # Low anomaly score
  "feature_importance": {              # What drove this prediction
    "responsiveness_feature_1": 0.25,
    "voltage_std": 0.18
  }
}

# Interpretation:
# - Equipment has ~45 cycles remaining (range: 38-52)
# - Currently in early degradation stage
# - Behavior is normal (not anomalous)
# - Plan maintenance in ~35-40 cycles (conservative)
            """,
            validation_criteria=[
                "Can identify RUL cycles in response",
                "Understands confidence interval meaning",
                "Can interpret degradation stage",
                "Knows when to take action based on results"
            ],
            next_step=TutorialStep.BATCH_PROCESSING,
            help_links=["result_interpretation", "maintenance_planning"]
        )
        
        self.tutorials[TutorialStep.BATCH_PROCESSING] = TutorialStepData(
            step=TutorialStep.BATCH_PROCESSING,
            title="Process Multiple Predictions",
            description="Learn how to efficiently process multiple capacitors",
            instructions=[
                "For monitoring multiple capacitors, use batch processing",
                "This is more efficient than individual requests",
                "You can process up to 100 predictions in one request",
                "Try the batch example below with two capacitors"
            ],
            code_example="""
# Batch prediction for multiple capacitors
curl -X POST "http://localhost:8000/batch_predict" \\
  -H "Content-Type: application/json" \\
  -d '{
    "predictions": [
      {
        "capacitor_id": "C1",
        "cycle_number": 50,
        "voltage_data": {
          "vl_series": [1.0, 1.1, 1.2, 1.1, 1.0],
          "vo_series": [0.9, 1.0, 1.1, 1.0, 0.9]
        }
      },
      {
        "capacitor_id": "C2",
        "cycle_number": 75,
        "voltage_data": {
          "vl_series": [1.1, 1.2, 1.3, 1.2, 1.1],
          "vo_series": [1.0, 1.1, 1.2, 1.1, 1.0]
        }
      }
    ]
  }'

# Response includes:
# - results: Array of individual predictions
# - success_count: Number of successful predictions
# - error_count: Number of failed predictions
# - total_processing_time_ms: Total time taken
            """,
            validation_criteria=[
                "Batch request returns 200 status",
                "Response contains 'results' array",
                "Can identify success_count and error_count",
                "Understands efficiency benefits of batching"
            ],
            next_step=TutorialStep.MONITORING_SETUP,
            help_links=["api_basics", "performance_optimization"]
        )
        
        self.tutorials[TutorialStep.MONITORING_SETUP] = TutorialStepData(
            step=TutorialStep.MONITORING_SETUP,
            title="Set Up Monitoring",
            description="Learn how to monitor system health and performance",
            instructions=[
                "Regular monitoring ensures reliable operation",
                "Check system health periodically",
                "Monitor prediction accuracy and false positive rates",
                "Set up alerts for critical degradation stages"
            ],
            code_example="""
# 1. Check system health regularly
curl http://localhost:8000/health

# 2. Get model performance information
curl http://localhost:8000/model_info

# 3. Monitor prediction logs
tail -f logs/api_predictions.jsonl

# 4. Set up automated health checks (example cron job)
# Add to crontab: */5 * * * * curl -s http://localhost:8000/health | grep -q "healthy" || echo "RUL system unhealthy" | mail admin@company.com

# 5. Create alerting rules for critical predictions
# Example: Alert when RUL < 10 cycles or degradation_stage == "critical"
            """,
            validation_criteria=[
                "Can check system health",
                "Understands model performance metrics",
                "Knows how to access prediction logs",
                "Can set up basic monitoring"
            ],
            next_step=TutorialStep.COMPLETION,
            help_links=["monitoring_setup", "maintenance_planning"]
        )
        
        self.tutorials[TutorialStep.COMPLETION] = TutorialStepData(
            step=TutorialStep.COMPLETION,
            title="Tutorial Complete!",
            description="You've successfully learned the basics of RUL prediction",
            instructions=[
                "Congratulations! You've completed the RUL prediction tutorial",
                "You now know how to:",
                "• Verify system health",
                "• Make single and batch predictions",
                "• Interpret results and plan maintenance",
                "• Set up basic monitoring",
                "",
                "Next steps:",
                "• Integrate with your maintenance system",
                "• Set up automated monitoring and alerting",
                "• Explore advanced features and customization"
            ],
            code_example=None,
            validation_criteria=[],
            next_step=None,
            help_links=["advanced_topics", "integration_guide", "maintenance_planning"]
        )
    
    def get_article(self, article_id: str) -> Optional[HelpArticle]:
        """Get a specific help article"""
        return self.articles.get(article_id)
    
    def search_articles(self, query: str, category: Optional[HelpCategory] = None) -> List[HelpArticle]:
        """Search help articles by query and category"""
        results = []
        query_lower = query.lower()
        
        for article in self.articles.values():
            # Filter by category if specified
            if category and article.category != category:
                continue
            
            # Search in title, content, and tags
            if (query_lower in article.title.lower() or
                query_lower in article.content.lower() or
                any(query_lower in tag.lower() for tag in article.tags)):
                results.append(article)
        
        return results
    
    def get_articles_by_category(self, category: HelpCategory) -> List[HelpArticle]:
        """Get all articles in a specific category"""
        return [article for article in self.articles.values() 
                if article.category == category]
    
    def get_tutorial_step(self, step: TutorialStep) -> Optional[TutorialStepData]:
        """Get tutorial step data"""
        return self.tutorials.get(step)
    
    def get_tutorial_progress(self, completed_steps: List[TutorialStep]) -> Dict[str, Any]:
        """Get tutorial progress information"""
        total_steps = len(self.tutorials)
        completed_count = len(completed_steps)
        
        # Find current step
        current_step = None
        for step in TutorialStep:
            if step not in completed_steps:
                current_step = step
                break
        
        return {
            "total_steps": total_steps,
            "completed_steps": completed_count,
            "progress_percentage": (completed_count / total_steps) * 100,
            "current_step": current_step.value if current_step else None,
            "is_complete": completed_count == total_steps
        }
    
    def get_contextual_help(self, context: str) -> List[HelpArticle]:
        """Get contextual help based on current user context"""
        context_mapping = {
            "api_error": ["troubleshooting_common", "api_basics"],
            "first_time": ["quick_start", "api_basics"],
            "interpretation": ["result_interpretation", "maintenance_planning"],
            "performance": ["performance_optimization", "troubleshooting_common"],
            "integration": ["integration_guide", "api_examples"]
        }
        
        article_ids = context_mapping.get(context, ["quick_start"])
        return [self.articles[aid] for aid in article_ids if aid in self.articles]
    
    def export_help_data(self) -> Dict[str, Any]:
        """Export all help data for API responses"""
        return {
            "articles": {
                aid: {
                    "id": article.id,
                    "title": article.title,
                    "category": article.category.value,
                    "content": article.content,
                    "tags": article.tags,
                    "related_articles": article.related_articles,
                    "difficulty": article.difficulty,
                    "estimated_time": article.estimated_time
                }
                for aid, article in self.articles.items()
            },
            "tutorials": {
                step.value: {
                    "step": tutorial.step.value,
                    "title": tutorial.title,
                    "description": tutorial.description,
                    "instructions": tutorial.instructions,
                    "code_example": tutorial.code_example,
                    "validation_criteria": tutorial.validation_criteria,
                    "next_step": tutorial.next_step.value if tutorial.next_step else None,
                    "help_links": tutorial.help_links
                }
                for step, tutorial in self.tutorials.items()
            },
            "categories": [cat.value for cat in HelpCategory],
            "tutorial_steps": [step.value for step in TutorialStep]
        }

# Global help system instance
help_system = HelpSystem()

def get_help_system() -> HelpSystem:
    """Get the global help system instance"""
    return help_system