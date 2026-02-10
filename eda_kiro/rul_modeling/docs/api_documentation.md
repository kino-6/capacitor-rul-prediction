# RUL Prediction API Documentation

## Overview

The RUL Prediction API provides RESTful endpoints for predicting the Remaining Useful Life of capacitors. This API is built with FastAPI and provides automatic OpenAPI documentation, high performance, and comprehensive error handling.

**Base URL**: `http://localhost:8000`
**API Version**: 1.0.0
**Documentation**: `http://localhost:8000/docs` (Interactive Swagger UI)
**Alternative Docs**: `http://localhost:8000/redoc` (ReDoc)

## Authentication

Currently, the API operates without authentication for development purposes. For production deployment, implement appropriate authentication mechanisms:

- API Keys
- JWT Tokens
- OAuth 2.0
- Basic Authentication

## Rate Limiting

- **Single Predictions**: 100 requests/minute per client
- **Batch Predictions**: 10 requests/minute per client
- **Health Checks**: Unlimited

## API Endpoints

### 1. Health Check

Check system health and readiness.

**Endpoint**: `GET /health`

**Response Model**:
```json
{
  "status": "healthy",
  "model_ready": true,
  "uptime_seconds": 3600.5,
  "version": "1.0.0"
}
```

**Example Request**:
```bash
curl -X GET "http://localhost:8000/health" \
  -H "accept: application/json"
```

**Example Response**:
```json
{
  "status": "healthy",
  "model_ready": true,
  "uptime_seconds": 3600.5,
  "version": "1.0.0"
}
```

**Status Codes**:
- `200`: System healthy
- `503`: System unavailable

### 2. Model Information

Get detailed information about loaded models.

**Endpoint**: `GET /model_info`

**Response Model**:
```json
{
  "model_version": "1.0.0",
  "model_types": ["xgboost", "lightgbm", "random_forest", "ensemble"],
  "feature_count": 55,
  "last_trained": "2024-01-15T10:00:00Z",
  "performance_metrics": {
    "rmse": 5.2,
    "mae": 3.8,
    "r2_score": 0.92,
    "fpr": 0.03
  }
}
```

**Example Request**:
```bash
curl -X GET "http://localhost:8000/model_info" \
  -H "accept: application/json"
```

**Status Codes**:
- `200`: Success
- `503`: Models not loaded

### 3. Single Prediction

Predict RUL for a single capacitor cycle.

**Endpoint**: `POST /predict`

**Request Model**:
```json
{
  "capacitor_id": "string",
  "cycle_number": 1,
  "voltage_data": {
    "vl_series": [1.0, 1.1, 1.2],
    "vo_series": [0.9, 1.0, 1.1]
  },
  "include_interpretability": true
}
```

**Response Model**:
```json
{
  "rul_cycles": 45,
  "rul_confidence_lower": 38,
  "rul_confidence_upper": 52,
  "degradation_score": 0.35,
  "degradation_stage": "early_degradation",
  "anomaly_flag": false,
  "anomaly_score": 0.12,
  "feature_importance": {
    "responsiveness_feature_1": 0.25,
    "voltage_std": 0.18
  },
  "timestamp": "2024-01-15T10:30:00",
  "model_version": "1.0.0",
  "processing_time_ms": 150.5
}
```

**Example Request**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -H "accept: application/json" \
  -d '{
    "capacitor_id": "C1",
    "cycle_number": 50,
    "voltage_data": {
      "vl_series": [1.0, 1.05, 1.1, 1.15, 1.2, 1.18, 1.15, 1.1, 1.05, 1.0],
      "vo_series": [0.9, 0.95, 1.0, 1.05, 1.1, 1.08, 1.05, 1.0, 0.95, 0.9]
    },
    "include_interpretability": true
  }'
```

**Field Descriptions**:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `capacitor_id` | string | Yes | Unique identifier for the capacitor |
| `cycle_number` | integer | Yes | Current cycle number (≥ 1) |
| `voltage_data.vl_series` | array[float] | Yes | Input voltage time series |
| `voltage_data.vo_series` | array[float] | Yes | Output voltage time series |
| `include_interpretability` | boolean | No | Include SHAP values (default: true) |

**Response Field Descriptions**:

| Field | Type | Description |
|-------|------|-------------|
| `rul_cycles` | integer | Predicted remaining cycles |
| `rul_confidence_lower` | integer | Lower bound of 95% confidence interval |
| `rul_confidence_upper` | integer | Upper bound of 95% confidence interval |
| `degradation_score` | float | Continuous degradation score (0-1) |
| `degradation_stage` | string | Stage: healthy, early_degradation, advanced_degradation, critical |
| `anomaly_flag` | boolean | Whether current behavior is anomalous |
| `anomaly_score` | float | Anomaly score (0-1, higher = more anomalous) |
| `feature_importance` | object | Feature importance scores for interpretability |
| `timestamp` | string | Prediction timestamp (ISO 8601) |
| `model_version` | string | Version of the model used |
| `processing_time_ms` | float | Processing time in milliseconds |

**Status Codes**:
- `200`: Prediction successful
- `400`: Invalid request data
- `422`: Validation error
- `500`: Internal server error
- `503`: Service unavailable

### 4. Batch Prediction

Process multiple predictions in parallel.

**Endpoint**: `POST /batch_predict`

**Request Model**:
```json
{
  "predictions": [
    {
      "capacitor_id": "C1",
      "cycle_number": 50,
      "voltage_data": {
        "vl_series": [1.0, 1.1, 1.2],
        "vo_series": [0.9, 1.0, 1.1]
      },
      "include_interpretability": true
    }
  ]
}
```

**Response Model**:
```json
{
  "results": [
    {
      "rul_cycles": 45,
      "rul_confidence_lower": 38,
      "rul_confidence_upper": 52,
      "degradation_score": 0.35,
      "degradation_stage": "early_degradation",
      "anomaly_flag": false,
      "anomaly_score": 0.12,
      "feature_importance": {},
      "timestamp": "2024-01-15T10:30:00",
      "model_version": "1.0.0",
      "processing_time_ms": 0
    }
  ],
  "total_processing_time_ms": 450.2,
  "success_count": 1,
  "error_count": 0
}
```

**Example Request**:
```bash
curl -X POST "http://localhost:8000/batch_predict" \
  -H "Content-Type: application/json" \
  -H "accept: application/json" \
  -d '{
    "predictions": [
      {
        "capacitor_id": "C1",
        "cycle_number": 50,
        "voltage_data": {
          "vl_series": [1.0, 1.05, 1.1, 1.15, 1.2],
          "vo_series": [0.9, 0.95, 1.0, 1.05, 1.1]
        }
      },
      {
        "capacitor_id": "C2",
        "cycle_number": 75,
        "voltage_data": {
          "vl_series": [1.1, 1.15, 1.2, 1.25, 1.3],
          "vo_series": [1.0, 1.05, 1.1, 1.15, 1.2]
        }
      }
    ]
  }'
```

**Constraints**:
- Maximum 100 predictions per batch
- Failed individual predictions don't fail the entire batch
- Results array may be shorter than input if some predictions fail

**Status Codes**:
- `200`: Batch processed (check individual results)
- `400`: Invalid request format
- `413`: Batch too large (> 100 items)
- `503`: Service unavailable

### 5. Root Endpoint

Basic API information.

**Endpoint**: `GET /`

**Example Response**:
```json
{
  "message": "RUL Prediction API",
  "version": "1.0.0",
  "docs": "/docs",
  "health": "/health"
}
```

## Interactive Examples

### Python Client Example

```python
import requests
import json
import numpy as np

# API base URL
BASE_URL = "http://localhost:8000"

class RULClient:
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url
    
    def health_check(self):
        """Check API health"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def predict_single(self, capacitor_id, cycle_number, vl_series, vo_series):
        """Make single prediction"""
        payload = {
            "capacitor_id": capacitor_id,
            "cycle_number": cycle_number,
            "voltage_data": {
                "vl_series": vl_series.tolist() if isinstance(vl_series, np.ndarray) else vl_series,
                "vo_series": vo_series.tolist() if isinstance(vo_series, np.ndarray) else vo_series
            },
            "include_interpretability": True
        }
        
        response = requests.post(
            f"{self.base_url}/predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Prediction failed: {response.status_code} - {response.text}")
    
    def predict_batch(self, predictions):
        """Make batch predictions"""
        payload = {"predictions": predictions}
        
        response = requests.post(
            f"{self.base_url}/batch_predict",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        return response.json()

# Usage example
client = RULClient()

# Check health
health = client.health_check()
print(f"System status: {health['status']}")

# Generate sample data
vl_data = np.linspace(1.0, 1.2, 100) + np.random.normal(0, 0.01, 100)
vo_data = np.linspace(0.9, 1.1, 100) + np.random.normal(0, 0.01, 100)

# Make prediction
result = client.predict_single("C1", 50, vl_data, vo_data)
print(f"RUL Prediction: {result['rul_cycles']} cycles")
print(f"Confidence: [{result['rul_confidence_lower']}, {result['rul_confidence_upper']}]")
print(f"Degradation Stage: {result['degradation_stage']}")
```

### JavaScript Client Example

```javascript
class RULClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async healthCheck() {
        const response = await fetch(`${this.baseUrl}/health`);
        return await response.json();
    }
    
    async predictSingle(capacitorId, cycleNumber, vlSeries, voSeries) {
        const payload = {
            capacitor_id: capacitorId,
            cycle_number: cycleNumber,
            voltage_data: {
                vl_series: vlSeries,
                vo_series: voSeries
            },
            include_interpretability: true
        };
        
        const response = await fetch(`${this.baseUrl}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            throw new Error(`Prediction failed: ${response.status}`);
        }
        
        return await response.json();
    }
    
    async predictBatch(predictions) {
        const payload = { predictions };
        
        const response = await fetch(`${this.baseUrl}/batch_predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });
        
        return await response.json();
    }
}

// Usage example
const client = new RULClient();

// Check health
client.healthCheck().then(health => {
    console.log(`System status: ${health.status}`);
});

// Generate sample data
const vlData = Array.from({length: 100}, (_, i) => 1.0 + i * 0.002 + Math.random() * 0.01);
const voData = Array.from({length: 100}, (_, i) => 0.9 + i * 0.002 + Math.random() * 0.01);

// Make prediction
client.predictSingle('C1', 50, vlData, voData)
    .then(result => {
        console.log(`RUL Prediction: ${result.rul_cycles} cycles`);
        console.log(`Confidence: [${result.rul_confidence_lower}, ${result.rul_confidence_upper}]`);
        console.log(`Degradation Stage: ${result.degradation_stage}`);
    })
    .catch(error => {
        console.error('Prediction failed:', error);
    });
```

### cURL Examples

#### Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

#### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "capacitor_id": "C1",
    "cycle_number": 50,
    "voltage_data": {
      "vl_series": [1.0, 1.05, 1.1, 1.15, 1.2, 1.18, 1.15, 1.1, 1.05, 1.0],
      "vo_series": [0.9, 0.95, 1.0, 1.05, 1.1, 1.08, 1.05, 1.0, 0.95, 0.9]
    }
  }'
```

#### Batch Prediction
```bash
curl -X POST "http://localhost:8000/batch_predict" \
  -H "Content-Type: application/json" \
  -d '{
    "predictions": [
      {
        "capacitor_id": "C1",
        "cycle_number": 50,
        "voltage_data": {
          "vl_series": [1.0, 1.1, 1.2],
          "vo_series": [0.9, 1.0, 1.1]
        }
      }
    ]
  }'
```

## Error Handling

### Error Response Format

All errors follow a consistent format:

```json
{
  "detail": "Error description",
  "error_code": "VALIDATION_ERROR",
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_123456"
}
```

### Common Error Codes

| HTTP Status | Error Code | Description | Solution |
|-------------|------------|-------------|----------|
| 400 | `INVALID_REQUEST` | Malformed request | Check request format |
| 400 | `INVALID_VOLTAGE_DATA` | Voltage data out of range | Validate voltage values |
| 400 | `MISMATCHED_SERIES_LENGTH` | VL and VO different lengths | Ensure equal array lengths |
| 422 | `VALIDATION_ERROR` | Pydantic validation failed | Check field types and constraints |
| 500 | `PREDICTION_ERROR` | Model prediction failed | Check logs, retry request |
| 500 | `FEATURE_EXTRACTION_ERROR` | Feature extraction failed | Validate input data quality |
| 503 | `MODEL_NOT_READY` | Models not loaded | Wait for model loading |
| 504 | `TIMEOUT_ERROR` | Request timeout | Reduce data size or retry |

### Error Handling Best Practices

1. **Always check HTTP status codes**
2. **Parse error details for specific issues**
3. **Implement retry logic with exponential backoff**
4. **Log errors for debugging**
5. **Validate input data before sending requests**

### Example Error Handling (Python)

```python
import requests
import time
from typing import Optional

def predict_with_retry(client, capacitor_id, cycle_number, vl_series, vo_series, max_retries=3):
    """Make prediction with retry logic"""
    for attempt in range(max_retries):
        try:
            result = client.predict_single(capacitor_id, cycle_number, vl_series, vo_series)
            return result
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                raise e
            
            # Exponential backoff
            wait_time = 2 ** attempt
            print(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
            time.sleep(wait_time)
    
    return None
```

## Performance Considerations

### Response Times

- **Single Prediction**: < 1 second (typical: 150-300ms)
- **Batch Prediction**: < 5 seconds for 100 items
- **Health Check**: < 50ms

### Optimization Tips

1. **Use batch predictions** for multiple items
2. **Cache results** when appropriate
3. **Compress large voltage series** if possible
4. **Use connection pooling** for multiple requests
5. **Implement client-side caching** for repeated requests

### Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Single predictions**: 100/minute per client
- **Batch predictions**: 10/minute per client
- **Rate limit headers** included in responses

## Monitoring and Logging

### Request Logging

All requests are logged with:
- Request ID
- Timestamp
- Client IP
- Processing time
- Response status

### Metrics Available

- Request count by endpoint
- Response time percentiles
- Error rates
- Model performance metrics

### Health Monitoring

Monitor these endpoints for system health:
- `GET /health` - Overall system status
- `GET /model_info` - Model status and metrics

## Security Considerations

### Input Validation

- All inputs are validated using Pydantic models
- Voltage data ranges are checked
- Array lengths are validated
- SQL injection protection (not applicable for this API)

### Rate Limiting

- Prevents abuse and ensures fair usage
- Configurable limits per client
- Automatic blocking of excessive requests

### CORS Configuration

- Currently allows all origins for development
- Configure restrictive CORS for production
- Use environment variables for configuration

### Recommended Production Security

1. **Enable HTTPS/TLS**
2. **Implement authentication** (API keys, JWT)
3. **Use rate limiting**
4. **Enable request logging**
5. **Regular security updates**
6. **Input sanitization**
7. **Error message sanitization**

---

## Support

**API Documentation**: http://localhost:8000/docs
**Technical Support**: support@rul-system.com
**GitHub Issues**: https://github.com/rul-system/issues

**Version**: 1.0.0
**Last Updated**: January 2024