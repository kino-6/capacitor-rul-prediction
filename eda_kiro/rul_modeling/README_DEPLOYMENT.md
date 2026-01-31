# RUL Prediction System - Deployment Guide

## Quick Start (5 minutes)

### 1. Build and Run
```bash
# Build the Docker image
docker-compose build

# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

### 2. Test API
```bash
# Health check
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "capacitor_id": "TEST_C1",
    "cycle_number": 10,
    "voltage_data": {
      "vl_series": [1.0, 1.1, 1.2],
      "vo_series": [0.9, 1.0, 1.1]
    }
  }'
```

## Performance Configuration

### High-Performance Setup
- **CPU**: 8+ cores recommended
- **Memory**: 8GB+ recommended
- **Workers**: Set to CPU cores
- **Redis**: For caching predictions

### Environment Variables
```bash
WORKERS=8                    # Number of API workers
MAX_REQUESTS=1000           # Requests per worker before restart
LOG_LEVEL=INFO              # Logging level
REDIS_URL=redis://redis:6379 # Redis connection
```

## API Endpoints

- `GET /health` - Health check
- `GET /model_info` - Model information
- `POST /predict` - Single prediction
- `POST /batch_predict` - Batch predictions (up to 100)

## Monitoring

### Logs
```bash
# View API logs
docker-compose logs -f rul-api

# View structured logs
tail -f logs/api_predictions.jsonl | jq
```

### Metrics
- Processing time per prediction
- Success/error rates
- Memory usage
- CPU utilization

## Scaling

### Horizontal Scaling
```bash
# Scale API service
docker-compose up -d --scale rul-api=4
```

### Load Balancing
Configure nginx for load balancing multiple API instances.

## Production Deployment

1. **Security**: Add authentication, HTTPS, rate limiting
2. **Monitoring**: Add Prometheus, Grafana
3. **Backup**: Regular model and log backups
4. **Updates**: Blue-green deployment strategy