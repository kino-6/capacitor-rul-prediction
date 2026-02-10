# RUL Prediction System - Troubleshooting Guide

## Table of Contents

1. [Quick Diagnostic Checklist](#quick-diagnostic-checklist)
2. [Common Issues and Solutions](#common-issues-and-solutions)
3. [System Health Diagnostics](#system-health-diagnostics)
4. [Performance Issues](#performance-issues)
5. [Data Quality Problems](#data-quality-problems)
6. [Model Performance Issues](#model-performance-issues)
7. [API and Integration Issues](#api-and-integration-issues)
8. [Deployment and Infrastructure](#deployment-and-infrastructure)
9. [Emergency Procedures](#emergency-procedures)
10. [Frequently Asked Questions](#frequently-asked-questions)

## Quick Diagnostic Checklist

When experiencing issues, run through this checklist first:

### ✅ System Status Check
```bash
# 1. Check API health
curl http://localhost:8000/health

# 2. Check model status
curl http://localhost:8000/model_info

# 3. Check system resources
docker stats rul-api
```

### ✅ Basic Functionality Test
```bash
# Test prediction with sample data
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "capacitor_id": "TEST",
    "cycle_number": 1,
    "voltage_data": {
      "vl_series": [1.0, 1.1, 1.2, 1.1, 1.0],
      "vo_series": [0.9, 1.0, 1.1, 1.0, 0.9]
    }
  }'
```

### ✅ Log Review
```bash
# Check recent logs
tail -n 50 logs/api_predictions.jsonl
tail -n 50 logs/system.log
docker-compose logs --tail=50 rul-api
```

## Common Issues and Solutions

### Issue 1: API Returns 503 Service Unavailable

**Symptoms:**
- Health check shows `model_ready: false`
- All prediction requests return 503
- Error message: "Predictor not initialized"

**Possible Causes:**
- Models not loaded properly
- Missing model files
- Insufficient memory
- Startup sequence incomplete

**Solutions:**

1. **Check model files exist:**
   ```bash
   ls -la models/
   # Should contain: rul_model.pkl, anomaly_model.pkl, feature_scaler.pkl
   ```

2. **Restart the service:**
   ```bash
   docker-compose restart rul-api
   # Wait 30-60 seconds for models to load
   ```

3. **Check memory usage:**
   ```bash
   docker stats rul-api
   # Ensure sufficient memory (>2GB recommended)
   ```

4. **Review startup logs:**
   ```bash
   docker-compose logs rul-api | grep -i "startup\|error\|model"
   ```

**Prevention:**
- Ensure adequate system resources
- Verify model files are present before deployment
- Implement health checks in deployment pipeline

### Issue 2: High False Positive Rate

**Symptoms:**
- Too many anomaly alerts for healthy equipment
- Anomaly scores consistently high (>0.7) for normal operation
- Operators losing trust in system alerts

**Possible Causes:**
- Anomaly detection thresholds too sensitive
- Model trained on limited data
- Data distribution shift
- Feature extraction issues

**Solutions:**

1. **Adjust anomaly thresholds:**
   ```python
   # In configuration
   ANOMALY_THRESHOLD = 0.7  # Increase from default 0.5
   FPR_TARGET = 0.03  # Maintain <5% target
   ```

2. **Retrain with more data:**
   ```bash
   python scripts/retrain_models.py --include-recent-data --fpr-target 0.03
   ```

3. **Analyze threshold performance:**
   ```python
   from true_rul.model_evaluator import ModelEvaluator
   evaluator = ModelEvaluator()
   evaluator.analyze_threshold_performance()
   ```

4. **Check data quality:**
   ```bash
   python scripts/validate_input_data.py --check-distribution
   ```

**Prevention:**
- Regular model retraining with recent data
- Continuous monitoring of FPR metrics
- Implement adaptive thresholding

### Issue 3: Slow Prediction Response Times

**Symptoms:**
- Predictions taking >2 seconds
- Timeout errors (504)
- High CPU usage during predictions

**Possible Causes:**
- Large input data size
- Insufficient system resources
- Model complexity
- Lack of caching

**Solutions:**

1. **Enable caching:**
   ```python
   # In API configuration
   ENABLE_PREDICTION_CACHE = True
   CACHE_TTL_SECONDS = 300
   ```

2. **Optimize batch size:**
   ```bash
   # Reduce batch size for large requests
   curl -X POST "/batch_predict" -d '{"predictions": [...]}' # Max 50 items
   ```

3. **Scale system resources:**
   ```yaml
   # In docker-compose.yml
   services:
     rul-api:
       deploy:
         resources:
           limits:
             cpus: '4.0'
             memory: 8G
   ```

4. **Use parallel processing:**
   ```python
   # Enable in configuration
   PARALLEL_PROCESSING = True
   MAX_WORKERS = 4
   ```

**Prevention:**
- Monitor response time metrics
- Implement auto-scaling
- Use load balancing for high traffic

### Issue 4: Inconsistent Predictions

**Symptoms:**
- Large variations in consecutive predictions for same capacitor
- Confidence intervals very wide
- Predictions don't follow expected degradation patterns

**Possible Causes:**
- Noisy input data
- Model instability
- Feature extraction inconsistencies
- Insufficient training data

**Solutions:**

1. **Apply data smoothing:**
   ```python
   from scipy.signal import savgol_filter
   
   # Smooth voltage data
   vl_smooth = savgol_filter(vl_series, window_length=5, polyorder=2)
   vo_smooth = savgol_filter(vo_series, window_length=5, polyorder=2)
   ```

2. **Use ensemble predictions:**
   ```python
   # Enable ensemble mode
   MODEL_TYPE = "ensemble"  # Instead of single model
   ENSEMBLE_WEIGHTS = {"xgboost": 0.4, "lightgbm": 0.4, "rf": 0.2}
   ```

3. **Increase confidence thresholds:**
   ```python
   # Flag predictions with low confidence
   MIN_CONFIDENCE_THRESHOLD = 0.8
   if confidence < MIN_CONFIDENCE_THRESHOLD:
       result.flags.append("low_confidence")
   ```

4. **Validate input data quality:**
   ```python
   def validate_voltage_data(vl_series, vo_series):
       # Check for outliers
       vl_std = np.std(vl_series)
       vo_std = np.std(vo_series)
       
       # Flag if too much variation
       if vl_std > 0.1 or vo_std > 0.1:
           return False, "High voltage variation detected"
       
       return True, "Data quality OK"
   ```

**Prevention:**
- Implement data quality checks
- Use rolling averages for trend analysis
- Regular model validation

### Issue 5: Memory Leaks and Resource Issues

**Symptoms:**
- Gradually increasing memory usage
- System becomes unresponsive over time
- Out of memory errors

**Possible Causes:**
- Memory leaks in model inference
- Large data caching without cleanup
- Accumulating log files
- Resource not properly released

**Solutions:**

1. **Restart service regularly:**
   ```bash
   # Add to cron job
   0 2 * * * docker-compose restart rul-api
   ```

2. **Implement memory monitoring:**
   ```python
   import psutil
   
   def check_memory_usage():
       memory = psutil.virtual_memory()
       if memory.percent > 85:
           logger.warning(f"High memory usage: {memory.percent}%")
           # Trigger cleanup or restart
   ```

3. **Clean up resources:**
   ```python
   # In prediction code
   try:
       result = model.predict(data)
   finally:
       # Explicitly clean up
       del data
       gc.collect()
   ```

4. **Rotate log files:**
   ```bash
   # Configure log rotation
   logrotate -f /etc/logrotate.d/rul-system
   ```

**Prevention:**
- Regular memory monitoring
- Implement resource limits
- Use memory profiling tools

## System Health Diagnostics

### Health Check Procedures

#### 1. API Health Verification
```bash
#!/bin/bash
# health_check.sh

echo "=== RUL System Health Check ==="

# Check API availability
echo "1. Checking API availability..."
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ API is responding"
else
    echo "❌ API is not responding"
    exit 1
fi

# Check model readiness
echo "2. Checking model readiness..."
MODEL_READY=$(curl -s http://localhost:8000/health | jq -r '.model_ready')
if [ "$MODEL_READY" = "true" ]; then
    echo "✅ Models are loaded"
else
    echo "❌ Models are not ready"
fi

# Check system resources
echo "3. Checking system resources..."
MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
echo "Memory usage: ${MEMORY_USAGE}%"

if (( $(echo "$MEMORY_USAGE > 90" | bc -l) )); then
    echo "⚠️  High memory usage"
else
    echo "✅ Memory usage normal"
fi

# Test prediction
echo "4. Testing prediction functionality..."
PREDICTION_RESULT=$(curl -s -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "capacitor_id": "HEALTH_CHECK",
    "cycle_number": 1,
    "voltage_data": {
      "vl_series": [1.0, 1.1, 1.2, 1.1, 1.0],
      "vo_series": [0.9, 1.0, 1.1, 1.0, 0.9]
    }
  }')

if echo "$PREDICTION_RESULT" | jq -e '.rul_cycles' > /dev/null; then
    echo "✅ Prediction test successful"
else
    echo "❌ Prediction test failed"
    echo "Response: $PREDICTION_RESULT"
fi

echo "=== Health Check Complete ==="
```

#### 2. Model Performance Monitoring
```python
# model_health_monitor.py
import json
import numpy as np
from datetime import datetime, timedelta
from true_rul.model_evaluator import ModelEvaluator

class ModelHealthMonitor:
    def __init__(self):
        self.evaluator = ModelEvaluator()
        self.thresholds = {
            'rmse': 10.0,
            'mae': 7.0,
            'r2_score': 0.8,
            'fpr': 0.05
        }
    
    def check_model_performance(self):
        """Check if model performance is within acceptable ranges"""
        results = {}
        
        # Load recent predictions
        recent_predictions = self.load_recent_predictions(days=7)
        
        if len(recent_predictions) < 10:
            results['status'] = 'insufficient_data'
            return results
        
        # Calculate metrics
        metrics = self.evaluator.calculate_metrics(recent_predictions)
        
        # Check against thresholds
        alerts = []
        for metric, value in metrics.items():
            threshold = self.thresholds.get(metric)
            if threshold:
                if metric in ['rmse', 'mae', 'fpr'] and value > threshold:
                    alerts.append(f"{metric} too high: {value:.3f} > {threshold}")
                elif metric == 'r2_score' and value < threshold:
                    alerts.append(f"{metric} too low: {value:.3f} < {threshold}")
        
        results['metrics'] = metrics
        results['alerts'] = alerts
        results['status'] = 'healthy' if not alerts else 'degraded'
        
        return results
    
    def load_recent_predictions(self, days=7):
        """Load predictions from the last N days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        predictions = []
        
        try:
            with open('logs/api_predictions.jsonl', 'r') as f:
                for line in f:
                    pred = json.loads(line)
                    pred_date = datetime.fromisoformat(pred['timestamp'])
                    if pred_date > cutoff_date:
                        predictions.append(pred)
        except FileNotFoundError:
            pass
        
        return predictions

# Usage
monitor = ModelHealthMonitor()
health_report = monitor.check_model_performance()
print(json.dumps(health_report, indent=2))
```

### Automated Monitoring Setup

#### 1. Prometheus Metrics
```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
prediction_counter = Counter('rul_predictions_total', 'Total predictions made')
prediction_duration = Histogram('rul_prediction_duration_seconds', 'Prediction processing time')
model_accuracy = Gauge('rul_model_accuracy', 'Current model accuracy')
false_positive_rate = Gauge('rul_false_positive_rate', 'Current false positive rate')

def setup_metrics():
    """Start metrics server"""
    start_http_server(8001)  # Metrics available at :8001/metrics

def record_prediction(processing_time, accuracy=None):
    """Record prediction metrics"""
    prediction_counter.inc()
    prediction_duration.observe(processing_time)
    if accuracy:
        model_accuracy.set(accuracy)
```

#### 2. Alerting Rules
```yaml
# alerting_rules.yml
groups:
  - name: rul_system_alerts
    rules:
      - alert: HighFalsePositiveRate
        expr: rul_false_positive_rate > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "RUL system false positive rate too high"
          description: "FPR is {{ $value }}, exceeding 5% threshold"
      
      - alert: ModelAccuracyDegraded
        expr: rul_model_accuracy < 0.8
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "RUL model accuracy degraded"
          description: "Model accuracy is {{ $value }}, below 80% threshold"
      
      - alert: PredictionLatencyHigh
        expr: histogram_quantile(0.95, rul_prediction_duration_seconds) > 2.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "RUL prediction latency high"
          description: "95th percentile latency is {{ $value }}s"
```

## Performance Issues

### Optimization Strategies

#### 1. Model Optimization
```python
# model_optimizer.py
import joblib
from sklearn.model_selection import GridSearchCV
import optuna

class ModelOptimizer:
    def __init__(self):
        self.best_params = {}
    
    def optimize_xgboost(self, X_train, y_train, X_val, y_val):
        """Optimize XGBoost hyperparameters"""
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
            
            model = xgb.XGBRegressor(**params, random_state=42)
            model.fit(X_train, y_train)
            
            predictions = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, predictions))
            
            return rmse
        
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)
        
        self.best_params['xgboost'] = study.best_params
        return study.best_params
    
    def create_optimized_model(self, model_type='xgboost'):
        """Create model with optimized parameters"""
        if model_type == 'xgboost' and 'xgboost' in self.best_params:
            return xgb.XGBRegressor(**self.best_params['xgboost'])
        else:
            # Return default model
            return xgb.XGBRegressor()
```

#### 2. Caching Implementation
```python
# caching.py
import redis
import json
import hashlib
from typing import Optional

class PredictionCache:
    def __init__(self, redis_host='localhost', redis_port=6379, ttl=300):
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
        self.ttl = ttl  # Time to live in seconds
    
    def _generate_key(self, capacitor_id: str, cycle_number: int, voltage_data: dict) -> str:
        """Generate cache key from input data"""
        data_str = f"{capacitor_id}_{cycle_number}_{json.dumps(voltage_data, sort_keys=True)}"
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def get_prediction(self, capacitor_id: str, cycle_number: int, voltage_data: dict) -> Optional[dict]:
        """Get cached prediction if available"""
        key = self._generate_key(capacitor_id, cycle_number, voltage_data)
        cached_result = self.redis_client.get(key)
        
        if cached_result:
            return json.loads(cached_result)
        return None
    
    def cache_prediction(self, capacitor_id: str, cycle_number: int, voltage_data: dict, result: dict):
        """Cache prediction result"""
        key = self._generate_key(capacitor_id, cycle_number, voltage_data)
        self.redis_client.setex(key, self.ttl, json.dumps(result))
    
    def clear_cache(self):
        """Clear all cached predictions"""
        for key in self.redis_client.scan_iter(match="*"):
            self.redis_client.delete(key)
```

## Data Quality Problems

### Data Validation Framework

```python
# data_validator.py
import numpy as np
from typing import Tuple, List, Dict
from dataclasses import dataclass

@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    quality_score: float

class DataValidator:
    def __init__(self):
        self.voltage_range = (0.3, 2.5)  # Acceptable voltage range
        self.min_series_length = 10
        self.max_series_length = 10000
        self.max_noise_ratio = 0.3
    
    def validate_voltage_data(self, vl_series: np.ndarray, vo_series: np.ndarray) -> ValidationResult:
        """Comprehensive voltage data validation"""
        errors = []
        warnings = []
        quality_score = 1.0
        
        # Check series lengths
        if len(vl_series) != len(vo_series):
            errors.append(f"Series length mismatch: VL={len(vl_series)}, VO={len(vo_series)}")
        
        if len(vl_series) < self.min_series_length:
            errors.append(f"Series too short: {len(vl_series)} < {self.min_series_length}")
        
        if len(vl_series) > self.max_series_length:
            warnings.append(f"Series very long: {len(vl_series)} > {self.max_series_length}")
            quality_score -= 0.1
        
        # Check voltage ranges
        vl_min, vl_max = np.min(vl_series), np.max(vl_series)
        vo_min, vo_max = np.min(vo_series), np.max(vo_series)
        
        if not (self.voltage_range[0] <= vl_min and vl_max <= self.voltage_range[1]):
            errors.append(f"VL voltage out of range: [{vl_min:.3f}, {vl_max:.3f}]")
        
        if not (self.voltage_range[0] <= vo_min and vo_max <= self.voltage_range[1]):
            errors.append(f"VO voltage out of range: [{vo_min:.3f}, {vo_max:.3f}]")
        
        # Check for NaN or infinite values
        if np.any(np.isnan(vl_series)) or np.any(np.isnan(vo_series)):
            errors.append("NaN values detected in voltage data")
        
        if np.any(np.isinf(vl_series)) or np.any(np.isinf(vo_series)):
            errors.append("Infinite values detected in voltage data")
        
        # Check noise levels
        vl_noise = self._estimate_noise_ratio(vl_series)
        vo_noise = self._estimate_noise_ratio(vo_series)
        
        if vl_noise > self.max_noise_ratio:
            warnings.append(f"High noise in VL series: {vl_noise:.2f}")
            quality_score -= 0.2
        
        if vo_noise > self.max_noise_ratio:
            warnings.append(f"High noise in VO series: {vo_noise:.2f}")
            quality_score -= 0.2
        
        # Check for constant values (sensor failure)
        if np.std(vl_series) < 1e-6:
            errors.append("VL series appears constant (possible sensor failure)")
        
        if np.std(vo_series) < 1e-6:
            errors.append("VO series appears constant (possible sensor failure)")
        
        is_valid = len(errors) == 0
        quality_score = max(0.0, quality_score)
        
        return ValidationResult(is_valid, errors, warnings, quality_score)
    
    def _estimate_noise_ratio(self, series: np.ndarray) -> float:
        """Estimate noise-to-signal ratio"""
        # Use high-frequency components as noise estimate
        from scipy.signal import butter, filtfilt
        
        # High-pass filter to isolate noise
        nyquist = 0.5 * len(series)
        high_cutoff = 0.3 * nyquist
        b, a = butter(4, high_cutoff / nyquist, btype='high')
        noise = filtfilt(b, a, series)
        
        signal_power = np.var(series)
        noise_power = np.var(noise)
        
        return noise_power / signal_power if signal_power > 0 else 0
```

### Data Cleaning Utilities

```python
# data_cleaner.py
import numpy as np
from scipy.signal import savgol_filter, medfilt
from scipy.interpolate import interp1d

class DataCleaner:
    def __init__(self):
        self.outlier_threshold = 3.0  # Standard deviations
    
    def clean_voltage_series(self, vl_series: np.ndarray, vo_series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Clean voltage data using multiple techniques"""
        # Remove outliers
        vl_clean = self._remove_outliers(vl_series)
        vo_clean = self._remove_outliers(vo_series)
        
        # Smooth data
        vl_smooth = self._smooth_series(vl_clean)
        vo_smooth = self._smooth_series(vo_clean)
        
        # Interpolate missing values
        vl_final = self._interpolate_missing(vl_smooth)
        vo_final = self._interpolate_missing(vo_smooth)
        
        return vl_final, vo_final
    
    def _remove_outliers(self, series: np.ndarray) -> np.ndarray:
        """Remove outliers using z-score method"""
        z_scores = np.abs((series - np.mean(series)) / np.std(series))
        return np.where(z_scores > self.outlier_threshold, np.nan, series)
    
    def _smooth_series(self, series: np.ndarray) -> np.ndarray:
        """Apply smoothing filter"""
        # Use median filter for spike removal
        series_filtered = medfilt(series, kernel_size=3)
        
        # Apply Savitzky-Golay filter for smoothing
        window_length = min(11, len(series) // 4)
        if window_length % 2 == 0:
            window_length += 1
        
        if window_length >= 3:
            series_smooth = savgol_filter(series_filtered, window_length, 2)
        else:
            series_smooth = series_filtered
        
        return series_smooth
    
    def _interpolate_missing(self, series: np.ndarray) -> np.ndarray:
        """Interpolate missing (NaN) values"""
        if not np.any(np.isnan(series)):
            return series
        
        # Find valid indices
        valid_indices = ~np.isnan(series)
        if np.sum(valid_indices) < 2:
            # Not enough valid points for interpolation
            return np.full_like(series, np.nanmean(series))
        
        # Interpolate
        x = np.arange(len(series))
        f = interp1d(x[valid_indices], series[valid_indices], 
                    kind='linear', fill_value='extrapolate')
        
        return f(x)
```

## Emergency Procedures

### System Recovery Procedures

#### 1. Complete System Failure
```bash
#!/bin/bash
# emergency_recovery.sh

echo "=== EMERGENCY SYSTEM RECOVERY ==="

# 1. Stop all services
echo "Stopping all services..."
docker-compose down

# 2. Check disk space
echo "Checking disk space..."
df -h
if [ $(df / | tail -1 | awk '{print $5}' | sed 's/%//') -gt 90 ]; then
    echo "WARNING: Low disk space detected"
    # Clean up logs
    find logs/ -name "*.log" -mtime +7 -delete
    find logs/ -name "*.jsonl" -mtime +7 -delete
fi

# 3. Check for corrupted files
echo "Checking model files..."
if [ ! -f "models/rul_model.pkl" ]; then
    echo "ERROR: RUL model file missing"
    echo "Restoring from backup..."
    cp backups/latest/rul_model.pkl models/
fi

# 4. Restart services
echo "Restarting services..."
docker-compose up -d

# 5. Wait for startup
echo "Waiting for system startup..."
sleep 30

# 6. Verify recovery
echo "Verifying system recovery..."
if curl -s http://localhost:8000/health | grep -q "healthy"; then
    echo "✅ System recovery successful"
else
    echo "❌ System recovery failed - manual intervention required"
    exit 1
fi
```

#### 2. Model Corruption Recovery
```python
# model_recovery.py
import os
import shutil
from datetime import datetime
import joblib

class ModelRecovery:
    def __init__(self, model_dir='models', backup_dir='backups'):
        self.model_dir = model_dir
        self.backup_dir = backup_dir
    
    def verify_model_integrity(self):
        """Check if models can be loaded"""
        model_files = ['rul_model.pkl', 'anomaly_model.pkl', 'feature_scaler.pkl']
        corrupted_files = []
        
        for model_file in model_files:
            model_path = os.path.join(self.model_dir, model_file)
            try:
                joblib.load(model_path)
                print(f"✅ {model_file} - OK")
            except Exception as e:
                print(f"❌ {model_file} - CORRUPTED: {e}")
                corrupted_files.append(model_file)
        
        return corrupted_files
    
    def restore_from_backup(self, corrupted_files):
        """Restore corrupted models from backup"""
        backup_timestamp = self._find_latest_backup()
        if not backup_timestamp:
            raise Exception("No backup found")
        
        backup_path = os.path.join(self.backup_dir, backup_timestamp)
        
        for model_file in corrupted_files:
            backup_file = os.path.join(backup_path, model_file)
            target_file = os.path.join(self.model_dir, model_file)
            
            if os.path.exists(backup_file):
                shutil.copy2(backup_file, target_file)
                print(f"Restored {model_file} from backup {backup_timestamp}")
            else:
                print(f"WARNING: Backup for {model_file} not found")
    
    def _find_latest_backup(self):
        """Find the most recent backup"""
        if not os.path.exists(self.backup_dir):
            return None
        
        backups = [d for d in os.listdir(self.backup_dir) 
                  if os.path.isdir(os.path.join(self.backup_dir, d))]
        
        if not backups:
            return None
        
        return max(backups)  # Assumes timestamp format allows string sorting

# Usage
recovery = ModelRecovery()
corrupted = recovery.verify_model_integrity()
if corrupted:
    recovery.restore_from_backup(corrupted)
```

### Backup and Recovery

#### 1. Automated Backup System
```bash
#!/bin/bash
# backup_system.sh

BACKUP_DIR="/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_PATH="$BACKUP_DIR/$TIMESTAMP"

echo "Creating backup: $BACKUP_PATH"

# Create backup directory
mkdir -p "$BACKUP_PATH"

# Backup models
cp -r models/ "$BACKUP_PATH/"

# Backup configuration
cp -r config/ "$BACKUP_PATH/"

# Backup recent logs (last 7 days)
find logs/ -name "*.log" -mtime -7 -exec cp {} "$BACKUP_PATH/" \;
find logs/ -name "*.jsonl" -mtime -7 -exec cp {} "$BACKUP_PATH/" \;

# Backup database (if applicable)
# pg_dump rul_database > "$BACKUP_PATH/database.sql"

# Create backup manifest
cat > "$BACKUP_PATH/manifest.txt" << EOF
Backup created: $(date)
System version: 1.0.0
Models included: $(ls models/)
Configuration files: $(ls config/)
Log files: $(ls logs/ | wc -l) files
EOF

# Compress backup
tar -czf "$BACKUP_PATH.tar.gz" -C "$BACKUP_DIR" "$TIMESTAMP"
rm -rf "$BACKUP_PATH"

# Clean old backups (keep last 30 days)
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +30 -delete

echo "Backup completed: $BACKUP_PATH.tar.gz"
```

## Frequently Asked Questions

### Q1: Why are my predictions inconsistent?

**A:** Inconsistent predictions can be caused by:
- Noisy input data - apply smoothing filters
- Model instability - use ensemble methods
- Insufficient training data - retrain with more data
- Feature extraction issues - validate feature computation

**Solution:** Enable ensemble mode and apply data preprocessing.

### Q2: How do I improve prediction accuracy?

**A:** To improve accuracy:
1. Collect more training data
2. Improve data quality (reduce noise, handle outliers)
3. Tune hyperparameters using cross-validation
4. Use ensemble methods
5. Add domain-specific features

### Q3: What should I do if FPR exceeds 5%?

**A:** If False Positive Rate is too high:
1. Adjust anomaly detection thresholds
2. Retrain models with recent data
3. Implement adaptive thresholding
4. Review data quality and preprocessing
5. Consider ensemble methods for anomaly detection

### Q4: How often should I retrain the models?

**A:** Retrain models when:
- Accuracy drops below 80% (R² < 0.8)
- FPR exceeds 5%
- New equipment types are introduced
- Significant changes in operating conditions
- Monthly scheduled retraining (recommended)

### Q5: What are the minimum system requirements?

**A:** Minimum requirements:
- **CPU**: 4 cores, 2.5 GHz
- **RAM**: 8 GB (16 GB recommended)
- **Storage**: 50 GB SSD
- **Network**: 1 Gbps
- **OS**: Linux (Ubuntu 20.04+), Docker support

### Q6: How do I scale the system for high traffic?

**A:** For scaling:
1. Use load balancing (nginx, HAProxy)
2. Deploy multiple API instances
3. Implement Redis caching
4. Use horizontal pod autoscaling (Kubernetes)
5. Optimize batch processing
6. Consider GPU acceleration for large models

### Q7: What security measures should I implement?

**A:** Security recommendations:
1. Enable HTTPS/TLS encryption
2. Implement API authentication (JWT, API keys)
3. Use rate limiting
4. Enable request logging and monitoring
5. Regular security updates
6. Network segmentation
7. Input validation and sanitization

### Q8: How do I monitor system performance?

**A:** Monitoring setup:
1. Use Prometheus for metrics collection
2. Set up Grafana dashboards
3. Implement alerting rules
4. Monitor response times, accuracy, and FPR
5. Track system resources (CPU, memory, disk)
6. Set up log aggregation (ELK stack)

---

## Getting Help

### Support Channels

1. **Documentation**: Check user manual and API docs first
2. **System Logs**: Review logs for error details
3. **Health Checks**: Run diagnostic scripts
4. **Technical Support**: support@rul-system.com
5. **Emergency Hotline**: +1-800-RUL-HELP (24/7)

### When Contacting Support

Please provide:
- System version and configuration
- Error messages and logs
- Steps to reproduce the issue
- System resource usage
- Recent changes or updates

**Document Version**: 1.0
**Last Updated**: January 2024