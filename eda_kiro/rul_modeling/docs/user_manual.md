# RUL Prediction System - User Manual

## Table of Contents

1. [Introduction](#introduction)
2. [System Overview](#system-overview)
3. [Getting Started](#getting-started)
4. [User Interfaces](#user-interfaces)
5. [Operating Procedures](#operating-procedures)
6. [Interpreting Results](#interpreting-results)
7. [Troubleshooting](#troubleshooting)
8. [Maintenance](#maintenance)
9. [Safety Guidelines](#safety-guidelines)

## Introduction

The RUL (Remaining Useful Life) Prediction System is an advanced machine learning platform designed to predict the remaining operational cycles of capacitors before failure. This system helps maintenance engineers and operators make informed decisions about equipment replacement and maintenance scheduling.

### Key Features

- **True RUL Prediction**: Predicts exact remaining cycles (not just binary classification)
- **High Precision**: Achieves False Positive Rate (FPR) < 5%
- **Staged Degradation**: Provides continuous degradation progression with confidence intervals
- **Real-time Processing**: Sub-second prediction response times
- **Interpretable Results**: SHAP values and feature importance for decision support

### Target Users

- **System Operators**: Monitor equipment health and receive alerts
- **Maintenance Engineers**: Plan maintenance schedules and resource allocation
- **Plant Managers**: Make strategic decisions about equipment replacement
- **Data Analysts**: Analyze degradation patterns and system performance

## System Overview

### Architecture Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Input    │───▶│  RUL Predictor  │───▶│   Results       │
│                 │    │                 │    │                 │
│ • Voltage Data  │    │ • Feature Ext.  │    │ • RUL Cycles    │
│ • Cycle Info    │    │ • ML Models     │    │ • Confidence    │
│ • Timestamps    │    │ • Anomaly Det.  │    │ • Degradation   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow

1. **Input**: Voltage time-series data (VL, VO) from capacitor cycles
2. **Processing**: Feature extraction and machine learning inference
3. **Output**: RUL prediction with confidence intervals and interpretability

### Supported Data Formats

- **ES12 Dataset**: NASA PCOE capacitor degradation data
- **Voltage Series**: Input (VL) and Output (VO) voltage measurements
- **Cycle Information**: Cycle numbers and timestamps

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Access to voltage measurement data
- Basic understanding of capacitor operation

### Quick Start Guide

#### 1. System Access

The system provides multiple interfaces:

- **REST API**: For programmatic access
- **Web Dashboard**: For interactive monitoring
- **Command Line**: For batch processing

#### 2. First Prediction

**Using the REST API:**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "capacitor_id": "C1",
    "cycle_number": 50,
    "voltage_data": {
      "vl_series": [1.0, 1.1, 1.2, ...],
      "vo_series": [0.9, 1.0, 1.1, ...]
    }
  }'
```

**Expected Response:**
```json
{
  "rul_cycles": 45,
  "rul_confidence_lower": 38,
  "rul_confidence_upper": 52,
  "degradation_score": 0.35,
  "degradation_stage": "early_degradation",
  "anomaly_flag": false,
  "anomaly_score": 0.12,
  "feature_importance": {...},
  "timestamp": "2024-01-15T10:30:00",
  "model_version": "1.0.0"
}
```

#### 3. Understanding Results

- **RUL Cycles**: Predicted remaining operational cycles
- **Confidence Interval**: Statistical range of uncertainty
- **Degradation Stage**: Current health status (healthy, early_degradation, advanced_degradation, critical)
- **Anomaly Flag**: Whether current behavior is anomalous

## User Interfaces

### 1. REST API

The primary interface for system integration.

**Base URL**: `http://localhost:8000`

**Key Endpoints**:
- `POST /predict` - Single prediction
- `POST /batch_predict` - Multiple predictions
- `GET /health` - System health check
- `GET /model_info` - Model information

**Authentication**: Currently open access (configure security as needed)

### 2. Web Dashboard

Interactive monitoring interface for operators.

**Access**: `http://localhost:8000/dashboard`

**Features**:
- Real-time prediction monitoring
- Historical trend visualization
- Alert management
- System health status

### 3. Command Line Interface

For batch processing and automation.

**Usage**:
```bash
python -m true_rul.cli predict --input data.csv --output results.json
python -m true_rul.cli batch --directory /path/to/data
```

## Operating Procedures

### Daily Operations

#### 1. System Health Check

**Morning Routine**:
1. Check system status: `GET /health`
2. Verify model readiness
3. Review overnight alerts
4. Check log files for errors

**Health Check Indicators**:
- ✅ `status: "healthy"` - System operational
- ✅ `model_ready: true` - Models loaded
- ⚠️ `uptime_seconds` - Monitor for unexpected restarts

#### 2. Data Input Procedures

**Voltage Data Requirements**:
- **Sampling Rate**: Consistent across measurements
- **Data Quality**: No missing values or outliers
- **Format**: Numeric arrays for VL and VO series
- **Metadata**: Capacitor ID and cycle number

**Data Validation Checklist**:
- [ ] Voltage values within expected range (0.5V - 2.0V)
- [ ] Series lengths match (VL and VO same length)
- [ ] No NaN or infinite values
- [ ] Timestamps in correct format

#### 3. Prediction Workflow

**Single Prediction Process**:
1. Prepare voltage data
2. Submit prediction request
3. Validate response format
4. Interpret results
5. Take appropriate action

**Batch Processing**:
1. Organize data files
2. Submit batch request (max 100 items)
3. Monitor processing progress
4. Review batch results
5. Handle any failures

### Alert Management

#### Alert Types

1. **Critical Alerts** (Immediate Action Required)
   - RUL < 10 cycles
   - Degradation stage: "critical"
   - System errors or failures

2. **Warning Alerts** (Monitor Closely)
   - RUL < 25 cycles
   - Degradation stage: "advanced_degradation"
   - High anomaly scores (> 0.8)

3. **Information Alerts** (Routine Monitoring)
   - Degradation stage transitions
   - Model performance updates
   - System maintenance notifications

#### Response Procedures

**Critical Alert Response**:
1. Verify alert validity
2. Check equipment status
3. Schedule immediate inspection
4. Prepare replacement parts
5. Notify maintenance team

**Warning Alert Response**:
1. Increase monitoring frequency
2. Review historical trends
3. Plan maintenance window
4. Update maintenance schedule

### Maintenance Scheduling

#### Predictive Maintenance Integration

**RUL-Based Scheduling**:
- **RUL > 50 cycles**: Normal operation, routine monitoring
- **RUL 25-50 cycles**: Increased monitoring, prepare maintenance
- **RUL 10-25 cycles**: Schedule maintenance window
- **RUL < 10 cycles**: Immediate maintenance required

**Confidence Considerations**:
- **High Confidence** (narrow interval): Trust prediction
- **Low Confidence** (wide interval): Increase monitoring, consider early maintenance

## Interpreting Results

### Prediction Output Explained

#### RUL Cycles
- **Definition**: Predicted number of operational cycles before failure
- **Range**: 0 to maximum observed cycles (typically 200)
- **Interpretation**: Higher values indicate healthier equipment

#### Confidence Intervals
- **Lower Bound**: Conservative estimate (95% confidence)
- **Upper Bound**: Optimistic estimate (95% confidence)
- **Width**: Indicates prediction uncertainty

**Example Interpretation**:
```json
{
  "rul_cycles": 45,
  "rul_confidence_lower": 38,
  "rul_confidence_upper": 52
}
```
- **Best Estimate**: 45 cycles remaining
- **Conservative Planning**: Plan for 38 cycles
- **Uncertainty**: ±7 cycles (relatively confident)

#### Degradation Stages

1. **Healthy** (degradation_score: 0.0-0.3)
   - Normal operation
   - No immediate concerns
   - Continue routine monitoring

2. **Early Degradation** (degradation_score: 0.3-0.6)
   - Initial signs of wear
   - Increase monitoring frequency
   - Plan future maintenance

3. **Advanced Degradation** (degradation_score: 0.6-0.8)
   - Significant degradation detected
   - Schedule maintenance soon
   - Prepare replacement parts

4. **Critical** (degradation_score: 0.8-1.0)
   - Failure imminent
   - Immediate action required
   - Consider emergency shutdown

#### Feature Importance

Shows which measurements most influenced the prediction:

```json
{
  "feature_importance": {
    "responsiveness_feature_1": 0.25,
    "voltage_std": 0.18,
    "frequency_peak": 0.15,
    "rolling_mean": 0.12,
    ...
  }
}
```

**Interpretation**:
- Higher values indicate more influential features
- Helps understand failure mechanisms
- Guides sensor placement and monitoring focus

### Anomaly Detection

#### Anomaly Flag
- **true**: Current behavior is anomalous
- **false**: Behavior within normal range

#### Anomaly Score
- **Range**: 0.0 (normal) to 1.0 (highly anomalous)
- **Threshold**: Typically 0.5 for binary classification
- **Interpretation**: Higher scores indicate more unusual behavior

**Action Guidelines**:
- **Score < 0.3**: Normal operation
- **Score 0.3-0.7**: Monitor closely
- **Score > 0.7**: Investigate immediately

### Trend Analysis

#### Historical Context
- Compare current prediction with previous cycles
- Look for acceleration in degradation
- Identify sudden changes in behavior

#### Pattern Recognition
- **Linear Degradation**: Steady decline over time
- **Accelerated Degradation**: Increasing rate of decline
- **Step Changes**: Sudden shifts in behavior

## Troubleshooting

### Common Issues

#### 1. Prediction Errors

**Symptom**: HTTP 500 errors or invalid predictions

**Possible Causes**:
- Invalid input data format
- Missing voltage measurements
- Model not loaded properly

**Solutions**:
1. Validate input data format
2. Check voltage data ranges
3. Restart the service
4. Check log files for detailed errors

**Example Fix**:
```bash
# Check service status
curl http://localhost:8000/health

# Restart service if needed
docker-compose restart rul-api

# Check logs
docker-compose logs rul-api
```

#### 2. High False Positive Rate

**Symptom**: Too many anomaly alerts for healthy equipment

**Possible Causes**:
- Threshold too sensitive
- Model drift
- Data quality issues

**Solutions**:
1. Adjust anomaly thresholds
2. Retrain models with recent data
3. Improve data preprocessing

#### 3. Slow Response Times

**Symptom**: Predictions taking > 1 second

**Possible Causes**:
- High system load
- Large batch requests
- Memory issues

**Solutions**:
1. Reduce batch sizes
2. Scale up system resources
3. Enable caching
4. Use parallel processing

#### 4. Inconsistent Predictions

**Symptom**: Large variations in consecutive predictions

**Possible Causes**:
- Noisy input data
- Model instability
- Feature extraction issues

**Solutions**:
1. Apply data smoothing
2. Use ensemble models
3. Increase confidence thresholds

### Diagnostic Procedures

#### System Health Diagnosis

1. **Check API Status**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Verify Model Loading**:
   ```bash
   curl http://localhost:8000/model_info
   ```

3. **Test Prediction**:
   ```bash
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d @test_data.json
   ```

4. **Review Logs**:
   ```bash
   tail -f logs/api_predictions.jsonl
   tail -f logs/system.log
   ```

#### Data Quality Checks

1. **Voltage Range Validation**:
   - VL series: 0.5V - 2.0V
   - VO series: 0.4V - 1.8V

2. **Series Length Consistency**:
   - VL and VO same length
   - Typical length: 100-1000 points

3. **Missing Data Detection**:
   - No NaN values
   - No infinite values
   - Complete time series

#### Performance Monitoring

1. **Response Time Tracking**:
   - Target: < 1 second
   - Monitor 95th percentile
   - Alert if > 2 seconds

2. **Accuracy Monitoring**:
   - Track prediction vs actual
   - Monitor confidence intervals
   - Alert on accuracy degradation

3. **Resource Usage**:
   - CPU utilization
   - Memory consumption
   - Disk space

### Error Codes

| Code | Description | Action |
|------|-------------|---------|
| 400 | Bad Request | Check input format |
| 404 | Not Found | Verify endpoint URL |
| 500 | Internal Error | Check logs, restart service |
| 503 | Service Unavailable | Models not loaded |
| 504 | Timeout | Reduce request size |

## Maintenance

### Routine Maintenance

#### Daily Tasks
- [ ] Check system health status
- [ ] Review prediction logs
- [ ] Monitor alert queue
- [ ] Verify data ingestion

#### Weekly Tasks
- [ ] Analyze prediction accuracy
- [ ] Review false positive rates
- [ ] Check system performance metrics
- [ ] Update documentation

#### Monthly Tasks
- [ ] Model performance evaluation
- [ ] System backup verification
- [ ] Security updates
- [ ] Capacity planning review

### Model Maintenance

#### Performance Monitoring
- Track prediction accuracy over time
- Monitor false positive/negative rates
- Analyze confidence interval coverage
- Review feature importance stability

#### Retraining Triggers
- Accuracy drops below threshold (R² < 0.8)
- FPR exceeds 5%
- Significant data distribution changes
- New equipment types introduced

#### Model Updates
1. Collect new training data
2. Validate data quality
3. Retrain models
4. Evaluate performance
5. Deploy updated models
6. Monitor post-deployment performance

### System Updates

#### Software Updates
- Regular security patches
- Dependency updates
- Feature enhancements
- Bug fixes

#### Hardware Maintenance
- Server health monitoring
- Storage capacity management
- Network performance optimization
- Backup system verification

## Safety Guidelines

### Operational Safety

#### Critical Warnings
⚠️ **Never ignore critical alerts** - Equipment failure can cause safety hazards
⚠️ **Verify predictions** - Use engineering judgment alongside system predictions
⚠️ **Maintain backups** - Ensure prediction history is preserved
⚠️ **Monitor system health** - Undetected system failures can lead to missed alerts

#### Best Practices
- Always validate unusual predictions
- Maintain manual override capabilities
- Keep backup monitoring systems
- Train operators on emergency procedures

#### Emergency Procedures

**System Failure**:
1. Switch to manual monitoring
2. Notify technical support
3. Implement backup procedures
4. Document incident details

**False Alert Storm**:
1. Verify system status
2. Adjust alert thresholds temporarily
3. Investigate root cause
4. Implement corrective measures

### Data Security

#### Access Control
- Implement user authentication
- Use role-based permissions
- Monitor access logs
- Regular security audits

#### Data Protection
- Encrypt sensitive data
- Secure API endpoints
- Regular backup procedures
- Incident response plan

---

## Support and Contact Information

**Technical Support**: support@rul-system.com
**Documentation**: https://docs.rul-system.com
**Emergency Contact**: +1-800-RUL-HELP

**System Version**: 1.0.0
**Document Version**: 1.0
**Last Updated**: January 2024