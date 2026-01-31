#!/usr/bin/env python3
"""
Demonstration of Prediction Aggregation and Confidence Estimation

This script shows how to use the PredictionAggregator and ConfidenceEstimator
classes to combine RUL predictions and anomaly detection results into unified
prediction results with confidence intervals.
"""

import numpy as np
from datetime import datetime
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from true_rul.prediction_aggregator import PredictionAggregator
from true_rul.confidence_estimator import ConfidenceEstimator
from true_rul.data_structures import PredictionResult


class MockRULModel:
    """Mock RUL regression model for demonstration"""
    
    def __init__(self, base_prediction=50.0, variance=5.0):
        self.base_prediction = base_prediction
        self.variance = variance
    
    def predict(self, x):
        """Mock predict method with some variance"""
        # Add some random noise to simulate model uncertainty
        noise = np.random.normal(0, self.variance)
        return max(0, self.base_prediction + noise)


def demonstrate_basic_usage():
    """Demonstrate basic usage of prediction aggregation"""
    print("=== Basic Prediction Aggregation Demo ===")
    
    # Initialize components
    aggregator = PredictionAggregator(model_version="demo-1.0")
    confidence_estimator = ConfidenceEstimator(method="ensemble")
    
    # Simulate RUL predictions from multiple models
    rul_models = [
        MockRULModel(base_prediction=48.0, variance=3.0),
        MockRULModel(base_prediction=50.0, variance=2.0),
        MockRULModel(base_prediction=52.0, variance=4.0)
    ]
    
    # Sample input (would be actual features in practice)
    sample_input = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    
    # Get RUL predictions
    rul_predictions = []
    for i, model in enumerate(rul_models):
        pred = model.predict(sample_input)
        rul_predictions.append(pred)
        print(f"Model {i+1} RUL prediction: {pred:.1f} cycles")
    
    # Calculate ensemble mean and confidence interval
    rul_mean = np.mean(rul_predictions)
    rul_lower, rul_upper = confidence_estimator.estimate_confidence_ensemble(rul_predictions)
    
    print(f"Ensemble RUL prediction: {rul_mean:.1f} cycles")
    print(f"95% Confidence interval: [{rul_lower:.1f}, {rul_upper:.1f}] cycles")
    
    # Simulate anomaly detection results
    anomaly_flag = False
    anomaly_score = 0.25
    feature_importance = {
        'voltage_response': 0.30,
        'frequency_features': 0.25,
        'statistical_features': 0.20,
        'trend_features': 0.15,
        'rolling_features': 0.10
    }
    
    print(f"Anomaly detected: {anomaly_flag}")
    print(f"Anomaly score: {anomaly_score:.3f}")
    
    # Aggregate all predictions
    result = aggregator.aggregate(
        rul_pred=rul_mean,
        rul_confidence_lower=rul_lower,
        rul_confidence_upper=rul_upper,
        anomaly_flag=anomaly_flag,
        anomaly_score=anomaly_score,
        feature_importance=feature_importance,
        capacitor_id="DEMO_C1",
        cycle_number=25
    )
    
    # Display final result
    print("\n--- Final Prediction Result ---")
    print(f"Capacitor ID: {result.capacitor_id}")
    print(f"Cycle Number: {result.cycle_number}")
    print(f"RUL Prediction: {result.rul_cycles} cycles")
    print(f"Confidence Interval: [{result.rul_confidence_lower}, {result.rul_confidence_upper}] cycles")
    print(f"Degradation Score: {result.degradation_score:.3f}")
    print(f"Degradation Stage: {result.degradation_stage}")
    print(f"Anomaly Flag: {result.anomaly_flag}")
    print(f"Anomaly Score: {result.anomaly_score:.3f}")
    print(f"Model Version: {result.model_version}")
    print(f"Timestamp: {result.timestamp}")
    
    return result


def demonstrate_different_scenarios():
    """Demonstrate prediction aggregation for different degradation scenarios"""
    print("\n=== Different Degradation Scenarios ===")
    
    aggregator = PredictionAggregator(model_version="demo-1.0")
    confidence_estimator = ConfidenceEstimator(method="ensemble")
    
    scenarios = [
        {
            'name': 'Healthy Capacitor',
            'rul_predictions': [180.0, 175.0, 185.0],
            'anomaly_score': 0.05,
            'anomaly_flag': False,
            'cycle_number': 15
        },
        {
            'name': 'Early Degradation',
            'rul_predictions': [120.0, 115.0, 125.0],
            'anomaly_score': 0.35,
            'anomaly_flag': False,
            'cycle_number': 80
        },
        {
            'name': 'Advanced Degradation',
            'rul_predictions': [45.0, 40.0, 50.0],
            'anomaly_score': 0.65,
            'anomaly_flag': True,
            'cycle_number': 155
        },
        {
            'name': 'Critical State',
            'rul_predictions': [8.0, 10.0, 6.0],
            'anomaly_score': 0.90,
            'anomaly_flag': True,
            'cycle_number': 192
        }
    ]
    
    feature_importance = {
        'voltage_response': 0.30,
        'frequency_features': 0.25,
        'statistical_features': 0.20,
        'trend_features': 0.15,
        'rolling_features': 0.10
    }
    
    for scenario in scenarios:
        print(f"\n--- {scenario['name']} ---")
        
        # Calculate confidence interval
        rul_mean = np.mean(scenario['rul_predictions'])
        rul_lower, rul_upper = confidence_estimator.estimate_confidence_ensemble(
            scenario['rul_predictions']
        )
        
        # Aggregate predictions
        result = aggregator.aggregate(
            rul_pred=rul_mean,
            rul_confidence_lower=rul_lower,
            rul_confidence_upper=rul_upper,
            anomaly_flag=scenario['anomaly_flag'],
            anomaly_score=scenario['anomaly_score'],
            feature_importance=feature_importance,
            capacitor_id=f"DEMO_{scenario['name'].replace(' ', '_').upper()}",
            cycle_number=scenario['cycle_number']
        )
        
        print(f"RUL: {result.rul_cycles} cycles [{result.rul_confidence_lower}-{result.rul_confidence_upper}]")
        print(f"Degradation: {result.degradation_score:.3f} ({result.degradation_stage})")
        print(f"Anomaly: {result.anomaly_flag} (score: {result.anomaly_score:.3f})")


def demonstrate_confidence_methods():
    """Demonstrate different confidence estimation methods"""
    print("\n=== Confidence Estimation Methods ===")
    
    # Sample predictions with different variance levels
    low_variance_predictions = [49.0, 50.0, 51.0, 49.5, 50.5]
    high_variance_predictions = [30.0, 50.0, 70.0, 40.0, 60.0]
    
    # Test ensemble variance method
    ensemble_estimator = ConfidenceEstimator(method="ensemble", confidence_level=0.95)
    
    print("Low Variance Predictions:", low_variance_predictions)
    low_var_lower, low_var_upper = ensemble_estimator.estimate_confidence_ensemble(low_variance_predictions)
    print(f"95% CI: [{low_var_lower:.1f}, {low_var_upper:.1f}] (width: {low_var_upper - low_var_lower:.1f})")
    
    print("\nHigh Variance Predictions:", high_variance_predictions)
    high_var_lower, high_var_upper = ensemble_estimator.estimate_confidence_ensemble(high_variance_predictions)
    print(f"95% CI: [{high_var_lower:.1f}, {high_var_upper:.1f}] (width: {high_var_upper - high_var_lower:.1f})")
    
    # Test different confidence levels
    print("\nDifferent Confidence Levels (same predictions):")
    for conf_level in [0.90, 0.95, 0.99]:
        lower, upper = ensemble_estimator.estimate_confidence_ensemble(
            high_variance_predictions, confidence_level=conf_level
        )
        print(f"{conf_level*100:.0f}% CI: [{lower:.1f}, {upper:.1f}] (width: {upper - lower:.1f})")


def demonstrate_degradation_history():
    """Demonstrate impact of degradation history on predictions"""
    print("\n=== Degradation History Impact ===")
    
    aggregator = PredictionAggregator(model_version="demo-1.0")
    
    # Base prediction parameters
    base_rul = 75.0
    base_anomaly_score = 0.3
    feature_importance = {'feature_1': 1.0}
    
    # Different degradation histories
    histories = {
        'Stable': [0.20, 0.21, 0.19, 0.20, 0.22],
        'Increasing': [0.10, 0.15, 0.25, 0.35, 0.45],
        'Decreasing': [0.40, 0.35, 0.25, 0.20, 0.15],
        'Volatile': [0.10, 0.30, 0.15, 0.35, 0.20]
    }
    
    for history_type, history in histories.items():
        result = aggregator.aggregate(
            rul_pred=base_rul,
            rul_confidence_lower=base_rul - 5,
            rul_confidence_upper=base_rul + 5,
            anomaly_flag=False,
            anomaly_score=base_anomaly_score,
            feature_importance=feature_importance,
            degradation_history=history,
            capacitor_id=f"HIST_{history_type.upper()}"
        )
        
        print(f"{history_type} History: {history}")
        print(f"  Final Degradation Score: {result.degradation_score:.3f}")
        print(f"  Degradation Stage: {result.degradation_stage}")


def demonstrate_serialization():
    """Demonstrate result serialization"""
    print("\n=== Result Serialization ===")
    
    aggregator = PredictionAggregator(model_version="demo-1.0")
    
    # Create a sample result
    result = aggregator.aggregate(
        rul_pred=100.0,
        rul_confidence_lower=95.0,
        rul_confidence_upper=105.0,
        anomaly_flag=False,
        anomaly_score=0.2,
        feature_importance={
            'voltage_response': 0.4,
            'frequency_features': 0.3,
            'statistical_features': 0.3
        },
        capacitor_id="SERIAL_DEMO_C1",
        cycle_number=50
    )
    
    # Convert to dictionary
    result_dict = result.to_dict()
    print("Dictionary format:")
    for key, value in result_dict.items():
        print(f"  {key}: {value}")
    
    # Convert to JSON
    result_json = result.to_json()
    print(f"\nJSON format:\n{result_json}")


def main():
    """Main demonstration function"""
    print("Prediction Aggregation and Confidence Estimation Demo")
    print("=" * 60)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    try:
        # Run all demonstrations
        demonstrate_basic_usage()
        demonstrate_different_scenarios()
        demonstrate_confidence_methods()
        demonstrate_degradation_history()
        demonstrate_serialization()
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        
    except Exception as e:
        print(f"Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())