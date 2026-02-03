"""
Integration script for FPR Monitoring with RUL Prediction System

This script integrates the FPR monitoring system with the existing
RUL prediction pipeline, enabling real-time monitoring of model performance.

Requirements: 10.3, 5.5
"""

import sys
import logging
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from true_rul.fpr_monitor import FPRMonitor, AlertConfig, create_fpr_monitor
from true_rul.fpr_dashboard import create_fpr_dashboard
from true_rul.rul_predictor import RULPredictor
from true_rul.data_loader import DataLoader
from true_rul.data_structures import CycleData, PredictionResult
from true_rul.structured_logger import configure_prediction_logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MonitoredRULPredictor:
    """
    RUL Predictor with integrated FPR monitoring
    
    This class wraps the standard RUL predictor and automatically
    records predictions for FPR monitoring and alerting.
    """
    
    def __init__(
        self,
        rul_predictor: RULPredictor,
        fpr_monitor: FPRMonitor,
        enable_ground_truth_collection: bool = False
    ):
        """
        Initialize monitored RUL predictor
        
        Args:
            rul_predictor: Base RUL predictor
            fpr_monitor: FPR monitor for recording predictions
            enable_ground_truth_collection: Whether to collect ground truth labels
        """
        self.rul_predictor = rul_predictor
        self.fpr_monitor = fpr_monitor
        self.enable_ground_truth_collection = enable_ground_truth_collection
        
        # Ground truth collection (for validation datasets)
        self.ground_truth_labels: Dict[str, bool] = {}
        
        logger.info("Monitored RUL Predictor initialized")
    
    def set_ground_truth_label(self, capacitor_id: str, cycle_number: int, is_anomaly: bool):
        """
        Set ground truth label for a specific prediction
        
        Args:
            capacitor_id: Capacitor identifier
            cycle_number: Cycle number
            is_anomaly: True if this cycle is anomalous
        """
        key = f"{capacitor_id}_{cycle_number}"
        self.ground_truth_labels[key] = is_anomaly
    
    def predict(
        self,
        cycle_data: CycleData,
        capacitor_id: str,
        cycle_number: int,
        ground_truth_anomaly: Optional[bool] = None
    ) -> PredictionResult:
        """
        Make prediction and record for FPR monitoring
        
        Args:
            cycle_data: Cycle data for prediction
            capacitor_id: Capacitor identifier
            cycle_number: Cycle number
            ground_truth_anomaly: Optional ground truth anomaly label
            
        Returns:
            Prediction result
        """
        # Make prediction using base predictor
        prediction_result = self.rul_predictor.predict(cycle_data)
        
        # Determine ground truth label
        true_anomaly_label = ground_truth_anomaly
        
        if true_anomaly_label is None and self.enable_ground_truth_collection:
            key = f"{capacitor_id}_{cycle_number}"
            true_anomaly_label = self.ground_truth_labels.get(key)
        
        # Record prediction for FPR monitoring
        self.fpr_monitor.record_prediction(
            prediction_result=prediction_result,
            true_anomaly_label=true_anomaly_label,
            capacitor_id=capacitor_id
        )
        
        return prediction_result
    
    def predict_batch(
        self,
        cycle_data_list: list,
        capacitor_ids: list,
        cycle_numbers: list,
        ground_truth_anomalies: Optional[list] = None
    ) -> list:
        """
        Make batch predictions with FPR monitoring
        
        Args:
            cycle_data_list: List of cycle data
            capacitor_ids: List of capacitor identifiers
            cycle_numbers: List of cycle numbers
            ground_truth_anomalies: Optional list of ground truth labels
            
        Returns:
            List of prediction results
        """
        results = []
        
        for i, cycle_data in enumerate(cycle_data_list):
            capacitor_id = capacitor_ids[i]
            cycle_number = cycle_numbers[i]
            ground_truth = ground_truth_anomalies[i] if ground_truth_anomalies else None
            
            result = self.predict(
                cycle_data=cycle_data,
                capacitor_id=capacitor_id,
                cycle_number=cycle_number,
                ground_truth_anomaly=ground_truth
            )
            results.append(result)
        
        return results


def setup_fpr_monitoring_for_validation():
    """
    Setup FPR monitoring for validation dataset evaluation
    
    This function demonstrates how to integrate FPR monitoring
    when evaluating the model on a validation dataset with known labels.
    """
    logger.info("Setting up FPR monitoring for validation dataset")
    
    # Configure logging
    configure_prediction_logging(log_file="logs/fpr_validation_monitoring.jsonl")
    
    # Create FPR monitor with production-like settings
    config = AlertConfig(
        fpr_threshold=0.05,
        fpr_warning_threshold=0.03,
        min_predictions_for_alert=10,
        alert_cooldown_minutes=15,
        drift_detection_window_hours=24,
        enable_email_alerts=False,  # Disable for testing
        enable_webhook_alerts=False
    )
    
    fpr_monitor = FPRMonitor(
        config=config,
        db_path="validation_fpr_monitoring.db",
        monitoring_window_minutes=30
    )
    
    # Start monitoring
    fpr_monitor.start_monitoring()
    
    try:
        # Load validation data
        data_loader = DataLoader()
        
        # For demonstration, we'll use a subset of ES12 data
        # In practice, you would load your actual validation dataset
        logger.info("Loading validation dataset...")
        
        # Create RUL predictor (assuming models are trained)
        rul_predictor = RULPredictor()
        
        # Check if models are available
        if not rul_predictor.is_ready():
            logger.warning("RUL predictor models not available. Using mock predictions.")
            # You would typically load trained models here
            # rul_predictor.load_models("path/to/trained/models")
        
        # Create monitored predictor
        monitored_predictor = MonitoredRULPredictor(
            rul_predictor=rul_predictor,
            fpr_monitor=fpr_monitor,
            enable_ground_truth_collection=True
        )
        
        # Simulate validation predictions with known ground truth
        # In practice, you would iterate through your actual validation dataset
        logger.info("Running validation predictions with FPR monitoring...")
        
        validation_results = []
        
        # Example: Process validation data
        # This is a simplified example - replace with your actual validation loop
        for capacitor_id in ["C1", "C2"]:  # Example capacitor IDs
            for cycle_num in range(1, 51):  # Example cycle range
                try:
                    # Create example cycle data (replace with actual data loading)
                    cycle_data = CycleData(
                        cycle_number=cycle_num,
                        vl_series=None,  # Would be loaded from actual data
                        vo_series=None,  # Would be loaded from actual data
                        timestamp=datetime.now().timestamp()
                    )
                    
                    # Determine ground truth (example logic - replace with actual labels)
                    # In ES12 dataset, cycles > 150 are typically considered degraded
                    is_anomaly = cycle_num > 150
                    
                    # Make prediction with monitoring
                    if rul_predictor.is_ready():
                        result = monitored_predictor.predict(
                            cycle_data=cycle_data,
                            capacitor_id=capacitor_id,
                            cycle_number=cycle_num,
                            ground_truth_anomaly=is_anomaly
                        )
                        validation_results.append(result)
                    
                except Exception as e:
                        logger.error(f"Error processing {capacitor_id} cycle {cycle_num}: {e}")
                        continue
        
        logger.info(f"Processed {len(validation_results)} validation predictions")
        
        # Wait for monitoring to process all predictions
        import time
        time.sleep(30)
        
        # Generate FPR report
        report = fpr_monitor.generate_fpr_report(hours=1)
        logger.info("Validation FPR Report:")
        print(report)
        
        # Check if FPR requirement is met
        current_metrics = fpr_monitor.calculate_current_fpr()
        if current_metrics:
            fpr_met = current_metrics.fpr < 0.05
            logger.info(f"FPR Requirement (<5%): {'✓ MET' if fpr_met else '✗ NOT MET'}")
            logger.info(f"Actual FPR: {current_metrics.fpr:.3f}")
        
        # Get active alerts
        active_alerts = fpr_monitor.db.get_active_alerts()
        if active_alerts:
            logger.warning(f"Active alerts: {len(active_alerts)}")
            for alert in active_alerts:
                logger.warning(f"  - {alert.alert_type}: {alert.message}")
        else:
            logger.info("No active alerts")
        
    finally:
        fpr_monitor.stop_monitoring()
    
    logger.info("Validation FPR monitoring completed")


def setup_production_fpr_monitoring():
    """
    Setup FPR monitoring for production deployment
    
    This function demonstrates how to set up FPR monitoring
    for a production RUL prediction system.
    """
    logger.info("Setting up production FPR monitoring")
    
    # Configure production logging
    configure_prediction_logging(
        log_file="logs/production_fpr_monitoring.jsonl",
        log_level=logging.INFO
    )
    
    # Create production FPR monitor configuration
    config = AlertConfig(
        fpr_threshold=0.05,
        fpr_warning_threshold=0.03,
        min_predictions_for_alert=20,
        alert_cooldown_minutes=30,
        drift_detection_window_hours=24,
        drift_threshold=0.02,
        
        # Email alerts (configure with actual SMTP settings)
        enable_email_alerts=True,
        smtp_server="smtp.company.com",
        smtp_port=587,
        smtp_username="rul_system@company.com",
        smtp_password="password",  # Use environment variable in production
        alert_recipients=["ops_team@company.com", "ml_team@company.com"],
        
        # Webhook alerts (configure with actual webhook URL)
        enable_webhook_alerts=True,
        webhook_url="https://hooks.slack.com/services/YOUR/WEBHOOK/URL",
        webhook_headers={"Content-Type": "application/json"}
    )
    
    # Create FPR monitor
    fpr_monitor = FPRMonitor(
        config=config,
        db_path="production_fpr_monitoring.db",
        monitoring_window_minutes=60  # 1-hour windows for production
    )
    
    # Add custom alert callback for additional integrations
    def custom_alert_handler(alert):
        """Custom alert handler for additional integrations"""
        logger.critical(f"PRODUCTION ALERT: {alert.message}")
        
        # Add custom integrations here:
        # - Send to monitoring systems (Prometheus, Grafana)
        # - Update incident management systems
        # - Trigger automated responses
        
        if alert.severity == "critical":
            # Example: Trigger automated model retraining
            logger.info("Critical alert detected - consider triggering model retraining")
    
    fpr_monitor.add_alert_callback(custom_alert_handler)
    
    # Start monitoring
    fpr_monitor.start_monitoring()
    
    # Create and start dashboard
    dashboard = create_fpr_dashboard(
        fpr_monitor=fpr_monitor,
        host="0.0.0.0",
        port=8080,
        enable_websocket=True
    )
    
    logger.info("Production FPR monitoring setup completed")
    logger.info("Dashboard available at http://localhost:8080")
    logger.info("Press Ctrl+C to stop")
    
    try:
        # Run dashboard (this will block)
        dashboard.run()
    except KeyboardInterrupt:
        logger.info("Production monitoring stopped by user")
    finally:
        fpr_monitor.stop_monitoring()


def create_monitoring_integration_example():
    """
    Create an example of how to integrate FPR monitoring
    with the existing API endpoints
    """
    logger.info("Creating monitoring integration example")
    
    # This example shows how to modify the existing API
    # to include FPR monitoring
    
    example_code = '''
# Example: Modified API endpoint with FPR monitoring

from fastapi import FastAPI
from true_rul.fpr_monitor import create_fpr_monitor
from true_rul.rul_predictor import RULPredictor

app = FastAPI()

# Initialize FPR monitoring
fpr_monitor = create_fpr_monitor(
    fpr_threshold=0.05,
    warning_threshold=0.03,
    monitoring_window_minutes=30
)
fpr_monitor.start_monitoring()

# Initialize RUL predictor
rul_predictor = RULPredictor()

@app.post("/predict")
async def predict_rul(request: PredictionRequest):
    """RUL prediction endpoint with FPR monitoring"""
    
    # Make prediction
    prediction_result = rul_predictor.predict(request.cycle_data)
    
    # Record prediction for FPR monitoring
    # Note: ground_truth_anomaly would typically be None in production
    # unless you have a feedback mechanism to collect true labels
    fpr_monitor.record_prediction(
        prediction_result=prediction_result,
        true_anomaly_label=None,  # Unknown in production
        capacitor_id=request.capacitor_id
    )
    
    return prediction_result

@app.get("/monitoring/fpr-status")
async def get_fpr_status():
    """Get current FPR monitoring status"""
    current_metrics = fpr_monitor.calculate_current_fpr()
    active_alerts = fpr_monitor.db.get_active_alerts()
    
    return {
        "current_fpr": current_metrics.fpr if current_metrics else None,
        "fpr_threshold": fpr_monitor.config.fpr_threshold,
        "active_alerts": len(active_alerts),
        "monitoring_active": fpr_monitor.is_monitoring
    }

@app.get("/monitoring/dashboard-data")
async def get_dashboard_data():
    """Get dashboard data for monitoring UI"""
    return fpr_monitor.get_dashboard_data(hours=24)
    '''
    
    # Save example to file
    example_path = Path("output/fpr_monitoring_integration_example.py")
    example_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(example_path, 'w') as f:
        f.write(example_code)
    
    logger.info(f"Integration example saved to {example_path}")


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Integrate FPR Monitoring with RUL System")
    parser.add_argument(
        "--mode",
        choices=["validation", "production", "example"],
        default="validation",
        help="Integration mode"
    )
    
    args = parser.parse_args()
    
    # Create output directories
    Path("logs").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)
    
    if args.mode == "validation":
        setup_fpr_monitoring_for_validation()
    elif args.mode == "production":
        setup_production_fpr_monitoring()
    elif args.mode == "example":
        create_monitoring_integration_example()
    
    logger.info("FPR monitoring integration completed!")


if __name__ == "__main__":
    main()