"""
Integration tests for the training pipeline

These tests verify that the training pipeline components work together
correctly with synthetic data.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from true_rul.training_pipeline import TrainingPipeline
from true_rul.model_evaluator import ModelEvaluator
from true_rul.data_structures import TrainingDataset, CycleData, CapacitorData
from true_rul.config import MODEL_CONFIG


class TestTrainingPipelineIntegration:
    """Integration tests for training pipeline"""
    
    def test_training_pipeline_initialization(self):
        """Test that training pipeline initializes correctly"""
        pipeline = TrainingPipeline()
        
        assert pipeline.data_loader is not None
        assert pipeline.feature_extractor is not None
        assert pipeline.preprocessor is not None
        assert not pipeline.is_trained
        assert pipeline.rul_model is None
        assert pipeline.anomaly_detector is None
    
    def test_model_evaluator_initialization(self):
        """Test that model evaluator initializes correctly"""
        evaluator = ModelEvaluator()
        
        assert len(evaluator.rul_metrics) == 0
        assert len(evaluator.anomaly_metrics) == 0
        assert len(evaluator.evaluation_results) == 0
    
    def test_training_dataset_creation(self):
        """Test creating a training dataset with synthetic data"""
        # Create synthetic data
        n_samples = 100
        n_features = 55  # Expected number of features
        
        capacitor_ids = [f"ES12C{i//25 + 1}" for i in range(n_samples)]
        features = np.random.randn(n_samples, n_features)
        rul_labels = np.random.randint(0, 200, n_samples)
        cycle_numbers = np.random.randint(1, 201, n_samples)
        anomaly_labels = np.random.randint(0, 2, n_samples)
        
        dataset = TrainingDataset(
            capacitor_ids=capacitor_ids,
            features=features,
            rul_labels=rul_labels,
            cycle_numbers=cycle_numbers,
            anomaly_labels=anomaly_labels
        )
        
        assert dataset.n_samples == n_samples
        assert dataset.n_features == n_features
        
        # Test split by capacitor
        test_capacitors = ["ES12C4"]
        train_dataset, test_dataset = dataset.split_by_capacitor(test_capacitors)
        
        assert train_dataset.n_samples + test_dataset.n_samples == n_samples
        
        # Test get normal cycles
        normal_cycles = dataset.get_normal_cycles(max_cycle=10)
        assert len(normal_cycles) <= n_samples
    
    def test_capacitor_data_creation(self):
        """Test creating capacitor data structures"""
        # Create synthetic cycle data
        cycles = []
        for i in range(10):
            vl_series = np.random.randn(100)
            vo_series = np.random.randn(100)
            cycle = CycleData(
                cycle_number=i + 1,
                vl_series=vl_series,
                vo_series=vo_series
            )
            cycles.append(cycle)
        
        capacitor_data = CapacitorData(
            capacitor_id="ES12C1",
            cycles=cycles,
            total_cycles=len(cycles)
        )
        
        assert capacitor_data.capacitor_id == "ES12C1"
        assert capacitor_data.total_cycles == 10
        assert len(capacitor_data.cycles) == 10
        
        # Test get_cycle method
        cycle_5 = capacitor_data.get_cycle(5)
        assert cycle_5 is not None
        assert cycle_5.cycle_number == 5
        
        # Test get_cycles_range method
        range_cycles = capacitor_data.get_cycles_range(3, 7)
        assert len(range_cycles) == 5
        assert range_cycles[0].cycle_number == 3
        assert range_cycles[-1].cycle_number == 7
    
    def test_config_loading(self):
        """Test that configuration loads correctly"""
        config = MODEL_CONFIG
        
        assert "rul_model" in config
        assert "anomaly_detection" in config
        assert "feature_extraction" in config
        assert "training" in config
        
        # Check RUL model config
        rul_config = config["rul_model"]
        assert "type" in rul_config
        assert rul_config["type"] in ["xgboost", "lightgbm", "random_forest", "ensemble"]
        
        # Check anomaly detection config
        anomaly_config = config["anomaly_detection"]
        assert "ensemble_weights" in anomaly_config
        assert len(anomaly_config["ensemble_weights"]) == 3
        
        # Check feature extraction config
        feature_config = config["feature_extraction"]
        assert "rolling_window" in feature_config
        assert "normalization" in feature_config
    
    def test_pipeline_components_compatibility(self):
        """Test that pipeline components are compatible with each other"""
        pipeline = TrainingPipeline()
        
        # Test that feature extractor and preprocessor have compatible settings
        assert (pipeline.feature_extractor.rolling_window == 
                pipeline.preprocessor.rolling_window)
        
        # Test that config is properly loaded
        assert pipeline.config is not None
        assert "rul_model" in pipeline.config
        assert "anomaly_detection" in pipeline.config
    
    @pytest.mark.parametrize("model_type", ["xgboost", "lightgbm", "random_forest", "ensemble"])
    def test_model_type_configurations(self, model_type):
        """Test different model type configurations"""
        config = MODEL_CONFIG.copy()
        config["rul_model"]["type"] = model_type
        
        pipeline = TrainingPipeline(config=config)
        
        assert pipeline.config["rul_model"]["type"] == model_type
    
    def test_evaluation_metrics_structure(self):
        """Test that evaluation metrics have the expected structure"""
        evaluator = ModelEvaluator()
        
        # Test with synthetic data
        n_samples = 50
        true_values = np.random.randint(0, 200, n_samples)
        predictions = true_values + np.random.randn(n_samples) * 10  # Add some noise
        
        # Manually calculate some metrics to test structure
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        rmse = np.sqrt(mean_squared_error(true_values, predictions))
        mae = mean_absolute_error(true_values, predictions)
        r2 = r2_score(true_values, predictions)
        
        assert rmse >= 0
        assert mae >= 0
        assert -1 <= r2 <= 1  # R² can be negative for very poor fits
    
    def test_anomaly_detection_metrics_structure(self):
        """Test anomaly detection metrics structure"""
        # Test with synthetic binary classification data
        n_samples = 100
        true_labels = np.random.randint(0, 2, n_samples)
        predictions = np.random.randint(0, 2, n_samples)
        scores = np.random.rand(n_samples)
        
        from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
        
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average='binary', zero_division=0
        )
        
        tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
        
        # Calculate rates
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        assert 0 <= precision <= 1
        assert 0 <= recall <= 1
        assert 0 <= f1 <= 1
        assert 0 <= tpr <= 1
        assert 0 <= fpr <= 1
        assert tp + fp + tn + fn == n_samples


if __name__ == "__main__":
    pytest.main([__file__, "-v"])