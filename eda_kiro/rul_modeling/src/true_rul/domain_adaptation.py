"""
Domain Adaptation Framework for True RUL Prediction System

This module implements domain adaptation capabilities including:
- Transfer learning for new capacitor types
- Domain-specific feature engineering pipelines
- Automated model adaptation for different operating conditions
- Few-shot learning for rapid deployment to new domains

Requirements: 1.1, 3.1, 8.4
"""

import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
from datetime import datetime
from pathlib import Path
import pickle
import json
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import defaultdict

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

from .data_structures import CycleData, CapacitorData, TrainingDataset
from .rul_regression_model import RULRegressionModel
from .ensemble_anomaly_detector import EnsembleAnomalyDetector
from .feature_extractor import FeatureExtractor
from .time_series_preprocessor import TimeSeriesPreprocessor

logger = logging.getLogger(__name__)


@dataclass
class DomainInfo:
    """Information about a specific domain"""
    domain_id: str
    domain_type: str  # 'capacitor_type', 'operating_condition', 'manufacturer', etc.
    characteristics: Dict[str, Any]
    sample_count: int
    feature_statistics: Optional[Dict[str, float]] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AdaptationResult:
    """Result of domain adaptation"""
    success: bool
    adapted_model: Optional[RULRegressionModel]
    adaptation_method: str
    source_domain: str
    target_domain: str
    performance_metrics: Dict[str, float]
    adaptation_time: float
    details: Dict[str, Any] = field(default_factory=dict)


class DomainFeatureExtractor(ABC):
    """Abstract base class for domain-specific feature extractors"""
    
    @abstractmethod
    def extract_domain_features(self, cycle_data: CycleData, domain_info: DomainInfo) -> np.ndarray:
        """Extract domain-specific features"""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get names of domain-specific features"""
        pass


class CapacitorTypeFeatureExtractor(DomainFeatureExtractor):
    """Feature extractor for different capacitor types"""
    
    def __init__(self):
        self.base_extractor = FeatureExtractor()
        self.capacitor_specific_features = {
            'electrolytic': self._extract_electrolytic_features,
            'ceramic': self._extract_ceramic_features,
            'tantalum': self._extract_tantalum_features,
            'film': self._extract_film_features
        }
    
    def extract_domain_features(self, cycle_data: CycleData, domain_info: DomainInfo) -> np.ndarray:
        """Extract capacitor type-specific features"""
        # Extract base features
        base_features = self.base_extractor.extract_features(cycle_data, [])
        
        # Extract capacitor-specific features
        capacitor_type = domain_info.characteristics.get('capacitor_type', 'unknown')
        
        if capacitor_type in self.capacitor_specific_features:
            specific_features = self.capacitor_specific_features[capacitor_type](cycle_data, domain_info)
            combined_features = np.concatenate([base_features, specific_features])
        else:
            logger.warning(f"Unknown capacitor type: {capacitor_type}. Using base features only.")
            combined_features = base_features
        
        return combined_features
    
    def _extract_electrolytic_features(self, cycle_data: CycleData, domain_info: DomainInfo) -> np.ndarray:
        """Extract features specific to electrolytic capacitors"""
        # Electrolytic capacitors are sensitive to temperature and ripple current
        vl, vo = cycle_data.vl_series, cycle_data.vo_series
        
        features = []
        
        # ESR-related features (important for electrolytics)
        esr_proxy = np.mean(np.abs(vl - vo))
        features.append(esr_proxy)
        
        # Leakage current proxy
        leakage_proxy = np.min(vo) / np.max(vl) if np.max(vl) > 0 else 0
        features.append(leakage_proxy)
        
        # Temperature sensitivity proxy (voltage stability)
        temp_sensitivity = np.std(vo) / np.mean(vo) if np.mean(vo) > 0 else 0
        features.append(temp_sensitivity)
        
        # Ripple handling (high-frequency response)
        ripple_response = np.std(np.diff(vo))
        features.append(ripple_response)
        
        return np.array(features)
    
    def _extract_ceramic_features(self, cycle_data: CycleData, domain_info: DomainInfo) -> np.ndarray:
        """Extract features specific to ceramic capacitors"""
        # Ceramic capacitors have voltage and temperature coefficients
        vl, vo = cycle_data.vl_series, cycle_data.vo_series
        
        features = []
        
        # Voltage coefficient effect
        voltage_coeff = np.polyfit(vl, vo, 1)[0] if len(vl) > 1 else 0
        features.append(voltage_coeff)
        
        # Dielectric absorption (slower response)
        absorption_proxy = np.mean(np.abs(np.diff(vo, n=2))) if len(vo) > 2 else 0
        features.append(absorption_proxy)
        
        # Piezoelectric effect proxy
        piezo_proxy = np.corrcoef(vl, vo)[0, 1] if len(vl) > 1 else 0
        features.append(piezo_proxy)
        
        # High-frequency stability
        hf_stability = 1.0 / (1.0 + np.std(vo[-10:]) / np.mean(vo[-10:])) if len(vo) >= 10 else 1.0
        features.append(hf_stability)
        
        return np.array(features)
    
    def _extract_tantalum_features(self, cycle_data: CycleData, domain_info: DomainInfo) -> np.ndarray:
        """Extract features specific to tantalum capacitors"""
        # Tantalum capacitors have unique failure modes
        vl, vo = cycle_data.vl_series, cycle_data.vo_series
        
        features = []
        
        # Surge current sensitivity
        surge_sensitivity = np.max(np.abs(np.diff(vl))) / np.mean(vl) if np.mean(vl) > 0 else 0
        features.append(surge_sensitivity)
        
        # Reverse voltage sensitivity
        reverse_voltage = np.sum(vo < 0) / len(vo)
        features.append(reverse_voltage)
        
        # Thermal runaway indicator
        thermal_indicator = np.std(vo) * np.mean(np.abs(vl))
        features.append(thermal_indicator)
        
        return np.array(features)
    
    def _extract_film_features(self, cycle_data: CycleData, domain_info: DomainInfo) -> np.ndarray:
        """Extract features specific to film capacitors"""
        # Film capacitors have excellent stability
        vl, vo = cycle_data.vl_series, cycle_data.vo_series
        
        features = []
        
        # Self-healing capability proxy
        self_healing = 1.0 - np.std(vo) / np.mean(vo) if np.mean(vo) > 0 else 0
        features.append(self_healing)
        
        # Moisture sensitivity
        moisture_sensitivity = np.corrcoef(np.arange(len(vo)), vo)[0, 1] if len(vo) > 1 else 0
        features.append(moisture_sensitivity)
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features"""
        base_names = [f"base_{i}" for i in range(55)]  # Assuming 55 base features
        
        specific_names = {
            'electrolytic': ['esr_proxy', 'leakage_proxy', 'temp_sensitivity', 'ripple_response'],
            'ceramic': ['voltage_coeff', 'absorption_proxy', 'piezo_proxy', 'hf_stability'],
            'tantalum': ['surge_sensitivity', 'reverse_voltage', 'thermal_indicator'],
            'film': ['self_healing', 'moisture_sensitivity']
        }
        
        # Return base names + all possible specific names
        all_specific = []
        for names in specific_names.values():
            all_specific.extend(names)
        
        return base_names + all_specific


class OperatingConditionFeatureExtractor(DomainFeatureExtractor):
    """Feature extractor for different operating conditions"""
    
    def __init__(self):
        self.base_extractor = FeatureExtractor()
    
    def extract_domain_features(self, cycle_data: CycleData, domain_info: DomainInfo) -> np.ndarray:
        """Extract operating condition-specific features"""
        base_features = self.base_extractor.extract_features(cycle_data, [])
        
        # Extract condition-specific features
        temperature = domain_info.characteristics.get('temperature', 25.0)  # Celsius
        humidity = domain_info.characteristics.get('humidity', 50.0)  # %RH
        voltage_stress = domain_info.characteristics.get('voltage_stress', 1.0)  # Ratio to rated
        
        condition_features = self._extract_condition_features(
            cycle_data, temperature, humidity, voltage_stress
        )
        
        return np.concatenate([base_features, condition_features])
    
    def _extract_condition_features(self, cycle_data: CycleData, 
                                  temperature: float, humidity: float, 
                                  voltage_stress: float) -> np.ndarray:
        """Extract features based on operating conditions"""
        vl, vo = cycle_data.vl_series, cycle_data.vo_series
        
        features = []
        
        # Temperature-dependent features
        temp_factor = (temperature - 25.0) / 50.0  # Normalized temperature
        temp_effect = np.std(vo) * (1.0 + temp_factor * 0.1)
        features.append(temp_effect)
        
        # Humidity-dependent features
        humidity_factor = humidity / 100.0
        humidity_effect = np.mean(np.abs(vl - vo)) * (1.0 + humidity_factor * 0.05)
        features.append(humidity_effect)
        
        # Voltage stress features
        stress_effect = np.max(vo) * voltage_stress
        features.append(stress_effect)
        
        # Combined environmental stress
        combined_stress = temp_factor * humidity_factor * voltage_stress
        features.append(combined_stress)
        
        # Aging acceleration factor
        aging_factor = np.exp((temperature - 25.0) / 10.0) * voltage_stress**2
        features.append(aging_factor)
        
        return np.array(features)
    
    def get_feature_names(self) -> List[str]:
        """Get names of condition-specific features"""
        base_names = [f"base_{i}" for i in range(55)]
        condition_names = [
            'temp_effect', 'humidity_effect', 'stress_effect', 
            'combined_stress', 'aging_factor'
        ]
        return base_names + condition_names


class TransferLearningAdapter:
    """Implements transfer learning for domain adaptation"""
    
    def __init__(self, source_model: RULRegressionModel):
        """
        Initialize transfer learning adapter
        
        Args:
            source_model: Pre-trained model from source domain
        """
        self.source_model = source_model
        self.adaptation_history: List[AdaptationResult] = []
        
        logger.info("TransferLearningAdapter initialized")
    
    def adapt_to_domain(self, 
                       target_data: TrainingDataset,
                       target_domain: DomainInfo,
                       adaptation_method: str = "fine_tuning",
                       learning_rate: float = 0.001,
                       freeze_layers: Optional[List[str]] = None) -> AdaptationResult:
        """
        Adapt source model to target domain
        
        Args:
            target_data: Training data from target domain
            target_domain: Information about target domain
            adaptation_method: Method for adaptation ('fine_tuning', 'feature_adaptation', 'layer_adaptation')
            learning_rate: Learning rate for adaptation
            freeze_layers: Layers to freeze during adaptation
            
        Returns:
            AdaptationResult with adapted model and metrics
        """
        start_time = datetime.now()
        logger.info(f"Starting domain adaptation using {adaptation_method}")
        
        try:
            if adaptation_method == "fine_tuning":
                adapted_model = self._fine_tune_model(target_data, learning_rate)
            elif adaptation_method == "feature_adaptation":
                adapted_model = self._adapt_features(target_data, target_domain)
            elif adaptation_method == "layer_adaptation":
                adapted_model = self._adapt_layers(target_data, freeze_layers)
            else:
                raise ValueError(f"Unknown adaptation method: {adaptation_method}")
            
            # Evaluate adapted model
            performance_metrics = self._evaluate_adaptation(adapted_model, target_data)
            
            adaptation_time = (datetime.now() - start_time).total_seconds()
            
            result = AdaptationResult(
                success=True,
                adapted_model=adapted_model,
                adaptation_method=adaptation_method,
                source_domain="source",  # Could be made more specific
                target_domain=target_domain.domain_id,
                performance_metrics=performance_metrics,
                adaptation_time=adaptation_time,
                details={
                    'learning_rate': learning_rate,
                    'target_samples': target_data.n_samples,
                    'freeze_layers': freeze_layers
                }
            )
            
            self.adaptation_history.append(result)
            logger.info(f"Domain adaptation completed successfully in {adaptation_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Domain adaptation failed: {e}")
            return AdaptationResult(
                success=False,
                adapted_model=None,
                adaptation_method=adaptation_method,
                source_domain="source",
                target_domain=target_domain.domain_id,
                performance_metrics={},
                adaptation_time=(datetime.now() - start_time).total_seconds(),
                details={'error': str(e)}
            )
    
    def _fine_tune_model(self, target_data: TrainingDataset, learning_rate: float) -> RULRegressionModel:
        """Fine-tune the entire model on target data"""
        # Create a copy of the source model
        adapted_model = RULRegressionModel(model_type=self.source_model.model_type)
        
        # For tree-based models, we'll retrain with combined data approach
        if self.source_model.model_type in ['xgboost', 'lightgbm', 'random_forest']:
            # Use target data to retrain (simplified approach)
            # In practice, you might want to combine source and target data
            val_size = max(1, target_data.n_samples // 5)
            train_size = target_data.n_samples - val_size
            
            train_features = target_data.features[:train_size]
            train_labels = target_data.rul_labels[:train_size]
            val_features = target_data.features[train_size:]
            val_labels = target_data.rul_labels[train_size:]
            
            adapted_model.train(
                train_features, train_labels, 
                val_features, val_labels,
                feature_names=self.source_model.feature_names
            )
        
        return adapted_model
    
    def _adapt_features(self, target_data: TrainingDataset, target_domain: DomainInfo) -> RULRegressionModel:
        """Adapt model by transforming features for target domain"""
        # Create feature adapter
        feature_adapter = DomainFeatureAdapter()
        
        # Learn feature transformation
        adapted_features = feature_adapter.fit_transform(
            target_data.features, target_domain
        )
        
        # Create new dataset with adapted features
        adapted_dataset = TrainingDataset(
            capacitor_ids=target_data.capacitor_ids,
            features=adapted_features,
            rul_labels=target_data.rul_labels,
            cycle_numbers=target_data.cycle_numbers,
            anomaly_labels=target_data.anomaly_labels
        )
        
        # Train new model on adapted features
        adapted_model = RULRegressionModel(model_type=self.source_model.model_type)
        
        val_size = max(1, adapted_dataset.n_samples // 5)
        train_size = adapted_dataset.n_samples - val_size
        
        adapted_model.train(
            adapted_dataset.features[:train_size],
            adapted_dataset.rul_labels[:train_size],
            adapted_dataset.features[train_size:],
            adapted_dataset.rul_labels[train_size:]
        )
        
        return adapted_model
    
    def _adapt_layers(self, target_data: TrainingDataset, freeze_layers: Optional[List[str]]) -> RULRegressionModel:
        """Adapt specific layers while freezing others"""
        # For tree-based models, this is similar to fine-tuning
        # In neural networks, you would freeze specific layers
        return self._fine_tune_model(target_data, 0.0001)  # Lower learning rate
    
    def _evaluate_adaptation(self, model: RULRegressionModel, test_data: TrainingDataset) -> Dict[str, float]:
        """Evaluate adapted model performance"""
        try:
            predictions = model.predict(test_data.features)
            
            # Calculate metrics
            mse = np.mean((predictions - test_data.rul_labels) ** 2)
            mae = np.mean(np.abs(predictions - test_data.rul_labels))
            
            # R² score
            ss_res = np.sum((test_data.rul_labels - predictions) ** 2)
            ss_tot = np.sum((test_data.rul_labels - np.mean(test_data.rul_labels)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(np.sqrt(mse)),
                'r2': float(r2)
            }
        except Exception as e:
            logger.error(f"Error evaluating adaptation: {e}")
            return {'error': str(e)}


class DomainFeatureAdapter(BaseEstimator, TransformerMixin):
    """Adapts features for different domains"""
    
    def __init__(self):
        self.domain_scalers: Dict[str, StandardScaler] = {}
        self.domain_transformers: Dict[str, PCA] = {}
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, domain_info: DomainInfo) -> 'DomainFeatureAdapter':
        """Fit adapter to domain data"""
        domain_id = domain_info.domain_id
        
        # Fit scaler for this domain
        scaler = StandardScaler()
        self.domain_scalers[domain_id] = scaler.fit(X)
        
        # Fit PCA for dimensionality reduction/transformation
        pca = PCA(n_components=min(50, X.shape[1]))  # Reduce to 50 components max
        X_scaled = scaler.transform(X)
        self.domain_transformers[domain_id] = pca.fit(X_scaled)
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray, domain_info: DomainInfo) -> np.ndarray:
        """Transform features for domain"""
        if not self.is_fitted:
            raise ValueError("Adapter must be fitted before transform")
        
        domain_id = domain_info.domain_id
        
        if domain_id not in self.domain_scalers:
            # Use a default transformation
            logger.warning(f"No scaler for domain {domain_id}, using default")
            scaler = StandardScaler().fit(X)
            pca = PCA(n_components=min(50, X.shape[1])).fit(scaler.transform(X))
        else:
            scaler = self.domain_scalers[domain_id]
            pca = self.domain_transformers[domain_id]
        
        X_scaled = scaler.transform(X)
        X_transformed = pca.transform(X_scaled)
        
        return X_transformed
    
    def fit_transform(self, X: np.ndarray, domain_info: DomainInfo) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(X, domain_info).transform(X, domain_info)


class FewShotLearningAdapter:
    """Implements few-shot learning for rapid domain adaptation"""
    
    def __init__(self, base_model: RULRegressionModel):
        """
        Initialize few-shot learning adapter
        
        Args:
            base_model: Base model to adapt
        """
        self.base_model = base_model
        self.support_sets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.prototypes: Dict[str, np.ndarray] = {}
        
        logger.info("FewShotLearningAdapter initialized")
    
    def add_support_set(self, domain_id: str, features: np.ndarray, labels: np.ndarray):
        """Add support set for a domain"""
        if len(features) != len(labels):
            raise ValueError("Features and labels must have same length")
        
        self.support_sets[domain_id] = (features, labels)
        
        # Compute prototype (mean of support features)
        self.prototypes[domain_id] = np.mean(features, axis=0)
        
        logger.info(f"Added support set for domain {domain_id} with {len(features)} samples")
    
    def predict_few_shot(self, query_features: np.ndarray, 
                        target_domain: str,
                        k_shot: int = 5) -> np.ndarray:
        """
        Make predictions using few-shot learning
        
        Args:
            query_features: Features to predict
            target_domain: Target domain ID
            k_shot: Number of support examples to use
            
        Returns:
            Predictions for query features
        """
        if target_domain not in self.support_sets:
            raise ValueError(f"No support set for domain {target_domain}")
        
        support_features, support_labels = self.support_sets[target_domain]
        
        # Use k-nearest neighbors in support set
        predictions = []
        
        for query in query_features:
            # Find k nearest support examples
            distances = np.linalg.norm(support_features - query, axis=1)
            k_nearest_indices = np.argsort(distances)[:k_shot]
            
            # Weighted average based on distance
            weights = 1.0 / (distances[k_nearest_indices] + 1e-8)
            weights = weights / np.sum(weights)
            
            prediction = np.sum(weights * support_labels[k_nearest_indices])
            predictions.append(prediction)
        
        return np.array(predictions)
    
    def adapt_with_few_shots(self, target_domain: str, 
                           query_features: np.ndarray,
                           k_shot: int = 5) -> AdaptationResult:
        """Adapt model using few-shot learning"""
        start_time = datetime.now()
        
        try:
            # Make predictions using few-shot learning
            predictions = self.predict_few_shot(query_features, target_domain, k_shot)
            
            # Create a simple adapted model (wrapper around few-shot predictions)
            adapted_model = FewShotModel(self, target_domain, k_shot)
            
            adaptation_time = (datetime.now() - start_time).total_seconds()
            
            result = AdaptationResult(
                success=True,
                adapted_model=adapted_model,
                adaptation_method="few_shot_learning",
                source_domain="base",
                target_domain=target_domain,
                performance_metrics={'predictions_made': len(predictions)},
                adaptation_time=adaptation_time,
                details={'k_shot': k_shot, 'support_set_size': len(self.support_sets[target_domain][0])}
            )
            
            logger.info(f"Few-shot adaptation completed in {adaptation_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Few-shot adaptation failed: {e}")
            return AdaptationResult(
                success=False,
                adapted_model=None,
                adaptation_method="few_shot_learning",
                source_domain="base",
                target_domain=target_domain,
                performance_metrics={},
                adaptation_time=(datetime.now() - start_time).total_seconds(),
                details={'error': str(e)}
            )


class FewShotModel:
    """Wrapper model for few-shot learning predictions"""
    
    def __init__(self, adapter: FewShotLearningAdapter, domain: str, k_shot: int):
        self.adapter = adapter
        self.domain = domain
        self.k_shot = k_shot
        self.is_trained = True
        self.model_type = "few_shot"
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using few-shot learning"""
        return self.adapter.predict_few_shot(X, self.domain, self.k_shot)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_type': 'few_shot',
            'domain': self.domain,
            'k_shot': self.k_shot,
            'is_trained': True
        }


class DomainAdaptationFramework:
    """
    Main framework for domain adaptation
    
    Coordinates different adaptation methods and manages domain information.
    """
    
    def __init__(self, base_model: RULRegressionModel):
        """
        Initialize domain adaptation framework
        
        Args:
            base_model: Base model to adapt from
        """
        self.base_model = base_model
        self.domains: Dict[str, DomainInfo] = {}
        self.feature_extractors: Dict[str, DomainFeatureExtractor] = {
            'capacitor_type': CapacitorTypeFeatureExtractor(),
            'operating_condition': OperatingConditionFeatureExtractor()
        }
        
        # Adaptation methods
        self.transfer_adapter = TransferLearningAdapter(base_model)
        self.few_shot_adapter = FewShotLearningAdapter(base_model)
        
        # Adaptation history
        self.adaptation_history: List[AdaptationResult] = []
        
        logger.info("DomainAdaptationFramework initialized")
    
    def register_domain(self, domain_info: DomainInfo):
        """Register a new domain"""
        self.domains[domain_info.domain_id] = domain_info
        logger.info(f"Registered domain: {domain_info.domain_id} ({domain_info.domain_type})")
    
    def adapt_to_domain(self, 
                       target_domain_id: str,
                       target_data: TrainingDataset,
                       adaptation_method: str = "auto",
                       **kwargs) -> AdaptationResult:
        """
        Adapt model to target domain
        
        Args:
            target_domain_id: ID of target domain
            target_data: Training data from target domain
            adaptation_method: Adaptation method ('transfer', 'few_shot', 'auto')
            **kwargs: Additional arguments for adaptation methods
            
        Returns:
            AdaptationResult with adapted model and metrics
        """
        if target_domain_id not in self.domains:
            raise ValueError(f"Domain {target_domain_id} not registered")
        
        target_domain = self.domains[target_domain_id]
        
        # Choose adaptation method
        if adaptation_method == "auto":
            adaptation_method = self._choose_adaptation_method(target_data, target_domain)
        
        logger.info(f"Adapting to domain {target_domain_id} using {adaptation_method}")
        
        # Perform adaptation
        if adaptation_method == "transfer":
            result = self.transfer_adapter.adapt_to_domain(
                target_data, target_domain, **kwargs
            )
        elif adaptation_method == "few_shot":
            # For few-shot, we need to set up support set first
            support_size = min(10, target_data.n_samples // 2)
            support_features = target_data.features[:support_size]
            support_labels = target_data.rul_labels[:support_size]
            
            self.few_shot_adapter.add_support_set(
                target_domain_id, support_features, support_labels
            )
            
            query_features = target_data.features[support_size:]
            result = self.few_shot_adapter.adapt_with_few_shots(
                target_domain_id, query_features, **kwargs
            )
        else:
            raise ValueError(f"Unknown adaptation method: {adaptation_method}")
        
        # Store result
        self.adaptation_history.append(result)
        
        return result
    
    def _choose_adaptation_method(self, target_data: TrainingDataset, 
                                target_domain: DomainInfo) -> str:
        """Automatically choose best adaptation method"""
        # Simple heuristic: use few-shot for small datasets, transfer for larger ones
        if target_data.n_samples < 20:
            return "few_shot"
        else:
            return "transfer"
    
    def extract_domain_features(self, cycle_data: CycleData, 
                              domain_id: str) -> np.ndarray:
        """Extract features specific to a domain"""
        if domain_id not in self.domains:
            raise ValueError(f"Domain {domain_id} not registered")
        
        domain_info = self.domains[domain_id]
        domain_type = domain_info.domain_type
        
        if domain_type in self.feature_extractors:
            extractor = self.feature_extractors[domain_type]
            return extractor.extract_domain_features(cycle_data, domain_info)
        else:
            logger.warning(f"No feature extractor for domain type {domain_type}")
            # Fall back to base feature extraction
            base_extractor = FeatureExtractor()
            return base_extractor.extract_features(cycle_data, [])
    
    def get_domain_similarity(self, domain1_id: str, domain2_id: str) -> float:
        """Compute similarity between two domains"""
        if domain1_id not in self.domains or domain2_id not in self.domains:
            return 0.0
        
        domain1 = self.domains[domain1_id]
        domain2 = self.domains[domain2_id]
        
        # Simple similarity based on domain type and characteristics
        if domain1.domain_type != domain2.domain_type:
            return 0.0
        
        # Compare characteristics
        common_keys = set(domain1.characteristics.keys()) & set(domain2.characteristics.keys())
        if not common_keys:
            return 0.5  # Default similarity
        
        similarities = []
        for key in common_keys:
            val1 = domain1.characteristics[key]
            val2 = domain2.characteristics[key]
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                # Numerical similarity
                max_val = max(abs(val1), abs(val2), 1.0)
                sim = 1.0 - abs(val1 - val2) / max_val
                similarities.append(sim)
            elif val1 == val2:
                similarities.append(1.0)
            else:
                similarities.append(0.0)
        
        return np.mean(similarities) if similarities else 0.0
    
    def get_adaptation_summary(self) -> Dict[str, Any]:
        """Get summary of all adaptations performed"""
        if not self.adaptation_history:
            return {'total_adaptations': 0}
        
        successful = [r for r in self.adaptation_history if r.success]
        
        return {
            'total_adaptations': len(self.adaptation_history),
            'successful_adaptations': len(successful),
            'success_rate': len(successful) / len(self.adaptation_history),
            'methods_used': list(set(r.adaptation_method for r in self.adaptation_history)),
            'domains_adapted': list(set(r.target_domain for r in self.adaptation_history)),
            'average_adaptation_time': np.mean([r.adaptation_time for r in successful]) if successful else 0.0
        }
    
    def save_framework(self, filepath: Path):
        """Save framework state"""
        state = {
            'domains': {k: {
                'domain_id': v.domain_id,
                'domain_type': v.domain_type,
                'characteristics': v.characteristics,
                'sample_count': v.sample_count,
                'feature_statistics': v.feature_statistics,
                'created_at': v.created_at.isoformat()
            } for k, v in self.domains.items()},
            'adaptation_history': [{
                'success': r.success,
                'adaptation_method': r.adaptation_method,
                'source_domain': r.source_domain,
                'target_domain': r.target_domain,
                'performance_metrics': r.performance_metrics,
                'adaptation_time': r.adaptation_time,
                'details': r.details
            } for r in self.adaptation_history]
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"Framework state saved to {filepath}")
    
    def load_framework(self, filepath: Path):
        """Load framework state"""
        with open(filepath, 'r') as f:
            state = json.load(f)
        
        # Restore domains
        for domain_id, domain_data in state.get('domains', {}).items():
            domain_info = DomainInfo(
                domain_id=domain_data['domain_id'],
                domain_type=domain_data['domain_type'],
                characteristics=domain_data['characteristics'],
                sample_count=domain_data['sample_count'],
                feature_statistics=domain_data.get('feature_statistics'),
                created_at=datetime.fromisoformat(domain_data['created_at'])
            )
            self.domains[domain_id] = domain_info
        
        # Restore adaptation history (without models)
        for result_data in state.get('adaptation_history', []):
            result = AdaptationResult(
                success=result_data['success'],
                adapted_model=None,  # Models are not serialized
                adaptation_method=result_data['adaptation_method'],
                source_domain=result_data['source_domain'],
                target_domain=result_data['target_domain'],
                performance_metrics=result_data['performance_metrics'],
                adaptation_time=result_data['adaptation_time'],
                details=result_data['details']
            )
            self.adaptation_history.append(result)
        
        logger.info(f"Framework state loaded from {filepath}")


def create_domain_adaptation_framework(base_model: RULRegressionModel) -> DomainAdaptationFramework:
    """
    Factory function to create domain adaptation framework
    
    Args:
        base_model: Base model to adapt from
        
    Returns:
        Configured DomainAdaptationFramework
    """
    return DomainAdaptationFramework(base_model)