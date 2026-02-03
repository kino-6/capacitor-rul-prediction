"""
Model Interpretability Enhancements for RUL Prediction System

This module implements advanced interpretability techniques including:
- LIME (Local Interpretable Model-agnostic Explanations)
- Counterfactual explanations for anomaly predictions
- Attention mechanisms for feature importance
- Causal inference analysis for feature relationships

Requirements: 9.1, 9.2, 9.3
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class LIMEExplainer:
    """
    Local Interpretable Model-agnostic Explanations (LIME) for RUL prediction
    
    LIME explains individual predictions by learning an interpretable model
    locally around the prediction. It perturbs the input and observes the
    changes in predictions to understand feature importance.
    
    Attributes:
        model: The black-box model to explain
        feature_names: Names of input features
        n_samples: Number of samples to generate for local explanation
        kernel_width: Width of the exponential kernel for weighting samples
        random_state: Random seed for reproducibility
    """
    
    def __init__(self,
                 model: Any,
                 feature_names: Optional[List[str]] = None,
                 n_samples: int = 1000,
                 kernel_width: float = 0.25,
                 random_state: int = 42):
        """
        Initialize LIME explainer
        
        Args:
            model: Model to explain (must have predict method)
            feature_names: Names of features
            n_samples: Number of samples for local explanation
            kernel_width: Kernel width for sample weighting
            random_state: Random seed
        """
        self.model = model
        self.feature_names = feature_names
        self.n_samples = n_samples
        self.kernel_width = kernel_width
        self.random_state = random_state
        
        np.random.seed(random_state)
        
        logger.info(f"Initialized LIME explainer with {n_samples} samples")
    
    def explain_instance(self,
                        instance: np.ndarray,
                        training_data: Optional[np.ndarray] = None,
                        num_features: int = 10) -> Dict[str, Any]:
        """
        Explain a single prediction instance
        
        Args:
            instance: Single instance to explain (1D array)
            training_data: Training data for generating perturbations
            num_features: Number of top features to include in explanation
            
        Returns:
            Dictionary with explanation details
        """
        if len(instance.shape) != 1:
            raise ValueError("Instance must be 1D array")
        
        logger.info(f"Explaining instance with {len(instance)} features")
        
        # Generate perturbed samples around the instance
        perturbed_samples = self._generate_perturbed_samples(instance, training_data)
        
        # Get predictions for perturbed samples
        predictions = self.model.predict(perturbed_samples)
        
        # Calculate distances and weights
        distances = euclidean_distances([instance], perturbed_samples)[0]
        weights = np.exp(-(distances ** 2) / (self.kernel_width ** 2))
        
        # Fit local linear model
        local_model = LinearRegression()
        local_model.fit(perturbed_samples, predictions, sample_weight=weights)
        
        # Get feature importance from local model
        feature_importance = local_model.coef_
        
        # Get top features
        if self.feature_names:
            feature_names = self.feature_names
        else:
            feature_names = [f"feature_{i}" for i in range(len(instance))]
        
        # Sort features by absolute importance
        importance_pairs = list(zip(feature_names, feature_importance))
        importance_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        
        top_features = importance_pairs[:num_features]
        
        # Calculate local prediction and fidelity
        local_prediction = local_model.predict([instance])[0]
        original_prediction = self.model.predict([instance])[0]
        fidelity = 1 - abs(local_prediction - original_prediction) / (abs(original_prediction) + 1e-8)
        
        explanation = {
            "instance": instance,
            "original_prediction": original_prediction,
            "local_prediction": local_prediction,
            "fidelity": fidelity,
            "top_features": top_features,
            "all_feature_importance": dict(importance_pairs),
            "local_model_r2": local_model.score(perturbed_samples, predictions, sample_weight=weights),
            "n_samples_used": len(perturbed_samples)
        }
        
        logger.info(f"LIME explanation completed. Fidelity: {fidelity:.3f}, R²: {explanation['local_model_r2']:.3f}")
        
        return explanation
    
    def _generate_perturbed_samples(self,
                                   instance: np.ndarray,
                                   training_data: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate perturbed samples around the instance
        
        Args:
            instance: Original instance
            training_data: Training data for realistic perturbations
            
        Returns:
            Array of perturbed samples
        """
        n_features = len(instance)
        perturbed_samples = np.zeros((self.n_samples, n_features))
        
        if training_data is not None:
            # Use training data statistics for realistic perturbations
            feature_means = np.mean(training_data, axis=0)
            feature_stds = np.std(training_data, axis=0)
        else:
            # Use instance-based perturbations
            feature_means = instance
            feature_stds = np.abs(instance) * 0.1 + 0.01  # 10% of value + small constant
        
        for i in range(self.n_samples):
            # Generate perturbation
            perturbation = np.random.normal(0, feature_stds)
            
            # Apply perturbation with some probability for each feature
            mask = np.random.random(n_features) < 0.5  # 50% chance to perturb each feature
            
            perturbed_sample = instance.copy()
            perturbed_sample[mask] = feature_means[mask] + perturbation[mask]
            
            perturbed_samples[i] = perturbed_sample
        
        return perturbed_samples
    
    def explain_batch(self,
                     instances: np.ndarray,
                     training_data: Optional[np.ndarray] = None,
                     num_features: int = 10) -> List[Dict[str, Any]]:
        """
        Explain multiple instances
        
        Args:
            instances: Multiple instances to explain (2D array)
            training_data: Training data for perturbations
            num_features: Number of top features per explanation
            
        Returns:
            List of explanations
        """
        explanations = []
        
        for i, instance in enumerate(instances):
            logger.info(f"Explaining instance {i+1}/{len(instances)}")
            explanation = self.explain_instance(instance, training_data, num_features)
            explanations.append(explanation)
        
        return explanations


class CounterfactualExplainer:
    """
    Counterfactual explanations for anomaly predictions
    
    Generates counterfactual examples that show what changes would be needed
    to flip the prediction from anomaly to normal (or vice versa).
    
    Attributes:
        model: The anomaly detection model to explain
        feature_names: Names of input features
        max_iterations: Maximum optimization iterations
        learning_rate: Learning rate for optimization
        distance_weight: Weight for distance penalty in optimization
    """
    
    def __init__(self,
                 model: Any,
                 feature_names: Optional[List[str]] = None,
                 max_iterations: int = 1000,
                 learning_rate: float = 0.01,
                 distance_weight: float = 1.0,
                 random_state: int = 42):
        """
        Initialize counterfactual explainer
        
        Args:
            model: Anomaly detection model
            feature_names: Names of features
            max_iterations: Maximum optimization iterations
            learning_rate: Learning rate for gradient descent
            distance_weight: Weight for distance penalty
            random_state: Random seed
        """
        self.model = model
        self.feature_names = feature_names
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        self.distance_weight = distance_weight
        self.random_state = random_state
        
        np.random.seed(random_state)
        
        logger.info("Initialized counterfactual explainer")
    
    def generate_counterfactual(self,
                               instance: np.ndarray,
                               target_class: int,
                               feature_ranges: Optional[Dict[int, Tuple[float, float]]] = None) -> Dict[str, Any]:
        """
        Generate counterfactual explanation for an instance
        
        Args:
            instance: Original instance
            target_class: Desired target class (0=normal, 1=anomaly)
            feature_ranges: Valid ranges for each feature
            
        Returns:
            Dictionary with counterfactual explanation
        """
        logger.info(f"Generating counterfactual for target class {target_class}")
        
        # Get original prediction
        if hasattr(self.model, 'predict_proba'):
            original_proba = self.model.predict_proba([instance])[0]
            original_class = np.argmax(original_proba)
        else:
            original_class = self.model.predict([instance])[0]
            original_proba = [1-original_class, original_class]  # Simple binary conversion
        
        if original_class == target_class:
            logger.warning("Instance already belongs to target class")
            return {
                "original_instance": instance,
                "counterfactual": instance,
                "original_class": original_class,
                "target_class": target_class,
                "success": False,
                "distance": 0.0,
                "changes": {}
            }
        
        # Initialize counterfactual as copy of original
        counterfactual = instance.copy().astype(float)
        
        # Optimization loop
        best_counterfactual = counterfactual.copy()
        best_distance = float('inf')
        
        for iteration in range(self.max_iterations):
            # Get current prediction
            if hasattr(self.model, 'predict_proba'):
                current_proba = self.model.predict_proba([counterfactual])[0]
                current_class = np.argmax(current_proba)
                target_proba = current_proba[target_class]
            else:
                current_class = self.model.predict([counterfactual])[0]
                target_proba = float(current_class == target_class)
            
            # Check if we've reached the target
            if current_class == target_class:
                distance = np.linalg.norm(counterfactual - instance)
                if distance < best_distance:
                    best_distance = distance
                    best_counterfactual = counterfactual.copy()
                break
            
            # Compute gradient approximation using finite differences
            gradient = self._compute_gradient(counterfactual, target_class)
            
            # Update counterfactual
            counterfactual += self.learning_rate * gradient
            
            # Apply feature range constraints if provided
            if feature_ranges:
                for feature_idx, (min_val, max_val) in feature_ranges.items():
                    counterfactual[feature_idx] = np.clip(counterfactual[feature_idx], min_val, max_val)
            
            # Early stopping if we're making progress
            if iteration % 100 == 0:
                distance = np.linalg.norm(counterfactual - instance)
                if distance < best_distance:
                    best_distance = distance
                    best_counterfactual = counterfactual.copy()
        
        # Final check
        if hasattr(self.model, 'predict_proba'):
            final_proba = self.model.predict_proba([best_counterfactual])[0]
            final_class = np.argmax(final_proba)
        else:
            final_class = self.model.predict([best_counterfactual])[0]
        
        success = (final_class == target_class)
        
        # Calculate changes
        changes = {}
        if self.feature_names:
            for i, (original_val, cf_val) in enumerate(zip(instance, best_counterfactual)):
                if abs(cf_val - original_val) > 1e-6:
                    changes[self.feature_names[i]] = {
                        "original": original_val,
                        "counterfactual": cf_val,
                        "change": cf_val - original_val
                    }
        
        result = {
            "original_instance": instance,
            "counterfactual": best_counterfactual,
            "original_class": original_class,
            "target_class": target_class,
            "final_class": final_class,
            "success": success,
            "distance": best_distance,
            "changes": changes,
            "iterations": iteration + 1
        }
        
        logger.info(f"Counterfactual generation {'succeeded' if success else 'failed'}. Distance: {best_distance:.3f}")
        
        return result
    
    def _compute_gradient(self, instance: np.ndarray, target_class: int) -> np.ndarray:
        """
        Compute gradient approximation using finite differences
        
        Args:
            instance: Current instance
            target_class: Target class
            
        Returns:
            Gradient approximation
        """
        gradient = np.zeros_like(instance)
        epsilon = 1e-5
        
        # Get baseline prediction
        if hasattr(self.model, 'predict_proba'):
            baseline_proba = self.model.predict_proba([instance])[0][target_class]
        else:
            baseline_pred = self.model.predict([instance])[0]
            baseline_proba = float(baseline_pred == target_class)
        
        # Compute partial derivatives
        for i in range(len(instance)):
            # Perturb feature i
            perturbed = instance.copy()
            perturbed[i] += epsilon
            
            # Get perturbed prediction
            if hasattr(self.model, 'predict_proba'):
                perturbed_proba = self.model.predict_proba([perturbed])[0][target_class]
            else:
                perturbed_pred = self.model.predict([perturbed])[0]
                perturbed_proba = float(perturbed_pred == target_class)
            
            # Compute gradient
            gradient[i] = (perturbed_proba - baseline_proba) / epsilon
        
        # Add distance penalty gradient
        distance_gradient = -self.distance_weight * (instance - instance)  # Zero for now
        
        return gradient + distance_gradient


class AttentionMechanism:
    """
    Attention mechanism for feature importance in time-series or sequential data
    
    This class implements attention weights that highlight which features
    or time steps are most important for the prediction.
    
    Attributes:
        attention_type: Type of attention mechanism ("additive" or "multiplicative")
        hidden_dim: Hidden dimension for attention computation
        temperature: Temperature parameter for softmax attention
    """
    
    def __init__(self,
                 attention_type: str = "additive",
                 hidden_dim: int = 64,
                 temperature: float = 1.0):
        """
        Initialize attention mechanism
        
        Args:
            attention_type: Type of attention ("additive" or "multiplicative")
            hidden_dim: Hidden dimension for attention computation
            temperature: Temperature for softmax
        """
        self.attention_type = attention_type
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        
        # Initialize attention parameters (simplified linear attention)
        self.W_attention = None
        self.v_attention = None
        
        logger.info(f"Initialized {attention_type} attention mechanism")
    
    def compute_attention_weights(self,
                                 features: np.ndarray,
                                 query: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute attention weights for features
        
        Args:
            features: Input features (n_samples, n_features) or (n_timesteps, n_features)
            query: Query vector for attention (optional)
            
        Returns:
            Attention weights
        """
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        n_samples, n_features = features.shape
        
        if self.attention_type == "additive":
            # Additive attention: score = v^T * tanh(W * features + U * query)
            if query is None:
                query = np.mean(features, axis=0)  # Use mean as default query
            
            # Simplified implementation without learned parameters
            # In practice, these would be learned during training
            scores = np.sum(features * query, axis=1)  # Dot product attention
            
        elif self.attention_type == "multiplicative":
            # Multiplicative attention: score = features * query
            if query is None:
                query = np.mean(features, axis=0)
            
            scores = np.dot(features, query)
            
        else:
            raise ValueError(f"Unknown attention type: {self.attention_type}")
        
        # Apply temperature and softmax
        scores = scores / self.temperature
        attention_weights = self._softmax(scores)
        
        return attention_weights
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Compute softmax function"""
        exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
        return exp_x / np.sum(exp_x)
    
    def visualize_attention(self,
                           features: np.ndarray,
                           attention_weights: np.ndarray,
                           feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create visualization data for attention weights
        
        Args:
            features: Input features
            attention_weights: Computed attention weights
            feature_names: Names of features
            
        Returns:
            Dictionary with visualization data
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(features.shape[-1])]
        
        # Create attention visualization data
        attention_data = []
        
        if len(features.shape) == 1:
            # Single sample
            for i, (feature_val, weight) in enumerate(zip(features, attention_weights)):
                attention_data.append({
                    "feature_name": feature_names[i] if i < len(feature_names) else f"feature_{i}",
                    "feature_value": feature_val,
                    "attention_weight": weight,
                    "weighted_value": feature_val * weight
                })
        else:
            # Multiple samples/timesteps
            for t, (feature_vec, weight) in enumerate(zip(features, attention_weights)):
                attention_data.append({
                    "timestep": t,
                    "features": feature_vec,
                    "attention_weight": weight,
                    "weighted_features": feature_vec * weight
                })
        
        return {
            "attention_data": attention_data,
            "max_attention": np.max(attention_weights),
            "min_attention": np.min(attention_weights),
            "attention_entropy": -np.sum(attention_weights * np.log(attention_weights + 1e-8))
        }


class CausalInferenceAnalyzer:
    """
    Causal inference analysis for feature relationships
    
    This class analyzes causal relationships between features and their
    impact on predictions using techniques like do-calculus and
    intervention analysis.
    
    Attributes:
        model: The prediction model to analyze
        feature_names: Names of input features
        causal_graph: Adjacency matrix representing causal relationships
    """
    
    def __init__(self,
                 model: Any,
                 feature_names: Optional[List[str]] = None):
        """
        Initialize causal inference analyzer
        
        Args:
            model: Prediction model to analyze
            feature_names: Names of features
        """
        self.model = model
        self.feature_names = feature_names
        self.causal_graph = None
        
        logger.info("Initialized causal inference analyzer")
    
    def discover_causal_structure(self,
                                 data: np.ndarray,
                                 method: str = "correlation") -> np.ndarray:
        """
        Discover causal structure between features
        
        Args:
            data: Training data for structure discovery
            method: Method for structure discovery ("correlation", "mutual_info")
            
        Returns:
            Adjacency matrix representing causal relationships
        """
        logger.info(f"Discovering causal structure using {method}")
        
        n_features = data.shape[1]
        causal_graph = np.zeros((n_features, n_features))
        
        if method == "correlation":
            # Use correlation as proxy for causal relationships
            correlation_matrix = np.corrcoef(data.T)
            
            # Threshold correlations to create causal graph
            threshold = 0.3  # Adjust based on domain knowledge
            causal_graph = (np.abs(correlation_matrix) > threshold).astype(float)
            
            # Remove self-loops
            np.fill_diagonal(causal_graph, 0)
            
        elif method == "mutual_info":
            # Use mutual information (simplified implementation)
            from sklearn.feature_selection import mutual_info_regression
            
            for i in range(n_features):
                for j in range(n_features):
                    if i != j:
                        mi = mutual_info_regression(data[:, [i]], data[:, j])
                        causal_graph[i, j] = mi[0]
            
            # Threshold mutual information
            threshold = np.percentile(causal_graph[causal_graph > 0], 75)
            causal_graph = (causal_graph > threshold).astype(float)
        
        self.causal_graph = causal_graph
        
        logger.info(f"Discovered causal graph with {np.sum(causal_graph)} edges")
        
        return causal_graph
    
    def compute_causal_effects(self,
                              instance: np.ndarray,
                              intervention_feature: int,
                              intervention_values: np.ndarray) -> Dict[str, Any]:
        """
        Compute causal effects of intervening on a feature
        
        Args:
            instance: Original instance
            intervention_feature: Index of feature to intervene on
            intervention_values: Values to set the feature to
            
        Returns:
            Dictionary with causal effect analysis
        """
        logger.info(f"Computing causal effects for feature {intervention_feature}")
        
        # Get baseline prediction
        baseline_prediction = self.model.predict([instance])[0]
        
        # Compute effects of interventions
        intervention_effects = []
        
        for intervention_value in intervention_values:
            # Create intervened instance
            intervened_instance = instance.copy()
            intervened_instance[intervention_feature] = intervention_value
            
            # Get prediction after intervention
            intervened_prediction = self.model.predict([intervened_instance])[0]
            
            # Compute causal effect
            causal_effect = intervened_prediction - baseline_prediction
            
            intervention_effects.append({
                "intervention_value": intervention_value,
                "prediction": intervened_prediction,
                "causal_effect": causal_effect
            })
        
        # Compute average treatment effect
        effects = [effect["causal_effect"] for effect in intervention_effects]
        average_treatment_effect = np.mean(effects)
        
        result = {
            "intervention_feature": intervention_feature,
            "feature_name": self.feature_names[intervention_feature] if self.feature_names else f"feature_{intervention_feature}",
            "baseline_prediction": baseline_prediction,
            "intervention_effects": intervention_effects,
            "average_treatment_effect": average_treatment_effect,
            "effect_variance": np.var(effects)
        }
        
        logger.info(f"Average treatment effect: {average_treatment_effect:.3f}")
        
        return result
    
    def analyze_feature_interactions(self,
                                   instance: np.ndarray,
                                   feature_pairs: Optional[List[Tuple[int, int]]] = None) -> Dict[str, Any]:
        """
        Analyze interactions between feature pairs
        
        Args:
            instance: Instance to analyze
            feature_pairs: Pairs of features to analyze (if None, analyze all pairs)
            
        Returns:
            Dictionary with interaction analysis
        """
        logger.info("Analyzing feature interactions")
        
        n_features = len(instance)
        
        if feature_pairs is None:
            # Analyze top feature pairs based on causal graph
            if self.causal_graph is not None:
                # Find pairs with strong causal relationships
                strong_edges = np.where(self.causal_graph > 0.5)
                feature_pairs = list(zip(strong_edges[0], strong_edges[1]))
            else:
                # Analyze a few random pairs
                feature_pairs = [(i, j) for i in range(min(5, n_features)) 
                               for j in range(i+1, min(5, n_features))]
        
        interaction_results = []
        
        for feature_i, feature_j in feature_pairs:
            # Compute individual effects
            instance_i_high = instance.copy()
            instance_i_high[feature_i] = instance[feature_i] * 1.2  # 20% increase
            
            instance_j_high = instance.copy()
            instance_j_high[feature_j] = instance[feature_j] * 1.2
            
            # Compute joint effect
            instance_both_high = instance.copy()
            instance_both_high[feature_i] = instance[feature_i] * 1.2
            instance_both_high[feature_j] = instance[feature_j] * 1.2
            
            # Get predictions
            baseline_pred = self.model.predict([instance])[0]
            pred_i = self.model.predict([instance_i_high])[0]
            pred_j = self.model.predict([instance_j_high])[0]
            pred_both = self.model.predict([instance_both_high])[0]
            
            # Compute interaction effect
            individual_effects = (pred_i - baseline_pred) + (pred_j - baseline_pred)
            joint_effect = pred_both - baseline_pred
            interaction_effect = joint_effect - individual_effects
            
            interaction_results.append({
                "feature_i": feature_i,
                "feature_j": feature_j,
                "feature_i_name": self.feature_names[feature_i] if self.feature_names else f"feature_{feature_i}",
                "feature_j_name": self.feature_names[feature_j] if self.feature_names else f"feature_{feature_j}",
                "individual_effect_i": pred_i - baseline_pred,
                "individual_effect_j": pred_j - baseline_pred,
                "joint_effect": joint_effect,
                "interaction_effect": interaction_effect
            })
        
        # Sort by interaction strength
        interaction_results.sort(key=lambda x: abs(x["interaction_effect"]), reverse=True)
        
        result = {
            "baseline_prediction": baseline_pred,
            "interactions": interaction_results,
            "strongest_interaction": interaction_results[0] if interaction_results else None
        }
        
        logger.info(f"Analyzed {len(interaction_results)} feature interactions")
        
        return result


class InterpretabilityDashboard:
    """
    Unified dashboard for all interpretability techniques
    
    This class provides a unified interface to all interpretability methods
    and can generate comprehensive explanations for predictions.
    """
    
    def __init__(self,
                 model: Any,
                 feature_names: Optional[List[str]] = None):
        """
        Initialize interpretability dashboard
        
        Args:
            model: Model to explain
            feature_names: Names of features
        """
        self.model = model
        self.feature_names = feature_names
        
        # Initialize all explainers
        self.lime_explainer = LIMEExplainer(model, feature_names)
        self.counterfactual_explainer = CounterfactualExplainer(model, feature_names)
        self.attention_mechanism = AttentionMechanism()
        self.causal_analyzer = CausalInferenceAnalyzer(model, feature_names)
        
        logger.info("Initialized interpretability dashboard")
    
    def comprehensive_explanation(self,
                                 instance: np.ndarray,
                                 training_data: Optional[np.ndarray] = None,
                                 include_counterfactuals: bool = True,
                                 include_causal: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for an instance
        
        Args:
            instance: Instance to explain
            training_data: Training data for context
            include_counterfactuals: Whether to include counterfactual explanations
            include_causal: Whether to include causal analysis
            
        Returns:
            Comprehensive explanation dictionary
        """
        logger.info("Generating comprehensive explanation")
        
        explanation = {
            "instance": instance,
            "prediction": self.model.predict([instance])[0]
        }
        
        # LIME explanation
        try:
            lime_explanation = self.lime_explainer.explain_instance(instance, training_data)
            explanation["lime"] = lime_explanation
        except Exception as e:
            logger.warning(f"LIME explanation failed: {e}")
            explanation["lime"] = None
        
        # Attention weights
        try:
            attention_weights = self.attention_mechanism.compute_attention_weights(instance)
            attention_viz = self.attention_mechanism.visualize_attention(
                instance, attention_weights, self.feature_names
            )
            explanation["attention"] = attention_viz
        except Exception as e:
            logger.warning(f"Attention analysis failed: {e}")
            explanation["attention"] = None
        
        # Counterfactual explanations
        if include_counterfactuals:
            try:
                # Try to generate counterfactual for opposite class
                current_class = self.model.predict([instance])[0]
                target_class = 1 - current_class if current_class in [0, 1] else 0
                
                counterfactual = self.counterfactual_explainer.generate_counterfactual(
                    instance, target_class
                )
                explanation["counterfactual"] = counterfactual
            except Exception as e:
                logger.warning(f"Counterfactual explanation failed: {e}")
                explanation["counterfactual"] = None
        
        # Causal analysis
        if include_causal and training_data is not None:
            try:
                # Discover causal structure if not already done
                if self.causal_analyzer.causal_graph is None:
                    self.causal_analyzer.discover_causal_structure(training_data)
                
                # Analyze top features
                if explanation["lime"] is not None:
                    top_features = explanation["lime"]["top_features"][:3]  # Top 3 features
                    causal_effects = []
                    
                    for feature_name, importance in top_features:
                        if self.feature_names:
                            feature_idx = self.feature_names.index(feature_name)
                        else:
                            feature_idx = int(feature_name.split('_')[1])
                        
                        # Analyze causal effect of this feature
                        intervention_values = np.linspace(
                            instance[feature_idx] * 0.8,
                            instance[feature_idx] * 1.2,
                            5
                        )
                        
                        causal_effect = self.causal_analyzer.compute_causal_effects(
                            instance, feature_idx, intervention_values
                        )
                        causal_effects.append(causal_effect)
                    
                    explanation["causal_effects"] = causal_effects
                
            except Exception as e:
                logger.warning(f"Causal analysis failed: {e}")
                explanation["causal_effects"] = None
        
        logger.info("Comprehensive explanation completed")
        
        return explanation