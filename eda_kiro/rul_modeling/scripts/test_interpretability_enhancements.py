#!/usr/bin/env python3
"""
Test script for interpretability enhancements

This script tests the advanced interpretability techniques including
LIME, counterfactual explanations, attention mechanisms, and causal inference.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression, make_classification
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from true_rul.interpretability_enhancements import (
    LIMEExplainer,
    CounterfactualExplainer,
    AttentionMechanism,
    CausalInferenceAnalyzer,
    InterpretabilityDashboard
)
from true_rul.rul_regression_model import RULRegressionModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_test_data():
    """Generate test data for interpretability testing"""
    logger.info("Generating test data...")
    
    # Regression data for RUL prediction
    X_reg, y_reg = make_regression(
        n_samples=200,
        n_features=10,
        noise=0.1,
        random_state=42
    )
    
    # Ensure RUL values are positive
    y_reg = np.abs(y_reg) + 1
    
    # Classification data for anomaly detection
    X_clf, y_clf = make_classification(
        n_samples=200,
        n_features=10,
        n_classes=2,
        n_redundant=0,
        random_state=42
    )
    
    # Split data
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.3, random_state=42
    )
    
    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
        X_clf, y_clf, test_size=0.3, random_state=42
    )
    
    return {
        'regression': {
            'X_train': X_reg_train,
            'y_train': y_reg_train,
            'X_test': X_reg_test,
            'y_test': y_reg_test
        },
        'classification': {
            'X_train': X_clf_train,
            'y_train': y_clf_train,
            'X_test': X_clf_test,
            'y_test': y_clf_test
        }
    }


def test_lime_explainer():
    """Test LIME explainer"""
    logger.info("Testing LIME explainer...")
    
    data = generate_test_data()
    reg_data = data['regression']
    
    # Train a simple model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(reg_data['X_train'], reg_data['y_train'])
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(reg_data['X_train'].shape[1])]
    
    # Initialize LIME explainer
    lime_explainer = LIMEExplainer(
        model=model,
        feature_names=feature_names,
        n_samples=500  # Reduced for testing
    )
    
    # Explain a single instance
    test_instance = reg_data['X_test'][0]
    explanation = lime_explainer.explain_instance(
        test_instance,
        training_data=reg_data['X_train'],
        num_features=5
    )
    
    logger.info(f"LIME explanation results:")
    logger.info(f"  Original prediction: {explanation['original_prediction']:.3f}")
    logger.info(f"  Local prediction: {explanation['local_prediction']:.3f}")
    logger.info(f"  Fidelity: {explanation['fidelity']:.3f}")
    logger.info(f"  Local model R²: {explanation['local_model_r2']:.3f}")
    logger.info(f"  Top features:")
    
    for feature_name, importance in explanation['top_features']:
        logger.info(f"    {feature_name}: {importance:.3f}")
    
    # Test batch explanation
    batch_explanations = lime_explainer.explain_batch(
        reg_data['X_test'][:3],
        training_data=reg_data['X_train'],
        num_features=3
    )
    
    logger.info(f"  Batch explanations: {len(batch_explanations)} instances")
    
    return lime_explainer


def test_counterfactual_explainer():
    """Test counterfactual explainer"""
    logger.info("Testing counterfactual explainer...")
    
    data = generate_test_data()
    clf_data = data['classification']
    
    # Train a classification model
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(clf_data['X_train'], clf_data['y_train'])
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(clf_data['X_train'].shape[1])]
    
    # Initialize counterfactual explainer
    cf_explainer = CounterfactualExplainer(
        model=model,
        feature_names=feature_names,
        max_iterations=100,  # Reduced for testing
        learning_rate=0.1
    )
    
    # Generate counterfactual for a test instance
    test_instance = clf_data['X_test'][0]
    original_class = model.predict([test_instance])[0]
    target_class = 1 - original_class  # Flip class
    
    counterfactual = cf_explainer.generate_counterfactual(
        test_instance,
        target_class
    )
    
    logger.info(f"Counterfactual explanation results:")
    logger.info(f"  Original class: {counterfactual['original_class']}")
    logger.info(f"  Target class: {counterfactual['target_class']}")
    logger.info(f"  Final class: {counterfactual['final_class']}")
    logger.info(f"  Success: {counterfactual['success']}")
    logger.info(f"  Distance: {counterfactual['distance']:.3f}")
    logger.info(f"  Iterations: {counterfactual['iterations']}")
    logger.info(f"  Number of changes: {len(counterfactual['changes'])}")
    
    if counterfactual['changes']:
        logger.info(f"  Top changes:")
        for feature_name, change_info in list(counterfactual['changes'].items())[:3]:
            logger.info(f"    {feature_name}: {change_info['original']:.3f} → {change_info['counterfactual']:.3f}")
    
    return cf_explainer


def test_attention_mechanism():
    """Test attention mechanism"""
    logger.info("Testing attention mechanism...")
    
    # Generate sample features
    np.random.seed(42)
    features = np.random.randn(10, 8)  # 10 timesteps, 8 features
    feature_names = [f"feature_{i}" for i in range(8)]
    
    # Test additive attention
    attention_additive = AttentionMechanism(attention_type="additive")
    weights_additive = attention_additive.compute_attention_weights(features)
    
    logger.info(f"Additive attention results:")
    logger.info(f"  Attention weights shape: {weights_additive.shape}")
    logger.info(f"  Weights sum: {np.sum(weights_additive):.3f}")
    logger.info(f"  Max weight: {np.max(weights_additive):.3f}")
    logger.info(f"  Min weight: {np.min(weights_additive):.3f}")
    
    # Test multiplicative attention
    attention_multiplicative = AttentionMechanism(attention_type="multiplicative")
    weights_multiplicative = attention_multiplicative.compute_attention_weights(features)
    
    logger.info(f"Multiplicative attention results:")
    logger.info(f"  Attention weights shape: {weights_multiplicative.shape}")
    logger.info(f"  Weights sum: {np.sum(weights_multiplicative):.3f}")
    
    # Test visualization
    single_feature = features[0]  # Single timestep
    single_weights = attention_additive.compute_attention_weights(single_feature.reshape(1, -1))
    
    viz_data = attention_additive.visualize_attention(
        single_feature,
        single_weights,
        feature_names
    )
    
    logger.info(f"Attention visualization:")
    logger.info(f"  Max attention: {viz_data['max_attention']:.3f}")
    logger.info(f"  Min attention: {viz_data['min_attention']:.3f}")
    logger.info(f"  Attention entropy: {viz_data['attention_entropy']:.3f}")
    
    return attention_additive


def test_causal_inference_analyzer():
    """Test causal inference analyzer"""
    logger.info("Testing causal inference analyzer...")
    
    data = generate_test_data()
    reg_data = data['regression']
    
    # Train a model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(reg_data['X_train'], reg_data['y_train'])
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(reg_data['X_train'].shape[1])]
    
    # Initialize causal analyzer
    causal_analyzer = CausalInferenceAnalyzer(
        model=model,
        feature_names=feature_names
    )
    
    # Discover causal structure
    causal_graph = causal_analyzer.discover_causal_structure(
        reg_data['X_train'],
        method="correlation"
    )
    
    logger.info(f"Causal structure discovery:")
    logger.info(f"  Causal graph shape: {causal_graph.shape}")
    logger.info(f"  Number of edges: {np.sum(causal_graph)}")
    logger.info(f"  Graph density: {np.sum(causal_graph) / (causal_graph.shape[0] * causal_graph.shape[1]):.3f}")
    
    # Compute causal effects
    test_instance = reg_data['X_test'][0]
    intervention_feature = 0
    intervention_values = np.linspace(
        test_instance[intervention_feature] * 0.5,
        test_instance[intervention_feature] * 1.5,
        5
    )
    
    causal_effects = causal_analyzer.compute_causal_effects(
        test_instance,
        intervention_feature,
        intervention_values
    )
    
    logger.info(f"Causal effects analysis:")
    logger.info(f"  Intervention feature: {causal_effects['feature_name']}")
    logger.info(f"  Baseline prediction: {causal_effects['baseline_prediction']:.3f}")
    logger.info(f"  Average treatment effect: {causal_effects['average_treatment_effect']:.3f}")
    logger.info(f"  Effect variance: {causal_effects['effect_variance']:.3f}")
    
    # Analyze feature interactions
    interactions = causal_analyzer.analyze_feature_interactions(
        test_instance,
        feature_pairs=[(0, 1), (1, 2), (2, 3)]
    )
    
    logger.info(f"Feature interactions analysis:")
    logger.info(f"  Number of interactions analyzed: {len(interactions['interactions'])}")
    
    if interactions['strongest_interaction']:
        strongest = interactions['strongest_interaction']
        logger.info(f"  Strongest interaction: {strongest['feature_i_name']} × {strongest['feature_j_name']}")
        logger.info(f"  Interaction effect: {strongest['interaction_effect']:.3f}")
    
    return causal_analyzer


def test_interpretability_dashboard():
    """Test comprehensive interpretability dashboard"""
    logger.info("Testing interpretability dashboard...")
    
    data = generate_test_data()
    reg_data = data['regression']
    
    # Train a model
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(reg_data['X_train'], reg_data['y_train'])
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(reg_data['X_train'].shape[1])]
    
    # Initialize dashboard
    dashboard = InterpretabilityDashboard(
        model=model,
        feature_names=feature_names
    )
    
    # Generate comprehensive explanation
    test_instance = reg_data['X_test'][0]
    
    comprehensive_explanation = dashboard.comprehensive_explanation(
        test_instance,
        training_data=reg_data['X_train'],
        include_counterfactuals=False,  # Skip for regression model
        include_causal=True
    )
    
    logger.info(f"Comprehensive explanation results:")
    logger.info(f"  Prediction: {comprehensive_explanation['prediction']:.3f}")
    
    # LIME results
    if comprehensive_explanation['lime']:
        lime_result = comprehensive_explanation['lime']
        logger.info(f"  LIME fidelity: {lime_result['fidelity']:.3f}")
        logger.info(f"  LIME top feature: {lime_result['top_features'][0][0]}")
    
    # Attention results
    if comprehensive_explanation['attention']:
        attention_result = comprehensive_explanation['attention']
        logger.info(f"  Attention entropy: {attention_result['attention_entropy']:.3f}")
    
    # Causal effects results
    if comprehensive_explanation['causal_effects']:
        causal_result = comprehensive_explanation['causal_effects'][0]
        logger.info(f"  Top causal effect: {causal_result['average_treatment_effect']:.3f}")
    
    return dashboard


def main():
    """Run all interpretability enhancement tests"""
    logger.info("Starting interpretability enhancement tests...")
    
    try:
        # Test each interpretability technique
        logger.info("\n" + "="*50)
        test_lime_explainer()
        
        logger.info("\n" + "="*50)
        test_counterfactual_explainer()
        
        logger.info("\n" + "="*50)
        test_attention_mechanism()
        
        logger.info("\n" + "="*50)
        test_causal_inference_analyzer()
        
        logger.info("\n" + "="*50)
        test_interpretability_dashboard()
        
        logger.info("\n" + "="*50)
        logger.info("All interpretability enhancement tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


if __name__ == "__main__":
    main()