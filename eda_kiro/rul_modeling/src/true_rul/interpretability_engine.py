"""
Interpretability Engine for RUL Prediction System

This module provides comprehensive interpretability features including:
- Feature importance aggregation across ensemble models
- SHAP value computation and analysis
- Diagnostic report generation for unusual predictions
- Visualization utilities for model explanations

Requirements: 9.1, 9.2, 9.3, 9.4, 9.5
"""

import logging
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

# Optional imports for visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    warnings.warn("Matplotlib/Seaborn not available. Plotting features disabled.")

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    warnings.warn("SHAP not available. SHAP plotting features disabled.")

from .data_structures import PredictionResult

logger = logging.getLogger(__name__)


class InterpretabilityEngine:
    """
    Engine for providing comprehensive model interpretability
    
    This class aggregates interpretability information from multiple models
    and provides unified explanations for predictions.
    """
    
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        enable_plotting: bool = True
    ):
        """
        Initialize interpretability engine
        
        Args:
            feature_names: List of feature names for interpretability
            enable_plotting: Whether to enable plotting features
        """
        self.feature_names = feature_names or []
        self.enable_plotting = enable_plotting and HAS_PLOTTING
        
        # Historical data for context
        self.prediction_history: List[Dict[str, Any]] = []
        self.feature_importance_history: List[Dict[str, float]] = []
        
        # Thresholds for diagnostic reports
        self.deviation_threshold = 2.0  # Standard deviations
        self.importance_threshold = 0.1  # Minimum importance to report
        
        logger.info(f"InterpretabilityEngine initialized with {len(self.feature_names)} features")
    
    def aggregate_feature_importance(
        self,
        importance_dict_list: List[Dict[str, float]],
        weights: Optional[List[float]] = None,
        normalize: bool = True
    ) -> Dict[str, float]:
        """
        Aggregate feature importance across multiple models
        
        Args:
            importance_dict_list: List of feature importance dictionaries
            weights: Optional weights for each model (default: equal weights)
            normalize: Whether to normalize importance scores to sum to 1.0
            
        Returns:
            Aggregated feature importance dictionary
        """
        if not importance_dict_list:
            return {}
        
        # Set equal weights if not provided
        if weights is None:
            weights = [1.0 / len(importance_dict_list)] * len(importance_dict_list)
        
        if len(weights) != len(importance_dict_list):
            raise ValueError("Number of weights must match number of importance dictionaries")
        
        # Aggregate importance scores
        aggregated = {}
        
        for importance_dict, weight in zip(importance_dict_list, weights):
            for feature, importance in importance_dict.items():
                if feature not in aggregated:
                    aggregated[feature] = 0.0
                aggregated[feature] += importance * weight
        
        # Normalize if requested
        if normalize and aggregated:
            total_importance = sum(aggregated.values())
            if total_importance > 0:
                aggregated = {
                    feature: importance / total_importance
                    for feature, importance in aggregated.items()
                }
        
        return aggregated
    
    def get_top_features(
        self,
        feature_importance: Dict[str, float],
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Get top K most important features
        
        Args:
            feature_importance: Feature importance dictionary
            top_k: Number of top features to return
            
        Returns:
            List of (feature_name, importance) tuples sorted by importance
        """
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        return sorted_features[:top_k]
    
    def analyze_shap_values(
        self,
        shap_values: np.ndarray,
        feature_names: Optional[List[str]] = None,
        sample_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Analyze SHAP values for a specific prediction
        
        Args:
            shap_values: SHAP values array (n_samples, n_features)
            feature_names: Feature names (uses self.feature_names if None)
            sample_idx: Index of sample to analyze
            
        Returns:
            Dictionary with SHAP analysis results
        """
        if feature_names is None:
            feature_names = self.feature_names
        
        if len(feature_names) != shap_values.shape[1]:
            logger.warning(
                f"Feature names length ({len(feature_names)}) doesn't match "
                f"SHAP values features ({shap_values.shape[1]})"
            )
            feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]
        
        sample_shap = shap_values[sample_idx]
        
        # Create feature importance from SHAP values
        shap_importance = {
            feature: float(abs(shap_val))
            for feature, shap_val in zip(feature_names, sample_shap)
        }
        
        # Get top positive and negative contributors
        positive_contrib = [
            (feature, float(shap_val))
            for feature, shap_val in zip(feature_names, sample_shap)
            if shap_val > 0
        ]
        negative_contrib = [
            (feature, float(shap_val))
            for feature, shap_val in zip(feature_names, sample_shap)
            if shap_val < 0
        ]
        
        positive_contrib.sort(key=lambda x: x[1], reverse=True)
        negative_contrib.sort(key=lambda x: x[1])
        
        return {
            "shap_values": sample_shap.tolist(),
            "feature_names": feature_names,
            "shap_importance": shap_importance,
            "top_positive_contributors": positive_contrib[:5],
            "top_negative_contributors": negative_contrib[:5],
            "total_shap_magnitude": float(np.sum(np.abs(sample_shap))),
            "shap_sum": float(np.sum(sample_shap))
        }
    
    def generate_diagnostic_report(
        self,
        prediction_result: PredictionResult,
        feature_importance: Dict[str, float],
        shap_analysis: Optional[Dict[str, Any]] = None,
        historical_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate diagnostic report for predictions with significant deviation
        
        Args:
            prediction_result: The prediction result to analyze
            feature_importance: Feature importance for this prediction
            shap_analysis: Optional SHAP analysis results
            historical_context: Optional historical context data
            
        Returns:
            Comprehensive diagnostic report
        """
        report = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "prediction_summary": {
                "rul_cycles": prediction_result.rul_cycles,
                "degradation_score": prediction_result.degradation_score,
                "degradation_stage": prediction_result.degradation_stage,
                "anomaly_flag": prediction_result.anomaly_flag,
                "anomaly_score": prediction_result.anomaly_score,
                "confidence_interval": [
                    prediction_result.rul_confidence_lower,
                    prediction_result.rul_confidence_upper
                ]
            },
            "interpretability_analysis": {},
            "deviation_analysis": {},
            "recommendations": []
        }
        
        # Feature importance analysis
        top_features = self.get_top_features(feature_importance, top_k=10)
        report["interpretability_analysis"]["top_features"] = [
            {"feature": feature, "importance": importance}
            for feature, importance in top_features
        ]
        
        # Check for unusual feature importance patterns
        unusual_features = [
            feature for feature, importance in feature_importance.items()
            if importance > self.importance_threshold * 3  # 3x normal threshold
        ]
        
        if unusual_features:
            report["interpretability_analysis"]["unusual_high_importance"] = unusual_features
            report["recommendations"].append(
                "High feature importance detected. Investigate data quality for: " +
                ", ".join(unusual_features[:3])
            )
        
        # SHAP analysis if available
        if shap_analysis:
            report["interpretability_analysis"]["shap_analysis"] = {
                "top_positive_contributors": shap_analysis["top_positive_contributors"][:3],
                "top_negative_contributors": shap_analysis["top_negative_contributors"][:3],
                "total_explanation_magnitude": shap_analysis["total_shap_magnitude"]
            }
        
        # Historical deviation analysis
        if historical_context and self.prediction_history:
            report["deviation_analysis"] = self._analyze_historical_deviation(
                prediction_result, historical_context
            )
        
        # Confidence analysis
        confidence_width = (
            prediction_result.rul_confidence_upper - prediction_result.rul_confidence_lower
        )
        if confidence_width > prediction_result.rul_cycles * 0.5:  # Wide confidence interval
            report["recommendations"].append(
                "Wide confidence interval detected. Consider additional monitoring."
            )
        
        # Anomaly analysis
        if prediction_result.anomaly_flag:
            report["recommendations"].append(
                f"Anomaly detected (score: {prediction_result.anomaly_score:.3f}). "
                "Immediate inspection recommended."
            )
        
        return report
    
    def _analyze_historical_deviation(
        self,
        current_prediction: PredictionResult,
        historical_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze deviation from historical patterns
        
        Args:
            current_prediction: Current prediction result
            historical_context: Historical context data
            
        Returns:
            Deviation analysis results
        """
        deviation_analysis = {
            "has_significant_deviation": False,
            "deviation_details": []
        }
        
        # Check RUL deviation
        if "historical_rul_mean" in historical_context:
            historical_mean = historical_context["historical_rul_mean"]
            historical_std = historical_context.get("historical_rul_std", historical_mean * 0.1)
            
            rul_deviation = abs(current_prediction.rul_cycles - historical_mean) / historical_std
            
            if rul_deviation > self.deviation_threshold:
                deviation_analysis["has_significant_deviation"] = True
                deviation_analysis["deviation_details"].append({
                    "metric": "rul_cycles",
                    "current_value": current_prediction.rul_cycles,
                    "historical_mean": historical_mean,
                    "deviation_magnitude": rul_deviation,
                    "severity": "high" if rul_deviation > 3.0 else "medium"
                })
        
        # Check degradation score deviation
        if "historical_degradation_mean" in historical_context:
            historical_mean = historical_context["historical_degradation_mean"]
            historical_std = historical_context.get("historical_degradation_std", 0.1)
            
            deg_deviation = abs(current_prediction.degradation_score - historical_mean) / historical_std
            
            if deg_deviation > self.deviation_threshold:
                deviation_analysis["has_significant_deviation"] = True
                deviation_analysis["deviation_details"].append({
                    "metric": "degradation_score",
                    "current_value": current_prediction.degradation_score,
                    "historical_mean": historical_mean,
                    "deviation_magnitude": deg_deviation,
                    "severity": "high" if deg_deviation > 3.0 else "medium"
                })
        
        return deviation_analysis
    
    def update_history(
        self,
        prediction_result: PredictionResult,
        feature_importance: Dict[str, float]
    ) -> None:
        """
        Update historical data for context
        
        Args:
            prediction_result: Prediction result to add to history
            feature_importance: Feature importance to add to history
        """
        # Add to prediction history
        self.prediction_history.append({
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "rul_cycles": prediction_result.rul_cycles,
            "degradation_score": prediction_result.degradation_score,
            "degradation_stage": prediction_result.degradation_stage,
            "anomaly_flag": prediction_result.anomaly_flag,
            "anomaly_score": prediction_result.anomaly_score
        })
        
        # Add to feature importance history
        self.feature_importance_history.append(feature_importance.copy())
        
        # Keep only recent history (last 100 predictions)
        if len(self.prediction_history) > 100:
            self.prediction_history = self.prediction_history[-100:]
            self.feature_importance_history = self.feature_importance_history[-100:]
    
    def get_historical_context(self) -> Dict[str, Any]:
        """
        Get historical context for deviation analysis
        
        Returns:
            Historical context dictionary
        """
        if not self.prediction_history:
            return {}
        
        # Calculate historical statistics
        rul_values = [p["rul_cycles"] for p in self.prediction_history]
        degradation_values = [p["degradation_score"] for p in self.prediction_history]
        
        context = {
            "n_historical_predictions": len(self.prediction_history),
            "historical_rul_mean": np.mean(rul_values),
            "historical_rul_std": np.std(rul_values),
            "historical_rul_min": np.min(rul_values),
            "historical_rul_max": np.max(rul_values),
            "historical_degradation_mean": np.mean(degradation_values),
            "historical_degradation_std": np.std(degradation_values),
            "recent_anomaly_rate": np.mean([
                p["anomaly_flag"] for p in self.prediction_history[-20:]
            ]) if len(self.prediction_history) >= 20 else 0.0
        }
        
        return context
    
    def create_feature_importance_plot(
        self,
        feature_importance: Dict[str, float],
        title: str = "Feature Importance",
        top_k: int = 15,
        figsize: Tuple[int, int] = (10, 8)
    ) -> Optional[Any]:
        """
        Create feature importance plot
        
        Args:
            feature_importance: Feature importance dictionary
            title: Plot title
            top_k: Number of top features to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure or None if plotting disabled
        """
        if not self.enable_plotting:
            logger.warning("Plotting disabled. Install matplotlib to enable.")
            return None
        
        top_features = self.get_top_features(feature_importance, top_k)
        
        if not top_features:
            logger.warning("No features to plot")
            return None
        
        features, importances = zip(*top_features)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create horizontal bar plot
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, importances, alpha=0.8)
        
        # Color bars based on importance magnitude
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()  # Top feature at top
        ax.set_xlabel('Importance')
        ax.set_title(title)
        
        # Add value labels on bars
        for i, (feature, importance) in enumerate(top_features):
            ax.text(importance + max(importances) * 0.01, i, 
                   f'{importance:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def create_shap_summary_plot(
        self,
        shap_values: np.ndarray,
        feature_values: np.ndarray,
        feature_names: Optional[List[str]] = None,
        max_display: int = 15
    ) -> Optional[Any]:
        """
        Create SHAP summary plot
        
        Args:
            shap_values: SHAP values array (n_samples, n_features)
            feature_values: Feature values array (n_samples, n_features)
            feature_names: Feature names
            max_display: Maximum number of features to display
            
        Returns:
            SHAP plot or None if SHAP not available
        """
        if not HAS_SHAP:
            logger.warning("SHAP not available. Install shap to enable SHAP plots.")
            return None
        
        if feature_names is None:
            feature_names = self.feature_names
        
        try:
            fig = plt.figure(figsize=(10, 8))
            shap.summary_plot(
                shap_values, 
                feature_values,
                feature_names=feature_names,
                max_display=max_display,
                show=False
            )
            return fig
        except Exception as e:
            logger.error(f"Failed to create SHAP summary plot: {e}")
            return None
    
    def create_shap_waterfall_plot(
        self,
        shap_values: np.ndarray,
        feature_values: np.ndarray,
        expected_value: float,
        sample_idx: int = 0,
        feature_names: Optional[List[str]] = None,
        max_display: int = 10
    ) -> Optional[Any]:
        """
        Create SHAP waterfall plot for a single prediction
        
        Args:
            shap_values: SHAP values array (n_samples, n_features)
            feature_values: Feature values array (n_samples, n_features)
            expected_value: Expected value (baseline prediction)
            sample_idx: Index of sample to plot
            feature_names: Feature names
            max_display: Maximum number of features to display
            
        Returns:
            SHAP plot or None if SHAP not available
        """
        if not HAS_SHAP:
            logger.warning("SHAP not available. Install shap to enable SHAP plots.")
            return None
        
        if feature_names is None:
            feature_names = self.feature_names
        
        try:
            fig = plt.figure(figsize=(10, 6))
            
            # Create explanation object for waterfall plot
            explanation = shap.Explanation(
                values=shap_values[sample_idx],
                base_values=expected_value,
                data=feature_values[sample_idx],
                feature_names=feature_names
            )
            
            shap.waterfall_plot(explanation, max_display=max_display, show=False)
            return fig
        except Exception as e:
            logger.error(f"Failed to create SHAP waterfall plot: {e}")
            return None
    
    def export_diagnostic_report(
        self,
        diagnostic_report: Dict[str, Any],
        filepath: str,
        format: str = "json"
    ) -> None:
        """
        Export diagnostic report to file
        
        Args:
            diagnostic_report: Diagnostic report dictionary
            filepath: Output file path
            format: Export format ("json" or "html")
        """
        if format.lower() == "json":
            import json
            with open(filepath, 'w') as f:
                json.dump(diagnostic_report, f, indent=2)
        
        elif format.lower() == "html":
            self._export_html_report(diagnostic_report, filepath)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Diagnostic report exported to {filepath}")
    
    def _export_html_report(
        self,
        diagnostic_report: Dict[str, Any],
        filepath: str
    ) -> None:
        """
        Export diagnostic report as HTML
        
        Args:
            diagnostic_report: Diagnostic report dictionary
            filepath: Output HTML file path
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>RUL Prediction Diagnostic Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ margin: 5px 0; }}
                .recommendation {{ background-color: #fff3cd; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                .deviation {{ background-color: #f8d7da; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>RUL Prediction Diagnostic Report</h1>
                <p>Generated: {diagnostic_report['timestamp']}</p>
            </div>
            
            <div class="section">
                <h2>Prediction Summary</h2>
                <div class="metric">RUL Cycles: {diagnostic_report['prediction_summary']['rul_cycles']}</div>
                <div class="metric">Degradation Score: {diagnostic_report['prediction_summary']['degradation_score']:.3f}</div>
                <div class="metric">Degradation Stage: {diagnostic_report['prediction_summary']['degradation_stage']}</div>
                <div class="metric">Anomaly Flag: {diagnostic_report['prediction_summary']['anomaly_flag']}</div>
                <div class="metric">Anomaly Score: {diagnostic_report['prediction_summary']['anomaly_score']:.3f}</div>
            </div>
            
            <div class="section">
                <h2>Top Contributing Features</h2>
                <table>
                    <tr><th>Feature</th><th>Importance</th></tr>
        """
        
        # Add top features table
        for feature_info in diagnostic_report['interpretability_analysis']['top_features'][:10]:
            html_content += f"<tr><td>{feature_info['feature']}</td><td>{feature_info['importance']:.4f}</td></tr>"
        
        html_content += """
                </table>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
        """
        
        # Add recommendations
        for recommendation in diagnostic_report['recommendations']:
            html_content += f'<div class="recommendation">{recommendation}</div>'
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(filepath, 'w') as f:
            f.write(html_content)
    
    def clear_history(self) -> None:
        """Clear historical data"""
        self.prediction_history.clear()
        self.feature_importance_history.clear()
        logger.info("Interpretability history cleared")
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics about interpretability engine state
        
        Returns:
            Summary statistics dictionary
        """
        return {
            "n_features": len(self.feature_names),
            "n_historical_predictions": len(self.prediction_history),
            "n_feature_importance_records": len(self.feature_importance_history),
            "plotting_enabled": self.enable_plotting,
            "shap_available": HAS_SHAP,
            "deviation_threshold": self.deviation_threshold,
            "importance_threshold": self.importance_threshold
        }