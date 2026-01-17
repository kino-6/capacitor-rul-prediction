"""
Response Feature Extractor for VL-VO relationship analysis.

Extracts features that quantify the relationship between input voltage (VL)
and output voltage (VO) to detect capacitor degradation.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from scipy import signal


class ResponseFeatureExtractor:
    """
    Extract response-based features from VL-VO relationship.
    
    Features are designed to capture:
    1. Energy transfer efficiency
    2. Waveform similarity
    3. Response delay
    4. Deviation from initial state
    5. Degradation rate
    """
    
    def __init__(self):
        """Initialize the response feature extractor."""
        self.initial_stats = {}
    
    def extract_features(
        self,
        vl: np.ndarray,
        vo: np.ndarray,
        capacitor_id: str,
        cycle: int,
        include_advanced: bool = False
    ) -> Dict[str, float]:
        """
        Extract all response features from VL and VO data.
        
        Args:
            vl: Input voltage array
            vo: Output voltage array
            capacitor_id: Capacitor identifier
            cycle: Cycle number
            include_advanced: Whether to include advanced features
        
        Returns:
            Dictionary of feature names and values
        """
        features = {}
        
        # Basic features (high priority)
        features.update(self._extract_energy_features(vl, vo))
        features.update(self._extract_waveform_features(vl, vo))
        features.update(self._extract_delay_features(vl, vo))
        
        # Deviation features (require initial stats)
        if capacitor_id in self.initial_stats:
            features.update(self._extract_deviation_features(
                features, capacitor_id
            ))
        
        # Advanced features (optional)
        if include_advanced:
            features.update(self._extract_advanced_features(vl, vo))
        
        # Store initial stats for first 10 cycles
        if cycle <= 10:
            self._update_initial_stats(capacitor_id, features)
        
        return features
    
    def _extract_energy_features(
        self,
        vl: np.ndarray,
        vo: np.ndarray
    ) -> Dict[str, float]:
        """Extract energy transfer features."""
        features = {}
        
        # Response efficiency (energy ratio)
        vl_energy = np.sum(vl ** 2)
        vo_energy = np.sum(vo ** 2)
        features['response_efficiency'] = (
            vo_energy / vl_energy if vl_energy > 0 else 0
        )
        
        # Voltage ratio (mean)
        vl_mean = np.mean(vl)
        vo_mean = np.mean(vo)
        features['voltage_ratio'] = (
            vo_mean / vl_mean if vl_mean != 0 else 0
        )
        
        # Peak voltage ratio
        vl_peak = np.max(np.abs(vl))
        vo_peak = np.max(np.abs(vo))
        features['peak_voltage_ratio'] = (
            vo_peak / vl_peak if vl_peak > 0 else 0
        )
        
        # RMS voltage ratio
        vl_rms = np.sqrt(np.mean(vl ** 2))
        vo_rms = np.sqrt(np.mean(vo ** 2))
        features['rms_voltage_ratio'] = (
            vo_rms / vl_rms if vl_rms > 0 else 0
        )
        
        return features
    
    def _extract_waveform_features(
        self,
        vl: np.ndarray,
        vo: np.ndarray
    ) -> Dict[str, float]:
        """Extract waveform similarity features."""
        features = {}
        
        # Waveform correlation
        if len(vl) > 1 and len(vo) > 1:
            corr_matrix = np.corrcoef(vl, vo)
            features['waveform_correlation'] = corr_matrix[0, 1]
        else:
            features['waveform_correlation'] = 0
        
        # VO variability (coefficient of variation)
        vo_std = np.std(vo)
        vo_mean_abs = np.mean(np.abs(vo))
        features['vo_variability'] = (
            vo_std / vo_mean_abs if vo_mean_abs > 0 else 0
        )
        
        # VL variability
        vl_std = np.std(vl)
        vl_mean_abs = np.mean(np.abs(vl))
        features['vl_variability'] = (
            vl_std / vl_mean_abs if vl_mean_abs > 0 else 0
        )
        
        return features
    
    def _extract_delay_features(
        self,
        vl: np.ndarray,
        vo: np.ndarray
    ) -> Dict[str, float]:
        """Extract response delay features."""
        features = {}
        
        # Response delay (cross-correlation peak)
        if len(vl) > 1 and len(vo) > 1:
            # Normalize signals
            vl_norm = vl - np.mean(vl)
            vo_norm = vo - np.mean(vo)
            
            # Cross-correlation
            cross_corr = np.correlate(vl_norm, vo_norm, mode='full')
            delay = np.argmax(cross_corr) - (len(vl) - 1)
            features['response_delay'] = delay
            
            # Normalized delay (as fraction of signal length)
            features['response_delay_normalized'] = delay / len(vl)
        else:
            features['response_delay'] = 0
            features['response_delay_normalized'] = 0
        
        return features
    
    def _extract_deviation_features(
        self,
        current_features: Dict[str, float],
        capacitor_id: str
    ) -> Dict[str, float]:
        """Extract deviation from initial state features."""
        features = {}
        
        initial = self.initial_stats[capacitor_id]
        
        # Check if initial stats are still lists (not yet averaged)
        # This happens during cycles 1-10 before averaging is complete
        if isinstance(initial['response_efficiency'], list):
            # Return zeros for deviation features during initial cycles
            features['efficiency_degradation_rate'] = 0.0
            features['voltage_ratio_deviation'] = 0.0
            features['correlation_shift'] = 0.0
            features['peak_voltage_ratio_deviation'] = 0.0
            return features
        
        # Efficiency degradation rate
        current_eff = current_features['response_efficiency']
        initial_eff = initial['response_efficiency']
        features['efficiency_degradation_rate'] = (
            (initial_eff - current_eff) / initial_eff 
            if initial_eff > 0 else 0
        )
        
        # Voltage ratio deviation
        current_vr = current_features['voltage_ratio']
        initial_vr = initial['voltage_ratio']
        features['voltage_ratio_deviation'] = (
            abs(current_vr - initial_vr) / abs(initial_vr)
            if initial_vr != 0 else 0
        )
        
        # Correlation shift
        current_corr = current_features['waveform_correlation']
        initial_corr = initial['waveform_correlation']
        features['correlation_shift'] = current_corr - initial_corr
        
        # Peak voltage ratio deviation
        current_pvr = current_features['peak_voltage_ratio']
        initial_pvr = initial['peak_voltage_ratio']
        features['peak_voltage_ratio_deviation'] = (
            abs(current_pvr - initial_pvr) / abs(initial_pvr)
            if initial_pvr != 0 else 0
        )
        
        return features
    
    def _extract_advanced_features(
        self,
        vl: np.ndarray,
        vo: np.ndarray
    ) -> Dict[str, float]:
        """Extract advanced features (optional)."""
        features = {}
        
        # Residual energy (linear fit residual)
        if len(vl) > 1 and len(vo) > 1:
            # Linear fit: vo = a * vl + b
            coeffs = np.polyfit(vl, vo, 1)
            vo_predicted = np.polyval(coeffs, vl)
            residual = vo - vo_predicted
            
            vo_energy = np.sum(vo ** 2)
            residual_energy = np.sum(residual ** 2)
            features['residual_energy_ratio'] = (
                residual_energy / vo_energy if vo_energy > 0 else 0
            )
        else:
            features['residual_energy_ratio'] = 0
        
        # Waveform complexity (sample entropy approximation)
        # Simplified version: use standard deviation of differences
        if len(vo) > 1:
            vo_diff = np.diff(vo)
            features['vo_complexity'] = np.std(vo_diff)
        else:
            features['vo_complexity'] = 0
        
        return features
    
    def _update_initial_stats(
        self,
        capacitor_id: str,
        features: Dict[str, float]
    ) -> None:
        """Update initial statistics for first 10 cycles."""
        if capacitor_id not in self.initial_stats:
            self.initial_stats[capacitor_id] = {
                'response_efficiency': [],
                'voltage_ratio': [],
                'waveform_correlation': [],
                'peak_voltage_ratio': []
            }
        
        stats = self.initial_stats[capacitor_id]
        stats['response_efficiency'].append(features['response_efficiency'])
        stats['voltage_ratio'].append(features['voltage_ratio'])
        stats['waveform_correlation'].append(features['waveform_correlation'])
        stats['peak_voltage_ratio'].append(features['peak_voltage_ratio'])
        
        # After 10 cycles, compute averages
        if len(stats['response_efficiency']) == 10:
            for key in stats:
                stats[key] = np.mean(stats[key])
    
    def get_feature_names(self, include_advanced: bool = False) -> list:
        """Get list of feature names."""
        basic_features = [
            # Energy features
            'response_efficiency',
            'voltage_ratio',
            'peak_voltage_ratio',
            'rms_voltage_ratio',
            # Waveform features
            'waveform_correlation',
            'vo_variability',
            'vl_variability',
            # Delay features
            'response_delay',
            'response_delay_normalized',
        ]
        
        deviation_features = [
            'efficiency_degradation_rate',
            'voltage_ratio_deviation',
            'correlation_shift',
            'peak_voltage_ratio_deviation',
        ]
        
        advanced_features = [
            'residual_energy_ratio',
            'vo_complexity',
        ]
        
        features = basic_features + deviation_features
        
        if include_advanced:
            features += advanced_features
        
        return features
    
    def reset_initial_stats(self) -> None:
        """Reset initial statistics."""
        self.initial_stats = {}


def extract_response_features_from_cycle(
    vl: np.ndarray,
    vo: np.ndarray,
    capacitor_id: str,
    cycle: int,
    extractor: Optional[ResponseFeatureExtractor] = None,
    include_advanced: bool = False
) -> Dict[str, float]:
    """
    Convenience function to extract response features from a single cycle.
    
    Args:
        vl: Input voltage array
        vo: Output voltage array
        capacitor_id: Capacitor identifier
        cycle: Cycle number
        extractor: Existing extractor instance (optional)
        include_advanced: Whether to include advanced features
    
    Returns:
        Dictionary of feature names and values
    """
    if extractor is None:
        extractor = ResponseFeatureExtractor()
    
    features = extractor.extract_features(
        vl, vo, capacitor_id, cycle, include_advanced
    )
    
    # Add metadata
    features['capacitor_id'] = capacitor_id
    features['cycle'] = cycle
    
    return features
