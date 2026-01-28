"""
Enhanced Feature Extractor for True RUL Prediction

Extends the existing ResponseFeatureExtractor with additional features:
- Statistical features (mean, std, skewness, kurtosis, etc.)
- Frequency domain features (FFT-based)
- Trend features (linear trends, acceleration)
- Rolling window features
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging

import numpy as np
from scipy import stats, signal
from scipy.fft import fft, fftfreq

# Import existing ResponseFeatureExtractor
sys.path.insert(0, str(Path(__file__).parent.parent))
from feature_extraction.response_extractor import ResponseFeatureExtractor

from .data_structures import CycleData, CapacitorData
from .config import MODEL_CONFIG

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Enhanced feature extractor combining existing and new features
    
    Features extracted:
    - 15 responsiveness features (existing)
    - 12 statistical features (new)
    - 10 frequency domain features (new)
    - 8 trend features (new)
    - 10 rolling window features (new)
    
    Total: ~55 features per cycle
    """
    
    def __init__(self, include_advanced: bool = True, rolling_window: int = 5):
        """
        Initialize feature extractor
        
        Args:
            include_advanced: Whether to include advanced features
            rolling_window: Window size for rolling features
        """
        self.include_advanced = include_advanced
        self.rolling_window = rolling_window
        
        # Initialize existing response feature extractor
        self.response_extractor = ResponseFeatureExtractor()
        
        logger.info(
            f"FeatureExtractor initialized: "
            f"advanced={include_advanced}, window={rolling_window}"
        )
    
    def extract_features(
        self,
        cycle: CycleData,
        capacitor_id: str,
        history: Optional[List[CycleData]] = None
    ) -> Dict[str, float]:
        """
        Extract all features from a single cycle
        
        Args:
            cycle: Current cycle data
            capacitor_id: Capacitor identifier
            history: Previous cycles for rolling features (optional)
            
        Returns:
            Dictionary of feature names to values
        """
        features = {}
        
        # 1. Responsiveness features (15 features) - existing
        response_features = self.extract_responsiveness_features(
            cycle, capacitor_id
        )
        features.update(response_features)
        
        # 2. Statistical features (12 features) - new
        stat_features = self.extract_statistical_features(cycle)
        features.update(stat_features)
        
        # 3. Frequency features (10 features) - new
        freq_features = self.extract_frequency_features(cycle)
        features.update(freq_features)
        
        # 4. Trend features (8 features) - new
        trend_features = self.extract_trend_features(cycle)
        features.update(trend_features)
        
        # 5. Rolling features (10 features) - new (if history available)
        if history is not None and len(history) > 0:
            rolling_features = self.extract_rolling_features(cycle, history)
            features.update(rolling_features)
        else:
            # Fill with zeros if no history
            rolling_features = self._get_empty_rolling_features()
            features.update(rolling_features)
        
        return features
    
    def extract_responsiveness_features(
        self,
        cycle: CycleData,
        capacitor_id: str
    ) -> Dict[str, float]:
        """
        Extract 15 existing responsiveness features
        
        Args:
            cycle: Cycle data
            capacitor_id: Capacitor identifier
            
        Returns:
            Dictionary of responsiveness features
        """
        features = self.response_extractor.extract_features(
            vl=cycle.vl_series,
            vo=cycle.vo_series,
            capacitor_id=capacitor_id,
            cycle=cycle.cycle_number,
            include_advanced=self.include_advanced
        )
        
        # Remove metadata fields
        features.pop('capacitor_id', None)
        features.pop('cycle', None)
        
        return features
    
    def extract_statistical_features(self, cycle: CycleData) -> Dict[str, float]:
        """
        Extract statistical features (mean, std, skewness, kurtosis, etc.)
        
        Args:
            cycle: Cycle data
            
        Returns:
            Dictionary of statistical features (12 features)
        """
        features = {}
        
        vl = cycle.vl_series
        vo = cycle.vo_series
        
        # Remove NaN values for statistics
        vl_clean = vl[~np.isnan(vl)]
        vo_clean = vo[~np.isnan(vo)]
        
        if len(vl_clean) == 0 or len(vo_clean) == 0:
            # Return zeros if no valid data
            return self._get_empty_statistical_features()
        
        # VL statistics (6 features)
        features['vl_skewness'] = float(stats.skew(vl_clean))
        features['vl_kurtosis'] = float(stats.kurtosis(vl_clean))
        features['vl_q25'] = float(np.percentile(vl_clean, 25))
        features['vl_q75'] = float(np.percentile(vl_clean, 75))
        features['vl_iqr'] = features['vl_q75'] - features['vl_q25']
        features['vl_rms'] = float(np.sqrt(np.mean(vl_clean ** 2)))
        
        # VO statistics (6 features)
        features['vo_skewness'] = float(stats.skew(vo_clean))
        features['vo_kurtosis'] = float(stats.kurtosis(vo_clean))
        features['vo_q25'] = float(np.percentile(vo_clean, 25))
        features['vo_q75'] = float(np.percentile(vo_clean, 75))
        features['vo_iqr'] = features['vo_q75'] - features['vo_q25']
        features['vo_rms'] = float(np.sqrt(np.mean(vo_clean ** 2)))
        
        return features
    
    def extract_frequency_features(self, cycle: CycleData) -> Dict[str, float]:
        """
        Extract frequency domain features using FFT
        
        Args:
            cycle: Cycle data
            
        Returns:
            Dictionary of frequency features (10 features)
        """
        features = {}
        
        vl = cycle.vl_series
        vo = cycle.vo_series
        
        # Remove NaN values
        vl_clean = vl[~np.isnan(vl)]
        vo_clean = vo[~np.isnan(vo)]
        
        if len(vl_clean) < 10 or len(vo_clean) < 10:
            # Not enough data for FFT
            return self._get_empty_frequency_features()
        
        # VL frequency features (5 features)
        vl_freq_features = self._compute_fft_features(vl_clean, 'vl')
        features.update(vl_freq_features)
        
        # VO frequency features (5 features)
        vo_freq_features = self._compute_fft_features(vo_clean, 'vo')
        features.update(vo_freq_features)
        
        return features
    
    def _compute_fft_features(
        self,
        signal_data: np.ndarray,
        prefix: str
    ) -> Dict[str, float]:
        """
        Compute FFT-based features for a signal
        
        Args:
            signal_data: Signal array
            prefix: Prefix for feature names ('vl' or 'vo')
            
        Returns:
            Dictionary of FFT features
        """
        features = {}
        
        # Compute FFT
        n = len(signal_data)
        fft_vals = fft(signal_data)
        fft_freqs = fftfreq(n)
        
        # Take positive frequencies only
        pos_mask = fft_freqs > 0
        fft_magnitude = np.abs(fft_vals[pos_mask])
        fft_freqs_pos = fft_freqs[pos_mask]
        
        if len(fft_magnitude) == 0:
            features[f'{prefix}_dominant_freq'] = 0.0
            features[f'{prefix}_spectral_energy'] = 0.0
            features[f'{prefix}_spectral_entropy'] = 0.0
            features[f'{prefix}_spectral_centroid'] = 0.0
            features[f'{prefix}_spectral_rolloff'] = 0.0
            return features
        
        # Dominant frequency
        dominant_idx = np.argmax(fft_magnitude)
        features[f'{prefix}_dominant_freq'] = float(fft_freqs_pos[dominant_idx])
        
        # Spectral energy
        features[f'{prefix}_spectral_energy'] = float(np.sum(fft_magnitude ** 2))
        
        # Spectral entropy
        psd = fft_magnitude ** 2
        psd_norm = psd / np.sum(psd) if np.sum(psd) > 0 else psd
        psd_norm = psd_norm[psd_norm > 0]  # Remove zeros for log
        features[f'{prefix}_spectral_entropy'] = float(
            -np.sum(psd_norm * np.log(psd_norm))
        )
        
        # Spectral centroid
        features[f'{prefix}_spectral_centroid'] = float(
            np.sum(fft_freqs_pos * fft_magnitude) / np.sum(fft_magnitude)
            if np.sum(fft_magnitude) > 0 else 0
        )
        
        # Spectral rolloff (95% of energy)
        cumsum_energy = np.cumsum(fft_magnitude ** 2)
        total_energy = cumsum_energy[-1]
        rolloff_idx = np.where(cumsum_energy >= 0.95 * total_energy)[0]
        if len(rolloff_idx) > 0:
            features[f'{prefix}_spectral_rolloff'] = float(
                fft_freqs_pos[rolloff_idx[0]]
            )
        else:
            features[f'{prefix}_spectral_rolloff'] = 0.0
        
        return features
    
    def extract_trend_features(self, cycle: CycleData) -> Dict[str, float]:
        """
        Extract trend features (linear trends, acceleration)
        
        Args:
            cycle: Cycle data
            
        Returns:
            Dictionary of trend features (8 features)
        """
        features = {}
        
        vl = cycle.vl_series
        vo = cycle.vo_series
        
        # Remove NaN values
        vl_clean = vl[~np.isnan(vl)]
        vo_clean = vo[~np.isnan(vo)]
        
        if len(vl_clean) < 3 or len(vo_clean) < 3:
            # Not enough data for trend analysis
            return self._get_empty_trend_features()
        
        # VL trend features (4 features)
        vl_trend_features = self._compute_trend_features(vl_clean, 'vl')
        features.update(vl_trend_features)
        
        # VO trend features (4 features)
        vo_trend_features = self._compute_trend_features(vo_clean, 'vo')
        features.update(vo_trend_features)
        
        return features
    
    def _compute_trend_features(
        self,
        signal_data: np.ndarray,
        prefix: str
    ) -> Dict[str, float]:
        """
        Compute trend features for a signal
        
        Args:
            signal_data: Signal array
            prefix: Prefix for feature names
            
        Returns:
            Dictionary of trend features
        """
        features = {}
        
        n = len(signal_data)
        x = np.arange(n)
        
        # Linear trend (slope)
        slope, intercept = np.polyfit(x, signal_data, 1)
        features[f'{prefix}_trend_slope'] = float(slope)
        
        # Trend strength (R²)
        y_pred = slope * x + intercept
        ss_res = np.sum((signal_data - y_pred) ** 2)
        ss_tot = np.sum((signal_data - np.mean(signal_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        features[f'{prefix}_trend_strength'] = float(r_squared)
        
        # Acceleration (second derivative approximation)
        if n >= 3:
            # Use central differences for second derivative
            second_diff = np.diff(signal_data, n=2)
            features[f'{prefix}_acceleration'] = float(np.mean(second_diff))
            features[f'{prefix}_acceleration_std'] = float(np.std(second_diff))
        else:
            features[f'{prefix}_acceleration'] = 0.0
            features[f'{prefix}_acceleration_std'] = 0.0
        
        return features
    
    def extract_rolling_features(
        self,
        current_cycle: CycleData,
        history: List[CycleData]
    ) -> Dict[str, float]:
        """
        Extract rolling window features from cycle history
        
        Args:
            current_cycle: Current cycle
            history: Previous cycles (ordered)
            
        Returns:
            Dictionary of rolling features (10 features)
        """
        features = {}
        
        # Get window of previous cycles
        window_size = min(self.rolling_window, len(history))
        if window_size == 0:
            return self._get_empty_rolling_features()
        
        window_cycles = history[-window_size:]
        
        # Extract key metrics from window
        response_efficiencies = []
        voltage_ratios = []
        waveform_correlations = []
        vo_variabilities = []
        vl_variabilities = []
        
        for cycle in window_cycles:
            # Extract basic features for each cycle in window
            cycle_features = self.response_extractor.extract_features(
                vl=cycle.vl_series,
                vo=cycle.vo_series,
                capacitor_id="temp",  # Not used for rolling features
                cycle=cycle.cycle_number,
                include_advanced=False
            )
            
            response_efficiencies.append(
                cycle_features.get('response_efficiency', 0)
            )
            voltage_ratios.append(cycle_features.get('voltage_ratio', 0))
            waveform_correlations.append(
                cycle_features.get('waveform_correlation', 0)
            )
            vo_variabilities.append(cycle_features.get('vo_variability', 0))
            vl_variabilities.append(cycle_features.get('vl_variability', 0))
        
        # Compute rolling statistics
        features['rolling_efficiency_mean'] = float(np.mean(response_efficiencies))
        features['rolling_efficiency_std'] = float(np.std(response_efficiencies))
        features['rolling_voltage_ratio_mean'] = float(np.mean(voltage_ratios))
        features['rolling_voltage_ratio_std'] = float(np.std(voltage_ratios))
        features['rolling_correlation_mean'] = float(np.mean(waveform_correlations))
        features['rolling_correlation_std'] = float(np.std(waveform_correlations))
        features['rolling_vo_variability_mean'] = float(np.mean(vo_variabilities))
        features['rolling_vl_variability_mean'] = float(np.mean(vl_variabilities))
        
        # Trend over window (change from first to last)
        if len(response_efficiencies) >= 2:
            features['rolling_efficiency_trend'] = float(
                response_efficiencies[-1] - response_efficiencies[0]
            )
            features['rolling_correlation_trend'] = float(
                waveform_correlations[-1] - waveform_correlations[0]
            )
        else:
            features['rolling_efficiency_trend'] = 0.0
            features['rolling_correlation_trend'] = 0.0
        
        return features
    
    def _get_empty_statistical_features(self) -> Dict[str, float]:
        """Get empty statistical features (all zeros)"""
        return {
            'vl_skewness': 0.0, 'vl_kurtosis': 0.0, 'vl_q25': 0.0,
            'vl_q75': 0.0, 'vl_iqr': 0.0, 'vl_rms': 0.0,
            'vo_skewness': 0.0, 'vo_kurtosis': 0.0, 'vo_q25': 0.0,
            'vo_q75': 0.0, 'vo_iqr': 0.0, 'vo_rms': 0.0,
        }
    
    def _get_empty_frequency_features(self) -> Dict[str, float]:
        """Get empty frequency features (all zeros)"""
        return {
            'vl_dominant_freq': 0.0, 'vl_spectral_energy': 0.0,
            'vl_spectral_entropy': 0.0, 'vl_spectral_centroid': 0.0,
            'vl_spectral_rolloff': 0.0,
            'vo_dominant_freq': 0.0, 'vo_spectral_energy': 0.0,
            'vo_spectral_entropy': 0.0, 'vo_spectral_centroid': 0.0,
            'vo_spectral_rolloff': 0.0,
        }
    
    def _get_empty_trend_features(self) -> Dict[str, float]:
        """Get empty trend features (all zeros)"""
        return {
            'vl_trend_slope': 0.0, 'vl_trend_strength': 0.0,
            'vl_acceleration': 0.0, 'vl_acceleration_std': 0.0,
            'vo_trend_slope': 0.0, 'vo_trend_strength': 0.0,
            'vo_acceleration': 0.0, 'vo_acceleration_std': 0.0,
        }
    
    def _get_empty_rolling_features(self) -> Dict[str, float]:
        """Get empty rolling features (all zeros)"""
        return {
            'rolling_efficiency_mean': 0.0, 'rolling_efficiency_std': 0.0,
            'rolling_voltage_ratio_mean': 0.0, 'rolling_voltage_ratio_std': 0.0,
            'rolling_correlation_mean': 0.0, 'rolling_correlation_std': 0.0,
            'rolling_vo_variability_mean': 0.0, 'rolling_vl_variability_mean': 0.0,
            'rolling_efficiency_trend': 0.0, 'rolling_correlation_trend': 0.0,
        }
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of all feature names
        
        Returns:
            List of feature names
        """
        # Get responsiveness feature names
        response_names = self.response_extractor.get_feature_names(
            include_advanced=self.include_advanced
        )
        
        # Statistical feature names
        stat_names = list(self._get_empty_statistical_features().keys())
        
        # Frequency feature names
        freq_names = list(self._get_empty_frequency_features().keys())
        
        # Trend feature names
        trend_names = list(self._get_empty_trend_features().keys())
        
        # Rolling feature names
        rolling_names = list(self._get_empty_rolling_features().keys())
        
        return response_names + stat_names + freq_names + trend_names + rolling_names
    
    @property
    def n_features(self) -> int:
        """Get total number of features"""
        return len(self.get_feature_names())
