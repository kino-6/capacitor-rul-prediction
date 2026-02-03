"""
Advanced Feature Extractor for Anomaly Detection

This module implements advanced feature extraction techniques including
wavelet-based features, time-frequency domain analysis, statistical
process control features, and change point detection.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from dataclasses import dataclass
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import pywt
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class AdvancedFeatureConfig:
    """Configuration for advanced feature extraction."""
    # Wavelet features
    wavelet_name: str = "db4"  # Daubechies wavelet
    wavelet_levels: int = 4
    extract_wavelet_energy: bool = True
    extract_wavelet_entropy: bool = True
    extract_wavelet_variance: bool = True
    
    # STFT features
    stft_window: str = "hann"
    stft_nperseg: int = 64
    stft_noverlap: Optional[int] = None
    extract_spectral_centroid: bool = True
    extract_spectral_bandwidth: bool = True
    extract_spectral_rolloff: bool = True
    extract_spectral_flux: bool = True
    
    # SPC features
    spc_window_size: int = 10
    extract_control_limits: bool = True
    extract_process_capability: bool = True
    extract_run_tests: bool = True
    
    # Change point detection
    change_point_method: str = "cusum"  # "cusum", "pelt", "window"
    change_point_threshold: float = 1.0
    extract_change_points: bool = True
    extract_change_magnitude: bool = True
    
    # General settings
    normalize_features: bool = True
    handle_nan: str = "zero"  # "zero", "mean", "drop"


class AdvancedFeatureExtractor:
    """
    Advanced feature extractor for anomaly detection.
    
    Extracts wavelet-based features, time-frequency domain features,
    statistical process control features, and change point features.
    """
    
    def __init__(self, config: AdvancedFeatureConfig):
        self.config = config
        self.feature_names: List[str] = []
        self.scaler: Optional[StandardScaler] = None
        
    def extract_all_features(self, vl_series: np.ndarray, vo_series: np.ndarray) -> np.ndarray:
        """
        Extract all advanced features from voltage time series.
        
        Args:
            vl_series: Load voltage time series
            vo_series: Output voltage time series
            
        Returns:
            Feature vector
        """
        features = []
        feature_names = []
        
        # Extract features for both series
        for series_name, series in [("VL", vl_series), ("VO", vo_series)]:
            # Wavelet features
            if any([self.config.extract_wavelet_energy, 
                   self.config.extract_wavelet_entropy,
                   self.config.extract_wavelet_variance]):
                wavelet_features, wavelet_names = self._extract_wavelet_features(series)
                features.extend(wavelet_features)
                feature_names.extend([f"{series_name}_{name}" for name in wavelet_names])
            
            # STFT features
            if any([self.config.extract_spectral_centroid,
                   self.config.extract_spectral_bandwidth,
                   self.config.extract_spectral_rolloff,
                   self.config.extract_spectral_flux]):
                stft_features, stft_names = self._extract_stft_features(series)
                features.extend(stft_features)
                feature_names.extend([f"{series_name}_{name}" for name in stft_names])
            
            # SPC features
            if any([self.config.extract_control_limits,
                   self.config.extract_process_capability,
                   self.config.extract_run_tests]):
                spc_features, spc_names = self._extract_spc_features(series)
                features.extend(spc_features)
                feature_names.extend([f"{series_name}_{name}" for name in spc_names])
            
            # Change point features
            if any([self.config.extract_change_points,
                   self.config.extract_change_magnitude]):
                cp_features, cp_names = self._extract_change_point_features(series)
                features.extend(cp_features)
                feature_names.extend([f"{series_name}_{name}" for name in cp_names])
        
        # Cross-series features
        cross_features, cross_names = self._extract_cross_series_features(vl_series, vo_series)
        features.extend(cross_features)
        feature_names.extend(cross_names)
        
        # Store feature names
        self.feature_names = feature_names
        
        # Convert to numpy array and handle NaN values
        features = np.array(features, dtype=np.float64)
        features = self._handle_nan_values(features)
        
        return features
    
    def _extract_wavelet_features(self, series: np.ndarray) -> Tuple[List[float], List[str]]:
        """Extract wavelet-based features."""
        features = []
        feature_names = []
        
        try:
            # Perform wavelet decomposition
            coeffs = pywt.wavedec(series, self.config.wavelet_name, level=self.config.wavelet_levels)
            
            # Energy features
            if self.config.extract_wavelet_energy:
                for i, coeff in enumerate(coeffs):
                    energy = np.sum(coeff ** 2)
                    features.append(energy)
                    level_name = "approx" if i == 0 else f"detail_{i}"
                    feature_names.append(f"wavelet_energy_{level_name}")
                
                # Relative energy
                total_energy = sum(np.sum(coeff ** 2) for coeff in coeffs)
                if total_energy > 0:
                    for i, coeff in enumerate(coeffs):
                        rel_energy = np.sum(coeff ** 2) / total_energy
                        features.append(rel_energy)
                        level_name = "approx" if i == 0 else f"detail_{i}"
                        feature_names.append(f"wavelet_rel_energy_{level_name}")
            
            # Entropy features
            if self.config.extract_wavelet_entropy:
                for i, coeff in enumerate(coeffs):
                    # Shannon entropy
                    coeff_abs = np.abs(coeff)
                    if np.sum(coeff_abs) > 0:
                        coeff_norm = coeff_abs / np.sum(coeff_abs)
                        coeff_norm = coeff_norm[coeff_norm > 0]  # Remove zeros
                        entropy = -np.sum(coeff_norm * np.log2(coeff_norm))
                    else:
                        entropy = 0
                    
                    features.append(entropy)
                    level_name = "approx" if i == 0 else f"detail_{i}"
                    feature_names.append(f"wavelet_entropy_{level_name}")
            
            # Variance features
            if self.config.extract_wavelet_variance:
                for i, coeff in enumerate(coeffs):
                    variance = np.var(coeff)
                    features.append(variance)
                    level_name = "approx" if i == 0 else f"detail_{i}"
                    feature_names.append(f"wavelet_variance_{level_name}")
        
        except Exception as e:
            logger.warning(f"Wavelet feature extraction failed: {e}")
            # Return zero features
            n_levels = self.config.wavelet_levels + 1
            n_features = 0
            if self.config.extract_wavelet_energy:
                n_features += 2 * n_levels  # Energy + relative energy
            if self.config.extract_wavelet_entropy:
                n_features += n_levels
            if self.config.extract_wavelet_variance:
                n_features += n_levels
            
            features = [0.0] * n_features
            feature_names = [f"wavelet_feature_{i}" for i in range(n_features)]
        
        return features, feature_names
    
    def _extract_stft_features(self, series: np.ndarray) -> Tuple[List[float], List[str]]:
        """Extract Short-Time Fourier Transform features."""
        features = []
        feature_names = []
        
        try:
            # Compute STFT
            nperseg = min(self.config.stft_nperseg, len(series))
            noverlap = self.config.stft_noverlap or nperseg // 2
            
            f, t, Zxx = signal.stft(
                series,
                window=self.config.stft_window,
                nperseg=nperseg,
                noverlap=noverlap
            )
            
            # Magnitude spectrogram
            magnitude = np.abs(Zxx)
            power = magnitude ** 2
            
            # Spectral centroid
            if self.config.extract_spectral_centroid:
                centroids = []
                for i in range(magnitude.shape[1]):
                    spectrum = magnitude[:, i]
                    if np.sum(spectrum) > 0:
                        centroid = np.sum(f * spectrum) / np.sum(spectrum)
                    else:
                        centroid = 0
                    centroids.append(centroid)
                
                features.extend([
                    np.mean(centroids),
                    np.std(centroids),
                    np.min(centroids),
                    np.max(centroids)
                ])
                feature_names.extend([
                    "spectral_centroid_mean",
                    "spectral_centroid_std",
                    "spectral_centroid_min",
                    "spectral_centroid_max"
                ])
            
            # Spectral bandwidth
            if self.config.extract_spectral_bandwidth:
                bandwidths = []
                for i in range(magnitude.shape[1]):
                    spectrum = magnitude[:, i]
                    if np.sum(spectrum) > 0:
                        centroid = np.sum(f * spectrum) / np.sum(spectrum)
                        bandwidth = np.sqrt(np.sum(((f - centroid) ** 2) * spectrum) / np.sum(spectrum))
                    else:
                        bandwidth = 0
                    bandwidths.append(bandwidth)
                
                features.extend([
                    np.mean(bandwidths),
                    np.std(bandwidths),
                    np.min(bandwidths),
                    np.max(bandwidths)
                ])
                feature_names.extend([
                    "spectral_bandwidth_mean",
                    "spectral_bandwidth_std",
                    "spectral_bandwidth_min",
                    "spectral_bandwidth_max"
                ])
            
            # Spectral rolloff
            if self.config.extract_spectral_rolloff:
                rolloffs = []
                rolloff_threshold = 0.85  # 85% of energy
                
                for i in range(magnitude.shape[1]):
                    spectrum = magnitude[:, i]
                    if np.sum(spectrum) > 0:
                        cumsum = np.cumsum(spectrum)
                        total = cumsum[-1]
                        rolloff_idx = np.where(cumsum >= rolloff_threshold * total)[0]
                        if len(rolloff_idx) > 0:
                            rolloff = f[rolloff_idx[0]]
                        else:
                            rolloff = f[-1]
                    else:
                        rolloff = 0
                    rolloffs.append(rolloff)
                
                features.extend([
                    np.mean(rolloffs),
                    np.std(rolloffs),
                    np.min(rolloffs),
                    np.max(rolloffs)
                ])
                feature_names.extend([
                    "spectral_rolloff_mean",
                    "spectral_rolloff_std",
                    "spectral_rolloff_min",
                    "spectral_rolloff_max"
                ])
            
            # Spectral flux
            if self.config.extract_spectral_flux:
                flux = []
                for i in range(1, magnitude.shape[1]):
                    diff = magnitude[:, i] - magnitude[:, i-1]
                    flux_val = np.sum(diff ** 2)
                    flux.append(flux_val)
                
                if flux:
                    features.extend([
                        np.mean(flux),
                        np.std(flux),
                        np.min(flux),
                        np.max(flux)
                    ])
                else:
                    features.extend([0, 0, 0, 0])
                
                feature_names.extend([
                    "spectral_flux_mean",
                    "spectral_flux_std",
                    "spectral_flux_min",
                    "spectral_flux_max"
                ])
        
        except Exception as e:
            logger.warning(f"STFT feature extraction failed: {e}")
            # Return zero features
            n_features = 0
            if self.config.extract_spectral_centroid:
                n_features += 4
            if self.config.extract_spectral_bandwidth:
                n_features += 4
            if self.config.extract_spectral_rolloff:
                n_features += 4
            if self.config.extract_spectral_flux:
                n_features += 4
            
            features = [0.0] * n_features
            feature_names = [f"stft_feature_{i}" for i in range(n_features)]
        
        return features, feature_names
    
    def _extract_spc_features(self, series: np.ndarray) -> Tuple[List[float], List[str]]:
        """Extract Statistical Process Control features."""
        features = []
        feature_names = []
        
        try:
            # Control limits
            if self.config.extract_control_limits:
                mean_val = np.mean(series)
                std_val = np.std(series)
                
                # Control limits (3-sigma)
                ucl = mean_val + 3 * std_val
                lcl = mean_val - 3 * std_val
                
                # Points outside control limits
                out_of_control = np.sum((series > ucl) | (series < lcl))
                out_of_control_ratio = out_of_control / len(series)
                
                features.extend([
                    ucl,
                    lcl,
                    out_of_control,
                    out_of_control_ratio
                ])
                feature_names.extend([
                    "spc_ucl",
                    "spc_lcl",
                    "spc_out_of_control_count",
                    "spc_out_of_control_ratio"
                ])
            
            # Process capability
            if self.config.extract_process_capability:
                # Assuming specification limits (can be adjusted)
                usl = np.mean(series) + 6 * np.std(series)  # Upper spec limit
                lsl = np.mean(series) - 6 * np.std(series)  # Lower spec limit
                
                # Cp and Cpk indices
                if std_val > 0:
                    cp = (usl - lsl) / (6 * std_val)
                    cpu = (usl - mean_val) / (3 * std_val)
                    cpl = (mean_val - lsl) / (3 * std_val)
                    cpk = min(cpu, cpl)
                else:
                    cp = cpk = 0
                
                features.extend([cp, cpk])
                feature_names.extend(["spc_cp", "spc_cpk"])
            
            # Run tests
            if self.config.extract_run_tests:
                # Test 1: One point beyond 3-sigma
                test1_violations = np.sum(np.abs(series - mean_val) > 3 * std_val)
                
                # Test 2: Nine consecutive points on same side of center line
                centered = series - mean_val
                signs = np.sign(centered)
                test2_violations = 0
                
                run_length = 1
                for i in range(1, len(signs)):
                    if signs[i] == signs[i-1]:
                        run_length += 1
                        if run_length >= 9:
                            test2_violations += 1
                    else:
                        run_length = 1
                
                # Test 3: Six consecutive increasing or decreasing points
                test3_violations = 0
                trend_length = 1
                
                for i in range(1, len(series)):
                    if series[i] > series[i-1]:
                        if trend_length > 0:
                            trend_length += 1
                        else:
                            trend_length = 1
                    elif series[i] < series[i-1]:
                        if trend_length < 0:
                            trend_length -= 1
                        else:
                            trend_length = -1
                    else:
                        trend_length = 1
                    
                    if abs(trend_length) >= 6:
                        test3_violations += 1
                
                features.extend([
                    test1_violations,
                    test2_violations,
                    test3_violations
                ])
                feature_names.extend([
                    "spc_test1_violations",
                    "spc_test2_violations",
                    "spc_test3_violations"
                ])
        
        except Exception as e:
            logger.warning(f"SPC feature extraction failed: {e}")
            # Return zero features
            n_features = 0
            if self.config.extract_control_limits:
                n_features += 4
            if self.config.extract_process_capability:
                n_features += 2
            if self.config.extract_run_tests:
                n_features += 3
            
            features = [0.0] * n_features
            feature_names = [f"spc_feature_{i}" for i in range(n_features)]
        
        return features, feature_names
    
    def _extract_change_point_features(self, series: np.ndarray) -> Tuple[List[float], List[str]]:
        """Extract change point detection features."""
        features = []
        feature_names = []
        
        try:
            if self.config.change_point_method == "cusum":
                change_points, magnitudes = self._cusum_change_detection(series)
            elif self.config.change_point_method == "window":
                change_points, magnitudes = self._window_change_detection(series)
            else:
                # Fallback to simple method
                change_points, magnitudes = self._simple_change_detection(series)
            
            # Change point features
            if self.config.extract_change_points:
                n_change_points = len(change_points)
                change_point_density = n_change_points / len(series)
                
                # Time between change points
                if len(change_points) > 1:
                    intervals = np.diff(change_points)
                    mean_interval = np.mean(intervals)
                    std_interval = np.std(intervals)
                else:
                    mean_interval = len(series)
                    std_interval = 0
                
                features.extend([
                    n_change_points,
                    change_point_density,
                    mean_interval,
                    std_interval
                ])
                feature_names.extend([
                    "change_points_count",
                    "change_points_density",
                    "change_points_mean_interval",
                    "change_points_std_interval"
                ])
            
            # Change magnitude features
            if self.config.extract_change_magnitude:
                if magnitudes:
                    features.extend([
                        np.mean(magnitudes),
                        np.std(magnitudes),
                        np.min(magnitudes),
                        np.max(magnitudes),
                        np.sum(magnitudes)
                    ])
                else:
                    features.extend([0, 0, 0, 0, 0])
                
                feature_names.extend([
                    "change_magnitude_mean",
                    "change_magnitude_std",
                    "change_magnitude_min",
                    "change_magnitude_max",
                    "change_magnitude_sum"
                ])
        
        except Exception as e:
            logger.warning(f"Change point feature extraction failed: {e}")
            # Return zero features
            n_features = 0
            if self.config.extract_change_points:
                n_features += 4
            if self.config.extract_change_magnitude:
                n_features += 5
            
            features = [0.0] * n_features
            feature_names = [f"change_point_feature_{i}" for i in range(n_features)]
        
        return features, feature_names
    
    def _cusum_change_detection(self, series: np.ndarray) -> Tuple[List[int], List[float]]:
        """CUSUM-based change point detection."""
        # Standardize series
        series_std = (series - np.mean(series)) / (np.std(series) + 1e-8)
        
        # CUSUM parameters
        h = self.config.change_point_threshold  # Decision threshold
        k = 0.5  # Reference value
        
        # Initialize CUSUM statistics
        s_pos = np.zeros(len(series_std))
        s_neg = np.zeros(len(series_std))
        
        change_points = []
        magnitudes = []
        
        for i in range(1, len(series_std)):
            s_pos[i] = max(0, s_pos[i-1] + series_std[i] - k)
            s_neg[i] = max(0, s_neg[i-1] - series_std[i] - k)
            
            if s_pos[i] > h or s_neg[i] > h:
                change_points.append(i)
                magnitudes.append(max(s_pos[i], s_neg[i]))
                # Reset CUSUM
                s_pos[i] = 0
                s_neg[i] = 0
        
        return change_points, magnitudes
    
    def _window_change_detection(self, series: np.ndarray) -> Tuple[List[int], List[float]]:
        """Window-based change point detection."""
        window_size = min(self.config.spc_window_size, len(series) // 4)
        change_points = []
        magnitudes = []
        
        for i in range(window_size, len(series) - window_size):
            # Compare statistics of left and right windows
            left_window = series[i-window_size:i]
            right_window = series[i:i+window_size]
            
            # T-test for mean difference
            try:
                t_stat, p_value = stats.ttest_ind(left_window, right_window)
                if p_value < 0.05:  # Significant difference
                    change_points.append(i)
                    magnitudes.append(abs(t_stat))
            except:
                continue
        
        return change_points, magnitudes
    
    def _simple_change_detection(self, series: np.ndarray) -> Tuple[List[int], List[float]]:
        """Simple change point detection based on derivatives."""
        # Compute first and second derivatives
        first_diff = np.diff(series)
        second_diff = np.diff(first_diff)
        
        # Find points with large second derivative
        threshold = self.config.change_point_threshold * np.std(second_diff)
        change_indices = np.where(np.abs(second_diff) > threshold)[0]
        
        change_points = (change_indices + 1).tolist()  # Adjust for diff offset
        magnitudes = np.abs(second_diff[change_indices]).tolist()
        
        return change_points, magnitudes
    
    def _extract_cross_series_features(self, vl_series: np.ndarray, vo_series: np.ndarray) -> Tuple[List[float], List[str]]:
        """Extract features that compare VL and VO series."""
        features = []
        feature_names = []
        
        try:
            # Cross-correlation
            correlation = np.corrcoef(vl_series, vo_series)[0, 1]
            if np.isnan(correlation):
                correlation = 0
            
            features.append(correlation)
            feature_names.append("cross_correlation")
            
            # Phase relationship (using cross-correlation)
            cross_corr = np.correlate(vl_series - np.mean(vl_series), 
                                    vo_series - np.mean(vo_series), mode='full')
            max_corr_idx = np.argmax(np.abs(cross_corr))
            phase_lag = max_corr_idx - len(vo_series) + 1
            
            features.extend([phase_lag, np.max(np.abs(cross_corr))])
            feature_names.extend(["phase_lag", "max_cross_correlation"])
            
            # Ratio statistics
            ratio = vl_series / (vo_series + 1e-8)  # Avoid division by zero
            features.extend([
                np.mean(ratio),
                np.std(ratio),
                np.min(ratio),
                np.max(ratio)
            ])
            feature_names.extend([
                "vl_vo_ratio_mean",
                "vl_vo_ratio_std",
                "vl_vo_ratio_min",
                "vl_vo_ratio_max"
            ])
            
            # Difference statistics
            diff = vl_series - vo_series
            features.extend([
                np.mean(diff),
                np.std(diff),
                np.min(diff),
                np.max(diff)
            ])
            feature_names.extend([
                "vl_vo_diff_mean",
                "vl_vo_diff_std",
                "vl_vo_diff_min",
                "vl_vo_diff_max"
            ])
        
        except Exception as e:
            logger.warning(f"Cross-series feature extraction failed: {e}")
            features = [0.0] * 11
            feature_names = [f"cross_feature_{i}" for i in range(11)]
        
        return features, feature_names
    
    def _handle_nan_values(self, features: np.ndarray) -> np.ndarray:
        """Handle NaN values in features."""
        if self.config.handle_nan == "zero":
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        elif self.config.handle_nan == "mean":
            # Replace NaN with mean of non-NaN values
            for i in range(len(features)):
                if np.isnan(features[i]) or np.isinf(features[i]):
                    # Use 0 as fallback if all values are NaN
                    features[i] = 0.0
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get names of extracted features."""
        return self.feature_names.copy()
    
    def fit_scaler(self, feature_matrix: np.ndarray) -> 'AdvancedFeatureExtractor':
        """Fit feature scaler on training data."""
        if self.config.normalize_features:
            self.scaler = StandardScaler()
            self.scaler.fit(feature_matrix)
        return self
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler."""
        if self.scaler is not None:
            return self.scaler.transform(features.reshape(1, -1)).flatten()
        return features


def create_advanced_feature_extractor(**kwargs) -> AdvancedFeatureExtractor:
    """
    Factory function to create an advanced feature extractor.
    
    Args:
        **kwargs: Configuration parameters
        
    Returns:
        Configured advanced feature extractor
    """
    config = AdvancedFeatureConfig(**kwargs)
    return AdvancedFeatureExtractor(config)