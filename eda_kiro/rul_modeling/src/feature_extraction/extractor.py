"""
Cycle-level feature extraction for capacitor degradation analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy import stats


class CycleFeatureExtractor:
    """Extract features from a single cycle of VL/VO data."""
    
    def __init__(self):
        """Initialize the feature extractor."""
        pass
    
    def extract_basic_stats(self, vl: np.ndarray, vo: np.ndarray) -> Dict[str, float]:
        """
        Extract basic statistical features from VL and VO.
        
        Args:
            vl: VL time series data (input voltage)
            vo: VO time series data (output voltage)
        
        Returns:
            Dictionary of basic statistical features
        """
        features = {}
        
        # VL (Input) statistics
        features['vl_mean'] = float(np.mean(vl))
        features['vl_std'] = float(np.std(vl))
        features['vl_min'] = float(np.min(vl))
        features['vl_max'] = float(np.max(vl))
        features['vl_range'] = features['vl_max'] - features['vl_min']
        features['vl_median'] = float(np.median(vl))
        features['vl_q25'] = float(np.percentile(vl, 25))
        features['vl_q75'] = float(np.percentile(vl, 75))
        
        # VO (Output) statistics
        features['vo_mean'] = float(np.mean(vo))
        features['vo_std'] = float(np.std(vo))
        features['vo_min'] = float(np.min(vo))
        features['vo_max'] = float(np.max(vo))
        features['vo_range'] = features['vo_max'] - features['vo_min']
        features['vo_median'] = float(np.median(vo))
        features['vo_q25'] = float(np.percentile(vo, 25))
        features['vo_q75'] = float(np.percentile(vo, 75))
        
        return features
    
    def extract_degradation_indicators(self, vl: np.ndarray, vo: np.ndarray) -> Dict[str, float]:
        """
        Extract degradation indicator features.
        
        Args:
            vl: VL time series data
            vo: VO time series data
        
        Returns:
            Dictionary of degradation indicators
        """
        features = {}
        
        vl_mean = np.mean(vl)
        vo_mean = np.mean(vo)
        vl_std = np.std(vl)
        vo_std = np.std(vo)
        vl_range = np.max(vl) - np.min(vl)
        vo_range = np.max(vo) - np.min(vo)
        
        # Voltage ratio (key degradation indicator from EDA)
        features['voltage_ratio'] = float(vo_mean / vl_mean) if vl_mean != 0 else 0.0
        
        # Voltage ratio std (variation in the ratio)
        voltage_ratio_series = vo / vl if np.all(vl != 0) else np.zeros_like(vo)
        features['voltage_ratio_std'] = float(np.std(voltage_ratio_series))
        
        # Response efficiency
        features['response_efficiency'] = float(vo_range / vl_range) if vl_range != 0 else 0.0
        
        # Signal attenuation
        features['signal_attenuation'] = float(1 - (vo_std / vl_std)) if vl_std != 0 else 0.0
        
        return features
    
    def extract_time_series_features(self, vl: np.ndarray, vo: np.ndarray) -> Dict[str, float]:
        """
        Extract time series features (trends, variability).
        
        Args:
            vl: VL time series data
            vo: VO time series data
        
        Returns:
            Dictionary of time series features
        """
        features = {}
        
        # Linear trend (slope of linear regression)
        x = np.arange(len(vl))
        
        # VL trend
        if len(vl) > 1:
            slope_vl, _, _, _, _ = stats.linregress(x, vl)
            features['vl_trend'] = float(slope_vl)
        else:
            features['vl_trend'] = 0.0
        
        # VO trend
        if len(vo) > 1:
            slope_vo, _, _, _, _ = stats.linregress(x, vo)
            features['vo_trend'] = float(slope_vo)
        else:
            features['vo_trend'] = 0.0
        
        # Coefficient of variation (CV = std / mean)
        vl_mean = np.mean(vl)
        vo_mean = np.mean(vo)
        
        features['vl_cv'] = float(np.std(vl) / vl_mean) if vl_mean != 0 else 0.0
        features['vo_cv'] = float(np.std(vo) / vo_mean) if vo_mean != 0 else 0.0
        
        return features
    
    def extract_cycle_info(self, cycle_num: int, total_cycles: int = 200) -> Dict[str, float]:
        """
        Extract cycle-related information.
        
        Args:
            cycle_num: Current cycle number
            total_cycles: Total number of cycles (default: 200)
        
        Returns:
            Dictionary of cycle information
        """
        features = {}
        
        features['cycle_number'] = float(cycle_num)
        features['cycle_normalized'] = float(cycle_num / total_cycles)
        
        return features
    
    def extract_historical_features(
        self, 
        history_df: Optional[pd.DataFrame], 
        window_size: int = 5
    ) -> Dict[str, float]:
        """
        Extract historical features from past cycles.
        
        Args:
            history_df: DataFrame containing features from past cycles
            window_size: Number of past cycles to consider
        
        Returns:
            Dictionary of historical features
        """
        features = {}
        
        if history_df is None or len(history_df) < window_size:
            # Not enough history, return zeros
            features['voltage_ratio_mean_last_5'] = 0.0
            features['voltage_ratio_std_last_5'] = 0.0
            features['voltage_ratio_trend_last_10'] = 0.0
            features['degradation_rate'] = 0.0
            return features
        
        # Past N cycles statistics
        recent = history_df.tail(window_size)
        features['voltage_ratio_mean_last_5'] = float(recent['voltage_ratio'].mean())
        features['voltage_ratio_std_last_5'] = float(recent['voltage_ratio'].std())
        
        # Trend over past 10 cycles
        if len(history_df) >= 10:
            recent_10 = history_df.tail(10)
            x = np.arange(len(recent_10))
            slope, _, _, _, _ = stats.linregress(x, recent_10['voltage_ratio'].values)
            features['voltage_ratio_trend_last_10'] = float(slope)
        else:
            features['voltage_ratio_trend_last_10'] = 0.0
        
        # Degradation rate
        if len(history_df) > 0:
            initial_ratio = history_df.iloc[0]['voltage_ratio']
            current_ratio = history_df.iloc[-1]['voltage_ratio']
            cycle_num = history_df.iloc[-1]['cycle_number']
            features['degradation_rate'] = float((current_ratio - initial_ratio) / cycle_num) if cycle_num != 0 else 0.0
        else:
            features['degradation_rate'] = 0.0
        
        return features
    
    def extract_all_features(
        self,
        vl: np.ndarray,
        vo: np.ndarray,
        cycle_num: int,
        history_df: Optional[pd.DataFrame] = None,
        total_cycles: int = 200
    ) -> Dict[str, float]:
        """
        Extract all features from a single cycle.
        
        Args:
            vl: VL time series data
            vo: VO time series data
            cycle_num: Current cycle number
            history_df: DataFrame containing features from past cycles
            total_cycles: Total number of cycles
        
        Returns:
            Dictionary containing all features
        """
        features = {}
        
        # Extract all feature groups
        features.update(self.extract_basic_stats(vl, vo))
        features.update(self.extract_degradation_indicators(vl, vo))
        features.update(self.extract_time_series_features(vl, vo))
        features.update(self.extract_cycle_info(cycle_num, total_cycles))
        features.update(self.extract_historical_features(history_df))
        
        return features
