"""Property-based tests for RULFeatureAnalyzer."""

import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings

from nasa_pcoe_eda.analysis.rul_features import RULFeatureAnalyzer


class TestRULFeatureAnalyzerProperties:
    """Property-based tests for RULFeatureAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = RULFeatureAnalyzer()

    # Feature: nasa-pcoe-eda, Property 20: 劣化率の単調性
    @given(st.data())
    @settings(max_examples=100)
    def test_degradation_rate_monotonicity(self, data):
        """
        任意の劣化トレンドを示す特徴量に対して、計算される変化率は
        時間の経過とともに一貫した方向性（増加または減少）を示す
        
        Property 20: Degradation Rate Monotonicity
        For any feature showing degradation trends, the computed change rate
        should show consistent directionality (increasing or decreasing) over time.
        """
        # Generate time series data with monotonic trend
        num_points = data.draw(st.integers(min_value=10, max_value=100))
        
        # Generate time values
        time_values = np.arange(num_points)
        
        # Generate monotonic trend (either increasing or decreasing)
        trend_direction = data.draw(st.sampled_from(['increasing', 'decreasing']))
        base_slope = data.draw(st.floats(min_value=0.1, max_value=2.0))
        
        if trend_direction == 'increasing':
            slope = base_slope
        else:
            slope = -base_slope
            
        # Generate feature values with monotonic trend + small noise
        noise_level = data.draw(st.floats(min_value=0.01, max_value=0.5))
        noise = np.random.normal(0, noise_level, num_points)
        
        initial_value = data.draw(st.floats(min_value=10, max_value=100))
        feature_values = initial_value + slope * time_values + noise
        
        # Create DataFrame
        df = pd.DataFrame({
            'time': time_values,
            'feature': feature_values
        })
        
        # Compute degradation rate
        rates = self.analyzer.compute_degradation_rates(df, ['feature'])
        
        if 'feature' in rates:
            computed_rate = rates['feature']
            
            # The computed rate should have the same sign as the original slope
            # (allowing for some numerical tolerance due to noise)
            if abs(slope) > 0.1:  # Only check for significant slopes
                if slope > 0:
                    assert computed_rate > -0.1, f"Expected positive rate for increasing trend, got {computed_rate}"
                else:
                    assert computed_rate < 0.1, f"Expected negative rate for decreasing trend, got {computed_rate}"

    # Feature: nasa-pcoe-eda, Property 21: RUL特徴量ランキングの順序性
    @given(st.data())
    @settings(max_examples=100)
    def test_rul_feature_ranking_order(self, data):
        """
        任意のデータセットに対して、RUL予測のための特徴量ランキングは、
        相関係数の絶対値の降順である
        
        Property 21: RUL Feature Ranking Order
        For any dataset, the feature ranking for RUL prediction should be
        in descending order of absolute correlation coefficients.
        """
        # Generate dataset with features having different correlations with RUL
        num_features = data.draw(st.integers(min_value=2, max_value=8))
        num_points = data.draw(st.integers(min_value=20, max_value=100))
        
        # Generate RUL values (decreasing over time)
        rul_values = np.linspace(100, 10, num_points)
        
        df_data = {'rul': rul_values}
        
        # Generate features with different correlation strengths
        for i in range(num_features):
            # Generate correlation strength
            correlation_strength = data.draw(st.floats(min_value=0.1, max_value=0.9))
            
            # Randomly choose positive or negative correlation
            sign = data.draw(st.sampled_from([1, -1]))
            
            # Generate feature values correlated with RUL
            noise_level = data.draw(st.floats(min_value=0.1, max_value=0.5))
            noise = np.random.normal(0, noise_level, num_points)
            
            # Create feature with desired correlation
            feature_values = sign * correlation_strength * rul_values + noise
            df_data[f'feature_{i}'] = feature_values
        
        df = pd.DataFrame(df_data)
        
        # Rank features for RUL
        ranking = self.analyzer.rank_features_for_rul(df, 'rul')
        
        if len(ranking) > 1:
            # Check that ranking is in descending order of absolute correlation
            for i in range(len(ranking) - 1):
                current_abs_corr = abs(ranking[i][1])
                next_abs_corr = abs(ranking[i + 1][1])
                
                assert current_abs_corr >= next_abs_corr, (
                    f"Ranking not in descending order: "
                    f"{ranking[i][0]}({current_abs_corr:.3f}) should be >= "
                    f"{ranking[i+1][0]}({next_abs_corr:.3f})"
                )
        
        # Also check that all correlations are valid (between -1 and 1)
        for feature_name, correlation in ranking:
            assert -1.0 <= correlation <= 1.0, (
                f"Invalid correlation {correlation} for feature {feature_name}"
            )