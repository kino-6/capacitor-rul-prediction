# ES12C4 Similar Input Response Analysis Report

## Overview

Detailed analysis of response changes for cycles with similar input patterns.
By plotting similar input cycles on the same scale, we can clearly observe
how output responses degrade over time while input patterns remain similar.

## Analysis Approach

1. **Input Similarity Clustering**: Group cycles with similar VL patterns
2. **Same-Scale Visualization**: Plot similar inputs on identical scales
3. **Response Comparison**: Compare VO responses within each cluster
4. **Degradation Quantification**: Measure response changes over time

## Cluster Analysis Results

### Cluster 1

**Cycles**: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25
**Time Span**: 24 cycles
**Cluster Size**: 25 cycles

#### Input Similarity
- **Early vs Late Input Correlation**: 0.879
- **Input Pattern Consistency**: Moderate

#### Response Changes
- **Voltage Ratio Change**: -135.8%
- **Output Amplitude Change**: +30.9%
- **Degradation Severity**: 🔴 **Severe** (>100% change)

#### Cycle-by-Cycle Analysis

| Cycle | VL Mean | VO Mean | Voltage Ratio | Change from First |
|-------|---------|---------|---------------|-------------------|
| 1 | -0.0026 | 0.0551 | -21.262 | -0.0% |
| 2 | 1.8742 | 1.9144 | 1.021 | -104.8% |
| 3 | 1.8152 | 2.3132 | 1.274 | -106.0% |
| 4 | 1.7614 | 2.6916 | 1.528 | -107.2% |
| 5 | 1.7128 | 3.0554 | 1.784 | -108.4% |
| 6 | 1.6667 | 3.4044 | 2.043 | -109.6% |
| 7 | 1.6246 | 3.7423 | 2.303 | -110.8% |
| 8 | 1.5838 | 4.0677 | 2.568 | -112.1% |
| 9 | 1.5454 | 4.3831 | 2.836 | -113.3% |
| 10 | 1.5093 | 4.6890 | 3.107 | -114.6% |
| 11 | 1.4733 | 4.9845 | 3.383 | -115.9% |
| 12 | 1.4401 | 5.2726 | 3.661 | -117.2% |
| 13 | 1.4073 | 5.5514 | 3.945 | -118.6% |
| 14 | 1.3753 | 5.8220 | 4.233 | -119.9% |
| 15 | 1.3444 | 6.0850 | 4.526 | -121.3% |
| 16 | 1.3159 | 6.3419 | 4.819 | -122.7% |
| 17 | 1.2895 | 6.5932 | 5.113 | -124.0% |
| 18 | 1.2643 | 6.8381 | 5.409 | -125.4% |
| 19 | 1.2408 | 7.0775 | 5.704 | -126.8% |
| 20 | 1.2148 | 7.3077 | 6.015 | -128.3% |
| 21 | 1.1925 | 7.5343 | 6.318 | -129.7% |
| 22 | 1.1686 | 7.7532 | 6.635 | -131.2% |
| 23 | 1.1458 | 7.9670 | 6.953 | -132.7% |
| 24 | 1.1252 | 8.1769 | 7.267 | -134.2% |
| 25 | 1.1021 | 8.3787 | 7.602 | -135.8% |

### Cluster 3

**Cycles**: 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38
**Time Span**: 12 cycles
**Cluster Size**: 13 cycles

#### Input Similarity
- **Early vs Late Input Correlation**: 0.886
- **Input Pattern Consistency**: Moderate

#### Response Changes
- **Voltage Ratio Change**: +53.4%
- **Output Amplitude Change**: -7.3%
- **Degradation Severity**: 🟡 **Moderate** (50-100% change)

#### Cycle-by-Cycle Analysis

| Cycle | VL Mean | VO Mean | Voltage Ratio | Change from First |
|-------|---------|---------|---------------|-------------------|
| 26 | 1.0826 | 8.5781 | 7.924 | +0.0% |
| 27 | 1.0625 | 8.7716 | 8.256 | +4.2% |
| 28 | 1.0424 | 8.9597 | 8.595 | +8.5% |
| 29 | 1.0248 | 9.1455 | 8.924 | +12.6% |
| 30 | 1.0048 | 9.3239 | 9.279 | +17.1% |
| 31 | 0.9879 | 9.5004 | 9.617 | +21.4% |
| 32 | 0.9703 | 9.6718 | 9.968 | +25.8% |
| 33 | 0.9525 | 9.8384 | 10.329 | +30.4% |
| 34 | 0.9372 | 10.0031 | 10.673 | +34.7% |
| 35 | 0.9194 | 10.1610 | 11.051 | +39.5% |
| 36 | 0.9051 | 10.3182 | 11.400 | +43.9% |
| 37 | 0.8893 | 10.4699 | 11.774 | +48.6% |
| 38 | 0.8732 | 10.6174 | 12.159 | +53.4% |

### Cluster 2

**Cycles**: 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50
**Time Span**: 11 cycles
**Cluster Size**: 12 cycles

#### Input Similarity
- **Early vs Late Input Correlation**: 0.943
- **Input Pattern Consistency**: High

#### Response Changes
- **Voltage Ratio Change**: +102.4%
- **Output Amplitude Change**: -21.3%
- **Degradation Severity**: 🔴 **Severe** (>100% change)

#### Cycle-by-Cycle Analysis

| Cycle | VL Mean | VO Mean | Voltage Ratio | Change from First |
|-------|---------|---------|---------------|-------------------|
| 39 | 0.8595 | 10.7634 | 12.522 | +0.0% |
| 40 | 0.8435 | 10.9035 | 12.926 | +3.2% |
| 41 | 0.8303 | 11.0426 | 13.299 | +6.2% |
| 42 | 0.8156 | 11.1765 | 13.703 | +9.4% |
| 43 | 0.8000 | 11.3060 | 14.132 | +12.9% |
| 44 | 0.7847 | 11.4327 | 14.569 | +16.3% |
| 45 | 0.7699 | 11.5564 | 15.011 | +19.9% |
| 46 | 0.7554 | 11.6774 | 15.459 | +23.5% |
| 47 | 0.7398 | 11.7942 | 15.942 | +27.3% |
| 48 | 0.6715 | 11.8577 | 17.658 | +41.0% |
| 49 | 0.5587 | 11.8592 | 21.226 | +69.5% |
| 50 | 0.4678 | 11.8568 | 25.346 | +102.4% |

## Key Findings

1. **Average Degradation**: 97.2% voltage ratio change across clusters
2. **Maximum Degradation**: 135.8% in worst-affected cluster
3. **Input Consistency**: Similar input patterns maintained across time
4. **Response Variability**: Clear degradation in output responses despite similar inputs
5. **Temporal Patterns**: Later cycles show progressively worse responses

## Visualization Benefits

- **Same-Scale Plotting**: Enables direct visual comparison of similar inputs
- **Response Isolation**: Clearly separates input consistency from output degradation
- **Temporal Tracking**: Shows progression of degradation over time
- **Quantitative Analysis**: Provides precise degradation measurements

## Conclusions

The same-scale visualization approach successfully demonstrates:

1. **Input Pattern Consistency**: VL patterns remain highly similar within clusters
2. **Output Response Degradation**: VO responses show clear degradation over time
3. **Measurable Changes**: Quantifiable degradation in voltage ratios and amplitudes
4. **Temporal Progression**: Systematic degradation pattern across operational cycles

---
Report Generated: 2026-01-03 18:30:06
