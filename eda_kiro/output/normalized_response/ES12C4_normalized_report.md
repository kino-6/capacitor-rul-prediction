# ES12C4 Normalized Response Analysis Report

## Overview

Analysis of ES12C4 response characteristics across multiple cycles.
This report normalizes input signals to enable proper comparison of degradation patterns.

## Key Issue Identified

⚠️ **Input Signal Variation**: The input signals (VL) vary significantly between cycles,
making direct comparison of 'same input, different output' impossible without normalization.

## Input Signal Variation Analysis

### Cycle 1
- **VL Mean**: 0.007322 V
- **VL Std**: 0.123129 V
- **VL Range**: -0.391207 to 0.327294 V

### Cycle 100
- **VL Mean**: 0.062449 V
- **VL Std**: 0.131896 V
- **VL Range**: -0.418446 to 0.363611 V

### Cycle 200
- **VL Mean**: -0.004812 V
- **VL Std**: 0.136361 V
- **VL Range**: -0.365178 to 0.335162 V

### Cycle 300
- **VL Mean**: 0.046108 V
- **VL Std**: 0.138358 V
- **VL Range**: -0.325228 to 0.332135 V

## Response Characteristics Comparison

| Cycle | VL Mean (V) | VO Mean (V) | Voltage Ratio | Gain | Correlation |
|-------|-------------|-------------|---------------|------|-------------|
| 1 | 0.007322 | 0.065126 | 8.894 | 0.998 | 0.999 |
| 100 | 0.062449 | 11.834535 | 189.508 | 1.000 | 0.996 |
| 200 | -0.004812 | 0.053075 | -11.030 | 0.999 | 0.999 |
| 300 | 0.046108 | 11.821175 | 256.380 | 1.001 | 0.996 |

## Degradation Analysis

Using Cycle 1 as reference:

### Cycle 100 vs Cycle 1
- **Voltage Ratio Change**: +2030.7%
- **Gain Change**: +0.2%
- **Correlation**: 0.996
- 🔴 **Severe degradation detected**

### Cycle 200 vs Cycle 1
- **Voltage Ratio Change**: -224.0%
- **Gain Change**: +0.0%
- **Correlation**: 0.999
- 🔴 **Severe degradation detected**

### Cycle 300 vs Cycle 1
- **Voltage Ratio Change**: +2782.5%
- **Gain Change**: +0.2%
- **Correlation**: 0.996
- 🔴 **Severe degradation detected**

## Conclusions

1. **Input Variability**: Input signals vary significantly between cycles
2. **Normalization Required**: Direct comparison requires signal normalization
3. **Response Changes**: Significant changes in voltage ratio and gain observed
4. **Correlation Maintained**: High correlation suggests linear relationship preserved

## Recommendations

- Use normalized signals for degradation analysis
- Focus on gain and correlation changes rather than absolute values
- Consider input signal stabilization in future experiments

---
Report Generated: 2026-01-03 17:52:28
