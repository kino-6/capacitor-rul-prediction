# ES12C4 Similar Input Cycles Analysis Report

## Overview

Analysis of ES12C4 to identify cycles with similar input patterns (VL)
and analyze degradation within these similar input groups.

## Data Summary

- **Total Valid Cycles**: 12
- **Cycle Range**: 39 to 50
- **Waveform Length**: 3000 points per cycle

## Clustering Results

- **Number of Clusters**: 3
- **Similarity Threshold**: 0.3 (distance)
- **Minimum Cluster Size**: 3 cycles

### Cluster Details

| Cluster ID | Size | Cycles | Time Span |
|------------|------|--------|----------|
| 1 | 25 | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25 | 24 |
| 3 | 13 | 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38 | 12 |
| 2 | 12 | 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50 | 11 |

## Degradation Analysis Results

### Cluster 1

- **Cycles**: 25 cycles over 24 cycle span
- **Total Degradation**: 135.8%
- **Degradation Rate**: 5.656% per cycle
- 🔴 **Severe degradation detected**

### Cluster 3

- **Cycles**: 13 cycles over 12 cycle span
- **Total Degradation**: 53.4%
- **Degradation Rate**: 4.454% per cycle
- 🔴 **Severe degradation detected**

### Cluster 2

- **Cycles**: 12 cycles over 11 cycle span
- **Total Degradation**: 102.4%
- **Degradation Rate**: 9.309% per cycle
- 🔴 **Severe degradation detected**

## Conclusions

1. **Similar Input Identification**: Successfully identified clusters of cycles with similar VL patterns
2. **Degradation Quantification**: Measured output changes within similar input groups
3. **Temporal Patterns**: Observed degradation progression over time
4. **Cluster Variability**: Different clusters show varying degradation rates

---
Report Generated: 2026-01-03 18:22:50
