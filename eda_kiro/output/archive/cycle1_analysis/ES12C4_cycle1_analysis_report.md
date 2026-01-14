# ES12C4 Cycle 1 Analysis Report

## Overview

Cycle 1 is the most dynamic cycle in the ES12C4 dataset with a dynamism score of 30.4. This analysis identifies cycles similar to Cycle 1 for degradation comparison.

## Cycle 1 Characteristics

- **VL Mean**: -0.0026V
- **VL Std**: 0.1236V
- **VL Range**: 0.6168V
- **VO Mean**: 0.0551V
- **VO Std**: 0.1236V
- **VO Range**: 0.6165V

![Cycle 1 Waveform](ES12C4_cycle1_waveform.png)

## Similar Cycles Found

Total cycles with similarity ≥ 0.5: 185

### Top 10 Most Similar Cycles

| Rank | Cycle | Similarity | Correlation | Time Gap | Degradation |
|------|-------|------------|-------------|----------|-------------|
| 1 | 4 | 0.787 | 0.987 | 3 | 107.2% |
| 2 | 6 | 0.786 | 0.986 | 5 | 109.6% |
| 3 | 8 | 0.785 | 0.983 | 7 | 112.1% |
| 4 | 11 | 0.785 | 0.980 | 10 | 115.9% |
| 5 | 9 | 0.785 | 0.982 | 8 | 113.3% |
| 6 | 3 | 0.785 | 0.986 | 2 | 106.0% |
| 7 | 7 | 0.784 | 0.984 | 6 | 110.8% |
| 8 | 10 | 0.784 | 0.980 | 9 | 114.6% |
| 9 | 5 | 0.784 | 0.985 | 4 | 108.4% |
| 10 | 12 | 0.784 | 0.977 | 11 | 117.2% |

## Detailed Comparison: Top 3 Pairs

### Pair 1: Cycle 1 vs Cycle 4

![Comparison](ES12C4_cycle1_vs_4_comparison.png)

- **Time Gap**: 3 cycles
- **Composite Similarity**: 0.787
- **Correlation**: 0.987
- **Amplitude Similarity**: 0.979
- **Offset Similarity**: -0.001
- **Degradation**: 107.2%

### Pair 2: Cycle 1 vs Cycle 6

![Comparison](ES12C4_cycle1_vs_6_comparison.png)

- **Time Gap**: 5 cycles
- **Composite Similarity**: 0.786
- **Correlation**: 0.986
- **Amplitude Similarity**: 0.978
- **Offset Similarity**: -0.002
- **Degradation**: 109.6%

### Pair 3: Cycle 1 vs Cycle 8

![Comparison](ES12C4_cycle1_vs_8_comparison.png)

- **Time Gap**: 7 cycles
- **Composite Similarity**: 0.785
- **Correlation**: 0.983
- **Amplitude Similarity**: 0.980
- **Offset Similarity**: -0.002
- **Degradation**: 112.1%

## Conclusion

Cycle 1 shows the most dynamic input pattern in the dataset. Found 185 cycles with reasonable similarity for degradation analysis.

Report generated: 2026-01-15 00:31:35
