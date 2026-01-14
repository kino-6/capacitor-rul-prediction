# ES12C4 True Input Similarity Analysis

## Problem Identified

**Correlation measures waveform shape similarity but ignores amplitude and offset differences.**

### Example from Previous Analysis
- Cycle 46: VL mean ≈ 0.7V, Cycle 96: VL mean ≈ 0.0V
- Correlation = 0.894 (high shape similarity)
- But amplitude difference is massive!
- This is NOT 'same input, different output'

## Solution: Comprehensive Similarity Metric

### Components:
1. **Shape Similarity** (50%): Correlation coefficient
2. **Amplitude Similarity** (30%): Standard deviation and range similarity
3. **Offset Similarity** (20%): Mean value similarity

## Results

- **Correlation-only pairs found**: 3960
- **True similar pairs found**: 406

### True Similar Pairs Found
**Pair 1**: Cycles 3-15
- Composite similarity: 0.933
- Shape similarity: 0.980
- Amplitude similarity: 0.983
- Offset similarity: 0.741
- Degradation: 255.2%

**Pair 2**: Cycles 4-15
- Composite similarity: 0.941
- Shape similarity: 0.983
- Amplitude similarity: 0.990
- Offset similarity: 0.763
- Degradation: 196.2%

**Pair 3**: Cycles 3-14
- Composite similarity: 0.939
- Shape similarity: 0.981
- Amplitude similarity: 0.990
- Offset similarity: 0.758
- Degradation: 232.2%

**Pair 4**: Cycles 3-17
- Composite similarity: 0.908
- Shape similarity: 0.953
- Amplitude similarity: 0.965
- Offset similarity: 0.710
- Degradation: 301.2%

**Pair 5**: Cycles 3-16
- Composite similarity: 0.918
- Shape similarity: 0.967
- Amplitude similarity: 0.964
- Offset similarity: 0.725
- Degradation: 278.2%

## Conclusion

The comprehensive similarity approach successfully addresses the critical flaw in correlation-only analysis by ensuring both shape AND amplitude consistency for reliable 'same input, different output' comparisons.

Report generated: 2026-01-05 14:03:22
