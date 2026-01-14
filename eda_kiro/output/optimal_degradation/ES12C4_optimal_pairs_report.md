# ES12C4 Optimal Degradation Comparison Pairs Report

## Overview

This report identifies optimal cycle pairs for degradation analysis by balancing:
1. **Input Similarity**: Cycles with similar VL patterns for fair comparison
2. **Temporal Separation**: Sufficient time gap to observe meaningful degradation
3. **Degradation Significance**: Measurable changes in response characteristics

## Selection Criteria

- **Minimum Input Similarity**: ≥0.7 (correlation coefficient)
- **Minimum Time Gap**: ≥10 cycles
- **Minimum Degradation**: ≥20% voltage ratio change
- **Composite Score**: Weighted combination (40% similarity + 30% time + 30% degradation)

## Results Summary

- **Total Optimal Pairs Found**: 3960
- **Average Input Similarity**: 0.869
- **Average Time Gap**: 39.5 cycles
- **Average Degradation**: 1629.3%
- **Score Range**: 0.518 - 0.958

## Top 10 Optimal Pairs

| Rank | Cycle Pair | Time Gap | Similarity | Degradation | Score | Early Ratio | Late Ratio |
|------|------------|----------|------------|-------------|-------|-------------|------------|
| 1 | 46-96 | 50 | 0.894 | 1217.6% | 0.958 | 15.46 | 203.68 |
| 2 | 47-97 | 50 | 0.893 | 1182.4% | 0.957 | 15.94 | 204.43 |
| 3 | 45-95 | 50 | 0.893 | 1248.2% | 0.957 | 15.01 | 202.37 |
| 4 | 46-97 | 51 | 0.891 | 1222.4% | 0.957 | 15.46 | 204.43 |
| 5 | 44-94 | 50 | 0.891 | 1282.7% | 0.956 | 14.57 | 201.45 |
| 6 | 45-96 | 51 | 0.890 | 1256.9% | 0.956 | 15.01 | 203.68 |
| 7 | 47-98 | 51 | 0.890 | 1185.5% | 0.956 | 15.94 | 204.93 |
| 8 | 46-98 | 52 | 0.889 | 1225.7% | 0.955 | 15.46 | 204.93 |
| 9 | 43-93 | 50 | 0.889 | 1307.7% | 0.955 | 14.13 | 198.94 |
| 10 | 44-95 | 51 | 0.888 | 1289.0% | 0.955 | 14.57 | 202.37 |

## Detailed Analysis of Top 5 Pairs

### Pair #1: Cycles 46 → 96

#### Characteristics
- **Time Separation**: 50 cycles
- **Input Similarity**: 0.894 (correlation)
- **Voltage Ratio Change**: 1217.6%
- **Composite Score**: 0.958

#### Response Changes
- **Early Cycle (46) Ratio**: 15.459
- **Late Cycle (96) Ratio**: 203.678
- **Change Direction**: +1217.6%
- **Degradation Severity**: 🔴 **Severe** (>100% change)

#### Why This Pair is Optimal
- **Similar Inputs**: High correlation (0.894) ensures fair comparison
- **Sufficient Time Gap**: 50 cycles allows degradation to manifest
- **Clear Degradation**: 1217.6% change is easily observable
- **Balanced Score**: Optimal trade-off between all criteria

### Pair #2: Cycles 47 → 97

#### Characteristics
- **Time Separation**: 50 cycles
- **Input Similarity**: 0.893 (correlation)
- **Voltage Ratio Change**: 1182.4%
- **Composite Score**: 0.957

#### Response Changes
- **Early Cycle (47) Ratio**: 15.942
- **Late Cycle (97) Ratio**: 204.429
- **Change Direction**: +1182.4%
- **Degradation Severity**: 🔴 **Severe** (>100% change)

#### Why This Pair is Optimal
- **Similar Inputs**: High correlation (0.893) ensures fair comparison
- **Sufficient Time Gap**: 50 cycles allows degradation to manifest
- **Clear Degradation**: 1182.4% change is easily observable
- **Balanced Score**: Optimal trade-off between all criteria

### Pair #3: Cycles 45 → 95

#### Characteristics
- **Time Separation**: 50 cycles
- **Input Similarity**: 0.893 (correlation)
- **Voltage Ratio Change**: 1248.2%
- **Composite Score**: 0.957

#### Response Changes
- **Early Cycle (45) Ratio**: 15.011
- **Late Cycle (95) Ratio**: 202.367
- **Change Direction**: +1248.2%
- **Degradation Severity**: 🔴 **Severe** (>100% change)

#### Why This Pair is Optimal
- **Similar Inputs**: High correlation (0.893) ensures fair comparison
- **Sufficient Time Gap**: 50 cycles allows degradation to manifest
- **Clear Degradation**: 1248.2% change is easily observable
- **Balanced Score**: Optimal trade-off between all criteria

### Pair #4: Cycles 46 → 97

#### Characteristics
- **Time Separation**: 51 cycles
- **Input Similarity**: 0.891 (correlation)
- **Voltage Ratio Change**: 1222.4%
- **Composite Score**: 0.957

#### Response Changes
- **Early Cycle (46) Ratio**: 15.459
- **Late Cycle (97) Ratio**: 204.429
- **Change Direction**: +1222.4%
- **Degradation Severity**: 🔴 **Severe** (>100% change)

#### Why This Pair is Optimal
- **Similar Inputs**: High correlation (0.891) ensures fair comparison
- **Sufficient Time Gap**: 51 cycles allows degradation to manifest
- **Clear Degradation**: 1222.4% change is easily observable
- **Balanced Score**: Optimal trade-off between all criteria

### Pair #5: Cycles 44 → 94

#### Characteristics
- **Time Separation**: 50 cycles
- **Input Similarity**: 0.891 (correlation)
- **Voltage Ratio Change**: 1282.7%
- **Composite Score**: 0.956

#### Response Changes
- **Early Cycle (44) Ratio**: 14.569
- **Late Cycle (94) Ratio**: 201.446
- **Change Direction**: +1282.7%
- **Degradation Severity**: 🔴 **Severe** (>100% change)

#### Why This Pair is Optimal
- **Similar Inputs**: High correlation (0.891) ensures fair comparison
- **Sufficient Time Gap**: 50 cycles allows degradation to manifest
- **Clear Degradation**: 1282.7% change is easily observable
- **Balanced Score**: Optimal trade-off between all criteria

## Methodology Benefits

1. **Avoids Trivial Comparisons**: Minimum time gap prevents comparing adjacent cycles
2. **Ensures Fair Comparison**: Similarity threshold maintains input consistency
3. **Focuses on Significant Changes**: Degradation threshold filters meaningful differences
4. **Balanced Selection**: Composite scoring prevents over-optimization of single criteria

## Recommendations

- Use top-ranked pairs for detailed degradation analysis
- Focus on pairs with high similarity (>0.8) for most reliable comparisons
- Consider pairs with larger time gaps for long-term degradation studies
- Validate findings across multiple optimal pairs to ensure robustness

---
Report Generated: 2026-01-03 18:33:42
