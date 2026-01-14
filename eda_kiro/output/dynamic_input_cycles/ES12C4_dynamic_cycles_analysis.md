# ES12C4 Dynamic Input Cycles Analysis

## Objective

Find cycles with more dynamic VL input patterns (not nearly-constant) to better observe input-output response relationships and degradation.

## Dynamism Metrics

- **Coefficient of Variation**: Normalized variability
- **Rate of Change**: How quickly the signal changes
- **Sign Changes**: Number of peaks and valleys (complexity)
- **Periodicity Ratio**: Presence of periodic patterns
- **Dynamism Score**: Weighted combination of above metrics

## Top 20 Most Dynamic Cycles

| Rank | Cycle | Dynamism | CV | Mean Change | Sign Changes | Periodicity |
|------|-------|----------|----|--------------|--------------|--------------|
| 1 | 1 | 30.4110 | 47.6675 | 0.1381 | 1996 | 0.0051 |
| 201 | 201 | 7.7771 | 12.7281 | 0.1583 | 1959 | 0.0050 |
| 191 | 191 | 6.5581 | 10.2342 | 0.1531 | 1973 | 0.0063 |
| 193 | 193 | 6.4850 | 10.1065 | 0.1529 | 1981 | 0.0061 |
| 188 | 188 | 6.4694 | 10.0965 | 0.1531 | 1976 | 0.0064 |
| 190 | 190 | 6.4291 | 10.0295 | 0.1529 | 1976 | 0.0063 |
| 194 | 194 | 6.4107 | 9.9835 | 0.1530 | 1966 | 0.0060 |
| 189 | 189 | 6.4041 | 9.9884 | 0.1532 | 1976 | 0.0064 |
| 192 | 192 | 6.3955 | 9.9713 | 0.1530 | 1983 | 0.0061 |
| 186 | 186 | 6.3929 | 9.9831 | 0.1532 | 1977 | 0.0065 |
| 195 | 195 | 6.3687 | 9.9160 | 0.1528 | 1970 | 0.0059 |
| 187 | 187 | 6.2603 | 9.7636 | 0.1531 | 1975 | 0.0064 |
| 196 | 196 | 6.2408 | 9.7084 | 0.1529 | 1969 | 0.0058 |
| 185 | 185 | 6.2092 | 9.6902 | 0.1530 | 1980 | 0.0065 |
| 179 | 179 | 6.1536 | 9.6361 | 0.1529 | 1974 | 0.0068 |
| 184 | 184 | 6.0918 | 9.5074 | 0.1530 | 1969 | 0.0064 |
| 180 | 180 | 6.0718 | 9.4961 | 0.1529 | 1983 | 0.0067 |
| 183 | 183 | 6.0549 | 9.4497 | 0.1531 | 1979 | 0.0064 |
| 197 | 197 | 6.0520 | 9.4038 | 0.1529 | 1967 | 0.0056 |
| 181 | 181 | 5.9897 | 9.3550 | 0.1530 | 1978 | 0.0066 |

## Dynamic Cycle Pairs for Degradation Analysis

No pairs found meeting the criteria.

## Conclusion

The ES12 dataset appears to contain primarily steady-state or nearly-constant input patterns. Dynamic input-output response analysis may be limited with this dataset.

Report generated: 2026-01-15 00:30:15
