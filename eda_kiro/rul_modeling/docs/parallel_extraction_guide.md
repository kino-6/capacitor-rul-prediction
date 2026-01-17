# Parallel Feature Extraction Guide

## Overview

The `ParallelFeatureExtractor` module provides high-performance feature extraction from multiple capacitors using multiprocessing. It's designed to efficiently process the ES12 dataset (8 capacitors, ~200 cycles each) by leveraging multiple CPU cores.

## Features

- **Parallel Processing**: Utilizes multiprocessing to extract features from multiple capacitors simultaneously
- **Progress Tracking**: Real-time progress reporting with cycle counts and percentages
- **Flexible Configuration**: Configurable number of processes, cycle limits, and feature options
- **History Support**: Optional historical feature extraction (disabled by default for Phase 1)
- **Automatic Saving**: Built-in CSV export with organized column ordering

## Quick Start

### Basic Usage

```python
from src.data_preparation.parallel_extractor import extract_es12_features

# Extract features from all ES12 capacitors
features_df = extract_es12_features(
    es12_path="../data/raw/ES12.mat",
    output_path="output/features/es12_features.csv",
    max_cycles=200,
    n_processes=8,
    include_history=False,
    progress_interval=20
)
```

### Advanced Usage

```python
from src.data_preparation.parallel_extractor import ParallelFeatureExtractor

# Initialize extractor
extractor = ParallelFeatureExtractor(
    es12_path="../data/raw/ES12.mat",
    n_processes=8,
    include_history=False
)

# Extract from specific capacitors
features_df = extractor.extract_all_capacitors(
    capacitor_ids=["ES12C1", "ES12C2", "ES12C3"],
    max_cycles=200,
    progress_interval=20
)

# Save results
extractor.save_features(features_df, "output/features/custom_features.csv")
```

## Configuration Parameters

### ParallelFeatureExtractor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `es12_path` | str | Required | Path to ES12.mat file |
| `n_processes` | int | CPU count | Number of parallel processes |
| `include_history` | bool | False | Whether to extract historical features |

### extract_all_capacitors

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `capacitor_ids` | List[str] | All ES12 | List of capacitor IDs to process |
| `max_cycles` | int | 200 | Maximum cycles per capacitor |
| `progress_interval` | int | 20 | Progress report frequency |

## Output Format

### CSV Structure

The output CSV file contains:

1. **Metadata Columns** (first):
   - `capacitor_id`: Capacitor identifier (e.g., "ES12C1")
   - `cycle`: Cycle number (1-indexed)

2. **Feature Columns** (alphabetically sorted):
   - Basic statistics (16 features): `vl_mean`, `vl_std`, `vl_min`, `vl_max`, `vl_range`, `vl_median`, `vl_q25`, `vl_q75`, `vo_mean`, `vo_std`, `vo_min`, `vo_max`, `vo_range`, `vo_median`, `vo_q25`, `vo_q75`
   - Degradation indicators (4 features): `voltage_ratio`, `voltage_ratio_std`, `response_efficiency`, `signal_attenuation`
   - Time series features (4 features): `vl_trend`, `vo_trend`, `vl_cv`, `vo_cv`
   - Cycle information (2 features): `cycle_number`, `cycle_normalized`

### Example Output

```csv
capacitor_id,cycle,cycle_normalized,cycle_number,response_efficiency,...
ES12C1,1,0.005,1.0,0.823,...
ES12C1,2,0.010,2.0,0.819,...
ES12C2,1,0.005,1.0,0.831,...
```

## Performance

### Benchmarks (M4 Pro, 14 cores)

| Configuration | Time | Throughput |
|---------------|------|------------|
| 1 capacitor, 200 cycles | ~9.5s | ~21 cycles/s |
| 2 capacitors, 200 cycles (parallel) | ~10.2s | ~39 cycles/s |
| 8 capacitors, 200 cycles (parallel) | ~3-4 min | ~7 cycles/s/capacitor |

### Optimization Tips

1. **Process Count**: Set `n_processes` to match the number of capacitors for optimal parallelization
2. **Progress Interval**: Use larger intervals (20-50) for faster processing with less I/O overhead
3. **History Features**: Disable `include_history` in Phase 1 to reduce processing time by ~30%

## Feature Descriptions

### Basic Statistics

- **Mean/Std/Min/Max**: Standard statistical measures for VL and VO
- **Range**: Max - Min for both signals
- **Median/Quartiles**: Robust central tendency and spread measures

### Degradation Indicators

- **voltage_ratio**: VO mean / VL mean (key degradation indicator)
- **voltage_ratio_std**: Standard deviation of point-wise voltage ratios
- **response_efficiency**: VO range / VL range
- **signal_attenuation**: 1 - (VO std / VL std)

### Time Series Features

- **vl_trend/vo_trend**: Linear regression slope over the cycle
- **vl_cv/vo_cv**: Coefficient of variation (std / mean)

### Cycle Information

- **cycle_number**: Absolute cycle number
- **cycle_normalized**: Cycle number / total_cycles (0-1 range)

## Error Handling

The extractor includes robust error handling:

- **File Not Found**: Raises `FileNotFoundError` if ES12.mat doesn't exist
- **Cycle Errors**: Logs errors and continues with remaining cycles
- **Process Failures**: Gracefully handles worker process failures

## Progress Output Example

```
======================================================================
Parallel Feature Extraction
======================================================================
Capacitors: 8 (ES12C1, ES12C2, ES12C3, ES12C4, ES12C5, ES12C6, ES12C7, ES12C8)
Cycles per capacitor: 200
Processes: 8
Include history: False
======================================================================

[ES12C1] Starting feature extraction...
[ES12C2] Starting feature extraction...
...
[ES12C1] Cycle 20/200 (10.0%) - Elapsed: 18.5s
[ES12C2] Cycle 20/200 (10.0%) - Elapsed: 18.7s
...
[ES12C1] Cycle 200/200 (100.0%) - Elapsed: 185.2s
[ES12C1] Completed in 185.2s (200 cycles extracted)
...

======================================================================
Extraction Complete!
======================================================================
Total samples: 1600
Total features: 28
Total time: 195.3s (3.3 minutes)
Average time per capacitor: 24.4s
======================================================================
```

## Integration with Pipeline

### Phase 1: Data Preparation

```python
# Step 1: Extract features
from src.data_preparation.parallel_extractor import extract_es12_features

features_df = extract_es12_features(
    es12_path="../data/raw/ES12.mat",
    output_path="output/features/es12_features.csv",
    max_cycles=200,
    n_processes=8,
    include_history=False
)

# Step 2: Add labels (next task)
from src.data_preparation.label_generator import LabelGenerator
# ... continue with labeling
```

## Testing

Run the test suite to verify functionality:

```bash
# Run all tests
pytest tests/test_parallel_extractor.py -v

# Run specific test
pytest tests/test_parallel_extractor.py::TestParallelFeatureExtractor::test_parallel_extraction_multiple_capacitors -v

# Quick verification test
python test_parallel_extractor.py
```

## Troubleshooting

### Issue: Slow Performance

**Solution**: Increase `n_processes` to match available CPU cores

```python
import multiprocessing as mp
n_cores = mp.cpu_count()
print(f"Available cores: {n_cores}")
```

### Issue: Memory Errors

**Solution**: Reduce `n_processes` or process capacitors in batches

```python
# Process in batches
for batch in [["ES12C1", "ES12C2"], ["ES12C3", "ES12C4"]]:
    features_df = extractor.extract_all_capacitors(
        capacitor_ids=batch,
        max_cycles=200
    )
```

### Issue: Missing Features

**Solution**: Verify CycleFeatureExtractor is properly configured

```python
from src.feature_extraction.extractor import CycleFeatureExtractor
extractor = CycleFeatureExtractor()
# Check available methods
print(dir(extractor))
```

## References

- [CycleFeatureExtractor Documentation](../src/feature_extraction/extractor.py)
- [ES12 Data Structure Guide](../../../docs/es12_data_structure_guide.md)
- [RUL Model Design](../.kiro/specs/rul_model_spec/design.md)

---

**Created**: 2026-01-16  
**Last Updated**: 2026-01-16  
**Version**: 1.0
