# ES12 Dataset Structure and Loading Guide

## Overview

The ES12.mat file contains capacitor degradation data from NASA PCOE (Prognostics Center of Excellence) Dataset No.12. This dataset includes Electrochemical Impedance Spectroscopy (EIS) measurements and transient response data for 8 capacitors undergoing electrical stress testing.

## File Structure

The ES12.mat file is a MATLAB v7.3 format file (HDF5-based) with the following hierarchical structure:

```
ES12/
├── EIS_Data/
│   ├── EIS_Reference_Table (4×73 object array)
│   ├── ES12C1/
│   │   └── EIS_Measurement/
│   │       ├── ColumNames (73×1 object references)
│   │       ├── Data (73×1 object references)
│   │       └── Header (73×1 object references)
│   ├── ES12C2/ ... ES12C8/ (similar structure)
├── Transient_Data/
│   ├── Serial_Date (77241×1 float64 - MATLAB serial dates)
│   ├── ES12C1/
│   │   ├── VL (77237×400 float64 - Load Voltage)
│   │   └── VO (77237×400 float64 - Output Voltage)
│   ├── ES12C2/ ... ES12C8/ (similar structure)
└── Initial_Date (22×1 uint16 - Initial measurement date)
```

## Data Content

### Capacitors
- **8 capacitors**: ES12C1 through ES12C8
- Each capacitor underwent the same electrical stress testing protocol
- Individual degradation patterns and failure modes

### EIS Data
- **73 measurements** per capacitor
- Complex reference structure requiring specialized parsing
- Contains impedance and phase information at various frequencies
- Currently extracted as metadata (measurement count)

### Transient Data
- **VL (Load Voltage)**: Input voltage measurements
- **VO (Output Voltage)**: Output voltage measurements
- **77,237 time points** per measurement
- **400 cycles** per capacitor (approximately)
- High-resolution time series data for degradation analysis

### Temporal Information
- **Serial_Date**: MATLAB serial date format (days since January 1, 0000)
- **Date Range**: November 18, 2015 to November 19, 2015
- **Duration**: Approximately 15 hours of continuous testing

## Data Loading

### Automatic ES12 Detection

The system automatically detects ES12 files and uses the specialized loader:

```python
from nasa_pcoe_eda.data.loader import DataLoader

loader = DataLoader()
df = loader.load_dataset("data/raw/ES12.mat")  # Auto-detects ES12 format
```

### Specialized ES12 Loader

For direct access to ES12-specific functionality:

```python
from nasa_pcoe_eda.data.es12_loader import ES12DataLoader

loader = ES12DataLoader()
df = loader.load_dataset("data/raw/ES12.mat")

# Access capacitor-specific data
cap_data = loader.get_capacitor_data('ES12C1')

# Access raw transient data
raw_data = loader.get_raw_transient_data('ES12C1')
vl_data = raw_data['VL']  # Shape: (77237, 400)
vo_data = raw_data['VO']  # Shape: (77237, 400)
```

## Processed Data Format

The loader converts the raw ES12 data into a structured DataFrame suitable for analysis:

### DataFrame Structure
- **3,120 rows**: 8 capacitors × 390 cycles (average)
- **15 columns**: Statistical summaries and metadata

### Key Columns
- `capacitor`: Capacitor identifier (ES12C1-ES12C8)
- `cycle`: Cycle number (1-based indexing)
- `vl_mean`, `vl_std`, `vl_min`, `vl_max`: Load voltage statistics
- `vo_mean`, `vo_std`, `vo_min`, `vo_max`: Output voltage statistics
- `vl_valid_points`, `vo_valid_points`: Number of valid measurements
- `voltage_ratio`: VO/VL ratio (efficiency indicator)
- `timestamp`: Converted Python datetime
- `eis_measurements`: Number of EIS measurements available

## Degradation Analysis Features

### Voltage Ratio Degradation
- **Initial efficiency**: VO/VL ratio at cycle 1
- **Degradation pattern**: Typically decreases over time
- **Failure indicator**: Significant ratio changes indicate degradation

### Statistical Trends
- **Mean voltage changes**: Track average voltage levels
- **Standard deviation changes**: Indicate response stability
- **Valid point counts**: Monitor data quality over time

### Individual Capacitor Patterns
- **ES12C1, ES12C2, ES12C4, ES12C8**: Show significant degradation
- **ES12C3, ES12C6**: More stable performance
- **ES12C5, ES12C7**: Mixed degradation patterns

## Usage Examples

### Basic Analysis
```python
# Load and examine data
loader = DataLoader()
df = loader.load_dataset("data/raw/ES12.mat")

print(f"Dataset shape: {df.shape}")
print(f"Capacitors: {sorted(df['capacitor'].unique())}")
print(f"Cycle range: {df['cycle'].min()} to {df['cycle'].max()}")

# Analyze degradation per capacitor
for cap in df['capacitor'].unique():
    cap_data = df[df['capacitor'] == cap]
    initial_ratio = cap_data['voltage_ratio'].iloc[0]
    final_ratio = cap_data['voltage_ratio'].iloc[-1]
    degradation = (final_ratio - initial_ratio) / initial_ratio * 100
    print(f"{cap}: {degradation:.1f}% voltage ratio change")
```

### Advanced Analysis
```python
# Use with existing analyzers
from nasa_pcoe_eda.analysis.rul_features import RULFeatureAnalyzer

rul_analyzer = RULFeatureAnalyzer()

# Create synthetic RUL (Remaining Useful Life)
max_cycle = df['cycle'].max()
df['rul'] = max_cycle - df['cycle']

# Rank features for RUL prediction
rul_features = rul_analyzer.rank_features_for_rul(df, 'rul')
print("Top RUL prediction features:")
for feat, score in rul_features[:5]:
    print(f"  {feat}: {score:.3f}")
```

## Data Quality Notes

### Strengths
- **High temporal resolution**: 77,237 time points per cycle
- **Multiple capacitors**: 8 units for comparative analysis
- **Complete coverage**: All capacitors have full datasets
- **Temporal consistency**: Synchronized measurements across capacitors

### Limitations
- **EIS complexity**: EIS data requires specialized domain knowledge
- **Large file size**: 1.8 GB file requires significant memory
- **Processing time**: Initial loading takes 15-20 seconds
- **Missing cycles**: Some capacitors have fewer than 400 cycles due to early failure

### Recommendations
- **Memory management**: Use statistical summaries for large-scale analysis
- **Cycle selection**: Focus on specific cycle ranges for detailed analysis
- **Comparative studies**: Leverage multiple capacitors for pattern validation
- **Temporal analysis**: Use timestamp information for time-based studies

## Integration with EDA System

The ES12 loader integrates seamlessly with all existing EDA components:

- **StatisticsAnalyzer**: Computes descriptive statistics
- **CorrelationAnalyzer**: Identifies feature relationships
- **OutlierDetector**: Detects anomalous measurements
- **RULFeatureAnalyzer**: Ranks features for degradation prediction
- **VisualizationEngine**: Creates degradation plots
- **ReportGenerator**: Includes ES12 analysis in reports

This integration enables comprehensive degradation analysis using the full EDA toolkit while maintaining the specialized handling required for the complex ES12 data structure.