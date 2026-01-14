# Warning Management Developer Guide

This guide provides detailed information about the warning management system in the NASA PCOE EDA project.

## Overview

The warning management system is designed to:
- Suppress known harmless warnings that clutter test output
- Categorize warnings by type and importance
- Provide configurable warning control
- Generate warning statistics and reports
- Maintain clean CI/CD output while preserving important warnings

## Architecture

### Core Components

1. **WarningManager**: Central warning management class
2. **WarningCategory**: Enumeration of warning types
3. **WarningConfig**: Configuration data class
4. **Context Managers**: For temporary warning suppression
5. **Environment Integration**: Automatic setup on package import

### Warning Categories

The system categorizes warnings into the following types:

- `FONT`: Japanese font-related warnings from matplotlib
- `MATPLOTLIB`: General matplotlib warnings
- `JAPANIZE`: Warnings from japanize-matplotlib library
- `DEPRECATION`: Deprecation warnings from dependencies
- `RUNTIME`: Runtime warnings from numerical computations
- `NUMPY`: NumPy-specific warnings
- `PANDAS`: Pandas-specific warnings
- `SCIPY`: SciPy-specific warnings
- `OTHER`: Uncategorized warnings

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NASA_PCOE_SUPPRESS_WARNINGS` | `true` | Master toggle for warning suppression |
| `NASA_PCOE_SUPPRESS_FONT` | `true` | Suppress font-related warnings |
| `NASA_PCOE_SUPPRESS_MATPLOTLIB` | `true` | Suppress matplotlib warnings |
| `NASA_PCOE_SUPPRESS_JAPANIZE` | `true` | Suppress japanize-matplotlib warnings |
| `NASA_PCOE_SUPPRESS_DEPRECATION` | `true` | Suppress deprecation warnings |
| `NASA_PCOE_LOG_WARNINGS` | `false` | Log suppressed warnings |
| `NASA_PCOE_ENV` | - | Environment name (development/testing/ci/production) |

### Configuration File

The `warning_config.yaml` file provides detailed configuration options:

```yaml
global:
  suppress_warnings: true
  log_suppressed_warnings: false

categories:
  font:
    suppress: true
    patterns:
      - ".*Glyph.*missing from current font.*"
      - ".*findfont.*"
```

## Usage Patterns

### Basic Usage

```python
from nasa_pcoe_eda.utils.warnings_manager import get_warning_manager

# Get the global warning manager
wm = get_warning_manager()

# Check configuration
config = wm.config
print(f"Font warnings suppressed: {config.suppress_font_warnings}")
```

### Context Managers

```python
from nasa_pcoe_eda.utils.warnings_manager import WarningCategory

# Suppress specific category temporarily
with wm.suppress_category(WarningCategory.FONT):
    # Font warnings are suppressed here
    plt.figure()
    plt.plot([1, 2, 3])
    plt.title("日本語タイトル")  # No font warnings

# Suppress all configured categories
with wm.suppress_all():
    # All configured warning categories are suppressed
    run_analysis()
```

### Statistics and Reporting

```python
# Get warning statistics
stats = wm.get_warning_stats()
print(f"Font warnings: {stats['font']}")
print(f"Deprecation warnings: {stats['deprecation']}")

# Generate comprehensive report
report = wm.generate_warning_report()
print(report)

# Reset statistics
wm.reset_stats()
```

## Integration with Visualization Engine

The `VisualizationEngine` class automatically uses warning suppression:

```python
class VisualizationEngine:
    def __init__(self):
        self._setup_warning_filters()  # Automatic setup
        
    @contextmanager
    def _suppress_font_warnings(self):
        """Context manager for font warning suppression."""
        with warnings.catch_warnings():
            # Suppress font-related warnings
            warnings.filterwarnings('ignore', ...)
            yield
    
    def plot_distributions(self, ...):
        with self._suppress_font_warnings():
            # Plotting code with suppressed warnings
            plt.figure()
            # ...
```

## Testing Integration

### pytest Configuration

The `pytest.ini` file configures warning filters for test execution:

```ini
[tool:pytest]
filterwarnings =
    # Suppress known harmless warnings
    ignore::UserWarning:matplotlib.*
    ignore::UserWarning:japanize_matplotlib.*
    
    # Allow important warnings from our code
    default::UserWarning:nasa_pcoe_eda.*
```

### Test-Specific Warning Control

```python
import pytest
from nasa_pcoe_eda.utils.warnings_manager import get_warning_manager

def test_with_warning_suppression():
    wm = get_warning_manager()
    
    with wm.suppress_category(WarningCategory.FONT):
        # Test code that would generate font warnings
        result = create_visualization()
        assert result is not None

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_with_pytest_filter():
    # This test ignores all UserWarnings
    pass
```

## CI/CD Integration

### GitHub Actions Example

```yaml
- name: Run tests with warning management
  run: |
    export NASA_PCOE_ENV=ci
    export NASA_PCOE_SUPPRESS_WARNINGS=true
    uv run pytest --tb=short
```

### Warning Report Generation

```python
# In CI scripts
from nasa_pcoe_eda import generate_warning_report

# Generate and save warning report
report = generate_warning_report()
with open('warning_report.txt', 'w') as f:
    f.write(report)
```

## Best Practices

### 1. Categorize New Warnings

When adding new warning filters:

```python
def _categorize_warning(self, message: str, category: type, module: str) -> WarningCategory:
    """Add logic to categorize new warning types."""
    if 'new_library' in module.lower():
        return WarningCategory.OTHER
    # ...
```

### 2. Environment-Specific Configuration

Use different configurations for different environments:

```python
# Development: Show more warnings
if os.getenv('NASA_PCOE_ENV') == 'development':
    config.suppress_runtime_warnings = False

# CI: Suppress all for clean output
elif os.getenv('NASA_PCOE_ENV') == 'ci':
    config.suppress_runtime_warnings = True
```

### 3. Temporary Suppression

Use context managers for temporary suppression:

```python
# Good: Temporary suppression
with suppress_font_warnings():
    create_japanese_plot()

# Avoid: Global suppression
warnings.filterwarnings('ignore')  # Too broad
```

### 4. Monitor Warning Trends

Regularly check warning statistics:

```python
# In test teardown or CI scripts
stats = get_warning_stats()
if stats['other'] > 10:  # Threshold for new warnings
    print(f"Warning: {stats['other']} uncategorized warnings detected")
```

## Troubleshooting

### Common Issues

1. **Warnings Still Appearing**
   - Check environment variables
   - Verify warning patterns match actual messages
   - Ensure warning manager is initialized

2. **Important Warnings Suppressed**
   - Review warning categories
   - Use more specific patterns
   - Check environment configuration

3. **Performance Impact**
   - Warning filtering has minimal overhead
   - Context managers are efficient
   - Statistics collection is lightweight

### Debugging

```python
# Enable debug logging
import logging
logging.getLogger('nasa_pcoe_eda.warnings').setLevel(logging.DEBUG)

# Check current filters
import warnings
print(warnings.filters)

# Test warning categorization
wm = get_warning_manager()
category = wm._categorize_warning("test message", UserWarning, "matplotlib.font_manager")
print(f"Categorized as: {category}")
```

## Future Enhancements

Potential improvements to the warning management system:

1. **Machine Learning Classification**: Automatically categorize new warnings
2. **Warning Trends Analysis**: Track warning patterns over time
3. **Integration with Monitoring**: Send alerts for unusual warning patterns
4. **Custom Warning Handlers**: User-defined warning processing
5. **Performance Metrics**: Track warning suppression performance impact

## Contributing

When contributing to the warning management system:

1. Add tests for new warning categories
2. Update documentation for new configuration options
3. Ensure backward compatibility
4. Test in different environments (dev/test/ci/prod)
5. Update the configuration schema if needed

## References

- [Python warnings module documentation](https://docs.python.org/3/library/warnings.html)
- [pytest warning management](https://docs.pytest.org/en/stable/how.html#warnings)
- [matplotlib warning handling](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html)