"""
Warning management utilities for NASA PCOE EDA system.

This module provides centralized warning management capabilities:
- Categorization of warnings by type and severity
- Configurable warning suppression
- Warning statistics and reporting
- Environment-based warning control
- CI/CD friendly warning management

The warning manager helps maintain clean test output while ensuring
important warnings are not missed.

Example usage:
    from nasa_pcoe_eda.utils.warnings_manager import WarningManager
    
    # Initialize with default settings
    wm = WarningManager()
    
    # Suppress font warnings during visualization
    with wm.suppress_category('font'):
        # Plotting code here
        pass
    
    # Get warning statistics
    stats = wm.get_warning_stats()
    print(f"Suppressed {stats['font']} font warnings")
"""

import warnings
import os
import logging
from typing import Dict, List, Optional, Set, Any
from contextlib import contextmanager
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum


class WarningCategory(Enum):
    """Categories of warnings for management purposes."""
    FONT = "font"
    DEPRECATION = "deprecation"
    RUNTIME = "runtime"
    MATPLOTLIB = "matplotlib"
    JAPANIZE = "japanize"
    NUMPY = "numpy"
    PANDAS = "pandas"
    SCIPY = "scipy"
    OTHER = "other"


@dataclass
class WarningConfig:
    """Configuration for warning management."""
    suppress_font_warnings: bool = True
    suppress_deprecation_warnings: bool = True
    suppress_runtime_warnings: bool = False
    suppress_matplotlib_warnings: bool = True
    suppress_japanize_warnings: bool = True
    log_suppressed_warnings: bool = False
    warning_log_level: int = logging.DEBUG
    max_warnings_per_category: int = 100
    categories_to_suppress: Set[WarningCategory] = field(default_factory=lambda: {
        WarningCategory.FONT,
        WarningCategory.MATPLOTLIB,
        WarningCategory.JAPANIZE
    })


class WarningManager:
    """Centralized warning management for the NASA PCOE EDA system."""
    
    def __init__(self, config: Optional[WarningConfig] = None):
        """
        Initialize the warning manager.
        
        Args:
            config: Warning configuration. If None, uses default config
                   modified by environment variables.
        """
        self.config = config or self._load_config_from_env()
        self._warning_counts: Dict[WarningCategory, int] = defaultdict(int)
        self._suppressed_warnings: List[Dict[str, Any]] = []
        self._original_showwarning = warnings.showwarning
        self._setup_warning_filters()
        self._setup_logging()
    
    def _load_config_from_env(self) -> WarningConfig:
        """Load configuration from environment variables."""
        config = WarningConfig()
        
        # Check main suppression flag
        suppress_warnings = os.getenv('NASA_PCOE_SUPPRESS_WARNINGS', 'true').lower() == 'true'
        
        if not suppress_warnings:
            # If main flag is false, disable all suppression
            config.suppress_font_warnings = False
            config.suppress_deprecation_warnings = False
            config.suppress_matplotlib_warnings = False
            config.suppress_japanize_warnings = False
            config.categories_to_suppress = set()
        else:
            # Fine-grained control
            config.suppress_font_warnings = os.getenv('NASA_PCOE_SUPPRESS_FONT', 'true').lower() == 'true'
            config.suppress_deprecation_warnings = os.getenv('NASA_PCOE_SUPPRESS_DEPRECATION', 'true').lower() == 'true'
            config.suppress_matplotlib_warnings = os.getenv('NASA_PCOE_SUPPRESS_MATPLOTLIB', 'true').lower() == 'true'
            config.suppress_japanize_warnings = os.getenv('NASA_PCOE_SUPPRESS_JAPANIZE', 'true').lower() == 'true'
        
        # Logging configuration
        config.log_suppressed_warnings = os.getenv('NASA_PCOE_LOG_WARNINGS', 'false').lower() == 'true'
        
        return config
    
    def _setup_logging(self):
        """Setup logging for warning management."""
        self.logger = logging.getLogger('nasa_pcoe_eda.warnings')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(self.config.warning_log_level)
    
    def _setup_warning_filters(self):
        """Setup warning filters based on configuration."""
        if self.config.suppress_font_warnings:
            self._add_font_filters()
        
        if self.config.suppress_deprecation_warnings:
            self._add_deprecation_filters()
        
        if self.config.suppress_matplotlib_warnings:
            self._add_matplotlib_filters()
        
        if self.config.suppress_japanize_warnings:
            self._add_japanize_filters()
        
        if self.config.suppress_runtime_warnings:
            self._add_runtime_filters()
    
    def _add_font_filters(self):
        """Add filters for font-related warnings."""
        font_patterns = [
            r'.*Glyph.*missing from current font.*',
            r'.*findfont.*',
            r'.*font.*not found.*',
            r'.*Japanese.*font.*',
            r'.*Font.*family.*not found.*',
            r'.*Substituting.*font.*'
        ]
        
        for pattern in font_patterns:
            warnings.filterwarnings('ignore', category=UserWarning, message=pattern)
    
    def _add_deprecation_filters(self):
        """Add filters for deprecation warnings."""
        deprecation_patterns = [
            r'.*pkg_resources.*',
            r'.*distutils.*',
            r'.*imp module.*',
            r'.*setuptools.*',
            r'.*collections\.abc.*'
        ]
        
        for pattern in deprecation_patterns:
            warnings.filterwarnings('ignore', category=DeprecationWarning, message=pattern)
    
    def _add_matplotlib_filters(self):
        """Add filters for matplotlib warnings."""
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.*')
        warnings.filterwarnings('ignore', category=DeprecationWarning, module='matplotlib.*')
    
    def _add_japanize_filters(self):
        """Add filters for japanize-matplotlib warnings."""
        warnings.filterwarnings('ignore', category=UserWarning, module='japanize_matplotlib.*')
        warnings.filterwarnings('ignore', category=DeprecationWarning, module='japanize_matplotlib.*')
    
    def _add_runtime_filters(self):
        """Add filters for runtime warnings."""
        runtime_patterns = [
            r'.*invalid value encountered.*',
            r'.*divide by zero encountered.*',
            r'.*Mean of empty slice.*',
            r'.*Degrees of freedom <= 0.*'
        ]
        
        for pattern in runtime_patterns:
            warnings.filterwarnings('ignore', category=RuntimeWarning, message=pattern)
    
    def _categorize_warning(self, message: str, category: type, module: str) -> WarningCategory:
        """Categorize a warning based on its properties."""
        message_lower = message.lower()
        module_lower = module.lower() if module else ""
        
        # Font-related warnings
        if any(keyword in message_lower for keyword in ['font', 'glyph', 'findfont']):
            return WarningCategory.FONT
        
        # Matplotlib warnings
        if 'matplotlib' in module_lower:
            return WarningCategory.MATPLOTLIB
        
        # Japanize warnings
        if 'japanize' in module_lower:
            return WarningCategory.JAPANIZE
        
        # NumPy warnings
        if 'numpy' in module_lower:
            return WarningCategory.NUMPY
        
        # Pandas warnings
        if 'pandas' in module_lower:
            return WarningCategory.PANDAS
        
        # SciPy warnings
        if 'scipy' in module_lower:
            return WarningCategory.SCIPY
        
        # Deprecation warnings
        if category == DeprecationWarning:
            return WarningCategory.DEPRECATION
        
        # Runtime warnings
        if category == RuntimeWarning:
            return WarningCategory.RUNTIME
        
        return WarningCategory.OTHER
    
    @contextmanager
    def suppress_category(self, category: WarningCategory):
        """Context manager to temporarily suppress warnings of a specific category."""
        with warnings.catch_warnings():
            if category == WarningCategory.FONT:
                self._add_font_filters()
            elif category == WarningCategory.MATPLOTLIB:
                self._add_matplotlib_filters()
            elif category == WarningCategory.JAPANIZE:
                self._add_japanize_filters()
            elif category == WarningCategory.DEPRECATION:
                self._add_deprecation_filters()
            elif category == WarningCategory.RUNTIME:
                self._add_runtime_filters()
            
            yield
    
    @contextmanager
    def suppress_all(self):
        """Context manager to temporarily suppress all configured warning categories."""
        with warnings.catch_warnings():
            for category in self.config.categories_to_suppress:
                if category == WarningCategory.FONT:
                    self._add_font_filters()
                elif category == WarningCategory.MATPLOTLIB:
                    self._add_matplotlib_filters()
                elif category == WarningCategory.JAPANIZE:
                    self._add_japanize_filters()
                elif category == WarningCategory.DEPRECATION:
                    self._add_deprecation_filters()
                elif category == WarningCategory.RUNTIME:
                    self._add_runtime_filters()
            
            yield
    
    def get_warning_stats(self) -> Dict[str, int]:
        """Get statistics about warnings by category."""
        return {category.value: count for category, count in self._warning_counts.items()}
    
    def get_suppressed_warnings(self) -> List[Dict[str, Any]]:
        """Get list of suppressed warnings with details."""
        return self._suppressed_warnings.copy()
    
    def reset_stats(self):
        """Reset warning statistics."""
        self._warning_counts.clear()
        self._suppressed_warnings.clear()
    
    def generate_warning_report(self) -> str:
        """Generate a comprehensive warning report."""
        stats = self.get_warning_stats()
        total_warnings = sum(stats.values())
        
        report = ["Warning Management Report", "=" * 25, ""]
        
        if total_warnings == 0:
            report.append("No warnings were suppressed.")
        else:
            report.append(f"Total suppressed warnings: {total_warnings}")
            report.append("")
            report.append("Breakdown by category:")
            
            for category, count in sorted(stats.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_warnings) * 100
                report.append(f"  {category}: {count} ({percentage:.1f}%)")
        
        report.append("")
        report.append("Configuration:")
        report.append(f"  Font warnings suppressed: {self.config.suppress_font_warnings}")
        report.append(f"  Deprecation warnings suppressed: {self.config.suppress_deprecation_warnings}")
        report.append(f"  Matplotlib warnings suppressed: {self.config.suppress_matplotlib_warnings}")
        report.append(f"  Japanize warnings suppressed: {self.config.suppress_japanize_warnings}")
        report.append(f"  Runtime warnings suppressed: {self.config.suppress_runtime_warnings}")
        
        return "\n".join(report)
    
    def log_warning_summary(self):
        """Log a summary of warning statistics."""
        stats = self.get_warning_stats()
        total = sum(stats.values())
        
        if total > 0:
            self.logger.info(f"Suppressed {total} warnings: {stats}")
        else:
            self.logger.debug("No warnings were suppressed")


# Global warning manager instance
_global_warning_manager: Optional[WarningManager] = None


def get_warning_manager() -> WarningManager:
    """Get the global warning manager instance."""
    global _global_warning_manager
    if _global_warning_manager is None:
        _global_warning_manager = WarningManager()
    return _global_warning_manager


def setup_warning_management():
    """Setup global warning management."""
    get_warning_manager()


def suppress_font_warnings():
    """Context manager to suppress font warnings."""
    return get_warning_manager().suppress_category(WarningCategory.FONT)


def suppress_matplotlib_warnings():
    """Context manager to suppress matplotlib warnings."""
    return get_warning_manager().suppress_category(WarningCategory.MATPLOTLIB)


def get_warning_stats() -> Dict[str, int]:
    """Get global warning statistics."""
    return get_warning_manager().get_warning_stats()


def generate_warning_report() -> str:
    """Generate global warning report."""
    return get_warning_manager().generate_warning_report()