"""Unit tests for WarningManager."""

import pytest
import warnings
from unittest.mock import patch
import os

from nasa_pcoe_eda.utils.warnings_manager import (
    WarningManager,
    WarningCategory,
    WarningConfig,
    get_warning_manager,
    setup_warning_management
)


class TestWarningManager:
    """Test cases for WarningManager."""
    
    def test_warning_manager_initialization(self):
        """Test that WarningManager initializes correctly."""
        wm = WarningManager()
        assert wm.config is not None
        assert isinstance(wm.config, WarningConfig)
        assert wm.config.suppress_font_warnings is True
    
    def test_warning_config_from_env(self):
        """Test that configuration is loaded from environment variables."""
        with patch.dict(os.environ, {'NASA_PCOE_SUPPRESS_WARNINGS': 'false'}):
            wm = WarningManager()
            assert wm.config.suppress_font_warnings is False
            assert wm.config.suppress_matplotlib_warnings is False
    
    def test_warning_categorization(self):
        """Test that warnings are categorized correctly."""
        wm = WarningManager()
        
        # Test font warning categorization
        category = wm._categorize_warning(
            "Glyph missing from font", 
            UserWarning, 
            "matplotlib.font_manager"
        )
        assert category == WarningCategory.FONT
        
        # Test matplotlib warning categorization
        category = wm._categorize_warning(
            "Some matplotlib warning", 
            UserWarning, 
            "matplotlib.pyplot"
        )
        assert category == WarningCategory.MATPLOTLIB
        
        # Test deprecation warning categorization
        category = wm._categorize_warning(
            "pkg_resources is deprecated", 
            DeprecationWarning, 
            "pkg_resources"
        )
        assert category == WarningCategory.DEPRECATION
    
    def test_suppress_category_context_manager(self):
        """Test that category suppression context manager works."""
        wm = WarningManager()
        
        # This should not raise any exceptions
        with wm.suppress_category(WarningCategory.FONT):
            # Simulate font warning (would normally be suppressed)
            pass
    
    def test_suppress_all_context_manager(self):
        """Test that suppress_all context manager works."""
        wm = WarningManager()
        
        # This should not raise any exceptions
        with wm.suppress_all():
            # All configured warnings should be suppressed
            pass
    
    def test_warning_stats(self):
        """Test warning statistics functionality."""
        wm = WarningManager()
        
        # Initially, stats should be empty
        stats = wm.get_warning_stats()
        assert isinstance(stats, dict)
        assert all(count == 0 for count in stats.values())
        
        # Reset stats should work
        wm.reset_stats()
        stats = wm.get_warning_stats()
        assert all(count == 0 for count in stats.values())
    
    def test_warning_report_generation(self):
        """Test warning report generation."""
        wm = WarningManager()
        
        report = wm.generate_warning_report()
        assert isinstance(report, str)
        assert "Warning Management Report" in report
        assert "Configuration:" in report
    
    def test_global_warning_manager(self):
        """Test global warning manager functionality."""
        # Get global instance
        wm1 = get_warning_manager()
        wm2 = get_warning_manager()
        
        # Should be the same instance
        assert wm1 is wm2
        
        # Setup should work without errors
        setup_warning_management()
    
    def test_warning_config_defaults(self):
        """Test default warning configuration values."""
        config = WarningConfig()
        
        assert config.suppress_font_warnings is True
        assert config.suppress_deprecation_warnings is True
        assert config.suppress_matplotlib_warnings is True
        assert config.suppress_japanize_warnings is True
        assert config.log_suppressed_warnings is False
        assert config.max_warnings_per_category == 100
    
    def test_environment_variable_override(self):
        """Test that environment variables override default config."""
        with patch.dict(os.environ, {
            'NASA_PCOE_SUPPRESS_WARNINGS': 'true',
            'NASA_PCOE_SUPPRESS_FONT': 'false',
            'NASA_PCOE_LOG_WARNINGS': 'true'
        }):
            wm = WarningManager()
            assert wm.config.suppress_font_warnings is False
            assert wm.config.log_suppressed_warnings is True
    
    def test_warning_filter_setup(self):
        """Test that warning filters are set up correctly."""
        wm = WarningManager()
        
        # Check that warning filters were added
        # This is a basic test - the actual filtering is tested through integration
        assert len(warnings.filters) > 0
    
    @pytest.mark.parametrize("category", [
        WarningCategory.FONT,
        WarningCategory.MATPLOTLIB,
        WarningCategory.DEPRECATION,
        WarningCategory.RUNTIME
    ])
    def test_category_suppression(self, category):
        """Test suppression of different warning categories."""
        wm = WarningManager()
        
        # Test that context manager works for each category
        with wm.suppress_category(category):
            # Should not raise any exceptions
            pass
    
    def test_warning_manager_logging(self):
        """Test warning manager logging functionality."""
        wm = WarningManager()
        
        # Test log warning summary (should not raise exceptions)
        wm.log_warning_summary()
        
        # Test with some mock statistics
        wm._warning_counts[WarningCategory.FONT] = 5
        wm.log_warning_summary()