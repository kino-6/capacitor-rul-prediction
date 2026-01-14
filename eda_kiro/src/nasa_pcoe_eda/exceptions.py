"""
Custom exception classes for the NASA PCOE EDA system.

This module defines a hierarchy of custom exceptions that provide specific
error handling for different components of the EDA system. Each exception
includes detailed error messages and context to help with debugging and
error recovery.

Exception Hierarchy:
- EDAError: Base exception for all EDA-related errors
  - DataLoadError: Data loading and file access errors
  - DataValidationError: Data validation and integrity errors
  - AnalysisError: Statistical analysis and computation errors
  - VisualizationError: Plotting and visualization errors

All exceptions inherit from EDAError, allowing for both specific and
general error handling patterns. The exceptions are designed to provide
clear, actionable error messages to help users resolve issues quickly.

Example usage:
    try:
        df = loader.load_dataset(path)
    except DataLoadError as e:
        logger.error(f"Failed to load dataset: {e}")
        # Handle data loading error
    except EDAError as e:
        logger.error(f"EDA system error: {e}")
        # Handle general EDA error
"""


class EDAError(Exception):
    """Base exception class for the EDA system."""

    pass


class DataLoadError(EDAError):
    """Exception raised when data loading fails."""

    pass


class DataValidationError(EDAError):
    """Exception raised when data validation fails."""

    pass


class AnalysisError(EDAError):
    """Exception raised when analysis operations fail."""

    pass


class VisualizationError(EDAError):
    """Exception raised when visualization operations fail."""

    pass
