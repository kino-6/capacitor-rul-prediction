"""
True RUL Prediction System

A comprehensive system for predicting remaining useful life (RUL) of capacitors
using interpretable machine learning models.
"""

__version__ = "0.2.0"

from .data_structures import CycleData, CapacitorData, PredictionResult, TrainingDataset
from .data_loader import DataLoader, load_es12_data

__all__ = [
    "CycleData",
    "CapacitorData",
    "PredictionResult",
    "TrainingDataset",
    "DataLoader",
    "load_es12_data",
]
