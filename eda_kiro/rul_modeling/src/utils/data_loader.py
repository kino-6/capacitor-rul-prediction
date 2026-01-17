"""
Simple data loader for ES12 dataset.
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Tuple


def load_es12_cycle_data(es12_path: str, capacitor_id: str, cycle: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load VL and VO data for a specific cycle from ES12.mat.
    
    Args:
        es12_path: Path to ES12.mat file
        capacitor_id: Capacitor ID (e.g., "ES12C1")
        cycle: Cycle number (1-indexed)
    
    Returns:
        Tuple of (VL, VO) numpy arrays
    """
    with h5py.File(es12_path, 'r') as f:
        # Navigate to transient data for the capacitor
        cap_group = f['ES12']['Transient_Data'][capacitor_id]
        
        # Get VL and VO data (shape: time_points × cycles)
        vl_data = cap_group['VL']
        vo_data = cap_group['VO']
        
        # Extract the specific cycle (cycle is 1-indexed, array is 0-indexed)
        # Data is stored as columns, so we use [:, cycle-1]
        vl = vl_data[:, cycle - 1]
        vo = vo_data[:, cycle - 1]
        
        # Find valid indices (where both VL and VO are not NaN)
        valid_mask = ~(np.isnan(vl) | np.isnan(vo))
        vl = vl[valid_mask]
        vo = vo[valid_mask]
    
    return vl, vo


def get_available_capacitors() -> list:
    """Get list of available capacitor IDs."""
    return [f'ES12C{i}' for i in range(1, 9)]
