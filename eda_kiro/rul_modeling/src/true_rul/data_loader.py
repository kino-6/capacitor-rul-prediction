"""
Data loader for ES12 dataset with new data structures
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional
import logging

import numpy as np

# Add parent directory to path to import existing ES12DataLoader
try:
    project_root = Path(__file__).parent.parent.parent.parent.parent / "eda_kiro"
    sys.path.insert(0, str(project_root))
    from src.nasa_pcoe_eda.data.es12_loader import ES12DataLoader as BaseES12Loader
except ImportError:
    # Fallback for testing - create a dummy loader
    class BaseES12Loader:
        def load_dataset(self, path):
            return {}
        def get_raw_transient_data(self, cap_id):
            return None

from .data_structures import CycleData, CapacitorData
from .config import ES12_CONFIG

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Data loader for ES12 dataset
    
    Wraps the existing ES12DataLoader and converts to new data structures
    """
    
    def __init__(self):
        """Initialize the data loader"""
        self.base_loader = BaseES12Loader()
        self._capacitor_data: Dict[str, CapacitorData] = {}
    
    def load_es12_dataset(self, data_path: Path) -> Dict[str, CapacitorData]:
        """
        Load ES12 dataset from disk
        
        Args:
            data_path: Path to ES12.mat file
            
        Returns:
            Dictionary mapping capacitor_id to CapacitorData object
            
        Raises:
            FileNotFoundError: If data file not found
            ValueError: If data cannot be parsed
        """
        logger.info(f"Loading ES12 dataset from {data_path}")
        
        # Convert to Path object if string
        if isinstance(data_path, str):
            data_path = Path(data_path)
        
        # Check if file exists
        if not data_path.exists():
            raise FileNotFoundError(f"ES12 data file not found: {data_path}")
        
        try:
            # Load using base loader
            df = self.base_loader.load_dataset(data_path)
            logger.info(f"Loaded {len(df)} cycle records from ES12 dataset")
            
            # Get raw transient data for each capacitor
            capacitor_data = {}
            
            for cap_id in ES12_CONFIG["capacitor_ids"]:
                logger.debug(f"Processing {cap_id}")
                
                # Get raw VL/VO data
                raw_data = self.base_loader.get_raw_transient_data(cap_id)
                
                if raw_data is None:
                    logger.warning(f"No raw data found for {cap_id}, skipping")
                    continue
                
                if 'VL' not in raw_data or 'VO' not in raw_data:
                    logger.warning(f"Missing VL or VO data for {cap_id}, skipping")
                    continue
                
                # Convert to CapacitorData
                cap_data = self._convert_to_capacitor_data(
                    cap_id, raw_data['VL'], raw_data['VO']
                )
                
                if cap_data is not None:
                    capacitor_data[cap_id] = cap_data
                    logger.info(
                        f"Loaded {cap_data.total_cycles} cycles for {cap_id}"
                    )
            
            if not capacitor_data:
                raise ValueError("No valid capacitor data found in dataset")
            
            self._capacitor_data = capacitor_data
            logger.info(
                f"Successfully loaded {len(capacitor_data)} capacitors"
            )
            
            return capacitor_data
            
        except Exception as e:
            logger.error(f"Failed to load ES12 dataset: {e}")
            raise ValueError(f"Failed to load ES12 dataset: {e}")
    
    def _convert_to_capacitor_data(
        self,
        cap_id: str,
        vl_data: np.ndarray,
        vo_data: np.ndarray
    ) -> Optional[CapacitorData]:
        """
        Convert raw VL/VO data to CapacitorData structure
        
        Args:
            cap_id: Capacitor identifier
            vl_data: VL data array (n_timepoints, n_cycles)
            vo_data: VO data array (n_timepoints, n_cycles)
            
        Returns:
            CapacitorData object or None if conversion fails
        """
        try:
            # VL/VO data shape: (n_timepoints, n_cycles)
            n_timepoints, n_cycles = vl_data.shape
            
            if vo_data.shape != vl_data.shape:
                logger.error(
                    f"VL and VO shape mismatch for {cap_id}: "
                    f"VL={vl_data.shape}, VO={vo_data.shape}"
                )
                return None
            
            # Create CycleData for each cycle
            cycles = []
            for cycle_idx in range(n_cycles):
                # Extract data for this cycle
                vl_series = vl_data[:, cycle_idx]
                vo_series = vo_data[:, cycle_idx]
                
                # Skip cycles with all NaN values
                if np.all(np.isnan(vl_series)) or np.all(np.isnan(vo_series)):
                    logger.debug(f"Skipping cycle {cycle_idx + 1} (all NaN)")
                    continue
                
                # Create CycleData (cycle_number is 1-based)
                try:
                    cycle_data = CycleData(
                        cycle_number=cycle_idx + 1,
                        vl_series=vl_series,
                        vo_series=vo_series,
                        timestamp=None  # Timestamps not currently used
                    )
                    cycles.append(cycle_data)
                except ValueError as e:
                    logger.warning(
                        f"Failed to create CycleData for {cap_id} "
                        f"cycle {cycle_idx + 1}: {e}"
                    )
                    continue
            
            if not cycles:
                logger.error(f"No valid cycles found for {cap_id}")
                return None
            
            # Create CapacitorData
            capacitor_data = CapacitorData(
                capacitor_id=cap_id,
                cycles=cycles,
                total_cycles=len(cycles)
            )
            
            return capacitor_data
            
        except Exception as e:
            logger.error(f"Failed to convert data for {cap_id}: {e}")
            return None
    
    def get_capacitor_cycles(self, capacitor_id: str) -> Optional[List[CycleData]]:
        """
        Get all cycles for a specific capacitor
        
        Args:
            capacitor_id: Capacitor identifier (e.g., "ES12C1")
            
        Returns:
            List of CycleData objects or None if not found
        """
        if capacitor_id not in self._capacitor_data:
            logger.warning(f"Capacitor {capacitor_id} not found in loaded data")
            return None
        
        return self._capacitor_data[capacitor_id].cycles
    
    def get_capacitor_data(self, capacitor_id: str) -> Optional[CapacitorData]:
        """
        Get CapacitorData for a specific capacitor
        
        Args:
            capacitor_id: Capacitor identifier
            
        Returns:
            CapacitorData object or None if not found
        """
        return self._capacitor_data.get(capacitor_id)
    
    @property
    def loaded_capacitors(self) -> List[str]:
        """Get list of loaded capacitor IDs"""
        return list(self._capacitor_data.keys())
    
    @property
    def is_loaded(self) -> bool:
        """Check if data has been loaded"""
        return len(self._capacitor_data) > 0


def load_es12_data(data_path: Path) -> Dict[str, CapacitorData]:
    """
    Convenience function to load ES12 dataset
    
    Args:
        data_path: Path to ES12.mat file
        
    Returns:
        Dictionary mapping capacitor_id to CapacitorData
    """
    loader = DataLoader()
    return loader.load_es12_dataset(data_path)
