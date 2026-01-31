"""
Unit tests for DataLoader
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys

# CRITICAL: Add parent project root FIRST before any true_rul imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# Then add src directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from true_rul.data_loader import DataLoader, load_es12_data
from true_rul.data_structures import CycleData, CapacitorData


class TestDataLoader:
    """Test suite for DataLoader"""
    
    @pytest.fixture
    def data_loader(self):
        """Create a DataLoader instance"""
        return DataLoader()
    
    @pytest.fixture
    def es12_data_path(self):
        """Get path to ES12 dataset"""
        # Path relative to project root
        project_root = Path(__file__).parent.parent.parent
        data_path = project_root / "data" / "raw" / "ES12.mat"
        return data_path
    
    def test_initialization(self, data_loader):
        """Test DataLoader initialization"""
        assert data_loader is not None
        assert data_loader.base_loader is not None
        assert len(data_loader._capacitor_data) == 0
        assert not data_loader.is_loaded
    
    def test_load_es12_dataset_valid(self, data_loader, es12_data_path):
        """Test loading valid ES12 dataset"""
        if not es12_data_path.exists():
            pytest.skip(f"ES12 data not found at {es12_data_path}")
        
        capacitor_data = data_loader.load_es12_dataset(es12_data_path)
        
        # Should return dictionary
        assert isinstance(capacitor_data, dict)
        
        # Should have capacitors
        assert len(capacitor_data) > 0
        
        # All values should be CapacitorData objects
        for cap_id, cap_data in capacitor_data.items():
            assert isinstance(cap_data, CapacitorData)
            assert cap_data.capacitor_id == cap_id
            assert cap_data.total_cycles > 0
            assert len(cap_data.cycles) == cap_data.total_cycles
    
    def test_load_es12_dataset_file_not_found(self, data_loader):
        """Test error handling for missing file"""
        non_existent_path = Path("/nonexistent/path/ES12.mat")
        
        with pytest.raises(FileNotFoundError, match="ES12 data file not found"):
            data_loader.load_es12_dataset(non_existent_path)
    
    def test_load_es12_dataset_invalid_file(self, data_loader):
        """Test error handling for invalid file"""
        # Create a temporary invalid file
        with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
            tmp.write(b"invalid data")
            tmp_path = Path(tmp.name)
        
        try:
            with pytest.raises(ValueError, match="Failed to load ES12 dataset"):
                data_loader.load_es12_dataset(tmp_path)
        finally:
            tmp_path.unlink()
    
    def test_load_es12_dataset_sets_loaded_flag(self, data_loader, es12_data_path):
        """Test that loading sets the is_loaded flag"""
        if not es12_data_path.exists():
            pytest.skip(f"ES12 data not found at {es12_data_path}")
        
        assert not data_loader.is_loaded
        
        data_loader.load_es12_dataset(es12_data_path)
        
        assert data_loader.is_loaded
    
    def test_load_es12_dataset_populates_internal_data(self, data_loader, es12_data_path):
        """Test that loading populates internal data structure"""
        if not es12_data_path.exists():
            pytest.skip(f"ES12 data not found at {es12_data_path}")
        
        capacitor_data = data_loader.load_es12_dataset(es12_data_path)
        
        # Internal data should match returned data
        assert len(data_loader._capacitor_data) == len(capacitor_data)
        
        for cap_id in capacitor_data.keys():
            assert cap_id in data_loader._capacitor_data
    
    def test_get_capacitor_cycles_valid(self, data_loader, es12_data_path):
        """Test getting cycles for a valid capacitor"""
        if not es12_data_path.exists():
            pytest.skip(f"ES12 data not found at {es12_data_path}")
        
        data_loader.load_es12_dataset(es12_data_path)
        
        # Get first capacitor ID
        cap_id = data_loader.loaded_capacitors[0]
        
        cycles = data_loader.get_capacitor_cycles(cap_id)
        
        assert cycles is not None
        assert isinstance(cycles, list)
        assert len(cycles) > 0
        
        # All should be CycleData objects
        for cycle in cycles:
            assert isinstance(cycle, CycleData)
    
    def test_get_capacitor_cycles_not_found(self, data_loader, es12_data_path):
        """Test getting cycles for non-existent capacitor"""
        if not es12_data_path.exists():
            pytest.skip(f"ES12 data not found at {es12_data_path}")
        
        data_loader.load_es12_dataset(es12_data_path)
        
        cycles = data_loader.get_capacitor_cycles("NONEXISTENT")
        
        assert cycles is None
    
    def test_get_capacitor_cycles_before_loading(self, data_loader):
        """Test getting cycles before loading data"""
        cycles = data_loader.get_capacitor_cycles("ES12C1")
        
        assert cycles is None
    
    def test_get_capacitor_data_valid(self, data_loader, es12_data_path):
        """Test getting CapacitorData for valid capacitor"""
        if not es12_data_path.exists():
            pytest.skip(f"ES12 data not found at {es12_data_path}")
        
        data_loader.load_es12_dataset(es12_data_path)
        
        cap_id = data_loader.loaded_capacitors[0]
        cap_data = data_loader.get_capacitor_data(cap_id)
        
        assert cap_data is not None
        assert isinstance(cap_data, CapacitorData)
        assert cap_data.capacitor_id == cap_id
    
    def test_get_capacitor_data_not_found(self, data_loader, es12_data_path):
        """Test getting CapacitorData for non-existent capacitor"""
        if not es12_data_path.exists():
            pytest.skip(f"ES12 data not found at {es12_data_path}")
        
        data_loader.load_es12_dataset(es12_data_path)
        
        cap_data = data_loader.get_capacitor_data("NONEXISTENT")
        
        assert cap_data is None
    
    def test_loaded_capacitors_property(self, data_loader, es12_data_path):
        """Test loaded_capacitors property"""
        if not es12_data_path.exists():
            pytest.skip(f"ES12 data not found at {es12_data_path}")
        
        # Before loading
        assert len(data_loader.loaded_capacitors) == 0
        
        # After loading
        data_loader.load_es12_dataset(es12_data_path)
        
        loaded = data_loader.loaded_capacitors
        assert isinstance(loaded, list)
        assert len(loaded) > 0
        
        # All should be strings
        for cap_id in loaded:
            assert isinstance(cap_id, str)
    
    def test_is_loaded_property(self, data_loader, es12_data_path):
        """Test is_loaded property"""
        if not es12_data_path.exists():
            pytest.skip(f"ES12 data not found at {es12_data_path}")
        
        # Before loading
        assert not data_loader.is_loaded
        
        # After loading
        data_loader.load_es12_dataset(es12_data_path)
        assert data_loader.is_loaded
    
    def test_data_structure_integrity_cycle_numbers(self, data_loader, es12_data_path):
        """Test that cycle numbers are sequential and 1-based"""
        if not es12_data_path.exists():
            pytest.skip(f"ES12 data not found at {es12_data_path}")
        
        capacitor_data = data_loader.load_es12_dataset(es12_data_path)
        
        for cap_id, cap_data in capacitor_data.items():
            for i, cycle in enumerate(cap_data.cycles):
                # Cycle numbers should be 1-based and sequential
                assert cycle.cycle_number == i + 1
    
    def test_data_structure_integrity_vl_vo_shapes(self, data_loader, es12_data_path):
        """Test that VL and VO have matching shapes"""
        if not es12_data_path.exists():
            pytest.skip(f"ES12 data not found at {es12_data_path}")
        
        capacitor_data = data_loader.load_es12_dataset(es12_data_path)
        
        for cap_id, cap_data in capacitor_data.items():
            for cycle in cap_data.cycles:
                # VL and VO should have same length
                assert len(cycle.vl_series) == len(cycle.vo_series)
                
                # Should be numpy arrays
                assert isinstance(cycle.vl_series, np.ndarray)
                assert isinstance(cycle.vo_series, np.ndarray)
                
                # Should not be empty
                assert len(cycle.vl_series) > 0
    
    def test_data_structure_integrity_no_all_nan_cycles(self, data_loader, es12_data_path):
        """Test that no cycles have all NaN values"""
        if not es12_data_path.exists():
            pytest.skip(f"ES12 data not found at {es12_data_path}")
        
        capacitor_data = data_loader.load_es12_dataset(es12_data_path)
        
        for cap_id, cap_data in capacitor_data.items():
            for cycle in cap_data.cycles:
                # Should not have all NaN values
                assert not np.all(np.isnan(cycle.vl_series))
                assert not np.all(np.isnan(cycle.vo_series))
    
    def test_data_structure_integrity_total_cycles_matches(self, data_loader, es12_data_path):
        """Test that total_cycles matches actual cycle count"""
        if not es12_data_path.exists():
            pytest.skip(f"ES12 data not found at {es12_data_path}")
        
        capacitor_data = data_loader.load_es12_dataset(es12_data_path)
        
        for cap_id, cap_data in capacitor_data.items():
            assert cap_data.total_cycles == len(cap_data.cycles)
    
    def test_data_structure_integrity_capacitor_ids(self, data_loader, es12_data_path):
        """Test that capacitor IDs are consistent"""
        if not es12_data_path.exists():
            pytest.skip(f"ES12 data not found at {es12_data_path}")
        
        capacitor_data = data_loader.load_es12_dataset(es12_data_path)
        
        for cap_id, cap_data in capacitor_data.items():
            # Dictionary key should match CapacitorData.capacitor_id
            assert cap_id == cap_data.capacitor_id
            
            # Capacitor ID should not be empty
            assert len(cap_id) > 0
    
    def test_convenience_function_load_es12_data(self, es12_data_path):
        """Test convenience function for loading ES12 data"""
        if not es12_data_path.exists():
            pytest.skip(f"ES12 data not found at {es12_data_path}")
        
        capacitor_data = load_es12_data(es12_data_path)
        
        assert isinstance(capacitor_data, dict)
        assert len(capacitor_data) > 0
        
        for cap_data in capacitor_data.values():
            assert isinstance(cap_data, CapacitorData)
    
    def test_multiple_loads_overwrite(self, data_loader, es12_data_path):
        """Test that loading multiple times overwrites previous data"""
        if not es12_data_path.exists():
            pytest.skip(f"ES12 data not found at {es12_data_path}")
        
        # Load first time
        data1 = data_loader.load_es12_dataset(es12_data_path)
        cap_ids_1 = set(data_loader.loaded_capacitors)
        
        # Load second time
        data2 = data_loader.load_es12_dataset(es12_data_path)
        cap_ids_2 = set(data_loader.loaded_capacitors)
        
        # Should have same capacitors
        assert cap_ids_1 == cap_ids_2
        
        # Should still be loaded
        assert data_loader.is_loaded
    
    def test_cycle_data_validation_in_loaded_data(self, data_loader, es12_data_path):
        """Test that loaded CycleData objects pass validation"""
        if not es12_data_path.exists():
            pytest.skip(f"ES12 data not found at {es12_data_path}")
        
        capacitor_data = data_loader.load_es12_dataset(es12_data_path)
        
        for cap_data in capacitor_data.values():
            for cycle in cap_data.cycles:
                # These should not raise exceptions
                assert cycle.cycle_number >= 1
                assert len(cycle.vl_series) == len(cycle.vo_series)
                assert len(cycle.vl_series) > 0
    
    def test_capacitor_data_validation_in_loaded_data(self, data_loader, es12_data_path):
        """Test that loaded CapacitorData objects pass validation"""
        if not es12_data_path.exists():
            pytest.skip(f"ES12 data not found at {es12_data_path}")
        
        capacitor_data = data_loader.load_es12_dataset(es12_data_path)
        
        for cap_data in capacitor_data.values():
            # These should not raise exceptions
            assert len(cap_data.capacitor_id) > 0
            assert cap_data.total_cycles == len(cap_data.cycles)
            
            # Cycles should be ordered
            for i, cycle in enumerate(cap_data.cycles):
                assert cycle.cycle_number == i + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
