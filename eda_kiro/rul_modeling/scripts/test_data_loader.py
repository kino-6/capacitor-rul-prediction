#!/usr/bin/env python3
"""
Quick test script for DataLoader
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from true_rul.data_loader import DataLoader
from true_rul.config import setup_logging, DATA_DIR

def main():
    """Test data loading"""
    setup_logging("test_data_loader.log")
    
    # Path to ES12 data
    data_path = Path("../data/raw/ES12.mat")
    
    if not data_path.exists():
        print(f"❌ ES12 data file not found: {data_path}")
        print("Please ensure ES12.mat is in data/raw/ directory")
        return
    
    print("🔄 Loading ES12 dataset...")
    loader = DataLoader()
    
    try:
        capacitor_data = loader.load_es12_dataset(data_path)
        
        print(f"✅ Successfully loaded {len(capacitor_data)} capacitors")
        print()
        
        # Display summary for each capacitor
        for cap_id, cap_data in capacitor_data.items():
            print(f"📊 {cap_id}:")
            print(f"   Total cycles: {cap_data.total_cycles}")
            
            # Check first cycle
            first_cycle = cap_data.get_cycle(1)
            if first_cycle:
                print(f"   First cycle VL shape: {first_cycle.vl_series.shape}")
                print(f"   First cycle VO shape: {first_cycle.vo_series.shape}")
            
            # Check last cycle
            last_cycle = cap_data.get_cycle(cap_data.total_cycles)
            if last_cycle:
                print(f"   Last cycle number: {last_cycle.cycle_number}")
            
            print()
        
        print("✅ Data loader test passed!")
        
    except Exception as e:
        print(f"❌ Data loader test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
