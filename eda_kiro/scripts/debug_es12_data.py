#!/usr/bin/env python3
"""
Debug ES12 data structure to understand the actual waveform data
"""

import h5py
import numpy as np
from pathlib import Path

def debug_es12_structure():
    """Debug the ES12 data structure"""
    data_path = Path("data/raw/ES12.mat")
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        return
    
    with h5py.File(data_path, 'r') as f:
        print("🔍 ES12 Data Structure Analysis")
        print("=" * 50)
        
        # Check transient data structure
        transient_group = f['ES12']['Transient_Data']
        print(f"📊 Transient Data Group Keys: {list(transient_group.keys())}")
        
        # Check serial dates
        serial_dates = transient_group['Serial_Date'][:]
        print(f"📅 Serial Dates Shape: {serial_dates.shape}")
        print(f"📅 Serial Date Range: {serial_dates.min()} to {serial_dates.max()}")
        print(f"📅 First 5 dates: {serial_dates[:5]}")
        
        # Check ES12C4 data
        cap_group = transient_group['ES12C4']
        print(f"\n🔋 ES12C4 Keys: {list(cap_group.keys())}")
        
        vl_data = cap_group['VL'][:]
        vo_data = cap_group['VO'][:]
        
        print(f"📈 VL Data Shape: {vl_data.shape}")
        print(f"📈 VO Data Shape: {vo_data.shape}")
        print(f"📈 VL Data Type: {vl_data.dtype}")
        print(f"📈 VO Data Type: {vo_data.dtype}")
        
        # Check for NaN values
        vl_nan_count = np.sum(np.isnan(vl_data))
        vo_nan_count = np.sum(np.isnan(vo_data))
        print(f"🚫 VL NaN count: {vl_nan_count} / {vl_data.size}")
        print(f"🚫 VO NaN count: {vo_nan_count} / {vo_data.size}")
        
        # Check first cycle data
        print(f"\n🔍 First Cycle Analysis:")
        vl_cycle1 = vl_data[:, 0]  # First cycle
        vo_cycle1 = vo_data[:, 0]
        
        print(f"📊 VL Cycle 1 - Min: {np.nanmin(vl_cycle1):.6f}, Max: {np.nanmax(vl_cycle1):.6f}")
        print(f"📊 VL Cycle 1 - Mean: {np.nanmean(vl_cycle1):.6f}, Std: {np.nanstd(vl_cycle1):.6f}")
        print(f"📊 VO Cycle 1 - Min: {np.nanmin(vo_cycle1):.6f}, Max: {np.nanmax(vo_cycle1):.6f}")
        print(f"📊 VO Cycle 1 - Mean: {np.nanmean(vo_cycle1):.6f}, Std: {np.nanstd(vo_cycle1):.6f}")
        
        # Check valid data points
        vl_valid = np.sum(~np.isnan(vl_cycle1))
        vo_valid = np.sum(~np.isnan(vo_cycle1))
        print(f"✅ VL Valid points: {vl_valid} / {len(vl_cycle1)}")
        print(f"✅ VO Valid points: {vo_valid} / {len(vo_cycle1)}")
        
        # Sample some actual values
        print(f"\n📋 Sample VL values (first 10): {vl_cycle1[:10]}")
        print(f"📋 Sample VO values (first 10): {vo_cycle1[:10]}")
        
        # Check different cycles
        print(f"\n🔄 Cycle Comparison:")
        for cycle_idx in [0, 99, 199, 299]:  # Cycles 1, 100, 200, 300
            if cycle_idx < vl_data.shape[1]:
                vl_cycle = vl_data[:, cycle_idx]
                vo_cycle = vo_data[:, cycle_idx]
                
                vl_mean = np.nanmean(vl_cycle)
                vo_mean = np.nanmean(vo_cycle)
                ratio = vo_mean / vl_mean if vl_mean != 0 else np.nan
                
                print(f"  Cycle {cycle_idx + 1}: VL={vl_mean:.6f}, VO={vo_mean:.6f}, Ratio={ratio:.6f}")

if __name__ == "__main__":
    debug_es12_structure()