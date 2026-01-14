#!/usr/bin/env python3
"""
Detailed analysis of ES12.mat file structure and content.

This script provides a deeper examination of the actual data content
to understand the measurement structure and data organization.
"""

import sys
from pathlib import Path
import numpy as np
import h5py

def analyze_eis_data(file_path):
    """Analyze EIS (Electrochemical Impedance Spectroscopy) data structure"""
    print("=== EIS Data Analysis ===")
    
    with h5py.File(str(file_path), 'r') as f:
        # Look at the EIS reference table first
        ref_table = f['ES12/EIS_Data/EIS_Reference_Table']
        print(f"EIS Reference Table shape: {ref_table.shape}")
        
        # Examine one capacitor's EIS data in detail
        cap_name = 'ES12C1'
        print(f"\n=== Analyzing {cap_name} EIS Data ===")
        
        eis_path = f'ES12/EIS_Data/{cap_name}/EIS_Measurement'
        
        # Get column names
        col_names_ref = f[f'{eis_path}/ColumNames']
        print(f"Column names reference shape: {col_names_ref.shape}")
        
        # Try to read some column names
        try:
            # Column names are stored as object references
            for i in range(min(10, col_names_ref.shape[0])):
                ref = col_names_ref[i, 0]
                if ref:
                    try:
                        col_name = f[ref][()]
                        if isinstance(col_name, np.ndarray):
                            # Convert from uint16 to string
                            col_str = ''.join(chr(x) for x in col_name.flatten() if x != 0)
                            print(f"  Column {i}: {col_str}")
                    except Exception as e:
                        print(f"  Column {i}: Could not read ({e})")
        except Exception as e:
            print(f"Could not read column names: {e}")
        
        # Get data references
        data_refs = f[f'{eis_path}/Data']
        print(f"\nData references shape: {data_refs.shape}")
        
        # Try to read some actual data
        try:
            for i in range(min(5, data_refs.shape[0])):
                ref = data_refs[i, 0]
                if ref:
                    try:
                        data = f[ref][()]
                        print(f"  Data {i} shape: {data.shape}, dtype: {data.dtype}")
                        if data.size < 20:
                            print(f"    Sample: {data.flatten()[:10]}")
                    except Exception as e:
                        print(f"  Data {i}: Could not read ({e})")
        except Exception as e:
            print(f"Could not read data: {e}")

def analyze_transient_data(file_path):
    """Analyze Transient data structure"""
    print("\n=== Transient Data Analysis ===")
    
    with h5py.File(str(file_path), 'r') as f:
        # Look at serial date first
        serial_date = f['ES12/Transient_Data/Serial_Date']
        print(f"Serial Date shape: {serial_date.shape}")
        print(f"Serial Date dtype: {serial_date.dtype}")
        
        # Sample some dates
        dates = serial_date[:]
        print(f"Date range: {dates.min()} to {dates.max()}")
        print(f"First 5 dates: {dates[:5].flatten()}")
        
        # Examine one capacitor's transient data
        cap_name = 'ES12C1'
        print(f"\n=== Analyzing {cap_name} Transient Data ===")
        
        vl_data = f[f'ES12/Transient_Data/{cap_name}/VL']
        vo_data = f[f'ES12/Transient_Data/{cap_name}/VO']
        
        print(f"VL data shape: {vl_data.shape}, dtype: {vl_data.dtype}")
        print(f"VO data shape: {vo_data.shape}, dtype: {vo_data.dtype}")
        
        # Sample some data
        vl_sample = vl_data[:5, :5]
        vo_sample = vo_data[:5, :5]
        
        print(f"VL sample (first 5x5):")
        print(vl_sample)
        print(f"VO sample (first 5x5):")
        print(vo_sample)
        
        # Check data ranges
        print(f"VL range: {vl_data[:].min()} to {vl_data[:].max()}")
        print(f"VO range: {vo_data[:].min()} to {vo_data[:].max()}")

def analyze_all_capacitors(file_path):
    """Analyze data availability for all capacitors"""
    print("\n=== All Capacitors Analysis ===")
    
    with h5py.File(str(file_path), 'r') as f:
        capacitors = ['ES12C1', 'ES12C2', 'ES12C3', 'ES12C4', 
                     'ES12C5', 'ES12C6', 'ES12C7', 'ES12C8']
        
        print("EIS Data availability:")
        for cap in capacitors:
            eis_path = f'ES12/EIS_Data/{cap}/EIS_Measurement'
            if eis_path in f:
                data_refs = f[f'{eis_path}/Data']
                print(f"  {cap}: {data_refs.shape[0]} measurements")
            else:
                print(f"  {cap}: No EIS data")
        
        print("\nTransient Data availability:")
        for cap in capacitors:
            vl_path = f'ES12/Transient_Data/{cap}/VL'
            vo_path = f'ES12/Transient_Data/{cap}/VO'
            if vl_path in f and vo_path in f:
                vl_shape = f[vl_path].shape
                vo_shape = f[vo_path].shape
                print(f"  {cap}: VL {vl_shape}, VO {vo_shape}")
            else:
                print(f"  {cap}: No transient data")

def main():
    file_path = Path("data/raw/ES12.mat")
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    print(f"Detailed analysis of: {file_path}")
    print(f"File size: {file_path.stat().st_size / (1024*1024):.2f} MB")
    
    try:
        analyze_eis_data(file_path)
        analyze_transient_data(file_path)
        analyze_all_capacitors(file_path)
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()