#!/usr/bin/env python3
"""
Script to analyze the structure of ES12.mat file.

This script examines the internal structure of the ES12.mat file to understand:
- File format (MATLAB v7.0 vs v7.3/HDF5)
- Data hierarchy and organization
- Capacitor data layout (ES12C1-ES12C8)
- EIS data structure
- Measurement parameters (frequencies, cycles, etc.)
"""

import sys
from pathlib import Path
import numpy as np

def analyze_with_scipy(file_path):
    """Analyze using scipy.io.loadmat (for MATLAB v7.0 files)"""
    try:
        from scipy.io import loadmat
        print("=== Analyzing with scipy.io.loadmat ===")
        
        mat_data = loadmat(str(file_path))
        
        print(f"Top-level keys: {list(mat_data.keys())}")
        
        # Filter out metadata keys
        data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        print(f"Data keys: {data_keys}")
        
        for key in data_keys:
            data = mat_data[key]
            print(f"\nKey: {key}")
            print(f"  Type: {type(data)}")
            print(f"  Shape: {getattr(data, 'shape', 'N/A')}")
            print(f"  Dtype: {getattr(data, 'dtype', 'N/A')}")
            
            if hasattr(data, 'dtype') and data.dtype.names:
                print(f"  Field names: {data.dtype.names}")
                
        return True
        
    except NotImplementedError as e:
        print(f"scipy.io.loadmat failed: {e}")
        print("This is likely a MATLAB v7.3 file, trying h5py...")
        return False
    except Exception as e:
        print(f"Error with scipy: {e}")
        return False

def analyze_with_h5py(file_path):
    """Analyze using h5py (for MATLAB v7.3/HDF5 files)"""
    try:
        import h5py
        print("\n=== Analyzing with h5py (MATLAB v7.3/HDF5) ===")
        
        with h5py.File(str(file_path), 'r') as f:
            print(f"Root keys: {list(f.keys())}")
            
            def print_structure(name, obj, level=0):
                indent = "  " * level
                if isinstance(obj, h5py.Group):
                    print(f"{indent}Group: {name}")
                    print(f"{indent}  Keys: {list(obj.keys())}")
                elif isinstance(obj, h5py.Dataset):
                    print(f"{indent}Dataset: {name}")
                    print(f"{indent}  Shape: {obj.shape}")
                    print(f"{indent}  Dtype: {obj.dtype}")
                    print(f"{indent}  Size: {obj.size}")
                    
                    # Show a sample of the data if it's small enough
                    if obj.size < 100 and obj.size > 0:
                        try:
                            sample = obj[()]
                            if isinstance(sample, np.ndarray) and sample.size < 20:
                                print(f"{indent}  Sample: {sample}")
                            elif not isinstance(sample, np.ndarray):
                                print(f"{indent}  Value: {sample}")
                        except Exception as e:
                            print(f"{indent}  Could not read sample: {e}")
            
            # Print the full structure
            print("\n=== Full Structure ===")
            f.visititems(print_structure)
            
            # Look for capacitor-specific data
            print("\n=== Looking for Capacitor Data ===")
            capacitor_keys = []
            
            def find_capacitor_data(name, obj):
                if 'ES12C' in name or 'C1' in name or 'C2' in name:
                    capacitor_keys.append(name)
                    print(f"Found capacitor-related data: {name}")
                    if isinstance(obj, h5py.Dataset):
                        print(f"  Shape: {obj.shape}, Dtype: {obj.dtype}")
            
            f.visititems(find_capacitor_data)
            
            if not capacitor_keys:
                print("No obvious capacitor-specific keys found. Checking all datasets...")
                
                def check_all_datasets(name, obj):
                    if isinstance(obj, h5py.Dataset) and obj.size > 100:
                        print(f"Large dataset: {name}")
                        print(f"  Shape: {obj.shape}, Dtype: {obj.dtype}, Size: {obj.size}")
                
                f.visititems(check_all_datasets)
        
        return True
        
    except ImportError:
        print("h5py not available")
        return False
    except Exception as e:
        print(f"Error with h5py: {e}")
        return False

def main():
    file_path = Path("data/raw/ES12.mat")
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        sys.exit(1)
    
    print(f"Analyzing file: {file_path}")
    print(f"File size: {file_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Try scipy first, then h5py
    success = analyze_with_scipy(file_path)
    if not success:
        success = analyze_with_h5py(file_path)
    
    if not success:
        print("Could not analyze the file with either method")
        sys.exit(1)

if __name__ == "__main__":
    main()