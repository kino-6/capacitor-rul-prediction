"""
ES10/ES14 Data Structure Analysis Script

This script analyzes the structure of ES10.mat and ES14.mat files and compares them with ES12.
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path

def analyze_mat_structure(filepath):
    """Analyze the structure of a .mat file"""
    print(f"\n{'='*80}")
    print(f"Analyzing: {filepath.name}")
    print('='*80)
    
    with h5py.File(filepath, 'r') as f:
        def print_structure(name, obj, indent=0):
            """Recursively print HDF5 structure"""
            prefix = "  " * indent
            if isinstance(obj, h5py.Dataset):
                print(f"{prefix}📄 {name}: shape={obj.shape}, dtype={obj.dtype}")
            elif isinstance(obj, h5py.Group):
                print(f"{prefix}📁 {name}/")
        
        print("\n🔍 File Structure:")
        f.visititems(print_structure)
        
        # Get root level keys
        root_keys = list(f.keys())
        print(f"\n📋 Root Level Keys: {root_keys}")
        
        # Analyze each capacitor
        capacitor_info = []
        
        for key in root_keys:
            if key.startswith('ES'):
                print(f"\n{'─'*80}")
                print(f"Capacitor: {key}")
                print('─'*80)
                
                cap_group = f[key]
                
                # Get all datasets in this capacitor
                datasets = {}
                def collect_datasets(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        datasets[name] = obj
                
                cap_group.visititems(collect_datasets)
                
                print(f"  Datasets found: {len(datasets)}")
                
                # Analyze each dataset
                for ds_name, ds in datasets.items():
                    print(f"    • {ds_name}:")
                    print(f"        Shape: {ds.shape}")
                    print(f"        Dtype: {ds.dtype}")
                    print(f"        Size: {ds.size} elements")
                    
                    # Try to get some sample data
                    if ds.size > 0 and ds.size < 1000000:  # Only for reasonably sized datasets
                        try:
                            data = ds[:]
                            if data.ndim == 1:
                                print(f"        Range: [{np.min(data):.4f}, {np.max(data):.4f}]")
                            elif data.ndim == 2:
                                print(f"        Dimensions: {data.shape[0]} x {data.shape[1]}")
                                if data.shape[0] > 0 and data.shape[1] > 0:
                                    print(f"        Sample (first row): {data[0, :min(5, data.shape[1])]}")
                        except Exception as e:
                            print(f"        (Could not read data: {e})")
                
                # Count cycles
                cycle_count = 0
                if 'cycle' in cap_group:
                    cycle_group = cap_group['cycle']
                    cycle_count = len([k for k in cycle_group.keys() if k.startswith('cycle')])
                
                capacitor_info.append({
                    'capacitor': key,
                    'num_datasets': len(datasets),
                    'num_cycles': cycle_count,
                    'datasets': list(datasets.keys())
                })
        
        return capacitor_info

def compare_datasets(es10_info, es12_info, es14_info):
    """Compare the structure of ES10, ES12, and ES14 datasets"""
    print("\n" + "="*80)
    print("DATASET COMPARISON")
    print("="*80)
    
    # Create comparison table
    comparison = []
    
    for dataset_name in ['ES10', 'ES12', 'ES14']:
        if dataset_name == 'ES10':
            info_list = es10_info
        elif dataset_name == 'ES12':
            info_list = es12_info
        else:
            info_list = es14_info
        
        num_capacitors = len(info_list)
        total_cycles = sum([cap['num_cycles'] for cap in info_list])
        avg_cycles = total_cycles / num_capacitors if num_capacitors > 0 else 0
        
        # Get common dataset names
        if info_list:
            common_datasets = set(info_list[0]['datasets'])
            for cap in info_list[1:]:
                common_datasets &= set(cap['datasets'])
        else:
            common_datasets = set()
        
        comparison.append({
            'Dataset': dataset_name,
            'Capacitors': num_capacitors,
            'Total Cycles': total_cycles,
            'Avg Cycles/Cap': f"{avg_cycles:.1f}",
            'Common Datasets': len(common_datasets)
        })
    
    comp_df = pd.DataFrame(comparison)
    print("\n📊 Summary Comparison:")
    print(comp_df.to_string(index=False))
    
    # Find common features across all datasets
    print("\n🔍 Common Features Analysis:")
    
    if es10_info and es12_info and es14_info:
        es10_datasets = set(es10_info[0]['datasets'])
        es12_datasets = set(es12_info[0]['datasets'])
        es14_datasets = set(es14_info[0]['datasets'])
        
        common_all = es10_datasets & es12_datasets & es14_datasets
        print(f"\n  Features common to ALL datasets: {len(common_all)}")
        for feat in sorted(common_all):
            print(f"    • {feat}")
        
        es12_only = es12_datasets - es10_datasets - es14_datasets
        if es12_only:
            print(f"\n  Features unique to ES12: {len(es12_only)}")
            for feat in sorted(es12_only):
                print(f"    • {feat}")
        
        es10_only = es10_datasets - es12_datasets - es14_datasets
        if es10_only:
            print(f"\n  Features unique to ES10: {len(es10_only)}")
            for feat in sorted(es10_only):
                print(f"    • {feat}")
        
        es14_only = es14_datasets - es12_datasets - es10_datasets
        if es14_only:
            print(f"\n  Features unique to ES14: {len(es14_only)}")
            for feat in sorted(es14_only):
                print(f"    • {feat}")
    
    return comp_df

def generate_structure_report(es10_info, es12_info, es14_info, comp_df):
    """Generate a detailed structure comparison report"""
    print("\n" + "="*80)
    print("GENERATING STRUCTURE COMPARISON REPORT")
    print("="*80)
    
    report = f"""# ES10/ES14 Data Structure Analysis Report

## 📅 Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Dataset Overview

### Summary Comparison

{comp_df.to_string(index=False)}

## 🔍 Detailed Analysis

### ES10 Dataset

"""
    
    if es10_info:
        report += f"""
- **Number of Capacitors**: {len(es10_info)}
- **Capacitor IDs**: {', '.join([cap['capacitor'] for cap in es10_info])}
- **Total Cycles**: {sum([cap['num_cycles'] for cap in es10_info])}
- **Average Cycles per Capacitor**: {sum([cap['num_cycles'] for cap in es10_info]) / len(es10_info):.1f}

**Available Datasets per Capacitor**:
"""
        if es10_info:
            for ds in sorted(es10_info[0]['datasets']):
                report += f"- {ds}\n"
    else:
        report += "\n⚠️ ES10 data not found or could not be analyzed.\n"
    
    report += """

### ES12 Dataset

"""
    
    if es12_info:
        report += f"""
- **Number of Capacitors**: {len(es12_info)}
- **Capacitor IDs**: {', '.join([cap['capacitor'] for cap in es12_info])}
- **Total Cycles**: {sum([cap['num_cycles'] for cap in es12_info])}
- **Average Cycles per Capacitor**: {sum([cap['num_cycles'] for cap in es12_info]) / len(es12_info):.1f}

**Available Datasets per Capacitor**:
"""
        if es12_info:
            for ds in sorted(es12_info[0]['datasets']):
                report += f"- {ds}\n"
    else:
        report += "\n⚠️ ES12 data not found or could not be analyzed.\n"
    
    report += """

### ES14 Dataset

"""
    
    if es14_info:
        report += f"""
- **Number of Capacitors**: {len(es14_info)}
- **Capacitor IDs**: {', '.join([cap['capacitor'] for cap in es14_info])}
- **Total Cycles**: {sum([cap['num_cycles'] for cap in es14_info])}
- **Average Cycles per Capacitor**: {sum([cap['num_cycles'] for cap in es14_info]) / len(es14_info):.1f}

**Available Datasets per Capacitor**:
"""
        if es14_info:
            for ds in sorted(es14_info[0]['datasets']):
                report += f"- {ds}\n"
    else:
        report += "\n⚠️ ES14 data not found or could not be analyzed.\n"
    
    # Common features analysis
    report += """

## 🔗 Cross-Dataset Compatibility

### Common Features

"""
    
    if es10_info and es12_info and es14_info:
        es10_datasets = set(es10_info[0]['datasets'])
        es12_datasets = set(es12_info[0]['datasets'])
        es14_datasets = set(es14_info[0]['datasets'])
        
        common_all = es10_datasets & es12_datasets & es14_datasets
        report += f"\n**Features available in ALL datasets** ({len(common_all)}):\n"
        for feat in sorted(common_all):
            report += f"- {feat}\n"
        
        report += "\n### Dataset-Specific Features\n"
        
        es12_only = es12_datasets - es10_datasets - es14_datasets
        if es12_only:
            report += f"\n**ES12 Only** ({len(es12_only)}):\n"
            for feat in sorted(es12_only):
                report += f"- {feat}\n"
        
        es10_only = es10_datasets - es12_datasets - es14_datasets
        if es10_only:
            report += f"\n**ES10 Only** ({len(es10_only)}):\n"
            for feat in sorted(es10_only):
                report += f"- {feat}\n"
        
        es14_only = es14_datasets - es12_datasets - es10_datasets
        if es14_only:
            report += f"\n**ES14 Only** ({len(es14_only)}):\n"
            for feat in sorted(es14_only):
                report += f"- {feat}\n"
    
    report += """

## 💡 Recommendations

### Data Compatibility

1. **Use Common Features**: Focus on features available in all datasets for cross-dataset training
2. **Feature Extraction Strategy**: Ensure the same 30 features can be extracted from all datasets
3. **Data Normalization**: Account for potential differences in measurement scales

### Next Steps

1. **Task 6.5**: Implement unified data loader for ES10/ES14
2. **Task 6.5**: Extract features from ES10/ES14 using the same pipeline
3. **Task 6.6**: Validate ES12 models on ES10/ES14 datasets

---

**Report Generated by**: Kiro AI Agent  
**Status**: Phase 2.5 - Data Structure Analysis Complete
"""
    
    # Save report
    output_dir = Path("docs")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "es10_es14_structure_analysis.md"
    
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✓ Saved: {report_path}")
    return report

def main():
    """Main execution function"""
    print("="*80)
    print("ES10/ES14 DATA STRUCTURE ANALYSIS")
    print("="*80)
    
    # Define file paths
    data_dir = Path("../data/raw")
    es10_path = data_dir / "ES10.mat"
    es12_path = data_dir / "ES12.mat"
    es14_path = data_dir / "ES14.mat"
    
    # Analyze each dataset
    es10_info = []
    es12_info = []
    es14_info = []
    
    if es10_path.exists():
        es10_info = analyze_mat_structure(es10_path)
    else:
        print(f"\n⚠️ ES10.mat not found at {es10_path}")
    
    if es12_path.exists():
        es12_info = analyze_mat_structure(es12_path)
    else:
        print(f"\n⚠️ ES12.mat not found at {es12_path}")
    
    if es14_path.exists():
        es14_info = analyze_mat_structure(es14_path)
    else:
        print(f"\n⚠️ ES14.mat not found at {es14_path}")
    
    # Compare datasets
    if es10_info or es12_info or es14_info:
        comp_df = compare_datasets(es10_info, es12_info, es14_info)
        
        # Generate report
        report = generate_structure_report(es10_info, es12_info, es14_info, comp_df)
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"\nGenerated: docs/es10_es14_structure_analysis.md")
        print("\nNext: Proceed to Task 6.5 (Unified Data Loader Implementation)")
    else:
        print("\n⚠️ No datasets could be analyzed. Please check file paths.")

if __name__ == "__main__":
    main()
