"""
Parallel feature extraction for multiple capacitors.

This module provides functionality to extract features from multiple capacitors
in parallel using multiprocessing, with progress tracking.
"""

import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import time

from ..feature_extraction.extractor import CycleFeatureExtractor
from ..utils.data_loader import load_es12_cycle_data, get_available_capacitors


class ParallelFeatureExtractor:
    """Extract features from multiple capacitors in parallel."""
    
    def __init__(
        self,
        es12_path: str,
        n_processes: Optional[int] = None,
        include_history: bool = False
    ):
        """
        Initialize the parallel feature extractor.
        
        Args:
            es12_path: Path to ES12.mat file
            n_processes: Number of parallel processes (default: CPU count)
            include_history: Whether to include historical features (default: False)
        """
        self.es12_path = es12_path
        self.n_processes = n_processes or mp.cpu_count()
        self.include_history = include_history
        
        # Verify file exists
        if not Path(es12_path).exists():
            raise FileNotFoundError(f"ES12 data file not found: {es12_path}")
    
    def extract_capacitor_features(
        self,
        capacitor_id: str,
        max_cycles: int = 200,
        progress_interval: int = 20
    ) -> pd.DataFrame:
        """
        Extract features from all cycles of a single capacitor.
        
        Args:
            capacitor_id: Capacitor ID (e.g., "ES12C1")
            max_cycles: Maximum number of cycles to process
            progress_interval: Print progress every N cycles
        
        Returns:
            DataFrame with features for all cycles
        """
        extractor = CycleFeatureExtractor()
        features_list = []
        
        print(f"[{capacitor_id}] Starting feature extraction...")
        start_time = time.time()
        
        for cycle in range(1, max_cycles + 1):
            try:
                # Load cycle data
                vl, vo = load_es12_cycle_data(self.es12_path, capacitor_id, cycle)
                
                # Extract features (without history for Phase 1)
                if self.include_history and len(features_list) > 0:
                    history_df = pd.DataFrame(features_list)
                    features = extractor.extract_all_features(
                        vl, vo, cycle, history_df=history_df, total_cycles=max_cycles
                    )
                else:
                    # Extract without history
                    features = {}
                    features.update(extractor.extract_basic_stats(vl, vo))
                    features.update(extractor.extract_degradation_indicators(vl, vo))
                    features.update(extractor.extract_time_series_features(vl, vo))
                    features.update(extractor.extract_cycle_info(cycle, max_cycles))
                
                # Add metadata
                features['capacitor_id'] = capacitor_id
                features['cycle'] = cycle
                
                features_list.append(features)
                
                # Progress reporting
                if cycle % progress_interval == 0 or cycle == max_cycles:
                    elapsed = time.time() - start_time
                    progress_pct = (cycle / max_cycles) * 100
                    print(f"[{capacitor_id}] Cycle {cycle}/{max_cycles} "
                          f"({progress_pct:.1f}%) - Elapsed: {elapsed:.1f}s")
            
            except Exception as e:
                print(f"[{capacitor_id}] Error at cycle {cycle}: {e}")
                continue
        
        total_time = time.time() - start_time
        print(f"[{capacitor_id}] Completed in {total_time:.1f}s "
              f"({len(features_list)} cycles extracted)")
        
        return pd.DataFrame(features_list)
    
    def extract_all_capacitors(
        self,
        capacitor_ids: Optional[List[str]] = None,
        max_cycles: int = 200,
        progress_interval: int = 20
    ) -> pd.DataFrame:
        """
        Extract features from multiple capacitors in parallel.
        
        Args:
            capacitor_ids: List of capacitor IDs (default: all ES12 capacitors)
            max_cycles: Maximum number of cycles to process per capacitor
            progress_interval: Print progress every N cycles
        
        Returns:
            DataFrame with features from all capacitors
        """
        if capacitor_ids is None:
            capacitor_ids = get_available_capacitors()
        
        print(f"\n{'='*70}")
        print(f"Parallel Feature Extraction")
        print(f"{'='*70}")
        print(f"Capacitors: {len(capacitor_ids)} ({', '.join(capacitor_ids)})")
        print(f"Cycles per capacitor: {max_cycles}")
        print(f"Processes: {self.n_processes}")
        print(f"Include history: {self.include_history}")
        print(f"{'='*70}\n")
        
        overall_start = time.time()
        
        # Create a pool of worker processes
        with mp.Pool(processes=self.n_processes) as pool:
            # Create tasks for each capacitor
            tasks = [
                (cap_id, max_cycles, progress_interval)
                for cap_id in capacitor_ids
            ]
            
            # Use starmap to pass multiple arguments
            results = pool.starmap(
                self._extract_capacitor_wrapper,
                tasks
            )
        
        # Combine all results
        all_features = pd.concat(results, ignore_index=True)
        
        overall_time = time.time() - overall_start
        
        print(f"\n{'='*70}")
        print(f"Extraction Complete!")
        print(f"{'='*70}")
        print(f"Total samples: {len(all_features)}")
        print(f"Total features: {len(all_features.columns)}")
        print(f"Total time: {overall_time:.1f}s ({overall_time/60:.1f} minutes)")
        print(f"Average time per capacitor: {overall_time/len(capacitor_ids):.1f}s")
        print(f"{'='*70}\n")
        
        return all_features
    
    def _extract_capacitor_wrapper(
        self,
        capacitor_id: str,
        max_cycles: int,
        progress_interval: int
    ) -> pd.DataFrame:
        """
        Wrapper method for multiprocessing.
        
        This is needed because multiprocessing can't pickle instance methods directly.
        """
        return self.extract_capacitor_features(
            capacitor_id,
            max_cycles,
            progress_interval
        )
    
    def save_features(
        self,
        features_df: pd.DataFrame,
        output_path: str,
        include_metadata: bool = True
    ) -> None:
        """
        Save extracted features to CSV.
        
        Args:
            features_df: DataFrame with extracted features
            output_path: Path to save CSV file
            include_metadata: Whether to include metadata columns
        """
        # Create output directory if it doesn't exist
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Reorder columns: metadata first, then features
        if include_metadata:
            metadata_cols = ['capacitor_id', 'cycle']
            feature_cols = [col for col in features_df.columns if col not in metadata_cols]
            ordered_cols = metadata_cols + sorted(feature_cols)
            features_df = features_df[ordered_cols]
        
        # Save to CSV
        features_df.to_csv(output_path, index=False)
        print(f"Features saved to: {output_path}")
        print(f"Shape: {features_df.shape}")


def extract_es12_features(
    es12_path: str,
    output_path: str,
    capacitor_ids: Optional[List[str]] = None,
    max_cycles: int = 200,
    n_processes: Optional[int] = None,
    include_history: bool = False,
    progress_interval: int = 20
) -> pd.DataFrame:
    """
    Convenience function to extract features from ES12 dataset.
    
    Args:
        es12_path: Path to ES12.mat file
        output_path: Path to save features CSV
        capacitor_ids: List of capacitor IDs (default: all ES12 capacitors)
        max_cycles: Maximum number of cycles to process
        n_processes: Number of parallel processes (default: CPU count)
        include_history: Whether to include historical features
        progress_interval: Print progress every N cycles
    
    Returns:
        DataFrame with extracted features
    """
    extractor = ParallelFeatureExtractor(
        es12_path=es12_path,
        n_processes=n_processes,
        include_history=include_history
    )
    
    features_df = extractor.extract_all_capacitors(
        capacitor_ids=capacitor_ids,
        max_cycles=max_cycles,
        progress_interval=progress_interval
    )
    
    extractor.save_features(features_df, output_path)
    
    return features_df
