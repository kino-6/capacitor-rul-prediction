"""
Parallel Batch Processing for RUL Predictions
High-performance parallel processing using multiprocessing and threading
"""

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Callable, Tuple
import numpy as np
import time
import logging
from tqdm import tqdm
from dataclasses import dataclass

from .data_structures import CycleData, PredictionResult
from .rul_predictor import RULPredictor

logger = logging.getLogger(__name__)

@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing"""
    max_workers: int = min(8, mp.cpu_count())
    chunk_size: int = 10
    use_threading: bool = True  # True for I/O bound, False for CPU bound
    show_progress: bool = True
    timeout_per_item: float = 5.0

class ParallelBatchProcessor:
    """
    High-performance parallel batch processor for RUL predictions
    
    Uses both threading and multiprocessing for optimal performance
    """
    
    def __init__(self, config: Optional[BatchProcessingConfig] = None):
        """Initialize parallel processor"""
        self.config = config or BatchProcessingConfig()
        self.predictor: Optional[RULPredictor] = None
        
        logger.info(f"Parallel processor initialized: {self.config.max_workers} workers")
    
    def set_predictor(self, predictor: RULPredictor) -> None:
        """Set the RUL predictor instance"""
        self.predictor = predictor
    
    def process_batch_parallel(
        self,
        cycle_data_list: List[CycleData],
        capacitor_ids: List[str],
        cycle_histories: Optional[List[List[CycleData]]] = None
    ) -> Tuple[List[PredictionResult], List[Exception]]:
        """
        Process batch of predictions in parallel
        
        Args:
            cycle_data_list: List of cycle data to process
            capacitor_ids: List of capacitor IDs
            cycle_histories: Optional list of cycle histories
            
        Returns:
            Tuple of (successful_results, errors)
        """
        if not self.predictor:
            raise RuntimeError("Predictor not set. Call set_predictor() first.")
        
        if len(cycle_data_list) != len(capacitor_ids):
            raise ValueError("cycle_data_list and capacitor_ids must have same length")
        
        start_time = time.time()
        
        # Prepare input data
        if cycle_histories is None:
            cycle_histories = [None] * len(cycle_data_list)
        
        inputs = list(zip(cycle_data_list, capacitor_ids, cycle_histories))
        
        # Choose executor based on configuration
        executor_class = ThreadPoolExecutor if self.config.use_threading else ProcessPoolExecutor
        
        results = []
        errors = []
        
        with executor_class(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_input = {
                executor.submit(self._process_single_item, inp): inp 
                for inp in inputs
            }
            
            # Process results with progress bar
            if self.config.show_progress:
                futures = tqdm(
                    as_completed(future_to_input, timeout=self.config.timeout_per_item * len(inputs)),
                    total=len(inputs),
                    desc="Processing predictions",
                    unit="pred"
                )
            else:
                futures = as_completed(future_to_input, timeout=self.config.timeout_per_item * len(inputs))
            
            for future in futures:
                try:
                    result = future.result(timeout=self.config.timeout_per_item)
                    results.append(result)
                except Exception as e:
                    errors.append(e)
                    logger.error(f"Prediction failed: {e}")
        
        total_time = time.time() - start_time
        success_rate = len(results) / len(inputs) * 100
        
        logger.info(
            f"Batch processing complete: {len(results)}/{len(inputs)} successful "
            f"({success_rate:.1f}%) in {total_time:.2f}s"
        )
        
        return results, errors
    
    def _process_single_item(
        self, 
        input_data: Tuple[CycleData, str, Optional[List[CycleData]]]
    ) -> PredictionResult:
        """Process a single prediction item"""
        cycle_data, capacitor_id, cycle_history = input_data
        
        return self.predictor.predict_with_error_handling(
            cycle_data=cycle_data,
            capacitor_id=capacitor_id,
            cycle_history=cycle_history
        )
    
    def process_batch_chunked(
        self,
        cycle_data_list: List[CycleData],
        capacitor_ids: List[str],
        cycle_histories: Optional[List[List[CycleData]]] = None
    ) -> Tuple[List[PredictionResult], List[Exception]]:
        """
        Process batch in chunks for memory efficiency
        
        Args:
            cycle_data_list: List of cycle data to process
            capacitor_ids: List of capacitor IDs
            cycle_histories: Optional list of cycle histories
            
        Returns:
            Tuple of (successful_results, errors)
        """
        if not self.predictor:
            raise RuntimeError("Predictor not set. Call set_predictor() first.")
        
        total_items = len(cycle_data_list)
        chunk_size = self.config.chunk_size
        
        all_results = []
        all_errors = []
        
        # Process in chunks
        for i in tqdm(range(0, total_items, chunk_size), desc="Processing chunks", disable=not self.config.show_progress):
            end_idx = min(i + chunk_size, total_items)
            
            chunk_cycle_data = cycle_data_list[i:end_idx]
            chunk_capacitor_ids = capacitor_ids[i:end_idx]
            chunk_histories = cycle_histories[i:end_idx] if cycle_histories else None
            
            # Process chunk in parallel
            chunk_results, chunk_errors = self.process_batch_parallel(
                chunk_cycle_data,
                chunk_capacitor_ids,
                chunk_histories
            )
            
            all_results.extend(chunk_results)
            all_errors.extend(chunk_errors)
        
        return all_results, all_errors
    
    def benchmark_processing_speed(
        self,
        n_samples: int = 100,
        n_features: int = 55
    ) -> Dict[str, float]:
        """
        Benchmark processing speed with synthetic data
        
        Args:
            n_samples: Number of samples to process
            n_features: Number of features per sample
            
        Returns:
            Benchmark results dictionary
        """
        logger.info(f"Running benchmark with {n_samples} samples...")
        
        # Generate synthetic data
        synthetic_data = []
        capacitor_ids = []
        
        for i in range(n_samples):
            cycle_data = CycleData(
                cycle_number=i + 1,
                vl_series=np.random.randn(100),
                vo_series=np.random.randn(100),
                timestamp=time.time()
            )
            synthetic_data.append(cycle_data)
            capacitor_ids.append(f"BENCH_C{i % 10}")  # 10 different capacitors
        
        # Benchmark different configurations
        results = {}
        
        # Sequential processing
        start_time = time.time()
        sequential_results = []
        for cycle_data, cap_id in zip(synthetic_data[:10], capacitor_ids[:10]):  # Small sample
            try:
                result = self.predictor.predict_with_error_handling(cycle_data, cap_id)
                sequential_results.append(result)
            except Exception as e:
                logger.error(f"Sequential processing error: {e}")
        
        sequential_time = time.time() - start_time
        results["sequential_time_per_item"] = sequential_time / len(sequential_results) if sequential_results else float('inf')
        
        # Parallel processing (threading)
        self.config.use_threading = True
        start_time = time.time()
        parallel_results, parallel_errors = self.process_batch_parallel(
            synthetic_data[:50], capacitor_ids[:50]  # Larger sample
        )
        parallel_time = time.time() - start_time
        results["parallel_threading_time_per_item"] = parallel_time / len(parallel_results) if parallel_results else float('inf')
        results["parallel_threading_success_rate"] = len(parallel_results) / 50
        
        # Calculate speedup
        if results["sequential_time_per_item"] > 0:
            results["speedup_factor"] = results["sequential_time_per_item"] / results["parallel_threading_time_per_item"]
        else:
            results["speedup_factor"] = 1.0
        
        results["max_workers"] = self.config.max_workers
        results["cpu_cores"] = mp.cpu_count()
        
        logger.info(f"Benchmark complete: {results['speedup_factor']:.2f}x speedup")
        
        return results

# Utility functions for common batch processing patterns
def process_capacitor_batch(
    predictor: RULPredictor,
    capacitor_data: Dict[str, List[CycleData]],
    max_workers: int = 8,
    show_progress: bool = True
) -> Dict[str, List[PredictionResult]]:
    """
    Process multiple capacitors in parallel
    
    Args:
        predictor: RUL predictor instance
        capacitor_data: Dictionary mapping capacitor_id to list of cycles
        max_workers: Maximum number of parallel workers
        show_progress: Whether to show progress bar
        
    Returns:
        Dictionary mapping capacitor_id to prediction results
    """
    config = BatchProcessingConfig(
        max_workers=max_workers,
        show_progress=show_progress,
        use_threading=True
    )
    
    processor = ParallelBatchProcessor(config)
    processor.set_predictor(predictor)
    
    results = {}
    
    for capacitor_id, cycles in tqdm(capacitor_data.items(), desc="Processing capacitors", disable=not show_progress):
        capacitor_ids = [capacitor_id] * len(cycles)
        
        cycle_results, errors = processor.process_batch_parallel(
            cycles, capacitor_ids
        )
        
        results[capacitor_id] = cycle_results
        
        if errors:
            logger.warning(f"Capacitor {capacitor_id}: {len(errors)} errors out of {len(cycles)} cycles")
    
    return results

def process_time_series_batch(
    predictor: RULPredictor,
    time_series_data: List[Tuple[str, List[CycleData]]],
    max_workers: int = 8,
    chunk_size: int = 20
) -> List[Tuple[str, List[PredictionResult]]]:
    """
    Process time series data for multiple capacitors
    
    Args:
        predictor: RUL predictor instance
        time_series_data: List of (capacitor_id, cycles) tuples
        max_workers: Maximum number of parallel workers
        chunk_size: Chunk size for processing
        
    Returns:
        List of (capacitor_id, results) tuples
    """
    config = BatchProcessingConfig(
        max_workers=max_workers,
        chunk_size=chunk_size,
        use_threading=True,
        show_progress=True
    )
    
    processor = ParallelBatchProcessor(config)
    processor.set_predictor(predictor)
    
    results = []
    
    for capacitor_id, cycles in time_series_data:
        # Create cycle histories for temporal context
        cycle_histories = []
        for i, cycle in enumerate(cycles):
            history = cycles[:i] if i > 0 else None
            cycle_histories.append(history)
        
        capacitor_ids = [capacitor_id] * len(cycles)
        
        cycle_results, errors = processor.process_batch_chunked(
            cycles, capacitor_ids, cycle_histories
        )
        
        results.append((capacitor_id, cycle_results))
        
        if errors:
            logger.warning(f"Time series {capacitor_id}: {len(errors)} errors")
    
    return results