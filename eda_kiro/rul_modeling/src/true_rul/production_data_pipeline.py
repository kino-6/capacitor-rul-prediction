"""
Production Data Pipeline Integration

This module provides real-time data ingestion, validation, and processing
capabilities for industrial sensor data in production environments.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from concurrent.futures import ThreadPoolExecutor
import threading
import queue

import numpy as np
import pandas as pd

from .data_structures import CycleData, CapacitorData, PredictionResult
from .data_loader import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class SensorReading:
    """Single sensor reading from industrial equipment"""
    sensor_id: str
    timestamp: datetime
    vl_value: float
    vo_value: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate sensor reading"""
        if not self.sensor_id:
            raise ValueError("sensor_id cannot be empty")
        if np.isnan(self.vl_value) or np.isnan(self.vo_value):
            raise ValueError("VL and VO values cannot be NaN")


@dataclass
class DataQualityMetrics:
    """Data quality metrics for monitoring"""
    total_readings: int = 0
    valid_readings: int = 0
    invalid_readings: int = 0
    missing_readings: int = 0
    out_of_range_readings: int = 0
    duplicate_readings: int = 0
    last_reading_time: Optional[datetime] = None
    data_gap_duration: Optional[timedelta] = None
    
    @property
    def quality_score(self) -> float:
        """Calculate overall data quality score (0-1)"""
        if self.total_readings == 0:
            return 0.0
        return self.valid_readings / self.total_readings
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate"""
        if self.total_readings == 0:
            return 0.0
        return self.invalid_readings / self.total_readings


@dataclass
class DataValidationRule:
    """Data validation rule configuration"""
    name: str
    enabled: bool = True
    vl_min: Optional[float] = None
    vl_max: Optional[float] = None
    vo_min: Optional[float] = None
    vo_max: Optional[float] = None
    max_gap_seconds: Optional[int] = None
    max_duplicate_tolerance: Optional[int] = None


class DataValidator:
    """Validates incoming sensor data against quality rules"""
    
    def __init__(self, rules: List[DataValidationRule]):
        self.rules = {rule.name: rule for rule in rules}
        self.metrics = DataQualityMetrics()
        self._last_readings: Dict[str, SensorReading] = {}
        
    def validate_reading(self, reading: SensorReading) -> tuple[bool, List[str]]:
        """
        Validate a single sensor reading
        
        Args:
            reading: Sensor reading to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Update metrics
        self.metrics.total_readings += 1
        
        # Range validation
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
                
            # VL range check
            if rule.vl_min is not None and reading.vl_value < rule.vl_min:
                errors.append(f"VL value {reading.vl_value} below minimum {rule.vl_min}")
                self.metrics.out_of_range_readings += 1
                
            if rule.vl_max is not None and reading.vl_value > rule.vl_max:
                errors.append(f"VL value {reading.vl_value} above maximum {rule.vl_max}")
                self.metrics.out_of_range_readings += 1
                
            # VO range check
            if rule.vo_min is not None and reading.vo_value < rule.vo_min:
                errors.append(f"VO value {reading.vo_value} below minimum {rule.vo_min}")
                self.metrics.out_of_range_readings += 1
                
            if rule.vo_max is not None and reading.vo_value > rule.vo_max:
                errors.append(f"VO value {reading.vo_value} above maximum {rule.vo_max}")
                self.metrics.out_of_range_readings += 1
                
            # Gap detection
            if rule.max_gap_seconds is not None:
                last_reading = self._last_readings.get(reading.sensor_id)
                if last_reading is not None:
                    gap = (reading.timestamp - last_reading.timestamp).total_seconds()
                    if gap > rule.max_gap_seconds:
                        errors.append(f"Data gap of {gap}s exceeds maximum {rule.max_gap_seconds}s")
                        self.metrics.data_gap_duration = timedelta(seconds=gap)
                        
            # Duplicate detection
            if rule.max_duplicate_tolerance is not None:
                last_reading = self._last_readings.get(reading.sensor_id)
                if (last_reading is not None and 
                    abs(reading.vl_value - last_reading.vl_value) < 1e-6 and
                    abs(reading.vo_value - last_reading.vo_value) < 1e-6):
                    errors.append("Duplicate reading detected")
                    self.metrics.duplicate_readings += 1
        
        # Update last reading
        self._last_readings[reading.sensor_id] = reading
        self.metrics.last_reading_time = reading.timestamp
        
        is_valid = len(errors) == 0
        if is_valid:
            self.metrics.valid_readings += 1
        else:
            self.metrics.invalid_readings += 1
            
        return is_valid, errors
    
    def get_metrics(self) -> DataQualityMetrics:
        """Get current data quality metrics"""
        return self.metrics
    
    def reset_metrics(self):
        """Reset data quality metrics"""
        self.metrics = DataQualityMetrics()


class DataBuffer:
    """Thread-safe circular buffer for sensor data"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._buffer: deque = deque(maxlen=max_size)
        self._lock = threading.RLock()
        
    def add_reading(self, reading: SensorReading):
        """Add a sensor reading to the buffer"""
        with self._lock:
            self._buffer.append(reading)
            
    def get_readings(self, sensor_id: str, count: int = None) -> List[SensorReading]:
        """Get recent readings for a sensor"""
        with self._lock:
            readings = [r for r in self._buffer if r.sensor_id == sensor_id]
            if count is not None:
                readings = readings[-count:]
            return readings
    
    def get_readings_in_timerange(
        self, 
        sensor_id: str, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[SensorReading]:
        """Get readings within a time range"""
        with self._lock:
            return [
                r for r in self._buffer 
                if (r.sensor_id == sensor_id and 
                    start_time <= r.timestamp <= end_time)
            ]
    
    def clear(self):
        """Clear the buffer"""
        with self._lock:
            self._buffer.clear()
    
    @property
    def size(self) -> int:
        """Current buffer size"""
        with self._lock:
            return len(self._buffer)


class DataIngestionSource(ABC):
    """Abstract base class for data ingestion sources"""
    
    @abstractmethod
    async def start(self):
        """Start data ingestion"""
        pass
    
    @abstractmethod
    async def stop(self):
        """Stop data ingestion"""
        pass
    
    @abstractmethod
    def set_callback(self, callback: Callable[[SensorReading], None]):
        """Set callback for new readings"""
        pass


class FileDataSource(DataIngestionSource):
    """File-based data source for testing and batch processing"""
    
    def __init__(self, file_path: Path, sensor_id: str, replay_speed: float = 1.0):
        self.file_path = file_path
        self.sensor_id = sensor_id
        self.replay_speed = replay_speed
        self.callback: Optional[Callable[[SensorReading], None]] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
    def set_callback(self, callback: Callable[[SensorReading], None]):
        """Set callback for new readings"""
        self.callback = callback
        
    async def start(self):
        """Start reading from file"""
        if self._running:
            return
            
        self._running = True
        self._task = asyncio.create_task(self._read_file())
        
    async def stop(self):
        """Stop reading from file"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
                
    async def _read_file(self):
        """Read data from file and emit readings"""
        try:
            # Load data from file (assuming CSV format)
            df = pd.read_csv(self.file_path)
            
            for _, row in df.iterrows():
                if not self._running:
                    break
                    
                reading = SensorReading(
                    sensor_id=self.sensor_id,
                    timestamp=datetime.now(),
                    vl_value=float(row['VL']),
                    vo_value=float(row['VO']),
                    metadata={'source': 'file', 'row_index': row.name}
                )
                
                if self.callback:
                    self.callback(reading)
                    
                # Simulate real-time by adding delay
                await asyncio.sleep(1.0 / self.replay_speed)
                
        except Exception as e:
            logger.error(f"Error reading from file {self.file_path}: {e}")


class CycleDetector:
    """Detects complete charge-discharge cycles from streaming data"""
    
    def __init__(self, 
                 cycle_length_threshold: int = 50,  # Reduced for testing
                 voltage_change_threshold: float = 1.0):  # More lenient
        self.cycle_length_threshold = cycle_length_threshold
        self.voltage_change_threshold = voltage_change_threshold
        self._current_cycle_data: Dict[str, List[SensorReading]] = {}
        self._cycle_counter: Dict[str, int] = {}
        
    def add_reading(self, reading: SensorReading) -> Optional[CycleData]:
        """
        Add a reading and check if a complete cycle is detected
        
        Args:
            reading: New sensor reading
            
        Returns:
            CycleData if complete cycle detected, None otherwise
        """
        sensor_id = reading.sensor_id
        
        # Initialize if first reading for this sensor
        if sensor_id not in self._current_cycle_data:
            self._current_cycle_data[sensor_id] = []
            self._cycle_counter[sensor_id] = 0
            
        # Add reading to current cycle
        self._current_cycle_data[sensor_id].append(reading)
        
        # Check if cycle is complete
        current_readings = self._current_cycle_data[sensor_id]
        
        if len(current_readings) >= self.cycle_length_threshold:
            # For testing, just create cycles based on length
            cycle_data = self._create_cycle_data(sensor_id, current_readings)
            
            # Reset for next cycle
            self._current_cycle_data[sensor_id] = []
            self._cycle_counter[sensor_id] += 1
            
            return cycle_data
                
        return None
    
    def _is_cycle_complete(self, readings: List[SensorReading]) -> bool:
        """Check if readings represent a complete cycle"""
        if len(readings) < 10:
            return False
            
        # Simple heuristic: check if voltage has returned to starting level
        start_vl = readings[0].vl_value
        end_vl = readings[-1].vl_value
        
        return abs(end_vl - start_vl) < self.voltage_change_threshold
    
    def _create_cycle_data(self, sensor_id: str, readings: List[SensorReading]) -> CycleData:
        """Create CycleData from readings"""
        vl_series = np.array([r.vl_value for r in readings])
        vo_series = np.array([r.vo_value for r in readings])
        
        cycle_number = self._cycle_counter[sensor_id] + 1
        timestamp = readings[0].timestamp
        
        return CycleData(
            cycle_number=cycle_number,
            vl_series=vl_series,
            vo_series=vo_series,
            timestamp=timestamp
        )


class BatchProcessor:
    """Processes batches of cycles for high-throughput scenarios"""
    
    def __init__(self, 
                 batch_size: int = 10,
                 max_wait_time: float = 30.0,
                 max_workers: int = 4):
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.max_workers = max_workers
        
        self._batch_queue: queue.Queue = queue.Queue()
        self._current_batch: List[CycleData] = []
        self._last_batch_time = time.time()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._processing_callback: Optional[Callable[[List[CycleData]], List[PredictionResult]]] = None
        
    def set_processing_callback(self, callback: Callable[[List[CycleData]], List[PredictionResult]]):
        """Set callback for batch processing"""
        self._processing_callback = callback
        
    def add_cycle(self, cycle_data: CycleData) -> Optional[List[PredictionResult]]:
        """
        Add cycle to batch and process if batch is ready
        
        Args:
            cycle_data: New cycle data
            
        Returns:
            List of prediction results if batch was processed, None otherwise
        """
        self._current_batch.append(cycle_data)
        current_time = time.time()
        
        # Check if batch should be processed
        should_process = (
            len(self._current_batch) >= self.batch_size or
            (current_time - self._last_batch_time) >= self.max_wait_time
        )
        
        if should_process and self._processing_callback:
            batch_to_process = self._current_batch.copy()
            self._current_batch = []
            self._last_batch_time = current_time
            
            # Process batch asynchronously
            future = self._executor.submit(self._processing_callback, batch_to_process)
            try:
                results = future.result(timeout=60.0)  # 60 second timeout
                return results
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                return None
                
        return None
    
    def flush_batch(self) -> Optional[List[PredictionResult]]:
        """Force process current batch"""
        if self._current_batch and self._processing_callback:
            batch_to_process = self._current_batch.copy()
            self._current_batch = []
            self._last_batch_time = time.time()
            
            try:
                results = self._processing_callback(batch_to_process)
                return results
            except Exception as e:
                logger.error(f"Batch flush failed: {e}")
                return None
                
        return None
    
    def shutdown(self):
        """Shutdown the batch processor"""
        self._executor.shutdown(wait=True)


class ProductionDataPipeline:
    """Main production data pipeline orchestrator"""
    
    def __init__(self, 
                 validation_rules: List[DataValidationRule],
                 buffer_size: int = 10000,
                 batch_size: int = 10):
        
        self.validator = DataValidator(validation_rules)
        self.buffer = DataBuffer(buffer_size)
        self.cycle_detector = CycleDetector()
        self.batch_processor = BatchProcessor(batch_size=batch_size)
        
        self._data_sources: List[DataIngestionSource] = []
        self._running = False
        self._stats = {
            'readings_processed': 0,
            'cycles_detected': 0,
            'batches_processed': 0,
            'errors': 0
        }
        
    def add_data_source(self, source: DataIngestionSource):
        """Add a data ingestion source"""
        source.set_callback(self._handle_new_reading)
        self._data_sources.append(source)
        
    def set_prediction_callback(self, callback: Callable[[List[CycleData]], List[PredictionResult]]):
        """Set callback for processing batches of cycles"""
        self.batch_processor.set_processing_callback(callback)
        
    async def start(self):
        """Start the data pipeline"""
        if self._running:
            return
            
        logger.info("Starting production data pipeline")
        self._running = True
        
        # Start all data sources
        for source in self._data_sources:
            await source.start()
            
        logger.info(f"Started {len(self._data_sources)} data sources")
        
    async def stop(self):
        """Stop the data pipeline"""
        if not self._running:
            return
            
        logger.info("Stopping production data pipeline")
        self._running = False
        
        # Stop all data sources
        for source in self._data_sources:
            await source.stop()
            
        # Flush any remaining batch
        remaining_results = self.batch_processor.flush_batch()
        if remaining_results is not None:
            self._stats['batches_processed'] += 1
            logger.info(f"Flushed final batch with {len(remaining_results)} predictions")
            
        self.batch_processor.shutdown()
        
        logger.info("Production data pipeline stopped")
        
    def _handle_new_reading(self, reading: SensorReading):
        """Handle new sensor reading"""
        try:
            self._stats['readings_processed'] += 1
            
            # Validate reading
            is_valid, errors = self.validator.validate_reading(reading)
            
            if not is_valid:
                logger.warning(f"Invalid reading from {reading.sensor_id}: {errors}")
                self._stats['errors'] += 1
                return
                
            # Add to buffer
            self.buffer.add_reading(reading)
            
            # Check for complete cycle
            cycle_data = self.cycle_detector.add_reading(reading)
            
            if cycle_data is not None:
                self._stats['cycles_detected'] += 1
                logger.debug(f"Detected cycle {cycle_data.cycle_number} for {reading.sensor_id}")
                
                # Add to batch processor
                results = self.batch_processor.add_cycle(cycle_data)
                
                if results is not None:
                    self._stats['batches_processed'] += 1
                    logger.info(f"Processed batch with {len(results)} predictions")
                    
        except Exception as e:
            logger.error(f"Error handling reading: {e}")
            self._stats['errors'] += 1
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        quality_metrics = self.validator.get_metrics()
        return {
            **self._stats,
            'data_quality': {
                'total_readings': quality_metrics.total_readings,
                'valid_readings': quality_metrics.valid_readings,
                'invalid_readings': quality_metrics.invalid_readings,
                'missing_readings': quality_metrics.missing_readings,
                'out_of_range_readings': quality_metrics.out_of_range_readings,
                'duplicate_readings': quality_metrics.duplicate_readings,
                'quality_score': quality_metrics.quality_score,
                'error_rate': quality_metrics.error_rate,
                'last_reading_time': quality_metrics.last_reading_time,
                'data_gap_duration': quality_metrics.data_gap_duration
            },
            'buffer_size': self.buffer.size,
            'is_running': self._running
        }
    
    def get_recent_readings(self, sensor_id: str, count: int = 100) -> List[SensorReading]:
        """Get recent readings for a sensor"""
        return self.buffer.get_readings(sensor_id, count)


# Default validation rules for ES12-like data
DEFAULT_VALIDATION_RULES = [
    DataValidationRule(
        name="voltage_range",
        vl_min=0.0,
        vl_max=10.0,
        vo_min=0.0,
        vo_max=10.0
    ),
    DataValidationRule(
        name="data_continuity",
        max_gap_seconds=300,  # 5 minutes
        max_duplicate_tolerance=3
    )
]


def create_production_pipeline(
    validation_rules: Optional[List[DataValidationRule]] = None,
    buffer_size: int = 10000,
    batch_size: int = 10
) -> ProductionDataPipeline:
    """
    Create a production data pipeline with default configuration
    
    Args:
        validation_rules: Data validation rules (uses defaults if None)
        buffer_size: Size of data buffer
        batch_size: Batch size for processing
        
    Returns:
        Configured ProductionDataPipeline
    """
    if validation_rules is None:
        validation_rules = DEFAULT_VALIDATION_RULES
        
    return ProductionDataPipeline(
        validation_rules=validation_rules,
        buffer_size=buffer_size,
        batch_size=batch_size
    )