"""
Comprehensive Testing Framework for RUL Prediction System

This module provides automated regression testing, performance benchmarking,
stress testing, and validation testing capabilities.
"""

import json
import time
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import gc

from .data_loader import DataLoader
from .feature_extractor import FeatureExtractor
from .rul_regression_model import RULRegressionModel
from .ensemble_anomaly_detector import EnsembleAnomalyDetector
from .rul_predictor import RULPredictor
from .model_evaluator import ModelEvaluator


@dataclass
class TestResult:
    """Test result data structure"""
    test_name: str
    status: str  # "passed", "failed", "error"
    execution_time: float
    memory_usage: float
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    details: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkResult:
    """Performance benchmark result"""
    test_name: str
    execution_time: float
    memory_usage: float
    throughput: float  # samples per second
    cpu_usage: float
    baseline_comparison: Optional[float] = None  # % change from baseline


@dataclass
class StressTestResult:
    """Stress test result"""
    test_name: str
    load_level: str
    success_rate: float
    avg_response_time: float
    max_response_time: float
    memory_peak: float
    cpu_peak: float
    errors: List[str]


class RegressionTester:
    """Automated regression testing for model updates"""
    
    def __init__(self, baseline_path: str, output_dir: str = "test_results"):
        self.baseline_path = Path(baseline_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for regression tests"""
        logger = logging.getLogger("regression_tester")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.output_dir / "regression_tests.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def load_baseline_results(self) -> Dict[str, Any]:
        """Load baseline test results"""
        if not self.baseline_path.exists():
            self.logger.warning(f"Baseline file not found: {self.baseline_path}")
            return {}
            
        with open(self.baseline_path, 'r') as f:
            return json.load(f)
            
    def save_baseline_results(self, results: Dict[str, Any]):
        """Save current results as new baseline"""
        with open(self.baseline_path, 'w') as f:
            json.dump(results, f, indent=2)
            
    def run_model_accuracy_regression(self, model_path: str, test_data_path: str) -> TestResult:
        """Test model accuracy regression"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            # Load test data
            data_loader = DataLoader()
            test_data = data_loader.load_es12_dataset(test_data_path)
            
            # Load model
            predictor = RULPredictor()
            predictor.load_models(model_path)
            
            # Run predictions
            predictions = []
            actuals = []
            
            for cap_id, cap_data in test_data.items():
                for cycle in cap_data.cycles:
                    if cycle.cycle_number > 10:  # Skip training cycles
                        pred_result = predictor.predict(cycle.vl_series, cycle.vo_series)
                        predictions.append(pred_result.rul_cycles)
                        actuals.append(cap_data.total_cycles - cycle.cycle_number)
            
            # Calculate metrics
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
            mae = np.mean(np.abs(predictions - actuals))
            r2 = 1 - np.sum((actuals - predictions) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2)
            
            metrics = {
                "rmse": float(rmse),
                "mae": float(mae),
                "r2": float(r2),
                "n_samples": len(predictions)
            }
            
            execution_time = time.time() - start_time
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 - start_memory
            
            return TestResult(
                test_name="model_accuracy_regression",
                status="passed",
                execution_time=execution_time,
                memory_usage=memory_usage,
                metrics=metrics
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 - start_memory
            
            return TestResult(
                test_name="model_accuracy_regression",
                status="error",
                execution_time=execution_time,
                memory_usage=memory_usage,
                error_message=str(e)
            )
            
    def run_fpr_regression(self, model_path: str, test_data_path: str) -> TestResult:
        """Test FPR regression"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            # Load test data
            data_loader = DataLoader()
            test_data = data_loader.load_es12_dataset(test_data_path)
            
            # Load model
            predictor = RULPredictor()
            predictor.load_models(model_path)
            
            # Test on normal cycles (1-10)
            normal_predictions = []
            
            for cap_id, cap_data in test_data.items():
                for cycle in cap_data.cycles[:10]:  # First 10 cycles are normal
                    pred_result = predictor.predict(cycle.vl_series, cycle.vo_series)
                    normal_predictions.append(pred_result.anomaly_flag)
            
            # Calculate FPR
            false_positives = sum(normal_predictions)
            total_normal = len(normal_predictions)
            fpr = false_positives / total_normal if total_normal > 0 else 0
            
            metrics = {
                "fpr": float(fpr),
                "false_positives": false_positives,
                "total_normal": total_normal
            }
            
            execution_time = time.time() - start_time
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 - start_memory
            
            status = "passed" if fpr < 0.05 else "failed"
            
            return TestResult(
                test_name="fpr_regression",
                status=status,
                execution_time=execution_time,
                memory_usage=memory_usage,
                metrics=metrics
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 - start_memory
            
            return TestResult(
                test_name="fpr_regression",
                status="error",
                execution_time=execution_time,
                memory_usage=memory_usage,
                error_message=str(e)
            )
            
    def compare_with_baseline(self, current_results: Dict[str, TestResult]) -> Dict[str, Dict[str, float]]:
        """Compare current results with baseline"""
        baseline = self.load_baseline_results()
        comparisons = {}
        
        for test_name, result in current_results.items():
            if test_name in baseline and result.metrics:
                baseline_metrics = baseline[test_name].get("metrics", {})
                current_metrics = result.metrics
                
                comparison = {}
                for metric, value in current_metrics.items():
                    if metric in baseline_metrics:
                        baseline_value = baseline_metrics[metric]
                        if baseline_value != 0:
                            change = ((value - baseline_value) / baseline_value) * 100
                            comparison[f"{metric}_change_pct"] = change
                        comparison[f"{metric}_baseline"] = baseline_value
                        comparison[f"{metric}_current"] = value
                
                comparisons[test_name] = comparison
                
        return comparisons


class PerformanceBenchmarker:
    """Performance benchmarking and comparison tools"""
    
    def __init__(self, output_dir: str = "benchmark_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for benchmarks"""
        logger = logging.getLogger("performance_benchmarker")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.output_dir / "benchmarks.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def benchmark_prediction_latency(self, model_path: str, n_samples: int = 1000) -> BenchmarkResult:
        """Benchmark prediction latency"""
        # Load model
        predictor = RULPredictor()
        predictor.load_models(model_path)
        
        # Generate synthetic test data
        np.random.seed(42)
        vl_data = np.random.randn(n_samples, 100)
        vo_data = np.random.randn(n_samples, 100)
        
        # Warm up
        for i in range(10):
            predictor.predict(vl_data[i], vo_data[i])
            
        # Benchmark
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        cpu_percent_start = psutil.cpu_percent()
        
        for i in range(n_samples):
            predictor.predict(vl_data[i], vo_data[i])
            
        execution_time = time.time() - start_time
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 - start_memory
        cpu_usage = psutil.cpu_percent() - cpu_percent_start
        throughput = n_samples / execution_time
        
        return BenchmarkResult(
            test_name="prediction_latency",
            execution_time=execution_time,
            memory_usage=memory_usage,
            throughput=throughput,
            cpu_usage=cpu_usage
        )
        
    def benchmark_batch_processing(self, model_path: str, batch_sizes: List[int]) -> List[BenchmarkResult]:
        """Benchmark batch processing performance"""
        results = []
        
        predictor = RULPredictor()
        predictor.load_models(model_path)
        
        for batch_size in batch_sizes:
            # Generate test data
            np.random.seed(42)
            vl_batch = [np.random.randn(100) for _ in range(batch_size)]
            vo_batch = [np.random.randn(100) for _ in range(batch_size)]
            
            # Benchmark
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            cpu_percent_start = psutil.cpu_percent()
            
            # Process batch
            batch_results = []
            for vl, vo in zip(vl_batch, vo_batch):
                result = predictor.predict(vl, vo)
                batch_results.append(result)
                
            execution_time = time.time() - start_time
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 - start_memory
            cpu_usage = psutil.cpu_percent() - cpu_percent_start
            throughput = batch_size / execution_time
            
            results.append(BenchmarkResult(
                test_name=f"batch_processing_size_{batch_size}",
                execution_time=execution_time,
                memory_usage=memory_usage,
                throughput=throughput,
                cpu_usage=cpu_usage
            ))
            
        return results
        
    def benchmark_memory_usage(self, model_path: str) -> BenchmarkResult:
        """Benchmark memory usage patterns"""
        gc.collect()  # Clean up before test
        
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_time = time.time()
        
        # Load model
        predictor = RULPredictor()
        predictor.load_models(model_path)
        
        model_load_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Run predictions
        np.random.seed(42)
        for i in range(100):
            vl = np.random.randn(100)
            vo = np.random.randn(100)
            predictor.predict(vl, vo)
            
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        execution_time = time.time() - start_time
        
        return BenchmarkResult(
            test_name="memory_usage",
            execution_time=execution_time,
            memory_usage=final_memory - start_memory,
            throughput=100 / execution_time,
            cpu_usage=0,  # Not measured for this test
            baseline_comparison=None
        )


class StressTester:
    """Stress testing for high-load scenarios"""
    
    def __init__(self, output_dir: str = "stress_test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for stress tests"""
        logger = logging.getLogger("stress_tester")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.output_dir / "stress_tests.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def concurrent_load_test(self, model_path: str, n_threads: int = 10, 
                           requests_per_thread: int = 100) -> StressTestResult:
        """Test concurrent load handling"""
        predictor = RULPredictor()
        predictor.load_models(model_path)
        
        # Generate test data
        np.random.seed(42)
        test_data = [(np.random.randn(100), np.random.randn(100)) 
                     for _ in range(n_threads * requests_per_thread)]
        
        results = []
        errors = []
        response_times = []
        
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_time = time.time()
        
        def worker(thread_data):
            thread_results = []
            thread_errors = []
            
            for vl, vo in thread_data:
                request_start = time.time()
                try:
                    result = predictor.predict(vl, vo)
                    response_time = time.time() - request_start
                    thread_results.append((result, response_time))
                except Exception as e:
                    thread_errors.append(str(e))
                    
            return thread_results, thread_errors
            
        # Split data among threads
        chunk_size = len(test_data) // n_threads
        thread_data_chunks = [test_data[i:i + chunk_size] 
                             for i in range(0, len(test_data), chunk_size)]
        
        # Run concurrent requests
        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            futures = [executor.submit(worker, chunk) for chunk in thread_data_chunks]
            
            for future in as_completed(futures):
                thread_results, thread_errors = future.result()
                results.extend(thread_results)
                errors.extend(thread_errors)
                response_times.extend([rt for _, rt in thread_results])
                
        execution_time = time.time() - start_time
        memory_peak = psutil.Process().memory_info().rss / 1024 / 1024
        
        success_rate = len(results) / (len(results) + len(errors)) if (len(results) + len(errors)) > 0 else 0
        avg_response_time = np.mean(response_times) if response_times else 0
        max_response_time = np.max(response_times) if response_times else 0
        
        return StressTestResult(
            test_name="concurrent_load_test",
            load_level=f"{n_threads}_threads_{requests_per_thread}_requests",
            success_rate=success_rate,
            avg_response_time=avg_response_time,
            max_response_time=max_response_time,
            memory_peak=memory_peak - start_memory,
            cpu_peak=psutil.cpu_percent(),
            errors=errors[:10]  # Keep only first 10 errors
        )
        
    def memory_stress_test(self, model_path: str, max_concurrent: int = 1000) -> StressTestResult:
        """Test memory usage under stress"""
        predictor = RULPredictor()
        predictor.load_models(model_path)
        
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_time = time.time()
        
        # Generate large amount of test data
        np.random.seed(42)
        test_data = [(np.random.randn(100), np.random.randn(100)) 
                     for _ in range(max_concurrent)]
        
        results = []
        errors = []
        memory_samples = []
        
        for i, (vl, vo) in enumerate(test_data):
            try:
                result = predictor.predict(vl, vo)
                results.append(result)
                
                # Sample memory usage every 100 requests
                if i % 100 == 0:
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    memory_samples.append(current_memory)
                    
            except Exception as e:
                errors.append(str(e))
                
        execution_time = time.time() - start_time
        memory_peak = max(memory_samples) if memory_samples else start_memory
        
        success_rate = len(results) / len(test_data)
        avg_response_time = execution_time / len(test_data)
        
        return StressTestResult(
            test_name="memory_stress_test",
            load_level=f"{max_concurrent}_concurrent_requests",
            success_rate=success_rate,
            avg_response_time=avg_response_time,
            max_response_time=avg_response_time,  # Not measured individually
            memory_peak=memory_peak - start_memory,
            cpu_peak=psutil.cpu_percent(),
            errors=errors[:10]
        )


class ValidationTester:
    """Validation testing with synthetic and real-world data"""
    
    def __init__(self, output_dir: str = "validation_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for validation tests"""
        logger = logging.getLogger("validation_tester")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.output_dir / "validation_tests.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def generate_synthetic_data(self, n_capacitors: int = 5, 
                              cycles_per_capacitor: int = 200) -> Dict[str, Any]:
        """Generate synthetic capacitor data for testing"""
        synthetic_data = {}
        
        np.random.seed(42)
        
        for cap_id in range(n_capacitors):
            cycles = []
            
            for cycle_num in range(1, cycles_per_capacitor + 1):
                # Generate degrading voltage patterns
                degradation_factor = 1 - (cycle_num / cycles_per_capacitor) * 0.3
                noise_level = 0.01 + (cycle_num / cycles_per_capacitor) * 0.05
                
                # VL series (input voltage)
                vl_base = np.sin(np.linspace(0, 4*np.pi, 100)) * degradation_factor
                vl_noise = np.random.normal(0, noise_level, 100)
                vl_series = vl_base + vl_noise
                
                # VO series (output voltage) - more degraded
                vo_base = np.sin(np.linspace(0, 4*np.pi, 100)) * degradation_factor * 0.8
                vo_noise = np.random.normal(0, noise_level * 1.2, 100)
                vo_series = vo_base + vo_noise
                
                cycles.append({
                    'cycle_number': cycle_num,
                    'vl_series': vl_series,
                    'vo_series': vo_series,
                    'timestamp': time.time() + cycle_num
                })
                
            synthetic_data[f"synthetic_C{cap_id}"] = {
                'capacitor_id': f"synthetic_C{cap_id}",
                'cycles': cycles,
                'total_cycles': cycles_per_capacitor
            }
            
        return synthetic_data
        
    def validate_synthetic_data(self, model_path: str) -> TestResult:
        """Validate model performance on synthetic data"""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            # Generate synthetic data
            synthetic_data = self.generate_synthetic_data()
            
            # Load model
            predictor = RULPredictor()
            predictor.load_models(model_path)
            
            # Test predictions
            predictions = []
            actuals = []
            anomaly_predictions = []
            
            for cap_id, cap_data in synthetic_data.items():
                for cycle in cap_data['cycles']:
                    if cycle['cycle_number'] > 10:  # Skip training cycles
                        pred_result = predictor.predict(
                            cycle['vl_series'], 
                            cycle['vo_series']
                        )
                        
                        predictions.append(pred_result.rul_cycles)
                        actuals.append(cap_data['total_cycles'] - cycle['cycle_number'])
                        anomaly_predictions.append(pred_result.anomaly_flag)
            
            # Calculate metrics
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            
            rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
            mae = np.mean(np.abs(predictions - actuals))
            
            # Check if model can detect degradation progression
            degradation_detected = np.mean(anomaly_predictions[-50:]) > np.mean(anomaly_predictions[:50])
            
            metrics = {
                "synthetic_rmse": float(rmse),
                "synthetic_mae": float(mae),
                "degradation_progression_detected": degradation_detected,
                "n_synthetic_samples": len(predictions)
            }
            
            execution_time = time.time() - start_time
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 - start_memory
            
            return TestResult(
                test_name="synthetic_data_validation",
                status="passed",
                execution_time=execution_time,
                memory_usage=memory_usage,
                metrics=metrics
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 - start_memory
            
            return TestResult(
                test_name="synthetic_data_validation",
                status="error",
                execution_time=execution_time,
                memory_usage=memory_usage,
                error_message=str(e)
            )
            
    def cross_dataset_validation(self, model_path: str, 
                               datasets: List[str]) -> List[TestResult]:
        """Validate model across multiple datasets"""
        results = []
        
        for dataset_path in datasets:
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            try:
                # Load dataset
                data_loader = DataLoader()
                test_data = data_loader.load_es12_dataset(dataset_path)
                
                # Load model
                predictor = RULPredictor()
                predictor.load_models(model_path)
                
                # Test predictions
                predictions = []
                actuals = []
                
                for cap_id, cap_data in test_data.items():
                    for cycle in cap_data.cycles:
                        if cycle.cycle_number > 10:
                            pred_result = predictor.predict(cycle.vl_series, cycle.vo_series)
                            predictions.append(pred_result.rul_cycles)
                            actuals.append(cap_data.total_cycles - cycle.cycle_number)
                
                # Calculate metrics
                predictions = np.array(predictions)
                actuals = np.array(actuals)
                
                rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
                mae = np.mean(np.abs(predictions - actuals))
                
                metrics = {
                    "cross_dataset_rmse": float(rmse),
                    "cross_dataset_mae": float(mae),
                    "dataset_path": dataset_path,
                    "n_samples": len(predictions)
                }
                
                execution_time = time.time() - start_time
                memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 - start_memory
                
                results.append(TestResult(
                    test_name=f"cross_dataset_validation_{Path(dataset_path).name}",
                    status="passed",
                    execution_time=execution_time,
                    memory_usage=memory_usage,
                    metrics=metrics
                ))
                
            except Exception as e:
                execution_time = time.time() - start_time
                memory_usage = psutil.Process().memory_info().rss / 1024 / 1024 - start_memory
                
                results.append(TestResult(
                    test_name=f"cross_dataset_validation_{Path(dataset_path).name}",
                    status="error",
                    execution_time=execution_time,
                    memory_usage=memory_usage,
                    error_message=str(e)
                ))
                
        return results


class ComprehensiveTestRunner:
    """Main test runner that orchestrates all testing components"""
    
    def __init__(self, output_dir: str = "comprehensive_test_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.regression_tester = RegressionTester(
            baseline_path=str(self.output_dir / "baseline_results.json"),
            output_dir=str(self.output_dir / "regression")
        )
        
        self.benchmarker = PerformanceBenchmarker(
            output_dir=str(self.output_dir / "benchmarks")
        )
        
        self.stress_tester = StressTester(
            output_dir=str(self.output_dir / "stress_tests")
        )
        
        self.validator = ValidationTester(
            output_dir=str(self.output_dir / "validation")
        )
        
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup main logger"""
        logger = logging.getLogger("comprehensive_test_runner")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.output_dir / "comprehensive_tests.log")
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def run_all_tests(self, model_path: str, test_data_path: str) -> Dict[str, Any]:
        """Run comprehensive test suite"""
        self.logger.info("Starting comprehensive test suite")
        
        all_results = {
            "timestamp": datetime.now().isoformat(),
            "model_path": model_path,
            "test_data_path": test_data_path,
            "regression_tests": {},
            "benchmarks": {},
            "stress_tests": {},
            "validation_tests": {}
        }
        
        # Regression tests
        self.logger.info("Running regression tests")
        try:
            accuracy_result = self.regression_tester.run_model_accuracy_regression(
                model_path, test_data_path
            )
            fpr_result = self.regression_tester.run_fpr_regression(
                model_path, test_data_path
            )
            
            all_results["regression_tests"] = {
                "accuracy": asdict(accuracy_result),
                "fpr": asdict(fpr_result)
            }
        except Exception as e:
            self.logger.error(f"Regression tests failed: {e}")
            all_results["regression_tests"]["error"] = str(e)
            
        # Performance benchmarks
        self.logger.info("Running performance benchmarks")
        try:
            latency_result = self.benchmarker.benchmark_prediction_latency(model_path)
            batch_results = self.benchmarker.benchmark_batch_processing(
                model_path, [1, 10, 50, 100]
            )
            memory_result = self.benchmarker.benchmark_memory_usage(model_path)
            
            all_results["benchmarks"] = {
                "latency": asdict(latency_result),
                "batch_processing": [asdict(r) for r in batch_results],
                "memory": asdict(memory_result)
            }
        except Exception as e:
            self.logger.error(f"Benchmarks failed: {e}")
            all_results["benchmarks"]["error"] = str(e)
            
        # Stress tests
        self.logger.info("Running stress tests")
        try:
            concurrent_result = self.stress_tester.concurrent_load_test(model_path)
            memory_stress_result = self.stress_tester.memory_stress_test(model_path)
            
            all_results["stress_tests"] = {
                "concurrent_load": asdict(concurrent_result),
                "memory_stress": asdict(memory_stress_result)
            }
        except Exception as e:
            self.logger.error(f"Stress tests failed: {e}")
            all_results["stress_tests"]["error"] = str(e)
            
        # Validation tests
        self.logger.info("Running validation tests")
        try:
            synthetic_result = self.validator.validate_synthetic_data(model_path)
            
            all_results["validation_tests"] = {
                "synthetic_data": asdict(synthetic_result)
            }
        except Exception as e:
            self.logger.error(f"Validation tests failed: {e}")
            all_results["validation_tests"]["error"] = str(e)
            
        # Save results
        results_file = self.output_dir / f"comprehensive_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
            
        self.logger.info(f"Comprehensive test suite completed. Results saved to {results_file}")
        
        return all_results
        
    def generate_test_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable test report"""
        report = []
        report.append("# Comprehensive Test Report")
        report.append(f"Generated: {results['timestamp']}")
        report.append(f"Model: {results['model_path']}")
        report.append("")
        
        # Regression tests
        if "regression_tests" in results:
            report.append("## Regression Tests")
            
            if "accuracy" in results["regression_tests"]:
                acc = results["regression_tests"]["accuracy"]
                report.append(f"- **Accuracy Test**: {acc['status'].upper()}")
                if acc.get("metrics"):
                    report.append(f"  - RMSE: {acc['metrics']['rmse']:.3f}")
                    report.append(f"  - MAE: {acc['metrics']['mae']:.3f}")
                    report.append(f"  - R²: {acc['metrics']['r2']:.3f}")
                    
            if "fpr" in results["regression_tests"]:
                fpr = results["regression_tests"]["fpr"]
                report.append(f"- **FPR Test**: {fpr['status'].upper()}")
                if fpr.get("metrics"):
                    report.append(f"  - FPR: {fpr['metrics']['fpr']:.3f}")
                    report.append(f"  - Target: < 0.05")
            report.append("")
            
        # Benchmarks
        if "benchmarks" in results:
            report.append("## Performance Benchmarks")
            
            if "latency" in results["benchmarks"]:
                lat = results["benchmarks"]["latency"]
                report.append(f"- **Prediction Latency**")
                report.append(f"  - Throughput: {lat['throughput']:.1f} predictions/sec")
                report.append(f"  - Memory Usage: {lat['memory_usage']:.1f} MB")
                
            if "memory" in results["benchmarks"]:
                mem = results["benchmarks"]["memory"]
                report.append(f"- **Memory Usage**")
                report.append(f"  - Peak Memory: {mem['memory_usage']:.1f} MB")
            report.append("")
            
        # Stress tests
        if "stress_tests" in results:
            report.append("## Stress Tests")
            
            if "concurrent_load" in results["stress_tests"]:
                conc = results["stress_tests"]["concurrent_load"]
                report.append(f"- **Concurrent Load Test**")
                report.append(f"  - Success Rate: {conc['success_rate']:.1%}")
                report.append(f"  - Avg Response Time: {conc['avg_response_time']:.3f}s")
                report.append(f"  - Peak Memory: {conc['memory_peak']:.1f} MB")
            report.append("")
            
        # Validation tests
        if "validation_tests" in results:
            report.append("## Validation Tests")
            
            if "synthetic_data" in results["validation_tests"]:
                syn = results["validation_tests"]["synthetic_data"]
                report.append(f"- **Synthetic Data Test**: {syn['status'].upper()}")
                if syn.get("metrics"):
                    report.append(f"  - RMSE: {syn['metrics']['synthetic_rmse']:.3f}")
                    report.append(f"  - MAE: {syn['metrics']['synthetic_mae']:.3f}")
            report.append("")
            
        return "\n".join(report)