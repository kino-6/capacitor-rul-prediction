#!/usr/bin/env python3
"""
Test script for the comprehensive testing framework
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from true_rul.testing_framework import (
    ComprehensiveTestRunner,
    RegressionTester,
    PerformanceBenchmarker,
    StressTester,
    ValidationTester
)


def test_individual_components():
    """Test individual testing components"""
    print("Testing individual components...")
    
    # Test data paths (using mock paths for now)
    model_path = "models/mock_model"
    test_data_path = "data/mock_test_data"
    
    try:
        # Test RegressionTester
        print("  Testing RegressionTester...")
        regression_tester = RegressionTester(
            baseline_path="test_baseline.json",
            output_dir="test_regression_output"
        )
        print("    ✓ RegressionTester initialized successfully")
        
        # Test PerformanceBenchmarker
        print("  Testing PerformanceBenchmarker...")
        benchmarker = PerformanceBenchmarker(output_dir="test_benchmark_output")
        print("    ✓ PerformanceBenchmarker initialized successfully")
        
        # Test StressTester
        print("  Testing StressTester...")
        stress_tester = StressTester(output_dir="test_stress_output")
        print("    ✓ StressTester initialized successfully")
        
        # Test ValidationTester
        print("  Testing ValidationTester...")
        validator = ValidationTester(output_dir="test_validation_output")
        
        # Test synthetic data generation
        synthetic_data = validator.generate_synthetic_data(n_capacitors=2, cycles_per_capacitor=50)
        assert len(synthetic_data) == 2
        assert all(len(cap_data['cycles']) == 50 for cap_data in synthetic_data.values())
        print("    ✓ ValidationTester and synthetic data generation working")
        
        print("✓ All individual components tested successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error testing individual components: {e}")
        return False


def test_comprehensive_runner():
    """Test the comprehensive test runner"""
    print("Testing ComprehensiveTestRunner...")
    
    try:
        # Initialize runner
        runner = ComprehensiveTestRunner(output_dir="test_comprehensive_output")
        print("  ✓ ComprehensiveTestRunner initialized successfully")
        
        # Test report generation with mock data
        mock_results = {
            "timestamp": "2024-01-01T12:00:00",
            "model_path": "mock_model",
            "test_data_path": "mock_data",
            "regression_tests": {
                "accuracy": {
                    "status": "passed",
                    "metrics": {"rmse": 5.2, "mae": 3.1, "r2": 0.85}
                },
                "fpr": {
                    "status": "passed",
                    "metrics": {"fpr": 0.03}
                }
            },
            "benchmarks": {
                "latency": {
                    "throughput": 150.5,
                    "memory_usage": 45.2
                },
                "memory": {
                    "memory_usage": 128.5
                }
            },
            "stress_tests": {
                "concurrent_load": {
                    "success_rate": 0.98,
                    "avg_response_time": 0.025,
                    "memory_peak": 256.8
                }
            },
            "validation_tests": {
                "synthetic_data": {
                    "status": "passed",
                    "metrics": {"synthetic_rmse": 4.8, "synthetic_mae": 2.9}
                }
            }
        }
        
        report = runner.generate_test_report(mock_results)
        assert "Comprehensive Test Report" in report
        assert "RMSE: 5.200" in report
        assert "FPR: 0.030" in report
        print("  ✓ Test report generation working")
        
        print("✓ ComprehensiveTestRunner tested successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error testing ComprehensiveTestRunner: {e}")
        return False


def test_data_structures():
    """Test data structures used by the testing framework"""
    print("Testing data structures...")
    
    try:
        from true_rul.testing_framework import TestResult, BenchmarkResult, StressTestResult
        
        # Test TestResult
        test_result = TestResult(
            test_name="test_accuracy",
            status="passed",
            execution_time=1.5,
            memory_usage=64.2,
            metrics={"rmse": 5.0, "mae": 3.2}
        )
        assert test_result.test_name == "test_accuracy"
        assert test_result.metrics["rmse"] == 5.0
        print("  ✓ TestResult structure working")
        
        # Test BenchmarkResult
        benchmark_result = BenchmarkResult(
            test_name="latency_test",
            execution_time=2.1,
            memory_usage=32.5,
            throughput=100.0,
            cpu_usage=15.2
        )
        assert benchmark_result.throughput == 100.0
        print("  ✓ BenchmarkResult structure working")
        
        # Test StressTestResult
        stress_result = StressTestResult(
            test_name="load_test",
            load_level="high",
            success_rate=0.95,
            avg_response_time=0.05,
            max_response_time=0.2,
            memory_peak=128.0,
            cpu_peak=80.0,
            errors=["timeout error"]
        )
        assert stress_result.success_rate == 0.95
        assert len(stress_result.errors) == 1
        print("  ✓ StressTestResult structure working")
        
        print("✓ All data structures tested successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error testing data structures: {e}")
        return False


def main():
    """Main test function"""
    print("=" * 60)
    print("Testing Comprehensive Testing Framework")
    print("=" * 60)
    
    all_passed = True
    
    # Test individual components
    if not test_individual_components():
        all_passed = False
    print()
    
    # Test comprehensive runner
    if not test_comprehensive_runner():
        all_passed = False
    print()
    
    # Test data structures
    if not test_data_structures():
        all_passed = False
    print()
    
    # Final result
    print("=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED - Comprehensive Testing Framework is working correctly")
        return 0
    else:
        print("✗ SOME TESTS FAILED - Please check the errors above")
        return 1


if __name__ == "__main__":
    exit(main())