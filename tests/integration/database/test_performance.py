#!/usr/bin/env python3
"""
Purpose: Performance and resource usage validation tests for hygiene system
Usage: python -m pytest tests/hygiene/test_performance.py
Requirements: pytest, psutil, time
"""

import unittest
import tempfile
import subprocess
import time
import psutil
import threading
import resource
from pathlib import Path
import shutil
import sys
import json
import os

# Add project root to path
# Path handled by pytest configuration

class TestPerformanceBenchmarks(unittest.TestCase):
    """Test performance benchmarks for hygiene system components"""
    
    def setUp(self):
        """Setup performance testing environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_project = self.temp_dir / "perf_test"
        self.test_project.mkdir()
        
        # Create large test dataset
        self.create_large_test_dataset()
        
        # Performance thresholds
        self.performance_thresholds = {
            'orchestrator_dry_run': 60.0,  # seconds
            'rule_13_detection': 30.0,     # seconds
            'git_hook_execution': 10.0,    # seconds
            'report_generation': 5.0,      # seconds
            'max_memory_mb': 100,          # MB
            'max_cpu_percent': 80          # percent
        }
        
    def tearDown(self):
        """Cleanup performance test environment"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            
    def create_large_test_dataset(self):
        """Create large dataset for performance testing"""
        # Create directory structure
        dirs = ['scripts', 'tests', 'logs', 'archive', 'data']
        for dir_name in dirs:
            (self.test_project / dir_name).mkdir()
            
        # Create many test files
        file_types = [
            ('.py', 'print("test python file")'),
            ('.sh', '#!/bin/bash\necho "test shell script"'),
            ('.txt', 'test text content'),
            ('.backup', 'backup file content'),
            ('.tmp', 'temporary file content'),
            ('.log', 'log file content')
        ]
        
        # Create 1000 files total
        for i in range(1000):
            file_type, content = file_types[i % len(file_types)]
            test_file = self.test_project / f"test_{i:04d}{file_type}"
            test_file.write_text(f"{content} - file {i}")
            
        # Create nested directory structure
        for i in range(10):
            nested_dir = self.test_project / f"nested_{i}"
            nested_dir.mkdir()
            
            for j in range(50):
                nested_file = nested_dir / f"nested_{j}.py"
                nested_file.write_text(f"# Nested file {i}/{j}\nprint('nested')")
                
    def measure_execution_time(self, func, *args, **kwargs):
        """Measure execution time of a function"""
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        return result, execution_time
        
    def measure_resource_usage(self, func, *args, **kwargs):
        """Measure resource usage during function execution"""
        # Get initial resource usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu_percent = process.cpu_percent()
        
        # Execute function with timing
        start_time = time.time()
        
        # Monitor resources during execution
        max_memory = initial_memory
        max_cpu = initial_cpu_percent
        
        def monitor_resources():
            nonlocal max_memory, max_cpu
            while not stop_monitoring:
                try:
                    memory_mb = process.memory_info().rss / 1024 / 1024
                    cpu_percent = process.cpu_percent()
                    
                    max_memory = max(max_memory, memory_mb)
                    max_cpu = max(max_cpu, cpu_percent)
                    
                    time.sleep(0.1)  # Sample every 100ms
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    break
                    
        stop_monitoring = False
        monitor_thread = threading.Thread(target=monitor_resources)
        monitor_thread.start()
        
        try:
            result = func(*args, **kwargs)
        finally:
            stop_monitoring = True
            monitor_thread.join(timeout=1)
            
        end_time = time.time()
        execution_time = end_time - start_time
        
        return result, {
            'execution_time': execution_time,
            'max_memory_mb': max_memory,
            'max_cpu_percent': max_cpu,
            'initial_memory_mb': initial_memory
        }
        
    def test_large_directory_scanning_performance(self):
        """Test performance of scanning large directories"""
        def scan_violations():
            violations = []
            patterns = ['*.backup', '*.tmp', '*.bak', '*~']
            
            for pattern in patterns:
                violations.extend(list(self.test_project.glob(f"**/{pattern}")))
                
            return violations
            
        violations, execution_time = self.measure_execution_time(scan_violations)
        
        # Performance assertions
        self.assertLess(execution_time, self.performance_thresholds['rule_13_detection'],
                       f"Directory scanning took too long: {execution_time:.2f}s")
        
        self.assertGreater(len(violations), 0, "Should detect violations in test dataset")
        
        print(f"Directory scanning performance: {execution_time:.2f}s for {len(violations)} violations")
        
    def test_file_processing_performance(self):
        """Test performance of processing many files"""
        def process_python_files():
            python_files = list(self.test_project.glob("**/*.py"))
            processed = []
            
            for py_file in python_files:
                try:
                    content = py_file.read_text()
                    # Simulate processing (check for documentation)
                    has_docstring = '"""' in content or "'''" in content
                    processed.append({
                        'file': str(py_file),
                        'has_docstring': has_docstring,
                        'size': len(content)
                    })
                except Exception:
                    continue
                    
            return processed
            
        processed_files, metrics = self.measure_resource_usage(process_python_files)
        
        # Performance assertions
        self.assertLess(metrics['execution_time'], 30.0,
                       f"File processing took too long: {metrics['execution_time']:.2f}s")
        
        self.assertLess(metrics['max_memory_mb'], self.performance_thresholds['max_memory_mb'],
                       f"Memory usage too high: {metrics['max_memory_mb']:.1f}MB")
        
        print(f"File processing performance: {metrics['execution_time']:.2f}s, "
              f"Memory: {metrics['max_memory_mb']:.1f}MB, "
              f"Files processed: {len(processed_files)}")
              
    def test_concurrent_processing_performance(self):
        """Test performance of concurrent operations"""
        import concurrent.futures
        
        def process_file(file_path):
            """Process a single file"""
            try:
                content = file_path.read_text()
                return {
                    'file': str(file_path),
                    'size': len(content),
                    'lines': len(content.split('\n'))
                }
            except Exception:
                return None
                
        def concurrent_processing():
            python_files = list(self.test_project.glob("**/*.py"))[:100]  # Limit for test
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(process_file, python_files))
                
            return [r for r in results if r is not None]
            
        results, metrics = self.measure_resource_usage(concurrent_processing)
        
        # Performance assertions
        self.assertLess(metrics['execution_time'], 20.0,
                       f"Concurrent processing took too long: {metrics['execution_time']:.2f}s")
        
        self.assertGreater(len(results), 0, "Should process files concurrently")
        
        print(f"Concurrent processing performance: {metrics['execution_time']:.2f}s, "
              f"Processed: {len(results)} files")

class TestMemoryUsagePatterns(unittest.TestCase):
    """Test memory usage patterns and optimization"""
    
    def setUp(self):
        """Setup memory testing environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        """Cleanup memory test environment"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            
    def test_memory_leak_detection(self):
        """Test for potential memory leaks in repeated operations"""
        import gc
        
        def create_and_process_data():
            """Create and process data that might cause memory leaks"""
            data = []
            
            # Create large data structure
            for i in range(1000):
                item = {
                    'id': i,
                    'content': f"data item {i}" * 100,
                    'metadata': {'created': time.time(), 'processed': False}
                }
                data.append(item)
                
            # Process data
            for item in data:
                item['processed'] = True
                item['content'] = item['content'].upper()
                
            return len(data)
            
        # Measure memory usage over multiple iterations
        memory_usage = []
        process = psutil.Process()
        
        for iteration in range(10):
            # Force garbage collection before measurement
            gc.collect()
            
            initial_memory = process.memory_info().rss / 1024 / 1024
            
            # Run operation
            result = create_and_process_data()
            
            # Force garbage collection after operation
            gc.collect()
            
            final_memory = process.memory_info().rss / 1024 / 1024
            memory_usage.append(final_memory - initial_memory)
            
        # Check for memory leaks (increasing trend)
        if len(memory_usage) > 5:
            recent_avg = sum(memory_usage[-5:]) / 5
            early_avg = sum(memory_usage[:5]) / 5
            
            memory_increase = recent_avg - early_avg
            
            # Allow for some variation but detect significant leaks
            self.assertLess(memory_increase, 10.0,  # 10MB increase threshold
                           f"Potential memory leak detected: {memory_increase:.1f}MB increase")
            
        print(f"Memory usage pattern: {memory_usage}")
        
    def test_large_dataset_memory_efficiency(self):
        """Test memory efficiency with large datasets"""
        def process_large_dataset_streaming():
            """Process large dataset using streaming/generator approach"""
            processed_count = 0
            
            # Simulate streaming processing
            def data_generator():
                for i in range(10000):
                    yield f"data_item_{i}" * 50
                    
            for item in data_generator():
                # Process item without storing all in memory
                processed_item = item.upper()
                processed_count += 1
                
                # Simulate some processing
                if processed_count % 1000 == 0:
                    pass  # Could yield progress here
                    
            return processed_count
            
        def process_large_dataset_batch():
            """Process large dataset by loading all into memory"""
            # Load all data into memory at once
            all_data = [f"data_item_{i}" * 50 for i in range(10000)]
            
            # Process all data
            processed_data = [item.upper() for item in all_data]
            
            return len(processed_data)
            
        # Test streaming approach
        streaming_result, streaming_metrics = self.measure_resource_usage(
            process_large_dataset_streaming
        )
        
        # Test batch approach
        batch_result, batch_metrics = self.measure_resource_usage(
            process_large_dataset_batch
        )
        
        # Streaming should use less memory
        self.assertLess(streaming_metrics['max_memory_mb'], 
                       batch_metrics['max_memory_mb'],
                       "Streaming approach should use less memory")
        
        # Both should process same amount of data
        self.assertEqual(streaming_result, batch_result)
        
        print(f"Streaming: {streaming_metrics['max_memory_mb']:.1f}MB, "
              f"Batch: {batch_metrics['max_memory_mb']:.1f}MB")

class TestScalabilityLimits(unittest.TestCase):
    """Test scalability limits and behavior under load"""
    
    def setUp(self):
        """Setup scalability testing environment"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        """Cleanup scalability test environment"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            
    def test_file_count_scalability(self):
        """Test handling of increasing file counts"""
        file_counts = [100, 500, 1000, 2000]
        processing_times = []
        
        for file_count in file_counts:
            # Create test files
            test_dir = self.temp_dir / f"scale_test_{file_count}"
            test_dir.mkdir()
            
            for i in range(file_count):
                test_file = test_dir / f"test_{i:06d}.py"
                test_file.write_text(f"# Test file {i}\nprint('test {i}')")
                
            # Measure processing time
            def process_files():
                python_files = list(test_dir.glob("*.py"))
                results = []
                
                for py_file in python_files:
                    content = py_file.read_text()
                    results.append({
                        'file': str(py_file),
                        'lines': len(content.split('\n'))
                    })
                    
                return results
                
            results, execution_time = self.measure_execution_time(process_files)
            processing_times.append(execution_time)
            
            # Cleanup
            shutil.rmtree(test_dir)
            
        # Check scalability (should be roughly linear)
        if len(processing_times) >= 2:
            # Calculate scaling factor
            time_ratio = processing_times[-1] / processing_times[0]
            file_ratio = file_counts[-1] / file_counts[0]
            
            # Should scale better than quadratic
            scaling_efficiency = time_ratio / file_ratio
            
            self.assertLess(scaling_efficiency, 2.0,
                           f"Poor scaling detected: {scaling_efficiency:.2f}x time for {file_ratio}x files")
            
        print(f"Scalability test - File counts: {file_counts}, Times: {processing_times}")
        
    def test_directory_depth_scalability(self):
        """Test handling of deep directory structures"""
        max_depth = 20
        
        # Create deep directory structure
        current_dir = self.temp_dir / "depth_test"
        current_dir.mkdir()
        
        for depth in range(max_depth):
            current_dir = current_dir / f"level_{depth:02d}"
            current_dir.mkdir()
            
            # Add files at each level
            for i in range(5):
                test_file = current_dir / f"file_{i}.py"
                test_file.write_text(f"# File at depth {depth}\nprint('depth {depth}')")
                
        # Test recursive operations
        def recursive_file_scan():
            all_files = list(self.temp_dir.glob("**/*.py"))
            return len(all_files)
            
        file_count, execution_time = self.measure_execution_time(recursive_file_scan)
        
        # Should handle deep structures reasonably
        self.assertLess(execution_time, 10.0,
                       f"Deep directory scanning took too long: {execution_time:.2f}s")
        
        self.assertEqual(file_count, max_depth * 5,
                        f"Should find all files in deep structure")
        
        print(f"Deep directory test: {file_count} files at {max_depth} levels in {execution_time:.2f}s")

class TestRealWorldPerformance(unittest.TestCase):
    """Test performance with real-world scenarios"""
    
    def setUp(self):
        """Setup real-world performance testing"""
        self.project_root = Path("/opt/sutazaiapp")
        
    def test_orchestrator_performance(self):
        """Test orchestrator performance with real system"""
        orchestrator_script = self.project_root / "scripts/agents/hygiene-agent-orchestrator.py"
        
        if not orchestrator_script.exists():
            self.skipTest("Orchestrator script not available")
            
        def run_orchestrator_dry_run():
            """Run orchestrator in dry run mode"""
            cmd = [sys.executable, str(orchestrator_script), "--rule=13", "--dry-run"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            return result.returncode == 0
            
        success, execution_time = self.measure_execution_time(run_orchestrator_dry_run)
        
        # Performance assertions
        self.assertTrue(success, "Orchestrator should execute successfully")
        self.assertLess(execution_time, 60.0,
                       f"Orchestrator execution too slow: {execution_time:.2f}s")
        
        print(f"Orchestrator performance: {execution_time:.2f}s")
        
    def test_test_suite_performance(self):
        """Test the test suite's own performance"""
        test_runner = self.project_root / "scripts/test-hygiene-system.py"
        
        if not test_runner.exists():
            self.skipTest("Test runner not available")
            
        def run_test_suite():
            """Run test suite setup only"""
            cmd = [sys.executable, str(test_runner), "--setup-only"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return result.returncode == 0
            
        success, execution_time = self.measure_execution_time(run_test_suite)
        
        # Performance assertions
        self.assertTrue(success, "Test suite should run successfully")
        self.assertLess(execution_time, 30.0,
                       f"Test suite setup too slow: {execution_time:.2f}s")
        
        print(f"Test suite performance: {execution_time:.2f}s")

class TestPerformanceReporting(unittest.TestCase):
    """Test performance reporting and metrics collection"""
    
    def setUp(self):
        """Setup performance reporting test"""
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def tearDown(self):
        """Cleanup performance reporting test"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            
    def test_performance_metrics_collection(self):
        """Test collection of performance metrics"""
        def collect_system_metrics():
            """Collect system performance metrics"""
            metrics = {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(interval=1),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            }
            return metrics
            
        metrics = collect_system_metrics()
        
        # Validate metrics structure
        required_fields = ['timestamp', 'cpu_percent', 'memory_percent', 'disk_usage']
        for field in required_fields:
            self.assertIn(field, metrics, f"Missing metric: {field}")
            self.assertIsInstance(metrics[field], (int, float))
            
        print(f"System metrics: {metrics}")
        
    def test_performance_report_generation(self):
        """Test generation of performance reports"""
        def generate_performance_report():
            """Generate comprehensive performance report"""
            # Simulate performance test results
            test_results = {
                'directory_scan': {'time': 2.5, 'files_processed': 1000},
                'file_processing': {'time': 5.1, 'files_processed': 500},
                'memory_usage': {'peak_mb': 45.2, 'average_mb': 32.1},
                'cpu_usage': {'peak_percent': 75.0, 'average_percent': 45.0}
            }
            
            # Create report
            report = {
                'timestamp': time.time(),
                'test_summary': {
                    'total_tests': len(test_results),
                    'total_time': sum(r.get('time', 0) for r in test_results.values()),
                    'passed_tests': len(test_results)  # All passed in this simulation
                },
                'performance_metrics': test_results,
                'system_info': {
                    'python_version': sys.version,
                    'platform': sys.platform,
                    'cpu_count': os.cpu_count()
                }
            }
            
            return report
            
        report = generate_performance_report()
        
        # Validate report structure
        required_sections = ['timestamp', 'test_summary', 'performance_metrics', 'system_info']
        for section in required_sections:
            self.assertIn(section, report, f"Missing report section: {section}")
            
        # Test report serialization
        report_json = json.dumps(report, indent=2)
        self.assertIsInstance(report_json, str)
        
        # Test report deserialization
        loaded_report = json.loads(report_json)
        self.assertEqual(loaded_report['test_summary']['total_tests'], 
                        report['test_summary']['total_tests'])
        
        print(f"Performance report generated: {len(report_json)} characters")

if __name__ == "__main__":
    unittest.main()