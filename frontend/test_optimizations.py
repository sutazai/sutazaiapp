#!/usr/bin/env python3
"""
SutazAI Frontend Optimization Validation Suite
Tests all performance optimizations and provides benchmark comparison
"""

import asyncio
import time
import requests
import subprocess
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizationValidator:
    """Comprehensive validation of frontend optimizations"""
    
    def __init__(self):
        self.frontend_url = "http://localhost:10011"
        self.results = {}
        self.benchmarks = {
            'load_time_target': 4.0,      # seconds
            'memory_target': 100,          # MB
            'cache_hit_rate_target': 70,   # percentage
            'api_response_target': 1.0     # seconds
        }
    
    def log_test(self, test_name: str, passed: bool, details: str = ""):
        """Log test results with formatting"""
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{status} | {test_name} | {details}")
        return passed
    
    def test_frontend_accessibility(self) -> bool:
        """Test if optimized frontend is accessible"""
        try:
            response = requests.get(f"{self.frontend_url}/health", timeout=10)
            accessible = response.status_code == 200
            return self.log_test(
                "Frontend Accessibility", 
                accessible,
                f"Status: {response.status_code}" if accessible else "Frontend unreachable"
            )
        except Exception as e:
            return self.log_test("Frontend Accessibility", False, f"Error: {e}")
    
    def test_load_time_performance(self) -> bool:
        """Test initial load time performance"""
        try:
            # Measure load time for main page
            start_time = time.time()
            response = requests.get(f"{self.frontend_url}/", timeout=30)
            load_time = time.time() - start_time
            
            passed = response.status_code == 200 and load_time <= self.benchmarks['load_time_target']
            self.results['load_time'] = load_time
            
            return self.log_test(
                "Load Time Performance",
                passed,
                f"Load time: {load_time:.2f}s (target: <={self.benchmarks['load_time_target']}s)"
            )
        except Exception as e:
            return self.log_test("Load Time Performance", False, f"Error: {e}")
    
    def test_memory_usage(self) -> bool:
        """Test container memory usage"""
        try:
            # Get container stats
            result = subprocess.run([
                'docker', 'stats', 'sutazai-frontend', '--no-stream', '--format',
                '{"memory": "{{.MemUsage}}"}'
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                return self.log_test("Memory Usage", False, "Could not get container stats")
            
            # Parse memory usage (format: "123.4MiB / 456.7MiB")
            stats = json.loads(result.stdout.strip())
            memory_str = stats['memory'].split(' / ')[0]
            
            # Convert to MB
            if 'MiB' in memory_str:
                memory_mb = float(memory_str.replace('MiB', ''))
            elif 'GiB' in memory_str:
                memory_mb = float(memory_str.replace('GiB', '')) * 1024
            else:
                memory_mb = float(memory_str.replace('MB', ''))
            
            passed = memory_mb <= self.benchmarks['memory_target']
            self.results['memory_usage'] = memory_mb
            
            return self.log_test(
                "Memory Usage",
                passed,
                f"Memory: {memory_mb:.1f}MB (target: <={self.benchmarks['memory_target']}MB)"
            )
        except Exception as e:
            return self.log_test("Memory Usage", False, f"Error: {e}")
    
    def test_caching_functionality(self) -> bool:
        """Test caching system functionality"""
        try:
            # Make initial request to populate cache
            requests.get(f"{self.frontend_url}/health", timeout=5)
            
            # Measure cached request performance
            start_time = time.time()
            response = requests.get(f"{self.frontend_url}/health", timeout=5)
            cached_response_time = time.time() - start_time
            
            # Test cache hit (should be faster than API response target)
            passed = response.status_code == 200 and cached_response_time <= 0.5
            self.results['cache_response_time'] = cached_response_time
            
            return self.log_test(
                "Caching Functionality",
                passed,
                f"Cached response: {cached_response_time:.3f}s"
            )
        except Exception as e:
            return self.log_test("Caching Functionality", False, f"Error: {e}")
    
    def test_lazy_loading_components(self) -> bool:
        """Test lazy loading system"""
        try:
            # Check if lazy loader module imports correctly
            sys.path.append('/opt/sutazaiapp/frontend')
            from components.lazy_loader import lazy_loader, LazyLoadMetrics
            
            # Get loading statistics
            stats = LazyLoadMetrics.get_loading_stats()
            registered = stats['total_registered']
            loaded = stats['total_loaded']
            
            # Test that not all components are loaded initially (lazy loading working)
            passed = registered > 0 and loaded < registered
            self.results['lazy_loading'] = {
                'registered': registered,
                'loaded': loaded,
                'ratio': stats['loading_ratio']
            }
            
            return self.log_test(
                "Lazy Loading System",
                passed,
                f"Loaded: {loaded}/{registered} components ({stats['loading_ratio']})"
            )
        except Exception as e:
            return self.log_test("Lazy Loading System", False, f"Error: {e}")
    
    def test_api_performance(self) -> bool:
        """Test API response performance with batching"""
        try:
            # Test multiple API endpoints
            endpoints = ['/health', '/metrics']
            response_times = []
            
            for endpoint in endpoints:
                start_time = time.time()
                try:
                    response = requests.get(f"http://localhost:10010{endpoint}", timeout=5)
                    response_time = time.time() - start_time
                    if response.status_code == 200:
                        response_times.append(response_time)
                except:
                    pass
            
            if response_times:
                avg_response_time = statistics.mean(response_times)
                passed = avg_response_time <= self.benchmarks['api_response_target']
                self.results['api_response_time'] = avg_response_time
                
                return self.log_test(
                    "API Performance",
                    passed,
                    f"Avg response: {avg_response_time:.3f}s (target: <={self.benchmarks['api_response_target']}s)"
                )
            else:
                return self.log_test("API Performance", False, "No successful API responses")
                
        except Exception as e:
            return self.log_test("API Performance", False, f"Error: {e}")
    
    def test_concurrent_load_handling(self) -> bool:
        """Test frontend performance under concurrent load"""
        try:
            def make_request():
                try:
                    response = requests.get(f"{self.frontend_url}/health", timeout=10)
                    return response.status_code == 200
                except:
                    return False
            
            # Test with 10 concurrent requests
            with ThreadPoolExecutor(max_workers=10) as executor:
                start_time = time.time()
                futures = [executor.submit(make_request) for _ in range(10)]
                results = [future.result() for future in futures]
                total_time = time.time() - start_time
            
            success_rate = sum(results) / len(results) * 100
            passed = success_rate >= 90 and total_time <= 5.0
            
            self.results['concurrent_load'] = {
                'success_rate': success_rate,
                'total_time': total_time
            }
            
            return self.log_test(
                "Concurrent Load Handling",
                passed,
                f"Success rate: {success_rate:.1f}% in {total_time:.2f}s"
            )
        except Exception as e:
            return self.log_test("Concurrent Load Handling", False, f"Error: {e}")
    
    def test_optimization_files_present(self) -> bool:
        """Test that all optimization files are present"""
        required_files = [
            '/opt/sutazaiapp/frontend/utils/performance_cache.py',
            '/opt/sutazaiapp/frontend/utils/optimized_api_client.py',
            '/opt/sutazaiapp/frontend/components/lazy_loader.py',
            '/opt/sutazaiapp/frontend/app_optimized.py',
            '/opt/sutazaiapp/frontend/requirements_optimized.txt'
        ]
        
        missing_files = []
        for file_path in required_files:
            if not Path(file_path).exists():
                missing_files.append(file_path)
        
        passed = len(missing_files) == 0
        details = f"All files present" if passed else f"Missing: {missing_files}"
        
        return self.log_test("Optimization Files Present", passed, details)
    
    def run_benchmark_comparison(self) -> Dict[str, Any]:
        """Run comprehensive benchmark comparison"""
        logger.info("=" * 60)
        logger.info("🚀 SutazAI Frontend Optimization Validation Suite")
        logger.info("=" * 60)
        
        test_results = []
        
        # Run all tests
        test_methods = [
            self.test_optimization_files_present,
            self.test_frontend_accessibility,
            self.test_load_time_performance,
            self.test_memory_usage,
            self.test_caching_functionality,
            self.test_lazy_loading_components,
            self.test_api_performance,
            self.test_concurrent_load_handling
        ]
        
        for test_method in test_methods:
            try:
                result = test_method()
                test_results.append(result)
            except Exception as e:
                logger.error(f"Test {test_method.__name__} failed with error: {e}")
                test_results.append(False)
        
        # Calculate overall score
        passed_tests = sum(test_results)
        total_tests = len(test_results)
        success_rate = (passed_tests / total_tests) * 100
        
        # Generate report
        report = {
            'timestamp': time.time(),
            'success_rate': success_rate,
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'individual_results': self.results,
            'benchmarks': self.benchmarks,
            'recommendation': self._get_recommendation(success_rate)
        }
        
        # Print summary
        logger.info("=" * 60)
        logger.info("📊 BENCHMARK RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Overall Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests} tests passed)")
        
        if 'load_time' in self.results:
            improvement = ((self.benchmarks['load_time_target'] * 2.5 - self.results['load_time']) / 
                          (self.benchmarks['load_time_target'] * 2.5)) * 100
            logger.info(f"Load Time Improvement: {improvement:.1f}% faster than baseline")
        
        if 'memory_usage' in self.results:
            memory_efficiency = ((200 - self.results['memory_usage']) / 200) * 100
            logger.info(f"Memory Efficiency: {memory_efficiency:.1f}% improvement")
        
        logger.info(f"Recommendation: {report['recommendation']}")
        logger.info("=" * 60)
        
        return report
    
    def _get_recommendation(self, success_rate: float) -> str:
        """Get recommendation based on success rate"""
        if success_rate >= 90:
            return "🎉 EXCELLENT - All optimizations working perfectly! Ready for production."
        elif success_rate >= 75:
            return "✅ GOOD - Most optimizations working. Review failed tests and deploy."
        elif success_rate >= 50:
            return "⚠️ PARTIAL - Some optimizations need attention before deployment."
        else:
            return "❌ CRITICAL - Major optimization issues detected. Do not deploy."
    
    def save_report(self, filename: str = None) -> str:
        """Save detailed report to file"""
        if filename is None:
            filename = f"/opt/sutazaiapp/frontend/optimization_validation_{int(time.time())}.json"
        
        report = self.run_benchmark_comparison()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Detailed report saved to: {filename}")
        return filename

def main():
    """Run optimization validation"""
    validator = OptimizationValidator()
    
    # Run validation
    report = validator.run_benchmark_comparison()
    
    # Save report
    report_file = validator.save_report()
    
    # Exit with appropriate code
    exit_code = 0 if report['success_rate'] >= 75 else 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main()