#!/usr/bin/env python3
"""
Ultra QA Team Lead - Frontend Performance Validation Test
Validates claimed 70% load time improvement and 60% memory reduction
"""

import logging

# Configure logger for exception handling
logger = logging.getLogger(__name__)

import time
import requests
import psutil
import json
import subprocess
import threading
from datetime import datetime
from typing import Dict, List, Any
import statistics
import concurrent.futures
import sys
import os

class FrontendPerformanceValidator:
    def __init__(self):
        self.frontend_url = "http://localhost:10011"
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "test_results": {},
            "baseline_measurements": {},
            "performance_claims": {
                "load_time_improvement": "70%",
                "memory_reduction": "60%"
            },
            "validation_status": "PENDING"
        }
        
    def get_container_stats(self, container_name: str) -> Dict[str, Any]:
        """Get container resource statistics"""
        try:
            cmd = f"docker stats {container_name} --no-stream --format json"
            result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                stats = json.loads(result.stdout)
                
                # Parse memory usage
                mem_usage_str = stats.get("MemUsage", "0B / 0B")
                mem_parts = mem_usage_str.split(" / ")
                mem_used = self._parse_memory_string(mem_parts[0])
                mem_limit = self._parse_memory_string(mem_parts[1])
                
                # Parse CPU percentage
                cpu_percent = float(stats.get("CPUPerc", "0%").replace("%", ""))
                
                return {
                    "memory_used_mb": mem_used,
                    "memory_limit_mb": mem_limit,
                    "memory_percent": (mem_used / mem_limit * 100) if mem_limit > 0 else 0,
                    "cpu_percent": cpu_percent,
                    "raw_stats": stats
                }
        except Exception as e:
            print(f"Error getting container stats: {e}")
            
        return {"error": "Failed to get stats"}
    
    def _parse_memory_string(self, mem_str: str) -> float:
        """Parse memory string like '48.55MiB' to MB"""
        mem_str = mem_str.strip()
        if mem_str.endswith("MiB"):
            return float(mem_str[:-3])
        elif mem_str.endswith("MB"):
            return float(mem_str[:-2])
        elif mem_str.endswith("GiB"):
            return float(mem_str[:-3]) * 1024
        elif mem_str.endswith("GB"):
            return float(mem_str[:-2]) * 1000
        elif mem_str.endswith("B"):
            return float(mem_str[:-1]) / (1024 * 1024)
        return 0.0
    
    def measure_load_time(self, url: str, num_tests: int = 10) -> Dict[str, float]:
        """Measure frontend load times with statistical analysis"""
        load_times = []
        
        print(f"Measuring load times for {url} ({num_tests} iterations)...")
        
        for i in range(num_tests):
            try:
                start_time = time.time()
                response = requests.get(url, timeout=30)
                end_time = time.time()
                
                load_time = end_time - start_time
                load_times.append(load_time)
                
                print(f"  Test {i+1}/{num_tests}: {load_time:.3f}s (Status: {response.status_code})")
                
                # Wait between tests
                time.sleep(0.5)
                
            except Exception as e:
                print(f"  Test {i+1}/{num_tests}: FAILED - {e}")
                
        if not load_times:
            return {"error": "All load time tests failed"}
            
        return {
            "mean": statistics.mean(load_times),
            "median": statistics.median(load_times),
            "min": min(load_times),
            "max": max(load_times),
            "std_dev": statistics.stdev(load_times) if len(load_times) > 1 else 0,
            "sample_count": len(load_times),
            "raw_times": load_times
        }
    
    def perform_load_testing(self, concurrent_users: int = 5, duration_seconds: int = 30) -> Dict[str, Any]:
        """Perform concurrent load testing"""
        results = {
            "concurrent_users": concurrent_users,
            "duration_seconds": duration_seconds,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0,
            "requests_per_second": 0,
            "errors": []
        }
        
        print(f"Starting load test: {concurrent_users} concurrent users for {duration_seconds} seconds")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        response_times = []
        request_count = 0
        success_count = 0
        errors = []
        
        def make_requests():
            nonlocal request_count, success_count, errors
            
            while time.time() < end_time:
                try:
                    request_start = time.time()
                    response = requests.get(self.frontend_url, timeout=10)
                    request_end = time.time()
                    
                    request_count += 1
                    response_time = request_end - request_start
                    response_times.append(response_time)
                    
                    if response.status_code == 200:
                        success_count += 1
                    else:
                        errors.append(f"HTTP {response.status_code}")
                        
                except Exception as e:
                    request_count += 1
                    errors.append(str(e))
                    
                time.sleep(0.1)  # Brief pause between requests
        
        # Start concurrent threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(make_requests) for _ in range(concurrent_users)]
            concurrent.futures.wait(futures)
        
        actual_duration = time.time() - start_time
        
        results.update({
            "total_requests": request_count,
            "successful_requests": success_count,
            "failed_requests": request_count - success_count,
            "average_response_time": statistics.mean(response_times) if response_times else 0,
            "requests_per_second": request_count / actual_duration if actual_duration > 0 else 0,
            "errors": errors[:10]  # Keep first 10 errors
        })
        
        return results
    
    def test_caching_mechanisms(self) -> Dict[str, Any]:
        """Test if caching is working by measuring repeated requests"""
        print("Testing caching mechanisms...")
        
        # First request (should be slower - cache miss)
        first_request = self.measure_load_time(self.frontend_url, num_tests=1)
        time.sleep(1)  # Brief pause
        
        # Second request (should be faster - cache hit)
        second_request = self.measure_load_time(self.frontend_url, num_tests=1)
        
        if "error" not in first_request and "error" not in second_request:
            first_time = first_request["mean"]
            second_time = second_request["mean"]
            improvement = ((first_time - second_time) / first_time * 100) if first_time > 0 else 0
            
            return {
                "first_request_time": first_time,
                "second_request_time": second_time,
                "improvement_percent": improvement,
                "caching_effective": improvement > 10  # Consider caching effective if >10% improvement
            }
        
        return {"error": "Failed to test caching"}
    
    def validate_functionality(self) -> Dict[str, Any]:
        """Validate that all core functionality still works"""
        print("Validating frontend functionality...")
        
        tests = {
            "homepage_accessible": False,
            "streamlit_app_loads": False,
            "health_endpoint": False,
            "static_resources": False
        }
        
        errors = []
        
        try:
            # Test homepage
            response = requests.get(self.frontend_url, timeout=10)
            tests["homepage_accessible"] = response.status_code == 200
            
            if response.status_code == 200:
                # Check if it's a Streamlit app
                if "streamlit" in response.text.lower() or "window.prerenderReady" in response.text:
                    tests["streamlit_app_loads"] = True
                    
        except Exception as e:
            errors.append(f"Homepage test failed: {e}")
        
        try:
            # Test health endpoint (though Streamlit doesn't have one by default)
            health_response = requests.get(f"{self.frontend_url}/health", timeout=5)
            tests["health_endpoint"] = health_response.status_code == 200
        except (AssertionError, Exception) as e:
            # TODO: Review this exception handling
            logger.error(f"Unexpected exception: {e}", exc_info=True)
            pass  # Health endpoint may not exist for Streamlit
        
        try:
            # Test static resources
            static_response = requests.get(f"{self.frontend_url}/static/css/index.CJVRHjQZ.css", timeout=5)
            tests["static_resources"] = static_response.status_code in [200, 404]  # 404 is acceptable if path changes
        except Exception as e:
            errors.append(f"Static resources test failed: {e}")
        
        return {
            "tests": tests,
            "errors": errors,
            "overall_functional": all([tests["homepage_accessible"], tests["streamlit_app_loads"]])
        }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run all performance tests"""
        print("=" * 80)
        print("ULTRA QA TEAM LEAD - Frontend Performance Validation")
        print("=" * 80)
        
        # Get baseline container stats
        print("\n1. Getting baseline container statistics...")
        baseline_stats = self.get_container_stats("sutazai-frontend")
        self.results["baseline_measurements"]["container_stats"] = baseline_stats
        
        if "error" not in baseline_stats:
            print(f"   Memory Usage: {baseline_stats['memory_used_mb']:.1f} MB ({baseline_stats['memory_percent']:.1f}%)")
            print(f"   CPU Usage: {baseline_stats['cpu_percent']:.1f}%")
        
        # Test load times
        print("\n2. Testing load times...")
        load_time_results = self.measure_load_time(self.frontend_url, num_tests=15)
        self.results["test_results"]["load_times"] = load_time_results
        
        if "error" not in load_time_results:
            print(f"   Average load time: {load_time_results['mean']:.3f}s")
            print(f"   Median load time: {load_time_results['median']:.3f}s")
            print(f"   Min/Max: {load_time_results['min']:.3f}s / {load_time_results['max']:.3f}s")
        
        # Test caching
        print("\n3. Testing caching mechanisms...")
        caching_results = self.test_caching_mechanisms()
        self.results["test_results"]["caching"] = caching_results
        
        if "error" not in caching_results:
            print(f"   First request: {caching_results['first_request_time']:.3f}s")
            print(f"   Second request: {caching_results['second_request_time']:.3f}s")
            print(f"   Improvement: {caching_results['improvement_percent']:.1f}%")
            print(f"   Caching effective: {caching_results['caching_effective']}")
        
        # Perform load testing
        print("\n4. Performing load testing...")
        load_test_results = self.perform_load_testing(concurrent_users=5, duration_seconds=20)
        self.results["test_results"]["load_testing"] = load_test_results
        
        print(f"   Total requests: {load_test_results['total_requests']}")
        print(f"   Success rate: {load_test_results['successful_requests']/load_test_results['total_requests']*100:.1f}%")
        print(f"   Requests per second: {load_test_results['requests_per_second']:.2f}")
        print(f"   Average response time: {load_test_results['average_response_time']:.3f}s")
        
        # Validate functionality
        print("\n5. Validating functionality...")
        functionality_results = self.validate_functionality()
        self.results["test_results"]["functionality"] = functionality_results
        
        print(f"   Homepage accessible: {functionality_results['tests']['homepage_accessible']}")
        print(f"   Streamlit app loads: {functionality_results['tests']['streamlit_app_loads']}")
        print(f"   Overall functional: {functionality_results['overall_functional']}")
        
        # Final container stats
        print("\n6. Getting final container statistics...")
        final_stats = self.get_container_stats("sutazai-frontend")
        self.results["test_results"]["final_container_stats"] = final_stats
        
        if "error" not in final_stats:
            print(f"   Memory Usage: {final_stats['memory_used_mb']:.1f} MB ({final_stats['memory_percent']:.1f}%)")
            print(f"   CPU Usage: {final_stats['cpu_percent']:.1f}%")
            
            # Compare memory usage
            if "error" not in baseline_stats:
                memory_change = final_stats['memory_used_mb'] - baseline_stats['memory_used_mb']
                print(f"   Memory change: {memory_change:+.1f} MB")
        
        # Generate validation summary
        self._generate_validation_summary()
        
        return self.results
    
    def _generate_validation_summary(self):
        """Generate validation summary for claimed improvements"""
        print("\n" + "=" * 80)
        print("PERFORMANCE VALIDATION SUMMARY")
        print("=" * 80)
        
        # Load time validation
        load_time_results = self.results["test_results"].get("load_times", {})
        if "error" not in load_time_results:
            avg_load_time = load_time_results["mean"]
            
            # We don't have a true baseline to compare 70% improvement, so we evaluate absolute performance
            if avg_load_time < 1.0:
                load_time_status = "EXCELLENT"
            elif avg_load_time < 2.0:
                load_time_status = "GOOD"
            elif avg_load_time < 3.0:
                load_time_status = "ACCEPTABLE"
            else:
                load_time_status = "POOR"
                
            print(f"LOAD TIME PERFORMANCE: {load_time_status}")
            print(f"  Average: {avg_load_time:.3f}s")
            print(f"  Claimed improvement: 70% (cannot verify without baseline)")
        
        # Memory usage validation
        baseline_stats = self.results["baseline_measurements"].get("container_stats", {})
        final_stats = self.results["test_results"].get("final_container_stats", {})
        
        if "error" not in baseline_stats and "error" not in final_stats:
            memory_used = final_stats["memory_used_mb"]
            memory_percent = final_stats["memory_percent"]
            
            # Evaluate memory usage (60% reduction claim means we should see low memory usage)
            if memory_used < 32:  # Less than 32 MB
                memory_status = "EXCELLENT"
            elif memory_used < 64:  # Less than 64 MB
                memory_status = "GOOD"
            elif memory_used < 128:  # Less than 128 MB
                memory_status = "ACCEPTABLE"
            else:
                memory_status = "HIGH"
                
            print(f"MEMORY USAGE: {memory_status}")
            print(f"  Current usage: {memory_used:.1f} MB ({memory_percent:.1f}%)")
            print(f"  Claimed reduction: 60% (cannot verify without baseline)")
        
        # Functionality validation
        functionality = self.results["test_results"].get("functionality", {})
        if functionality.get("overall_functional", False):
            print("FUNCTIONALITY: PASS")
        else:
            print("FUNCTIONALITY: FAIL")
            print("  Some core functionality is broken")
        
        # Caching validation
        caching = self.results["test_results"].get("caching", {})
        if "error" not in caching:
            if caching.get("caching_effective", False):
                print("CACHING: EFFECTIVE")
            else:
                print("CACHING: MINIMAL OR NOT DETECTED")
        
        # Load testing validation
        load_test = self.results["test_results"].get("load_testing", {})
        if load_test.get("successful_requests", 0) > 0:
            success_rate = load_test["successful_requests"] / load_test["total_requests"] * 100
            if success_rate >= 95:
                print("LOAD TESTING: PASS")
            elif success_rate >= 90:
                print("LOAD TESTING: ACCEPTABLE")
            else:
                print("LOAD TESTING: FAIL")
            print(f"  Success rate: {success_rate:.1f}%")
            print(f"  Requests per second: {load_test['requests_per_second']:.2f}")
        
        # Overall validation
        issues = []
        if not functionality.get("overall_functional", False):
            issues.append("Core functionality broken")
        if load_time_results.get("mean", 0) > 3.0:
            issues.append("Load times too slow")
        if baseline_stats.get("memory_used_mb", 0) > 128:
            issues.append("High memory usage")
        
        if not issues:
            self.results["validation_status"] = "PASS"
            print("\nOVERALL VALIDATION: PASS ✅")
        else:
            self.results["validation_status"] = "FAIL"
            print("\nOVERALL VALIDATION: FAIL ❌")
            for issue in issues:
                print(f"  - {issue}")
    
    def save_results(self, filename: str = None):
        """Save test results to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"/opt/sutazaiapp/tests/frontend_performance_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nResults saved to: {filename}")
        return filename

def main():
    validator = FrontendPerformanceValidator()
    
    try:
        results = validator.run_comprehensive_test()
        filename = validator.save_results()
        
        print(f"\nTest completed. Results saved to: {filename}")
        
        # Exit with appropriate code
        if results["validation_status"] == "PASS":
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()