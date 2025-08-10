#!/usr/bin/env python3
"""
Ultra QA Team Lead - Frontend Stress Testing and Validation
Extended testing for extreme load scenarios and memory pressure
"""

import logging

# Configure logger for exception handling
logger = logging.getLogger(__name__)

import time
import requests
import json
import subprocess
import threading
from datetime import datetime
from typing import Dict, List, Any
import statistics
import concurrent.futures
import sys
import os
import psutil

class FrontendStressValidator:
    def __init__(self):
        self.frontend_url = "http://localhost:10011"
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "stress_tests": {},
            "performance_analysis": {},
            "validation_status": "PENDING"
        }
        
    def extreme_load_test(self, concurrent_users: int = 20, duration_seconds: int = 60) -> Dict[str, Any]:
        """Extreme concurrent load testing"""
        results = {
            "concurrent_users": concurrent_users,
            "duration_seconds": duration_seconds,
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
            "error_rates": {},
            "memory_samples": []
        }
        
        print(f"Starting extreme load test: {concurrent_users} concurrent users for {duration_seconds} seconds")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        response_times = []
        request_count = 0
        success_count = 0
        errors = {}
        memory_samples = []
        
        # Memory monitoring thread
        def monitor_memory():
            while time.time() < end_time:
                try:
                    stats = self.get_container_stats("sutazai-frontend")
                    if "error" not in stats:
                        memory_samples.append({
                            "timestamp": time.time() - start_time,
                            "memory_mb": stats["memory_used_mb"],
                            "memory_percent": stats["memory_percent"],
                            "cpu_percent": stats["cpu_percent"]
                        })
                except (AssertionError, Exception) as e:
                    # Suppressed exception (was bare except)
                    logger.debug(f"Suppressed exception: {e}")
                    pass
                time.sleep(2)
        
        monitor_thread = threading.Thread(target=monitor_memory)
        monitor_thread.start()
        
        def make_aggressive_requests():
            nonlocal request_count, success_count, errors
            
            while time.time() < end_time:
                try:
                    request_start = time.time()
                    response = requests.get(self.frontend_url, timeout=5)
                    request_end = time.time()
                    
                    request_count += 1
                    response_time = request_end - request_start
                    response_times.append(response_time)
                    
                    if response.status_code == 200:
                        success_count += 1
                    else:
                        error_key = f"HTTP_{response.status_code}"
                        errors[error_key] = errors.get(error_key, 0) + 1
                        
                except requests.exceptions.Timeout:
                    request_count += 1
                    errors["TIMEOUT"] = errors.get("TIMEOUT", 0) + 1
                except requests.exceptions.ConnectionError:
                    request_count += 1
                    errors["CONNECTION_ERROR"] = errors.get("CONNECTION_ERROR", 0) + 1
                except Exception as e:
                    request_count += 1
                    error_key = str(type(e).__name__)
                    errors[error_key] = errors.get(error_key, 0) + 1
                    
                # Minimal sleep to create realistic load
                time.sleep(0.01)
        
        # Start aggressive concurrent threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = [executor.submit(make_aggressive_requests) for _ in range(concurrent_users)]
            concurrent.futures.wait(futures)
        
        monitor_thread.join()
        
        actual_duration = time.time() - start_time
        
        # Calculate statistics
        if response_times:
            results.update({
                "total_requests": request_count,
                "successful_requests": success_count,
                "failed_requests": request_count - success_count,
                "success_rate_percent": (success_count / request_count * 100) if request_count > 0 else 0,
                "requests_per_second": request_count / actual_duration,
                "average_response_time": statistics.mean(response_times),
                "median_response_time": statistics.median(response_times),
                "p95_response_time": self.percentile(response_times, 95),
                "p99_response_time": self.percentile(response_times, 99),
                "max_response_time": max(response_times),
                "error_rates": errors,
                "memory_samples": memory_samples,
                "actual_duration": actual_duration
            })
        
        return results
    
    def memory_pressure_test(self, iterations: int = 100) -> Dict[str, Any]:
        """Test frontend under memory pressure scenarios"""
        print(f"Testing memory pressure with {iterations} rapid requests...")
        
        memory_snapshots = []
        response_times = []
        
        # Get initial memory
        initial_stats = self.get_container_stats("sutazai-frontend")
        
        for i in range(iterations):
            # Make request and measure
            start_time = time.time()
            try:
                response = requests.get(self.frontend_url, timeout=10)
                end_time = time.time()
                
                response_times.append(end_time - start_time)
                
                # Get memory stats every 10 requests
                if i % 10 == 0:
                    stats = self.get_container_stats("sutazai-frontend")
                    if "error" not in stats:
                        memory_snapshots.append({
                            "iteration": i,
                            "memory_mb": stats["memory_used_mb"],
                            "memory_percent": stats["memory_percent"],
                            "cpu_percent": stats["cpu_percent"]
                        })
                
            except Exception as e:
                print(f"  Request {i+1} failed: {e}")
            
            # Brief pause
            time.sleep(0.05)
        
        # Get final memory
        final_stats = self.get_container_stats("sutazai-frontend")
        
        return {
            "iterations": iterations,
            "initial_memory_mb": initial_stats.get("memory_used_mb", 0),
            "final_memory_mb": final_stats.get("memory_used_mb", 0),
            "memory_change_mb": final_stats.get("memory_used_mb", 0) - initial_stats.get("memory_used_mb", 0),
            "memory_snapshots": memory_snapshots,
            "average_response_time": statistics.mean(response_times) if response_times else 0,
            "response_time_degradation": self.calculate_degradation(response_times)
        }
    
    def calculate_degradation(self, response_times: List[float]) -> Dict[str, float]:
        """Calculate performance degradation over time"""
        if len(response_times) < 10:
            return {"error": "Insufficient data"}
        
        # Compare first 10% vs last 10%
        first_chunk = response_times[:len(response_times)//10]
        last_chunk = response_times[-len(response_times)//10:]
        
        first_avg = statistics.mean(first_chunk)
        last_avg = statistics.mean(last_chunk)
        
        degradation_percent = ((last_avg - first_avg) / first_avg * 100) if first_avg > 0 else 0
        
        return {
            "first_chunk_avg": first_avg,
            "last_chunk_avg": last_avg,
            "degradation_percent": degradation_percent,
            "performance_stable": abs(degradation_percent) < 20
        }
    
    def percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data"""
        if not data:
            return 0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * percentile / 100
        f = int(k)
        c = f + 1 if f + 1 < len(sorted_data) else f
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
    
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
    
    def run_stress_validation(self) -> Dict[str, Any]:
        """Run comprehensive stress testing"""
        print("=" * 80)
        print("ULTRA QA TEAM LEAD - Frontend Stress Testing Validation")
        print("=" * 80)
        
        # Test 1: Extreme Load Testing
        print("\n1. Extreme Load Testing (20 concurrent users, 60 seconds)...")
        extreme_load_results = self.extreme_load_test(concurrent_users=20, duration_seconds=60)
        self.results["stress_tests"]["extreme_load"] = extreme_load_results
        
        if extreme_load_results.get("total_requests", 0) > 0:
            print(f"   Total requests: {extreme_load_results['total_requests']}")
            print(f"   Success rate: {extreme_load_results['success_rate_percent']:.1f}%")
            print(f"   Requests per second: {extreme_load_results['requests_per_second']:.2f}")
            print(f"   Average response time: {extreme_load_results['average_response_time']:.3f}s")
            print(f"   P95 response time: {extreme_load_results['p95_response_time']:.3f}s")
            print(f"   P99 response time: {extreme_load_results['p99_response_time']:.3f}s")
            
            if extreme_load_results["error_rates"]:
                print(f"   Errors: {extreme_load_results['error_rates']}")
        
        # Test 2: Memory Pressure Testing
        print("\n2. Memory Pressure Testing (100 rapid requests)...")
        memory_pressure_results = self.memory_pressure_test(iterations=100)
        self.results["stress_tests"]["memory_pressure"] = memory_pressure_results
        
        print(f"   Initial memory: {memory_pressure_results['initial_memory_mb']:.1f} MB")
        print(f"   Final memory: {memory_pressure_results['final_memory_mb']:.1f} MB")
        print(f"   Memory change: {memory_pressure_results['memory_change_mb']:+.1f} MB")
        print(f"   Average response time: {memory_pressure_results['average_response_time']:.3f}s")
        
        degradation = memory_pressure_results["response_time_degradation"]
        if "error" not in degradation:
            print(f"   Performance degradation: {degradation['degradation_percent']:+.1f}%")
            print(f"   Performance stable: {degradation['performance_stable']}")
        
        # Analysis
        self._generate_stress_analysis()
        
        return self.results
    
    def _generate_stress_analysis(self):
        """Generate comprehensive stress test analysis"""
        print("\n" + "=" * 80)
        print("STRESS TESTING ANALYSIS")
        print("=" * 80)
        
        extreme_load = self.results["stress_tests"].get("extreme_load", {})
        memory_pressure = self.results["stress_tests"].get("memory_pressure", {})
        
        # Extreme Load Analysis
        if extreme_load.get("total_requests", 0) > 0:
            success_rate = extreme_load["success_rate_percent"]
            rps = extreme_load["requests_per_second"]
            p95_time = extreme_load["p95_response_time"]
            
            print("EXTREME LOAD TEST RESULTS:")
            
            # Success rate evaluation
            if success_rate >= 99:
                print("  Success Rate: EXCELLENT (≥99%)")
            elif success_rate >= 95:
                print("  Success Rate: GOOD (≥95%)")
            elif success_rate >= 90:
                print("  Success Rate: ACCEPTABLE (≥90%)")
            else:
                print("  Success Rate: POOR (<90%)")
            print(f"    Actual: {success_rate:.1f}%")
            
            # Throughput evaluation
            if rps >= 100:
                print("  Throughput: EXCELLENT (≥100 RPS)")
            elif rps >= 50:
                print("  Throughput: GOOD (≥50 RPS)")
            elif rps >= 25:
                print("  Throughput: ACCEPTABLE (≥25 RPS)")
            else:
                print("  Throughput: POOR (<25 RPS)")
            print(f"    Actual: {rps:.1f} RPS")
            
            # Response time evaluation
            if p95_time <= 0.1:
                print("  P95 Response Time: EXCELLENT (≤100ms)")
            elif p95_time <= 0.5:
                print("  P95 Response Time: GOOD (≤500ms)")
            elif p95_time <= 1.0:
                print("  P95 Response Time: ACCEPTABLE (≤1s)")
            else:
                print("  P95 Response Time: POOR (>1s)")
            print(f"    Actual: {p95_time:.3f}s")
        
        # Memory Pressure Analysis
        if memory_pressure.get("iterations", 0) > 0:
            memory_change = memory_pressure["memory_change_mb"]
            degradation = memory_pressure["response_time_degradation"]
            
            print("\nMEMORY PRESSURE TEST RESULTS:")
            
            # Memory stability
            if abs(memory_change) <= 5:
                print("  Memory Stability: EXCELLENT (≤5MB change)")
            elif abs(memory_change) <= 10:
                print("  Memory Stability: GOOD (≤10MB change)")
            elif abs(memory_change) <= 20:
                print("  Memory Stability: ACCEPTABLE (≤20MB change)")
            else:
                print("  Memory Stability: POOR (>20MB change)")
            print(f"    Memory change: {memory_change:+.1f} MB")
            
            # Performance stability
            if "error" not in degradation:
                if degradation["performance_stable"]:
                    print("  Performance Stability: STABLE")
                else:
                    print("  Performance Stability: UNSTABLE")
                print(f"    Degradation: {degradation['degradation_percent']:+.1f}%")
        
        # Overall Assessment
        issues = []
        
        if extreme_load.get("success_rate_percent", 0) < 95:
            issues.append("Low success rate under extreme load")
        if extreme_load.get("p95_response_time", 0) > 1.0:
            issues.append("High P95 response times")
        if abs(memory_pressure.get("memory_change_mb", 0)) > 20:
            issues.append("Excessive memory usage changes")
        if not memory_pressure.get("response_time_degradation", {}).get("performance_stable", True):
            issues.append("Performance degradation under load")
        
        print("\nOVERALL STRESS TEST ASSESSMENT:")
        if not issues:
            self.results["validation_status"] = "EXCELLENT"
            print("  Status: EXCELLENT ✅")
            print("  The frontend handles extreme load conditions very well")
        elif len(issues) <= 1:
            self.results["validation_status"] = "GOOD"
            print("  Status: GOOD ✅")
            print("  Minor issues under extreme conditions")
        else:
            self.results["validation_status"] = "NEEDS_IMPROVEMENT"
            print("  Status: NEEDS IMPROVEMENT ⚠️")
            for issue in issues:
                print(f"    - {issue}")
    
    def save_results(self, filename: str = None):
        """Save stress test results to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"/opt/sutazaiapp/tests/frontend_stress_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nStress test results saved to: {filename}")
        return filename

def main():
    validator = FrontendStressValidator()
    
    try:
        results = validator.run_stress_validation()
        filename = validator.save_results()
        
        print(f"\nStress testing completed. Results saved to: {filename}")
        
        # Exit with appropriate code
        if results["validation_status"] in ["EXCELLENT", "GOOD"]:
            sys.exit(0)
        else:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nStress test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nStress test failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()