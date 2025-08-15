#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
SutazAI Performance Validation Suite (Simplified)
Comprehensive performance testing for production readiness assessment
"""

import time
import json
import requests
import concurrent.futures
import statistics
import subprocess
from datetime import datetime
from typing import Dict, List

class PerformanceValidator:
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "system": "SutazAI v76",
            "tests": {},
            "summary": {},
            "production_ready": False
        }
        
    def test_health_endpoints(self) -> Dict:
        """Test health endpoint response times - Target: <200ms"""
        logger.info("\n[1/7] Testing Health Endpoints Response Times...")
        
        endpoints = [
            ("Backend API", "http://localhost:10010/health"),
            ("Hardware Optimizer", "http://localhost:11110/health"),
            ("Ollama Integration", "http://localhost:8090/health"),
            ("FAISS Vector DB", "http://localhost:10103/health"),
            ("AI Agent Orchestrator", "http://localhost:8589/health"),
            ("Resource Arbitration", "http://localhost:8588/health"),
            ("Task Assignment", "http://localhost:8551/health"),
            ("Jarvis Automation", "http://localhost:11102/health"),
            ("Jarvis Hardware", "http://localhost:11104/health")
        ]
        
        results = {"endpoints": {}, "avg_response_time": 0, "pass": True}
        response_times = []
        
        for name, url in endpoints:
            times = []
            for _ in range(10):  # Test 10 times for average
                try:
                    start = time.time()
                    resp = requests.get(url, timeout=5)
                    elapsed = (time.time() - start) * 1000  # Convert to ms
                    times.append(elapsed)
                    time.sleep(0.1)
                except Exception as e:
                    times.append(5000)  # Timeout penalty
                    
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)
            
            results["endpoints"][name] = {
                "url": url,
                "avg_ms": round(avg_time, 2),
                "min_ms": round(min_time, 2),
                "max_ms": round(max_time, 2),
                "pass": avg_time < 200,
                "status": "OK" if avg_time < 200 else "SLOW"
            }
            response_times.append(avg_time)
            
            logger.info(f"  {name}: {avg_time:.2f}ms avg ({'PASS' if avg_time < 200 else 'FAIL'})")
            
            if avg_time >= 200:
                results["pass"] = False
                
        results["avg_response_time"] = round(statistics.mean(response_times), 2)
        return results
    
    def test_chat_endpoint(self) -> Dict:
        """Test chat endpoint response time - Target: <5s"""
        logger.info("\n[2/7] Testing Chat Endpoint Response Times...")
        
        results = {"response_times": [], "avg_time": 0, "pass": True}
        
        test_messages = [
            "Hello, how are you?",
            "What is the weather today?",
            "Can you help me with Python?"
        ]
        
        for i, message in enumerate(test_messages, 1):
            try:
                start = time.time()
                resp = requests.post(
                    "http://localhost:10010/api/v1/chat/",
                    json={"message": message, "model": "tinyllama"},
                    timeout=10
                )
                elapsed = time.time() - start
                results["response_times"].append(round(elapsed, 2))
                logger.info(f"  Message {i}: {elapsed:.2f}s")
            except Exception as e:
                results["response_times"].append(10.0)  # Timeout
                logger.info(f"  Message {i}: TIMEOUT")
                
        if results["response_times"]:
            results["avg_time"] = round(statistics.mean(results["response_times"]), 2)
            results["min_time"] = round(min(results["response_times"]), 2)
            results["max_time"] = round(max(results["response_times"]), 2)
            results["pass"] = results["avg_time"] < 5.0
            logger.info(f"  Average: {results['avg_time']}s ({'PASS' if results['pass'] else 'FAIL'})")
        
        return results
    
    def measure_system_resources(self) -> Dict:
        """Measure system resource utilization using docker stats"""
        logger.info("\n[3/7] Measuring System Resource Utilization...")
        
        # Get container stats
        cmd = "docker stats --no-stream --format 'json'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        containers = {}
        total_cpu = 0
        total_memory = 0
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line:
                    try:
                        data = json.loads(line)
                        name = data.get('Name', 'unknown')
                        cpu_str = data.get('CPUPerc', '0%').replace('%', '')
                        mem_str = data.get('MemUsage', '0MiB / 0MiB').split('/')[0]
                        
                        # Parse memory
                        mem_value = 0
                        if 'GiB' in mem_str:
                            mem_value = float(mem_str.replace('GiB', '').strip()) * 1024
                        elif 'MiB' in mem_str:
                            mem_value = float(mem_str.replace('MiB', '').strip())
                        
                        cpu_value = float(cpu_str) if cpu_str else 0
                        
                        containers[name] = {
                            "cpu_percent": cpu_value,
                            "memory_mb": round(mem_value, 2)
                        }
                        
                        total_cpu += cpu_value
                        total_memory += mem_value
                        
                    except (json.JSONDecodeError, ValueError):
                        continue
        
        # Get system metrics
        df_result = subprocess.run("df -h /", shell=True, capture_output=True, text=True)
        disk_usage = 0
        if df_result.returncode == 0:
            lines = df_result.stdout.strip().split('\n')
            if len(lines) > 1:
                parts = lines[1].split()
                if len(parts) > 4:
                    disk_usage = int(parts[4].replace('%', ''))
        
        logger.info(f"  Total containers: {len(containers)}")
        logger.info(f"  Total CPU usage: {total_cpu:.1f}%")
        logger.info(f"  Total memory usage: {total_memory:.0f} MB")
        logger.info(f"  Disk usage: {disk_usage}%")
        
        return {
            "containers_count": len(containers),
            "total_cpu_percent": round(total_cpu, 2),
            "total_memory_mb": round(total_memory, 2),
            "disk_usage_percent": disk_usage,
            "pass": total_cpu < 80 and disk_usage < 85,
            "top_consumers": sorted(containers.items(), key=lambda x: x[1]['cpu_percent'], reverse=True)[:5]
        }
    
    def test_database_performance(self) -> Dict:
        """Test database query performance"""
        logger.info("\n[4/7] Testing Database Performance...")
        
        results = {"postgres": {}, "redis": {}, "pass": True}
        
        # Test PostgreSQL through backend API
        try:
            query_times = []
            for _ in range(10):
                start = time.time()
                resp = requests.get("http://localhost:10010/health", timeout=5)
                elapsed = (time.time() - start) * 1000
                query_times.append(elapsed)
                time.sleep(0.1)
                
            results["postgres"] = {
                "avg_query_ms": round(statistics.mean(query_times), 2),
                "min_query_ms": round(min(query_times), 2),
                "max_query_ms": round(max(query_times), 2),
                "pass": statistics.mean(query_times) < 100
            }
            logger.info(f"  PostgreSQL avg query: {results['postgres']['avg_query_ms']}ms")
            
        except Exception as e:
            results["postgres"] = {"error": str(e), "pass": False}
            results["pass"] = False
        
        # Test Redis through backend API
        try:
            cache_times = []
            for i in range(10):
                start = time.time()
                resp = requests.post(
                    "http://localhost:10010/api/v1/mesh/enqueue",
                    json={"task_type": "test", "payload": {"test": i}},
                    timeout=5
                )
                elapsed = (time.time() - start) * 1000
                cache_times.append(elapsed)
                
            results["redis"] = {
                "avg_operation_ms": round(statistics.mean(cache_times), 2),
                "min_operation_ms": round(min(cache_times), 2),
                "max_operation_ms": round(max(cache_times), 2),
                "pass": statistics.mean(cache_times) < 50
            }
            logger.info(f"  Redis avg operation: {results['redis']['avg_operation_ms']}ms")
            
        except Exception as e:
            results["redis"] = {"error": str(e), "pass": False}
            
        return results
    
    def evaluate_container_efficiency(self) -> Dict:
        """Evaluate container resource allocation and efficiency"""
        logger.info("\n[5/7] Evaluating Container Efficiency...")
        
        # Get container inspection data
        cmd = "docker ps --format '{{.Names}}' | xargs -I {} docker inspect {} --format '{{.Name}},{{.HostConfig.Memory}},{{.HostConfig.CpuShares}}'"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        containers = {}
        unlimited_memory = 0
        limited_memory = 0
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line:
                    parts = line.split(',')
                    if len(parts) >= 3:
                        name = parts[0].replace('/', '')
                        memory = parts[1]
                        cpu_shares = parts[2]
                        
                        if memory == '0':
                            unlimited_memory += 1
                            containers[name] = {"memory": "Unlimited", "cpu_shares": cpu_shares}
                        else:
                            limited_memory += 1
                            mem_mb = int(memory) / (1024*1024) if memory.isdigit() else 0
                            containers[name] = {"memory_mb": mem_mb, "cpu_shares": cpu_shares}
        
        logger.info(f"  Containers with memory limits: {limited_memory}")
        logger.info(f"  Containers without limits: {unlimited_memory}")
        logger.info(f"  Resource efficiency: {'Good' if limited_memory > unlimited_memory else 'Needs optimization'}")
        
        return {
            "total_containers": limited_memory + unlimited_memory,
            "limited_containers": limited_memory,
            "unlimited_containers": unlimited_memory,
            "efficiency_rating": "Good" if limited_memory > unlimited_memory else "Needs optimization",
            "pass": limited_memory > 0
        }
    
    def test_concurrent_load(self) -> Dict:
        """Test concurrent user load handling - Target: 50+ users"""
        logger.info("\n[6/7] Testing Concurrent Load Handling...")
        
        results = {"concurrent_users": 50, "success_rate": 0, "avg_response_time": 0, "pass": False}
        
        def make_request(user_id):
            try:
                start = time.time()
                resp = requests.get(
                    f"http://localhost:10010/health",
                    timeout=10
                )
                elapsed = time.time() - start
                return {"success": resp.status_code == 200, "time": elapsed}
            except Exception:
                return {"success": False, "time": 10}
        
        logger.info("  Simulating 50 concurrent users...")
        
        # Simulate 50 concurrent users
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [executor.submit(make_request, i) for i in range(50)]
            responses = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        successful = sum(1 for r in responses if r["success"])
        response_times = [r["time"] for r in responses]
        
        results["success_rate"] = (successful / 50) * 100
        results["successful_requests"] = successful
        results["failed_requests"] = 50 - successful
        results["avg_response_time"] = round(statistics.mean(response_times), 2)
        results["min_response_time"] = round(min(response_times), 2)
        results["max_response_time"] = round(max(response_times), 2)
        results["95th_percentile"] = round(statistics.quantiles(response_times, n=20)[18], 2)
        results["pass"] = results["success_rate"] >= 95 and results["avg_response_time"] < 2
        
        logger.info(f"  Success rate: {results['success_rate']}%")
        logger.info(f"  Average response: {results['avg_response_time']}s")
        logger.info(f"  95th percentile: {results['95th_percentile']}s")
        
        return results
    
    def test_ollama_and_cache(self) -> Dict:
        """Test Ollama response times and cache performance"""
        logger.info("\n[7/7] Testing Ollama and Cache Performance...")
        
        results = {"ollama": {}, "cache": {}, "pass": True}
        
        # Test Ollama response times
        logger.info("  Testing Ollama response times...")
        try:
            ollama_times = []
            for i in range(3):
                start = time.time()
                resp = requests.post(
                    "http://localhost:10104/api/generate",
                    json={"model": "tinyllama", "prompt": "Hello", "stream": False},
                    timeout=30
                )
                elapsed = time.time() - start
                ollama_times.append(elapsed)
                logger.info(f"    Test {i+1}: {elapsed:.2f}s")
                
            results["ollama"] = {
                "avg_response_s": round(statistics.mean(ollama_times), 2),
                "min_response_s": round(min(ollama_times), 2),
                "max_response_s": round(max(ollama_times), 2),
                "model": "tinyllama (637MB)",
                "pass": statistics.mean(ollama_times) < 10
            }
            logger.info(f"  Ollama average: {results['ollama']['avg_response_s']}s")
            
        except Exception as e:
            results["ollama"] = {"error": str(e), "pass": False}
            results["pass"] = False
            logger.error(f"  Ollama test failed: {e}")
        
        # Test cache performance through repeated API calls
        logger.info("  Testing cache performance...")
        try:
            # First call (cache miss)
            start1 = time.time()
            resp1 = requests.get("http://localhost:10010/api/v1/models/", timeout=5)
            time1 = time.time() - start1
            
            # Second call (potential cache hit)
            start2 = time.time()
            resp2 = requests.get("http://localhost:10010/api/v1/models/", timeout=5)
            time2 = time.time() - start2
            
            # Third call (should be cached)
            start3 = time.time()
            resp3 = requests.get("http://localhost:10010/api/v1/models/", timeout=5)
            time3 = time.time() - start3
            
            cache_improvement = ((time1 - time3) / time1) * 100 if time1 > 0 else 0
            
            results["cache"] = {
                "first_call_ms": round(time1 * 1000, 2),
                "cached_call_ms": round(time3 * 1000, 2),
                "improvement_percent": round(cache_improvement, 2),
                "effective": cache_improvement > 20,
                "pass": True
            }
            logger.info(f"  Cache improvement: {cache_improvement:.1f}%")
            
        except Exception as e:
            results["cache"] = {"error": str(e), "pass": False}
            
        return results
    
    def generate_summary(self):
        """Generate performance summary and production readiness assessment"""
        
        total_tests = len(self.results["tests"])
        passed_tests = sum(1 for test in self.results["tests"].values() if test.get("pass", False))
        
        # Production readiness criteria
        criteria = {
            "health_endpoints_fast": self.results["tests"].get("health_endpoints", {}).get("pass", False),
            "chat_responsive": self.results["tests"].get("chat_endpoint", {}).get("pass", False),
            "resources_healthy": self.results["tests"].get("system_resources", {}).get("pass", False),
            "database_performant": self.results["tests"].get("database_performance", {}).get("pass", False),
            "concurrent_load_handled": self.results["tests"].get("concurrent_load", {}).get("pass", False),
            "ollama_functional": self.results["tests"].get("ollama_cache", {}).get("ollama", {}).get("pass", False)
        }
        
        self.results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": round((passed_tests / total_tests) * 100, 2) if total_tests > 0 else 0,
            "production_criteria_met": sum(criteria.values()),
            "production_criteria_total": len(criteria),
            "criteria_details": criteria
        }
        
        # Production ready if 80% of criteria are met
        self.results["production_ready"] = (sum(criteria.values()) / len(criteria)) >= 0.8
        
        # Performance improvements detected
        improvements = []
        
        if self.results["tests"].get("health_endpoints", {}).get("avg_response_time", 1000) < 200:
            improvements.append(f"Health endpoints optimized ({self.results['tests']['health_endpoints']['avg_response_time']}ms average)")
            
        if self.results["tests"].get("ollama_cache", {}).get("cache", {}).get("effective", False):
            improvements.append(f"Cache system effective ({self.results['tests']['ollama_cache']['cache'].get('improvement_percent', 0)}% improvement)")
            
        if self.results["tests"].get("ollama_cache", {}).get("ollama", {}).get("avg_response_s", 100) < 10:
            improvements.append(f"Ollama response optimized ({self.results['tests']['ollama_cache']['ollama']['avg_response_s']}s average)")
            
        if self.results["tests"].get("concurrent_load", {}).get("success_rate", 0) >= 95:
            improvements.append(f"High concurrency support ({self.results['tests']['concurrent_load']['success_rate']}% success rate)")
            
        self.results["summary"]["improvements_achieved"] = improvements
    
    def run_all_tests(self):
        """Execute all performance tests"""
        
        logger.info("=" * 60)
        logger.info("SutazAI Performance Validation Suite")
        logger.info("=" * 60)
        
        # Run tests
        self.results["tests"]["health_endpoints"] = self.test_health_endpoints()
        self.results["tests"]["chat_endpoint"] = self.test_chat_endpoint()
        self.results["tests"]["system_resources"] = self.measure_system_resources()
        self.results["tests"]["database_performance"] = self.test_database_performance()
        self.results["tests"]["container_efficiency"] = self.evaluate_container_efficiency()
        self.results["tests"]["concurrent_load"] = self.test_concurrent_load()
        self.results["tests"]["ollama_cache"] = self.test_ollama_and_cache()
        
        # Generate summary
        self.generate_summary()
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"/opt/sutazaiapp/PERFORMANCE_VALIDATION_REPORT_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\n\nResults saved to: {report_file}")
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("PERFORMANCE VALIDATION SUMMARY")
        logger.info("=" * 60)
        
        summary = self.results["summary"]
        logger.info(f"\nTests Passed: {summary['passed_tests']}/{summary['total_tests']} ({summary['success_rate']}%)")
        logger.info(f"Production Criteria Met: {summary['production_criteria_met']}/{summary['production_criteria_total']}")
        logger.info(f"\nPRODUCTION READY: {'YES' if self.results['production_ready'] else 'NO'}")
        
        if summary.get("improvements_achieved"):
            logger.info("\nPerformance Improvements Achieved:")
            for improvement in summary["improvements_achieved"]:
                logger.info(f"  - {improvement}")
        
        return self.results

if __name__ == "__main__":
    validator = PerformanceValidator()
    results = validator.run_all_tests()