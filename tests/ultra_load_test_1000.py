#!/usr/bin/env python3
"""
ULTRA Load Test - 1000+ Concurrent Users
Test system performance under extreme load conditions
Target: <2s response time, 95%+ success rate, zero failures
"""

import asyncio
import aiohttp
import time
import json
import statistics
import concurrent.futures
import threading
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/tests/ultra_load_test_results.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class LoadTestResult:
    """Store load test results"""
    endpoint: str
    method: str
    concurrent_users: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p95_response_time: float
    p99_response_time: float
    requests_per_second: float
    error_rate: float
    test_duration: float
    errors: List[str]

class UltraLoadTester:
    """Ultra comprehensive load testing system"""
    
    def __init__(self):
        self.base_url = "http://localhost"
        self.endpoints = [
            {"path": ":10010/health", "method": "GET", "name": "backend_health"},
            {"path": ":10010/metrics", "method": "GET", "name": "backend_metrics"},
            {"path": ":10010/api/v1/models/", "method": "GET", "name": "models_list"},
            {"path": ":10011/", "method": "GET", "name": "frontend_home"},
            {"path": ":10104/api/tags", "method": "GET", "name": "ollama_tags"},
            {"path": ":10101/", "method": "GET", "name": "qdrant_home"},
            {"path": ":10100/api/v1/heartbeat", "method": "GET", "name": "chromadb_heartbeat"},
            {"path": ":10103/health", "method": "GET", "name": "faiss_health"},
            {"path": ":8589/health", "method": "GET", "name": "agent_orchestrator"},
            {"path": ":8090/health", "method": "GET", "name": "ollama_integration"},
            {"path": ":11110/health", "method": "GET", "name": "hardware_optimizer"},
            {"path": ":8588/health", "method": "GET", "name": "resource_arbitration"},
            {"path": ":8551/health", "method": "GET", "name": "task_coordinator"}
        ]
        self.results: List[LoadTestResult] = []
        
    async def make_request(self, session: aiohttp.ClientSession, url: str, method: str = "GET") -> Dict[str, Any]:
        """Make a single HTTP request and measure performance"""
        start_time = time.time()
        
        try:
            timeout = aiohttp.ClientTimeout(total=30)  # 30 second timeout
            
            if method.upper() == "GET":
                async with session.get(url, timeout=timeout) as response:
                    response_time = time.time() - start_time
                    content = await response.text()
                    
                    return {
                        "success": True,
                        "status_code": response.status,
                        "response_time": response_time,
                        "content_length": len(content),
                        "error": None
                    }
            elif method.upper() == "POST":
                async with session.post(url, json={}, timeout=timeout) as response:
                    response_time = time.time() - start_time
                    content = await response.text()
                    
                    return {
                        "success": True,
                        "status_code": response.status,
                        "response_time": response_time,
                        "content_length": len(content),
                        "error": None
                    }
                    
        except Exception as e:
            response_time = time.time() - start_time
            return {
                "success": False,
                "status_code": 0,
                "response_time": response_time,
                "content_length": 0,
                "error": str(e)
            }
    
    async def load_test_endpoint(self, endpoint: Dict[str, str], concurrent_users: int, requests_per_user: int) -> LoadTestResult:
        """Run load test against a single endpoint"""
        url = f"{self.base_url}{endpoint['path']}"
        method = endpoint.get('method', 'GET')
        endpoint_name = endpoint['name']
        
        logger.info(f"Starting load test for {endpoint_name}: {concurrent_users} users, {requests_per_user} requests each")
        
        # Track all metrics
        all_response_times = []
        successful_requests = 0
        failed_requests = 0
        errors = []
        start_time = time.time()
        
        # Create connector with appropriate limits
        connector = aiohttp.TCPConnector(
            limit=concurrent_users * 2,  # Connection pool size
            limit_per_host=concurrent_users * 2,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            
            async def user_simulation():
                """Simulate a single user making multiple requests"""
                user_response_times = []
                user_errors = []
                user_successful = 0
                user_failed = 0
                
                for _ in range(requests_per_user):
                    result = await self.make_request(session, url, method)
                    
                    user_response_times.append(result['response_time'])
                    
                    if result['success'] and 200 <= result['status_code'] < 400:
                        user_successful += 1
                    else:
                        user_failed += 1
                        if result['error']:
                            user_errors.append(result['error'])
                    
                    # Small delay between requests from same user
                    await asyncio.sleep(0.01)
                
                return user_response_times, user_successful, user_failed, user_errors
            
            # Create all user tasks
            user_tasks = [user_simulation() for _ in range(concurrent_users)]
            
            # Execute all users concurrently
            user_results = await asyncio.gather(*user_tasks, return_exceptions=True)
            
            # Aggregate results
            for result in user_results:
                if isinstance(result, Exception):
                    logger.error(f"User simulation failed: {result}")
                    failed_requests += requests_per_user
                    errors.append(str(result))
                else:
                    response_times, user_successful, user_failed, user_errors = result
                    all_response_times.extend(response_times)
                    successful_requests += user_successful
                    failed_requests += user_failed
                    errors.extend(user_errors)
        
        end_time = time.time()
        test_duration = end_time - start_time
        total_requests = concurrent_users * requests_per_user
        
        # Calculate metrics
        if all_response_times:
            avg_response_time = statistics.mean(all_response_times)
            min_response_time = min(all_response_times)
            max_response_time = max(all_response_times)
            
            # Calculate percentiles
            sorted_times = sorted(all_response_times)
            p95_index = int(0.95 * len(sorted_times))
            p99_index = int(0.99 * len(sorted_times))
            p95_response_time = sorted_times[p95_index] if p95_index < len(sorted_times) else max_response_time
            p99_response_time = sorted_times[p99_index] if p99_index < len(sorted_times) else max_response_time
        else:
            avg_response_time = min_response_time = max_response_time = p95_response_time = p99_response_time = 0
        
        requests_per_second = total_requests / test_duration if test_duration > 0 else 0
        error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0
        
        result = LoadTestResult(
            endpoint=endpoint_name,
            method=method,
            concurrent_users=concurrent_users,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            test_duration=test_duration,
            errors=errors[:10]  # Keep only first 10 errors
        )
        
        logger.info(f"Completed {endpoint_name}: {successful_requests}/{total_requests} success, "
                   f"avg={avg_response_time:.3f}s, p95={p95_response_time:.3f}s, "
                   f"rps={requests_per_second:.1f}, error_rate={error_rate:.1f}%")
        
        return result
    
    def run_progressive_load_test(self) -> Dict[str, Any]:
        """Run progressive load test with increasing concurrent users"""
        logger.info("Starting ULTRA Progressive Load Test")
        
        test_scenarios = [
            {"users": 10, "requests": 10, "name": "warmup"},
            {"users": 50, "requests": 20, "name": "light_load"},
            {"users": 100, "requests": 20, "name": "medium_load"},
            {"users": 250, "requests": 20, "name": "heavy_load"},
            {"users": 500, "requests": 20, "name": "stress_load"},
            {"users": 1000, "requests": 10, "name": "ultra_load"},
            {"users": 1500, "requests": 5, "name": "extreme_load"}
        ]
        
        all_results = []
        
        for scenario in test_scenarios:
            logger.info(f"\n=== Running {scenario['name'].upper()} TEST ===")
            logger.info(f"Concurrent Users: {scenario['users']}, Requests per User: {scenario['requests']}")
            
            scenario_results = []
            
            # Test critical endpoints under this load
            critical_endpoints = [
                {"path": ":10010/health", "method": "GET", "name": "backend_health"},
                {"path": ":10010/metrics", "method": "GET", "name": "backend_metrics"},
                {"path": ":10011/", "method": "GET", "name": "frontend_home"},
                {"path": ":10104/api/tags", "method": "GET", "name": "ollama_tags"},
                {"path": ":11110/health", "method": "GET", "name": "hardware_optimizer"}
            ]
            
            # Run tests for critical endpoints
            for endpoint in critical_endpoints:
                try:
                    result = asyncio.run(
                        self.load_test_endpoint(endpoint, scenario['users'], scenario['requests'])
                    )
                    scenario_results.append(result)
                    self.results.append(result)
                    
                    # Check if we're hitting performance targets
                    if result.avg_response_time > 2.0:
                        logger.warning(f"PERFORMANCE WARNING: {endpoint['name']} avg response time "
                                     f"{result.avg_response_time:.3f}s exceeds 2s target")
                    
                    if result.error_rate > 5.0:
                        logger.warning(f"ERROR RATE WARNING: {endpoint['name']} error rate "
                                     f"{result.error_rate:.1f}% exceeds 5% threshold")
                    
                except Exception as e:
                    logger.error(f"Failed to test {endpoint['name']}: {e}")
            
            # Brief pause between scenarios
            time.sleep(2)
            
            all_results.append({
                "scenario": scenario['name'],
                "users": scenario['users'],
                "requests_per_user": scenario['requests'],
                "results": scenario_results
            })
        
        return {
            "test_type": "progressive_load_test",
            "timestamp": datetime.now().isoformat(),
            "total_scenarios": len(test_scenarios),
            "scenarios": all_results,
            "summary": self.generate_summary()
        }
    
    def run_sustained_load_test(self) -> Dict[str, Any]:
        """Run sustained load test at target capacity"""
        logger.info("Starting ULTRA Sustained Load Test - 1000 users for 5 minutes")
        
        # Sustained test parameters
        concurrent_users = 1000
        test_duration_minutes = 5
        requests_per_minute = 12  # 1 request every 5 seconds per user
        
        # Calculate total requests
        total_requests_per_user = test_duration_minutes * requests_per_minute
        
        # Test all endpoints under sustained load
        sustained_results = []
        
        for endpoint in self.endpoints:
            try:
                logger.info(f"Sustained test on {endpoint['name']}: {concurrent_users} users, "
                           f"{total_requests_per_user} requests each over {test_duration_minutes} minutes")
                
                result = asyncio.run(
                    self.load_test_endpoint(endpoint, concurrent_users, total_requests_per_user)
                )
                sustained_results.append(result)
                self.results.append(result)
                
            except Exception as e:
                logger.error(f"Failed sustained test for {endpoint['name']}: {e}")
        
        return {
            "test_type": "sustained_load_test",
            "timestamp": datetime.now().isoformat(),
            "concurrent_users": concurrent_users,
            "test_duration_minutes": test_duration_minutes,
            "results": sustained_results,
            "summary": self.generate_summary()
        }
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive test summary"""
        if not self.results:
            return {"status": "no_results"}
        
        # Overall metrics
        total_requests = sum(r.total_requests for r in self.results)
        total_successful = sum(r.successful_requests for r in self.results)
        total_failed = sum(r.failed_requests for r in self.results)
        
        overall_success_rate = (total_successful / total_requests * 100) if total_requests > 0 else 0
        overall_error_rate = (total_failed / total_requests * 100) if total_requests > 0 else 0
        
        # Response time analysis
        all_avg_times = [r.avg_response_time for r in self.results if r.avg_response_time > 0]
        all_p95_times = [r.p95_response_time for r in self.results if r.p95_response_time > 0]
        
        avg_response_time = statistics.mean(all_avg_times) if all_avg_times else 0
        avg_p95_time = statistics.mean(all_p95_times) if all_p95_times else 0
        
        # Performance targets
        performance_targets = {
            "avg_response_time_target": 2.0,
            "p95_response_time_target": 3.0,
            "success_rate_target": 95.0,
            "error_rate_target": 5.0
        }
        
        # Check if targets are met
        targets_met = {
            "avg_response_time": avg_response_time <= performance_targets["avg_response_time_target"],
            "p95_response_time": avg_p95_time <= performance_targets["p95_response_time_target"],
            "success_rate": overall_success_rate >= performance_targets["success_rate_target"],
            "error_rate": overall_error_rate <= performance_targets["error_rate_target"]
        }
        
        all_targets_met = all(targets_met.values())
        
        # Generate performance grade
        grade_score = 0
        grade_score += 25 if targets_met["avg_response_time"] else 0
        grade_score += 25 if targets_met["p95_response_time"] else 0
        grade_score += 25 if targets_met["success_rate"] else 0
        grade_score += 25 if targets_met["error_rate"] else 0
        
        if grade_score == 100:
            performance_grade = "A+ (ULTRA PERFECT)"
        elif grade_score >= 90:
            performance_grade = "A (EXCELLENT)"
        elif grade_score >= 80:
            performance_grade = "B (GOOD)"
        elif grade_score >= 70:
            performance_grade = "C (ACCEPTABLE)"
        else:
            performance_grade = "F (NEEDS IMPROVEMENT)"
        
        return {
            "performance_grade": performance_grade,
            "grade_score": grade_score,
            "all_targets_met": all_targets_met,
            "total_requests": total_requests,
            "successful_requests": total_successful,
            "failed_requests": total_failed,
            "overall_success_rate": round(overall_success_rate, 2),
            "overall_error_rate": round(overall_error_rate, 2),
            "avg_response_time": round(avg_response_time, 3),
            "avg_p95_response_time": round(avg_p95_time, 3),
            "performance_targets": performance_targets,
            "targets_met": targets_met,
            "endpoint_count": len(set(r.endpoint for r in self.results)),
            "test_scenarios": len([r for r in self.results]),
            "max_concurrent_users": max(r.concurrent_users for r in self.results) if self.results else 0
        }
    
    def save_results(self, test_data: Dict[str, Any]):
        """Save test results to JSON file"""
        filename = f"/opt/sutazaiapp/tests/ultra_load_test_results_{int(time.time())}.json"
        
        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            serializable_results.append(asdict(result))
        
        output_data = {
            **test_data,
            "detailed_results": serializable_results,
            "test_execution": {
                "start_time": datetime.now().isoformat(),
                "python_version": sys.version,
                "script_path": __file__
            }
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"Test results saved to: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return None

def main():
    """Run the ULTRA load test suite"""
    logger.info("🚀 ULTRA LOAD TEST - 1000+ CONCURRENT USERS")
    logger.info("=" * 60)
    
    tester = UltraLoadTester()
    
    # Run progressive load test
    logger.info("\n📈 Running Progressive Load Test...")
    progressive_results = tester.run_progressive_load_test()
    
    # Save progressive results
    progressive_file = tester.save_results(progressive_results)
    
    # Run sustained load test
    logger.info("\n⏱️  Running Sustained Load Test...")
    sustained_results = tester.run_sustained_load_test()
    
    # Save sustained results
    sustained_file = tester.save_results(sustained_results)
    
    # Final summary
    summary = tester.generate_summary()
    
    logger.info("\n" + "=" * 60)
    logger.info("🎯 ULTRA LOAD TEST RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Performance Grade: {summary['performance_grade']}")
    logger.info(f"Grade Score: {summary['grade_score']}/100")
    logger.info(f"All Targets Met: {'✅ YES' if summary['all_targets_met'] else '❌ NO'}")
    logger.info(f"Total Requests: {summary['total_requests']:,}")
    logger.info(f"Success Rate: {summary['overall_success_rate']:.1f}%")
    logger.error(f"Error Rate: {summary['overall_error_rate']:.1f}%")
    logger.info(f"Avg Response Time: {summary['avg_response_time']:.3f}s")
    logger.info(f"P95 Response Time: {summary['avg_p95_response_time']:.3f}s")
    logger.info(f"Max Concurrent Users: {summary['max_concurrent_users']}")
    
    # Performance targets check
    logger.info("\n📊 Performance Targets:")
    targets = summary['targets_met']
    logger.info(f"Response Time < 2s: {'✅' if targets['avg_response_time'] else '❌'}")
    logger.info(f"P95 Response Time < 3s: {'✅' if targets['p95_response_time'] else '❌'}")
    logger.info(f"Success Rate > 95%: {'✅' if targets['success_rate'] else '❌'}")
    logger.error(f"Error Rate < 5%: {'✅' if targets['error_rate'] else '❌'}")
    
    logger.info(f"\n📄 Results saved to:")
    if progressive_file:
        logger.info(f"  - {progressive_file}")
    if sustained_file:
        logger.info(f"  - {sustained_file}")
    
    # Return exit code based on success
    return 0 if summary['all_targets_met'] else 1

if __name__ == "__main__":
    exit(main())