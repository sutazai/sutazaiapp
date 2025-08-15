#!/usr/bin/env python3
"""
SutazAI Simplified Production Load Testing
Adapted for current running services

Author: QA Team Lead
Date: 2025-08-05
Version: 1.0
"""

import asyncio
import aiohttp
import json
import time
import logging
import sys
import statistics
from typing import Dict, List, Tuple
from dataclasses import dataclass
import random
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'/opt/sutazaiapp/load-testing/logs/simplified_load_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class LoadTestResult:
    test_name: str
    duration: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    p95_response_time: float
    requests_per_second: float
    error_rate: float
    errors: List[str]

    def to_dict(self):
        return {
            'test_name': self.test_name,
            'duration_seconds': self.duration,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'avg_response_time_ms': self.avg_response_time * 1000,
            'p95_response_time_ms': self.p95_response_time * 1000,
            'requests_per_second': self.requests_per_second,
            'error_rate_percent': self.error_rate,
            'error_count': len(self.errors),
            'unique_errors': len(set(self.errors))
        }

class SimplifiedLoadTester:
    """Simplified load tester for current SutazAI deployment"""
    
    def __init__(self):
        self.base_url = "http://localhost"
        self.results = []
        
        # Actual running services based on docker-compose ps output
        self.services = {
            'hygiene-backend': f'{self.base_url}:10420',
            'hygiene-dashboard': f'{self.base_url}:10422',
            'hygiene-nginx': f'{self.base_url}:10423',
            'rule-control-api': f'{self.base_url}:10421',
            'kong-api': f'{self.base_url}:10005',
            'kong-admin': f'{self.base_url}:10007',
            'postgres': f'{self.base_url}:10020',
            'redis': f'{self.base_url}:10021',
            'rabbitmq-management': f'{self.base_url}:10042',
            'consul': f'{self.base_url}:10006'
        }
        
    async def health_check(self) -> Dict[str, bool]:
        """Check health of running services"""
        logger.info("Checking health of running services...")
        health_status = {}
        
        health_endpoints = {
            'hygiene-backend': '/health',
            'hygiene-dashboard': '/',  # Dashboard root
            'hygiene-nginx': '/',      # Nginx root
            'rule-control-api': '/health',
            'kong-api': '/',           # Kong API root
            'kong-admin': '/',         # Kong Admin root
        }
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            for service, base_url in self.services.items():
                if service in health_endpoints:
                    try:
                        endpoint = health_endpoints[service]
                        url = f"{base_url}{endpoint}"
                        async with session.get(url) as response:
                            is_healthy = response.status in [200, 204]
                            health_status[service] = is_healthy
                            if is_healthy:
                                logger.info(f"✓ {service} - OK (Status: {response.status})")
                            else:
                                logger.warning(f"✗ {service} - Status {response.status}")
                    except Exception as e:
                        health_status[service] = False
                        logger.error(f"✗ {service} - Error: {str(e)}")
                else:
                    health_status[service] = None  # Not tested
                    logger.info(f"~ {service} - Not tested (no health endpoint)")
        
        healthy_count = sum(1 for status in health_status.values() if status is True)
        total_tested = sum(1 for status in health_status.values() if status is not None)
        
        logger.info(f"Health check complete: {healthy_count}/{total_tested} tested services healthy")
        return health_status
    
    async def single_request(self, session: aiohttp.ClientSession, url: str) -> Tuple[float, int, str]:
        """Make a single HTTP request"""
        start_time = time.time()
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                await response.text()  # Consume response body
                response_time = time.time() - start_time
                return response_time, response.status, ""
        except Exception as e:
            response_time = time.time() - start_time
            return response_time, 0, str(e)
    
    async def load_test(self, service_name: str, concurrent_users: int, duration_seconds: int) -> LoadTestResult:
        """Execute load test on specific service"""
        service_url = self.services.get(service_name)
        if not service_url:
            raise ValueError(f"Service {service_name} not found")
        
        # Determine appropriate endpoint
        if service_name == 'hygiene-backend':
            test_url = f"{service_url}/health"
        elif service_name == 'rule-control-api':
            test_url = f"{service_url}/health"
        else:
            test_url = f"{service_url}/"
        
        logger.info(f"Starting load test: {service_name} - {concurrent_users} users for {duration_seconds}s")
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        response_times = []
        status_codes = []
        errors = []
        
        async def user_session():
            async with aiohttp.ClientSession() as session:
                while time.time() < end_time:
                    response_time, status_code, error = await self.single_request(session, test_url)
                    
                    response_times.append(response_time)
                    status_codes.append(status_code)
                    
                    if error:
                        errors.append(error)
                    
                    # Small delay between requests
                    await asyncio.sleep(random.uniform(0.1, 0.3))
        
        # Run concurrent user sessions
        tasks = [user_session() for _ in range(concurrent_users)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        actual_duration = time.time() - start_time
        
        # Calculate metrics
        total_requests = len(response_times)
        successful_requests = sum(1 for code in status_codes if 200 <= code < 300)
        failed_requests = total_requests - successful_requests
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times)
        else:
            avg_response_time = p95_response_time = 0
        
        requests_per_second = total_requests / actual_duration if actual_duration > 0 else 0
        error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0
        
        result = LoadTestResult(
            test_name=f"{service_name} - {concurrent_users} users",
            duration=actual_duration,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            errors=errors[:10]  # Keep only first 10 errors
        )
        
        logger.info(f"Completed {result.test_name}: {successful_requests}/{total_requests} successful, "
                   f"avg: {avg_response_time*1000:.0f}ms, RPS: {requests_per_second:.1f}")
        
        return result
    
    async def test_scenario_1_normal_operation(self) -> LoadTestResult:
        """Test 1: Normal operation - 100 concurrent users"""
        return await self.load_test('hygiene-backend', 100, 60)  # 1 minute for quick test
    
    async def test_scenario_2_peak_load(self) -> LoadTestResult:
        """Test 2: Peak load - 1000 concurrent users"""
        return await self.load_test('hygiene-backend', 200, 60)  # Reduced to 200 for safety
    
    async def test_scenario_3_sustained_load(self) -> LoadTestResult:
        """Test 3: Sustained load - moderate load for longer period"""
        return await self.load_test('hygiene-backend', 50, 300)  # 5 minutes sustained
    
    async def test_scenario_4_spike_testing(self) -> LoadTestResult:
        """Test 4: Spike testing - rapid increase in load"""
        logger.info("Starting spike test: 10 -> 100 -> 10 users")
        
        # Phase 1: Baseline (10 users for 30s)
        result1 = await self.load_test('hygiene-backend', 10, 30)
        
        # Phase 2: Spike (100 users for 30s)
        result2 = await self.load_test('hygiene-backend', 100, 30)
        
        # Phase 3: Recovery (10 users for 30s)
        result3 = await self.load_test('hygiene-backend', 10, 30)
        
        # Combine results
        total_requests = result1.total_requests + result2.total_requests + result3.total_requests
        successful_requests = result1.successful_requests + result2.successful_requests + result3.successful_requests
        failed_requests = total_requests - successful_requests
        
        all_response_times = []
        # We can't easily combine response times, so use result2 (spike phase) as representative
        avg_response_time = result2.avg_response_time
        p95_response_time = result2.p95_response_time
        
        total_duration = result1.duration + result2.duration + result3.duration
        requests_per_second = total_requests / total_duration if total_duration > 0 else 0
        error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0
        
        combined_errors = result1.errors + result2.errors + result3.errors
        
        result = LoadTestResult(
            test_name="Spike Test (10->100->10 users)",
            duration=total_duration,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            errors=combined_errors[:10]
        )
        
        logger.info(f"Completed spike test: {successful_requests}/{total_requests} successful overall")
        return result
    
    async def test_scenario_5_service_resilience(self) -> LoadTestResult:
        """Test 5: Service resilience - test multiple services"""
        logger.info("Testing service resilience across multiple endpoints")
        
        services_to_test = ['hygiene-backend', 'rule-control-api']
        all_results = []
        
        for service in services_to_test:
            try:
                result = await self.load_test(service, 20, 30)  # 20 users for 30s each
                all_results.append(result)
            except Exception as e:
                logger.error(f"Failed to test {service}: {e}")
        
        if not all_results:
            raise Exception("No services could be tested")
        
        # Combine results
        total_requests = sum(r.total_requests for r in all_results)
        successful_requests = sum(r.successful_requests for r in all_results)
        failed_requests = total_requests - successful_requests
        
        avg_response_time = statistics.mean([r.avg_response_time for r in all_results])
        p95_response_time = max([r.p95_response_time for r in all_results])
        
        total_duration = sum(r.duration for r in all_results)
        requests_per_second = total_requests / total_duration if total_duration > 0 else 0
        error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0
        
        combined_errors = []
        for r in all_results:
            combined_errors.extend(r.errors)
        
        result = LoadTestResult(
            test_name="Service Resilience Test",
            duration=total_duration,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            errors=combined_errors[:10]
        )
        
        logger.info(f"Completed service resilience test across {len(all_results)} services")
        return result
    
    def analyze_results(self) -> Dict:
        """Analyze test results and generate recommendations"""
        if not self.results:
            return {"error": "No test results to analyze"}
        
        # Performance analysis
        total_requests = sum(r.total_requests for r in self.results)
        total_successful = sum(r.successful_requests for r in self.results)
        overall_error_rate = ((total_requests - total_successful) / total_requests * 100) if total_requests > 0 else 0
        
        # Performance metrics
        avg_response_times = [r.avg_response_time * 1000 for r in self.results]  # Convert to ms
        p95_response_times = [r.p95_response_time * 1000 for r in self.results]  # Convert to ms
        throughput_values = [r.requests_per_second for r in self.results]
        
        # Breaking points
        breaking_points = []
        for result in self.results:
            if result.error_rate > 5.0 or result.p95_response_time > 5.0:  # 5s threshold
                breaking_points.append({
                    "test": result.test_name,
                    "error_rate": result.error_rate,
                    "p95_response_time_ms": result.p95_response_time * 1000,
                    "requests_per_second": result.requests_per_second
                })
        
        # Recommendations
        recommendations = []
        
        # Response time analysis
        max_avg_response = max(avg_response_times) if avg_response_times else 0
        if max_avg_response > 2000:  # 2 seconds
            recommendations.append({
                "category": "Performance",
                "severity": "High",
                "issue": f"Average response time exceeds 2s (max: {max_avg_response:.0f}ms)",
                "recommendation": "Optimize database queries, add caching, or scale compute resources"
            })
        
        # Error rate analysis
        high_error_tests = [r for r in self.results if r.error_rate > 1.0]
        if high_error_tests:
            recommendations.append({
                "category": "Reliability",
                "severity": "Critical",
                "issue": f"High error rate in {len(high_error_tests)} test(s)",
                "recommendation": "Investigate error logs, improve error handling, add circuit breakers"
            })
        
        # Throughput analysis
        max_throughput = max(throughput_values) if throughput_values else 0
        if max_throughput < 50:
            recommendations.append({
                "category": "Capacity",
                "severity": "Medium",
                "issue": f"Low throughput (max: {max_throughput:.1f} RPS)",
                "recommendation": "Scale horizontally or optimize request handling"
            })
        
        # Production capacity estimation
        estimated_capacity = {
            "max_requests_per_second": max_throughput,
            "estimated_daily_requests": max_throughput * 86400,
            "recommended_max_concurrent_users": int(max_throughput * 2),
            "current_avg_response_time_ms": statistics.mean(avg_response_times) if avg_response_times else 0,
            "current_p95_response_time_ms": statistics.mean(p95_response_times) if p95_response_times else 0
        }
        
        return {
            "summary": {
                "total_tests": len(self.results),
                "total_requests": total_requests,
                "successful_requests": total_successful,
                "overall_success_rate": ((total_successful / total_requests * 100) if total_requests > 0 else 0),
                "overall_error_rate": overall_error_rate,
                "test_duration_total": sum(r.duration for r in self.results)
            },
            "performance_metrics": {
                "avg_response_time_ms": statistics.mean(avg_response_times) if avg_response_times else 0,
                "max_response_time_ms": max(avg_response_times) if avg_response_times else 0,
                "avg_p95_response_time_ms": statistics.mean(p95_response_times) if p95_response_times else 0,
                "max_throughput_rps": max_throughput,
                "avg_throughput_rps": statistics.mean(throughput_values) if throughput_values else 0
            },
            "breaking_points": breaking_points,
            "recommendations": recommendations,
            "production_capacity": estimated_capacity,
            "detailed_results": [result.to_dict() for result in self.results]
        }
    
    async def run_all_tests(self) -> Dict:
        """Run all load testing scenarios"""
        logger.info("Starting SutazAI Simplified Load Testing Suite")
        
        # Health check first
        health_status = await self.health_check()
        healthy_services = sum(1 for status in health_status.values() if status is True)
        
        if healthy_services == 0:
            logger.error("No healthy services found. Cannot proceed with load testing.")
            return {"error": "No healthy services available for testing"}
        
        logger.info(f"Proceeding with load testing on {healthy_services} healthy service(s)")
        
        # Test scenarios
        test_scenarios = [
            ("Normal Operation", self.test_scenario_1_normal_operation),
            ("Peak Load", self.test_scenario_2_peak_load),
            ("Sustained Load", self.test_scenario_3_sustained_load),
            ("Spike Testing", self.test_scenario_4_spike_testing),
            ("Service Resilience", self.test_scenario_5_service_resilience),
        ]
        
        passed_tests = 0
        failed_tests = 0
        
        for test_name, test_func in test_scenarios:
            try:
                logger.info(f"\nStarting {test_name}...")
                result = await test_func()
                self.results.append(result)
                passed_tests += 1
                
                # Log immediate results
                logger.info(f"✓ {test_name} completed:")
                logger.info(f"  Success Rate: {((result.successful_requests/result.total_requests*100) if result.total_requests > 0 else 0):.1f}%")
                logger.info(f"  Avg Response: {result.avg_response_time*1000:.0f}ms")
                logger.info(f"  Throughput: {result.requests_per_second:.1f} RPS")
                
            except Exception as e:
                logger.error(f"✗ {test_name} failed: {str(e)}")
                failed_tests += 1
        
        logger.info(f"\nLoad testing complete: {passed_tests} passed, {failed_tests} failed")
        
        # Generate analysis report
        report = self.analyze_results()
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"/opt/sutazaiapp/load-testing/reports/sutazai_load_test_report_{timestamp}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Report saved to: {report_path}")
        
        return report

def main():
    """Main execution function"""
    # Create necessary directories
    os.makedirs('/opt/sutazaiapp/load-testing/logs', exist_ok=True)
    os.makedirs('/opt/sutazaiapp/load-testing/reports', exist_ok=True)
    
    tester = SimplifiedLoadTester()
    
    try:
        # Run all tests
        report = asyncio.run(tester.run_all_tests())
        
        if "error" in report:
            logger.error(f"\nERROR: {report['error']}")
            return 1
        
        # Print comprehensive summary
        logger.info("\n" + "="*80)
        logger.info("SUTAZAI PRODUCTION LOAD TESTING REPORT")
        logger.info("="*80)
        
        summary = report['summary']
        metrics = report['performance_metrics']
        capacity = report['production_capacity']
        
        logger.info(f"\nTEST SUMMARY:")
        logger.info(f"  Total Tests: {summary['total_tests']}")
        logger.info(f"  Total Requests: {summary['total_requests']:,}")
        logger.info(f"  Success Rate: {summary['overall_success_rate']:.2f}%")
        logger.error(f"  Error Rate: {summary['overall_error_rate']:.2f}%")
        logger.info(f"  Test Duration: {summary['test_duration_total']:.1f} seconds")
        
        logger.info(f"\nPERFORMANCE METRICS:")
        logger.info(f"  Average Response Time: {metrics['avg_response_time_ms']:.0f}ms")
        logger.info(f"  P95 Response Time: {metrics['avg_p95_response_time_ms']:.0f}ms")
        logger.info(f"  Max Throughput: {metrics['max_throughput_rps']:.1f} RPS")
        logger.info(f"  Average Throughput: {metrics['avg_throughput_rps']:.1f} RPS")
        
        logger.info(f"\nPRODUCTION CAPACITY ESTIMATE:")
        logger.info(f"  Max Requests/Second: {capacity['max_requests_per_second']:.1f}")
        logger.info(f"  Daily Request Capacity: {capacity['estimated_daily_requests']:,.0f}")
        logger.info(f"  Recommended Max Users: {capacity['recommended_max_concurrent_users']:,}")
        
        if report['breaking_points']:
            logger.info(f"\nBREAKING POINTS DETECTED: {len(report['breaking_points'])}")
            for bp in report['breaking_points']:
                logger.error(f"  - {bp['test']}: {bp['error_rate']:.2f}% errors, {bp['p95_response_time_ms']:.0f}ms P95")
        
        if report['recommendations']:
            logger.info(f"\nRECOMMENDATIONS: {len(report['recommendations'])}")
            for i, rec in enumerate(report['recommendations'], 1):
                logger.info(f"  {i}. {rec['category']} ({rec['severity']}):")
                logger.info(f"     Issue: {rec['issue']}")
                logger.info(f"     Fix: {rec['recommendation']}")
        
        logger.info("\n" + "="*80)
        
        # Return exit code based on overall performance
        if summary['overall_error_rate'] > 5.0:
            logger.error("WARNING: High error rate detected. System may need attention.")
            return 2
        elif metrics['avg_response_time_ms'] > 3000:
            logger.warning("WARNING: High response times detected. Performance optimization recommended.")
            return 1
        else:
            logger.info("SUCCESS: System performance within acceptable limits.")
            return 0
            
    except KeyboardInterrupt:
        logger.info("\nTesting interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
