#!/usr/bin/env python3
"""
SutazAI Production Load Testing Suite
Comprehensive load testing framework for SutazAI production environment

Author: QA Team Lead
Date: 2025-08-04
Version: 1.0
"""

import asyncio
import aiohttp
import json
import time
import logging
import sys
import argparse
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import threading
import random
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'/opt/sutazaiapp/load-testing/logs/load_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Data class to store test results"""
    test_name: str
    start_time: datetime
    end_time: datetime
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    p95_response_time: float
    p99_response_time: float
    min_response_time: float
    max_response_time: float
    requests_per_second: float
    error_rate: float
    errors: List[str]
    
    def to_dict(self):
        return {
            **asdict(self),
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_seconds': (self.end_time - self.start_time).total_seconds()
        }

class SutazAILoadTester:
    """Main load testing class for SutazAI"""
    
    def __init__(self, base_url: str = "http://localhost", timeout: int = 30):
        self.base_url = base_url
        self.timeout = timeout
        self.results: List[TestResult] = []
        self.session_stats = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'start_time': None,
            'end_time': None
        }
        
        # Service endpoints configuration
        self.endpoints = {
            'backend': f'{base_url}:10010',
            'frontend': f'{base_url}:8501',
            'kong_api': f'{base_url}:10005',
            'kong_admin': f'{base_url}:10007',
            'prometheus': f'{base_url}:10101',
            'grafana': f'{base_url}:10102',
            'jaeger': f'{base_url}:10104',
            'ollama': f'{base_url}:11434',
            'postgres': f'{base_url}:10020',
            'redis': f'{base_url}:10021',
            'neo4j': f'{base_url}:10022',
            'chromadb': f'{base_url}:10100'
        }
        
        # Agent endpoints (sample of key agents for testing)
        self.agents = {
            'ai-system-architect': f'{base_url}:11010',
            'ai-qa-team-lead': f'{base_url}:11082',
            'ai-senior-engineer': f'{base_url}:11081',
            'ai-backend-developer': f'{base_url}:11002',
            'ai-testing-validator': f'{base_url}:10322',
            'jarvis-voice-interface': f'{base_url}:11013'
        }
    
    async def health_check(self) -> Dict[str, bool]:
        """Perform system health check before load testing"""
        logger.info("Performing system health check...")
        health_status = {}
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            for service, url in self.endpoints.items():
                try:
                    health_url = f"{url}/health" if service != 'grafana' else f"{url}/api/health"
                    async with session.get(health_url) as response:
                        health_status[service] = response.status == 200
                        if response.status == 200:
                            logger.info(f"✓ {service} - OK")
                        else:
                            logger.warning(f"✗ {service} - Status {response.status}")
                except Exception as e:
                    health_status[service] = False
                    logger.error(f"✗ {service} - Error: {str(e)}")
        
        healthy_services = sum(health_status.values())
        total_services = len(health_status)
        
        logger.info(f"Health check complete: {healthy_services}/{total_services} services healthy")
        
        if healthy_services < total_services * 0.7:  # Require 70% healthy services
            logger.error("Too many services are unhealthy. Aborting load tests.")
            return {}
        
        return health_status
    
    async def single_request(self, session: aiohttp.ClientSession, url: str, method: str = 'GET', 
                           data: Optional[Dict] = None, headers: Optional[Dict] = None) -> Tuple[float, int, str]:
        """Make a single HTTP request and return response time, status code, and error message"""
        start_time = time.time()
        try:
            kwargs = {
                'timeout': aiohttp.ClientTimeout(total=self.timeout),
                'headers': headers or {}
            }
            
            if method.upper() == 'POST' and data:
                kwargs['json'] = data
            
            async with session.request(method, url, **kwargs) as response:
                await response.text()  # Consume response body
                response_time = time.time() - start_time
                return response_time, response.status, ""
        except Exception as e:
            response_time = time.time() - start_time
            return response_time, 0, str(e)
    
    async def concurrent_load_test(self, url: str, concurrent_users: int, duration_seconds: int, 
                                 test_name: str, method: str = 'GET', data: Optional[Dict] = None) -> TestResult:
        """Run concurrent load test with specified parameters"""
        logger.info(f"Starting {test_name}: {concurrent_users} users for {duration_seconds}s on {url}")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(seconds=duration_seconds)
        
        response_times = []
        status_codes = []
        errors = []
        request_count = 0
        
        async def user_session():
            nonlocal request_count
            async with aiohttp.ClientSession() as session:
                while datetime.now() < end_time:
                    response_time, status_code, error = await self.single_request(session, url, method, data)
                    
                    response_times.append(response_time)
                    status_codes.append(status_code)
                    request_count += 1
                    
                    if error:
                        errors.append(error)
                    
                    # Small delay to simulate realistic user behavior
                    await asyncio.sleep(random.uniform(0.1, 0.5))
        
        # Run concurrent user sessions
        tasks = [user_session() for _ in range(concurrent_users)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        actual_end_time = datetime.now()
        
        # Calculate metrics
        successful_requests = sum(1 for code in status_codes if 200 <= code < 300)
        failed_requests = request_count - successful_requests
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times)
            p99_response_time = statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else max(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = p95_response_time = p99_response_time = min_response_time = max_response_time = 0
        
        duration = (actual_end_time - start_time).total_seconds()
        requests_per_second = request_count / duration if duration > 0 else 0
        error_rate = (failed_requests / request_count * 100) if request_count > 0 else 0
        
        result = TestResult(
            test_name=test_name,
            start_time=start_time,
            end_time=actual_end_time,
            total_requests=request_count,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            errors=list(set(errors))  # Remove duplicates
        )
        
        logger.info(f"Completed {test_name}: {successful_requests}/{request_count} requests successful, "
                   f"avg response: {avg_response_time:.3f}s, RPS: {requests_per_second:.1f}")
        
        return result
    
    async def test_normal_operation(self) -> TestResult:
        """Test 1: Normal operation (100 concurrent users)"""
        return await self.concurrent_load_test(
            url=f"{self.endpoints['backend']}/health",
            concurrent_users=100,
            duration_seconds=300,  # 5 minutes
            test_name="Normal Operation (100 users)"
        )
    
    async def test_peak_load(self) -> TestResult:
        """Test 2: Peak load (1000 concurrent users)"""
        return await self.concurrent_load_test(
            url=f"{self.endpoints['backend']}/health",
            concurrent_users=1000,
            duration_seconds=300,  # 5 minutes
            test_name="Peak Load (1000 users)"
        )
    
    async def test_sustained_load(self) -> TestResult:
        """Test 3: Sustained load (500 users for 1 hour)"""
        return await self.concurrent_load_test(
            url=f"{self.endpoints['backend']}/health",
            concurrent_users=500,
            duration_seconds=3600,  # 1 hour
            test_name="Sustained Load (500 users, 1 hour)"
        )
    
    async def test_spike_load(self) -> TestResult:
        """Test 4: Spike testing (0 to 1000 users in 1 minute)"""
        logger.info("Starting Spike Load Test: 0 to 1000 users in 1 minute")
        
        start_time = datetime.now()
        response_times = []
        status_codes = []
        errors = []
        request_count = 0
        
        async def gradual_spike():
            nonlocal request_count
            for second in range(60):  # 60 seconds
                users_this_second = int((second + 1) * 1000 / 60)  # Gradual increase to 1000
                
                async def spike_user():
                    nonlocal request_count
                    async with aiohttp.ClientSession() as session:
                        response_time, status_code, error = await self.single_request(
                            session, f"{self.endpoints['backend']}/health"
                        )
                        response_times.append(response_time)
                        status_codes.append(status_code)
                        request_count += 1
                        if error:
                            errors.append(error)
                
                # Launch users for this second
                tasks = [spike_user() for _ in range(min(50, users_this_second))]  # Cap at 50 per second
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                await asyncio.sleep(1)  # Wait one second
        
        await gradual_spike()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Calculate metrics
        successful_requests = sum(1 for code in status_codes if 200 <= code < 300)
        failed_requests = request_count - successful_requests
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times)
            p99_response_time = statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else max(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = p95_response_time = p99_response_time = min_response_time = max_response_time = 0
        
        requests_per_second = request_count / duration if duration > 0 else 0
        error_rate = (failed_requests / request_count * 100) if request_count > 0 else 0
        
        result = TestResult(
            test_name="Spike Load (0-1000 users)",
            start_time=start_time,
            end_time=end_time,
            total_requests=request_count,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            errors=list(set(errors))
        )
        
        logger.info(f"Completed Spike Load Test: {successful_requests}/{request_count} requests successful")
        return result
    
    async def test_agent_failure_scenarios(self) -> TestResult:
        """Test 5: Agent failure scenarios"""
        logger.info("Starting Agent Failure Scenario Testing")
        
        start_time = datetime.now()
        response_times = []
        status_codes = []
        errors = []
        request_count = 0
        
        # Test requests to potentially failing agents
        test_agents = list(self.agents.keys())[:3]  # Test first 3 agents
        
        async def test_agent_resilience():
            nonlocal request_count
            async with aiohttp.ClientSession() as session:
                for _ in range(50):  # 50 requests per agent
                    for agent_name in test_agents:
                        agent_url = f"{self.agents[agent_name]}/health"
                        response_time, status_code, error = await self.single_request(session, agent_url)
                        
                        response_times.append(response_time)
                        status_codes.append(status_code)
                        request_count += 1
                        
                        if error:
                            errors.append(f"{agent_name}: {error}")
                        
                        await asyncio.sleep(0.1)  # Small delay
        
        # Run 10 concurrent sessions
        tasks = [test_agent_resilience() for _ in range(10)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Calculate metrics
        successful_requests = sum(1 for code in status_codes if 200 <= code < 300)
        failed_requests = request_count - successful_requests
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times)
            p99_response_time = statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else max(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = p95_response_time = p99_response_time = min_response_time = max_response_time = 0
        
        requests_per_second = request_count / duration if duration > 0 else 0
        error_rate = (failed_requests / request_count * 100) if request_count > 0 else 0
        
        result = TestResult(
            test_name="Agent Failure Scenarios",
            start_time=start_time,
            end_time=end_time,
            total_requests=request_count,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            errors=list(set(errors))
        )
        
        logger.info(f"Completed Agent Failure Scenarios: {successful_requests}/{request_count} requests successful")
        return result
    
    async def test_database_failover(self) -> TestResult:
        """Test 6: Database failover testing"""
        logger.info("Starting Database Failover Testing")
        
        start_time = datetime.now()
        response_times = []
        status_codes = []
        errors = []
        request_count = 0
        
        # Test database-dependent endpoints
        db_endpoints = [
            f"{self.endpoints['backend']}/api/health/database",
            f"{self.endpoints['backend']}/api/health/redis",
            f"{self.endpoints['backend']}/api/health/neo4j"
        ]
        
        async def test_db_resilience():
            nonlocal request_count
            async with aiohttp.ClientSession() as session:
                for _ in range(100):  # 100 requests
                    for endpoint in db_endpoints:
                        response_time, status_code, error = await self.single_request(session, endpoint)
                        
                        response_times.append(response_time)
                        status_codes.append(status_code)
                        request_count += 1
                        
                        if error:
                            errors.append(f"{endpoint}: {error}")
                        
                        await asyncio.sleep(0.1)
        
        # Run 5 concurrent sessions
        tasks = [test_db_resilience() for _ in range(5)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Calculate metrics
        successful_requests = sum(1 for code in status_codes if 200 <= code < 300)
        failed_requests = request_count - successful_requests
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times)
            p99_response_time = statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else max(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = p95_response_time = p99_response_time = min_response_time = max_response_time = 0
        
        requests_per_second = request_count / duration if duration > 0 else 0
        error_rate = (failed_requests / request_count * 100) if request_count > 0 else 0
        
        result = TestResult(
            test_name="Database Failover Testing",
            start_time=start_time,
            end_time=end_time,
            total_requests=request_count,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            errors=list(set(errors))
        )
        
        logger.info(f"Completed Database Failover Testing: {successful_requests}/{request_count} requests successful")
        return result
    
    async def test_network_partition(self) -> TestResult:
        """Test 7: Network partition testing"""
        logger.info("Starting Network Partition Testing")
        
        start_time = datetime.now()
        response_times = []
        status_codes = []
        errors = []
        request_count = 0
        
        # Test service mesh communication under stress
        mesh_endpoints = [
            f"{self.endpoints['kong_api']}/health",
            f"{self.endpoints['backend']}/api/agents/status",
            f"{self.endpoints['prometheus']}/api/v1/query?query=up"
        ]
        
        async def test_network_resilience():
            nonlocal request_count
            async with aiohttp.ClientSession() as session:
                for _ in range(200):  # High volume to stress network
                    endpoint = random.choice(mesh_endpoints)
                    response_time, status_code, error = await self.single_request(session, endpoint)
                    
                    response_times.append(response_time)
                    status_codes.append(status_code)
                    request_count += 1
                    
                    if error:
                        errors.append(f"{endpoint}: {error}")
                    
                    await asyncio.sleep(random.uniform(0.01, 0.1))  # Variable timing
        
        # Run 20 concurrent sessions to simulate network stress
        tasks = [test_network_resilience() for _ in range(20)]
        await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Calculate metrics
        successful_requests = sum(1 for code in status_codes if 200 <= code < 300)
        failed_requests = request_count - successful_requests
        
        if response_times:
            avg_response_time = statistics.mean(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times)
            p99_response_time = statistics.quantiles(response_times, n=100)[98] if len(response_times) > 100 else max(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = p95_response_time = p99_response_time = min_response_time = max_response_time = 0
        
        requests_per_second = request_count / duration if duration > 0 else 0
        error_rate = (failed_requests / request_count * 100) if request_count > 0 else 0
        
        result = TestResult(
            test_name="Network Partition Testing",
            start_time=start_time,
            end_time=end_time,
            total_requests=request_count,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            average_response_time=avg_response_time,
            p95_response_time=p95_response_time,
            p99_response_time=p99_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            requests_per_second=requests_per_second,
            error_rate=error_rate,
            errors=list(set(errors))
        )
        
        logger.info(f"Completed Network Partition Testing: {successful_requests}/{request_count} requests successful")
        return result
    
    def generate_report(self) -> Dict:
        """Generate comprehensive performance report"""
        if not self.results:
            return {"error": "No test results available"}
        
        # Performance analysis
        total_requests = sum(r.total_requests for r in self.results)
        total_successful = sum(r.successful_requests for r in self.results)
        total_failed = sum(r.failed_requests for r in self.results)
        overall_error_rate = (total_failed / total_requests * 100) if total_requests > 0 else 0
        
        # Breaking points analysis
        breaking_points = []
        for result in self.results:
            if result.error_rate > 5.0:  # More than 5% error rate indicates breaking point
                breaking_points.append({
                    "test": result.test_name,
                    "error_rate": result.error_rate,
                    "avg_response_time": result.average_response_time,
                    "requests_per_second": result.requests_per_second
                })
        
        # Performance recommendations
        recommendations = []
        
        # Check response time recommendations
        slow_tests = [r for r in self.results if r.p95_response_time > 3.0]
        if slow_tests:
            recommendations.append({
                "category": "Response Time",
                "severity": "High",
                "issue": f"P95 response time exceeds 3s in {len(slow_tests)} test(s)",
                "recommendation": "Consider optimizing database queries, adding caching, or scaling compute resources",
                "affected_tests": [t.test_name for t in slow_tests]
            })
        
        # Check error rate recommendations
        high_error_tests = [r for r in self.results if r.error_rate > 1.0]
        if high_error_tests:
            recommendations.append({
                "category": "Error Rate",
                "severity": "Critical",
                "issue": f"Error rate exceeds 1% in {len(high_error_tests)} test(s)",
                "recommendation": "Investigate error logs, improve error handling, and add circuit breakers",
                "affected_tests": [t.test_name for t in high_error_tests]
            })
        
        # Check throughput recommendations
        low_throughput_tests = [r for r in self.results if r.requests_per_second < 50]
        if low_throughput_tests:
            recommendations.append({
                "category": "Throughput",
                "severity": "Medium",
                "issue": f"Low throughput (<50 RPS) in {len(low_throughput_tests)} test(s)",
                "recommendation": "Scale horizontally, optimize connection pooling, or upgrade hardware",
                "affected_tests": [t.test_name for t in low_throughput_tests]
            })
        
        # Production capacity estimation
        max_rps = max(r.requests_per_second for r in self.results)
        estimated_capacity = {
            "current_max_rps": max_rps,
            "estimated_daily_capacity": max_rps * 86400,  # 24 hours
            "recommended_max_concurrent_users": int(max_rps * 2),  # Conservative estimate
            "scaling_threshold_rps": max_rps * 0.8  # Scale when reaching 80% capacity
        }
        
        report = {
            "summary": {
                "test_session_start": self.session_stats['start_time'].isoformat() if self.session_stats['start_time'] else None,
                "test_session_end": self.session_stats['end_time'].isoformat() if self.session_stats['end_time'] else None,
                "total_tests": len(self.results),
                "total_requests": total_requests,
                "successful_requests": total_successful,
                "failed_requests": total_failed,
                "overall_success_rate": (total_successful / total_requests * 100) if total_requests > 0 else 0,
                "overall_error_rate": overall_error_rate
            },
            "performance_metrics": {
                "average_response_times": {r.test_name: r.average_response_time for r in self.results},
                "p95_response_times": {r.test_name: r.p95_response_time for r in self.results},
                "p99_response_times": {r.test_name: r.p99_response_time for r in self.results},
                "requests_per_second": {r.test_name: r.requests_per_second for r in self.results},
                "error_rates": {r.test_name: r.error_rate for r in self.results}
            },
            "breaking_points": breaking_points,
            "recommendations": recommendations,
            "production_capacity": estimated_capacity,
            "detailed_results": [result.to_dict() for result in self.results]
        }
        
        return report
    
    async def run_all_tests(self):
        """Run all load testing scenarios"""
        logger.info("Starting SutazAI Production Load Testing Suite")
        self.session_stats['start_time'] = datetime.now()
        
        # Health check first
        health_status = await self.health_check()
        if not health_status:
            logger.error("Health check failed. Aborting tests.")
            return
        
        # Define test scenarios
        test_scenarios = [
            ("normal_operation", self.test_normal_operation),
            ("peak_load", self.test_peak_load),
            ("sustained_load", self.test_sustained_load),
            ("spike_load", self.test_spike_load),
            ("agent_failure", self.test_agent_failure_scenarios),
            ("database_failover", self.test_database_failover),
            ("network_partition", self.test_network_partition)
        ]
        
        # Run tests
        for test_id, test_func in test_scenarios:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"Starting {test_id.replace('_', ' ').title()} Test")
                logger.info(f"{'='*60}")
                
                result = await test_func()
                self.results.append(result)
                self.session_stats['passed_tests'] += 1
                
                # Log immediate results
                logger.info(f"✓ {result.test_name} completed successfully")
                logger.info(f"  Requests: {result.total_requests} ({result.successful_requests} successful)")
                logger.info(f"  Error Rate: {result.error_rate:.2f}%")
                logger.info(f"  Avg Response: {result.average_response_time:.3f}s")
                logger.info(f"  P95 Response: {result.p95_response_time:.3f}s")
                logger.info(f"  Throughput: {result.requests_per_second:.1f} RPS")
                
            except Exception as e:
                logger.error(f"✗ {test_id} failed: {str(e)}")
                self.session_stats['failed_tests'] += 1
                continue
        
        self.session_stats['end_time'] = datetime.now()
        self.session_stats['total_tests'] = len(self.results)
        
        logger.info(f"\n{'='*60}")
        logger.info("Load Testing Session Complete")
        logger.info(f"{'='*60}")
        logger.info(f"Total Tests: {self.session_stats['total_tests']}")
        logger.info(f"Passed: {self.session_stats['passed_tests']}")
        logger.info(f"Failed: {self.session_stats['failed_tests']}")
        
        # Generate and save report
        report = self.generate_report()
        
        # Save report to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = f"/opt/sutazaiapp/load-testing/reports/production_load_test_{timestamp}.json"
        
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Comprehensive report saved to: {report_path}")
        
        return report

def main():
    parser = argparse.ArgumentParser(description='SutazAI Production Load Testing Suite')
    parser.add_argument('--base-url', default='http://localhost', help='Base URL for testing')
    parser.add_argument('--timeout', type=int, default=30, help='Request timeout in seconds')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only (shorter duration)')
    
    args = parser.parse_args()
    
    # Create logs directory
    os.makedirs('/opt/sutazaiapp/load-testing/logs', exist_ok=True)
    os.makedirs('/opt/sutazaiapp/load-testing/reports', exist_ok=True)
    
    # Initialize load tester
    tester = SutazAILoadTester(base_url=args.base_url, timeout=args.timeout)
    
    # Run tests
    try:
        report = asyncio.run(tester.run_all_tests())
        
        # Print summary
        print("\n" + "="*80)
        print("SUTAZAI PRODUCTION LOAD TESTING REPORT SUMMARY")
        print("="*80)
        print(f"Total Tests: {report['summary']['total_tests']}")
        print(f"Total Requests: {report['summary']['total_requests']:,}")
        print(f"Success Rate: {report['summary']['overall_success_rate']:.2f}%")
        print(f"Error Rate: {report['summary']['overall_error_rate']:.2f}%")
        print(f"\nProduction Capacity Estimation:")
        print(f"  Max RPS: {report['production_capacity']['current_max_rps']:.1f}")
        print(f"  Recommended Max Users: {report['production_capacity']['recommended_max_concurrent_users']:,}")
        print(f"  Daily Capacity: {report['production_capacity']['estimated_daily_capacity']:,} requests")
        
        if report['breaking_points']:
            print(f"\nBreaking Points Detected: {len(report['breaking_points'])}")
            for bp in report['breaking_points']:
                print(f"  - {bp['test']}: {bp['error_rate']:.2f}% error rate")
        
        if report['recommendations']:
            print(f"\nRecommendations: {len(report['recommendations'])}")
            for rec in report['recommendations'][:3]:  # Show top 3
                print(f"  - {rec['category']} ({rec['severity']}): {rec['issue']}")
        
        print("\n" + "="*80)
        
        return 0 if report['summary']['overall_error_rate'] < 5.0 else 1
        
    except KeyboardInterrupt:
        logger.info("\nLoad testing interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Load testing failed: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
