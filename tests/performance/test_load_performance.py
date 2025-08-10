#!/usr/bin/env python3
"""
Comprehensive Performance and Load Tests for SutazAI System
Validates performance requirements per Rules 1-19
"""

import pytest
import asyncio
import httpx
import time
import statistics
import json
import psutil
import threading
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import resource
import gc

# Test configuration
BASE_URL = os.getenv('TEST_BASE_URL', 'http://localhost:10010')
TEST_TIMEOUT = 120.0
MAX_CONCURRENT_USERS = 50
TEST_DURATION = 60  # seconds
PERFORMANCE_THRESHOLD_MS = 200  # Health endpoint should respond in <200ms
CHAT_THRESHOLD_MS = 5000  # Chat should respond in <5s
MEMORY_THRESHOLD_MB = 1000  # Process should not exceed 1GB


class PerformanceMetrics:
    """Collect and analyze performance metrics"""
    
    def __init__(self):
        self.response_times = []
        self.success_count = 0
        self.error_count = 0
        self.memory_samples = []
        self.cpu_samples = []
        self.start_time = None
        self.end_time = None
    
    def add_response_time(self, response_time_ms: float, success: bool = True):
        """Add a response time measurement"""
        self.response_times.append(response_time_ms)
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    def add_system_metrics(self, memory_mb: float, cpu_percent: float):
        """Add system resource measurements"""
        self.memory_samples.append(memory_mb)
        self.cpu_samples.append(cpu_percent)
    
    def start_test(self):
        """Mark test start time"""
        self.start_time = time.time()
    
    def end_test(self):
        """Mark test end time"""
        self.end_time = time.time()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics"""
        if not self.response_times:
            return {"error": "No response times recorded"}
        
        total_requests = self.success_count + self.error_count
        duration = (self.end_time - self.start_time) if self.end_time and self.start_time else 0
        
        return {
            "total_requests": total_requests,
            "successful_requests": self.success_count,
            "failed_requests": self.error_count,
            "success_rate": (self.success_count / total_requests) * 100 if total_requests > 0 else 0,
            "test_duration_seconds": duration,
            "requests_per_second": total_requests / duration if duration > 0 else 0,
            "response_times": {
                "min_ms": min(self.response_times),
                "max_ms": max(self.response_times),
                "avg_ms": statistics.mean(self.response_times),
                "median_ms": statistics.median(self.response_times),
                "p95_ms": self._percentile(self.response_times, 95),
                "p99_ms": self._percentile(self.response_times, 99)
            },
            "memory_usage": {
                "max_mb": max(self.memory_samples) if self.memory_samples else 0,
                "avg_mb": statistics.mean(self.memory_samples) if self.memory_samples else 0
            },
            "cpu_usage": {
                "max_percent": max(self.cpu_samples) if self.cpu_samples else 0,
                "avg_percent": statistics.mean(self.cpu_samples) if self.cpu_samples else 0
            }
        }
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of data"""
        sorted_data = sorted(data)
        index = (percentile / 100.0) * (len(sorted_data) - 1)
        lower = int(index)
        upper = lower + 1
        weight = index - lower
        
        if upper >= len(sorted_data):
            return sorted_data[-1]
        
        return sorted_data[lower] * (1 - weight) + sorted_data[upper] * weight


class LoadGenerator:
    """Generate load for performance testing"""
    
    def __init__(self, base_url: str, metrics: PerformanceMetrics):
        self.base_url = base_url
        self.metrics = metrics
        self.running = False
    
    async def single_request(self, endpoint: str, method: str = 'GET', data: Dict = None) -> Tuple[bool, float]:
        """Make a single request and measure response time"""
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
                if method.upper() == 'GET':
                    response = await client.get(f"{self.base_url}{endpoint}")
                elif method.upper() == 'POST':
                    response = await client.post(f"{self.base_url}{endpoint}", json=data)
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                success = response.status_code in [200, 201, 202]
                response_time = (time.time() - start_time) * 1000
                
                return success, response_time
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return False, response_time
    
    async def user_simulation(self, user_id: int, duration: int):
        """Simulate a user session for specified duration"""
        end_time = time.time() + duration
        request_count = 0
        
        while time.time() < end_time and self.running:
            # Simulate user behavior with different endpoints
            endpoints = [
                ('/health', 'GET', None),
                ('/api/v1/models/', 'GET', None),
                ('/api/v1/chat/', 'POST', {'message': f'Hello from user {user_id}', 'model': 'tinyllama'}),
                ('/metrics', 'GET', None)
            ]
            
            for endpoint, method, data in endpoints:
                if not self.running or time.time() >= end_time:
                    break
                
                success, response_time = await self.single_request(endpoint, method, data)
                self.metrics.add_response_time(response_time, success)
                
                request_count += 1
                
                # Small delay between requests (simulate user think time)
                await asyncio.sleep(0.1)
        
        return request_count
    
    async def concurrent_load_test(self, concurrent_users: int, duration: int):
        """Run concurrent load test with specified users and duration"""
        self.running = True
        self.metrics.start_test()
        
        # Start system monitoring
        monitor_task = asyncio.create_task(self._monitor_system(duration))
        
        # Create user simulation tasks
        user_tasks = [
            asyncio.create_task(self.user_simulation(i, duration))
            for i in range(concurrent_users)
        ]
        
        try:
            # Wait for all users to complete
            await asyncio.gather(*user_tasks)
        finally:
            self.running = False
            await monitor_task
            self.metrics.end_test()
    
    async def _monitor_system(self, duration: int):
        """Monitor system resources during test"""
        end_time = time.time() + duration
        
        while time.time() < end_time and self.running:
            try:
                # Get memory usage (in MB)
                memory_info = psutil.virtual_memory()
                memory_mb = memory_info.used / (1024 * 1024)
                
                # Get CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                
                self.metrics.add_system_metrics(memory_mb, cpu_percent)
                
                await asyncio.sleep(2)  # Sample every 2 seconds
            except Exception:
                await asyncio.sleep(2)  # Continue monitoring even if there's an error


@pytest.mark.performance
class TestBasicPerformance:
    """Test basic performance requirements"""
    
    @pytest.mark.asyncio
    async def test_health_endpoint_response_time(self):
        """Test health endpoint meets response time requirements"""
        metrics = PerformanceMetrics()
        load_gen = LoadGenerator(BASE_URL, metrics)
        
        # Make 50 requests to health endpoint
        for i in range(50):
            success, response_time = await load_gen.single_request('/health')
            metrics.add_response_time(response_time, success)
        
        summary = metrics.get_summary()
        
        # Assertions
        assert summary['success_rate'] >= 95.0, f"Success rate {summary['success_rate']:.1f}% below 95%"
        assert summary['response_times']['avg_ms'] < PERFORMANCE_THRESHOLD_MS, \
            f"Average response time {summary['response_times']['avg_ms']:.1f}ms exceeds {PERFORMANCE_THRESHOLD_MS}ms"
        assert summary['response_times']['p95_ms'] < PERFORMANCE_THRESHOLD_MS * 2, \
            f"95th percentile {summary['response_times']['p95_ms']:.1f}ms exceeds {PERFORMANCE_THRESHOLD_MS * 2}ms"
    
    @pytest.mark.asyncio
    async def test_chat_endpoint_response_time(self):
        """Test chat endpoint meets response time requirements"""
        metrics = PerformanceMetrics()
        load_gen = LoadGenerator(BASE_URL, metrics)
        
        # Make 10 chat requests (fewer due to longer processing time)
        for i in range(10):
            chat_data = {
                'message': f'Performance test message {i}',
                'model': 'tinyllama'
            }
            success, response_time = await load_gen.single_request('/api/v1/chat/', 'POST', chat_data)
            metrics.add_response_time(response_time, success)
        
        summary = metrics.get_summary()
        
        # Assertions (more lenient for chat endpoint)
        assert summary['success_rate'] >= 80.0, f"Success rate {summary['success_rate']:.1f}% below 80%"
        assert summary['response_times']['avg_ms'] < CHAT_THRESHOLD_MS, \
            f"Average response time {summary['response_times']['avg_ms']:.1f}ms exceeds {CHAT_THRESHOLD_MS}ms"
    
    @pytest.mark.asyncio
    async def test_concurrent_users_performance(self):
        """Test performance with concurrent users"""
        metrics = PerformanceMetrics()
        load_gen = LoadGenerator(BASE_URL, metrics)
        
        # Test with 10 concurrent users for 30 seconds
        await load_gen.concurrent_load_test(concurrent_users=10, duration=30)
        
        summary = metrics.get_summary()
        
        # Assertions
        assert summary['success_rate'] >= 85.0, f"Success rate {summary['success_rate']:.1f}% below 85%"
        assert summary['requests_per_second'] >= 5.0, \
            f"RPS {summary['requests_per_second']:.1f} below minimum 5.0"
        assert summary['response_times']['p95_ms'] < 2000, \
            f"95th percentile {summary['response_times']['p95_ms']:.1f}ms exceeds 2000ms"


@pytest.mark.performance
@pytest.mark.slow
class TestLoadPerformance:
    """Test system performance under various load conditions"""
    
    @pytest.mark.asyncio
    async def test_moderate_load(self):
        """Test performance under moderate load (20 concurrent users)"""
        metrics = PerformanceMetrics()
        load_gen = LoadGenerator(BASE_URL, metrics)
        
        await load_gen.concurrent_load_test(concurrent_users=20, duration=60)
        
        summary = metrics.get_summary()
        
        # Save results for analysis
        self._save_test_results('moderate_load', summary)
        
        # Assertions
        assert summary['success_rate'] >= 80.0
        assert summary['requests_per_second'] >= 8.0
        assert summary['response_times']['avg_ms'] < 1000
        assert summary['memory_usage']['max_mb'] < MEMORY_THRESHOLD_MB
    
    @pytest.mark.asyncio
    async def test_high_load(self):
        """Test performance under high load (50 concurrent users)"""
        metrics = PerformanceMetrics()
        load_gen = LoadGenerator(BASE_URL, metrics)
        
        await load_gen.concurrent_load_test(concurrent_users=50, duration=90)
        
        summary = metrics.get_summary()
        
        # Save results for analysis
        self._save_test_results('high_load', summary)
        
        # Assertions (more lenient under high load)
        assert summary['success_rate'] >= 70.0
        assert summary['requests_per_second'] >= 10.0
        assert summary['response_times']['p99_ms'] < 5000
    
    @pytest.mark.asyncio
    async def test_stress_test(self):
        """Stress test to find system breaking point"""
        metrics = PerformanceMetrics()
        load_gen = LoadGenerator(BASE_URL, metrics)
        
        # Gradually increase load
        user_counts = [10, 20, 30, 40, 50]
        results = {}
        
        for user_count in user_counts:
            test_metrics = PerformanceMetrics()
            test_load_gen = LoadGenerator(BASE_URL, test_metrics)
            
            await test_load_gen.concurrent_load_test(concurrent_users=user_count, duration=30)
            
            summary = test_metrics.get_summary()
            results[user_count] = summary
            
            # Stop if success rate drops below 50%
            if summary['success_rate'] < 50.0:
                break
        
        # Save stress test results
        self._save_test_results('stress_test', results)
        
        # Find maximum sustainable load
        max_sustainable_users = max([users for users, result in results.items() 
                                   if result['success_rate'] >= 70.0])
        
        assert max_sustainable_users >= 20, f"System can only sustain {max_sustainable_users} users"
    
    @pytest.mark.asyncio
    async def test_endurance_test(self):
        """Test system stability over extended period"""
        metrics = PerformanceMetrics()
        load_gen = LoadGenerator(BASE_URL, metrics)
        
        # Run for 10 minutes with moderate load
        await load_gen.concurrent_load_test(concurrent_users=15, duration=600)
        
        summary = metrics.get_summary()
        
        # Save results
        self._save_test_results('endurance_test', summary)
        
        # Verify system remains stable
        assert summary['success_rate'] >= 85.0
        assert summary['requests_per_second'] >= 5.0
        
        # Check for memory leaks (memory should be relatively stable)
        memory_samples = metrics.memory_samples
        if len(memory_samples) > 10:
            first_quarter = memory_samples[:len(memory_samples)//4]
            last_quarter = memory_samples[-len(memory_samples)//4:]
            
            avg_early = statistics.mean(first_quarter)
            avg_late = statistics.mean(last_quarter)
            
            # Memory growth should be less than 50% over the test
            memory_growth_ratio = avg_late / avg_early
            assert memory_growth_ratio < 1.5, f"Memory grew by {(memory_growth_ratio-1)*100:.1f}%"
    
    def _save_test_results(self, test_name: str, results: Dict[str, Any]):
        """Save test results to file for analysis"""
        results_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"performance_{test_name}_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump({
                'test_name': test_name,
                'timestamp': timestamp,
                'results': results
            }, f, indent=2)
        
        print(f"Performance results saved to: {filepath}")


@pytest.mark.performance
class TestMemoryPerformance:
    """Test memory usage and garbage collection performance"""
    
    @pytest.mark.asyncio
    async def test_memory_usage_patterns(self):
        """Test memory usage patterns during various operations"""
        import tracemalloc
        
        tracemalloc.start()
        initial_memory = tracemalloc.get_traced_memory()[0]
        
        # Perform various operations
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # Health checks
            for _ in range(100):
                await client.get(f"{BASE_URL}/health")
            
            health_memory = tracemalloc.get_traced_memory()[0]
            
            # Chat operations
            for i in range(10):
                await client.post(
                    f"{BASE_URL}/api/v1/chat/",
                    json={'message': f'Memory test {i}', 'model': 'tinyllama'}
                )
            
            chat_memory = tracemalloc.get_traced_memory()[0]
            
            # Force garbage collection
            gc.collect()
            final_memory = tracemalloc.get_traced_memory()[0]
        
        tracemalloc.stop()
        
        # Analyze memory usage
        health_growth = (health_memory - initial_memory) / (1024 * 1024)  # MB
        chat_growth = (chat_memory - health_memory) / (1024 * 1024)  # MB
        final_growth = (final_memory - initial_memory) / (1024 * 1024)  # MB
        
        # Assertions
        assert health_growth < 50, f"Health operations used {health_growth:.1f}MB"
        assert chat_growth < 100, f"Chat operations used {chat_growth:.1f}MB"
        assert final_growth < 150, f"Total memory growth {final_growth:.1f}MB exceeds limit"
    
    def test_garbage_collection_efficiency(self):
        """Test garbage collection efficiency"""
        import weakref
        
        # Create objects that should be garbage collected
        objects = []
        weak_refs = []
        
        for i in range(1000):
            obj = {'data': f'test_object_{i}', 'large_data': 'x' * 1000}
            objects.append(obj)
            weak_refs.append(weakref.ref(obj))
        
        # Clear strong references
        objects.clear()
        
        # Force garbage collection
        gc.collect()
        
        # Count objects still alive
        alive_count = sum(1 for ref in weak_refs if ref() is not None)
        
        # Most objects should be garbage collected
        assert alive_count < 100, f"{alive_count} objects not garbage collected"


@pytest.mark.performance
class TestDatabasePerformance:
    """Test database performance characteristics"""
    
    @pytest.mark.asyncio
    async def test_database_connection_performance(self):
        """Test database connection performance through API"""
        response_times = []
        
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            for i in range(50):
                start_time = time.time()
                
                # Test endpoint that uses database
                response = await client.get(f"{BASE_URL}/health")
                
                response_time = (time.time() - start_time) * 1000
                response_times.append(response_time)
                
                assert response.status_code == 200
        
        # Analyze performance
        avg_response_time = statistics.mean(response_times)
        p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        
        assert avg_response_time < 100, f"Average DB response time {avg_response_time:.1f}ms > 100ms"
        assert p95_response_time < 200, f"P95 DB response time {p95_response_time:.1f}ms > 200ms"
    
    @pytest.mark.asyncio
    async def test_cache_performance(self):
        """Test cache performance improvements"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            # First request (cache miss)
            start_time = time.time()
            first_response = await client.get(f"{BASE_URL}/api/v1/models/")
            first_time = (time.time() - start_time) * 1000
            
            # Second request (potential cache hit)
            start_time = time.time()
            second_response = await client.get(f"{BASE_URL}/api/v1/models/")
            second_time = (time.time() - start_time) * 1000
            
            # Both should succeed
            assert first_response.status_code == 200
            assert second_response.status_code == 200
            
            # Results should be consistent
            assert first_response.json() == second_response.json()
            
            # Performance improvement not guaranteed but both should be reasonable
            assert first_time < 5000  # First request should be reasonable
            assert second_time < 5000  # Second request should be reasonable


@pytest.mark.performance
class TestNetworkPerformance:
    """Test network performance characteristics"""
    
    @pytest.mark.asyncio
    async def test_connection_pooling_efficiency(self):
        """Test connection pooling reduces overhead"""
        # Test with connection reuse (normal client)
        reuse_times = []
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            for _ in range(20):
                start_time = time.time()
                response = await client.get(f"{BASE_URL}/health")
                response_time = (time.time() - start_time) * 1000
                reuse_times.append(response_time)
                assert response.status_code == 200
        
        # Test without connection reuse (new client each time)
        no_reuse_times = []
        for _ in range(20):
            start_time = time.time()
            async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
                response = await client.get(f"{BASE_URL}/health")
            response_time = (time.time() - start_time) * 1000
            no_reuse_times.append(response_time)
            assert response.status_code == 200
        
        # Connection reuse should generally be faster
        avg_reuse = statistics.mean(reuse_times)
        avg_no_reuse = statistics.mean(no_reuse_times)
        
        # Both should be reasonable, with reuse potentially faster
        assert avg_reuse < 1000, f"Connection reuse avg {avg_reuse:.1f}ms too slow"
        assert avg_no_reuse < 2000, f"No reuse avg {avg_no_reuse:.1f}ms too slow"
    
    @pytest.mark.asyncio
    async def test_large_response_handling(self):
        """Test handling of large responses"""
        async with httpx.AsyncClient(timeout=TEST_TIMEOUT) as client:
            start_time = time.time()
            
            # Request that might return large response (like metrics)
            response = await client.get(f"{BASE_URL}/metrics")
            
            response_time = (time.time() - start_time) * 1000
            response_size = len(response.content)
            
            assert response.status_code == 200
            assert response_time < 5000, f"Large response took {response_time:.1f}ms"
            
            # Calculate throughput (MB/s)
            if response_size > 1024:  # If response is > 1KB
                throughput_mbps = (response_size / (1024 * 1024)) / (response_time / 1000)
                assert throughput_mbps > 1.0, f"Throughput {throughput_mbps:.1f} MB/s too low"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
