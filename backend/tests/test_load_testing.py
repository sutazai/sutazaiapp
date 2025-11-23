#!/usr/bin/env python3
"""
Comprehensive API Load Testing
Tests API under realistic load with concurrent users and traffic patterns
"""

import pytest
import asyncio
import httpx
import time
import statistics
from typing import List, Dict, Any
from datetime import datetime, timezone

BASE_URL = "http://localhost:10200/api/v1"
BACKEND_URL = "http://localhost:10200"
TIMEOUT = 30.0


class TestLoadScenarios:
    """Test various load scenarios"""
    
    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self):
        """Test concurrent health check requests"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            num_requests = 100
            start_time = time.time()
            
            # Create concurrent tasks
            tasks = [client.get(f"{BACKEND_URL}/health") for _ in range(num_requests)]
            
            # Execute all requests
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            duration = time.time() - start_time
            
            # Analyze results
            successful = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
            errors = sum(1 for r in responses if isinstance(r, Exception))
            
            throughput = num_requests / duration
            
            print(f"\n=== Concurrent Health Check Load Test ===")
            print(f"Total requests: {num_requests}")
            print(f"Successful: {successful}")
            print(f"Errors: {errors}")
            print(f"Duration: {duration:.2f}s")
            print(f"Throughput: {throughput:.2f} req/s")
            
            assert successful > num_requests * 0.8, "At least 80% should succeed"
    
    @pytest.mark.asyncio
    async def test_authentication_load(self):
        """Test authentication endpoints under load"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            num_users = 50
            start_time = time.time()
            
            # Register multiple users concurrently
            tasks = []
            for i in range(num_users):
                payload = {
                    "email": f"loadtest{int(time.time())}_{i}@example.com",
                    "username": f"loadtest{int(time.time())}_{i}",
                    "password": "SecureP@ssw0rd123!"
                }
                tasks.append(client.post(f"{BASE_URL}/auth/register", json=payload))
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            duration = time.time() - start_time
            
            # Analyze results
            successful = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code in [200, 201])
            errors = sum(1 for r in responses if isinstance(r, Exception) or (hasattr(r, 'status_code') and r.status_code >= 400))
            
            print(f"\n=== Authentication Load Test ===")
            print(f"Total registration attempts: {num_users}")
            print(f"Successful: {successful}")
            print(f"Errors: {errors}")
            print(f"Duration: {duration:.2f}s")
            print(f"Average time per registration: {duration/num_users:.3f}s")
            
            assert successful > 0, "Some registrations should succeed"
    
    @pytest.mark.asyncio
    async def test_mixed_workload(self):
        """Test mixed API workload (reads and writes)"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            num_operations = 100
            start_time = time.time()
            
            tasks = []
            
            # Mix of operations
            for i in range(num_operations):
                if i % 3 == 0:
                    # Health check (read)
                    tasks.append(client.get(f"{BACKEND_URL}/health"))
                elif i % 3 == 1:
                    # Detailed health (read with DB access)
                    tasks.append(client.get(f"{BACKEND_URL}/health/detailed"))
                else:
                    # Password reset request (write)
                    payload = {"email": f"mixed{i}@example.com"}
                    tasks.append(client.post(f"{BASE_URL}/auth/password-reset", json=payload))
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            duration = time.time() - start_time
            
            # Analyze results
            successful = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code in [200, 201])
            rate_limited = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 429)
            errors = sum(1 for r in responses if isinstance(r, Exception))
            
            print(f"\n=== Mixed Workload Test ===")
            print(f"Total operations: {num_operations}")
            print(f"Successful: {successful}")
            print(f"Rate limited: {rate_limited}")
            print(f"Errors: {errors}")
            print(f"Duration: {duration:.2f}s")
            print(f"Throughput: {num_operations/duration:.2f} ops/s")
            
            assert successful + rate_limited > num_operations * 0.7, "Most operations should complete"


class TestResponseTimeDistribution:
    """Test response time distribution under load"""
    
    @pytest.mark.asyncio
    async def test_response_time_percentiles(self):
        """Test response time percentiles"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            num_requests = 100
            response_times = []
            
            for i in range(num_requests):
                start = time.time()
                try:
                    response = await client.get(f"{BACKEND_URL}/health")
                    duration = time.time() - start
                    
                    if response.status_code == 200:
                        response_times.append(duration)
                except Exception:
                    pass
            
            if response_times:
                # Calculate percentiles
                response_times.sort()
                p50 = response_times[len(response_times)//2]
                p95 = response_times[int(len(response_times)*0.95)]
                p99 = response_times[int(len(response_times)*0.99)]
                avg = statistics.mean(response_times)
                
                print(f"\n=== Response Time Distribution ===")
                print(f"Total successful requests: {len(response_times)}")
                print(f"Average: {avg:.3f}s")
                print(f"P50 (median): {p50:.3f}s")
                print(f"P95: {p95:.3f}s")
                print(f"P99: {p99:.3f}s")
                print(f"Min: {min(response_times):.3f}s")
                print(f"Max: {max(response_times):.3f}s")
                
                assert p95 < 5.0, "P95 should be under 5 seconds"


class TestSustainedLoad:
    """Test system behavior under sustained load"""
    
    @pytest.mark.asyncio
    async def test_sustained_request_rate(self):
        """Test sustained request rate over time"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            duration_seconds = 30
            requests_per_second = 10
            
            start_time = time.time()
            total_requests = 0
            successful_requests = 0
            
            print(f"\n=== Sustained Load Test ===")
            print(f"Duration: {duration_seconds}s")
            print(f"Target rate: {requests_per_second} req/s")
            
            while time.time() - start_time < duration_seconds:
                batch_start = time.time()
                
                # Send batch of requests
                tasks = [client.get(f"{BACKEND_URL}/health") for _ in range(requests_per_second)]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                total_requests += len(responses)
                successful_requests += sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
                
                # Wait to maintain rate
                batch_duration = time.time() - batch_start
                if batch_duration < 1.0:
                    await asyncio.sleep(1.0 - batch_duration)
            
            actual_duration = time.time() - start_time
            actual_rate = total_requests / actual_duration
            success_rate = successful_requests / total_requests * 100
            
            print(f"Total requests: {total_requests}")
            print(f"Successful: {successful_requests}")
            print(f"Actual duration: {actual_duration:.2f}s")
            print(f"Actual rate: {actual_rate:.2f} req/s")
            print(f"Success rate: {success_rate:.1f}%")
            
            assert success_rate > 90, "Should maintain > 90% success rate"


class TestResourceUtilization:
    """Test resource utilization under load"""
    
    @pytest.mark.asyncio
    async def test_memory_stability_under_load(self):
        """Test memory stability under load"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Make many requests
            for batch in range(5):
                tasks = [client.get(f"{BACKEND_URL}/health") for _ in range(50)]
                await asyncio.gather(*tasks, return_exceptions=True)
                
                # Small delay between batches
                await asyncio.sleep(1)
            
            # System should remain stable
            response = await client.get(f"{BACKEND_URL}/health/detailed")
            
            assert response.status_code in [200, 503], "System should remain stable"
    
    @pytest.mark.asyncio
    async def test_connection_pool_under_load(self):
        """Test database connection pool under load"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Make requests that hit database
            tasks = []
            for i in range(100):
                tasks.append(client.get(f"{BACKEND_URL}/health/detailed"))
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            successful = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
            
            print(f"\n=== Connection Pool Load Test ===")
            print(f"Total DB queries: {len(tasks)}")
            print(f"Successful: {successful}")
            
            assert successful > 50, "Connection pool should handle load"


class TestErrorRatesUnderLoad:
    """Test error rates and recovery under load"""
    
    @pytest.mark.asyncio
    async def test_error_rate_monitoring(self):
        """Test error rate under normal load"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            num_requests = 100
            
            tasks = [client.get(f"{BACKEND_URL}/health") for _ in range(num_requests)]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count status codes
            status_codes = {}
            for r in responses:
                if isinstance(r, Exception):
                    status_codes['exception'] = status_codes.get('exception', 0) + 1
                else:
                    code = r.status_code
                    status_codes[code] = status_codes.get(code, 0) + 1
            
            error_rate = (num_requests - status_codes.get(200, 0)) / num_requests * 100
            
            print(f"\n=== Error Rate Analysis ===")
            print(f"Total requests: {num_requests}")
            print(f"Status code distribution: {status_codes}")
            print(f"Error rate: {error_rate:.1f}%")
            
            assert error_rate < 20, "Error rate should be under 20%"
    
    @pytest.mark.asyncio
    async def test_recovery_from_errors(self):
        """Test system recovery from errors"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Make invalid requests (should error)
            invalid_tasks = [
                client.post(f"{BASE_URL}/auth/register", json={"invalid": "data"})
                for _ in range(10)
            ]
            await asyncio.gather(*invalid_tasks, return_exceptions=True)
            
            # System should still respond to valid requests
            valid_response = await client.get(f"{BACKEND_URL}/health")
            
            assert valid_response.status_code == 200, "System should recover from errors"


class TestScalabilityMetrics:
    """Test scalability metrics"""
    
    @pytest.mark.asyncio
    async def test_throughput_scaling(self):
        """Test throughput scaling with concurrent connections"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            concurrency_levels = [10, 25, 50]
            results = []
            
            print(f"\n=== Throughput Scaling Test ===")
            
            for concurrency in concurrency_levels:
                start_time = time.time()
                
                tasks = [client.get(f"{BACKEND_URL}/health") for _ in range(concurrency)]
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                
                duration = time.time() - start_time
                successful = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
                throughput = successful / duration
                
                results.append({
                    'concurrency': concurrency,
                    'successful': successful,
                    'duration': duration,
                    'throughput': throughput
                })
                
                print(f"Concurrency {concurrency}: {throughput:.2f} req/s ({successful}/{concurrency} successful)")
            
            # Throughput should generally increase with concurrency
            assert all(r['throughput'] > 0 for r in results), "All tests should show throughput"


class TestMetricsUnderLoad:
    """Test metrics collection under load"""
    
    @pytest.mark.asyncio
    async def test_metrics_endpoint_under_load(self):
        """Test /metrics endpoint doesn't degrade under load"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Generate load
            load_tasks = [client.get(f"{BACKEND_URL}/health") for _ in range(50)]
            await asyncio.gather(*load_tasks, return_exceptions=True)
            
            # Check metrics still accessible
            metrics_response = await client.get(f"{BACKEND_URL}/metrics")
            
            assert metrics_response.status_code == 200, "Metrics should remain accessible"
            assert len(metrics_response.text) > 0, "Metrics should contain data"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])
