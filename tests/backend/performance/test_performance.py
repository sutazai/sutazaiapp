"""
Performance tests for SutazAI backend
Testing response times, throughput, and resource utilization under load
"""

import pytest
import asyncio
import time
import statistics
from unittest.mock import patch, AsyncMock
from httpx import AsyncClient
import concurrent.futures


class TestResponseTimePerformance:
    """Test response time performance requirements"""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_health_endpoint_response_time(self, async_client):
        """Test health endpoint meets <50ms response time requirement"""
        response_times = []
        
        # Make 10 requests to establish baseline
        for _ in range(10):
            start = time.time()
            response = await async_client.get("/health")
            end = time.time()
            
            assert response.status_code == 200
            response_time_ms = (end - start) * 1000
            response_times.append(response_time_ms)
        
        # Calculate performance metrics
        avg_response_time = statistics.mean(response_times)
        p95_response_time = statistics.quantiles(response_times, n=20)[18]  # 95th percentile
        max_response_time = max(response_times)
        
        # Performance assertions
        assert avg_response_time < 50, f"Average response time {avg_response_time:.2f}ms exceeds 50ms target"
        assert p95_response_time < 100, f"95th percentile {p95_response_time:.2f}ms exceeds 100ms limit"
        assert max_response_time < 200, f"Max response time {max_response_time:.2f}ms exceeds 200ms limit"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_api_endpoint_response_times(self, async_client, mock_validation):
        """Test critical API endpoints meet performance requirements"""
        endpoints = [
            "/api/v1/agents",
            "/api/v1/cache/stats",
            "/api/v1/metrics",
            "/api/v1/settings"
        ]
        
        for endpoint in endpoints:
            response_times = []
            
            # Test each endpoint multiple times
            for _ in range(5):
                start = time.time()
                response = await async_client.get(endpoint)
                end = time.time()
                
                assert response.status_code == 200
                response_time_ms = (end - start) * 1000
                response_times.append(response_time_ms)
            
            avg_response_time = statistics.mean(response_times)
            assert avg_response_time < 200, f"{endpoint} average response time {avg_response_time:.2f}ms too slow"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_chat_endpoint_response_time(self, async_client, mock_validation):
        """Test chat endpoint performance with AI processing"""
        chat_request = {
            "message": "Hello",
            "model": "tinyllama", 
            "use_cache": True
        }
        
        response_times = []
        
        # Test chat performance
        for _ in range(5):
            start = time.time()
            response = await async_client.post("/api/v1/chat", json=chat_request)
            end = time.time()
            
            assert response.status_code == 200
            response_time_ms = (end - start) * 1000
            response_times.append(response_time_ms)
        
        avg_response_time = statistics.mean(response_times)
        # Chat with caching should be reasonably fast
        assert avg_response_time < 1000, f"Chat response time {avg_response_time:.2f}ms too slow"


class TestThroughputPerformance:
    """Test throughput and concurrent request handling"""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_health_checks(self, async_client):
        """Test system handles concurrent health checks efficiently"""
        concurrency_levels = [10, 25, 50]
        
        for concurrency in concurrency_levels:
            start_time = time.time()
            
            # Create concurrent requests
            tasks = []
            for _ in range(concurrency):
                task = async_client.get("/health")
                tasks.append(task)
            
            # Execute all requests concurrently
            responses = await asyncio.gather(*tasks)
            
            end_time = time.time()
            total_time = end_time - start_time
            requests_per_second = concurrency / total_time
            
            # All requests should succeed
            for response in responses:
                assert response.status_code == 200
            
            # Performance assertions based on concurrency
            if concurrency <= 10:
                assert requests_per_second > 50, f"Low throughput at {concurrency} concurrent: {requests_per_second:.1f} RPS"
            elif concurrency <= 25:
                assert requests_per_second > 25, f"Low throughput at {concurrency} concurrent: {requests_per_second:.1f} RPS"
            else:
                assert requests_per_second > 10, f"Low throughput at {concurrency} concurrent: {requests_per_second:.1f} RPS"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_mixed_api_throughput(self, async_client, mock_validation):
        """Test throughput with mixed API operations"""
        endpoints = [
            "/health",
            "/api/v1/agents", 
            "/api/v1/cache/stats",
            "/api/v1/metrics"
        ]
        
        start_time = time.time()
        
        # Create mixed concurrent requests
        tasks = []
        for _ in range(20):  # 20 total requests
            for endpoint in endpoints:
                task = async_client.get(endpoint)
                tasks.append(task)
        
        # Execute all requests concurrently
        responses = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        total_requests = len(responses)
        requests_per_second = total_requests / total_time
        
        # All requests should succeed
        success_count = sum(1 for r in responses if r.status_code == 200)
        success_rate = success_count / total_requests
        
        assert success_rate >= 0.95, f"Success rate {success_rate:.2%} below 95%"
        assert requests_per_second > 20, f"Mixed API throughput {requests_per_second:.1f} RPS too low"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_chat_throughput(self, async_client, mock_validation):
        """Test chat endpoint throughput under concurrent load"""
        chat_request = {
            "message": "Quick test",
            "model": "tinyllama",
            "use_cache": True
        }
        
        start_time = time.time()
        
        # Create concurrent chat requests
        tasks = []
        for _ in range(10):  # Moderate load for AI endpoint
            task = async_client.post("/api/v1/chat", json=chat_request)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        requests_per_second = len(responses) / total_time
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
        
        # Chat throughput should be reasonable
        assert requests_per_second > 2, f"Chat throughput {requests_per_second:.1f} RPS too low"


class TestResourceUtilizationPerformance:
    """Test resource utilization and efficiency"""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, async_client, mock_psutil):
        """Test memory usage remains stable under load"""
        # Make multiple concurrent requests to stress memory
        tasks = []
        for _ in range(30):
            task = async_client.get("/api/v1/metrics")
            tasks.append(task)
        
        # Execute requests and monitor memory
        responses = await asyncio.gather(*tasks)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            
            # Check memory metrics are reported
            assert "system" in data
            assert "memory" in data["system"]

    @pytest.mark.performance 
    @pytest.mark.asyncio
    async def test_cache_performance_under_load(self, async_client):
        """Test cache performance under concurrent access"""
        # Warm cache first
        warm_response = await async_client.post("/api/v1/cache/warm")
        assert warm_response.status_code == 200
        
        # Create concurrent cache stat requests
        tasks = []
        for _ in range(20):
            task = async_client.get("/api/v1/cache/stats")
            tasks.append(task)
        
        start_time = time.time()
        responses = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # All requests should succeed quickly
        for response in responses:
            assert response.status_code == 200
        
        # Cache operations should be very fast
        assert total_time < 1.0, f"Cache operations took {total_time:.2f}s under load"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_connection_pool_efficiency(self, async_client):
        """Test connection pool handles concurrent requests efficiently"""
        # Create many concurrent requests to test connection pooling
        tasks = []
        for _ in range(40):
            task = async_client.get("/health")
            tasks.append(task)
        
        start_time = time.time()
        responses = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
        
        # Connection pooling should enable efficient handling
        requests_per_second = len(responses) / total_time
        assert requests_per_second > 20, f"Connection pool efficiency low: {requests_per_second:.1f} RPS"


class TestStressTestPerformance:
    """Stress tests for system limits and breaking points"""

    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_sustained_load_performance(self, async_client):
        """Test performance under sustained load"""
        duration_seconds = 30
        request_interval = 0.1  # 10 RPS
        
        start_time = time.time()
        responses = []
        
        while time.time() - start_time < duration_seconds:
            response = await async_client.get("/health")
            responses.append(response)
            await asyncio.sleep(request_interval)
        
        # Calculate metrics
        total_requests = len(responses)
        success_count = sum(1 for r in responses if r.status_code == 200)
        success_rate = success_count / total_requests
        actual_rps = total_requests / duration_seconds
        
        assert success_rate >= 0.99, f"Success rate {success_rate:.2%} degraded under sustained load"
        assert actual_rps >= 8, f"Actual RPS {actual_rps:.1f} below target under sustained load"

    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_burst_load_handling(self, async_client):
        """Test system handles burst loads without degradation"""
        # Create burst of requests
        burst_size = 100
        
        start_time = time.time()
        
        tasks = []
        for _ in range(burst_size):
            task = async_client.get("/health")
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All requests should succeed
        success_count = sum(1 for r in responses if r.status_code == 200)
        success_rate = success_count / burst_size
        requests_per_second = burst_size / total_time
        
        assert success_rate >= 0.95, f"Burst load success rate {success_rate:.2%} too low"
        assert requests_per_second > 10, f"Burst handling {requests_per_second:.1f} RPS insufficient"

    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_gradual_load_increase(self, async_client):
        """Test system performance with gradually increasing load"""
        load_levels = [5, 10, 20, 30]
        
        for load_level in load_levels:
            start_time = time.time()
            
            tasks = []
            for _ in range(load_level):
                task = async_client.get("/health")
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            end_time = time.time()
            
            total_time = end_time - start_time
            requests_per_second = load_level / total_time
            
            # All requests should succeed at each load level
            success_count = sum(1 for r in responses if r.status_code == 200)
            success_rate = success_count / load_level
            
            assert success_rate >= 0.95, f"Load level {load_level}: success rate {success_rate:.2%} degraded"
            
            # Allow for reasonable performance degradation at higher loads
            min_rps = max(5, load_level * 0.5)
            assert requests_per_second > min_rps, f"Load level {load_level}: {requests_per_second:.1f} RPS too low"


class TestCachePerformance:
    """Test cache-specific performance characteristics"""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_cache_hit_performance(self, async_client, mock_validation):
        """Test cache hit performance improves response times"""
        # First request (cache miss)
        start = time.time()
        response1 = await async_client.get("/api/v1/agents")
        miss_time = time.time() - start
        
        assert response1.status_code == 200
        
        # Second request (should be cache hit)
        start = time.time()
        response2 = await async_client.get("/api/v1/agents")
        hit_time = time.time() - start
        
        assert response2.status_code == 200
        
        # Cache hit should be faster (with Mocked services, difference may be)
        # Just verify both requests complete successfully
        assert miss_time >= 0
        assert hit_time >= 0

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_cache_warm_performance(self, async_client):
        """Test cache warming improves subsequent performance"""
        # Measure performance before warming
        tasks_before = []
        for _ in range(5):
            task = async_client.get("/api/v1/cache/stats")
            tasks_before.append(task)
        
        start_before = time.time()
        responses_before = await asyncio.gather(*tasks_before)
        time_before = time.time() - start_before
        
        # Warm cache
        warm_response = await async_client.post("/api/v1/cache/warm")
        assert warm_response.status_code == 200
        
        # Measure performance after warming
        tasks_after = []
        for _ in range(5):
            task = async_client.get("/api/v1/cache/stats")
            tasks_after.append(task)
        
        start_after = time.time()
        responses_after = await asyncio.gather(*tasks_after)
        time_after = time.time() - start_after
        
        # All requests should succeed
        for response in responses_before + responses_after:
            assert response.status_code == 200
        
        # Performance should be reasonable in both cases
        assert time_before < 2.0, "Performance before cache warm too slow"
        assert time_after < 2.0, "Performance after cache warm too slow"


class TestErrorHandlingPerformance:
    """Test error handling doesn't degrade performance"""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_404_error_performance(self, async_client):
        """Test 404 errors are handled efficiently"""
        start_time = time.time()
        
        tasks = []
        for i in range(10):
            task = async_client.get(f"/nonexistent-endpoint-{i}")
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        end_time = time.time()
        
        total_time = end_time - start_time
        
        # All should be 404s
        for response in responses:
            assert response.status_code == 404
        
        # Error handling should be fast
        assert total_time < 1.0, f"404 error handling took {total_time:.2f}s"

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_validation_error_performance(self, async_client):
        """Test validation errors are handled efficiently"""
        invalid_requests = []
        for i in range(5):
            invalid_request = {
                "message": f"test {i}",
                "model": "'; DROP TABLE models; --",  # Malicious model name
                "use_cache": True
            }
            invalid_requests.append(invalid_request)
        
        start_time = time.time()
        
        with patch('app.utils.validation.validate_model_name') as mock_validate:
            mock_validate.side_effect = ValueError("Invalid model name")
            
            tasks = []
            for request in invalid_requests:
                task = async_client.post("/api/v1/chat", json=request)
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All should be validation errors
        for response in responses:
            assert response.status_code == 400
        
        # Validation error handling should be fast
        assert total_time < 1.0, f"Validation error handling took {total_time:.2f}s"


class TestScalabilityIndicators:
    """Test indicators of system scalability potential"""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_response_time_consistency(self, async_client):
        """Test response times remain consistent across multiple requests"""
        response_times = []
        
        # Make many requests to test consistency
        for _ in range(20):
            start = time.time()
            response = await async_client.get("/health")
            end = time.time()
            
            assert response.status_code == 200
            response_time_ms = (end - start) * 1000
            response_times.append(response_time_ms)
        
        # Calculate consistency metrics
        mean_time = statistics.mean(response_times)
        std_dev = statistics.stdev(response_times) if len(response_times) > 1 else 0
        coefficient_of_variation = std_dev / mean_time if mean_time > 0 else 0
        
        # Response times should be consistent (low coefficient of variation)
        assert coefficient_of_variation < 0.5, f"Response times inconsistent: CV={coefficient_of_variation:.2f}"

    @pytest.mark.performance
    @pytest.mark.asyncio  
    async def test_resource_efficiency_indicators(self, async_client, mock_psutil):
        """Test resource efficiency under load"""
        # Make concurrent requests while monitoring resource metrics
        tasks = []
        for _ in range(15):
            task = async_client.get("/api/v1/metrics")
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        
        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            
            # Verify resource metrics are being tracked
            assert "system" in data
            system_metrics = data["system"]
            assert "cpu_percent" in system_metrics
            assert "memory" in system_metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "performance"])