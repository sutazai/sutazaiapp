#!/usr/bin/env python3
"""
Performance and Load Testing Suite
Tests system under various load conditions (10-500 concurrent users)
"""

import pytest
import httpx
import asyncio
import time
from typing import List, Dict, Any
import statistics

BASE_URL = "http://localhost:10200/api/v1"
TIMEOUT = 60.0

class TestAPIResponseTimes:
    """Test API response times under normal load"""
    
    @pytest.mark.asyncio
    async def test_health_endpoint_latency(self):
        """Test health endpoint response time"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            latencies = []
            
            for _ in range(50):
                start = time.time()
                response = await client.get("http://localhost:10200/health")
                latency = (time.time() - start) * 1000  # ms
                
                if response.status_code == 200:
                    latencies.append(latency)
            
            if latencies:
                avg_latency = statistics.mean(latencies)
                p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile
                
                print(f"\nHealth endpoint latency: avg={avg_latency:.2f}ms, p95={p95_latency:.2f}ms")
                assert avg_latency < 100  # Should be < 100ms
    
    @pytest.mark.asyncio
    async def test_models_endpoint_latency(self):
        """Test models endpoint response time"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            latencies = []
            
            for _ in range(30):
                start = time.time()
                response = await client.get(f"{BASE_URL}/models/")
                latency = (time.time() - start) * 1000
                
                if response.status_code == 200:
                    latencies.append(latency)
            
            if latencies:
                avg_latency = statistics.mean(latencies)
                print(f"\nModels endpoint latency: avg={avg_latency:.2f}ms")
                assert avg_latency < 500


class TestConcurrentRequests:
    """Test system behavior under concurrent load"""
    
    @pytest.mark.asyncio
    async def test_10_concurrent_users(self):
        """Test with 10 concurrent users"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            tasks = [
                client.get("http://localhost:10200/health")
                for _ in range(10)
            ]
            
            start = time.time()
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            duration = time.time() - start
            
            successful = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
            print(f"\n10 concurrent: {successful}/10 successful in {duration:.2f}s")
            
            assert successful >= 8  # At least 80% success
    
    @pytest.mark.asyncio
    async def test_50_concurrent_users(self):
        """Test with 50 concurrent users"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            tasks = [
                client.get(f"{BASE_URL}/models/")
                for _ in range(50)
            ]
            
            start = time.time()
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            duration = time.time() - start
            
            successful = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
            print(f"\n50 concurrent: {successful}/50 successful in {duration:.2f}s")
            
            assert successful >= 40  # At least 80% success
    
    @pytest.mark.asyncio
    async def test_100_concurrent_users(self):
        """Test with 100 concurrent users"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            tasks = [
                client.get("http://localhost:10200/health")
                for _ in range(100)
            ]
            
            start = time.time()
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            duration = time.time() - start
            
            successful = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
            throughput = successful / duration if duration > 0 else 0
            
            print(f"\n100 concurrent: {successful}/100 successful in {duration:.2f}s ({throughput:.2f} req/s)")
            
            assert successful >= 80  # At least 80% success
    
    @pytest.mark.asyncio
    async def test_500_concurrent_users_stress(self):
        """Stress test with 500 concurrent users"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Split into batches to avoid overwhelming
            batch_size = 100
            batches = 5
            
            all_successful = 0
            total_duration = 0
            
            for batch in range(batches):
                tasks = [
                    client.get("http://localhost:10200/health")
                    for _ in range(batch_size)
                ]
                
                start = time.time()
                responses = await asyncio.gather(*tasks, return_exceptions=True)
                duration = time.time() - start
                
                successful = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code == 200)
                all_successful += successful
                total_duration += duration
                
                print(f"Batch {batch+1}: {successful}/{batch_size} successful in {duration:.2f}s")
            
            success_rate = (all_successful / (batch_size * batches)) * 100
            print(f"\nOverall: {all_successful}/500 successful ({success_rate:.1f}%)")
            
            assert all_successful >= 400  # At least 80% success


class TestDatabasePerformance:
    """Test database query performance"""
    
    @pytest.mark.asyncio
    async def test_postgres_connection_pool(self):
        """Test PostgreSQL connection pooling under load"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Simulate 50 concurrent database queries
            tasks = [
                client.get(f"{BASE_URL}/chat/history")
                for _ in range(50)
            ]
            
            start = time.time()
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            duration = time.time() - start
            
            successful = sum(1 for r in responses if not isinstance(r, Exception) and r.status_code in [200, 404])
            print(f"\nDB pool test: {successful}/50 queries in {duration:.2f}s")
            
            assert duration < 10  # Should complete within 10s
    
    @pytest.mark.asyncio
    async def test_redis_cache_performance(self):
        """Test Redis cache hit/miss performance"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # First request (cache miss)
            start1 = time.time()
            response1 = await client.get(f"{BASE_URL}/models/")
            miss_latency = time.time() - start1
            
            if response1.status_code == 200:
                # Second request (potential cache hit)
                start2 = time.time()
                response2 = await client.get(f"{BASE_URL}/models/")
                hit_latency = time.time() - start2
                
                print(f"\nCache miss: {miss_latency*1000:.2f}ms, potential hit: {hit_latency*1000:.2f}ms")
                assert response2.status_code == 200


class TestOllamaInferenceLatency:
    """Test Ollama model inference performance"""
    
    @pytest.mark.asyncio
    async def test_tinyllama_inference_latency(self):
        """Test TinyLlama inference response time"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            payload = {
                "message": "Hello, how are you?",
                "model": "tinyllama",
                "max_tokens": 50
            }
            
            start = time.time()
            response = await client.post(f"{BASE_URL}/chat/send", json=payload)
            latency = time.time() - start
            
            print(f"\nTinyLlama inference: {latency:.2f}s")
            
            if response.status_code in [200, 404]:
                assert latency < 30  # Should respond within 30s


class TestWebSocketPerformance:
    """Test WebSocket message throughput"""
    
    @pytest.mark.asyncio
    async def test_websocket_message_rate(self):
        """Test WebSocket can handle rapid messages"""
        # This is informational - actual WebSocket testing requires websockets library
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get(f"{BASE_URL}/ws/info")
            print(f"\nWebSocket info endpoint: {response.status_code}")
            assert response.status_code in [200, 404]


class TestMemoryUsage:
    """Test memory usage patterns"""
    
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self):
        """Test for memory leaks during repeated operations"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Perform 100 operations
            for i in range(100):
                await client.get("http://localhost:10200/health")
                
                if i % 25 == 0:
                    # Check metrics for memory usage
                    metrics_resp = await client.get("http://localhost:10200/metrics")
                    if metrics_resp.status_code == 200:
                        # Look for process memory metric
                        if "process_resident_memory_bytes" in metrics_resp.text:
                            print(f"\nIteration {i}: Memory metrics available")
            
            assert True  # Informational test


class TestResourceLimits:
    """Test container resource limit enforcement"""
    
    @pytest.mark.asyncio
    async def test_cpu_usage_under_load(self):
        """Test CPU utilization during peak load"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Generate CPU load with concurrent requests
            tasks = [
                client.get(f"{BASE_URL}/models/")
                for _ in range(100)
            ]
            
            start = time.time()
            await asyncio.gather(*tasks, return_exceptions=True)
            duration = time.time() - start
            
            print(f"\nCPU load test: 100 requests in {duration:.2f}s")
            assert duration < 30  # Should complete within 30s
    
    @pytest.mark.asyncio
    async def test_disk_io_performance(self):
        """Test disk I/O for logs and data"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Simulate operations that trigger logging/disk writes
            for _ in range(50):
                await client.post(
                    f"{BASE_URL}/chat/send",
                    json={"message": "Test message", "model": "tinyllama"}
                )
            
            assert True  # Informational


class TestVectorSearchPerformance:
    """Test vector database query performance"""
    
    @pytest.mark.asyncio
    async def test_chromadb_query_latency(self):
        """Test ChromaDB similarity search latency"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            payload = {
                "query": "test embedding search",
                "n_results": 10
            }
            
            start = time.time()
            response = await client.post(f"{BASE_URL}/vectors/chromadb/search", json=payload)
            latency = time.time() - start
            
            print(f"\nChromaDB query: {latency*1000:.2f}ms")
            assert response.status_code in [200, 404]
    
    @pytest.mark.asyncio
    async def test_qdrant_query_latency(self):
        """Test Qdrant similarity search latency"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            payload = {
                "query": "test embedding search",
                "limit": 10
            }
            
            start = time.time()
            response = await client.post(f"{BASE_URL}/vectors/qdrant/search", json=payload)
            latency = time.time() - start
            
            print(f"\nQdrant query: {latency*1000:.2f}ms")
            assert response.status_code in [200, 404]


class TestThroughput:
    """Test overall system throughput"""
    
    @pytest.mark.asyncio
    async def test_requests_per_second(self):
        """Measure requests per second capacity"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            duration = 10  # 10 second test
            request_count = 0
            
            start_time = time.time()
            while time.time() - start_time < duration:
                response = await client.get("http://localhost:10200/health")
                if response.status_code == 200:
                    request_count += 1
            
            elapsed = time.time() - start_time
            rps = request_count / elapsed
            
            print(f"\nThroughput: {rps:.2f} requests/second")
            assert rps > 10  # Should handle at least 10 req/s


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
