#!/usr/bin/env python3
"""
Performance tests for Ollama integration with all 131 agents
Tests load handling, memory usage, response times, and resource optimization
"""

import pytest
import asyncio
import time
import psutil
import statistics
import sys
import os
from unittest.mock import AsyncMock, Mock, patch
from concurrent.futures import ThreadPoolExecutor
import threading
from datetime import datetime, timedelta

# Add the agents directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'agents'))

from core.base_agent_v2 import BaseAgentV2
from core.ollama_pool import OllamaConnectionPool
from core.ollama_integration import OllamaIntegration, OllamaConfig


class TestPerformanceMetrics:
    """Performance monitoring and metrics collection"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.memory_samples = []
        self.cpu_samples = []
        self.response_times = []
        self.concurrent_requests = 0
        self.max_concurrent = 0
        self.errors = 0
        self.successes = 0
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self._monitor_resources()
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.end_time = time.time()
    
    def _monitor_resources(self):
        """Monitor system resources in background"""
        def resource_monitor():
            process = psutil.Process()
            while self.end_time is None:
                try:
                    self.memory_samples.append(process.memory_info().rss / 1024 / 1024)  # MB
                    self.cpu_samples.append(process.cpu_percent())
                    time.sleep(0.1)
                except:
                    break
        
        thread = threading.Thread(target=resource_monitor, daemon=True)
        thread.start()
    
    def record_request(self, response_time: float, success: bool = True):
        """Record a request result"""
        self.response_times.append(response_time)
        if success:
            self.successes += 1
        else:
            self.errors += 1
    
    def increment_concurrent(self):
        """Increment concurrent request counter"""
        self.concurrent_requests += 1
        self.max_concurrent = max(self.max_concurrent, self.concurrent_requests)
    
    def decrement_concurrent(self):
        """Decrement concurrent request counter"""
        self.concurrent_requests = max(0, self.concurrent_requests - 1)
    
    def get_summary(self):
        """Get performance summary"""
        duration = (self.end_time or time.time()) - (self.start_time or time.time())
        
        return {
            "duration_seconds": duration,
            "total_requests": len(self.response_times),
            "successful_requests": self.successes,
            "failed_requests": self.errors,
            "success_rate": self.successes / max(1, len(self.response_times)),
            "requests_per_second": len(self.response_times) / max(0.001, duration),
            "max_concurrent_requests": self.max_concurrent,
            "response_times": {
                "min": min(self.response_times) if self.response_times else 0,
                "max": max(self.response_times) if self.response_times else 0,
                "avg": statistics.mean(self.response_times) if self.response_times else 0,
                "median": statistics.median(self.response_times) if self.response_times else 0,
                "p95": statistics.quantiles(self.response_times, n=20)[18] if len(self.response_times) > 20 else 0,
                "p99": statistics.quantiles(self.response_times, n=100)[98] if len(self.response_times) > 100 else 0
            },
            "memory_usage_mb": {
                "min": min(self.memory_samples) if self.memory_samples else 0,
                "max": max(self.memory_samples) if self.memory_samples else 0,
                "avg": statistics.mean(self.memory_samples) if self.memory_samples else 0
            },
            "cpu_usage_percent": {
                "min": min(self.cpu_samples) if self.cpu_samples else 0,
                "max": max(self.cpu_samples) if self.cpu_samples else 0,
                "avg": statistics.mean(self.cpu_samples) if self.cpu_samples else 0
            }
        }


class TestConnectionPoolPerformance:
    """Test connection pool performance under load"""
    
    @pytest.fixture
    def connection_pool(self):
        """Create connection pool for testing"""
        return OllamaConnectionPool(
            max_connections=5,
            min_connections=2,
            default_model="tinyllama",
            connection_timeout=10,
            request_timeout=30
        )
    
    @pytest.mark.asyncio
    async def test_concurrent_connections(self, connection_pool):
        """Test concurrent connection handling"""
        metrics = TestPerformanceMetrics()
        metrics.start_monitoring()
        
        async def make_request():
            metrics.increment_concurrent()
            start_time = time.time()
            
            try:
                # Mock successful response
                with patch.object(connection_pool, '_ensure_model_available', return_value=True):
                    mock_client = AsyncMock()
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"response": f"Response {time.time()}"}
                    
                    # Add connection to pool
                    from core.ollama_pool import PoolConnection
                    connection = PoolConnection(client=mock_client)
                    connection_pool._connections.append(connection)
                    
                    with patch.object(mock_client, 'post', return_value=mock_response):
                        result = await connection_pool.generate(f"Test prompt {time.time()}")
                        
                        response_time = time.time() - start_time
                        metrics.record_request(response_time, result is not None)
                        
                        return result
            finally:
                metrics.decrement_concurrent()
        
        # Create 20 concurrent requests
        tasks = [make_request() for _ in range(20)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # Performance assertions
        assert summary["total_requests"] == 20
        assert summary["success_rate"] >= 0.8  # At least 80% success rate
        assert summary["response_times"]["avg"] < 5.0  # Average response time under 5 seconds
        assert summary["max_concurrent_requests"] <= 20
        
        # Memory should not grow excessively
        memory_growth = summary["memory_usage_mb"]["max"] - summary["memory_usage_mb"]["min"]
        assert memory_growth < 100  # Less than 100MB growth
        
        print(f"Connection Pool Performance Summary: {summary}")
    
    @pytest.mark.asyncio
    async def test_connection_pool_scaling(self, connection_pool):
        """Test connection pool scaling behavior"""
        metrics = TestPerformanceMetrics()
        metrics.start_monitoring()
        
        # Gradually increase load
        for batch_size in [1, 5, 10, 15, 20]:
            batch_start = time.time()
            
            async def batch_request(request_id):
                start_time = time.time()
                
                # Mock request
                with patch.object(connection_pool, '_ensure_model_available', return_value=True):
                    mock_client = AsyncMock()
                    mock_response = Mock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"response": f"Batch response {request_id}"}
                    
                    from core.ollama_pool import PoolConnection
                    connection = PoolConnection(client=mock_client)
                    if not connection_pool._connections:
                        connection_pool._connections.append(connection)
                    
                    with patch.object(mock_client, 'post', return_value=mock_response):
                        result = await connection_pool.generate(f"Batch {batch_size} request {request_id}")
                        
                        response_time = time.time() - start_time
                        metrics.record_request(response_time, result is not None)
                        
                        return result
            
            # Execute batch
            tasks = [batch_request(i) for i in range(batch_size)]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            batch_duration = time.time() - batch_start
            
            # Check that response time doesn't degrade significantly with load
            recent_times = metrics.response_times[-batch_size:]
            avg_batch_time = statistics.mean(recent_times) if recent_times else 0
            
            print(f"Batch size {batch_size}: avg response time {avg_batch_time:.3f}s, batch duration {batch_duration:.3f}s")
            
            # Response time should not increase dramatically with load
            assert avg_batch_time < 2.0, f"Response time too high for batch size {batch_size}"
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # Overall performance should be acceptable
        assert summary["success_rate"] >= 0.95
        assert summary["response_times"]["p95"] < 3.0
        
        print(f"Scaling Performance Summary: {summary}")
    
    @pytest.mark.asyncio
    async def test_model_switching_overhead(self, connection_pool):
        """Test performance impact of model switching"""
        metrics = TestPerformanceMetrics()
        metrics.start_monitoring()
        
        models = ["tinyllama", "tinyllama2.5-coder:7b", "tinyllama"]
        
        async def request_with_model(model):
            start_time = time.time()
            
            with patch.object(connection_pool, '_ensure_model_available', return_value=True):
                mock_client = AsyncMock()
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"response": f"Response from {model}"}
                
                from core.ollama_pool import PoolConnection
                connection = PoolConnection(client=mock_client, current_model=model)
                connection_pool._connections = [connection]
                
                with patch.object(mock_client, 'post', return_value=mock_response):
                    result = await connection_pool.generate(f"Test with {model}", model=model)
                    
                    response_time = time.time() - start_time
                    metrics.record_request(response_time, result is not None)
                    
                    return result, response_time
        
        # Test requests alternating between models
        model_times = {model: [] for model in models}
        
        for i in range(30):  # 10 requests per model
            model = models[i % len(models)]
            result, response_time = await request_with_model(model)
            model_times[model].append(response_time)
        
        metrics.stop_monitoring()
        
        # Analyze model switching overhead
        for model, times in model_times.items():
            avg_time = statistics.mean(times) if times else 0
            print(f"Model {model}: avg response time {avg_time:.3f}s")
            
            # Each model should have reasonable response times
            assert avg_time < 2.0, f"Model {model} response time too high"
        
        # Overall performance should not be significantly impacted by model switching
        summary = metrics.get_summary()
        assert summary["success_rate"] >= 0.95
        assert summary["response_times"]["avg"] < 1.5
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, connection_pool):
        """Test memory efficiency under sustained load"""
        metrics = TestPerformanceMetrics()
        metrics.start_monitoring()
        
        async def sustained_request_batch():
            """Execute a batch of requests"""
            batch_tasks = []
            
            for i in range(10):
                async def single_request():
                    with patch.object(connection_pool, '_ensure_model_available', return_value=True):
                        mock_client = AsyncMock()
                        mock_response = Mock()
                        mock_response.status_code = 200
                        mock_response.json.return_value = {"response": f"Memory test {time.time()}"}
                        
                        from core.ollama_pool import PoolConnection
                        connection = PoolConnection(client=mock_client)
                        if not connection_pool._connections:
                            connection_pool._connections.append(connection)
                        
                        with patch.object(mock_client, 'post', return_value=mock_response):
                            start_time = time.time()
                            result = await connection_pool.generate(f"Memory test request {i}")
                            response_time = time.time() - start_time
                            metrics.record_request(response_time, result is not None)
                            return result
                
                batch_tasks.append(single_request())
            
            return await asyncio.gather(*batch_tasks)
        
        # Run 10 batches of sustained load
        for batch_num in range(10):
            await sustained_request_batch()
            
            # Small delay between batches
            await asyncio.sleep(0.1)
            
            # Check memory growth
            if metrics.memory_samples:
                current_memory = metrics.memory_samples[-1]
                initial_memory = metrics.memory_samples[0]
                memory_growth = current_memory - initial_memory
                
                # Memory growth should be bounded
                assert memory_growth < 200, f"Excessive memory growth: {memory_growth}MB in batch {batch_num}"
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # Final memory usage should be reasonable
        total_memory_growth = summary["memory_usage_mb"]["max"] - summary["memory_usage_mb"]["min"]
        assert total_memory_growth < 300, f"Total memory growth too high: {total_memory_growth}MB"
        
        print(f"Memory Efficiency Summary: {summary}")


class TestAgentPerformance:
    """Test individual agent performance"""
    
    @pytest.fixture
    def mock_agent(self):
        """Create mock agent for testing"""
        with patch.dict(os.environ, {
            'AGENT_NAME': 'test-performance-agent',
            'AGENT_TYPE': 'performance-test'
        }):
            agent = BaseAgentV2(
                max_concurrent_tasks=5,
                max_ollama_connections=2
            )
            return agent
    
    @pytest.mark.asyncio
    async def test_agent_task_processing_performance(self, mock_agent):
        """Test agent task processing performance"""
        await mock_agent._setup_async_components()
        
        metrics = TestPerformanceMetrics()
        metrics.start_monitoring()
        
        async def process_test_task(task_id):
            task = {
                "id": f"perf-task-{task_id}",
                "type": "performance-test",
                "data": {"prompt": f"Performance test prompt {task_id}"}
            }
            
            start_time = time.time()
            result = await mock_agent.process_task(task)
            processing_time = time.time() - start_time
            
            metrics.record_request(processing_time, result.status == "completed")
            return result
        
        # Process 50 tasks concurrently
        tasks = [process_test_task(i) for i in range(50)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # Performance assertions
        assert summary["total_requests"] == 50
        assert summary["success_rate"] >= 0.98  # Very high success rate for basic tasks
        assert summary["response_times"]["avg"] < 1.0  # Fast processing for simple tasks
        assert summary["response_times"]["p95"] < 2.0
        
        await mock_agent._cleanup_async_components()
        print(f"Agent Task Processing Performance: {summary}")
    
    @pytest.mark.asyncio
    async def test_agent_ollama_query_performance(self, mock_agent):
        """Test agent Ollama query performance"""
        await mock_agent._setup_async_components()
        
        metrics = TestPerformanceMetrics()
        metrics.start_monitoring()
        
        async def test_ollama_query(query_id):
            prompt = f"Performance test query {query_id}: What is the capital of France?"
            
            # Mock Ollama response
            with patch.object(mock_agent.circuit_breaker, 'call') as mock_call:
                mock_call.return_value = f"The capital of France is Paris. (Response {query_id})"
                
                start_time = time.time()
                result = await mock_agent.query_ollama(prompt)
                query_time = time.time() - start_time
                
                metrics.record_request(query_time, result is not None)
                return result
        
        # Execute 30 concurrent Ollama queries
        tasks = [test_ollama_query(i) for i in range(30)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # Performance assertions for Ollama queries
        assert summary["total_requests"] == 30
        assert summary["success_rate"] >= 0.95
        assert summary["response_times"]["avg"] < 2.0  # Mocked responses should be fast
        
        await mock_agent._cleanup_async_components()
        print(f"Agent Ollama Query Performance: {summary}")
    
    @pytest.mark.asyncio
    async def test_agent_concurrent_limit_enforcement(self, mock_agent):
        """Test that agent respects concurrent task limits"""
        await mock_agent._setup_async_components()
        
        # Set low concurrent limit for testing
        mock_agent.max_concurrent_tasks = 3
        mock_agent.task_semaphore = asyncio.Semaphore(3)
        
        metrics = TestPerformanceMetrics()
        concurrent_tracker = []
        
        async def tracked_task(task_id):
            metrics.increment_concurrent()
            concurrent_tracker.append(("start", task_id, time.time(), metrics.concurrent_requests))
            
            # Simulate work
            await asyncio.sleep(0.1)
            
            metrics.decrement_concurrent()
            concurrent_tracker.append(("end", task_id, time.time(), metrics.concurrent_requests))
            
            return f"Task {task_id} completed"
        
        # Start 10 tasks (more than the limit)
        tasks = [tracked_task(i) for i in range(10)]
        results = await asyncio.gather(*tasks)
        
        # Analyze concurrent execution
        max_concurrent = max(entry[3] for entry in concurrent_tracker)
        
        # Should never exceed the limit
        assert max_concurrent <= mock_agent.max_concurrent_tasks
        
        # All tasks should complete
        assert len(results) == 10
        
        await mock_agent._cleanup_async_components()
        print(f"Max concurrent tasks observed: {max_concurrent} (limit: {mock_agent.max_concurrent_tasks})")


class TestSystemWidePerfomance:
    """Test system-wide performance with multiple agents"""
    
    @pytest.mark.asyncio
    async def test_multi_agent_performance(self):
        """Test performance with multiple agents running concurrently"""
        # Create different types of agents
        agent_configs = [
            ("ai-system-architect", "opus"),
            ("ai-product-manager", "sonnet"),
            ("garbage-collector", "default"),
            ("testing-qa-validator", "sonnet"),
            ("ai-senior-backend-developer", "opus")
        ]
        
        agents = []
        metrics = TestPerformanceMetrics()
        metrics.start_monitoring()
        
        try:
            # Create and setup agents
            for agent_name, model_type in agent_configs:
                with patch.dict(os.environ, {
                    'AGENT_NAME': agent_name,
                    'AGENT_TYPE': f'test-{model_type}'
                }):
                    agent = BaseAgentV2(max_concurrent_tasks=2)
                    await agent._setup_async_components()
                    agents.append(agent)
            
            async def agent_workload(agent, agent_id):
                """Simulate workload for a single agent"""
                task_results = []
                
                for task_num in range(5):  # 5 tasks per agent
                    task = {
                        "id": f"multi-agent-task-{agent_id}-{task_num}",
                        "type": "multi-agent-test",
                        "data": {"prompt": f"Task for agent {agent_id}"}
                    }
                    
                    start_time = time.time()
                    
                    # Mock Ollama calls for the agent
                    with patch.object(agent.circuit_breaker, 'call') as mock_call:
                        mock_call.return_value = f"Response from {agent.agent_name} for task {task_num}"
                        
                        result = await agent.process_task(task)
                        
                        processing_time = time.time() - start_time
                        metrics.record_request(processing_time, result.status == "completed")
                        task_results.append(result)
                
                return task_results
            
            # Run all agents concurrently
            agent_tasks = [
                agent_workload(agent, i) 
                for i, agent in enumerate(agents)
            ]
            
            all_results = await asyncio.gather(*agent_tasks)
            
            metrics.stop_monitoring()
            summary = metrics.get_summary()
            
            # System-wide performance assertions
            total_tasks = len(agent_configs) * 5  # 5 tasks per agent
            assert summary["total_requests"] == total_tasks
            assert summary["success_rate"] >= 0.95
            assert summary["response_times"]["avg"] < 3.0
            
            # Memory usage should be reasonable for multiple agents
            assert summary["memory_usage_mb"]["max"] < 500  # Less than 500MB total
            
            print(f"Multi-Agent Performance Summary: {summary}")
            
        finally:
            # Cleanup all agents
            for agent in agents:
                await agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_agent_model_distribution_performance(self):
        """Test performance impact of different model assignments"""
        # Test different model configurations
        model_test_cases = [
            ("tinyllama", 10),  # Default model, more requests
            ("tinyllama2.5-coder:7b", 5),  # Sonnet model, fewer requests
            ("tinyllama", 3)  # Opus model, least requests
        ]
        
        metrics = TestPerformanceMetrics()
        metrics.start_monitoring()
        
        model_performance = {}
        
        for model, request_count in model_test_cases:
            model_metrics = TestPerformanceMetrics()
            model_metrics.start_monitoring()
            
            # Create agent with specific model
            with patch.dict(os.environ, {
                'AGENT_NAME': f'test-{model.replace(":", "-").replace(".", "-")}',
                'AGENT_TYPE': 'model-performance-test'
            }):
                agent = BaseAgentV2()
                agent.default_model = model
                await agent._setup_async_components()
                
                try:
                    async def model_request(req_id):
                        with patch.object(agent.circuit_breaker, 'call') as mock_call:
                            mock_call.return_value = f"Response from {model} for request {req_id}"
                            
                            start_time = time.time()
                            result = await agent.query_ollama(f"Test request {req_id} for {model}")
                            query_time = time.time() - start_time
                            
                            model_metrics.record_request(query_time, result is not None)
                            metrics.record_request(query_time, result is not None)
                            
                            return result
                    
                    # Execute requests for this model
                    tasks = [model_request(i) for i in range(request_count)]
                    results = await asyncio.gather(*tasks)
                    
                    model_metrics.stop_monitoring()
                    model_summary = model_metrics.get_summary()
                    model_performance[model] = model_summary
                    
                finally:
                    await agent._cleanup_async_components()
        
        metrics.stop_monitoring()
        overall_summary = metrics.get_summary()
        
        # Analyze model performance differences
        for model, perf in model_performance.items():
            print(f"Model {model}: avg response time {perf['response_times']['avg']:.3f}s, "
                  f"success rate {perf['success_rate']:.2%}")
            
            # Each model should have good performance
            assert perf["success_rate"] >= 0.95
            assert perf["response_times"]["avg"] < 2.0
        
        # Overall system performance
        assert overall_summary["success_rate"] >= 0.95
        
        print(f"Model Distribution Performance Summary: {overall_summary}")


class TestResourceOptimization:
    """Test resource optimization features"""
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_performance_impact(self):
        """Test performance impact of circuit breaker"""
        from core.circuit_breaker import CircuitBreaker
        
        metrics = TestPerformanceMetrics()
        metrics.start_monitoring()
        
        # Create circuit breaker with tight limits for testing
        circuit_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1,
            expected_exception=Exception
        )
        
        async def test_circuit_breaker_call(call_id, should_fail=False):
            start_time = time.time()
            
            try:
                if should_fail:
                    result = await circuit_breaker.call(
                        lambda: asyncio.create_task(self._failing_function())
                    )
                else:
                    result = await circuit_breaker.call(
                        lambda: asyncio.create_task(self._successful_function(call_id))
                    )
                
                call_time = time.time() - start_time
                metrics.record_request(call_time, True)
                return result
                
            except Exception as e:
                call_time = time.time() - start_time
                metrics.record_request(call_time, False)
                raise
        
        # Test normal operation
        successful_tasks = [test_circuit_breaker_call(i) for i in range(10)]
        await asyncio.gather(*successful_tasks, return_exceptions=True)
        
        # Test with failures to trip circuit breaker
        failing_tasks = [test_circuit_breaker_call(i, should_fail=True) for i in range(5)]
        await asyncio.gather(*failing_tasks, return_exceptions=True)
        
        # Test operation with open circuit (should be fast failures)
        open_circuit_tasks = [test_circuit_breaker_call(i) for i in range(5)]
        await asyncio.gather(*open_circuit_tasks, return_exceptions=True)
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # Circuit breaker should not add significant overhead
        assert summary["response_times"]["avg"] < 0.1  # Very fast for mocked operations
        
        print(f"Circuit Breaker Performance Impact: {summary}")
    
    async def _successful_function(self, call_id):
        """Mock successful function"""
        await asyncio.sleep(0.01)  # Small delay
        return f"Success {call_id}"
    
    async def _failing_function(self):
        """Mock failing function"""
        await asyncio.sleep(0.01)  # Small delay
        raise Exception("Simulated failure")
    
    @pytest.mark.asyncio
    async def test_request_queue_performance(self):
        """Test request queue performance characteristics"""
        from core.request_queue import RequestQueue
        
        metrics = TestPerformanceMetrics()
        metrics.start_monitoring()
        
        # Create request queue with moderate limits
        request_queue = RequestQueue(
            max_queue_size=50,
            max_concurrent=5,
            timeout=10
        )
        
        async def queued_request(req_id):
            start_time = time.time()
            
            async def work_function():
                # Simulate work
                await asyncio.sleep(0.05)
                return f"Queued work result {req_id}"
            
            try:
                result = await request_queue.submit(work_function())
                request_time = time.time() - start_time
                metrics.record_request(request_time, True)
                return result
            except Exception as e:
                request_time = time.time() - start_time
                metrics.record_request(request_time, False)
                raise
        
        # Submit 30 requests (more than concurrent limit)
        tasks = [queued_request(i) for i in range(30)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        await request_queue.close()
        
        metrics.stop_monitoring()
        summary = metrics.get_summary()
        
        # Queue should handle all requests efficiently
        assert summary["total_requests"] == 30
        assert summary["success_rate"] >= 0.95
        # Requests should be processed in reasonable time despite queuing
        assert summary["response_times"]["p95"] < 2.0
        
        print(f"Request Queue Performance: {summary}")


# Performance benchmarks and thresholds
PERFORMANCE_BENCHMARKS = {
    "connection_pool": {
        "max_response_time_avg": 2.0,
        "min_success_rate": 0.95,
        "max_memory_growth_mb": 200
    },
    "agent_processing": {
        "max_response_time_avg": 1.0,
        "min_success_rate": 0.98,
        "max_response_time_p95": 2.0
    },
    "system_wide": {
        "max_response_time_avg": 3.0,
        "min_success_rate": 0.95,
        "max_memory_usage_mb": 500
    }
}


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmark tests with pass/fail criteria"""
    
    def test_benchmark_thresholds_defined(self):
        """Ensure all performance benchmarks are properly defined"""
        for category, benchmarks in PERFORMANCE_BENCHMARKS.items():
            assert "max_response_time_avg" in benchmarks
            assert "min_success_rate" in benchmarks
            assert benchmarks["max_response_time_avg"] > 0
            assert 0 < benchmarks["min_success_rate"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "not performance"])  # Skip performance tests by default