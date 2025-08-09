#!/usr/bin/env python3
"""
Comprehensive failure scenario tests for Ollama integration
Tests resilience, error handling, recovery, and fault tolerance
"""

import pytest
import asyncio
import httpx
import sys
import os
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime, timedelta
import json
import time

# Add the agents directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'agents'))

from agents.core.base_agent import BaseAgentV2, AgentStatus
from core.ollama_pool import OllamaConnectionPool, ConnectionState
from core.ollama_integration import OllamaIntegration
from core.circuit_breaker import CircuitBreaker, CircuitBreakerState


class TestOllamaServiceFailures:
    """Test failures related to Ollama service unavailability"""
    
    @pytest.fixture
    def ollama_integration(self):
        return OllamaIntegration(base_url="http://test-ollama:10104")
    
    @pytest.mark.asyncio
    async def test_ollama_service_down(self, ollama_integration):
        """Test behavior when Ollama service is completely down"""
        # Mock connection error
        with patch.object(ollama_integration.client, 'get', side_effect=httpx.ConnectError("Connection refused")):
            result = await ollama_integration.ensure_model_available("tinyllama")
            assert result is False
        
        # Generation should fail gracefully
        with patch.object(ollama_integration.client, 'post', side_effect=httpx.ConnectError("Connection refused")):
            result = await ollama_integration.generate("test prompt")
            assert result is None
    
    @pytest.mark.asyncio
    async def test_ollama_service_timeout(self, ollama_integration):
        """Test behavior when Ollama service times out"""
        # Mock timeout error
        with patch.object(ollama_integration.client, 'post', side_effect=httpx.TimeoutException("Request timeout")):
            result = await ollama_integration.generate("test prompt")
            assert result is None
        
        # Chat should also handle timeouts
        messages = [{"role": "user", "content": "Hello"}]
        with patch.object(ollama_integration.client, 'post', side_effect=httpx.TimeoutException("Request timeout")):
            result = await ollama_integration.chat(messages)
            assert result is None
    
    @pytest.mark.asyncio
    async def test_ollama_service_intermittent_failures(self, ollama_integration):
        """Test behavior with intermittent Ollama service failures"""
        call_count = 0
        
        def intermittent_failure(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            
            if call_count % 3 == 0:  # Fail every 3rd call
                raise httpx.ConnectError("Intermittent failure")
            
            # Success case
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": f"Success on attempt {call_count}"}
            return mock_response
        
        with patch.object(ollama_integration, 'ensure_model_available', return_value=True):
            with patch.object(ollama_integration.client, 'post', side_effect=intermittent_failure):
                
                # Multiple attempts should show mixed results
                results = []
                for i in range(9):
                    result = await ollama_integration.generate(f"test prompt {i}")
                    results.append(result is not None)
                
                # Should have some successes and some failures
                successes = sum(results)
                failures = len(results) - successes
                
                assert successes > 0, "Should have some successful requests"
                assert failures > 0, "Should have some failed requests"
    
    @pytest.mark.asyncio
    async def test_ollama_partial_model_availability(self, ollama_integration):
        """Test behavior when only some models are available"""
        # Mock model list with only some models available
        def mock_model_response(*args, **kwargs):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "models": [
                    {"name": "tinyllama:latest"},
                    # tinyllama2.5-coder and tinyllama-r1 missing
                ]
            }
            return mock_response
        
        # Mock failed model pull
        def mock_failed_pull(*args, **kwargs):
            mock_response = Mock()
            mock_response.status_code = 404
            return mock_response
        
        with patch.object(ollama_integration.client, 'get', side_effect=mock_model_response):
            with patch.object(ollama_integration.client, 'post', side_effect=mock_failed_pull):
                
                # Available model should work
                result = await ollama_integration.ensure_model_available("tinyllama")
                assert result is True
                
                # Unavailable model should fail
                result = await ollama_integration.ensure_model_available("tinyllama2.5-coder:7b")
                assert result is False


class TestConnectionPoolFailures:
    """Test connection pool failure scenarios"""
    
    @pytest.fixture
    def connection_pool(self):
        return OllamaConnectionPool(
            max_connections=3,
            min_connections=1,
            default_model="tinyllama",
            connection_timeout=5,
            request_timeout=10
        )
    
    @pytest.mark.asyncio
    async def test_connection_pool_all_connections_fail(self, connection_pool):
        """Test behavior when all connections in pool fail"""
        # Create failing connections
        for i in range(3):
            mock_client = AsyncMock()
            mock_client.get.side_effect = httpx.ConnectError("Connection failed")
            
            from core.ollama_pool import PoolConnection, ConnectionState
            connection = PoolConnection(client=mock_client, state=ConnectionState.ERROR)
            connection_pool._connections.append(connection)
        
        # Health check should fail
        result = await connection_pool.health_check()
        assert result is False
        
        # Generation should fail
        result = await connection_pool.generate("test prompt")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_connection_pool_partial_failures(self, connection_pool):
        """Test connection pool with some failed connections"""
        from core.ollama_pool import PoolConnection, ConnectionState
        
        # Create mix of working and failing connections
        # Working connection
        working_client = AsyncMock()
        working_response = Mock()
        working_response.status_code = 200
        working_response.json.return_value = {"response": "Working connection response"}
        working_client.post.return_value = working_response
        working_client.get.return_value = working_response
        
        working_conn = PoolConnection(client=working_client, state=ConnectionState.IDLE)
        connection_pool._connections.append(working_conn)
        
        # Failing connection
        failing_client = AsyncMock()
        failing_client.get.side_effect = httpx.ConnectError("Connection failed")
        
        failing_conn = PoolConnection(client=failing_client, state=ConnectionState.ERROR)
        connection_pool._connections.append(failing_conn)
        
        # Pool should use working connection
        with patch.object(connection_pool, '_ensure_model_available', return_value=True):
            result = await connection_pool.generate("test prompt")
            assert result == "Working connection response"
    
    @pytest.mark.asyncio
    async def test_connection_pool_recovery_after_failure(self, connection_pool):
        """Test connection pool recovery after failures"""
        from core.ollama_pool import PoolConnection, ConnectionState
        
        # Start with a failing connection
        failing_client = AsyncMock()
        failing_client.get.side_effect = httpx.ConnectError("Initial failure")
        
        failing_conn = PoolConnection(client=failing_client, state=ConnectionState.ERROR)
        connection_pool._connections = [failing_conn]
        
        # Simulate recovery by creating new working connection
        working_client = AsyncMock()
        working_response = Mock()
        working_response.status_code = 200
        working_response.json.return_value = {"response": "Recovered response"}
        working_client.get.return_value = working_response
        working_client.post.return_value = working_response
        
        # Mock connection creation to return working connection
        async def mock_create_connection():
            return PoolConnection(client=working_client, state=ConnectionState.IDLE)
        
        with patch.object(connection_pool, '_create_connection', side_effect=mock_create_connection):
            with patch.object(connection_pool, '_ensure_model_available', return_value=True):
                result = await connection_pool.generate("test prompt")
                assert result == "Recovered response"
    
    @pytest.mark.asyncio
    async def test_connection_pool_cleanup_failed_connections(self, connection_pool):
        """Test that failed connections are properly cleaned up"""
        from core.ollama_pool import PoolConnection, ConnectionState
        
        # Add several failed connections
        for i in range(5):
            mock_client = AsyncMock()
            failed_conn = PoolConnection(client=mock_client, state=ConnectionState.ERROR)
            connection_pool._connections.append(failed_conn)
        
        connection_pool.stats.total_connections = 5
        
        # Run cleanup
        await connection_pool._cleanup_connections()
        
        # All failed connections should be removed
        assert len(connection_pool._connections) == 0
        assert connection_pool.stats.total_connections == 0


class TestCircuitBreakerFailures:
    """Test circuit breaker failure handling"""
    
    @pytest.fixture
    def circuit_breaker(self):
        return CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1,
            expected_exception=Exception
        )
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_on_consecutive_failures(self, circuit_breaker):
        """Test circuit breaker opens after consecutive failures"""
        async def failing_function():
            raise Exception("Simulated failure")
        
        # First few failures should be allowed through
        for i in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(failing_function)
        
        # Circuit should now be open
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Further calls should fail immediately without calling function
        start_time = time.time()
        with pytest.raises(Exception):
            await circuit_breaker.call(failing_function)
        call_time = time.time() - start_time
        
        # Should be very fast (immediate failure)
        assert call_time < 0.01
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_half_open_recovery(self, circuit_breaker):
        """Test circuit breaker recovery through half-open state"""
        async def initially_failing_then_working():
            if not hasattr(initially_failing_then_working, 'call_count'):
                initially_failing_then_working.call_count = 0
            
            initially_failing_then_working.call_count += 1
            
            # Fail first 3 calls, then succeed
            if initially_failing_then_working.call_count <= 3:
                raise Exception("Initial failure")
            return "Recovery success"
        
        # Trip the circuit breaker
        for i in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(initially_failing_then_working)
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Next call should succeed and close circuit
        result = await circuit_breaker.call(initially_failing_then_working)
        assert result == "Recovery success"
        assert circuit_breaker.state == CircuitBreakerState.CLOSED
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_stays_open_on_continued_failures(self, circuit_breaker):
        """Test circuit breaker stays open if failures continue during recovery"""
        async def always_failing():
            raise Exception("Persistent failure")
        
        # Trip the circuit breaker
        for i in range(3):
            with pytest.raises(Exception):
                await circuit_breaker.call(always_failing)
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN
        
        # Wait for recovery timeout
        await asyncio.sleep(1.1)
        
        # Recovery attempt should fail and keep circuit open
        with pytest.raises(Exception):
            await circuit_breaker.call(always_failing)
        
        assert circuit_breaker.state == CircuitBreakerState.OPEN


class TestAgentFailureScenarios:
    """Test agent-level failure scenarios"""
    
    @pytest.fixture
    def test_agent(self):
        with patch.dict(os.environ, {
            'AGENT_NAME': 'test-failure-agent',
            'AGENT_TYPE': 'failure-test',
            'BACKEND_URL': 'http://test-backend:8000',
            'OLLAMA_URL': 'http://test-ollama:10104'
        }):
            return BaseAgent(max_concurrent_tasks=2)
    
    @pytest.mark.asyncio
    async def test_agent_backend_unavailable(self, test_agent):
        """Test agent behavior when backend is unavailable"""
        await test_agent._setup_async_components()
        
        # Mock backend connection failure
        with patch.object(test_agent.http_client, 'post', side_effect=httpx.ConnectError("Backend unavailable")):
            # Registration should fail
            result = await test_agent.register_with_coordinator()
            assert result is False
            
            # Heartbeat should fail gracefully
            test_agent.shutdown_event.set()  # Exit quickly
            await test_agent.send_heartbeat()  # Should not raise exception
            
            # Task completion reporting should fail gracefully
            from agents.core.base_agent import TaskResult
            task_result = TaskResult(
                task_id="test-task",
                status="completed",
                result={"output": "test"},
                processing_time=1.0
            )
            
            await test_agent.report_task_complete(task_result)  # Should not raise exception
        
        await test_agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_agent_ollama_completely_unavailable(self, test_agent):
        """Test agent behavior when Ollama is completely unavailable"""
        await test_agent._setup_async_components()
        
        # Mock Ollama circuit breaker always failing
        with patch.object(test_agent.circuit_breaker, 'call', side_effect=Exception("Ollama unavailable")):
            # Ollama queries should return None
            result = await test_agent.query_ollama("test prompt")
            assert result is None
            assert test_agent.metrics.ollama_failures == 1
            
            # Chat queries should also return None
            messages = [{"role": "user", "content": "Hello"}]
            result = await test_agent.query_ollama_chat(messages)
            assert result is None
            assert test_agent.metrics.ollama_failures == 2
        
        await test_agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_agent_task_processing_exception(self, test_agent):
        """Test agent behavior when task processing throws exceptions"""
        # Override process_task to throw exception
        original_process_task = test_agent.process_task
        
        async def failing_process_task(task):
            if task.get("id") == "failing-task":
                raise ValueError("Task processing failed")
            return await original_process_task(task)
        
        test_agent.process_task = failing_process_task
        
        # Test task that should fail
        failing_task = {
            "id": "failing-task",
            "type": "test",
            "data": {"prompt": "test"}
        }
        
        result = await test_agent.process_task(failing_task)
        
        assert result.status == "failed"
        assert result.error is not None
        assert "Task processing failed" in result.error
    
    @pytest.mark.asyncio
    async def test_agent_health_check_failures(self, test_agent):
        """Test agent health check with various failure conditions"""
        await test_agent._setup_async_components()
        
        # Mock Ollama pool health check failure
        with patch.object(test_agent.ollama_pool, 'health_check', return_value=False):
            health_status = await test_agent.health_check()
            
            assert health_status["healthy"] is False
            assert health_status["ollama_healthy"] is False
        
        # Mock backend health check failure
        with patch.object(test_agent.ollama_pool, 'health_check', return_value=True):
            mock_response = Mock()
            mock_response.status_code = 500
            
            with patch.object(test_agent.http_client, 'get', return_value=mock_response):
                health_status = await test_agent.health_check()
                
                assert health_status["healthy"] is False
                assert health_status["backend_healthy"] is False
        
        # Mock health check exception
        with patch.object(test_agent.ollama_pool, 'health_check', side_effect=Exception("Health check error")):
            health_status = await test_agent.health_check()
            
            assert health_status["healthy"] is False
            assert "error" in health_status
        
        await test_agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_agent_graceful_shutdown_with_active_tasks(self, test_agent):
        """Test agent graceful shutdown when tasks are still active"""
        await test_agent._setup_async_components()
        
        # Mock long-running task
        async def long_running_task(task):
            await asyncio.sleep(2)  # Simulate long processing
            return await test_agent.process_task(task)
        
        original_process_task = test_agent.process_task
        test_agent.process_task = long_running_task
        
        # Start a task
        task = {"id": "long-task", "type": "test", "data": {}}
        task_coroutine = test_agent._task_wrapper(task)
        task_future = asyncio.create_task(task_coroutine)
        
        # Give task time to start
        await asyncio.sleep(0.1)
        
        # Verify task is active
        assert "long-task" in test_agent.active_tasks
        
        # Trigger shutdown
        test_agent.shutdown_event.set()
        
        # Wait a bit then check active tasks
        await asyncio.sleep(0.5)
        assert len(test_agent.active_tasks) > 0  # Task should still be running
        
        # Wait for task to complete or timeout
        try:
            await asyncio.wait_for(task_future, timeout=3)
        except asyncio.TimeoutError:
            pass
        
        await test_agent._cleanup_async_components()


class TestNetworkFailures:
    """Test various network failure scenarios"""
    
    @pytest.mark.asyncio
    async def test_network_partition_scenario(self):
        """Test behavior during network partition (agent can't reach services)"""
        agent = BaseAgent()
        await agent._setup_async_components()
        
        # Simulate network partition - all network calls fail
        async def network_partition_side_effect(*args, **kwargs):
            raise httpx.ConnectError("Network unreachable")
        
        with patch.object(agent.http_client, 'get', side_effect=network_partition_side_effect):
            with patch.object(agent.http_client, 'post', side_effect=network_partition_side_effect):
                with patch.object(agent.circuit_breaker, 'call', side_effect=Exception("Network unreachable")):
                    
                    # All operations should fail gracefully
                    registration = await agent.register_with_coordinator()
                    assert registration is False
                    
                    task = await agent.get_next_task()
                    assert task is None
                    
                    health = await agent.health_check()
                    assert health["healthy"] is False
                    
                    ollama_result = await agent.query_ollama("test")
                    assert ollama_result is None
        
        await agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_slow_network_scenario(self):
        """Test behavior with very slow network responses"""
        agent = BaseAgent()
        await agent._setup_async_components()
        
        async def slow_network_response(*args, **kwargs):
            await asyncio.sleep(2)  # Simulate slow response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": "Slow response"}
            return mock_response
        
        # Test with slow backend responses
        with patch.object(agent.http_client, 'post', side_effect=slow_network_response):
            start_time = time.time()
            
            # Should handle slow responses
            result = await agent.register_with_coordinator()
            
            response_time = time.time() - start_time
            assert response_time >= 2.0  # Should have waited for slow response
            assert result is True  # But still succeed
        
        await agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_dns_resolution_failure(self):
        """Test behavior when DNS resolution fails"""
        # Create agent with invalid hostname
        with patch.dict(os.environ, {
            'BACKEND_URL': 'http://nonexistent-backend-host:8000',
            'OLLAMA_URL': 'http://nonexistent-ollama-host:10104'
        }):
            agent = BaseAgent()
            await agent._setup_async_components()
            
            # DNS failures should be handled gracefully
            registration = await agent.register_with_coordinator()
            assert registration is False
            
            health = await agent.health_check()
            assert health["healthy"] is False
            
            await agent._cleanup_async_components()


class TestResourceExhaustionScenarios:
    """Test behavior under resource exhaustion"""
    
    @pytest.mark.asyncio
    async def test_memory_pressure_scenario(self):
        """Test behavior under memory pressure (simulated)"""
        agent = BaseAgent()
        await agent._setup_async_components()
        
        # Simulate memory pressure by creating large objects
        memory_hogs = []
        
        async def memory_intensive_task(task):
            # Simulate memory allocation
            memory_hogs.append(b'x' * 1024 * 1024)  # 1MB allocation
            
            try:
                return await agent.process_task(task)
            finally:
                # Clean up to prevent actual memory issues
                if memory_hogs:
                    memory_hogs.pop()
        
        original_process_task = agent.process_task
        agent.process_task = memory_intensive_task
        
        # Process multiple tasks
        tasks = [
            {"id": f"memory-task-{i}", "type": "test", "data": {}}
            for i in range(10)
        ]
        
        results = []
        for task in tasks:
            result = await agent.process_task(task)
            results.append(result)
        
        # All tasks should complete despite memory pressure
        assert all(r.status == "completed" for r in results)
        
        await agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_connection_exhaustion_scenario(self):
        """Test behavior when connection limits are reached"""
        pool = OllamaConnectionPool(
            max_connections=2,  # Very low limit
            min_connections=1,
            default_model="tinyllama"
        )
        
        # Create concurrent requests that exceed connection limit
        async def concurrent_request(req_id):
            # Mock successful response
            with patch.object(pool, '_ensure_model_available', return_value=True):
                mock_client = AsyncMock()
                mock_response = Mock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"response": f"Response {req_id}"}
                
                from core.ollama_pool import PoolConnection
                if len(pool._connections) < pool.max_connections:
                    connection = PoolConnection(client=mock_client)
                    pool._connections.append(connection)
                
                with patch.object(mock_client, 'post', return_value=mock_response):
                    return await pool.generate(f"Request {req_id}")
        
        # Submit more requests than connection limit
        tasks = [concurrent_request(i) for i in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Some requests should succeed, others may fail gracefully
        successful_results = [r for r in results if isinstance(r, str)]
        failed_results = [r for r in results if isinstance(r, Exception) or r is None]
        
        # At least some should succeed
        assert len(successful_results) > 0
        
        await pool.close()


class TestDataCorruptionScenarios:
    """Test handling of corrupted or malformed data"""
    
    @pytest.mark.asyncio
    async def test_malformed_json_responses(self):
        """Test handling of malformed JSON responses from services"""
        agent = BaseAgent()
        await agent._setup_async_components()
        
        # Mock malformed JSON response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.text = "Invalid JSON response"
        
        with patch.object(agent.http_client, 'get', return_value=mock_response):
            task = await agent.get_next_task()
            assert task is None  # Should handle gracefully
        
        await agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_corrupted_configuration_data(self):
        """Test behavior with corrupted configuration"""
        # Create corrupted config file
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{"invalid": json content}')  # Invalid JSON
            temp_path = f.name
        
        try:
            # Agent should handle corrupted config gracefully
            agent = BaseAgent(config_path=temp_path)
            
            # Should fall back to default config
            assert agent.config["capabilities"] == []
            assert agent.config["max_retries"] == 3
            
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_incomplete_task_data(self):
        """Test handling of incomplete or malformed task data"""
        agent = BaseAgent()
        
        # Test with missing required fields
        incomplete_tasks = [
            {},  # Empty task
            {"id": "test"},  # Missing type and data
            {"type": "test"},  # Missing id and data
            {"id": "test", "type": "test"},  # Missing data
            {"id": "test", "type": "test", "data": None},  # Null data
        ]
        
        for task in incomplete_tasks:
            # Should handle incomplete tasks gracefully
            result = await agent.process_task(task)
            
            # Should still return a valid TaskResult
            assert result.task_id is not None
            assert result.status in ["completed", "failed"]
            assert isinstance(result.result, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])