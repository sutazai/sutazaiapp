#!/usr/bin/env python3
"""
Comprehensive unit tests for BaseAgentV2
Tests all aspects of the enhanced base agent implementation
"""

import pytest
import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime, timedelta
import tempfile
import httpx

# Add the agents directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'agents'))

from core.base_agent_v2 import (
    BaseAgentV2, AgentStatus, AgentMetrics, TaskResult,
    BaseAgent  # Backward compatibility alias
)


class TestBaseAgentV2:
    """Test suite for BaseAgentV2 class"""
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary config file for testing"""
        config_data = {
            "capabilities": ["test", "example"],
            "max_retries": 2,
            "timeout": 60,
            "batch_size": 5
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name
        
        yield temp_path
        
        # Cleanup
        os.unlink(temp_path)
    
    @pytest.fixture
    def agent(self, temp_config):
        """Create BaseAgentV2 instance for testing"""
        with patch.dict(os.environ, {
            'AGENT_NAME': 'test-agent',
            'AGENT_TYPE': 'testing',
            'BACKEND_URL': 'http://test-backend:8000',
            'OLLAMA_URL': 'http://test-ollama:10104'
        }):
            return BaseAgentV2(
                config_path=temp_config,
                max_concurrent_tasks=2,
                max_ollama_connections=1,
                health_check_interval=10,
                heartbeat_interval=10
            )
    
    def test_initialization(self, agent):
        """Test agent initialization"""
        assert agent.agent_name == 'test-agent'
        assert agent.agent_type == 'testing'
        assert agent.agent_version == "2.0.0"
        assert agent.backend_url == 'http://test-backend:8000'
        assert agent.ollama_url == 'http://test-ollama:10104'
        assert agent.status == AgentStatus.INITIALIZING
        assert agent.max_concurrent_tasks == 2
        assert isinstance(agent.metrics, AgentMetrics)
    
    def test_load_config_success(self, temp_config):
        """Test successful config loading"""
        agent = BaseAgentV2(config_path=temp_config)
        assert agent.config["capabilities"] == ["test", "example"]
        assert agent.config["max_retries"] == 2
        assert agent.config["timeout"] == 60
        assert agent.config["batch_size"] == 5
    
    def test_load_config_missing_file(self):
        """Test config loading with missing file"""
        agent = BaseAgentV2(config_path="/nonexistent/config.json")
        # Should load default config
        assert agent.config["capabilities"] == []
        assert agent.config["max_retries"] == 3
        assert agent.config["timeout"] == 300
        assert agent.config["batch_size"] == 10
    
    def test_load_config_invalid_json(self):
        """Test config loading with invalid JSON"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name
        
        try:
            agent = BaseAgentV2(config_path=temp_path)
            # Should load default config
            assert agent.config["capabilities"] == []
        finally:
            os.unlink(temp_path)
    
    @pytest.mark.asyncio
    async def test_setup_async_components(self, agent):
        """Test async components setup"""
        await agent._setup_async_components()
        
        assert agent.http_client is not None
        assert agent.ollama_pool is not None
        assert agent.circuit_breaker is not None
        assert agent.request_queue is not None
        
        # Cleanup
        await agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_cleanup_async_components(self, agent):
        """Test async components cleanup"""
        await agent._setup_async_components()
        await agent._cleanup_async_components()
        
        # Components should be cleaned up but not None (for safety)
        assert agent.http_client is not None
        assert agent.ollama_pool is not None
    
    @pytest.mark.asyncio
    async def test_register_with_coordinator_success(self, agent):
        """Test successful registration with coordinator"""
        await agent._setup_async_components()
        
        mock_response = Mock()
        mock_response.status_code = 200
        
        with patch.object(agent.http_client, 'post', return_value=mock_response):
            result = await agent.register_with_coordinator()
            assert result is True
            assert agent.status == AgentStatus.REGISTERING
        
        await agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_register_with_coordinator_failure(self, agent):
        """Test failed registration with coordinator"""
        await agent._setup_async_components()
        
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        
        with patch.object(agent.http_client, 'post', return_value=mock_response):
            result = await agent.register_with_coordinator()
            assert result is False
        
        await agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_register_with_coordinator_exception(self, agent):
        """Test registration with network exception"""
        await agent._setup_async_components()
        
        with patch.object(agent.http_client, 'post', side_effect=httpx.ConnectError("Connection failed")):
            result = await agent.register_with_coordinator()
            assert result is False
        
        await agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_send_heartbeat_success(self, agent):
        """Test successful heartbeat sending"""
        await agent._setup_async_components()
        
        mock_response = Mock()
        mock_response.status_code = 200
        
        with patch.object(agent.http_client, 'post', return_value=mock_response):
            # Set shutdown event to exit heartbeat loop quickly
            agent.shutdown_event.set()
            await agent.send_heartbeat()
        
        await agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_send_heartbeat_failure(self, agent):
        """Test heartbeat sending with failure"""
        await agent._setup_async_components()
        
        mock_response = Mock()
        mock_response.status_code = 500
        
        with patch.object(agent.http_client, 'post', return_value=mock_response):
            agent.shutdown_event.set()
            await agent.send_heartbeat()
        
        await agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_get_next_task_success(self, agent):
        """Test successful task retrieval"""
        await agent._setup_async_components()
        
        task_data = {
            "id": "test-task-123",
            "type": "test",
            "data": {"prompt": "test prompt"}
        }
        
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = task_data
        
        with patch.object(agent.http_client, 'get', return_value=mock_response):
            task = await agent.get_next_task()
            assert task == task_data
            assert agent.metrics.tasks_queued == 1
        
        await agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_get_next_task_no_tasks(self, agent):
        """Test task retrieval when no tasks available"""
        await agent._setup_async_components()
        
        mock_response = Mock()
        mock_response.status_code = 204
        
        with patch.object(agent.http_client, 'get', return_value=mock_response):
            task = await agent.get_next_task()
            assert task is None
        
        await agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_get_next_task_error(self, agent):
        """Test task retrieval with error"""
        await agent._setup_async_components()
        
        with patch.object(agent.http_client, 'get', side_effect=httpx.ConnectError("Connection failed")):
            task = await agent.get_next_task()
            assert task is None
        
        await agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_report_task_complete_success(self, agent):
        """Test successful task completion reporting"""
        await agent._setup_async_components()
        
        task_result = TaskResult(
            task_id="test-task-123",
            status="completed",
            result={"output": "test output"},
            processing_time=1.5
        )
        
        mock_response = Mock()
        mock_response.status_code = 200
        
        with patch.object(agent.http_client, 'post', return_value=mock_response):
            await agent.report_task_complete(task_result)
        
        await agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_report_task_complete_failure(self, agent):
        """Test task completion reporting with failure"""
        await agent._setup_async_components()
        
        task_result = TaskResult(
            task_id="test-task-123",
            status="failed",
            result={"error": "test error"},
            processing_time=0.5,
            error="test error"
        )
        
        mock_response = Mock()
        mock_response.status_code = 500
        
        with patch.object(agent.http_client, 'post', return_value=mock_response):
            await agent.report_task_complete(task_result)
        
        await agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_process_task_success(self, agent):
        """Test successful task processing"""
        task = {
            "id": "test-task-123",
            "type": "test",
            "data": {"prompt": "test prompt"}
        }
        
        result = await agent.process_task(task)
        
        assert result.task_id == "test-task-123"
        assert result.status == "completed"
        assert result.result["status"] == "success"
        assert result.result["agent_name"] == agent.agent_name
        assert result.processing_time > 0
    
    @pytest.mark.asyncio
    async def test_process_task_exception(self, agent):
        """Test task processing with exception"""
        task = {
            "id": "test-task-123",
            "type": "test",
            "data": {"prompt": "test prompt"}
        }
        
        # Mock process_task to raise exception
        with patch.object(agent, 'process_task') as mock_process:
            mock_process.side_effect = Exception("Processing error")
            
            # Create new agent to test the actual exception handling
            test_agent = BaseAgentV2()
            result = await test_agent.process_task(task)
            
            assert result.task_id == "test-task-123"
            assert result.status == "failed"
            assert "error" in result.result
            assert result.error is not None
    
    @pytest.mark.asyncio
    async def test_query_ollama_success(self, agent):
        """Test successful Ollama query"""
        await agent._setup_async_components()
        
        # Mock circuit breaker call
        with patch.object(agent.circuit_breaker, 'call', return_value="Ollama response"):
            result = await agent.query_ollama("test prompt")
            assert result == "Ollama response"
            assert agent.metrics.ollama_requests == 1
        
        await agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_query_ollama_failure(self, agent):
        """Test Ollama query with failure"""
        await agent._setup_async_components()
        
        with patch.object(agent.circuit_breaker, 'call', side_effect=Exception("Ollama error")):
            result = await agent.query_ollama("test prompt")
            assert result is None
            assert agent.metrics.ollama_failures == 1
        
        await agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_query_ollama_chat_success(self, agent):
        """Test successful Ollama chat query"""
        await agent._setup_async_components()
        
        messages = [{"role": "user", "content": "Hello"}]
        
        with patch.object(agent.circuit_breaker, 'call', return_value="Chat response"):
            result = await agent.query_ollama_chat(messages)
            assert result == "Chat response"
            assert agent.metrics.ollama_requests == 1
        
        await agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_query_ollama_chat_failure(self, agent):
        """Test Ollama chat query with failure"""
        await agent._setup_async_components()
        
        messages = [{"role": "user", "content": "Hello"}]
        
        with patch.object(agent.circuit_breaker, 'call', side_effect=Exception("Chat error")):
            result = await agent.query_ollama_chat(messages)
            assert result is None
            assert agent.metrics.ollama_failures == 1
        
        await agent._cleanup_async_components()
    
    def test_update_metrics_completed(self, agent):
        """Test metrics update for completed task"""
        task_result = TaskResult(
            task_id="test-task",
            status="completed",
            result={"output": "test"},
            processing_time=2.0
        )
        
        initial_processed = agent.metrics.tasks_processed
        initial_total_time = agent.metrics.total_processing_time
        
        asyncio.run(agent._update_metrics(task_result))
        
        assert agent.metrics.tasks_processed == initial_processed + 1
        assert agent.metrics.total_processing_time == initial_total_time + 2.0
        assert agent.metrics.avg_processing_time == agent.metrics.total_processing_time / agent.metrics.tasks_processed
    
    def test_update_metrics_failed(self, agent):
        """Test metrics update for failed task"""
        task_result = TaskResult(
            task_id="test-task",
            status="failed",
            result={"error": "test error"},
            processing_time=1.0,
            error="test error"
        )
        
        initial_failed = agent.metrics.tasks_failed
        
        asyncio.run(agent._update_metrics(task_result))
        
        assert agent.metrics.tasks_failed == initial_failed + 1
    
    @pytest.mark.asyncio
    async def test_task_wrapper_success(self, agent):
        """Test task wrapper with successful processing"""
        await agent._setup_async_components()
        
        task = {"id": "test-task", "type": "test"}
        
        # Mock successful processing
        mock_result = TaskResult(
            task_id="test-task",
            status="completed",
            result={"output": "success"},
            processing_time=1.0
        )
        
        with patch.object(agent, 'process_task', return_value=mock_result), \
             patch.object(agent, 'report_task_complete') as mock_report:
            
            await agent._task_wrapper(task)
            
            # Verify task was reported
            mock_report.assert_called_once_with(mock_result)
            
            # Verify status changes
            assert agent.status == AgentStatus.ACTIVE  # Should return to active after completion
        
        await agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_task_wrapper_exception(self, agent):
        """Test task wrapper with processing exception"""
        await agent._setup_async_components()
        
        task = {"id": "test-task", "type": "test"}
        
        with patch.object(agent, 'process_task', side_effect=Exception("Processing error")), \
             patch.object(agent, 'report_task_complete') as mock_report:
            
            await agent._task_wrapper(task)
            
            # Verify error was reported
            mock_report.assert_called_once()
            call_args = mock_report.call_args[0][0]
            assert call_args.status == "failed"
            assert call_args.error == "Processing error"
        
        await agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, agent):
        """Test successful health check"""
        await agent._setup_async_components()
        
        # Mock healthy ollama pool
        with patch.object(agent.ollama_pool, 'health_check', return_value=True):
            # Mock healthy backend
            mock_response = Mock()
            mock_response.status_code = 200
            
            with patch.object(agent.http_client, 'get', return_value=mock_response):
                health_status = await agent.health_check()
                
                assert health_status["healthy"] is True
                assert health_status["agent_name"] == agent.agent_name
                assert health_status["ollama_healthy"] is True
                assert health_status["backend_healthy"] is True
        
        await agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_health_check_ollama_unhealthy(self, agent):
        """Test health check with unhealthy Ollama"""
        await agent._setup_async_components()
        
        with patch.object(agent.ollama_pool, 'health_check', return_value=False):
            mock_response = Mock()
            mock_response.status_code = 200
            
            with patch.object(agent.http_client, 'get', return_value=mock_response):
                health_status = await agent.health_check()
                
                assert health_status["healthy"] is False
                assert health_status["ollama_healthy"] is False
                assert health_status["backend_healthy"] is True
        
        await agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_health_check_backend_unhealthy(self, agent):
        """Test health check with unhealthy backend"""
        await agent._setup_async_components()
        
        with patch.object(agent.ollama_pool, 'health_check', return_value=True):
            mock_response = Mock()
            mock_response.status_code = 500
            
            with patch.object(agent.http_client, 'get', return_value=mock_response):
                health_status = await agent.health_check()
                
                assert health_status["healthy"] is False
                assert health_status["ollama_healthy"] is True
                assert health_status["backend_healthy"] is False
        
        await agent._cleanup_async_components()
    
    @pytest.mark.asyncio
    async def test_health_check_exception(self, agent):
        """Test health check with exception"""
        await agent._setup_async_components()
        
        with patch.object(agent.ollama_pool, 'health_check', side_effect=Exception("Health check error")):
            health_status = await agent.health_check()
            
            assert health_status["healthy"] is False
            assert "error" in health_status
        
        await agent._cleanup_async_components()
    
    def test_backward_compatibility_alias(self):
        """Test that BaseAgent alias works"""
        agent = BaseAgent()
        assert isinstance(agent, BaseAgentV2)
    
    def test_query_ollama_sync_not_running(self):
        """Test sync Ollama query when no event loop is running"""
        agent = BaseAgentV2()
        
        # Mock asyncio.get_event_loop() to return a mock that's not running
        mock_loop = Mock()
        mock_loop.is_running.return_value = False
        mock_loop.run_until_complete.return_value = "sync response"
        
        with patch('asyncio.get_event_loop', return_value=mock_loop):
            result = agent.query_ollama_sync("test prompt")
            # This should log a warning and attempt to use run_until_complete
            # In our mock setup, it will return None due to setup requirements
            assert result is None  # Because agent components aren't set up
    
    def test_query_ollama_sync_running_loop(self):
        """Test sync Ollama query when event loop is already running"""
        agent = BaseAgentV2()
        
        # Mock asyncio.get_event_loop() to return a mock that's running
        mock_loop = Mock()
        mock_loop.is_running.return_value = True
        
        with patch('asyncio.get_event_loop', return_value=mock_loop):
            result = agent.query_ollama_sync("test prompt")
            assert result is None  # Should return None when loop is running


class TestAgentStatus:
    """Test AgentStatus enum"""
    
    def test_agent_status_values(self):
        """Test that all agent status values are correct"""
        assert AgentStatus.INITIALIZING.value == "initializing"
        assert AgentStatus.REGISTERING.value == "registering"
        assert AgentStatus.ACTIVE.value == "active"
        assert AgentStatus.BUSY.value == "busy"
        assert AgentStatus.ERROR.value == "error"
        assert AgentStatus.SHUTTING_DOWN.value == "shutting_down"
        assert AgentStatus.STOPPED.value == "stopped"


class TestAgentMetrics:
    """Test AgentMetrics dataclass"""
    
    def test_agent_metrics_defaults(self):
        """Test AgentMetrics default values"""
        metrics = AgentMetrics()
        
        assert metrics.tasks_processed == 0
        assert metrics.tasks_failed == 0
        assert metrics.tasks_queued == 0
        assert metrics.total_processing_time == 0.0
        assert metrics.avg_processing_time == 0.0
        assert metrics.last_task_time is None
        assert metrics.memory_usage_mb == 0.0
        assert metrics.cpu_usage_percent == 0.0
        assert metrics.ollama_requests == 0
        assert metrics.ollama_failures == 0
        assert metrics.circuit_breaker_trips == 0
        assert isinstance(metrics.startup_time, datetime)


class TestTaskResult:
    """Test TaskResult dataclass"""
    
    def test_task_result_creation(self):
        """Test TaskResult creation"""
        result = TaskResult(
            task_id="test-123",
            status="completed",
            result={"output": "test"},
            processing_time=1.5,
            error="test error"
        )
        
        assert result.task_id == "test-123"
        assert result.status == "completed"
        assert result.result == {"output": "test"}
        assert result.processing_time == 1.5
        assert result.error == "test error"
        assert isinstance(result.timestamp, datetime)
    
    def test_task_result_defaults(self):
        """Test TaskResult default values"""
        result = TaskResult(
            task_id="test-123",
            status="completed",
            result={"output": "test"},
            processing_time=1.5
        )
        
        assert result.error is None
        assert isinstance(result.timestamp, datetime)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])