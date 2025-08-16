"""
Unit tests for Redis-based service mesh bus operations.
Tests all functions in backend/app/mesh/redis_bus.py with Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Tested Redis.
"""
import json
import os
import pytest
from unittest.Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test import Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test, patch, MagicRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test
from typing import Dict, Any, List, Tuple

# Test data and fixtures
@pytest.fixture
def Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis():
    """Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test Redis client for unit tests."""
    redis_Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
    redis_Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test.ping.return_value = True
    redis_Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test.set.return_value = True
    redis_Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test.get.return_value = None
    redis_Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test.exists.return_value = False
    redis_Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test.expire.return_value = True
    redis_Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test.xadd.return_value = "1699999999999-0"
    redis_Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test.xrevrange.return_value = []
    redis_Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test.xgroup_create.return_value = True
    redis_Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test.xreadgroup.return_value = []
    redis_Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test.xack.return_value = 1
    redis_Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test.scan.return_value = (0, [])
    # Set up pipeline context manager properly
    pipeline_Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
    pipeline_Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test.__enter__ = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test(return_value=redis_Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test)
    pipeline_Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test.__exit__ = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test(return_value=None)
    redis_Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test.pipeline.return_value = pipeline_Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test
    redis_Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test.execute.return_value = []
    return redis_Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test

@pytest.fixture
def Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis_pool():
    """Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test Redis connection pool."""
    pool_Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
    return pool_Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test

@pytest.fixture
def sample_task_payload():
    """Sample task payload for testing."""
    return {
        "task_id": "test-task-123",
        "task_type": "data_processing", 
        "agent_id": "test-agent",
        "payload": {
            "input_data": "test data",
            "parameters": {"param1": "value1"}
        },
        "priority": "normal",
        "timestamp": "2024-01-01T00:00:00Z"
    }

@pytest.fixture
def sample_agent_data():
    """Sample agent registration data."""
    return {
        "agent_id": "test-agent-123",
        "agent_type": "data_processor",
        "meta": {
            "version": "1.0.0",
            "capabilities": ["text_analysis", "data_transform"],
            "max_concurrent_tasks": 5
        }
    }

class TestRedisConnection:
    """Test Redis connection management."""
    
    @patch.dict(os.environ, {"REDIS_URL": "redis://test:6379/0"})
    def test_redis_url_from_env(self):
        """Test Redis URL configuration from environment."""
        from backend.app.mesh.redis_bus import _redis_url
        assert _redis_url() == "redis://test:6379/0"
    
    def test_redis_url_default(self):
        """Test default Redis URL when environment variable not set."""
        with patch.dict(os.environ, {}, clear=True):
            from backend.app.mesh.redis_bus import _redis_url
            assert _redis_url() == "redis://redis:6379/0"
    
    @patch('backend.app.mesh.redis_bus.redis.ConnectionPool.from_url')
    @patch('backend.app.mesh.redis_bus.redis.Redis')
    def test_get_redis_creates_pool(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis_class, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_pool_from_url, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis_pool):
        """Test Redis connection pool creation."""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_pool_from_url.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis_pool
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis_class.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        
        from backend.app.mesh.redis_bus import get_redis
        
        # Reset global pool
        import backend.app.mesh.redis_bus
        backend.app.mesh.redis_bus._redis_pool = None
        
        client = get_redis()
        
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_pool_from_url.assert_called_once()
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis_class.assert_called_once_with(connection_pool=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis_pool)
        assert client is not None
    
    @patch('backend.app.mesh.redis_bus.redis.ConnectionPool.from_url')
    @patch('backend.app.mesh.redis_bus.redis.Redis')
    def test_get_redis_reuses_pool(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis_class, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_pool_from_url, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis_pool):
        """Test Redis connection pool reuse."""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_pool_from_url.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis_pool
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis_class.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        
        from backend.app.mesh.redis_bus import get_redis
        
        # Reset global pool and call twice
        import backend.app.mesh.redis_bus
        backend.app.mesh.redis_bus._redis_pool = None
        
        client1 = get_redis()
        client2 = get_redis()
        
        # Pool should only be created once
        assert Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_pool_from_url.call_count == 1
        assert Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis_class.call_count == 2  # But client created each time

class TestStreamKeys:
    """Test stream key generation functions."""
    
    def test_task_stream_key(self):
        """Test task stream key generation."""
        from backend.app.mesh.redis_bus import task_stream
        assert task_stream("data_processing") == "stream:tasks:data_processing"
        assert task_stream("ai_inference") == "stream:tasks:ai_inference"
    
    def test_result_stream_key(self):
        """Test result stream key generation."""
        from backend.app.mesh.redis_bus import result_stream
        assert result_stream("data_processing") == "stream:results:data_processing"
        assert result_stream("ai_inference") == "stream:results:ai_inference"
    
    def test_dead_stream_key(self):
        """Test dead letter stream key generation."""
        from backend.app.mesh.redis_bus import dead_stream
        assert dead_stream("data_processing") == "stream:dead:data_processing"
        assert dead_stream("ai_inference") == "stream:dead:ai_inference"
    
    def test_agent_key(self):
        """Test agent registry key generation."""
        from backend.app.mesh.redis_bus import agent_key
        assert agent_key("agent-123") == "mesh:agent:agent-123"
        assert agent_key("data-processor-1") == "mesh:agent:data-processor-1"

class TestAgentRegistry:
    """Test agent registration and management."""
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_register_agent(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis, sample_agent_data):
        """Test agent registration."""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis
        
        from backend.app.mesh.redis_bus import register_agent
        
        register_agent(
            agent_id=sample_agent_data["agent_id"],
            agent_type=sample_agent_data["agent_type"],
            ttl_seconds=120,
            meta=sample_agent_data["meta"]
        )
        
        expected_data = {
            "agent_id": sample_agent_data["agent_id"],
            "agent_type": sample_agent_data["agent_type"],
            "meta": sample_agent_data["meta"]
        }
        
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.set.assert_called_once_with(
            f"mesh:agent:{sample_agent_data['agent_id']}",
            json.dumps(expected_data),
            ex=120
        )
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_register_agent_default_ttl(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis):
        """Test agent registration with default TTL."""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis
        
        from backend.app.mesh.redis_bus import register_agent
        
        register_agent("test-agent", "test-type")
        
        # Should use default TTL of 60 seconds
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.set.assert_called_once()
        args, kwargs = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.set.call_args
        assert kwargs["ex"] == 60
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_heartbeat_agent_exists(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis):
        """Test agent heartbeat when agent exists."""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.exists.return_value = True
        
        from backend.app.mesh.redis_bus import heartbeat_agent
        
        heartbeat_agent("test-agent", 90)
        
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.exists.assert_called_once_with("mesh:agent:test-agent")
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.expire.assert_called_once_with("mesh:agent:test-agent", 90)
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_heartbeat_agent_not_exists(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis):
        """Test agent heartbeat when agent doesn't exist."""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.exists.return_value = False
        
        from backend.app.mesh.redis_bus import heartbeat_agent
        
        heartbeat_agent("test-agent", 90)
        
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.exists.assert_called_once_with("mesh:agent:test-agent")
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.expire.assert_not_called()
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_list_agents_empty(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis):
        """Test listing agents when none are registered."""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.scan.return_value = (0, [])
        
        from backend.app.mesh.redis_bus import list_agents
        
        agents = list_agents()
        
        assert agents == []
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.scan.assert_called_once_with(0, match="mesh:agent:*", count=100)
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_list_agents_with_data(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis, sample_agent_data):
        """Test listing agents with registered agents."""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis
        
        # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test scan to return agent keys
        agent_keys = ["mesh:agent:agent-1", "mesh:agent:agent-2"]
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.scan.return_value = (0, agent_keys)
        
        # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test pipeline execution to return agent data
        agent_data = [json.dumps(sample_agent_data), json.dumps(sample_agent_data)]
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.execute.return_value = agent_data
        
        from backend.app.mesh.redis_bus import list_agents
        
        agents = list_agents()
        
        assert len(agents) == 2
        assert all(agent == sample_agent_data for agent in agents)
        
        # Verify pipeline usage for batch fetch
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.pipeline.assert_called_once()
        assert Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.get.call_count == 2
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_list_agents_with_invalid_json(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis):
        """Test listing agents with some invalid JSON data."""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis
        
        agent_keys = ["mesh:agent:agent-1", "mesh:agent:agent-2"]
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.scan.return_value = (0, agent_keys)
        
        # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test one valid and one invalid JSON
        agent_data = ['{"valid": "json"}', 'invalid json']
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.execute.return_value = agent_data
        
        from backend.app.mesh.redis_bus import list_agents
        
        agents = list_agents()
        
        # Should only return the valid agent
        assert len(agents) == 1
        assert agents[0] == {"valid": "json"}

class TestTaskEnqueuing:
    """Test task enqueuing functionality."""
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_enqueue_task_basic(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis, sample_task_payload):
        """Test basic task enqueuing."""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.xadd.return_value = "1699999999999-0"
        
        from backend.app.mesh.redis_bus import enqueue_task
        
        # Reset stream cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        msg_id = enqueue_task("data_processing", sample_task_payload)
        
        assert msg_id == "1699999999999-0"
        
        # Verify stream creation with consumer group
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.xgroup_create.assert_called_once_with(
            "stream:tasks:data_processing", "default", id="$", mkstream=True
        )
        
        # Verify task addition
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.xadd.assert_called_once_with(
            "stream:tasks:data_processing",
            {"json": json.dumps(sample_task_payload)},
            maxlen=10000,
            approximate=True
        )
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_enqueue_task_group_exists(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis, sample_task_payload):
        """Test task enqueuing when consumer group already exists."""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.xadd.return_value = "1699999999999-1"
        
        # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test group already exists error
        import redis
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.xgroup_create.side_effect = redis.exceptions.ResponseError("BUSYGROUP")
        
        from backend.app.mesh.redis_bus import enqueue_task
        
        # Reset stream cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        msg_id = enqueue_task("data_processing", sample_task_payload)
        
        assert msg_id == "1699999999999-1"
        # Should still succeed despite group creation error
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.xadd.assert_called_once()
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_enqueue_task_custom_maxlen(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis, sample_task_payload):
        """Test task enqueuing with custom maxlen."""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.xadd.return_value = "1699999999999-2"
        
        from backend.app.mesh.redis_bus import enqueue_task
        
        # Reset stream cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        msg_id = enqueue_task("data_processing", sample_task_payload, maxlen=5000)
        
        # Verify custom maxlen is used
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.xadd.assert_called_once_with(
            "stream:tasks:data_processing",
            {"json": json.dumps(sample_task_payload)},
            maxlen=5000,
            approximate=True
        )
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_enqueue_task_caching(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis, sample_task_payload):
        """Test stream caching optimization."""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.xadd.return_value = "1699999999999-0"
        
        from backend.app.mesh.redis_bus import enqueue_task
        
        # Reset stream cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        # First call should create group
        enqueue_task("data_processing", sample_task_payload)
        assert Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.xgroup_create.call_count == 1
        
        # Second call should not create group (cached)
        enqueue_task("data_processing", sample_task_payload)
        assert Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.xgroup_create.call_count == 1  # Still 1

class TestResultTailing:
    """Test result tailing functionality."""
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_tail_results_empty(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis):
        """Test tailing results when no results exist."""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.execute.return_value = [[]]  # Empty pipeline result
        
        from backend.app.mesh.redis_bus import tail_results
        
        results = tail_results("data_processing")
        
        assert results == []
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.pipeline.assert_called_once()
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.xrevrange.assert_called_once_with("stream:results:data_processing", count=10)
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_tail_results_with_data(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis):
        """Test tailing results with actual data."""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis
        
        # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test result data
        result_data = {"status": "completed", "result": "processed successfully"}
        raw_results = [
            ("1699999999999-0", {"json": json.dumps(result_data)}),
            ("1699999999999-1", {"json": json.dumps(result_data)})
        ]
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.execute.return_value = [raw_results]
        
        from backend.app.mesh.redis_bus import tail_results
        
        results = tail_results("data_processing", count=2)
        
        assert len(results) == 2
        assert results[0] == ("1699999999999-0", result_data)
        assert results[1] == ("1699999999999-1", result_data)
        
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.xrevrange.assert_called_once_with("stream:results:data_processing", count=2)
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_tail_results_invalid_json(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis):
        """Test tailing results with invalid JSON data."""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis
        
        # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test result with invalid JSON
        raw_results = [
            ("1699999999999-0", {"json": "invalid json"}),
            ("1699999999999-1", {"field1": "value1", "field2": "value2"})  # Raw fields
        ]
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.execute.return_value = [raw_results]
        
        from backend.app.mesh.redis_bus import tail_results
        
        results = tail_results("data_processing")
        
        assert len(results) == 2
        # First should fall back to raw fields when JSON parsing fails
        assert results[0] == ("1699999999999-0", {"json": "invalid json"})
        # Second should use empty dict when no "json" key (default "{}" parses to {})
        assert results[1] == ("1699999999999-1", {})

class TestConsumerGroups:
    """Test consumer group management."""
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_create_consumer_group(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis):
        """Test consumer group creation."""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis
        
        from backend.app.mesh.redis_bus import create_consumer_group
        
        create_consumer_group("data_processing", "workers")
        
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.xgroup_create.assert_called_once_with(
            "stream:tasks:data_processing", "workers", id="$", mkstream=True
        )
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_create_consumer_group_already_exists(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis):
        """Test consumer group creation when group already exists."""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis
        
        import redis
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.xgroup_create.side_effect = redis.exceptions.ResponseError("BUSYGROUP Consumer Group name already exists")
        
        from backend.app.mesh.redis_bus import create_consumer_group
        
        # Should not raise exception
        create_consumer_group("data_processing", "workers")
        
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.xgroup_create.assert_called_once()
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_create_consumer_group_other_error(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis):
        """Test consumer group creation with unexpected error."""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis
        
        import redis
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.xgroup_create.side_effect = redis.exceptions.ResponseError("WRONGTYPE Operation against a key holding the wrong kind of value")
        
        from backend.app.mesh.redis_bus import create_consumer_group
        
        # Should raise the exception
        with pytest.raises(redis.exceptions.ResponseError):
            create_consumer_group("data_processing", "workers")
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_read_group_empty(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis):
        """Test reading from consumer group with no messages."""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.xreadgroup.return_value = []
        
        from backend.app.mesh.redis_bus import read_group
        
        messages = read_group("data_processing", "workers", "worker-1")
        
        assert messages == []
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.xreadgroup.assert_called_once_with(
            "workers", "worker-1", {"stream:tasks:data_processing": ">"}, count=1, block=1000
        )
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_read_group_with_messages(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis, sample_task_payload):
        """Test reading from consumer group with messages."""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis
        
        # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test message data
        stream_data = [
            ("stream:tasks:data_processing", [
                ("1699999999999-0", {"json": json.dumps(sample_task_payload)}),
                ("1699999999999-1", {"json": json.dumps(sample_task_payload)})
            ])
        ]
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.xreadgroup.return_value = stream_data
        
        from backend.app.mesh.redis_bus import read_group
        
        messages = read_group("data_processing", "workers", "worker-1", count=2)
        
        assert len(messages) == 2
        assert messages[0] == ("1699999999999-0", sample_task_payload)
        assert messages[1] == ("1699999999999-1", sample_task_payload)

    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_read_group_with_invalid_json(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis):
        """Test reading from consumer group with invalid JSON messages."""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis
        
        # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test message data with invalid JSON
        stream_data = [
            ("stream:tasks:data_processing", [
                ("1699999999999-0", {"json": "invalid json"}),
                ("1699999999999-1", {"field1": "value1", "field2": "value2"})  # No json field
            ])
        ]
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.xreadgroup.return_value = stream_data
        
        from backend.app.mesh.redis_bus import read_group
        
        messages = read_group("data_processing", "workers", "worker-1")
        
        assert len(messages) == 2
        # First should fall back to raw fields when JSON parsing fails
        assert messages[0] == ("1699999999999-0", {"json": "invalid json"})
        # Second should use empty dict when no "json" key (default "{}" parses to {})
        assert messages[1] == ("1699999999999-1", {})
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_ack_message(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis):
        """Test acknowledging processed message."""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.xack.return_value = 1
        
        from backend.app.mesh.redis_bus import ack
        
        result = ack("data_processing", "workers", "1699999999999-0")
        
        assert result == 1
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.xack.assert_called_once_with(
            "stream:tasks:data_processing", "workers", "1699999999999-0"
        )

class TestDeadLetterQueue:
    """Test dead letter queue functionality."""
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_move_to_dead(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis, sample_task_payload):
        """Test moving failed message to dead letter queue."""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.xadd.return_value = "1699999999999-dead-0"
        
        from backend.app.mesh.redis_bus import move_to_dead
        
        dead_msg_id = move_to_dead("data_processing", "1699999999999-0", sample_task_payload)
        
        assert dead_msg_id == "1699999999999-dead-0"
        
        expected_payload = {
            "id": "1699999999999-0",
            "payload": sample_task_payload
        }
        
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.xadd.assert_called_once_with(
            "stream:dead:data_processing",
            {"json": json.dumps(expected_payload)},
            maxlen=10000,
            approximate=True
        )

class TestPerformanceOptimizations:
    """Test performance optimization features."""
    
    def test_connection_pool_settings(self):
        """Test Redis connection pool optimization settings."""
        with patch('backend.app.mesh.redis_bus.redis.ConnectionPool.from_url') as Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_pool:
            with patch('backend.app.mesh.redis_bus.redis.Redis') as Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis_class:
                # Reset global pool to force creation
                import backend.app.mesh.redis_bus
                backend.app.mesh.redis_bus._redis_pool = None
                
                from backend.app.mesh.redis_bus import get_redis
                get_redis()
                
                # Verify optimized pool settings
                Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_pool.assert_called_once()
                args, kwargs = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_pool.call_args
                
                assert kwargs['max_connections'] == 50
                assert kwargs['socket_connect_timeout'] == 5
                assert kwargs['socket_timeout'] == 5
                assert kwargs['socket_keepalive'] is True
                assert kwargs['health_check_interval'] == 30

    def test_get_redis_async_creates_pool(self):
        """Test async Redis connection pool creation."""
        with patch('backend.app.mesh.redis_bus.redis_async.ConnectionPool.from_url') as Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_pool:
            with patch('backend.app.mesh.redis_bus.redis_async.Redis') as Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis_class:
                # Reset global async pool
                import backend.app.mesh.redis_bus
                backend.app.mesh.redis_bus._redis_async_pool = None
                
                import asyncio
                async def test_async():
                    from backend.app.mesh.redis_bus import get_redis_async
                    await get_redis_async()
                    return Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_pool.call_args
                
                # Run async test
                call_args = asyncio.run(test_async())
                
                # Verify async pool settings
                Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_pool.assert_called_once()
                args, kwargs = call_args
                assert kwargs['max_connections'] == 50
                assert kwargs['socket_keepalive'] is True
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_pipeline_usage_in_list_agents(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis):
        """Test pipeline usage for batch operations in list_agents."""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.scan.return_value = (0, ["mesh:agent:1", "mesh:agent:2"])
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.execute.return_value = ['{"id": "1"}', '{"id": "2"}']
        
        from backend.app.mesh.redis_bus import list_agents
        
        list_agents()
        
        # Verify pipeline is used for batch fetching
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.pipeline.assert_called_once()
        assert Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.get.call_count == 2  # Batched get operations
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_pipeline_usage_in_tail_results(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis):
        """Test pipeline usage for batch operations in tail_results."""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.execute.return_value = [[]]
        
        from backend.app.mesh.redis_bus import tail_results
        
        tail_results("data_processing")
        
        # Verify pipeline is used
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.pipeline.assert_called_once()
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.xrevrange.assert_called_once()

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_enqueue_task_redis_error(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis, sample_task_payload):
        """Test task enqueuing with Redis connection error."""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis
        
        import redis
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.xadd.side_effect = redis.exceptions.ConnectionError("Connection failed")
        
        from backend.app.mesh.redis_bus import enqueue_task
        
        # Reset stream cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        with pytest.raises(redis.exceptions.ConnectionError):
            enqueue_task("data_processing", sample_task_payload)
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_list_agents_redis_error(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis):
        """Test listing agents with Redis connection error."""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis
        
        import redis
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.scan.side_effect = redis.exceptions.ConnectionError("Connection failed")
        
        from backend.app.mesh.redis_bus import list_agents
        
        with pytest.raises(redis.exceptions.ConnectionError):
            list_agents()
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_tail_results_redis_error(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis):
        """Test tailing results with Redis connection error."""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis
        
        import redis
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.pipeline.side_effect = redis.exceptions.ConnectionError("Connection failed")
        
        from backend.app.mesh.redis_bus import tail_results
        
        with pytest.raises(redis.exceptions.ConnectionError):
            tail_results("data_processing")
    
    def test_empty_topic_handling(self):
        """Test handling of empty topic strings."""
        from backend.app.mesh.redis_bus import task_stream, result_stream, dead_stream
        
        # Empty topics should still generate valid keys
        assert task_stream("") == "stream:tasks:"
        assert result_stream("") == "stream:results:"
        assert dead_stream("") == "stream:dead:"
    
    def test_special_characters_in_topic(self):
        """Test handling of special characters in topic names."""
        from backend.app.mesh.redis_bus import task_stream, result_stream, dead_stream
        
        topic = "data:processing-v2_test"
        assert task_stream(topic) == "stream:tasks:data:processing-v2_test"
        assert result_stream(topic) == "stream:results:data:processing-v2_test"
        assert dead_stream(topic) == "stream:dead:data:processing-v2_test"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])