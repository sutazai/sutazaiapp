"""
Unit tests for Redis-based service mesh bus operations.
Tests all functions in backend/app/mesh/redis_bus.py with Mocked Redis.
"""
import json
import os
import pytest
from unittest.Mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Tuple

# Test data and fixtures
@pytest.fixture
def Mock_redis():
    """Mock Redis client for unit tests."""
    redis_Mock = Mock()
    redis_Mock.ping.return_value = True
    redis_Mock.set.return_value = True
    redis_Mock.get.return_value = None
    redis_Mock.exists.return_value = False
    redis_Mock.expire.return_value = True
    redis_Mock.xadd.return_value = "1699999999999-0"
    redis_Mock.xrevrange.return_value = []
    redis_Mock.xgroup_create.return_value = True
    redis_Mock.xreadgroup.return_value = []
    redis_Mock.xack.return_value = 1
    redis_Mock.scan.return_value = (0, [])
    # Set up pipeline context manager properly
    pipeline_Mock = Mock()
    pipeline_Mock.__enter__ = Mock(return_value=redis_Mock)
    pipeline_Mock.__exit__ = Mock(return_value=None)
    redis_Mock.pipeline.return_value = pipeline_Mock
    redis_Mock.execute.return_value = []
    return redis_Mock

@pytest.fixture
def Mock_redis_pool():
    """Mock Redis connection pool."""
    pool_Mock = Mock()
    return pool_Mock

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
    def test_get_redis_creates_pool(self, Mock_redis_class, Mock_pool_from_url, Mock_redis_pool):
        """Test Redis connection pool creation."""
        Mock_pool_from_url.return_value = Mock_redis_pool
        Mock_redis_class.return_value = Mock()
        
        from backend.app.mesh.redis_bus import get_redis
        
        # Reset global pool
        import backend.app.mesh.redis_bus
        backend.app.mesh.redis_bus._redis_pool = None
        
        client = get_redis()
        
        Mock_pool_from_url.assert_called_once()
        Mock_redis_class.assert_called_once_with(connection_pool=Mock_redis_pool)
        assert client is not None
    
    @patch('backend.app.mesh.redis_bus.redis.ConnectionPool.from_url')
    @patch('backend.app.mesh.redis_bus.redis.Redis')
    def test_get_redis_reuses_pool(self, Mock_redis_class, Mock_pool_from_url, Mock_redis_pool):
        """Test Redis connection pool reuse."""
        Mock_pool_from_url.return_value = Mock_redis_pool
        Mock_redis_class.return_value = Mock()
        
        from backend.app.mesh.redis_bus import get_redis
        
        # Reset global pool and call twice
        import backend.app.mesh.redis_bus
        backend.app.mesh.redis_bus._redis_pool = None
        
        client1 = get_redis()
        client2 = get_redis()
        
        # Pool should only be created once
        assert Mock_pool_from_url.call_count == 1
        assert Mock_redis_class.call_count == 2  # But client created each time

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
    def test_register_agent(self, Mock_get_redis, Mock_redis, sample_agent_data):
        """Test agent registration."""
        Mock_get_redis.return_value = Mock_redis
        
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
        
        Mock_redis.set.assert_called_once_with(
            f"mesh:agent:{sample_agent_data['agent_id']}",
            json.dumps(expected_data),
            ex=120
        )
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_register_agent_default_ttl(self, Mock_get_redis, Mock_redis):
        """Test agent registration with default TTL."""
        Mock_get_redis.return_value = Mock_redis
        
        from backend.app.mesh.redis_bus import register_agent
        
        register_agent("test-agent", "test-type")
        
        # Should use default TTL of 60 seconds
        Mock_redis.set.assert_called_once()
        args, kwargs = Mock_redis.set.call_args
        assert kwargs["ex"] == 60
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_heartbeat_agent_exists(self, Mock_get_redis, Mock_redis):
        """Test agent heartbeat when agent exists."""
        Mock_get_redis.return_value = Mock_redis
        Mock_redis.exists.return_value = True
        
        from backend.app.mesh.redis_bus import heartbeat_agent
        
        heartbeat_agent("test-agent", 90)
        
        Mock_redis.exists.assert_called_once_with("mesh:agent:test-agent")
        Mock_redis.expire.assert_called_once_with("mesh:agent:test-agent", 90)
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_heartbeat_agent_not_exists(self, Mock_get_redis, Mock_redis):
        """Test agent heartbeat when agent doesn't exist."""
        Mock_get_redis.return_value = Mock_redis
        Mock_redis.exists.return_value = False
        
        from backend.app.mesh.redis_bus import heartbeat_agent
        
        heartbeat_agent("test-agent", 90)
        
        Mock_redis.exists.assert_called_once_with("mesh:agent:test-agent")
        Mock_redis.expire.assert_not_called()
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_list_agents_empty(self, Mock_get_redis, Mock_redis):
        """Test listing agents when none are registered."""
        Mock_get_redis.return_value = Mock_redis
        Mock_redis.scan.return_value = (0, [])
        
        from backend.app.mesh.redis_bus import list_agents
        
        agents = list_agents()
        
        assert agents == []
        Mock_redis.scan.assert_called_once_with(0, match="mesh:agent:*", count=100)
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_list_agents_with_data(self, Mock_get_redis, Mock_redis, sample_agent_data):
        """Test listing agents with registered agents."""
        Mock_get_redis.return_value = Mock_redis
        
        # Mock scan to return agent keys
        agent_keys = ["mesh:agent:agent-1", "mesh:agent:agent-2"]
        Mock_redis.scan.return_value = (0, agent_keys)
        
        # Mock pipeline execution to return agent data
        agent_data = [json.dumps(sample_agent_data), json.dumps(sample_agent_data)]
        Mock_redis.execute.return_value = agent_data
        
        from backend.app.mesh.redis_bus import list_agents
        
        agents = list_agents()
        
        assert len(agents) == 2
        assert all(agent == sample_agent_data for agent in agents)
        
        # Verify pipeline usage for batch fetch
        Mock_redis.pipeline.assert_called_once()
        assert Mock_redis.get.call_count == 2
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_list_agents_with_invalid_json(self, Mock_get_redis, Mock_redis):
        """Test listing agents with some invalid JSON data."""
        Mock_get_redis.return_value = Mock_redis
        
        agent_keys = ["mesh:agent:agent-1", "mesh:agent:agent-2"]
        Mock_redis.scan.return_value = (0, agent_keys)
        
        # Mock one valid and one invalid JSON
        agent_data = ['{"valid": "json"}', 'invalid json']
        Mock_redis.execute.return_value = agent_data
        
        from backend.app.mesh.redis_bus import list_agents
        
        agents = list_agents()
        
        # Should only return the valid agent
        assert len(agents) == 1
        assert agents[0] == {"valid": "json"}

class TestTaskEnqueuing:
    """Test task enqueuing functionality."""
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_enqueue_task_basic(self, Mock_get_redis, Mock_redis, sample_task_payload):
        """Test basic task enqueuing."""
        Mock_get_redis.return_value = Mock_redis
        Mock_redis.xadd.return_value = "1699999999999-0"
        
        from backend.app.mesh.redis_bus import enqueue_task
        
        # Reset stream cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        msg_id = enqueue_task("data_processing", sample_task_payload)
        
        assert msg_id == "1699999999999-0"
        
        # Verify stream creation with consumer group
        Mock_redis.xgroup_create.assert_called_once_with(
            "stream:tasks:data_processing", "default", id="$", mkstream=True
        )
        
        # Verify task addition
        Mock_redis.xadd.assert_called_once_with(
            "stream:tasks:data_processing",
            {"json": json.dumps(sample_task_payload)},
            maxlen=10000,
            approximate=True
        )
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_enqueue_task_group_exists(self, Mock_get_redis, Mock_redis, sample_task_payload):
        """Test task enqueuing when consumer group already exists."""
        Mock_get_redis.return_value = Mock_redis
        Mock_redis.xadd.return_value = "1699999999999-1"
        
        # Mock group already exists error
        import redis
        Mock_redis.xgroup_create.side_effect = redis.exceptions.ResponseError("BUSYGROUP")
        
        from backend.app.mesh.redis_bus import enqueue_task
        
        # Reset stream cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        msg_id = enqueue_task("data_processing", sample_task_payload)
        
        assert msg_id == "1699999999999-1"
        # Should still succeed despite group creation error
        Mock_redis.xadd.assert_called_once()
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_enqueue_task_custom_maxlen(self, Mock_get_redis, Mock_redis, sample_task_payload):
        """Test task enqueuing with custom maxlen."""
        Mock_get_redis.return_value = Mock_redis
        Mock_redis.xadd.return_value = "1699999999999-2"
        
        from backend.app.mesh.redis_bus import enqueue_task
        
        # Reset stream cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        msg_id = enqueue_task("data_processing", sample_task_payload, maxlen=5000)
        
        # Verify custom maxlen is used
        Mock_redis.xadd.assert_called_once_with(
            "stream:tasks:data_processing",
            {"json": json.dumps(sample_task_payload)},
            maxlen=5000,
            approximate=True
        )
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_enqueue_task_caching(self, Mock_get_redis, Mock_redis, sample_task_payload):
        """Test stream caching optimization."""
        Mock_get_redis.return_value = Mock_redis
        Mock_redis.xadd.return_value = "1699999999999-0"
        
        from backend.app.mesh.redis_bus import enqueue_task
        
        # Reset stream cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        # First call should create group
        enqueue_task("data_processing", sample_task_payload)
        assert Mock_redis.xgroup_create.call_count == 1
        
        # Second call should not create group (cached)
        enqueue_task("data_processing", sample_task_payload)
        assert Mock_redis.xgroup_create.call_count == 1  # Still 1

class TestResultTailing:
    """Test result tailing functionality."""
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_tail_results_empty(self, Mock_get_redis, Mock_redis):
        """Test tailing results when no results exist."""
        Mock_get_redis.return_value = Mock_redis
        Mock_redis.execute.return_value = [[]]  # Empty pipeline result
        
        from backend.app.mesh.redis_bus import tail_results
        
        results = tail_results("data_processing")
        
        assert results == []
        Mock_redis.pipeline.assert_called_once()
        Mock_redis.xrevrange.assert_called_once_with("stream:results:data_processing", count=10)
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_tail_results_with_data(self, Mock_get_redis, Mock_redis):
        """Test tailing results with actual data."""
        Mock_get_redis.return_value = Mock_redis
        
        # Mock result data
        result_data = {"status": "completed", "result": "processed successfully"}
        raw_results = [
            ("1699999999999-0", {"json": json.dumps(result_data)}),
            ("1699999999999-1", {"json": json.dumps(result_data)})
        ]
        Mock_redis.execute.return_value = [raw_results]
        
        from backend.app.mesh.redis_bus import tail_results
        
        results = tail_results("data_processing", count=2)
        
        assert len(results) == 2
        assert results[0] == ("1699999999999-0", result_data)
        assert results[1] == ("1699999999999-1", result_data)
        
        Mock_redis.xrevrange.assert_called_once_with("stream:results:data_processing", count=2)
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_tail_results_invalid_json(self, Mock_get_redis, Mock_redis):
        """Test tailing results with invalid JSON data."""
        Mock_get_redis.return_value = Mock_redis
        
        # Mock result with invalid JSON
        raw_results = [
            ("1699999999999-0", {"json": "invalid json"}),
            ("1699999999999-1", {"field1": "value1", "field2": "value2"})  # Raw fields
        ]
        Mock_redis.execute.return_value = [raw_results]
        
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
    def test_create_consumer_group(self, Mock_get_redis, Mock_redis):
        """Test consumer group creation."""
        Mock_get_redis.return_value = Mock_redis
        
        from backend.app.mesh.redis_bus import create_consumer_group
        
        create_consumer_group("data_processing", "workers")
        
        Mock_redis.xgroup_create.assert_called_once_with(
            "stream:tasks:data_processing", "workers", id="$", mkstream=True
        )
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_create_consumer_group_already_exists(self, Mock_get_redis, Mock_redis):
        """Test consumer group creation when group already exists."""
        Mock_get_redis.return_value = Mock_redis
        
        import redis
        Mock_redis.xgroup_create.side_effect = redis.exceptions.ResponseError("BUSYGROUP Consumer Group name already exists")
        
        from backend.app.mesh.redis_bus import create_consumer_group
        
        # Should not raise exception
        create_consumer_group("data_processing", "workers")
        
        Mock_redis.xgroup_create.assert_called_once()
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_create_consumer_group_other_error(self, Mock_get_redis, Mock_redis):
        """Test consumer group creation with unexpected error."""
        Mock_get_redis.return_value = Mock_redis
        
        import redis
        Mock_redis.xgroup_create.side_effect = redis.exceptions.ResponseError("WRONGTYPE Operation against a key holding the wrong kind of value")
        
        from backend.app.mesh.redis_bus import create_consumer_group
        
        # Should raise the exception
        with pytest.raises(redis.exceptions.ResponseError):
            create_consumer_group("data_processing", "workers")
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_read_group_empty(self, Mock_get_redis, Mock_redis):
        """Test reading from consumer group with no messages."""
        Mock_get_redis.return_value = Mock_redis
        Mock_redis.xreadgroup.return_value = []
        
        from backend.app.mesh.redis_bus import read_group
        
        messages = read_group("data_processing", "workers", "worker-1")
        
        assert messages == []
        Mock_redis.xreadgroup.assert_called_once_with(
            "workers", "worker-1", {"stream:tasks:data_processing": ">"}, count=1, block=1000
        )
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_read_group_with_messages(self, Mock_get_redis, Mock_redis, sample_task_payload):
        """Test reading from consumer group with messages."""
        Mock_get_redis.return_value = Mock_redis
        
        # Mock message data
        stream_data = [
            ("stream:tasks:data_processing", [
                ("1699999999999-0", {"json": json.dumps(sample_task_payload)}),
                ("1699999999999-1", {"json": json.dumps(sample_task_payload)})
            ])
        ]
        Mock_redis.xreadgroup.return_value = stream_data
        
        from backend.app.mesh.redis_bus import read_group
        
        messages = read_group("data_processing", "workers", "worker-1", count=2)
        
        assert len(messages) == 2
        assert messages[0] == ("1699999999999-0", sample_task_payload)
        assert messages[1] == ("1699999999999-1", sample_task_payload)

    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_read_group_with_invalid_json(self, Mock_get_redis, Mock_redis):
        """Test reading from consumer group with invalid JSON messages."""
        Mock_get_redis.return_value = Mock_redis
        
        # Mock message data with invalid JSON
        stream_data = [
            ("stream:tasks:data_processing", [
                ("1699999999999-0", {"json": "invalid json"}),
                ("1699999999999-1", {"field1": "value1", "field2": "value2"})  # No json field
            ])
        ]
        Mock_redis.xreadgroup.return_value = stream_data
        
        from backend.app.mesh.redis_bus import read_group
        
        messages = read_group("data_processing", "workers", "worker-1")
        
        assert len(messages) == 2
        # First should fall back to raw fields when JSON parsing fails
        assert messages[0] == ("1699999999999-0", {"json": "invalid json"})
        # Second should use empty dict when no "json" key (default "{}" parses to {})
        assert messages[1] == ("1699999999999-1", {})
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_ack_message(self, Mock_get_redis, Mock_redis):
        """Test acknowledging processed message."""
        Mock_get_redis.return_value = Mock_redis
        Mock_redis.xack.return_value = 1
        
        from backend.app.mesh.redis_bus import ack
        
        result = ack("data_processing", "workers", "1699999999999-0")
        
        assert result == 1
        Mock_redis.xack.assert_called_once_with(
            "stream:tasks:data_processing", "workers", "1699999999999-0"
        )

class TestDeadLetterQueue:
    """Test dead letter queue functionality."""
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_move_to_dead(self, Mock_get_redis, Mock_redis, sample_task_payload):
        """Test moving failed message to dead letter queue."""
        Mock_get_redis.return_value = Mock_redis
        Mock_redis.xadd.return_value = "1699999999999-dead-0"
        
        from backend.app.mesh.redis_bus import move_to_dead
        
        dead_msg_id = move_to_dead("data_processing", "1699999999999-0", sample_task_payload)
        
        assert dead_msg_id == "1699999999999-dead-0"
        
        expected_payload = {
            "id": "1699999999999-0",
            "payload": sample_task_payload
        }
        
        Mock_redis.xadd.assert_called_once_with(
            "stream:dead:data_processing",
            {"json": json.dumps(expected_payload)},
            maxlen=10000,
            approximate=True
        )

class TestPerformanceOptimizations:
    """Test performance optimization features."""
    
    def test_connection_pool_settings(self):
        """Test Redis connection pool optimization settings."""
        with patch('backend.app.mesh.redis_bus.redis.ConnectionPool.from_url') as Mock_pool:
            with patch('backend.app.mesh.redis_bus.redis.Redis') as Mock_redis_class:
                # Reset global pool to force creation
                import backend.app.mesh.redis_bus
                backend.app.mesh.redis_bus._redis_pool = None
                
                from backend.app.mesh.redis_bus import get_redis
                get_redis()
                
                # Verify optimized pool settings
                Mock_pool.assert_called_once()
                args, kwargs = Mock_pool.call_args
                
                assert kwargs['max_connections'] == 50
                assert kwargs['socket_connect_timeout'] == 5
                assert kwargs['socket_timeout'] == 5
                assert kwargs['socket_keepalive'] is True
                assert kwargs['health_check_interval'] == 30

    def test_get_redis_async_creates_pool(self):
        """Test async Redis connection pool creation."""
        with patch('backend.app.mesh.redis_bus.redis_async.ConnectionPool.from_url') as Mock_pool:
            with patch('backend.app.mesh.redis_bus.redis_async.Redis') as Mock_redis_class:
                # Reset global async pool
                import backend.app.mesh.redis_bus
                backend.app.mesh.redis_bus._redis_async_pool = None
                
                import asyncio
                async def test_async():
                    from backend.app.mesh.redis_bus import get_redis_async
                    await get_redis_async()
                    return Mock_pool.call_args
                
                # Run async test
                call_args = asyncio.run(test_async())
                
                # Verify async pool settings
                Mock_pool.assert_called_once()
                args, kwargs = call_args
                assert kwargs['max_connections'] == 50
                assert kwargs['socket_keepalive'] is True
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_pipeline_usage_in_list_agents(self, Mock_get_redis, Mock_redis):
        """Test pipeline usage for batch operations in list_agents."""
        Mock_get_redis.return_value = Mock_redis
        Mock_redis.scan.return_value = (0, ["mesh:agent:1", "mesh:agent:2"])
        Mock_redis.execute.return_value = ['{"id": "1"}', '{"id": "2"}']
        
        from backend.app.mesh.redis_bus import list_agents
        
        list_agents()
        
        # Verify pipeline is used for batch fetching
        Mock_redis.pipeline.assert_called_once()
        assert Mock_redis.get.call_count == 2  # Batched get operations
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_pipeline_usage_in_tail_results(self, Mock_get_redis, Mock_redis):
        """Test pipeline usage for batch operations in tail_results."""
        Mock_get_redis.return_value = Mock_redis
        Mock_redis.execute.return_value = [[]]
        
        from backend.app.mesh.redis_bus import tail_results
        
        tail_results("data_processing")
        
        # Verify pipeline is used
        Mock_redis.pipeline.assert_called_once()
        Mock_redis.xrevrange.assert_called_once()

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_enqueue_task_redis_error(self, Mock_get_redis, Mock_redis, sample_task_payload):
        """Test task enqueuing with Redis connection error."""
        Mock_get_redis.return_value = Mock_redis
        
        import redis
        Mock_redis.xadd.side_effect = redis.exceptions.ConnectionError("Connection failed")
        
        from backend.app.mesh.redis_bus import enqueue_task
        
        # Reset stream cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        with pytest.raises(redis.exceptions.ConnectionError):
            enqueue_task("data_processing", sample_task_payload)
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_list_agents_redis_error(self, Mock_get_redis, Mock_redis):
        """Test listing agents with Redis connection error."""
        Mock_get_redis.return_value = Mock_redis
        
        import redis
        Mock_redis.scan.side_effect = redis.exceptions.ConnectionError("Connection failed")
        
        from backend.app.mesh.redis_bus import list_agents
        
        with pytest.raises(redis.exceptions.ConnectionError):
            list_agents()
    
    @patch('backend.app.mesh.redis_bus.get_redis')
    def test_tail_results_redis_error(self, Mock_get_redis, Mock_redis):
        """Test tailing results with Redis connection error."""
        Mock_get_redis.return_value = Mock_redis
        
        import redis
        Mock_redis.pipeline.side_effect = redis.exceptions.ConnectionError("Connection failed")
        
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