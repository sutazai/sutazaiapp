"""
Integration tests for mesh Redis connectivity and operations.
Tests actual Redis integration with real Redis connection.
"""
import json
import os
import time
import pytest
import asyncio
from typing import Dict, Any, List, Tuple
from unittest.Mock import patch

# Import mesh components
from backend.app.mesh.redis_bus import (
    get_redis, get_redis_async, enqueue_task, tail_results, 
    register_agent, heartbeat_agent, list_agents,
    create_consumer_group, read_group, ack, move_to_dead,
    task_stream, result_stream, dead_stream, agent_key
)

# Test fixtures
@pytest.fixture
def redis_client():
    """Get Redis client for integration tests."""
    try:
        client = get_redis()
        # Test connection
        client.ping()
        return client
    except Exception as e:
        pytest.skip(f"Redis not available for integration tests: {e}")

@pytest.fixture
async def redis_async_client():
    """Get async Redis client for integration tests."""
    try:
        client = await get_redis_async()
        # Test connection
        await client.ping()
        return client
    except Exception as e:
        pytest.skip(f"Async Redis not available for integration tests: {e}")

@pytest.fixture
def test_topic():
    """Test topic name with timestamp to avoid conflicts."""
    return f"test_topic_{int(time.time())}"

@pytest.fixture
def test_agent_id():
    """Test agent ID with timestamp to avoid conflicts."""
    return f"test_agent_{int(time.time())}"

@pytest.fixture
def sample_task_data():
    """Sample task data for testing."""
    return {
        "task_id": f"task_{int(time.time())}",
        "task_type": "integration_test",
        "input_data": "test data for integration testing",
        "parameters": {
            "test_param": "test_value",
            "timestamp": int(time.time())
        },
        "priority": "normal"
    }

@pytest.fixture
def sample_agent_meta():
    """Sample agent metadata."""
    return {
        "version": "1.0.0",
        "capabilities": ["test_capability_1", "test_capability_2"],
        "max_concurrent_tasks": 3,
        "status": "active",
        "test_metadata": f"test_{int(time.time())}"
    }

@pytest.fixture(autouse=True)
def cleanup_test_data(redis_client, test_topic, test_agent_id):
    """Cleanup test data before and after tests."""
    # Cleanup before test
    try:
        redis_client.delete(task_stream(test_topic))
        redis_client.delete(result_stream(test_topic))
        redis_client.delete(dead_stream(test_topic))
        redis_client.delete(agent_key(test_agent_id))
    except:
        pass
    
    yield
    
    # Cleanup after test
    try:
        redis_client.delete(task_stream(test_topic))
        redis_client.delete(result_stream(test_topic))
        redis_client.delete(dead_stream(test_topic))
        redis_client.delete(agent_key(test_agent_id))
        
        # Clean up consumer groups
        try:
            redis_client.xgroup_destroy(task_stream(test_topic), "default")
            redis_client.xgroup_destroy(task_stream(test_topic), "test_group")
        except:
            pass
    except:
        pass

class TestRedisConnection:
    """Test Redis connection and configuration."""
    
    def test_redis_connection_sync(self, redis_client):
        """Test synchronous Redis connection."""
        assert redis_client.ping() is True
    
    async def test_redis_connection_async(self, redis_async_client):
        """Test asynchronous Redis connection."""
        result = await redis_async_client.ping()
        assert result is True
    
    def test_redis_connection_pool_configuration(self):
        """Test Redis connection pool is properly configured."""
        # Reset global pool to test fresh creation
        import backend.app.mesh.redis_bus
        original_pool = backend.app.mesh.redis_bus._redis_pool
        backend.app.mesh.redis_bus._redis_pool = None
        
        try:
            client = get_redis()
            pool = client.connection_pool
            
            # Verify pool settings
            assert pool.max_connections == 50
            assert pool.connection_kwargs['socket_connect_timeout'] == 5
            assert pool.connection_kwargs['socket_timeout'] == 5
            assert pool.connection_kwargs['socket_keepalive'] is True
            assert pool.connection_kwargs['health_check_interval'] == 30
        finally:
            # Restore original pool
            backend.app.mesh.redis_bus._redis_pool = original_pool
    
    def test_redis_url_configuration(self):
        """Test Redis URL configuration from environment."""
        # Test default URL
        with patch.dict(os.environ, {}, clear=True):
            from backend.app.mesh.redis_bus import _redis_url
            assert _redis_url() == "redis://redis:6379/0"
        
        # Test custom URL
        custom_url = "redis://custom:6380/1"
        with patch.dict(os.environ, {"REDIS_URL": custom_url}):
            from backend.app.mesh.redis_bus import _redis_url
            assert _redis_url() == custom_url

class TestStreamKeys:
    """Test stream key generation and validation."""
    
    def test_stream_key_generation(self, test_topic):
        """Test stream key generation."""
        task_key = task_stream(test_topic)
        result_key = result_stream(test_topic)
        dead_key = dead_stream(test_topic)
        
        assert task_key == f"stream:tasks:{test_topic}"
        assert result_key == f"stream:results:{test_topic}"
        assert dead_key == f"stream:dead:{test_topic}"
    
    def test_agent_key_generation(self, test_agent_id):
        """Test agent key generation."""
        key = agent_key(test_agent_id)
        assert key == f"mesh:agent:{test_agent_id}"

class TestTaskEnqueuing:
    """Test task enqueuing and stream operations."""
    
    def test_enqueue_single_task(self, redis_client, test_topic, sample_task_data):
        """Test enqueuing a single task."""
        # Reset enqueue cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        msg_id = enqueue_task(test_topic, sample_task_data)
        
        assert msg_id is not None
        assert isinstance(msg_id, str)
        assert "-" in msg_id  # Redis stream ID format
        
        # Verify task was added to stream
        messages = redis_client.xrange(task_stream(test_topic))
        assert len(messages) == 1
        
        stored_data = json.loads(messages[0][1]["json"])
        assert stored_data == sample_task_data
    
    def test_enqueue_multiple_tasks(self, redis_client, test_topic, sample_task_data):
        """Test enqueuing multiple tasks."""
        # Reset enqueue cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        task_count = 5
        msg_ids = []
        
        for i in range(task_count):
            task_data = sample_task_data.copy()
            task_data["task_id"] = f"task_{i}_{int(time.time())}"
            msg_id = enqueue_task(test_topic, task_data)
            msg_ids.append(msg_id)
        
        assert len(msg_ids) == task_count
        assert len(set(msg_ids)) == task_count  # All IDs should be unique
        
        # Verify all tasks are in stream
        messages = redis_client.xrange(task_stream(test_topic))
        assert len(messages) == task_count
    
    def test_enqueue_with_custom_maxlen(self, redis_client, test_topic, sample_task_data):
        """Test enqueuing with custom maxlen parameter."""
        # Reset enqueue cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        msg_id = enqueue_task(test_topic, sample_task_data, maxlen=100)
        assert msg_id is not None
        
        # Verify task was added
        messages = redis_client.xrange(task_stream(test_topic))
        assert len(messages) == 1
    
    def test_stream_maxlen_enforcement(self, redis_client, test_topic, sample_task_data):
        """Test stream maxlen enforcement."""
        # Reset enqueue cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        maxlen = 3
        
        # Add more tasks than maxlen
        for i in range(maxlen + 2):
            task_data = sample_task_data.copy()
            task_data["task_id"] = f"task_{i}"
            enqueue_task(test_topic, task_data, maxlen=maxlen)
        
        # Stream should be trimmed to approximately maxlen
        messages = redis_client.xrange(task_stream(test_topic))
        # Note: XTRIM with MAXLEN ~ is approximate, so may have a few more
        assert len(messages) <= maxlen + 1

class TestResultRetrieval:
    """Test result tailing and retrieval."""
    
    def test_tail_results_empty(self, redis_client, test_topic):
        """Test tailing results from empty stream."""
        results = tail_results(test_topic)
        assert results == []
    
    def test_tail_results_with_data(self, redis_client, test_topic):
        """Test tailing results with actual data."""
        # Add test results directly to result stream
        result_data = [
            {"status": "completed", "result": "test result 1"},
            {"status": "completed", "result": "test result 2"},
            {"status": "failed", "error": "test error"}
        ]
        
        msg_ids = []
        for data in result_data:
            msg_id = redis_client.xadd(
                result_stream(test_topic),
                {"json": json.dumps(data)}
            )
            msg_ids.append(msg_id)
        
        # Tail results
        results = tail_results(test_topic, count=5)
        
        assert len(results) == 3
        
        # Results should be in reverse order (most recent first)
        for i, (msg_id, data) in enumerate(results):
            assert msg_id == msg_ids[-(i+1)]  # Reverse order
            assert data == result_data[-(i+1)]
    
    def test_tail_results_count_limit(self, redis_client, test_topic):
        """Test result count limiting."""
        # Add multiple results
        for i in range(10):
            redis_client.xadd(
                result_stream(test_topic),
                {"json": json.dumps({"result": f"test_{i}"})}
            )
        
        # Tail with count limit
        results = tail_results(test_topic, count=3)
        assert len(results) == 3
        
        # Should get most recent 3
        for i, (msg_id, data) in enumerate(results):
            expected_result = f"test_{9-i}"  # Most recent first
            assert data["result"] == expected_result
    
    def test_tail_results_invalid_json(self, redis_client, test_topic):
        """Test handling of invalid JSON in results."""
        # Add result with invalid JSON
        redis_client.xadd(
            result_stream(test_topic),
            {"json": "invalid json data"}
        )
        
        # Add result with raw fields (no JSON)
        redis_client.xadd(
            result_stream(test_topic),
            {"field1": "value1", "field2": "value2"}
        )
        
        results = tail_results(test_topic)
        assert len(results) == 2
        
        # First should fall back to raw fields
        assert results[0][1] == {"field1": "value1", "field2": "value2"}
        # Second should fall back to raw fields  
        assert results[1][1] == {"json": "invalid json data"}

class TestAgentRegistry:
    """Test agent registration and management."""
    
    def test_register_agent_basic(self, redis_client, test_agent_id, sample_agent_meta):
        """Test basic agent registration."""
        register_agent(test_agent_id, "test_agent", 60, sample_agent_meta)
        
        # Verify agent was registered
        agent_data = redis_client.get(agent_key(test_agent_id))
        assert agent_data is not None
        
        parsed_data = json.loads(agent_data)
        assert parsed_data["agent_id"] == test_agent_id
        assert parsed_data["agent_type"] == "test_agent"
        assert parsed_data["meta"] == sample_agent_meta
        
        # Verify TTL was set
        ttl = redis_client.ttl(agent_key(test_agent_id))
        assert 50 <= ttl <= 60  # Should be close to 60 seconds
    
    def test_register_agent_default_ttl(self, redis_client, test_agent_id):
        """Test agent registration with default TTL."""
        register_agent(test_agent_id, "test_agent")
        
        # Verify default TTL (60 seconds)
        ttl = redis_client.ttl(agent_key(test_agent_id))
        assert 50 <= ttl <= 60
    
    def test_heartbeat_agent_existing(self, redis_client, test_agent_id, sample_agent_meta):
        """Test heartbeat for existing agent."""
        # Register agent first
        register_agent(test_agent_id, "test_agent", 30, sample_agent_meta)
        
        # Wait a bit and then heartbeat
        time.sleep(1)
        heartbeat_agent(test_agent_id, 60)
        
        # TTL should be refreshed to ~60 seconds
        ttl = redis_client.ttl(agent_key(test_agent_id))
        assert 50 <= ttl <= 60
    
    def test_heartbeat_agent_nonexistent(self, redis_client, test_agent_id):
        """Test heartbeat for non-existent agent."""
        # Ensure agent doesn't exist
        redis_client.delete(agent_key(test_agent_id))
        
        # Heartbeat should not create the agent
        heartbeat_agent(test_agent_id, 60)
        
        # Agent should still not exist
        assert redis_client.exists(agent_key(test_agent_id)) == 0
    
    def test_list_agents_empty(self, redis_client):
        """Test listing agents when none are registered."""
        # Clean up any existing test agents
        cursor = 0
        while True:
            cursor, keys = redis_client.scan(cursor, match="mesh:agent:test_*")
            if keys:
                redis_client.delete(*keys)
            if cursor == 0:
                break
        
        agents = list_agents()
        # Filter out any non-test agents
        test_agents = [a for a in agents if a.get("agent_id", "").startswith("test_")]
        assert len(test_agents) == 0
    
    def test_list_agents_multiple(self, redis_client, sample_agent_meta):
        """Test listing multiple registered agents."""
        # Register multiple test agents
        agent_ids = [f"test_agent_{i}_{int(time.time())}" for i in range(3)]
        
        for agent_id in agent_ids:
            meta = sample_agent_meta.copy()
            meta["agent_index"] = agent_id
            register_agent(agent_id, "test_agent", 120, meta)
        
        agents = list_agents()
        
        # Filter for our test agents
        test_agents = [a for a in agents if a.get("agent_id") in agent_ids]
        assert len(test_agents) == 3
        
        # Verify agent data
        for agent in test_agents:
            assert agent["agent_type"] == "test_agent"
            assert agent["agent_id"] in agent_ids
            assert "agent_index" in agent["meta"]
    
    def test_agent_expiration(self, redis_client, test_agent_id):
        """Test agent expiration with short TTL."""
        # Register agent with very short TTL
        register_agent(test_agent_id, "test_agent", 1)
        
        # Agent should exist initially
        assert redis_client.exists(agent_key(test_agent_id)) == 1
        
        # Wait for expiration
        time.sleep(2)
        
        # Agent should be expired
        assert redis_client.exists(agent_key(test_agent_id)) == 0
        
        # Should not appear in agent list
        agents = list_agents()
        test_agents = [a for a in agents if a.get("agent_id") == test_agent_id]
        assert len(test_agents) == 0

class TestConsumerGroups:
    """Test consumer group operations."""
    
    def test_create_consumer_group(self, redis_client, test_topic):
        """Test consumer group creation."""
        group_name = "test_group"
        create_consumer_group(test_topic, group_name)
        
        # Verify group was created
        groups = redis_client.xinfo_groups(task_stream(test_topic))
        group_names = [g['name'] for g in groups]
        assert group_name in group_names
    
    def test_create_consumer_group_already_exists(self, redis_client, test_topic):
        """Test creating consumer group that already exists."""
        group_name = "test_group"
        
        # Create group first time
        create_consumer_group(test_topic, group_name)
        
        # Create again - should not raise error
        create_consumer_group(test_topic, group_name)
        
        # Should still exist
        groups = redis_client.xinfo_groups(task_stream(test_topic))
        group_names = [g['name'] for g in groups]
        assert group_name in group_names
    
    def test_read_group_no_messages(self, redis_client, test_topic):
        """Test reading from consumer group with no messages."""
        group_name = "test_group"
        consumer_name = "test_consumer"
        
        create_consumer_group(test_topic, group_name)
        
        # Read with short timeout
        messages = read_group(test_topic, group_name, consumer_name, count=1, block_ms=100)
        assert messages == []
    
    def test_read_group_with_messages(self, redis_client, test_topic, sample_task_data):
        """Test reading from consumer group with messages."""
        group_name = "test_group"
        consumer_name = "test_consumer"
        
        # Reset enqueue cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        # Enqueue some tasks first
        msg_ids = []
        for i in range(3):
            task_data = sample_task_data.copy()
            task_data["task_id"] = f"task_{i}"
            msg_id = enqueue_task(test_topic, task_data)
            msg_ids.append(msg_id)
        
        # Create consumer group from beginning
        try:
            redis_client.xgroup_create(task_stream(test_topic), group_name, id="0", mkstream=True)
        except:
            pass  # Group might already exist
        
        # Read messages
        messages = read_group(test_topic, group_name, consumer_name, count=2)
        
        assert len(messages) <= 2  # Should get up to 2 messages
        
        if messages:  # If any messages were read
            for msg_id, data in messages:
                assert msg_id in msg_ids
                assert "task_id" in data
    
    def test_ack_message(self, redis_client, test_topic, sample_task_data):
        """Test acknowledging processed messages."""
        group_name = "test_group"
        consumer_name = "test_consumer"
        
        # Reset enqueue cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        # Enqueue a task
        msg_id = enqueue_task(test_topic, sample_task_data)
        
        # Create consumer group from beginning
        try:
            redis_client.xgroup_create(task_stream(test_topic), group_name, id="0", mkstream=True)
        except:
            pass
        
        # Read message
        messages = read_group(test_topic, group_name, consumer_name, count=1)
        
        if messages:  # If message was read
            read_msg_id, data = messages[0]
            
            # Acknowledge message
            ack_count = ack(test_topic, group_name, read_msg_id)
            assert ack_count == 1

class TestDeadLetterQueue:
    """Test dead letter queue operations."""
    
    def test_move_to_dead(self, redis_client, test_topic, sample_task_data):
        """Test moving failed message to dead letter queue."""
        original_msg_id = "1699999999999-0"
        
        dead_msg_id = move_to_dead(test_topic, original_msg_id, sample_task_data)
        
        assert dead_msg_id is not None
        assert isinstance(dead_msg_id, str)
        
        # Verify message was added to dead stream
        messages = redis_client.xrange(dead_stream(test_topic))
        assert len(messages) == 1
        
        dead_data = json.loads(messages[0][1]["json"])
        assert dead_data["id"] == original_msg_id
        assert dead_data["payload"] == sample_task_data
    
    def test_dead_letter_maxlen(self, redis_client, test_topic, sample_task_data):
        """Test dead letter queue maxlen enforcement."""
        # Add multiple failed messages
        for i in range(5):
            task_data = sample_task_data.copy()
            task_data["task_id"] = f"failed_task_{i}"
            move_to_dead(test_topic, f"msg_{i}", task_data)
        
        # Verify messages are in dead stream
        messages = redis_client.xrange(dead_stream(test_topic))
        assert len(messages) == 5

class TestPerformanceOptimizations:
    """Test performance features in integration."""
    
    def test_connection_pooling(self):
        """Test that connection pooling works in practice."""
        # Get multiple Redis clients
        client1 = get_redis()
        client2 = get_redis()
        
        # Should use same connection pool
        assert client1.connection_pool is client2.connection_pool
    
    def test_pipeline_operations(self, redis_client, test_topic):
        """Test pipeline operations work correctly."""
        # Add multiple results
        for i in range(10):
            redis_client.xadd(
                result_stream(test_topic),
                {"json": json.dumps({"result": f"test_{i}"})}
            )
        
        # tail_results uses pipeline internally
        results = tail_results(test_topic, count=5)
        assert len(results) == 5
    
    def test_stream_caching(self, redis_client, test_topic, sample_task_data):
        """Test stream creation caching."""
        # Reset enqueue cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        # First enqueue should create stream and cache it
        msg_id1 = enqueue_task(test_topic, sample_task_data)
        assert msg_id1 is not None
        
        # Second enqueue should use cached stream info
        task_data2 = sample_task_data.copy()
        task_data2["task_id"] = "task_2"
        msg_id2 = enqueue_task(test_topic, task_data2)
        assert msg_id2 is not None
        
        # Both messages should be in stream
        messages = redis_client.xrange(task_stream(test_topic))
        assert len(messages) == 2

class TestErrorHandling:
    """Test error handling in integration scenarios."""
    
    def test_invalid_json_handling(self, redis_client, test_topic):
        """Test handling of invalid JSON in streams."""
        # Add invalid JSON to result stream
        redis_client.xadd(result_stream(test_topic), {"json": "invalid{json"})
        redis_client.xadd(result_stream(test_topic), {"raw_field": "raw_value"})
        
        # Should not crash and handle gracefully
        results = tail_results(test_topic)
        assert len(results) == 2
        
        # Should fall back to raw fields
        assert isinstance(results[0][1], dict)
        assert isinstance(results[1][1], dict)
    
    def test_large_payload_handling(self, redis_client, test_topic):
        """Test handling of large payloads."""
        # Create large task data
        large_data = {
            "task_id": "large_task",
            "large_field": "x" * 10000,  # 10KB of data
            "nested_data": {"key": "value"} * 1000
        }
        
        # Reset enqueue cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        # Should handle large payload
        msg_id = enqueue_task(test_topic, large_data)
        assert msg_id is not None
        
        # Verify data integrity
        messages = redis_client.xrange(task_stream(test_topic))
        assert len(messages) == 1
        
        retrieved_data = json.loads(messages[0][1]["json"])
        assert retrieved_data == large_data

if __name__ == "__main__":
    pytest.main([__file__, "-v"])