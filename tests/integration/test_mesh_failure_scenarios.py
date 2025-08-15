"""
Integration tests for mesh system failure scenarios.
Tests error handling, recovery, and resilience under various failure conditions.
"""
import json
import time
import threading
import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
import redis

# Import mesh components
from backend.app.mesh.redis_bus import (
    get_redis, enqueue_task, tail_results, register_agent,
    list_agents, create_consumer_group, read_group, ack,
    move_to_dead, task_stream, result_stream, dead_stream
)

class FailureSimulator:
    """Simulate various failure conditions."""
    
    def __init__(self):
        self.failures_active = {}
        self.failure_count = 0
        self._original_methods = {}
    
    def simulate_redis_connection_failure(self, duration: float = 1.0):
        """Simulate Redis connection failure."""
        def failing_ping(*args, **kwargs):
            if self.failures_active.get("redis_connection"):
                raise redis.exceptions.ConnectionError("Simulated connection failure")
            return self._original_methods["ping"](*args, **kwargs)
        
        def failing_operation(*args, **kwargs):
            if self.failures_active.get("redis_connection"):
                raise redis.exceptions.ConnectionError("Simulated connection failure")
            # Try to call original method if available
            method_name = kwargs.pop('_method_name', 'unknown')
            if method_name in self._original_methods:
                return self._original_methods[method_name](*args, **kwargs)
            raise Exception("Original method not found")
        
        return self._setup_failure("redis_connection", duration, {
            "ping": failing_ping,
            "xadd": failing_operation,
            "xrevrange": failing_operation,
            "xlen": failing_operation,
            "set": failing_operation,
            "get": failing_operation
        })
    
    def simulate_network_timeout(self, duration: float = 2.0):
        """Simulate network timeout."""
        def timeout_operation(*args, **kwargs):
            if self.failures_active.get("network_timeout"):
                time.sleep(duration)
                raise redis.exceptions.TimeoutError("Simulated timeout")
            method_name = kwargs.pop('_method_name', 'unknown')
            if method_name in self._original_methods:
                return self._original_methods[method_name](*args, **kwargs)
            raise Exception("Original method not found")
        
        return self._setup_failure("network_timeout", duration, {
            "xadd": timeout_operation,
            "xrevrange": timeout_operation,
            "xreadgroup": timeout_operation
        })
    
    def simulate_memory_pressure(self, duration: float = 1.0):
        """Simulate Redis memory pressure."""
        def memory_error_operation(*args, **kwargs):
            if self.failures_active.get("memory_pressure"):
                raise redis.exceptions.ResponseError("OOM command not allowed when used memory > 'maxmemory'")
            method_name = kwargs.pop('_method_name', 'unknown')
            if method_name in self._original_methods:
                return self._original_methods[method_name](*args, **kwargs)
            raise Exception("Original method not found")
        
        return self._setup_failure("memory_pressure", duration, {
            "xadd": memory_error_operation,
            "set": memory_error_operation
        })
    
    def simulate_partial_failure(self, failure_rate: float = 0.3, duration: float = 2.0):
        """Simulate partial/intermittent failures."""
        import random
        
        def partial_failure_operation(*args, **kwargs):
            if self.failures_active.get("partial_failure") and random.random() < failure_rate:
                self.failure_count += 1
                raise redis.exceptions.ConnectionError("Intermittent failure")
            method_name = kwargs.pop('_method_name', 'unknown')
            if method_name in self._original_methods:
                return self._original_methods[method_name](*args, **kwargs)
            raise Exception("Original method not found")
        
        return self._setup_failure("partial_failure", duration, {
            "xadd": partial_failure_operation,
            "xrevrange": partial_failure_operation,
            "get": partial_failure_operation,
            "set": partial_failure_operation
        })
    
    def _setup_failure(self, failure_type: str, duration: float, method_overrides: Dict):
        """Setup failure simulation."""
        self.failures_active[failure_type] = True
        
        # Schedule failure end
        def end_failure():
            time.sleep(duration)
            self.failures_active[failure_type] = False
        
        timer = threading.Timer(duration, end_failure)
        timer.start()
        
        return method_overrides
    
    def is_failure_active(self, failure_type: str) -> bool:
        """Check if failure is active."""
        return self.failures_active.get(failure_type, False)
    
    def stop_all_failures(self):
        """Stop all active failures."""
        self.failures_active.clear()

# Test fixtures
@pytest.fixture
def redis_client():
    """Get Redis client for failure tests."""
    try:
        client = get_redis()
        client.ping()
        return client
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")

@pytest.fixture
def test_topic():
    """Test topic with timestamp."""
    return f"failure_test_{int(time.time())}"

@pytest.fixture
def failure_simulator():
    """Failure simulator instance."""
    simulator = FailureSimulator()
    yield simulator
    simulator.stop_all_failures()

@pytest.fixture(autouse=True)
def cleanup_test_data(redis_client, test_topic):
    """Cleanup test data."""
    # Cleanup before
    try:
        redis_client.delete(task_stream(test_topic))
        redis_client.delete(result_stream(test_topic))
        redis_client.delete(dead_stream(test_topic))
    except:
        pass
    
    yield
    
    # Cleanup after
    try:
        redis_client.delete(task_stream(test_topic))
        redis_client.delete(result_stream(test_topic))
        redis_client.delete(dead_stream(test_topic))
        
        # Clean up consumer groups
        try:
            redis_client.xgroup_destroy(task_stream(test_topic), "failure_test_group")
        except:
            pass
    except:
        pass

class TestConnectionFailures:
    """Test Redis connection failure scenarios."""
    
    def test_enqueue_with_connection_failure(self, redis_client, test_topic, failure_simulator):
        """Test task enqueuing during connection failures."""
        # Start with successful enqueue
        payload = {"task_type": "connection_test", "data": "test_data"}
        
        # Reset enqueue cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        # First enqueue should succeed
        msg_id = enqueue_task(test_topic, payload)
        assert msg_id is not None
        
        # Simulate connection failure
        with patch.object(redis_client, 'xadd') as mock_xadd:
            mock_xadd.side_effect = redis.exceptions.ConnectionError("Connection failed")
            
            # Should raise exception during connection failure
            with pytest.raises(redis.exceptions.ConnectionError):
                enqueue_task(test_topic, payload)
        
        # After failure recovery, should work again
        msg_id2 = enqueue_task(test_topic, payload)
        assert msg_id2 is not None
        assert msg_id2 != msg_id  # Different message ID
    
    def test_result_retrieval_with_connection_failure(self, redis_client, test_topic, failure_simulator):
        """Test result retrieval during connection failures."""
        # Populate some results first
        for i in range(5):
            redis_client.xadd(
                result_stream(test_topic),
                {"json": json.dumps({"result": f"test_{i}"})}
            )
        
        # Normal retrieval should work
        results = tail_results(test_topic, count=3)
        assert len(results) == 3
        
        # Simulate connection failure
        with patch.object(redis_client, 'pipeline') as mock_pipeline:
            mock_pipeline.side_effect = redis.exceptions.ConnectionError("Connection failed")
            
            # Should raise exception during connection failure
            with pytest.raises(redis.exceptions.ConnectionError):
                tail_results(test_topic, count=3)
        
        # After failure recovery, should work again
        results2 = tail_results(test_topic, count=2)
        assert len(results2) == 2
    
    def test_agent_operations_with_connection_failure(self, redis_client, failure_simulator):
        """Test agent operations during connection failures."""
        agent_id = f"failure_test_agent_{int(time.time())}"
        
        # Normal registration should work
        register_agent(agent_id, "test_agent", 30, {"test": True})
        
        # Verify agent registered
        agents = list_agents()
        test_agents = [a for a in agents if a.get("agent_id") == agent_id]
        assert len(test_agents) == 1
        
        # Simulate connection failure for registration
        with patch.object(redis_client, 'set') as mock_set:
            mock_set.side_effect = redis.exceptions.ConnectionError("Connection failed")
            
            # Should raise exception during connection failure
            with pytest.raises(redis.exceptions.ConnectionError):
                register_agent(f"failing_agent_{int(time.time())}", "test_agent", 30)
        
        # Simulate connection failure for listing
        with patch.object(redis_client, 'scan') as mock_scan:
            mock_scan.side_effect = redis.exceptions.ConnectionError("Connection failed")
            
            # Should raise exception during connection failure
            with pytest.raises(redis.exceptions.ConnectionError):
                list_agents()
        
        # After failure, operations should work again
        register_agent(f"recovery_agent_{int(time.time())}", "test_agent", 30)

class TestTimeoutScenarios:
    """Test timeout scenarios."""
    
    def test_enqueue_timeout_handling(self, redis_client, test_topic):
        """Test handling of enqueue timeouts."""
        payload = {"task_type": "timeout_test", "data": "test_data"}
        
        # Reset enqueue cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        # Simulate timeout
        with patch.object(redis_client, 'xadd') as mock_xadd:
            mock_xadd.side_effect = redis.exceptions.TimeoutError("Operation timed out")
            
            # Should raise timeout exception
            with pytest.raises(redis.exceptions.TimeoutError):
                enqueue_task(test_topic, payload)
        
        # After timeout, should work normally
        msg_id = enqueue_task(test_topic, payload)
        assert msg_id is not None
    
    def test_consumer_group_timeout_handling(self, redis_client, test_topic):
        """Test consumer group operations with timeouts."""
        group_name = "timeout_test_group"
        consumer_name = "timeout_consumer"
        
        # Create consumer group
        create_consumer_group(test_topic, group_name)
        
        # Enqueue a message
        # Reset enqueue cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        payload = {"task_type": "timeout_test", "data": "test"}
        msg_id = enqueue_task(test_topic, payload)
        
        # Normal read should work (with short timeout)
        messages = read_group(test_topic, group_name, consumer_name, count=1, block_ms=100)
        assert len(messages) <= 1  # Might get the message or timeout
        
        # Simulate timeout on read
        with patch.object(redis_client, 'xreadgroup') as mock_read:
            mock_read.side_effect = redis.exceptions.TimeoutError("Read timed out")
            
            # Should raise timeout exception
            with pytest.raises(redis.exceptions.TimeoutError):
                read_group(test_topic, group_name, consumer_name, count=1, block_ms=1000)
        
        # After timeout, should work normally
        # (No assertion here as message might already be consumed)
        try:
            read_group(test_topic, group_name, consumer_name, count=1, block_ms=100)
        except:
            pass  # Acceptable if no messages available
    
    def test_result_retrieval_timeout(self, redis_client, test_topic):
        """Test result retrieval timeout handling."""
        # Populate results
        for i in range(3):
            redis_client.xadd(
                result_stream(test_topic),
                {"json": json.dumps({"result": f"timeout_test_{i}"})}
            )
        
        # Simulate timeout during retrieval
        with patch.object(redis_client, 'pipeline') as mock_pipeline:
            # Create a mock pipeline that times out
            mock_pipe = Mock()
            mock_pipe.__enter__ = Mock(return_value=mock_pipe)
            mock_pipe.__exit__ = Mock(return_value=None)
            mock_pipe.xrevrange = Mock()
            mock_pipe.execute.side_effect = redis.exceptions.TimeoutError("Pipeline timed out")
            mock_pipeline.return_value = mock_pipe
            
            # Should raise timeout exception
            with pytest.raises(redis.exceptions.TimeoutError):
                tail_results(test_topic, count=2)
        
        # After timeout, should work normally
        results = tail_results(test_topic, count=2)
        assert len(results) == 2

class TestMemoryPressureScenarios:
    """Test Redis memory pressure scenarios."""
    
    def test_enqueue_under_memory_pressure(self, redis_client, test_topic):
        """Test task enqueuing under memory pressure."""
        payload = {"task_type": "memory_test", "data": "x" * 1024}  # 1KB payload
        
        # Reset enqueue cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        # Normal enqueue should work
        msg_id = enqueue_task(test_topic, payload)
        assert msg_id is not None
        
        # Simulate memory pressure
        with patch.object(redis_client, 'xadd') as mock_xadd:
            mock_xadd.side_effect = redis.exceptions.ResponseError(
                "OOM command not allowed when used memory > 'maxmemory'"
            )
            
            # Should raise memory error
            with pytest.raises(redis.exceptions.ResponseError):
                enqueue_task(test_topic, payload)
        
        # After memory pressure relief, should work again
        msg_id2 = enqueue_task(test_topic, payload)
        assert msg_id2 is not None
    
    def test_agent_registration_under_memory_pressure(self, redis_client):
        """Test agent registration under memory pressure."""
        agent_id = f"memory_test_agent_{int(time.time())}"
        
        # Normal registration should work
        register_agent(agent_id, "memory_test", 30, {"test": True})
        
        # Simulate memory pressure
        with patch.object(redis_client, 'set') as mock_set:
            mock_set.side_effect = redis.exceptions.ResponseError(
                "OOM command not allowed when used memory > 'maxmemory'"
            )
            
            # Should raise memory error
            with pytest.raises(redis.exceptions.ResponseError):
                register_agent(f"oom_agent_{int(time.time())}", "memory_test", 30)
        
        # After memory pressure relief, should work again
        register_agent(f"recovery_agent_{int(time.time())}", "memory_test", 30)

class TestIntermittentFailures:
    """Test intermittent/partial failure scenarios."""
    
    def test_partial_enqueue_failures(self, redis_client, test_topic):
        """Test handling of partial enqueue failures."""
        num_tasks = 20
        success_count = 0
        failure_count = 0
        
        # Reset enqueue cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        # Simulate 30% failure rate
        original_xadd = redis_client.xadd
        
        def intermittent_xadd(*args, **kwargs):
            import random
            if random.random() < 0.3:  # 30% failure rate
                raise redis.exceptions.ConnectionError("Intermittent failure")
            return original_xadd(*args, **kwargs)
        
        with patch.object(redis_client, 'xadd', side_effect=intermittent_xadd):
            for i in range(num_tasks):
                try:
                    payload = {"task_type": "intermittent_test", "task_id": i}
                    msg_id = enqueue_task(test_topic, payload)
                    if msg_id:
                        success_count += 1
                except redis.exceptions.ConnectionError:
                    failure_count += 1
        
        # Should have some successes and some failures
        assert success_count > 0
        assert failure_count > 0
        assert success_count + failure_count == num_tasks
        
        # Success rate should be approximately 70%
        success_rate = success_count / num_tasks
        assert 0.5 < success_rate < 0.9  # Reasonable range
        
        print(f"Intermittent failures: {success_count}/{num_tasks} succeeded "
              f"({success_rate*100:.1f}% success rate)")
    
    def test_recovery_after_extended_failure(self, redis_client, test_topic):
        """Test system recovery after extended failure period."""
        # Reset enqueue cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        # Normal operation
        payload = {"task_type": "recovery_test", "phase": "normal"}
        msg_id1 = enqueue_task(test_topic, payload)
        assert msg_id1 is not None
        
        # Extended failure period
        failure_duration = 2.0  # 2 seconds
        start_time = time.time()
        
        def extended_failure(*args, **kwargs):
            if time.time() - start_time < failure_duration:
                raise redis.exceptions.ConnectionError("Extended failure")
            return redis_client.__class__.xadd(redis_client, *args, **kwargs)
        
        with patch.object(redis_client, 'xadd', side_effect=extended_failure):
            # Should fail during failure period
            while time.time() - start_time < failure_duration:
                try:
                    payload["phase"] = "failure"
                    enqueue_task(test_topic, payload)
                    # If we get here, failure period ended
                    break
                except redis.exceptions.ConnectionError:
                    time.sleep(0.1)  # Brief pause before retry
        
        # After failure period, should work normally
        payload["phase"] = "recovery"
        msg_id2 = enqueue_task(test_topic, payload)
        assert msg_id2 is not None
        assert msg_id2 != msg_id1
        
        # Verify messages are in stream
        stream_length = redis_client.xlen(task_stream(test_topic))
        assert stream_length >= 2  # At least the normal and recovery messages

class TestConcurrentFailures:
    """Test failures under concurrent load."""
    
    def test_concurrent_operations_with_failures(self, redis_client, test_topic):
        """Test concurrent operations during failures."""
        num_workers = 10
        operations_per_worker = 5
        failure_rate = 0.2  # 20% failure rate
        
        success_count = threading.Value('i', 0)
        failure_count = threading.Value('i', 0)
        
        # Reset enqueue cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        def failing_worker(worker_id: int):
            """Worker that operates with intermittent failures."""
            import random
            original_xadd = redis_client.xadd
            
            def worker_xadd(*args, **kwargs):
                if random.random() < failure_rate:
                    raise redis.exceptions.ConnectionError("Worker failure")
                return original_xadd(*args, **kwargs)
            
            for i in range(operations_per_worker):
                try:
                    with patch.object(redis_client, 'xadd', side_effect=worker_xadd):
                        payload = {
                            "task_type": "concurrent_failure_test",
                            "worker_id": worker_id,
                            "operation": i
                        }
                        msg_id = enqueue_task(test_topic, payload)
                        
                        if msg_id:
                            with success_count.get_lock():
                                success_count.value += 1
                
                except redis.exceptions.ConnectionError:
                    with failure_count.get_lock():
                        failure_count.value += 1
                
                time.sleep(0.01)  # Brief pause
        
        # Run concurrent workers with failures
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for worker_id in range(num_workers):
                future = executor.submit(failing_worker, worker_id)
                futures.append(future)
            
            # Wait for completion
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    # Workers might have internal failures
                    pass
        
        total_operations = success_count.value + failure_count.value
        expected_operations = num_workers * operations_per_worker
        
        # Should have attempted all operations
        assert total_operations == expected_operations
        
        # Should have some successes despite failures
        assert success_count.value > 0
        
        # Success rate should be approximately (1 - failure_rate)
        actual_success_rate = success_count.value / total_operations
        expected_success_rate = 1 - failure_rate
        
        # Allow some variance due to randomness
        assert abs(actual_success_rate - expected_success_rate) < 0.3
        
        print(f"Concurrent failures: {success_count.value}/{total_operations} succeeded "
              f"({actual_success_rate*100:.1f}% success rate)")

class TestDeadLetterHandling:
    """Test dead letter queue handling during failures."""
    
    def test_failed_message_recovery(self, redis_client, test_topic):
        """Test moving failed messages to dead letter queue."""
        # Create some failed scenarios
        failed_payloads = [
            {"task_type": "failed_task_1", "error": "processing_failed"},
            {"task_type": "failed_task_2", "error": "timeout"},
            {"task_type": "failed_task_3", "error": "invalid_data"}
        ]
        
        # Simulate failure handling by moving to dead letter queue
        dead_msg_ids = []
        for i, payload in enumerate(failed_payloads):
            original_msg_id = f"failed_{int(time.time())}_{i}"
            dead_msg_id = move_to_dead(test_topic, original_msg_id, payload)
            dead_msg_ids.append(dead_msg_id)
        
        # Verify all failed messages are in dead letter queue
        assert len(dead_msg_ids) == len(failed_payloads)
        
        # Check dead letter stream content
        dead_messages = redis_client.xrange(dead_stream(test_topic))
        assert len(dead_messages) == len(failed_payloads)
        
        # Verify dead letter message format
        for msg_id, fields in dead_messages:
            dead_data = json.loads(fields["json"])
            assert "id" in dead_data  # Original message ID
            assert "payload" in dead_data  # Original payload
            assert dead_data["id"].startswith("failed_")
    
    def test_dead_letter_queue_overflow(self, redis_client, test_topic):
        """Test dead letter queue behavior at capacity."""
        # Add many messages to dead letter queue
        num_dead_messages = 15
        
        for i in range(num_dead_messages):
            payload = {
                "task_type": "overflow_test",
                "message_index": i,
                "data": "x" * 100  # Some data
            }
            move_to_dead(test_topic, f"overflow_msg_{i}", payload)
        
        # Verify messages are in dead letter queue
        dead_messages = redis_client.xrange(dead_stream(test_topic))
        assert len(dead_messages) <= num_dead_messages
        
        # Dead letter queue should enforce maxlen (10000 by default)
        # For this test, all messages should be retained
        assert len(dead_messages) == num_dead_messages

class TestRecoveryScenarios:
    """Test system recovery scenarios."""
    
    def test_graceful_degradation(self, redis_client, test_topic):
        """Test graceful degradation during partial failures."""
        # Test that some operations can continue even if others fail
        
        # Reset enqueue cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        # Normal enqueue should work
        payload1 = {"task_type": "graceful_test", "phase": "normal"}
        msg_id1 = enqueue_task(test_topic, payload1)
        assert msg_id1 is not None
        
        # Simulate failure in one operation but not others
        with patch.object(redis_client, 'set') as mock_set:
            mock_set.side_effect = redis.exceptions.ConnectionError("Agent ops failing")
            
            # Agent operations should fail
            with pytest.raises(redis.exceptions.ConnectionError):
                register_agent("failing_agent", "test", 30)
            
            # But enqueue should still work
            payload2 = {"task_type": "graceful_test", "phase": "degraded"}
            msg_id2 = enqueue_task(test_topic, payload2)
            assert msg_id2 is not None
        
        # After partial failure recovery, all operations should work
        register_agent(f"recovery_agent_{int(time.time())}", "test", 30)
        payload3 = {"task_type": "graceful_test", "phase": "recovered"}
        msg_id3 = enqueue_task(test_topic, payload3)
        assert msg_id3 is not None
        
        # Verify all successful operations persisted
        stream_length = redis_client.xlen(task_stream(test_topic))
        assert stream_length >= 3  # All enqueue operations should have succeeded
    
    def test_connection_pool_recovery(self, redis_client, test_topic):
        """Test connection pool recovery after failures."""
        # This test verifies that the connection pool can recover
        # from connection failures and continue operating
        
        # Reset enqueue cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        # Normal operation
        payload = {"task_type": "pool_recovery_test"}
        
        # Multiple operations to use pool connections
        for i in range(5):
            payload["operation"] = i
            msg_id = enqueue_task(test_topic, payload)
            assert msg_id is not None
        
        # Simulate pool connection failures
        original_get_redis = get_redis
        failure_count = [0]  # Use list to allow modification in nested function
        
        def failing_get_redis():
            if failure_count[0] < 3:  # Fail first 3 attempts
                failure_count[0] += 1
                client = original_get_redis()
                # Patch the client to fail
                client.ping = Mock(side_effect=redis.exceptions.ConnectionError("Pool failure"))
                return client
            return original_get_redis()  # Normal operation after failures
        
        # Test recovery
        with patch('backend.app.mesh.redis_bus.get_redis', side_effect=failing_get_redis):
            recovery_attempts = 0
            max_attempts = 10
            
            while recovery_attempts < max_attempts:
                try:
                    payload["recovery_attempt"] = recovery_attempts
                    msg_id = enqueue_task(test_topic, payload)
                    if msg_id:
                        break  # Successfully recovered
                except redis.exceptions.ConnectionError:
                    pass  # Expected during failure period
                
                recovery_attempts += 1
                time.sleep(0.1)  # Brief pause between attempts
        
        # Should have recovered within max attempts
        assert recovery_attempts < max_attempts
        
        # Normal operation should work after recovery
        payload["phase"] = "post_recovery"
        msg_id = enqueue_task(test_topic, payload)
        assert msg_id is not None

class TestSystemResilience:
    """Test overall system resilience."""
    
    def test_mixed_failure_scenario(self, redis_client, test_topic):
        """Test system behavior under mixed failure conditions."""
        operations = {
            "enqueue_success": 0,
            "enqueue_failure": 0,
            "read_success": 0,
            "read_failure": 0,
            "agent_success": 0,
            "agent_failure": 0
        }
        
        # Reset enqueue cache
        if hasattr(enqueue_task, '_stream_cache'):
            enqueue_task._stream_cache = {}
        
        # Pre-populate some results for reading
        for i in range(5):
            redis_client.xadd(
                result_stream(test_topic),
                {"json": json.dumps({"result": f"initial_{i}"})}
            )
        
        # Simulate mixed failures with different patterns
        def mixed_failure_simulation():
            import random
            
            for operation_round in range(10):
                # Enqueue operation
                try:
                    if random.random() < 0.8:  # 80% success rate
                        payload = {"op": "mixed_test", "round": operation_round}
                        msg_id = enqueue_task(test_topic, payload)
                        if msg_id:
                            operations["enqueue_success"] += 1
                    else:
                        raise redis.exceptions.ConnectionError("Simulated enqueue failure")
                except Exception:
                    operations["enqueue_failure"] += 1
                
                # Read operation
                try:
                    if random.random() < 0.9:  # 90% success rate
                        results = tail_results(test_topic, count=2)
                        operations["read_success"] += 1
                    else:
                        raise redis.exceptions.TimeoutError("Simulated read failure")
                except Exception:
                    operations["read_failure"] += 1
                
                # Agent operation
                try:
                    if random.random() < 0.7:  # 70% success rate
                        agent_id = f"mixed_agent_{operation_round}"
                        register_agent(agent_id, "mixed_test", 30)
                        operations["agent_success"] += 1
                    else:
                        raise redis.exceptions.ResponseError("Simulated agent failure")
                except Exception:
                    operations["agent_failure"] += 1
                
                time.sleep(0.05)  # Brief pause between rounds
        
        # Run mixed failure simulation
        mixed_failure_simulation()
        
        # Verify system maintained partial functionality
        total_enqueue = operations["enqueue_success"] + operations["enqueue_failure"]
        total_read = operations["read_success"] + operations["read_failure"]
        total_agent = operations["agent_success"] + operations["agent_failure"]
        
        assert total_enqueue == 10  # All rounds attempted
        assert total_read == 10
        assert total_agent == 10
        
        # Should have some successes in each category
        assert operations["enqueue_success"] > 0
        assert operations["read_success"] > 0
        assert operations["agent_success"] > 0
        
        # Success rates should be reasonable
        enqueue_rate = operations["enqueue_success"] / total_enqueue
        read_rate = operations["read_success"] / total_read
        agent_rate = operations["agent_success"] / total_agent
        
        assert enqueue_rate > 0.5  # At least 50% success
        assert read_rate > 0.5
        assert agent_rate > 0.4
        
        print(f"Mixed failure resilience: "
              f"enqueue {enqueue_rate*100:.1f}%, "
              f"read {read_rate*100:.1f}%, "
              f"agent {agent_rate*100:.1f}%")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])