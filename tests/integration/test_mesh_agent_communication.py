"""
Integration tests for mesh agent communication and workflows.
Tests end-to-end agent communication through Redis mesh system.
"""
import json
import time
import asyncio
import threading
import pytest
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import mesh components
from app.mesh.redis_bus import (
    get_redis, enqueue_task, tail_results, register_agent, 
    heartbeat_agent, list_agents, create_consumer_group, 
    read_group, ack, move_to_dead, task_stream, result_stream
)

class MockAgent:
    """Mock agent for testing communication patterns."""
    
    def __init__(self, agent_id: str, agent_type: str, capabilities: List[str]):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities
        self.is_running = False
        self.processed_tasks = []
        self.errors = []
        self.redis_client = None
        
    def start(self):
        """Start the Mock agent."""
        self.is_running = True
        self.redis_client = get_redis()
        
        # Register agent
        register_agent(
            self.agent_id,
            self.agent_type,
            ttl_seconds=120,
            meta={
                "capabilities": self.capabilities,
                "status": "active",
                "version": "1.0.0"
            }
        )
    
    def stop(self):
        """Stop the Mock agent."""
        self.is_running = False
        
    def process_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task and return result."""
        try:
            task_type = task_data.get("task_type", "unknown")
            
            if task_type == "echo":
                result = {
                    "status": "completed",
                    "result": task_data.get("input_data", ""),
                    "processed_by": self.agent_id,
                    "timestamp": time.time()
                }
            elif task_type == "transform":
                input_data = task_data.get("input_data", "")
                result = {
                    "status": "completed", 
                    "result": input_data.upper(),
                    "processed_by": self.agent_id,
                    "timestamp": time.time()
                }
            elif task_type == "error":
                raise Exception("Simulated processing error")
            elif task_type == "slow":
                time.sleep(task_data.get("delay", 1))
                result = {
                    "status": "completed",
                    "result": "slow task completed",
                    "processed_by": self.agent_id,
                    "delay": task_data.get("delay", 1),
                    "timestamp": time.time()
                }
            else:
                result = {
                    "status": "error",
                    "error": f"Unknown task type: {task_type}",
                    "processed_by": self.agent_id,
                    "timestamp": time.time()
                }
            
            self.processed_tasks.append((task_data, result))
            return result
            
        except Exception as e:
            error_result = {
                "status": "error",
                "error": str(e),
                "processed_by": self.agent_id,
                "timestamp": time.time()
            }
            self.errors.append((task_data, str(e)))
            return error_result
    
    def run_worker(self, topic: str, group: str, max_tasks: int = 10):
        """Run agent worker loop."""
        consumer_name = f"{self.agent_id}_consumer"
        
        try:
            create_consumer_group(topic, group)
        except:
            pass  # Group might already exist
        
        tasks_processed = 0
        
        while self.is_running and tasks_processed < max_tasks:
            try:
                # Send heartbeat
                heartbeat_agent(self.agent_id, 120)
                
                # Read messages
                messages = read_group(topic, group, consumer_name, count=1, block_ms=1000)
                
                for msg_id, task_data in messages:
                    if not self.is_running:
                        break
                    
                    # Process task
                    result = self.process_task(task_data)
                    
                    # Send result
                    self.redis_client.xadd(
                        result_stream(topic),
                        {"json": json.dumps(result)}
                    )
                    
                    # Acknowledge task
                    ack(topic, group, msg_id)
                    
                    tasks_processed += 1
                    
                    if tasks_processed >= max_tasks:
                        break
                        
            except Exception as e:
                self.errors.append(("worker_error", str(e)))
                time.sleep(0.1)  # Brief pause on error

# Test fixtures
@pytest.fixture
def redis_client():
    """Get Redis client for integration tests."""
    try:
        client = get_redis()
        client.ping()
        return client
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")

@pytest.fixture
def test_topic():
    """Test topic with timestamp."""
    return f"agent_comm_test_{int(time.time())}"

@pytest.fixture
def test_group():
    """Test consumer group name."""
    return "test_agents"

@pytest.fixture
def mock_agents():
    """Create Mock agents for testing."""
    timestamp = int(time.time())
    agents = [
        MockAgent(f"echo_agent_{timestamp}", "echo", ["echo", "transform"]),
        MockAgent(f"transform_agent_{timestamp}", "transform", ["transform"]),
        MockAgent(f"slow_agent_{timestamp}", "slow", ["slow_processing"])
    ]
    return agents

@pytest.fixture(autouse=True)
def cleanup_test_data(redis_client, test_topic):
    """Cleanup test data before and after tests."""
    # Cleanup before
    try:
        redis_client.delete(task_stream(test_topic))
        redis_client.delete(result_stream(test_topic))
    except:
        pass
    
    yield
    
    # Cleanup after
    try:
        redis_client.delete(task_stream(test_topic))
        redis_client.delete(result_stream(test_topic))
        
        # Clean up consumer groups
        try:
            redis_client.xgroup_destroy(task_stream(test_topic), "test_agents")
        except:
            pass
    except:
        pass

class TestBasicAgentCommunication:
    """Test basic agent communication patterns."""
    
    def test_single_agent_task_processing(self, redis_client, test_topic, test_group, mock_agents):
        """Test single agent processing tasks."""
        agent = mock_agents[0]  # echo_agent
        agent.start()
        
        try:
            # Enqueue test tasks
            test_tasks = [
                {"task_type": "echo", "input_data": "hello world"},
                {"task_type": "transform", "input_data": "test string"}
            ]
            
            # Reset enqueue cache
            if hasattr(enqueue_task, '_stream_cache'):
                enqueue_task._stream_cache = {}
            
            for task in test_tasks:
                enqueue_task(test_topic, task)
            
            # Run agent worker
            agent.run_worker(test_topic, test_group, max_tasks=2)
            
            # Verify tasks were processed
            assert len(agent.processed_tasks) == 2
            assert len(agent.errors) == 0
            
            # Check results
            results = tail_results(test_topic, count=5)
            assert len(results) >= 2
            
            # Verify result content
            result_data = [result[1] for result in results]
            
            # Find echo result
            echo_result = next((r for r in result_data if r.get("result") == "hello world"), None)
            assert echo_result is not None
            assert echo_result["status"] == "completed"
            assert echo_result["processed_by"] == agent.agent_id
            
            # Find transform result
            transform_result = next((r for r in result_data if r.get("result") == "TEST STRING"), None)
            assert transform_result is not None
            assert transform_result["status"] == "completed"
            
        finally:
            agent.stop()
    
    def test_multiple_agents_task_distribution(self, redis_client, test_topic, test_group, mock_agents):
        """Test task distribution among multiple agents."""
        # Start multiple agents
        active_agents = mock_agents[:2]  # echo_agent and transform_agent
        
        for agent in active_agents:
            agent.start()
        
        try:
            # Enqueue multiple tasks
            test_tasks = [
                {"task_type": "echo", "input_data": f"message_{i}"} for i in range(5)
            ]
            
            # Reset enqueue cache
            if hasattr(enqueue_task, '_stream_cache'):
                enqueue_task._stream_cache = {}
            
            for task in test_tasks:
                enqueue_task(test_topic, task)
            
            # Run agents in parallel
            with ThreadPoolExecutor(max_workers=len(active_agents)) as executor:
                futures = []
                for agent in active_agents:
                    future = executor.submit(agent.run_worker, test_topic, test_group, 5)
                    futures.append(future)
                
                # Wait for completion
                for future in as_completed(futures, timeout=10):
                    try:
                        future.result()
                    except Exception as e:
                        pytest.fail(f"Agent worker failed: {e}")
            
            # Verify task distribution
            total_processed = sum(len(agent.processed_tasks) for agent in active_agents)
            assert total_processed == len(test_tasks)
            
            # Verify no errors
            total_errors = sum(len(agent.errors) for agent in active_agents)
            assert total_errors == 0
            
            # Check results
            results = tail_results(test_topic, count=10)
            assert len(results) >= len(test_tasks)
            
        finally:
            for agent in active_agents:
                agent.stop()
    
    def test_agent_heartbeat_and_registry(self, redis_client, mock_agents):
        """Test agent heartbeat and registry functionality."""
        agent = mock_agents[0]
        agent.start()
        
        try:
            # Check agent appears in registry
            agents = list_agents()
            test_agents = [a for a in agents if a.get("agent_id") == agent.agent_id]
            assert len(test_agents) == 1
            
            registered_agent = test_agents[0]
            assert registered_agent["agent_type"] == agent.agent_type
            assert registered_agent["meta"]["capabilities"] == agent.capabilities
            assert registered_agent["meta"]["status"] == "active"
            
            # Test heartbeat updates TTL
            initial_ttl = redis_client.ttl(f"mesh:agent:{agent.agent_id}")
            time.sleep(1)
            heartbeat_agent(agent.agent_id, 120)
            updated_ttl = redis_client.ttl(f"mesh:agent:{agent.agent_id}")
            
            # TTL should be refreshed
            assert updated_ttl > initial_ttl
            
        finally:
            agent.stop()

class TestErrorHandlingAndRecovery:
    """Test error handling and recovery scenarios."""
    
    def test_task_processing_errors(self, redis_client, test_topic, test_group, mock_agents):
        """Test handling of task processing errors."""
        agent = mock_agents[0]
        agent.start()
        
        try:
            # Enqueue tasks including error-inducing ones
            test_tasks = [
                {"task_type": "echo", "input_data": "success_task"},
                {"task_type": "error", "input_data": "this will fail"},
                {"task_type": "echo", "input_data": "another_success"}
            ]
            
            # Reset enqueue cache
            if hasattr(enqueue_task, '_stream_cache'):
                enqueue_task._stream_cache = {}
            
            for task in test_tasks:
                enqueue_task(test_topic, task)
            
            # Run agent worker
            agent.run_worker(test_topic, test_group, max_tasks=3)
            
            # Verify all tasks were processed
            assert len(agent.processed_tasks) == 3
            assert len(agent.errors) == 1  # One error should be recorded
            
            # Check results include both success and error
            results = tail_results(test_topic, count=5)
            result_data = [result[1] for result in results]
            
            success_results = [r for r in result_data if r.get("status") == "completed"]
            error_results = [r for r in result_data if r.get("status") == "error"]
            
            assert len(success_results) >= 2
            assert len(error_results) >= 1
            
        finally:
            agent.stop()
    
    def test_dead_letter_queue_usage(self, redis_client, test_topic, test_group, mock_agents):
        """Test moving failed messages to dead letter queue."""
        agent = mock_agents[0]
        agent.start()
        
        try:
            # Enqueue a task
            task_data = {"task_type": "echo", "input_data": "test_message"}
            
            # Reset enqueue cache
            if hasattr(enqueue_task, '_stream_cache'):
                enqueue_task._stream_cache = {}
            
            msg_id = enqueue_task(test_topic, task_data)
            
            # Manually move to dead letter queue
            dead_msg_id = move_to_dead(test_topic, msg_id, task_data)
            assert dead_msg_id is not None
            
            # Verify message in dead letter stream
            dead_stream_key = f"stream:dead:{test_topic}"
            dead_messages = redis_client.xrange(dead_stream_key)
            assert len(dead_messages) == 1
            
            dead_data = json.loads(dead_messages[0][1]["json"])
            assert dead_data["id"] == msg_id
            assert dead_data["payload"] == task_data
            
        finally:
            agent.stop()
    
    def test_agent_recovery_after_failure(self, redis_client, test_topic, test_group, mock_agents):
        """Test agent recovery after simulated failure."""
        agent = mock_agents[0]
        agent.start()
        
        try:
            # Enqueue initial tasks
            # Reset enqueue cache
            if hasattr(enqueue_task, '_stream_cache'):
                enqueue_task._stream_cache = {}
            
            for i in range(3):
                enqueue_task(test_topic, {"task_type": "echo", "input_data": f"message_{i}"})
            
            # Process some tasks
            agent.run_worker(test_topic, test_group, max_tasks=2)
            
            # Simulate agent restart
            agent.stop()
            time.sleep(0.5)
            agent.start()
            
            # Agent should re-register and be able to process remaining tasks
            remaining_tasks = 1
            agent.run_worker(test_topic, test_group, max_tasks=remaining_tasks)
            
            # Verify agent is still in registry after restart
            agents = list_agents()
            test_agents = [a for a in agents if a.get("agent_id") == agent.agent_id]
            assert len(test_agents) == 1
            
        finally:
            agent.stop()

class TestPerformanceAndScalability:
    """Test performance and scalability scenarios."""
    
    def test_high_throughput_processing(self, redis_client, test_topic, test_group, mock_agents):
        """Test high-throughput task processing."""
        agent = mock_agents[0]
        agent.start()
        
        try:
            # Enqueue many tasks
            num_tasks = 50
            start_time = time.time()
            
            # Reset enqueue cache
            if hasattr(enqueue_task, '_stream_cache'):
                enqueue_task._stream_cache = {}
            
            for i in range(num_tasks):
                enqueue_task(test_topic, {
                    "task_type": "echo", 
                    "input_data": f"bulk_message_{i}",
                    "batch_id": "high_throughput_test"
                })
            
            enqueue_time = time.time() - start_time
            
            # Process all tasks
            start_time = time.time()
            agent.run_worker(test_topic, test_group, max_tasks=num_tasks)
            processing_time = time.time() - start_time
            
            # Verify all tasks processed
            assert len(agent.processed_tasks) == num_tasks
            assert len(agent.errors) == 0
            
            # Performance assertions
            assert enqueue_time < 10.0  # Should enqueue 50 tasks in < 10 seconds
            assert processing_time < 30.0  # Should process 50 tasks in < 30 seconds
            
            # Verify results
            results = tail_results(test_topic, count=num_tasks)
            assert len(results) >= num_tasks
            
        finally:
            agent.stop()
    
    def test_concurrent_agent_processing(self, redis_client, test_topic, test_group, mock_agents):
        """Test concurrent processing by multiple agents."""
        # Use multiple agents
        active_agents = mock_agents[:3]
        
        for agent in active_agents:
            agent.start()
        
        try:
            # Enqueue tasks for concurrent processing
            num_tasks = 30
            
            # Reset enqueue cache
            if hasattr(enqueue_task, '_stream_cache'):
                enqueue_task._stream_cache = {}
            
            for i in range(num_tasks):
                task_type = "echo" if i % 2 == 0 else "transform"
                enqueue_task(test_topic, {
                    "task_type": task_type,
                    "input_data": f"concurrent_task_{i}"
                })
            
            # Run agents concurrently
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=len(active_agents)) as executor:
                futures = []
                for agent in active_agents:
                    future = executor.submit(agent.run_worker, test_topic, test_group, 15)
                    futures.append(future)
                
                # Wait for completion
                for future in as_completed(futures, timeout=20):
                    try:
                        future.result()
                    except Exception as e:
                        pytest.fail(f"Concurrent agent failed: {e}")
            
            processing_time = time.time() - start_time
            
            # Verify all tasks processed
            total_processed = sum(len(agent.processed_tasks) for agent in active_agents)
            total_errors = sum(len(agent.errors) for agent in active_agents)
            
            assert total_processed == num_tasks
            assert total_errors == 0
            
            # Concurrent processing should be faster than sequential
            assert processing_time < 15.0  # Should be faster with concurrency
            
            # Verify load distribution
            for agent in active_agents:
                # Each agent should process some tasks
                assert len(agent.processed_tasks) > 0
                
        finally:
            for agent in active_agents:
                agent.stop()
    
    def test_slow_task_handling(self, redis_client, test_topic, test_group, mock_agents):
        """Test handling of slow/long-running tasks."""
        agent = mock_agents[2]  # slow_agent
        agent.start()
        
        try:
            # Enqueue slow tasks
            slow_tasks = [
                {"task_type": "slow", "delay": 0.5},
                {"task_type": "slow", "delay": 1.0},
                {"task_type": "slow", "delay": 0.3}
            ]
            
            # Reset enqueue cache
            if hasattr(enqueue_task, '_stream_cache'):
                enqueue_task._stream_cache = {}
            
            for task in slow_tasks:
                enqueue_task(test_topic, task)
            
            # Process tasks
            start_time = time.time()
            agent.run_worker(test_topic, test_group, max_tasks=3)
            total_time = time.time() - start_time
            
            # Verify all tasks completed
            assert len(agent.processed_tasks) == 3
            assert len(agent.errors) == 0
            
            # Should take at least the sum of delays
            expected_min_time = 0.5 + 1.0 + 0.3
            assert total_time >= expected_min_time
            
            # Check results
            results = tail_results(test_topic, count=5)
            result_data = [result[1] for result in results]
            
            slow_results = [r for r in result_data if "delay" in r]
            assert len(slow_results) >= 3
            
        finally:
            agent.stop()

class TestMessageReliability:
    """Test message delivery reliability."""
    
    def test_message_acknowledgment(self, redis_client, test_topic, test_group, mock_agents):
        """Test proper message acknowledgment."""
        agent = mock_agents[0]
        agent.start()
        
        try:
            # Enqueue tasks
            # Reset enqueue cache
            if hasattr(enqueue_task, '_stream_cache'):
                enqueue_task._stream_cache = {}
            
            for i in range(3):
                enqueue_task(test_topic, {"task_type": "echo", "input_data": f"ack_test_{i}"})
            
            # Process tasks (agent should acknowledge them)
            agent.run_worker(test_topic, test_group, max_tasks=3)
            
            # Verify tasks were acknowledged (should not be in pending)
            try:
                create_consumer_group(test_topic, test_group)
            except:
                pass
                
            # Check pending messages - should be empty after acknowledgment
            group_info = redis_client.xinfo_groups(task_stream(test_topic))
            test_group_info = next((g for g in group_info if g['name'] == test_group), None)
            
            if test_group_info:
                # Pending count should be 0 after acknowledgment
                assert test_group_info['pending'] == 0
            
        finally:
            agent.stop()
    
    def test_unacknowledged_message_recovery(self, redis_client, test_topic, test_group, mock_agents):
        """Test recovery of unacknowledged messages."""
        agent = mock_agents[0]
        
        try:
            # Create consumer group
            create_consumer_group(test_topic, test_group)
            
            # Enqueue a task
            # Reset enqueue cache
            if hasattr(enqueue_task, '_stream_cache'):
                enqueue_task._stream_cache = {}
            
            task_data = {"task_type": "echo", "input_data": "unack_test"}
            enqueue_task(test_topic, task_data)
            
            # Read message without acknowledgment
            consumer_name = f"{agent.agent_id}_consumer"
            messages = read_group(test_topic, test_group, consumer_name, count=1, block_ms=100)
            
            assert len(messages) == 1
            msg_id, data = messages[0]
            assert data == task_data
            
            # Check pending messages - should have 1 pending
            group_info = redis_client.xinfo_groups(task_stream(test_topic))
            test_group_info = next((g for g in group_info if g['name'] == test_group), None)
            
            if test_group_info:
                assert test_group_info['pending'] == 1
            
            # Acknowledge the message
            ack_count = ack(test_topic, test_group, msg_id)
            assert ack_count == 1
            
            # Pending count should now be 0
            group_info = redis_client.xinfo_groups(task_stream(test_topic))
            test_group_info = next((g for g in group_info if g['name'] == test_group), None)
            
            if test_group_info:
                assert test_group_info['pending'] == 0
                
        finally:
            agent.stop()

class TestAgentLifecycle:
    """Test complete agent lifecycle scenarios."""
    
    def test_agent_registration_expiration(self, redis_client, mock_agents):
        """Test agent registration and expiration."""
        agent = mock_agents[0]
        
        # Register with short TTL
        register_agent(agent.agent_id, agent.agent_type, ttl_seconds=2, meta={"test": True})
        
        # Should appear in agent list
        agents = list_agents()
        test_agents = [a for a in agents if a.get("agent_id") == agent.agent_id]
        assert len(test_agents) == 1
        
        # Wait for expiration
        time.sleep(3)
        
        # Should no longer appear in agent list
        agents = list_agents()
        test_agents = [a for a in agents if a.get("agent_id") == agent.agent_id]
        assert len(test_agents) == 0
    
    def test_agent_graceful_shutdown(self, redis_client, test_topic, test_group, mock_agents):
        """Test graceful agent shutdown."""
        agent = mock_agents[0]
        agent.start()
        
        try:
            # Enqueue tasks
            # Reset enqueue cache
            if hasattr(enqueue_task, '_stream_cache'):
                enqueue_task._stream_cache = {}
            
            for i in range(5):
                enqueue_task(test_topic, {"task_type": "echo", "input_data": f"shutdown_test_{i}"})
            
            # Start processing in background
            def worker_thread():
                agent.run_worker(test_topic, test_group, max_tasks=10)
            
            thread = threading.Thread(target=worker_thread)
            thread.start()
            
            # Let it process some tasks
            time.sleep(1)
            
            # Signal shutdown
            agent.stop()
            
            # Wait for graceful completion
            thread.join(timeout=5)
            
            # Agent should have processed some tasks before shutdown
            assert len(agent.processed_tasks) > 0
            assert len(agent.processed_tasks) <= 5
            
        finally:
            agent.stop()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])