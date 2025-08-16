#!/usr/bin/env python3
"""
Unit and integration tests for Task Assignment Coordinator
"""
import asyncio
import json
import pytest
from datetime import datetime, timedelta
from unittest.Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test import Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test, AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test, patch, MagicRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test
import heapq
import sys
import os

# Add paths for imports
sys.path.append('/opt/sutazaiapp/agents')
sys.path.append('/opt/sutazaiapp/agents/task_assignment_coordinator')

from task_assignment_coordinator.app import (
    TaskAssignmentCoordinator, PrioritizedTask, QueuedTask,
    TaskMetrics, AssignmentStrategy, Priority, AGENT_ID
)
from core.messaging import TaskMessage, StatusMessage, MessageType


class TestTaskAssignmentCoordinator:
    """Test suite for Task Assignment Coordinator"""
    
    @pytest.fixture
    async def coordinator(self):
        """Create coordinator instance with Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Tested dependencies"""
        coord = TaskAssignmentCoordinator()
        
        # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test Redis client
        coord.redis_client = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        coord.redis_client.ping = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test(return_value=True)
        coord.redis_client.zadd = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        coord.redis_client.zrem = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        coord.redis_client.zrange = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test(return_value=[])
        coord.redis_client.hset = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        coord.redis_client.hget = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test(return_value=None)
        coord.redis_client.hdel = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        
        # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test message processor
        coord.message_processor = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        coord.message_processor.start = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        coord.message_processor.stop = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        coord.message_processor.rabbitmq_client = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        coord.message_processor.rabbitmq_client.publish_message = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        coord.message_processor.rabbitmq_client.publish_error = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        
        return coord
    
    @pytest.mark.asyncio
    async def test_enqueue_task_success(self, coordinator):
        """Test successful task enqueueing"""
        task_msg = TaskMessage(
            message_id="msg-001",
            message_type=MessageType.TASK_REQUEST,
            source_agent="client",
            task_id="task-001",
            task_type="processing",
            payload={"data": "test"},
            priority=Priority.HIGH
        )
        
        await coordinator.enqueue_task(task_msg)
        
        # Verify task added to queue
        assert len(coordinator.task_queue) == 1
        
        # Verify priority queue ordering (negative for max heap)
        task = coordinator.task_queue[0]
        assert task.priority == -Priority.HIGH
        assert task.task.task_id == "task-001"
        
        # Verify metrics updated
        assert coordinator.metrics.total_received == 1
        assert coordinator.metrics.queue_depth == 1
        
        # Verify Redis storage
        coordinator.redis_client.zadd.assert_called_once()
        coordinator.redis_client.hset.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_enqueue_task_queue_full(self, coordinator):
        """Test enqueueing when queue is full"""
        # Fill queue to max
        coordinator.task_queue = [Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test() for _ in range(10000)]
        
        task_msg = TaskMessage(
            message_id="msg-002",
            message_type=MessageType.TASK_REQUEST,
            source_agent="client",
            task_id="task-002",
            task_type="processing",
            payload={},
            priority=Priority.NORMAL
        )
        
        with pytest.raises(Exception, match="Queue full"):
            await coordinator.enqueue_task(task_msg)
    
    @pytest.mark.asyncio
    async def test_priority_queue_ordering(self, coordinator):
        """Test that tasks are dequeued in priority order"""
        # Add tasks with different priorities
        tasks = [
            TaskMessage(
                message_id=f"msg-{i}",
                message_type=MessageType.TASK_REQUEST,
                source_agent="client",
                task_id=f"task-{i}",
                task_type="processing",
                payload={},
                priority=priority
            )
            for i, priority in enumerate([
                Priority.LOW,
                Priority.CRITICAL,
                Priority.NORMAL,
                Priority.HIGH
            ])
        ]
        
        for task in tasks:
            await coordinator.enqueue_task(task)
        
        # Dequeue and verify order (CRITICAL, HIGH, NORMAL, LOW)
        dequeued_priorities = []
        while coordinator.task_queue:
            task = await coordinator.dequeue_task()
            dequeued_priorities.append(task.priority)
        
        assert dequeued_priorities == [
            Priority.CRITICAL,
            Priority.HIGH, 
            Priority.NORMAL,
            Priority.LOW
        ]
    
    @pytest.mark.asyncio
    async def test_select_agent_round_robin(self, coordinator):
        """Test round-robin agent selection strategy"""
        coordinator.strategy.strategy_type = "round_robin"
        coordinator.agent_load = {
            "agent-1": 0.5,
            "agent-2": 0.3,
            "agent-3": 0.7
        }
        
        task = TaskMessage(
            message_id="msg-rr",
            message_type=MessageType.TASK_REQUEST,
            source_agent="client",
            task_id="task-rr",
            task_type="processing",
            payload={},
            priority=Priority.NORMAL
        )
        
        agent = await coordinator.select_agent(task)
        
        # Should select first available agent
        assert agent in ["agent-1", "agent-2", "agent-3"]
    
    @pytest.mark.asyncio
    async def test_select_agent_least_loaded(self, coordinator):
        """Test least-loaded agent selection strategy"""
        coordinator.strategy.strategy_type = "least_loaded"
        coordinator.agent_load = {
            "agent-1": 0.6,
            "agent-2": 0.2,  # Least loaded
            "agent-3": 0.8
        }
        
        task = TaskMessage(
            message_id="msg-ll",
            message_type=MessageType.TASK_REQUEST,
            source_agent="client",
            task_id="task-ll",
            task_type="processing",
            payload={},
            priority=Priority.NORMAL
        )
        
        agent = await coordinator.select_agent(task)
        assert agent == "agent-2"
    
    @pytest.mark.asyncio
    async def test_select_agent_capability_match(self, coordinator):
        """Test capability-based agent selection"""
        coordinator.strategy.strategy_type = "capability_match"
        coordinator.agent_load = {
            "agent-1": 0.5,
            "agent-2": 0.3,
            "agent-3": 0.4
        }
        coordinator.agent_capabilities = {
            "agent-1": ["nlp", "ml"],
            "agent-2": ["data_processing"],
            "agent-3": ["nlp", "ml", "data_processing"]  # Most capable
        }
        
        task = TaskMessage(
            message_id="msg-cm",
            message_type=MessageType.TASK_REQUEST,
            source_agent="client",
            task_id="task-cm",
            task_type="nlp",
            payload={},
            priority=Priority.NORMAL
        )
        
        agent = await coordinator.select_agent(task)
        # Should select agent-3 (most capabilities with nlp)
        assert agent == "agent-3"
    
    @pytest.mark.asyncio
    async def test_select_agent_no_available(self, coordinator):
        """Test agent selection when all are overloaded"""
        coordinator.agent_load = {
            "agent-1": 0.9,
            "agent-2": 0.95,
            "agent-3": 0.85
        }
        coordinator.strategy.load_threshold = 0.8
        
        task = TaskMessage(
            message_id="msg-na",
            message_type=MessageType.TASK_REQUEST,
            source_agent="client",
            task_id="task-na",
            task_type="processing",
            payload={},
            priority=Priority.NORMAL
        )
        
        agent = await coordinator.select_agent(task)
        assert agent is None
    
    @pytest.mark.asyncio
    async def test_handle_failed_task_retry(self, coordinator):
        """Test retrying failed tasks"""
        # Create a queued task
        queued_task = QueuedTask(
            task_id="task-fail",
            task_type="processing",
            payload={"data": "test"},
            priority=Priority.NORMAL,
            source_agent="client",
            retry_count=1
        )
        
        coordinator.processing_tasks["task-fail"] = queued_task
        coordinator.strategy.max_retries = 3
        
        await coordinator.handle_failed_task("task-fail")
        
        # Verify task requeued
        assert len(coordinator.task_queue) == 1
        
        # Verify retry count incremented
        assert queued_task.retry_count == 2
        
        # Task should be removed from processing
        assert "task-fail" not in coordinator.processing_tasks
    
    @pytest.mark.asyncio
    async def test_handle_failed_task_max_retries(self, coordinator):
        """Test handling task that exceeded max retries"""
        queued_task = QueuedTask(
            task_id="task-max-fail",
            task_type="processing",
            payload={},
            priority=Priority.NORMAL,
            source_agent="client",
            retry_count=3  # Already at max
        )
        
        coordinator.processing_tasks["task-max-fail"] = queued_task
        coordinator.strategy.max_retries = 3
        
        await coordinator.handle_failed_task("task-max-fail")
        
        # Task should not be requeued
        assert len(coordinator.task_queue) == 0
        
        # Metrics should show failure
        assert coordinator.metrics.total_failed == 1
        
        # Task removed from processing
        assert "task-max-fail" not in coordinator.processing_tasks
    
    @pytest.mark.asyncio
    async def test_register_agent_capabilities(self, coordinator):
        """Test registering agent capabilities"""
        await coordinator.register_agent_capabilities(
            "agent-new",
            ["nlp", "ml", "data_processing"]
        )
        
        assert "agent-new" in coordinator.agent_capabilities
        assert coordinator.agent_capabilities["agent-new"] == ["nlp", "ml", "data_processing"]
        assert coordinator.agent_load["agent-new"] == 0.0
        
        # Verify Redis storage
        coordinator.redis_client.hset.assert_called_with(
            "agent:capabilities",
            "agent-new",
            json.dumps(["nlp", "ml", "data_processing"])
        )
    
    @pytest.mark.asyncio
    async def test_update_metrics_completed(self, coordinator):
        """Test updating metrics for completed tasks"""
        # Setup initial metrics
        coordinator.metrics.total_completed = 5
        coordinator.metrics.average_processing_time = 10.0
        
        # Create a processing task
        queued_task = QueuedTask(
            task_id="task-complete",
            task_type="processing",
            payload={},
            priority=Priority.NORMAL,
            source_agent="client",
            queued_at=datetime.utcnow() - timedelta(seconds=15)
        )
        coordinator.processing_tasks["task-complete"] = queued_task
        coordinator.agent_load["agent-1"] = 0.5
        
        # Update metrics with completion
        status_msg = StatusMessage(
            message_id="status-001",
            message_type=MessageType.TASK_STATUS,
            source_agent="agent-1",
            task_id="task-complete",
            status="completed"
        )
        
        await coordinator.update_metrics(status_msg)
        
        assert coordinator.metrics.total_completed == 6
        # Average should be updated: (10.0 * 5 + 15) / 6 = 10.83...
        assert coordinator.metrics.average_processing_time == pytest.approx(10.83, 0.01)
        assert coordinator.agent_load["agent-1"] == 0.4
        assert "task-complete" not in coordinator.processing_tasks
    
    @pytest.mark.asyncio
    async def test_timeout_monitor(self, coordinator):
        """Test timeout monitoring for tasks"""
        # Create tasks with different timeout states
        current_time = datetime.utcnow()
        
        coordinator.processing_tasks = {
            "task-timeout-1": QueuedTask(
                task_id="task-timeout-1",
                task_type="processing",
                payload={},
                priority=Priority.NORMAL,
                source_agent="client",
                timeout_at=current_time - timedelta(seconds=10)  # Already timed out
            ),
            "task-ok": QueuedTask(
                task_id="task-ok",
                task_type="processing",
                payload={},
                priority=Priority.NORMAL,
                source_agent="client",
                timeout_at=current_time + timedelta(seconds=100)  # Not timed out
            )
        }
        
        coordinator.running = True
        coordinator.strategy.max_retries = 3
        
        # Run one iteration of timeout monitor
        async def run_once():
            # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test the handle_failed_task to avoid side effects
            coordinator.handle_failed_task = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
            
            # Run timeout check logic directly
            current_time = datetime.utcnow()
            timed_out_tasks = []
            
            for task_id, queued_task in coordinator.processing_tasks.items():
                if queued_task.timeout_at and current_time > queued_task.timeout_at:
                    timed_out_tasks.append(task_id)
                    coordinator.metrics.total_timeout += 1
            
            for task_id in timed_out_tasks:
                await coordinator.handle_failed_task(task_id)
        
        await run_once()
        
        # Verify timeout detected and handled
        assert coordinator.metrics.total_timeout == 1
        coordinator.handle_failed_task.assert_called_once_with("task-timeout-1")
    
    @pytest.mark.asyncio
    async def test_process_assignments_batch(self, coordinator):
        """Test batch processing of assignments"""
        coordinator.strategy.batch_size = 3
        coordinator.agent_load = {
            "agent-1": 0.2,
            "agent-2": 0.3
        }
        
        # Add 5 tasks to queue
        for i in range(5):
            task = TaskMessage(
                message_id=f"msg-{i}",
                message_type=MessageType.TASK_REQUEST,
                source_agent="client",
                task_id=f"task-{i}",
                task_type="processing",
                payload={},
                priority=Priority.NORMAL
            )
            await coordinator.enqueue_task(task)
        
        await coordinator.process_assignments()
        
        # Should process only batch_size tasks
        assert len(coordinator.task_queue) == 2  # 5 - 3 = 2
        assert len(coordinator.processing_tasks) == 3
        assert coordinator.metrics.total_assigned == 3
    
    @pytest.mark.asyncio
    async def test_load_queued_tasks(self, coordinator):
        """Test loading queued tasks from Redis on startup"""
        # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test Redis data
        coordinator.redis_client.zrange.return_value = ["task-1", "task-2"]
        
        task1_data = TaskMessage(
            message_id="msg-1",
            message_type=MessageType.TASK_REQUEST,
            source_agent="client",
            task_id="task-1",
            task_type="processing",
            payload={},
            priority=Priority.HIGH
        ).json()
        
        task2_data = TaskMessage(
            message_id="msg-2",
            message_type=MessageType.TASK_REQUEST,
            source_agent="client",
            task_id="task-2",
            task_type="analysis",
            payload={},
            priority=Priority.NORMAL
        ).json()
        
        coordinator.redis_client.hget.side_effect = [task1_data, task2_data]
        
        await coordinator.load_queued_tasks()
        
        assert len(coordinator.task_queue) == 2
        assert coordinator.metrics.queue_depth == 2
        
        # Verify priority ordering
        task1 = heapq.heappop(coordinator.task_queue)
        assert task1.task.priority == Priority.HIGH


class TestCoordinatorIntegration:
    """Integration tests for coordinator with real message processing"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_task_flow(self):
        """Test complete task flow through coordinator"""
        with patch('aio_pika.connect_robust') as Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_connect:
            # Setup Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test RabbitMQ
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_connection = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_channel = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_exchange = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_queue = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
            
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_connect.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_connection
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_connection.channel.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_channel
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_channel.declare_exchange.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_exchange
            Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_channel.declare_queue.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_queue
            
            coordinator = TaskAssignmentCoordinator()
            
            # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test Redis
            with patch('redis.asyncio.from_url') as Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis:
                Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis_client = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
                Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis_client
                Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis_client.ping.return_value = True
                Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_redis_client.zrange.return_value = []
                
                # Initialize
                await coordinator.initialize()
                
                # Register agents
                await coordinator.register_agent_capabilities(
                    "processor-1",
                    ["text_processing", "data_analysis"]
                )
                await coordinator.register_agent_capabilities(
                    "processor-2",
                    ["ml_inference"]
                )
                
                # Submit tasks
                task1 = TaskMessage(
                    message_id="msg-int-1",
                    message_type=MessageType.TASK_REQUEST,
                    source_agent="client",
                    task_id="task-int-1",
                    task_type="text_processing",
                    payload={"text": "Process this"},
                    priority=Priority.HIGH
                )
                
                task2 = TaskMessage(
                    message_id="msg-int-2",
                    message_type=MessageType.TASK_REQUEST,
                    source_agent="client",
                    task_id="task-int-2",
                    task_type="ml_inference",
                    payload={"model": "bert"},
                    priority=Priority.NORMAL
                )
                
                # Process task requests
                await coordinator.message_processor.handle_task_request(task1.dict())
                await coordinator.message_processor.handle_task_request(task2.dict())
                
                # Verify queue state
                assert coordinator.metrics.total_received == 2
                assert coordinator.metrics.queue_depth == 2
                
                # Process assignments
                await coordinator.process_assignments()
                
                # Verify assignments made
                assert coordinator.metrics.total_assigned > 0
                
                # Simulate task completion
                status = StatusMessage(
                    message_id="status-int-1",
                    message_type=MessageType.TASK_STATUS,
                    source_agent="processor-1",
                    task_id="task-int-1",
                    status="completed"
                )
                
                await coordinator.message_processor.handle_status_update(status.dict())
                
                # Verify metrics updated
                assert coordinator.metrics.total_completed == 1
                
                # Cleanup
                await coordinator.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])