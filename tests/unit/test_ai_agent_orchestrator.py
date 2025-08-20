#!/usr/bin/env python3
"""
Unit and integration tests for AI Agent Orchestrator
"""
import asyncio
import json
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import sys
import os

# Add paths for imports
sys.path.append('/opt/sutazaiapp/agents')
sys.path.append('/opt/sutazaiapp/agents/ai_agent_orchestrator')

from ai_agent_orchestrator.app import (
    AIAgentOrchestrator, RegisteredAgent, TaskAssignment,
    AgentCapability, TaskRequest, Priority, AGENT_ID
)
from core.messaging import TaskMessage, StatusMessage, MessageType


class TestAIAgentOrchestrator:
    """Test suite for AI Agent Orchestrator"""
    
    @pytest.fixture
    async def orchestrator(self):
        """Create orchestrator instance with Mocked dependencies"""
        orch = AIAgentOrchestrator()
        
        # Mock Redis client
        orch.redis_client = AsyncMock()
        orch.redis_client.ping = AsyncMock(return_value=True)
        orch.redis_client.keys = AsyncMock(return_value=[])
        orch.redis_client.setex = AsyncMock()
        orch.redis_client.get = AsyncMock(return_value=None)
        orch.redis_client.delete = AsyncMock()
        
        # Mock message processor
        orch.message_processor = AsyncMock()
        orch.message_processor.start = AsyncMock()
        orch.message_processor.stop = AsyncMock()
        orch.message_processor.rabbitmq_client = AsyncMock()
        
        return orch
    
    @pytest.mark.asyncio
    async def test_register_agent_success(self, orchestrator):
        """Test successful agent registration"""
        agent_data = {
            "agent_id": "test-agent-1",
            "agent_type": "processor",
            "capabilities": [
                {"capability_type": "nlp", "proficiency": 0.9},
                {"capability_type": "data_analysis", "proficiency": 0.8}
            ],
            "endpoint": "http://test-agent:8080"
        }
        
        agent = await orchestrator.register_agent(agent_data)
        
        assert agent.agent_id == "test-agent-1"
        assert agent.agent_type == "processor"
        assert len(agent.capabilities) == 2
        assert agent.status == "online"
        assert orchestrator.registered_agents["test-agent-1"] == agent
        assert "nlp" in orchestrator.agent_capabilities
        assert "test-agent-1" in orchestrator.agent_capabilities["nlp"]
    
    @pytest.mark.asyncio
    async def test_find_best_agent_with_capabilities(self, orchestrator):
        """Test finding best agent based on capabilities"""
        # Register test agents
        agent1 = RegisteredAgent(
            agent_id="agent-1",
            agent_type="processor",
            capabilities=[
                AgentCapability(capability_type="nlp", proficiency=0.9),
                AgentCapability(capability_type="ml", proficiency=0.7)
            ],
            endpoint="http://agent1:8080",
            status="online",
            current_load=0.3,
            tasks_completed=10,
            tasks_failed=1
        )
        
        agent2 = RegisteredAgent(
            agent_id="agent-2",
            agent_type="processor",
            capabilities=[
                AgentCapability(capability_type="nlp", proficiency=0.8)
            ],
            endpoint="http://agent2:8080",
            status="online",
            current_load=0.1,
            tasks_completed=5,
            tasks_failed=0
        )
        
        orchestrator.registered_agents = {
            "agent-1": agent1,
            "agent-2": agent2
        }
        orchestrator.agent_capabilities = {
            "nlp": {"agent-1", "agent-2"},
            "ml": {"agent-1"}
        }
        
        # Find best agent for NLP task
        best_agent = await orchestrator.find_best_agent("nlp_task", ["nlp"])
        
        assert best_agent is not None
        # Agent 2 should be selected due to lower load
        assert best_agent.agent_id == "agent-2"
    
    @pytest.mark.asyncio
    async def test_find_best_agent_no_available(self, orchestrator):
        """Test finding agent when none are available"""
        # Register offline agent
        agent = RegisteredAgent(
            agent_id="agent-1",
            agent_type="processor",
            capabilities=[],
            endpoint="http://agent1:8080",
            status="offline"
        )
        
        orchestrator.registered_agents = {"agent-1": agent}
        
        best_agent = await orchestrator.find_best_agent("task", [])
        assert best_agent is None
    
    @pytest.mark.asyncio
    async def test_store_and_get_assignment(self, orchestrator):
        """Test storing and retrieving task assignments"""
        assignment = TaskAssignment(
            task_id="task-123",
            task_type="processing",
            assigned_agent="agent-1",
            priority=Priority.HIGH,
            status="pending"
        )
        
        await orchestrator.store_assignment(assignment)
        
        # Verify stored in memory
        assert orchestrator.task_assignments["task-123"] == assignment
        
        # Verify Redis call
        orchestrator.redis_client.setex.assert_called_once()
        
        # Test retrieval
        retrieved = await orchestrator.get_assignment("task-123")
        assert retrieved == assignment
    
    @pytest.mark.asyncio
    async def test_update_task_status_completed(self, orchestrator):
        """Test updating task status to completed"""
        # Create initial assignment
        assignment = TaskAssignment(
            task_id="task-456",
            task_type="analysis",
            assigned_agent="agent-1",
            priority=Priority.NORMAL,
            status="processing",
            started_at=datetime.utcnow()
        )
        
        # Create agent
        agent = RegisteredAgent(
            agent_id="agent-1",
            agent_type="processor",
            capabilities=[],
            endpoint="http://agent1:8080",
            current_load=0.5,
            tasks_completed=5,
            tasks_failed=2
        )
        
        orchestrator.task_assignments["task-456"] = assignment
        orchestrator.registered_agents["agent-1"] = agent
        
        # Update status to completed
        await orchestrator.update_task_status("task-456", "completed", {"result": "success"})
        
        # Verify assignment updated
        assert assignment.status == "completed"
        assert assignment.completed_at is not None
        
        # Verify agent metrics updated
        assert agent.current_load == 0.4  # Reduced by 0.1
        assert agent.tasks_completed == 6
    
    @pytest.mark.asyncio
    async def test_update_task_status_failed(self, orchestrator):
        """Test updating task status to failed"""
        assignment = TaskAssignment(
            task_id="task-789",
            task_type="processing",
            assigned_agent="agent-2",
            priority=Priority.LOW,
            status="processing"
        )
        
        agent = RegisteredAgent(
            agent_id="agent-2",
            agent_type="processor",
            capabilities=[],
            endpoint="http://agent2:8080",
            current_load=0.6,
            tasks_completed=10,
            tasks_failed=3
        )
        
        orchestrator.task_assignments["task-789"] = assignment
        orchestrator.registered_agents["agent-2"] = agent
        
        await orchestrator.update_task_status("task-789", "failed", {"error": "timeout"})
        
        assert assignment.status == "failed"
        assert agent.tasks_failed == 4
        assert agent.current_load == 0.5
    
    @pytest.mark.asyncio
    async def test_retry_task(self, orchestrator):
        """Test retrying a failed task"""
        assignment = TaskAssignment(
            task_id="task-retry",
            task_type="processing",
            assigned_agent="agent-1",
            priority=Priority.NORMAL,
            status="failed",
            retry_count=1
        )
        
        # Create available agent
        agent = RegisteredAgent(
            agent_id="agent-2",
            agent_type="processor",
            capabilities=[],
            endpoint="http://agent2:8080",
            status="online",
            current_load=0.2
        )
        
        orchestrator.registered_agents["agent-2"] = agent
        orchestrator.task_assignments["task-retry"] = assignment
        
        await orchestrator.retry_task(assignment)
        
        # Verify retry count increased
        assert assignment.retry_count == 2
        assert assignment.status == "pending"
        assert assignment.assigned_agent == "agent-2"
        
        # Verify task republished
        orchestrator.message_processor.rabbitmq_client.publish_task.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_update_agent_status_existing(self, orchestrator):
        """Test updating status of existing agent"""
        agent = RegisteredAgent(
            agent_id="agent-existing",
            agent_type="processor",
            capabilities=[],
            endpoint="http://existing:8080",
            status="online",
            current_load=0.5
        )
        
        orchestrator.registered_agents["agent-existing"] = agent
        
        await orchestrator.update_agent_status(
            "agent-existing", 
            "online", 
            0.7,
            [{"capability_type": "new_cap"}]
        )
        
        assert agent.status == "online"
        assert agent.current_load == 0.7
        assert agent.last_heartbeat is not None
    
    @pytest.mark.asyncio
    async def test_update_agent_status_new_agent(self, orchestrator):
        """Test auto-registering new agent from heartbeat"""
        await orchestrator.update_agent_status(
            "agent-new",
            "online",
            0.0,
            [{"capability_type": "processing"}]
        )
        
        assert "agent-new" in orchestrator.registered_agents
        agent = orchestrator.registered_agents["agent-new"]
        assert agent.status == "online"
        assert agent.current_load == 0.0
    
    @pytest.mark.asyncio
    async def test_health_monitor_marks_offline(self, orchestrator):
        """Test health monitor marking stale agents offline"""
        # Create agent with old heartbeat
        old_time = datetime.utcnow() - timedelta(minutes=5)
        agent = RegisteredAgent(
            agent_id="agent-stale",
            agent_type="processor",
            capabilities=[],
            endpoint="http://stale:8080",
            status="online",
            last_heartbeat=old_time
        )
        
        orchestrator.registered_agents["agent-stale"] = agent
        orchestrator.running = True
        
        # Run one iteration of health monitor
        async def run_once():
            await orchestrator.health_monitor()
            orchestrator.running = False  # Stop after one iteration
        
        # Run with timeout
        try:
            await asyncio.wait_for(run_once(), timeout=1.0)
        except asyncio.TimeoutError:
            pass
        
        # Agent should be marked offline
        assert agent.status == "offline"
        assert agent.current_load == 0.0
    
    @pytest.mark.asyncio 
    async def test_get_status(self, orchestrator):
        """Test getting orchestrator status"""
        # Setup test data
        orchestrator.registered_agents = {
            "agent-1": RegisteredAgent(
                agent_id="agent-1",
                agent_type="processor",
                capabilities=[],
                endpoint="http://agent1:8080",
                status="online"
            ),
            "agent-2": RegisteredAgent(
                agent_id="agent-2",
                agent_type="processor",
                capabilities=[],
                endpoint="http://agent2:8080",
                status="offline"
            )
        }
        
        orchestrator.task_assignments = {
            "task-1": TaskAssignment(
                task_id="task-1",
                task_type="processing",
                assigned_agent="agent-1",
                priority=Priority.NORMAL,
                status="processing"
            ),
            "task-2": TaskAssignment(
                task_id="task-2",
                task_type="analysis",
                assigned_agent="agent-2",
                priority=Priority.HIGH,
                status="completed"
            ),
            "task-3": TaskAssignment(
                task_id="task-3",
                task_type="processing",
                assigned_agent="agent-1",
                priority=Priority.LOW,
                status="failed"
            )
        }
        
        status = await orchestrator.get_status()
        
        assert status["status"] == "healthy"
        assert status["registered_agents"] == 2
        assert status["online_agents"] == 1
        assert status["active_tasks"] == 1
        assert status["pending_tasks"] == 0
        assert status["completed_tasks"] == 1
        assert status["failed_tasks"] == 1


class TestOrchestratorIntegration:
    """Integration tests for orchestrator with real RabbitMQ"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_task_lifecycle(self):
        """Test complete task lifecycle with Mocked RabbitMQ"""
        with patch('aio_pika.connect_robust') as mock_connect:
            # Setup Mock RabbitMQ connection
            mock_connection = AsyncMock()
            mock_channel = AsyncMock()
            mock_exchange = AsyncMock()
            mock_queue = AsyncMock()
            
            mock_connect.return_value = mock_connection
            mock_connection.channel.return_value = mock_channel
            mock_channel.declare_exchange.return_value = mock_exchange
            mock_channel.declare_queue.return_value = mock_queue
            
            # Create orchestrator
            orchestrator = AIAgentOrchestrator()
            
            # Mock Redis
            with patch('redis.asyncio.from_url') as mock_redis:
                mock_redis_client = AsyncMock()
                mock_redis.return_value = mock_redis_client
                mock_redis_client.ping.return_value = True
                mock_redis_client.keys.return_value = []
                
                # Initialize
                await orchestrator.initialize()
                
                # Register an agent
                agent_data = {
                    "agent_id": "test-processor",
                    "agent_type": "processor",
                    "capabilities": [
                        {"capability_type": "text_processing", "proficiency": 0.95}
                    ]
                }
                
                agent = await orchestrator.register_agent(agent_data)
                assert agent.agent_id == "test-processor"
                
                # Submit a task
                task_msg = TaskMessage(
                    message_id="msg-123",
                    message_type=MessageType.TASK_REQUEST,
                    source_agent="client",
                    task_id="task-001",
                    task_type="text_processing",
                    payload={"text": "Process this"},
                    priority=Priority.NORMAL
                )
                
                # Process task request
                await orchestrator.message_processor.handle_task_request(task_msg.dict())
                
                # Verify assignment created
                assignment = await orchestrator.get_assignment("task-001")
                assert assignment is not None
                assert assignment.assigned_agent == "test-processor"
                
                # Simulate task completion
                await orchestrator.update_task_status(
                    "task-001",
                    "completed",
                    {"result": "processed"}
                )
                
                # Verify final state
                assert assignment.status == "completed"
                assert agent.tasks_completed == 1
                
                # Cleanup
                await orchestrator.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])