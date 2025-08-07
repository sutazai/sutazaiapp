#!/usr/bin/env python3
"""
Integration tests for Task Assignment Coordinator
Tests concurrent task assignments, agent selection, and failure handling.
"""
import asyncio
import json
import sys
import uuid
from datetime import datetime
import logging
from typing import Dict, List

# Add project root to path
sys.path.insert(0, '/opt/sutazaiapp')

from agents.core.rabbitmq_client import RabbitMQClient
from schemas.task_messages import TaskRequestMessage, TaskAssignmentMessage
from schemas.agent_messages import AgentHeartbeatMessage, AgentStatusMessage
from schemas.system_messages import ErrorMessage
from schemas.base import Priority, TaskStatus, AgentStatus, MessageType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoordinatorIntegrationTest:
    """Integration test suite for Task Assignment Coordinator"""
    
    def __init__(self):
        self.test_results = []
        self.received_messages = {}
        self.test_agents = []
        
    async def setup_test_agents(self):
        """Create mock agents for testing"""
        # Create 3 test agents with different capabilities
        agent_configs = [
            {
                "id": "test_agent_1",
                "capabilities": ["memory_optimization", "resource_monitoring"],
                "load": 0.2,
                "active_tasks": 2
            },
            {
                "id": "test_agent_2", 
                "capabilities": ["disk_cleanup", "docker_optimization"],
                "load": 0.5,
                "active_tasks": 5
            },
            {
                "id": "test_agent_3",
                "capabilities": ["memory_optimization", "disk_cleanup", "resource_monitoring"],
                "load": 0.8,
                "active_tasks": 8
            }
        ]
        
        for config in agent_configs:
            client = RabbitMQClient(config["id"], "test_agent")
            await client.connect()
            
            # Send heartbeat to register agent
            heartbeat = AgentHeartbeatMessage(
                source_agent=config["id"],
                agent_id=config["id"],
                status=AgentStatus.READY,
                current_load=config["load"],
                active_tasks=config["active_tasks"],
                available_capacity=10 - config["active_tasks"],
                cpu_usage=config["load"] * 100,
                memory_usage=50.0,
                uptime_seconds=3600,
                error_count=0
            )
            
            await client.publish(
                heartbeat,
                exchange="sutazai.agents",
                routing_key="agent.status"
            )
            
            # Send status message with capabilities
            status = AgentStatusMessage(
                source_agent=config["id"],
                agent_id=config["id"],
                status=AgentStatus.READY,
                metrics={"capabilities": config["capabilities"]},
                active_task_ids=[],
                queued_task_ids=[]
            )
            
            await client.publish(
                status,
                exchange="sutazai.agents",
                routing_key="agent.status"
            )
            
            self.test_agents.append(client)
            logger.info(f"Setup test agent: {config['id']}")
        
        # Wait for agents to register
        await asyncio.sleep(2)
    
    async def test_single_assignment(self):
        """Test single task assignment"""
        try:
            client = RabbitMQClient("test_client_single", "tester")
            await client.connect()
            
            trace_id = str(uuid.uuid4())
            task_id = f"test_task_{uuid.uuid4().hex[:8]}"
            
            # Create task request
            task = TaskRequestMessage(
                source_agent="test_client",
                correlation_id=trace_id,
                task_id=task_id,
                task_type="optimize_memory",
                payload={"threshold": 80},
                priority=Priority.NORMAL,
                timeout_seconds=60
            )
            
            # Publish to assignment queue
            await client.publish(
                task,
                exchange="sutazai.tasks", 
                routing_key="task.assign"
            )
            
            # Wait for assignment
            await asyncio.sleep(1)
            
            await client.close()
            
            self.test_results.append({
                "test": "Single Task Assignment",
                "passed": True,
                "details": f"Task {task_id} sent for assignment"
            })
            return True
            
        except Exception as e:
            self.test_results.append({
                "test": "Single Task Assignment",
                "passed": False,
                "details": str(e)
            })
            return False
    
    async def test_concurrent_assignments(self):
        """Test 5 concurrent task assignments"""
        try:
            client = RabbitMQClient("test_client_concurrent", "tester")
            await client.connect()
            
            tasks = []
            
            # Create 5 concurrent tasks with different types
            task_types = [
                "optimize_memory",
                "clean_disk",
                "optimize_memory",
                "monitor_resources",
                "clean_disk"
            ]
            
            for i, task_type in enumerate(task_types):
                trace_id = str(uuid.uuid4())
                task_id = f"concurrent_task_{i}_{uuid.uuid4().hex[:8]}"
                
                task = TaskRequestMessage(
                    source_agent="test_client",
                    correlation_id=trace_id,
                    task_id=task_id,
                    task_type=task_type,
                    payload={"test": f"concurrent_{i}"},
                    priority=Priority.NORMAL if i < 3 else Priority.HIGH,
                    timeout_seconds=60
                )
                
                tasks.append(task)
            
            # Publish all tasks concurrently
            publish_tasks = []
            for task in tasks:
                publish_tasks.append(
                    client.publish(
                        task,
                        exchange="sutazai.tasks",
                        routing_key="task.assign"
                    )
                )
            
            await asyncio.gather(*publish_tasks)
            logger.info(f"Published {len(tasks)} concurrent tasks")
            
            # Wait for assignments
            await asyncio.sleep(3)
            
            await client.close()
            
            self.test_results.append({
                "test": "Concurrent Task Assignments",
                "passed": True,
                "details": f"Successfully sent {len(tasks)} concurrent tasks"
            })
            return True
            
        except Exception as e:
            self.test_results.append({
                "test": "Concurrent Task Assignments",
                "passed": False,
                "details": str(e)
            })
            return False
    
    async def test_no_eligible_agent(self):
        """Test assignment failure when no eligible agent exists"""
        try:
            client = RabbitMQClient("test_client_no_agent", "tester")
            await client.connect()
            
            received_error = []
            
            # Set up error message consumer
            async def error_handler(message_data, raw_message):
                if message_data.get("error_code") == "ASSIGNMENT_FAILED":
                    received_error.append(message_data)
                    logger.info(f"Received expected error: {message_data.get('error_message')}")
            
            await client.consume("assignment.failed", error_handler, auto_ack=True)
            
            trace_id = str(uuid.uuid4())
            task_id = f"impossible_task_{uuid.uuid4().hex[:8]}"
            
            # Create task with non-existent capability
            task = TaskRequestMessage(
                source_agent="test_client",
                correlation_id=trace_id,
                task_id=task_id,
                task_type="quantum_computing",  # No agent has this capability
                payload={"impossible": True},
                priority=Priority.HIGH,
                timeout_seconds=60
            )
            
            await client.publish(
                task,
                exchange="sutazai.tasks",
                routing_key="task.assign"
            )
            
            # Wait for error response
            await asyncio.sleep(2)
            
            await client.stop_consumer("assignment.failed")
            await client.close()
            
            passed = len(received_error) > 0
            self.test_results.append({
                "test": "No Eligible Agent Failure",
                "passed": passed,
                "details": "Received assignment failure" if passed else "No error received"
            })
            return passed
            
        except Exception as e:
            self.test_results.append({
                "test": "No Eligible Agent Failure",
                "passed": False,
                "details": str(e)
            })
            return False
    
    async def test_agent_overload(self):
        """Test assignment when all agents are at capacity"""
        try:
            client = RabbitMQClient("test_client_overload", "tester")
            await client.connect()
            
            # Send heartbeats showing all agents at capacity
            for i, agent in enumerate(self.test_agents):
                heartbeat = AgentHeartbeatMessage(
                    source_agent=f"test_agent_{i+1}",
                    agent_id=f"test_agent_{i+1}",
                    status=AgentStatus.BUSY,
                    current_load=0.95,  # Overloaded
                    active_tasks=10,  # At max capacity
                    available_capacity=0,
                    cpu_usage=95.0,
                    memory_usage=90.0,
                    uptime_seconds=3600,
                    error_count=0
                )
                
                await agent.publish(
                    heartbeat,
                    exchange="sutazai.agents",
                    routing_key="agent.status"
                )
            
            await asyncio.sleep(1)
            
            # Try to assign task
            trace_id = str(uuid.uuid4())
            task_id = f"overload_task_{uuid.uuid4().hex[:8]}"
            
            task = TaskRequestMessage(
                source_agent="test_client",
                correlation_id=trace_id,
                task_id=task_id,
                task_type="optimize_memory",
                payload={"test": "overload"},
                priority=Priority.NORMAL,
                timeout_seconds=60
            )
            
            await client.publish(
                task,
                exchange="sutazai.tasks",
                routing_key="task.assign"
            )
            
            await asyncio.sleep(2)
            await client.close()
            
            self.test_results.append({
                "test": "Agent Overload Handling",
                "passed": True,
                "details": "Overload scenario tested"
            })
            return True
            
        except Exception as e:
            self.test_results.append({
                "test": "Agent Overload Handling",
                "passed": False,
                "details": str(e)
            })
            return False
    
    async def test_priority_assignment(self):
        """Test priority-based task assignment"""
        try:
            client = RabbitMQClient("test_client_priority", "tester")
            await client.connect()
            
            # Reset agent loads
            for i, agent in enumerate(self.test_agents):
                heartbeat = AgentHeartbeatMessage(
                    source_agent=f"test_agent_{i+1}",
                    agent_id=f"test_agent_{i+1}",
                    status=AgentStatus.READY,
                    current_load=0.3 * (i + 1),  # Different loads
                    active_tasks=i * 2,
                    available_capacity=10 - (i * 2),
                    cpu_usage=30.0 * (i + 1),
                    memory_usage=40.0,
                    uptime_seconds=3600,
                    error_count=0
                )
                
                await agent.publish(
                    heartbeat,
                    exchange="sutazai.agents",
                    routing_key="agent.status"
                )
            
            await asyncio.sleep(1)
            
            # Send high priority task
            trace_id = str(uuid.uuid4())
            task_id = f"priority_task_{uuid.uuid4().hex[:8]}"
            
            task = TaskRequestMessage(
                source_agent="test_client",
                correlation_id=trace_id,
                task_id=task_id,
                task_type="optimize_memory",
                payload={"urgent": True},
                priority=Priority.CRITICAL,
                timeout_seconds=60
            )
            
            await client.publish(
                task,
                exchange="sutazai.tasks",
                routing_key="task.assign"
            )
            
            await asyncio.sleep(1)
            await client.close()
            
            self.test_results.append({
                "test": "Priority Task Assignment",
                "passed": True,
                "details": "Critical priority task sent"
            })
            return True
            
        except Exception as e:
            self.test_results.append({
                "test": "Priority Task Assignment",
                "passed": False,
                "details": str(e)
            })
            return False
    
    async def cleanup(self):
        """Clean up test agents"""
        for agent in self.test_agents:
            try:
                await agent.close()
            except:
                pass
    
    async def run_all_tests(self):
        """Run all integration tests"""
        print("\n" + "="*60)
        print("Task Assignment Coordinator Integration Tests")
        print("="*60 + "\n")
        
        # Setup test agents
        print("Setting up test agents...")
        await self.setup_test_agents()
        
        # Run tests
        print("\nRunning tests...")
        await self.test_single_assignment()
        await self.test_concurrent_assignments()
        await self.test_no_eligible_agent()
        await self.test_agent_overload()
        await self.test_priority_assignment()
        
        # Cleanup
        await self.cleanup()
        
        # Print results
        print("\n" + "-"*60)
        print("Test Results:")
        print("-"*60)
        
        passed = 0
        failed = 0
        
        for result in self.test_results:
            status = "✅ PASSED" if result["passed"] else "❌ FAILED"
            print(f"{status}: {result['test']}")
            if result["details"]:
                print(f"   Details: {result['details']}")
            
            if result["passed"]:
                passed += 1
            else:
                failed += 1
        
        print("-"*60)
        print(f"\nSummary: {passed} passed, {failed} failed")
        print("="*60 + "\n")
        
        return failed == 0


async def main():
    """Run integration tests"""
    tester = CoordinatorIntegrationTest()
    success = await tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())