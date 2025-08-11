#!/usr/bin/env python3
"""
Integration tests for RabbitMQ messaging between agents.
Tests real message flow using the centralized schemas.
"""
import asyncio
import json
import sys
import os
from datetime import datetime
import logging

# Add project root to path
sys.path.insert(0, '/opt/sutazaiapp')

from agents.core.rabbitmq_client import RabbitMQClient
from schemas.task_messages import TaskRequestMessage, TaskCompletionMessage
from schemas.agent_messages import AgentRegistrationMessage, AgentHeartbeatMessage
from schemas.resource_messages import ResourceRequestMessage, ResourceStatusMessage
from schemas.base import Priority, TaskStatus, AgentStatus, ResourceType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MessagingIntegrationTest:
    """Test suite for inter-agent messaging"""
    
    def __init__(self):
        self.test_results = []
        self.received_messages = []
        
    async def test_connection(self):
        """Test RabbitMQ connection"""
        try:
            client = RabbitMQClient("test_client", "tester")
            connected = await client.connect()
            await client.close()
            
            self.test_results.append({
                "test": "RabbitMQ Connection",
                "passed": connected,
                "details": "Connected successfully" if connected else "Connection failed"
            })
            return connected
        except Exception as e:
            self.test_results.append({
                "test": "RabbitMQ Connection",
                "passed": False,
                "details": str(e)
            })
            return False
    
    async def test_agent_registration(self):
        """Test agent registration message"""
        try:
            client = RabbitMQClient("test_agent", "tester")
            await client.connect()
            
            # Send registration
            registration = AgentRegistrationMessage(
                source_agent="test_agent",
                agent_id="test_agent",
                agent_type="tester",
                capabilities=["testing", "validation"],
                version="1.0.0",
                host="localhost",
                port=9999,
                max_concurrent_tasks=5
            )
            
            await client.publish(registration)
            await asyncio.sleep(1)  # Give time for message to propagate
            await client.close()
            
            self.test_results.append({
                "test": "Agent Registration",
                "passed": True,
                "details": "Registration message sent successfully"
            })
            return True
            
        except Exception as e:
            self.test_results.append({
                "test": "Agent Registration",
                "passed": False,
                "details": str(e)
            })
            return False
    
    async def test_task_flow(self):
        """Test task request and completion flow"""
        try:
            # Create publisher and consumer
            publisher = RabbitMQClient("task_publisher", "publisher")
            consumer = RabbitMQClient("task_consumer", "consumer")
            
            await publisher.connect()
            await consumer.connect()
            
            received = []
            
            # Message handler
            async def handle_message(message_data, raw_message):
                received.append(message_data)
                logger.info(f"Received message: {message_data.get('message_type')}")
            
            # Start consuming
            await consumer.consume("tasks.normal_priority", handle_message, auto_ack=True)
            
            # Send task request
            task = TaskRequestMessage(
                source_agent="task_publisher",
                task_id="test_task_001",
                task_type="test_operation",
                payload={"test": "data"},
                priority=Priority.NORMAL,
                timeout_seconds=60
            )
            
            await publisher.publish(task)
            
            # Wait for message
            await asyncio.sleep(2)
            
            # Cleanup
            await consumer.stop_consumer("tasks.normal_priority")
            await publisher.close()
            await consumer.close()
            
            passed = len(received) > 0
            self.test_results.append({
                "test": "Task Message Flow",
                "passed": passed,
                "details": f"Received {len(received)} messages" if passed else "No messages received"
            })
            return passed
            
        except Exception as e:
            self.test_results.append({
                "test": "Task Message Flow",
                "passed": False,
                "details": str(e)
            })
            return False
    
    async def test_resource_messages(self):
        """Test resource request and status messages"""
        try:
            client = RabbitMQClient("resource_tester", "tester")
            await client.connect()
            
            # Send resource request
            request = ResourceRequestMessage(
                source_agent="resource_tester",
                request_id="res_test_001",
                requesting_agent="resource_tester",
                resources={ResourceType.CPU: 2.0, ResourceType.MEMORY: 4.0},
                duration_seconds=60
            )
            
            await client.publish(request)
            
            # Send resource status
            status = ResourceStatusMessage(
                source_agent="resource_tester",
                total_capacity={ResourceType.CPU: 8, ResourceType.MEMORY: 16},
                available_capacity={ResourceType.CPU: 6, ResourceType.MEMORY: 12},
                allocated_capacity={ResourceType.CPU: 2, ResourceType.MEMORY: 4},
                reserved_capacity={},
                active_allocations=[],
                pending_requests=1
            )
            
            await client.publish(status)
            await asyncio.sleep(1)
            await client.close()
            
            self.test_results.append({
                "test": "Resource Messages",
                "passed": True,
                "details": "Resource messages sent successfully"
            })
            return True
            
        except Exception as e:
            self.test_results.append({
                "test": "Resource Messages",
                "passed": False,
                "details": str(e)
            })
            return False
    
    async def test_heartbeat(self):
        """Test agent heartbeat message"""
        try:
            client = RabbitMQClient("heartbeat_agent", "tester")
            await client.connect()
            
            heartbeat = AgentHeartbeatMessage(
                source_agent="heartbeat_agent",
                agent_id="heartbeat_agent",
                status=AgentStatus.READY,
                current_load=0.3,
                active_tasks=3,
                available_capacity=7,
                cpu_usage=25.5,
                memory_usage=45.2,
                uptime_seconds=3600.0,
                error_count=0
            )
            
            await client.publish(heartbeat)
            await asyncio.sleep(1)
            await client.close()
            
            self.test_results.append({
                "test": "Agent Heartbeat",
                "passed": True,
                "details": "Heartbeat sent successfully"
            })
            return True
            
        except Exception as e:
            self.test_results.append({
                "test": "Agent Heartbeat",
                "passed": False,
                "details": str(e)
            })
            return False
    
    async def run_all_tests(self):
        """Run all integration tests"""
        print("\n" + "="*50)
        print("RabbitMQ Messaging Integration Tests")
        print("="*50 + "\n")
        
        # Run tests
        await self.test_connection()
        await self.test_agent_registration()
        await self.test_task_flow()
        await self.test_resource_messages()
        await self.test_heartbeat()
        
        # Print results
        print("\nTest Results:")
        print("-" * 50)
        
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
        
        print("-" * 50)
        print(f"\nSummary: {passed} passed, {failed} failed")
        print("="*50 + "\n")
        
        return failed == 0


async def main():
    """Run integration tests"""
    tester = MessagingIntegrationTest()
    success = await tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())