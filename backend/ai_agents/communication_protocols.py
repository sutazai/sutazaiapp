"""
Real-Time Agent Communication Protocols
Provides WebSocket-based communication, message queuing, and event-driven coordination.
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from backend.app.schemas.message_types import MessageType, MessagePriority
from backend.app.schemas.message_model import Message as Message
from typing import Any, Dict, List, Optional, Set, Union, Callable, Awaitable
import aiohttp
from aiohttp import web, WSMsgType
import aioredis
from universal_client import AgentType, Priority

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Using canonical MessageType and MessagePriority


class DeliveryMode(Enum):
    """Message delivery modes."""
    FIRE_AND_FORGET = "fire_and_forget"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"
    REQUEST_RESPONSE = "request_response"


# Use canonical Message from schemas.message_model


@dataclass
class AgentConnection:
    """Represents a connection to an agent."""
    agent_id: str
    websocket: web.WebSocketResponse
    last_heartbeat: datetime = field(default_factory=datetime.now)
    capabilities: List[str] = field(default_factory=list)
    status: str = "connected"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    async def send_message(self, message: Message):
        """Send message to agent via WebSocket."""
        try:
            await self.websocket.send_str(json.dumps(message.to_dict()))
        except Exception as e:
            logger.error(f"Failed to send message to {self.agent_id}: {str(e)}")
            raise
    
    def is_alive(self, timeout: int = 60) -> bool:
        """Check if connection is alive based on heartbeat."""
        elapsed = (datetime.now() - self.last_heartbeat).total_seconds()
        return elapsed < timeout


class MessageBus(ABC):
    """Abstract message bus interface."""
    
    @abstractmethod
    async def publish(self, message: Message):
        """Publish a message."""
        pass
    
    @abstractmethod
    async def subscribe(self, pattern: str, handler: Callable[[Message], Awaitable[None]]):
        """Subscribe to messages matching pattern."""
        pass
    
    @abstractmethod
    async def unsubscribe(self, pattern: str):
        """Unsubscribe from pattern."""
        pass
    
    @abstractmethod
    async def send_direct(self, recipient: str, message: Message):
        """Send message directly to recipient."""
        pass


class InMemoryMessageBus(MessageBus):
    """In-memory message bus implementation."""
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.message_queues: Dict[str, deque] = defaultdict(deque)
        self.pending_responses: Dict[str, asyncio.Future] = {}
        self.message_history: List[Message] = []
        self.max_history = 10000
    
    async def publish(self, message: Message):
        """Publish message to all matching subscribers."""
        self._add_to_history(message)
        
        # Handle direct recipients
        if message.recipient:
            await self._deliver_direct(message)
        else:
            # Broadcast to pattern matching subscribers
            await self._broadcast(message)
    
    async def subscribe(self, pattern: str, handler: Callable[[Message], Awaitable[None]]):
        """Subscribe to messages matching pattern."""
        self.subscribers[pattern].append(handler)
    
    async def unsubscribe(self, pattern: str):
        """Unsubscribe from pattern."""
        if pattern in self.subscribers:
            del self.subscribers[pattern]
    
    async def send_direct(self, recipient: str, message: Message):
        """Send message directly to recipient."""
        message.recipient = recipient
        await self.publish(message)
    
    async def _deliver_direct(self, message: Message):
        """Deliver message to specific recipient."""
        recipient = message.recipient
        
        # Add to recipient's queue
        self.message_queues[recipient].append(message)
        
        # Notify subscribers
        for pattern, handlers in self.subscribers.items():
            if self._pattern_matches(pattern, message):
                for handler in handlers:
                    try:
                        await handler(message)
                    except Exception as e:
                        logger.error(f"Handler error for pattern {pattern}: {str(e)}")
    
    async def _broadcast(self, message: Message):
        """Broadcast message to all matching subscribers."""
        for pattern, handlers in self.subscribers.items():
            if self._pattern_matches(pattern, message):
                for handler in handlers:
                    try:
                        await handler(message)
                    except Exception as e:
                        logger.error(f"Handler error for pattern {pattern}: {str(e)}")
    
    def _pattern_matches(self, pattern: str, message: Message) -> bool:
        """Check if message matches subscription pattern."""
        # Simple pattern matching - could be enhanced
        if pattern == "*":
            return True
        
        if pattern == message.type.value:
            return True
        
        if pattern.endswith("*"):
            prefix = pattern[:-1]
            return message.type.value.startswith(prefix)
        
        return False
    
    def _add_to_history(self, message: Message):
        """Add message to history."""
        self.message_history.append(message)
        
        # Trim history if too large
        if len(self.message_history) > self.max_history:
            self.message_history = self.message_history[-self.max_history//2:]
    
    def get_messages_for_agent(self, agent_id: str, limit: int = 100) -> List[Message]:
        """Get messages for a specific agent."""
        messages = list(self.message_queues[agent_id])
        return messages[-limit:] if limit else messages
    
    def clear_agent_queue(self, agent_id: str):
        """Clear message queue for agent."""
        self.message_queues[agent_id].clear()


class RedisMessageBus(MessageBus):
    """Redis-based message bus for distributed systems."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis: Optional[aioredis.Redis] = None
        self.pubsub: Optional[aioredis.client.PubSub] = None
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
    
    async def connect(self):
        """Connect to Redis."""
        self.redis = aioredis.from_url(self.redis_url)
        self.pubsub = self.redis.pubsub()
        
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.pubsub:
            await self.pubsub.close()
        if self.redis:
            await self.redis.close()
    
    async def publish(self, message: Message):
        """Publish message to Redis."""
        if not self.redis:
            await self.connect()
        
        channel = message.recipient or "broadcast"
        message_data = json.dumps(message.to_dict())
        
        await self.redis.publish(channel, message_data)
        
        # Store in message queue for recipient
        if message.recipient:
            queue_key = f"queue:{message.recipient}"
            await self.redis.lpush(queue_key, message_data)
            
            # Set TTL for message queue
            await self.redis.expire(queue_key, 3600)  # 1 hour
    
    async def subscribe(self, pattern: str, handler: Callable[[Message], Awaitable[None]]):
        """Subscribe to Redis channels."""
        if not self.pubsub:
            await self.connect()
        
        self.subscribers[pattern].append(handler)
        await self.pubsub.psubscribe(pattern)
        
        # Start listening task if not already running
        if not hasattr(self, '_listen_task'):
            self._listen_task = asyncio.create_task(self._listen_loop())
    
    async def unsubscribe(self, pattern: str):
        """Unsubscribe from Redis channels."""
        if pattern in self.subscribers:
            del self.subscribers[pattern]
        
        if self.pubsub:
            await self.pubsub.punsubscribe(pattern)
    
    async def send_direct(self, recipient: str, message: Message):
        """Send message directly to recipient."""
        message.recipient = recipient
        await self.publish(message)
    
    async def _listen_loop(self):
        """Listen for Redis messages."""
        while True:
            try:
                message = await self.pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'pmessage':
                    pattern = message['pattern'].decode()
                    data = json.loads(message['data'].decode())
                    msg = Message.from_dict(data)
                    
                    # Call matching handlers
                    for handler in self.subscribers.get(pattern, []):
                        try:
                            await handler(msg)
                        except Exception as e:
                            logger.error(f"Handler error: {str(e)}")
                            
            except Exception as e:
                logger.error(f"Redis listen error: {str(e)}")
                await asyncio.sleep(1)
    
    async def get_queued_messages(self, agent_id: str, limit: int = 100) -> List[Message]:
        """Get queued messages for agent."""
        if not self.redis:
            await self.connect()
        
        queue_key = f"queue:{agent_id}"
        messages_data = await self.redis.lrange(queue_key, 0, limit - 1)
        
        messages = []
        for data in messages_data:
            try:
                message = Message.from_dict(json.loads(data))
                if not message.is_expired():
                    messages.append(message)
            except Exception as e:
                logger.error(f"Failed to parse message: {str(e)}")
        
        return messages
    
    async def clear_agent_queue(self, agent_id: str):
        """Clear message queue for agent."""
        if not self.redis:
            await self.connect()
        
        queue_key = f"queue:{agent_id}"
        await self.redis.delete(queue_key)


class CommunicationHub:
    """Central hub for agent communication."""
    
    def __init__(
        self,
        message_bus: Optional[MessageBus] = None,
        enable_websocket: bool = True,
        websocket_port: int = 8080
    ):
        self.message_bus = message_bus or InMemoryMessageBus()
        self.enable_websocket = enable_websocket
        self.websocket_port = websocket_port
        
        # Connection management
        self.connections: Dict[str, AgentConnection] = {}
        self.connection_groups: Dict[str, Set[str]] = defaultdict(set)
        
        # Request-response tracking
        self.pending_requests: Dict[str, asyncio.Future] = {}
        self.request_timeouts: Dict[str, asyncio.Task] = {}
        
        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "connections_established": 0,
            "connections_lost": 0,
            "errors": 0
        }
        
        # WebSocket application
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
    
    async def start(self):
        """Start the communication hub."""
        logger.info("Starting Communication Hub...")
        
        # Subscribe to all messages for routing
        await self.message_bus.subscribe("*", self._handle_message)
        
        if self.enable_websocket:
            await self._start_websocket_server()
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_loop())
        
        logger.info(f"Communication Hub started on port {self.websocket_port}")
    
    async def stop(self):
        """Stop the communication hub."""
        logger.info("Stopping Communication Hub...")
        
        # Close all connections
        for connection in list(self.connections.values()):
            try:
                await connection.websocket.close()
            except Exception:
                pass
        
        # Stop WebSocket server
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        
        logger.info("Communication Hub stopped")
    
    async def _start_websocket_server(self):
        """Start WebSocket server for agent connections."""
        self.app = web.Application()
        self.app.router.add_get('/agent/{agent_id}', self._websocket_handler)
        self.app.router.add_get('/health', self._health_handler)
        
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        self.site = web.TCPSite(self.runner, 'localhost', self.websocket_port)
        await self.site.start()
    
    async def _websocket_handler(self, request: web.Request):
        """Handle WebSocket connections from agents."""
        agent_id = request.match_info['agent_id']
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        # Create connection
        connection = AgentConnection(agent_id=agent_id, websocket=ws)
        self.connections[agent_id] = connection
        self.stats["connections_established"] += 1
        
        logger.info(f"Agent {agent_id} connected")
        
        try:
            # Send queued messages
            await self._send_queued_messages(agent_id)
            
            # Handle incoming messages
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        message = Message.from_dict(data)
                        message.sender = agent_id
                        
                        await self._process_incoming_message(message, connection)
                        
                    except Exception as e:
                        logger.error(f"Error processing message from {agent_id}: {str(e)}")
                        self.stats["errors"] += 1
                        
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f"WebSocket error from {agent_id}: {ws.exception()}")
                    
        except Exception as e:
            logger.error(f"WebSocket handler error for {agent_id}: {str(e)}")
            
        finally:
            # Clean up connection
            if agent_id in self.connections:
                del self.connections[agent_id]
            self.stats["connections_lost"] += 1
            logger.info(f"Agent {agent_id} disconnected")
        
        return ws
    
    async def _health_handler(self, request: web.Request):
        """Health check endpoint."""
        return web.json_response({
            "status": "healthy",
            "connections": len(self.connections),
            "stats": self.stats
        })
    
    async def _send_queued_messages(self, agent_id: str):
        """Send queued messages to newly connected agent."""
        if hasattr(self.message_bus, 'get_messages_for_agent'):
            messages = self.message_bus.get_messages_for_agent(agent_id)
            
            for message in messages:
                if not message.is_expired():
                    try:
                        await self.connections[agent_id].send_message(message)
                    except Exception as e:
                        logger.error(f"Failed to send queued message to {agent_id}: {str(e)}")
            
            # Clear queue after sending
            if hasattr(self.message_bus, 'clear_agent_queue'):
                self.message_bus.clear_agent_queue(agent_id)
    
    async def _process_incoming_message(self, message: Message, connection: AgentConnection):
        """Process incoming message from agent."""
        self.stats["messages_received"] += 1
        
        # Update heartbeat
        connection.last_heartbeat = datetime.now()
        
        # Handle special message types
        if message.type == MessageType.HEARTBEAT:
            await self._handle_heartbeat(message, connection)
        elif message.type == MessageType.AGENT_REGISTER:
            await self._handle_agent_register(message, connection)
        elif message.type == MessageType.RESPONSE:
            await self._handle_response(message)
        else:
            # Forward to message bus
            await self.message_bus.publish(message)
    
    async def _handle_heartbeat(self, message: Message, connection: AgentConnection):
        """Handle heartbeat message."""
        # Update connection metadata if provided
        if "capabilities" in message.payload:
            connection.capabilities = message.payload["capabilities"]
        if "status" in message.payload:
            connection.status = message.payload["status"]
        
        # Send heartbeat response
        response = message.create_response({"status": "ok", "timestamp": datetime.now().isoformat()})
        await connection.send_message(response)
    
    async def _handle_agent_register(self, message: Message, connection: AgentConnection):
        """Handle agent registration."""
        registration_data = message.payload
        
        connection.capabilities = registration_data.get("capabilities", [])
        connection.metadata.update(registration_data.get("metadata", {}))
        
        # Add to groups based on capabilities
        for capability in connection.capabilities:
            self.connection_groups[capability].add(connection.agent_id)
        
        # Send registration confirmation
        response = message.create_response({
            "status": "registered",
            "agent_id": connection.agent_id,
            "hub_info": {
                "connected_agents": len(self.connections),
                "available_capabilities": list(self.connection_groups.keys())
            }
        })
        await connection.send_message(response)
        
        logger.info(f"Agent {connection.agent_id} registered with capabilities: {connection.capabilities}")
    
    async def _handle_response(self, message: Message):
        """Handle response message for request-response pattern."""
        correlation_id = message.correlation_id
        
        if correlation_id in self.pending_requests:
            future = self.pending_requests[correlation_id]
            if not future.done():
                future.set_result(message)
            
            # Clean up
            del self.pending_requests[correlation_id]
            if correlation_id in self.request_timeouts:
                self.request_timeouts[correlation_id].cancel()
                del self.request_timeouts[correlation_id]
    
    async def _handle_message(self, message: Message):
        """Handle messages from message bus."""
        if message.recipient and message.recipient in self.connections:
            # Direct message to connected agent
            try:
                await self.connections[message.recipient].send_message(message)
                self.stats["messages_sent"] += 1
            except Exception as e:
                logger.error(f"Failed to send message to {message.recipient}: {str(e)}")
                self.stats["errors"] += 1
        elif message.recipient:
            # Agent not connected - message will be queued by message bus
            pass
        else:
            # Broadcast message
            await self._broadcast_message(message)
    
    async def _broadcast_message(self, message: Message):
        """Broadcast message to all connected agents."""
        failed_connections = []
        
        for agent_id, connection in self.connections.items():
            try:
                await connection.send_message(message)
                self.stats["messages_sent"] += 1
            except Exception as e:
                logger.error(f"Failed to broadcast to {agent_id}: {str(e)}")
                failed_connections.append(agent_id)
                self.stats["errors"] += 1
        
        # Clean up failed connections
        for agent_id in failed_connections:
            if agent_id in self.connections:
                del self.connections[agent_id]
    
    async def _cleanup_loop(self):
        """Periodic cleanup of expired resources."""
        while True:
            try:
                await asyncio.sleep(30)  # Run every 30 seconds
                
                # Clean up dead connections
                dead_connections = []
                for agent_id, connection in self.connections.items():
                    if not connection.is_alive():
                        dead_connections.append(agent_id)
                
                for agent_id in dead_connections:
                    del self.connections[agent_id]
                    logger.info(f"Cleaned up dead connection: {agent_id}")
                    
                    # Remove from groups
                    for group_agents in self.connection_groups.values():
                        group_agents.discard(agent_id)
                
                # Clean up expired pending requests
                expired_requests = []
                for request_id, future in self.pending_requests.items():
                    if future.done():
                        expired_requests.append(request_id)
                
                for request_id in expired_requests:
                    del self.pending_requests[request_id]
                    if request_id in self.request_timeouts:
                        self.request_timeouts[request_id].cancel()
                        del self.request_timeouts[request_id]
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {str(e)}")
    
    async def send_message(self, message: Message) -> Optional[Message]:
        """Send message and optionally wait for response."""
        if message.delivery_mode == DeliveryMode.REQUEST_RESPONSE:
            return await self._send_request_response(message)
        else:
            await self.message_bus.publish(message)
            return None
    
    async def _send_request_response(self, message: Message, timeout: int = 30) -> Message:
        """Send request and wait for response."""
        # Set up response tracking
        future = asyncio.Future()
        self.pending_requests[message.id] = future
        
        # Set up timeout
        timeout_task = asyncio.create_task(self._request_timeout(message.id, timeout))
        self.request_timeouts[message.id] = timeout_task
        
        # Send message
        await self.message_bus.publish(message)
        
        try:
            # Wait for response
            response = await future
            return response
        except asyncio.TimeoutError:
            raise TimeoutError(f"Request {message.id} timed out after {timeout} seconds")
        finally:
            # Clean up
            if message.id in self.pending_requests:
                del self.pending_requests[message.id]
            if message.id in self.request_timeouts:
                self.request_timeouts[message.id].cancel()
                del self.request_timeouts[message.id]
    
    async def _request_timeout(self, request_id: str, timeout: int):
        """Handle request timeout."""
        await asyncio.sleep(timeout)
        
        if request_id in self.pending_requests:
            future = self.pending_requests[request_id]
            if not future.done():
                future.set_exception(asyncio.TimeoutError())
    
    def broadcast_to_capability_group(self, capability: str, message: Message):
        """Broadcast message to all agents with specific capability."""
        if capability in self.connection_groups:
            for agent_id in self.connection_groups[capability]:
                message_copy = Message(
                    id=f"{message.id}_{agent_id}",
                    type=message.type,
                    sender=message.sender,
                    recipient=agent_id,
                    payload=message.payload.copy(),
                    priority=message.priority,
                    delivery_mode=message.delivery_mode
                )
                asyncio.create_task(self.send_message(message_copy))
    
    def get_connected_agents(self) -> List[Dict[str, Any]]:
        """Get list of connected agents."""
        agents = []
        for agent_id, connection in self.connections.items():
            agents.append({
                "agent_id": agent_id,
                "capabilities": connection.capabilities,
                "status": connection.status,
                "last_heartbeat": connection.last_heartbeat.isoformat(),
                "metadata": connection.metadata
            })
        return agents
    
    def get_capability_groups(self) -> Dict[str, List[str]]:
        """Get agents grouped by capabilities."""
        return {
            capability: list(agents)
            for capability, agents in self.connection_groups.items()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get communication statistics."""
        return {
            **self.stats,
            "active_connections": len(self.connections),
            "capability_groups": len(self.connection_groups),
            "pending_requests": len(self.pending_requests)
        }


# High-level communication helpers
class AgentCommunicator:
    """High-level interface for agent communication."""
    
    def __init__(self, agent_id: str, communication_hub: CommunicationHub):
        self.agent_id = agent_id
        self.hub = communication_hub
    
    async def send_task_assignment(
        self,
        target_agent: str,
        task_description: str,
        parameters: Dict[str, Any],
        priority: Priority = Priority.MEDIUM
    ) -> Message:
        """Send task assignment to another agent."""
        message = Message(
            id=f"task_{uuid.uuid4().hex[:8]}",
            type=MessageType.TASK_ASSIGNMENT,
            sender=self.agent_id,
            recipient=target_agent,
            payload={
                "task_description": task_description,
                "parameters": parameters,
                "priority": priority.value
            },
            priority=MessagePriority(priority.value),
            delivery_mode=DeliveryMode.REQUEST_RESPONSE
        )
        
        return await self.hub.send_message(message)
    
    async def send_task_completion(
        self,
        task_id: str,
        result: Any,
        execution_time: float = 0.0
    ):
        """Send task completion notification."""
        message = Message(
            id=f"complete_{uuid.uuid4().hex[:8]}",
            type=MessageType.TASK_COMPLETION,
            sender=self.agent_id,
            payload={
                "task_id": task_id,
                "result": result,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            },
            priority=MessagePriority.HIGH
        )
        
        await self.hub.send_message(message)
    
    async def send_task_failure(
        self,
        task_id: str,
        error: str,
        retry_possible: bool = True
    ):
        """Send task failure notification."""
        message = Message(
            id=f"failure_{uuid.uuid4().hex[:8]}",
            type=MessageType.TASK_FAILURE,
            sender=self.agent_id,
            payload={
                "task_id": task_id,
                "error": error,
                "retry_possible": retry_possible,
                "timestamp": datetime.now().isoformat()
            },
            priority=MessagePriority.URGENT
        )
        
        await self.hub.send_message(message)
    
    async def request_collaboration(
        self,
        capability_needed: str,
        task_description: str,
        context: Dict[str, Any]
    ) -> List[Message]:
        """Request collaboration from agents with specific capability."""
        message = Message(
            id=f"collab_{uuid.uuid4().hex[:8]}",
            type=MessageType.COLLABORATION_REQUEST,
            sender=self.agent_id,
            payload={
                "capability_needed": capability_needed,
                "task_description": task_description,
                "context": context
            },
            priority=MessagePriority.HIGH,
            delivery_mode=DeliveryMode.REQUEST_RESPONSE
        )
        
        # Broadcast to capability group
        self.hub.broadcast_to_capability_group(capability_needed, message)
        
        # Wait for responses (simplified - could be enhanced)
        await asyncio.sleep(5)  # Give agents time to respond
        return []  # Would collect actual responses
    
    async def share_knowledge(
        self,
        knowledge_type: str,
        knowledge_data: Dict[str, Any],
        target_agents: Optional[List[str]] = None
    ):
        """Share knowledge with other agents."""
        message = Message(
            id=f"knowledge_{uuid.uuid4().hex[:8]}",
            type=MessageType.KNOWLEDGE_SHARE,
            sender=self.agent_id,
            payload={
                "knowledge_type": knowledge_type,
                "knowledge_data": knowledge_data,
                "timestamp": datetime.now().isoformat()
            },
            priority=MessagePriority.NORMAL
        )
        
        if target_agents:
            for agent_id in target_agents:
                message.recipient = agent_id
                await self.hub.send_message(message)
        else:
            # Broadcast to all
            await self.hub.send_message(message)
    
    async def send_heartbeat(self, capabilities: List[str], status: str = "active"):
        """Send heartbeat with current status."""
        message = Message(
            id=f"heartbeat_{uuid.uuid4().hex[:8]}",
            type=MessageType.HEARTBEAT,
            sender=self.agent_id,
            payload={
                "capabilities": capabilities,
                "status": status,
                "timestamp": datetime.now().isoformat()
            },
            priority=MessagePriority.LOW
        )
        
        await self.hub.send_message(message)


# Example usage and testing
if __name__ == "__main__":
    async def main():
        """Example usage of the Communication Protocols."""
        
        # Start communication hub
        hub = CommunicationHub()
        await hub.start()
        
        try:
            # Create communicators for testing
            agent1 = AgentCommunicator("test-agent-1", hub)
            agent2 = AgentCommunicator("test-agent-2", hub)
            
            # Send heartbeats
            await agent1.send_heartbeat(["code_analysis", "testing"], "active")
            await agent2.send_heartbeat(["deployment", "monitoring"], "active")
            
            # Test task assignment
            await agent1.send_task_assignment(
                "test-agent-2",
                "Deploy application to staging",
                {"app_name": "test-app", "environment": "staging"}
            )
            
            # Test knowledge sharing
            await agent1.share_knowledge(
                "best_practices",
                {"code_review": "Always check for security issues"},
                ["test-agent-2"]
            )
            
            # Show statistics
            stats = hub.get_statistics()
            print(f"Communication stats: {stats}")
            
            connected_agents = hub.get_connected_agents()
            print(f"Connected agents: {len(connected_agents)}")
            
            await asyncio.sleep(5)  # Let messages process
            
        finally:
            await hub.stop()
    
    # Run the example
    asyncio.run(main())
