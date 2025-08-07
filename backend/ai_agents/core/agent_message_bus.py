"""
Agent Message Bus - Advanced Inter-Agent Communication System
============================================================

This module provides a sophisticated message bus for inter-agent communication
using Redis pub/sub, message queues, and advanced routing capabilities.
It's completely independent and supports complex agent workflows.

Features:
- Redis-based pub/sub messaging
- Message queues with priority handling
- Message routing and filtering
- Broadcast and multicast messaging
- Message persistence and replay
- Event-driven architecture
- Message acknowledgments and retries
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

import aioredis
from pydantic import BaseModel

from .base_agent import AgentMessage, AgentStatus
from backend.app.schemas.message_types import MessageType, MessagePriority


# Using canonical MessagePriority and MessageType


class RoutingStrategy(Enum):
    """Message routing strategies"""
    DIRECT = "direct"           # Send to specific agent
    BROADCAST = "broadcast"     # Send to all agents
    MULTICAST = "multicast"     # Send to agents with specific capabilities
    ROUND_ROBIN = "round_robin" # Distribute among capable agents
    LOAD_BALANCE = "load_balance" # Send to least busy capable agent
    FAILOVER = "failover"       # Try agents in order until success


@dataclass
class MessageRoute:
    """Defines how messages should be routed"""
    strategy: RoutingStrategy
    target_agents: Optional[List[str]] = None
    required_capabilities: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    max_attempts: int = 3
    timeout_seconds: int = 30


@dataclass
class MessageStats:
    """Message statistics"""
    total_sent: int = 0
    total_received: int = 0
    total_failed: int = 0
    total_retried: int = 0
    average_latency_ms: float = 0.0
    peak_queue_size: int = 0
    current_queue_size: int = 0


class MessageFilter:
    """Filters messages based on criteria"""
    
    def __init__(self, criteria: Dict[str, Any]):
        self.criteria = criteria
    
    def match(self, message: AgentMessage) -> bool:
        """Check if message matches filter criteria"""
        for key, value in self.criteria.items():
            if key == "sender_id" and message.sender_id != value:
                return False
            elif key == "message_type" and message.message_type != value:
                return False
            elif key == "priority" and message.priority != value:
                return False
            elif key == "content_contains":
                if not any(str(value).lower() in str(v).lower() 
                          for v in message.content.values()):
                    return False
            elif key == "metadata" and isinstance(value, dict):
                for meta_key, meta_value in value.items():
                    if message.metadata.get(meta_key) != meta_value:
                        return False
        return True


class AgentMessageBus:
    """
    Advanced Message Bus for Inter-Agent Communication
    
    Provides sophisticated messaging capabilities including:
    - Pub/sub messaging with Redis
    - Priority queues
    - Message routing strategies
    - Persistence and replay
    - Advanced filtering and routing
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379", 
                 namespace: str = "sutazai"):
        self.redis_url = redis_url
        self.namespace = namespace
        self.redis: Optional[aioredis.Redis] = None
        self.pubsub: Optional[aioredis.client.PubSub] = None
        
        # Message handling
        self.message_handlers: Dict[str, List[Callable]] = {}
        self.message_filters: Dict[str, MessageFilter] = {}
        self.routing_rules: Dict[str, MessageRoute] = {}
        
        # Statistics and monitoring
        self.stats = MessageStats()
        self.active_subscriptions: Set[str] = set()
        self.message_history: List[AgentMessage] = []
        self.max_history_size = 1000
        
        # Agent registry tracking
        self.known_agents: Dict[str, Dict[str, Any]] = {}
        self.agent_capabilities: Dict[str, Set[str]] = {}
        self.agent_status: Dict[str, AgentStatus] = {}
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        self.logger = logging.getLogger("agent_message_bus")
    
    async def initialize(self) -> bool:
        """Initialize the message bus"""
        try:
            self.logger.info("Initializing Agent Message Bus")
            
            # Connect to Redis
            self.redis = aioredis.from_url(self.redis_url, decode_responses=True)
            await self.redis.ping()
            
            # Initialize pub/sub
            self.pubsub = self.redis.pubsub()
            
            # Start background tasks
            self._start_background_tasks()
            
            # Load existing agent registry
            await self._load_agent_registry()
            
            self.logger.info("Agent Message Bus initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize message bus: {e}")
            return False
    
    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        tasks = [
            self._message_listener(),
            self._agent_registry_monitor(),
            self._stats_collector(),
            self._cleanup_expired_messages()
        ]
        
        for coro in tasks:
            task = asyncio.create_task(coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
    
    async def _load_agent_registry(self):
        """Load agent registry from Redis"""
        try:
            agents_data = await self.redis.hgetall(f"{self.namespace}:agent_registry")
            
            for agent_id, agent_info_json in agents_data.items():
                agent_info = json.loads(agent_info_json)
                self.known_agents[agent_id] = agent_info
                self.agent_capabilities[agent_id] = set(agent_info.get("capabilities", []))
                
                try:
                    self.agent_status[agent_id] = AgentStatus(agent_info.get("status", "offline"))
                except ValueError:
                    self.agent_status[agent_id] = AgentStatus.OFFLINE
                    
        except Exception as e:
            self.logger.error(f"Error loading agent registry: {e}")
    
    async def _message_listener(self):
        """Background task to listen for messages"""
        try:
            await self.pubsub.subscribe(f"{self.namespace}:messages")
            
            async for message in self.pubsub.listen():
                if message["type"] == "message":
                    try:
                        await self._process_incoming_message(message["data"])
                    except Exception as e:
                        self.logger.error(f"Error processing message: {e}")
                        
        except Exception as e:
            self.logger.error(f"Message listener error: {e}")
    
    async def _agent_registry_monitor(self):
        """Monitor agent registry changes"""
        try:
            await self.pubsub.subscribe(f"{self.namespace}:agent_events")
            
            async for message in self.pubsub.listen():
                if message["type"] == "message":
                    try:
                        event_data = json.loads(message["data"])
                        await self._handle_agent_event(event_data)
                    except Exception as e:
                        self.logger.error(f"Error handling agent event: {e}")
                        
        except Exception as e:
            self.logger.error(f"Agent registry monitor error: {e}")
    
    async def _stats_collector(self):
        """Collect message statistics"""
        while not self._shutdown_event.is_set():
            try:
                # Update queue sizes
                queue_size = await self.redis.llen(f"{self.namespace}:message_queue")
                self.stats.current_queue_size = queue_size
                self.stats.peak_queue_size = max(self.stats.peak_queue_size, queue_size)
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Stats collector error: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_expired_messages(self):
        """Clean up expired messages"""
        while not self._shutdown_event.is_set():
            try:
                # Clean up message history
                if len(self.message_history) > self.max_history_size:
                    self.message_history = self.message_history[-self.max_history_size:]
                
                # Clean up expired Redis keys
                pattern = f"{self.namespace}:message:*"
                keys = await self.redis.keys(pattern)
                
                for key in keys:
                    ttl = await self.redis.ttl(key)
                    if ttl == -1:  # No expiration set
                        await self.redis.expire(key, 3600)  # Set 1 hour expiration
                
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(60)
    
    async def _process_incoming_message(self, message_data: str):
        """Process incoming message from Redis"""
        try:
            message_dict = json.loads(message_data)
            message = AgentMessage.from_dict(message_dict)
            
            # Update stats
            self.stats.total_received += 1
            
            # Add to history
            self.message_history.append(message)
            
            # Apply filters
            if not self._should_process_message(message):
                return
            
            # Call registered handlers
            handlers = self.message_handlers.get(message.message_type, [])
            handlers.extend(self.message_handlers.get("*", []))  # Wildcard handlers
            
            for handler in handlers:
                try:
                    await handler(message)
                except Exception as e:
                    self.logger.error(f"Message handler error: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error processing incoming message: {e}")
    
    def _should_process_message(self, message: AgentMessage) -> bool:
        """Check if message should be processed based on filters"""
        for filter_name, message_filter in self.message_filters.items():
            if not message_filter.match(message):
                return False
        return True
    
    async def _handle_agent_event(self, event_data: Dict[str, Any]):
        """Handle agent lifecycle events"""
        event_type = event_data.get("type")
        agent_id = event_data.get("agent_id")
        
        if event_type == "agent_registered":
            agent_info = event_data.get("agent_info", {})
            self.known_agents[agent_id] = agent_info
            self.agent_capabilities[agent_id] = set(agent_info.get("capabilities", []))
            self.agent_status[agent_id] = AgentStatus(agent_info.get("status", "idle"))
            
        elif event_type == "agent_status_changed":
            new_status = event_data.get("status")
            if agent_id in self.agent_status:
                try:
                    self.agent_status[agent_id] = AgentStatus(new_status)
                except ValueError:
                    pass
                    
        elif event_type == "agent_unregistered":
            self.known_agents.pop(agent_id, None)
            self.agent_capabilities.pop(agent_id, None)
            self.agent_status.pop(agent_id, None)
    
    async def send_message(self, message: AgentMessage, 
                          routing: Optional[MessageRoute] = None) -> str:
        """Send a message through the bus"""
        try:
            message_id = message.id
            
            # Apply routing if specified
            if routing:
                await self._route_message(message, routing)
            else:
                # Direct send
                await self._send_direct_message(message)
            
            # Update stats
            self.stats.total_sent += 1
            
            # Store message for potential replay
            await self._store_message(message)
            
            return message_id
            
        except Exception as e:
            self.stats.total_failed += 1
            self.logger.error(f"Failed to send message {message.id}: {e}")
            raise
    
    async def _route_message(self, message: AgentMessage, routing: MessageRoute):
        """Route message based on strategy"""
        if routing.strategy == RoutingStrategy.DIRECT:
            await self._send_direct_message(message)
            
        elif routing.strategy == RoutingStrategy.BROADCAST:
            await self._send_broadcast_message(message)
            
        elif routing.strategy == RoutingStrategy.MULTICAST:
            target_agents = await self._find_capable_agents(routing.required_capabilities or [])
            for agent_id in target_agents:
                agent_message = AgentMessage(
                    id=str(uuid.uuid4()),
                    sender_id=message.sender_id,
                    receiver_id=agent_id,
                    message_type=message.message_type,
                    content=message.content,
                    priority=message.priority,
                    created_at=message.created_at,
                    expires_at=message.expires_at,
                    metadata=message.metadata
                )
                await self._send_direct_message(agent_message)
                
        elif routing.strategy == RoutingStrategy.ROUND_ROBIN:
            await self._send_round_robin(message, routing)
            
        elif routing.strategy == RoutingStrategy.LOAD_BALANCE:
            await self._send_load_balanced(message, routing)
            
        elif routing.strategy == RoutingStrategy.FAILOVER:
            await self._send_with_failover(message, routing)
    
    async def _send_direct_message(self, message: AgentMessage):
        """Send message directly to specified receiver"""
        channel = f"{self.namespace}:agent:{message.receiver_id}"
        await self.redis.publish(channel, json.dumps(message.to_dict()))
    
    async def _send_broadcast_message(self, message: AgentMessage):
        """Send message to all agents"""
        channel = f"{self.namespace}:agent:broadcast"
        await self.redis.publish(channel, json.dumps(message.to_dict()))
    
    async def _find_capable_agents(self, required_capabilities: List[str]) -> List[str]:
        """Find agents with required capabilities"""
        capable_agents = []
        
        for agent_id, capabilities in self.agent_capabilities.items():
            if all(cap in capabilities for cap in required_capabilities):
                # Check if agent is available
                status = self.agent_status.get(agent_id, AgentStatus.OFFLINE)
                if status in [AgentStatus.IDLE, AgentStatus.ACTIVE]:
                    capable_agents.append(agent_id)
        
        return capable_agents
    
    async def _send_round_robin(self, message: AgentMessage, routing: MessageRoute):
        """Send message using round-robin to capable agents"""
        capable_agents = await self._find_capable_agents(routing.required_capabilities or [])
        
        if not capable_agents:
            raise RuntimeError("No capable agents available")
        
        # Simple round-robin selection
        selected_agent = capable_agents[self.stats.total_sent % len(capable_agents)]
        
        message.receiver_id = selected_agent
        await self._send_direct_message(message)
    
    async def _send_load_balanced(self, message: AgentMessage, routing: MessageRoute):
        """Send message to least busy capable agent"""
        capable_agents = await self._find_capable_agents(routing.required_capabilities or [])
        
        if not capable_agents:
            raise RuntimeError("No capable agents available")
        
        # Get agent loads (active tasks)
        agent_loads = {}
        for agent_id in capable_agents:
            heartbeat_key = f"{self.namespace}:heartbeat:{agent_id}"
            heartbeat_data = await self.redis.get(heartbeat_key)
            
            if heartbeat_data:
                try:
                    heartbeat = json.loads(heartbeat_data)
                    agent_loads[agent_id] = heartbeat.get("active_tasks", 0)
                except:
                    agent_loads[agent_id] = 0
            else:
                agent_loads[agent_id] = float('inf')  # Agent not responding
        
        # Select least busy agent
        selected_agent = min(agent_loads.keys(), key=lambda x: agent_loads[x])
        
        message.receiver_id = selected_agent
        await self._send_direct_message(message)
    
    async def _send_with_failover(self, message: AgentMessage, routing: MessageRoute):
        """Send message with failover to backup agents"""
        target_agents = routing.target_agents or await self._find_capable_agents(
            routing.required_capabilities or []
        )
        
        if not target_agents:
            raise RuntimeError("No target agents available")
        
        last_error = None
        for agent_id in target_agents:
            try:
                message.receiver_id = agent_id
                await self._send_direct_message(message)
                return  # Success, stop trying
                
            except Exception as e:
                last_error = e
                continue
        
        # All agents failed
        raise RuntimeError(f"All failover attempts failed. Last error: {last_error}")
    
    async def _store_message(self, message: AgentMessage):
        """Store message for potential replay"""
        message_key = f"{self.namespace}:message:{message.id}"
        await self.redis.setex(
            message_key,
            3600,  # 1 hour TTL
            json.dumps(message.to_dict())
        )
    
    async def subscribe_to_messages(self, agent_id: str, 
                                   message_handler: Callable[[AgentMessage], None]):
        """Subscribe agent to receive messages"""
        # Subscribe to agent-specific channel
        await self.pubsub.subscribe(f"{self.namespace}:agent:{agent_id}")
        
        # Subscribe to broadcast channel
        await self.pubsub.subscribe(f"{self.namespace}:agent:broadcast")
        
        # Register handler
        if agent_id not in self.message_handlers:
            self.message_handlers[agent_id] = []
        self.message_handlers[agent_id].append(message_handler)
        
        self.active_subscriptions.add(agent_id)
        self.logger.info(f"Subscribed agent {agent_id} to messages")
    
    def register_message_handler(self, message_type: str, 
                                handler: Callable[[AgentMessage], None]):
        """Register a handler for specific message types"""
        if message_type not in self.message_handlers:
            self.message_handlers[message_type] = []
        self.message_handlers[message_type].append(handler)
    
    def add_message_filter(self, filter_name: str, criteria: Dict[str, Any]):
        """Add a message filter"""
        self.message_filters[filter_name] = MessageFilter(criteria)
    
    def remove_message_filter(self, filter_name: str):
        """Remove a message filter"""
        self.message_filters.pop(filter_name, None)
    
    def add_routing_rule(self, pattern: str, route: MessageRoute):
        """Add a routing rule for message patterns"""
        self.routing_rules[pattern] = route
    
    async def get_message_history(self, agent_id: str = None, 
                                 message_type: str = None,
                                 limit: int = 100) -> List[AgentMessage]:
        """Get message history with optional filtering"""
        messages = self.message_history[-limit:]
        
        if agent_id:
            messages = [m for m in messages 
                       if m.sender_id == agent_id or m.receiver_id == agent_id]
        
        if message_type:
            messages = [m for m in messages if m.message_type == message_type]
        
        return messages
    
    async def replay_messages(self, agent_id: str, from_time: datetime):
        """Replay messages for an agent from a specific time"""
        messages = [m for m in self.message_history 
                   if m.created_at >= from_time and 
                   (m.receiver_id == agent_id or m.receiver_id == "broadcast")]
        
        for message in messages:
            await self._send_direct_message(message)
    
    def get_stats(self) -> MessageStats:
        """Get message bus statistics"""
        return self.stats
    
    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific agent"""
        return self.known_agents.get(agent_id)
    
    def get_all_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all known agents"""
        return self.known_agents.copy()
    
    async def ping_agent(self, agent_id: str, timeout: float = 5.0) -> bool:
        """Ping an agent to check if it's responsive"""
        ping_message = AgentMessage(
            sender_id="message_bus",
            receiver_id=agent_id,
            message_type="ping",
            content={"timestamp": datetime.utcnow().isoformat()}
        )
        
        try:
            await self.send_message(ping_message)
            
            # Wait for pong response
            start_time = time.time()
            while time.time() - start_time < timeout:
                # Check for pong in recent messages
                recent_messages = self.message_history[-10:]
                for msg in recent_messages:
                    if (msg.sender_id == agent_id and 
                        msg.message_type == "pong" and
                        msg.content.get("ping_id") == ping_message.id):
                        return True
                
                await asyncio.sleep(0.1)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error pinging agent {agent_id}: {e}")
            return False
    
    async def broadcast_system_message(self, message_type: str, content: Dict[str, Any]):
        """Broadcast a system message to all agents"""
        system_message = AgentMessage(
            sender_id="system",
            receiver_id="broadcast",
            message_type=message_type,
            content=content,
            priority=1  # High priority for system messages
        )
        
        await self.send_message(system_message)
    
    async def shutdown(self):
        """Shutdown the message bus"""
        self.logger.info("Shutting down Agent Message Bus")
        
        # Signal shutdown to background tasks
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Close Redis connections
        if self.pubsub:
            await self.pubsub.close()
        if self.redis:
            await self.redis.close()
        
        self.logger.info("Agent Message Bus shutdown complete")


# Global message bus instance
_message_bus_instance: Optional[AgentMessageBus] = None


def get_message_bus() -> AgentMessageBus:
    """Get the global message bus instance"""
    global _message_bus_instance
    if _message_bus_instance is None:
        _message_bus_instance = AgentMessageBus()
    return _message_bus_instance


def set_message_bus(message_bus: AgentMessageBus):
    """Set the global message bus instance"""
    global _message_bus_instance
    _message_bus_instance = message_bus


# Convenience functions
async def send_message(sender_id: str, receiver_id: str, message_type: str,
                      content: Dict[str, Any], priority: int = 5) -> str:
    """Send message using global message bus"""
    message_bus = get_message_bus()
    message = AgentMessage(
        sender_id=sender_id,
        receiver_id=receiver_id,
        message_type=message_type,
        content=content,
        priority=priority
    )
    return await message_bus.send_message(message)


async def broadcast_message(sender_id: str, message_type: str,
                           content: Dict[str, Any], priority: int = 5) -> str:
    """Broadcast message using global message bus"""
    return await send_message(sender_id, "broadcast", message_type, content, priority)
