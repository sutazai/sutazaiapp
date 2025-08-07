"""
SutazAI Message Bus for Real-time Agent Communication
Handles pub/sub messaging, task routing, and inter-agent coordination
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
from dataclasses import asdict
from enum import Enum
import redis.asyncio as redis
from collections import defaultdict

logger = logging.getLogger(__name__)

from backend.app.schemas.message_types import MessageType
from backend.app.schemas.message_model import Message

class MessageBus:
    """
    High-performance message bus for agent communication
    
    Features:
    - Pub/Sub messaging with Redis
    - Message routing and filtering
    - Priority queues
    - Message persistence
    - Dead letter queues
    - Message correlation
    - Broadcast and unicast messaging
    """
    
    def __init__(self, redis_url: str = "redis://:redis_password@localhost:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.running = False
        
        # Message routing
        self.routing_table: Dict[str, str] = {}
        self.message_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        
        # Performance metrics
        self.metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_failed": 0,
            "avg_delivery_time": 0.0
        }
    
    async def initialize(self):
        """Initialize the message bus"""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            
            # Start background tasks
            asyncio.create_task(self._message_listener())
            asyncio.create_task(self._metrics_collector())
            
            self.running = True
            logger.info("Message bus initialized successfully")
            
        except Exception as e:
            logger.error(f"Message bus initialization failed: {e}")
            raise
    
    async def send_message(self, message: Message) -> bool:
        """Send a message to the bus"""
        try:
            # Serialize message (canonical shape)
            message_data = json.dumps(message.to_dict(), default=str)
            
            # Determine routing
            if message.recipient == "*":
                # Broadcast message
                await self.redis_client.publish("broadcast", message_data)
            else:
                # Direct message
                channel = f"agent:{message.recipient}"
                await self.redis_client.publish(channel, message_data)
            
            # Store in persistence layer if needed
            if message.ttl and message.ttl > 0:
                await self.redis_client.setex(
                    f"message:{message.id}",
                    message.ttl,
                    message_data
                )
            
            self.metrics["messages_sent"] += 1
            logger.debug(f"Message sent: {message.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message {message.id}: {e}")
            self.metrics["messages_failed"] += 1
            return False
    
    async def subscribe_to_agent_messages(self, agent_id: str, handler: Callable):
        """Subscribe to messages for a specific agent"""
        channel = f"agent:{agent_id}"
        self.subscribers[channel].append(handler)
        
        # Subscribe to Redis channel
        pubsub = self.redis_client.pubsub()
        await pubsub.subscribe(channel)
        
        # Start listener for this channel
        asyncio.create_task(self._channel_listener(pubsub, channel))
        
        logger.info(f"Subscribed to messages for agent: {agent_id}")
    
    async def subscribe_to_broadcasts(self, handler: Callable):
        """Subscribe to broadcast messages"""
        self.subscribers["broadcast"].append(handler)
        
        # Subscribe to Redis broadcast channel
        pubsub = self.redis_client.pubsub()
        await pubsub.subscribe("broadcast")
        
        # Start listener for broadcast channel
        asyncio.create_task(self._channel_listener(pubsub, "broadcast"))
        
        logger.info("Subscribed to broadcast messages")
    
    async def subscribe_to_message_type(self, message_type: MessageType, handler: Callable):
        """Subscribe to specific message types"""
        self.message_handlers[message_type].append(handler)
        logger.info(f"Subscribed to message type: {message_type.value}")
    
    async def _channel_listener(self, pubsub, channel: str):
        """Listen for messages on a specific channel"""
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        # Parse message
                        message_data = json.loads(message["data"])
                        msg = Message.from_dict(message_data)
                        
                        # Route to handlers
                        await self._route_message(msg, channel)
                        
                        self.metrics["messages_received"] += 1
                        
                    except Exception as e:
                        logger.error(f"Failed to process message on {channel}: {e}")
                        
        except Exception as e:
            logger.error(f"Channel listener error for {channel}: {e}")
    
    async def _route_message(self, message: Message, channel: str):
        """Route message to appropriate handlers"""
        # Route to channel subscribers
        for handler in self.subscribers.get(channel, []):
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Handler error: {e}")
        
        # Route to message type handlers
        for handler in self.message_handlers.get(message.type, []):
            try:
                await handler(message)
            except Exception as e:
                logger.error(f"Message type handler error: {e}")
    
    async def _message_listener(self):
        """Main message listener loop"""
        while self.running:
            try:
                # Process any queued messages
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Message listener error: {e}")
                await asyncio.sleep(1)
    
    async def _metrics_collector(self):
        """Collect message bus metrics"""
        while self.running:
            try:
                # Store metrics in Redis
                await self.redis_client.hset(
                    "message_bus_metrics",
                    "current",
                    json.dumps(self.metrics)
                )
                
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Metrics collector error: {e}")
                await asyncio.sleep(10)
    
    async def send_task_assignment(self, agent_id: str, task_data: Dict[str, Any]) -> str:
        """Send task assignment to an agent"""
        import uuid

        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.TASK_ASSIGNMENT,
            sender="orchestrator",
            recipient=agent_id,
            payload=task_data,
            timestamp=datetime.now(),
            requires_response=True,
        )
        
        await self.send_message(message)
        return message.id
    
    async def send_task_result(self, orchestrator_id: str, task_id: str, result: Dict[str, Any]) -> str:
        """Send task result back to orchestrator"""
        import uuid

        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.TASK_RESULT,
            sender="agent",
            recipient=orchestrator_id,
            payload={
                "task_id": task_id,
                "result": result,
            },
            timestamp=datetime.now(),
        )
        
        await self.send_message(message)
        return message.id
    
    async def send_heartbeat(self, agent_id: str, health_data: Dict[str, Any]) -> str:
        """Send agent heartbeat"""
        import uuid

        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.HEARTBEAT,
            sender=agent_id,
            recipient="orchestrator",
            payload=health_data,
            timestamp=datetime.now(),
        )
        
        await self.send_message(message)
        return message.id
    
    async def send_status_update(self, agent_id: str, status_data: Dict[str, Any]) -> str:
        """Send agent status update"""
        import uuid

        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.STATUS_UPDATE,
            sender=agent_id,
            recipient="orchestrator",
            payload=status_data,
            timestamp=datetime.now(),
        )
        
        await self.send_message(message)
        return message.id
    
    async def broadcast_system_notification(self, notification: str, data: Dict[str, Any] = None) -> str:
        """Broadcast system notification to all agents"""
        import uuid

        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.SYSTEM_NOTIFICATION,
            sender="system",
            recipient="*",
            payload={
                "notification": notification,
                "data": data or {},
            },
            timestamp=datetime.now(),
        )
        
        await self.send_message(message)
        return message.id
    
    async def request_coordination(self, sender_id: str, coordination_type: str, data: Dict[str, Any]) -> str:
        """Send coordination request"""
        import uuid

        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.COORDINATION_REQUEST,
            sender=sender_id,
            recipient="orchestrator",
            payload={
                "coordination_type": coordination_type,
                "data": data,
            },
            timestamp=datetime.now(),
            requires_response=True,
        )
        
        await self.send_message(message)
        return message.id
    
    async def send_coordination_response(self, recipient_id: str, correlation_id: str, response_data: Dict[str, Any]) -> str:
        """Send coordination response"""
        import uuid

        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.COORDINATION_RESPONSE,
            sender="orchestrator",
            recipient=recipient_id,
            payload=response_data,
            timestamp=datetime.now(),
            correlation_id=correlation_id,
        )
        
        await self.send_message(message)
        return message.id
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get message bus metrics"""
        return {
            **self.metrics,
            "active_subscriptions": sum(len(handlers) for handlers in self.subscribers.values()),
            "message_types_handled": len(self.message_handlers),
            "running": self.running
        }
    
    async def stop(self):
        """Stop the message bus"""
        self.running = False
        if self.redis_client:
            await self.redis_client.close()
        logger.info("Message bus stopped")

# Singleton instance
message_bus = MessageBus()
