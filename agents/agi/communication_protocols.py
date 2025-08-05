#!/usr/bin/env python3
"""
Inter-Agent Communication Protocols for AGI Orchestration
Handles all communication between agents in the AGI system
"""

import asyncio
import json
import logging
import redis.asyncio as aioredis
import hashlib
import time
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in the AGI communication system"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    TASK_UPDATE = "task_update"
    TASK_COMPLETION = "task_completion"
    AGENT_HEARTBEAT = "agent_heartbeat"
    AGENT_STATUS = "agent_status"
    CONSENSUS_VOTE = "consensus_vote"
    CONSENSUS_REQUEST = "consensus_request"
    COORDINATION_EVENT = "coordination_event"
    EMERGENT_BEHAVIOR = "emergent_behavior"
    SAFETY_ALERT = "safety_alert"
    PERFORMANCE_METRIC = "performance_metric"
    META_LEARNING = "meta_learning"
    RESOURCE_REQUEST = "resource_request"
    SYSTEM_COMMAND = "system_command"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class DeliveryMode(Enum):
    """Message delivery modes"""
    FIRE_AND_FORGET = "fire_and_forget"
    ACKNOWLEDGMENT = "acknowledgment"
    RELIABLE = "reliable"
    ORDERED = "ordered"


@dataclass
class Message:
    """Represents a message in the AGI communication system"""
    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: Optional[str]  # None for broadcast
    priority: MessagePriority
    delivery_mode: DeliveryMode
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    ttl: int = 3600  # Time to live in seconds
    created_at: datetime = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender_id": self.sender_id,
            "recipient_id": self.recipient_id,
            "priority": self.priority.value,
            "delivery_mode": self.delivery_mode.value,
            "payload": self.payload,
            "correlation_id": self.correlation_id,
            "reply_to": self.reply_to,
            "ttl": self.ttl,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """Create message from dictionary"""
        return cls(
            message_id=data["message_id"],
            message_type=MessageType(data["message_type"]),
            sender_id=data["sender_id"],
            recipient_id=data.get("recipient_id"),
            priority=MessagePriority(data["priority"]),
            delivery_mode=DeliveryMode(data["delivery_mode"]),
            payload=data["payload"],
            correlation_id=data.get("correlation_id"),
            reply_to=data.get("reply_to"),
            ttl=data.get("ttl", 3600),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
            retry_count=data.get("retry_count", 0),
            max_retries=data.get("max_retries", 3)
        )


@dataclass
class CommunicationChannel:
    """Represents a communication channel"""
    channel_id: str
    name: str
    description: str
    message_types: Set[MessageType]
    participants: Set[str]
    retention_period: int = 3600  # seconds
    max_message_size: int = 1048576  # 1MB
    compression_enabled: bool = True
    encryption_enabled: bool = False


class CommunicationProtocol:
    """
    Advanced communication protocol for AGI agent coordination
    """
    
    def __init__(self, 
                 redis_url: str = "redis://redis:6379/1",
                 agent_id: str = None):
        
        self.redis_url = redis_url
        self.agent_id = agent_id or f"agi_protocol_{uuid.uuid4().hex[:8]}"
        self.redis_client: Optional[aioredis.Redis] = None
        
        # Communication channels
        self.channels: Dict[str, CommunicationChannel] = {}
        self.subscriptions: Dict[str, Set[Callable]] = {}
        self.message_handlers: Dict[MessageType, Callable] = {}
        
        # Message tracking
        self.pending_messages: Dict[str, Message] = {}
        self.message_history: List[Message] = []
        self.delivery_confirmations: Dict[str, datetime] = {}
        
        # Performance metrics
        self.metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "messages_failed": 0,
            "avg_delivery_time": 0.0,
            "active_connections": 0
        }
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        # Initialize default channels
        self._initialize_default_channels()
    
    async def initialize(self):
        """Initialize the communication protocol"""
        
        try:
            # Connect to Redis
            self.redis_client = aioredis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Register agent
            await self._register_agent()
            
            logger.info(f"Communication protocol initialized for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize communication protocol: {e}")
            raise
    
    def _initialize_default_channels(self):
        """Initialize default communication channels"""
        
        default_channels = [
            CommunicationChannel(
                channel_id="agi:tasks",
                name="Task Communication",
                description="Task requests, responses, and updates",
                message_types={
                    MessageType.TASK_REQUEST,
                    MessageType.TASK_RESPONSE,
                    MessageType.TASK_UPDATE,
                    MessageType.TASK_COMPLETION
                },
                participants=set(),
                retention_period=86400  # 24 hours
            ),
            CommunicationChannel(
                channel_id="agi:agents",
                name="Agent Communication",
                description="Agent heartbeats and status updates",
                message_types={
                    MessageType.AGENT_HEARTBEAT,
                    MessageType.AGENT_STATUS
                },
                participants=set(),
                retention_period=3600  # 1 hour
            ),
            CommunicationChannel(
                channel_id="agi:consensus",
                name="Consensus Communication",
                description="Consensus voting and requests",
                message_types={
                    MessageType.CONSENSUS_VOTE,
                    MessageType.CONSENSUS_REQUEST
                },
                participants=set(),
                retention_period=1800  # 30 minutes
            ),
            CommunicationChannel(
                channel_id="agi:coordination",
                name="Coordination Events",
                description="Agent coordination and collaboration events",
                message_types={
                    MessageType.COORDINATION_EVENT,
                    MessageType.EMERGENT_BEHAVIOR
                },
                participants=set(),
                retention_period=7200  # 2 hours
            ),
            CommunicationChannel(
                channel_id="agi:safety",
                name="Safety Alerts",
                description="Safety alerts and critical system events",
                message_types={
                    MessageType.SAFETY_ALERT,
                    MessageType.SYSTEM_COMMAND
                },
                participants=set(),
                retention_period=604800,  # 7 days
                compression_enabled=False,  # Keep safety messages uncompressed
                encryption_enabled=True    # Encrypt safety messages
            ),
            CommunicationChannel(
                channel_id="agi:performance",
                name="Performance Metrics",
                description="Performance metrics and monitoring data",
                message_types={
                    MessageType.PERFORMANCE_METRIC,
                    MessageType.META_LEARNING
                },
                participants=set(),
                retention_period=86400  # 24 hours
            ),
            CommunicationChannel(
                channel_id="agi:resources",
                name="Resource Management",
                description="Resource requests and allocation updates",
                message_types={
                    MessageType.RESOURCE_REQUEST
                },
                participants=set(),
                retention_period=3600  # 1 hour
            )
        ]
        
        for channel in default_channels:
            self.channels[channel.channel_id] = channel
    
    async def _start_background_tasks(self):
        """Start background communication tasks"""
        
        # Message listener task
        listener_task = asyncio.create_task(self._message_listener())
        self._background_tasks.add(listener_task)
        
        # Message cleanup task
        cleanup_task = asyncio.create_task(self._message_cleanup())
        self._background_tasks.add(cleanup_task)
        
        # Heartbeat task
        heartbeat_task = asyncio.create_task(self._heartbeat_sender())
        self._background_tasks.add(heartbeat_task)
        
        # Retry handler task
        retry_task = asyncio.create_task(self._retry_handler())
        self._background_tasks.add(retry_task)
        
        # Metrics collector task
        metrics_task = asyncio.create_task(self._metrics_collector())
        self._background_tasks.add(metrics_task)
        
        logger.info("Background communication tasks started")
    
    async def _register_agent(self):
        """Register this agent in the communication system"""
        
        registration_message = Message(
            message_id=self._generate_message_id(),
            message_type=MessageType.AGENT_STATUS,
            sender_id=self.agent_id,
            recipient_id=None,  # Broadcast
            priority=MessagePriority.MEDIUM,
            delivery_mode=DeliveryMode.FIRE_AND_FORGET,
            payload={
                "status": "online",
                "capabilities": [],
                "supported_message_types": [mt.value for mt in MessageType],
                "agent_type": "communication_protocol",
                "version": "1.0.0",
                "registration_time": datetime.utcnow().isoformat()
            }
        )
        
        await self.send_message(registration_message, "agi:agents")
        logger.info(f"Agent {self.agent_id} registered in communication system")
    
    async def send_message(self, message: Message, channel_id: str) -> bool:
        """Send a message to a specific channel"""
        
        try:
            # Validate channel
            if channel_id not in self.channels:
                logger.error(f"Unknown channel: {channel_id}")
                return False
            
            channel = self.channels[channel_id]
            
            # Validate message type for channel
            if message.message_type not in channel.message_types:
                logger.error(f"Message type {message.message_type} not allowed in channel {channel_id}")
                return False
            
            # Serialize message
            message_data = json.dumps(message.to_dict())
            
            # Check message size
            if len(message_data) > channel.max_message_size:
                logger.error(f"Message too large: {len(message_data)} > {channel.max_message_size}")
                return False
            
            # Compress if enabled
            if channel.compression_enabled and len(message_data) > 1024:
                message_data = await self._compress_message(message_data)
            
            # Encrypt if enabled
            if channel.encryption_enabled:
                message_data = await self._encrypt_message(message_data)
            
            # Send based on delivery mode
            if message.delivery_mode == DeliveryMode.FIRE_AND_FORGET:
                await self._send_fire_and_forget(message_data, channel_id)
            
            elif message.delivery_mode == DeliveryMode.ACKNOWLEDGMENT:
                success = await self._send_with_acknowledgment(message, message_data, channel_id)
                if not success:
                    return False
            
            elif message.delivery_mode == DeliveryMode.RELIABLE:
                await self._send_reliable(message, message_data, channel_id)
            
            elif message.delivery_mode == DeliveryMode.ORDERED:
                await self._send_ordered(message, message_data, channel_id)
            
            # Update metrics
            self.metrics["messages_sent"] += 1
            
            # Store in history
            self.message_history.append(message)
            if len(self.message_history) > 10000:  # Keep last 10k messages
                self.message_history = self.message_history[-10000:]
            
            logger.debug(f"Message sent: {message.message_id} to {channel_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message {message.message_id}: {e}")
            self.metrics["messages_failed"] += 1
            return False
    
    async def _send_fire_and_forget(self, message_data: str, channel_id: str):
        """Send message with fire-and-forget delivery"""
        await self.redis_client.publish(channel_id, message_data)
    
    async def _send_with_acknowledgment(self, message: Message, message_data: str, channel_id: str) -> bool:
        """Send message with acknowledgment requirement"""
        
        # Store for acknowledgment tracking
        self.pending_messages[message.message_id] = message
        
        # Send message
        await self.redis_client.publish(channel_id, message_data)
        
        # Wait for acknowledgment (with timeout)
        timeout = 30  # 30 seconds
        start_time = time.time()
        
        while message.message_id not in self.delivery_confirmations:
            if time.time() - start_time > timeout:
                logger.warning(f"Acknowledgment timeout for message {message.message_id}")
                self.pending_messages.pop(message.message_id, None)
                return False
            
            await asyncio.sleep(0.1)
        
        # Remove from pending
        self.pending_messages.pop(message.message_id, None)
        self.delivery_confirmations.pop(message.message_id, None)
        
        return True
    
    async def _send_reliable(self, message: Message, message_data: str, channel_id: str):
        """Send message with reliable delivery (retries on failure)"""
        
        # Store for retry handling
        self.pending_messages[message.message_id] = message
        
        # Send message
        await self.redis_client.publish(channel_id, message_data)
        
        # Reliable delivery will be handled by retry handler
    
    async def _send_ordered(self, message: Message, message_data: str, channel_id: str):
        """Send message with ordered delivery guarantee"""
        
        # Use Redis streams for ordered delivery
        stream_key = f"{channel_id}:ordered"
        
        await self.redis_client.xadd(
            stream_key,
            {
                "message_id": message.message_id,
                "sender_id": message.sender_id,
                "data": message_data
            }
        )
    
    async def subscribe_to_channel(self, channel_id: str, 
                                 message_handler: Callable[[Message], None]):
        """Subscribe to a communication channel"""
        
        if channel_id not in self.channels:
            logger.error(f"Unknown channel: {channel_id}")
            return
        
        if channel_id not in self.subscriptions:
            self.subscriptions[channel_id] = set()
        
        self.subscriptions[channel_id].add(message_handler)
        
        # Add agent to channel participants
        self.channels[channel_id].participants.add(self.agent_id)
        
        logger.info(f"Subscribed to channel {channel_id}")
    
    async def unsubscribe_from_channel(self, channel_id: str, 
                                     message_handler: Callable[[Message], None]):
        """Unsubscribe from a communication channel"""
        
        if channel_id in self.subscriptions:
            self.subscriptions[channel_id].discard(message_handler)
            
            if not self.subscriptions[channel_id]:
                del self.subscriptions[channel_id]
                
                # Remove agent from channel participants
                if channel_id in self.channels:
                    self.channels[channel_id].participants.discard(self.agent_id)
        
        logger.info(f"Unsubscribed from channel {channel_id}")
    
    def register_message_handler(self, message_type: MessageType, 
                                handler: Callable[[Message], None]):
        """Register a handler for a specific message type"""
        self.message_handlers[message_type] = handler
    
    async def broadcast_message(self, message: Message):
        """Broadcast a message to all relevant channels"""
        
        # Find channels that support this message type
        relevant_channels = [
            channel_id for channel_id, channel in self.channels.items()
            if message.message_type in channel.message_types
        ]
        
        # Send to all relevant channels
        success_count = 0
        for channel_id in relevant_channels:
            if await self.send_message(message, channel_id):
                success_count += 1
        
        logger.info(f"Broadcast message {message.message_id} to {success_count}/{len(relevant_channels)} channels")
        return success_count > 0
    
    async def send_task_request(self, task_id: str, task_description: str, 
                              target_agents: List[str] = None) -> str:
        """Send a task request message"""
        
        message = Message(
            message_id=self._generate_message_id(),
            message_type=MessageType.TASK_REQUEST,
            sender_id=self.agent_id,
            recipient_id=None if not target_agents else ",".join(target_agents),
            priority=MessagePriority.MEDIUM,
            delivery_mode=DeliveryMode.ACKNOWLEDGMENT,
            payload={
                "task_id": task_id,
                "description": task_description,
                "requested_at": datetime.utcnow().isoformat(),
                "target_agents": target_agents or [],
                "timeout": 3600
            },
            ttl=3600
        )
        
        await self.send_message(message, "agi:tasks")
        return message.message_id
    
    async def send_task_response(self, task_id: str, result: Dict[str, Any], 
                               correlation_id: str):
        """Send a task response message"""
        
        message = Message(
            message_id=self._generate_message_id(),
            message_type=MessageType.TASK_RESPONSE,
            sender_id=self.agent_id,
            recipient_id=None,  # Will be routed by correlation
            priority=MessagePriority.MEDIUM,
            delivery_mode=DeliveryMode.RELIABLE,
            payload={
                "task_id": task_id,
                "result": result,
                "completed_at": datetime.utcnow().isoformat(),
                "success": result.get("success", True),
                "execution_time": result.get("execution_time", 0)
            },
            correlation_id=correlation_id,
            ttl=7200
        )
        
        await self.send_message(message, "agi:tasks")
        return message.message_id
    
    async def send_consensus_vote(self, proposal_id: str, vote: float, 
                                confidence: float, reasoning: str):
        """Send a consensus vote message"""
        
        message = Message(
            message_id=self._generate_message_id(),
            message_type=MessageType.CONSENSUS_VOTE,
            sender_id=self.agent_id,
            recipient_id=None,  # Broadcast to consensus system
            priority=MessagePriority.HIGH,
            delivery_mode=DeliveryMode.RELIABLE,
            payload={
                "proposal_id": proposal_id,
                "vote": vote,
                "confidence": confidence,
                "reasoning": reasoning,
                "voter_id": self.agent_id,
                "timestamp": datetime.utcnow().isoformat()
            },
            ttl=1800
        )
        
        await self.send_message(message, "agi:consensus")
        return message.message_id
    
    async def send_emergent_behavior_alert(self, behavior_type: str, 
                                         participants: List[str], 
                                         impact_score: float,
                                         evidence: Dict[str, Any]):
        """Send an emergent behavior alert"""
        
        message = Message(
            message_id=self._generate_message_id(),
            message_type=MessageType.EMERGENT_BEHAVIOR,
            sender_id=self.agent_id,
            recipient_id=None,
            priority=MessagePriority.HIGH,
            delivery_mode=DeliveryMode.RELIABLE,
            payload={
                "behavior_type": behavior_type,
                "participants": participants,
                "impact_score": impact_score,
                "evidence": evidence,
                "detection_time": datetime.utcnow().isoformat(),
                "detector_id": self.agent_id
            },
            ttl=7200
        )
        
        await self.send_message(message, "agi:coordination")
        return message.message_id
    
    async def send_safety_alert(self, alert_level: str, alert_message: str, 
                              alert_data: Dict[str, Any]):
        """Send a safety alert message"""
        
        priority_map = {
            "info": MessagePriority.LOW,
            "warning": MessagePriority.MEDIUM,
            "critical": MessagePriority.CRITICAL,
            "emergency": MessagePriority.EMERGENCY
        }
        
        message = Message(
            message_id=self._generate_message_id(),
            message_type=MessageType.SAFETY_ALERT,
            sender_id=self.agent_id,
            recipient_id=None,  # Broadcast to all safety monitors
            priority=priority_map.get(alert_level, MessagePriority.HIGH),
            delivery_mode=DeliveryMode.RELIABLE,
            payload={
                "alert_level": alert_level,
                "alert_message": alert_message,
                "alert_data": alert_data,
                "timestamp": datetime.utcnow().isoformat(),
                "source_agent": self.agent_id
            },
            ttl=86400  # Keep safety alerts for 24 hours
        )
        
        await self.send_message(message, "agi:safety")
        return message.message_id
    
    async def send_performance_metric(self, metric_name: str, metric_value: float,
                                    metric_metadata: Dict[str, Any] = None):
        """Send a performance metric message"""
        
        message = Message(
            message_id=self._generate_message_id(),
            message_type=MessageType.PERFORMANCE_METRIC,
            sender_id=self.agent_id,
            recipient_id=None,
            priority=MessagePriority.LOW,
            delivery_mode=DeliveryMode.FIRE_AND_FORGET,
            payload={
                "metric_name": metric_name,
                "metric_value": metric_value,
                "metadata": metric_metadata or {},
                "timestamp": datetime.utcnow().isoformat(),
                "agent_id": self.agent_id
            },
            ttl=86400
        )
        
        await self.send_message(message, "agi:performance")
        return message.message_id
    
    async def _message_listener(self):
        """Background task to listen for incoming messages"""
        
        try:
            pubsub = self.redis_client.pubsub()
            
            # Subscribe to all channels we're interested in
            for channel_id in self.subscriptions.keys():
                await pubsub.subscribe(channel_id)
            
            async for message in pubsub.listen():
                if message["type"] == "message":
                    await self._handle_incoming_message(
                        message["channel"], 
                        message["data"]
                    )
                    
        except Exception as e:
            logger.error(f"Message listener error: {e}")
    
    async def _handle_incoming_message(self, channel_id: str, message_data: str):
        """Handle an incoming message"""
        
        try:
            # Decrypt if needed
            if self.channels[channel_id].encryption_enabled:
                message_data = await self._decrypt_message(message_data)
            
            # Decompress if needed
            if self.channels[channel_id].compression_enabled:
                message_data = await self._decompress_message(message_data)
            
            # Parse message
            message_dict = json.loads(message_data)
            message = Message.from_dict(message_dict)
            
            # Update metrics
            self.metrics["messages_received"] += 1
            
            # Send acknowledgment if required
            if message.delivery_mode == DeliveryMode.ACKNOWLEDGMENT:
                await self._send_acknowledgment(message, channel_id)
            
            # Route message to handlers
            await self._route_message(message, channel_id)
            
        except Exception as e:
            logger.error(f"Error handling incoming message: {e}")
    
    async def _route_message(self, message: Message, channel_id: str):
        """Route message to appropriate handlers"""
        
        # Call channel-specific handlers
        if channel_id in self.subscriptions:
            for handler in self.subscriptions[channel_id]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(message)
                    else:
                        handler(message)
                except Exception as e:
                    logger.error(f"Error in message handler: {e}")
        
        # Call message type-specific handlers
        if message.message_type in self.message_handlers:
            handler = self.message_handlers[message.message_type]
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                logger.error(f"Error in message type handler: {e}")
    
    async def _send_acknowledgment(self, message: Message, channel_id: str):
        """Send acknowledgment for a received message"""
        
        ack_message = Message(
            message_id=self._generate_message_id(),
            message_type=MessageType.SYSTEM_COMMAND,
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            priority=MessagePriority.HIGH,
            delivery_mode=DeliveryMode.FIRE_AND_FORGET,
            payload={
                "command": "acknowledgment",
                "original_message_id": message.message_id,
                "ack_timestamp": datetime.utcnow().isoformat()
            }
        )
        
        # Send acknowledgment on a special channel
        await self._send_fire_and_forget(
            json.dumps(ack_message.to_dict()),
            f"{channel_id}:ack"
        )
    
    async def _heartbeat_sender(self):
        """Send periodic heartbeat messages"""
        
        while not self._shutdown_event.is_set():
            try:
                heartbeat_message = Message(
                    message_id=self._generate_message_id(),
                    message_type=MessageType.AGENT_HEARTBEAT,
                    sender_id=self.agent_id,
                    recipient_id=None,
                    priority=MessagePriority.LOW,
                    delivery_mode=DeliveryMode.FIRE_AND_FORGET,
                    payload={
                        "status": "alive",
                        "timestamp": datetime.utcnow().isoformat(),
                        "metrics": self.metrics.copy(),
                        "active_subscriptions": list(self.subscriptions.keys())
                    }
                )
                
                await self.send_message(heartbeat_message, "agi:agents")
                
            except Exception as e:
                logger.error(f"Heartbeat sender error: {e}")
            
            await asyncio.sleep(30)  # Send heartbeat every 30 seconds
    
    async def _retry_handler(self):
        """Handle message retries for reliable delivery"""
        
        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.utcnow()
                retry_messages = []
                
                # Find messages that need retry
                for message_id, message in self.pending_messages.items():
                    if message.delivery_mode == DeliveryMode.RELIABLE:
                        age = (current_time - message.created_at).total_seconds()
                        
                        if age > 60 and message.retry_count < message.max_retries:  # Retry after 60 seconds
                            retry_messages.append(message)
                
                # Retry messages
                for message in retry_messages:
                    message.retry_count += 1
                    
                    # Find appropriate channel
                    for channel_id, channel in self.channels.items():
                        if message.message_type in channel.message_types:
                            await self.send_message(message, channel_id)
                            break
                    
                    logger.info(f"Retried message {message.message_id} (attempt {message.retry_count})")
                
                # Clean up messages that exceeded max retries
                expired_messages = [
                    msg_id for msg_id, msg in self.pending_messages.items()
                    if msg.retry_count >= msg.max_retries
                ]
                
                for msg_id in expired_messages:
                    self.pending_messages.pop(msg_id, None)
                    self.metrics["messages_failed"] += 1
                    logger.warning(f"Message {msg_id} exceeded max retries")
                
            except Exception as e:
                logger.error(f"Retry handler error: {e}")
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def _message_cleanup(self):
        """Clean up expired messages and maintain storage limits"""
        
        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.utcnow()
                
                # Clean up message history
                if len(self.message_history) > 10000:
                    self.message_history = self.message_history[-5000:]  # Keep last 5000
                
                # Clean up pending messages based on TTL
                expired_pending = [
                    msg_id for msg_id, msg in self.pending_messages.items()
                    if (current_time - msg.created_at).total_seconds() > msg.ttl
                ]
                
                for msg_id in expired_pending:
                    self.pending_messages.pop(msg_id, None)
                
                # Clean up delivery confirmations
                expired_confirmations = [
                    msg_id for msg_id, timestamp in self.delivery_confirmations.items()
                    if (current_time - timestamp).total_seconds() > 3600  # 1 hour
                ]
                
                for msg_id in expired_confirmations:
                    self.delivery_confirmations.pop(msg_id, None)
                
                logger.debug(f"Cleaned up {len(expired_pending)} pending messages and {len(expired_confirmations)} confirmations")
                
            except Exception as e:
                logger.error(f"Message cleanup error: {e}")
            
            await asyncio.sleep(300)  # Clean up every 5 minutes
    
    async def _metrics_collector(self):
        """Collect and update communication metrics"""
        
        while not self._shutdown_event.is_set():
            try:
                # Update active connections
                self.metrics["active_connections"] = len(self.subscriptions)
                
                # Calculate average delivery time (simplified)
                # In production, would track actual delivery times
                if self.metrics["messages_sent"] > 0:
                    self.metrics["avg_delivery_time"] = 0.5  # Placeholder
                
                # Send metrics to performance channel
                await self.send_performance_metric(
                    "communication_metrics",
                    0.0,  # Overall health score
                    self.metrics
                )
                
            except Exception as e:
                logger.error(f"Metrics collector error: {e}")
            
            await asyncio.sleep(60)  # Collect metrics every minute
    
    async def _compress_message(self, message_data: str) -> str:
        """Compress message data (placeholder implementation)"""
        # In production, would use actual compression like gzip
        return message_data
    
    async def _decompress_message(self, message_data: str) -> str:
        """Decompress message data (placeholder implementation)"""
        # In production, would use actual decompression
        return message_data
    
    async def _encrypt_message(self, message_data: str) -> str:
        """Encrypt message data (placeholder implementation)"""
        # In production, would use actual encryption
        return message_data
    
    async def _decrypt_message(self, message_data: str) -> str:
        """Decrypt message data (placeholder implementation)"""
        # In production, would use actual decryption
        return message_data
    
    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        timestamp = str(int(time.time() * 1000))
        random_part = uuid.uuid4().hex[:8]
        return f"{self.agent_id}_{timestamp}_{random_part}"
    
    async def get_communication_status(self) -> Dict[str, Any]:
        """Get current communication status"""
        
        return {
            "agent_id": self.agent_id,
            "timestamp": datetime.utcnow().isoformat(),
            "channels": {
                channel_id: {
                    "name": channel.name,
                    "participants": len(channel.participants),
                    "message_types": [mt.value for mt in channel.message_types]
                }
                for channel_id, channel in self.channels.items()
            },
            "subscriptions": list(self.subscriptions.keys()),
            "pending_messages": len(self.pending_messages),
            "message_history_size": len(self.message_history),
            "metrics": self.metrics.copy(),
            "background_tasks": len(self._background_tasks)
        }
    
    async def shutdown(self):
        """Gracefully shutdown the communication protocol"""
        
        logger.info("Shutting down communication protocol")
        
        # Send offline status
        offline_message = Message(
            message_id=self._generate_message_id(),
            message_type=MessageType.AGENT_STATUS,
            sender_id=self.agent_id,
            recipient_id=None,
            priority=MessagePriority.MEDIUM,
            delivery_mode=DeliveryMode.FIRE_AND_FORGET,
            payload={
                "status": "offline",
                "timestamp": datetime.utcnow().isoformat(),
                "reason": "graceful_shutdown"
            }
        )
        
        await self.send_message(offline_message, "agi:agents")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Communication protocol shutdown complete")


# Utility functions for common communication patterns

async def create_task_coordination_session(agents: List[str], 
                                         session_id: str,
                                         communication: CommunicationProtocol) -> str:
    """Create a task coordination session between agents"""
    
    session_message = Message(
        message_id=communication._generate_message_id(),
        message_type=MessageType.COORDINATION_EVENT,
        sender_id=communication.agent_id,
        recipient_id=None,
        priority=MessagePriority.HIGH,
        delivery_mode=DeliveryMode.RELIABLE,
        payload={
            "event_type": "coordination_session_start",
            "session_id": session_id,
            "participants": agents,
            "coordinator": communication.agent_id,
            "start_time": datetime.utcnow().isoformat()
        }
    )
    
    await communication.send_message(session_message, "agi:coordination")
    return session_message.message_id


async def request_consensus(proposal_id: str, 
                          proposal_data: Dict[str, Any],
                          communication: CommunicationProtocol) -> str:
    """Request consensus on a proposal"""
    
    consensus_request = Message(
        message_id=communication._generate_message_id(),
        message_type=MessageType.CONSENSUS_REQUEST,
        sender_id=communication.agent_id,
        recipient_id=None,
        priority=MessagePriority.HIGH,
        delivery_mode=DeliveryMode.RELIABLE,
        payload={
            "proposal_id": proposal_id,
            "proposal_data": proposal_data,
            "requester": communication.agent_id,
            "requested_at": datetime.utcnow().isoformat(),
            "voting_deadline": (datetime.utcnow() + timedelta(minutes=5)).isoformat(),
            "minimum_votes": 3
        }
    )
    
    await communication.send_message(consensus_request, "agi:consensus")
    return consensus_request.message_id


if __name__ == "__main__":
    # Example usage
    async def main():
        protocol = CommunicationProtocol(agent_id="test_agent")
        await protocol.initialize()
        
        # Subscribe to task channel
        async def task_handler(message: Message):
            print(f"Received task message: {message.payload}")
        
        await protocol.subscribe_to_channel("agi:tasks", task_handler)
        
        # Send a test task request
        await protocol.send_task_request(
            "test_task_001",
            "Test task for communication protocol",
            ["agent1", "agent2"]
        )
        
        # Wait a bit
        await asyncio.sleep(10)
        
        # Get status
        status = await protocol.get_communication_status()
        print("Communication Status:")
        print(json.dumps(status, indent=2))
        
        await protocol.shutdown()
    
    asyncio.run(main())