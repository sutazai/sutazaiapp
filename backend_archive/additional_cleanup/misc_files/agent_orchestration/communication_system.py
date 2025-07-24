#!/usr/bin/env python3
"""
Communication System - Inter-agent communication infrastructure
"""

import asyncio
import logging
import threading
import uuid
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable, Set
from enum import Enum
from collections import defaultdict, deque
import weakref

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """Message types"""
    DIRECT = "direct"
    BROADCAST = "broadcast"
    GROUP = "group"
    SYSTEM = "system"
    COORDINATION = "coordination"
    NOTIFICATION = "notification"
    REQUEST = "request"
    RESPONSE = "response"
    HEARTBEAT = "heartbeat"

class MessagePriority(Enum):
    """Message priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4

class DeliveryMode(Enum):
    """Message delivery modes"""
    IMMEDIATE = "immediate"
    RELIABLE = "reliable"
    ORDERED = "ordered"
    BEST_EFFORT = "best_effort"

@dataclass
class Message:
    """Inter-agent message"""
    message_id: str
    sender_id: str
    recipient_id: Optional[str] = None
    group_id: Optional[str] = None
    message_type: MessageType = MessageType.DIRECT
    priority: MessagePriority = MessagePriority.NORMAL
    delivery_mode: DeliveryMode = DeliveryMode.IMMEDIATE
    content: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class MessageChannel:
    """Communication channel"""
    channel_id: str
    channel_type: str
    participants: Set[str] = field(default_factory=set)
    message_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    channel_config: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True

@dataclass
class CommunicationConfig:
    """Configuration for communication system"""
    max_message_size: int = 1024 * 1024  # 1MB
    max_queue_size: int = 10000
    message_timeout: float = 30.0
    heartbeat_interval: float = 10.0
    enable_message_persistence: bool = True
    enable_message_encryption: bool = True
    enable_message_compression: bool = True
    enable_delivery_confirmation: bool = True
    enable_message_ordering: bool = True
    max_channels: int = 1000
    max_participants_per_channel: int = 100
    message_history_size: int = 1000
    enable_message_routing: bool = True
    enable_load_balancing: bool = True

class CommunicationSystem:
    """Inter-agent communication system"""
    
    def __init__(self, config: CommunicationConfig = None):
        self.config = config or CommunicationConfig()
        
        # Message infrastructure
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.message_handlers: Dict[str, Dict[MessageType, Callable]] = defaultdict(dict)
        self.message_channels: Dict[str, MessageChannel] = {}
        
        # Routing and delivery
        self.message_routing: Dict[str, str] = {}  # message_id -> route
        self.delivery_confirmations: Dict[str, Dict[str, Any]] = {}
        self.pending_messages: Dict[str, Message] = {}
        
        # Agent registry
        self.active_agents: Set[str] = set()
        self.agent_capabilities: Dict[str, Dict[str, Any]] = {}
        self.agent_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        
        # Performance metrics
        self.communication_metrics = {
            "total_messages": 0,
            "delivered_messages": 0,
            "failed_messages": 0,
            "average_delivery_time": 0.0,
            "active_channels": 0,
            "message_throughput": 0.0
        }
        
        # Threading
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # Background tasks
        self._delivery_task = None
        self._heartbeat_task = None
        self._cleanup_task = None
        
        logger.info("Communication system initialized")
    
    async def initialize(self) -> bool:
        """Initialize communication system"""
        try:
            # Start background tasks
            self._start_message_delivery()
            self._start_heartbeat()
            self._start_cleanup()
            
            logger.info("Communication system initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Communication system initialization failed: {e}")
            return False
    
    def _start_message_delivery(self):
        """Start message delivery background task"""
        def delivery_loop():
            while not self._shutdown_event.is_set():
                try:
                    asyncio.run(self._process_message_delivery())
                    self._shutdown_event.wait(0.1)  # High frequency processing
                except Exception as e:
                    logger.error(f"Message delivery error: {e}")
                    self._shutdown_event.wait(1)
        
        self._delivery_task = threading.Thread(target=delivery_loop, daemon=True)
        self._delivery_task.start()
    
    def _start_heartbeat(self):
        """Start heartbeat background task"""
        def heartbeat_loop():
            while not self._shutdown_event.is_set():
                try:
                    asyncio.run(self._send_heartbeats())
                    self._shutdown_event.wait(self.config.heartbeat_interval)
                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")
                    self._shutdown_event.wait(30)
        
        self._heartbeat_task = threading.Thread(target=heartbeat_loop, daemon=True)
        self._heartbeat_task.start()
    
    def _start_cleanup(self):
        """Start cleanup background task"""
        def cleanup_loop():
            while not self._shutdown_event.is_set():
                try:
                    asyncio.run(self._cleanup_expired_messages())
                    self._shutdown_event.wait(60)  # Run every minute
                except Exception as e:
                    logger.error(f"Cleanup error: {e}")
                    self._shutdown_event.wait(60)
        
        self._cleanup_task = threading.Thread(target=cleanup_loop, daemon=True)
        self._cleanup_task.start()
    
    async def register_agent(self, agent_id: str, capabilities: Dict[str, Any] = None) -> bool:
        """Register agent with communication system"""
        try:
            with self._lock:
                # Add agent to registry
                self.active_agents.add(agent_id)
                self.agent_capabilities[agent_id] = capabilities or {}
                
                # Create message queue
                if agent_id not in self.message_queues:
                    self.message_queues[agent_id] = asyncio.Queue(maxsize=self.config.max_queue_size)
                
                logger.info(f"Agent registered: {agent_id}")
                return True
                
        except Exception as e:
            logger.error(f"Agent registration failed: {e}")
            return False
    
    async def unregister_agent(self, agent_id: str) -> bool:
        """Unregister agent from communication system"""
        try:
            with self._lock:
                # Remove from registry
                self.active_agents.discard(agent_id)
                if agent_id in self.agent_capabilities:
                    del self.agent_capabilities[agent_id]
                
                # Clear message queue
                if agent_id in self.message_queues:
                    # Process remaining messages
                    while not self.message_queues[agent_id].empty():
                        try:
                            self.message_queues[agent_id].get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    del self.message_queues[agent_id]
                
                # Remove from channels
                for channel in self.message_channels.values():
                    channel.participants.discard(agent_id)
                
                # Clear subscriptions
                if agent_id in self.agent_subscriptions:
                    del self.agent_subscriptions[agent_id]
                
                logger.info(f"Agent unregistered: {agent_id}")
                return True
                
        except Exception as e:
            logger.error(f"Agent unregistration failed: {e}")
            return False
    
    async def send_message(self, message: Message) -> bool:
        """Send message to recipient(s)"""
        try:
            # Validate message
            if not self._validate_message(message):
                logger.warning(f"Message validation failed: {message.message_id}")
                return False
            
            # Apply message processing
            processed_message = await self._process_outgoing_message(message)
            
            # Route message
            if message.message_type == MessageType.DIRECT:
                return await self._send_direct_message(processed_message)
            elif message.message_type == MessageType.BROADCAST:
                return await self._send_broadcast_message(processed_message)
            elif message.message_type == MessageType.GROUP:
                return await self._send_group_message(processed_message)
            else:
                # Default to direct message
                return await self._send_direct_message(processed_message)
                
        except Exception as e:
            logger.error(f"Message sending failed: {e}")
            return False
    
    async def _send_direct_message(self, message: Message) -> bool:
        """Send direct message to recipient"""
        try:
            if not message.recipient_id:
                logger.warning("Direct message missing recipient")
                return False
            
            if message.recipient_id not in self.active_agents:
                logger.warning(f"Recipient not active: {message.recipient_id}")
                return False
            
            # Add to recipient's queue
            recipient_queue = self.message_queues[message.recipient_id]
            
            if message.delivery_mode == DeliveryMode.IMMEDIATE:
                # Try immediate delivery
                try:
                    recipient_queue.put_nowait(message)
                    await self._confirm_delivery(message)
                    return True
                except asyncio.QueueFull:
                    logger.warning(f"Recipient queue full: {message.recipient_id}")
                    return False
            else:
                # Reliable delivery
                await recipient_queue.put(message)
                await self._confirm_delivery(message)
                return True
                
        except Exception as e:
            logger.error(f"Direct message delivery failed: {e}")
            return False
    
    async def _send_broadcast_message(self, message: Message) -> bool:
        """Send broadcast message to all agents"""
        try:
            success_count = 0
            total_agents = len(self.active_agents)
            
            for agent_id in self.active_agents.copy():
                if agent_id != message.sender_id:  # Don't send to sender
                    message_copy = self._copy_message(message)
                    message_copy.recipient_id = agent_id
                    
                    if await self._send_direct_message(message_copy):
                        success_count += 1
            
            # Consider broadcast successful if delivered to majority
            return success_count >= (total_agents * 0.5)
            
        except Exception as e:
            logger.error(f"Broadcast message delivery failed: {e}")
            return False
    
    async def _send_group_message(self, message: Message) -> bool:
        """Send group message to channel participants"""
        try:
            if not message.group_id:
                logger.warning("Group message missing group ID")
                return False
            
            if message.group_id not in self.message_channels:
                logger.warning(f"Group channel not found: {message.group_id}")
                return False
            
            channel = self.message_channels[message.group_id]
            success_count = 0
            
            for participant_id in channel.participants:
                if participant_id != message.sender_id:  # Don't send to sender
                    message_copy = self._copy_message(message)
                    message_copy.recipient_id = participant_id
                    
                    if await self._send_direct_message(message_copy):
                        success_count += 1
            
            # Add to channel history
            channel.message_history.append(message)
            channel.last_activity = datetime.now(timezone.utc)
            
            # Consider group message successful if delivered to majority
            return success_count >= (len(channel.participants) * 0.5)
            
        except Exception as e:
            logger.error(f"Group message delivery failed: {e}")
            return False
    
    async def receive_message(self, agent_id: str, timeout: float = None) -> Optional[Message]:
        """Receive message for agent"""
        try:
            if agent_id not in self.message_queues:
                return None
            
            queue = self.message_queues[agent_id]
            
            if timeout:
                message = await asyncio.wait_for(queue.get(), timeout=timeout)
            else:
                message = await queue.get()
            
            # Process incoming message
            processed_message = await self._process_incoming_message(message)
            
            # Update metrics
            self.communication_metrics["delivered_messages"] += 1
            
            return processed_message
            
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Message receiving failed: {e}")
            return None
    
    async def create_channel(self, channel_config: Dict[str, Any]) -> str:
        """Create communication channel"""
        try:
            with self._lock:
                if len(self.message_channels) >= self.config.max_channels:
                    raise RuntimeError("Maximum channels reached")
                
                channel_id = f"channel_{uuid.uuid4().hex[:8]}"
                
                # Create channel
                channel = MessageChannel(
                    channel_id=channel_id,
                    channel_type=channel_config.get("type", "general"),
                    channel_config=channel_config
                )
                
                # Add initial participants
                initial_participants = channel_config.get("participants", [])
                for participant in initial_participants:
                    if participant in self.active_agents:
                        channel.participants.add(participant)
                
                self.message_channels[channel_id] = channel
                self.communication_metrics["active_channels"] += 1
                
                logger.info(f"Channel created: {channel_id}")
                return channel_id
                
        except Exception as e:
            logger.error(f"Channel creation failed: {e}")
            raise
    
    async def join_channel(self, channel_id: str, agent_id: str) -> bool:
        """Join agent to channel"""
        try:
            with self._lock:
                if channel_id not in self.message_channels:
                    return False
                
                channel = self.message_channels[channel_id]
                
                if len(channel.participants) >= self.config.max_participants_per_channel:
                    logger.warning(f"Channel participant limit reached: {channel_id}")
                    return False
                
                if agent_id not in self.active_agents:
                    logger.warning(f"Agent not active: {agent_id}")
                    return False
                
                channel.participants.add(agent_id)
                channel.last_activity = datetime.now(timezone.utc)
                
                logger.info(f"Agent {agent_id} joined channel {channel_id}")
                return True
                
        except Exception as e:
            logger.error(f"Channel join failed: {e}")
            return False
    
    async def leave_channel(self, channel_id: str, agent_id: str) -> bool:
        """Remove agent from channel"""
        try:
            with self._lock:
                if channel_id not in self.message_channels:
                    return False
                
                channel = self.message_channels[channel_id]
                channel.participants.discard(agent_id)
                channel.last_activity = datetime.now(timezone.utc)
                
                # Remove empty channels
                if not channel.participants:
                    del self.message_channels[channel_id]
                    self.communication_metrics["active_channels"] -= 1
                
                logger.info(f"Agent {agent_id} left channel {channel_id}")
                return True
                
        except Exception as e:
            logger.error(f"Channel leave failed: {e}")
            return False
    
    def register_message_handler(self, agent_id: str, message_type: MessageType, handler: Callable):
        """Register message handler for agent"""
        try:
            with self._lock:
                self.message_handlers[agent_id][message_type] = handler
                logger.info(f"Message handler registered for {agent_id}: {message_type}")
                
        except Exception as e:
            logger.error(f"Message handler registration failed: {e}")
    
    def unregister_message_handler(self, agent_id: str, message_type: MessageType):
        """Unregister message handler for agent"""
        try:
            with self._lock:
                if agent_id in self.message_handlers:
                    self.message_handlers[agent_id].pop(message_type, None)
                    
        except Exception as e:
            logger.error(f"Message handler unregistration failed: {e}")
    
    def _validate_message(self, message: Message) -> bool:
        """Validate message"""
        try:
            # Check message size
            message_size = len(json.dumps(message.content, default=str))
            if message_size > self.config.max_message_size:
                logger.warning(f"Message too large: {message_size} bytes")
                return False
            
            # Check sender is active
            if message.sender_id not in self.active_agents:
                logger.warning(f"Sender not active: {message.sender_id}")
                return False
            
            # Check expiration
            if message.expires_at and message.expires_at < datetime.now(timezone.utc):
                logger.warning(f"Message expired: {message.message_id}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Message validation failed: {e}")
            return False
    
    async def _process_outgoing_message(self, message: Message) -> Message:
        """Process outgoing message"""
        try:
            # Apply encryption if enabled
            if self.config.enable_message_encryption:
                message = await self._encrypt_message(message)
            
            # Apply compression if enabled
            if self.config.enable_message_compression:
                message = await self._compress_message(message)
            
            # Add routing information
            if self.config.enable_message_routing:
                message = await self._add_routing_info(message)
            
            # Update metrics
            self.communication_metrics["total_messages"] += 1
            
            return message
            
        except Exception as e:
            logger.error(f"Outgoing message processing failed: {e}")
            raise
    
    async def _process_incoming_message(self, message: Message) -> Message:
        """Process incoming message"""
        try:
            # Remove routing information
            message = await self._remove_routing_info(message)
            
            # Apply decompression if enabled
            if self.config.enable_message_compression:
                message = await self._decompress_message(message)
            
            # Apply decryption if enabled
            if self.config.enable_message_encryption:
                message = await self._decrypt_message(message)
            
            return message
            
        except Exception as e:
            logger.error(f"Incoming message processing failed: {e}")
            raise
    
    async def _encrypt_message(self, message: Message) -> Message:
        """Encrypt message content"""
        # Simplified encryption - in practice, use proper encryption
        return message
    
    async def _decrypt_message(self, message: Message) -> Message:
        """Decrypt message content"""
        # Simplified decryption - in practice, use proper decryption
        return message
    
    async def _compress_message(self, message: Message) -> Message:
        """Compress message content"""
        # Simplified compression - in practice, use proper compression
        return message
    
    async def _decompress_message(self, message: Message) -> Message:
        """Decompress message content"""
        # Simplified decompression - in practice, use proper decompression
        return message
    
    async def _add_routing_info(self, message: Message) -> Message:
        """Add routing information to message"""
        message.metadata["routing"] = {
            "route_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        return message
    
    async def _remove_routing_info(self, message: Message) -> Message:
        """Remove routing information from message"""
        message.metadata.pop("routing", None)
        return message
    
    async def _confirm_delivery(self, message: Message):
        """Confirm message delivery"""
        if self.config.enable_delivery_confirmation:
            confirmation = {
                "message_id": message.message_id,
                "recipient": message.recipient_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "delivered"
            }
            self.delivery_confirmations[message.message_id] = confirmation
    
    def _copy_message(self, message: Message) -> Message:
        """Create a copy of message"""
        return Message(
            message_id=str(uuid.uuid4()),
            sender_id=message.sender_id,
            recipient_id=message.recipient_id,
            group_id=message.group_id,
            message_type=message.message_type,
            priority=message.priority,
            delivery_mode=message.delivery_mode,
            content=message.content.copy(),
            metadata=message.metadata.copy(),
            timestamp=message.timestamp,
            expires_at=message.expires_at,
            correlation_id=message.correlation_id,
            reply_to=message.reply_to
        )
    
    async def _process_message_delivery(self):
        """Process message delivery"""
        try:
            # Process pending messages
            for message_id, message in list(self.pending_messages.items()):
                if message.expires_at and message.expires_at < datetime.now(timezone.utc):
                    # Message expired
                    del self.pending_messages[message_id]
                    self.communication_metrics["failed_messages"] += 1
                    continue
                
                # Retry delivery
                if message.retry_count < message.max_retries:
                    if await self.send_message(message):
                        del self.pending_messages[message_id]
                    else:
                        message.retry_count += 1
                else:
                    # Max retries reached
                    del self.pending_messages[message_id]
                    self.communication_metrics["failed_messages"] += 1
                    
        except Exception as e:
            logger.error(f"Message delivery processing failed: {e}")
    
    async def _send_heartbeats(self):
        """Send heartbeat messages"""
        try:
            for agent_id in self.active_agents.copy():
                heartbeat = Message(
                    message_id=f"heartbeat_{uuid.uuid4().hex[:8]}",
                    sender_id="system",
                    recipient_id=agent_id,
                    message_type=MessageType.HEARTBEAT,
                    priority=MessagePriority.LOW,
                    content={"timestamp": datetime.now(timezone.utc).isoformat()}
                )
                
                # Send heartbeat (don't wait for delivery)
                asyncio.create_task(self._send_direct_message(heartbeat))
                
        except Exception as e:
            logger.error(f"Heartbeat sending failed: {e}")
    
    async def _cleanup_expired_messages(self):
        """Clean up expired messages and confirmations"""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Clean up delivery confirmations
            expired_confirmations = [
                msg_id for msg_id, confirmation in self.delivery_confirmations.items()
                if (current_time - datetime.fromisoformat(confirmation["timestamp"])).total_seconds() > 3600
            ]
            
            for msg_id in expired_confirmations:
                del self.delivery_confirmations[msg_id]
            
            # Clean up inactive channels
            inactive_channels = [
                channel_id for channel_id, channel in self.message_channels.items()
                if (current_time - channel.last_activity).total_seconds() > 3600 and not channel.participants
            ]
            
            for channel_id in inactive_channels:
                del self.message_channels[channel_id]
                self.communication_metrics["active_channels"] -= 1
                
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")
    
    async def start(self) -> bool:
        """Start communication system"""
        try:
            logger.info("Starting communication system...")
            return True
        except Exception as e:
            logger.error(f"Failed to start communication system: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop communication system"""
        try:
            logger.info("Stopping communication system...")
            
            # Signal shutdown
            self._shutdown_event.set()
            
            # Clear all queues
            for queue in self.message_queues.values():
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop communication system: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get communication system status"""
        with self._lock:
            return {
                "active_agents": len(self.active_agents),
                "active_channels": len(self.message_channels),
                "pending_messages": len(self.pending_messages),
                "delivery_confirmations": len(self.delivery_confirmations),
                "metrics": self.communication_metrics,
                "config": {
                    "max_message_size": self.config.max_message_size,
                    "max_queue_size": self.config.max_queue_size,
                    "max_channels": self.config.max_channels
                }
            }
    
    def get_channel_info(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """Get channel information"""
        with self._lock:
            channel = self.message_channels.get(channel_id)
            if channel:
                return {
                    "channel_id": channel.channel_id,
                    "channel_type": channel.channel_type,
                    "participants": list(channel.participants),
                    "message_count": len(channel.message_history),
                    "created_at": channel.created_at.isoformat(),
                    "last_activity": channel.last_activity.isoformat(),
                    "is_active": channel.is_active
                }
            return None
    
    def health_check(self) -> bool:
        """Check communication system health"""
        try:
            return (
                len(self.active_agents) <= 10000 and  # Reasonable limit
                len(self.message_channels) <= self.config.max_channels and
                len(self.pending_messages) <= 1000  # Reasonable limit
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

# Factory function
def create_communication_system(config: Optional[Dict[str, Any]] = None) -> CommunicationSystem:
    """Create communication system instance"""
    if config:
        comm_config = CommunicationConfig(**config)
    else:
        comm_config = CommunicationConfig()
    
    return CommunicationSystem(config=comm_config)