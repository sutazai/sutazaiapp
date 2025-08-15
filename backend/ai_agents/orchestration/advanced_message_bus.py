"""
Advanced Multi-Agent Communication and Message Bus System
=========================================================

A sophisticated message bus system enabling real-time communication, coordination,
and knowledge sharing between all 38 AI agents in the SutazAI ecosystem.

Key Features:
- High-performance Redis-based message bus
- Multi-pattern communication (point-to-point, broadcast, pub-sub, request-response)
- Advanced message routing with priority queues
- Real-time event streaming and notifications
- Message persistence and replay capabilities
- Agent discovery and presence management
- Load balancing and failover support
- Message encryption and security
- Performance monitoring and analytics
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Any, Optional, Set, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from backend.app.schemas.message_types import MessageType, MessagePriority
import redis.asyncio as redis
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import pickle
import zlib
import hashlib
import hmac
from cryptography.fernet import Fernet

logger = logging.getLogger(__name__)


# Using canonical MessageType and MessagePriority


class CommunicationPattern(Enum):
    """Communication patterns"""
    POINT_TO_POINT = "point_to_point"      # Direct agent-to-agent
    BROADCAST = "broadcast"                # One-to-many
    MULTICAST = "multicast"               # One-to-group
    PUBLISH_SUBSCRIBE = "pub_sub"         # Topic-based
    REQUEST_RESPONSE = "request_response"  # Synchronous RPC
    PIPELINE = "pipeline"                 # Sequential processing
    SCATTER_GATHER = "scatter_gather"     # Parallel processing + aggregation


@dataclass
class Message:
    """Enhanced message structure"""
    id: str
    sender_id: str
    recipient_id: Optional[str] = None    # None for broadcast
    message_type: MessageType = MessageType.CHAT_MESSAGE
    pattern: CommunicationPattern = CommunicationPattern.POINT_TO_POINT
    priority: MessagePriority = MessagePriority.NORMAL
    topic: Optional[str] = None           # For pub-sub pattern
    correlation_id: Optional[str] = None  # For request-response
    payload: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    encrypted: bool = False
    compressed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization"""
        data = asdict(self)
        data['message_type'] = self.message_type.value
        data['message_type_name'] = self.message_type.name
        data['pattern'] = self.pattern.value
        data['priority'] = self.priority.value
        data['priority_name'] = self.priority.name
        data['timestamp'] = self.timestamp.isoformat()
        if self.expires_at:
            data['expires_at'] = self.expires_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary"""
        # Prefer by name when provided to avoid enum value drift
        mt = data.get('message_type_name') or data.get('message_type', 'chat_message')
        try:
            data['message_type'] = MessageType(mt) if isinstance(mt, str) and mt.lower() == mt else MessageType[mt]
        except Exception:
            data['message_type'] = MessageType.CHAT_MESSAGE
        data['pattern'] = CommunicationPattern(data.get('pattern', 'point_to_point'))
        pr = data.get('priority_name') if 'priority_name' in data else data.get('priority', 3)
        try:
            data['priority'] = MessagePriority(pr) if isinstance(pr, int) else MessagePriority[pr]
        except Exception:
            data['priority'] = MessagePriority.NORMAL
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if data.get('expires_at'):
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
        return cls(**data)


@dataclass
class AgentPresence:
    """Agent presence information"""
    agent_id: str
    status: str = "online"
    capabilities: List[str] = field(default_factory=list)
    current_load: int = 0
    max_load: int = 10
    last_seen: datetime = field(default_factory=datetime.now)
    endpoint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MessageStats:
    """Message statistics and metrics"""
    total_sent: int = 0
    total_received: int = 0
    total_failed: int = 0
    average_latency: float = 0.0
    peak_throughput: int = 0
    error_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)


class AdvancedMessageBus:
    """
    Advanced message bus for multi-agent communication
    """
    
    def __init__(self, 
                 redis_url: str = "redis://redis:6379",
                 encryption_key: Optional[bytes] = None):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub_client: Optional[redis.Redis] = None
        
        # Encryption setup
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher = Fernet(self.encryption_key)
        
        # Agent Management
        self.agent_presence: Dict[str, AgentPresence] = {}
        self.agent_subscriptions: Dict[str, Set[str]] = defaultdict(set)
        self.message_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Message Management
        self.pending_messages: Dict[str, Message] = {}
        self.message_history: deque = deque(maxlen=10000)
        self.response_futures: Dict[str, asyncio.Future] = {}
        
        # Performance Tracking
        self.stats: Dict[str, MessageStats] = defaultdict(MessageStats)
        self.performance_metrics: Dict[str, Any] = {}
        
        # Configuration
        self.config = {
            "max_message_size": 10 * 1024 * 1024,  # 10MB
            "message_ttl": 3600,  # 1 hour
            "retry_delay": 5,     # 5 seconds
            "compression_threshold": 1024,  # 1KB
            "encryption_enabled": True,
            "persistence_enabled": True,
            "metrics_enabled": True
        }
        
        # Background Tasks
        self.background_tasks: List[asyncio.Task] = []
        self.running = False
        
        logger.info("Advanced Message Bus initialized")
    
    async def initialize(self):
        """Initialize the message bus system"""
        logger.info("🚀 Initializing Advanced Message Bus...")
        
        # Connect to Redis
        self.redis_client = await redis.from_url(self.redis_url, decode_responses=False)
        self.pubsub_client = await redis.from_url(self.redis_url, decode_responses=False)
        
        # Initialize pub-sub
        self.pubsub = self.pubsub_client.pubsub()
        
        # Start background services
        self.background_tasks = [
            asyncio.create_task(self._message_processor()),
            asyncio.create_task(self._presence_monitor()),
            asyncio.create_task(self._cleanup_expired_messages()),
            asyncio.create_task(self._metrics_collector()),
            asyncio.create_task(self._pubsub_listener())
        ]
        
        self.running = True
        logger.info("✅ Advanced Message Bus ready")
    
    async def shutdown(self):
        """Shutdown the message bus"""
        logger.info("🛑 Shutting down Advanced Message Bus...")
        
        self.running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Close Redis connections
        if self.redis_client:
            await self.redis_client.close()
        if self.pubsub_client:
            await self.pubsub_client.close()
        
        logger.info("✅ Advanced Message Bus shutdown complete")
    
    # ==================== Agent Management ====================
    
    async def register_agent(self, agent_id: str, capabilities: List[str], 
                           endpoint: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """Register an agent with the message bus"""
        
        presence = AgentPresence(
            agent_id=agent_id,
            capabilities=capabilities,
            endpoint=endpoint,
            metadata=metadata or {}
        )
        
        self.agent_presence[agent_id] = presence
        
        # Store in Redis for persistence
        await self.redis_client.hset(
            "agent_presence",
            agent_id,
            json.dumps(asdict(presence), default=str)
        )
        
        # Announce agent presence
        await self.broadcast_message(Message(
            id=str(uuid.uuid4()),
            sender_id="message_bus",
            message_type=MessageType.AGENT_ANNOUNCEMENT,
            pattern=CommunicationPattern.BROADCAST,
            payload={
                "action": "agent_joined",
                "agent_id": agent_id,
                "capabilities": capabilities,
                "endpoint": endpoint
            }
        ))
        
        logger.info(f"Agent registered: {agent_id}")
    
    async def unregister_agent(self, agent_id: str):
        """Unregister an agent from the message bus"""
        
        if agent_id in self.agent_presence:
            del self.agent_presence[agent_id]
        
        # Remove from Redis
        await self.redis_client.hdel("agent_presence", agent_id)
        
        # Clean up subscriptions
        if agent_id in self.agent_subscriptions:
            del self.agent_subscriptions[agent_id]
        
        # Announce agent departure
        await self.broadcast_message(Message(
            id=str(uuid.uuid4()),
            sender_id="message_bus",
            message_type=MessageType.AGENT_ANNOUNCEMENT,
            pattern=CommunicationPattern.BROADCAST,
            payload={
                "action": "agent_left",
                "agent_id": agent_id
            }
        ))
        
        logger.info(f"Agent unregistered: {agent_id}")
    
    async def update_agent_presence(self, agent_id: str, status: str = "online", 
                                  current_load: Optional[int] = None):
        """Update agent presence information"""
        
        if agent_id not in self.agent_presence:
            logger.warning(f"Unknown agent: {agent_id}")
            return
        
        presence = self.agent_presence[agent_id]
        presence.status = status
        presence.last_seen = datetime.now()
        
        if current_load is not None:
            presence.current_load = current_load
        
        # Update in Redis
        await self.redis_client.hset(
            "agent_presence",
            agent_id,
            json.dumps(asdict(presence), default=str)
        )
    
    # ==================== Message Sending ====================
    
    async def send_message(self, message: Message) -> bool:
        """Send a message through the bus"""
        
        try:
            # Validate message
            if not await self._validate_message(message):
                return False
            
            # Set message ID if not provided
            if not message.id:
                message.id = str(uuid.uuid4())
            
            # Apply compression if needed
            if await self._should_compress(message):
                message = await self._compress_message(message)
            
            # Apply encryption if enabled
            if self.config["encryption_enabled"] and not message.encrypted:
                message = await self._encrypt_message(message)
            
            # Route message based on pattern
            success = await self._route_message(message)
            
            # Update statistics
            if success:
                self.stats[message.sender_id].total_sent += 1
                self.message_history.append(message)
            else:
                self.stats[message.sender_id].total_failed += 1
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            self.stats[message.sender_id].total_failed += 1
            return False
    
    async def send_request(self, message: Message, timeout: float = 30.0) -> Optional[Message]:
        """Send a request message and wait for response"""
        
        # Set up correlation ID for response tracking
        correlation_id = str(uuid.uuid4())
        message.correlation_id = correlation_id
        message.pattern = CommunicationPattern.REQUEST_RESPONSE
        
        # Create future for response
        response_future = asyncio.Future()
        self.response_futures[correlation_id] = response_future
        
        try:
            # Send the request
            success = await self.send_message(message)
            if not success:
                return None
            
            # Wait for response
            response = await asyncio.wait_for(response_future, timeout=timeout)
            return response
            
        except asyncio.TimeoutError:
            logger.warning(f"Request timeout for correlation_id: {correlation_id}")
            return None
        
        finally:
            # Clean up future
            if correlation_id in self.response_futures:
                del self.response_futures[correlation_id]
    
    async def broadcast_message(self, message: Message) -> bool:
        """Broadcast a message to all agents"""
        
        message.pattern = CommunicationPattern.BROADCAST
        message.recipient_id = None
        
        return await self.send_message(message)
    
    async def multicast_message(self, message: Message, recipients: List[str]) -> bool:
        """Send message to multiple specific recipients"""
        
        message.pattern = CommunicationPattern.MULTICAST
        message.metadata["recipients"] = recipients
        
        return await self.send_message(message)
    
    async def publish_to_topic(self, topic: str, message: Message) -> bool:
        """Publish a message to a topic"""
        
        message.pattern = CommunicationPattern.PUBLISH_SUBSCRIBE
        message.topic = topic
        
        return await self.send_message(message)
    
    # ==================== Message Routing ====================
    
    async def _route_message(self, message: Message) -> bool:
        """Route message based on communication pattern"""
        
        try:
            if message.pattern == CommunicationPattern.POINT_TO_POINT:
                return await self._route_point_to_point(message)
            
            elif message.pattern == CommunicationPattern.BROADCAST:
                return await self._route_broadcast(message)
            
            elif message.pattern == CommunicationPattern.MULTICAST:
                return await self._route_multicast(message)
            
            elif message.pattern == CommunicationPattern.PUBLISH_SUBSCRIBE:
                return await self._route_pub_sub(message)
            
            elif message.pattern == CommunicationPattern.REQUEST_RESPONSE:
                return await self._route_request_response(message)
            
            elif message.pattern == CommunicationPattern.PIPELINE:
                return await self._route_pipeline(message)
            
            elif message.pattern == CommunicationPattern.SCATTER_GATHER:
                return await self._route_scatter_gather(message)
            
            else:
                logger.error(f"Unknown communication pattern: {message.pattern}")
                return False
                
        except Exception as e:
            logger.error(f"Message routing failed: {e}")
            return False
    
    async def _route_point_to_point(self, message: Message) -> bool:
        """Route point-to-point message"""
        
        if not message.recipient_id:
            logger.error("Point-to-point message requires recipient_id")
            return False
        
        # Check if recipient is online
        if message.recipient_id not in self.agent_presence:
            logger.warning(f"Recipient not found: {message.recipient_id}")
            return False
        
        # Store message in recipient's queue
        queue_key = f"agent_queue:{message.recipient_id}"
        message_data = await self._serialize_message(message)
        
        # Use priority queue based on message priority
        priority_score = message.priority.value
        await self.redis_client.zadd(
            queue_key, 
            {message_data: priority_score}
        )
        
        # Set TTL on the queue
        await self.redis_client.expire(queue_key, self.config["message_ttl"])
        
        # Notify recipient
        await self.redis_client.publish(
            f"agent_notify:{message.recipient_id}",
            json.dumps({"action": "new_message", "message_id": message.id})
        )
        
        return True
    
    async def _route_broadcast(self, message: Message) -> bool:
        """Route broadcast message to all agents"""
        
        success_count = 0
        
        for agent_id in self.agent_presence:
            if agent_id != message.sender_id:  # Don't send to sender
                # Create copy for each recipient
                agent_message = Message(**asdict(message))
                agent_message.recipient_id = agent_id
                agent_message.pattern = CommunicationPattern.POINT_TO_POINT
                
                if await self._route_point_to_point(agent_message):
                    success_count += 1
        
        return success_count > 0
    
    async def _route_multicast(self, message: Message) -> bool:
        """Route multicast message to specific recipients"""
        
        recipients = message.metadata.get("recipients", [])
        success_count = 0
        
        for recipient_id in recipients:
            if recipient_id in self.agent_presence:
                # Create copy for each recipient
                agent_message = Message(**asdict(message))
                agent_message.recipient_id = recipient_id
                agent_message.pattern = CommunicationPattern.POINT_TO_POINT
                
                if await self._route_point_to_point(agent_message):
                    success_count += 1
        
        return success_count > 0
    
    async def _route_pub_sub(self, message: Message) -> bool:
        """Route pub-sub message to topic subscribers"""
        
        if not message.topic:
            logger.error("Pub-sub message requires topic")
            return False
        
        # Get topic subscribers
        subscribers = await self.redis_client.smembers(f"topic_subscribers:{message.topic}")
        
        if not subscribers:
            logger.warning(f"No subscribers for topic: {message.topic}")
            return False
        
        # Send to all subscribers
        success_count = 0
        for subscriber in subscribers:
            subscriber_id = subscriber.decode() if isinstance(subscriber, bytes) else subscriber
            
            if subscriber_id in self.agent_presence and subscriber_id != message.sender_id:
                # Create copy for each subscriber
                agent_message = Message(**asdict(message))
                agent_message.recipient_id = subscriber_id
                agent_message.pattern = CommunicationPattern.POINT_TO_POINT
                
                if await self._route_point_to_point(agent_message):
                    success_count += 1
        
        return success_count > 0
    
    async def _route_request_response(self, message: Message) -> bool:
        """Route request-response message"""
        
        # If this is a response, handle it specially
        if message.metadata.get("is_response"):
            correlation_id = message.correlation_id
            if correlation_id in self.response_futures:
                future = self.response_futures[correlation_id]
                if not future.done():
                    future.set_result(message)
                return True
        
        # Otherwise, route as point-to-point
        return await self._route_point_to_point(message)
    
    # ==================== Message Processing ====================
    
    async def _message_processor(self):
        """Background message processor"""
        
        while self.running:
            try:
                processed_count = 0
                
                # Process messages for each agent
                for agent_id in list(self.agent_presence.keys()):
                    queue_key = f"agent_queue:{agent_id}"
                    
                    # Get highest priority message
                    messages = await self.redis_client.zrevrange(
                        queue_key, 0, 0, withscores=True
                    )
                    
                    if messages:
                        message_data, priority = messages[0]
                        
                        # Remove from queue
                        await self.redis_client.zrem(queue_key, message_data)
                        
                        # Deserialize and process
                        try:
                            message = await self._deserialize_message(message_data)
                            await self._deliver_message(agent_id, message)
                            processed_count += 1
                            
                        except Exception as e:
                            logger.error(f"Failed to process message for {agent_id}: {e}")
                
                if processed_count == 0:
                    await asyncio.sleep(0.1)  # Short sleep if no messages
                
            except Exception as e:
                logger.error(f"Message processor error: {e}")
                await asyncio.sleep(1)
    
    async def _deliver_message(self, agent_id: str, message: Message):
        """Deliver message to specific agent"""
        
        try:
            # Update statistics
            self.stats[agent_id].total_received += 1
            
            # Decrypt if needed
            if message.encrypted:
                message = await self._decrypt_message(message)
            
            # Decompress if needed
            if message.compressed:
                message = await self._decompress_message(message)
            
            # Call registered handlers
            handlers = self.message_handlers.get(agent_id, [])
            for handler in handlers:
                try:
                    await handler(message)
                except Exception as e:
                    logger.error(f"Message handler error for {agent_id}: {e}")
            
            # If no handlers, use HTTP delivery
            if not handlers:
                await self._deliver_via_http(agent_id, message)
                
        except Exception as e:
            logger.error(f"Message delivery failed for {agent_id}: {e}")
            
            # Retry logic
            if message.retry_count < message.max_retries:
                message.retry_count += 1
                await asyncio.sleep(self.config["retry_delay"])
                
                # Re-queue message
                queue_key = f"agent_queue:{agent_id}"
                message_data = await self._serialize_message(message)
                priority_score = message.priority.value
                await self.redis_client.zadd(
                    queue_key, 
                    {message_data: priority_score}
                )
    
    async def _deliver_via_http(self, agent_id: str, message: Message):
        """Deliver message via HTTP to agent endpoint"""
        
        presence = self.agent_presence.get(agent_id)
        if not presence or not presence.endpoint:
            logger.warning(f"No endpoint for agent {agent_id}")
            return
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{presence.endpoint}/message",
                    json=message.to_dict(),
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    
                    if response.status == 200:
                        logger.debug(f"Message delivered to {agent_id}")
                    else:
                        logger.error(f"HTTP delivery failed for {agent_id}: {response.status}")
                        
        except Exception as e:
            logger.error(f"HTTP delivery error for {agent_id}: {e}")
    
    # ==================== Subscription Management ====================
    
    async def subscribe_to_topic(self, agent_id: str, topic: str):
        """Subscribe agent to a topic"""
        
        await self.redis_client.sadd(f"topic_subscribers:{topic}", agent_id)
        self.agent_subscriptions[agent_id].add(topic)
        
        logger.info(f"Agent {agent_id} subscribed to topic: {topic}")
    
    async def unsubscribe_from_topic(self, agent_id: str, topic: str):
        """Unsubscribe agent from a topic"""
        
        await self.redis_client.srem(f"topic_subscribers:{topic}", agent_id)
        if agent_id in self.agent_subscriptions:
            self.agent_subscriptions[agent_id].discard(topic)
        
        logger.info(f"Agent {agent_id} unsubscribed from topic: {topic}")
    
    async def register_message_handler(self, agent_id: str, handler: Callable[[Message], None]):
        """Register a message handler for an agent"""
        
        self.message_handlers[agent_id].append(handler)
        logger.info(f"Message handler registered for {agent_id}")
    
    # ==================== Message Serialization ====================
    
    async def _serialize_message(self, message: Message) -> bytes:
        """Serialize message for storage/transmission"""
        
        data = message.to_dict()
        json_data = json.dumps(data, default=str).encode()
        
        if len(json_data) > self.config["max_message_size"]:
            raise ValueError("Message too large")
        
        return json_data
    
    async def _deserialize_message(self, data: bytes) -> Message:
        """Deserialize message from storage/transmission"""
        
        json_data = data.decode() if isinstance(data, bytes) else data
        message_dict = json.loads(json_data)
        
        return Message.from_dict(message_dict)
    
    # ==================== Compression and Encryption ====================
    
    async def _should_compress(self, message: Message) -> bool:
        """Determine if message should be compressed"""
        
        if message.compressed:
            return False
        
        # Estimate message size
        estimated_size = len(json.dumps(message.payload, default=str))
        return estimated_size > self.config["compression_threshold"]
    
    async def _compress_message(self, message: Message) -> Message:
        """Compress message payload"""
        
        try:
            # Compress payload
            payload_json = json.dumps(message.payload, default=str)
            compressed_payload = zlib.compress(payload_json.encode())
            
            # Store compressed payload
            message.payload = {"__compressed__": compressed_payload.hex()}
            message.compressed = True
            
            return message
            
        except Exception as e:
            logger.error(f"Message compression failed: {e}")
            return message
    
    async def _decompress_message(self, message: Message) -> Message:
        """Decompress message payload"""
        
        try:
            if "__compressed__" in message.payload:
                compressed_data = bytes.fromhex(message.payload["__compressed__"])
                decompressed_data = zlib.decompress(compressed_data)
                message.payload = json.loads(decompressed_data.decode())
                message.compressed = False
            
            return message
            
        except Exception as e:
            logger.error(f"Message decompression failed: {e}")
            return message
    
    async def _encrypt_message(self, message: Message) -> Message:
        """Encrypt message payload"""
        
        try:
            # Encrypt payload
            payload_json = json.dumps(message.payload, default=str)
            encrypted_payload = self.cipher.encrypt(payload_json.encode())
            
            # Store encrypted payload
            message.payload = {"__encrypted__": encrypted_payload.hex()}
            message.encrypted = True
            
            return message
            
        except Exception as e:
            logger.error(f"Message encryption failed: {e}")
            return message
    
    async def _decrypt_message(self, message: Message) -> Message:
        """Decrypt message payload"""
        
        try:
            if "__encrypted__" in message.payload:
                encrypted_data = bytes.fromhex(message.payload["__encrypted__"])
                decrypted_data = self.cipher.decrypt(encrypted_data)
                message.payload = json.loads(decrypted_data.decode())
                message.encrypted = False
            
            return message
            
        except Exception as e:
            logger.error(f"Message decryption failed: {e}")
            return message
    
    # ==================== Background Services ====================
    
    async def _pubsub_listener(self):
        """Listen for pub-sub notifications"""
        
        while self.running:
            try:
                # Subscribe to all agent notification channels
                for agent_id in self.agent_presence:
                    await self.pubsub.subscribe(f"agent_notify:{agent_id}")
                
                # Listen for messages
                async for message in self.pubsub.listen():
                    if message["type"] == "message":
                        try:
                            data = json.loads(message["data"])
                            logger.debug(f"Pub-sub notification: {data}")
                        except Exception as e:
                            logger.error(f"Pub-sub message parsing error: {e}")
                            
            except Exception as e:
                logger.error(f"Pub-sub listener error: {e}")
                await asyncio.sleep(5)
    
    async def _presence_monitor(self):
        """Monitor agent presence and health"""
        
        while self.running:
            try:
                current_time = datetime.now()
                stale_agents = []
                
                for agent_id, presence in self.agent_presence.items():
                    # Check for stale presence (no heartbeat in 5 minutes)
                    if current_time - presence.last_seen > timedelta(minutes=5):
                        stale_agents.append(agent_id)
                        presence.status = "offline"
                
                # Clean up stale agents
                for agent_id in stale_agents:
                    logger.warning(f"Agent {agent_id} appears to be offline")
                    # Optionally unregister or mark as unavailable
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Presence monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_expired_messages(self):
        """Clean up expired messages"""
        
        while self.running:
            try:
                current_time = datetime.now()
                
                # Clean up expired pending messages
                expired_messages = []
                for msg_id, message in self.pending_messages.items():
                    if message.expires_at and current_time > message.expires_at:
                        expired_messages.append(msg_id)
                
                for msg_id in expired_messages:
                    del self.pending_messages[msg_id]
                
                # Clean up expired response futures
                expired_futures = []
                for correlation_id, future in self.response_futures.items():
                    if future.done():
                        expired_futures.append(correlation_id)
                
                for correlation_id in expired_futures:
                    del self.response_futures[correlation_id]
                
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(300)
    
    async def _metrics_collector(self):
        """Collect performance metrics"""
        
        while self.running:
            try:
                # Calculate system-wide metrics
                total_sent = sum(stats.total_sent for stats in self.stats.values())
                total_received = sum(stats.total_received for stats in self.stats.values())
                total_failed = sum(stats.total_failed for stats in self.stats.values())
                
                error_rate = total_failed / max(total_sent, 1)
                
                # Update performance metrics
                self.performance_metrics.update({
                    "total_messages_sent": total_sent,
                    "total_messages_received": total_received,
                    "total_messages_failed": total_failed,
                    "error_rate": error_rate,
                    "active_agents": len([p for p in self.agent_presence.values() if p.status == "online"]),
                    "total_agents": len(self.agent_presence),
                    "active_subscriptions": sum(len(subs) for subs in self.agent_subscriptions.values()),
                    "timestamp": datetime.now().isoformat()
                })
                
                # Store in Redis
                await self.redis_client.hset(
                    "message_bus:metrics",
                    mapping={
                        k: json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                        for k, v in self.performance_metrics.items()
                    }
                )
                
                logger.info(
                    f"Message bus metrics - Sent: {total_sent}, "
                    f"Received: {total_received}, Failed: {total_failed}, "
                    f"Error Rate: {error_rate:.2%}, Active Agents: {len(self.agent_presence)}"
                )
                
                await asyncio.sleep(60)  # Collect metrics every minute
                
            except Exception as e:
                logger.error(f"Metrics collector error: {e}")
                await asyncio.sleep(60)
    
    # ==================== Utility Methods ====================
    
    async def _validate_message(self, message: Message) -> bool:
        """Validate message before sending"""
        
        if not message.sender_id:
            logger.error("Message missing sender_id")
            return False
        
        if message.pattern == CommunicationPattern.POINT_TO_POINT and not message.recipient_id:
            logger.error("Point-to-point message missing recipient_id")
            return False
        
        if message.pattern == CommunicationPattern.PUBLISH_SUBSCRIBE and not message.topic:
            logger.error("Pub-sub message missing topic")
            return False
        
        return True
    
    # ==================== Public API Methods ====================
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get message bus system status"""
        
        return {
            "running": self.running,
            "total_agents": len(self.agent_presence),
            "online_agents": len([p for p in self.agent_presence.values() if p.status == "online"]),
            "total_messages_processed": len(self.message_history),
            "active_subscriptions": sum(len(subs) for subs in self.agent_subscriptions.values()),
            "performance_metrics": self.performance_metrics,
            "config": self.config
        }
    
    async def get_agent_presence(self) -> Dict[str, Any]:
        """Get all agent presence information"""
        
        return {
            agent_id: {
                "status": presence.status,
                "capabilities": presence.capabilities,
                "current_load": presence.current_load,
                "last_seen": presence.last_seen.isoformat(),
                "endpoint": presence.endpoint
            }
            for agent_id, presence in self.agent_presence.items()
        }
    
    async def get_message_stats(self, agent_id: Optional[str] = None) -> Dict[str, Any]:
        """Get message statistics"""
        
        if agent_id:
            if agent_id in self.stats:
                stats = self.stats[agent_id]
                return asdict(stats)
            else:
                return {}
        else:
            return {
                agent_id: asdict(stats)
                for agent_id, stats in self.stats.items()
            }


# ==================== Factory Function ====================

def create_message_bus(redis_url: str = "redis://redis:6379",
                      encryption_key: Optional[bytes] = None) -> AdvancedMessageBus:
    """Factory function to create message bus"""
    return AdvancedMessageBus(redis_url, encryption_key)


# ==================== Message Factory ====================

class MessageFactory:
    """Factory for creating common message types"""
    
    @staticmethod
    def create_task_assignment(sender_id: str, recipient_id: str, 
                             task_data: Dict[str, Any]) -> Message:
        """Create task assignment message"""
        return Message(
            id=str(uuid.uuid4()),
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=MessageType.TASK_ASSIGNMENT,
            pattern=CommunicationPattern.POINT_TO_POINT,
            priority=MessagePriority.HIGH,
            payload=task_data
        )
    
    @staticmethod
    def create_coordination_request(sender_id: str, coordination_data: Dict[str, Any]) -> Message:
        """Create coordination request message"""
        return Message(
            id=str(uuid.uuid4()),
            sender_id=sender_id,
            message_type=MessageType.COORDINATION_REQUEST,
            pattern=CommunicationPattern.BROADCAST,
            priority=MessagePriority.HIGH,
            payload=coordination_data
        )
    
    @staticmethod
    def create_knowledge_share(sender_id: str, topic: str, 
                             knowledge_data: Dict[str, Any]) -> Message:
        """Create knowledge sharing message"""
        return Message(
            id=str(uuid.uuid4()),
            sender_id=sender_id,
            message_type=MessageType.KNOWLEDGE_SHARE,
            pattern=CommunicationPattern.PUBLISH_SUBSCRIBE,
            topic=topic,
            priority=MessagePriority.NORMAL,
            payload=knowledge_data
        )
    
    @staticmethod
    def create_status_update(sender_id: str, status: str, 
                           metadata: Optional[Dict[str, Any]] = None) -> Message:
        """Create status update message"""
        return Message(
            id=str(uuid.uuid4()),
            sender_id=sender_id,
            message_type=MessageType.SYSTEM_UPDATE,
            pattern=CommunicationPattern.BROADCAST,
            priority=MessagePriority.NORMAL,
            payload={"status": status, "metadata": metadata or {}}
        )


# ==================== Example Usage ====================

async def example_message_bus():
    """Example of using the advanced message bus"""
    
    # Initialize message bus
    bus = create_message_bus("redis://redis:6379")
    await bus.initialize()
    
    # Register agents
    await bus.register_agent(
        "agent1",
        capabilities=["reasoning", "code_generation"],
        endpoint="http://agent1:8080"
    )
    
    await bus.register_agent(
        "agent2", 
        capabilities=["testing", "deployment"],
        endpoint="http://agent2:8080"
    )
    
    # Subscribe to topics
    await bus.subscribe_to_topic("agent1", "development")
    await bus.subscribe_to_topic("agent2", "development")
    
    # Send various types of messages
    
    # Point-to-point message
    p2p_msg = MessageFactory.create_task_assignment(
        "orchestrator", "agent1",
        {"task": "Generate code for API endpoint"}
    )
    await bus.send_message(p2p_msg)
    
    # Broadcast message
    broadcast_msg = MessageFactory.create_status_update(
        "orchestrator", "system_update",
        {"update": "New deployment available"}
    )
    await bus.broadcast_message(broadcast_msg)
    
    # Publish to topic
    topic_msg = MessageFactory.create_knowledge_share(
        "agent1", "development",
        {"knowledge": "Best practices for API design"}
    )
    await bus.publish_to_topic("development", topic_msg)
    
    # Request-response pattern
    request_msg = Message(
        id=str(uuid.uuid4()),
        sender_id="orchestrator",
        recipient_id="agent2",
        message_type=MessageType.CAPABILITY_QUERY,
        pattern=CommunicationPattern.REQUEST_RESPONSE,
        payload={"query": "What are your current capabilities?"}
    )
    
    response = await bus.send_request(request_msg, timeout=10.0)
    if response:
        logger.info(f"Received response: {response.payload}")
    
    # Get system status
    status = await bus.get_system_status()
    logger.info(f"System status: {status}")
    
    await bus.shutdown()


if __name__ == "__main__":
    asyncio.run(example_message_bus())
