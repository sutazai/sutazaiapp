#!/usr/bin/env python3
"""
Shared RabbitMQ messaging module for all agents.
Provides consistent message schemas and communication patterns.
"""
import json
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from enum import Enum
import aio_pika
from aio_pika import ExchangeType, Message, DeliveryMode
from pydantic import BaseModel, Field
import os

logger = logging.getLogger(__name__)

# Message Types
class MessageType(str, Enum):
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    TASK_STATUS = "task_status"
    RESOURCE_REQUEST = "resource_request"
    RESOURCE_RESPONSE = "resource_response"
    AGENT_REGISTRATION = "agent_registration"
    AGENT_HEARTBEAT = "agent_heartbeat"
    ERROR = "error"

# Priority Levels
class Priority(int, Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3

# Message Schemas
class BaseMessage(BaseModel):
    """Base message schema for all agent communications"""
    message_id: str
    message_type: MessageType
    source_agent: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None
    priority: Priority = Priority.NORMAL
    
    class Config:
        use_enum_values = True

class TaskMessage(BaseMessage):
    """Task-related message schema"""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    target_agent: Optional[str] = None
    timeout_seconds: int = 300
    retry_count: int = 0
    max_retries: int = 3

class ResourceMessage(BaseMessage):
    """Resource allocation message schema"""
    resource_type: str  # cpu, memory, gpu, etc.
    resource_amount: float
    resource_unit: str  # cores, GB, etc.
    duration_seconds: Optional[int] = None
    exclusive: bool = False

class StatusMessage(BaseMessage):
    """Status update message schema"""
    task_id: str
    status: str  # pending, processing, completed, failed
    progress: Optional[float] = None  # 0.0 to 1.0
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ErrorMessage(BaseMessage):
    """Error message schema"""
    error_code: str
    error_message: str
    error_details: Optional[Dict[str, Any]] = None
    original_message_id: Optional[str] = None


class RabbitMQClient:
    """Async RabbitMQ client for agent messaging"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.connection = None
        self.channel = None
        self.exchange = None
        self.queue = None
        self.consumer_tag = None
        self.callbacks = {}
        
        # Connection parameters from environment
        self.rabbitmq_url = os.getenv(
            "RABBITMQ_URL", 
            "amqp://guest:guest@rabbitmq:5672/"
        )
        
    async def connect(self):
        """Establish connection to RabbitMQ"""
        try:
            self.connection = await aio_pika.connect_robust(
                self.rabbitmq_url,
                loop=asyncio.get_event_loop()
            )
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=10)
            
            # Declare main exchange
            self.exchange = await self.channel.declare_exchange(
                "sutazai.agents",
                ExchangeType.TOPIC,
                durable=True
            )
            
            # Declare agent-specific queue
            self.queue = await self.channel.declare_queue(
                f"agent.{self.agent_id}",
                durable=True,
                arguments={
                    "x-message-ttl": 3600000,  # 1 hour TTL
                    "x-max-length": 1000  # Max 1000 messages
                }
            )
            
            # Bind queue to exchange with routing patterns
            await self.queue.bind(self.exchange, f"agent.{self.agent_id}.*")
            await self.queue.bind(self.exchange, "agent.all.*")
            await self.queue.bind(self.exchange, f"task.{self.agent_id}.*")
            
            logger.info(f"RabbitMQ connected for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise
    
    async def disconnect(self):
        """Close RabbitMQ connection"""
        try:
            if self.consumer_tag:
                await self.queue.cancel(self.consumer_tag)
            if self.connection:
                await self.connection.close()
            logger.info(f"RabbitMQ disconnected for agent {self.agent_id}")
        except Exception as e:
            logger.error(f"Error disconnecting from RabbitMQ: {e}")
    
    async def publish_message(
        self, 
        message: BaseMessage, 
        routing_key: str,
        exchange_name: str = None
    ):
        """Publish a message to RabbitMQ"""
        try:
            exchange = self.exchange if not exchange_name else await self.channel.get_exchange(exchange_name)
            
            # Serialize message
            message_body = message.json().encode()
            
            # Create AMQP message with properties
            amqp_message = Message(
                body=message_body,
                delivery_mode=DeliveryMode.PERSISTENT,
                priority=message.priority,
                correlation_id=message.correlation_id,
                timestamp=datetime.utcnow(),
                headers={
                    "source_agent": self.agent_id,
                    "message_type": message.message_type,
                    "message_id": message.message_id
                }
            )
            
            # Publish with confirmation
            await exchange.publish(
                amqp_message,
                routing_key=routing_key
            )
            
            logger.debug(f"Published message {message.message_id} to {routing_key}")
            
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            raise
    
    async def consume_messages(self, callback: Callable):
        """Start consuming messages from the queue"""
        try:
            async with self.queue.iterator() as queue_iter:
                self.consumer_tag = queue_iter.consumer_tag
                
                async for message in queue_iter:
                    async with message.process():
                        try:
                            # Parse message
                            body = json.loads(message.body.decode())
                            message_type = body.get("message_type")
                            
                            # Route to appropriate handler
                            if message_type in self.callbacks:
                                await self.callbacks[message_type](body)
                            else:
                                await callback(body)
                                
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                            # Message will be requeued due to exception
                            raise
                            
        except asyncio.CancelledError:
            logger.info("Message consumption cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in message consumer: {e}")
            raise
    
    def register_handler(self, message_type: MessageType, callback: Callable):
        """Register a callback for a specific message type"""
        self.callbacks[message_type] = callback
        logger.info(f"Registered handler for {message_type}")
    
    async def publish_task(
        self,
        task_id: str,
        task_type: str,
        payload: Dict[str, Any],
        target_agent: Optional[str] = None,
        priority: Priority = Priority.NORMAL
    ):
        """Publish a task request"""
        message = TaskMessage(
            message_id=f"{self.agent_id}_{task_id}_{datetime.utcnow().timestamp()}",
            message_type=MessageType.TASK_REQUEST,
            source_agent=self.agent_id,
            task_id=task_id,
            task_type=task_type,
            payload=payload,
            target_agent=target_agent,
            priority=priority
        )
        
        routing_key = f"task.{target_agent if target_agent else 'all'}.request"
        await self.publish_message(message, routing_key)
        return message
    
    async def publish_status(
        self,
        task_id: str,
        status: str,
        progress: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """Publish a status update"""
        message = StatusMessage(
            message_id=f"{self.agent_id}_status_{datetime.utcnow().timestamp()}",
            message_type=MessageType.TASK_STATUS,
            source_agent=self.agent_id,
            task_id=task_id,
            status=status,
            progress=progress,
            details=details
        )
        
        routing_key = f"agent.all.status"
        await self.publish_message(message, routing_key)
        return message
    
    async def request_resource(
        self,
        resource_type: str,
        resource_amount: float,
        resource_unit: str,
        duration_seconds: Optional[int] = None,
        exclusive: bool = False
    ):
        """Request resource allocation"""
        message = ResourceMessage(
            message_id=f"{self.agent_id}_resource_{datetime.utcnow().timestamp()}",
            message_type=MessageType.RESOURCE_REQUEST,
            source_agent=self.agent_id,
            resource_type=resource_type,
            resource_amount=resource_amount,
            resource_unit=resource_unit,
            duration_seconds=duration_seconds,
            exclusive=exclusive
        )
        
        routing_key = "resource.arbitrator.request"
        await self.publish_message(message, routing_key)
        return message
    
    async def publish_error(
        self,
        error_code: str,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None,
        original_message_id: Optional[str] = None
    ):
        """Publish an error message"""
        message = ErrorMessage(
            message_id=f"{self.agent_id}_error_{datetime.utcnow().timestamp()}",
            message_type=MessageType.ERROR,
            source_agent=self.agent_id,
            error_code=error_code,
            error_message=error_message,
            error_details=error_details,
            original_message_id=original_message_id
        )
        
        routing_key = "agent.all.error"
        await self.publish_message(message, routing_key)
        return message


class MessageProcessor:
    """Base class for processing messages"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.rabbitmq_client = RabbitMQClient(agent_id)
        
    async def start(self):
        """Start the message processor"""
        await self.rabbitmq_client.connect()
        
        # Register handlers
        self.rabbitmq_client.register_handler(
            MessageType.TASK_REQUEST, 
            self.handle_task_request
        )
        self.rabbitmq_client.register_handler(
            MessageType.RESOURCE_RESPONSE,
            self.handle_resource_response
        )
        
        # Start consuming
        await self.rabbitmq_client.consume_messages(self.handle_message)
    
    async def stop(self):
        """Stop the message processor"""
        await self.rabbitmq_client.disconnect()
    
    async def handle_message(self, message: Dict[str, Any]):
        """Default message handler - override in subclasses"""
        logger.info(f"Received message: {message.get('message_id')}")
    
    async def handle_task_request(self, message: Dict[str, Any]):
        """Handle task request - override in subclasses"""
        logger.info(f"Received task request: {message.get('task_id')}")
    
    async def handle_resource_response(self, message: Dict[str, Any]):
        """Handle resource response - override in subclasses"""
        logger.info(f"Received resource response for: {message.get('resource_type')}")