#!/usr/bin/env python3
"""
Enhanced RabbitMQ client with real message handling for all agents.
Uses centralized schemas and queue configuration.
"""
import json
import logging
import asyncio
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime
import aio_pika
from aio_pika import ExchangeType, Message, DeliveryMode
import os
import sys

# Add schemas to path
sys.path.insert(0, '/opt/sutazaiapp')
from schemas.queue_config import ExchangeConfig, QueueConfig, RoutingKeys, QueueArguments, AGENT_QUEUE_MAP
from schemas.base import BaseMessage

logger = logging.getLogger(__name__)


class RabbitMQClient:
    """
    Production RabbitMQ client for agent communication.
    Handles connection management, message publishing, and consumption.
    """
    
    def __init__(self, agent_id: str, agent_type: str = None):
        self.agent_id = agent_id
        self.agent_type = agent_type or agent_id
        self.connection = None
        self.channel = None
        self.exchanges = {}
        self.queues = {}
        self.consumers = {}
        self.message_handlers = {}
        self.url = os.getenv('RABBITMQ_URL', 'amqp://guest:guest@rabbitmq:5672/')
        
    async def connect(self):
        """Establish connection to RabbitMQ"""
        try:
            self.connection = await aio_pika.connect_robust(
                self.url,
                client_properties={
                    'client_id': self.agent_id,
                    'agent_type': self.agent_type
                }
            )
            self.channel = await self.connection.channel()
            await self.channel.set_qos(prefetch_count=10)
            
            # Setup exchanges
            await self._setup_exchanges()
            
            # Setup queues for this agent
            await self._setup_queues()
            
            logger.info(f"RabbitMQ connected for agent {self.agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            return False
    
    async def _setup_exchanges(self):
        """Create all required exchanges"""
        exchange_configs = [
            (ExchangeConfig.MAIN, ExchangeType.TOPIC),
            (ExchangeConfig.AGENTS, ExchangeType.TOPIC),
            (ExchangeConfig.TASKS, ExchangeType.TOPIC),
            (ExchangeConfig.RESOURCES, ExchangeType.TOPIC),
            (ExchangeConfig.SYSTEM, ExchangeType.TOPIC),
            (ExchangeConfig.DLX, ExchangeType.FANOUT),
        ]
        
        for exchange_name, exchange_type in exchange_configs:
            exchange = await self.channel.declare_exchange(
                exchange_name,
                exchange_type,
                durable=True
            )
            self.exchanges[exchange_name] = exchange
            
    async def _setup_queues(self):
        """Setup queues based on agent configuration"""
        # Get queue config for this agent
        agent_config = AGENT_QUEUE_MAP.get(self.agent_type, {})
        subscribe_queues = agent_config.get('subscribe', [])
        
        # Always create agent-specific queue
        agent_queue_name = QueueConfig.agent_queue(self.agent_id)
        if agent_queue_name not in subscribe_queues:
            subscribe_queues.append(agent_queue_name)
        
        # Create and bind queues
        for queue_name in subscribe_queues:
            queue = await self.channel.declare_queue(
                queue_name,
                durable=True,
                arguments=QueueArguments.standard_queue()
            )
            self.queues[queue_name] = queue
            
            # Bind to appropriate exchange
            if "agent" in queue_name:
                await queue.bind(
                    ExchangeConfig.AGENTS,
                    routing_key=f"agent.{self.agent_id}.#"
                )
            elif "task" in queue_name:
                await queue.bind(
                    ExchangeConfig.TASKS,
                    routing_key="task.#"
                )
            elif "resource" in queue_name:
                await queue.bind(
                    ExchangeConfig.RESOURCES,
                    routing_key="resource.#"
                )
            elif "system" in queue_name:
                await queue.bind(
                    ExchangeConfig.SYSTEM,
                    routing_key="system.#"
                )
    
    async def publish(
        self,
        message: BaseMessage,
        exchange: str = None,
        routing_key: str = None
    ):
        """
        Publish a message to RabbitMQ.
        
        Args:
            message: Message object to publish
            exchange: Exchange name (auto-determined if not provided)
            routing_key: Routing key (auto-determined if not provided)
        """
        try:
            if not self.channel:
                await self.connect()
            
            # Auto-determine exchange and routing key based on message type
            if not exchange:
                exchange = self._get_exchange_for_message(message)
            
            if not routing_key:
                routing_key = self._get_routing_key_for_message(message)
            
            # Serialize message
            message_body = message.json().encode()
            
            # Create AMQP message
            amqp_message = Message(
                message_body,
                delivery_mode=DeliveryMode.PERSISTENT,
                headers={
                    'source_agent': self.agent_id,
                    'message_type': message.message_type,
                    'timestamp': datetime.utcnow().isoformat()
                },
                expiration=str(message.ttl * 1000) if message.ttl else None
            )
            
            # Publish
            exchange_obj = self.exchanges.get(exchange)
            if exchange_obj:
                await exchange_obj.publish(amqp_message, routing_key=routing_key)
                logger.debug(f"Published {message.message_type} to {exchange}/{routing_key}")
            else:
                logger.error(f"Exchange {exchange} not found")
                
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            raise
    
    async def consume(
        self,
        queue_name: str,
        handler: Callable,
        auto_ack: bool = False
    ):
        """
        Start consuming messages from a queue.
        
        Args:
            queue_name: Name of queue to consume from
            handler: Async function to handle messages
            auto_ack: Whether to auto-acknowledge messages
        """
        try:
            queue = self.queues.get(queue_name)
            if not queue:
                logger.error(f"Queue {queue_name} not found")
                return
            
            async def process_message(message: aio_pika.IncomingMessage):
                async with message.process(ignore_processed=True):
                    try:
                        # Deserialize message
                        body = json.loads(message.body.decode())
                        
                        # Call handler
                        await handler(body, message)
                        
                        if not auto_ack:
                            await message.ack()
                            
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        if not auto_ack:
                            await message.nack(requeue=True)
            
            # Start consuming
            consumer = await queue.consume(process_message, no_ack=auto_ack)
            self.consumers[queue_name] = consumer
            logger.info(f"Started consuming from {queue_name}")
            
        except Exception as e:
            logger.error(f"Failed to start consumer: {e}")
            raise
    
    async def stop_consumer(self, queue_name: str):
        """Stop consuming from a specific queue"""
        consumer = self.consumers.get(queue_name)
        if consumer:
            await consumer.cancel()
            del self.consumers[queue_name]
            logger.info(f"Stopped consuming from {queue_name}")
    
    async def register_handler(
        self,
        message_type: str,
        handler: Callable
    ):
        """Register a handler for a specific message type"""
        self.message_handlers[message_type] = handler
        logger.debug(f"Registered handler for {message_type}")
    
    async def process_by_type(self, message_data: Dict, raw_message: aio_pika.IncomingMessage):
        """Process message based on its type using registered handlers"""
        message_type = message_data.get('message_type')
        handler = self.message_handlers.get(message_type)
        
        if handler:
            await handler(message_data, raw_message)
        else:
            logger.warning(f"No handler registered for message type: {message_type}")
    
    def _get_exchange_for_message(self, message: BaseMessage) -> str:
        """Determine appropriate exchange based on message type"""
        msg_type = message.message_type
        
        if msg_type.startswith("agent"):
            return ExchangeConfig.AGENTS
        elif msg_type.startswith("task"):
            return ExchangeConfig.TASKS
        elif msg_type.startswith("resource"):
            return ExchangeConfig.RESOURCES
        elif msg_type.startswith("system"):
            return ExchangeConfig.SYSTEM
        else:
            return ExchangeConfig.MAIN
    
    def _get_routing_key_for_message(self, message: BaseMessage) -> str:
        """Generate routing key based on message type and content"""
        msg_type = message.message_type
        
        if msg_type == "agent.registration":
            return f"agent.{self.agent_id}.registration"
        elif msg_type == "agent.heartbeat":
            return f"agent.{self.agent_id}.heartbeat"
        elif msg_type == "task.request":
            return f"task.request.{message.priority}"
        elif msg_type == "task.assignment":
            target = getattr(message, 'assigned_agent', '*')
            return f"task.assignment.{target}"
        elif msg_type == "resource.request":
            return "resource.request.*"
        elif msg_type == "system.alert":
            severity = getattr(message, 'severity', 'info')
            return f"system.alert.{severity}"
        else:
            return msg_type.replace(".", ".")
    
    async def close(self):
        """Close RabbitMQ connection"""
        try:
            # Stop all consumers
            for queue_name in list(self.consumers.keys()):
                await self.stop_consumer(queue_name)
            
            # Close connection
            if self.connection:
                await self.connection.close()
                
            logger.info(f"RabbitMQ connection closed for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Error closing connection: {e}")
    
    async def health_check(self) -> bool:
        """Check if RabbitMQ connection is healthy"""
        try:
            if not self.connection or self.connection.is_closed:
                return False
            
            # Try to declare a temporary queue
            temp_queue = await self.channel.declare_queue(
                f"health_check_{self.agent_id}",
                auto_delete=True
            )
            await temp_queue.delete()
            return True
            
        except Exception:
            return False