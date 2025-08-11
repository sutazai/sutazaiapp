#!/usr/bin/env python3
"""
Base agent class with integrated RabbitMQ messaging capabilities.
All agents should inherit from this class for consistent messaging.
"""
import asyncio
import logging
import signal
import sys
from datetime import datetime
import os

# Add schemas to path
sys.path.insert(0, '/opt/sutazaiapp')
from agents.core.rabbitmq_client import RabbitMQClient
from schemas.agent_messages import AgentRegistrationMessage, AgentHeartbeatMessage
from schemas.base import AgentStatus
from schemas.system_messages import ErrorMessage
from schemas.base import Priority

logger = logging.getLogger(__name__)


class MessagingAgent:
    """
    Base class for agents with RabbitMQ messaging capabilities.
    Provides automatic registration, heartbeat, and message handling.
    """
    
    def __init__(
        self,
        agent_id: str,
        agent_type: str,
        capabilities: list = None,
        version: str = "1.0.0",
        port: int = 8080
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.capabilities = capabilities or []
        self.version = version
        self.port = port
        self.host = os.getenv("HOSTNAME", "localhost")
        
        # Agent state
        self.status = AgentStatus.INITIALIZING
        self.start_time = datetime.utcnow()
        self.active_tasks = {}
        self.metrics = {
            "tasks_processed": 0,
            "tasks_failed": 0,
            "total_processing_time": 0,
            "error_count": 0
        }
        
        # RabbitMQ client
        self.rabbitmq = RabbitMQClient(agent_id, agent_type)
        self.heartbeat_task = None
        self.shutdown_event = asyncio.Event()
        
        # Message handlers
        self.message_handlers = {}
        
        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.{agent_id}")
        
    async def initialize(self):
        """Initialize agent and connect to RabbitMQ"""
        try:
            # Connect to RabbitMQ
            connected = await self.rabbitmq.connect()
            if not connected:
                self.logger.error("Failed to connect to RabbitMQ")
                return False
            
            # Register default message handlers
            await self._register_default_handlers()
            
            # Send registration message
            await self._register_with_system()
            
            # Start heartbeat
            self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            # Update status
            self.status = AgentStatus.READY
            self.logger.info(f"Agent {self.agent_id} initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agent: {e}")
            self.status = AgentStatus.ERROR
            return False
    
    async def _register_with_system(self):
        """Register agent with the system"""
        try:
            registration = AgentRegistrationMessage(
                source_agent=self.agent_id,
                agent_id=self.agent_id,
                agent_type=self.agent_type,
                capabilities=self.capabilities,
                version=self.version,
                host=self.host,
                port=self.port,
                max_concurrent_tasks=10,
                supported_message_types=list(self.message_handlers.keys())
            )
            
            await self.rabbitmq.publish(registration)
            self.logger.info(f"Agent {self.agent_id} registered with system")
            
        except Exception as e:
            self.logger.error(f"Failed to register agent: {e}")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        heartbeat_interval = 30  # seconds
        
        while not self.shutdown_event.is_set():
            try:
                # Calculate metrics
                cpu_usage = 0.0
                memory_usage = 0.0
                
                try:
                    import psutil
                    process = psutil.Process()
                    cpu_usage = process.cpu_percent()
                    memory_usage = process.memory_percent()
                except Exception as e:
                    # Suppressed exception (was bare except)
                    logger.debug(f"Suppressed exception: {e}")
                    pass
                
                # Send heartbeat
                heartbeat = AgentHeartbeatMessage(
                    source_agent=self.agent_id,
                    agent_id=self.agent_id,
                    status=self.status,
                    current_load=len(self.active_tasks) / 10.0,  # Assuming max 10 tasks
                    active_tasks=len(self.active_tasks),
                    available_capacity=10 - len(self.active_tasks),
                    cpu_usage=cpu_usage,
                    memory_usage=memory_usage,
                    uptime_seconds=(datetime.utcnow() - self.start_time).total_seconds(),
                    error_count=self.metrics["error_count"]
                )
                
                await self.rabbitmq.publish(heartbeat)
                
                # Wait for next heartbeat
                await asyncio.sleep(heartbeat_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(heartbeat_interval)
    
    async def _register_default_handlers(self):
        """Register default message handlers"""
        # Override in subclasses to add specific handlers
        pass
    
    async def register_handler(
        self,
        message_type: str,
        handler: Callable
    ):
        """
        Register a message handler.
        
        Args:
            message_type: Type of message to handle
            handler: Async function to handle the message
        """
        self.message_handlers[message_type] = handler
        await self.rabbitmq.register_handler(message_type, handler)
        self.logger.debug(f"Registered handler for {message_type}")
    
    async def send_error(
        self,
        error_message: str,
        error_code: str = "GENERAL_ERROR",
        task_id: Optional[str] = None,
        severity: Priority = Priority.HIGH
    ):
        """Send error message to system"""
        try:
            error = ErrorMessage(
                source_agent=self.agent_id,
                error_id=f"{self.agent_id}_{datetime.utcnow().timestamp()}",
                error_code=error_code,
                error_message=error_message,
                error_type="agent_error",
                severity=severity,
                affected_task_id=task_id,
                affected_agent_id=self.agent_id,
                retry_possible=True
            )
            
            await self.rabbitmq.publish(error)
            self.metrics["error_count"] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to send error message: {e}")
    
    async def start_consuming(self):
        """Start consuming messages from assigned queues"""
        try:
            # Get queue configuration
            from schemas.queue_config import AGENT_QUEUE_MAP, QueueConfig
            
            agent_config = AGENT_QUEUE_MAP.get(self.agent_type, {})
            subscribe_queues = agent_config.get('subscribe', [])
            
            # Always consume from agent-specific queue
            agent_queue = QueueConfig.agent_queue(self.agent_id)
            if agent_queue not in subscribe_queues:
                subscribe_queues.append(agent_queue)
            
            # Start consumers
            for queue_name in subscribe_queues:
                await self.rabbitmq.consume(
                    queue_name,
                    self.rabbitmq.process_by_type,
                    auto_ack=False
                )
                self.logger.info(f"Started consuming from {queue_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to start consumers: {e}")
            raise
    
    async def shutdown(self):
        """Gracefully shutdown the agent"""
        try:
            self.logger.info(f"Shutting down agent {self.agent_id}")
            
            # Update status
            self.status = AgentStatus.SHUTDOWN
            
            # Signal shutdown
            self.shutdown_event.set()
            
            # Cancel heartbeat
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
                try:
                    await self.heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            # Close RabbitMQ connection
            await self.rabbitmq.close()
            
            self.logger.info(f"Agent {self.agent_id} shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(sig, frame):
            asyncio.create_task(self.shutdown())
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def run(self):
        """Main agent run loop - override in subclasses"""
        try:
            # Initialize
            if not await self.initialize():
                return
            
            # Start consuming messages
            await self.start_consuming()
            
            # Setup signal handlers
            self.setup_signal_handlers()
            
            # Keep running until shutdown
            await self.shutdown_event.wait()
            
        except Exception as e:
            self.logger.error(f"Agent run loop error: {e}")
            await self.shutdown()