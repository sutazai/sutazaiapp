"""
automation Agent Communication Bus
==========================

High-performance, Redis-based communication system for 28+ AI agents
with intelligent message routing, resource-aware task distribution,
and shared memory management.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from backend.app.schemas.message_types import MessageType, MessagePriority
import redis.asyncio as redis
from collections import defaultdict
import weakref

logger = logging.getLogger(__name__)


# Using canonical MessageType and MessagePriority


@dataclass
class AgentMessage:
    """Standardized message format for inter-agent communication"""
    id: str = None
    sender_id: str = None
    recipient_id: str = None  # None for broadcast
    message_type: MessageType = MessageType.TASK_REQUEST
    priority: MessagePriority = MessagePriority.NORMAL
    payload: Dict[str, Any] = None
    timestamp: float = None
    correlation_id: str = None  # For request-response tracking
    expires_at: float = None
    retry_count: int = 0
    max_retries: int = 3
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.payload is None:
            self.payload = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage"""
        return {
            'id': self.id,
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'message_type': self.message_type.value,
            'message_type_name': self.message_type.name,
            'priority': self.priority.value,
            'priority_name': self.priority.name,
            'payload': json.dumps(self.payload),
            'timestamp': self.timestamp,
            'correlation_id': self.correlation_id,
            'expires_at': self.expires_at,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create message from dictionary"""
        # Prefer robust parsing by name if available
        mt = data.get('message_type_name') or data.get('message_type')
        pr = data.get('priority_name') if 'priority_name' in data else data.get('priority')
        try:
            message_type = MessageType(mt) if isinstance(mt, str) and mt.lower() == mt else MessageType[mt]
        except Exception:
            message_type = MessageType.TASK_REQUEST
        try:
            if isinstance(pr, str):
                priority = MessagePriority[pr]
            else:
                priority = MessagePriority(pr)
        except Exception:
            priority = MessagePriority.NORMAL
        return cls(
            id=data['id'],
            sender_id=data['sender_id'],
            recipient_id=data.get('recipient_id'),
            message_type=message_type,
            priority=priority,
            payload=json.loads(data['payload']),
            timestamp=data['timestamp'],
            correlation_id=data.get('correlation_id'),
            expires_at=data.get('expires_at'),
            retry_count=data.get('retry_count', 0),
            max_retries=data.get('max_retries', 3)
        )


class AgentCommunicationBus:
    """
    High-performance communication bus for automation agents
    
    Features:
    - Priority-based message queuing
    - Resource-aware task distribution
    - Message persistence and retry logic
    - Pub/sub for real-time updates
    - Dead letter queue for failed messages
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self.redis_client = None
        self.message_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self.agent_capabilities: Dict[str, Set[str]] = {}
        self.agent_loads: Dict[str, float] = defaultdict(float)
        self.agent_last_seen: Dict[str, float] = {}
        self.running = False
        self.worker_tasks: List[asyncio.Task] = []
        
        # Channel names
        self.channels = {
            'task_queue': 'agi:tasks',
            'priority_queue': 'agi:priority_tasks',
            'broadcast': 'agi:broadcast',
            'agent_status': 'agi:agent_status',
            'emergency': 'agi:emergency',
            'dead_letter': 'agi:dead_letter',
            'metrics': 'agi:metrics'
        }
    
    async def initialize(self):
        """Initialize Redis connection and start background workers"""
        self.redis_client = redis.from_url(self.redis_url)
        
        # Test connection
        await self.redis_client.ping()
        logger.info("Connected to Redis communication bus")
        
        # Start background workers
        self.running = True
        self.worker_tasks = [
            asyncio.create_task(self._message_processor()),
            asyncio.create_task(self._priority_processor()),
            asyncio.create_task(self._health_monitor()),
            asyncio.create_task(self._cleanup_expired_messages())
        ]
        
        logger.info("Agent communication bus initialized")
    
    async def shutdown(self):
        """Clean shutdown of communication bus"""
        self.running = False
        
        # Cancel worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for graceful shutdown
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Agent communication bus shut down")
    
    async def register_agent(self, agent_id: str, capabilities: Set[str], 
                           current_load: float = 0.0):
        """Register an agent with its capabilities"""
        self.agent_capabilities[agent_id] = capabilities
        self.agent_loads[agent_id] = current_load
        self.agent_last_seen[agent_id] = time.time()
        
        # Store in Redis for persistence
        await self.redis_client.hset(
            "agi:agents",
            agent_id,
            json.dumps({
                'capabilities': list(capabilities),
                'load': current_load,
                'last_seen': time.time(),
                'status': 'active'
            })
        )
        
        # Broadcast agent registration
        await self.broadcast_message(AgentMessage(
            sender_id="communication_bus",
            message_type=MessageType.STATUS_UPDATE,
            payload={
                'event': 'agent_registered',
                'agent_id': agent_id,
                'capabilities': list(capabilities)
            }
        ))
    
    async def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        self.agent_capabilities.pop(agent_id, None)
        self.agent_loads.pop(agent_id, None)
        self.agent_last_seen.pop(agent_id, None)
        
        await self.redis_client.hdel("agi:agents", agent_id)
        
        logger.info(f"Agent {agent_id} unregistered")
    
    async def send_message(self, message: AgentMessage) -> bool:
        """Send a message through the communication bus"""
        try:
            # Set expiration if not set
            if message.expires_at is None:
                message.expires_at = time.time() + 300  # 5 minutes default
            
            # Route message based on priority and recipient
            if message.priority in [MessagePriority.CRITICAL, MessagePriority.HIGH]:
                queue_name = self.channels['priority_queue']
            else:
                queue_name = self.channels['task_queue']
            
            # Add to appropriate queue
            await self.redis_client.lpush(queue_name, json.dumps(message.to_dict()))
            
            # Store message for tracking
            await self.redis_client.hset(
                f"agi:messages:{message.id}",
                mapping=message.to_dict()
            )
            
            # Set expiration for cleanup
            await self.redis_client.expire(f"agi:messages:{message.id}", 3600)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to send message {message.id}: {e}")
            return False
    
    async def broadcast_message(self, message: AgentMessage):
        """Broadcast message to all active agents"""
        message.recipient_id = None  # Broadcast indicator
        
        # Use pub/sub for real-time broadcasting
        await self.redis_client.publish(
            self.channels['broadcast'],
            json.dumps(message.to_dict())
        )
    
    async def request_task_execution(self, task: Dict[str, Any], 
                                   required_capabilities: Set[str],
                                   priority: MessagePriority = MessagePriority.NORMAL,
                                   timeout: int = 300) -> Optional[Dict[str, Any]]:
        """Request task execution from capable agents"""
        
        # Find suitable agents
        suitable_agents = await self._find_suitable_agents(required_capabilities)
        
        if not suitable_agents:
            logger.warning(f"No agents found with capabilities: {required_capabilities}")
            return None
        
        # Select best agent based on current load
        best_agent = min(suitable_agents, key=lambda a: self.agent_loads[a])
        
        # Create task request message
        message = AgentMessage(
            sender_id="task_router",
            recipient_id=best_agent,
            message_type=MessageType.TASK_REQUEST,
            priority=priority,
            payload={
                'task': task,
                'required_capabilities': list(required_capabilities),
                'timeout': timeout
            },
            correlation_id=str(uuid.uuid4())
        )
        
        # Send message
        success = await self.send_message(message)
        if not success:
            return None
        
        # Wait for response
        return await self._wait_for_response(message.correlation_id, timeout)
    
    async def collaborative_task_execution(self, task: Dict[str, Any],
                                         agent_roles: Dict[str, Set[str]],
                                         coordination_strategy: str = "sequential") -> Dict[str, Any]:
        """Execute complex tasks requiring multiple agents"""
        
        collaboration_id = str(uuid.uuid4())
        results = {}
        
        if coordination_strategy == "sequential":
            # Execute tasks in sequence
            for role, capabilities in agent_roles.items():
                subtask = task.get(role, {})
                result = await self.request_task_execution(
                    subtask, capabilities, MessagePriority.HIGH
                )
                results[role] = result
                
                # Pass results to next agent
                if result:
                    task['previous_results'] = results
        
        elif coordination_strategy == "parallel":
            # Execute tasks in parallel
            tasks = []
            for role, capabilities in agent_roles.items():
                subtask = task.get(role, {})
                task_coroutine = self.request_task_execution(
                    subtask, capabilities, MessagePriority.HIGH
                )
                tasks.append((role, task_coroutine))
            
            # Wait for all tasks to complete
            for role, task_coroutine in tasks:
                result = await task_coroutine
                results[role] = result
        
        return {
            'collaboration_id': collaboration_id,
            'strategy': coordination_strategy,
            'results': results,
            'success': all(r is not None for r in results.values())
        }
    
    async def update_agent_load(self, agent_id: str, load: float):
        """Update agent's current load"""
        self.agent_loads[agent_id] = load
        self.agent_last_seen[agent_id] = time.time()
        
        # Update in Redis
        agent_data = await self.redis_client.hget("agi:agents", agent_id)
        if agent_data:
            data = json.loads(agent_data)
            data['load'] = load
            data['last_seen'] = time.time()
            await self.redis_client.hset("agi:agents", agent_id, json.dumps(data))
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system communication metrics"""
        
        # Count messages in queues
        task_queue_size = await self.redis_client.llen(self.channels['task_queue'])
        priority_queue_size = await self.redis_client.llen(self.channels['priority_queue'])
        dead_letter_size = await self.redis_client.llen(self.channels['dead_letter'])
        
        # Agent statistics
        active_agents = len([a for a, last_seen in self.agent_last_seen.items() 
                           if time.time() - last_seen < 60])
        
        avg_load = sum(self.agent_loads.values()) / len(self.agent_loads) if self.agent_loads else 0
        
        return {
            'task_queue_size': task_queue_size,
            'priority_queue_size': priority_queue_size,
            'dead_letter_size': dead_letter_size,
            'active_agents': active_agents,
            'total_registered_agents': len(self.agent_capabilities),
            'average_agent_load': avg_load,
            'timestamp': time.time()
        }
    
    # Internal Methods
    
    async def _find_suitable_agents(self, required_capabilities: Set[str]) -> List[str]:
        """Find agents with required capabilities"""
        suitable_agents = []
        
        for agent_id, capabilities in self.agent_capabilities.items():
            if required_capabilities.issubset(capabilities):
                # Check if agent is still active (heartbeat within last minute)
                if time.time() - self.agent_last_seen.get(agent_id, 0) < 60:
                    suitable_agents.append(agent_id)
        
        return suitable_agents
    
    async def _wait_for_response(self, correlation_id: str, timeout: int) -> Optional[Dict[str, Any]]:
        """Wait for response message with correlation ID"""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Check for response message
            response_key = f"agi:responses:{correlation_id}"
            response_data = await self.redis_client.get(response_key)
            
            if response_data:
                await self.redis_client.delete(response_key)
                return json.loads(response_data)
            
            await asyncio.sleep(0.1)  # Check every 100ms
        
        logger.warning(f"Timeout waiting for response to correlation_id: {correlation_id}")
        return None
    
    async def _message_processor(self):
        """Background worker to process regular messages"""
        while self.running:
            try:
                # Block for up to 1 second waiting for messages
                message_data = await self.redis_client.brpop(
                    self.channels['task_queue'], timeout=1
                )
                
                if message_data:
                    _, raw_message = message_data
                    await self._process_message(json.loads(raw_message))
                    
            except Exception as e:
                logger.error(f"Error in message processor: {e}")
                await asyncio.sleep(1)
    
    async def _priority_processor(self):
        """Background worker to process priority messages"""
        while self.running:
            try:
                message_data = await self.redis_client.brpop(
                    self.channels['priority_queue'], timeout=1
                )
                
                if message_data:
                    _, raw_message = message_data
                    await self._process_message(json.loads(raw_message))
                    
            except Exception as e:
                logger.error(f"Error in priority processor: {e}")
                await asyncio.sleep(1)
    
    async def _process_message(self, message_data: Dict[str, Any]):
        """Process individual message"""
        try:
            message = AgentMessage.from_dict(message_data)
            
            # Check if message has expired
            if message.expires_at and time.time() > message.expires_at:
                await self._move_to_dead_letter(message, "expired")
                return
            
            # Route message to appropriate handler
            handlers = self.message_handlers.get(message.message_type.value, [])
            
            if handlers:
                for handler in handlers:
                    try:
                        await handler(message)
                    except Exception as e:
                        logger.error(f"Handler error for message {message.id}: {e}")
            else:
                logger.warning(f"No handlers for message type: {message.message_type.value}")
                
        except Exception as e:
            logger.error(f"Error processing message: {e}")
    
    async def _health_monitor(self):
        """Monitor agent health and communication system"""
        while self.running:
            try:
                current_time = time.time()
                
                # Check for inactive agents
                for agent_id, last_seen in list(self.agent_last_seen.items()):
                    if current_time - last_seen > 120:  # 2 minutes inactive
                        logger.warning(f"Agent {agent_id} appears inactive")
                        await self._handle_inactive_agent(agent_id)
                
                # Update system metrics
                metrics = await self.get_system_metrics()
                await self.redis_client.hset(
                    self.channels['metrics'],
                    'latest',
                    json.dumps(metrics)
                )
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(30)
    
    async def _cleanup_expired_messages(self):
        """Clean up expired messages and old data"""
        while self.running:
            try:
                # Clean up expired message records
                # This would typically involve scanning keys with TTL
                # For now, we rely on Redis TTL for automatic cleanup
                
                await asyncio.sleep(300)  # Clean every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in cleanup: {e}")
                await asyncio.sleep(300)
    
    async def _move_to_dead_letter(self, message: AgentMessage, reason: str):
        """Move message to dead letter queue"""
        dead_letter_data = {
            'original_message': message.to_dict(),
            'reason': reason,
            'moved_at': time.time()
        }
        
        await self.redis_client.lpush(
            self.channels['dead_letter'],
            json.dumps(dead_letter_data)
        )
    
    async def _handle_inactive_agent(self, agent_id: str):
        """Handle inactive agent cleanup"""
        # Remove from active tracking
        self.agent_loads.pop(agent_id, None)
        
        # Mark as inactive in Redis
        agent_data = await self.redis_client.hget("agi:agents", agent_id)
        if agent_data:
            data = json.loads(agent_data)
            data['status'] = 'inactive'
            data['inactive_since'] = time.time()
            await self.redis_client.hset("agi:agents", agent_id, json.dumps(data))
    
    def register_message_handler(self, message_type: MessageType, handler: Callable):
        """Register a message handler for specific message types"""
        self.message_handlers[message_type.value].append(handler)
