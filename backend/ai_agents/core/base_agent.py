"""
Universal Base Agent - Core Foundation for All AI Agents
========================================================

This module provides the base class for all AI agents in the SutazAI system.
It's completely independent from Claude and works with local Ollama models,
Redis messaging, and provides a universal interface for all agent types.

Features:
- Local Ollama model integration
- Redis-based messaging
- Universal agent capabilities
- Self-contained operation
- Extensible architecture
"""

import asyncio
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Callable
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager

import redis
import aioredis
import httpx
from pydantic import BaseModel, Field


class AgentStatus(Enum):
    """Agent operational status"""
    INITIALIZING = "initializing"
    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


class AgentCapability(Enum):
    """Standard agent capabilities"""
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    SECURITY_ANALYSIS = "security_analysis"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    COMMUNICATION = "communication"
    REASONING = "reasoning"
    LEARNING = "learning"
    ORCHESTRATION = "orchestration"
    DATA_PROCESSING = "data_processing"
    FILE_OPERATIONS = "file_operations"
    API_INTEGRATION = "api_integration"
    AUTONOMOUS_EXECUTION = "autonomous_execution"


@dataclass
class AgentMessage:
    """Universal message format for inter-agent communication"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: str = ""  # Can be "broadcast" for all agents
    message_type: str = "request"  # request, response, notification, event
    content: Dict[str, Any] = Field(default_factory=dict)
    priority: int = 5  # 1-10, 1 is highest priority
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type,
            "content": self.content,
            "priority": self.priority,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        data = data.copy()
        if "created_at" in data:
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("expires_at"):
            data["expires_at"] = datetime.fromisoformat(data["expires_at"])
        return cls(**data)


@dataclass
class AgentConfig:
    """Universal agent configuration"""
    agent_id: str
    agent_type: str
    name: str
    description: str
    capabilities: List[AgentCapability]
    model_config: Dict[str, Any]
    redis_config: Dict[str, Any]
    max_concurrent_tasks: int = 5
    heartbeat_interval: int = 30
    message_timeout: int = 300
    auto_retry: bool = True
    max_retries: int = 3
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BaseAgent(ABC):
    """
    Universal Base Agent Class
    
    This is the foundation for all AI agents in the SutazAI system.
    All agents inherit from this class and gain universal capabilities.
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.agent_id = config.agent_id
        self.agent_type = config.agent_type
        self.name = config.name
        self.status = AgentStatus.INITIALIZING
        self.capabilities = set(config.capabilities)
        
        # Redis connection for messaging
        self.redis_client: Optional[redis.Redis] = None
        self.async_redis: Optional[aioredis.Redis] = None
        
        # Ollama client for local AI models
        self.ollama_client: Optional[httpx.AsyncClient] = None
        self.ollama_base_url = config.model_config.get("ollama_url", "http://localhost:10104")
        self.model_name = config.model_config.get("model", "tinyllama")
        
        # Internal state
        self.active_tasks: Set[str] = set()
        self.message_handlers: Dict[str, Callable] = {}
        self.startup_time = datetime.utcnow()
        self.last_heartbeat = datetime.utcnow()
        self.error_count = 0
        self.task_count = 0
        
        # Logging
        self.logger = logging.getLogger(f"agent.{self.agent_id}")
        
        # Initialize default message handlers
        self._register_default_handlers()
    
    async def initialize(self) -> bool:
        """Initialize the agent and all its connections"""
        try:
            self.logger.info(f"Initializing agent {self.agent_id}")
            
            # Initialize Redis connection
            await self._initialize_redis()
            
            # Initialize Ollama client
            await self._initialize_ollama()
            
            # Register with agent registry
            await self._register_with_registry()
            
            # Start background tasks
            asyncio.create_task(self._heartbeat_loop())
            asyncio.create_task(self._message_listener())
            
            # Call agent-specific initialization
            await self.on_initialize()
            
            self.status = AgentStatus.IDLE
            self.logger.info(f"Agent {self.agent_id} initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agent {self.agent_id}: {e}")
            self.status = AgentStatus.ERROR
            return False
    
    async def _initialize_redis(self):
        """Initialize Redis connections for messaging"""
        redis_config = self.config.redis_config
        redis_url = redis_config.get("url", "redis://localhost:6379")
        
        # Synchronous Redis client
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        
        # Asynchronous Redis client
        self.async_redis = aioredis.from_url(redis_url, decode_responses=True)
        
        # Test connection
        await self.async_redis.ping()
        self.logger.info("Redis connection established")
    
    async def _initialize_ollama(self):
        """Initialize Ollama client for local AI models"""
        self.ollama_client = httpx.AsyncClient(
            base_url=self.ollama_base_url,
            timeout=300.0
        )
        
        # Verify model is available
        try:
            response = await self.ollama_client.get("/api/tags")
            models = [model["name"] for model in response.json().get("models", [])]
            
            if self.model_name not in models:
                self.logger.warning(f"Model {self.model_name} not found. Available: {models}")
                # Try to pull the model
                await self._pull_model()
            
            self.logger.info(f"Ollama client initialized with model: {self.model_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Ollama client: {e}")
            raise
    
    async def _pull_model(self):
        """Pull the required model from Ollama"""
        try:
            self.logger.info(f"Pulling model: {self.model_name}")
            response = await self.ollama_client.post(
                "/api/pull",
                json={"name": self.model_name}
            )
            if response.status_code == 200:
                self.logger.info(f"Model {self.model_name} pulled successfully")
            else:
                self.logger.error(f"Failed to pull model: {response.text}")
        except Exception as e:
            self.logger.error(f"Error pulling model: {e}")
    
    async def _register_with_registry(self):
        """Register this agent with the global agent registry"""
        agent_info = {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "name": self.name,
            "status": self.status.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "startup_time": self.startup_time.isoformat(),
            "model_name": self.model_name,
            "max_concurrent_tasks": self.config.max_concurrent_tasks
        }
        
        await self.async_redis.hset(
            "agent_registry",
            self.agent_id,
            json.dumps(agent_info)
        )
        
        self.logger.info(f"Agent {self.agent_id} registered in registry")
    
    def _register_default_handlers(self):
        """Register default message handlers"""
        self.message_handlers.update({
            "ping": self._handle_ping,
            "status": self._handle_status_request,
            "capabilities": self._handle_capabilities_request,
            "shutdown": self._handle_shutdown,
            "task_cancel": self._handle_task_cancel,
            "health_check": self._handle_health_check
        })
    
    async def _heartbeat_loop(self):
        """Background task that sends regular heartbeats"""
        while self.status != AgentStatus.OFFLINE:
            try:
                await self._send_heartbeat()
                await asyncio.sleep(self.config.heartbeat_interval)
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5)
    
    async def _send_heartbeat(self):
        """Send heartbeat to indicate agent is alive"""
        self.last_heartbeat = datetime.utcnow()
        heartbeat_data = {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "active_tasks": len(self.active_tasks),
            "error_count": self.error_count,
            "task_count": self.task_count,
            "timestamp": self.last_heartbeat.isoformat()
        }
        
        await self.async_redis.setex(
            f"heartbeat:{self.agent_id}",
            self.config.heartbeat_interval * 2,  # TTL is 2x heartbeat interval
            json.dumps(heartbeat_data)
        )
    
    async def _message_listener(self):
        """Background task that listens for incoming messages"""
        pubsub = self.async_redis.pubsub()
        
        # Subscribe to agent-specific channel and broadcast channel
        await pubsub.subscribe(f"agent:{self.agent_id}")
        await pubsub.subscribe("agent:broadcast")
        
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        await self._process_message(message["data"])
                    except Exception as e:
                        self.logger.error(f"Error processing message: {e}")
        except Exception as e:
            self.logger.error(f"Message listener error: {e}")
    
    async def _process_message(self, message_data: str):
        """Process incoming message"""
        try:
            message = AgentMessage.from_dict(json.loads(message_data))
            
            # Check if message is expired
            if message.expires_at and datetime.utcnow() > message.expires_at:
                self.logger.debug(f"Ignoring expired message {message.id}")
                return
            
            # Check if we have a handler for this message type
            handler = self.message_handlers.get(message.message_type)
            if handler:
                await handler(message)
            else:
                # Let subclass handle unknown message types
                await self.on_message_received(message)
                
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
    
    async def send_message(self, receiver_id: str, message_type: str, 
                          content: Dict[str, Any], priority: int = 5) -> str:
        """Send a message to another agent or broadcast"""
        message = AgentMessage(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            priority=priority
        )
        
        # Determine channel
        if receiver_id == "broadcast":
            channel = "agent:broadcast"
        else:
            channel = f"agent:{receiver_id}"
        
        await self.async_redis.publish(channel, json.dumps(message.to_dict()))
        self.logger.debug(f"Sent message {message.id} to {receiver_id}")
        return message.id
    
    async def query_model(self, prompt: str, system_prompt: str = None, 
                         temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """Query the local Ollama model"""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            }
            
            response = await self.ollama_client.post("/api/generate", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                self.logger.error(f"Ollama query failed: {response.text}")
                return ""
                
        except Exception as e:
            self.logger.error(f"Error querying model: {e}")
            return ""
    
    async def execute_task(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task (to be implemented by subclasses)"""
        if len(self.active_tasks) >= self.config.max_concurrent_tasks:
            return {
                "success": False,
                "error": "Agent at maximum capacity",
                "task_id": task_id
            }
        
        self.active_tasks.add(task_id)
        self.status = AgentStatus.ACTIVE
        
        try:
            # Call agent-specific task execution
            result = await self.on_task_execute(task_id, task_data)
            self.task_count += 1
            return result
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"Task {task_id} failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id
            }
        finally:
            self.active_tasks.discard(task_id)
            self.status = AgentStatus.IDLE if not self.active_tasks else AgentStatus.ACTIVE
    
    # Message handlers
    async def _handle_ping(self, message: AgentMessage):
        """Handle ping request"""
        await self.send_message(
            message.sender_id,
            "pong",
            {"timestamp": datetime.utcnow().isoformat()}
        )
    
    async def _handle_status_request(self, message: AgentMessage):
        """Handle status request"""
        status_info = {
            "agent_id": self.agent_id,
            "status": self.status.value,
            "active_tasks": len(self.active_tasks),
            "capabilities": [cap.value for cap in self.capabilities],
            "uptime": (datetime.utcnow() - self.startup_time).total_seconds(),
            "error_count": self.error_count,
            "task_count": self.task_count
        }
        
        await self.send_message(message.sender_id, "status_response", status_info)
    
    async def _handle_capabilities_request(self, message: AgentMessage):
        """Handle capabilities request"""
        capabilities_info = {
            "agent_id": self.agent_id,
            "capabilities": [cap.value for cap in self.capabilities],
            "max_concurrent_tasks": self.config.max_concurrent_tasks,
            "model_name": self.model_name
        }
        
        await self.send_message(message.sender_id, "capabilities_response", capabilities_info)
    
    async def _handle_shutdown(self, message: AgentMessage):
        """Handle shutdown request"""
        self.logger.info(f"Shutdown requested by {message.sender_id}")
        await self.shutdown()
    
    async def _handle_task_cancel(self, message: AgentMessage):
        """Handle task cancellation request"""
        task_id = message.content.get("task_id")
        if task_id in self.active_tasks:
            self.active_tasks.discard(task_id)
            await self.send_message(
                message.sender_id,
                "task_cancelled",
                {"task_id": task_id, "success": True}
            )
        else:
            await self.send_message(
                message.sender_id,
                "task_cancelled",
                {"task_id": task_id, "success": False, "error": "Task not found"}
            )
    
    async def _handle_health_check(self, message: AgentMessage):
        """Handle health check request"""
        health_info = {
            "agent_id": self.agent_id,
            "healthy": self.status not in [AgentStatus.ERROR, AgentStatus.OFFLINE],
            "status": self.status.value,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "error_count": self.error_count,
            "active_tasks": len(self.active_tasks)
        }
        
        await self.send_message(message.sender_id, "health_response", health_info)
    
    async def shutdown(self):
        """Gracefully shutdown the agent"""
        self.logger.info(f"Shutting down agent {self.agent_id}")
        self.status = AgentStatus.OFFLINE
        
        # Cancel active tasks
        for task_id in self.active_tasks.copy():
            self.active_tasks.discard(task_id)
        
        # Call agent-specific shutdown
        await self.on_shutdown()
        
        # Remove from registry
        await self.async_redis.hdel("agent_registry", self.agent_id)
        await self.async_redis.delete(f"heartbeat:{self.agent_id}")
        
        # Close connections
        if self.ollama_client:
            await self.ollama_client.aclose()
        if self.async_redis:
            await self.async_redis.close()
    
    # Abstract methods for subclasses to implement
    @abstractmethod
    async def on_initialize(self):
        """Called after basic initialization is complete"""
        pass
    
    @abstractmethod
    async def on_task_execute(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent-specific task logic"""
        pass
    
    @abstractmethod
    async def on_message_received(self, message: AgentMessage):
        """Handle unknown message types"""
        pass
    
    @abstractmethod
    async def on_shutdown(self):
        """Perform agent-specific cleanup"""
        pass
    
    # Utility methods
    def add_capability(self, capability: AgentCapability):
        """Add a capability to this agent"""
        self.capabilities.add(capability)
    
    def remove_capability(self, capability: AgentCapability):
        """Remove a capability from this agent"""
        self.capabilities.discard(capability)
    
    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if agent has a specific capability"""
        return capability in self.capabilities
    
    def register_message_handler(self, message_type: str, handler: Callable):
        """Register a custom message handler"""
        self.message_handlers[message_type] = handler
    
    async def get_agent_info(self) -> Dict[str, Any]:
        """Get comprehensive agent information"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type,
            "name": self.name,
            "status": self.status.value,
            "capabilities": [cap.value for cap in self.capabilities],
            "active_tasks": len(self.active_tasks),
            "max_concurrent_tasks": self.config.max_concurrent_tasks,
            "uptime": (datetime.utcnow() - self.startup_time).total_seconds(),
            "error_count": self.error_count,
            "task_count": self.task_count,
            "model_name": self.model_name,
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "startup_time": self.startup_time.isoformat()
        }