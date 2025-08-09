#!/usr/bin/env python3
"""
Universal Base Agent for SutazAI System
=======================================

This is the CANONICAL base class for all AI agents in the SutazAI system.
All agent implementations should inherit from this class.

Key Features:
- Full async/await support with no threading
- Ollama integration with connection pooling and circuit breaker
- Redis-based messaging system for inter-agent communication  
- Request queue management for parallel limits
- Health check capabilities and comprehensive metrics
- Backward compatibility with existing agent patterns
- Resource-efficient for limited hardware environments
- Support for both standalone and orchestrated operation modes

Architecture:
- Async-first design with proper resource management
- Circuit breaker pattern for resilience
- Connection pooling for efficiency
- Comprehensive error handling and recovery
- Modular design supporting different operation modes
"""

import os
import sys
import json
import asyncio
import logging
import uuid
import time
from typing import Dict, Any, Optional, List, Union, Callable, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
import signal

# Optional dependencies
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None
    HTTPX_AVAILABLE = False

# Import enhanced components if available
try:
    from .ollama_pool import OllamaConnectionPool
    from .circuit_breaker import CircuitBreaker  
    from .request_queue import RequestQueue
    from .ollama_integration import OllamaConfig
    ENHANCED_COMPONENTS_AVAILABLE = True
except ImportError:
    ENHANCED_COMPONENTS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class AgentStatus(Enum):
    """Agent lifecycle status"""
    INITIALIZING = "initializing"
    REGISTERING = "registering" 
    IDLE = "idle"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"
    OFFLINE = "offline"


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
class AgentMetrics:
    """Agent performance metrics"""
    tasks_processed: int = 0
    tasks_failed: int = 0
    tasks_queued: int = 0
    total_processing_time: float = 0.0
    avg_processing_time: float = 0.0
    last_task_time: Optional[datetime] = None
    startup_time: datetime = field(default_factory=datetime.utcnow)
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    ollama_requests: int = 0
    ollama_failures: int = 0
    circuit_breaker_trips: int = 0


@dataclass
class TaskResult:
    """Standardized task result"""
    task_id: str
    status: str
    result: Dict[str, Any]
    processing_time: float
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentMessage:
    """Universal message format for inter-agent communication"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = ""
    receiver_id: str = ""  # Can be "broadcast" for all agents
    message_type: str = "request"  # request, response, notification, event
    content: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5  # 1-10, 1 is highest priority
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
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
    capabilities: List[AgentCapability] = field(default_factory=list)
    model_config: Dict[str, Any] = field(default_factory=dict)
    redis_config: Dict[str, Any] = field(default_factory=dict)
    max_concurrent_tasks: int = 3
    heartbeat_interval: int = 30
    health_check_interval: int = 30
    message_timeout: int = 300
    auto_retry: bool = True
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent:
    """
    Universal Base Agent Class
    
    This is the canonical base class for all AI agents in the SutazAI system.
    It provides a unified interface that supports both enhanced async operation
    with connection pooling and circuit breakers, as well as simple synchronous
    operation for backward compatibility.
    
    The agent can operate in two modes:
    1. Enhanced Mode: Full async with Ollama pools, circuit breakers, Redis messaging
    2. Simple Mode: Basic operation with direct HTTP calls
    
    All agents should inherit from this class and implement the abstract methods.
    """
    
    def __init__(self, 
                 agent_id: str = None,
                 name: str = None,
                 port: int = None,
                 description: str = None,
                 config_path: str = '/app/config.json',
                 max_concurrent_tasks: int = 3,
                 max_ollama_connections: int = 2,
                 health_check_interval: int = 30,
                 heartbeat_interval: int = 30,
                 config: AgentConfig = None):
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load configuration from file if path provided
        if config_path and os.path.exists(config_path):
            self.config_data = self._load_config(config_path)
        else:
            self.config_data = self._get_default_config()
        
        # Agent identification - support both constructor patterns
        self.agent_id = agent_id or os.getenv('AGENT_NAME', 'base-agent')
        self.agent_name = name or self.agent_id or os.getenv('AGENT_NAME', 'base-agent')
        self.name = self.agent_name  # Alias for compatibility
        self.agent_type = os.getenv('AGENT_TYPE', 'base')
        self.agent_version = "3.0.0"  # Consolidated version
        self.port = port or int(os.getenv("PORT", "8080"))
        self.description = description or "SutazAI Universal Agent"
        
        # Service endpoints
        self.backend_url = os.getenv('BACKEND_URL', 'http://localhost:8000')
        # Support multiple URL env vars for compatibility
        self.ollama_url = (os.getenv('OLLAMA_URL') or 
                          os.getenv('OLLAMA_BASE_URL', 'http://localhost:10104'))
        
        # Agent state
        self.status = AgentStatus.INITIALIZING
        self.metrics = AgentMetrics()
        self.shutdown_event = asyncio.Event()
        
        # Configuration parameters
        self.max_concurrent_tasks = max_concurrent_tasks
        self.health_check_interval = health_check_interval
        self.heartbeat_interval = heartbeat_interval
        
        # Model configuration
        if ENHANCED_COMPONENTS_AVAILABLE:
            try:
                self.model_config = OllamaConfig.get_model_config(self.agent_name)
                self.default_model = self.model_config["model"]
            except:
                self.model_config = {"model": "tinyllama", "temperature": 0.7}
                self.default_model = "tinyllama"
        else:
            self.model_config = {"model": "tinyllama", "temperature": 0.7}
            self.default_model = "tinyllama"
        
        # Enhanced components (initialized in setup if available)
        self.http_client = None
        self.ollama_pool = None
        self.circuit_breaker = None
        self.request_queue = None
        self.redis_client = None
        self.async_redis = None
        
        # Task tracking
        self.active_tasks = set()
        self.task_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.capabilities = set()
        
        # Message handling for inter-agent communication
        self.message_handlers: Dict[str, Callable] = {}
        self._register_default_handlers()
        
        # Background tasks
        self._background_tasks = set()
        
        self.logger.info(f"Initializing {self.agent_name} v{self.agent_version} with model {self.default_model}")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load agent configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.logger.debug(f"Loaded config from {config_path}")
                return config
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            "capabilities": [],
            "max_retries": 3,
            "timeout": 300,
            "batch_size": 10
        }
    
    async def initialize(self) -> bool:
        """Initialize the agent and all its connections"""
        try:
            self.logger.info(f"Initializing agent {self.agent_id}")
            
            # Setup async components (enhanced or basic)
            await self._setup_async_components()
            
            # Initialize Redis if enhanced components available
            if ENHANCED_COMPONENTS_AVAILABLE:
                await self._initialize_redis()
            
            # Register with coordinator if backend available
            if not await self.register_with_coordinator():
                self.logger.warning("Failed to register with coordinator, continuing anyway...")
            
            # Start background tasks
            if ENHANCED_COMPONENTS_AVAILABLE:
                asyncio.create_task(self._heartbeat_loop())
                asyncio.create_task(self._message_listener())
                asyncio.create_task(self._periodic_health_check())
            
            # Call agent-specific initialization
            await self.on_initialize()
            
            self.status = AgentStatus.IDLE
            self.logger.info(f"Agent {self.agent_id} initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agent {self.agent_id}: {e}")
            self.status = AgentStatus.ERROR
            return False
    
    async def _setup_async_components(self):
        """Initialize async components (enhanced or basic)"""
        try:
            # Setup HTTP client if httpx is available
            if HTTPX_AVAILABLE:
                timeout = httpx.Timeout(30.0)
                self.http_client = httpx.AsyncClient(timeout=timeout)
            else:
                self.http_client = None
                self.logger.warning("httpx not available - HTTP functionality disabled")
            
            if ENHANCED_COMPONENTS_AVAILABLE:
                # Setup enhanced components
                self.ollama_pool = OllamaConnectionPool(
                    base_url=self.ollama_url,
                    max_connections=2,  # Conservative for limited hardware
                    default_model=self.default_model
                )
                
                self.circuit_breaker = CircuitBreaker(
                    failure_threshold=5,
                    recovery_timeout=60,
                    expected_exception=Exception
                )
                
                self.request_queue = RequestQueue(
                    max_queue_size=100,
                    max_concurrent=self.max_concurrent_tasks,
                    timeout=300,
                    enable_background=False  # For clean teardown in tests
                )
                
                self.logger.info("Enhanced async components initialized")
            else:
                self.logger.info("Basic async components initialized")
                
        except Exception as e:
            self.logger.error(f"Failed to setup async components: {e}")
            raise
    
    async def _initialize_redis(self):
        """Initialize Redis connections for messaging (if available)"""
        try:
            import redis.asyncio as aioredis
            import redis
            
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
            
            # Synchronous Redis client
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            
            # Asynchronous Redis client  
            self.async_redis = aioredis.from_url(redis_url, decode_responses=True)
            
            # Test connection
            await self.async_redis.ping()
            self.logger.info("Redis connection established")
            
        except ImportError:
            self.logger.warning("Redis not available, messaging disabled")
        except Exception as e:
            self.logger.warning(f"Redis connection failed: {e}, messaging disabled")
    
    async def _cleanup_async_components(self):
        """Cleanup async components"""
        try:
            if self.http_client:
                await self.http_client.aclose()
                
            if self.ollama_pool:
                await self.ollama_pool.close()
                
            if self.request_queue:
                await self.request_queue.close()
                
            if self.async_redis:
                await self.async_redis.close()
                
            self.logger.info("Async components cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    async def register_with_coordinator(self) -> bool:
        """Register this agent with the coordinator"""
        if not self.http_client:
            self.logger.debug("HTTP client not available, skipping registration")
            return False
            
        try:
            self.status = AgentStatus.REGISTERING
            
            registration_data = {
                "agent_name": self.agent_name,
                "agent_type": self.agent_type,
                "agent_version": self.agent_version,
                "capabilities": [cap.value for cap in self.capabilities],
                "status": self.status.value,
                "model": self.default_model,
                "max_concurrent_tasks": self.max_concurrent_tasks,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            response = await self.http_client.post(
                f"{self.backend_url}/api/agents/register",
                json=registration_data
            )
            
            if response.status_code == 200:
                self.logger.info(f"Successfully registered {self.agent_name}")
                return True
            else:
                self.logger.error(f"Registration failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Registration error: {e}")
            return False
    
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
        while self.status != AgentStatus.OFFLINE and not self.shutdown_event.is_set():
            try:
                await self._send_heartbeat()
                await asyncio.wait_for(
                    self.shutdown_event.wait(), 
                    timeout=self.heartbeat_interval
                )
                break  # Shutdown event was set
            except asyncio.TimeoutError:
                continue  # Normal timeout, continue heartbeat loop
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(5)
    
    async def _send_heartbeat(self):
        """Send heartbeat to coordinator"""
        try:
            heartbeat_data = {
                "agent_name": self.agent_name,
                "status": self.status.value,
                "tasks_processed": self.metrics.tasks_processed,
                "tasks_failed": self.metrics.tasks_failed,
                "tasks_queued": self.metrics.tasks_queued,
                "memory_usage_mb": self.metrics.memory_usage_mb,
                "cpu_usage_percent": self.metrics.cpu_usage_percent,
                "avg_processing_time": self.metrics.avg_processing_time,
                "circuit_breaker_trips": self.metrics.circuit_breaker_trips,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Try to send to coordinator if HTTP client available
            if self.http_client:
                try:
                    response = await self.http_client.post(
                        f"{self.backend_url}/api/agents/heartbeat",
                        json=heartbeat_data
                    )
                    
                    if response.status_code == 200:
                        self.logger.debug("Heartbeat sent successfully")
                    else:
                        self.logger.warning(f"Heartbeat failed: {response.status_code}")
                except:
                    pass  # Heartbeat failure is not critical
            
            # Also store in Redis if available
            if self.async_redis:
                try:
                    await self.async_redis.setex(
                        f"heartbeat:{self.agent_id}",
                        self.heartbeat_interval * 2,
                        json.dumps(heartbeat_data)
                    )
                except:
                    pass
                    
        except Exception as e:
            self.logger.error(f"Heartbeat error: {e}")
    
    async def _message_listener(self):
        """Background task that listens for incoming messages"""
        if not self.async_redis:
            return
            
        try:
            pubsub = self.async_redis.pubsub()
            
            # Subscribe to agent-specific channel and broadcast channel
            await pubsub.subscribe(f"agent:{self.agent_id}")
            await pubsub.subscribe("agent:broadcast")
            
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
        if not self.async_redis:
            self.logger.warning("Redis not available, cannot send message")
            return ""
            
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
    
    async def query_ollama(self, 
                          prompt: str, 
                          model: Optional[str] = None,
                          system: Optional[str] = None,
                          **kwargs) -> Optional[str]:
        """
        Query Ollama using enhanced or basic method
        
        Uses connection pool and circuit breaker if available,
        otherwise falls back to direct HTTP calls
        """
        model = model or self.default_model
        
        try:
            if ENHANCED_COMPONENTS_AVAILABLE and self.ollama_pool and self.circuit_breaker:
                # Use enhanced method with connection pool and circuit breaker
                config = {**self.model_config, **kwargs}
                config.pop('model', None)  # Remove model from config to avoid duplicate
                
                response = await self.circuit_breaker.call(
                    self.ollama_pool.generate,
                    prompt=prompt,
                    model=model,
                    system=system,
                    **config
                )
                
                self.metrics.ollama_requests += 1
                return response
            else:
                # Use basic direct HTTP method
                return await self._query_ollama_direct(prompt, model, system, **kwargs)
            
        except Exception as e:
            self.metrics.ollama_failures += 1
            self.logger.error(f"Ollama query failed: {e}")
            return None
    
    async def _query_ollama_direct(self, prompt: str, model: str = None, system: str = None, **kwargs) -> Optional[str]:
        """Direct Ollama query without enhanced components"""
        if not self.http_client:
            self.logger.error("HTTP client not available for Ollama query")
            return None
            
        model = model or self.default_model
        
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "num_predict": kwargs.get("max_tokens", 2000)
                }
            }
            
            if system:
                payload["system"] = system
            
            response = await self.http_client.post(
                f"{self.ollama_url}/api/generate", 
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "")
            else:
                self.logger.error(f"Ollama query failed: {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error querying Ollama: {e}")
            return None
    
    async def query_ollama_chat(self, 
                               messages: List[Dict[str, str]], 
                               model: Optional[str] = None,
                               **kwargs) -> Optional[str]:
        """Query Ollama chat endpoint"""
        model = model or self.default_model
        
        try:
            if ENHANCED_COMPONENTS_AVAILABLE and self.ollama_pool and self.circuit_breaker:
                # Use enhanced method
                config = {**self.model_config, **kwargs}
                config.pop('model', None)
                
                response = await self.circuit_breaker.call(
                    self.ollama_pool.chat,
                    messages=messages,
                    model=model,
                    **config
                )
                
                self.metrics.ollama_requests += 1
                return response
            else:
                # Use basic direct HTTP method
                return await self._query_ollama_chat_direct(messages, model, **kwargs)
            
        except Exception as e:
            self.metrics.ollama_failures += 1
            self.logger.error(f"Ollama chat query failed: {e}")
            return None
    
    async def _query_ollama_chat_direct(self, messages: List[Dict[str, str]], model: str = None, **kwargs) -> Optional[str]:
        """Direct Ollama chat query without enhanced components"""
        if not self.http_client:
            self.logger.error("HTTP client not available for Ollama chat")
            return None
            
        model = model or self.default_model
        
        try:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "num_predict": kwargs.get("max_tokens", 2000)
                }
            }
            
            response = await self.http_client.post(
                f"{self.ollama_url}/api/chat",
                json=payload
            )
            
            if response.status_code == 200:
                result = response.json()
                message = result.get("message", {})
                return message.get("content", "")
            else:
                self.logger.error(f"Ollama chat failed: {response.text}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error querying Ollama chat: {e}")
            return None
    
    async def get_next_task(self) -> Optional[Dict[str, Any]]:
        """Fetch next task from the coordinator"""
        if not self.http_client:
            return None
            
        try:
            response = await self.http_client.get(
                f"{self.backend_url}/api/tasks/next/{self.agent_type}"
            )
            
            if response.status_code == 200:
                task = response.json()
                if task:
                    self.logger.info(f"Received task: {task.get('id', 'unknown')}")
                    self.metrics.tasks_queued += 1
                    return task
            elif response.status_code == 204:
                # No tasks available
                return None
            else:
                self.logger.warning(f"Unexpected response getting task: {response.status_code}")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting task: {e}")
            return None
    
    async def report_task_complete(self, task_result: TaskResult):
        """Report task completion to coordinator"""
        if not self.http_client:
            self.logger.debug("HTTP client not available, skipping task completion report")
            return
            
        try:
            completion_data = {
                "task_id": task_result.task_id,
                "agent_name": self.agent_name,
                "status": task_result.status,
                "result": task_result.result,
                "processing_time": task_result.processing_time,
                "error": task_result.error,
                "timestamp": task_result.timestamp.isoformat()
            }
            
            response = await self.http_client.post(
                f"{self.backend_url}/api/tasks/complete",
                json=completion_data
            )
            
            if response.status_code == 200:
                self.logger.info(f"Task {task_result.task_id} reported as {task_result.status}")
            else:
                self.logger.error(f"Failed to report task completion: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error reporting task completion: {e}")
    
    async def process_task(self, task: Dict[str, Any]) -> TaskResult:
        """
        Process a task - base implementation that calls agent-specific logic
        
        This method provides the standard task processing wrapper.
        Subclasses should override on_task_execute() for custom logic.
        """
        start_time = datetime.utcnow()
        task_id = task.get("id", "unknown")
        
        try:
            self.logger.info(f"Processing task: {task_id}")
            
            # Call agent-specific processing logic
            result = await self.on_task_execute(task_id, task)
            
            # If result is already a TaskResult, return it
            if isinstance(result, TaskResult):
                return result
            
            # Otherwise wrap in TaskResult
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            return TaskResult(
                task_id=task_id,
                status="completed",
                result=result,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            self.logger.error(f"Error processing task {task_id}: {e}")
            
            return TaskResult(
                task_id=task_id,
                status="failed",
                result={"error": str(e)},
                processing_time=processing_time,
                error=str(e)
            )
    
    async def execute_task(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task with capacity checking"""
        if len(self.active_tasks) >= self.max_concurrent_tasks:
            return {
                "success": False,
                "error": "Agent at maximum capacity",
                "task_id": task_id
            }
        
        self.active_tasks.add(task_id)
        self.status = AgentStatus.ACTIVE
        
        try:
            result = await self.on_task_execute(task_id, task_data)
            self.metrics.tasks_processed += 1
            return result
        except Exception as e:
            self.metrics.tasks_failed += 1
            self.logger.error(f"Task {task_id} failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "task_id": task_id
            }
        finally:
            self.active_tasks.discard(task_id)
            self.status = AgentStatus.IDLE if not self.active_tasks else AgentStatus.ACTIVE
    
    async def _update_metrics(self, task_result: TaskResult):
        """Update agent metrics after task completion"""
        if task_result.status == "completed":
            self.metrics.tasks_processed += 1
        else:
            self.metrics.tasks_failed += 1
        
        self.metrics.tasks_queued = max(0, self.metrics.tasks_queued - 1)
        self.metrics.total_processing_time += task_result.processing_time
        
        # Calculate average processing time
        total_tasks = self.metrics.tasks_processed + self.metrics.tasks_failed
        if total_tasks > 0:
            self.metrics.avg_processing_time = self.metrics.total_processing_time / total_tasks
        
        self.metrics.last_task_time = task_result.timestamp
        
        # Update circuit breaker trips if available
        if hasattr(self.circuit_breaker, 'trip_count'):
            self.metrics.circuit_breaker_trips = self.circuit_breaker.trip_count
    
    async def _task_wrapper(self, task: Dict[str, Any]):
        """Wrapper for task processing with semaphore and error handling"""
        task_id = task.get("id", "unknown")
        
        async with self.task_semaphore:
            self.active_tasks.add(task_id)
            self.status = AgentStatus.BUSY
            
            try:
                # Process the task
                task_result = await self.process_task(task)
                
                # Update metrics
                await self._update_metrics(task_result)
                
                # Report completion
                await self.report_task_complete(task_result)
                
            except Exception as e:
                self.logger.error(f"Task wrapper error for {task_id}: {e}")
                
                # Create error result
                error_result = TaskResult(
                    task_id=task_id,
                    status="failed",
                    result={"error": str(e)},
                    processing_time=0.0,
                    error=str(e)
                )
                
                await self._update_metrics(error_result)
                await self.report_task_complete(error_result)
                
            finally:
                self.active_tasks.discard(task_id)
                if len(self.active_tasks) == 0:
                    self.status = AgentStatus.ACTIVE
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check and return status"""
        try:
            # Check Ollama connectivity
            ollama_healthy = await self._check_ollama_health()
            
            # Check backend connectivity
            backend_healthy = await self._check_backend_health()
            
            # Calculate uptime
            uptime = (datetime.utcnow() - self.metrics.startup_time).total_seconds()
            
            health_status = {
                "agent_name": self.agent_name,
                "agent_version": self.agent_version,
                "status": self.status.value,
                "healthy": ollama_healthy and backend_healthy,
                "uptime_seconds": uptime,
                "tasks_processed": self.metrics.tasks_processed,
                "tasks_failed": self.metrics.tasks_failed,
                "active_tasks": len(self.active_tasks),
                "avg_processing_time": self.metrics.avg_processing_time,
                "ollama_healthy": ollama_healthy,
                "backend_healthy": backend_healthy,
                "circuit_breaker_status": getattr(self.circuit_breaker, 'state', {}).get('value', 'unknown') if self.circuit_breaker else "unavailable",
                "model": self.default_model,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            return health_status
            
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            return {
                "agent_name": self.agent_name,
                "healthy": False,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _check_ollama_health(self) -> bool:
        """Check Ollama health"""
        try:
            if self.ollama_pool and hasattr(self.ollama_pool, 'health_check'):
                return await self.ollama_pool.health_check()
            elif self.http_client:
                # Basic health check via direct HTTP
                response = await self.http_client.get(f"{self.ollama_url}/api/tags")
                return response.status_code == 200
            else:
                return False
        except:
            return False
    
    async def _check_backend_health(self) -> bool:
        """Check backend health"""
        if not self.http_client:
            return False
            
        try:
            response = await self.http_client.get(f"{self.backend_url}/health")
            return response.status_code == 200
        except:
            return False
    
    async def _periodic_health_check(self):
        """Periodic health check background task"""
        while not self.shutdown_event.is_set():
            try:
                health_status = await self.health_check()
                
                if not health_status.get("healthy", False):
                    self.logger.warning(f"Health check failed: {health_status}")
                    self.status = AgentStatus.ERROR
                elif self.status == AgentStatus.ERROR:
                    # Recover from error state
                    self.status = AgentStatus.ACTIVE
                    self.logger.info("Recovered from error state")
                
            except Exception as e:
                self.logger.error(f"Periodic health check error: {e}")
            
            try:
                await asyncio.wait_for(
                    self.shutdown_event.wait(), 
                    timeout=self.health_check_interval
                )
                break
            except asyncio.TimeoutError:
                continue
    
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
            "uptime": (datetime.utcnow() - self.metrics.startup_time).total_seconds(),
            "error_count": self.metrics.tasks_failed,
            "task_count": self.metrics.tasks_processed
        }
        
        await self.send_message(message.sender_id, "status_response", status_info)
    
    async def _handle_capabilities_request(self, message: AgentMessage):
        """Handle capabilities request"""
        capabilities_info = {
            "agent_id": self.agent_id,
            "capabilities": [cap.value for cap in self.capabilities],
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "model_name": self.default_model
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
            "last_heartbeat": datetime.utcnow().isoformat(),
            "error_count": self.metrics.tasks_failed,
            "active_tasks": len(self.active_tasks)
        }
        
        await self.send_message(message.sender_id, "health_response", health_info)
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""
        def signal_handler():
            self.logger.info("Received shutdown signal")
            self.shutdown_event.set()
        
        if sys.platform != "win32":
            loop = asyncio.get_event_loop()
            for sig in [signal.SIGTERM, signal.SIGINT]:
                loop.add_signal_handler(sig, signal_handler)
    
    async def run_async(self):
        """Main async run loop for the agent"""
        try:
            # Initialize the agent
            if not await self.initialize():
                raise RuntimeError("Failed to initialize agent")
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            self.status = AgentStatus.ACTIVE
            self.logger.info(f"{self.agent_name} is running and waiting for tasks...")
            
            # Main task processing loop
            while not self.shutdown_event.is_set():
                try:
                    # Get next task
                    task = await self.get_next_task()
                    
                    if task:
                        # Process task asynchronously
                        task_coroutine = self._task_wrapper(task)
                        asyncio.create_task(task_coroutine)
                    else:
                        # No task available, wait a bit
                        try:
                            await asyncio.wait_for(
                                self.shutdown_event.wait(), 
                                timeout=5.0
                            )
                        except asyncio.TimeoutError:
                            continue
                        
                except Exception as e:
                    self.logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(10)
            
        except Exception as e:
            self.logger.error(f"Fatal error in run_async: {e}")
            self.status = AgentStatus.ERROR
            raise
            
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Gracefully shutdown the agent"""
        self.logger.info(f"Shutting down agent {self.agent_id}")
        self.status = AgentStatus.SHUTTING_DOWN
        
        # Cancel active tasks
        for task_id in self.active_tasks.copy():
            self.active_tasks.discard(task_id)
        
        # Call agent-specific shutdown
        await self.on_shutdown()
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for active tasks to complete (with timeout)
        if self.active_tasks:
            self.logger.info(f"Waiting for {len(self.active_tasks)} active tasks to complete...")
            
            max_wait = 30  # seconds
            start_time = asyncio.get_event_loop().time()
            
            while self.active_tasks and (asyncio.get_event_loop().time() - start_time) < max_wait:
                await asyncio.sleep(1)
            
            if self.active_tasks:
                self.logger.warning(f"Shutting down with {len(self.active_tasks)} tasks still active")
        
        # Remove from registry if Redis available
        if self.async_redis:
            try:
                await self.async_redis.hdel("agent_registry", self.agent_id)
                await self.async_redis.delete(f"heartbeat:{self.agent_id}")
            except:
                pass
        
        # Cleanup async components
        await self._cleanup_async_components()
        
        self.status = AgentStatus.STOPPED
        self.logger.info(f"{self.agent_name} shutdown complete")
    
    def run(self):
        """Main entry point - run the agent"""
        try:
            asyncio.run(self.run_async())
        except KeyboardInterrupt:
            self.logger.info("Agent stopped by user")
        except Exception as e:
            self.logger.error(f"Agent run error: {e}")
            raise
    
    def start(self):
        """Compatibility method for existing agents"""
        if asyncio.iscoroutinefunction(self.on_initialize):
            # Async start
            self.run()
        else:
            # Simple synchronous start for basic agents
            self.status = AgentStatus.ACTIVE
            self.logger.info(f"Agent {self.name} started in standalone mode")
            
            try:
                asyncio.run(self._standalone_loop())
            except KeyboardInterrupt:
                self.logger.info("Agent stopped by user")
            except Exception as e:
                self.logger.error(f"Agent error: {e}")
    
    async def _standalone_loop(self):
        """Simple standalone event loop"""
        self.logger.info(f"Agent {self.name} running on port {self.port}")
        
        # In standalone mode, just keep the agent alive
        while not self.shutdown_event.is_set():
            await asyncio.sleep(10)
            self.logger.debug(f"Agent {self.name} heartbeat - processed {self.metrics.tasks_processed} tasks")
    
    # Backward compatibility methods
    def query_ollama_sync(self, prompt: str, model: str = None) -> Optional[str]:
        """
        Backward compatibility method for synchronous Ollama queries
        
        This is provided for compatibility with existing agents that may
        still use synchronous calls. New agents should use query_ollama().
        """
        self.logger.warning("Using deprecated sync method. Please use async query_ollama() instead.")
        
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, we can't use run()
                self.logger.error("Cannot use sync method from async context")
                return None
            else:
                return loop.run_until_complete(self.query_ollama(prompt, model))
        except Exception as e:
            self.logger.error(f"Sync Ollama query error: {e}")
            return None
    
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
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "uptime": (datetime.utcnow() - self.metrics.startup_time).total_seconds(),
            "error_count": self.metrics.tasks_failed,
            "task_count": self.metrics.tasks_processed,
            "model_name": self.default_model,
            "last_heartbeat": datetime.utcnow().isoformat(),
            "startup_time": self.metrics.startup_time.isoformat()
        }
    
    # Abstract methods for subclasses to implement
    async def on_initialize(self):
        """Called after basic initialization is complete"""
        pass
    
    async def on_task_execute(self, task_id: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent-specific task logic
        
        Subclasses should override this method to implement their specific
        task processing logic. The base implementation provides a simple echo.
        
        Args:
            task_id: Unique identifier for the task
            task_data: Task data and parameters
            
        Returns:
            Dict containing the task result
        """
        # Default implementation - simple echo
        await asyncio.sleep(0.1)  # Simulate processing
        
        return {
            "status": "success",
            "message": f"Task processed by {self.agent_name} v{self.agent_version}",
            "task_id": task_id,
            "agent_name": self.agent_name,
            "model_used": self.default_model,
            "processed_at": datetime.utcnow().isoformat(),
            "task_data": task_data
        }
    
    async def on_message_received(self, message: AgentMessage):
        """Handle unknown message types"""
        self.logger.debug(f"Received unhandled message type: {message.message_type}")
        pass
    
    async def on_shutdown(self):
        """Perform agent-specific cleanup"""
        pass


# Backward compatibility aliases
BaseAgentV2 = BaseAgent


if __name__ == "__main__":
    # Simple test when run directly
    agent = BaseAgent()
    agent.run()