#!/usr/bin/env python3
"""
Enhanced Base Agent for SutazAI System v2.0
Provides optimized async core functionality for all 131 agent types with Ollama integration

Key Features:
- Full async/await support with no threading
- Integrated Ollama connection pooling and circuit breaker
- Request queue management for parallel limits
- Health check capabilities and comprehensive metrics
- Backward compatible with existing agents
- Resource-efficient for limited hardware
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import httpx
import signal
from contextlib import asynccontextmanager

# Import our enhanced components
from .ollama_pool import OllamaConnectionPool
from .circuit_breaker import CircuitBreaker
from .request_queue import RequestQueue
from .ollama_integration import OllamaConfig

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class AgentStatus(Enum):
    """Agent lifecycle status"""
    INITIALIZING = "initializing"
    REGISTERING = "registering"
    ACTIVE = "active"
    BUSY = "busy"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


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


class BaseAgentV2:
    """
    Enhanced base class for all SutazAI agents with async Ollama integration
    
    This class provides:
    - Async-first architecture with proper connection pooling
    - Circuit breaker pattern for resilience
    - Request queue management for resource limits
    - Comprehensive metrics and health monitoring
    - Backward compatibility with existing agent patterns
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
                 heartbeat_interval: int = 30):
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config = self._load_config(config_path)
        
        # Agent identification - support both constructor patterns
        self.agent_id = agent_id or os.getenv('AGENT_NAME', 'base-agent-v2')
        self.agent_name = name or self.agent_id or os.getenv('AGENT_NAME', 'base-agent-v2')
        self.name = self.agent_name  # Alias for compatibility
        self.agent_type = os.getenv('AGENT_TYPE', 'base')
        self.agent_version = "2.0.0"
        self.port = port or int(os.getenv("PORT", "8080"))
        self.description = description or "SutazAI Agent"
        
        # Service endpoints
        self.backend_url = os.getenv('BACKEND_URL', 'http://localhost:8000')
        self.ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:10104')
        
        # Agent state
        self.status = AgentStatus.INITIALIZING
        self.metrics = AgentMetrics()
        self.shutdown_event = asyncio.Event()
        
        # Configuration
        self.max_concurrent_tasks = max_concurrent_tasks
        self.health_check_interval = health_check_interval
        self.heartbeat_interval = heartbeat_interval
        
        # Get model configuration for this agent
        self.model_config = OllamaConfig.get_model_config(self.agent_name)
        self.default_model = self.model_config["model"]
        
        # Async components (initialized in setup)
        self.http_client = None
        self.ollama_pool = None
        self.circuit_breaker = None
        self.request_queue = None
        
        # Task tracking
        self.active_tasks = set()
        self.task_semaphore = asyncio.Semaphore(max_concurrent_tasks)
        
        # Background tasks
        self._background_tasks = set()
        
        self.logger.info(f"Initializing {self.agent_name} v{self.agent_version} with model {self.default_model}")
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load agent configuration from JSON file"""
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    self.logger.debug(f"Loaded config from {config_path}")
                    return config
            except Exception as e:
                self.logger.error(f"Failed to load config from {config_path}: {e}")
        
        # Return default config
        return {
            "capabilities": [],
            "max_retries": 3,
            "timeout": 300,
            "batch_size": 10
        }
    
    async def _setup_async_components(self):
        """Initialize async components"""
        try:
            # HTTP client for API calls
            timeout = httpx.Timeout(30.0)
            self.http_client = httpx.AsyncClient(timeout=timeout)
            
            # Ollama connection pool
            self.ollama_pool = OllamaConnectionPool(
                base_url=self.ollama_url,
                max_connections=2,  # Conservative for limited hardware
                default_model=self.default_model
            )
            
            # Circuit breaker for resilience
            self.circuit_breaker = CircuitBreaker(
                failure_threshold=5,
                recovery_timeout=60,
                expected_exception=Exception
            )
            
            # Request queue for managing concurrent requests
            self.request_queue = RequestQueue(
                max_queue_size=100,
                max_concurrent=self.max_concurrent_tasks,
                timeout=300
            )
            
            self.logger.info("Async components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to setup async components: {e}")
            raise
    
    async def _cleanup_async_components(self):
        """Cleanup async components"""
        try:
            if self.http_client:
                await self.http_client.aclose()
            
            if self.ollama_pool:
                await self.ollama_pool.close()
            
            if self.request_queue:
                await self.request_queue.close()
                
            self.logger.info("Async components cleaned up")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    async def register_with_coordinator(self) -> bool:
        """Register this agent with the coordinator"""
        try:
            self.status = AgentStatus.REGISTERING
            
            registration_data = {
                "agent_name": self.agent_name,
                "agent_type": self.agent_type,
                "agent_version": self.agent_version,
                "capabilities": self.config.get("capabilities", []),
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
    
    async def send_heartbeat(self):
        """Send periodic heartbeat to coordinator"""
        while not self.shutdown_event.is_set():
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
                
                response = await self.http_client.post(
                    f"{self.backend_url}/api/agents/heartbeat",
                    json=heartbeat_data
                )
                
                if response.status_code == 200:
                    self.logger.debug("Heartbeat sent successfully")
                else:
                    self.logger.warning(f"Heartbeat failed: {response.status_code}")
                    
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
            
            try:
                await asyncio.wait_for(
                    self.shutdown_event.wait(), 
                    timeout=self.heartbeat_interval
                )
                break  # Shutdown event was set
            except asyncio.TimeoutError:
                continue  # Normal timeout, continue heartbeat loop
    
    async def get_next_task(self) -> Optional[Dict[str, Any]]:
        """Fetch next task from the coordinator"""
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
        Process a task - to be overridden by specific agents
        
        This is the main method that specific agents should override.
        The base implementation provides a simple echo response.
        """
        start_time = datetime.utcnow()
        task_id = task.get("id", "unknown")
        
        try:
            self.logger.info(f"Processing task: {task_id}")
            
            # Simulate some processing
            await asyncio.sleep(0.1)
            
            # Base implementation - just echo the task
            result = {
                "status": "success",
                "message": f"Task processed by {self.agent_name} v{self.agent_version}",
                "task_id": task_id,
                "agent_name": self.agent_name,
                "model_used": self.default_model,
                "processed_at": datetime.utcnow().isoformat()
            }
            
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
    
    async def query_ollama(self, 
                          prompt: str, 
                          model: Optional[str] = None,
                          system: Optional[str] = None,
                          **kwargs) -> Optional[str]:
        """
        Query Ollama using connection pool and circuit breaker
        
        This method uses the enhanced Ollama integration with:
        - Connection pooling for efficiency
        - Circuit breaker for resilience
        - Proper async handling
        """
        model = model or self.default_model
        
        try:
            # Use circuit breaker to protect against Ollama failures
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
            
        except Exception as e:
            self.metrics.ollama_failures += 1
            self.logger.error(f"Ollama query failed: {e}")
            return None
    
    async def query_ollama_chat(self, 
                               messages: List[Dict[str, str]], 
                               model: Optional[str] = None,
                               **kwargs) -> Optional[str]:
        """Query Ollama chat endpoint using connection pool"""
        model = model or self.default_model
        
        try:
            config = {**self.model_config, **kwargs}
            config.pop('model', None)  # Remove model from config to avoid duplicate
            response = await self.circuit_breaker.call(
                self.ollama_pool.chat,
                messages=messages,
                model=model,
                **config
            )
            
            self.metrics.ollama_requests += 1
            return response
            
        except Exception as e:
            self.metrics.ollama_failures += 1
            self.logger.error(f"Ollama chat query failed: {e}")
            return None
    
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
        
        # Update circuit breaker trips
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
            ollama_healthy = await self.ollama_pool.health_check()
            
            # Check backend connectivity
            backend_healthy = False
            try:
                response = await self.http_client.get(f"{self.backend_url}/health")
                backend_healthy = response.status_code == 200
            except:
                pass
            
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
                "circuit_breaker_status": self.circuit_breaker.state.value if self.circuit_breaker else "unknown",
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
            # Setup async components
            await self._setup_async_components()
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            # Register with coordinator
            if not await self.register_with_coordinator():
                self.logger.warning("Failed to register with coordinator, continuing anyway...")
            
            self.status = AgentStatus.ACTIVE
            
            # Start background tasks
            heartbeat_task = asyncio.create_task(self.send_heartbeat())
            health_task = asyncio.create_task(self._periodic_health_check())
            
            self._background_tasks.add(heartbeat_task)
            self._background_tasks.add(health_task)
            
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
            self.status = AgentStatus.SHUTTING_DOWN
            
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
        self.run()
    
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


# Backward compatibility alias
BaseAgent = BaseAgentV2


if __name__ == "__main__":
    agent = BaseAgentV2()
    agent.run()