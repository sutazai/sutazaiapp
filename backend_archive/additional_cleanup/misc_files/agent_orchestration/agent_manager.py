#!/usr/bin/env python3
"""
Enhanced Agent Manager - Advanced agent lifecycle management
"""

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Callable, Union
from enum import Enum
import json
import psutil
import torch

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Agent status enumeration"""
    INITIALIZING = "initializing"
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    TERMINATING = "terminating"
    TERMINATED = "terminated"

class AgentType(Enum):
    """Agent type enumeration"""
    BASIC = "basic"
    REASONING = "reasoning"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"
    COLLABORATIVE = "collaborative"
    SPECIALIST = "specialist"
    AUTONOMOUS = "autonomous"

@dataclass
class AgentConfig:
    """Agent configuration"""
    agent_type: AgentType = AgentType.BASIC
    max_memory_mb: int = 1024
    max_cpu_percent: float = 25.0
    timeout_seconds: int = 300
    enable_learning: bool = True
    enable_memory: bool = True
    enable_reasoning: bool = True
    model_name: str = "default"
    capabilities: List[str] = field(default_factory=list)
    restrictions: List[str] = field(default_factory=list)
    context_window: int = 4096
    temperature: float = 0.7
    max_tokens: int = 1000

@dataclass
class AgentManagerConfig:
    """Agent manager configuration"""
    max_agents: int = 100
    max_concurrent_agents: int = 50
    agent_pool_size: int = 10
    resource_monitoring_interval: float = 5.0
    health_check_interval: float = 30.0
    enable_agent_pooling: bool = True
    enable_load_balancing: bool = True
    enable_resource_monitoring: bool = True
    enable_auto_scaling: bool = True
    default_agent_config: AgentConfig = field(default_factory=AgentConfig)

@dataclass
class AgentMetrics:
    """Agent performance metrics"""
    agent_id: str
    start_time: datetime
    last_activity: datetime
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    average_response_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    error_rate: float = 0.0
    uptime: float = 0.0

class AgentInstance:
    """Individual agent instance"""
    
    def __init__(self, agent_id: str, config: AgentConfig):
        self.agent_id = agent_id
        self.config = config
        self.status = AgentStatus.INITIALIZING
        self.created_at = datetime.now(timezone.utc)
        self.last_activity = self.created_at
        
        # Agent state
        self.current_task = None
        self.task_queue = asyncio.Queue()
        self.context = {}
        self.memory = {}
        
        # Performance metrics
        self.metrics = AgentMetrics(
            agent_id=agent_id,
            start_time=self.created_at,
            last_activity=self.created_at
        )
        
        # Threading
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # Agent components
        self.model = None
        self.reasoning_engine = None
        self.memory_system = None
        
        logger.info(f"Agent instance created: {agent_id}")
    
    async def initialize(self) -> bool:
        """Initialize agent instance"""
        try:
            with self._lock:
                if self.status != AgentStatus.INITIALIZING:
                    return True
                
                # Initialize model
                # This would integrate with the model manager
                self.model = f"model_{self.config.model_name}"
                
                # Initialize reasoning engine if enabled
                if self.config.enable_reasoning:
                    self.reasoning_engine = "reasoning_engine"
                
                # Initialize memory system if enabled
                if self.config.enable_memory:
                    self.memory_system = "memory_system"
                
                self.status = AgentStatus.IDLE
                logger.info(f"Agent initialized: {self.agent_id}")
                return True
                
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            self.status = AgentStatus.ERROR
            return False
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task"""
        try:
            with self._lock:
                if self.status != AgentStatus.IDLE:
                    raise RuntimeError(f"Agent not ready: {self.status}")
                
                self.status = AgentStatus.RUNNING
                self.current_task = task
                self.last_activity = datetime.now(timezone.utc)
                
                # Task execution logic would go here
                # This is a simplified implementation
                result = {
                    "task_id": task.get("task_id", str(uuid.uuid4())),
                    "status": "completed",
                    "result": f"Task executed by agent {self.agent_id}",
                    "execution_time": 0.1,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                # Update metrics
                self.metrics.total_tasks += 1
                self.metrics.successful_tasks += 1
                self.metrics.last_activity = datetime.now(timezone.utc)
                
                self.status = AgentStatus.IDLE
                self.current_task = None
                
                return result
                
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            self.metrics.failed_tasks += 1
            self.status = AgentStatus.ERROR
            raise
    
    async def pause(self) -> bool:
        """Pause agent"""
        try:
            with self._lock:
                if self.status == AgentStatus.RUNNING:
                    self.status = AgentStatus.PAUSED
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to pause agent: {e}")
            return False
    
    async def resume(self) -> bool:
        """Resume agent"""
        try:
            with self._lock:
                if self.status == AgentStatus.PAUSED:
                    self.status = AgentStatus.IDLE
                    return True
                return False
        except Exception as e:
            logger.error(f"Failed to resume agent: {e}")
            return False
    
    async def terminate(self) -> bool:
        """Terminate agent"""
        try:
            with self._lock:
                self.status = AgentStatus.TERMINATING
                
                # Cleanup resources
                self.model = None
                self.reasoning_engine = None
                self.memory_system = None
                
                # Signal shutdown
                self._shutdown_event.set()
                
                self.status = AgentStatus.TERMINATED
                logger.info(f"Agent terminated: {self.agent_id}")
                return True
                
        except Exception as e:
            logger.error(f"Agent termination failed: {e}")
            self.status = AgentStatus.ERROR
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        with self._lock:
            return {
                "agent_id": self.agent_id,
                "status": self.status.value,
                "agent_type": self.config.agent_type.value,
                "created_at": self.created_at.isoformat(),
                "last_activity": self.last_activity.isoformat(),
                "current_task": self.current_task,
                "uptime": (datetime.now(timezone.utc) - self.created_at).total_seconds(),
                "metrics": {
                    "total_tasks": self.metrics.total_tasks,
                    "successful_tasks": self.metrics.successful_tasks,
                    "failed_tasks": self.metrics.failed_tasks,
                    "error_rate": self.metrics.error_rate,
                    "cpu_usage": self.metrics.cpu_usage,
                    "memory_usage": self.metrics.memory_usage
                }
            }
    
    def health_check(self) -> bool:
        """Check agent health"""
        try:
            with self._lock:
                return (
                    self.status not in [AgentStatus.ERROR, AgentStatus.TERMINATED] and
                    self.model is not None
                )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

class EnhancedAgentManager:
    """Enhanced agent manager with advanced features"""
    
    def __init__(self, config: AgentManagerConfig = None, registry=None, 
                 resource_manager=None, communication_system=None):
        self.config = config or AgentManagerConfig()
        self.registry = registry
        self.resource_manager = resource_manager
        self.communication_system = communication_system
        
        # Agent management
        self.agents: Dict[str, AgentInstance] = {}
        self.agent_pool: List[str] = []
        self.active_agents: Dict[str, AgentInstance] = {}
        
        # Load balancing
        self.load_balancer = None
        self.task_queue = asyncio.Queue()
        
        # Performance monitoring
        self.system_metrics = {
            "total_agents": 0,
            "active_agents": 0,
            "idle_agents": 0,
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "average_response_time": 0.0,
            "cpu_usage": 0.0,
            "memory_usage": 0.0
        }
        
        # Threading
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # Background tasks
        self._monitoring_task = None
        self._health_check_task = None
        
        logger.info("Enhanced agent manager initialized")
    
    async def initialize(self) -> bool:
        """Initialize agent manager"""
        try:
            # Initialize agent pool if enabled
            if self.config.enable_agent_pooling:
                await self._initialize_agent_pool()
            
            # Start background monitoring
            if self.config.enable_resource_monitoring:
                self._start_monitoring()
            
            # Start health checking
            self._start_health_checking()
            
            logger.info("Agent manager initialization completed")
            return True
            
        except Exception as e:
            logger.error(f"Agent manager initialization failed: {e}")
            return False
    
    async def _initialize_agent_pool(self):
        """Initialize agent pool"""
        for i in range(self.config.agent_pool_size):
            agent_id = f"pool_agent_{i}"
            agent_config = self.config.default_agent_config
            
            agent = AgentInstance(agent_id, agent_config)
            if await agent.initialize():
                self.agents[agent_id] = agent
                self.agent_pool.append(agent_id)
                
                # Register with registry if available
                if self.registry:
                    await self.registry.register_agent(agent_id, agent.get_status())
    
    def _start_monitoring(self):
        """Start resource monitoring"""
        def monitoring_loop():
            while not self._shutdown_event.is_set():
                try:
                    self._collect_metrics()
                    self._shutdown_event.wait(self.config.resource_monitoring_interval)
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    self._shutdown_event.wait(30)
        
        self._monitoring_task = threading.Thread(target=monitoring_loop, daemon=True)
        self._monitoring_task.start()
    
    def _start_health_checking(self):
        """Start health checking"""
        def health_check_loop():
            while not self._shutdown_event.is_set():
                try:
                    asyncio.run(self._check_agent_health())
                    self._shutdown_event.wait(self.config.health_check_interval)
                except Exception as e:
                    logger.error(f"Health check error: {e}")
                    self._shutdown_event.wait(60)
        
        self._health_check_task = threading.Thread(target=health_check_loop, daemon=True)
        self._health_check_task.start()
    
    async def create_agent(self, agent_config: Dict[str, Any]) -> Optional[str]:
        """Create a new agent"""
        try:
            with self._lock:
                if len(self.agents) >= self.config.max_agents:
                    logger.warning("Maximum agent limit reached")
                    return None
                
                # Create agent configuration
                config = AgentConfig(**agent_config)
                
                # Generate agent ID
                agent_id = f"agent_{uuid.uuid4().hex[:8]}"
                
                # Create agent instance
                agent = AgentInstance(agent_id, config)
                
                # Initialize agent
                if await agent.initialize():
                    self.agents[agent_id] = agent
                    self.system_metrics["total_agents"] += 1
                    
                    # Register with registry if available
                    if self.registry:
                        await self.registry.register_agent(agent_id, agent.get_status())
                    
                    logger.info(f"Agent created: {agent_id}")
                    return agent_id
                else:
                    logger.error(f"Failed to initialize agent: {agent_id}")
                    return None
                    
        except Exception as e:
            logger.error(f"Agent creation failed: {e}")
            return None
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Destroy an agent"""
        try:
            with self._lock:
                if agent_id not in self.agents:
                    logger.warning(f"Agent not found: {agent_id}")
                    return False
                
                agent = self.agents[agent_id]
                
                # Terminate agent
                if await agent.terminate():
                    # Remove from collections
                    del self.agents[agent_id]
                    if agent_id in self.active_agents:
                        del self.active_agents[agent_id]
                    if agent_id in self.agent_pool:
                        self.agent_pool.remove(agent_id)
                    
                    self.system_metrics["total_agents"] -= 1
                    
                    # Unregister from registry if available
                    if self.registry:
                        await self.registry.unregister_agent(agent_id)
                    
                    logger.info(f"Agent destroyed: {agent_id}")
                    return True
                else:
                    logger.error(f"Failed to terminate agent: {agent_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Agent destruction failed: {e}")
            return False
    
    async def assign_task(self, task: Dict[str, Any]) -> Optional[str]:
        """Assign task to an available agent"""
        try:
            # Find available agent
            agent_id = await self._find_available_agent()
            if not agent_id:
                logger.warning("No available agents for task assignment")
                return None
            
            agent = self.agents[agent_id]
            
            # Execute task
            result = await agent.execute_task(task)
            
            # Update metrics
            self.system_metrics["total_tasks"] += 1
            self.system_metrics["successful_tasks"] += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Task assignment failed: {e}")
            self.system_metrics["failed_tasks"] += 1
            return None
    
    async def _find_available_agent(self) -> Optional[str]:
        """Find available agent for task assignment"""
        with self._lock:
            # Check idle agents first
            for agent_id, agent in self.agents.items():
                if agent.status == AgentStatus.IDLE:
                    return agent_id
            
            # Check agent pool
            if self.agent_pool:
                for agent_id in self.agent_pool:
                    if agent_id in self.agents and self.agents[agent_id].status == AgentStatus.IDLE:
                        return agent_id
            
            # If auto-scaling is enabled, create new agent
            if self.config.enable_auto_scaling and len(self.agents) < self.config.max_agents:
                agent_id = await self.create_agent(self.config.default_agent_config.__dict__)
                return agent_id
            
            return None
    
    def _collect_metrics(self):
        """Collect system metrics"""
        try:
            with self._lock:
                # Count agent statuses
                idle_agents = sum(1 for agent in self.agents.values() if agent.status == AgentStatus.IDLE)
                active_agents = sum(1 for agent in self.agents.values() if agent.status == AgentStatus.RUNNING)
                
                self.system_metrics.update({
                    "total_agents": len(self.agents),
                    "active_agents": active_agents,
                    "idle_agents": idle_agents
                })
                
                # Collect agent metrics
                total_cpu = 0
                total_memory = 0
                agent_count = 0
                
                for agent in self.agents.values():
                    if agent.health_check():
                        # This would normally integrate with actual resource monitoring
                        agent.metrics.cpu_usage = psutil.cpu_percent(interval=None)
                        agent.metrics.memory_usage = psutil.virtual_memory().percent
                        
                        total_cpu += agent.metrics.cpu_usage
                        total_memory += agent.metrics.memory_usage
                        agent_count += 1
                
                if agent_count > 0:
                    self.system_metrics["cpu_usage"] = total_cpu / agent_count
                    self.system_metrics["memory_usage"] = total_memory / agent_count
                    
        except Exception as e:
            logger.error(f"Metrics collection failed: {e}")
    
    async def _check_agent_health(self):
        """Check health of all agents"""
        try:
            with self._lock:
                unhealthy_agents = []
                
                for agent_id, agent in self.agents.items():
                    if not agent.health_check():
                        unhealthy_agents.append(agent_id)
                        logger.warning(f"Unhealthy agent detected: {agent_id}")
                
                # Handle unhealthy agents
                for agent_id in unhealthy_agents:
                    if self.agents[agent_id].status == AgentStatus.ERROR:
                        # Attempt to restart or destroy
                        logger.info(f"Attempting to restart unhealthy agent: {agent_id}")
                        await self.destroy_agent(agent_id)
                        
                        # Create replacement if needed
                        if self.config.enable_auto_scaling:
                            await self.create_agent(self.config.default_agent_config.__dict__)
                            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    async def start(self) -> bool:
        """Start agent manager"""
        try:
            logger.info("Starting agent manager...")
            
            # Start all agents
            for agent_id, agent in self.agents.items():
                if agent.status == AgentStatus.TERMINATED:
                    await agent.initialize()
            
            logger.info("Agent manager started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start agent manager: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop agent manager"""
        try:
            logger.info("Stopping agent manager...")
            
            # Stop all agents
            for agent_id, agent in self.agents.items():
                await agent.terminate()
            
            # Signal shutdown
            self._shutdown_event.set()
            
            logger.info("Agent manager stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop agent manager: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check agent manager health"""
        try:
            # Check system health
            healthy_agents = sum(1 for agent in self.agents.values() if agent.health_check())
            total_agents = len(self.agents)
            
            if total_agents == 0:
                return True  # No agents is valid
            
            health_ratio = healthy_agents / total_agents
            return health_ratio >= 0.8  # 80% healthy agents required
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent manager status"""
        with self._lock:
            return {
                "total_agents": len(self.agents),
                "active_agents": len([a for a in self.agents.values() if a.status == AgentStatus.RUNNING]),
                "idle_agents": len([a for a in self.agents.values() if a.status == AgentStatus.IDLE]),
                "error_agents": len([a for a in self.agents.values() if a.status == AgentStatus.ERROR]),
                "metrics": self.system_metrics,
                "config": {
                    "max_agents": self.config.max_agents,
                    "max_concurrent_agents": self.config.max_concurrent_agents,
                    "agent_pool_size": self.config.agent_pool_size
                }
            }
    
    def get_active_agents(self) -> List[str]:
        """Get list of active agent IDs"""
        with self._lock:
            return list(self.active_agents.keys())
    
    def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get information about specific agent"""
        with self._lock:
            if agent_id in self.agents:
                return self.agents[agent_id].get_status()
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics"""
        with self._lock:
            return self.system_metrics.copy()

# Factory function
def create_enhanced_agent_manager(config: Optional[Dict[str, Any]] = None) -> EnhancedAgentManager:
    """Create enhanced agent manager instance"""
    if config:
        agent_config = AgentManagerConfig(**config)
    else:
        agent_config = AgentManagerConfig()
    
    return EnhancedAgentManager(config=agent_config)