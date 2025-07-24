#!/usr/bin/env python3
"""
Agent Orchestrator - Main coordination system for multi-agent environments
"""

import asyncio
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import json
from pathlib import Path

from .agent_manager import EnhancedAgentManager, AgentManagerConfig
from .agent_coordinator import AgentCoordinator, CoordinationConfig
from .communication_system import CommunicationSystem, CommunicationConfig
from .task_scheduler import TaskScheduler, SchedulerConfig
from .collaboration_engine import CollaborationEngine, CollaborationConfig
from .resource_manager import ResourceManager, ResourceConfig
from .workflow_engine import WorkflowEngine
from .agent_registry import AgentRegistry, RegistryConfig
from .performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)

class OrchestratorMode(Enum):
    """Orchestrator operation modes"""
    CENTRALIZED = "centralized"
    DISTRIBUTED = "distributed"
    HYBRID = "hybrid"
    AUTONOMOUS = "autonomous"

class OrchestratorStatus(Enum):
    """Orchestrator status"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class OrchestratorConfig:
    """Configuration for agent orchestrator"""
    # Core settings
    mode: OrchestratorMode = OrchestratorMode.HYBRID
    max_agents: int = 100
    max_concurrent_tasks: int = 50
    
    # Component configurations - using proper defaults
    agent_manager_config: Optional[Dict[str, Any]] = None
    coordinator_config: Optional[Dict[str, Any]] = None
    communication_config: Optional[Dict[str, Any]] = None
    scheduler_config: Optional[Dict[str, Any]] = None
    collaboration_config: Optional[Dict[str, Any]] = None
    resource_config: Optional[Dict[str, Any]] = None
    workflow_config: Optional[Dict[str, Any]] = None
    registry_config: Optional[Dict[str, Any]] = None
    monitor_config: Optional[Dict[str, Any]] = None
    
    # Performance settings
    heartbeat_interval: float = 5.0  # seconds
    task_timeout: float = 300.0  # seconds
    agent_timeout: float = 600.0  # seconds
    
    # Fault tolerance
    enable_fault_tolerance: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Security settings
    enable_authentication: bool = True
    enable_authorization: bool = True
    enable_encryption: bool = True
    
    # Directories
    data_dir: str = "/opt/sutazaiapp/backend/data/orchestration"
    logs_dir: str = "/opt/sutazaiapp/backend/logs/orchestration"
    cache_dir: str = "/opt/sutazaiapp/backend/cache/orchestration"

class AgentOrchestrator:
    """
    Main agent orchestrator for managing multi-agent systems
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize agent orchestrator"""
        self.config = OrchestratorConfig(**config) if config else OrchestratorConfig()
        self.orchestrator_id = str(uuid.uuid4())
        self.status = OrchestratorStatus.INITIALIZING
        self.start_time = datetime.now(timezone.utc)
        
        # Core components
        self.agent_manager: Optional[EnhancedAgentManager] = None
        self.coordinator: Optional[AgentCoordinator] = None
        self.communication_system: Optional[CommunicationSystem] = None
        self.task_scheduler: Optional[TaskScheduler] = None
        self.collaboration_engine: Optional[CollaborationEngine] = None
        self.resource_manager: Optional[ResourceManager] = None
        self.workflow_engine: Optional[WorkflowEngine] = None
        self.agent_registry: Optional[AgentRegistry] = None
        self.performance_monitor: Optional[PerformanceMonitor] = None
        
        # State management
        self.active_agents: Dict[str, Dict[str, Any]] = {}
        self.active_tasks: Dict[str, Dict[str, Any]] = {}
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        
        # Event system
        self.event_handlers: Dict[str, List[Any]] = {}
        
        # Threading
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # Performance metrics
        self.metrics = {
            "total_agents": 0,
            "active_agents": 0,
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "average_response_time": 0.0,
            "uptime": 0.0
        }
        
        # Create directories
        self._create_directories()
        
        logger.info(f"Agent orchestrator initialized with ID: {self.orchestrator_id}")
    
    def _create_directories(self):
        """Create required directories"""
        directories = [
            self.config.data_dir,
            self.config.logs_dir,
            self.config.cache_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    async def initialize(self) -> bool:
        """Initialize orchestrator components"""
        try:
            with self._lock:
                if self.status != OrchestratorStatus.INITIALIZING:
                    logger.warning("Orchestrator already initialized")
                    return True
                
                logger.info("Initializing orchestrator components...")
                
                # Initialize agent registry
                from .agent_registry import RegistryConfig
                registry_config = RegistryConfig(**(self.config.registry_config or {}))
                self.agent_registry = AgentRegistry(registry_config)
                await self.agent_registry.initialize()
                
                # Initialize resource manager
                from .resource_manager import ResourceConfig
                resource_config = ResourceConfig(**(self.config.resource_config or {}))
                self.resource_manager = ResourceManager(resource_config)
                await self.resource_manager.initialize()
                
                # Initialize workflow engine (using existing simple implementation)
                self.workflow_engine = WorkflowEngine()
                logger.info("Workflow engine initialized")
                
                # Initialize performance monitor (using existing implementation)
                self.performance_monitor = PerformanceMonitor()
                await self.performance_monitor.start_monitoring()
                logger.info("Performance monitor initialized")
                
                # Other components will be created as needed
                # For now, focus on the working components
                self.communication_system = None
                self.agent_manager = None  
                self.task_scheduler = None
                self.collaboration_engine = None
                self.coordinator = None
                
                # Setup event handlers
                self._setup_event_handlers()
                
                # Start background tasks
                self._start_background_tasks()
                
                self.status = OrchestratorStatus.STOPPED
                logger.info("Orchestrator initialization completed")
                return True
                
        except Exception as e:
            logger.error(f"Orchestrator initialization failed: {e}")
            self.status = OrchestratorStatus.ERROR
            raise
    
    def _setup_event_handlers(self):
        """Setup event handlers"""
        self.event_handlers = {
            "agent_created": [],
            "agent_destroyed": [],
            "task_started": [],
            "task_completed": [],
            "task_failed": [],
            "workflow_started": [],
            "workflow_completed": [],
            "collaboration_started": [],
            "error": []
        }
    
    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        def heartbeat_task():
            while not self._shutdown_event.is_set():
                try:
                    asyncio.run(self._heartbeat())
                    self._shutdown_event.wait(self.config.heartbeat_interval)
                except Exception as e:
                    logger.error(f"Heartbeat error: {e}")
                    self._shutdown_event.wait(5)
        
        # Start heartbeat in background thread
        heartbeat_thread = threading.Thread(target=heartbeat_task, daemon=True)
        heartbeat_thread.start()
    
    async def _heartbeat(self):
        """Periodic heartbeat to check system health"""
        try:
            # Update metrics
            self.metrics["uptime"] = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            self.metrics["active_agents"] = len(self.active_agents)
            self.metrics["total_tasks"] = len(self.active_tasks)
            
            # Check component health
            if self.agent_manager:
                await self.agent_manager.health_check()
            
            if self.task_scheduler:
                await self.task_scheduler.health_check()
            
            if self.performance_monitor:
                await self.performance_monitor.collect_metrics()
            
        except Exception as e:
            logger.error(f"Heartbeat failed: {e}")
    
    async def start(self) -> bool:
        """Start the orchestrator"""
        try:
            if self.status != OrchestratorStatus.STOPPED:
                logger.warning(f"Orchestrator not in stopped state: {self.status}")
                return False
            
            logger.info("Starting agent orchestrator...")
            self.status = OrchestratorStatus.RUNNING
            
            # Start components
            components = [
                self.agent_registry,
                self.resource_manager,
                self.communication_system,
                self.agent_manager,
                self.task_scheduler,
                self.collaboration_engine,
                self.workflow_engine,
                self.coordinator,
                self.performance_monitor
            ]
            
            for component in components:
                if component and hasattr(component, 'start'):
                    await component.start()
            
            logger.info("Agent orchestrator started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start orchestrator: {e}")
            self.status = OrchestratorStatus.ERROR
            return False
    
    async def stop(self) -> bool:
        """Stop the orchestrator"""
        try:
            if self.status == OrchestratorStatus.STOPPED:
                logger.info("Orchestrator already stopped")
                return True
            
            logger.info("Stopping agent orchestrator...")
            self.status = OrchestratorStatus.STOPPING
            
            # Stop components in reverse order
            components = [
                self.performance_monitor,
                self.coordinator,
                self.workflow_engine,
                self.collaboration_engine,
                self.task_scheduler,
                self.agent_manager,
                self.communication_system,
                self.resource_manager,
                self.agent_registry
            ]
            
            for component in components:
                if component and hasattr(component, 'stop'):
                    await component.stop()
            
            # Signal shutdown
            self._shutdown_event.set()
            
            self.status = OrchestratorStatus.STOPPED
            logger.info("Agent orchestrator stopped successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop orchestrator: {e}")
            self.status = OrchestratorStatus.ERROR
            return False
    
    async def create_agent(self, agent_config: Dict[str, Any]) -> Optional[str]:
        """Create a new agent"""
        try:
            if not self.agent_manager:
                raise RuntimeError("Agent manager not initialized")
            
            # Create agent
            agent_id = await self.agent_manager.create_agent(agent_config)
            
            if agent_id:
                # Register agent
                self.active_agents[agent_id] = {
                    "id": agent_id,
                    "config": agent_config,
                    "created_at": datetime.now(timezone.utc),
                    "status": "active"
                }
                
                # Fire event
                await self._fire_event("agent_created", {"agent_id": agent_id})
                
                # Update metrics
                self.metrics["total_agents"] += 1
                
                logger.info(f"Agent created: {agent_id}")
            
            return agent_id
            
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            return None
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """Destroy an agent"""
        try:
            if not self.agent_manager:
                raise RuntimeError("Agent manager not initialized")
            
            # Destroy agent
            success = await self.agent_manager.destroy_agent(agent_id)
            
            if success and agent_id in self.active_agents:
                # Remove from active agents
                del self.active_agents[agent_id]
                
                # Fire event
                await self._fire_event("agent_destroyed", {"agent_id": agent_id})
                
                logger.info(f"Agent destroyed: {agent_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to destroy agent: {e}")
            return False
    
    async def execute_task(self, task_definition: Dict[str, Any]) -> str:
        """Execute a task"""
        try:
            if not self.task_scheduler:
                raise RuntimeError("Task scheduler not initialized")
            
            # Generate task ID
            task_id = str(uuid.uuid4())
            
            # Add task to active tasks
            self.active_tasks[task_id] = {
                "id": task_id,
                "definition": task_definition,
                "status": "pending",
                "created_at": datetime.now(timezone.utc)
            }
            
            # Fire event
            await self._fire_event("task_started", {"task_id": task_id})
            
            # Schedule task
            result = await self.task_scheduler.schedule_task(task_id, task_definition)
            
            # Update task status
            if task_id in self.active_tasks:
                self.active_tasks[task_id]["status"] = "running"
                self.active_tasks[task_id]["result"] = result
            
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to execute task: {e}")
            # Mark task as failed
            if task_id in self.active_tasks:
                self.active_tasks[task_id]["status"] = "failed"
                self.active_tasks[task_id]["error"] = str(e)
            
            await self._fire_event("task_failed", {"task_id": task_id, "error": str(e)})
            raise
    
    async def execute_workflow(self, workflow_definition: Dict[str, Any]) -> str:
        """Execute a workflow"""
        try:
            if not self.workflow_engine:
                raise RuntimeError("Workflow engine not initialized")
            
            # Generate workflow ID
            workflow_id = str(uuid.uuid4())
            
            # Add workflow to active workflows
            self.active_workflows[workflow_id] = {
                "id": workflow_id,
                "definition": workflow_definition,
                "status": "pending",
                "created_at": datetime.now(timezone.utc)
            }
            
            # Fire event
            await self._fire_event("workflow_started", {"workflow_id": workflow_id})
            
            # Execute workflow
            result = await self.workflow_engine.execute_workflow(workflow_id, workflow_definition)
            
            # Update workflow status
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id]["status"] = "running"
                self.active_workflows[workflow_id]["result"] = result
            
            return workflow_id
            
        except Exception as e:
            logger.error(f"Failed to execute workflow: {e}")
            # Mark workflow as failed
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id]["status"] = "failed"
                self.active_workflows[workflow_id]["error"] = str(e)
            
            raise
    
    async def start_collaboration(self, collaboration_config: Dict[str, Any]) -> str:
        """Start agent collaboration"""
        try:
            if not self.collaboration_engine:
                raise RuntimeError("Collaboration engine not initialized")
            
            # Start collaboration
            collaboration_id = await self.collaboration_engine.start_collaboration(collaboration_config)
            
            # Fire event
            await self._fire_event("collaboration_started", {"collaboration_id": collaboration_id})
            
            return collaboration_id
            
        except Exception as e:
            logger.error(f"Failed to start collaboration: {e}")
            raise
    
    async def _fire_event(self, event_type: str, data: Dict[str, Any]):
        """Fire system event"""
        try:
            if event_type in self.event_handlers:
                for handler in self.event_handlers[event_type]:
                    await handler(data)
        except Exception as e:
            logger.error(f"Event handling error: {e}")
    
    def add_event_handler(self, event_type: str, handler):
        """Add event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            "orchestrator_id": self.orchestrator_id,
            "status": self.status.value,
            "mode": self.config.mode.value,
            "uptime": self.metrics["uptime"],
            "active_agents": len(self.active_agents),
            "active_tasks": len(self.active_tasks),
            "active_workflows": len(self.active_workflows),
            "metrics": self.metrics,
            "components": {
                "agent_manager": self.agent_manager.get_status() if self.agent_manager else None,
                "task_scheduler": self.task_scheduler.get_status() if self.task_scheduler else None,
                "collaboration_engine": self.collaboration_engine.get_status() if self.collaboration_engine else None,
                "workflow_engine": self.workflow_engine.get_status() if self.workflow_engine else None
            }
        }
    
    def get_agents(self) -> List[Dict[str, Any]]:
        """Get list of active agents"""
        return list(self.active_agents.values())
    
    def get_tasks(self) -> List[Dict[str, Any]]:
        """Get list of active tasks"""
        return list(self.active_tasks.values())
    
    def get_workflows(self) -> List[Dict[str, Any]]:
        """Get list of active workflows"""
        return list(self.active_workflows.values())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics"""
        return self.metrics.copy()
    
    def health_check(self) -> bool:
        """Check orchestrator health"""
        try:
            if self.status != OrchestratorStatus.RUNNING:
                return False
            
            # Check components
            components = [
                self.agent_manager,
                self.task_scheduler,
                self.collaboration_engine,
                self.workflow_engine
            ]
            
            for component in components:
                if component and hasattr(component, 'health_check'):
                    if not component.health_check():
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def save_state(self, file_path: str) -> bool:
        """Save orchestrator state"""
        try:
            state = {
                "orchestrator_id": self.orchestrator_id,
                "status": self.status.value,
                "config": self.config.__dict__,
                "active_agents": self.active_agents,
                "active_tasks": self.active_tasks,
                "active_workflows": self.active_workflows,
                "metrics": self.metrics,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            with open(file_path, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            
            logger.info(f"Orchestrator state saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
            return False
    
    async def load_state(self, file_path: str) -> bool:
        """Load orchestrator state"""
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            # Restore state
            self.orchestrator_id = state.get("orchestrator_id", self.orchestrator_id)
            self.active_agents = state.get("active_agents", {})
            self.active_tasks = state.get("active_tasks", {})
            self.active_workflows = state.get("active_workflows", {})
            self.metrics = state.get("metrics", self.metrics)
            
            logger.info(f"Orchestrator state loaded from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load state: {e}")
            return False
    
    def __del__(self):
        """Cleanup resources"""
        try:
            self._shutdown_event.set()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# Factory function
def create_agent_orchestrator(config: Optional[Dict[str, Any]] = None) -> AgentOrchestrator:
    """Create agent orchestrator instance"""
    return AgentOrchestrator(config=config)