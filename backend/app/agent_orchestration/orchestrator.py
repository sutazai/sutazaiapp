"""
Unified Agent Orchestrator
Manages and coordinates all AI agents in the system
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio
from collections import defaultdict

logger = logging.getLogger(__name__)

class UnifiedAgentOrchestrator:
    """Central orchestrator for all AI agents in the system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.agents = {}
        self.tasks = defaultdict(list)
        self.workflows = {}
        self.active_sessions = {}
        self.metrics = {
            "total_tasks": 0,
            "completed_tasks": 0,
            "failed_tasks": 0,
            "active_agents": 0
        }
        self.initialized = False
        
    async def initialize(self):
        """Initialize the orchestrator and all subsystems"""
        if self.initialized:
            return
            
        logger.info("Initializing Unified Agent Orchestrator")
        
        # Initialize agent discovery
        await self._init_agent_discovery()
        
        # Initialize message bus
        await self._init_message_bus()
        
        # Initialize task router
        await self._init_task_router()
        
        # Initialize workflow engine
        await self._init_workflow_engine()
        
        self.initialized = True
        logger.info("Agent Orchestrator initialized successfully")
        
    async def _init_agent_discovery(self):
        """Initialize agent discovery system"""
        try:
            from app.orchestration.agent_discovery import AgentDiscovery
            self.agent_discovery = AgentDiscovery(self.config.get("discovery", {}))
            await self.agent_discovery.start()
            logger.info("Agent discovery initialized")
        except Exception as e:
            logger.warning(f"Agent discovery initialization skipped: {e}")
            self.agent_discovery = None
            
    async def _init_message_bus(self):
        """Initialize message bus for inter-agent communication"""
        try:
            from app.orchestration.message_bus import MessageBus
            self.message_bus = MessageBus(self.config.get("message_bus", {}))
            await self.message_bus.start()
            logger.info("Message bus initialized")
        except Exception as e:
            logger.warning(f"Message bus initialization skipped: {e}")
            self.message_bus = None
            
    async def _init_task_router(self):
        """Initialize task router"""
        try:
            from app.orchestration.task_router import TaskRouter
            self.task_router = TaskRouter(self.config.get("task_router", {}))
            await self.task_router.initialize()
            logger.info("Task router initialized")
        except Exception as e:
            logger.warning(f"Task router initialization skipped: {e}")
            self.task_router = None
            
    async def _init_workflow_engine(self):
        """Initialize workflow engine"""
        try:
            from app.orchestration.workflow_engine import WorkflowEngine
            self.workflow_engine = WorkflowEngine(self.config.get("workflow", {}))
            await self.workflow_engine.initialize()
            logger.info("Workflow engine initialized")
        except Exception as e:
            logger.warning(f"Workflow engine initialization skipped: {e}")
            self.workflow_engine = None
            
    async def register_agent(self, agent_id: str, agent_info: Dict[str, Any]) -> bool:
        """Register a new agent with the orchestrator"""
        try:
            self.agents[agent_id] = {
                "info": agent_info,
                "status": "active",
                "registered_at": datetime.now().isoformat(),
                "last_heartbeat": datetime.now().isoformat()
            }
            self.metrics["active_agents"] = len(self.agents)
            logger.info(f"Agent {agent_id} registered successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            return False
            
    async def submit_task(self, task: Dict[str, Any]) -> str:
        """Submit a task for execution"""
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.metrics['total_tasks']}"
        
        task_data = {
            "id": task_id,
            "status": "pending",
            "submitted_at": datetime.now().isoformat(),
            "task": task
        }
        
        self.tasks[task_id] = task_data
        self.metrics["total_tasks"] += 1
        
        # Route task if router is available
        if self.task_router:
            asyncio.create_task(self._route_task(task_id, task))
        
        logger.info(f"Task {task_id} submitted")
        return task_id
        
    async def _route_task(self, task_id: str, task: Dict[str, Any]):
        """Route task to appropriate agent"""
        try:
            agent_id = await self.task_router.route(task)
            if agent_id and agent_id in self.agents:
                self.tasks[task_id]["status"] = "assigned"
                self.tasks[task_id]["assigned_to"] = agent_id
                logger.info(f"Task {task_id} assigned to agent {agent_id}")
            else:
                self.tasks[task_id]["status"] = "unassigned"
                logger.warning(f"No suitable agent found for task {task_id}")
        except Exception as e:
            logger.error(f"Failed to route task {task_id}: {e}")
            self.tasks[task_id]["status"] = "routing_failed"
            
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a submitted task"""
        return self.tasks.get(task_id)
        
    async def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents"""
        return [
            {
                "id": agent_id,
                **agent_data
            }
            for agent_id, agent_data in self.agents.items()
        ]
        
    async def get_metrics(self) -> Dict[str, Any]:
        """Get orchestrator metrics"""
        return {
            **self.metrics,
            "agents": len(self.agents),
            "pending_tasks": sum(1 for t in self.tasks.values() if t.get("status") == "pending"),
            "assigned_tasks": sum(1 for t in self.tasks.values() if t.get("status") == "assigned"),
            "timestamp": datetime.now().isoformat()
        }
        
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on orchestrator"""
        health = {
            "status": "healthy" if self.initialized else "unhealthy",
            "initialized": self.initialized,
            "components": {
                "agent_discovery": bool(self.agent_discovery),
                "message_bus": bool(self.message_bus),
                "task_router": bool(self.task_router),
                "workflow_engine": bool(self.workflow_engine)
            },
            "metrics": await self.get_metrics()
        }
        return health
        
    async def shutdown(self):
        """Gracefully shutdown the orchestrator"""
        logger.info("Shutting down Agent Orchestrator")
        
        # Shutdown components
        if self.agent_discovery:
            await self.agent_discovery.stop()
        if self.message_bus:
            await self.message_bus.stop()
        if self.workflow_engine:
            await self.workflow_engine.shutdown()
            
        self.initialized = False
        logger.info("Agent Orchestrator shut down complete")

# Create singleton instance
_orchestrator_instance = None

def get_orchestrator() -> UnifiedAgentOrchestrator:
    """Get the singleton orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = UnifiedAgentOrchestrator()
    return _orchestrator_instance