"""
Agent Orchestrator - Manages all AI agents
"""
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import uuid

from app.core.logging import get_logger
from app.core.config import settings
from app.models.agent import AgentTask
from app.services.base_service import BaseService
from app.agents.registry import AgentRegistry
from app.core.exceptions import AgentNotFoundError, AgentExecutionError

logger = get_logger(__name__)

class AgentOrchestrator(BaseService):
    """
    Orchestrates AI agent execution and collaboration
    """
    
    def __init__(self):
        super().__init__()
        self.registry = AgentRegistry()
        self.active_tasks: Dict[str, AgentTask] = {}
        self.task_queue = asyncio.Queue(maxsize=settings.MAX_CONCURRENT_AGENTS)
        self.workers: List[asyncio.Task] = []
        
    async def initialize(self):
        """Initialize the orchestrator"""
        logger.info("Initializing Agent Orchestrator...")
        
        # Load all agents
        await self.registry.load_agents()
        
        # Start worker tasks
        for i in range(settings.MAX_CONCURRENT_AGENTS):
            worker = asyncio.create_task(self._worker(f"worker-{i}"))
            self.workers.append(worker)
            
        logger.info(f"Started {len(self.workers)} agent workers")
        
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up Agent Orchestrator...")
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
            
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        
    async def execute_task(
        self,
        agent_name: str,
        task_description: str,
        parameters: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> str:
        """
        Execute a task using specified agent
        """
        # Validate agent exists
        if not self.registry.agent_exists(agent_name):
            raise AgentNotFoundError(f"Agent '{agent_name}' not found")
            
        # Create task
        task_id = str(uuid.uuid4())
        task = AgentTask(
            id=task_id,
            agent_name=agent_name,
            task_description=task_description,
            parameters=parameters or {},
            user_id=user_id,
            status="queued",
            created_at=datetime.utcnow()
        )
        
        # Store task
        self.active_tasks[task_id] = task
        
        # Queue task
        await self.task_queue.put(task)
        
        logger.info(f"Queued task {task_id} for agent {agent_name}")
        
        return task_id
        
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a task"""
        task = self.active_tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
            
        return {
            "id": task.id,
            "agent_name": task.agent_name,
            "status": task.status,
            "created_at": task.created_at.isoformat(),
            "completed_at": task.completed_at.isoformat() if task.completed_at else None,
            "result": task.result
        }
        
    async def cancel_task(self, task_id: str):
        """Cancel a running task"""
        task = self.active_tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
            
        if task.status in ["completed", "failed"]:
            raise ValueError(f"Task {task_id} already {task.status}")
            
        task.status = "cancelled"
        task.completed_at = datetime.utcnow()
        
    async def list_agents(self) -> List[Dict[str, Any]]:
        """List all available agents"""
        return self.registry.list_agents()
        
    async def get_agent_info(self, agent_name: str) -> Dict[str, Any]:
        """Get information about a specific agent"""
        return self.registry.get_agent_info(agent_name)
        
    async def _worker(self, worker_id: str):
        """Worker task that processes agent tasks"""
        logger.info(f"Agent worker {worker_id} started")
        
        while True:
            try:
                # Get task from queue
                task = await self.task_queue.get()
                
                # Update task status
                task.status = "running"
                
                logger.info(f"Worker {worker_id} executing task {task.id}")
                
                # Get agent
                agent = self.registry.get_agent(task.agent_name)
                
                # Execute task
                try:
                    result = await agent.execute(
                        task.task_description,
                        task.parameters
                    )
                    
                    # Update task with result
                    task.status = "completed"
                    task.result = result
                    task.completed_at = datetime.utcnow()
                    
                except Exception as e:
                    logger.error(f"Agent execution error: {e}")
                    task.status = "failed"
                    task.result = {"error": str(e)}
                    task.completed_at = datetime.utcnow()
                    
                # Mark task as done
                self.task_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                
        logger.info(f"Agent worker {worker_id} stopped")
        
    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            "active_agents": len(self.registry.agents),
            "queued_tasks": self.task_queue.qsize(),
            "active_tasks": len([t for t in self.active_tasks.values() if t.status == "running"]),
            "workers": len(self.workers)
        }

# Create singleton instance
agent_orchestrator = AgentOrchestrator()