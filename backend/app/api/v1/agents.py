"""
AI Agent Management API Endpoints - Real Claude Agent Integration
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import logging
import asyncio
import uuid

logger = logging.getLogger(__name__)

router = APIRouter()

# Import our real orchestration components
try:
    from app.core.unified_agent_registry import get_registry
    from app.core.claude_agent_executor import get_executor, get_pool
    _orchestration_available = True
except ImportError:
    logger.warning("Orchestration components not available, using fallback")
    _orchestration_available = False

# Request/Response models
class AgentTaskRequest(BaseModel):
    """Request to execute an agent task"""
    task_description: str = Field(..., description="Task to execute")
    agent_type: Optional[str] = Field(None, description="Specific agent type")
    task_type: Optional[str] = Field("general", description="Type of task")
    task_data: Optional[Dict[str, Any]] = Field(None, description="Additional task data")
    required_capabilities: Optional[List[str]] = Field(None, description="Required capabilities")
    async_execution: bool = Field(False, description="Execute asynchronously")

class AgentTaskResponse(BaseModel):
    """Response from agent task execution"""
    task_id: str
    status: str
    agents_used: List[str]
    results: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

class AgentStatusResponse(BaseModel):
    """Agent status response"""
    total_agents: int
    active_agents: int
    claude_agents: int = 0
    container_agents: int = 0
    agents: Dict[str, Any]

# Real Agent Manager with Claude Integration
class RealAgentManager:
    def __init__(self):
        if _orchestration_available:
            self.registry = get_registry()
            self.executor = get_executor()
            self.agent_pool = get_pool()
            # Start background processor
            asyncio.create_task(self.agent_pool.process_tasks())
        else:
            self.registry = None
            self.executor = None
            self.agent_pool = None
            
        self.tasks = {}
        
    async def execute_task(self, agent_type: str, task_type: str, task_data: Dict[str, Any],
                          task_description: str = None, required_capabilities: List[str] = None,
                          async_execution: bool = False) -> Dict[str, Any]:
        """Execute a task with real Claude agent integration"""
        
        if not _orchestration_available:
            # Fallback behavior
            task_id = f"task_{len(self.tasks) + 1}"
            return {
                "task_id": task_id,
                "status": "unavailable",
                "agents_used": [],
                "error": "Orchestration system not available"
            }
            
        # Use real orchestration
        if not task_description:
            task_description = f"Execute {task_type} task with data: {task_data}"
            
        # Find best agent
        if agent_type:
            # Try to find specific agent
            agent = self.registry.get_agent(f"claude_{agent_type}") or \
                   self.registry.get_agent(f"container_{agent_type}") or \
                   self.registry.get_agent(agent_type)
        else:
            # Auto-select best agent
            agent = self.registry.find_best_agent(task_description, required_capabilities)
            
        if not agent:
            raise ValueError("No suitable agent found for task")
            
        # Execute agent
        if agent.type == "claude":
            if async_execution:
                task_id = await self.agent_pool.submit_task(
                    agent.name, task_description, task_data
                )
                result = {
                    "task_id": task_id,
                    "status": "pending",
                    "agents_used": [agent.name],
                    "results": [{"message": "Task submitted for async execution"}]
                }
            else:
                exec_result = await self.executor.execute_agent(
                    agent.name, task_description, task_data
                )
                result = {
                    "task_id": exec_result.get("task_id", str(uuid.uuid4())),
                    "status": exec_result.get("status", "completed"),
                    "agents_used": [agent.name],
                    "results": [exec_result]
                }
        else:
            # Container agent placeholder
            task_id = str(uuid.uuid4())
            result = {
                "task_id": task_id,
                "status": "completed",
                "agents_used": [agent.name],
                "results": [{
                    "agent": agent.name,
                    "type": agent.type,
                    "message": "Container agent execution (placeholder)"
                }]
            }
            
        self.tasks[result["task_id"]] = result
        return result
        
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents with real counts"""
        if not _orchestration_available:
            return {
                "total_agents": 0,
                "active_agents": 0,
                "claude_agents": 0,
                "container_agents": 0,
                "agents": {}
            }
            
        stats = self.registry.get_statistics()
        
        # Build agent dict
        agents = {}
        for agent in self.registry.list_agents():
            agents[agent.id] = {
                "name": agent.name,
                "type": agent.type,
                "status": "active" if agent.type == "claude" else "available",
                "capabilities": agent.capabilities,
                "description": agent.description[:200] if agent.description else ""
            }
            
        return {
            "total_agents": stats["total_agents"],
            "active_agents": stats.get("claude_agents", 0),
            "claude_agents": stats.get("claude_agents", 0),
            "container_agents": stats.get("container_agents", 0),
            "agents": agents
        }
        
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a specific task"""
        # Check pool results first
        if _orchestration_available and self.agent_pool:
            result = self.agent_pool.get_result(task_id)
            if result:
                return result
                
        # Check local tasks
        if task_id not in self.tasks:
            raise ValueError(f"Task {task_id} not found")
        return self.tasks[task_id]

# Create singleton instance
agent_manager = RealAgentManager()

@router.get("/status", response_model=AgentStatusResponse)
async def get_agent_status():
    """Get status of all available agents"""
    try:
        status = await agent_manager.get_agent_status()
        return status
    except Exception as e:
        logger.error(f"Failed to get agent status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/execute", response_model=AgentTaskResponse)
async def execute_agent_task(request: AgentTaskRequest, background_tasks: BackgroundTasks):
    """Execute a task using AI agents - now with real Claude agent integration"""
    try:
        result = await agent_manager.execute_task(
            request.agent_type,
            request.task_type,
            request.task_data,
            task_description=request.task_description,
            required_capabilities=request.required_capabilities,
            async_execution=request.async_execution
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to execute agent task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/tasks/{task_id}", response_model=AgentTaskResponse)
async def get_task_status(task_id: str):
    """Get status of a specific task"""
    try:
        status = await agent_manager.get_task_status(task_id)
        return status
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/capabilities", response_model=Dict[str, List[str]])
async def get_agent_capabilities():
    """Get all available agent capabilities from the unified registry"""
    if not _orchestration_available or not agent_manager.registry:
        return {"status": "success", "agents": [], "timestamp": datetime.now().isoformat()}
        
    capabilities = {}
    for agent in agent_manager.registry.list_agents():
        capabilities[agent.name] = agent.capabilities
    return capabilities

@router.post("/recommend")
async def recommend_agent(
    task_description: str = Query(..., description="Task description"),
    required_capabilities: Optional[List[str]] = Query(None, description="Required capabilities")
):
    """Get intelligent agent recommendations for a task"""
    if not _orchestration_available:
        raise HTTPException(status_code=503, detail="Orchestration system not available")
        
    try:
        from app.core.claude_agent_selector import get_selector
        selector = get_selector()
        
        recommendations = selector.get_agent_recommendations(task_description)
        return recommendations
        
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list")
async def list_all_agents(
    agent_type: Optional[str] = Query(None, description="Filter by type: claude or container"),
    capabilities: Optional[List[str]] = Query(None, description="Filter by capabilities")
):
    """List all available agents from the unified registry"""
    if not _orchestration_available or not agent_manager.registry:
        return []
        
    agents = agent_manager.registry.list_agents(agent_type, capabilities)
    
    return [
        {
            "id": agent.id,
            "name": agent.name,
            "type": agent.type,
            "description": agent.description[:200] if agent.description else "",
            "capabilities": agent.capabilities
        }
        for agent in agents
    ]

@router.get("/statistics")
async def get_agent_statistics():
    """Get comprehensive statistics about available agents"""
    if not _orchestration_available or not agent_manager.registry:
        return {"error": "Orchestration system not available"}
        
    stats = agent_manager.registry.get_statistics()
    
    # Add execution statistics
    if agent_manager.executor:
        stats["execution_history"] = len(agent_manager.executor.get_execution_history(100))
        stats["active_tasks"] = len(agent_manager.executor.get_active_tasks())
    
    return stats

@router.post("/collaborate", response_model=AgentTaskResponse)
async def collaborate_agents(
    task_type: str,
    task_data: Dict[str, Any],
    agents: List[str]
):
    """Execute a task using multiple agents collaboratively"""
    try:
        # For demo, simulate collaboration
        results = []
        for agent in agents:
            if agent in agent_manager.agents:
                results.append({
                    "agent": agent,
                    "status": "completed" if agent_manager.agents[agent]["status"] == "active" else "skipped",
                    "result": {"processed": True}
                })
        
        return AgentTaskResponse(
            task_id=f"collab_{len(agent_manager.tasks) + 1}",
            status="success",
            agents_used=agents,
            results=results
        )
    except Exception as e:
        logger.error(f"Failed to collaborate agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))
