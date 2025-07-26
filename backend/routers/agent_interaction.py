"""
Agent Interaction Router for SutazAI AGI System

This module provides enterprise-level agent interaction endpoints,
consolidating functionality from various agent-related routers.
"""

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
import uuid

# Import agent orchestration components
try:
    from backend.agent_orchestration.orchestrator import AgentOrchestrator
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False
    AgentOrchestrator = None

# Import agent manager
try:
    from backend.ai_agents.agent_manager import AgentManager
    AGENT_MANAGER_AVAILABLE = True
except ImportError:
    AGENT_MANAGER_AVAILABLE = False
    AgentManager = None

# Import monitoring
try:
    from backend.monitoring.monitoring import MonitoringService
    monitoring_service = MonitoringService()
except ImportError:
    monitoring_service = None


# Request/Response Models
class AgentTaskRequest(BaseModel):
    """Request model for agent task execution"""
    agent_name: str = Field(..., description="Name of the agent to execute task")
    task_type: str = Field(..., description="Type of task to execute")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Task parameters")
    priority: int = Field(default=5, ge=1, le=10, description="Task priority (1-10)")
    timeout: Optional[int] = Field(default=300, description="Task timeout in seconds")


class AgentTaskResponse(BaseModel):
    """Response model for agent task execution"""
    task_id: str = Field(..., description="Unique task identifier")
    agent_name: str = Field(..., description="Name of the executing agent")
    status: str = Field(..., description="Task status")
    created_at: datetime = Field(..., description="Task creation timestamp")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result if completed")
    error: Optional[str] = Field(None, description="Error message if failed")


class AgentCollaborationRequest(BaseModel):
    """Request model for multi-agent collaboration"""
    task_description: str = Field(..., description="Description of the collaborative task")
    required_agents: List[str] = Field(..., description="List of required agents")
    coordination_strategy: str = Field(default="sequential", description="Coordination strategy")
    shared_context: Dict[str, Any] = Field(default_factory=dict, description="Shared context data")


class AgentCollaborationResponse(BaseModel):
    """Response model for multi-agent collaboration"""
    collaboration_id: str = Field(..., description="Unique collaboration identifier")
    participating_agents: List[str] = Field(..., description="List of participating agents")
    status: str = Field(..., description="Collaboration status")
    results: Dict[str, Any] = Field(default_factory=dict, description="Collaboration results")


class AgentStatusResponse(BaseModel):
    """Response model for agent status"""
    agent_name: str = Field(..., description="Agent name")
    status: str = Field(..., description="Agent status (active/inactive/busy)")
    current_task: Optional[str] = Field(None, description="Current task if any")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    performance_metrics: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")


# Create router
router = APIRouter(
    prefix="/agents/interaction",
    tags=["agent-interaction"],
    responses={404: {"description": "Not found"}},
)


# Dependency to get orchestrator
def get_orchestrator():
    """Get the agent orchestrator instance"""
    if not ORCHESTRATOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Agent orchestrator not available")
    # In a real implementation, this would return a singleton instance
    return AgentOrchestrator()


# Dependency to get agent manager
def get_agent_manager():
    """Get the agent manager instance"""
    if not AGENT_MANAGER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Agent manager not available")
    # In a real implementation, this would return a singleton instance
    return AgentManager()


@router.post("/execute", response_model=AgentTaskResponse)
async def execute_agent_task(
    request: AgentTaskRequest,
    background_tasks: BackgroundTasks,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """
    Execute a task with a specific agent.
    
    This endpoint allows direct task execution with a named agent,
    providing enterprise-level control over agent operations.
    """
    task_id = str(uuid.uuid4())
    
    # Log the task execution request
    if monitoring_service:
        monitoring_service.log_event(
            event_type="agent_task_requested",
            message=f"Task requested for agent {request.agent_name}",
            details={"task_type": request.task_type, "task_id": task_id}
        )
    
    # Execute task in background
    background_tasks.add_task(
        orchestrator.execute_task,
        agent_name=request.agent_name,
        task_type=request.task_type,
        parameters=request.parameters,
        task_id=task_id
    )
    
    return AgentTaskResponse(
        task_id=task_id,
        agent_name=request.agent_name,
        status="pending",
        created_at=datetime.utcnow()
    )


@router.post("/collaborate", response_model=AgentCollaborationResponse)
async def create_agent_collaboration(
    request: AgentCollaborationRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """
    Create a multi-agent collaboration for complex tasks.
    
    This endpoint orchestrates multiple agents to work together
    on tasks that require diverse capabilities.
    """
    collaboration_id = str(uuid.uuid4())
    
    try:
        # Initiate collaboration
        result = await orchestrator.coordinate_agents(
            task_description=request.task_description,
            required_agents=request.required_agents,
            strategy=request.coordination_strategy,
            context=request.shared_context
        )
        
        return AgentCollaborationResponse(
            collaboration_id=collaboration_id,
            participating_agents=request.required_agents,
            status="in_progress",
            results=result
        )
    except Exception as e:
        if monitoring_service:
            monitoring_service.log_event(
                event_type="collaboration_failed",
                message=f"Collaboration failed: {str(e)}",
                severity="error",
                details={"collaboration_id": collaboration_id}
            )
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{agent_name}", response_model=AgentStatusResponse)
async def get_agent_status(
    agent_name: str,
    agent_manager: AgentManager = Depends(get_agent_manager)
):
    """
    Get the current status and metrics for a specific agent.
    
    Returns detailed information about an agent's current state,
    capabilities, and performance metrics.
    """
    try:
        agent_info = agent_manager.get_agent_info(agent_name)
        
        if not agent_info:
            raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
        
        return AgentStatusResponse(
            agent_name=agent_name,
            status=agent_info.get("status", "unknown"),
            current_task=agent_info.get("current_task"),
            capabilities=agent_info.get("capabilities", []),
            performance_metrics=agent_info.get("metrics", {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=List[AgentStatusResponse])
async def list_all_agents(
    agent_manager: AgentManager = Depends(get_agent_manager)
):
    """
    List all available agents with their status.
    
    Returns a comprehensive list of all agents in the system
    along with their current status and capabilities.
    """
    try:
        agents = agent_manager.list_agents()
        
        response = []
        for agent in agents:
            response.append(AgentStatusResponse(
                agent_name=agent["name"],
                status=agent.get("status", "unknown"),
                current_task=agent.get("current_task"),
                capabilities=agent.get("capabilities", []),
                performance_metrics=agent.get("metrics", {})
            ))
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/optimize/{agent_name}")
async def optimize_agent(
    agent_name: str,
    optimization_params: Optional[Dict[str, Any]] = None,
    agent_manager: AgentManager = Depends(get_agent_manager)
):
    """
    Trigger optimization for a specific agent.
    
    This endpoint initiates performance optimization for an agent,
    adjusting parameters based on historical performance data.
    """
    try:
        result = await agent_manager.optimize_agent(
            agent_name=agent_name,
            params=optimization_params or {}
        )
        
        if monitoring_service:
            monitoring_service.log_event(
                event_type="agent_optimized",
                message=f"Agent {agent_name} optimization completed",
                details=result
            )
        
        return {
            "agent_name": agent_name,
            "optimization_status": "completed",
            "improvements": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/communication/history/{agent_name}")
async def get_agent_communication_history(
    agent_name: str,
    limit: int = 100,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """
    Get communication history for an agent.
    
    Returns the history of inter-agent communications
    for analysis and debugging purposes.
    """
    try:
        history = orchestrator.get_communication_history(
            agent_name=agent_name,
            limit=limit
        )
        
        return {
            "agent_name": agent_name,
            "communication_count": len(history),
            "communications": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/broadcast")
async def broadcast_to_agents(
    message: str,
    target_agents: Optional[List[str]] = None,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """
    Broadcast a message to multiple agents.
    
    Sends a broadcast message to all agents or a specific subset,
    useful for system-wide notifications or coordinated actions.
    """
    try:
        result = await orchestrator.broadcast_message(
            message=message,
            targets=target_agents
        )
        
        return {
            "broadcast_id": str(uuid.uuid4()),
            "message": message,
            "recipients": result.get("recipients", []),
            "delivery_status": result.get("status", "sent")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Export router
__all__ = ['router']