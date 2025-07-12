from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import asyncio
from uuid import uuid4

from agents import agent_orchestrator
from agents.base_agent import BaseAgent, Task, TaskPriority, AgentStatus
from api.auth import get_current_user, require_admin
from api.database import db_manager

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/")
async def list_agents(current_user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """List all registered agents and their status."""
    try:
        agents_info = []
        for agent_id, agent in agent_orchestrator.agents.items():
            agent_info = {
                "id": agent_id,
                "name": getattr(agent, 'name', agent.__class__.__name__),
                "type": agent.__class__.__name__,
                "status": getattr(agent, 'status', 'unknown'),
                "capabilities": getattr(agent, 'capabilities', []),
                "tasks_completed": getattr(agent, 'tasks_completed', 0),
                "tasks_failed": getattr(agent, 'tasks_failed', 0),
                "created_at": getattr(agent, 'created_at', datetime.utcnow()).isoformat()
            }
            agents_info.append(agent_info)
        
        await db_manager.log_system_event(
            "info", "agents", "Listed agents", 
            {"user": current_user.get("username"), "agent_count": len(agents_info)}
        )
        
        return {
            "agents": agents_info,
            "total": len(agents_info),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error listing agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{agent_id}")
async def get_agent(agent_id: str, current_user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Get details about a specific agent."""
    try:
        if agent_id not in agent_orchestrator.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        agent = agent_orchestrator.agents[agent_id]
        agent_info = {
            "id": agent_id,
            "name": getattr(agent, 'name', agent.__class__.__name__),
            "type": agent.__class__.__name__,
            "status": getattr(agent, 'status', 'unknown'),
            "capabilities": getattr(agent, 'capabilities', []),
            "tasks_completed": getattr(agent, 'tasks_completed', 0),
            "tasks_failed": getattr(agent, 'tasks_failed', 0),
            "total_runtime": getattr(agent, 'total_runtime', 0),
            "configuration": getattr(agent, 'configuration', {}),
            "created_at": getattr(agent, 'created_at', datetime.utcnow()).isoformat()
        }
        
        return agent_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{agent_id}/tasks")
async def submit_task_to_agent(
    agent_id: str,
    task_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
) -> Dict[str, Any]:
    """Submit a task to a specific agent."""
    try:
        if agent_id not in agent_orchestrator.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        
        # Create task
        task = Task(
            id=str(uuid4()),
            name=task_data.get("name", "Unnamed Task"),
            description=task_data.get("description", ""),
            priority=TaskPriority(task_data.get("priority", "medium")),
            agent_type=agent_id,
            parameters=task_data.get("parameters", {}),
            created_at=datetime.utcnow(),
            status=AgentStatus.PENDING
        )
        
        # Submit task
        task_id = await agent_orchestrator.submit_task(task)
        
        await db_manager.log_system_event(
            "info", "agents", f"Task submitted to agent {agent_id}",
            {"user": current_user.get("username"), "task_id": task_id, "agent_id": agent_id}
        )
        
        return {
            "task_id": task_id,
            "agent_id": agent_id,
            "status": "submitted",
            "timestamp": datetime.utcnow().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error submitting task to agent {agent_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/orchestrator/status")
async def get_orchestrator_status(current_user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Get the status of the agent orchestrator."""
    try:
        status = agent_orchestrator.get_orchestrator_status()
        return status
    except Exception as e:
        logger.error(f"Error getting orchestrator status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/orchestrator/start")
async def start_orchestrator(current_user: dict = Depends(require_admin)) -> Dict[str, Any]:
    """Start the agent orchestrator (admin only)."""
    try:
        if hasattr(agent_orchestrator, 'start'):
            await agent_orchestrator.start()
        
        await db_manager.log_system_event(
            "info", "agents", "Orchestrator started",
            {"user": current_user.get("username")}
        )
        
        return {
            "status": "started",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error starting orchestrator: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/orchestrator/stop")
async def stop_orchestrator(current_user: dict = Depends(require_admin)) -> Dict[str, Any]:
    """Stop the agent orchestrator (admin only)."""
    try:
        if hasattr(agent_orchestrator, 'stop'):
            await agent_orchestrator.stop()
        
        await db_manager.log_system_event(
            "info", "agents", "Orchestrator stopped",
            {"user": current_user.get("username")}
        )
        
        return {
            "status": "stopped",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error stopping orchestrator: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/capabilities")
async def get_agent_capabilities(current_user: dict = Depends(get_current_user)) -> Dict[str, Any]:
    """Get aggregated capabilities of all agents."""
    try:
        all_capabilities = set()
        for agent in agent_orchestrator.agents.values():
            capabilities = getattr(agent, 'capabilities', [])
            all_capabilities.update(capabilities)
        
        return {
            "capabilities": list(all_capabilities),
            "agents_count": len(agent_orchestrator.agents),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting agent capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))
