"""
Agent Management Router

This module provides REST API endpoints for managing AI agents.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime

from ..agent_manager import AgentManager

router = APIRouter(prefix="/agents", tags=["agents"])


# Dependency to get agent manager instance
def get_agent_manager() -> AgentManager:
    """Get the agent manager instance."""
    return AgentManager()


class AgentConfig(BaseModel):
    """Agent configuration model."""

    type: str
    config: Optional[Dict[str, Any]] = None


class AgentStatusResponse(BaseModel):
    """Agent status response model."""

    status: str
    metrics: Dict[str, Any]


class AgentMetricsResponse(BaseModel):
    """Agent metrics response model."""

    cpu_percent: float
    memory_percent: float
    last_active: datetime
    execution_count: int
    error_count: int
    avg_execution_time: float


@router.post("/", response_model=str)
async def create_agent(
    config: AgentConfig, manager: AgentManager = Depends(get_agent_manager)
) -> str:
    """
    Create a new agent.

    Args:
        config: Agent configuration
        manager: Agent manager instance

    Returns:
        str: Agent ID
    """
    try:
        return manager.create_agent(config.type, config.config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{agent_id}/start")
async def start_agent(
    agent_id: str, manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, str]:
    """
    Start an agent.

    Args:
        agent_id: Agent ID to start
        manager: Agent manager instance

    Returns:
        Dict[str, str]: Status message
    """
    try:
        manager.start_agent(agent_id)
        return {"status": "success", "message": f"Agent {agent_id} started"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{agent_id}/stop")
async def stop_agent(
    agent_id: str, manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, str]:
    """
    Stop an agent.

    Args:
        agent_id: Agent ID to stop
        manager: Agent manager instance

    Returns:
        Dict[str, str]: Status message
    """
    try:
        manager.stop_agent(agent_id)
        return {"status": "success", "message": f"Agent {agent_id} stopped"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{agent_id}/pause")
async def pause_agent(
    agent_id: str, manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, str]:
    """
    Pause an agent.

    Args:
        agent_id: Agent ID to pause
        manager: Agent manager instance

    Returns:
        Dict[str, str]: Status message
    """
    try:
        manager.pause_agent(agent_id)
        return {"status": "success", "message": f"Agent {agent_id} paused"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{agent_id}/resume")
async def resume_agent(
    agent_id: str, manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, str]:
    """
    Resume a paused agent.

    Args:
        agent_id: Agent ID to resume
        manager: Agent manager instance

    Returns:
        Dict[str, str]: Status message
    """
    try:
        manager.resume_agent(agent_id)
        return {"status": "success", "message": f"Agent {agent_id} resumed"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{agent_id}/status", response_model=AgentStatusResponse)
async def get_agent_status(
    agent_id: str, manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, Any]:
    """
    Get agent status.

    Args:
        agent_id: Agent ID to get status for
        manager: Agent manager instance

    Returns:
        Dict[str, Any]: Agent status information
    """
    try:
        return manager.get_agent_status(agent_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{agent_id}/metrics", response_model=AgentMetricsResponse)
async def get_agent_metrics(
    agent_id: str, manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, Any]:
    """
    Get agent metrics.

    Args:
        agent_id: Agent ID to get metrics for
        manager: Agent manager instance

    Returns:
        Dict[str, Any]: Agent metrics
    """
    try:
        return manager.get_agent_metrics(agent_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/active", response_model=List[str])
async def get_active_agents(
    manager: AgentManager = Depends(get_agent_manager),
) -> List[str]:
    """
    Get list of active agents.

    Args:
        manager: Agent manager instance

    Returns:
        List[str]: List of active agent IDs
    """
    try:
        return manager.get_active_agents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
