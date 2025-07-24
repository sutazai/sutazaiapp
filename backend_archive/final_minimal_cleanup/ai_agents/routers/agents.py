"""
Agent Management Router

This module provides REST API endpoints for managing AI agents.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Path, Body, status
from pydantic import BaseModel
from datetime import datetime

from backend.ai_agents.agent_manager import AgentManager
from backend.models.agent_models import AgentStatusResponse as AgentStatusResponseModel

# Assuming these Pydantic models are defined elsewhere (e.g., schemas or models file)
class AgentInfo(BaseModel):
    id: str
    type: str
    status: str
    config: Dict[str, Any]

class AgentConfigCreate(BaseModel):
    type: str
    config: Optional[Dict[str, Any]] = None

class TaskInput(BaseModel):
    task_data: Dict[str, Any]
    timeout: Optional[int] = None

class TaskOutput(BaseModel):
    result: Dict[str, Any]
    status: str


# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])


# Dependency to get agent manager instance
def get_agent_manager() -> AgentManager:
    """Get the agent manager instance (dependency)."""
    # Replace with actual dependency injection logic
    # ...
    logging.warning("Dependency get_agent_manager is using a placeholder implementation.")
    raise NotImplementedError("AgentManager dependency not properly configured.")
    # return AgentManager() # type: ignore[no-any-return] # Keep raise for now


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


@router.post("/", response_model=AgentInfo, status_code=status.HTTP_201_CREATED)
def create_agent(
    agent_config: AgentConfigCreate,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> AgentInfo:
    """Create a new agent."""
    try:
        agent_id = agent_manager.create_agent(agent_config.type, agent_config.config)
        agent_status = agent_manager.get_agent_status(agent_id)
        # Construct AgentInfo response
        return AgentInfo(
            id=agent_id,
            type=agent_config.type, # Or get from agent_status if available
            status=agent_status.get("status", "UNKNOWN"), # Get status safely
            config=agent_status.get("config", {}) # Get config safely
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating agent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create agent")


@router.get("/", response_model=List[AgentInfo])
def list_agents(
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> List[AgentInfo]:
    """List all active agents."""
    try:
        agent_ids = agent_manager.get_active_agents()
        agent_infos = []
        for agent_id in agent_ids:
            try:
                status = agent_manager.get_agent_status(agent_id)
                agent_infos.append(AgentInfo(
                    id=agent_id,
                    type=status.get("type", "UNKNOWN"),
                    status=status.get("status", "UNKNOWN"),
                    config=status.get("config", {})
                ))
            except ValueError:
                 logger.warning(f"Could not get status for listed agent {agent_id}")
                 continue # Skip agent if status retrieval fails
        return agent_infos
    except Exception as e:
        logger.error(f"Error listing agents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list agents")


@router.get("/{agent_id}", response_model=AgentInfo)
def get_agent_info(
    agent_id: str = Path(...),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> AgentInfo:
    """Get information about a specific agent."""
    try:
        status = agent_manager.get_agent_status(agent_id)
        return AgentInfo(
            id=agent_id,
            type=status.get("type", "UNKNOWN"),
            status=status.get("status", "UNKNOWN"),
            config=status.get("config", {})
        )
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    except Exception as e:
        logger.error(f"Error getting agent info for {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get agent information")


@router.post("/{agent_id}/start")
def start_agent(
    agent_id: str = Path(...),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, str]:
    """Start a specific agent."""
    try:
        agent_manager.start_agent(agent_id)
        return {"status": "success", "message": f"Agent {agent_id} started"}
    except ValueError as e:
        raise HTTPException(status_code=404 if "not found" in str(e).lower() else 400, detail=str(e))
    except Exception as e:
        logger.error(f"Error starting agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to start agent")


@router.post("/{agent_id}/stop")
def stop_agent(
    agent_id: str = Path(...),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, str]:
    """Stop a specific agent."""
    try:
        agent_manager.stop_agent(agent_id)
        return {"status": "success", "message": f"Agent {agent_id} stopped"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error stopping agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to stop agent")


@router.post("/{agent_id}/execute", response_model=TaskOutput)
def execute_agent_task(
    agent_id: str = Path(...),
    task_input: TaskInput = Body(...),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> TaskOutput:
    """Execute a task on a specific agent."""
    try:
        result = agent_manager.execute_task(agent_id, task_input.task_data, task_input.timeout)
        # Assuming execute_task returns a dict like {"result": ..., "status": ...}
        return TaskOutput(**result)
    except ValueError as e: # Agent not found or not running
        raise HTTPException(status_code=404 if "not found" in str(e).lower() else 400, detail=str(e))
    except TimeoutError as e:
         raise HTTPException(status_code=408, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing task on agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to execute task")


@router.delete("/{agent_id}")
def terminate_agent(
    agent_id: str = Path(...),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, str]:
    """Terminate and remove an agent."""
    try:
        # Assuming AgentManager needs a terminate_agent method
        # If stop_agent also removes, use that. Otherwise implement terminate.
        # For now, using stop as placeholder for removal.
        agent_manager.stop_agent(agent_id)
        # Add logic to remove agent from self.agents dict etc. if stop doesn't do it.
        # Example: agent_manager.remove_agent(agent_id)
        return {"status": "success", "message": f"Agent {agent_id} terminated"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error terminating agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to terminate agent")


@router.get("/{agent_id}/status", response_model=AgentStatusResponseModel)
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
        status_dict = manager.get_agent_status(agent_id)
        return status_dict
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
