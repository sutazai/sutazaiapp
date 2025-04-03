"""
Agent Configuration Router

This module provides REST API endpoints for managing agent configurations.
"""

import os
import json
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime

from ai_agents.agent_manager import AgentManager
from ai_agents.dependencies import get_agent_manager
from ai_agents.agent_factory import AgentFactory
from ai_agents.utils.models import AgentConfigCreate, AgentConfigUpdate, AgentConfigResponse
from ai_agents.utils.enums import AgentStatus
from ai_agents.base_agent import BaseAgent

router = APIRouter(prefix="/config", tags=["config"])


# Dependency to get agent manager instance
def get_agent_manager() -> AgentManager:
    """Get the agent manager instance."""
    return AgentManager()


class AgentConfig(BaseModel):
    """Agent configuration model."""

    type: str
    config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class ConfigUpdate(BaseModel):
    """Configuration update model."""

    config: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


@router.get("/agents", response_model=List[str])
async def get_available_agent_types(
    manager: AgentManager = Depends(get_agent_manager),
) -> List[str]:
    """
    Get list of available agent types.

    Args:
        manager: Agent manager instance

    Returns:
        List[str]: List of available agent types
    """
    try:
        return list(manager.factory.agent_classes.keys())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_type}", response_model=Dict[str, Any])
async def get_agent_config_schema(
    agent_type: str, manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, Any]:
    """
    Get configuration schema for an agent type.

    Args:
        agent_type: Type of agent
        manager: Agent manager instance

    Returns:
        Dict[str, Any]: Configuration schema
    """
    try:
        if agent_type not in manager.factory.agent_classes:
            raise HTTPException(
                status_code=404, detail=f"Agent type {agent_type} not found"
            )

        agent_class = manager.factory.agent_classes[agent_type]
        return {
            "type": agent_type,
            "schema": agent_class.get_config_schema(),
            "capabilities": agent_class.get_capabilities(),
            "metadata": {
                "version": agent_class.get_version(),
                "description": agent_class.get_description(),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}/config", response_model=Dict[str, Any])
async def get_agent_config(
    agent_id: str, manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, Any]:
    """
    Get configuration for a specific agent.

    Args:
        agent_id: Agent ID to get config for
        manager: Agent manager instance

    Returns:
        Dict[str, Any]: Agent configuration
    """
    try:
        if agent_id not in manager.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        agent = manager.agents[agent_id]
        return {
            "type": agent.get_type(),
            "config": agent.get_config(),
            "metadata": agent.get_metadata(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/agents/{agent_id}/config", response_model=Dict[str, Any])
async def update_agent_config(
    agent_id: str,
    update: ConfigUpdate,
    manager: AgentManager = Depends(get_agent_manager),
) -> Dict[str, Any]:
    """
    Update configuration for a specific agent.

    Args:
        agent_id: Agent ID to update config for
        update: Configuration update
        manager: Agent manager instance

    Returns:
        Dict[str, Any]: Updated configuration
    """
    try:
        if agent_id not in manager.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        agent = manager.agents[agent_id]

        # Stop the agent if it's running
        if manager.agent_status[agent_id] in ["running", "paused"]:
            manager.stop_agent(agent_id)

        # Update configuration
        agent.update_config(update.config)
        if update.metadata:
            agent.update_metadata(update.metadata)

        # Reinitialize the agent
        agent.initialize()
        manager.agent_status[agent_id] = "ready"

        return {
            "type": agent.get_type(),
            "config": agent.get_config(),
            "metadata": agent.get_metadata(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system", response_model=Dict[str, Any])
async def get_system_config(
    manager: AgentManager = Depends(get_agent_manager),
) -> Dict[str, Any]:
    """
    Get system-wide configuration.

    Args:
        manager: Agent manager instance

    Returns:
        Dict[str, Any]: System configuration
    """
    try:
        return {
            "config_path": manager.config_path,
            "max_retries": manager._max_retries,
            "retry_delay": manager._retry_delay,
            "health_check_interval": manager.health_check.check_interval,
            "thresholds": manager.health_check.thresholds,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/system", response_model=Dict[str, Any])
async def update_system_config(
    config: Dict[str, Any], manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, Any]:
    """
    Update system-wide configuration.

    Args:
        config: Configuration update
        manager: Agent manager instance

    Returns:
        Dict[str, Any]: Updated system configuration
    """
    try:
        # Update retry settings
        if "max_retries" in config:
            manager._max_retries = config["max_retries"]
        if "retry_delay" in config:
            manager._retry_delay = config["retry_delay"]

        # Update health check settings
        if "health_check_interval" in config:
            manager.health_check.check_interval = config["health_check_interval"]
        if "thresholds" in config:
            for check_type, thresholds in config["thresholds"].items():
                manager.health_check.set_threshold(
                    check_type, thresholds["warning"], thresholds["critical"]
                )

        return {
            "config_path": manager.config_path,
            "max_retries": manager._max_retries,
            "retry_delay": manager._retry_delay,
            "health_check_interval": manager.health_check.check_interval,
            "thresholds": manager.health_check.thresholds,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/backup")
async def backup_configurations(
    backup_path: Optional[str] = None,
    manager: AgentManager = Depends(get_agent_manager),
) -> Dict[str, str]:
    """
    Backup all agent configurations.

    Args:
        backup_path: Optional path to save backup
        manager: Agent manager instance

    Returns:
        Dict[str, str]: Backup status
    """
    try:
        if backup_path is None:
            backup_path = os.path.join(
                os.path.dirname(manager.config_path),
                f"agents_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
            )

        # Collect all configurations
        backup_data = {"timestamp": datetime.utcnow().isoformat(), "agents": {}}

        for agent_id, agent in manager.agents.items():
            backup_data["agents"][agent_id] = {
                "type": agent.get_type(),
                "config": agent.get_config(),
                "metadata": agent.get_metadata(),
            }

        # Save backup
        with open(backup_path, "w") as f:
            json.dump(backup_data, f, indent=2)

        return {
            "status": "success",
            "message": f"Configuration backup saved to {backup_path}",
            "backup_path": backup_path,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/restore")
async def restore_configurations(
    backup_path: str, manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, str]:
    """
    Restore agent configurations from backup.

    Args:
        backup_path: Path to backup file
        manager: Agent manager instance

    Returns:
        Dict[str, str]: Restore status
    """
    try:
        if not os.path.exists(backup_path):
            raise HTTPException(
                status_code=404, detail=f"Backup file not found: {backup_path}"
            )

        # Load backup
        with open(backup_path, "r") as f:
            backup_data = json.load(f)

        # Stop all agents
        for agent_id in list(manager.agents.keys()):
            manager.stop_agent(agent_id)

        # Restore configurations
        for agent_id, agent_data in backup_data["agents"].items():
            if agent_id in manager.agents:
                agent = manager.agents[agent_id]
                agent.update_config(agent_data["config"])
                if "metadata" in agent_data:
                    agent.update_metadata(agent_data["metadata"])
                agent.initialize()
                manager.agent_status[agent_id] = "ready"

        return {
            "status": "success",
            "message": "Configuration restore completed",
            "restored_agents": list(backup_data["agents"].keys()),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents", response_model=AgentConfigResponse, status_code=201)
def create_agent_config(
    config_create: AgentConfigCreate,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> AgentConfigResponse:
    """Create a new agent configuration."""
    try:
        config = agent_manager.create_agent_config(config_create.dict())
        return AgentConfigResponse(**config.dict())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create agent config: {e}")


@router.get("/agents/{agent_id}", response_model=AgentConfigResponse)
def get_agent_config(
    agent_id: str,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> AgentConfigResponse:
    """Get the configuration for a specific agent."""
    config = agent_manager.get_agent_config(agent_id)
    if not config:
        raise HTTPException(status_code=404, detail=f"Agent config {agent_id} not found")
    return AgentConfigResponse(**config.dict())


@router.put("/agents/{agent_id}", response_model=AgentConfigResponse)
def update_agent_config(
    agent_id: str,
    agent_factory: AgentFactory = Depends(lambda: AgentFactory(get_agent_manager())),
) -> AgentConfigResponse:
    """Update the configuration for a specific agent."""
    try:
        if agent_id not in agent_factory.agent_classes:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        agent_class = agent_factory.get_agent_class(agent_id)
        config = agent_class.get_config()
        metadata = agent_class.get_metadata()

        # Stop the agent if it's running
        if agent_factory.agent_status[agent_id] in ["running", "paused"]:
            agent_factory.stop_agent(agent_id)

        # Update configuration
        config_update = ConfigUpdate(config=config, metadata=metadata)
        agent_factory.update_agent_config(agent_id, config_update)

        # Reinitialize the agent
        agent_factory.initialize_agent(agent_id)
        agent_factory.agent_status[agent_id] = "ready"

        return AgentConfigResponse(
            type=agent_id,
            config=config,
            metadata=metadata,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/types", response_model=List[Dict[str, str]])
def list_available_agent_types(
    agent_factory: AgentFactory = Depends(lambda: AgentFactory(get_agent_manager()))
) -> List[Dict[str, str]]:
    """List all available agent types and their descriptions."""
    types_info = []
    agent_types = agent_factory.list_available_agent_types()
    for agent_type in agent_types:
        try:
            agent_class = agent_factory.get_agent_class(agent_type)
            description = getattr(agent_class, '__doc__', 'No description available').strip().split('\n')[0]
            types_info.append({"type": agent_type, "description": description})
        except Exception:
            types_info.append({"type": agent_type, "description": "Error loading description"})
    return types_info
