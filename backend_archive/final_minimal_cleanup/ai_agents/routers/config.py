"""
Agent Configuration Router

This module provides REST API endpoints for managing agent configurations.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Callable
from fastapi import APIRouter, HTTPException, Depends, Path, Body
from pydantic import BaseModel, Field
from datetime import datetime

from backend.ai_agents.agent_manager import AgentManager
from backend.ai_agents.agent_factory import AgentFactory
from backend.ai_agents.utils.models import AgentConfigCreate, AgentConfigResponse
from backend.ai_agents.agent_status import AgentStatus

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/config", tags=["config"])


# Dependency to get agent manager instance
def get_agent_manager() -> AgentManager:
    """Dependency function to get the AgentManager instance."""
    # In a real app, this would return the singleton instance
    logger.warning("Dependency get_agent_manager is using a placeholder implementation.")
    # Option 1: Raise error if not initialized
    raise NotImplementedError("AgentManager instance not available")
    # Option 2: Return a default/mock instance if suitable for testing/dev
    # return AgentManager() # Requires AgentManager to be initializable like this


def get_agent_factory() -> AgentFactory:
    """Dependency function to get the AgentFactory instance."""
    logger.warning("Dependency get_agent_factory is using a placeholder implementation.")
    raise NotImplementedError("AgentFactory instance not available")
    # return AgentFactory() # Placeholder


class AgentConfig(BaseModel):
    """Agent configuration model."""

    type: str
    config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "type": "ExampleAgent",
                    "config": {"setting1": "value1"},
                    "metadata": {"version": "1.0"}
                }
            ]
        }
    }


class ConfigUpdate(BaseModel):
    """Configuration update model."""

    config: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None


class ConfigResponse(BaseModel):
    config: Dict[str, Any]


class AvailableAgentsResponse(BaseModel):
    agent_types: List[str]


class AgentSchemaResponse(BaseModel):
    schema_: Dict[str, Any] = Field(..., alias="schema")


class AgentDetailsResponse(BaseModel):
    type: str
    schema: Dict[str, Any]
    capabilities: List[str]
    version: str
    description: str


class AgentStatusResponse(BaseModel):
    status: str


@router.get("/agents/available", response_model=AvailableAgentsResponse)
def list_available_agent_types(agent_manager: AgentManager = Depends(get_agent_manager)):
    """List all available agent types that can be created."""
    try:
        agent_types = agent_manager.factory.get_available_agents()
        return AvailableAgentsResponse(agent_types=agent_types)
    except AttributeError:
        logger.error("AgentFactory or get_available_agents method not found on AgentManager.")
        raise HTTPException(status_code=501, detail="Agent type listing not available.")
    except Exception as e:
        logger.error(f"Error listing available agent types: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list available agent types")


@router.get("/config/schema/{agent_type}", response_model=Dict[str, Any])
def get_agent_config_schema(agent_type: str, agent_manager: AgentManager = Depends(get_agent_manager)):
    """Get the configuration schema for a specific agent type."""
    schema: Any = None
    try:
        agent_class = agent_manager.factory.agent_classes.get(agent_type)
        if not agent_class:
            raise HTTPException(status_code=404, detail=f"Agent type {agent_type} not found")
        if hasattr(agent_class, 'get_config_schema'):
            schema = agent_class.get_config_schema()
        else:
            schema = {}
        if schema is None:
            raise HTTPException(status_code=404, detail=f"Schema not found for agent type: {agent_type}")
        return schema
    except AttributeError:
        logger.error(f"AgentFactory or get_agent_class method not found for {agent_type}.")
        raise HTTPException(status_code=501, detail="Agent schema retrieval not available.")
    except Exception as e:
        logger.error(f"Error getting config schema for {agent_type}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/capabilities/{agent_type}", response_model=Dict[str, Any])
def get_agent_capabilities(agent_type: str, agent_manager: AgentManager = Depends(get_agent_manager)):
    """Get the capabilities for a specific agent type."""
    capabilities: Any = None
    try:
        agent_class = agent_manager.factory.agent_classes.get(agent_type)
        if not agent_class:
            raise HTTPException(status_code=404, detail=f"Agent type {agent_type} not found")
        if hasattr(agent_class, 'get_capabilities'):
            # Ignore call-arg error for now, acknowledge potential runtime issue
            capabilities = agent_class.get_capabilities() # type: ignore[call-arg]
        else:
            capabilities = []
        if capabilities is None:
            raise HTTPException(status_code=404, detail=f"Capabilities not found for agent type: {agent_type}")
        return {"capabilities": capabilities}
    except AttributeError:
        logger.error(f"AgentFactory or get_agent_class method not found for {agent_type}.")
        raise HTTPException(status_code=501, detail="Agent capability retrieval not available.")
    except Exception as e:
        logger.error(f"Error getting capabilities for {agent_type}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/agents/{agent_type}/details", response_model=AgentDetailsResponse)
def get_agent_type_details(agent_type: str, agent_manager: AgentManager = Depends(get_agent_manager)):
    """Get detailed information about a specific agent type (schema, capabilities, etc.)."""
    try:
        agent_class = agent_manager.factory.agent_classes.get(agent_type)
        if not agent_class:
            raise HTTPException(status_code=404, detail=f"Agent type {agent_type} not found")

        schema_func: Callable[[], Dict[str, Any]] = getattr(agent_class, 'get_config_schema', lambda: {})
        capabilities_func: Callable[[], List[str]] = getattr(agent_class, 'get_capabilities', lambda: [])
        version_func: Callable[[], str] = getattr(agent_class, 'get_version', lambda: "N/A")
        description_func: Callable[[], str] = getattr(agent_class, 'get_description', lambda: "No description")

        schema: Dict[str, Any] = schema_func()
        capabilities: List[str] = capabilities_func()
        version: str = version_func()
        description: str = description_func()

        return AgentDetailsResponse(
            type=agent_type,
            schema=schema,
            capabilities=capabilities,
            version=version,
            description=description,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except AttributeError:
        logger.error(f"AgentFactory or get_agent_class method not found for {agent_type}.")
        raise HTTPException(status_code=501, detail="Agent detail retrieval not available.")
    except Exception as e:
        logger.error(f"Error getting details for agent type {agent_type}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get agent details")


@router.get("/agents/{agent_id}/config", response_model=ConfigResponse)
def get_agent_config_route(
    agent_id: str = Path(...),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> ConfigResponse:
    """Get the current configuration of a specific agent."""
    try:
        if agent_id not in agent_manager.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        agent_instance = agent_manager.agents[agent_id]
        config = agent_instance.config.model_dump()
        return ConfigResponse(config=config)
    except AttributeError:
        logger.error(f"Agent instance {agent_id} found, but lacks 'config' attribute or model_dump method.")
        raise HTTPException(status_code=500, detail="Failed to retrieve agent configuration format.")
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting config for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get agent configuration")


@router.put("/agents/{agent_id}/config", response_model=ConfigResponse)
def update_agent_config_route(
    agent_id: str = Path(...),
    config_update: ConfigUpdate = Body(...),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> ConfigResponse:
    """Update the configuration of a specific agent."""
    try:
        if not hasattr(agent_manager, 'update_agent_config'):
            raise HTTPException(status_code=501, detail="update_agent_config not implemented in AgentManager")
        updated_config_dict = agent_manager.update_agent_config(agent_id, config_update.config, config_update.metadata)
        return ConfigResponse(config=updated_config_dict)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating config for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update agent configuration")


@router.get("/agents/{agent_id}/status", response_model=AgentStatusResponse)
def get_agent_status_route(
    agent_id: str = Path(...),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> AgentStatusResponse:
    """Get the current status of a specific agent."""
    try:
        status_dict = agent_manager.get_agent_status(agent_id)
        status_str = status_dict.get("status", "UNKNOWN")
        if hasattr(status_str, 'value'):
            status_str = status_str.value
        return AgentStatusResponse(status=str(status_str))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting status for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get agent status")


@router.get("/system", response_model=Dict[str, Any])
async def get_system_config(
    manager: AgentManager = Depends(get_agent_manager),
) -> Dict[str, Any]:
    """
    Get system-wide configuration.
    """
    try:
        max_retries = getattr(manager, '_max_retries', None)
        retry_delay = getattr(manager, '_retry_delay', None)
        health_check_interval = getattr(getattr(manager, 'health_check', None), 'check_interval', None)
        thresholds = getattr(getattr(manager, 'health_check', None), 'thresholds', None)
        config_path = getattr(manager, 'config_path', None)

        return {
            "config_path": config_path,
            "max_retries": max_retries,
            "retry_delay": retry_delay,
            "health_check_interval": health_check_interval,
            "thresholds": thresholds,
        }
    except Exception as e:
        logger.error(f"Error getting system config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve system config: {e}")


@router.put("/system", response_model=Dict[str, Any])
async def update_system_config(
    config: Dict[str, Any], manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, Any]:
    """
    Update system-wide configuration.
    """
    try:
        updated_values = {}
        if hasattr(manager, '_max_retries') and "max_retries" in config:
            manager._max_retries = config["max_retries"]
            updated_values["max_retries"] = manager._max_retries
        if hasattr(manager, '_retry_delay') and "retry_delay" in config:
            manager._retry_delay = config["retry_delay"]
            updated_values["retry_delay"] = manager._retry_delay

        if hasattr(manager, 'health_check'):
            if "health_check_interval" in config:
                manager.health_check.check_interval = config["health_check_interval"]
                updated_values["health_check_interval"] = manager.health_check.check_interval
            if "thresholds" in config and isinstance(config["thresholds"], dict):
                for check_type, thresholds in config["thresholds"].items():
                    if isinstance(thresholds, dict) and "warning" in thresholds and "critical" in thresholds:
                        manager.health_check.set_threshold(
                            check_type, thresholds["warning"], thresholds["critical"]
                        )
                updated_values["thresholds"] = manager.health_check.thresholds # type: ignore[assignment]

        current_config = await get_system_config(manager)
        current_config.update(updated_values)
        return current_config

    except AttributeError as e:
        logger.error(f"Error updating system config due to missing attribute: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid system component for update: {e}")
    except Exception as e:
        logger.error(f"Error updating system config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update system config: {e}")


@router.post("/backup")
async def backup_configurations(
    backup_path: Optional[str] = None,
    manager: AgentManager = Depends(get_agent_manager),
) -> Dict[str, str]:
    """
    Backup all agent configurations.
    """
    try:
        config_path = getattr(manager, 'config_path', None)
        if backup_path is None:
            if config_path:
                backup_dir = os.path.dirname(config_path) or "."
            else:
                backup_dir = "."
            backup_path = os.path.join(
                backup_dir,
                f"agents_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
            )

        backup_data = {"timestamp": datetime.utcnow().isoformat(), "agents": {}}

        for agent_id, agent_instance in manager.agents.items():
            if hasattr(agent_instance, 'config'):
                agent_config_dict = agent_instance.config.model_dump()
                if 'type' not in agent_config_dict and hasattr(agent_instance.config, 'type'):
                    agent_config_dict['type'] = agent_instance.config.type
                backup_data["agents"][agent_id] = agent_config_dict # type: ignore[index]
            else:
                logger.warning(f"Agent {agent_id} instance lacks config attribute, skipping backup.")

        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        with open(backup_path, "w") as f:
            json.dump(backup_data, f, indent=2)

        return {
            "status": "success",
            "message": f"Configuration backup saved to {backup_path}",
            "backup_path": backup_path,
        }
    except AttributeError as e:
        logger.error(f"Error backing up config due to missing attribute: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to backup config: {e}")
    except Exception as e:
        logger.error(f"Error backing up config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to backup config: {e}")


@router.post("/restore")
async def restore_configurations(
    backup_path: str, manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, Any]:
    """
    Restore agent configurations from backup.
    """
    try:
        if not os.path.exists(backup_path):
            raise HTTPException(
                status_code=404, detail=f"Backup file not found: {backup_path}"
            )

        with open(backup_path, "r") as f:
            backup_data = json.load(f)

        if "agents" not in backup_data:
            raise HTTPException(status_code=400, detail="Invalid backup file format: missing 'agents' key")

        restored_agents = []
        failed_agents = {}
        for agent_id, agent_config_dict in backup_data["agents"].items():
            try:
                if agent_id in manager.agents:
                    logger.info(f"Updating existing agent {agent_id} from backup...")
                    if hasattr(manager, 'update_agent_config'):
                        manager.update_agent_config(agent_id, agent_config_dict)
                        restored_agents.append(agent_id)
                    else:
                        logger.warning(f"AgentManager has no update_agent_config method. Cannot update {agent_id}.")
                        failed_agents[agent_id] = "Update method missing"
                else:
                    logger.info(f"Creating new agent {agent_id} from backup...")
                    agent_type = agent_config_dict.get('type')
                    if not agent_type:
                        raise ValueError("Missing 'type' in backup config")
                    manager.create_agent(agent_type, config=agent_config_dict)
                    restored_agents.append(agent_id)
                if agent_id in manager.agent_status:
                    manager.agent_status[agent_id] = AgentStatus.READY

            except Exception as restore_e:
                logger.error(f"Failed to restore/create agent {agent_id}: {restore_e}")
                failed_agents[agent_id] = str(restore_e)

        message = f"Configuration restore completed. Restored: {len(restored_agents)}. Failed: {len(failed_agents)}."
        if failed_agents:
            message += f" Failures: {failed_agents}"

        return {
            "status": "success" if not failed_agents else "partial_success",
            "message": message,
            "restored_agents": restored_agents,
            "failed_agents": failed_agents,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error restoring config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to restore config: {e}")


@router.post("/agents", response_model=AgentConfigResponse, status_code=201)
def create_agent_config(
    config_create: AgentConfigCreate,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> AgentConfigResponse:
    """Create a new agent configuration."""
    try:
        logger.warning("create_agent_config endpoint is likely redundant/incorrect.")
        raise HTTPException(status_code=501, detail="Not Implemented: Config creation separate from agent instantiation.")
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
    try:
        if agent_id not in agent_manager.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
        agent_instance = agent_manager.agents[agent_id]
        config = agent_instance.config.model_dump()
        if not config:
            raise HTTPException(status_code=404, detail=f"Agent config {agent_id} not found")
        return AgentConfigResponse(**config)
    except AttributeError:
        raise HTTPException(status_code=500, detail="Failed to retrieve agent configuration format.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agent config: {e}")


@router.put("/agents/{agent_id}", response_model=AgentConfigResponse)
def update_agent_config(
    agent_id: str,
) -> AgentConfigResponse:
    """Update the configuration for a specific agent."""
    logger.warning(f"PUT /config/agents/{agent_id} called - this endpoint seems incorrectly defined.")
    raise HTTPException(status_code=501, detail="Endpoint deprecated or incorrectly defined. Use PUT /agents/{agent_id}/config instead.")


@router.get("/types", response_model=List[str])
def list_agent_types(
    agent_manager: AgentManager = Depends(get_agent_manager)
):
    """List all available agent types."""
    try:
        agent_types = agent_manager.factory.get_available_agents()
        return agent_types
    except AttributeError:
        logger.error("AgentFactory or get_available_agents method not found on AgentManager.")
        raise HTTPException(status_code=501, detail="Agent type listing not available.")
    except Exception as e:
        logger.error(f"Error listing agent types: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
