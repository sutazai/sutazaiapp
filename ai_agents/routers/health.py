"""
Agent Health Router

This module provides REST API endpoints for monitoring agent health and system status.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime

from ai_agents.agent_manager import AgentManager
from ai_agents.dependencies import get_agent_manager, get_health_check
from ai_agents.utils.health_check import HealthCheck, HealthStatus, AgentHealthStatus, SystemHealthStatus

router = APIRouter(prefix="/health", tags=["health"])


# Dependency to get agent manager instance
def get_agent_manager() -> AgentManager:
    """Get the agent manager instance."""
    return AgentManager()


class HealthCheck(BaseModel):
    """Health check model."""

    check_id: str
    name: str
    status: str
    message: str
    details: Optional[Dict[str, Any]] = None
    last_check: datetime


class AgentHealth(BaseModel):
    """Agent health model."""

    agent_id: str
    status: str
    checks: List[HealthCheck]
    last_update: datetime


class SystemHealth(BaseModel):
    """System health model."""

    status: str
    checks: List[HealthCheck]
    agents: List[AgentHealth]
    last_update: datetime


@router.get("/", response_model=HealthStatus)
def get_overall_health(
    agent_manager: AgentManager = Depends(get_agent_manager),
    health_checker: HealthCheck = Depends(get_health_check)
) -> HealthStatus:
    """Get the overall health status of the agent system."""
    return health_checker.get_overall_status()


@router.get("/agents", response_model=Dict[str, AgentHealthStatus])
def get_all_agents_health(
    agent_manager: AgentManager = Depends(get_agent_manager),
    health_checker: HealthCheck = Depends(get_health_check)
) -> Dict[str, AgentHealthStatus]:
    """Get health status for all agents."""
    return health_checker.get_all_agent_statuses()


@router.get("/agents/{agent_id}", response_model=AgentHealthStatus)
def get_agent_health(
    agent_id: str,
    agent_manager: AgentManager = Depends(get_agent_manager),
    health_checker: HealthCheck = Depends(get_health_check)
) -> AgentHealthStatus:
    """Get health status for a specific agent."""
    health_status = health_checker.check_agent_health(agent_id)
    if not health_status:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    return health_status


@router.get("/system", response_model=SystemHealthStatus)
def get_system_health(
    health_checker: HealthCheck = Depends(get_health_check)
) -> SystemHealthStatus:
    """Get health status of system components (dependencies, resources)."""
    return health_checker.check_system_health()


@router.post("/check/agent/{agent_id}", response_model=AgentHealthStatus)
def run_agent_health_check(
    agent_id: str,
    health_checker: HealthCheck = Depends(get_health_check)
) -> AgentHealthStatus:
    """Manually trigger a health check for a specific agent."""
    health_status = health_checker.check_agent_health(agent_id, force_check=True)
    if not health_status:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    return health_status


@router.post("/check/system", response_model=SystemHealthStatus)
def run_system_health_check(
    health_checker: HealthCheck = Depends(get_health_check)
) -> SystemHealthStatus:
    """Manually trigger a health check for system components."""
    return health_checker.check_system_health(force_check=True)


# Example: Endpoint to get detailed dependency status
@router.get("/dependencies", response_model=Dict[str, Any])
def get_dependency_status(
     health_checker: HealthCheck = Depends(get_health_check)
) -> Dict[str, Any]:
     """Get detailed status of external dependencies."""
     # Assuming HealthCheck has a method for this
     if hasattr(health_checker, 'check_dependencies'):
         return health_checker.check_dependencies()
     else:
         return {"status": "ok", "detail": "Dependency check method not implemented"}


@router.get("/checks", response_model=List[HealthCheck])
async def get_available_health_checks(
    manager: AgentManager = Depends(get_agent_manager),
) -> List[HealthCheck]:
    """
    Get list of available health checks.

    Args:
        manager: Agent manager instance

    Returns:
        List[HealthCheck]: List of available health checks
    """
    try:
        checks = []
        for check_name, check_func in manager.health_check.checks.items():
            result = check_func()
            checks.append(
                HealthCheck(
                    check_id=check_name,
                    name=check_name,
                    status=result["status"],
                    message=result["message"],
                    details=result.get("details"),
                    last_check=datetime.utcnow(),
                )
            )
        return checks
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
