"""
Agent Health Router

This module provides REST API endpoints for monitoring agent health and system status.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime

from ..agent_manager import AgentManager
from ..health_check import HealthStatus

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


@router.get("/agents", response_model=List[AgentHealth])
async def get_all_agent_health(
    manager: AgentManager = Depends(get_agent_manager),
) -> List[AgentHealth]:
    """
    Get health status for all agents.

    Args:
        manager: Agent manager instance

    Returns:
        List[AgentHealth]: List of agent health statuses
    """
    try:
        health_statuses = []
        for agent_id, agent in manager.agents.items():
            checks = manager.health_check.check_agent(agent_id)
            status = (
                "healthy"
                if all(c["status"] == HealthStatus.OK for c in checks)
                else "unhealthy"
            )

            health_statuses.append(
                AgentHealth(
                    agent_id=agent_id,
                    status=status,
                    checks=[
                        HealthCheck(
                            check_id=f"{agent_id}_{i}",
                            name=check["name"],
                            status=check["status"],
                            message=check["message"],
                            details=check.get("details"),
                            last_check=datetime.utcnow(),
                        )
                        for i, check in enumerate(checks)
                    ],
                    last_update=datetime.utcnow(),
                )
            )
        return health_statuses
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}", response_model=AgentHealth)
async def get_agent_health(
    agent_id: str, manager: AgentManager = Depends(get_agent_manager)
) -> AgentHealth:
    """
    Get health status for a specific agent.

    Args:
        agent_id: Agent ID to get health for
        manager: Agent manager instance

    Returns:
        AgentHealth: Agent health status
    """
    try:
        if agent_id not in manager.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        checks = manager.health_check.check_agent(agent_id)
        status = (
            "healthy"
            if all(c["status"] == HealthStatus.OK for c in checks)
            else "unhealthy"
        )

        return AgentHealth(
            agent_id=agent_id,
            status=status,
            checks=[
                HealthCheck(
                    check_id=f"{agent_id}_{i}",
                    name=check["name"],
                    status=check["status"],
                    message=check["message"],
                    details=check.get("details"),
                    last_check=datetime.utcnow(),
                )
                for i, check in enumerate(checks)
            ],
            last_update=datetime.utcnow(),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system", response_model=SystemHealth)
async def get_system_health(
    manager: AgentManager = Depends(get_agent_manager),
) -> SystemHealth:
    """
    Get system-wide health status.

    Args:
        manager: Agent manager instance

    Returns:
        SystemHealth: System health status
    """
    try:
        # Get system checks
        system_checks = manager.health_check.check_system()

        # Get agent health statuses
        agent_health = []
        for agent_id, agent in manager.agents.items():
            checks = manager.health_check.check_agent(agent_id)
            status = (
                "healthy"
                if all(c["status"] == HealthStatus.OK for c in checks)
                else "unhealthy"
            )

            agent_health.append(
                AgentHealth(
                    agent_id=agent_id,
                    status=status,
                    checks=[
                        HealthCheck(
                            check_id=f"{agent_id}_{i}",
                            name=check["name"],
                            status=check["status"],
                            message=check["message"],
                            details=check.get("details"),
                            last_check=datetime.utcnow(),
                        )
                        for i, check in enumerate(checks)
                    ],
                    last_update=datetime.utcnow(),
                )
            )

        # Determine overall system status
        all_checks = system_checks + [
            check for agent in agent_health for check in agent.checks
        ]
        status = (
            "healthy"
            if all(c["status"] == HealthStatus.OK for c in all_checks)
            else "unhealthy"
        )

        return SystemHealth(
            status=status,
            checks=[
                HealthCheck(
                    check_id=f"system_{i}",
                    name=check["name"],
                    status=check["status"],
                    message=check["message"],
                    details=check.get("details"),
                    last_check=datetime.utcnow(),
                )
                for i, check in enumerate(system_checks)
            ],
            agents=agent_health,
            last_update=datetime.utcnow(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/{agent_id}/check")
async def run_agent_health_check(
    agent_id: str, manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, str]:
    """
    Run health check for a specific agent.

    Args:
        agent_id: Agent ID to check
        manager: Agent manager instance

    Returns:
        Dict[str, str]: Check status
    """
    try:
        if agent_id not in manager.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        # Run health check
        checks = manager.health_check.check_agent(agent_id)
        status = (
            "healthy"
            if all(c["status"] == HealthStatus.OK for c in checks)
            else "unhealthy"
        )

        return {
            "status": "success",
            "message": f"Health check completed for agent {agent_id}",
            "agent_status": status,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system/check")
async def run_system_health_check(
    manager: AgentManager = Depends(get_agent_manager),
) -> Dict[str, str]:
    """
    Run system-wide health check.

    Args:
        manager: Agent manager instance

    Returns:
        Dict[str, str]: Check status
    """
    try:
        # Run system health check
        system_checks = manager.health_check.check_system()

        # Run agent health checks
        agent_checks = []
        for agent_id in manager.agents:
            checks = manager.health_check.check_agent(agent_id)
            agent_checks.extend(checks)

        # Determine overall status
        all_checks = system_checks + agent_checks
        status = (
            "healthy"
            if all(c["status"] == HealthStatus.OK for c in all_checks)
            else "unhealthy"
        )

        return {
            "status": "success",
            "message": "System health check completed",
            "system_status": status,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
