"""
Agent Health Router

This module provides REST API endpoints for monitoring agent health and system status.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel
from datetime import datetime

from ai_agents.agent_manager import AgentManager
from ai_agents.dependencies import get_agent_manager
from ai_agents.health_check import HealthStatus

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/health", tags=["health"])


# --- Pydantic Models ---
# Renamed from HealthStatus to avoid conflict with the constants class
class HealthStatusResponse(BaseModel):
    status: str # ok, warning, critical
    components: Dict[str, Any] # Simplified: should reflect actual check results
    timestamp: datetime


class ComponentStatus(BaseModel):
    status: str
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None


class AgentHealth(BaseModel):
    agent_id: str
    status: str # ok, warning, critical, unknown
    last_check: Optional[datetime] = None
    details: Optional[Dict[str, Any]] = None

# Define SystemHealthStatus if not imported
class SystemHealthStatus(BaseModel):
    overall_status: str
    checks: Dict[str, ComponentStatus]
    timestamp: datetime

# Define AgentHealthStatus if not imported
class AgentHealthStatus(BaseModel):
    agent_id: str
    status: str
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime

# Define HealthCheckInfo model for the /checks endpoint
class HealthCheckInfo(BaseModel):
    check_id: str
    description: Optional[str] = None
    status: str
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    last_check: Optional[datetime] = None

# --- Helper Function ---
def _get_health_checker(agent_manager: AgentManager) -> Any: # Use Any temporarily for HealthCheck
    if not hasattr(agent_manager, 'health_check') or agent_manager.health_check is None:
        raise HTTPException(status_code=501, detail="Health check service not available")
    return agent_manager.health_check

# --- Endpoints ---

@router.get("/", response_model=HealthStatusResponse)
def get_overall_health(
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> HealthStatusResponse:
    """Get the overall health status of the agent system by aggregating check results."""
    health_checker = _get_health_checker(agent_manager)
    try:
        # Use get_status() instead of get_overall_status()
        health_status_data = health_checker.get_status()

        # Adapt the response to HealthStatusResponse model
        return HealthStatusResponse(
            status=health_status_data.get('status', HealthStatus.UNKNOWN),
            # Pass the full check results as components for now
            components=health_status_data.get('checks', {}),
            timestamp=datetime.fromisoformat(health_status_data.get('timestamp')) if health_status_data.get('timestamp') else datetime.utcnow()
        )
    except Exception as e:
        logger.error(f"Error retrieving overall health status: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Failed to retrieve health status")


@router.get("/components/{component_name}", response_model=ComponentStatus)
def get_component_health(
    component_name: str,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> ComponentStatus:
    """Get the health status of a specific registered check/component."""
    health_checker = _get_health_checker(agent_manager)
    try:
        # Get status from results, not check_component()
        component_result = health_checker.results.get(component_name)
        if component_result is None:
             # Check if it's a valid check ID even if no result yet
             if component_name not in health_checker.checks:
                 raise HTTPException(status_code=404, detail=f"Component '{component_name}' not found or not monitored")
             else:
                 # Return Unknown if check exists but hasn't run
                 return ComponentStatus(status=HealthStatus.UNKNOWN, message="Check has not run yet", timestamp=datetime.utcnow())

        # Adapt the stored result to ComponentStatus model
        return ComponentStatus(
            status=component_result.get('status', HealthStatus.UNKNOWN),
            message=component_result.get('message'),
            details=component_result.get('details'),
            timestamp=datetime.fromisoformat(component_result.get('timestamp')) if component_result.get('timestamp') else None
        )
    except Exception as e:
        logger.error(f"Error retrieving health for component {component_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve component health")


@router.get("/agents/{agent_id}", response_model=AgentHealth)
def get_agent_health_status(
    agent_id: str,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> AgentHealth:
    """Get the health status of a specific agent by running its dedicated check."""
    health_checker = _get_health_checker(agent_manager)
    try:
        # Get agent instance
        agent = agent_manager.agents.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

        if not hasattr(health_checker, 'check_agent_health'):
             raise HTTPException(status_code=501, detail="check_agent_health not implemented")

        # Call check_agent_health with agent instance
        health_info = health_checker.check_agent_health(agent_id, agent)

        # check_agent_health returns a dict, adapt to AgentHealth model
        return AgentHealth(
             agent_id=agent_id,
             status=health_info.get('status', HealthStatus.UNKNOWN),
             # Assuming last_check is part of details or timestamp?
             last_check=datetime.fromisoformat(health_info.get('timestamp')) if health_info.get('timestamp') else None, # Get timestamp if available
             details=health_info.get('details') # Get details if available
         )
    except ValueError as e: # Handle specific errors if check_agent_health raises them
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving health for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve agent health")


@router.get("/system", response_model=SystemHealthStatus)
def get_system_health(
    # Use agent_manager to get health_checker
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> SystemHealthStatus:
    """Get health status of default system checks (resources, runtime, network)."""
    health_checker = _get_health_checker(agent_manager)
    system_check_ids = ["system_resources", "python_runtime", "network_connectivity"]
    system_checks_results: Dict[str, ComponentStatus] = {}
    overall_status = HealthStatus.OK

    for check_id in system_check_ids:
        result = health_checker.results.get(check_id)
        if result:
            status = result.get('status', HealthStatus.UNKNOWN)
            system_checks_results[check_id] = ComponentStatus(
                status=status,
                message=result.get('message'),
                details=result.get('details'),
                timestamp=datetime.fromisoformat(result.get('timestamp')) if result.get('timestamp') else None
            )
            if status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
            elif status == HealthStatus.WARNING and overall_status == HealthStatus.OK:
                overall_status = HealthStatus.WARNING
            # If status is UNKNOWN, let overall status potentially become UNKNOWN below

        else:
            system_checks_results[check_id] = ComponentStatus(status=HealthStatus.UNKNOWN, message="Check has not run")
            status = HealthStatus.UNKNOWN

        # Update overall status based on current check's status
        if status == HealthStatus.CRITICAL:
             overall_status = HealthStatus.CRITICAL
        elif status == HealthStatus.WARNING and overall_status != HealthStatus.CRITICAL:
             overall_status = HealthStatus.WARNING
        elif status == HealthStatus.UNKNOWN and overall_status == HealthStatus.OK:
             overall_status = HealthStatus.UNKNOWN

    return SystemHealthStatus(
        overall_status=overall_status,
        checks=system_checks_results,
        timestamp=datetime.utcnow()
    )


@router.post("/check/agent/{agent_id}", response_model=AgentHealthStatus)
def run_agent_health_check(
    agent_id: str,
    # Use agent_manager to get health_checker
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> AgentHealthStatus:
    """Manually trigger a health check for a specific agent."""
    health_checker = _get_health_checker(agent_manager)
    try:
        agent = agent_manager.agents.get(agent_id)
        if agent is None:
            raise HTTPException(status_code=404, detail=f"Agent '{agent_id}' not found")

        if not hasattr(health_checker, 'check_agent_health'):
            raise HTTPException(status_code=501, detail="check_agent_health not implemented")

        # Call check_agent_health - force_check is not an argument in the provided definition
        health_info = health_checker.check_agent_health(agent_id, agent)

        # Adapt to AgentHealthStatus model
        return AgentHealthStatus(
            agent_id=agent_id,
            status=health_info.get('status', HealthStatus.UNKNOWN),
            message=health_info.get('message'),
            details=health_info.get('details'),
            timestamp=datetime.utcnow() # Use current time as check was just run
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error running health check for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to run agent health check")


@router.post("/check/system", response_model=SystemHealthStatus)
def run_system_health_check(
    # Use agent_manager to get health_checker
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> SystemHealthStatus:
    """Manually trigger default system health checks and update results."""
    health_checker = _get_health_checker(agent_manager)
    system_check_ids = ["system_resources", "python_runtime", "network_connectivity"]
    try:
        # Manually run the default check functions and update results
        now = datetime.utcnow()
        for check_id in system_check_ids:
            check_func_name = f"_check_{check_id}"
            if hasattr(health_checker, check_func_name) and callable(getattr(health_checker, check_func_name)):
                try:
                    result = getattr(health_checker, check_func_name)()
                    health_checker.results[check_id] = {
                        "status": result.get("status", HealthStatus.UNKNOWN),
                        "message": result.get("message", ""),
                        "details": result.get("details", {}),
                        "timestamp": now.isoformat(),
                    }
                    if check_id in health_checker.checks:
                         health_checker.checks[check_id]["last_run"] = now
                except Exception as check_e:
                    logger.error(f"Error running manual system check {check_id}: {check_e}", exc_info=True)
                    health_checker.results[check_id] = {
                         "status": HealthStatus.CRITICAL,
                         "message": f"Check failed: {str(check_e)}",
                         "details": {},
                         "timestamp": now.isoformat(),
                    }
                    if check_id in health_checker.checks:
                         health_checker.checks[check_id]["last_run"] = now

        # Return the updated system health status by calling the GET endpoint logic
        return get_system_health(agent_manager)

    except Exception as e:
        logger.error(f"Error running system health checks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to run system health checks")


@router.get("/dependencies", response_model=Dict[str, Any])
def get_dependency_status(
    # Use agent_manager to get health_checker
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, Any]:
    """Get detailed status of external dependencies (Placeholder)."""
    health_checker = _get_health_checker(agent_manager)
    # Placeholder: Actual implementation depends on how dependencies are checked
    # Check if a specific check like 'external_services' exists
    dep_check_result = health_checker.results.get("external_services") # Example check ID
    if dep_check_result:
        # Return the ComponentStatus model for consistency
        return ComponentStatus(
             status=dep_check_result.get('status', HealthStatus.UNKNOWN),
             message=dep_check_result.get('message'),
             details=dep_check_result.get('details'),
             timestamp=datetime.fromisoformat(dep_check_result.get('timestamp')) if dep_check_result.get('timestamp') else None
         ).dict()
    else:
        # Return a placeholder if no specific dependency check is registered/run
        return ComponentStatus(
            status= HealthStatus.UNKNOWN,
            message= "Dependency check ('external_services') not implemented or not run yet.",
            timestamp= datetime.utcnow()
        ).dict()

# Use HealthCheckInfo model for the response
@router.get("/checks", response_model=List[HealthCheckInfo])
async def get_available_health_checks(
    manager: AgentManager = Depends(get_agent_manager),
) -> List[HealthCheckInfo]:
    """
    Get list of available health checks and their last known status.
    """
    health_checker = _get_health_checker(manager)
    try:
        check_infos = []
        # Iterate through registered checks
        for check_id, check_config in health_checker.checks.items():
            # Get the latest result for this check
            result = health_checker.results.get(check_id)
            # Parse timestamp safely
            last_check_time = None
            if result and result.get("timestamp"):
                 try:
                      last_check_time = datetime.fromisoformat(result.get("timestamp"))
                 except (ValueError, TypeError):
                      logger.warning(f"Could not parse timestamp for check {check_id}: {result.get('timestamp')}")

            info = HealthCheckInfo(
                check_id=check_id,
                description=check_config.get("description"),
                status=result.get("status", HealthStatus.UNKNOWN) if result else HealthStatus.UNKNOWN,
                message=result.get("message") if result else None,
                details=result.get("details") if result else None,
                last_check=last_check_time
            )
            check_infos.append(info)
        return check_infos
    except Exception as e:
        logger.error(f"Error listing available health checks: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
