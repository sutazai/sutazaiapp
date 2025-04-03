"""
Agent Monitoring Router

This module provides REST API endpoints for monitoring agent performance and metrics.
"""

from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime

from ai_agents.agent_manager import AgentManager
from ai_agents.dependencies import get_agent_manager
from ai_agents.utils.models import AgentMetrics, SystemMetrics, OverallMetrics
from ai_agents.utils.enums import AgentStatus

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


# Dependency to get agent manager instance
def get_agent_manager() -> AgentManager:
    """Get the agent manager instance."""
    return AgentManager()


class AgentMetrics(BaseModel):
    """Agent metrics model."""

    agent_id: str
    cpu_usage: float
    memory_usage: float
    last_active: datetime
    execution_count: int
    error_count: int
    avg_execution_time: float
    status: str


class SystemMetrics(BaseModel):
    """System metrics model."""

    total_agents: int
    active_agents: int
    total_cpu_usage: float
    total_memory_usage: float
    error_rate: float
    avg_response_time: float


class Alert(BaseModel):
    """Alert model."""

    alert_id: str
    agent_id: str
    type: str
    severity: str
    message: str
    created_at: datetime
    resolved_at: Optional[datetime] = None
    status: str


@router.get("/agents", response_model=Dict[str, AgentMetrics])
def get_all_agents_metrics(
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, AgentMetrics]:
    """Get performance metrics for all active agents."""
    metrics = agent_manager.get_all_agent_metrics()
    # Convert AgentStatus enum to string for response model compatibility
    for agent_id in metrics:
         if hasattr(metrics[agent_id], 'status') and isinstance(metrics[agent_id].status, AgentStatus):
              metrics[agent_id] = metrics[agent_id].copy(update={'status': metrics[agent_id].status.value})
    return metrics


@router.get("/agents/{agent_id}", response_model=AgentMetrics)
def get_agent_metrics(
    agent_id: str,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> AgentMetrics:
    """Get performance metrics for a specific agent."""
    metrics = agent_manager.get_agent_metrics(agent_id)
    if not metrics:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found or no metrics available")
    # Convert AgentStatus enum to string
    if hasattr(metrics, 'status') and isinstance(metrics.status, AgentStatus):
         metrics = metrics.copy(update={'status': metrics.status.value})
    return metrics


@router.get("/system", response_model=SystemMetrics)
def get_system_metrics(
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> SystemMetrics:
    """Get overall system performance metrics."""
    return agent_manager.get_system_metrics()


@router.get("/overall", response_model=OverallMetrics)
def get_overall_metrics(
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> OverallMetrics:
    """Get combined agent and system metrics."""
    all_metrics = agent_manager.get_overall_metrics()
    # Convert AgentStatus enum to string for response model compatibility
    if all_metrics and hasattr(all_metrics, 'agents') and all_metrics.agents:
         updated_agents = {}
         for agent_id, metrics in all_metrics.agents.items():
             if hasattr(metrics, 'status') and isinstance(metrics.status, AgentStatus):
                 updated_agents[agent_id] = metrics.copy(update={'status': metrics.status.value})
             else:
                 updated_agents[agent_id] = metrics
         all_metrics = all_metrics.copy(update={'agents': updated_agents})
    return all_metrics


# Add routes for historical data, filtering, etc.
@router.get("/history/{agent_id}")
def get_agent_history(
    agent_id: str,
    limit: int = 100,
    agent_manager: AgentManager = Depends(get_agent_manager)
):
    """Get historical performance metrics for an agent."""
    # This would likely involve querying a time-series DB or logs
    history = agent_manager.get_agent_metric_history(agent_id, limit)
    if not history:
         raise HTTPException(status_code=404, detail=f"No history found for agent {agent_id}")
    return history


@router.get("/agents/status/{status}", response_model=List[AgentMetrics])
def get_agents_by_status(
    status: AgentStatus, # Use Enum for path parameter
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> List[AgentMetrics]:
    """Get agents by their current status."""
    agents = agent_manager.get_agents_by_status(status)
    # Convert AgentStatus enum to string for response model compatibility
    result = []
    for metrics in agents:
         if hasattr(metrics, 'status') and isinstance(metrics.status, AgentStatus):
             result.append(metrics.copy(update={'status': metrics.status.value}))
         else:
             result.append(metrics)
    return result


@router.get("/agents/top/cpu", response_model=List[AgentMetrics])
def get_top_cpu_agents(
    limit: int = 5,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> List[AgentMetrics]:
    """Get agents consuming the most CPU."""
    agents = agent_manager.get_top_agents_by_metric("cpu_usage", limit, descending=True)
    # Convert AgentStatus enum to string
    result = []
    for metrics in agents:
         if hasattr(metrics, 'status') and isinstance(metrics.status, AgentStatus):
             result.append(metrics.copy(update={'status': metrics.status.value}))
         else:
             result.append(metrics)
    return result


@router.get("/agents/top/memory", response_model=List[AgentMetrics])
def get_top_memory_agents(
    limit: int = 5,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> List[AgentMetrics]:
    """Get agents consuming the most Memory."""
    agents = agent_manager.get_top_agents_by_metric("memory_usage", limit, descending=True)
    # Convert AgentStatus enum to string
    result = []
    for metrics in agents:
         if hasattr(metrics, 'status') and isinstance(metrics.status, AgentStatus):
             result.append(metrics.copy(update={'status': metrics.status.value}))
         else:
             result.append(metrics)
    return result


@router.get("/agents/errors/rate", response_model=List[AgentMetrics])
def get_agents_by_error_rate(
    limit: int = 5,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> List[AgentMetrics]:
    """Get agents with the highest error rate."""
    agents = agent_manager.get_top_agents_by_metric("error_rate", limit, descending=True)
    # Convert AgentStatus enum to string
    result = []
    for metrics in agents:
         if hasattr(metrics, 'status') and isinstance(metrics.status, AgentStatus):
             result.append(metrics.copy(update={'status': metrics.status.value}))
         else:
             result.append(metrics)
    return result


@router.get("/alerts", response_model=List[Alert])
async def get_active_alerts(
    manager: AgentManager = Depends(get_agent_manager),
) -> List[Alert]:
    """
    Get active alerts for all agents.

    Args:
        manager: Agent manager instance

    Returns:
        List[Alert]: List of active alerts
    """
    try:
        alerts = []

        for agent_id, agent in manager.agents.items():
            metrics = manager.get_agent_metrics(agent_id)
            status = manager.agent_status[agent_id]

            # Check for high CPU usage
            if metrics.cpu_usage > 80:
                alerts.append(
                    Alert(
                        alert_id=f"cpu_{agent_id}",
                        agent_id=agent_id,
                        type="high_cpu",
                        severity="warning",
                        message=f"High CPU usage: {metrics.cpu_usage}%",
                        created_at=datetime.utcnow(),
                        status="active",
                    )
                )

            # Check for high memory usage
            if metrics.memory_usage > 80:
                alerts.append(
                    Alert(
                        alert_id=f"memory_{agent_id}",
                        agent_id=agent_id,
                        type="high_memory",
                        severity="warning",
                        message=f"High memory usage: {metrics.memory_usage}%",
                        created_at=datetime.utcnow(),
                        status="active",
                    )
                )

            # Check for high error rate
            if (
                metrics.error_count > 0
                and metrics.error_count / metrics.execution_count > 0.1
            ):
                alerts.append(
                    Alert(
                        alert_id=f"error_{agent_id}",
                        agent_id=agent_id,
                        type="high_error_rate",
                        severity="critical",
                        message=f"High error rate: {metrics.error_count / metrics.execution_count:.2%}",
                        created_at=datetime.utcnow(),
                        status="active",
                    )
                )

            # Check for agent in error state
            if status == "error":
                alerts.append(
                    Alert(
                        alert_id=f"state_{agent_id}",
                        agent_id=agent_id,
                        type="error_state",
                        severity="critical",
                        message="Agent in error state",
                        created_at=datetime.utcnow(),
                        status="active",
                    )
                )

        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/alerts/{alert_id}/resolve")
async def resolve_alert(
    alert_id: str, manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, str]:
    """
    Resolve a specific alert.

    Args:
        alert_id: Alert ID to resolve
        manager: Agent manager instance

    Returns:
        Dict[str, str]: Resolution status
    """
    try:
        # TODO: Implement alert storage and resolution
        # For now, just return success
        return {
            "status": "success",
            "message": f"Alert {alert_id} resolved successfully",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
