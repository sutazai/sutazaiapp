"""
Agent Monitoring Router

This module provides REST API endpoints for monitoring agent performance and metrics.
"""

from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from datetime import datetime

from ..agent_manager import AgentManager

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


@router.get("/agents/metrics", response_model=List[AgentMetrics])
async def get_all_agent_metrics(
    manager: AgentManager = Depends(get_agent_manager),
) -> List[AgentMetrics]:
    """
    Get metrics for all agents.

    Args:
        manager: Agent manager instance

    Returns:
        List[AgentMetrics]: List of agent metrics
    """
    try:
        metrics = []
        for agent_id, agent in manager.agents.items():
            agent_metrics = manager.get_agent_metrics(agent_id)
            metrics.append(
                AgentMetrics(
                    agent_id=agent_id,
                    cpu_usage=agent_metrics.cpu_usage,
                    memory_usage=agent_metrics.memory_usage,
                    last_active=agent_metrics.last_active,
                    execution_count=agent_metrics.execution_count,
                    error_count=agent_metrics.error_count,
                    avg_execution_time=agent_metrics.avg_execution_time,
                    status=manager.agent_status[agent_id],
                )
            )
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}/metrics", response_model=AgentMetrics)
async def get_agent_metrics(
    agent_id: str, manager: AgentManager = Depends(get_agent_manager)
) -> AgentMetrics:
    """
    Get metrics for a specific agent.

    Args:
        agent_id: Agent ID to get metrics for
        manager: Agent manager instance

    Returns:
        AgentMetrics: Agent metrics
    """
    try:
        if agent_id not in manager.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        agent_metrics = manager.get_agent_metrics(agent_id)
        return AgentMetrics(
            agent_id=agent_id,
            cpu_usage=agent_metrics.cpu_usage,
            memory_usage=agent_metrics.memory_usage,
            last_active=agent_metrics.last_active,
            execution_count=agent_metrics.execution_count,
            error_count=agent_metrics.error_count,
            avg_execution_time=agent_metrics.avg_execution_time,
            status=manager.agent_status[agent_id],
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/metrics", response_model=SystemMetrics)
async def get_system_metrics(
    manager: AgentManager = Depends(get_agent_manager),
) -> SystemMetrics:
    """
    Get system-wide metrics.

    Args:
        manager: Agent manager instance

    Returns:
        SystemMetrics: System metrics
    """
    try:
        total_agents = len(manager.agents)
        active_agents = len(
            [
                agent_id
                for agent_id, status in manager.agent_status.items()
                if status in ["running", "paused"]
            ]
        )

        total_cpu = 0.0
        total_memory = 0.0
        total_errors = 0
        total_executions = 0
        total_time = 0.0

        for agent_id in manager.agents:
            metrics = manager.get_agent_metrics(agent_id)
            total_cpu += metrics.cpu_usage
            total_memory += metrics.memory_usage
            total_errors += metrics.error_count
            total_executions += metrics.execution_count
            total_time += metrics.avg_execution_time * metrics.execution_count

        return SystemMetrics(
            total_agents=total_agents,
            active_agents=active_agents,
            total_cpu_usage=total_cpu,
            total_memory_usage=total_memory,
            error_rate=total_errors / total_executions if total_executions > 0 else 0,
            avg_response_time=total_time / total_executions
            if total_executions > 0
            else 0,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}/metrics/history", response_model=List[AgentMetrics])
async def get_agent_metrics_history(
    agent_id: str,
    window: Optional[int] = 100,
    manager: AgentManager = Depends(get_agent_manager),
) -> List[AgentMetrics]:
    """
    Get historical metrics for a specific agent.

    Args:
        agent_id: Agent ID to get history for
        window: Number of historical points to return
        manager: Agent manager instance

    Returns:
        List[AgentMetrics]: Historical metrics
    """
    try:
        if agent_id not in manager.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        # TODO: Implement metrics history storage and retrieval
        # For now, return current metrics
        agent_metrics = manager.get_agent_metrics(agent_id)
        return [
            AgentMetrics(
                agent_id=agent_id,
                cpu_usage=agent_metrics.cpu_usage,
                memory_usage=agent_metrics.memory_usage,
                last_active=agent_metrics.last_active,
                execution_count=agent_metrics.execution_count,
                error_count=agent_metrics.error_count,
                avg_execution_time=agent_metrics.avg_execution_time,
                status=manager.agent_status[agent_id],
            )
        ]
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
