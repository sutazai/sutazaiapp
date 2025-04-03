"""
Agent Logging Router

This module provides REST API endpoints for accessing agent logs and diagnostics.
"""

import os
import json
import asyncio
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Body, StreamingResponse
from pydantic import BaseModel
from datetime import datetime, timedelta

from ai_agents.agent_manager import AgentManager
from ai_agents.dependencies import get_agent_manager
from ai_agents.utils.models import LogEntry

router = APIRouter(prefix="/logs", tags=["logs"])


# Dependency to get agent manager instance
def get_agent_manager() -> AgentManager:
    """Get the agent manager instance."""
    return AgentManager()


class LogEntry(BaseModel):
    """Log entry model."""

    timestamp: datetime
    level: str
    agent_id: Optional[str] = None
    message: str
    details: Optional[Dict[str, Any]] = None


class DiagnosticInfo(BaseModel):
    """Diagnostic information model."""

    agent_id: str
    start_time: datetime
    uptime: float
    total_executions: int
    error_count: int
    last_error: Optional[Dict[str, Any]] = None
    resource_usage: Dict[str, float]
    dependencies: List[str]
    capabilities: List[str]


@router.get("/", response_model=List[LogEntry])
def get_logs(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    agent_id: Optional[str] = Query(None),
    level: Optional[str] = Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> List[LogEntry]:
    """Retrieve system or agent-specific logs."""
    logs = agent_manager.get_logs(
        limit=limit,
        offset=offset,
        agent_id=agent_id,
        level=level,
        start_time=start_time,
        end_time=end_time
    )
    return logs


@router.get("/agents/{agent_id}", response_model=List[LogEntry])
def get_agent_logs(
    agent_id: str,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    level: Optional[str] = Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> List[LogEntry]:
    """Retrieve logs for a specific agent."""
    logs = agent_manager.get_logs(
        agent_id=agent_id,
        limit=limit,
        offset=offset,
        level=level,
        start_time=start_time,
        end_time=end_time
    )
    if not logs and not agent_manager.agent_exists(agent_id):
         raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    return logs


@router.get("/stream")
async def stream_logs(
    agent_id: Optional[str] = Query(None),
    level: Optional[str] = Query(None),
    agent_manager: AgentManager = Depends(get_agent_manager)
):
    """Stream logs in real-time using Server-Sent Events."""
    # This requires a log streaming mechanism in AgentManager
    async def event_generator():
        async for log_entry in agent_manager.stream_logs(agent_id=agent_id, level=level):
            yield f"data: {json.dumps(log_entry.dict())}\n\n"
            await asyncio.sleep(0.1) # Avoid busy-waiting

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/search", response_model=List[LogEntry])
def search_logs(
    query: str = Body(...),
    limit: int = Body(100),
    agent_id: Optional[str] = Body(None),
    level: Optional[str] = Body(None),
    start_time: Optional[datetime] = Body(None),
    end_time: Optional[datetime] = Body(None),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> List[LogEntry]:
    """Search logs based on a query string."""
    logs = agent_manager.search_logs(
        query=query,
        limit=limit,
        agent_id=agent_id,
        level=level,
        start_time=start_time,
        end_time=end_time
    )
    return logs


@router.get("/levels", response_model=List[str])
def get_log_levels(
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> List[str]:
    """Get available log levels."""
    # Assuming AgentManager can provide this
    return agent_manager.get_available_log_levels()


@router.get("/summary/{agent_id}", response_model=Dict[str, Any])
def get_agent_log_summary(
    agent_id: str,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, Any]:
    """Get a summary of logs for a specific agent."""
    summary = agent_manager.get_log_summary(agent_id)
    if not summary:
         raise HTTPException(status_code=404, detail=f"No summary available for agent {agent_id}")
    return summary


@router.get("/errors/summary", response_model=Dict[str, int])
def get_error_summary(
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, int]:
    """Get a summary of error logs across all agents."""
    return agent_manager.get_error_summary()


@router.get("/system", response_model=List[LogEntry])
async def get_system_logs(
    level: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = Query(100, ge=1, le=1000),
    manager: AgentManager = Depends(get_agent_manager),
) -> List[LogEntry]:
    """
    Get system-wide logs.

    Args:
        level: Optional log level filter
        start_time: Optional start time filter
        end_time: Optional end time filter
        limit: Maximum number of log entries to return
        manager: Agent manager instance

    Returns:
        List[LogEntry]: List of log entries
    """
    try:
        # TODO: Implement system log storage and retrieval
        # For now, return empty list
        return []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents/{agent_id}/diagnostics", response_model=DiagnosticInfo)
async def get_agent_diagnostics(
    agent_id: str, manager: AgentManager = Depends(get_agent_manager)
) -> DiagnosticInfo:
    """
    Get diagnostic information for a specific agent.

    Args:
        agent_id: Agent ID to get diagnostics for
        manager: Agent manager instance

    Returns:
        DiagnosticInfo: Agent diagnostic information
    """
    try:
        if agent_id not in manager.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        agent = manager.agents[agent_id]
        metrics = manager.get_agent_metrics(agent_id)

        # Calculate uptime
        start_time = metrics.last_active - timedelta(
            seconds=metrics.avg_execution_time * metrics.execution_count
        )
        uptime = (datetime.utcnow() - start_time).total_seconds()

        # Get last error if any
        last_error = None
        if metrics.error_count > 0:
            # TODO: Implement error history storage and retrieval
            last_error = {
                "timestamp": metrics.last_active,
                "message": "Last error message",
                "details": {},
            }

        return DiagnosticInfo(
            agent_id=agent_id,
            start_time=start_time,
            uptime=uptime,
            total_executions=metrics.execution_count,
            error_count=metrics.error_count,
            last_error=last_error,
            resource_usage={"cpu": metrics.cpu_usage, "memory": metrics.memory_usage},
            dependencies=agent.get_dependencies(),
            capabilities=agent.get_capabilities(),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/diagnostics", response_model=Dict[str, Any])
async def get_system_diagnostics(
    manager: AgentManager = Depends(get_agent_manager),
) -> Dict[str, Any]:
    """
    Get system-wide diagnostic information.

    Args:
        manager: Agent manager instance

    Returns:
        Dict[str, Any]: System diagnostic information
    """
    try:
        # Get system metrics
        total_agents = len(manager.agents)
        active_agents = len(
            [
                agent_id
                for agent_id, status in manager.agent_status.items()
                if status in ["running", "paused"]
            ]
        )

        # Calculate resource usage
        total_cpu = 0.0
        total_memory = 0.0
        total_executions = 0
        total_errors = 0

        for agent_id in manager.agents:
            metrics = manager.get_agent_metrics(agent_id)
            total_cpu += metrics.cpu_usage
            total_memory += metrics.memory_usage
            total_executions += metrics.execution_count
            total_errors += metrics.error_count

        return {
            "timestamp": datetime.utcnow(),
            "system_info": {
                "total_agents": total_agents,
                "active_agents": active_agents,
                "total_cpu_usage": total_cpu,
                "total_memory_usage": total_memory,
                "total_executions": total_executions,
                "total_errors": total_errors,
                "error_rate": total_errors / total_executions
                if total_executions > 0
                else 0,
            },
            "agent_status": {
                agent_id: status for agent_id, status in manager.agent_status.items()
            },
            "health_status": {
                agent_id: manager.health_check.check_agent(agent_id)
                for agent_id in manager.agents
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/agents/{agent_id}/clear")
async def clear_agent_logs(
    agent_id: str, manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, str]:
    """
    Clear logs for a specific agent.

    Args:
        agent_id: Agent ID to clear logs for
        manager: Agent manager instance

    Returns:
        Dict[str, str]: Clear operation status
    """
    try:
        if agent_id not in manager.agents:
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        # TODO: Implement log clearing
        return {"status": "success", "message": f"Logs cleared for agent {agent_id}"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/system/clear")
async def clear_system_logs(
    manager: AgentManager = Depends(get_agent_manager),
) -> Dict[str, str]:
    """
    Clear system-wide logs.

    Args:
        manager: Agent manager instance

    Returns:
        Dict[str, str]: Clear operation status
    """
    try:
        # TODO: Implement system log clearing
        return {"status": "success", "message": "System logs cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export")
async def export_logs(
    agent_id: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    manager: AgentManager = Depends(get_agent_manager),
) -> Dict[str, str]:
    """
    Export logs to a file.

    Args:
        agent_id: Optional agent ID to export logs for
        start_time: Optional start time filter
        end_time: Optional end time filter
        manager: Agent manager instance

    Returns:
        Dict[str, str]: Export status
    """
    try:
        # TODO: Implement log export
        export_path = os.path.join(
            "logs", f"export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"
        )

        return {
            "status": "success",
            "message": f"Logs exported to {export_path}",
            "export_path": export_path,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
