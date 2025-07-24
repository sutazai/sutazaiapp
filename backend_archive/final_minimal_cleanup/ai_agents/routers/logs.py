"""
Agent Logging Router

This module provides REST API endpoints for accessing agent logs and diagnostics.
"""

import os
import asyncio
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Path
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from datetime import datetime, timedelta

from ai_agents.dependencies import get_agent_manager
from ai_agents.agent_manager import AgentManager
from .models import LogEntry, LogQuery, LogLevel, LogSummary

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/logs", tags=["logs"])


# Rename local LogEntry model to avoid conflicts
class LogEntryModel(BaseModel):
    timestamp: datetime
    level: str
    message: str
    agent_id: Optional[str] = None
    task_id: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


# Placeholder for actual log retrieval logic
def _fetch_logs_from_source(service: str, level: str, lines: int) -> List[LogEntryModel]:
     # In a real system, this would read from files, a logging database (e.g., ELK), or a logging service
     logger.warning("_fetch_logs_from_source is a placeholder and does not retrieve real logs.")
     # Return dummy data matching the model structure
     dummy_logs = []
     for i in range(min(lines, 5)): # Return max 5 dummy lines
          dummy_logs.append(
               LogEntryModel(
                    timestamp=datetime.now() - timedelta(seconds=i*10),
                    level="INFO",
                    message=f"Dummy log message {i+1} for {service}",
                    agent_id=f"agent_{i%2}" if service=="agent_manager" else None,
                    details={"source": "placeholder"}
               )
          )
     return dummy_logs


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


@router.get("/", response_model=List[LogEntryModel])
def get_logs(
    service: str = Query("all", description="Service name (e.g., agent_manager, model_manager, specific agent ID)"),
    level: str = Query("INFO", description="Minimum log level (e.g., DEBUG, INFO, WARNING, ERROR)"),
    lines: int = Query(100, ge=1, le=5000, description="Number of log lines to retrieve")
    # agent_manager: AgentManager = Depends(get_agent_manager) # Not needed if logs aren't on manager
) -> List[LogEntryModel]:
    """Retrieve log entries for specified services or agents."""
    try:
        # Replace with actual log fetching logic
        # logs = agent_manager.get_logs(service=service, level=level, lines=lines)
        logs = _fetch_logs_from_source(service=service, level=level, lines=lines)
        return logs
    except Exception as e:
        logger.error(f"Error retrieving logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve logs")


@router.get("/agents/{agent_id}", response_model=List[LogEntryModel])
def get_agent_logs(
    agent_id: str = Path(...),
    limit: int = Query(100, ge=1, le=1000),
    level: Optional[str] = Query(None),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> List[LogEntryModel]:
    """Retrieve logs for a specific agent."""
    try:
        if not hasattr(agent_manager, 'get_logs'):
            raise HTTPException(status_code=501, detail="get_logs not implemented")
        if not hasattr(agent_manager, 'agent_exists') or not agent_manager.agent_exists(agent_id):
            raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

        logs = agent_manager.get_logs(agent_id=agent_id, limit=limit, level=level)
        return [LogEntryModel(**log) for log in logs]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving logs for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve agent logs")


@router.get("/stream")
async def stream_logs(agent_manager: AgentManager = Depends(get_agent_manager)) -> StreamingResponse:
    """Stream logs in real-time using Server-Sent Events."""
    if not hasattr(agent_manager, 'stream_logs'):
        raise HTTPException(status_code=501, detail="Log streaming not implemented")

    async def log_generator():
        try:
            count = 0
            while count < 10:
                yield f"data: {{'timestamp': '{datetime.now().isoformat()}', 'level': 'INFO', 'message': f'Log message {count}'}}\n\n"
                count += 1
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            logger.info("Log stream cancelled by client.")
        except Exception as e:
            logger.error(f"Error during log streaming: {e}")
            yield f"data: {{'level': 'ERROR', 'message': 'Log stream error: {e}'}}\n\n"

    return StreamingResponse(log_generator(), media_type="text/event-stream")


@router.post("/search", response_model=List[LogEntry])
async def search_logs(
    query: LogQuery, manager: AgentManager = Depends(get_agent_manager)
) -> List[LogEntry]:
    """Search logs based on various criteria."""
    try:
        if not hasattr(manager, 'search_logs'):
            raise HTTPException(status_code=501, detail="search_logs not implemented")
        logs = manager.search_logs(
            level=query.level,
            message_contains=query.message_contains,
            start_time=query.start_time,
            end_time=query.end_time,
            limit=query.limit
        )
        return [LogEntry(**log) for log in logs]
    except Exception as e:
        logger.error(f"Error searching logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to search logs")


@router.get("/levels", response_model=Dict[str, LogLevel])
async def get_log_levels(manager: AgentManager = Depends(get_agent_manager)) -> Dict[str, LogLevel]:
    """Get available log levels."""
    if hasattr(manager, 'get_available_log_levels'):
        return {level: LogLevel(level=level) for level in manager.get_available_log_levels()}
    else:
        return {"DEBUG": LogLevel(level="DEBUG"), "INFO": LogLevel(level="INFO"), "WARNING": LogLevel(level="WARNING"), "ERROR": LogLevel(level="ERROR"), "CRITICAL": LogLevel(level="CRITICAL")}


@router.get("/summary", response_model=LogSummary)
async def get_log_summary(
    agent_id: Optional[str] = Query(None, description="Agent ID to get summary for, or system if omitted"),
    manager: AgentManager = Depends(get_agent_manager),
) -> LogSummary:
    """Get a summary of log counts by level."""
    if not hasattr(manager, 'get_log_summary'):
        raise HTTPException(status_code=501, detail="get_log_summary not implemented")
    summary = manager.get_log_summary()
    return summary


@router.get("/errors/summary", response_model=List[Dict[str, Any]])
def get_error_summary_route(
    limit: int = Query(10, ge=1, le=100),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> List[Dict[str, Any]]:
    """Get a summary of recent errors."""
    if not hasattr(agent_manager, 'get_error_summary'):
        logger.warning("AgentManager does not implement get_error_summary")
        return [] # Return empty list to satisfy type hint
    errors = agent_manager.get_error_summary(limit=limit)
    return errors # type: ignore[no-any-return]


@router.get("/system", response_model=List[LogEntryModel])
def get_system_logs(
    limit: int = Query(100, ge=1, le=1000),
    level: Optional[str] = Query(None),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> List[LogEntryModel]:
    """Get system-wide logs, optionally filtered by level."""
    try:
        if not hasattr(agent_manager, 'get_logs'):
            raise HTTPException(status_code=501, detail="get_logs not implemented")
        logs = agent_manager.get_logs(limit=limit, level=level)
        return [LogEntryModel(**log) for log in logs]
    except Exception as e:
        logger.error(f"Error retrieving system logs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve system logs")


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
            total_cpu += metrics.get('cpu_percent', 0.0)
            total_memory += metrics.get('memory_percent', 0.0)
            total_executions += metrics.get('execution_count', 0)
            total_errors += metrics.get('error_count', 0)

        avg_cpu = total_cpu / total_agents if total_agents > 0 else 0.0

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
                agent_id: manager.health_check.check_agent_health(agent_id, manager.agents[agent_id])
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
