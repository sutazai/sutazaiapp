"""
Agent Monitoring Router

This module provides REST API endpoints for monitoring agent performance and metrics.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from datetime import datetime

from ai_agents.agent_manager import AgentManager
from ai_agents.dependencies import get_agent_manager
from ai_agents.utils.models import OverallMetrics
from .models import AgentMetricsSnapshot, AgentMetricHistory

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/monitoring", tags=["monitoring"])

# Define missing Pydantic models
# Remove local definitions, as they are imported above
# class AgentMetricsSnapshot(BaseModel):
#    ...
#
# class AgentMetricHistory(BaseModel):
#    ...


@router.get("/agents", response_model=List[AgentMetricsSnapshot])
async def get_all_agents_metrics(
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> List[AgentMetricsSnapshot]:
    """Get current performance metrics for all managed agents."""
    try:
        # Assuming AgentManager stores metrics in agent_metrics dict
        return list(agent_manager.agent_metrics.values())
    except Exception as e:
        logger.error(f"Error getting all agent metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve agent metrics")


@router.get("/agents/{agent_id}", response_model=AgentMetricsSnapshot)
async def get_single_agent_metrics(
    agent_id: str,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> AgentMetricsSnapshot:
    """Get current performance metrics for a specific agent."""
    try:
        # Use the method from AgentManager
        metrics = agent_manager.get_agent_metrics(agent_id)
        return AgentMetricsSnapshot(**metrics) # Convert dict back to model if needed
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting metrics for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve agent metrics")


@router.get("/system", response_model=Dict[str, Any])
async def get_system_metrics_route( # Renamed
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, Any]:
    """Get overall system performance metrics."""
    try:
        # Call the correct method on AgentManager
        if not hasattr(agent_manager, 'get_system_health'):
             logger.warning("AgentManager does not implement get_system_health")
             return {} # Return empty dict to satisfy type hint
        metrics = agent_manager.get_system_health()
        # Ensure the returned dict matches SystemMetrics model
        return metrics # type: ignore[no-any-return]
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve system metrics")


@router.get("/overall", response_model=OverallMetrics)
def get_overall_metrics_route( # Renamed
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> OverallMetrics:
    """Get overall aggregated metrics for the agent system."""
    try:
        # This likely needs to be implemented in AgentManager
        if not hasattr(agent_manager, 'get_overall_metrics'):
            raise HTTPException(status_code=501, detail="get_overall_metrics not implemented")
        metrics = agent_manager.get_overall_metrics()
        return OverallMetrics(**metrics)
    except Exception as e:
        logger.error(f"Error getting overall metrics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve overall metrics")


@router.get("/agents/{agent_id}/history", response_model=AgentMetricHistory)
async def get_agent_history(
    agent_id: str,
    limit: Optional[int] = Query(100, ge=1, le=1000, description="Number of historical points to return"),
    start_time: Optional[datetime] = Query(None, description="Filter history from this time"),
    end_time: Optional[datetime] = Query(None, description="Filter history up to this time"),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> AgentMetricHistory:
    """Get historical performance metrics for a specific agent."""
    try:
        # This likely needs to be implemented in AgentManager
        if not hasattr(agent_manager, 'get_agent_metric_history'):
             raise HTTPException(status_code=501, detail="get_agent_metric_history not implemented")
        history = agent_manager.get_agent_metric_history(agent_id, limit, start_time, end_time)
        # Assume history is returned in a format compatible with AgentMetricHistory
        return AgentMetricHistory(**history)
    except ValueError as e:
         raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting metric history for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve metric history")


@router.get("/agents/status/{status}", response_model=List[str])
def get_agents_by_status_route(
    status: str, # Consider using AgentStatus Enum here
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> List[str]:
    """Get a list of agent IDs by their current status."""
    try:
        # This likely needs to be implemented in AgentManager
        if not hasattr(agent_manager, 'get_agents_by_status'):
             logger.warning("AgentManager does not implement get_agents_by_status")
             return [] # Return empty list to satisfy type hint
        # TODO: Validate status string against AgentStatus enum if applicable
        agent_ids = agent_manager.get_agents_by_status(status)
        return agent_ids # type: ignore[no-any-return]
    except Exception as e:
        logger.error(f"Error getting agents by status {status}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve agents by status")


# Example Top N routes (implement corresponding methods in AgentManager)
@router.get("/top/cpu", response_model=List[Dict[str, Any]])
def get_top_agents_by_cpu(
    limit: int = Query(5, ge=1, le=50),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> List[Dict[str, Any]]:
    """Get agents with the highest CPU usage."""
    if not hasattr(agent_manager, 'get_top_agents_by_metric'):
        logger.warning("AgentManager does not implement get_top_agents_by_metric")
        return [] # Return empty list to satisfy type hint
    return agent_manager.get_top_agents_by_metric("cpu_percent", limit, ascending=False) # type: ignore[no-any-return]


@router.get("/top/memory", response_model=List[Dict[str, Any]])
def get_top_agents_by_memory(
    limit: int = Query(5, ge=1, le=50),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> List[Dict[str, Any]]:
    """Get agents with the highest memory usage."""
    if not hasattr(agent_manager, 'get_top_agents_by_metric'):
        logger.warning("AgentManager does not implement get_top_agents_by_metric")
        return [] # Return empty list to satisfy type hint
    return agent_manager.get_top_agents_by_metric("memory_percent", limit, ascending=False) # type: ignore[no-any-return]


@router.get("/top/errors", response_model=List[Dict[str, Any]])
def get_top_agents_by_errors(
    limit: int = Query(5, ge=1, le=50),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> List[Dict[str, Any]]:
    """Get agents with the highest error count."""
    if not hasattr(agent_manager, 'get_top_agents_by_metric'):
        logger.warning("AgentManager does not implement get_top_agents_by_metric")
        return [] # Return empty list to satisfy type hint
    return agent_manager.get_top_agents_by_metric("error_count", limit, ascending=False) # type: ignore[no-any-return]


# Functions to extract specific metrics (used by examples above, implement in AgentManager)
# These are placeholders assuming get_top_agents_by_metric exists
# def _get_cpu_usage(agent_id: str, metrics: Dict[str, AgentMetrics]) -> float:
#     return metrics.get(agent_id, AgentMetrics()).cpu_percent

# def _get_memory_usage(agent_id: str, metrics: Dict[str, AgentMetrics]) -> float:
#     return metrics.get(agent_id, AgentMetrics()).memory_percent

# def _get_error_rate(agent_id: str, metrics: Dict[str, AgentMetrics]) -> float:
#     agent_metrics = metrics.get(agent_id, AgentMetrics())
#     if agent_metrics.execution_count == 0:
#         return 0.0
#     return agent_metrics.error_count / agent_metrics.execution_count
