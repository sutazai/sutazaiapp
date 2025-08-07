"""
Canonical agent protocol models.

Source of truth: IMPORTANT/COMPREHENSIVE_ENGINEERING_STANDARDS.md
- Eliminate duplicate class definitions across modules
- Provide distinct schema variants to preserve existing API surfaces

Do not import external libraries beyond Pydantic and stdlib.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# Variant A: Detailed single-agent task request/response (agent_interaction)
class TaskRequestDetailed(BaseModel):
    """Request model for agent task execution (detailed variant)."""

    agent_name: str = Field(..., description="Name of the agent to execute task")
    task_type: str = Field(..., description="Type of task to execute")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Task parameters"
    )
    priority: int = Field(default=5, ge=1, le=10, description="Task priority (1-10)")
    timeout: Optional[int] = Field(
        default=300, description="Task timeout in seconds"
    )


class TaskResponseDetailed(BaseModel):
    """Response model for agent task execution (detailed variant)."""

    task_id: str = Field(..., description="Unique task identifier")
    agent_name: str = Field(..., description="Name of the executing agent")
    status: str = Field(..., description="Task status")
    created_at: datetime = Field(..., description="Task creation timestamp")
    result: Optional[Dict[str, Any]] = Field(
        None, description="Task result if completed"
    )
    error: Optional[str] = Field(None, description="Error message if failed")


class AgentStatusSingle(BaseModel):
    """Status model for a single agent."""

    agent_name: str = Field(..., description="Agent name")
    status: str = Field(..., description="Agent status (active/inactive/busy)")
    current_task: Optional[str] = Field(None, description="Current task if any")
    capabilities: List[str] = Field(
        default_factory=list, description="Agent capabilities"
    )
    performance_metrics: Dict[str, float] = Field(
        default_factory=dict, description="Performance metrics"
    )


# Variant B: Basic multi-agent task and status (api/v1/agents)
class TaskRequestBasic(BaseModel):
    """Request model for agent task execution (basic variant)."""

    agent_type: str
    task_type: str
    task_data: Dict[str, Any]
    preferred_agents: Optional[List[str]] = None


class TaskResponseBasic(BaseModel):
    task_id: str
    status: str
    agents_used: List[str]
    results: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None


class AgentStatusSummary(BaseModel):
    total_agents: int
    active_agents: int
    agents: Dict[str, Dict[str, Any]]

