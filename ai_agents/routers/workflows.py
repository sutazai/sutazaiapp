"""
Workflow Router Module

This module provides REST API endpoints for managing workflows between multiple agents.
Supports creating workflows, adding tasks, managing dependencies, and executing workflows.
"""

from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body
from pydantic import BaseModel, Field
from datetime import datetime

from ..agent_manager import AgentManager
from ..dependencies import get_agent_manager


router = APIRouter()


class WorkflowTaskBase(BaseModel):
    """Base model for workflow task."""

    agent_id: str = Field(..., description="ID of the agent to execute the task")
    task_type: str = Field(..., description="Type of task to execute")
    parameters: Dict[str, Any] = Field(
        default_factory=dict, description="Task parameters"
    )
    dependencies: List[str] = Field(
        default_factory=list, description="IDs of tasks this task depends on"
    )


class WorkflowTaskResponse(WorkflowTaskBase):
    """Response model for workflow task."""

    task_id: str = Field(..., description="Task ID")
    status: str = Field(..., description="Task status")
    created_at: datetime = Field(..., description="Time the task was created")
    started_at: Optional[datetime] = Field(
        None, description="Time the task was started"
    )
    completed_at: Optional[datetime] = Field(
        None, description="Time the task was completed"
    )
    result: Optional[Dict[str, Any]] = Field(None, description="Task result")
    error: Optional[str] = Field(None, description="Error message if task failed")


class WorkflowCreate(BaseModel):
    """Model for creating a workflow."""

    name: str = Field(..., description="Workflow name")
    description: str = Field(default="", description="Workflow description")
    initial_tasks: List[WorkflowTaskBase] = Field(
        default_factory=list, description="Initial tasks for the workflow"
    )


class WorkflowResponse(BaseModel):
    """Response model for workflow."""

    workflow_id: str = Field(..., description="Workflow ID")
    name: str = Field(..., description="Workflow name")
    description: str = Field(..., description="Workflow description")
    created_at: datetime = Field(..., description="Time the workflow was created")
    status: str = Field(..., description="Workflow status")
    tasks: Dict[str, WorkflowTaskResponse] = Field(
        default_factory=dict, description="Tasks in the workflow"
    )


class WorkflowExecuteOptions(BaseModel):
    """Model for workflow execution options."""

    async_execution: bool = Field(
        default=True, description="Whether to execute asynchronously"
    )


@router.post("", response_model=WorkflowResponse)
async def create_workflow(
    workflow: WorkflowCreate, agent_manager: AgentManager = Depends(get_agent_manager)
):
    """
    Create a new workflow with optional initial tasks.
    """
    try:
        # Create the workflow
        workflow_id = agent_manager.create_workflow(
            name=workflow.name, description=workflow.description
        )

        # Add initial tasks
        for task in workflow.initial_tasks:
            agent_manager.add_workflow_task(
                workflow_id=workflow_id,
                agent_id=task.agent_id,
                task_type=task.task_type,
                parameters=task.parameters,
                dependencies=task.dependencies,
            )

        # Get the workflow status
        return agent_manager.get_workflow_status(workflow_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("", response_model=List[WorkflowResponse])
async def list_workflows(
    status: Optional[str] = Query(None, description="Filter by workflow status"),
    limit: int = Query(10, description="Maximum number of workflows to return"),
    offset: int = Query(0, description="Offset for pagination"),
    agent_manager: AgentManager = Depends(get_agent_manager),
):
    """
    List all workflows with optional filtering.
    """
    try:
        # TODO: Implement workflow listing with filtering
        # This is a placeholder for now
        return []
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: str = Path(..., description="Workflow ID to get"),
    agent_manager: AgentManager = Depends(get_agent_manager),
):
    """
    Get a specific workflow by ID.
    """
    try:
        return agent_manager.get_workflow_status(workflow_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Workflow not found: {str(e)}")


@router.post("/{workflow_id}/tasks", response_model=WorkflowTaskResponse)
async def add_workflow_task(
    workflow_id: str = Path(..., description="Workflow ID to add task to"),
    task: WorkflowTaskBase = Body(..., description="Task to add"),
    agent_manager: AgentManager = Depends(get_agent_manager),
):
    """
    Add a task to an existing workflow.
    """
    try:
        task_id = agent_manager.add_workflow_task(
            workflow_id=workflow_id,
            agent_id=task.agent_id,
            task_type=task.task_type,
            parameters=task.parameters,
            dependencies=task.dependencies,
        )

        # Get the updated workflow status
        workflow = agent_manager.get_workflow_status(workflow_id)
        return workflow["tasks"][task_id]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{workflow_id}/execute", response_model=Dict[str, Any])
async def execute_workflow(
    workflow_id: str = Path(..., description="Workflow ID to execute"),
    options: WorkflowExecuteOptions = Body(
        default=WorkflowExecuteOptions(), description="Execution options"
    ),
    agent_manager: AgentManager = Depends(get_agent_manager),
):
    """
    Execute a workflow with the specified options.
    """
    try:
        result = agent_manager.execute_workflow(
            workflow_id=workflow_id, async_execution=options.async_execution
        )

        if options.async_execution:
            return {"status": "executing", "workflow_id": workflow_id}
        else:
            return {"status": "completed", "workflow_id": workflow_id, "result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{workflow_id}", response_model=Dict[str, Any])
async def cancel_workflow(
    workflow_id: str = Path(..., description="Workflow ID to cancel"),
    agent_manager: AgentManager = Depends(get_agent_manager),
):
    """
    Cancel a running workflow.
    """
    try:
        success = agent_manager.cancel_workflow(workflow_id)
        if success:
            return {"status": "cancelled", "workflow_id": workflow_id}
        else:
            raise HTTPException(status_code=400, detail="Failed to cancel workflow")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{workflow_id}/tasks/{task_id}", response_model=WorkflowTaskResponse)
async def get_workflow_task(
    workflow_id: str = Path(..., description="Workflow ID"),
    task_id: str = Path(..., description="Task ID to get"),
    agent_manager: AgentManager = Depends(get_agent_manager),
):
    """
    Get a specific task in a workflow.
    """
    try:
        workflow = agent_manager.get_workflow_status(workflow_id)
        if task_id not in workflow["tasks"]:
            raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

        return workflow["tasks"][task_id]
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))
