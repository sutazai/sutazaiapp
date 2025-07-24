"""
Workflow Router Module

This module provides REST API endpoints for managing workflows between multiple agents.
Supports creating workflows, adding tasks, managing dependencies, and executing workflows.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Path, Body, status
from pydantic import BaseModel, Field
from datetime import datetime

from ai_agents.agent_manager import AgentManager
from ai_agents.dependencies import get_agent_manager

# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/workflows", tags=["workflows"])


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


class WorkflowDefinition(BaseModel):
    name: str
    description: Optional[str] = None
    tasks: List[Dict[str, Any]] # Detailed task definition schema needed


class WorkflowInstance(BaseModel):
    workflow_id: str
    status: str
    created_at: str # Use datetime if possible
    # Add other relevant fields


class TaskUpdate(BaseModel):
    status: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@router.post("/", response_model=WorkflowInstance, status_code=status.HTTP_201_CREATED)
def create_workflow_route(
    workflow_def: WorkflowDefinition,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> WorkflowInstance:
    """Define and create a new workflow instance."""
    try:
        if not hasattr(agent_manager, 'workflow_engine'):
             raise HTTPException(status_code=501, detail="Workflow engine not available")

        # 1. Create the workflow shell
        workflow_id = agent_manager.workflow_engine.create_workflow(
            name=workflow_def.name,
            description=workflow_def.description or "", # Pass empty string if None
            # metadata can be passed if needed, e.g., metadata=workflow_def.metadata
        )

        # 2. Add tasks to the created workflow
        task_ids_map = {} # Maps definition task ID (if any) to engine task ID
        for task_def in workflow_def.tasks:
             engine_task_id = agent_manager.workflow_engine.add_task(
                  workflow_id=workflow_id,
                  agent_id=task_def.get("agent_id"),
                  task_type=task_def.get("task_type", "unknown"),
                  parameters=task_def.get("parameters", {}),
                  dependencies=task_def.get("dependencies", []), # Initial dependencies
                  # Pass other relevant parameters from task_def if WorkflowEngine.add_task supports them
             )
             # Store mapping if needed for setting cross-dependencies later
             original_task_id = task_def.get("id")
             if original_task_id:
                  task_ids_map[original_task_id] = engine_task_id

        # Optional: Add cross-task dependencies if defined in workflow_def using mapped IDs
        # This requires a more complex definition in WorkflowDefinition

        # 3. Fetch status of the created workflow
        status = agent_manager.workflow_engine.get_workflow_status(workflow_id)
        if not status:
             # Should not happen if creation succeeded, but handle defensively
             raise HTTPException(status_code=500, detail="Failed to retrieve status after creation")
        return WorkflowInstance(**status)

    except ValueError as ve:
        # Specific error from WorkflowEngine (e.g., invalid ID)
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error creating workflow {workflow_def.name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create workflow: {str(e)}")


@router.get("/", response_model=List[WorkflowInstance])
def list_workflows(
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> List[WorkflowInstance]:
    """List all defined workflow instances."""
    try:
        if not hasattr(agent_manager, 'workflow_engine'):
             raise HTTPException(status_code=501, detail="Workflow engine not available")

        # Access the workflows dictionary and get status for each
        workflows_dict = agent_manager.workflow_engine.workflows
        workflow_statuses = []
        for wf_id in workflows_dict:
             status = agent_manager.workflow_engine.get_workflow_status(wf_id)
             if status:
                  workflow_statuses.append(status)

        return [WorkflowInstance(**wf_status) for wf_status in workflow_statuses]
    except Exception as e:
        logger.error(f"Error listing workflows: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list workflows")


@router.get("/{workflow_id}", response_model=WorkflowInstance) # Or a more detailed model
def get_workflow_details(
    workflow_id: str,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> WorkflowInstance:
    """Get details and status of a specific workflow instance."""
    try:
        if not hasattr(agent_manager, 'workflow_engine'):
             raise HTTPException(status_code=501, detail="Workflow engine not available")

        status = agent_manager.workflow_engine.get_workflow_status(workflow_id)
        if not status:
             raise HTTPException(status_code=404, detail=f"Workflow {workflow_id} not found")
        return WorkflowInstance(**status)
    except ValueError as e:
         raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting workflow details for {workflow_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get workflow details")


@router.post("/{workflow_id}/execute")
def execute_workflow_route(
    workflow_id: str,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, Any]:
    """Execute a defined workflow instance."""
    try:
        if not hasattr(agent_manager, 'workflow_engine'):
             raise HTTPException(status_code=501, detail="Workflow engine not available")

        result = agent_manager.workflow_engine.execute_workflow(workflow_id)
        return {"status": "execution started", "workflow_id": workflow_id, "result": result} # Adjust response as needed
    except ValueError as e:
         raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing workflow {workflow_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to execute workflow")


@router.post("/{workflow_id}/cancel")
def cancel_workflow_route(
    workflow_id: str,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, str]:
    """Cancel a running workflow instance."""
    try:
        if not hasattr(agent_manager, 'workflow_engine'):
             raise HTTPException(status_code=501, detail="Workflow engine not available")

        success = agent_manager.workflow_engine.cancel_workflow(workflow_id)
        if not success:
             raise HTTPException(status_code=400, detail=f"Could not cancel workflow {workflow_id} (may not exist or already completed)")
        return {"status": "success", "message": f"Workflow {workflow_id} cancellation requested"}
    except ValueError as e:
         raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error cancelling workflow {workflow_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to cancel workflow")


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


# Add other potential workflow endpoints:
# - Pause/Resume workflow
# - Get task status within a workflow
# - Update workflow definition (if mutable)
# - Delete workflow definition
