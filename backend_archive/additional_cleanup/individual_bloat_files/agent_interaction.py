"""
Agent Interaction Router

This module provides REST API endpoints for human interaction with agents,
enabling approval workflows, notifications, and task delegation.
"""

from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body
from pydantic import BaseModel, Field
from enum import Enum

from ai_agents.interaction.human_interaction import (
    InteractionManager,
    HumanInteractionPoint,
    InteractionType,
)
from ai_agents.dependencies import get_interaction_manager


router = APIRouter()


class InteractionTypeEnum(str, Enum):
    """Enum for interaction types."""

    APPROVAL = "approval"
    DECISION = "decision"
    INFORMATION = "information"
    INPUT = "input"
    ESCALATION = "escalation"


class InteractionStatusEnum(str, Enum):
    """Enum for interaction status."""

    PENDING = "pending"
    ACKNOWLEDGED = "acknowledged"
    COMPLETED = "completed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class OptionModel(BaseModel):
    """Model for interaction option."""

    id: str = Field(..., description="Option ID")
    label: str = Field(..., description="Option label")
    value: Any = Field(..., description="Option value")
    description: Optional[str] = Field(None, description="Option description")


class InteractionCreateModel(BaseModel):
    """Model for creating an interaction request."""

    interaction_type: InteractionTypeEnum = Field(
        ..., description="Type of interaction"
    )
    title: str = Field(..., description="Interaction title")
    description: str = Field(..., description="Interaction description")
    agent_id: str = Field(..., description="ID of the requesting agent")
    user_id: Optional[str] = Field(None, description="ID of the target user")
    task_id: Optional[str] = Field(None, description="Associated task ID")
    workflow_id: Optional[str] = Field(None, description="Associated workflow ID")
    data: Dict[str, Any] = Field(
        default={}, description="Additional data for the interaction"
    )
    options: List[OptionModel] = Field(
        default=[], description="Options for the interaction"
    )
    priority: int = Field(3, description="Priority level (1-5, 1 is highest)")
    timeout_seconds: Optional[int] = Field(None, description="Timeout in seconds")


class InteractionResponseModel(BaseModel):
    """Model for responding to an interaction request."""

    user_id: str = Field(..., description="ID of the responding user")
    response: Dict[str, Any] = Field(..., description="Response data")


class InteractionViewModel(BaseModel):
    """Model for viewing an interaction request."""

    request_id: str = Field(..., description="Request ID")
    interaction_type: str = Field(..., description="Type of interaction")
    title: str = Field(..., description="Interaction title")
    description: str = Field(..., description="Interaction description")
    agent_id: str = Field(..., description="ID of the requesting agent")
    user_id: Optional[str] = Field(None, description="ID of the target user")
    task_id: Optional[str] = Field(None, description="Associated task ID")
    workflow_id: Optional[str] = Field(None, description="Associated workflow ID")
    data: Dict[str, Any] = Field(..., description="Additional data for the interaction")
    options: List[Dict[str, Any]] = Field(
        ..., description="Options for the interaction"
    )
    status: str = Field(..., description="Current status")
    priority: int = Field(..., description="Priority level (1-5, 1 is highest)")
    created_at: str = Field(..., description="Creation timestamp")
    expires_at: Optional[str] = Field(None, description="Expiration timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")
    response: Optional[Dict[str, Any]] = Field(
        None, description="Response data if completed"
    )


@router.post("/interactions", response_model=Dict[str, str])
async def create_interaction(
    interaction: InteractionCreateModel = Body(...),
    interaction_manager: InteractionManager = Depends(get_interaction_manager),
):
    """
    Create a new interaction request.
    """
    try:
        # Convert enum to InteractionType
        interaction_type = InteractionType(interaction.interaction_type.value)

        # Create HumanInteractionPoint
        hip = HumanInteractionPoint(
            interaction_type=interaction_type,
            title=interaction.title,
            description=interaction.description,
            agent_id=interaction.agent_id,
            options=[option.dict() for option in interaction.options],
            data=interaction.data,
            timeout=interaction.timeout_seconds,
            priority=interaction.priority,
            user_id=interaction.user_id,
            task_id=interaction.task_id,
            workflow_id=interaction.workflow_id,
        )

        # Create interaction
        request_id = interaction_manager.create_interaction(hip)

        return {"request_id": request_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/interactions", response_model=List[InteractionViewModel])
async def get_pending_interactions(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    interaction_type: Optional[InteractionTypeEnum] = Query(
        None, description="Filter by interaction type"
    ),
    status: Optional[InteractionStatusEnum] = Query(
        None, description="Filter by status"
    ),
    interaction_manager: InteractionManager = Depends(get_interaction_manager),
):
    """
    Get pending interaction requests.
    """
    try:
        # Convert enum to InteractionType if provided
        interaction_type_enum = None
        if interaction_type:
            interaction_type_enum = InteractionType(interaction_type.value)

        # Get pending interactions
        interactions = interaction_manager.get_pending_interactions(
            user_id=user_id, interaction_type=interaction_type_enum
        )

        # Filter by status if provided
        if status:
            interactions = [
                interaction
                for interaction in interactions
                if interaction["status"] == status.value
            ]

        return interactions
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/interactions/{request_id}", response_model=InteractionViewModel)
async def get_interaction_status(
    request_id: str = Path(..., description="Interaction request ID"),
    interaction_manager: InteractionManager = Depends(get_interaction_manager),
):
    """
    Get the status of an interaction request.
    """
    try:
        interaction = interaction_manager.get_interaction_status(request_id)
        if not interaction:
            raise HTTPException(
                status_code=404, detail=f"Interaction not found: {request_id}"
            )

        return interaction
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/interactions/{request_id}/acknowledge", response_model=Dict[str, bool])
async def acknowledge_interaction(
    request_id: str = Path(..., description="Interaction request ID"),
    user_id: str = Query(..., description="ID of the acknowledging user"),
    interaction_manager: InteractionManager = Depends(get_interaction_manager),
):
    """
    Acknowledge an interaction request.
    """
    try:
        success = interaction_manager.acknowledge_interaction(
            request_id=request_id, user_id=user_id
        )

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Failed to acknowledge interaction: {request_id}",
            )

        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/interactions/{request_id}/respond", response_model=Dict[str, bool])
async def respond_to_interaction(
    request_id: str = Path(..., description="Interaction request ID"),
    response: InteractionResponseModel = Body(...),
    interaction_manager: InteractionManager = Depends(get_interaction_manager),
):
    """
    Respond to an interaction request.
    """
    try:
        success = interaction_manager.respond_to_interaction(
            request_id=request_id, user_id=response.user_id, response=response.response
        )

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Failed to respond to interaction: {request_id}",
            )

        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/interactions/{request_id}", response_model=Dict[str, bool])
async def cancel_interaction(
    request_id: str = Path(..., description="Interaction request ID"),
    interaction_manager: InteractionManager = Depends(get_interaction_manager),
):
    """
    Cancel an interaction request.
    """
    try:
        success = interaction_manager.cancel_interaction(request_id=request_id)

        if not success:
            raise HTTPException(
                status_code=404, detail=f"Failed to cancel interaction: {request_id}"
            )

        return {"success": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
