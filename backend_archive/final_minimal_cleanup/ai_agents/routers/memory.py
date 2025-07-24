"""
Memory Router Module

This module provides REST API endpoints for managing agent memory and shared memory systems.
Supports creating, retrieving, updating, and searching memory entries.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, status
from pydantic import BaseModel, Field, validator
from enum import Enum

from ai_agents.agent_manager import AgentManager
from ai_agents.dependencies import get_agent_manager
from ai_agents.memory.memory_models import MemoryItem, MemoryQuery, MemoryUpdate
from ai_agents.memory.shared_memory_models import SharedMemoryQuery


# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/memory", tags=["memory"])


class MemoryTypeEnum(str, Enum):
    """Enum for memory types."""

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    WORKING = "working"
    EPISODIC = "episodic"


class MemoryEntryBase(BaseModel):
    """Base model for memory entry."""

    content: Dict[str, Any] = Field(..., description="Memory content")
    tags: Optional[List[str]] = Field(None, description="Tags for the memory entry")
    importance: float = Field(0.5, description="Importance of the memory (0.0 to 1.0)")
    source: Optional[str] = Field(None, description="Source of the memory")

    @validator("importance")
    def validate_importance(cls, v):
        """Validate that importance is between 0.0 and 1.0."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Importance must be between 0.0 and 1.0")
        return v


class AgentMemoryCreate(MemoryEntryBase):
    """Model for creating an agent memory entry."""

    memory_type: MemoryTypeEnum = Field(
        MemoryTypeEnum.SHORT_TERM, description="Type of memory"
    )


class SharedMemoryCreate(MemoryEntryBase):
    """Model for creating a shared memory entry."""

    creator_id: str = Field(..., description="ID of the agent creating the memory")
    access_control: Optional[Dict[str, str]] = Field(
        None, description="Access control settings"
    )


class AgentMemoryResponse(MemoryEntryBase):
    """Response model for agent memory."""

    entry_id: str = Field(..., description="Memory entry ID")
    memory_type: str = Field(..., description="Type of memory")
    timestamp: float = Field(..., description="Timestamp when the memory was created")
    last_accessed: float = Field(
        ..., description="Timestamp when the memory was last accessed"
    )
    access_count: int = Field(
        ..., description="Number of times the memory has been accessed"
    )


class SharedMemoryResponse(MemoryEntryBase):
    """Response model for shared memory."""

    entry_id: str = Field(..., description="Memory entry ID")
    creator_id: str = Field(..., description="ID of the agent that created the entry")
    timestamp: float = Field(..., description="Timestamp when the memory was created")
    last_accessed: float = Field(
        ..., description="Timestamp when the memory was last accessed"
    )
    access_count: int = Field(
        ..., description="Number of times the memory has been accessed"
    )
    version: int = Field(..., description="Version of the entry")
    access_control: Dict[str, str] = Field(..., description="Access control settings")


class MemorySummary(BaseModel):
    """Model for memory summary."""

    total_entries: int = Field(..., description="Total number of memory entries")
    memory_types: Dict[str, int] = Field(
        ..., description="Count of entries by memory type"
    )
    recent_entries: List[Dict[str, Any]] = Field(
        ..., description="Recent memory entries"
    )
    memory_size: int = Field(..., description="Total size of memory in bytes")


class SharedMemorySpace(BaseModel):
    """Model for shared memory space."""

    name: str = Field(..., description="Name of the shared memory space")
    description: str = Field(
        default="", description="Description of the shared memory space"
    )
    max_entries: int = Field(1000, description="Maximum number of entries allowed")

    @validator("max_entries")
    def validate_max_entries(cls, v):
        """Validate that max_entries is a positive number."""
        if v <= 0:
            raise ValueError("Maximum entries must be a positive number")
        return v


# Agent Memory Endpoints


@router.post("/agents/{agent_id}", response_model=MemoryItem)
def add_agent_memory(
    agent_id: str,
    item: MemoryItem,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> MemoryItem:
    """Add an item to an agent's memory."""
    try:
        memory = agent_manager.memory_manager.get_memory(agent_id)
        if not memory:
            raise HTTPException(status_code=404, detail=f"Memory not found for agent {agent_id}")
        added_item = memory.add_memory(item)
        return added_item
    except Exception as e:
        logger.error(f"Error adding memory for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to add memory: {e}")


@router.get("/agents/{agent_id}", response_model=List[AgentMemoryResponse])
def get_agent_memory_route(
    agent_id: str,
    limit: int = Query(10, ge=1, le=1000),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> List[AgentMemoryResponse]:
    """Get recent memory items for an agent."""
    try:
        memory = agent_manager.memory_manager.get_memory(agent_id)
        if not memory:
            raise HTTPException(status_code=404, detail=f"Memory not found for agent {agent_id}")
        recent_entries = memory.get_recent_memories(limit=limit)
        return [AgentMemoryResponse(**entry.to_dict()) for entry in recent_entries] # type: ignore[no-any-return]
    except Exception as e:
        logger.error(f"Error getting memory for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get agent memory")


@router.post("/agents/{agent_id}/search", response_model=List[MemoryItem])
def search_agent_memory(
    agent_id: str,
    query: MemoryQuery,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> List[MemoryItem]:
    """Search an agent's memory."""
    try:
        memory = agent_manager.memory_manager.get_memory(agent_id)
        if not memory:
            raise HTTPException(status_code=404, detail=f"Memory not found for agent {agent_id}")
        return memory.search_memory(query.query, query.limit, query.threshold)
    except Exception as e:
        logger.error(f"Error searching memory for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to search agent memory")


@router.put("/agents/{agent_id}/{memory_id}", response_model=MemoryItem)
def update_agent_memory(
    agent_id: str,
    memory_id: str,
    update: MemoryUpdate,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> MemoryItem:
    """Update a specific memory item for an agent."""
    try:
        memory = agent_manager.memory_manager.get_memory(agent_id)
        if not memory:
            raise HTTPException(status_code=404, detail=f"Memory not found for agent {agent_id}")
        updated = memory.update_memory(memory_id, update.content, update.metadata)
        if not updated:
             raise HTTPException(status_code=404, detail=f"Memory item {memory_id} not found")
        return updated
    except Exception as e:
        logger.error(f"Error updating memory {memory_id} for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update memory")


@router.delete("/agents/{agent_id}/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_agent_memory(
    agent_id: str,
    memory_id: str,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> None:
    """Delete a specific memory item for an agent."""
    try:
        memory = agent_manager.memory_manager.get_memory(agent_id)
        if not memory:
            raise HTTPException(status_code=404, detail=f"Memory not found for agent {agent_id}")
        if not memory.delete_memory(memory_id):
             pass # Or log warning
        return None
    except Exception as e:
        logger.error(f"Error deleting memory {memory_id} for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to delete memory")


@router.post("/agents/{agent_id}/clear")
def clear_agent_memory(
    agent_id: str,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, str]:
    """Clear all memory for an agent."""
    try:
        memory = agent_manager.memory_manager.get_memory(agent_id)
        if not memory:
            raise HTTPException(status_code=404, detail=f"Memory not found for agent {agent_id}")
        memory.clear_memory()
        return {"status": "success", "message": f"Memory cleared for agent {agent_id}"}
    except Exception as e:
        logger.error(f"Error clearing memory for agent {agent_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to clear memory")


# Shared Memory Endpoints


@router.post("/shared/{memory_name}", response_model=SharedMemoryResponse)
def add_shared_memory_entry(
    memory_name: str,
    item_create: SharedMemoryCreate,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> SharedMemoryResponse:
    """Add an item to a specific shared memory space."""
    shared_memory_manager = agent_manager.shared_memory_manager
    if not shared_memory_manager:
         raise HTTPException(status_code=501, detail="Shared Memory Manager not available")
    try:
        memory = shared_memory_manager.get_memory(memory_name)
        if not memory:
             raise HTTPException(status_code=404, detail=f"Shared memory space '{memory_name}' not found")
        entry_id = memory.add_entry(
            creator_id=item_create.creator_id,
            content=item_create.content,
            tags=set(item_create.tags) if item_create.tags else None,
            importance=item_create.importance,
            access_control=item_create.access_control
        )
        new_entry = memory.get_entry(entry_id)
        if not new_entry:
             raise HTTPException(status_code=500, detail="Failed to retrieve created entry")
        return SharedMemoryResponse(**new_entry.to_dict())
    except ValueError as ve:
         raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error adding entry to shared memory '{memory_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to add shared memory entry")


@router.get("/shared/{memory_name}/{item_key}", response_model=SharedMemoryResponse)
def get_shared_memory_entry(
    memory_name: str,
    item_key: str,
    agent_id: Optional[str] = Query(None, description="Requesting agent ID for access control"),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> SharedMemoryResponse:
    """Get a specific item from shared memory."""
    shared_memory_manager = agent_manager.shared_memory_manager
    if not shared_memory_manager:
         raise HTTPException(status_code=501, detail="Shared Memory Manager not available")
    memory = shared_memory_manager.get_memory(memory_name)
    if not memory:
        raise HTTPException(status_code=404, detail=f"Shared memory space '{memory_name}' not found")
    item = memory.get_entry(item_key, agent_id=agent_id)
    if not item:
        raise HTTPException(status_code=404, detail=f"Item '{item_key}' not found or access denied")
    return SharedMemoryResponse(**item.to_dict())


@router.put("/shared/{memory_name}/{item_key}", response_model=SharedMemoryResponse)
def update_shared_memory_entry(
    memory_name: str,
    item_key: str,
    item_update: MemoryUpdate,
    agent_id: str = Query(..., description="Updating agent ID for access control"),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> SharedMemoryResponse:
    """Update a specific item in shared memory."""
    shared_memory_manager = agent_manager.shared_memory_manager
    if not shared_memory_manager:
         raise HTTPException(status_code=501, detail="Shared Memory Manager not available")
    memory = shared_memory_manager.get_memory(memory_name)
    if not memory:
        raise HTTPException(status_code=404, detail=f"Shared memory space '{memory_name}' not found")
    updated = memory.update_entry(
         entry_id=item_key,
            agent_id=agent_id,
         content=item_update.content,
         tags=set(item_update.metadata.get('tags')) if item_update.metadata and 'tags' in item_update.metadata else None,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Item '{item_key}' not found or update failed (access denied?)")
    updated_entry = memory.get_entry(item_key)
    if not updated_entry:
         raise HTTPException(status_code=500, detail="Failed to retrieve updated entry")
    return SharedMemoryResponse(**updated_entry.to_dict())


@router.delete("/shared/{memory_name}/{item_key}", status_code=status.HTTP_204_NO_CONTENT)
def delete_shared_memory_entry(
    memory_name: str,
    item_key: str,
    agent_id: str = Query(..., description="Deleting agent ID for access control"),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> None:
    """Delete a specific item from shared memory."""
    shared_memory_manager = agent_manager.shared_memory_manager
    if not shared_memory_manager:
         raise HTTPException(status_code=501, detail="Shared Memory Manager not available")
    memory = shared_memory_manager.get_memory(memory_name)
    if not memory:
        raise HTTPException(status_code=404, detail=f"Shared memory space '{memory_name}' not found")
    deleted = memory.delete_entry(item_key, agent_id=agent_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Item '{item_key}' not found or delete failed (access denied?)")
    return None


@router.post("/shared/{memory_name}/search", response_model=List[SharedMemoryResponse])
def search_shared_memory(
    memory_name: str,
    query: SharedMemoryQuery,
    agent_id: Optional[str] = Query(None, description="Requesting agent ID for access control"),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> List[SharedMemoryResponse]:
    """Search shared memory."""
    shared_memory_manager = agent_manager.shared_memory_manager
    if not shared_memory_manager:
         raise HTTPException(status_code=501, detail="Shared Memory Manager not available")
    memory = shared_memory_manager.get_memory(memory_name)
    if not memory:
        raise HTTPException(status_code=404, detail=f"Shared memory space '{memory_name}' not found")
    if query.tags:
         results = memory.search_by_tags(tags=set(query.tags), agent_id=agent_id, require_all=False)
    elif query.query:
         results = memory.search_by_content(query=query.query, agent_id=agent_id)
    else:
         results = memory.get_all_entries(agent_id=agent_id)[:query.limit or 100]

    return [SharedMemoryResponse(**entry.to_dict()) for entry in results]


@router.get("/shared/{memory_name}/list", response_model=List[str])
def list_shared_memory_keys(
    memory_name: str,
    agent_id: Optional[str] = Query(None, description="Requesting agent ID for access control"),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> List[str]:
    """List all item keys in a shared memory space (respecting access)."""
    shared_memory_manager = agent_manager.shared_memory_manager
    if not shared_memory_manager:
         raise HTTPException(status_code=501, detail="Shared Memory Manager not available")
    memory = shared_memory_manager.get_memory(memory_name)
    if not memory:
        raise HTTPException(status_code=404, detail=f"Shared memory space '{memory_name}' not found")
    entries = memory.get_all_entries(agent_id=agent_id)
    return [entry.entry_id for entry in entries]


@router.post("/shared/{memory_name}/clear")
def clear_shared_memory(
    memory_name: str,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, Any]:
    """Clear all items in a shared memory space."""
    shared_memory_manager = agent_manager.shared_memory_manager
    if not shared_memory_manager:
         raise HTTPException(status_code=501, detail="Shared Memory Manager not available")
    memory = shared_memory_manager.get_memory(memory_name)
    if not memory:
        raise HTTPException(status_code=404, detail=f"Shared memory space '{memory_name}' not found")
    count = memory.clear()
    return {"status": "success", "message": f"Cleared {count} items from {memory_name}"}
