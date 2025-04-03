"""
Memory Router Module

This module provides REST API endpoints for managing agent memory and shared memory systems.
Supports creating, retrieving, updating, and searching memory entries.
"""

import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Path, Body, status
from pydantic import BaseModel, Field, validator
from enum import Enum

from ai_agents.agent_manager import AgentManager
from ai_agents.dependencies import get_agent_manager
from ai_agents.memory.memory_models import MemoryItem, MemoryQuery, MemoryUpdate
from ai_agents.memory.shared_memory_models import SharedMemoryItem, SharedMemoryQuery


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
    memory = agent_manager.get_agent_memory(agent_id)
    if not memory:
        raise HTTPException(status_code=404, detail=f"Memory not found for agent {agent_id}")
    try:
        added_item = memory.add_memory(item)
        return added_item
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add memory: {e}")


@router.get("/agents/{agent_id}", response_model=List[MemoryItem])
def get_agent_memory(
    agent_id: str,
    limit: int = Query(100, ge=1, le=1000),
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> List[MemoryItem]:
    """Get memory items for an agent."""
    memory = agent_manager.get_agent_memory(agent_id)
    if not memory:
        raise HTTPException(status_code=404, detail=f"Memory not found for agent {agent_id}")
    return memory.get_memory(limit=limit)


@router.post("/agents/{agent_id}/search", response_model=List[MemoryItem])
def search_agent_memory(
    agent_id: str,
    query: MemoryQuery,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> List[MemoryItem]:
    """Search an agent's memory."""
    memory = agent_manager.get_agent_memory(agent_id)
    if not memory:
        raise HTTPException(status_code=404, detail=f"Memory not found for agent {agent_id}")
    return memory.search_memory(query.query, query.limit, query.threshold)


@router.put("/agents/{agent_id}/{memory_id}", response_model=MemoryItem)
def update_agent_memory(
    agent_id: str,
    memory_id: str,
    update: MemoryUpdate,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> MemoryItem:
    """Update a specific memory item for an agent."""
    memory = agent_manager.get_agent_memory(agent_id)
    if not memory:
        raise HTTPException(status_code=404, detail=f"Memory not found for agent {agent_id}")
    updated = memory.update_memory(memory_id, update.content, update.metadata)
    if not updated:
         raise HTTPException(status_code=404, detail=f"Memory item {memory_id} not found")
    return updated # Assuming update_memory returns the updated item


@router.delete("/agents/{agent_id}/{memory_id}", status_code=204)
def delete_agent_memory(
    agent_id: str,
    memory_id: str,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> None:
    """Delete a specific memory item for an agent."""
    memory = agent_manager.get_agent_memory(agent_id)
    if not memory:
        raise HTTPException(status_code=404, detail=f"Memory not found for agent {agent_id}")
    if not memory.delete_memory(memory_id):
         raise HTTPException(status_code=404, detail=f"Memory item {memory_id} not found")
    return None


@router.post("/agents/{agent_id}/clear")
def clear_agent_memory(
    agent_id: str,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, str]:
    """Clear all memory for an agent."""
    memory = agent_manager.get_agent_memory(agent_id)
    if not memory:
        raise HTTPException(status_code=404, detail=f"Memory not found for agent {agent_id}")
    memory.clear_memory()
    return {"status": "success", "message": f"Memory cleared for agent {agent_id}"}


# Shared Memory Endpoints


@router.post("/shared", response_model=SharedMemoryItem)
def add_shared_memory(
    item: SharedMemoryItem,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> SharedMemoryItem:
    """Add an item to the shared memory."""
    shared_memory = agent_manager.shared_memory_manager
    if not shared_memory:
         raise HTTPException(status_code=503, detail="Shared memory manager not available")
    try:
        added_item = shared_memory.add_item(item)
        return added_item
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add shared memory: {e}")


@router.get("/shared/{item_key}", response_model=SharedMemoryItem)
def get_shared_memory(
    item_key: str,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> SharedMemoryItem:
    """Get an item from shared memory by key."""
    shared_memory = agent_manager.shared_memory_manager
    if not shared_memory:
         raise HTTPException(status_code=503, detail="Shared memory manager not available")
    item = shared_memory.get_item(item_key)
    if not item:
        raise HTTPException(status_code=404, detail=f"Shared memory item '{item_key}' not found")
    return item


@router.put("/shared/{item_key}", response_model=SharedMemoryItem)
def update_shared_memory(
    item_key: str,
    item_update: SharedMemoryItem, # Assume full item replacement for simplicity
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> SharedMemoryItem:
    """Update an item in shared memory."""
    shared_memory = agent_manager.shared_memory_manager
    if not shared_memory:
         raise HTTPException(status_code=503, detail="Shared memory manager not available")
    updated = shared_memory.update_item(item_key, item_update)
    if not updated:
        raise HTTPException(status_code=404, detail=f"Shared memory item '{item_key}' not found for update")
    return updated


@router.delete("/shared/{item_key}", status_code=204)
def delete_shared_memory(
    item_key: str,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> None:
    """Delete an item from shared memory."""
    shared_memory = agent_manager.shared_memory_manager
    if not shared_memory:
         raise HTTPException(status_code=503, detail="Shared memory manager not available")
    if not shared_memory.delete_item(item_key):
        raise HTTPException(status_code=404, detail=f"Shared memory item '{item_key}' not found")
    return None


@router.post("/shared/search", response_model=List[SharedMemoryItem])
def search_shared_memory(
    query: SharedMemoryQuery,
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> List[SharedMemoryItem]:
    """Search the shared memory based on tags or content."""
    shared_memory = agent_manager.shared_memory_manager
    if not shared_memory:
         raise HTTPException(status_code=503, detail="Shared memory manager not available")
    return shared_memory.search_items(query.query, query.tags, query.limit)


@router.get("/shared/keys", response_model=List[str])
def list_shared_memory_keys(
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> List[str]:
    """List all keys currently in shared memory."""
    shared_memory = agent_manager.shared_memory_manager
    if not shared_memory:
         raise HTTPException(status_code=503, detail="Shared memory manager not available")
    return shared_memory.list_keys()


@router.post("/shared/clear")
def clear_shared_memory(
    agent_manager: AgentManager = Depends(get_agent_manager)
) -> Dict[str, str]:
    """Clear all items from shared memory."""
    shared_memory = agent_manager.shared_memory_manager
    if not shared_memory:
         raise HTTPException(status_code=503, detail="Shared memory manager not available")
    shared_memory.clear_all()
    return {"status": "success", "message": "Shared memory cleared"}
