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

from ..agent_manager import AgentManager
from ..dependencies import get_agent_manager
from ..memory.agent_memory import MemoryType


# Configure logging
logger = logging.getLogger(__name__)

router = APIRouter()


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


@router.post(
    "/agents/{agent_id}",
    response_model=AgentMemoryResponse,
    status_code=status.HTTP_201_CREATED,
)
async def add_agent_memory(
    agent_id: str = Path(..., description="Agent ID"),
    memory: AgentMemoryCreate = Body(..., description="Memory to add"),
    agent_manager: AgentManager = Depends(get_agent_manager),
):
    """
    Add a new memory entry for an agent.
    """
    try:
        # Convert enum to string for memory_type
        memory_type_str = memory.memory_type.value

        # Convert tags list to set if provided

        logger.info(f"Adding memory for agent {agent_id}, type: {memory_type_str}")

        entry_id = agent_manager.add_agent_memory(
            agent_id=agent_id,
            content=memory.content,
            memory_type=memory_type_str,
            tags=memory.tags,
            importance=memory.importance,
        )

        # Get the memory agent
        memory_agent = agent_manager.get_memory(agent_id)
        if not memory_agent:
            logger.error(f"Memory not found for agent: {agent_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory not found for agent: {agent_id}",
            )

        # Get the entry
        entry = memory_agent.get_memory(entry_id)
        if not entry:
            logger.error(f"Memory entry not found: {entry_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Memory entry not found: {entry_id}",
            )

        # Convert to response model
        response = {
            "entry_id": entry.entry_id,
            "content": entry.content,
            "tags": list(entry.tags),
            "memory_type": entry.memory_type.value,
            "importance": entry.importance,
            "source": entry.source,
            "timestamp": entry.timestamp,
            "last_accessed": entry.last_accessed,
            "access_count": entry.access_count,
        }

        logger.info(f"Successfully added memory entry {entry_id} for agent {agent_id}")
        return response
    except ValueError as ve:
        # Handle validation errors
        logger.error(f"Validation error adding memory for agent {agent_id}: {str(ve)}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(ve)
        )
    except Exception as e:
        # Handle other errors
        logger.error(f"Error adding memory for agent {agent_id}: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/agents/{agent_id}", response_model=List[AgentMemoryResponse])
async def get_agent_memories(
    agent_id: str = Path(..., description="Agent ID"),
    memory_type: Optional[MemoryTypeEnum] = Query(
        None, description="Filter by memory type"
    ),
    limit: int = Query(10, description="Maximum number of entries to return"),
    agent_manager: AgentManager = Depends(get_agent_manager),
):
    """
    Get memory entries for an agent.
    """
    try:
        # Get the memory agent
        memory_agent = agent_manager.get_memory(agent_id)
        if not memory_agent:
            raise HTTPException(
                status_code=404, detail=f"Memory not found for agent: {agent_id}"
            )

        # Convert enum to MemoryType if provided
        memory_type_enum = None
        if memory_type:
            try:
                memory_type_enum = MemoryType(memory_type.value)
            except ValueError:
                raise HTTPException(
                    status_code=400, detail=f"Invalid memory type: {memory_type}"
                )

        # Get recent memories
        entries = memory_agent.get_recent_memories(
            limit=limit, memory_type=memory_type_enum
        )

        # Convert to response models
        return [
            {
                "entry_id": entry.entry_id,
                "content": entry.content,
                "tags": list(entry.tags),
                "memory_type": entry.memory_type.value,
                "importance": entry.importance,
                "source": entry.source,
                "timestamp": entry.timestamp,
                "last_accessed": entry.last_accessed,
                "access_count": entry.access_count,
            }
            for entry in entries
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/agents/{agent_id}/entries/{entry_id}", response_model=AgentMemoryResponse)
async def get_agent_memory_entry(
    agent_id: str = Path(..., description="Agent ID"),
    entry_id: str = Path(..., description="Memory entry ID"),
    agent_manager: AgentManager = Depends(get_agent_manager),
):
    """
    Get a specific memory entry for an agent.
    """
    try:
        # Get the memory agent
        memory_agent = agent_manager.get_memory(agent_id)
        if not memory_agent:
            raise HTTPException(
                status_code=404, detail=f"Memory not found for agent: {agent_id}"
            )

        # Get the entry
        entry = memory_agent.get_memory(entry_id)
        if not entry:
            raise HTTPException(
                status_code=404, detail=f"Memory entry not found: {entry_id}"
            )

        # Convert to response model
        return {
            "entry_id": entry.entry_id,
            "content": entry.content,
            "tags": list(entry.tags),
            "memory_type": entry.memory_type.value,
            "importance": entry.importance,
            "source": entry.source,
            "timestamp": entry.timestamp,
            "last_accessed": entry.last_accessed,
            "access_count": entry.access_count,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/agents/{agent_id}/entries/{entry_id}", response_model=Dict[str, Any])
async def delete_agent_memory_entry(
    agent_id: str = Path(..., description="Agent ID"),
    entry_id: str = Path(..., description="Memory entry ID"),
    agent_manager: AgentManager = Depends(get_agent_manager),
):
    """
    Delete a specific memory entry for an agent.
    """
    try:
        # Get the memory agent
        memory_agent = agent_manager.get_memory(agent_id)
        if not memory_agent:
            raise HTTPException(
                status_code=404, detail=f"Memory not found for agent: {agent_id}"
            )

        # Delete the entry
        success = memory_agent.delete_memory(entry_id)
        if not success:
            raise HTTPException(
                status_code=404, detail=f"Memory entry not found: {entry_id}"
            )

        return {"success": True, "message": f"Memory entry {entry_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/agents/{agent_id}/search", response_model=List[AgentMemoryResponse])
async def search_agent_memory(
    agent_id: str = Path(..., description="Agent ID"),
    query: str = Query(..., description="Search query"),
    memory_type: Optional[MemoryTypeEnum] = Query(
        None, description="Filter by memory type"
    ),
    agent_manager: AgentManager = Depends(get_agent_manager),
):
    """
    Search memory entries for an agent.
    """
    try:
        # Get the memory agent
        memory_agent = agent_manager.get_memory(agent_id)
        if not memory_agent:
            raise HTTPException(
                status_code=404, detail=f"Memory not found for agent: {agent_id}"
            )

        # Convert enum to MemoryType if provided
        memory_type_enum = None
        if memory_type:
            try:
                memory_type_enum = MemoryType(memory_type.value)
            except ValueError:
                raise HTTPException(
                    status_code=400, detail=f"Invalid memory type: {memory_type}"
                )

        # Search by content
        entries = memory_agent.search_by_content(
            query=query, memory_type=memory_type_enum
        )

        # Convert to response models
        return [
            {
                "entry_id": entry.entry_id,
                "content": entry.content,
                "tags": list(entry.tags),
                "memory_type": entry.memory_type.value,
                "importance": entry.importance,
                "source": entry.source,
                "timestamp": entry.timestamp,
                "last_accessed": entry.last_accessed,
                "access_count": entry.access_count,
            }
            for entry in entries
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/agents/{agent_id}/tags", response_model=List[AgentMemoryResponse])
async def search_agent_memory_by_tags(
    agent_id: str = Path(..., description="Agent ID"),
    tags: List[str] = Query(..., description="Tags to search for"),
    require_all: bool = Query(False, description="Whether all tags must match"),
    memory_type: Optional[MemoryTypeEnum] = Query(
        None, description="Filter by memory type"
    ),
    agent_manager: AgentManager = Depends(get_agent_manager),
):
    """
    Search memory entries for an agent by tags.
    """
    try:
        # Get the memory agent
        memory_agent = agent_manager.get_memory(agent_id)
        if not memory_agent:
            raise HTTPException(
                status_code=404, detail=f"Memory not found for agent: {agent_id}"
            )

        # Convert enum to MemoryType if provided
        memory_type_enum = None
        if memory_type:
            try:
                memory_type_enum = MemoryType(memory_type.value)
            except ValueError:
                raise HTTPException(
                    status_code=400, detail=f"Invalid memory type: {memory_type}"
                )

        # Search by tags
        entries = memory_agent.search_by_tags(
            tags=set(tags), memory_type=memory_type_enum, require_all=require_all
        )

        # Convert to response models
        return [
            {
                "entry_id": entry.entry_id,
                "content": entry.content,
                "tags": list(entry.tags),
                "memory_type": entry.memory_type.value,
                "importance": entry.importance,
                "source": entry.source,
                "timestamp": entry.timestamp,
                "last_accessed": entry.last_accessed,
                "access_count": entry.access_count,
            }
            for entry in entries
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/agents/{agent_id}/summary", response_model=MemorySummary)
async def get_agent_memory_summary(
    agent_id: str = Path(..., description="Agent ID"),
    memory_type: Optional[MemoryTypeEnum] = Query(
        None, description="Filter by memory type"
    ),
    max_entries: int = Query(
        20, description="Maximum number of entries to include in summary"
    ),
    agent_manager: AgentManager = Depends(get_agent_manager),
):
    """
    Get a summary of memory entries for an agent.
    """
    try:
        # Get the memory agent
        memory_agent = agent_manager.get_memory(agent_id)
        if not memory_agent:
            raise HTTPException(
                status_code=404, detail=f"Memory not found for agent: {agent_id}"
            )

        # Convert enum to MemoryType if provided
        memory_type_enum = None
        if memory_type:
            try:
                memory_type_enum = MemoryType(memory_type.value)
            except ValueError:
                raise HTTPException(
                    status_code=400, detail=f"Invalid memory type: {memory_type}"
                )

        # Get summary
        summary = memory_agent.summarize_memory(
            memory_type=memory_type_enum, max_entries=max_entries
        )

        return summary
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Shared Memory Endpoints


@router.post("/shared", response_model=Dict[str, Any])
async def create_shared_memory_space(
    memory_space: SharedMemorySpace = Body(
        ..., description="Shared memory space to create"
    ),
    agent_manager: AgentManager = Depends(get_agent_manager),
):
    """
    Create a new shared memory space.
    """
    try:
        # Get the shared memory manager
        shared_manager = agent_manager.get_shared_memory_manager()

        # Create shared memory space
        shared_memory = shared_manager.create_memory(
            name=memory_space.name,
            description=memory_space.description,
            max_entries=memory_space.max_entries,
        )

        return {
            "name": shared_memory.name,
            "description": shared_memory.description,
            "max_entries": shared_memory.max_entries,
            "created_at": shared_memory.created_at,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/shared", response_model=List[Dict[str, Any]])
async def list_shared_memory_spaces(
    agent_manager: AgentManager = Depends(get_agent_manager),
):
    """
    List all shared memory spaces.
    """
    try:
        # Get the shared memory manager
        shared_manager = agent_manager.get_shared_memory_manager()

        # Get all memory spaces
        memory_names = shared_manager.get_memory_names()
        memories = []

        for name in memory_names:
            memory = shared_manager.get_memory(name)
            if memory:
                memories.append(
                    {
                        "name": memory.name,
                        "description": memory.description,
                        "max_entries": memory.max_entries,
                        "created_at": memory.created_at,
                        "entry_count": len(memory.entries),
                    }
                )

        return memories
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/shared/{memory_name}", response_model=SharedMemoryResponse)
async def add_shared_memory_entry(
    memory_name: str = Path(..., description="Shared memory space name"),
    memory: SharedMemoryCreate = Body(..., description="Memory to add"),
    agent_manager: AgentManager = Depends(get_agent_manager),
):
    """
    Add a new entry to a shared memory space.
    """
    try:
        # Convert tags list to set if provided

        entry_id = agent_manager.add_shared_memory(
            memory_space=memory_name,
            content=memory.content,
            creator_id=memory.creator_id,
            tags=memory.tags,
            importance=memory.importance,
            access_control=memory.access_control,
        )

        # Get the shared memory manager
        shared_manager = agent_manager.get_shared_memory_manager()

        # Get the shared memory space
        shared_memory = shared_manager.get_memory(memory_name)
        if not shared_memory:
            raise HTTPException(
                status_code=404, detail=f"Shared memory space not found: {memory_name}"
            )

        # Get the entry
        entry = shared_memory.get_entry(entry_id, agent_id=memory.creator_id)
        if not entry:
            raise HTTPException(
                status_code=404, detail=f"Shared memory entry not found: {entry_id}"
            )

        # Convert to response model
        return {
            "entry_id": entry.entry_id,
            "content": entry.content,
            "tags": list(entry.tags),
            "importance": entry.importance,
            "source": entry.source,
            "creator_id": entry.creator_id,
            "timestamp": entry.timestamp,
            "last_accessed": entry.last_accessed,
            "access_count": entry.access_count,
            "version": entry.version,
            "access_control": entry.access_control,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/shared/{memory_name}", response_model=List[SharedMemoryResponse])
async def get_shared_memory_entries(
    memory_name: str = Path(..., description="Shared memory space name"),
    agent_id: str = Query(..., description="ID of the agent requesting the entries"),
    agent_manager: AgentManager = Depends(get_agent_manager),
):
    """
    Get entries from a shared memory space.
    """
    try:
        # Get the shared memory manager
        shared_manager = agent_manager.get_shared_memory_manager()

        # Get the shared memory space
        shared_memory = shared_manager.get_memory(memory_name)
        if not shared_memory:
            raise HTTPException(
                status_code=404, detail=f"Shared memory space not found: {memory_name}"
            )

        # Get all entries
        entries = shared_memory.get_all_entries(agent_id=agent_id)

        # Convert to response models
        return [
            {
                "entry_id": entry.entry_id,
                "content": entry.content,
                "tags": list(entry.tags),
                "importance": entry.importance,
                "source": entry.source,
                "creator_id": entry.creator_id,
                "timestamp": entry.timestamp,
                "last_accessed": entry.last_accessed,
                "access_count": entry.access_count,
                "version": entry.version,
                "access_control": entry.access_control,
            }
            for entry in entries
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get(
    "/shared/{memory_name}/entries/{entry_id}", response_model=SharedMemoryResponse
)
async def get_shared_memory_entry(
    memory_name: str = Path(..., description="Shared memory space name"),
    entry_id: str = Path(..., description="Memory entry ID"),
    agent_id: str = Query(..., description="ID of the agent requesting the entry"),
    agent_manager: AgentManager = Depends(get_agent_manager),
):
    """
    Get a specific entry from a shared memory space.
    """
    try:
        # Get the shared memory manager
        shared_manager = agent_manager.get_shared_memory_manager()

        # Get the shared memory space
        shared_memory = shared_manager.get_memory(memory_name)
        if not shared_memory:
            raise HTTPException(
                status_code=404, detail=f"Shared memory space not found: {memory_name}"
            )

        # Get the entry
        entry = shared_memory.get_entry(entry_id, agent_id=agent_id)
        if not entry:
            raise HTTPException(
                status_code=404, detail=f"Shared memory entry not found: {entry_id}"
            )

        # Convert to response model
        return {
            "entry_id": entry.entry_id,
            "content": entry.content,
            "tags": list(entry.tags),
            "importance": entry.importance,
            "source": entry.source,
            "creator_id": entry.creator_id,
            "timestamp": entry.timestamp,
            "last_accessed": entry.last_accessed,
            "access_count": entry.access_count,
            "version": entry.version,
            "access_control": entry.access_control,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put(
    "/shared/{memory_name}/entries/{entry_id}", response_model=SharedMemoryResponse
)
async def update_shared_memory_entry(
    memory_name: str = Path(..., description="Shared memory space name"),
    entry_id: str = Path(..., description="Memory entry ID"),
    memory: MemoryEntryBase = Body(..., description="Updated memory"),
    agent_id: str = Query(..., description="ID of the agent updating the entry"),
    agent_manager: AgentManager = Depends(get_agent_manager),
):
    """
    Update a specific entry in a shared memory space.
    """
    try:
        # Get the shared memory manager
        shared_manager = agent_manager.get_shared_memory_manager()

        # Get the shared memory space
        shared_memory = shared_manager.get_memory(memory_name)
        if not shared_memory:
            raise HTTPException(
                status_code=404, detail=f"Shared memory space not found: {memory_name}"
            )

        # Convert tags list to set if provided

        # Update the entry
        success = shared_memory.update_entry(
            entry_id=entry_id,
            content=memory.content,
            agent_id=agent_id,
            tags=memory.tags,
            importance=memory.importance,
        )

        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Failed to update shared memory entry: {entry_id}",
            )

        # Get the updated entry
        entry = shared_memory.get_entry(entry_id, agent_id=agent_id)
        if not entry:
            raise HTTPException(
                status_code=404, detail=f"Shared memory entry not found: {entry_id}"
            )

        # Convert to response model
        return {
            "entry_id": entry.entry_id,
            "content": entry.content,
            "tags": list(entry.tags),
            "importance": entry.importance,
            "source": entry.source,
            "creator_id": entry.creator_id,
            "timestamp": entry.timestamp,
            "last_accessed": entry.last_accessed,
            "access_count": entry.access_count,
            "version": entry.version,
            "access_control": entry.access_control,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete(
    "/shared/{memory_name}/entries/{entry_id}", response_model=Dict[str, Any]
)
async def delete_shared_memory_entry(
    memory_name: str = Path(..., description="Shared memory space name"),
    entry_id: str = Path(..., description="Memory entry ID"),
    agent_id: str = Query(..., description="ID of the agent deleting the entry"),
    agent_manager: AgentManager = Depends(get_agent_manager),
):
    """
    Delete a specific entry from a shared memory space.
    """
    try:
        # Get the shared memory manager
        shared_manager = agent_manager.get_shared_memory_manager()

        # Get the shared memory space
        shared_memory = shared_manager.get_memory(memory_name)
        if not shared_memory:
            raise HTTPException(
                status_code=404, detail=f"Shared memory space not found: {memory_name}"
            )

        # Delete the entry
        success = shared_memory.delete_entry(entry_id, agent_id=agent_id)
        if not success:
            raise HTTPException(
                status_code=404,
                detail=f"Failed to delete shared memory entry: {entry_id}",
            )

        return {"success": True, "message": f"Shared memory entry {entry_id} deleted"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/shared/{memory_name}/search", response_model=List[SharedMemoryResponse])
async def search_shared_memory(
    memory_name: str = Path(..., description="Shared memory space name"),
    query: str = Query(..., description="Search query"),
    agent_id: str = Query(..., description="ID of the agent searching"),
    agent_manager: AgentManager = Depends(get_agent_manager),
):
    """
    Search entries in a shared memory space.
    """
    try:
        # Get the shared memory manager
        shared_manager = agent_manager.get_shared_memory_manager()

        # Get the shared memory space
        shared_memory = shared_manager.get_memory(memory_name)
        if not shared_memory:
            raise HTTPException(
                status_code=404, detail=f"Shared memory space not found: {memory_name}"
            )

        # Search by content
        entries = shared_memory.search_by_content(query=query, agent_id=agent_id)

        # Convert to response models
        return [
            {
                "entry_id": entry.entry_id,
                "content": entry.content,
                "tags": list(entry.tags),
                "importance": entry.importance,
                "source": entry.source,
                "creator_id": entry.creator_id,
                "timestamp": entry.timestamp,
                "last_accessed": entry.last_accessed,
                "access_count": entry.access_count,
                "version": entry.version,
                "access_control": entry.access_control,
            }
            for entry in entries
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/shared/{memory_name}/tags", response_model=List[SharedMemoryResponse])
async def search_shared_memory_by_tags(
    memory_name: str = Path(..., description="Shared memory space name"),
    tags: List[str] = Query(..., description="Tags to search for"),
    require_all: bool = Query(False, description="Whether all tags must match"),
    agent_id: str = Query(..., description="ID of the agent searching"),
    agent_manager: AgentManager = Depends(get_agent_manager),
):
    """
    Search entries in a shared memory space by tags.
    """
    try:
        # Get the shared memory manager
        shared_manager = agent_manager.get_shared_memory_manager()

        # Get the shared memory space
        shared_memory = shared_manager.get_memory(memory_name)
        if not shared_memory:
            raise HTTPException(
                status_code=404, detail=f"Shared memory space not found: {memory_name}"
            )

        # Search by tags
        entries = shared_memory.search_by_tags(
            tags=set(tags), agent_id=agent_id, require_all=require_all
        )

        # Convert to response models
        return [
            {
                "entry_id": entry.entry_id,
                "content": entry.content,
                "tags": list(entry.tags),
                "importance": entry.importance,
                "source": entry.source,
                "creator_id": entry.creator_id,
                "timestamp": entry.timestamp,
                "last_accessed": entry.last_accessed,
                "access_count": entry.access_count,
                "version": entry.version,
                "access_control": entry.access_control,
            }
            for entry in entries
        ]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/shared/{memory_name}", response_model=Dict[str, Any])
async def delete_shared_memory_space(
    memory_name: str = Path(..., description="Shared memory space name"),
    agent_manager: AgentManager = Depends(get_agent_manager),
):
    """
    Delete a shared memory space.
    """
    try:
        # Get the shared memory manager
        shared_manager = agent_manager.get_shared_memory_manager()

        # Delete the memory space
        success = shared_manager.delete_memory(memory_name)
        if not success:
            raise HTTPException(
                status_code=404, detail=f"Shared memory space not found: {memory_name}"
            )

        return {
            "success": True,
            "message": f"Shared memory space {memory_name} deleted",
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
