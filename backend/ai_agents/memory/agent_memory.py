"""
Agent Memory Management
Handles individual agent memory storage and retrieval
"""

import logging
import uuid
from enum import Enum
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Types of memory storage"""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"

@dataclass
class MemoryEntry:
    """Individual memory entry"""
    entry_id: str
    content: Dict[str, Any]
    memory_type: MemoryType
    tags: Set[str]
    importance: float
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    
    def __post_init__(self):
        if isinstance(self.memory_type, str):
            self.memory_type = MemoryType(self.memory_type)

class AgentMemory:
    """Memory management for a single agent"""
    
    def __init__(self, agent_id: str, max_short_term: int = 100, max_long_term: int = 1000):
        self.agent_id = agent_id
        self.max_short_term = max_short_term
        self.max_long_term = max_long_term
        self.memories: Dict[str, MemoryEntry] = {}
        self.type_indices: Dict[MemoryType, Set[str]] = {mt: set() for mt in MemoryType}
        self.tag_indices: Dict[str, Set[str]] = {}
        
    def add_memory(self, content: Dict[str, Any], memory_type: MemoryType, 
                   tags: Optional[Set[str]] = None, importance: float = 0.5) -> str:
        """Add a new memory entry"""
        entry_id = str(uuid.uuid4())
        tags = tags or set()
        
        entry = MemoryEntry(
            entry_id=entry_id,
            content=content,
            memory_type=memory_type,
            tags=tags,
            importance=importance,
            created_at=datetime.now(),
            last_accessed=datetime.now()
        )
        
        self.memories[entry_id] = entry
        self.type_indices[memory_type].add(entry_id)
        
        # Update tag indices
        for tag in tags:
            if tag not in self.tag_indices:
                self.tag_indices[tag] = set()
            self.tag_indices[tag].add(entry_id)
            
        # Cleanup if needed
        self._cleanup_memory_if_needed()
        
        logger.debug(f"Added {memory_type.value} memory for agent {self.agent_id}")
        return entry_id
        
    def get_memory(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory entry"""
        if entry_id in self.memories:
            entry = self.memories[entry_id]
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            return entry
        return None
        
    def search_memories(self, memory_type: Optional[MemoryType] = None, 
                       tags: Optional[Set[str]] = None, 
                       min_importance: Optional[float] = None,
                       limit: int = 50) -> List[MemoryEntry]:
        """Search memories by criteria"""
        candidate_ids = set(self.memories.keys())
        
        # Filter by memory type
        if memory_type:
            candidate_ids &= self.type_indices[memory_type]
            
        # Filter by tags
        if tags:
            for tag in tags:
                if tag in self.tag_indices:
                    candidate_ids &= self.tag_indices[tag]
                else:
                    candidate_ids = set()  # Tag not found
                    break
                    
        # Filter by importance and convert to entries
        results = []
        for entry_id in candidate_ids:
            entry = self.memories[entry_id]
            if min_importance is None or entry.importance >= min_importance:
                results.append(entry)
                
        # Sort by importance and recency
        results.sort(key=lambda e: (e.importance, e.last_accessed), reverse=True)
        
        # Update access info for returned results
        for entry in results[:limit]:
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            
        return results[:limit]
        
    def update_memory(self, entry_id: str, content: Optional[Dict[str, Any]] = None,
                     tags: Optional[Set[str]] = None, importance: Optional[float] = None) -> bool:
        """Update an existing memory entry"""
        if entry_id not in self.memories:
            return False
            
        entry = self.memories[entry_id]
        
        # Update content
        if content is not None:
            entry.content.update(content)
            
        # Update tags
        if tags is not None:
            # Remove old tag references
            for old_tag in entry.tags:
                if old_tag in self.tag_indices:
                    self.tag_indices[old_tag].discard(entry_id)
                    
            # Add new tag references
            entry.tags = tags
            for tag in tags:
                if tag not in self.tag_indices:
                    self.tag_indices[tag] = set()
                self.tag_indices[tag].add(entry_id)
                
        # Update importance
        if importance is not None:
            entry.importance = importance
            
        entry.last_accessed = datetime.now()
        return True
        
    def delete_memory(self, entry_id: str) -> bool:
        """Delete a memory entry"""
        if entry_id not in self.memories:
            return False
            
        entry = self.memories[entry_id]
        
        # Remove from indices
        self.type_indices[entry.memory_type].discard(entry_id)
        for tag in entry.tags:
            if tag in self.tag_indices:
                self.tag_indices[tag].discard(entry_id)
                
        # Remove entry
        del self.memories[entry_id]
        
        logger.debug(f"Deleted memory {entry_id} for agent {self.agent_id}")
        return True
        
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        type_counts = {mt.value: len(ids) for mt, ids in self.type_indices.items()}
        
        return {
            "agent_id": self.agent_id,
            "total_memories": len(self.memories),
            "memory_types": type_counts,
            "unique_tags": len(self.tag_indices),
            "max_short_term": self.max_short_term,
            "max_long_term": self.max_long_term,
            "avg_importance": sum(e.importance for e in self.memories.values()) / len(self.memories) if self.memories else 0
        }
        
    def _cleanup_memory_if_needed(self):
        """Clean up old memories if limits are exceeded"""
        # Clean up short-term memory
        short_term_ids = self.type_indices[MemoryType.SHORT_TERM]
        if len(short_term_ids) > self.max_short_term:
            # Remove oldest, least important short-term memories
            short_term_entries = [self.memories[eid] for eid in short_term_ids]
            short_term_entries.sort(key=lambda e: (e.importance, e.last_accessed))
            
            to_remove = len(short_term_entries) - self.max_short_term
            for entry in short_term_entries[:to_remove]:
                self.delete_memory(entry.entry_id)
                
        # Clean up long-term memory
        long_term_ids = self.type_indices[MemoryType.LONG_TERM]
        if len(long_term_ids) > self.max_long_term:
            long_term_entries = [self.memories[eid] for eid in long_term_ids]
            long_term_entries.sort(key=lambda e: (e.importance, e.access_count, e.last_accessed))
            
            to_remove = len(long_term_entries) - self.max_long_term
            for entry in long_term_entries[:to_remove]:
                self.delete_memory(entry.entry_id)

class MemoryManager:
    """Manager for all agent memories"""
    
    def __init__(self):
        self.agent_memories: Dict[str, AgentMemory] = {}
        
    def create_memory(self, agent_id: str, max_short_term: int = 100, max_long_term: int = 1000) -> AgentMemory:
        """Create memory storage for an agent"""
        if agent_id in self.agent_memories:
            return self.agent_memories[agent_id]
            
        memory = AgentMemory(agent_id, max_short_term, max_long_term)
        self.agent_memories[agent_id] = memory
        
        logger.info(f"Created memory storage for agent {agent_id}")
        return memory
        
    def get_memory(self, agent_id: str) -> Optional[AgentMemory]:
        """Get memory storage for an agent"""
        return self.agent_memories.get(agent_id)
        
    def delete_memory(self, agent_id: str) -> bool:
        """Delete all memory for an agent"""
        if agent_id in self.agent_memories:
            del self.agent_memories[agent_id]
            logger.info(f"Deleted memory storage for agent {agent_id}")
            return True
        return False
        
    def get_global_stats(self) -> Dict[str, Any]:
        """Get statistics for all agent memories"""
        total_memories = sum(len(mem.memories) for mem in self.agent_memories.values())
        
        return {
            "total_agents": len(self.agent_memories),
            "total_memories": total_memories,
            "agent_stats": {aid: mem.get_memory_stats() for aid, mem in self.agent_memories.items()}
        }