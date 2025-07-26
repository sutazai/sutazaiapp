"""
Shared Memory Management
Handles shared memory spaces for inter-agent communication
"""

import logging
import uuid
from typing import Dict, Any, List, Optional, Set
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class SharedMemoryEntry:
    """Entry in shared memory space"""
    entry_id: str
    content: Dict[str, Any]
    creator_id: str
    tags: Set[str]
    importance: float
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    access_control: Optional[Dict[str, str]] = None  # agent_id -> permission

class SharedMemorySpace:
    """A shared memory space accessible by multiple agents"""
    
    def __init__(self, name: str, description: str = "", max_entries: int = 500):
        self.name = name
        self.description = description
        self.max_entries = max_entries
        self.entries: Dict[str, SharedMemoryEntry] = {}
        self.tag_indices: Dict[str, Set[str]] = {}
        self.creator_indices: Dict[str, Set[str]] = {}
        self.created_at = datetime.now()
        
    def add_entry(self, content: Dict[str, Any], creator_id: str,
                  tags: Optional[Set[str]] = None, importance: float = 0.5,
                  access_control: Optional[Dict[str, str]] = None) -> str:
        """Add entry to shared memory"""
        entry_id = str(uuid.uuid4())
        tags = tags or set()
        
        entry = SharedMemoryEntry(
            entry_id=entry_id,
            content=content,
            creator_id=creator_id,
            tags=tags,
            importance=importance,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_control=access_control
        )
        
        self.entries[entry_id] = entry
        
        # Update indices
        for tag in tags:
            if tag not in self.tag_indices:
                self.tag_indices[tag] = set()
            self.tag_indices[tag].add(entry_id)
            
        if creator_id not in self.creator_indices:
            self.creator_indices[creator_id] = set()
        self.creator_indices[creator_id].add(entry_id)
        
        # Cleanup if needed
        self._cleanup_if_needed()
        
        logger.debug(f"Added entry to shared memory space '{self.name}' by {creator_id}")
        return entry_id
        
    def get_entry(self, entry_id: str, accessor_id: str) -> Optional[SharedMemoryEntry]:
        """Get entry if accessible by the requesting agent"""
        if entry_id not in self.entries:
            return None
            
        entry = self.entries[entry_id]
        
        # Check access control
        if not self._can_access(entry, accessor_id, "read"):
            return None
            
        entry.last_accessed = datetime.now()
        entry.access_count += 1
        return entry
        
    def search_entries(self, accessor_id: str, tags: Optional[Set[str]] = None,
                      creator_id: Optional[str] = None, min_importance: Optional[float] = None,
                      limit: int = 50) -> List[SharedMemoryEntry]:
        """Search entries accessible by the requesting agent"""
        candidate_ids = set(self.entries.keys())
        
        # Filter by tags
        if tags:
            for tag in tags:
                if tag in self.tag_indices:
                    candidate_ids &= self.tag_indices[tag]
                else:
                    candidate_ids = set()
                    break
                    
        # Filter by creator
        if creator_id and creator_id in self.creator_indices:
            candidate_ids &= self.creator_indices[creator_id]
            
        # Filter by access control and importance
        results = []
        for entry_id in candidate_ids:
            entry = self.entries[entry_id]
            
            if not self._can_access(entry, accessor_id, "read"):
                continue
                
            if min_importance is None or entry.importance >= min_importance:
                results.append(entry)
                
        # Sort by importance and recency
        results.sort(key=lambda e: (e.importance, e.last_accessed), reverse=True)
        
        # Update access info
        for entry in results[:limit]:
            entry.last_accessed = datetime.now()
            entry.access_count += 1
            
        return results[:limit]
        
    def update_entry(self, entry_id: str, accessor_id: str, content: Optional[Dict[str, Any]] = None,
                    tags: Optional[Set[str]] = None, importance: Optional[float] = None) -> bool:
        """Update entry if accessible"""
        if entry_id not in self.entries:
            return False
            
        entry = self.entries[entry_id]
        
        if not self._can_access(entry, accessor_id, "write"):
            return False
            
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
        
    def delete_entry(self, entry_id: str, accessor_id: str) -> bool:
        """Delete entry if accessible"""
        if entry_id not in self.entries:
            return False
            
        entry = self.entries[entry_id]
        
        if not self._can_access(entry, accessor_id, "delete"):
            return False
            
        # Remove from indices
        for tag in entry.tags:
            if tag in self.tag_indices:
                self.tag_indices[tag].discard(entry_id)
                
        if entry.creator_id in self.creator_indices:
            self.creator_indices[entry.creator_id].discard(entry_id)
            
        del self.entries[entry_id]
        
        logger.debug(f"Deleted entry {entry_id} from shared memory space '{self.name}'")
        return True
        
    def get_stats(self) -> Dict[str, Any]:
        """Get memory space statistics"""
        return {
            "name": self.name,
            "description": self.description,
            "total_entries": len(self.entries),
            "unique_tags": len(self.tag_indices),
            "unique_creators": len(self.creator_indices),
            "max_entries": self.max_entries,
            "created_at": self.created_at.isoformat(),
            "avg_importance": sum(e.importance for e in self.entries.values()) / len(self.entries) if self.entries else 0
        }
        
    def _can_access(self, entry: SharedMemoryEntry, accessor_id: str, operation: str) -> bool:
        """Check if agent can perform operation on entry"""
        # Creator always has full access
        if entry.creator_id == accessor_id:
            return True
            
        # No access control means public access
        if not entry.access_control:
            return True
            
        # Check specific permissions
        permission = entry.access_control.get(accessor_id, "none")
        
        if operation == "read":
            return permission in ["read", "write", "admin"]
        elif operation == "write":
            return permission in ["write", "admin"]
        elif operation == "delete":
            return permission == "admin"
            
        return False
        
    def _cleanup_if_needed(self):
        """Clean up old entries if limit exceeded"""
        if len(self.entries) <= self.max_entries:
            return
            
        # Remove oldest, least important entries
        entries_list = list(self.entries.values())
        entries_list.sort(key=lambda e: (e.importance, e.access_count, e.last_accessed))
        
        to_remove = len(entries_list) - self.max_entries
        for entry in entries_list[:to_remove]:
            # Remove from indices
            for tag in entry.tags:
                if tag in self.tag_indices:
                    self.tag_indices[tag].discard(entry.entry_id)
                    
            if entry.creator_id in self.creator_indices:
                self.creator_indices[entry.creator_id].discard(entry.entry_id)
                
            del self.entries[entry.entry_id]

class SharedMemoryManager:
    """Manager for all shared memory spaces"""
    
    def __init__(self):
        self.memory_spaces: Dict[str, SharedMemorySpace] = {}
        
    def create_memory(self, name: str, description: str = "", max_entries: int = 500) -> SharedMemorySpace:
        """Create a new shared memory space"""
        if name in self.memory_spaces:
            return self.memory_spaces[name]
            
        space = SharedMemorySpace(name, description, max_entries)
        self.memory_spaces[name] = space
        
        logger.info(f"Created shared memory space '{name}'")
        return space
        
    def get_memory(self, name: str) -> Optional[SharedMemorySpace]:
        """Get shared memory space by name"""
        return self.memory_spaces.get(name)
        
    def delete_memory(self, name: str) -> bool:
        """Delete shared memory space"""
        if name in self.memory_spaces:
            del self.memory_spaces[name]
            logger.info(f"Deleted shared memory space '{name}'")
            return True
        return False
        
    def list_memory_spaces(self) -> List[str]:
        """List all memory space names"""
        return list(self.memory_spaces.keys())
        
    def memory_exists(self, name: str) -> bool:
        """Check if memory space exists"""
        return name in self.memory_spaces
        
    def get_global_stats(self) -> Dict[str, Any]:
        """Get statistics for all memory spaces"""
        total_entries = sum(len(space.entries) for space in self.memory_spaces.values())
        
        return {
            "total_spaces": len(self.memory_spaces),
            "total_entries": total_entries,
            "space_stats": {name: space.get_stats() for name, space in self.memory_spaces.items()}
        }