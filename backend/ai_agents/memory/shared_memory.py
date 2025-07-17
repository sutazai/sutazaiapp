"""
Shared Memory Management

This module provides functionality for managing shared memory between agents,
allowing for collaborative information sharing and communication.
"""

import time
import json
import logging
import threading
import uuid
import sys
from enum import Enum
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict


# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class SharedMemoryEntry:
    """
    An entry in shared memory.
    """

    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    creator_id: str = field(default="system")
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    tags: Set[str] = field(default_factory=set)
    source: Optional[str] = None
    importance: float = 0.5  # 0.0 to 1.0
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    version: int = 1
    access_control: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert shared memory entry to dictionary."""
        data = asdict(self)
        data["tags"] = list(self.tags)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SharedMemoryEntry":
        """Create shared memory entry from dictionary."""
        data_copy = data.copy()  # Create a copy to avoid modifying the original

        # Handle tags conversion
        if "tags" in data_copy and isinstance(data_copy["tags"], list):
            data_copy["tags"] = set(data_copy["tags"])

        return cls(**data_copy)

    def update_access_stats(self) -> None:
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1

    def calculate_size(self) -> int:
        """
        Calculate the memory size in bytes.

        Returns:
            int: Size in bytes
        """
        # Convert to JSON to get approximate size
        try:
            return sys.getsizeof(json.dumps(self.to_dict()))
        except Exception as e:
            logger.warning(f"Error calculating memory size: {str(e)}")
            return 0

    def __eq__(self, other) -> bool:
        """Check if two shared memory entries are equal."""
        if not isinstance(other, SharedMemoryEntry):
            return False
        return self.entry_id == other.entry_id


class AccessLevel(Enum):
    """Access level for shared memory."""

    READ = "read"
    WRITE = "write"
    ADMIN = "admin"


class SharedMemory:
    """
    Shared memory system for allowing agents to share information.
    """

    def __init__(self, name: str, description: str = "", max_entries: int = 1000):
        """
        Initialize the shared memory system.

        Args:
            name: Name of the shared memory space
            description: Description of the shared memory space
            max_entries: Maximum number of entries allowed
        """
        self.name = name
        self.description = description
        self.max_entries = max_entries
        self.entries: Dict[str, SharedMemoryEntry] = {}
        self.tags_index: Dict[str, Set[str]] = {}  # Tag -> Set of entry IDs
        self.lock = threading.RLock()
        self.watchers: Dict[str, List[Callable[[str, Dict[str, Any]], None]]] = {}
        self.total_size_bytes = 0
        self.created_at = time.time()

        # Add history tracking per entry
        self._entry_history: Dict[str, List[SharedMemoryEntry]] = defaultdict(list)
        # Index entries created by each agent
        self._agent_entries: Dict[str, Set[str]] = defaultdict(set)

        logger.info(
            f"Initialized shared memory '{name}' with capacity for {max_entries} entries"
        )

    def add_entry(
        self,
        creator_id: str,
        content: Dict[str, Any],
        tags: Optional[Set[str]] = None,
        source: Optional[str] = None,
        importance: float = 0.5,
        access_control: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Add a new entry to shared memory.

        Args:
            creator_id: ID of the agent creating the entry
            content: Entry content
            tags: Tags to associate with the entry
            source: Source of the entry
            importance: Importance of the entry (0.0 to 1.0)
            access_control: Access control settings

        Returns:
            str: Entry ID

        Raises:
            ValueError: If shared memory is full
        """
        with self.lock:
            try:
                # Check if we're at capacity
                if len(self.entries) >= self.max_entries:
                    # Try to make space by removing least important/oldest entries
                    if not self._cleanup_entries():
                        raise ValueError(f"Shared memory '{self.name}' is full and couldn't make space")

                # Create entry
                entry = SharedMemoryEntry(
                    creator_id=creator_id,
                    content=content,
                    tags=tags or set(),
                    source=source,
                    importance=importance,
                    access_control=access_control or {},
                )

                # Store entry
                self.entries[entry.entry_id] = entry

                # Update size tracking
                self.total_size_bytes += entry.calculate_size()

                # Update tags index
                for tag in entry.tags:
                    if tag not in self.tags_index:
                        self.tags_index[tag] = set()
                    self.tags_index[tag].add(entry.entry_id)

                # Add to agent index
                self._agent_entries[creator_id].add(entry.entry_id)

                # Add to history
                self._entry_history[entry.entry_id].append(entry)

                # Notify watchers
                self._notify_watchers("add", entry.entry_id, entry.to_dict())

                logger.debug(
                    f"Added entry {entry.entry_id} to shared memory '{self.name}'"
                )
                return entry.entry_id

            except Exception as e:
                logger.error(
                    f"Error adding entry to shared memory '{self.name}': {str(e)}"
                )
                raise

    def get_entry(
        self, entry_id: str, agent_id: Optional[str] = None, update_stats: bool = True
    ) -> Optional[SharedMemoryEntry]:
        """
        Get an entry from shared memory.

        Args:
            entry_id: ID of the entry to get
            agent_id: ID of the agent requesting the entry (for access control)
            update_stats: Whether to update access statistics

        Returns:
            Optional[SharedMemoryEntry]: The entry, or None if not found or access denied
        """
        with self.lock:
            entry = self.entries.get(entry_id)

            if not entry:
                return None

            # Check access control if agent_id is provided
            if agent_id and not self._check_access(agent_id, entry, AccessLevel.READ):
                logger.warning(
                    f"Access denied for agent {agent_id} to read entry {entry_id}"
                )
                return None

            # Update access stats if requested
            if update_stats:
                entry.update_access_stats()

            return entry

    def update_entry(
        self,
        entry_id: str,
        agent_id: str,
        content: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
        importance: Optional[float] = None,
    ) -> bool:
        """
        Update an entry in shared memory.

        Args:
            entry_id: ID of the entry to update
            agent_id: ID of the agent updating the entry
            content: New content (if provided)
            tags: New tags (if provided)
            importance: New importance (if provided)

        Returns:
            bool: True if updated successfully, False otherwise
        """
        with self.lock:
            try:
                entry = self.entries.get(entry_id)

                if not entry:
                    logger.warning(f"Entry {entry_id} not found for update")
                    return False

                # Check access control
                if not self._check_access(agent_id, entry, AccessLevel.WRITE):
                    logger.warning(
                        f"Access denied for agent {agent_id} to update entry {entry_id}"
                    )
                    return False

                # Track original size for updating total
                original_size = entry.calculate_size()

                # Update content if provided
                if content is not None:
                    entry.content = content
                    entry.version += 1

                # Update tags if provided
                if tags is not None:
                    # Remove old tag references
                    for tag in entry.tags:
                        if tag in self.tags_index and entry_id in self.tags_index[tag]:
                            self.tags_index[tag].remove(entry_id)
                            # Clean up empty tag sets
                            if not self.tags_index[tag]:
                                del self.tags_index[tag]

                    # Set new tags
                    entry.tags = tags

                    # Update tags index
                    for tag in entry.tags:
                        if tag not in self.tags_index:
                            self.tags_index[tag] = set()
                        self.tags_index[tag].add(entry_id)

                # Update importance if provided
                if importance is not None:
                    if 0.0 <= importance <= 1.0:
                        entry.importance = importance
                    else:
                        logger.warning(f"Invalid importance value: {importance}")

                # Update timestamp and calculate new size
                entry.timestamp = time.time()
                new_size = entry.calculate_size()

                # Update total size
                self.total_size_bytes = self.total_size_bytes - original_size + new_size

                # Store previous version in history
                self._entry_history[entry_id].append(entry)

                # Notify watchers
                self._notify_watchers("update", entry_id, entry.to_dict())

                logger.debug(f"Updated entry {entry_id} in shared memory '{self.name}'")
                return True

            except Exception as e:
                logger.error(f"Error updating entry {entry_id}: {str(e)}")
                return False

    def delete_entry(self, entry_id: str, agent_id: str) -> bool:
        """
        Delete an entry from shared memory.

        Args:
            entry_id: ID of the entry to delete
            agent_id: ID of the agent deleting the entry

        Returns:
            bool: True if entry was deleted, False otherwise
        """
        with self.lock:
            entry = self.entries.get(entry_id)
            if not entry:
                return False

            # Check access control
            if not self._check_access(agent_id, entry, AccessLevel.WRITE):
                logger.warning(
                    f"Access denied for agent {agent_id} to delete entry {entry_id}"
                )
                return False

            # Store final state in history before deleting
            self._entry_history[entry_id].append(entry)

            # Remove entry from tag indexes
            for tag in entry.tags:
                if tag in self.tags_index and entry_id in self.tags_index[tag]:
                    self.tags_index[tag].remove(entry_id)
                    if not self.tags_index[tag]:
                        del self.tags_index[tag]

            # Remove entry from agent index
            if entry.creator_id in self._agent_entries and entry_id in self._agent_entries[entry.creator_id]:
                self._agent_entries[entry.creator_id].remove(entry_id)
                if not self._agent_entries[entry.creator_id]:
                    del self._agent_entries[entry.creator_id]

            # Remove entry
            del self.entries[entry_id]

            return True

    def search_by_tags(
        self, tags: Set[str], agent_id: Optional[str] = None, require_all: bool = False
    ) -> List[SharedMemoryEntry]:
        """
        Search for entries by tags.

        Args:
            tags: Tags to search for
            agent_id: ID of the agent searching (for access control)
            require_all: Whether all tags must match (True) or any tag (False)

        Returns:
            List[SharedMemoryEntry]: List of matching entries
        """
        matching_entries = []
        with self.lock:
            # Find entries with all required tags or any tag
            candidate_ids = set()
            if require_all:
                # Start with the set from the first tag, then intersect
                if tags:
                    first_tag = next(iter(tags))
                    candidate_ids = self.tags_index.get(first_tag, set()).copy()
                    for tag in tags:
                        candidate_ids.intersection_update(self.tags_index.get(tag, set()))
            else:
                # Union of sets for each tag
                for tag in tags:
                    candidate_ids.update(self.tags_index.get(tag, set()))

            # Retrieve and filter by access control
            for entry_id in candidate_ids:
                entry = self.get_entry(entry_id, agent_id=agent_id, update_stats=False)
                if entry:
                    matching_entries.append(entry)
        return matching_entries

    def search_by_content(
        self, query: str, agent_id: Optional[str] = None
    ) -> List[SharedMemoryEntry]:
        """
        Search for entries by content.

        Args:
            query: Query string to search for
            agent_id: ID of the agent searching (for access control)

        Returns:
            List[SharedMemoryEntry]: Matching entries
        """
        with self.lock:
            results = []

            for entry in self.entries.values():
                # Check access control if agent_id is provided
                if agent_id and entry.access_control:
                    if (
                        agent_id not in entry.access_control
                        and "*" not in entry.access_control
                    ):
                        continue

                # Check content
                content_str = json.dumps(entry.content).lower()
                if query.lower() in content_str:
                    # Update access stats
                    entry.update_access_stats()
                    results.append(entry)

            return results

    def get_agent_entries(self, agent_id: str) -> List[SharedMemoryEntry]:
        """
        Get entries created by an agent.

        Args:
            agent_id: ID of the agent

        Returns:
            List[SharedMemoryEntry]: List of entries created by the agent
        """
        agent_created_entries = []
        with self.lock:
            entry_ids = self._agent_entries.get(agent_id, set())
            for entry_id in entry_ids:
                # Check access before adding
                accessible_entry = self.get_entry(entry_id, agent_id=agent_id, update_stats=False)
                if accessible_entry:
                     agent_created_entries.append(accessible_entry)
        return agent_created_entries

    def update_access_control(
        self, entry_id: str, agent_id: str, access_control: Dict[str, str]
    ) -> bool:
        """
        Update the access control of an entry.

        Args:
            entry_id: ID of the entry to update
            agent_id: ID of the agent updating the entry
            access_control: New access control map (agent_id -> permission)

        Returns:
            bool: True if access control was updated, False otherwise
        """
        with self.lock:
            entry = self.entries.get(entry_id)
            if not entry:
                return False

            # Check owner access
            if agent_id != entry.creator_id:
                if entry.access_control.get(agent_id) != "owner":
                    logger.warning(
                        f"Agent {agent_id} attempted to update access control for entry {entry_id}"
                    )
                    return False

            # Update access control
            entry.access_control = access_control
            return True

    def get_all_entries(
        self, agent_id: Optional[str] = None
    ) -> List[SharedMemoryEntry]:
        """
        Get all entries in shared memory.

        Args:
            agent_id: ID of the agent requesting entries (for access control)

        Returns:
            List[SharedMemoryEntry]: All accessible entries
        """
        with self.lock:
            results = []

            for entry in self.entries.values():
                # Check access control if agent_id is provided
                if agent_id and entry.access_control:
                    if (
                        agent_id not in entry.access_control
                        and "*" not in entry.access_control
                    ):
                        continue

                # Update access stats
                entry.update_access_stats()
                results.append(entry)

            return results

    def get_entry_history(
        self, entry_id: str, agent_id: Optional[str] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get the version history of an entry.

        Args:
            entry_id: ID of the entry
            agent_id: ID of the agent requesting history (for access control)

        Returns:
            Optional[List[Dict[str, Any]]]: History of the entry, or None if not found/denied
        """
        with self.lock:
            # First, check if the requester can even read the *current* entry
            current_entry = self.get_entry(entry_id, agent_id=agent_id, update_stats=False)
            if not current_entry:
                 logger.warning(f"Denied access or entry {entry_id} not found for history request by {agent_id}")
                 return None

            # Return history if available
            history_list = self._entry_history.get(entry_id, [])
            return [entry.to_dict() for entry in history_list]

    def clear(self) -> int:
        """
        Clear all entries in shared memory.

        Returns:
            int: Number of entries cleared
        """
        with self.lock:
            count = len(self.entries)
            self.entries.clear()
            self.tags_index.clear()
            self._agent_entries.clear()
            self._entry_history.clear()
            self.total_size_bytes = 0
            return count

    # --- Watcher Methods ---

    def add_watcher(
        self, event_type: str, callback: Callable[[str, Dict[str, Any]], None]
    ):
        """Add a watcher for a specific event type (e.g., 'add', 'update', 'delete')."""
        with self.lock:
            if event_type not in self.watchers:
                self.watchers[event_type] = []
            self.watchers[event_type].append(callback)

    def remove_watcher(
        self, event_type: str, callback: Callable[[str, Dict[str, Any]], None]
    ):
        """Remove a watcher."""
        with self.lock:
            if event_type in self.watchers and callback in self.watchers[event_type]:
                self.watchers[event_type].remove(callback)
                if not self.watchers[event_type]:
                    del self.watchers[event_type]

    def _notify_watchers(
        self, event_type: str, entry_id: str, entry_data: Dict[str, Any]
    ):
        """Notify registered watchers about an event."""
        with self.lock:
            if event_type in self.watchers:
                for callback in self.watchers[event_type]:
                    try:
                        callback(entry_id, entry_data)
                    except Exception as e:
                        logger.error(
                            f"Error in watcher callback for event {event_type}: {e}"
                        )

    # --- Internal Helper Methods ---

    def _cleanup_entries(self, num_to_remove: int = 10) -> bool:
        """
        Remove oldest or least important entries to make space.

        Args:
            num_to_remove: Number of entries to attempt removing.

        Returns:
            bool: True if space was successfully made, False otherwise.
        """
        if len(self.entries) < self.max_entries:
            return True  # No need to clean up

        # Sort entries by importance (lowest first) and then timestamp (oldest first)
        sorted_entries = sorted(
            self.entries.values(), key=lambda e: (e.importance, e.timestamp)
        )

        removed_count = 0
        for entry in sorted_entries[:num_to_remove]:
            # Attempt to delete the entry (requires ADMIN or owner rights generally)
            # For cleanup, assume system has rights - or implement specific cleanup logic
            if self.delete_entry(entry.entry_id, agent_id="system_cleanup"): # Use a system ID
                removed_count += 1

        logger.info(f"Cleaned up {removed_count} entries from shared memory '{self.name}'")
        return removed_count > 0

    def _check_access(self, agent_id: str, entry: SharedMemoryEntry, required_level: AccessLevel) -> bool:
        """
        Check if an agent has the required access level for an entry.

        Args:
            agent_id: ID of the agent requesting access
            entry: The shared memory entry
            required_level: The required access level (READ, WRITE, ADMIN)

        Returns:
            bool: True if access is granted, False otherwise
        """
        if not entry.access_control:  # No specific controls = public access
            return True

        agent_level_str = entry.access_control.get(agent_id)
        wildcard_level_str = entry.access_control.get("*") # Check for wildcard

        # Determine the highest level granted (agent specific or wildcard)
        effective_level_str = agent_level_str or wildcard_level_str

        if not effective_level_str:
            return False # No access defined for this agent or wildcard

        try:
            effective_level = AccessLevel(effective_level_str)
        except ValueError:
            logger.warning(f"Invalid access level string '{effective_level_str}' in entry {entry.entry_id}")
            return False

        # Check if the effective level meets the required level
        if required_level == AccessLevel.READ:
            return effective_level in [AccessLevel.READ, AccessLevel.WRITE, AccessLevel.ADMIN]
        elif required_level == AccessLevel.WRITE:
            return effective_level in [AccessLevel.WRITE, AccessLevel.ADMIN]
        elif required_level == AccessLevel.ADMIN:
            return effective_level == AccessLevel.ADMIN

        return False # Should not be reached


class SharedMemoryManager:
    """
    Manager for shared memory spaces.

    This class provides functionality for managing multiple shared memory spaces,
    enabling different collaboration contexts between agents.
    """

    def __init__(self):
        """Initialize the shared memory manager."""
        self.memories: Dict[str, SharedMemory] = {}
        self.lock = threading.RLock()

    def create_memory(
        self, name: str, description: str = "", max_entries: int = 1000
    ) -> SharedMemory:
        """
        Create a new shared memory space.

        Args:
            name: Name of the shared memory space
            description: Description of the shared memory space
            max_entries: Maximum number of entries

        Returns:
            SharedMemory: Created shared memory space

        Raises:
            ValueError: If a shared memory space with the given name already exists
        """
        with self.lock:
            if name in self.memories:
                raise ValueError(f"Shared memory space '{name}' already exists")

            memory = SharedMemory(
                name=name, description=description, max_entries=max_entries
            )

            self.memories[name] = memory
            return memory

    def get_memory(self, name: str) -> Optional[SharedMemory]:
        """
        Get a shared memory space by name.

        Args:
            name: Name of the shared memory space

        Returns:
            Optional[SharedMemory]: Shared memory space or None if not found
        """
        with self.lock:
            return self.memories.get(name)

    def delete_memory(self, name: str) -> bool:
        """
        Delete a shared memory space.

        Args:
            name: Name of the shared memory space

        Returns:
            bool: True if shared memory space was deleted, False otherwise
        """
        with self.lock:
            if name not in self.memories:
                return False

            del self.memories[name]
            return True

    def get_memory_names(self) -> List[str]:
        """
        Get the names of all shared memory spaces.

        Returns:
            List[str]: List of memory names
        """
        with self.lock:
            return list(self.memories.keys())

    def get_all_memories(self) -> Dict[str, SharedMemory]:
        """
        Get all shared memory spaces.

        Returns:
            Dict[str, SharedMemory]: Dictionary of memory names to memory spaces
        """
        with self.lock:
            return self.memories.copy()
