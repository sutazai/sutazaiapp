"""
Agent Memory Management

This module provides functionality for managing agent memory, including
short-term and long-term memory, as well as memory search and retrieval.
"""

import time
import json
import logging
import threading
import uuid
import sys
from enum import Enum
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from collections import deque

# Configure logging
logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Type of memory."""

    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    WORKING = "working"
    EPISODIC = "episodic"


@dataclass
class MemoryEntry:
    """
    An entry in agent memory.
    """

    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    memory_type: MemoryType = MemoryType.SHORT_TERM
    content: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    tags: Set[str] = field(default_factory=set)
    source: Optional[str] = None
    importance: float = 0.5  # 0.0 to 1.0
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert memory entry to dictionary."""
        # Use asdict but handle the Set conversion for tags
        data = asdict(self)
        data["memory_type"] = self.memory_type.value
        data["tags"] = list(self.tags)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryEntry":
        """Create memory entry from dictionary."""
        data_copy = data.copy()  # Create a copy to avoid modifying the original

        # Handle memory_type conversion
        if isinstance(data_copy.get("memory_type"), str):
            try:
                data_copy["memory_type"] = MemoryType(data_copy["memory_type"])
            except ValueError:
                logger.error(f"Invalid memory type: {data_copy.get('memory_type')}")
                # Default to SHORT_TERM if invalid
                data_copy["memory_type"] = MemoryType.SHORT_TERM

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
        """Check if two memory entries are equal."""
        if not isinstance(other, MemoryEntry):
            return False
        return self.entry_id == other.entry_id


class Memory:
    """
    Agent memory system for storing and retrieving information.

    This class provides functionality for managing different types of memory,
    including short-term and long-term memory, as well as memory search and retrieval.
    """

    def __init__(
        self,
        agent_id: str,
        max_short_term_entries: int = 100,
        max_long_term_entries: int = 1000,
    ):
        """
        Initialize the memory system.

        Args:
            agent_id: ID of the agent this memory belongs to
            max_short_term_entries: Maximum number of short-term memory entries
            max_long_term_entries: Maximum number of long-term memory entries
        """
        self.agent_id = agent_id
        self.max_short_term_entries = max_short_term_entries
        self.max_long_term_entries = max_long_term_entries
        self.entries: Dict[str, MemoryEntry] = {}
        self.short_term_queue = deque(maxlen=max_short_term_entries)
        self.long_term_entries: Dict[str, MemoryEntry] = {}
        self.working_memory: Dict[str, MemoryEntry] = {}
        self.episodic_memory: Dict[str, List[MemoryEntry]] = {}
        self.lock = threading.RLock()
        self.total_size_bytes = 0

        logger.info(
            f"Initialized memory for agent {agent_id} with {max_short_term_entries} "
            f"short-term and {max_long_term_entries} long-term entry capacity"
        )

    def add_memory(
        self,
        content: Dict[str, Any],
        memory_type: MemoryType = MemoryType.SHORT_TERM,
        tags: Optional[Set[str]] = None,
        source: Optional[str] = None,
        importance: float = 0.5,
    ) -> str:
        """
        Add a new memory entry.

        Args:
            content: Memory content
            memory_type: Type of memory
            tags: Tags to associate with the memory
            source: Source of the memory
            importance: Importance of the memory (0.0 to 1.0)

        Returns:
            str: Entry ID
        """
        with self.lock:
            try:
                # Create entry
                entry = MemoryEntry(
                    memory_type=memory_type,
                    content=content,
                    tags=tags or set(),
                    source=source,
                    importance=importance,
                )

                # Store entry
                self.entries[entry.entry_id] = entry

                # Update total size
                self.total_size_bytes += entry.calculate_size()

                # Add to appropriate data structure based on memory type
                if memory_type == MemoryType.SHORT_TERM:
                    self._handle_short_term_memory(entry)
                elif memory_type == MemoryType.LONG_TERM:
                    self._handle_long_term_memory(entry)
                elif memory_type == MemoryType.WORKING:
                    self.working_memory[entry.entry_id] = entry
                elif memory_type == MemoryType.EPISODIC:
                    self._handle_episodic_memory(entry)

                logger.debug(
                    f"Added memory entry {entry.entry_id} of type {memory_type.value}"
                )
                return entry.entry_id
            except Exception as e:
                logger.error(f"Error adding memory: {str(e)}")
                raise

    def _handle_short_term_memory(self, entry: MemoryEntry) -> None:
        """
        Handle short-term memory management.

        Args:
            entry: Memory entry to add to short-term memory
        """
        self.short_term_queue.append(entry.entry_id)

        # If we're over capacity, remove oldest entries from short-term
        while len(self.short_term_queue) > self.max_short_term_entries:
            oldest_id = self.short_term_queue.popleft()

            # Consider moving to long-term based on importance
            oldest_entry = self.entries.get(oldest_id)
            if oldest_entry and oldest_entry.importance > 0.7:
                self._move_to_long_term(oldest_id)
            elif oldest_id in self.entries:
                # Remove entry and update total size
                if oldest_id in self.entries:
                    self.total_size_bytes -= self.entries[oldest_id].calculate_size()
                    del self.entries[oldest_id]

    def _handle_long_term_memory(self, entry: MemoryEntry) -> None:
        """
        Handle long-term memory management.

        Args:
            entry: Memory entry to add to long-term memory
        """
        self.long_term_entries[entry.entry_id] = entry

        # If we're over capacity, remove least important entries from long-term
        if len(self.long_term_entries) > self.max_long_term_entries:
            # Sort by importance (lowest first) and remove least important
            least_important = sorted(
                self.long_term_entries.items(),
                key=lambda x: (
                    x[1].importance,
                    -x[1].timestamp,
                ),  # Sort by importance, then oldest
            )

            to_remove = least_important[0][0]  # Get ID of least important entry

            # Remove from long-term and entries
            del self.long_term_entries[to_remove]

            # Update total size and remove from main entries dict
            if to_remove in self.entries:
                self.total_size_bytes -= self.entries[to_remove].calculate_size()
                del self.entries[to_remove]

    def _handle_episodic_memory(self, entry: MemoryEntry) -> None:
        """
        Handle episodic memory management.

        Args:
            entry: Memory entry to add to episodic memory
        """
        # Use the source as the episode ID
        episode_id = entry.source or "default"

        # Initialize episode if it doesn't exist
        if episode_id not in self.episodic_memory:
            self.episodic_memory[episode_id] = []

        # Add entry to episode
        self.episodic_memory[episode_id].append(entry)

    def get_memory(self, entry_id: str) -> Optional[MemoryEntry]:
        """
        Get a memory entry by ID.

        Args:
            entry_id: Entry ID to retrieve

        Returns:
            Optional[MemoryEntry]: Memory entry or None if not found
        """
        with self.lock:
            entry = self.entries.get(entry_id)
            if entry:
                entry.update_access_stats()
            return entry

    def update_memory(
        self,
        entry_id: str,
        content: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
        importance: Optional[float] = None,
    ) -> bool:
        """
        Update a memory entry.

        Args:
            entry_id: Entry ID to update
            content: New content (if provided)
            tags: New tags (if provided)
            importance: New importance (if provided)

        Returns:
            bool: True if entry was updated, False otherwise
        """
        with self.lock:
            entry = self.entries.get(entry_id)
            if not entry:
                return False

            # Update fields if provided
            if content is not None:
                entry.content = content

            if tags is not None:
                entry.tags = tags

            if importance is not None:
                entry.importance = importance

            # Update timestamp
            entry.timestamp = time.time()
            entry.update_access_stats()

            return True

    def delete_memory(self, entry_id: str) -> bool:
        """
        Delete a memory entry.

        Args:
            entry_id: Entry ID to delete

        Returns:
            bool: True if entry was deleted, False otherwise
        """
        with self.lock:
            if entry_id not in self.entries:
                return False

            entry = self.entries[entry_id]

            # Remove from appropriate data structure based on memory type
            if entry.memory_type == MemoryType.SHORT_TERM:
                try:
                    self.short_term_queue.remove(entry_id)
                except ValueError:
                    pass

            elif entry.memory_type == MemoryType.LONG_TERM:
                if entry_id in self.long_term_entries:
                    del self.long_term_entries[entry_id]

            elif entry.memory_type == MemoryType.WORKING:
                if entry_id in self.working_memory:
                    del self.working_memory[entry_id]

            elif entry.memory_type == MemoryType.EPISODIC:
                # Remove from episodic memory
                for source_key, entries in self.episodic_memory.items():
                    for i, e in enumerate(entries):
                        if e.entry_id == entry_id:
                            entries.pop(i)
                            break

            # Remove from entries
            del self.entries[entry_id]

            # Update total size
            self.total_size_bytes -= entry.calculate_size()

            return True

    def search_by_tags(
        self,
        tags: Set[str],
        memory_type: Optional[MemoryType] = None,
        require_all: bool = False,
    ) -> List[MemoryEntry]:
        """
        Search for memory entries by tags.

        Args:
            tags: Tags to search for
            memory_type: Memory type to filter by (optional)
            require_all: Whether all tags must match (True) or any tag (False)

        Returns:
            List[MemoryEntry]: Matching memory entries
        """
        with self.lock:
            results = []

            for entry in self.entries.values():
                # Filter by memory type if provided
                if memory_type and entry.memory_type != memory_type:
                    continue

                # Check tags
                if require_all and not tags.issubset(entry.tags):
                    continue
                elif not require_all and not tags.intersection(entry.tags):
                    continue

                # Update access stats
                entry.update_access_stats()

                results.append(entry)

            return results

    def search_by_content(
        self, query: str, memory_type: Optional[MemoryType] = None
    ) -> List[MemoryEntry]:
        """
        Search for memory entries by content.

        Args:
            query: Query string to search for
            memory_type: Memory type to filter by (optional)

        Returns:
            List[MemoryEntry]: Matching memory entries
        """
        with self.lock:
            results = []

            # TODO: Implement more sophisticated content search
            # This is a simple implementation that checks if the query is in the string
            # representation of the content

            for entry in self.entries.values():
                # Filter by memory type if provided
                if memory_type and entry.memory_type != memory_type:
                    continue

                # Check content
                content_str = json.dumps(entry.content).lower()
                if query.lower() in content_str:
                    # Update access stats
                    entry.update_access_stats()

                    results.append(entry)

            return results

    def get_recent_memories(
        self, limit: int = 10, memory_type: Optional[MemoryType] = None
    ) -> List[MemoryEntry]:
        """
        Get the most recent memory entries.

        Args:
            limit: Maximum number of entries to return
            memory_type: Memory type to filter by (optional)

        Returns:
            List[MemoryEntry]: Recent memory entries
        """
        with self.lock:
            # Filter by memory type if provided
            entries = list(self.entries.values())
            if memory_type:
                entries = [e for e in entries if e.memory_type == memory_type]

            # Sort by timestamp (most recent first)
            entries.sort(key=lambda e: e.timestamp, reverse=True)

            # Update access stats
            for entry in entries[:limit]:
                entry.update_access_stats()

            return entries[:limit]

    def get_important_memories(
        self, limit: int = 10, memory_type: Optional[MemoryType] = None
    ) -> List[MemoryEntry]:
        """
        Get the most important memory entries.

        Args:
            limit: Maximum number of entries to return
            memory_type: Memory type to filter by (optional)

        Returns:
            List[MemoryEntry]: Important memory entries
        """
        with self.lock:
            # Filter by memory type if provided
            entries = list(self.entries.values())
            if memory_type:
                entries = [e for e in entries if e.memory_type == memory_type]

            # Sort by importance (most important first)
            entries.sort(key=lambda e: e.importance, reverse=True)

            # Update access stats
            for entry in entries[:limit]:
                entry.update_access_stats()

            return entries[:limit]

    def clear_memory(self, memory_type: Optional[MemoryType] = None) -> int:
        """
        Clear memory entries.

        Args:
            memory_type: Memory type to clear (None for all)

        Returns:
            int: Number of entries cleared
        """
        with self.lock:
            if memory_type is None:
                # Clear all memory
                count = len(self.entries)
                self.entries.clear()
                self.short_term_queue.clear()
                self.long_term_entries.clear()
                self.working_memory.clear()
                self.episodic_memory.clear()
                self.total_size_bytes = 0
                return count

            # Clear specific memory type
            to_delete = [
                entry_id
                for entry_id, entry in self.entries.items()
                if entry.memory_type == memory_type
            ]

            # Remove entries
            for entry_id in to_delete:
                self.delete_memory(entry_id)

            return len(to_delete)

    def _move_to_long_term(self, entry_id: str) -> bool:
        """
        Move a memory entry from short-term to long-term memory.

        Args:
            entry_id: Entry ID to move

        Returns:
            bool: True if entry was moved, False otherwise
        """
        with self.lock:
            if entry_id not in self.entries:
                return False

            entry = self.entries[entry_id]

            # Check if entry is in short-term memory
            if entry.memory_type != MemoryType.SHORT_TERM:
                return False

            # Update memory type
            entry.memory_type = MemoryType.LONG_TERM

            # Add to long-term memory
            self.long_term_entries[entry_id] = entry

            return True

    def summarize_memory(
        self, memory_type: Optional[MemoryType] = None, max_entries: int = 20
    ) -> Dict[str, Any]:
        """
        Generate a summary of memory contents.

        Args:
            memory_type: Memory type to summarize (None for all)
            max_entries: Maximum number of entries to include

        Returns:
            Dict[str, Any]: Memory summary
        """
        with self.lock:
            # Filter by memory type if provided
            if memory_type:
                entries = [
                    e for e in self.entries.values() if e.memory_type == memory_type
                ]
            else:
                entries = list(self.entries.values())

            # Sort by importance and recency
            entries.sort(key=lambda e: (e.importance, e.timestamp), reverse=True)

            # Prepare summary
            summary = {
                "total_entries": len(entries),
                "memory_types": {
                    memory_type.value: len(
                        [e for e in entries if e.memory_type == memory_type]
                    )
                    for memory_type in MemoryType
                },
                "recent_entries": [e.to_dict() for e in entries[:max_entries]],
                "memory_size": sum(len(json.dumps(e.content)) for e in entries),
            }

            return summary


class MemoryManager:
    """
    Manager for agent memories.

    This class provides functionality for managing memory systems for
    multiple agents, including creation, access, and maintenance.
    """

    def __init__(self):
        """Initialize the memory manager."""
        self.memories: Dict[str, Memory] = {}
        self.lock = threading.RLock()

    def create_memory(
        self,
        agent_id: str,
        max_short_term_entries: int = 100,
        max_long_term_entries: int = 1000,
    ) -> Memory:
        """
        Create a new memory system for an agent.

        Args:
            agent_id: ID of the agent
            max_short_term_entries: Maximum number of short-term memory entries
            max_long_term_entries: Maximum number of long-term memory entries

        Returns:
            Memory: Created memory system

        Raises:
            ValueError: If agent already has a memory system
        """
        with self.lock:
            if agent_id in self.memories:
                raise ValueError(f"Agent {agent_id} already has a memory system")

            memory = Memory(
                agent_id=agent_id,
                max_short_term_entries=max_short_term_entries,
                max_long_term_entries=max_long_term_entries,
            )

            self.memories[agent_id] = memory
            return memory

    def get_memory(self, agent_id: str) -> Optional[Memory]:
        """
        Get the memory system for an agent.

        Args:
            agent_id: ID of the agent

        Returns:
            Optional[Memory]: Memory system or None if not found
        """
        with self.lock:
            return self.memories.get(agent_id)

    def delete_memory(self, agent_id: str) -> bool:
        """
        Delete the memory system for an agent.

        Args:
            agent_id: ID of the agent

        Returns:
            bool: True if memory system was deleted, False otherwise
        """
        with self.lock:
            if agent_id not in self.memories:
                return False

            del self.memories[agent_id]
            return True

    def get_agent_ids(self) -> List[str]:
        """
        Get the IDs of agents with memory systems.

        Returns:
            List[str]: List of agent IDs
        """
        with self.lock:
            return list(self.memories.keys())
