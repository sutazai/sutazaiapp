#!/usr/bin/env python3
"""
MCP State Management System

Persistent state tracking and recovery system for MCP orchestration. Provides
atomic state operations, distributed state synchronization, and crash recovery
with Redis-backed storage and comprehensive state history tracking.

Author: Claude AI Assistant (ai-agent-orchestrator)
Created: 2025-08-15 12:02:00 UTC
Version: 1.0.0
"""

import json
import pickle
import asyncio
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import hashlib

try:
    import redis.asyncio as redis
except ImportError:
    redis = None  # Handle gracefully if Redis not available

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import MCPAutomationConfig


class StateType(Enum):
    """State data types."""
    SYSTEM = "system"              # System-level state
    WORKFLOW = "workflow"          # Workflow execution state
    SERVICE = "service"            # Service state
    CONFIGURATION = "configuration"  # Configuration state
    CACHE = "cache"                # Cached data
    SESSION = "session"            # Session state
    CHECKPOINT = "checkpoint"      # Recovery checkpoints
    METRIC = "metric"              # Metrics and counters


class StateStatus(Enum):
    """State lifecycle status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    ARCHIVED = "archived"
    CORRUPTED = "corrupted"


@dataclass
class StateEntry:
    """Individual state entry."""
    key: str
    value: Any
    type: StateType
    status: StateStatus = StateStatus.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    version: int = 1
    checksum: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        """Check if state has expired."""
        if self.expires_at:
            return datetime.now(timezone.utc) > self.expires_at
        return False
        
    def calculate_checksum(self) -> str:
        """Calculate checksum for state value."""
        if isinstance(self.value, (dict, list)):
            data = json.dumps(self.value, sort_keys=True)
        else:
            data = str(self.value)
        return hashlib.sha256(data.encode()).hexdigest()
        
    def verify_integrity(self) -> bool:
        """Verify state integrity using checksum."""
        if not self.checksum:
            return True
        return self.calculate_checksum() == self.checksum


@dataclass
class StateSnapshot:
    """Point-in-time state snapshot."""
    id: str
    timestamp: datetime
    states: Dict[str, StateEntry]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "states": {k: asdict(v) for k, v in self.states.items()},
            "metadata": self.metadata
        }


@dataclass
class SystemState:
    """Complete system state."""
    orchestrator: Dict[str, Any] = field(default_factory=dict)
    workflows: Dict[str, Any] = field(default_factory=dict)
    services: Dict[str, Any] = field(default_factory=dict)
    events: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class StateManager:
    """
    Centralized state management for MCP orchestration.
    
    Provides persistent state storage, atomic operations, distributed
    synchronization, and crash recovery capabilities.
    """
    
    def __init__(self, config: Optional[MCPAutomationConfig] = None):
        """Initialize state manager."""
        self.config = config or MCPAutomationConfig()
        self.logger = self._setup_logging()
        
        # State storage
        self._states: Dict[str, StateEntry] = {}
        self._state_history: Dict[str, List[StateEntry]] = {}
        self._snapshots: List[StateSnapshot] = []
        
        # Redis connection
        self._redis: Optional[redis.Redis] = None
        self._redis_prefix = "mcp:orchestrator:"
        
        # State synchronization
        self._sync_interval = 30  # seconds
        self._sync_task: Optional[asyncio.Task] = None
        
        # Persistence
        self._persist_path = Path("/opt/sutazaiapp/data/orchestrator/state")
        self._persist_interval = 300  # 5 minutes
        self._persist_task: Optional[asyncio.Task] = None
        
        # Locks for atomic operations
        self._locks: Dict[str, asyncio.Lock] = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Configure logging."""
        logger = logging.getLogger("mcp.state_manager")
        logger.setLevel(self.config.log_level.value.upper())
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    async def initialize(self) -> None:
        """Initialize state manager."""
        self.logger.info("Initializing state manager...")
        
        # Create persistence directory
        self._persist_path.mkdir(parents=True, exist_ok=True)
        
        # Connect to Redis if available
        await self._connect_redis()
        
        # Load persisted state
        await self._load_persisted_state()
        
        # Start background tasks
        self._sync_task = asyncio.create_task(self._sync_loop())
        self._persist_task = asyncio.create_task(self._persist_loop())
        
        self.logger.info("State manager initialized")
        
    async def _connect_redis(self) -> None:
        """Connect to Redis if available."""
        if redis:
            try:
                self._redis = redis.Redis(
                    host="localhost",
                    port=10001,
                    decode_responses=False,  # We'll handle encoding
                    socket_connect_timeout=5
                )
                
                # Test connection
                await self._redis.ping()
                self.logger.info("Connected to Redis for distributed state")
                
            except Exception as e:
                self.logger.warning(f"Redis not available, using local state only: {e}")
                self._redis = None
        else:
            self.logger.warning("Redis module not available, using local state only")
            
    async def _load_persisted_state(self) -> None:
        """Load state from persistent storage."""
        try:
            state_file = self._persist_path / "state.json"
            if state_file.exists():
                with open(state_file) as f:
                    data = json.load(f)
                    
                # Restore states
                for key, entry_data in data.get("states", {}).items():
                    entry = StateEntry(
                        key=entry_data["key"],
                        value=entry_data["value"],
                        type=StateType[entry_data["type"]],
                        status=StateStatus[entry_data["status"]],
                        created_at=datetime.fromisoformat(entry_data["created_at"]),
                        updated_at=datetime.fromisoformat(entry_data["updated_at"]),
                        expires_at=datetime.fromisoformat(entry_data["expires_at"]) if entry_data.get("expires_at") else None,
                        version=entry_data.get("version", 1),
                        checksum=entry_data.get("checksum"),
                        metadata=entry_data.get("metadata", {})
                    )
                    
                    if not entry.is_expired():
                        self._states[key] = entry
                        
                self.logger.info(f"Loaded {len(self._states)} states from persistence")
                
        except Exception as e:
            self.logger.error(f"Failed to load persisted state: {e}")
            
    async def set_state(
        self,
        key: str,
        value: Any,
        type: StateType = StateType.SYSTEM,
        ttl: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Set a state value."""
        try:
            # Get or create lock
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
                
            async with self._locks[key]:
                # Create or update entry
                if key in self._states:
                    entry = self._states[key]
                    entry.value = value
                    entry.updated_at = datetime.now(timezone.utc)
                    entry.version += 1
                else:
                    entry = StateEntry(
                        key=key,
                        value=value,
                        type=type,
                        metadata=metadata or {}
                    )
                    
                # Set expiration
                if ttl:
                    entry.expires_at = datetime.now(timezone.utc) + timedelta(seconds=ttl)
                    
                # Calculate checksum
                entry.checksum = entry.calculate_checksum()
                
                # Store locally
                self._states[key] = entry
                
                # Track history
                if key not in self._state_history:
                    self._state_history[key] = []
                self._state_history[key].append(entry)
                
                # Store in Redis if available
                if self._redis:
                    try:
                        redis_key = f"{self._redis_prefix}{key}"
                        redis_value = pickle.dumps(entry)
                        
                        if ttl:
                            await self._redis.setex(redis_key, ttl, redis_value)
                        else:
                            await self._redis.set(redis_key, redis_value)
                            
                    except Exception as e:
                        self.logger.error(f"Failed to store state in Redis: {e}")
                        
                self.logger.debug(f"Set state: {key} (v{entry.version})")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to set state {key}: {e}")
            return False
            
    async def get_state(
        self,
        key: str,
        default: Any = None
    ) -> Any:
        """Get a state value."""
        try:
            # Check local cache first
            if key in self._states:
                entry = self._states[key]
                if not entry.is_expired() and entry.verify_integrity():
                    return entry.value
                else:
                    # Remove expired/corrupted entry
                    del self._states[key]
                    
            # Try Redis if available
            if self._redis:
                try:
                    redis_key = f"{self._redis_prefix}{key}"
                    redis_value = await self._redis.get(redis_key)
                    
                    if redis_value:
                        entry = pickle.loads(redis_value)
                        if not entry.is_expired() and entry.verify_integrity():
                            # Cache locally
                            self._states[key] = entry
                            return entry.value
                            
                except Exception as e:
                    self.logger.error(f"Failed to get state from Redis: {e}")
                    
            return default
            
        except Exception as e:
            self.logger.error(f"Failed to get state {key}: {e}")
            return default
            
    async def delete_state(self, key: str) -> bool:
        """Delete a state value."""
        try:
            # Remove from local storage
            if key in self._states:
                del self._states[key]
                
            # Remove from Redis
            if self._redis:
                try:
                    redis_key = f"{self._redis_prefix}{key}"
                    await self._redis.delete(redis_key)
                except Exception as e:
                    self.logger.error(f"Failed to delete state from Redis: {e}")
                    
            self.logger.debug(f"Deleted state: {key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete state {key}: {e}")
            return False
            
    async def exists(self, key: str) -> bool:
        """Check if state exists."""
        if key in self._states:
            entry = self._states[key]
            return not entry.is_expired()
            
        if self._redis:
            try:
                redis_key = f"{self._redis_prefix}{key}"
                return await self._redis.exists(redis_key) > 0
            except:
                pass
                
        return False
        
    async def get_keys(
        self,
        pattern: Optional[str] = None,
        type: Optional[StateType] = None
    ) -> List[str]:
        """Get all state keys matching pattern."""
        keys = []
        
        # Filter local keys
        for key, entry in self._states.items():
            if entry.is_expired():
                continue
            if type and entry.type != type:
                continue
            if pattern and not self._matches_pattern(key, pattern):
                continue
            keys.append(key)
            
        # Get Redis keys if available
        if self._redis and pattern:
            try:
                redis_pattern = f"{self._redis_prefix}{pattern}"
                redis_keys = await self._redis.keys(redis_pattern)
                for redis_key in redis_keys:
                    key = redis_key.decode().replace(self._redis_prefix, "")
                    if key not in keys:
                        keys.append(key)
            except Exception as e:
                self.logger.error(f"Failed to get keys from Redis: {e}")
                
        return keys
        
    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches pattern (simple wildcard support)."""
        import fnmatch
        return fnmatch.fnmatch(key, pattern)
        
    async def create_snapshot(
        self,
        id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> StateSnapshot:
        """Create a state snapshot."""
        try:
            snapshot = StateSnapshot(
                id=id or str(datetime.now(timezone.utc).timestamp()),
                timestamp=datetime.now(timezone.utc),
                states=dict(self._states),  # Shallow copy
                metadata=metadata or {}
            )
            
            self._snapshots.append(snapshot)
            
            # Keep only last 10 snapshots
            if len(self._snapshots) > 10:
                self._snapshots = self._snapshots[-10:]
                
            self.logger.info(f"Created snapshot: {snapshot.id}")
            return snapshot
            
        except Exception as e:
            self.logger.error(f"Failed to create snapshot: {e}")
            raise
            
    async def restore_snapshot(self, snapshot_id: str) -> bool:
        """Restore from a snapshot."""
        try:
            snapshot = next((s for s in self._snapshots if s.id == snapshot_id), None)
            if not snapshot:
                self.logger.error(f"Snapshot not found: {snapshot_id}")
                return False
                
            # Clear current state
            self._states.clear()
            
            # Restore snapshot state
            self._states = dict(snapshot.states)
            
            # Sync to Redis
            if self._redis:
                for key, entry in self._states.items():
                    try:
                        redis_key = f"{self._redis_prefix}{key}"
                        redis_value = pickle.dumps(entry)
                        await self._redis.set(redis_key, redis_value)
                    except Exception as e:
                        self.logger.error(f"Failed to sync state to Redis: {e}")
                        
            self.logger.info(f"Restored from snapshot: {snapshot_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to restore snapshot: {e}")
            return False
            
    async def get_system_state(self) -> SystemState:
        """Get complete system state."""
        state = SystemState()
        
        # Collect states by type
        for key, entry in self._states.items():
            if entry.is_expired():
                continue
                
            if key.startswith("orchestrator:"):
                state.orchestrator[key] = entry.value
            elif key.startswith("workflow:"):
                state.workflows[key] = entry.value
            elif key.startswith("service:"):
                state.services[key] = entry.value
            elif key.startswith("event:"):
                state.events[key] = entry.value
            elif key.startswith("metric:"):
                state.metrics[key] = entry.value
                
        return state
        
    async def clear_expired(self) -> int:
        """Clear expired states."""
        expired_count = 0
        expired_keys = []
        
        for key, entry in self._states.items():
            if entry.is_expired():
                expired_keys.append(key)
                expired_count += 1
                
        for key in expired_keys:
            await self.delete_state(key)
            
        if expired_count > 0:
            self.logger.info(f"Cleared {expired_count} expired states")
            
        return expired_count
        
    async def get_statistics(self) -> Dict[str, Any]:
        """Get state manager statistics."""
        return {
            "total_states": len(self._states),
            "states_by_type": {
                state_type.value: sum(1 for e in self._states.values() if e.type == state_type)
                for state_type in StateType
            },
            "expired_states": sum(1 for e in self._states.values() if e.is_expired()),
            "corrupted_states": sum(1 for e in self._states.values() if not e.verify_integrity()),
            "total_snapshots": len(self._snapshots),
            "redis_connected": self._redis is not None,
            "total_locks": len(self._locks),
            "history_entries": sum(len(h) for h in self._state_history.values())
        }
        
    async def _sync_loop(self) -> None:
        """Background task to sync state with Redis."""
        while True:
            try:
                await asyncio.sleep(self._sync_interval)
                
                if self._redis:
                    # Sync local changes to Redis
                    for key, entry in self._states.items():
                        try:
                            redis_key = f"{self._redis_prefix}{key}"
                            redis_value = pickle.dumps(entry)
                            
                            if entry.expires_at:
                                ttl = int((entry.expires_at - datetime.now(timezone.utc)).total_seconds())
                                if ttl > 0:
                                    await self._redis.setex(redis_key, ttl, redis_value)
                            else:
                                await self._redis.set(redis_key, redis_value)
                                
                        except Exception as e:
                            self.logger.error(f"Sync error for {key}: {e}")
                            
                    self.logger.debug("State synchronized with Redis")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Sync loop error: {e}")
                
    async def _persist_loop(self) -> None:
        """Background task to persist state to disk."""
        while True:
            try:
                await asyncio.sleep(self._persist_interval)
                await self._persist_state()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Persist loop error: {e}")
                
    async def _persist_state(self) -> None:
        """Persist state to disk."""
        try:
            # Prepare data
            data = {
                "states": {},
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            for key, entry in self._states.items():
                if not entry.is_expired():
                    data["states"][key] = {
                        "key": entry.key,
                        "value": entry.value,
                        "type": entry.type.value,
                        "status": entry.status.value,
                        "created_at": entry.created_at.isoformat(),
                        "updated_at": entry.updated_at.isoformat(),
                        "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
                        "version": entry.version,
                        "checksum": entry.checksum,
                        "metadata": entry.metadata
                    }
                    
            # Write to file
            state_file = self._persist_path / "state.json"
            temp_file = state_file.with_suffix(".tmp")
            
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)
                
            # Atomic rename
            temp_file.rename(state_file)
            
            self.logger.debug(f"Persisted {len(data['states'])} states to disk")
            
        except Exception as e:
            self.logger.error(f"Failed to persist state: {e}")
            
    async def shutdown(self) -> None:
        """Shutdown state manager."""
        self.logger.info("Shutting down state manager...")
        
        # Cancel background tasks
        if self._sync_task:
            self._sync_task.cancel()
        if self._persist_task:
            self._persist_task.cancel()
            
        # Final persistence
        await self._persist_state()
        
        # Close Redis connection
        if self._redis:
            await self._redis.close()
            
        self.logger.info("State manager shutdown complete")