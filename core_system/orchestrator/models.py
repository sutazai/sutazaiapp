#!/usr/bin/env python3.11
"""
Models for the Supreme AI Orchestrator

This module defines the data models used by the orchestrator system.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, Optional


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = auto()
    IN_PROGRESS = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class AgentStatus(Enum):
    """AI agent status"""
    IDLE = auto()
    BUSY = auto()
    ERROR = auto()
    OFFLINE = auto()


class SyncStatus(Enum):
    """Synchronization status"""
    SUCCESS = auto()
    PARTIAL = auto()
    FAILED = auto()
    IN_PROGRESS = auto()


@dataclass
class Task:
    """Task data model"""
    id: str
    type: str
    parameters: Dict[str, Any]
    priority: int = 0
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = datetime.now()
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class Agent:
    """AI agent data model"""
    id: str
    type: str
    capabilities: list[str]
    status: AgentStatus = AgentStatus.IDLE
    current_task: Optional[str] = None
    last_heartbeat: datetime = datetime.now()
    last_updated: datetime = datetime.now()
    error_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SyncData:
    """Synchronization data model"""
    timestamp: datetime
    server_id: str
    tasks: Dict[str, Task]
    agents: Dict[str, Agent]
    metadata: Dict[str, Any]


@dataclass
class ServerConfig:
    """Server configuration model"""
    id: str
    host: str
    port: int
    is_primary: bool
    sync_port: int
    api_key: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    primary_server: str
    secondary_server: str
    sync_interval: int
    max_agents: int
    task_timeout: int


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    agent_id: str
    agent_type: str
    capabilities: list[str]
    resources: dict[str, int]
    priority: int = 0


@dataclass
class TaskConfig:
    """Configuration for a task."""
    task_id: str
    task_type: str
    requirements: list[str]
    priority: int = 0
    timeout: int = 3600  # 1 hour default timeout