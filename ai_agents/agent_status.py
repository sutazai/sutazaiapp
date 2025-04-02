from enum import Enum

class AgentStatus(Enum):
    """Status of an agent."""

    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped" 