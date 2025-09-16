"""Frontend Services"""

# Import BackendClient from the fixed version
try:
    from .backend_client_fixed import BackendClient
    backend_client = BackendClient()
except ImportError:
    # Fallback if fixed version not available
    from .backend_client import BackendClient, backend_client

from .agent_orchestrator import AgentOrchestrator, Agent, Task, AgentStatus, TaskStatus, TaskPriority

__all__ = [
    "BackendClient", 
    "backend_client",
    "AgentOrchestrator",
    "Agent",
    "Task",
    "AgentStatus",
    "TaskStatus",
    "TaskPriority"
]