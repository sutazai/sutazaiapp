"""
AI Agents Package

This package provides a framework for building, managing, and orchestrating
intelligent agents for various tasks.
"""

# Import core components
from .agent_manager import AgentManager
from .dependencies import (
    get_agent_communication,
    get_agent_manager,
    get_interaction_manager,
    get_performance_metrics,
    get_workflow_engine,
    initialize_agent_dependencies,
)

# Import protocol components
from .protocols.agent_communication import AgentCommunication
from .protocols.message_protocol import Message, MessageProtocol, MessageType

# Import memory components
from .memory.agent_memory import Memory, MemoryEntry, MemoryManager, MemoryType
from .memory.shared_memory import SharedMemory, SharedMemoryEntry, SharedMemoryManager

# Import interaction components
from .interaction.human_interaction import (
    HumanInteractionPoint,
    InteractionManager,
    InteractionRequest,
    InteractionResponse,
    InteractionStatus,
    InteractionType,
)

# Import orchestration components
from .orchestrator.workflow_engine import (
    TaskStatus,
    Workflow,
    WorkflowEngine,
    WorkflowTask,
)

# Import metrics components
from .metrics.performance_metrics import (
    AgentMetricSummary,
    MetricPoint,
    MetricType,
    MetricsAnalyzer,
    PerformanceMetrics,
)

# Remove incorrect initialization call (should happen in main app startup)
# initialize_dependencies()

__all__ = [
    # Core components
    "AgentManager",
    "AgentCommunication",
    "AgentMetricSummary",
    # Functions
    "get_agent_communication",
    "get_agent_manager",
    "get_interaction_manager",
    "get_performance_metrics",
    "get_workflow_engine",
    "initialize_core_dependencies", # Expose correct initializers if needed by app
    "initialize_agent_dependencies",
    # Human Interaction
    "HumanInteractionPoint",
    "InteractionManager",
    "InteractionRequest",
    "InteractionResponse",
    "InteractionStatus",
    "InteractionType",
    # Memory components
    "Memory",
    "MemoryEntry",
    "MemoryManager",
    "MemoryType",
    # Message components
    "Message",
    "MessageProtocol",
    "MessageType",
    # Metrics components
    "MetricPoint",
    "MetricType",
    "MetricsAnalyzer",
    "PerformanceMetrics",
    # Shared Memory
    "SharedMemory",
    "SharedMemoryEntry",
    "SharedMemoryManager",
    # Workflow components
    "TaskStatus",
    "Workflow",
    "WorkflowEngine",
    "WorkflowTask",
]
