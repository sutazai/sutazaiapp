"""
Dependencies Module

This module provides dependency injection functions for FastAPI routes.
"""

import logging
import threading
from typing import Optional


from .agent_manager import AgentManager
from .protocols.agent_communication import AgentCommunication
from .interaction.human_interaction import InteractionManager
from .metrics.performance_metrics import PerformanceMetrics
from .orchestrator.workflow_engine import WorkflowEngine
from .health_check import HealthCheck
from .memory.agent_memory import MemoryManager
from .memory.shared_memory import SharedMemoryManager

# Configure logging
logger = logging.getLogger(__name__)

# Global instances
_agent_manager: Optional[AgentManager] = None
_agent_communication: Optional[AgentCommunication] = None
_interaction_manager: Optional[InteractionManager] = None
_performance_metrics: Optional[PerformanceMetrics] = None
_workflow_engine: Optional[WorkflowEngine] = None
_memory_manager: Optional[MemoryManager] = None
_shared_memory_manager: Optional[SharedMemoryManager] = None
_health_check: Optional[HealthCheck] = None

# Thread lock for safe initialization
_init_lock = threading.RLock()


def initialize_dependencies():
    """Initialize global dependency instances."""
    global \
        _agent_manager, \
        _agent_communication, \
        _interaction_manager, \
        _performance_metrics, \
        _workflow_engine, \
        _memory_manager, \
        _shared_memory_manager, \
        _health_check

    with _init_lock:
        if _agent_communication is not None:
            return
        try:
            # Initialize components sequentially
            logger.info("Initializing Agent Communication...")
            _agent_communication = AgentCommunication(max_queue_size=1000)
            _agent_communication.start()
            logger.info("Agent Communication initialized and started")

            logger.info("Initializing Interaction Manager...")
            _interaction_manager = InteractionManager(_agent_communication)
            _interaction_manager.start()
            logger.info("Interaction Manager initialized and started")

            logger.info("Initializing Workflow Engine...")
            _workflow_engine = WorkflowEngine(_agent_communication)
            _workflow_engine.start()
            logger.info("Workflow Engine initialized and started")

            logger.info("Initializing Memory Manager...")
            _memory_manager = MemoryManager()
            logger.info("Memory Manager initialized")

            logger.info("Initializing Shared Memory Manager...")
            _shared_memory_manager = SharedMemoryManager()
            _shared_memory_manager.create_memory(
                name="global", description="Global shared memory space for all agents"
            )
            logger.info("Shared Memory Manager initialized with 'global' space")

            logger.info("Initializing Health Check...")
            _health_check = HealthCheck()
            _health_check.start()
            logger.info("Health Check initialized and started")

            logger.info("Initializing Performance Metrics...")
            _performance_metrics = PerformanceMetrics(window_size=1000)
            logger.info("Performance Metrics initialized")

            # Initialize Agent Manager *last*, passing other components
            logger.info("Initializing Agent Manager...")
            _agent_manager = AgentManager(
                agent_communication=_agent_communication,
                interaction_manager=_interaction_manager,
                workflow_engine=_workflow_engine,
                memory_manager=_memory_manager,
                shared_memory_manager=_shared_memory_manager,
                health_check=_health_check,
            )
            logger.info("Agent Manager initialized")

            logger.info("All dependencies initialized successfully")

        except Exception as e:
            logger.exception(f"Error initializing dependencies: {str(e)}")
            raise RuntimeError(f"Failed to initialize dependencies: {str(e)}") from e


def get_agent_manager() -> AgentManager:
    """
    Get the AgentManager instance.

    Returns:
        AgentManager: The global AgentManager instance

    Raises:
        RuntimeError: If initialization fails
    """
    if _agent_manager is None:
        initialize_dependencies()
    return _agent_manager


def get_agent_communication() -> AgentCommunication:
    """
    Get the AgentCommunication instance.

    Returns:
        AgentCommunication: The global AgentCommunication instance

    Raises:
        RuntimeError: If initialization fails
    """
    if _agent_communication is None:
        initialize_dependencies()
    return _agent_communication


def get_interaction_manager() -> InteractionManager:
    """
    Get the InteractionManager instance.

    Returns:
        InteractionManager: The global InteractionManager instance

    Raises:
        RuntimeError: If initialization fails
    """
    if _interaction_manager is None:
        initialize_dependencies()
    return _interaction_manager


def get_performance_metrics() -> PerformanceMetrics:
    """
    Get the PerformanceMetrics instance.

    Returns:
        PerformanceMetrics: The global PerformanceMetrics instance

    Raises:
        RuntimeError: If initialization fails
    """
    if _performance_metrics is None:
        initialize_dependencies()
    return _performance_metrics


def get_workflow_engine() -> WorkflowEngine:
    """
    Get the WorkflowEngine instance.

    Returns:
        WorkflowEngine: The global WorkflowEngine instance

    Raises:
        RuntimeError: If initialization fails
    """
    if _workflow_engine is None:
        initialize_dependencies()
    return _workflow_engine


def get_memory_manager() -> MemoryManager:
    """
    Get the MemoryManager instance.
    """
    if _memory_manager is None:
        initialize_dependencies()
    return _memory_manager


def get_shared_memory_manager() -> SharedMemoryManager:
    """
    Get the SharedMemoryManager instance.
    """
    if _shared_memory_manager is None:
        initialize_dependencies()
    return _shared_memory_manager


def get_health_check() -> HealthCheck:
    """
    Get the HealthCheck instance.
    """
    if _health_check is None:
        initialize_dependencies()
    return _health_check
