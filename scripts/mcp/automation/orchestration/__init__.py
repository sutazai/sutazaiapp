#!/usr/bin/env python3
"""
MCP Orchestration Service

Comprehensive orchestration system for MCP automation components.
Provides central coordination, workflow management, service discovery,
event handling, and policy enforcement for the MCP ecosystem.

Author: Claude AI Assistant (ai-agent-orchestrator)
Created: 2025-08-15 12:04:00 UTC
Version: 1.0.0
"""

from .orchestrator import (
    MCPOrchestrator,
    OrchestrationMode,
    OrchestratorStatus,
    OrchestrationMetrics,
    OrchestrationContext
)

from .workflow_engine import (
    WorkflowEngine,
    WorkflowDefinition,
    WorkflowStep,
    WorkflowExecution,
    WorkflowStatus,
    StepType
)

from .service_registry import (
    ServiceRegistry,
    ServiceInfo,
    ServiceStatus,
    ServiceType,
    ServiceHealth,
    ServiceDependency
)

from .event_manager import (
    EventManager,
    EventBus,
    Event,
    EventType,
    EventPriority,
    EventFilter,
    DeliveryMode,
    Subscription
)

from .api_gateway import (
    create_api_app,
    WorkflowRequest,
    WorkflowResponse,
    ServiceUpdateRequest,
    EventPublishRequest,
    UpdateCheckRequest,
    CleanupRequest,
    SystemControlRequest
)

from .policy_engine import (
    PolicyEngine,
    PolicySet,
    PolicyRule,
    PolicyType,
    PolicyAction,
    PolicySeverity,
    PolicyViolation
)

from .state_manager import (
    StateManager,
    StateEntry,
    StateSnapshot,
    SystemState,
    StateType,
    StateStatus
)

__version__ = "1.0.0"

__all__ = [
    # Orchestrator
    "MCPOrchestrator",
    "OrchestrationMode",
    "OrchestratorStatus",
    "OrchestrationMetrics",
    "OrchestrationContext",
    
    # Workflow Engine
    "WorkflowEngine",
    "WorkflowDefinition",
    "WorkflowStep",
    "WorkflowExecution",
    "WorkflowStatus",
    "StepType",
    
    # Service Registry
    "ServiceRegistry",
    "ServiceInfo",
    "ServiceStatus",
    "ServiceType",
    "ServiceHealth",
    "ServiceDependency",
    
    # Event Management
    "EventManager",
    "EventBus",
    "Event",
    "EventType",
    "EventPriority",
    "EventFilter",
    "DeliveryMode",
    "Subscription",
    
    # API Gateway
    "create_api_app",
    "WorkflowRequest",
    "WorkflowResponse",
    "ServiceUpdateRequest",
    "EventPublishRequest",
    "UpdateCheckRequest",
    "CleanupRequest",
    "SystemControlRequest",
    
    # Policy Engine
    "PolicyEngine",
    "PolicySet",
    "PolicyRule",
    "PolicyType",
    "PolicyAction",
    "PolicySeverity",
    "PolicyViolation",
    
    # State Manager
    "StateManager",
    "StateEntry",
    "StateSnapshot",
    "SystemState",
    "StateType",
    "StateStatus",
]