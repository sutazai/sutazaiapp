"""
Centralized message schemas for inter-agent communication.
All message contracts are defined here to ensure consistency.
"""
from .agent_messages import *
from .task_messages import *
from .resource_messages import *
from .system_messages import *

__all__ = [
    # Agent Messages
    'AgentRegistrationMessage',
    'AgentHeartbeatMessage',
    'AgentStatusMessage',
    'AgentCapabilityMessage',
    
    # Task Messages
    'TaskRequestMessage',
    'TaskResponseMessage',
    'TaskStatusUpdateMessage',
    'TaskAssignmentMessage',
    'TaskCompletionMessage',
    
    # Resource Messages
    'ResourceRequestMessage',
    'ResourceAllocationMessage',
    'ResourceReleaseMessage',
    'ResourceStatusMessage',
    
    # System Messages
    'SystemHealthMessage',
    'SystemAlertMessage',
    'ErrorMessage',
    'AcknowledgementMessage',
    
    # Enums
    'MessageType',
    'Priority',
    'TaskStatus',
    'ResourceType',
    'AgentStatus'
]