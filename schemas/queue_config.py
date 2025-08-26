"""
Centralized RabbitMQ queue and exchange configuration.
All queue names and routing keys are defined here.
"""
from typing import Dict, List
from enum import Enum


class ExchangeConfig:
    """Exchange configuration constants"""
    MAIN = "sutazai.main"
    AGENTS = "sutazai.agents"
    TASKS = "sutazai.tasks"
    RESOURCES = "sutazai.resources"
    SYSTEM = "sutazai.system"
    DLX = "sutazai.dlx"  # Dead letter exchange


class QueueConfig:
    """Queue naming configuration"""
    # Agent-specific queues
    AGENT_PREFIX = "agent"
    
    # Task queues
    TASK_HIGH_PRIORITY = "tasks.high_priority"
    TASK_NORMAL_PRIORITY = "tasks.normal_priority"
    TASK_LOW_PRIORITY = "tasks.low_priority"
    TASK_RESULTS = "tasks.results"
    
    # Resource queues
    RESOURCE_REQUESTS = "resources.requests"
    RESOURCE_ALLOCATIONS = "resources.allocations"
    
    # System queues
    SYSTEM_HEALTH = "system.health"
    SYSTEM_ALERTS = "system.alerts"
    SYSTEM_ERRORS = "system.errors"
    
    # Dead letter queue
    DLQ = "dlq.messages"
    
    @staticmethod
    def agent_queue(agent_id: str) -> str:
        """Generate agent-specific queue name"""
        return f"{QueueConfig.AGENT_PREFIX}.{agent_id}"


class RoutingKeys:
    """Routing key patterns for topic exchanges"""
    
    # Agent routing patterns
    AGENT_ALL = "agent.#"
    AGENT_SPECIFIC = "agent.{agent_id}.#"
    AGENT_REGISTRATION = "agent.*.registration"
    AGENT_HEARTBEAT = "agent.*.heartbeat"
    AGENT_STATUS = "agent.*.status"
    
    # Task routing patterns
    TASK_ALL = "task.#"
    TASK_REQUEST = "task.request.{priority}"
    TASK_ASSIGNMENT = "task.assignment.{agent_id}"
    TASK_STATUS = "task.status.{task_id}"
    TASK_COMPLETION = "task.completion.{task_id}"
    
    # Resource routing patterns
    RESOURCE_ALL = "resource.#"
    RESOURCE_REQUEST = "resource.request.{resource_type}"
    RESOURCE_ALLOCATION = "resource.allocation.{agent_id}"
    RESOURCE_RELEASE = "resource.release.{allocation_id}"
    
    # System routing patterns
    SYSTEM_ALL = "system.#"
    SYSTEM_HEALTH = "system.health.{component}"
    SYSTEM_ALERT = "system.alert.{severity}"
    SYSTEM_ERROR = "system.error.{agent_id}"


class MessageTTL:
    """Message time-to-live settings (in seconds)"""
    DEFAULT = 3600  # 1 hour
    HEARTBEAT = 60  # 1 minute
    TASK_REQUEST = 300  # 5 minutes
    RESOURCE_REQUEST = 120  # 2 minutes
    SYSTEM_ALERT = 7200  # 2 hours
    ERROR = 86400  # 24 hours


class QueueArguments:
    """Standard queue arguments"""
    
    @staticmethod
    def standard_queue() -> Dict:
        """Standard queue with DLX"""
        return {
            "x-message-ttl": MessageTTL.DEFAULT * 1000,
            "x-dead-letter-exchange": ExchangeConfig.DLX,
            "x-max-length": 10000
        }
    
    @staticmethod
    def priority_queue() -> Dict:
        """Priority queue with DLX"""
        return {
            "x-max-priority": 10,
            "x-message-ttl": MessageTTL.DEFAULT * 1000,
            "x-dead-letter-exchange": ExchangeConfig.DLX,
            "x-max-length": 10000
        }
    
    @staticmethod
    def durable_queue() -> Dict:
        """Durable queue for critical messages"""
        return {
            "x-message-ttl": MessageTTL.DEFAULT * 1000,
            "x-dead-letter-exchange": ExchangeConfig.DLX,
            "x-max-length": 50000,
            "durable": True
        }


# Agent to Queue Mapping
AGENT_QUEUE_MAP = {
    "ai_agent_orchestrator": {
        "subscribe": [
            QueueConfig.TASK_RESULTS,
            QueueConfig.SYSTEM_HEALTH,
            QueueConfig.agent_queue("orchestrator")
        ],
        "publish": [
            QueueConfig.TASK_HIGH_PRIORITY,
            QueueConfig.TASK_NORMAL_PRIORITY,
            QueueConfig.SYSTEM_ALERTS
        ]
    },
    "task_assignment_coordinator": {
        "subscribe": [
            QueueConfig.TASK_HIGH_PRIORITY,
            QueueConfig.TASK_NORMAL_PRIORITY,
            QueueConfig.TASK_LOW_PRIORITY,
            QueueConfig.agent_queue("coordinator")
        ],
        "publish": [
            QueueConfig.TASK_RESULTS,
            QueueConfig.RESOURCE_REQUESTS
        ]
    },
    "resource_arbitration_agent": {
        "subscribe": [
            QueueConfig.RESOURCE_REQUESTS,
            QueueConfig.agent_queue("arbitrator")
        ],
        "publish": [
            QueueConfig.RESOURCE_ALLOCATIONS,
            QueueConfig.SYSTEM_ALERTS
        ]
    },
    "hardware_resource_optimizer": {
        "subscribe": [
            QueueConfig.RESOURCE_REQUESTS,
            QueueConfig.SYSTEM_HEALTH,
            QueueConfig.agent_queue("hardware_optimizer")
        ],
        "publish": [
            QueueConfig.RESOURCE_ALLOCATIONS,
            QueueConfig.SYSTEM_HEALTH
        ]
    },
    "multi_agent_coordinator": {
        "subscribe": [
            QueueConfig.agent_queue("multi_coordinator"),
            QueueConfig.SYSTEM_HEALTH
        ],
        "publish": [
            QueueConfig.TASK_HIGH_PRIORITY,
            QueueConfig.TASK_NORMAL_PRIORITY,
            QueueConfig.SYSTEM_ALERTS
        ]
    }
}