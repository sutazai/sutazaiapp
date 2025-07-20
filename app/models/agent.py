from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any

class AgentType(Enum):
    """Different types of AI agents"""
    CODE_GENERATOR = "code_generator"
    SECURITY_ANALYST = "security_analyst"
    DOCUMENT_PROCESSOR = "document_processor"
    FINANCIAL_ANALYST = "financial_analyst"
    WEB_AUTOMATOR = "web_automator"
    GENERAL_ASSISTANT = "general_assistant"
    TASK_COORDINATOR = "task_coordinator"
    SYSTEM_MONITOR = "system_monitor"
    DATA_SCIENTIST = "data_scientist"
    DEVOPS_ENGINEER = "devops_engineer"
    AIDER = "aider"
    GPT_ENGINEER = "gpt_engineer"
    SEMGREP = "semgrep"

class AgentStatus(Enum):
    """Agent status states"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"
    INITIALIZING = "initializing"

@dataclass
class Agent:
    """Represents an AI agent"""
    id: str
    name: str
    type: AgentType
    status: AgentStatus = AgentStatus.INITIALIZING
    capabilities: List[str] = field(default_factory=list)
    current_task: Optional[str] = None
    completed_tasks: int = 0
    created_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    config: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
