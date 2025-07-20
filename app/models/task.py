from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class Task:
    """Represents a task for agents to execute"""
    id: str
    description: str
    type: str
    priority: int = 5
    assigned_agent: Optional[str] = None
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
