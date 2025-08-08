from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class TaskRequest(BaseModel):
    """Task request model for Jarvis services"""
    command: str
    context: Optional[Dict[str, Any]] = {}
    voice_enabled: bool = False
    plugins: Optional[List[str]] = []


class TaskResponse(BaseModel):
    """Task response model for Jarvis services"""
    result: Any
    status: str
    execution_time: float
    agents_used: List[str]
    voice_response: Optional[str] = None

