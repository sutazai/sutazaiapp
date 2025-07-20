"""
Data Models for SutazAI Backend
==============================

Pydantic models and enums for the SutazAI system.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from dataclasses import dataclass


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentStatus(Enum):
    """Agent status"""
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"
    INITIALIZING = "initializing"


class AgentType(Enum):
    """Agent types"""
    CHAT = "chat"
    CODE_GENERATOR = "code_generator"
    DOCUMENT_PROCESSOR = "document_processor"
    WEB_AUTOMATOR = "web_automator"
    SECURITY_ANALYST = "security_analyst"
    DATA_ANALYST = "data_analyst"
    GENERAL_ASSISTANT = "general_assistant"


@dataclass
class Task:
    """Task data class"""
    id: str
    description: str
    task_type: str
    priority: int
    metadata: Dict[str, Any]
    status: TaskStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assigned_agent_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


@dataclass
class Agent:
    """Agent data class"""
    id: str
    name: str
    agent_type: AgentType
    status: AgentStatus
    capabilities: List[str]
    created_at: datetime
    last_activity: Optional[datetime] = None
    current_task_id: Optional[str] = None
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_execution_time: float = 0.0
    configuration: Optional[Dict[str, Any]] = None


# Pydantic models for API
class TaskCreateRequest(BaseModel):
    """Request to create a new task"""
    description: str = Field(..., min_length=1, max_length=5000)
    task_type: str = Field(..., description="Type of task to execute")
    priority: int = Field(default=5, ge=1, le=10)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class TaskResponse(BaseModel):
    """Task response model"""
    id: str
    description: str
    task_type: str
    priority: int
    status: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    assigned_agent_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None


class AgentResponse(BaseModel):
    """Agent response model"""
    id: str
    name: str
    agent_type: str
    status: str
    capabilities: List[str]
    created_at: datetime
    last_activity: Optional[datetime] = None
    current_task_id: Optional[str] = None
    completed_tasks: int = 0
    failed_tasks: int = 0
    total_execution_time: float = 0.0
    performance_metrics: Optional[Dict[str, Any]] = None


class ChatMessage(BaseModel):
    """Chat message model"""
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., min_length=1, max_length=10000)
    timestamp: Optional[datetime] = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class Conversation(BaseModel):
    """Conversation model"""
    id: str
    title: Optional[str] = None
    messages: List[ChatMessage] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ModelInfo(BaseModel):
    """AI model information"""
    name: str
    size: Optional[str] = None
    format: Optional[str] = None
    family: Optional[str] = None
    parameters: Optional[str] = None
    quantization_level: Optional[str] = None
    status: str = "available"
    capabilities: List[str] = Field(default_factory=list)


class SystemHealth(BaseModel):
    """System health information"""
    status: str
    timestamp: datetime
    uptime_seconds: float
    version: str
    components: Dict[str, str] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    alerts: List[str] = Field(default_factory=list)


class PerformanceMetrics(BaseModel):
    """Performance metrics model"""
    requests_per_second: float
    average_response_time: float
    error_rate: float
    active_connections: int
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float] = Field(default_factory=dict)


class DocumentInfo(BaseModel):
    """Document information"""
    id: str
    filename: str
    content_type: str
    size: int
    uploaded_at: datetime
    processed_at: Optional[datetime] = None
    status: str = "uploaded"
    metadata: Dict[str, Any] = Field(default_factory=dict)
    extracted_text: Optional[str] = None
    summary: Optional[str] = None


class CodeGenerationRequest(BaseModel):
    """Code generation request"""
    prompt: str = Field(..., min_length=1, max_length=5000)
    language: str = Field(default="python")
    framework: Optional[str] = None
    style: Optional[str] = None
    test_generation: bool = Field(default=False)
    documentation: bool = Field(default=True)


class CodeGenerationResponse(BaseModel):
    """Code generation response"""
    code: str
    language: str
    framework: Optional[str] = None
    tests: Optional[str] = None
    documentation: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)


class SecurityScanRequest(BaseModel):
    """Security scan request"""
    target_type: str = Field(..., description="Type of target: code, file, url")
    target_data: str = Field(..., description="Target data to scan")
    scan_type: str = Field(default="comprehensive")
    include_recommendations: bool = Field(default=True)


class SecurityScanResponse(BaseModel):
    """Security scan response"""
    scan_id: str
    target_type: str
    scan_type: str
    status: str
    findings: List[Dict[str, Any]] = Field(default_factory=list)
    risk_score: float
    recommendations: List[str] = Field(default_factory=list)
    scanned_at: datetime = Field(default_factory=datetime.now)


class WebAutomationRequest(BaseModel):
    """Web automation request"""
    action_type: str = Field(..., description="Type of action: scrape, form_fill, navigate")
    target_url: str = Field(..., description="Target URL")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    wait_for_load: bool = Field(default=True)
    timeout_seconds: int = Field(default=30)


class WebAutomationResponse(BaseModel):
    """Web automation response"""
    action_id: str
    action_type: str
    target_url: str
    status: str
    result_data: Optional[Dict[str, Any]] = None
    screenshot_url: Optional[str] = None
    executed_at: datetime = Field(default_factory=datetime.now)
    execution_time: float


class DataAnalysisRequest(BaseModel):
    """Data analysis request"""
    data_source: str = Field(..., description="Data source: file, database, api")
    analysis_type: str = Field(..., description="Type of analysis to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    output_format: str = Field(default="json")


class DataAnalysisResponse(BaseModel):
    """Data analysis response"""
    analysis_id: str
    analysis_type: str
    status: str
    results: Dict[str, Any] = Field(default_factory=dict)
    visualizations: List[str] = Field(default_factory=list)
    insights: List[str] = Field(default_factory=list)
    confidence_score: Optional[float] = None
    analyzed_at: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """Standard error response"""
    error_code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None


class SuccessResponse(BaseModel):
    """Standard success response"""
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None