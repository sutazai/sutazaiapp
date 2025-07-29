#!/usr/bin/env python3
"""
Brain State Model for AGI/ASI System
Defines the state that flows through the LangGraph orchestration
"""

from typing import Dict, List, Any, Optional, TypedDict
from datetime import datetime
from enum import Enum
import uuid


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    PLANNING = "planning"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    IMPROVING = "improving"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentType(str, Enum):
    """Available agent types in the system"""
    # Core AI Agents
    AUTOGEN = "autogen"
    CREWAI = "crewai"
    LANGCHAIN = "langchain"
    AUTOGPT = "autogpt"
    GPT_ENGINEER = "gpt-engineer"
    
    # Specialized Agents
    BROWSER_USE = "browser-use"
    SEMGREP = "semgrep"
    DOCUMIND = "documind"
    METAGPT = "metagpt"
    CAMEL = "camel"
    
    # Code Agents
    AIDER = "aider"
    GPTPILOT = "gpt-pilot"
    DEVIKA = "devika"
    OPENDEVIN = "opendevin"
    
    # Research Agents
    GPTRESEARCHER = "gpt-researcher"
    STORM = "storm"
    KHOJ = "khoj"
    
    # Analysis Agents
    PANDAS_AI = "pandas-ai"
    INTERPRETER = "open-interpreter"
    
    # Creative Agents
    FABRIC = "fabric"
    TXTAI = "txtai"
    
    # Security Agents
    PANDASEC = "pandasec"
    NUCLEI = "nuclei"


class MemoryEntry(TypedDict):
    """Structure for memory entries"""
    id: str
    timestamp: datetime
    content: str
    embedding: Optional[List[float]]
    metadata: Dict[str, Any]
    score: float
    agent_source: str


class TaskResult(TypedDict):
    """Result from an agent task"""
    agent: str
    task_id: str
    output: Any
    success: bool
    error: Optional[str]
    execution_time: float
    resources_used: Dict[str, float]  # CPU, memory, GPU
    quality_score: Optional[float]


class ImprovementPatch(TypedDict):
    """Patch for self-improvement"""
    id: str
    description: str
    files_changed: List[str]
    diff: str
    test_results: Dict[str, Any]
    estimated_impact: float
    pr_url: Optional[str]


class BrainState(TypedDict):
    """Main state object for the Brain's LangGraph workflow"""
    # Request context
    request_id: str
    user_input: str
    timestamp: datetime
    
    # Planning phase
    task_plan: List[Dict[str, Any]]
    selected_agents: List[AgentType]
    resource_allocation: Dict[str, Dict[str, float]]
    
    # Execution phase
    status: TaskStatus
    current_step: int
    agent_results: List[TaskResult]
    
    # Memory integration
    retrieved_memories: List[MemoryEntry]
    new_memories: List[MemoryEntry]
    
    # Evaluation phase
    quality_scores: Dict[str, float]
    overall_score: float
    performance_metrics: Dict[str, Any]
    
    # Self-improvement
    needs_improvement: bool
    improvement_suggestions: List[str]
    patches: List[ImprovementPatch]
    
    # Meta-learning
    model_adaptations: Dict[str, Any]
    learned_patterns: List[Dict[str, Any]]
    
    # System state
    gpu_available: bool
    memory_usage: float
    active_models: List[str]
    error_log: List[Dict[str, Any]]
    
    # Output
    final_output: Any
    output_format: str
    confidence: float


class AgentCapability(TypedDict):
    """Agent capability definition"""
    name: str
    type: AgentType
    capabilities: List[str]
    resource_requirements: Dict[str, float]
    average_latency: float
    success_rate: float
    specializations: List[str]


class BrainConfig(TypedDict):
    """Configuration for the Brain system"""
    # Hardware constraints
    max_memory_gb: float
    gpu_memory_gb: float
    cpu_cores: int
    
    # Model settings
    default_embedding_model: str
    default_reasoning_model: str
    default_coding_model: str
    
    # Thresholds
    min_quality_score: float
    improvement_threshold: float
    memory_retention_days: int
    
    # Parallelism
    max_concurrent_agents: int
    max_model_instances: int
    
    # Self-improvement
    auto_improve: bool
    pr_batch_size: int
    require_human_approval: bool