#!/usr/bin/env python3
"""
SutazAI - Advanced AI System
A comprehensive, well-structured AI system for enterprise and research applications
"""

__version__ = "1.0.0"
__author__ = "SutazAI Team"
__description__ = "Advanced AI System with Neural Processing, Agent Orchestration, and Knowledge Management"

# Core System Components
from .core import (
    SutazAICore,
    SystemConfig,
    SystemStatus,
    SystemMetrics
)

# Model Management
from .models import (
    ModelManager,
    ModelRegistry,
    ModelLoader,
    ModelOptimizer,
    InferenceEngine
)

# Neural Processing
from .neural import (
    NeuralProcessor,
    NeuralNetwork,
    BiologicalModeling,
    NeuromorphicEngine
)

# Agent System
from .agents import (
    AgentOrchestrator,
    AgentFramework,
    AgentManager,
    BaseAgent,
    AgentProtocol
)

# Knowledge Management
from .knowledge import (
    KnowledgeEngine,
    KnowledgeGraph,
    SemanticSearch,
    VectorStore,
    DocumentProcessor
)

# Security and Ethics
from .security import (
    SecurityManager,
    EthicsFramework,
    AccessControl,
    AuditSystem
)

# Monitoring and Observability
from .monitoring import (
    SystemMonitor,
    PerformanceMetrics,
    HealthChecker,
    AlertManager
)

# Utilities
from .utils import (
    Logger,
    ConfigManager,
    CacheManager,
    ValidationEngine
)

__all__ = [
    # Core
    "SutazAICore",
    "SystemConfig", 
    "SystemStatus",
    "SystemMetrics",
    
    # Models
    "ModelManager",
    "ModelRegistry",
    "ModelLoader",
    "ModelOptimizer",
    "InferenceEngine",
    
    # Neural
    "NeuralProcessor",
    "NeuralNetwork",
    "BiologicalModeling",
    "NeuromorphicEngine",
    
    # Agents
    "AgentOrchestrator",
    "AgentFramework",
    "AgentManager",
    "BaseAgent",
    "AgentProtocol",
    
    # Knowledge
    "KnowledgeEngine",
    "KnowledgeGraph",
    "SemanticSearch",
    "VectorStore",
    "DocumentProcessor",
    
    # Security
    "SecurityManager",
    "EthicsFramework",
    "AccessControl",
    "AuditSystem",
    
    # Monitoring
    "SystemMonitor",
    "PerformanceMetrics",
    "HealthChecker",
    "AlertManager",
    
    # Utils
    "Logger",
    "ConfigManager",
    "CacheManager",
    "ValidationEngine"
]

# Factory functions for easy system initialization
def create_sutazai_system(config_path: str = None) -> SutazAICore:
    """Create a complete SutazAI system instance"""
    return SutazAICore(config_path=config_path)

def create_model_manager(config: dict = None) -> ModelManager:
    """Create a model manager instance"""
    return ModelManager(config=config)

def create_agent_orchestrator(config: dict = None) -> AgentOrchestrator:
    """Create an agent orchestrator instance"""
    return AgentOrchestrator(config=config)

def create_knowledge_engine(config: dict = None) -> KnowledgeEngine:
    """Create a knowledge engine instance"""
    return KnowledgeEngine(config=config)

# System information
def get_system_info() -> dict:
    """Get SutazAI system information"""
    return {
        "name": "SutazAI",
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "components": [
            "Core System",
            "Model Management",
            "Neural Processing",
            "Agent Orchestration",
            "Knowledge Management",
            "Security & Ethics",
            "Monitoring & Observability",
            "Utilities"
        ]
    }