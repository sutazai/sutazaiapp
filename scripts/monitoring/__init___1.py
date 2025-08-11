#!/usr/bin/env python3
"""
SutazAI Multi-Modal Fusion System

A comprehensive multi-modal fusion coordination system for the SutazAI platform
that enables seamless integration and understanding across different data modalities.

Key Components:
- Multi-Modal Fusion Coordinator: Main fusion processing engine
- Unified Representation Framework: Cross-modal representation learning
- Cross-Modal Learning System: Advanced attention and alignment mechanisms
- Real-Time Processing Pipeline: High-performance streaming fusion
- Visualization Tools: Comprehensive monitoring and debugging interface

Features:
- Early, Late, and Hybrid fusion strategies
- Temporal synchronization across modalities
- Cross-modal attention mechanisms
- Real-time processing with auto-scaling
- Integration with SutazAI's 69 AI agents
- Support for Text, Voice, Visual, and Sensor modalities
- WebSocket-based real-time monitoring
- Performance optimization for high concurrency

Author: SutazAI Multi-Modal Fusion System
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "SutazAI Multi-Modal Fusion System"

# Core components
from .core.multi_modal_fusion_coordinator import (
    MultiModalFusionCoordinator,
    ModalityType,
    ModalityData,
    FusionResult,
    FusionStrategy,
    TemporalSynchronizer,
    CrossModalAttention,
    EarlyFusionProcessor,
    LateFusionProcessor
)

from .core.unified_representation import (
    UnifiedRepresentationFramework,
    UnifiedRepresentation,
    RepresentationLevel,
    CrossModalEncoder,
    SemanticSpaceMapper,
    AdaptiveRepresentationLearner
)

from .core.cross_modal_learning import (
    CrossModalLearningSystem,
    ContrastiveLearningModule,
    CrossModalAttentionLearner,
    ModalityTransferLearner,
    CrossModalSample,
    LearningStrategy,
    LearningMetrics
)

# Pipeline components
from .pipeline.realtime_fusion_pipeline import (
    RealTimeFusionPipeline,
    ProcessingRequest,
    ProcessingResponse,
    ProcessingPriority,
    PipelineStage,
    PipelineMetrics,
    StreamBuffer,
    LoadBalancer,
    AutoScaler
)

# Visualization components
from .visualization.fusion_visualizer import (
    FusionVisualizer,
    DataCollector,
    VisualizationConfig
)

# Convenience imports for common usage patterns
from .core.multi_modal_fusion_coordinator import ModalityType as Modality
from .core.multi_modal_fusion_coordinator import FusionStrategy as Strategy

# Module metadata
__all__ = [
    # Core classes
    "MultiModalFusionCoordinator",
    "UnifiedRepresentationFramework", 
    "CrossModalLearningSystem",
    "RealTimeFusionPipeline",
    "FusionVisualizer",
    
    # Data types
    "ModalityType",
    "ModalityData",
    "FusionResult",
    "FusionStrategy",
    "UnifiedRepresentation",
    "RepresentationLevel",
    "ProcessingRequest",
    "ProcessingResponse",
    "ProcessingPriority",
    "LearningStrategy",
    "LearningMetrics",
    "PipelineMetrics",
    
    # Processing components
    "TemporalSynchronizer",
    "CrossModalAttention",
    "EarlyFusionProcessor",
    "LateFusionProcessor",
    "CrossModalEncoder",
    "SemanticSpaceMapper",
    "ContrastiveLearningModule",
    "CrossModalAttentionLearner",
    
    # Pipeline components
    "StreamBuffer",
    "LoadBalancer",
    "AutoScaler",
    
    # Visualization components
    "DataCollector",
    "VisualizationConfig",
    
    # Convenience aliases
    "Modality",
    "Strategy"
]

# Version information
VERSION_INFO = {
    "major": 1,
    "minor": 0,
    "patch": 0,
    "release": "stable"
}

def get_version():
    """Get the current version string"""
    return __version__

def get_version_info():
    """Get detailed version information"""
    return VERSION_INFO.copy()

# System requirements check (lightweight, lazy)
def check_requirements(lightweight: bool = True):
    """Check if all required dependencies appear available.

    When lightweight=True, uses importlib.util.find_spec to avoid importing heavy packages
    at module import time. Returns a tuple (ok: bool, missing: list[str]).
    """
    import importlib.util

    required_packages = [
        "torch",
        "numpy",
        "asyncio",
        "websockets",
        "streamlit",
        "plotly",
        "pandas",
        "scikit-learn",
    ]

    missing_packages = []

    if lightweight:
        for package in required_packages:
            if importlib.util.find_spec(package) is None:
                missing_packages.append(package)
        return (len(missing_packages) == 0, missing_packages)
    else:
        # Fallback to strict check that actually imports modules (heavier)
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        return (len(missing_packages) == 0, missing_packages)

# Call check_requirements() explicitly from runtime paths that truly need full fusion features.

# Configuration defaults
DEFAULT_CONFIG = {
    "temporal_window": 2.0,
    "sync_tolerance": 0.2,
    "feature_dim": 768,
    "max_workers": 12,
    "batch_size": 32,
    "cache_size": 1000,
    "enable_real_time": True,
    "enable_visualization": True,
    "websocket_port": 8765
}

def create_fusion_system(config=None):
    """
    Convenience function to create a complete fusion system
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Dictionary containing all main system components
    """
    if config is None:
        config = DEFAULT_CONFIG.copy()
    
    # Create main components
    fusion_coordinator = MultiModalFusionCoordinator()
    representation_framework = UnifiedRepresentationFramework()
    learning_system = CrossModalLearningSystem()
    pipeline = RealTimeFusionPipeline()
    visualizer = FusionVisualizer()
    
    return {
        "coordinator": fusion_coordinator,
        "representations": representation_framework,
        "learning": learning_system,
        "pipeline": pipeline,
        "visualizer": visualizer,
        "config": config
    }

# Integration helpers
def integrate_with_sutazai(agent_orchestrator_url=None, ollama_url=None):
    """
    Helper function to integrate fusion system with SutazAI infrastructure
    
    Args:
        agent_orchestrator_url: URL of the agent orchestrator
        ollama_url: URL of the Ollama service
        
    Returns:
        Configuration dictionary for SutazAI integration
    """
    integration_config = {
        "agent_orchestrator_url": agent_orchestrator_url or "http://backend:8000/api/v1/agents",
        "ollama_url": ollama_url or "http://ollama:11434",
        "jarvis_url": "http://jarvis:8080",
        "chromadb_url": "http://chromadb:8000",
        "qdrant_url": "http://qdrant:6333",
        "neo4j_url": "bolt://neo4j:7687",
        "enable_agent_integration": True,
        "enable_knowledge_graph": True,
        "enable_vector_storage": True
    }
    
    return integration_config

# Module initialization message
import logging
logger = logging.getLogger(__name__)
logger.debug(f"SutazAI Multi-Modal Fusion System v{__version__} loaded (lazy requirements check)")
