"""
SutazAI Knowledge Graph System
=============================

A comprehensive knowledge graph system for the SutazAI platform that provides:
- Agent capability mapping and discovery
- Service dependency tracking
- Data flow visualization
- System architecture insights
- Real-time graph updates
- Intelligent query interfaces

Components:
- schema.py: Core graph schema and ontology definitions
- graph_builder.py: Knowledge extraction and graph construction
- neo4j_manager.py: Neo4j database integration
- query_engine.py: Graph query and reasoning engine
- visualization.py: Graph visualization interfaces
- real_time_updater.py: Real-time graph synchronization
"""

from .schema import (
    NodeType,
    RelationshipType, 
    NodeProperties,
    AgentNode,
    ServiceNode,
    DatabaseNode,
    WorkflowNode,
    CapabilityNode,
    ModelNode,
    DocumentNode,
    RelationshipProperties,
    KnowledgeGraphSchema,
    CAPABILITY_CATEGORIES,
    SERVICE_TYPES
)

__version__ = "1.0.0"
__author__ = "SutazAI Team"
__email__ = "dev@sutazai.com"

__all__ = [
    "NodeType",
    "RelationshipType",
    "NodeProperties", 
    "AgentNode",
    "ServiceNode",
    "DatabaseNode",
    "WorkflowNode",
    "CapabilityNode",
    "ModelNode",
    "DocumentNode",
    "RelationshipProperties",
    "KnowledgeGraphSchema",
    "CAPABILITY_CATEGORIES",
    "SERVICE_TYPES"
]