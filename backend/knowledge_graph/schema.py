"""
Knowledge Graph Schema for SutazAI Platform
==========================================

Comprehensive schema definition for the SutazAI knowledge graph system.
Supports agent capabilities, service dependencies, data flow patterns,
and system architecture components.

Core Concepts:
- Agents: AI agents with capabilities and relationships
- Services: Backend services and their dependencies
- Data Sources: Databases, APIs, and data stores
- Infrastructure: Containers, networks, and deployments
- Workflows: Processing flows and orchestration patterns
- Knowledge: Documents, models, and learned information
"""

from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from pydantic import BaseModel, Field
import uuid


class NodeType(Enum):
    """Core node types in the knowledge graph"""
    AGENT = "agent"
    SERVICE = "service"
    DATABASE = "database"
    API_ENDPOINT = "api_endpoint"
    CONTAINER = "container"
    WORKFLOW = "workflow"
    CAPABILITY = "capability"
    MODEL = "model"
    DOCUMENT = "document"
    CONCEPT = "concept"
    INFRASTRUCTURE = "infrastructure"
    NAMESPACE = "namespace"
    DEPENDENCY = "dependency"
    INTERFACE = "interface"
    DATA_FLOW = "data_flow"
    EVENT = "event"
    METRIC = "metric"
    CONFIGURATION = "configuration"


class RelationshipType(Enum):
    """Relationship types between nodes"""
    # Agent relationships
    HAS_CAPABILITY = "has_capability"
    ORCHESTRATES = "orchestrates"
    COMMUNICATES_WITH = "communicates_with"
    DELEGATES_TO = "delegates_to"
    MONITORS = "monitors"
    
    # Service relationships
    DEPENDS_ON = "depends_on"
    PROVIDES = "provides"
    CONSUMES = "consumes"
    EXPOSES = "exposes"
    IMPLEMENTS = "implements"
    
    # Data relationships
    READS_FROM = "reads_from"
    WRITES_TO = "writes_to"
    TRANSFORMS = "transforms"
    STORES = "stores"
    INDEXES = "indexes"
    
    # Infrastructure relationships
    DEPLOYED_ON = "deployed_on"
    CONTAINS = "contains"
    ROUTES_TO = "routes_to"
    SCALES = "scales"
    BACKS_UP = "backs_up"
    
    # Workflow relationships
    TRIGGERS = "triggers"
    FOLLOWS = "follows"
    BRANCHES_TO = "branches_to"
    JOINS_FROM = "joins_from"
    
    # Knowledge relationships
    SPECIALIZES = "specializes"
    GENERALIZES = "generalizes"
    RELATES_TO = "relates_to"
    DERIVES_FROM = "derives_from"
    INFLUENCES = "influences"
    
    # Temporal relationships
    CREATED_BEFORE = "created_before"
    MODIFIED_AFTER = "modified_after"
    SUPERSEDES = "supersedes"
    VERSION_OF = "version_of"


@dataclass
class NodeProperties:
    """Base properties for all nodes"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    type: NodeType = NodeType.CONCEPT
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    version: str = "1.0.0"
    status: str = "active"


@dataclass
class AgentNode(NodeProperties):
    """AI Agent node properties"""
    type: NodeType = NodeType.AGENT
    agent_type: str = ""
    capabilities: Set[str] = field(default_factory=set)
    model_config: Dict[str, Any] = field(default_factory=dict)
    host_info: Dict[str, Any] = field(default_factory=dict)
    max_concurrent_tasks: int = 5
    health_status: str = "healthy"
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    specializations: List[str] = field(default_factory=list)


@dataclass
class ServiceNode(NodeProperties):
    """Service node properties"""
    type: NodeType = NodeType.SERVICE
    service_type: str = ""
    port: Optional[int] = None
    endpoints: List[str] = field(default_factory=list)
    health_check_url: str = ""
    deployment_config: Dict[str, Any] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    scaling_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DatabaseNode(NodeProperties):
    """Database node properties"""
    type: NodeType = NodeType.DATABASE
    database_type: str = ""  # postgresql, redis, neo4j, etc.
    connection_string: str = ""
    schema_info: Dict[str, Any] = field(default_factory=dict)
    tables_collections: List[str] = field(default_factory=list)
    indexes: List[str] = field(default_factory=list)
    backup_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowNode(NodeProperties):
    """Workflow node properties"""
    type: NodeType = NodeType.WORKFLOW
    workflow_type: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    trigger_conditions: List[str] = field(default_factory=list)
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    execution_time_ms: Optional[float] = None
    success_rate: float = 100.0


@dataclass
class CapabilityNode(NodeProperties):
    """Capability node properties"""
    type: NodeType = NodeType.CAPABILITY
    capability_type: str = ""
    required_models: List[str] = field(default_factory=list)
    required_resources: Dict[str, Any] = field(default_factory=dict)
    complexity_level: int = 1  # 1-10 scale
    prerequisites: List[str] = field(default_factory=list)


@dataclass
class ModelNode(NodeProperties):
    """AI Model node properties"""
    type: NodeType = NodeType.MODEL
    model_type: str = ""
    framework: str = ""
    model_size: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    supported_tasks: List[str] = field(default_factory=list)


@dataclass
class DocumentNode(NodeProperties):
    """Document/Knowledge node properties"""
    type: NodeType = NodeType.DOCUMENT
    document_type: str = ""
    file_path: str = ""
    content_summary: str = ""
    keywords: List[str] = field(default_factory=list)
    language: str = "en"
    word_count: int = 0
    last_indexed: Optional[datetime] = None


@dataclass
class RelationshipProperties:
    """Properties for relationships between nodes"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    type: RelationshipType = RelationshipType.RELATES_TO
    weight: float = 1.0
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)


class KnowledgeGraphSchema:
    """Complete schema definition for the SutazAI knowledge graph"""
    
    def __init__(self):
        self.node_types = {
            NodeType.AGENT: AgentNode,
            NodeType.SERVICE: ServiceNode,
            NodeType.DATABASE: DatabaseNode,
            NodeType.WORKFLOW: WorkflowNode,
            NodeType.CAPABILITY: CapabilityNode,
            NodeType.MODEL: ModelNode,
            NodeType.DOCUMENT: DocumentNode,
            # Default to base NodeProperties for other types
        }
        
        self.relationship_rules = self._define_relationship_rules()
        self.ontology = self._define_ontology()
    
    def _define_relationship_rules(self) -> Dict[str, List[RelationshipType]]:
        """Define valid relationships between node types"""
        return {
            f"{NodeType.AGENT.value}->{NodeType.CAPABILITY.value}": [
                RelationshipType.HAS_CAPABILITY
            ],
            f"{NodeType.AGENT.value}->{NodeType.AGENT.value}": [
                RelationshipType.ORCHESTRATES,
                RelationshipType.COMMUNICATES_WITH,
                RelationshipType.DELEGATES_TO,
                RelationshipType.MONITORS
            ],
            f"{NodeType.AGENT.value}->{NodeType.SERVICE.value}": [
                RelationshipType.USES,
                RelationshipType.MONITORS,
                RelationshipType.ORCHESTRATES
            ],
            f"{NodeType.AGENT.value}->{NodeType.MODEL.value}": [
                RelationshipType.USES,
                RelationshipType.IMPLEMENTS
            ],
            f"{NodeType.SERVICE.value}->{NodeType.SERVICE.value}": [
                RelationshipType.DEPENDS_ON,
                RelationshipType.COMMUNICATES_WITH
            ],
            f"{NodeType.SERVICE.value}->{NodeType.DATABASE.value}": [
                RelationshipType.READS_FROM,
                RelationshipType.WRITES_TO,
                RelationshipType.DEPENDS_ON
            ],
            f"{NodeType.WORKFLOW.value}->{NodeType.AGENT.value}": [
                RelationshipType.USES,
                RelationshipType.TRIGGERS
            ],
            f"{NodeType.WORKFLOW.value}->{NodeType.SERVICE.value}": [
                RelationshipType.USES,
                RelationshipType.TRIGGERS
            ],
            f"{NodeType.DATABASE.value}->{NodeType.DATABASE.value}": [
                RelationshipType.REPLICATES,
                RelationshipType.BACKS_UP
            ]
        }
    
    def _define_ontology(self) -> Dict[str, Any]:
        """Define the ontology structure"""
        return {
            "namespaces": {
                "sutazai": "http://sutazai.com/ontology/",
                "agent": "http://sutazai.com/ontology/agent/",
                "service": "http://sutazai.com/ontology/service/",
                "capability": "http://sutazai.com/ontology/capability/",
                "workflow": "http://sutazai.com/ontology/workflow/"
            },
            "classes": {
                "Agent": {
                    "subclasses": [
                        "CodeGenerationAgent",
                        "SecurityAgent", 
                        "TestingAgent",
                        "MonitoringAgent",
                        "OrchestrationAgent"
                    ],
                    "properties": [
                        "hasCapability",
                        "hasModel",
                        "hasSpecialization"
                    ]
                },
                "Service": {
                    "subclasses": [
                        "APIService",
                        "DatabaseService",
                        "MessageService",
                        "ComputeService"
                    ],
                    "properties": [
                        "exposesEndpoint",
                        "dependsOn",
                        "hasDeployment"
                    ]
                },
                "Capability": {
                    "subclasses": [
                        "CodeCapability",
                        "AnalysisCapability",
                        "CommunicationCapability",
                        "ProcessingCapability"
                    ]
                }
            },
            "properties": {
                "hasCapability": {
                    "domain": "Agent",
                    "range": "Capability"
                },
                "dependsOn": {
                    "domain": "Service",
                    "range": "Service"
                },
                "orchestrates": {
                    "domain": "Agent",
                    "range": ["Agent", "Service", "Workflow"]
                }
            }
        }
    
    def create_node(self, node_type: NodeType, **kwargs) -> NodeProperties:
        """Create a new node with proper type"""
        node_class = self.node_types.get(node_type, NodeProperties)
        return node_class(type=node_type, **kwargs)
    
    def validate_relationship(self, source_type: NodeType, 
                            target_type: NodeType, 
                            rel_type: RelationshipType) -> bool:
        """Validate if a relationship is allowed"""
        rule_key = f"{source_type.value}->{target_type.value}"
        allowed_relationships = self.relationship_rules.get(rule_key, [])
        return rel_type in allowed_relationships
    
    def get_node_schema(self, node_type: NodeType) -> Dict[str, Any]:
        """Get JSON schema for a node type"""
        node_class = self.node_types.get(node_type, NodeProperties)
        
        # Generate basic schema from dataclass
        schema = {
            "type": "object",
            "properties": {},
            "required": ["id", "name", "type"]
        }
        
        # Add properties based on node type
        if node_type == NodeType.AGENT:
            schema["properties"].update({
                "agent_type": {"type": "string"},
                "capabilities": {"type": "array", "items": {"type": "string"}},
                "model_config": {"type": "object"},
                "health_status": {"type": "string", "enum": ["healthy", "warning", "critical", "unresponsive"]}
            })
        elif node_type == NodeType.SERVICE:
            schema["properties"].update({
                "service_type": {"type": "string"},
                "port": {"type": "integer"},
                "endpoints": {"type": "array", "items": {"type": "string"}},
                "health_check_url": {"type": "string"}
            })
        elif node_type == NodeType.DATABASE:
            schema["properties"].update({
                "database_type": {"type": "string"},
                "tables_collections": {"type": "array", "items": {"type": "string"}},
                "indexes": {"type": "array", "items": {"type": "string"}}
            })
        
        return schema
    
    def get_cypher_schema(self) -> str:
        """Generate Cypher schema for Neo4j"""
        cypher_statements = []
        
        # Create constraints for unique IDs
        for node_type in NodeType:
            cypher_statements.append(
                f"CREATE CONSTRAINT {node_type.value}_id_unique IF NOT EXISTS "
                f"FOR (n:{node_type.value.title()}) REQUIRE n.id IS UNIQUE"
            )
        
        # Create indexes for common properties
        common_indexes = [
            ("name", "String"),
            ("type", "String"),
            ("created_at", "DateTime"),
            ("status", "String")
        ]
        
        for node_type in NodeType:
            for prop, prop_type in common_indexes:
                cypher_statements.append(
                    f"CREATE INDEX {node_type.value}_{prop}_index IF NOT EXISTS "
                    f"FOR (n:{node_type.value.title()}) ON (n.{prop})"
                )
        
        return "\n".join(cypher_statements)
    
    def get_rdf_schema(self) -> str:
        """Generate RDF/Turtle schema"""
        turtle_schema = """
@prefix sutazai: <http://sutazai.com/ontology/> .
@prefix agent: <http://sutazai.com/ontology/agent/> .
@prefix service: <http://sutazai.com/ontology/service/> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .

# Agent Classes
agent:Agent a owl:Class ;
    rdfs:label "AI Agent" ;
    rdfs:comment "An autonomous AI agent in the SutazAI system" .

agent:CodeGenerationAgent a owl:Class ;
    rdfs:subClassOf agent:Agent ;
    rdfs:label "Code Generation Agent" .

agent:SecurityAgent a owl:Class ;
    rdfs:subClassOf agent:Agent ;
    rdfs:label "Security Analysis Agent" .

# Service Classes  
service:Service a owl:Class ;
    rdfs:label "System Service" ;
    rdfs:comment "A service component in the SutazAI system" .

service:APIService a owl:Class ;
    rdfs:subClassOf service:Service ;
    rdfs:label "API Service" .

# Properties
sutazai:hasCapability a owl:ObjectProperty ;
    rdfs:domain agent:Agent ;
    rdfs:range sutazai:Capability .

sutazai:dependsOn a owl:ObjectProperty ;
    rdfs:domain service:Service ;
    rdfs:range service:Service .

sutazai:orchestrates a owl:ObjectProperty ;
    rdfs:domain agent:Agent .
        """
        return turtle_schema


# Predefined capability categories for the SutazAI system
CAPABILITY_CATEGORIES = {
    "code": [
        "code_generation",
        "code_analysis", 
        "code_review",
        "refactoring",
        "debugging",
        "testing"
    ],
    "security": [
        "security_analysis",
        "vulnerability_scanning",
        "compliance_checking",
        "threat_modeling",
        "incident_response"
    ],
    "infrastructure": [
        "deployment",
        "monitoring",
        "scaling",
        "backup",
        "disaster_recovery"
    ],
    "data": [
        "data_processing",
        "data_analysis",
        "data_transformation",
        "data_validation",
        "data_visualization"
    ],
    "communication": [
        "message_routing",
        "protocol_translation",
        "api_integration",
        "event_handling"
    ],
    "intelligence": [
        "reasoning",
        "learning",
        "planning",
        "decision_making",
        "optimization"
    ]
}

# Service type classifications
SERVICE_TYPES = {
    "api": [
        "rest_api",
        "graphql_api", 
        "websocket_api",
        "grpc_service"
    ],
    "data": [
        "database",
        "cache",
        "message_queue",
        "search_engine",
        "data_warehouse"
    ],
    "compute": [
        "worker_service",
        "batch_processor",
        "stream_processor",
        "ml_inference"
    ],
    "infrastructure": [
        "load_balancer",
        "api_gateway",
        "service_mesh",
        "monitoring",
        "logging"
    ]
}