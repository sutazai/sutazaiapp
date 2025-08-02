#!/usr/bin/env python3
"""
SutazAI Knowledge Graph Builder - Comprehensive Implementation

This module builds sophisticated knowledge graphs for SutazAI codebase hygiene standards,
enabling advanced reasoning, compliance validation, and automated enforcement.
"""

import os
import sys
import json
import yaml
import logging
import asyncio
import networkx as nx
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import re
import hashlib

# External dependencies for advanced NLP and graph processing
try:
    import spacy
    import torch
    import torch.nn as nn
    from transformers import AutoModel, AutoTokenizer
    from torch_geometric.nn import GATConv, GCNConv
    from torch_geometric.data import Data
    import neo4j
    from neo4j import GraphDatabase
    import numpy as np
    from rdflib import Graph, Namespace, RDF, RDFS, OWL, URIRef, Literal
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as e:
    logging.warning(f"Optional dependency not available: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('knowledge_graph_builder')

class EnforcementLevel(Enum):
    """Enforcement levels for hygiene standards"""
    BLOCKING = "BLOCKING"
    WARNING = "WARNING"
    GUIDANCE = "GUIDANCE"

class EntityType(Enum):
    """Types of entities in the knowledge graph"""
    STANDARD = "Standard"
    AGENT = "Agent"
    PROCESS = "Process"
    TOOL = "Tool"
    METRIC = "Metric"
    VIOLATION = "Violation"
    REQUIREMENT = "Requirement"
    RULE = "Rule"

class RelationType(Enum):
    """Types of relationships in the knowledge graph"""
    ENFORCES = "enforces"
    DEPENDS_ON = "depends_on"
    VALIDATES = "validates"
    IMPLEMENTS = "implements"
    MONITORS = "monitors"
    TRIGGERS = "triggers"
    REQUIRES = "requires"
    BLOCKS = "blocks"
    WARNS = "warns"
    VIOLATES = "violates"
    COMPLIES_WITH = "complies_with"
    PART_OF = "part_of"
    IS_A = "is_a"

@dataclass
class Entity:
    """Represents an entity in the knowledge graph"""
    id: str
    type: EntityType
    name: str
    description: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Generate unique ID if not provided"""
        if not self.id:
            content = f"{self.type.value}_{self.name}_{self.description}"
            self.id = hashlib.md5(content.encode()).hexdigest()

@dataclass
class Relationship:
    """Represents a relationship between entities"""
    source_id: str
    target_id: str
    type: RelationType
    confidence: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)

@dataclass
class ComplianceCheck:
    """Represents a compliance validation result"""
    entity_id: str
    standard_id: str
    status: str  # "compliant", "warning", "violation"
    message: str
    severity: EnforcementLevel
    timestamp: datetime = field(default_factory=datetime.now)
    evidence: List[str] = field(default_factory=list)

class SutazaiStandardsExtractor:
    """Extracts hygiene standards and rules from SutazAI documentation"""
    
    def __init__(self):
        self.standards = []
        self.rules = []
        self.entities = []
        self.relationships = []
        
        # Initialize NLP model if available
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not available, using basic text processing")
            self.nlp = None
    
    def extract_from_claude_md(self, file_path: str) -> Tuple[List[Entity], List[Relationship]]:
        """Extract standards and rules from CLAUDE.md"""
        logger.info(f"Extracting standards from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        entities = []
        relationships = []
        
        # Extract major rules (Rule 1, Rule 2, etc.)
        rule_pattern = r'📌 Rule (\d+): ([^\n]+)\n(.*?)(?=📌 Rule \d+:|---|\Z)'
        rules = re.findall(rule_pattern, content, re.DOTALL)
        
        for rule_num, rule_title, rule_content in rules:
            rule_entity = Entity(
                id=f"rule_{rule_num}",
                type=EntityType.RULE,
                name=f"Rule {rule_num}: {rule_title}",
                description=rule_content.strip(),
                properties={
                    "rule_number": int(rule_num),
                    "enforcement_level": self._determine_enforcement_level(rule_content),
                    "category": self._categorize_rule(rule_title, rule_content)
                }
            )
            entities.append(rule_entity)
        
        # Extract standards sections
        standards_pattern = r'(🔧|🧠|🚫|📌|🐳|🚀|🔍|⚡|🔧)\s*([^📌🔧🧠🚫🐳🚀🔍⚡\n]+)\n(.*?)(?=🔧|🧠|🚫|📌|🐳|🚀|🔍|⚡|\Z)'
        standards = re.findall(standards_pattern, content, re.DOTALL)
        
        for icon, title, section_content in standards:
            if len(title.strip()) > 5:  # Filter out short matches
                standard_entity = Entity(
                    id=self._generate_id(title),
                    type=EntityType.STANDARD,
                    name=title.strip(),
                    description=section_content.strip(),
                    properties={
                        "icon": icon,
                        "enforcement_level": self._determine_enforcement_level(section_content),
                        "category": self._categorize_standard(title, section_content)
                    }
                )
                entities.append(standard_entity)
        
        # Extract enforcement levels
        blocking_pattern = r'🚨 BLOCKING.*?\n(.*?)(?=⚠️|📋|\Z)'
        warning_pattern = r'⚠️ WARNING.*?\n(.*?)(?=🚨|📋|\Z)'
        guidance_pattern = r'📋 GUIDANCE.*?\n(.*?)(?=🚨|⚠️|\Z)'
        
        for pattern, level in [(blocking_pattern, EnforcementLevel.BLOCKING),
                               (warning_pattern, EnforcementLevel.WARNING),
                               (guidance_pattern, EnforcementLevel.GUIDANCE)]:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                items = re.findall(r'- ([^\n]+)', match)
                for item in items:
                    enforcement_entity = Entity(
                        id=self._generate_id(item),
                        type=EntityType.REQUIREMENT,
                        name=item.strip(),
                        description=f"{level.value} requirement",
                        properties={
                            "enforcement_level": level,
                            "category": "enforcement"
                        }
                    )
                    entities.append(enforcement_entity)
        
        # Extract relationships between entities
        relationships.extend(self._extract_relationships(entities, content))
        
        return entities, relationships
    
    def extract_from_docker_configs(self, config_paths: List[str]) -> Tuple[List[Entity], List[Relationship]]:
        """Extract entities and relationships from Docker configuration files"""
        logger.info(f"Extracting from Docker configs: {config_paths}")
        
        entities = []
        relationships = []
        
        for config_path in config_paths:
            if not os.path.exists(config_path):
                continue
                
            with open(config_path, 'r', encoding='utf-8') as f:
                try:
                    config = yaml.safe_load(f)
                except yaml.YAMLError as e:
                    logger.warning(f"Error parsing {config_path}: {e}")
                    continue
            
            if 'services' in config:
                for service_name, service_config in config['services'].items():
                    # Create agent entity
                    agent_entity = Entity(
                        id=f"agent_{service_name}",
                        type=EntityType.AGENT,
                        name=service_name,
                        description=f"SutazAI agent: {service_name}",
                        properties={
                            "container_name": service_config.get('container_name', ''),
                            "image": service_config.get('image', ''),
                            "resource_limits": service_config.get('deploy', {}).get('resources', {}).get('limits', {}),
                            "health_check": service_config.get('healthcheck', {}),
                            "environment": service_config.get('environment', []),
                            "networks": service_config.get('networks', [])
                        }
                    )
                    entities.append(agent_entity)
                    
                    # Create relationships for dependencies
                    if 'depends_on' in service_config:
                        for dependency in service_config['depends_on']:
                            dep_relationship = Relationship(
                                source_id=agent_entity.id,
                                target_id=f"agent_{dependency}",
                                type=RelationType.DEPENDS_ON,
                                properties={"type": "service_dependency"}
                            )
                            relationships.append(dep_relationship)
                    
                    # Create relationships for external links
                    if 'external_links' in service_config:
                        for link in service_config['external_links']:
                            link_parts = link.split(':')
                            if len(link_parts) == 2:
                                link_relationship = Relationship(
                                    source_id=agent_entity.id,
                                    target_id=f"service_{link_parts[0]}",
                                    type=RelationType.DEPENDS_ON,
                                    properties={"type": "external_link", "alias": link_parts[1]}
                                )
                                relationships.append(link_relationship)
        
        return entities, relationships
    
    def _determine_enforcement_level(self, content: str) -> EnforcementLevel:
        """Determine enforcement level based on content analysis"""
        content_lower = content.lower()
        
        blocking_keywords = ['must', 'mandatory', 'required', 'critical', 'blocking', 'stops deployment', 'prevents']
        warning_keywords = ['should', 'warning', 'requires review', 'recommended']
        
        blocking_score = sum(1 for keyword in blocking_keywords if keyword in content_lower)
        warning_score = sum(1 for keyword in warning_keywords if keyword in content_lower)
        
        if blocking_score > warning_score:
            return EnforcementLevel.BLOCKING
        elif warning_score > 0:
            return EnforcementLevel.WARNING
        else:
            return EnforcementLevel.GUIDANCE
    
    def _categorize_rule(self, title: str, content: str) -> str:
        """Categorize rules based on their content"""
        title_lower = title.lower()
        content_lower = content.lower()
        
        if 'fantasy' in title_lower or 'fantasy' in content_lower:
            return "professional_standards"
        elif 'break' in title_lower or 'functionality' in title_lower:
            return "stability"
        elif 'professional' in title_lower or 'project' in title_lower:
            return "professional_standards"
        elif 'docker' in title_lower or 'deployment' in title_lower:
            return "technical_standards"
        elif 'chaos' in title_lower or 'resilience' in title_lower:
            return "process_standards"
        else:
            return "general"
    
    def _categorize_standard(self, title: str, content: str) -> str:
        """Categorize standards based on their content"""
        title_lower = title.lower()
        content_lower = content.lower()
        
        if any(keyword in title_lower for keyword in ['hygiene', 'clean', 'organization']):
            return "codebase_hygiene"
        elif any(keyword in title_lower for keyword in ['docker', 'container', 'deployment']):
            return "technical_standards"
        elif any(keyword in title_lower for keyword in ['monitoring', 'health', 'performance']):
            return "process_standards"
        elif any(keyword in title_lower for keyword in ['documentation', 'team', 'collaboration']):
            return "team_standards"
        else:
            return "general"
    
    def _generate_id(self, text: str) -> str:
        """Generate a unique ID from text"""
        clean_text = re.sub(r'[^\w\s]', '', text.lower())
        clean_text = re.sub(r'\s+', '_', clean_text.strip())
        return hashlib.md5(clean_text.encode()).hexdigest()[:12]
    
    def _extract_relationships(self, entities: List[Entity], content: str) -> List[Relationship]:
        """Extract relationships between entities based on content analysis"""
        relationships = []
        
        # Create enforcement relationships
        for entity in entities:
            if entity.type == EntityType.RULE:
                enforcement_level = entity.properties.get('enforcement_level')
                if enforcement_level == EnforcementLevel.BLOCKING:
                    # Rules with blocking enforcement block deployment
                    block_relationship = Relationship(
                        source_id=entity.id,
                        target_id="deployment_process",
                        type=RelationType.BLOCKS,
                        confidence=1.0,
                        properties={"reason": "blocking_rule"}
                    )
                    relationships.append(block_relationship)
        
        # Create dependency relationships between standards
        dependency_patterns = [
            (r'depends on', RelationType.DEPENDS_ON),
            (r'requires', RelationType.REQUIRES),
            (r'enforces', RelationType.ENFORCES),
            (r'validates', RelationType.VALIDATES)
        ]
        
        for pattern, rel_type in dependency_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                # Extract context around the match to identify entities
                context_start = max(0, match.start() - 100)
                context_end = min(len(content), match.end() + 100)
                context = content[context_start:context_end]
                
                # This is a simplified approach - in practice, you'd use more sophisticated NLP
                # to identify the specific entities being related
        
        return relationships

class KnowledgeGraphBuilder:
    """Main knowledge graph builder for SutazAI codebase hygiene standards"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.graph = nx.MultiDiGraph()
        self.rdf_graph = Graph()
        self.entities = {}
        self.relationships = []
        
        # Initialize components
        self.extractor = SutazaiStandardsExtractor()
        self.neo4j_driver = None
        self._initialize_neo4j()
        
        # Initialize namespaces for RDF
        self.SUTAZAI = Namespace("http://sutazai.ai/ontology/")
        self.rdf_graph.bind("sutazai", self.SUTAZAI)
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "neo4j": {
                "uri": "bolt://localhost:7687",
                "user": "neo4j",
                "password": "sutazai123"
            },
            "extraction": {
                "confidence_threshold": 0.7,
                "entity_types": [t.value for t in EntityType],
                "relationship_types": [t.value for t in RelationType]
            },
            "reasoning": {
                "max_hops": 5,
                "inference_rules": ["transitive_closure", "inverse_relations"]
            },
            "visualization": {
                "max_nodes": 1000,
                "layout": "force_directed",
                "color_by": "type"
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _initialize_neo4j(self):
        """Initialize Neo4j connection"""
        try:
            neo4j_config = self.config.get('neo4j', {})
            self.neo4j_driver = GraphDatabase.driver(
                neo4j_config.get('uri', 'bolt://localhost:7687'),
                auth=(
                    neo4j_config.get('user', 'neo4j'),
                    neo4j_config.get('password', 'sutazai123')
                )
            )
            logger.info("Neo4j connection established")
        except Exception as e:
            logger.warning(f"Could not connect to Neo4j: {e}")
    
    async def build_comprehensive_graph(self, sources: Dict[str, Any]) -> nx.MultiDiGraph:
        """Build comprehensive knowledge graph from multiple sources"""
        logger.info("Building comprehensive knowledge graph")
        
        all_entities = []
        all_relationships = []
        
        # Extract from CLAUDE.md
        if 'claude_md' in sources:
            claude_entities, claude_relationships = self.extractor.extract_from_claude_md(
                sources['claude_md']
            )
            all_entities.extend(claude_entities)
            all_relationships.extend(claude_relationships)
        
        # Extract from Docker configs
        if 'docker_configs' in sources:
            docker_entities, docker_relationships = self.extractor.extract_from_docker_configs(
                sources['docker_configs']
            )
            all_entities.extend(docker_entities)
            all_relationships.extend(docker_relationships)
        
        # Build NetworkX graph
        self._build_networkx_graph(all_entities, all_relationships)
        
        # Build RDF graph
        self._build_rdf_graph(all_entities, all_relationships)
        
        # Store in Neo4j if available
        if self.neo4j_driver:
            await self._store_in_neo4j(all_entities, all_relationships)
        
        # Perform graph analysis and enrichment
        await self._enrich_graph()
        
        logger.info(f"Knowledge graph built with {len(all_entities)} entities and {len(all_relationships)} relationships")
        return self.graph
    
    def _build_networkx_graph(self, entities: List[Entity], relationships: List[Relationship]):
        """Build NetworkX graph from entities and relationships"""
        # Add entities as nodes
        for entity in entities:
            self.entities[entity.id] = entity
            self.graph.add_node(
                entity.id,
                type=entity.type.value,
                name=entity.name,
                description=entity.description,
                **entity.properties
            )
        
        # Add relationships as edges
        for relationship in relationships:
            if relationship.source_id in self.entities and relationship.target_id in self.entities:
                self.graph.add_edge(
                    relationship.source_id,
                    relationship.target_id,
                    type=relationship.type.value,
                    confidence=relationship.confidence,
                    **relationship.properties
                )
        
        self.relationships = relationships
    
    def _build_rdf_graph(self, entities: List[Entity], relationships: List[Relationship]):
        """Build RDF graph for semantic queries"""
        # Add entities as RDF triples
        for entity in entities:
            entity_uri = URIRef(self.SUTAZAI[entity.id])
            self.rdf_graph.add((entity_uri, RDF.type, URIRef(self.SUTAZAI[entity.type.value])))
            self.rdf_graph.add((entity_uri, RDFS.label, Literal(entity.name)))
            self.rdf_graph.add((entity_uri, RDFS.comment, Literal(entity.description)))
            
            # Add properties
            for prop, value in entity.properties.items():
                prop_uri = URIRef(self.SUTAZAI[prop])
                self.rdf_graph.add((entity_uri, prop_uri, Literal(str(value))))
        
        # Add relationships as RDF triples
        for relationship in relationships:
            source_uri = URIRef(self.SUTAZAI[relationship.source_id])
            target_uri = URIRef(self.SUTAZAI[relationship.target_id])
            predicate_uri = URIRef(self.SUTAZAI[relationship.type.value])
            
            self.rdf_graph.add((source_uri, predicate_uri, target_uri))
    
    async def _store_in_neo4j(self, entities: List[Entity], relationships: List[Relationship]):
        """Store graph in Neo4j database"""
        if not self.neo4j_driver:
            return
        
        with self.neo4j_driver.session() as session:
            # Clear existing data
            session.run("MATCH (n) DETACH DELETE n")
            
            # Create entities
            for entity in entities:
                session.run(
                    f"CREATE (e:{entity.type.value} {{id: $id, name: $name, description: $description}}) "
                    "SET e += $properties",
                    id=entity.id,
                    name=entity.name,
                    description=entity.description,
                    properties=entity.properties
                )
            
            # Create relationships
            for relationship in relationships:
                session.run(
                    f"MATCH (a {{id: $source_id}}), (b {{id: $target_id}}) "
                    f"CREATE (a)-[r:{relationship.type.value.upper()} {{confidence: $confidence}}]->(b) "
                    "SET r += $properties",
                    source_id=relationship.source_id,
                    target_id=relationship.target_id,
                    confidence=relationship.confidence,
                    properties=relationship.properties
                )
    
    async def _enrich_graph(self):
        """Enrich graph with additional relationships and properties"""
        # Add transitive relationships
        self._add_transitive_relationships()
        
        # Calculate centrality measures
        self._calculate_centrality_measures()
        
        # Detect communities/clusters
        self._detect_communities()
        
        # Add inferred relationships
        await self._infer_relationships()
    
    def _add_transitive_relationships(self):
        """Add transitive relationships (e.g., if A depends on B and B depends on C, then A depends on C)"""
        transitive_relations = [RelationType.DEPENDS_ON, RelationType.REQUIRES]
        
        for rel_type in transitive_relations:
            # Find all paths of length 2 for this relationship type
            edges = [(u, v) for u, v, d in self.graph.edges(data=True) 
                    if d.get('type') == rel_type.value]
            
            for u, v in edges:
                for v2, w in edges:
                    if v == v2 and u != w and not self.graph.has_edge(u, w):
                        # Add transitive relationship
                        self.graph.add_edge(
                            u, w,
                            type=rel_type.value,
                            confidence=0.8,
                            inferred=True,
                            inference_type="transitive"
                        )
    
    def _calculate_centrality_measures(self):
        """Calculate various centrality measures for nodes"""
        try:
            # PageRank centrality
            pagerank = nx.pagerank(self.graph)
            
            # Betweenness centrality
            betweenness = nx.betweenness_centrality(self.graph)
            
            # Degree centrality
            degree = nx.degree_centrality(self.graph)
            
            # Add centrality measures to node attributes
            for node in self.graph.nodes():
                self.graph.nodes[node]['pagerank'] = pagerank.get(node, 0)
                self.graph.nodes[node]['betweenness'] = betweenness.get(node, 0)
                self.graph.nodes[node]['degree_centrality'] = degree.get(node, 0)
                
        except Exception as e:
            logger.warning(f"Error calculating centrality measures: {e}")
    
    def _detect_communities(self):
        """Detect communities/clusters in the graph"""
        try:
            # Convert to undirected graph for community detection
            undirected = self.graph.to_undirected()
            
            # Use a simple community detection algorithm
            # In practice, you might use more sophisticated algorithms like Louvain
            communities = list(nx.connected_components(undirected))
            
            # Add community information to nodes
            for i, community in enumerate(communities):
                for node in community:
                    self.graph.nodes[node]['community'] = i
                    
        except Exception as e:
            logger.warning(f"Error detecting communities: {e}")
    
    async def _infer_relationships(self):
        """Infer new relationships based on patterns and rules"""
        # Infer compliance relationships
        for entity_id, entity in self.entities.items():
            if entity.type == EntityType.AGENT:
                # Check if agent implements any standards
                for standard_id, standard in self.entities.items():
                    if standard.type == EntityType.STANDARD:
                        compliance_score = self._calculate_compliance_score(entity, standard)
                        if compliance_score > 0.7:
                            compliance_rel = Relationship(
                                source_id=entity_id,
                                target_id=standard_id,
                                type=RelationType.COMPLIES_WITH,
                                confidence=compliance_score,
                                properties={"inferred": True}
                            )
                            self.relationships.append(compliance_rel)
                            self.graph.add_edge(
                                entity_id, standard_id,
                                type=RelationType.COMPLIES_WITH.value,
                                confidence=compliance_score,
                                inferred=True
                            )
    
    def _calculate_compliance_score(self, agent: Entity, standard: Entity) -> float:
        """Calculate compliance score between agent and standard"""
        # This is a simplified implementation
        # In practice, you'd implement sophisticated compliance checking
        
        score = 0.0
        
        # Check if agent has health checks (required by standards)
        if 'health_check' in agent.properties and agent.properties['health_check']:
            score += 0.3
        
        # Check if agent has resource limits
        if 'resource_limits' in agent.properties and agent.properties['resource_limits']:
            score += 0.3
        
        # Check if agent follows naming conventions
        if 'sutazai-' in agent.properties.get('container_name', ''):
            score += 0.2
        
        # Check if agent is on the sutazai network
        if 'sutazai-network' in str(agent.properties.get('networks', [])):
            score += 0.2
        
        return min(score, 1.0)
    
    async def validate_compliance(self, entity_id: str) -> List[ComplianceCheck]:
        """Validate compliance for a specific entity"""
        compliance_checks = []
        
        if entity_id not in self.entities:
            return compliance_checks
        
        entity = self.entities[entity_id]
        
        if entity.type == EntityType.AGENT:
            # Check agent compliance with various standards
            compliance_checks.extend(await self._check_agent_compliance(entity))
        elif entity.type == EntityType.STANDARD:
            # Check if standard is being followed by entities
            compliance_checks.extend(await self._check_standard_adherence(entity))
        
        return compliance_checks
    
    async def _check_agent_compliance(self, agent: Entity) -> List[ComplianceCheck]:
        """Check agent compliance with standards"""
        checks = []
        
        # Check health check requirement
        if not agent.properties.get('health_check'):
            checks.append(ComplianceCheck(
                entity_id=agent.id,
                standard_id="health_check_requirement",
                status="violation",
                message="Agent does not have health check configured",
                severity=EnforcementLevel.BLOCKING,
                evidence=["Missing healthcheck section in Docker config"]
            ))
        
        # Check resource limits
        resource_limits = agent.properties.get('resource_limits', {})
        if not resource_limits.get('memory') or not resource_limits.get('cpus'):
            checks.append(ComplianceCheck(
                entity_id=agent.id,
                standard_id="resource_limits_requirement",
                status="warning",
                message="Agent does not have complete resource limits",
                severity=EnforcementLevel.WARNING,
                evidence=["Missing or incomplete resource limits in deploy section"]
            ))
        
        # Check naming conventions
        container_name = agent.properties.get('container_name', '')
        if not container_name.startswith('sutazai-'):
            checks.append(ComplianceCheck(
                entity_id=agent.id,
                standard_id="naming_convention",
                status="warning",
                message="Agent container name does not follow naming convention",
                severity=EnforcementLevel.WARNING,
                evidence=[f"Container name '{container_name}' should start with 'sutazai-'"]
            ))
        
        return checks
    
    async def _check_standard_adherence(self, standard: Entity) -> List[ComplianceCheck]:
        """Check how well a standard is being adhered to across the system"""
        checks = []
        
        # Count entities that comply with this standard
        compliant_entities = []
        non_compliant_entities = []
        
        for rel in self.relationships:
            if (rel.target_id == standard.id and 
                rel.type == RelationType.COMPLIES_WITH and 
                rel.confidence > 0.7):
                compliant_entities.append(rel.source_id)
            elif (rel.target_id == standard.id and 
                  rel.type == RelationType.VIOLATES):
                non_compliant_entities.append(rel.source_id)
        
        total_applicable = len(compliant_entities) + len(non_compliant_entities)
        if total_applicable > 0:
            compliance_rate = len(compliant_entities) / total_applicable
            
            if compliance_rate < 0.8:
                severity = EnforcementLevel.WARNING if compliance_rate > 0.5 else EnforcementLevel.BLOCKING
                checks.append(ComplianceCheck(
                    entity_id=standard.id,
                    standard_id=standard.id,
                    status="warning" if compliance_rate > 0.5 else "violation",
                    message=f"Standard compliance rate is {compliance_rate:.1%}",
                    severity=severity,
                    evidence=[f"{len(non_compliant_entities)} entities not compliant"]
                ))
        
        return checks
    
    def query_graph(self, query: str, query_type: str = "cypher") -> List[Dict[str, Any]]:
        """Execute queries against the knowledge graph"""
        if query_type == "cypher" and self.neo4j_driver:
            return self._execute_cypher_query(query)
        elif query_type == "sparql":
            return self._execute_sparql_query(query)
        elif query_type == "networkx":
            return self._execute_networkx_query(query)
        else:
            raise ValueError(f"Unsupported query type: {query_type}")
    
    def _execute_cypher_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute Cypher query against Neo4j"""
        if not self.neo4j_driver:
            return []
        
        with self.neo4j_driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]
    
    def _execute_sparql_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute SPARQL query against RDF graph"""
        try:
            results = self.rdf_graph.query(query)
            return [dict(row.asdict()) for row in results]
        except Exception as e:
            logger.error(f"SPARQL query error: {e}")
            return []
    
    def _execute_networkx_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute NetworkX-based queries"""
        # This would implement a custom query language for NetworkX
        # For now, return basic graph statistics
        return [{
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "is_connected": nx.is_connected(self.graph.to_undirected())
        }]
    
    def generate_compliance_report(self, format: str = "json") -> Dict[str, Any]:
        """Generate comprehensive compliance report"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_entities": len(self.entities),
                "total_relationships": len(self.relationships),
                "total_standards": len([e for e in self.entities.values() if e.type == EntityType.STANDARD]),
                "total_agents": len([e for e in self.entities.values() if e.type == EntityType.AGENT])
            },
            "compliance_by_category": {},
            "violations": [],
            "recommendations": []
        }
        
        # Analyze compliance by category
        categories = set()
        for entity in self.entities.values():
            if entity.type in [EntityType.STANDARD, EntityType.RULE]:
                category = entity.properties.get('category', 'general')
                categories.add(category)
        
        for category in categories:
            category_entities = [e for e in self.entities.values() 
                               if e.properties.get('category') == category]
            
            compliant_count = 0
            total_count = len(category_entities)
            
            # This is simplified - in practice you'd run actual compliance checks
            for entity in category_entities:
                compliance_score = self._calculate_overall_compliance_score(entity)
                if compliance_score > 0.7:
                    compliant_count += 1
            
            report["compliance_by_category"][category] = {
                "total": total_count,
                "compliant": compliant_count,
                "compliance_rate": compliant_count / total_count if total_count > 0 else 0
            }
        
        return report
    
    def _calculate_overall_compliance_score(self, entity: Entity) -> float:
        """Calculate overall compliance score for an entity"""
        # Simplified compliance scoring
        base_score = 0.5
        
        # Add points for having proper documentation
        if entity.description and len(entity.description) > 50:
            base_score += 0.2
        
        # Add points for having proper categorization
        if entity.properties.get('category'):
            base_score += 0.1
        
        # Add points based on enforcement level appropriateness
        enforcement_level = entity.properties.get('enforcement_level')
        if enforcement_level in [EnforcementLevel.BLOCKING, EnforcementLevel.WARNING]:
            base_score += 0.2
        
        return min(base_score, 1.0)
    
    def export_graph(self, format: str, output_path: str):
        """Export knowledge graph in various formats"""
        if format == "graphml":
            nx.write_graphml(self.graph, output_path)
        elif format == "gexf":
            nx.write_gexf(self.graph, output_path)
        elif format == "json":
            data = nx.node_link_data(self.graph)
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == "rdf":
            with open(output_path, 'w') as f:
                f.write(self.rdf_graph.serialize(format='turtle'))
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def get_visualization_data(self, layout: str = "force") -> Dict[str, Any]:
        """Get data formatted for visualization"""
        if layout == "force":
            pos = nx.spring_layout(self.graph, k=1/np.sqrt(self.graph.number_of_nodes()))
        elif layout == "hierarchical":
            try:
                pos = nx.nx_agraph.graphviz_layout(self.graph, prog="dot")
            except:
                pos = nx.spring_layout(self.graph)
        else:
            pos = nx.kamada_kawai_layout(self.graph)
        
        # Prepare nodes data
        nodes = []
        for node, data in self.graph.nodes(data=True):
            entity = self.entities.get(node)
            if entity:
                nodes.append({
                    "id": node,
                    "name": entity.name,
                    "type": entity.type.value,
                    "category": data.get('category', 'general'),
                    "x": pos[node][0] * 1000 if node in pos else 0,
                    "y": pos[node][1] * 1000 if node in pos else 0,
                    "size": data.get('degree_centrality', 0.1) * 100,
                    "enforcement_level": data.get('enforcement_level'),
                    "properties": dict(data)
                })
        
        # Prepare edges data
        edges = []
        for source, target, data in self.graph.edges(data=True):
            edges.append({
                "source": source,
                "target": target,
                "type": data.get('type', 'unknown'),
                "confidence": data.get('confidence', 1.0),
                "properties": dict(data)
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "stats": {
                "total_nodes": len(nodes),
                "total_edges": len(edges),
                "density": nx.density(self.graph)
            }
        }

# CLI Interface and main entry point
def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='SutazAI Knowledge Graph Builder')
    parser.add_argument('command', choices=['build', 'query', 'validate', 'report', 'export', 'visualize'],
                       help='Command to execute')
    parser.add_argument('--source', type=str, help='Source file or directory')
    parser.add_argument('--comprehensive', action='store_true', help='Build comprehensive graph from all sources')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--output', type=str, help='Output file or directory')
    parser.add_argument('--format', type=str, default='json', help='Output format')
    parser.add_argument('--entity', type=str, help='Entity ID for validation')
    parser.add_argument('--query-text', type=str, help='Query text')
    parser.add_argument('--query-type', type=str, default='cypher', help='Query type (cypher, sparql, networkx)')
    
    args = parser.parse_args()
    
    # Initialize knowledge graph builder
    builder = KnowledgeGraphBuilder(config_path=args.config)
    
    if args.command == 'build':
        sources = {}
        
        if args.comprehensive:
            # Build from all available sources
            sources['claude_md'] = '/opt/sutazaiapp/CLAUDE.md'
            sources['docker_configs'] = [
                '/opt/sutazaiapp/docker-compose.complete-agents.yml',
                '/opt/sutazaiapp/docker-compose.agents-simple.yml'
            ]
        elif args.source:
            if args.source.endswith('.md'):
                sources['claude_md'] = args.source
            elif 'docker-compose' in args.source:
                sources['docker_configs'] = [args.source]
        
        # Build the graph
        graph = asyncio.run(builder.build_comprehensive_graph(sources))
        print(f"Knowledge graph built with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")
    
    elif args.command == 'query':
        if not args.query_text:
            print("Error: --query-text required for query command")
            return
        
        results = builder.query_graph(args.query_text, args.query_type)
        print(json.dumps(results, indent=2))
    
    elif args.command == 'validate':
        if not args.entity:
            print("Error: --entity required for validate command")
            return
        
        compliance_checks = asyncio.run(builder.validate_compliance(args.entity))
        
        for check in compliance_checks:
            print(f"Entity: {check.entity_id}")
            print(f"Standard: {check.standard_id}")
            print(f"Status: {check.status}")
            print(f"Severity: {check.severity.value}")
            print(f"Message: {check.message}")
            print(f"Evidence: {', '.join(check.evidence)}")
            print("---")
    
    elif args.command == 'report':
        report = builder.generate_compliance_report(format=args.format)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(report, f, indent=2)
        else:
            print(json.dumps(report, indent=2))
    
    elif args.command == 'export':
        if not args.output:
            print("Error: --output required for export command")
            return
        
        builder.export_graph(args.format, args.output)
        print(f"Graph exported to {args.output} in {args.format} format")
    
    elif args.command == 'visualize':
        viz_data = builder.get_visualization_data()
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(viz_data, f, indent=2)
        else:
            print(json.dumps(viz_data, indent=2))

if __name__ == '__main__':
    main()