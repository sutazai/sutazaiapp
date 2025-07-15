"""
Knowledge Graph (KG)
Centralized repository for storing and retrieving system knowledge, algorithms, and data structures
"""

import asyncio
import logging
import json
import time
import uuid
import hashlib
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import networkx as nx
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class KnowledgeType(str, Enum):
    ALGORITHM = "algorithm"
    DATA_STRUCTURE = "data_structure"
    ARCHITECTURE = "architecture"
    PATTERN = "pattern"
    CONCEPT = "concept"
    RELATIONSHIP = "relationship"
    METADATA = "metadata"

class AccessLevel(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    CONFIDENTIAL = "confidential"

class RelationType(str, Enum):
    DEPENDS_ON = "depends_on"
    IMPLEMENTS = "implements"
    EXTENDS = "extends"
    USES = "uses"
    CONTAINS = "contains"
    SIMILAR_TO = "similar_to"
    REFERENCES = "references"

@dataclass
class KnowledgeNode:
    """Individual knowledge node in the graph"""
    id: str
    name: str
    knowledge_type: KnowledgeType
    content: Dict[str, Any]
    metadata: Dict[str, Any]
    access_level: AccessLevel
    created_at: float
    updated_at: float
    version: int = 1
    tags: List[str] = None
    confidence: float = 1.0
    usage_count: int = 0
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

@dataclass
class KnowledgeRelation:
    """Relationship between knowledge nodes"""
    id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    strength: float  # 0.0 to 1.0
    properties: Dict[str, Any]
    created_at: float
    validated: bool = True
    confidence: float = 1.0

@dataclass
class QueryResult:
    """Result of a knowledge query"""
    nodes: List[KnowledgeNode]
    relations: List[KnowledgeRelation]
    paths: List[List[str]]
    relevance_scores: Dict[str, float]
    query_time: float
    total_results: int

class KnowledgeGraph:
    """
    Advanced Knowledge Graph system for Sutazai
    Manages all system knowledge including algorithms, patterns, and relationships
    """
    
    # Hardcoded authorization
    AUTHORIZED_USER = "chrissuta01@gmail.com"
    
    def __init__(self, data_dir: str = "/opt/sutazaiapp/data/knowledge_graph"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Core graph storage
        self.graph = nx.MultiDiGraph()
        self.nodes = {}  # node_id -> KnowledgeNode
        self.relations = {}  # relation_id -> KnowledgeRelation
        
        # Indexing and search
        self.content_index = defaultdict(set)  # keyword -> node_ids
        self.type_index = defaultdict(set)     # type -> node_ids
        self.tag_index = defaultdict(set)      # tag -> node_ids
        self.temporal_index = []               # time-ordered node_ids
        
        # Caching and performance
        self.query_cache = {}
        self.similarity_cache = {}
        self.path_cache = {}
        
        # Analytics
        self.usage_metrics = {
            "total_queries": 0,
            "cache_hits": 0,
            "average_query_time": 0.0,
            "popular_nodes": {},
            "query_patterns": defaultdict(int)
        }
        
        # Initialize system
        self._load_existing_knowledge()
        self._initialize_core_knowledge()
        self._build_indexes()
        
    def _load_existing_knowledge(self):
        """Load existing knowledge from disk"""
        try:
            # Load nodes
            nodes_file = self.data_dir / "nodes.json"
            if nodes_file.exists():
                with open(nodes_file, 'r') as f:
                    data = json.load(f)
                    for node_data in data.get("nodes", []):
                        node = KnowledgeNode(**node_data)
                        self.nodes[node.id] = node
                        self.graph.add_node(node.id, **asdict(node))
            
            # Load relations
            relations_file = self.data_dir / "relations.json"
            if relations_file.exists():
                with open(relations_file, 'r') as f:
                    data = json.load(f)
                    for relation_data in data.get("relations", []):
                        relation = KnowledgeRelation(**relation_data)
                        self.relations[relation.id] = relation
                        self.graph.add_edge(
                            relation.source_id,
                            relation.target_id,
                            key=relation.id,
                            **asdict(relation)
                        )
            
            # Load metrics
            metrics_file = self.data_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    self.usage_metrics.update(json.load(f))
            
            logger.info(f"✅ Loaded {len(self.nodes)} nodes and {len(self.relations)} relations")
            
        except Exception as e:
            logger.error(f"Failed to load existing knowledge: {e}")
    
    def _initialize_core_knowledge(self):
        """Initialize core system knowledge"""
        try:
            if not self.nodes:  # Only initialize if empty
                core_knowledge = [
                    {
                        "name": "Sutazai Core System",
                        "knowledge_type": KnowledgeType.ARCHITECTURE,
                        "content": {
                            "description": "Main Sutazai AGI system architecture",
                            "components": ["CGM", "KG", "ACM"],
                            "design_patterns": ["modular", "distributed", "self-improving"],
                            "technologies": ["Python", "PyTorch", "NetworkX", "FastAPI"]
                        },
                        "tags": ["core", "architecture", "sutazai"],
                        "access_level": AccessLevel.INTERNAL
                    },
                    {
                        "name": "Code Generation Module",
                        "knowledge_type": KnowledgeType.ALGORITHM,
                        "content": {
                            "description": "Neural code generation with meta-learning",
                            "input": "task specifications",
                            "output": "generated code",
                            "techniques": ["transformer", "meta-learning", "reinforcement"],
                            "complexity": "O(n*m) where n=input_length, m=output_length"
                        },
                        "tags": ["cgm", "generation", "neural"],
                        "access_level": AccessLevel.INTERNAL
                    },
                    {
                        "name": "Neural Network Architecture",
                        "knowledge_type": KnowledgeType.DATA_STRUCTURE,
                        "content": {
                            "description": "Transformer-based neural network for code generation",
                            "layers": ["embedding", "transformer_encoder", "output_linear"],
                            "parameters": {"hidden_size": 512, "num_layers": 6, "num_heads": 8},
                            "activation": "GELU",
                            "optimization": "AdamW"
                        },
                        "tags": ["neural", "transformer", "architecture"],
                        "access_level": AccessLevel.INTERNAL
                    },
                    {
                        "name": "Meta-Learning Algorithm",
                        "knowledge_type": KnowledgeType.ALGORITHM,
                        "content": {
                            "description": "Few-shot learning for rapid adaptation",
                            "algorithm": "Model-Agnostic Meta-Learning (MAML)",
                            "steps": ["inner_loop_adaptation", "outer_loop_optimization"],
                            "learning_rate": {"inner": 0.01, "outer": 0.001},
                            "adaptation_steps": 5
                        },
                        "tags": ["meta-learning", "adaptation", "maml"],
                        "access_level": AccessLevel.RESTRICTED
                    },
                    {
                        "name": "Security Authorization Pattern",
                        "knowledge_type": KnowledgeType.PATTERN,
                        "content": {
                            "description": "Hardcoded authorization for system control",
                            "authorized_user": "chrissuta01@gmail.com",
                            "permissions": ["system_shutdown", "critical_changes", "data_access"],
                            "verification_methods": ["email_check", "session_validation", "mfa"],
                            "security_level": "maximum"
                        },
                        "tags": ["security", "authorization", "critical"],
                        "access_level": AccessLevel.CONFIDENTIAL
                    }
                ]
                
                for knowledge_data in core_knowledge:
                    await self.add_knowledge_node(
                        name=knowledge_data["name"],
                        knowledge_type=knowledge_data["knowledge_type"],
                        content=knowledge_data["content"],
                        tags=knowledge_data["tags"],
                        access_level=knowledge_data["access_level"],
                        user_id=self.AUTHORIZED_USER
                    )
                
                # Create relationships
                await self._create_core_relationships()
                
                logger.info("✅ Core knowledge initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize core knowledge: {e}")
    
    async def _create_core_relationships(self):
        """Create relationships between core knowledge nodes"""
        try:
            # Find core nodes
            sutazai_core = self._find_node_by_name("Sutazai Core System")
            cgm_node = self._find_node_by_name("Code Generation Module")
            neural_arch = self._find_node_by_name("Neural Network Architecture")
            meta_learning = self._find_node_by_name("Meta-Learning Algorithm")
            security_pattern = self._find_node_by_name("Security Authorization Pattern")
            
            if not all([sutazai_core, cgm_node, neural_arch, meta_learning, security_pattern]):
                return
            
            relationships = [
                (sutazai_core.id, cgm_node.id, RelationType.CONTAINS, 1.0),
                (cgm_node.id, neural_arch.id, RelationType.USES, 0.9),
                (cgm_node.id, meta_learning.id, RelationType.IMPLEMENTS, 0.8),
                (sutazai_core.id, security_pattern.id, RelationType.IMPLEMENTS, 1.0),
                (neural_arch.id, meta_learning.id, RelationType.DEPENDS_ON, 0.7)
            ]
            
            for source_id, target_id, relation_type, strength in relationships:
                await self.add_relation(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type,
                    strength=strength,
                    properties={"auto_generated": True},
                    user_id=self.AUTHORIZED_USER
                )
                
        except Exception as e:
            logger.error(f"Failed to create core relationships: {e}")
    
    def _find_node_by_name(self, name: str) -> Optional[KnowledgeNode]:
        """Find a node by name"""
        for node in self.nodes.values():
            if node.name == name:
                return node
        return None
    
    def _build_indexes(self):
        """Build search indexes"""
        try:
            self.content_index.clear()
            self.type_index.clear()
            self.tag_index.clear()
            self.temporal_index.clear()
            
            for node in self.nodes.values():
                # Content indexing
                content_text = json.dumps(node.content).lower()
                words = content_text.split()
                for word in words:
                    if len(word) > 2:  # Skip short words
                        self.content_index[word].add(node.id)
                
                # Type indexing
                self.type_index[node.knowledge_type.value].add(node.id)
                
                # Tag indexing
                for tag in node.tags:
                    self.tag_index[tag.lower()].add(node.id)
                
                # Temporal indexing
                self.temporal_index.append((node.created_at, node.id))
            
            # Sort temporal index
            self.temporal_index.sort(key=lambda x: x[0])
            
            logger.info("✅ Search indexes built")
            
        except Exception as e:
            logger.error(f"Failed to build indexes: {e}")
    
    async def add_knowledge_node(
        self, 
        name: str, 
        knowledge_type: KnowledgeType, 
        content: Dict[str, Any], 
        user_id: str,
        tags: List[str] = None,
        access_level: AccessLevel = AccessLevel.INTERNAL,
        metadata: Dict[str, Any] = None
    ) -> str:
        """Add a new knowledge node"""
        try:
            # Authorization check for confidential knowledge
            if access_level == AccessLevel.CONFIDENTIAL and user_id != self.AUTHORIZED_USER:
                raise PermissionError("Only authorized user can create confidential knowledge")
            
            node_id = str(uuid.uuid4())
            current_time = time.time()
            
            node = KnowledgeNode(
                id=node_id,
                name=name,
                knowledge_type=knowledge_type,
                content=content,
                metadata=metadata or {},
                access_level=access_level,
                created_at=current_time,
                updated_at=current_time,
                tags=tags or []
            )
            
            # Add to storage
            self.nodes[node_id] = node
            self.graph.add_node(node_id, **asdict(node))
            
            # Update indexes
            self._update_indexes_for_node(node)
            
            logger.info(f"✅ Added knowledge node: {name} ({node_id})")
            return node_id
            
        except Exception as e:
            logger.error(f"Failed to add knowledge node: {e}")
            raise
    
    async def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        strength: float,
        user_id: str,
        properties: Dict[str, Any] = None
    ) -> str:
        """Add a relationship between knowledge nodes"""
        try:
            # Validate nodes exist
            if source_id not in self.nodes or target_id not in self.nodes:
                raise ValueError("Source or target node does not exist")
            
            # Authorization check
            source_node = self.nodes[source_id]
            target_node = self.nodes[target_id]
            
            if (source_node.access_level == AccessLevel.CONFIDENTIAL or 
                target_node.access_level == AccessLevel.CONFIDENTIAL) and user_id != self.AUTHORIZED_USER:
                raise PermissionError("Only authorized user can create relations with confidential knowledge")
            
            relation_id = str(uuid.uuid4())
            
            relation = KnowledgeRelation(
                id=relation_id,
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                strength=strength,
                properties=properties or {},
                created_at=time.time()
            )
            
            # Add to storage
            self.relations[relation_id] = relation
            self.graph.add_edge(
                source_id,
                target_id,
                key=relation_id,
                **asdict(relation)
            )
            
            # Clear path cache as graph structure changed
            self.path_cache.clear()
            
            logger.info(f"✅ Added relation: {source_id} -> {target_id} ({relation_type.value})")
            return relation_id
            
        except Exception as e:
            logger.error(f"Failed to add relation: {e}")
            raise
    
    def _update_indexes_for_node(self, node: KnowledgeNode):
        """Update search indexes for a specific node"""
        # Content indexing
        content_text = json.dumps(node.content).lower()
        words = content_text.split()
        for word in words:
            if len(word) > 2:
                self.content_index[word].add(node.id)
        
        # Type indexing
        self.type_index[node.knowledge_type.value].add(node.id)
        
        # Tag indexing
        for tag in node.tags:
            self.tag_index[tag.lower()].add(node.id)
        
        # Temporal indexing
        self.temporal_index.append((node.created_at, node.id))
        self.temporal_index.sort(key=lambda x: x[0])
    
    async def query_knowledge(
        self,
        query: str,
        knowledge_types: List[KnowledgeType] = None,
        tags: List[str] = None,
        access_level: AccessLevel = AccessLevel.PUBLIC,
        user_id: str = "anonymous",
        limit: int = 50
    ) -> QueryResult:
        """Query knowledge graph with advanced search"""
        try:
            start_time = time.time()
            self.usage_metrics["total_queries"] += 1
            
            # Check cache
            cache_key = self._generate_cache_key(query, knowledge_types, tags, access_level, limit)
            if cache_key in self.query_cache:
                self.usage_metrics["cache_hits"] += 1
                cached_result = self.query_cache[cache_key]
                cached_result.query_time = time.time() - start_time
                return cached_result
            
            # Determine accessible access levels based on user
            accessible_levels = self._get_accessible_levels(user_id)
            if access_level not in accessible_levels:
                accessible_levels = [AccessLevel.PUBLIC]
            
            # Find candidate nodes
            candidate_nodes = set()
            
            # Text search
            query_words = query.lower().split()
            for word in query_words:
                if word in self.content_index:
                    candidate_nodes.update(self.content_index[word])
            
            # Type filter
            if knowledge_types:
                type_nodes = set()
                for kt in knowledge_types:
                    type_nodes.update(self.type_index.get(kt.value, set()))
                if candidate_nodes:
                    candidate_nodes &= type_nodes
                else:
                    candidate_nodes = type_nodes
            
            # Tag filter
            if tags:
                tag_nodes = set()
                for tag in tags:
                    tag_nodes.update(self.tag_index.get(tag.lower(), set()))
                if candidate_nodes:
                    candidate_nodes &= tag_nodes
                else:
                    candidate_nodes = tag_nodes
            
            # Access level filter
            filtered_nodes = []
            for node_id in candidate_nodes:
                node = self.nodes[node_id]
                if node.access_level in accessible_levels:
                    filtered_nodes.append(node)
            
            # Calculate relevance scores
            relevance_scores = {}
            for node in filtered_nodes:
                score = self._calculate_relevance_score(node, query, query_words)
                relevance_scores[node.id] = score
            
            # Sort by relevance
            filtered_nodes.sort(key=lambda n: relevance_scores[n.id], reverse=True)
            
            # Limit results
            result_nodes = filtered_nodes[:limit]
            
            # Find relevant relations
            result_node_ids = {node.id for node in result_nodes}
            relevant_relations = []
            for relation in self.relations.values():
                if (relation.source_id in result_node_ids and 
                    relation.target_id in result_node_ids):
                    relevant_relations.append(relation)
            
            # Find paths between nodes
            paths = []
            if len(result_nodes) > 1:
                paths = self._find_relevant_paths(result_node_ids)
            
            # Update usage metrics
            for node in result_nodes:
                node.usage_count += 1
                self.usage_metrics["popular_nodes"][node.id] = node.usage_count
            
            query_time = time.time() - start_time
            self.usage_metrics["average_query_time"] = (
                (self.usage_metrics["average_query_time"] * (self.usage_metrics["total_queries"] - 1) + query_time) /
                self.usage_metrics["total_queries"]
            )
            
            # Create result
            result = QueryResult(
                nodes=result_nodes,
                relations=relevant_relations,
                paths=paths[:10],  # Limit paths
                relevance_scores=relevance_scores,
                query_time=query_time,
                total_results=len(filtered_nodes)
            )
            
            # Cache result
            self.query_cache[cache_key] = result
            if len(self.query_cache) > 1000:  # Limit cache size
                oldest_key = min(self.query_cache.keys())
                del self.query_cache[oldest_key]
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to query knowledge: {e}")
            return QueryResult([], [], [], {}, time.time() - start_time, 0)
    
    def _get_accessible_levels(self, user_id: str) -> List[AccessLevel]:
        """Get access levels accessible to user"""
        if user_id == self.AUTHORIZED_USER:
            return [AccessLevel.PUBLIC, AccessLevel.INTERNAL, AccessLevel.RESTRICTED, AccessLevel.CONFIDENTIAL]
        elif user_id != "anonymous":
            return [AccessLevel.PUBLIC, AccessLevel.INTERNAL]
        else:
            return [AccessLevel.PUBLIC]
    
    def _generate_cache_key(self, query: str, knowledge_types: List[KnowledgeType], tags: List[str], access_level: AccessLevel, limit: int) -> str:
        """Generate cache key for query"""
        key_data = {
            "query": query,
            "types": [kt.value for kt in knowledge_types] if knowledge_types else [],
            "tags": tags or [],
            "access": access_level.value,
            "limit": limit
        }
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def _calculate_relevance_score(self, node: KnowledgeNode, query: str, query_words: List[str]) -> float:
        """Calculate relevance score for a node"""
        score = 0.0
        
        # Exact name match
        if query.lower() in node.name.lower():
            score += 1.0
        
        # Content word matches
        content_text = json.dumps(node.content).lower()
        word_matches = sum(1 for word in query_words if word in content_text)
        score += (word_matches / len(query_words)) * 0.8
        
        # Tag matches
        tag_matches = sum(1 for word in query_words if word in [tag.lower() for tag in node.tags])
        score += (tag_matches / len(query_words)) * 0.6
        
        # Usage popularity boost
        score += min(0.2, node.usage_count * 0.01)
        
        # Confidence boost
        score += node.confidence * 0.1
        
        return score
    
    def _find_relevant_paths(self, node_ids: Set[str]) -> List[List[str]]:
        """Find relevant paths between result nodes"""
        paths = []
        node_list = list(node_ids)
        
        for i in range(len(node_list)):
            for j in range(i + 1, len(node_list)):
                source = node_list[i]
                target = node_list[j]
                
                try:
                    # Find shortest path
                    if nx.has_path(self.graph, source, target):
                        path = nx.shortest_path(self.graph, source, target)
                        if len(path) <= 5:  # Only include short paths
                            paths.append(path)
                except:
                    continue
        
        return paths[:10]  # Limit number of paths
    
    async def get_node_details(self, node_id: str, user_id: str = "anonymous") -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific node"""
        try:
            if node_id not in self.nodes:
                return None
            
            node = self.nodes[node_id]
            
            # Check access permissions
            accessible_levels = self._get_accessible_levels(user_id)
            if node.access_level not in accessible_levels:
                return {"error": "Access denied"}
            
            # Get related nodes
            related_nodes = []
            for relation_id, relation in self.relations.items():
                if relation.source_id == node_id:
                    related_node = self.nodes.get(relation.target_id)
                    if related_node and related_node.access_level in accessible_levels:
                        related_nodes.append({
                            "node": asdict(related_node),
                            "relation": asdict(relation),
                            "direction": "outgoing"
                        })
                elif relation.target_id == node_id:
                    related_node = self.nodes.get(relation.source_id)
                    if related_node and related_node.access_level in accessible_levels:
                        related_nodes.append({
                            "node": asdict(related_node),
                            "relation": asdict(relation),
                            "direction": "incoming"
                        })
            
            # Update usage count
            node.usage_count += 1
            
            return {
                "node": asdict(node),
                "related_nodes": related_nodes,
                "centrality_metrics": self._calculate_node_centrality(node_id),
                "last_accessed": time.time()
            }
            
        except Exception as e:
            logger.error(f"Failed to get node details: {e}")
            return {"error": str(e)}
    
    def _calculate_node_centrality(self, node_id: str) -> Dict[str, float]:
        """Calculate centrality metrics for a node"""
        try:
            metrics = {}
            
            # Degree centrality
            metrics["degree"] = self.graph.degree(node_id)
            
            # In-degree and out-degree
            metrics["in_degree"] = self.graph.in_degree(node_id)
            metrics["out_degree"] = self.graph.out_degree(node_id)
            
            # Betweenness centrality (simplified)
            total_nodes = len(self.graph.nodes())
            if total_nodes > 2:
                try:
                    betweenness = nx.betweenness_centrality(self.graph)
                    metrics["betweenness"] = betweenness.get(node_id, 0.0)
                except:
                    metrics["betweenness"] = 0.0
            else:
                metrics["betweenness"] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate centrality: {e}")
            return {}
    
    async def update_knowledge_node(
        self,
        node_id: str,
        updates: Dict[str, Any],
        user_id: str
    ) -> bool:
        """Update an existing knowledge node"""
        try:
            if node_id not in self.nodes:
                return False
            
            node = self.nodes[node_id]
            
            # Check permissions
            accessible_levels = self._get_accessible_levels(user_id)
            if node.access_level not in accessible_levels:
                raise PermissionError("Access denied")
            
            # Apply updates
            if "content" in updates:
                node.content.update(updates["content"])
            if "tags" in updates:
                node.tags = updates["tags"]
            if "metadata" in updates:
                node.metadata.update(updates["metadata"])
            
            node.updated_at = time.time()
            node.version += 1
            
            # Update graph
            self.graph.nodes[node_id].update(asdict(node))
            
            # Rebuild indexes
            self._build_indexes()
            
            # Clear caches
            self.query_cache.clear()
            
            logger.info(f"✅ Updated knowledge node: {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update knowledge node: {e}")
            return False
    
    async def suggest_related_knowledge(self, node_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Suggest related knowledge based on content similarity and graph structure"""
        try:
            if node_id not in self.nodes:
                return []
            
            source_node = self.nodes[node_id]
            suggestions = []
            
            # Graph-based suggestions (direct neighbors)
            neighbors = set()
            for relation in self.relations.values():
                if relation.source_id == node_id:
                    neighbors.add(relation.target_id)
                elif relation.target_id == node_id:
                    neighbors.add(relation.source_id)
            
            # Content-based suggestions
            source_content = json.dumps(source_node.content).lower()
            source_words = set(source_content.split())
            
            for candidate_id, candidate_node in self.nodes.items():
                if candidate_id == node_id:
                    continue
                
                similarity_score = 0.0
                
                # Direct connection boost
                if candidate_id in neighbors:
                    similarity_score += 0.5
                
                # Content similarity
                candidate_content = json.dumps(candidate_node.content).lower()
                candidate_words = set(candidate_content.split())
                
                if source_words and candidate_words:
                    jaccard_similarity = len(source_words & candidate_words) / len(source_words | candidate_words)
                    similarity_score += jaccard_similarity * 0.4
                
                # Tag similarity
                source_tags = set(tag.lower() for tag in source_node.tags)
                candidate_tags = set(tag.lower() for tag in candidate_node.tags)
                
                if source_tags and candidate_tags:
                    tag_similarity = len(source_tags & candidate_tags) / len(source_tags | candidate_tags)
                    similarity_score += tag_similarity * 0.3
                
                # Type similarity
                if source_node.knowledge_type == candidate_node.knowledge_type:
                    similarity_score += 0.2
                
                if similarity_score > 0.1:  # Minimum threshold
                    suggestions.append({
                        "node": asdict(candidate_node),
                        "similarity_score": similarity_score,
                        "reason": self._generate_suggestion_reason(source_node, candidate_node, similarity_score)
                    })
            
            # Sort by similarity and limit
            suggestions.sort(key=lambda x: x["similarity_score"], reverse=True)
            return suggestions[:limit]
            
        except Exception as e:
            logger.error(f"Failed to suggest related knowledge: {e}")
            return []
    
    def _generate_suggestion_reason(self, source_node: KnowledgeNode, candidate_node: KnowledgeNode, score: float) -> str:
        """Generate human-readable reason for suggestion"""
        reasons = []
        
        if source_node.knowledge_type == candidate_node.knowledge_type:
            reasons.append(f"Same type ({source_node.knowledge_type.value})")
        
        common_tags = set(tag.lower() for tag in source_node.tags) & set(tag.lower() for tag in candidate_node.tags)
        if common_tags:
            reasons.append(f"Common tags: {', '.join(list(common_tags)[:3])}")
        
        if score > 0.7:
            reasons.append("High content similarity")
        elif score > 0.4:
            reasons.append("Moderate content similarity")
        
        return "; ".join(reasons) if reasons else "Related content"
    
    async def get_knowledge_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics about the knowledge graph"""
        try:
            analytics = {
                "overview": {
                    "total_nodes": len(self.nodes),
                    "total_relations": len(self.relations),
                    "total_queries": self.usage_metrics["total_queries"],
                    "cache_hit_rate": (self.usage_metrics["cache_hits"] / max(self.usage_metrics["total_queries"], 1)) * 100,
                    "average_query_time": self.usage_metrics["average_query_time"]
                },
                "knowledge_distribution": {
                    "by_type": {kt.value: len(nodes) for kt, nodes in self.type_index.items()},
                    "by_access_level": {},
                    "by_tag": dict(sorted(
                        {tag: len(nodes) for tag, nodes in self.tag_index.items()}.items(),
                        key=lambda x: x[1], reverse=True
                    )[:20])  # Top 20 tags
                },
                "graph_metrics": {
                    "density": nx.density(self.graph),
                    "average_clustering": nx.average_clustering(self.graph) if len(self.graph) > 2 else 0.0,
                    "number_of_components": nx.number_weakly_connected_components(self.graph)
                },
                "popular_nodes": [
                    {
                        "id": node_id,
                        "name": self.nodes[node_id].name,
                        "usage_count": usage_count
                    }
                    for node_id, usage_count in sorted(
                        self.usage_metrics["popular_nodes"].items(),
                        key=lambda x: x[1], reverse=True
                    )[:10]
                ],
                "recent_activity": self._get_recent_activity()
            }
            
            # Calculate access level distribution
            for node in self.nodes.values():
                level = node.access_level.value
                analytics["knowledge_distribution"]["by_access_level"][level] = \
                    analytics["knowledge_distribution"]["by_access_level"].get(level, 0) + 1
            
            return analytics
            
        except Exception as e:
            logger.error(f"Failed to get analytics: {e}")
            return {"error": str(e)}
    
    def _get_recent_activity(self) -> List[Dict[str, Any]]:
        """Get recent activity in the knowledge graph"""
        recent_nodes = sorted(
            self.nodes.values(),
            key=lambda n: n.updated_at,
            reverse=True
        )[:10]
        
        return [
            {
                "id": node.id,
                "name": node.name,
                "type": node.knowledge_type.value,
                "updated_at": node.updated_at,
                "action": "updated" if node.version > 1 else "created"
            }
            for node in recent_nodes
        ]
    
    async def save_knowledge_graph(self):
        """Save knowledge graph to disk"""
        try:
            # Save nodes
            nodes_data = {
                "nodes": [asdict(node) for node in self.nodes.values()],
                "saved_at": time.time()
            }
            with open(self.data_dir / "nodes.json", 'w') as f:
                json.dump(nodes_data, f, indent=2, default=str)
            
            # Save relations
            relations_data = {
                "relations": [asdict(relation) for relation in self.relations.values()],
                "saved_at": time.time()
            }
            with open(self.data_dir / "relations.json", 'w') as f:
                json.dump(relations_data, f, indent=2, default=str)
            
            # Save metrics
            with open(self.data_dir / "metrics.json", 'w') as f:
                json.dump(self.usage_metrics, f, indent=2, default=str)
            
            # Save graph structure for NetworkX
            nx.write_gpickle(self.graph, self.data_dir / "graph.gpickle")
            
            logger.info("✅ Knowledge graph saved to disk")
            
        except Exception as e:
            logger.error(f"Failed to save knowledge graph: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            # Save before cleanup
            asyncio.create_task(self.save_knowledge_graph())
            
            # Clear caches
            self.query_cache.clear()
            self.similarity_cache.clear()
            self.path_cache.clear()
            
            logger.info("✅ Knowledge graph cleaned up")
            
        except Exception as e:
            logger.error(f"Failed to cleanup: {e}")

# Global instance
knowledge_graph = KnowledgeGraph()

# Convenience functions
async def add_knowledge(name: str, knowledge_type: KnowledgeType, content: Dict[str, Any], user_id: str, **kwargs) -> str:
    """Add knowledge to the graph"""
    return await knowledge_graph.add_knowledge_node(name, knowledge_type, content, user_id, **kwargs)

async def query_knowledge(query: str, user_id: str = "anonymous", **kwargs) -> QueryResult:
    """Query the knowledge graph"""
    return await knowledge_graph.query_knowledge(query, user_id=user_id, **kwargs)

async def get_node_details(node_id: str, user_id: str = "anonymous") -> Optional[Dict[str, Any]]:
    """Get detailed node information"""
    return await knowledge_graph.get_node_details(node_id, user_id)

async def suggest_related(node_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get related knowledge suggestions"""
    return await knowledge_graph.suggest_related_knowledge(node_id, limit)