#!/usr/bin/env python3
"""
Knowledge Graph Engine
Dynamic knowledge graph construction and management for SutazAI
"""

import asyncio
import json
import logging
import networkx as nx
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from pathlib import Path
import sqlite3
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

@dataclass
class GraphNode:
    """Knowledge graph node representation"""
    id: str
    label: str
    node_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None
    confidence: float = 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass 
class GraphEdge:
    """Knowledge graph edge representation"""
    source_id: str
    target_id: str
    relation_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    confidence: float = 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

class KnowledgeGraphEngine:
    """
    Dynamic knowledge graph engine with semantic reasoning capabilities
    """
    
    def __init__(self, storage_path: str = "data/knowledge_graph.db"):
        """Initialize knowledge graph engine"""
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Graph storage
        self.graph = nx.MultiDiGraph()
        self.nodes: Dict[str, GraphNode] = {}
        self.edges: List[GraphEdge] = []
        
        # Semantic embedding model
        self.embedding_model = None
        self.embedding_cache: Dict[str, np.ndarray] = {}
        
        # Database connection
        self.db_connection = None
        
        # Configuration
        self.similarity_threshold = 0.8
        self.max_embedding_cache = 10000
        
        # Initialize components
        self._initialize_database()
        self._initialize_embedding_model()
        self._load_graph()
        
        logger.info("Knowledge Graph Engine initialized")
    
    def _initialize_database(self):
        """Initialize SQLite database for persistent storage"""
        try:
            self.db_connection = sqlite3.connect(self.storage_path, check_same_thread=False)
            cursor = self.db_connection.cursor()
            
            # Create nodes table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS nodes (
                    id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    node_type TEXT NOT NULL,
                    properties TEXT,
                    embedding BLOB,
                    confidence REAL DEFAULT 1.0,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)
            
            # Create edges table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id TEXT NOT NULL,
                    target_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    properties TEXT,
                    weight REAL DEFAULT 1.0,
                    confidence REAL DEFAULT 1.0,
                    created_at TEXT,
                    FOREIGN KEY (source_id) REFERENCES nodes (id),
                    FOREIGN KEY (target_id) REFERENCES nodes (id)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes (node_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_relation ON edges (relation_type)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_source ON edges (source_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_target ON edges (target_id)")
            
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _initialize_embedding_model(self):
        """Initialize sentence transformer for embeddings"""
        try:
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize embedding model: {e}")
    
    def _load_graph(self):
        """Load graph from database"""
        try:
            cursor = self.db_connection.cursor()
            
            # Load nodes
            cursor.execute("SELECT * FROM nodes")
            for row in cursor.fetchall():
                node_id, label, node_type, properties_json, embedding_blob, confidence, created_at, updated_at = row
                
                properties = json.loads(properties_json) if properties_json else {}
                embedding = np.frombuffer(embedding_blob) if embedding_blob else None
                
                node = GraphNode(
                    id=node_id,
                    label=label,
                    node_type=node_type,
                    properties=properties,
                    embedding=embedding,
                    confidence=confidence,
                    created_at=datetime.fromisoformat(created_at),
                    updated_at=datetime.fromisoformat(updated_at)
                )
                
                self.nodes[node_id] = node
                self.graph.add_node(node_id, **node.properties)
            
            # Load edges
            cursor.execute("SELECT * FROM edges")
            for row in cursor.fetchall():
                _, source_id, target_id, relation_type, properties_json, weight, confidence, created_at = row
                
                properties = json.loads(properties_json) if properties_json else {}
                
                edge = GraphEdge(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type,
                    properties=properties,
                    weight=weight,
                    confidence=confidence,
                    created_at=datetime.fromisoformat(created_at)
                )
                
                self.edges.append(edge)
                self.graph.add_edge(source_id, target_id, 
                                  relation=relation_type, 
                                  weight=weight,
                                  **properties)
            
            logger.info(f"Loaded graph with {len(self.nodes)} nodes and {len(self.edges)} edges")
            
        except Exception as e:
            logger.error(f"Failed to load graph: {e}")
    
    def _generate_embedding(self, text: str) -> Optional[np.ndarray]:
        """Generate embedding for text"""
        if not self.embedding_model:
            return None
        
        # Check cache
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        try:
            embedding = self.embedding_model.encode(text)
            
            # Cache management
            if len(self.embedding_cache) >= self.max_embedding_cache:
                # Remove oldest entry
                oldest_key = next(iter(self.embedding_cache))
                del self.embedding_cache[oldest_key]
            
            self.embedding_cache[text] = embedding
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    async def add_node(self, node: GraphNode, update_if_exists: bool = True) -> bool:
        """Add or update node in knowledge graph"""
        try:
            # Generate embedding for label
            if not node.embedding and node.label:
                node.embedding = self._generate_embedding(node.label)
            
            # Check if node exists
            if node.id in self.nodes:
                if update_if_exists:
                    await self.update_node(node.id, node.properties)
                return True
            
            # Add to memory
            self.nodes[node.id] = node
            self.graph.add_node(node.id, **node.properties)
            
            # Persist to database
            cursor = self.db_connection.cursor()
            embedding_blob = node.embedding.tobytes() if node.embedding is not None else None
            
            cursor.execute("""
                INSERT INTO nodes (id, label, node_type, properties, embedding, confidence, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                node.id,
                node.label,
                node.node_type,
                json.dumps(node.properties),
                embedding_blob,
                node.confidence,
                node.created_at.isoformat(),
                node.updated_at.isoformat()
            ))
            
            self.db_connection.commit()
            
            # Auto-discover relationships
            await self._discover_relationships(node)
            
            logger.debug(f"Added node: {node.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add node {node.id}: {e}")
            return False
    
    async def add_edge(self, edge: GraphEdge) -> bool:
        """Add edge to knowledge graph"""
        try:
            # Verify nodes exist
            if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
                logger.error(f"Cannot add edge: missing nodes {edge.source_id} or {edge.target_id}")
                return False
            
            # Add to memory
            self.edges.append(edge)
            self.graph.add_edge(edge.source_id, edge.target_id,
                              relation=edge.relation_type,
                              weight=edge.weight,
                              **edge.properties)
            
            # Persist to database
            cursor = self.db_connection.cursor()
            cursor.execute("""
                INSERT INTO edges (source_id, target_id, relation_type, properties, weight, confidence, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                edge.source_id,
                edge.target_id,
                edge.relation_type,
                json.dumps(edge.properties),
                edge.weight,
                edge.confidence,
                edge.created_at.isoformat()
            ))
            
            self.db_connection.commit()
            
            logger.debug(f"Added edge: {edge.source_id} -> {edge.target_id} ({edge.relation_type})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add edge: {e}")
            return False
    
    async def _discover_relationships(self, new_node: GraphNode):
        """Automatically discover relationships with existing nodes"""
        if not new_node.embedding is not None:
            return
        
        try:
            for existing_id, existing_node in self.nodes.items():
                if existing_id == new_node.id or existing_node.embedding is None:
                    continue
                
                # Calculate similarity
                similarity = cosine_similarity([new_node.embedding], [existing_node.embedding])[0][0]
                
                if similarity > self.similarity_threshold:
                    # Create similarity edge
                    edge = GraphEdge(
                        source_id=new_node.id,
                        target_id=existing_id,
                        relation_type="similar_to",
                        properties={"similarity": float(similarity)},
                        weight=similarity,
                        confidence=similarity
                    )
                    
                    await self.add_edge(edge)
            
        except Exception as e:
            logger.error(f"Failed to discover relationships: {e}")
    
    async def update_node(self, node_id: str, properties: Dict[str, Any]) -> bool:
        """Update node properties"""
        try:
            if node_id not in self.nodes:
                return False
            
            node = self.nodes[node_id]
            node.properties.update(properties)
            node.updated_at = datetime.now(timezone.utc)
            
            # Update graph
            self.graph.nodes[node_id].update(properties)
            
            # Update database
            cursor = self.db_connection.cursor()
            cursor.execute("""
                UPDATE nodes SET properties = ?, updated_at = ? WHERE id = ?
            """, (
                json.dumps(node.properties),
                node.updated_at.isoformat(),
                node_id
            ))
            
            self.db_connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to update node {node_id}: {e}")
            return False
    
    def find_nodes(self, node_type: Optional[str] = None, 
                  properties: Optional[Dict[str, Any]] = None,
                  limit: int = 100) -> List[GraphNode]:
        """Find nodes by type and/or properties"""
        try:
            results = []
            
            for node in self.nodes.values():
                # Filter by type
                if node_type and node.node_type != node_type:
                    continue
                
                # Filter by properties
                if properties:
                    match = True
                    for key, value in properties.items():
                        if key not in node.properties or node.properties[key] != value:
                            match = False
                            break
                    if not match:
                        continue
                
                results.append(node)
                
                if len(results) >= limit:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to find nodes: {e}")
            return []
    
    def find_related_nodes(self, node_id: str, relation_type: Optional[str] = None,
                          max_hops: int = 1) -> List[Tuple[GraphNode, str, int]]:
        """Find nodes related to given node"""
        try:
            if node_id not in self.nodes:
                return []
            
            related = []
            visited = {node_id}
            queue = [(node_id, 0)]
            
            while queue:
                current_id, hops = queue.pop(0)
                
                if hops >= max_hops:
                    continue
                
                # Find connected nodes
                for edge in self.edges:
                    target_id = None
                    edge_relation = edge.relation_type
                    
                    if edge.source_id == current_id:
                        target_id = edge.target_id
                    elif edge.target_id == current_id:
                        target_id = edge.source_id
                    
                    if target_id and target_id not in visited:
                        if not relation_type or edge_relation == relation_type:
                            if target_id in self.nodes:
                                related.append((self.nodes[target_id], edge_relation, hops + 1))
                                visited.add(target_id)
                                queue.append((target_id, hops + 1))
            
            return related
            
        except Exception as e:
            logger.error(f"Failed to find related nodes: {e}")
            return []
    
    def search_by_embedding(self, query: str, limit: int = 10) -> List[Tuple[GraphNode, float]]:
        """Search nodes by semantic similarity"""
        try:
            query_embedding = self._generate_embedding(query)
            if query_embedding is None:
                return []
            
            similarities = []
            
            for node in self.nodes.values():
                if node.embedding is not None:
                    similarity = cosine_similarity([query_embedding], [node.embedding])[0][0]
                    similarities.append((node, float(similarity)))
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities[:limit]
            
        except Exception as e:
            logger.error(f"Failed to search by embedding: {e}")
            return []
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        try:
            stats = {
                "nodes": len(self.nodes),
                "edges": len(self.edges),
                "node_types": {},
                "relation_types": {},
                "connected_components": nx.number_connected_components(self.graph.to_undirected()),
                "average_degree": sum(dict(self.graph.degree()).values()) / len(self.nodes) if self.nodes else 0
            }
            
            # Count node types
            for node in self.nodes.values():
                stats["node_types"][node.node_type] = stats["node_types"].get(node.node_type, 0) + 1
            
            # Count relation types
            for edge in self.edges:
                stats["relation_types"][edge.relation_type] = stats["relation_types"].get(edge.relation_type, 0) + 1
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get graph statistics: {e}")
            return {}
    
    async def merge_similar_nodes(self, similarity_threshold: float = 0.95):
        """Merge nodes with high similarity"""
        try:
            merged_count = 0
            
            node_list = list(self.nodes.values())
            
            for i, node1 in enumerate(node_list):
                if node1.embedding is None:
                    continue
                
                for j, node2 in enumerate(node_list[i+1:], i+1):
                    if node2.embedding is None or node1.id not in self.nodes or node2.id not in self.nodes:
                        continue
                    
                    # Calculate similarity
                    similarity = cosine_similarity([node1.embedding], [node2.embedding])[0][0]
                    
                    if similarity > similarity_threshold:
                        # Merge node2 into node1
                        await self._merge_nodes(node1.id, node2.id)
                        merged_count += 1
            
            logger.info(f"Merged {merged_count} similar nodes")
            
        except Exception as e:
            logger.error(f"Failed to merge similar nodes: {e}")
    
    async def _merge_nodes(self, keep_id: str, remove_id: str):
        """Merge two nodes, keeping the first and removing the second"""
        try:
            if keep_id not in self.nodes or remove_id not in self.nodes:
                return
            
            keep_node = self.nodes[keep_id]
            remove_node = self.nodes[remove_id]
            
            # Merge properties
            keep_node.properties.update(remove_node.properties)
            keep_node.updated_at = datetime.now(timezone.utc)
            
            # Redirect edges
            new_edges = []
            for edge in self.edges:
                if edge.source_id == remove_id:
                    edge.source_id = keep_id
                if edge.target_id == remove_id:
                    edge.target_id = keep_id
                    
                # Avoid self-loops
                if edge.source_id != edge.target_id:
                    new_edges.append(edge)
            
            self.edges = new_edges
            
            # Remove from memory and database
            del self.nodes[remove_id]
            self.graph.remove_node(remove_id)
            
            cursor = self.db_connection.cursor()
            cursor.execute("DELETE FROM nodes WHERE id = ?", (remove_id,))
            cursor.execute("UPDATE edges SET source_id = ? WHERE source_id = ?", (keep_id, remove_id))
            cursor.execute("UPDATE edges SET target_id = ? WHERE target_id = ?", (keep_id, remove_id))
            self.db_connection.commit()
            
        except Exception as e:
            logger.error(f"Failed to merge nodes: {e}")
    
    def export_graph(self, format: str = "json") -> Union[str, Dict]:
        """Export knowledge graph"""
        try:
            if format == "json":
                return {
                    "nodes": [
                        {
                            "id": node.id,
                            "label": node.label,
                            "type": node.node_type,
                            "properties": node.properties,
                            "confidence": node.confidence
                        }
                        for node in self.nodes.values()
                    ],
                    "edges": [
                        {
                            "source": edge.source_id,
                            "target": edge.target_id,
                            "relation": edge.relation_type,
                            "weight": edge.weight,
                            "properties": edge.properties
                        }
                        for edge in self.edges
                    ]
                }
            elif format == "networkx":
                return self.graph
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
        except Exception as e:
            logger.error(f"Failed to export graph: {e}")
            return {} if format == "json" else nx.Graph()
    
    def __del__(self):
        """Cleanup database connection"""
        try:
            if self.db_connection:
                self.db_connection.close()
        except:
            pass