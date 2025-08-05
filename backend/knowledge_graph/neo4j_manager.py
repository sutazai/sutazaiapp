"""
Neo4j Knowledge Graph Manager
============================

Manages the Neo4j graph database for the SutazAI knowledge graph system.
Provides high-level operations for storing, querying, and maintaining
the knowledge graph with optimized performance and reliability.

Features:
- Connection management and health monitoring
- Bulk import/export operations
- Index and constraint management
- Transaction handling
- Query optimization
- Backup and recovery
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from neo4j import GraphDatabase, Record
from neo4j.exceptions import ServiceUnavailable, TransientError

from .schema import (
    NodeType, RelationshipType, NodeProperties, RelationshipProperties,
    KnowledgeGraphSchema, AgentNode, ServiceNode, DatabaseNode
)


class Neo4jManager:
    """
    Neo4j database manager for the SutazAI knowledge graph.
    
    Handles all Neo4j operations including connection management,
    schema creation, data operations, and query execution.
    """
    
    def __init__(self, uri: str = "bolt://localhost:7687", 
                 username: str = "neo4j", 
                 password: str = "password",
                 database: str = "sutazai"):
        
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        
        self.driver = None
        self.schema = KnowledgeGraphSchema()
        self.logger = logging.getLogger("neo4j_manager")
        
        # Performance settings
        self.batch_size = 1000
        self.max_retries = 3
        self.retry_delay = 1.0
        
        # Statistics
        self.stats = {
            "nodes_created": 0,
            "relationships_created": 0,
            "queries_executed": 0,
            "errors": 0,
            "avg_query_time": 0.0
        }
    
    async def initialize(self) -> bool:
        """Initialize the Neo4j connection and schema"""
        try:
            self.logger.info("Initializing Neo4j connection")
            
            # Create driver
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_lifetime=300,
                max_connection_pool_size=50,
                connection_acquisition_timeout=120
            )
            
            # Verify connectivity
            await self._verify_connectivity()
            
            # Initialize schema
            await self._initialize_schema()
            
            self.logger.info("Neo4j manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Neo4j manager: {e}")
            return False
    
    async def _verify_connectivity(self):
        """Verify Neo4j database connectivity"""
        with self.driver.session(database=self.database) as session:
            result = session.run("RETURN 1 as test")
            record = result.single()
            if record["test"] != 1:
                raise ServiceUnavailable("Neo4j connectivity test failed")
    
    async def _initialize_schema(self):
        """Initialize database schema with constraints and indexes"""
        cypher_schema = self.schema.get_cypher_schema()
        statements = cypher_schema.split("\n")
        
        with self.driver.session(database=self.database) as session:
            for statement in statements:
                if statement.strip():
                    try:
                        session.run(statement)
                        self.logger.debug(f"Executed schema statement: {statement[:50]}...")
                    except Exception as e:
                        # Some constraints might already exist
                        self.logger.debug(f"Schema statement warning: {e}")
    
    async def create_node(self, node: NodeProperties) -> bool:
        """Create a single node in the graph"""
        try:
            # Convert node to properties
            props = self._node_to_properties(node)
            
            # Create Cypher query
            label = node.type.value.title()
            cypher = f"""
            CREATE (n:{label} $props)
            RETURN n.id as id
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher, props=props)
                record = result.single()
                
                if record:
                    self.stats["nodes_created"] += 1
                    self.logger.debug(f"Created node {record['id']}")
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to create node: {e}")
            self.stats["errors"] += 1
            return False
    
    async def create_nodes_batch(self, nodes: List[NodeProperties]) -> int:
        """Create multiple nodes in batches for better performance"""
        created_count = 0
        
        try:
            # Group nodes by type for optimized batch processing
            nodes_by_type = {}
            for node in nodes:
                node_type = node.type.value
                if node_type not in nodes_by_type:
                    nodes_by_type[node_type] = []
                nodes_by_type[node_type].append(node)
            
            # Process each type in batches
            with self.driver.session(database=self.database) as session:
                for node_type, type_nodes in nodes_by_type.items():
                    label = node_type.title()
                    
                    # Process in batches
                    for i in range(0, len(type_nodes), self.batch_size):
                        batch = type_nodes[i:i + self.batch_size]
                        
                        # Convert nodes to properties
                        props_list = [self._node_to_properties(node) for node in batch]
                        
                        cypher = f"""
                        UNWIND $nodes as nodeData
                        CREATE (n:{label})
                        SET n = nodeData
                        RETURN count(n) as created
                        """
                        
                        result = session.run(cypher, nodes=props_list)
                        record = result.single()
                        
                        if record:
                            batch_created = record["created"]
                            created_count += batch_created
                            self.stats["nodes_created"] += batch_created
                            
                            self.logger.debug(f"Created batch of {batch_created} {label} nodes")
            
            return created_count
            
        except Exception as e:
            self.logger.error(f"Failed to create nodes batch: {e}")
            self.stats["errors"] += 1
            return created_count
    
    async def create_relationship(self, relationship: RelationshipProperties) -> bool:
        """Create a single relationship in the graph"""
        try:
            props = self._relationship_to_properties(relationship)
            rel_type = relationship.type.value.upper()
            
            cypher = f"""
            MATCH (a {{id: $source_id}})
            MATCH (b {{id: $target_id}})
            CREATE (a)-[r:{rel_type} $props]->(b)
            RETURN r.id as id
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher, 
                                   source_id=relationship.source_id,
                                   target_id=relationship.target_id,
                                   props=props)
                record = result.single()
                
                if record:
                    self.stats["relationships_created"] += 1
                    self.logger.debug(f"Created relationship {record['id']}")
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to create relationship: {e}")
            self.stats["errors"] += 1
            return False
    
    async def create_relationships_batch(self, relationships: List[RelationshipProperties]) -> int:
        """Create multiple relationships in batches"""
        created_count = 0
        
        try:
            # Group relationships by type
            rels_by_type = {}
            for rel in relationships:
                rel_type = rel.type.value
                if rel_type not in rels_by_type:
                    rels_by_type[rel_type] = []
                rels_by_type[rel_type].append(rel)
            
            with self.driver.session(database=self.database) as session:
                for rel_type, type_rels in rels_by_type.items():
                    rel_label = rel_type.upper()
                    
                    # Process in batches
                    for i in range(0, len(type_rels), self.batch_size):
                        batch = type_rels[i:i + self.batch_size]
                        
                        # Prepare relationship data
                        rel_data = []
                        for rel in batch:
                            rel_data.append({
                                "source_id": rel.source_id,
                                "target_id": rel.target_id,
                                "props": self._relationship_to_properties(rel)
                            })
                        
                        cypher = f"""
                        UNWIND $rels as relData
                        MATCH (a {{id: relData.source_id}})
                        MATCH (b {{id: relData.target_id}})
                        CREATE (a)-[r:{rel_label}]->(b)
                        SET r = relData.props
                        RETURN count(r) as created
                        """
                        
                        result = session.run(cypher, rels=rel_data)
                        record = result.single()
                        
                        if record:
                            batch_created = record["created"]
                            created_count += batch_created
                            self.stats["relationships_created"] += batch_created
                            
                            self.logger.debug(f"Created batch of {batch_created} {rel_label} relationships")
            
            return created_count
            
        except Exception as e:
            self.logger.error(f"Failed to create relationships batch: {e}")
            self.stats["errors"] += 1
            return created_count
    
    async def find_nodes(self, node_type: Optional[NodeType] = None,
                        filters: Optional[Dict[str, Any]] = None,
                        limit: int = 100) -> List[Dict[str, Any]]:
        """Find nodes matching criteria"""
        try:
            # Build query
            where_clauses = []
            params = {}
            
            if filters:
                for key, value in filters.items():
                    param_name = f"param_{key}"
                    where_clauses.append(f"n.{key} = ${param_name}")
                    params[param_name] = value
            
            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            if node_type:
                label = node_type.value.title()
                cypher = f"""
                MATCH (n:{label})
                WHERE {where_clause}
                RETURN n
                LIMIT {limit}
                """
            else:
                cypher = f"""
                MATCH (n)
                WHERE {where_clause}
                RETURN n
                LIMIT {limit}
                """
            
            start_time = time.time()
            
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher, **params)
                records = result.data()
                
                self.stats["queries_executed"] += 1
                query_time = time.time() - start_time
                self._update_avg_query_time(query_time)
                
                return [record["n"] for record in records]
            
        except Exception as e:
            self.logger.error(f"Failed to find nodes: {e}")
            self.stats["errors"] += 1
            return []
    
    async def find_relationships(self, source_id: Optional[str] = None,
                               target_id: Optional[str] = None,
                               rel_type: Optional[RelationshipType] = None,
                               limit: int = 100) -> List[Dict[str, Any]]:
        """Find relationships matching criteria"""
        try:
            where_clauses = []
            params = {}
            
            if source_id:
                where_clauses.append("a.id = $source_id")
                params["source_id"] = source_id
            
            if target_id:
                where_clauses.append("b.id = $target_id")
                params["target_id"] = target_id
            
            where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
            
            if rel_type:
                rel_label = rel_type.value.upper()
                cypher = f"""
                MATCH (a)-[r:{rel_label}]->(b)
                WHERE {where_clause}
                RETURN a, r, b
                LIMIT {limit}
                """
            else:
                cypher = f"""
                MATCH (a)-[r]->(b)
                WHERE {where_clause}
                RETURN a, r, b
                LIMIT {limit}
                """
            
            start_time = time.time()
            
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher, **params)
                records = result.data()
                
                self.stats["queries_executed"] += 1
                query_time = time.time() - start_time
                self._update_avg_query_time(query_time)
                
                return records
            
        except Exception as e:
            self.logger.error(f"Failed to find relationships: {e}")
            self.stats["errors"] += 1
            return []
    
    async def execute_cypher(self, cypher: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute custom Cypher query"""
        try:
            start_time = time.time()
            
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher, parameters or {})
                records = result.data()
                
                self.stats["queries_executed"] += 1
                query_time = time.time() - start_time
                self._update_avg_query_time(query_time)
                
                return records
            
        except Exception as e:
            self.logger.error(f"Failed to execute Cypher query: {e}")
            self.stats["errors"] += 1
            return []
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get comprehensive graph statistics"""
        try:
            stats_queries = {
                "total_nodes": "MATCH (n) RETURN count(n) as count",
                "total_relationships": "MATCH ()-[r]->() RETURN count(r) as count",
                "node_counts_by_type": """
                    MATCH (n) 
                    RETURN labels(n)[0] as node_type, count(n) as count 
                    ORDER BY count DESC
                """,
                "relationship_counts_by_type": """
                    MATCH ()-[r]->() 
                    RETURN type(r) as rel_type, count(r) as count 
                    ORDER BY count DESC
                """,
                "degree_distribution": """
                    MATCH (n)
                    WITH n, size((n)-[]->()) + size((n)<-[]-()) as degree
                    RETURN degree, count(n) as node_count
                    ORDER BY degree
                """
            }
            
            graph_stats = {}
            
            with self.driver.session(database=self.database) as session:
                for stat_name, query in stats_queries.items():
                    result = session.run(query)
                    
                    if stat_name in ["total_nodes", "total_relationships"]:
                        record = result.single()
                        graph_stats[stat_name] = record["count"] if record else 0
                    else:
                        graph_stats[stat_name] = result.data()
            
            # Add manager statistics
            graph_stats["manager_stats"] = self.stats.copy()
            
            return graph_stats
            
        except Exception as e:
            self.logger.error(f"Failed to get graph statistics: {e}")
            return {"error": str(e)}
    
    async def clear_graph(self) -> bool:
        """Clear all nodes and relationships (use with caution!)"""
        try:
            with self.driver.session(database=self.database) as session:
                # Delete all relationships first
                session.run("MATCH ()-[r]->() DELETE r")
                
                # Then delete all nodes
                session.run("MATCH (n) DELETE n")
                
                self.logger.info("Graph cleared successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to clear graph: {e}")
            return False
    
    def _node_to_properties(self, node: NodeProperties) -> Dict[str, Any]:
        """Convert node object to Neo4j properties"""
        props = {
            "id": node.id,
            "name": node.name,
            "description": node.description,
            "type": node.type.value,
            "created_at": node.created_at.isoformat(),
            "updated_at": node.updated_at.isoformat(),
            "version": node.version,
            "status": node.status
        }
        
        # Add metadata
        if node.metadata:
            props["metadata"] = json.dumps(node.metadata)
        
        # Add tags
        if node.tags:
            props["tags"] = list(node.tags)
        
        # Add type-specific properties
        if isinstance(node, AgentNode):
            props.update({
                "agent_type": node.agent_type,
                "capabilities": list(node.capabilities),
                "max_concurrent_tasks": node.max_concurrent_tasks,
                "health_status": node.health_status
            })
            if node.model_config:
                props["model_config"] = json.dumps(node.model_config)
            if node.performance_metrics:
                props["performance_metrics"] = json.dumps(node.performance_metrics)
        
        elif isinstance(node, ServiceNode):
            props.update({
                "service_type": node.service_type,
                "endpoints": node.endpoints,
                "health_check_url": node.health_check_url
            })
            if node.port:
                props["port"] = node.port
        
        elif isinstance(node, DatabaseNode):
            props.update({
                "database_type": node.database_type,
                "connection_string": node.connection_string,
                "tables_collections": node.tables_collections,
                "indexes": node.indexes
            })
        
        return props
    
    def _relationship_to_properties(self, relationship: RelationshipProperties) -> Dict[str, Any]:
        """Convert relationship object to Neo4j properties"""
        props = {
            "id": relationship.id,
            "weight": relationship.weight,
            "confidence": relationship.confidence,
            "created_at": relationship.created_at.isoformat()
        }
        
        if relationship.metadata:
            props["metadata"] = json.dumps(relationship.metadata)
        
        if relationship.properties:
            props.update(relationship.properties)
        
        return props
    
    def _update_avg_query_time(self, query_time: float):
        """Update average query time statistic"""
        current_avg = self.stats["avg_query_time"]
        query_count = self.stats["queries_executed"]
        
        # Calculate running average
        self.stats["avg_query_time"] = ((current_avg * (query_count - 1)) + query_time) / query_count
    
    async def shutdown(self):
        """Shutdown the Neo4j manager"""
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j manager shutdown complete")


# Global manager instance
_neo4j_manager: Optional[Neo4jManager] = None


def get_neo4j_manager() -> Optional[Neo4jManager]:
    """Get the global Neo4j manager instance"""
    return _neo4j_manager


def set_neo4j_manager(manager: Neo4jManager):
    """Set the global Neo4j manager instance"""
    global _neo4j_manager
    _neo4j_manager = manager