#!/usr/bin/env python3
"""
Knowledge graph operations
Extracted from knowledge_manager.py for modularity
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
import numpy as np
from neo4j import AsyncGraphDatabase
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

from .models import Entity, Relationship, KnowledgeQuery, EntityType, RelationType

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """Core knowledge graph operations"""
    
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.driver = AsyncGraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.graph_cache = nx.DiGraph()
        self.last_cache_update = None
        
    async def close(self):
        """Close database connection"""
        await self.driver.close()
        
    async def create_entity(self, entity: Entity) -> str:
        """Create a new entity in the knowledge graph"""
        async with self.driver.session() as session:
            query = """
            CREATE (e:Entity {
                id: $id,
                name: $name,
                type: $type,
                description: $description,
                properties: $properties,
                embedding: $embedding,
                created_at: $created_at,
                updated_at: $updated_at,
                confidence: $confidence,
                source: $source
            })
            RETURN e.id
            """
            
            result = await session.run(query, {
                "id": entity.id,
                "name": entity.name,
                "type": entity.entity_type.value,
                "description": entity.description,
                "properties": entity.properties,
                "embedding": entity.embedding,
                "created_at": entity.created_at.isoformat(),
                "updated_at": entity.updated_at.isoformat(),
                "confidence": entity.confidence,
                "source": entity.source
            })
            
            record = await result.single()
            await self._invalidate_cache()
            return record["e.id"]
            
    async def create_relationship(self, relationship: Relationship) -> str:
        """Create a new relationship in the knowledge graph"""
        async with self.driver.session() as session:
            query = """
            MATCH (a:Entity {id: $source_id})
            MATCH (b:Entity {id: $target_id})
            CREATE (a)-[r:RELATES {
                id: $id,
                type: $type,
                properties: $properties,
                weight: $weight,
                confidence: $confidence,
                created_at: $created_at
            }]->(b)
            RETURN r.id
            """
            
            result = await session.run(query, {
                "source_id": relationship.source_id,
                "target_id": relationship.target_id,
                "id": relationship.id,
                "type": relationship.relation_type.value,
                "properties": relationship.properties,
                "weight": relationship.weight,
                "confidence": relationship.confidence,
                "created_at": relationship.created_at.isoformat()
            })
            
            record = await result.single()
            await self._invalidate_cache()
            return record["r.id"]
            
    async def search_entities(self, query: KnowledgeQuery) -> List[Entity]:
        """Search for entities based on query"""
        async with self.driver.session() as session:
            cypher_query = """
            MATCH (e:Entity)
            WHERE e.name CONTAINS $query_text
            OR e.description CONTAINS $query_text
            """
            
            if query.entity_types:
                type_filter = " OR ".join([f"e.type = '{t.value}'" for t in query.entity_types])
                cypher_query += f" AND ({type_filter})"
                
            if query.min_confidence > 0:
                cypher_query += f" AND e.confidence >= {query.min_confidence}"
                
            cypher_query += f" RETURN e LIMIT {query.limit}"
            
            result = await session.run(cypher_query, {"query_text": query.query_text})
            
            entities = []
            async for record in result:
                entity_data = record["e"]
                entity = Entity(
                    id=entity_data["id"],
                    name=entity_data["name"],
                    entity_type=EntityType(entity_data["type"]),
                    description=entity_data.get("description", ""),
                    properties=entity_data.get("properties", {}),
                    embedding=entity_data.get("embedding"),
                    confidence=entity_data.get("confidence", 1.0),
                    source=entity_data.get("source", "unknown")
                )
                entities.append(entity)
                
            return entities
            
    async def find_related_entities(self, entity_id: str, max_depth: int = 2) -> List[Tuple[Entity, float]]:
        """Find entities related to the given entity"""
        async with self.driver.session() as session:
            query = """
            MATCH (start:Entity {id: $entity_id})
            MATCH (start)-[r*1..$max_depth]-(related:Entity)
            WHERE related.id <> start.id
            RETURN DISTINCT related, reduce(weight = 1.0, rel in r | weight * rel.weight) as path_weight
            ORDER BY path_weight DESC
            LIMIT 20
            """
            
            result = await session.run(query, {
                "entity_id": entity_id,
                "max_depth": max_depth
            })
            
            related_entities = []
            async for record in result:
                entity_data = record["related"]
                weight = record["path_weight"]
                
                entity = Entity(
                    id=entity_data["id"],
                    name=entity_data["name"],
                    entity_type=EntityType(entity_data["type"]),
                    description=entity_data.get("description", ""),
                    properties=entity_data.get("properties", {}),
                    confidence=entity_data.get("confidence", 1.0)
                )
                
                related_entities.append((entity, weight))
                
            return related_entities
            
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        async with self.driver.session() as session:
            # Count entities by type
            entity_query = """
            MATCH (e:Entity)
            RETURN e.type as type, count(e) as count
            """
            
            # Count relationships by type
            rel_query = """
            MATCH ()-[r:RELATES]->()
            RETURN r.type as type, count(r) as count
            """
            
            entity_result = await session.run(entity_query)
            entity_counts = {}
            total_entities = 0
            
            async for record in entity_result:
                entity_type = record["type"]
                count = record["count"]
                entity_counts[entity_type] = count
                total_entities += count
                
            rel_result = await session.run(rel_query)
            rel_counts = {}
            total_relationships = 0
            
            async for record in rel_result:
                rel_type = record["type"]
                count = record["count"]
                rel_counts[rel_type] = count
                total_relationships += count
                
            return {
                "total_entities": total_entities,
                "total_relationships": total_relationships,
                "entity_type_counts": entity_counts,
                "relation_type_counts": rel_counts,
                "last_updated": datetime.now()
            }
            
    async def _invalidate_cache(self):
        """Invalidate the NetworkX cache"""
        self.graph_cache.clear()
        self.last_cache_update = None
        
    async def _update_cache(self):
        """Update the NetworkX cache from Neo4j"""
        async with self.driver.session() as session:
            # Get all entities
            entity_query = "MATCH (e:Entity) RETURN e"
            entity_result = await session.run(entity_query)
            
            async for record in entity_result:
                entity = record["e"]
                self.graph_cache.add_node(
                    entity["id"],
                    name=entity["name"],
                    type=entity["type"]
                )
                
            # Get all relationships
            rel_query = "MATCH (a:Entity)-[r:RELATES]->(b:Entity) RETURN a.id, b.id, r"
            rel_result = await session.run(rel_query)
            
            async for record in rel_result:
                source_id = record["a.id"]
                target_id = record["b.id"]
                rel = record["r"]
                
                self.graph_cache.add_edge(
                    source_id,
                    target_id,
                    weight=rel.get("weight", 1.0),
                    type=rel.get("type", "relates_to")
                )
                
            self.last_cache_update = datetime.now()
            
    async def analyze_graph_structure(self) -> Dict[str, Any]:
        """Analyze the structure of the knowledge graph"""
        if not self.last_cache_update or \
           (datetime.now() - self.last_cache_update).seconds > 3600:
            await self._update_cache()
            
        analysis = {
            "node_count": self.graph_cache.number_of_nodes(),
            "edge_count": self.graph_cache.number_of_edges(),
            "density": nx.density(self.graph_cache),
            "connected_components": nx.number_weakly_connected_components(self.graph_cache)
        }
        
        if self.graph_cache.number_of_nodes() > 0:
            try:
                analysis["average_clustering"] = nx.average_clustering(self.graph_cache.to_undirected())
                analysis["diameter"] = nx.diameter(self.graph_cache.to_undirected()) if nx.is_connected(self.graph_cache.to_undirected()) else "N/A"
            except:
                analysis["average_clustering"] = "N/A"
                analysis["diameter"] = "N/A"
                
        return analysis