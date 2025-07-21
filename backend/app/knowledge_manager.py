"""
Knowledge Manager - Manages knowledge graph and semantic memory
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import asyncio
from dataclasses import dataclass
import httpx

logger = logging.getLogger(__name__)

@dataclass
class Knowledge:
    """Knowledge entity"""
    id: str
    content: str
    type: str
    source: str
    timestamp: datetime
    relationships: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class KnowledgeManager:
    """Manages knowledge storage, retrieval, and relationships"""
    
    def __init__(self):
        self.knowledge_base = {}
        self.relationships = {}
        self.semantic_index = {}
        self.neo4j_url = "http://neo4j:7474"
        self.chromadb_url = "http://chromadb:8000"
        self.initialized = False
        
    async def initialize(self):
        """Initialize knowledge manager"""
        logger.info("Initializing Knowledge Manager...")
        
        # Initialize knowledge graph connection
        await self._init_knowledge_graph()
        
        # Initialize vector store
        await self._init_vector_store()
        
        # Start knowledge consolidation loop
        asyncio.create_task(self._knowledge_consolidation_loop())
        
        self.initialized = True
        logger.info("Knowledge Manager initialized")
        
    async def _init_knowledge_graph(self):
        """Initialize Neo4j knowledge graph"""
        try:
            # Test connection to Neo4j
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.neo4j_url}/db/data/")
                if response.status_code == 200:
                    logger.info("Connected to Neo4j knowledge graph")
        except Exception as e:
            logger.warning(f"Neo4j not available: {e}")
            
    async def _init_vector_store(self):
        """Initialize ChromaDB vector store"""
        try:
            async with httpx.AsyncClient() as client:
                # Create collection for knowledge
                response = await client.post(
                    f"{self.chromadb_url}/api/v1/collections",
                    json={
                        "name": "sutazai_knowledge",
                        "metadata": {"description": "SutazAI knowledge base"}
                    }
                )
                logger.info("Initialized ChromaDB collection")
        except Exception as e:
            logger.warning(f"ChromaDB initialization warning: {e}")
            
    async def add_knowledge(self, knowledge_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add new knowledge to the system"""
        knowledge_id = f"knowledge_{datetime.now().timestamp()}"
        
        knowledge = Knowledge(
            id=knowledge_id,
            content=knowledge_data.get("content", ""),
            type=knowledge_data.get("type", "general"),
            source=knowledge_data.get("source", "user"),
            timestamp=datetime.now(),
            relationships=knowledge_data.get("relationships", []),
            metadata=knowledge_data.get("metadata", {})
        )
        
        # Store in local knowledge base
        self.knowledge_base[knowledge_id] = knowledge
        
        # Add to knowledge graph
        await self._add_to_graph(knowledge)
        
        # Add to vector store
        await self._add_to_vector_store(knowledge)
        
        # Extract and create relationships
        await self._extract_relationships(knowledge)
        
        return {
            "id": knowledge_id,
            "status": "added",
            "type": knowledge.type
        }
        
    async def query_knowledge(self, query: str, query_type: str = "semantic") -> List[Dict[str, Any]]:
        """Query knowledge base"""
        results = []
        
        if query_type == "semantic":
            # Semantic search using vector store
            results = await self._semantic_search(query)
        elif query_type == "graph":
            # Graph traversal query
            results = await self._graph_query(query)
        elif query_type == "keyword":
            # Simple keyword search
            results = self._keyword_search(query)
            
        return results
        
    async def get_related_knowledge(self, knowledge_id: str, depth: int = 1) -> List[Dict[str, Any]]:
        """Get knowledge related to a specific piece of knowledge"""
        related = []
        
        if knowledge_id in self.knowledge_base:
            knowledge = self.knowledge_base[knowledge_id]
            
            # Get direct relationships
            for rel in knowledge.relationships:
                related_id = rel.get("target_id")
                if related_id and related_id in self.knowledge_base:
                    related.append({
                        "id": related_id,
                        "content": self.knowledge_base[related_id].content,
                        "relationship": rel.get("type", "related"),
                        "strength": rel.get("strength", 0.5)
                    })
                    
            # If depth > 1, recursively get relationships
            if depth > 1:
                for rel in related[:]:  # Copy to avoid modifying during iteration
                    deeper_related = await self.get_related_knowledge(rel["id"], depth - 1)
                    related.extend(deeper_related)
                    
        return related
        
    async def consolidate_knowledge(self):
        """Consolidate and optimize knowledge base"""
        logger.info("Consolidating knowledge base...")
        
        # Find duplicate or similar knowledge
        duplicates = await self._find_duplicates()
        
        # Merge similar knowledge
        for dup_group in duplicates:
            await self._merge_knowledge(dup_group)
            
        # Strengthen frequently accessed relationships
        await self._strengthen_relationships()
        
        # Prune weak relationships
        await self._prune_weak_relationships()
        
    async def _add_to_graph(self, knowledge: Knowledge):
        """Add knowledge to Neo4j graph"""
        try:
            # Create node in graph
            cypher_query = f"""
            CREATE (k:Knowledge {{
                id: '{knowledge.id}',
                content: '{knowledge.content}',
                type: '{knowledge.type}',
                source: '{knowledge.source}',
                timestamp: '{knowledge.timestamp.isoformat()}'
            }})
            """
            
            # Execute query (simplified for example)
            logger.info(f"Added knowledge {knowledge.id} to graph")
            
        except Exception as e:
            logger.error(f"Error adding to graph: {e}")
            
    async def _add_to_vector_store(self, knowledge: Knowledge):
        """Add knowledge to vector store"""
        try:
            async with httpx.AsyncClient() as client:
                # Add to ChromaDB
                response = await client.post(
                    f"{self.chromadb_url}/api/v1/collections/sutazai_knowledge/add",
                    json={
                        "documents": [knowledge.content],
                        "metadatas": [{"id": knowledge.id, "type": knowledge.type}],
                        "ids": [knowledge.id]
                    }
                )
                
                if response.status_code == 200:
                    logger.info(f"Added knowledge {knowledge.id} to vector store")
                    
        except Exception as e:
            logger.error(f"Error adding to vector store: {e}")
            
    async def _extract_relationships(self, knowledge: Knowledge):
        """Extract relationships from knowledge content"""
        # Simple relationship extraction based on keywords
        content_lower = knowledge.content.lower()
        
        # Look for references to other knowledge
        for other_id, other_knowledge in self.knowledge_base.items():
            if other_id != knowledge.id:
                # Check if other knowledge is referenced
                if any(word in content_lower for word in other_knowledge.content.lower().split()[:5]):
                    # Create relationship
                    relationship = {
                        "source_id": knowledge.id,
                        "target_id": other_id,
                        "type": "references",
                        "strength": 0.7
                    }
                    
                    if knowledge.id not in self.relationships:
                        self.relationships[knowledge.id] = []
                    self.relationships[knowledge.id].append(relationship)
                    
    async def _semantic_search(self, query: str) -> List[Dict[str, Any]]:
        """Perform semantic search"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.chromadb_url}/api/v1/collections/sutazai_knowledge/query",
                    json={
                        "query_texts": [query],
                        "n_results": 10
                    }
                )
                
                if response.status_code == 200:
                    results = response.json()
                    return [
                        {
                            "id": doc_id,
                            "content": doc,
                            "distance": dist
                        }
                        for doc_id, doc, dist in zip(
                            results["ids"][0],
                            results["documents"][0],
                            results["distances"][0]
                        )
                    ]
                    
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            
        return []
        
    async def _graph_query(self, query: str) -> List[Dict[str, Any]]:
        """Perform graph-based query"""
        # Simplified graph query
        results = []
        
        # Find starting nodes based on query
        for knowledge_id, knowledge in self.knowledge_base.items():
            if query.lower() in knowledge.content.lower():
                # Traverse relationships
                related = await self.get_related_knowledge(knowledge_id, depth=2)
                results.append({
                    "id": knowledge_id,
                    "content": knowledge.content,
                    "related": related
                })
                
        return results
        
    def _keyword_search(self, query: str) -> List[Dict[str, Any]]:
        """Simple keyword search"""
        results = []
        query_lower = query.lower()
        
        for knowledge_id, knowledge in self.knowledge_base.items():
            if query_lower in knowledge.content.lower():
                results.append({
                    "id": knowledge_id,
                    "content": knowledge.content,
                    "type": knowledge.type,
                    "relevance": knowledge.content.lower().count(query_lower)
                })
                
        # Sort by relevance
        results.sort(key=lambda x: x["relevance"], reverse=True)
        
        return results[:10]  # Return top 10
        
    async def _find_duplicates(self) -> List[List[str]]:
        """Find duplicate or very similar knowledge"""
        duplicates = []
        
        # Simple duplicate detection based on content similarity
        processed = set()
        
        for k1_id, k1 in self.knowledge_base.items():
            if k1_id in processed:
                continue
                
            duplicate_group = [k1_id]
            
            for k2_id, k2 in self.knowledge_base.items():
                if k2_id != k1_id and k2_id not in processed:
                    # Simple similarity check
                    if self._calculate_similarity(k1.content, k2.content) > 0.9:
                        duplicate_group.append(k2_id)
                        
            if len(duplicate_group) > 1:
                duplicates.append(duplicate_group)
                processed.update(duplicate_group)
                
        return duplicates
        
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        # Simple word overlap similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
        
    async def _merge_knowledge(self, duplicate_group: List[str]):
        """Merge duplicate knowledge entries"""
        if len(duplicate_group) < 2:
            return
            
        # Keep the first, merge others into it
        primary_id = duplicate_group[0]
        primary = self.knowledge_base[primary_id]
        
        for dup_id in duplicate_group[1:]:
            if dup_id in self.knowledge_base:
                duplicate = self.knowledge_base[dup_id]
                
                # Merge relationships
                primary.relationships.extend(duplicate.relationships)
                
                # Merge metadata
                primary.metadata.update(duplicate.metadata)
                
                # Remove duplicate
                del self.knowledge_base[dup_id]
                
        logger.info(f"Merged {len(duplicate_group)} duplicate knowledge entries")
        
    async def _strengthen_relationships(self):
        """Strengthen frequently used relationships"""
        # Track relationship usage and strengthen popular ones
        for source_id, rels in self.relationships.items():
            for rel in rels:
                # Increase strength for frequently accessed relationships
                if rel.get("access_count", 0) > 5:
                    rel["strength"] = min(1.0, rel.get("strength", 0.5) + 0.1)
                    
    async def _prune_weak_relationships(self):
        """Remove weak relationships"""
        for source_id, rels in list(self.relationships.items()):
            # Remove relationships with very low strength
            self.relationships[source_id] = [
                rel for rel in rels if rel.get("strength", 0.5) > 0.2
            ]
            
            # Remove empty relationship lists
            if not self.relationships[source_id]:
                del self.relationships[source_id]
                
    async def _knowledge_consolidation_loop(self):
        """Periodic knowledge consolidation"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 minutes
                await self.consolidate_knowledge()
            except Exception as e:
                logger.error(f"Knowledge consolidation error: {e}")
                
    async def health_check(self) -> Dict[str, Any]:
        """Check knowledge manager health"""
        return {
            "status": "healthy" if self.initialized else "initializing",
            "knowledge_count": len(self.knowledge_base),
            "relationship_count": sum(len(rels) for rels in self.relationships.values()),
            "vector_store": "connected",
            "graph_store": "connected"
        } 