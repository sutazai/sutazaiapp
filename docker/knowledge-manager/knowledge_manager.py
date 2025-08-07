#!/usr/bin/env python3
"""
Knowledge Manager for SutazAI automation/advanced automation System
Manages knowledge graphs, semantic relationships, and information retrieval
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import numpy as np
from neo4j import AsyncGraphDatabase
import aiohttp
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from elasticsearch import AsyncElasticsearch
import hashlib
from collections import defaultdict
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SutazAI Knowledge Manager", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EntityType(Enum):
    CONCEPT = "concept"
    PERSON = "person"
    PLACE = "place"
    EVENT = "event"
    OBJECT = "object"
    PROCESS = "process"
    ABSTRACT = "abstract"
    TEMPORAL = "temporal"
    RELATIONSHIP = "relationship"

class RelationType(Enum):
    IS_A = "is_a"
    PART_OF = "part_of"
    RELATED_TO = "related_to"
    CAUSES = "causes"
    PRECEDED_BY = "preceded_by"
    SIMILAR_TO = "similar_to"
    OPPOSITE_OF = "opposite_of"
    INSTANCE_OF = "instance_of"
    PROPERTY_OF = "property_of"
    LOCATED_IN = "located_in"

@dataclass
class Entity:
    """Knowledge entity with properties and metadata"""
    id: str
    name: str
    entity_type: EntityType
    properties: Dict[str, Any] = field(default_factory=dict)
    embeddings: Optional[np.ndarray] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    confidence: float = 1.0
    source: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "confidence": self.confidence,
            "source": self.source
        }

@dataclass
class Relationship:
    """Relationship between entities"""
    source_id: str
    target_id: str
    relation_type: RelationType
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)

class KnowledgeGraph:
    """Advanced knowledge graph with Neo4j backend"""
    
    def __init__(self, uri: str, user: str, password: str):
        self.driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self.nlp = spacy.load("en_core_web_sm")
        self.embedding_model = None  # Will be initialized
        
    async def close(self):
        await self.driver.close()
        
    async def create_entity(self, entity: Entity) -> str:
        """Create an entity in the knowledge graph"""
        async with self.driver.session() as session:
            query = """
            CREATE (e:Entity {
                id: $id,
                name: $name,
                entity_type: $entity_type,
                properties: $properties,
                created_at: datetime($created_at),
                updated_at: datetime($updated_at),
                confidence: $confidence,
                source: $source
            })
            RETURN e.id as id
            """
            
            result = await session.run(
                query,
                id=entity.id,
                name=entity.name,
                entity_type=entity.entity_type.value,
                properties=json.dumps(entity.properties),
                created_at=entity.created_at.isoformat(),
                updated_at=entity.updated_at.isoformat(),
                confidence=entity.confidence,
                source=entity.source
            )
            
            record = await result.single()
            return record["id"]
    
    async def create_relationship(self, relationship: Relationship) -> bool:
        """Create a relationship between entities"""
        async with self.driver.session() as session:
            query = f"""
            MATCH (a:Entity {{id: $source_id}})
            MATCH (b:Entity {{id: $target_id}})
            CREATE (a)-[r:{relationship.relation_type.value.upper()} {{
                properties: $properties,
                confidence: $confidence,
                created_at: datetime($created_at)
            }}]->(b)
            RETURN r
            """
            
            result = await session.run(
                query,
                source_id=relationship.source_id,
                target_id=relationship.target_id,
                properties=json.dumps(relationship.properties),
                confidence=relationship.confidence,
                created_at=relationship.created_at.isoformat()
            )
            
            return await result.single() is not None
    
    async def query_path(self, start_id: str, end_id: str, 
                        max_depth: int = 5) -> List[Dict[str, Any]]:
        """Find paths between two entities"""
        async with self.driver.session() as session:
            query = """
            MATCH path = shortestPath((start:Entity {id: $start_id})-[*..%d]-(end:Entity {id: $end_id}))
            RETURN path
            """ % max_depth
            
            result = await session.run(
                query,
                start_id=start_id,
                end_id=end_id
            )
            
            paths = []
            async for record in result:
                path = record["path"]
                path_data = {
                    "nodes": [node["name"] for node in path.nodes],
                    "relationships": [rel.type for rel in path.relationships],
                    "length": len(path.relationships)
                }
                paths.append(path_data)
            
            return paths
    
    async def semantic_search(self, query: str, limit: int = 10) -> List[Entity]:
        """Search entities using semantic similarity"""
        # Extract entities from query
        doc = self.nlp(query)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        async with self.driver.session() as session:
            # First, try exact matches
            exact_query = """
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($query)
            RETURN e
            LIMIT $limit
            """
            
            result = await session.run(
                exact_query,
                query=query,
                limit=limit
            )
            
            entities = []
            async for record in result:
                node = record["e"]
                entity = Entity(
                    id=node["id"],
                    name=node["name"],
                    entity_type=EntityType(node["entity_type"]),
                    properties=json.loads(node["properties"]),
                    confidence=node["confidence"],
                    source=node["source"]
                )
                entities.append(entity)
            
            return entities
    
    async def get_subgraph(self, entity_id: str, depth: int = 2) -> Dict[str, Any]:
        """Get subgraph around an entity"""
        async with self.driver.session() as session:
            query = """
            MATCH (center:Entity {id: $entity_id})
            OPTIONAL MATCH path = (center)-[*..%d]-(connected)
            WITH center, collect(path) as paths
            RETURN center, paths
            """ % depth
            
            result = await session.run(query, entity_id=entity_id)
            record = await result.single()
            
            if not record:
                return None
            
            # Build subgraph representation
            subgraph = {
                "center": record["center"]["name"],
                "nodes": set(),
                "edges": []
            }
            
            for path in record["paths"]:
                if path:
                    for node in path.nodes:
                        subgraph["nodes"].add(node["name"])
                    
                    for i, rel in enumerate(path.relationships):
                        subgraph["edges"].append({
                            "source": path.nodes[i]["name"],
                            "target": path.nodes[i+1]["name"],
                            "type": rel.type
                        })
            
            subgraph["nodes"] = list(subgraph["nodes"])
            return subgraph

class SemanticMemory:
    """Semantic memory system with vector similarity search"""
    
    def __init__(self, elasticsearch_url: str):
        self.es = AsyncElasticsearch([elasticsearch_url])
        self.index_name = "sutazai_semantic_memory"
        
    async def initialize(self):
        """Initialize Elasticsearch index"""
        index_body = {
            "mappings": {
                "properties": {
                    "id": {"type": "keyword"},
                    "content": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 768,
                        "index": True,
                        "similarity": "cosine"
                    },
                    "metadata": {"type": "object"},
                    "timestamp": {"type": "date"},
                    "importance": {"type": "float"},
                    "access_count": {"type": "integer"}
                }
            }
        }
        
        if not await self.es.indices.exists(index=self.index_name):
            await self.es.indices.create(index=self.index_name, body=index_body)
    
    async def store_memory(self, content: str, embedding: List[float], 
                          metadata: Dict[str, Any] = None) -> str:
        """Store a memory with its embedding"""
        memory_id = hashlib.sha256(content.encode()).hexdigest()
        
        doc = {
            "id": memory_id,
            "content": content,
            "embedding": embedding,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "importance": self._calculate_importance(content, metadata),
            "access_count": 0
        }
        
        await self.es.index(index=self.index_name, id=memory_id, body=doc)
        return memory_id
    
    async def retrieve_similar(self, embedding: List[float], 
                             k: int = 10) -> List[Dict[str, Any]]:
        """Retrieve k most similar memories"""
        query = {
            "size": k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": embedding}
                    }
                }
            }
        }
        
        response = await self.es.search(index=self.index_name, body=query)
        
        memories = []
        for hit in response["hits"]["hits"]:
            memory = hit["_source"]
            memory["score"] = hit["_score"]
            memories.append(memory)
            
            # Update access count
            await self._update_access_count(memory["id"])
        
        return memories
    
    async def consolidate_memories(self):
        """Consolidate and compress memories"""
        # Get all memories
        query = {
            "size": 1000,
            "query": {"match_all": {}},
            "sort": [{"importance": "desc"}]
        }
        
        response = await self.es.search(index=self.index_name, body=query)
        memories = [hit["_source"] for hit in response["hits"]["hits"]]
        
        # Group similar memories
        similarity_threshold = 0.8
        groups = []
        
        for memory in memories:
            added_to_group = False
            
            for group in groups:
                # Check similarity with group representative
                similarity = self._calculate_similarity(
                    memory["embedding"],
                    group[0]["embedding"]
                )
                
                if similarity > similarity_threshold:
                    group.append(memory)
                    added_to_group = True
                    break
            
            if not added_to_group:
                groups.append([memory])
        
        # Consolidate each group
        consolidated = []
        for group in groups:
            if len(group) > 1:
                # Merge similar memories
                merged = self._merge_memories(group)
                consolidated.append(merged)
            else:
                consolidated.append(group[0])
        
        # Update index with consolidated memories
        # (Implementation depends on specific consolidation strategy)
        
        return len(consolidated)
    
    def _calculate_importance(self, content: str, metadata: Dict[str, Any]) -> float:
        """Calculate importance score for a memory"""
        # Base importance on content length and metadata
        base_score = min(len(content) / 1000, 1.0)
        
        # Boost for certain metadata flags
        if metadata:
            if metadata.get("is_goal"):
                base_score *= 2.0
            if metadata.get("is_learning"):
                base_score *= 1.5
            if metadata.get("emotional_valence"):
                base_score *= 1.2
        
        return min(base_score, 1.0)
    
    def _calculate_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        return cosine_similarity([emb1], [emb2])[0][0]
    
    def _merge_memories(self, memories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Merge multiple similar memories into one"""
        # Simple merging strategy - can be made more sophisticated
        contents = [m["content"] for m in memories]
        merged_content = " ".join(set(contents))
        
        # Average embeddings
        embeddings = np.array([m["embedding"] for m in memories])
        merged_embedding = np.mean(embeddings, axis=0).tolist()
        
        # Merge metadata
        merged_metadata = {}
        for memory in memories:
            merged_metadata.update(memory.get("metadata", {}))
        
        # Sum importance and access counts
        total_importance = sum(m["importance"] for m in memories)
        total_access = sum(m["access_count"] for m in memories)
        
        return {
            "id": hashlib.sha256(merged_content.encode()).hexdigest(),
            "content": merged_content,
            "embedding": merged_embedding,
            "metadata": merged_metadata,
            "timestamp": datetime.now().isoformat(),
            "importance": min(total_importance, 1.0),
            "access_count": total_access
        }
    
    async def _update_access_count(self, memory_id: str):
        """Update access count for a memory"""
        await self.es.update(
            index=self.index_name,
            id=memory_id,
            body={"script": {"source": "ctx._source.access_count += 1"}}
        )

class KnowledgeManager:
    """Main knowledge management system"""
    
    def __init__(self, neo4j_uri: str, es_url: str):
        self.knowledge_graph = KnowledgeGraph(
            neo4j_uri, "neo4j", "sutazai_neo4j_2024"
        )
        self.semantic_memory = SemanticMemory(es_url)
        self.entity_extractor = pipeline(
            "ner",
            model="dbmdz/bert-large-cased-finetuned-conll03-english"
        )
        self.embedding_model = pipeline(
            "feature-extraction",
            model="sentence-transformers/all-mpnet-base-v2"
        )
        
    async def initialize(self):
        """Initialize all components"""
        await self.semantic_memory.initialize()
        logger.info("Knowledge Manager initialized")
    
    async def process_text(self, text: str, source: str = None) -> Dict[str, Any]:
        """Process text to extract entities and relationships"""
        # Extract entities
        entities = self.entity_extractor(text)
        
        # Extract relationships using NLP
        doc = self.knowledge_graph.nlp(text)
        
        created_entities = []
        created_relationships = []
        
        # Create entities in knowledge graph
        for ent in entities:
            entity_id = hashlib.sha256(
                f"{ent['word']}:{ent['entity_group']}".encode()
            ).hexdigest()[:16]
            
            entity = Entity(
                id=entity_id,
                name=ent['word'],
                entity_type=self._map_entity_type(ent['entity_group']),
                properties={"score": ent['score']},
                confidence=ent['score'],
                source=source
            )
            
            try:
                await self.knowledge_graph.create_entity(entity)
                created_entities.append(entity.to_dict())
            except Exception as e:
                logger.error(f"Error creating entity: {e}")
        
        # Extract and create relationships
        # This is simplified - in production, use more sophisticated methods
        for sent in doc.sents:
            # Look for subject-verb-object patterns
            for token in sent:
                if token.dep_ == "ROOT":
                    subject = [t for t in token.children if t.dep_ == "nsubj"]
                    obj = [t for t in token.children if t.dep_ == "dobj"]
                    
                    if subject and obj:
                        rel = Relationship(
                            source_id=hashlib.sha256(
                                f"{subject[0].text}:ENTITY".encode()
                            ).hexdigest()[:16],
                            target_id=hashlib.sha256(
                                f"{obj[0].text}:ENTITY".encode()
                            ).hexdigest()[:16],
                            relation_type=RelationType.RELATED_TO,
                            properties={"verb": token.text}
                        )
                        
                        try:
                            await self.knowledge_graph.create_relationship(rel)
                            created_relationships.append({
                                "source": subject[0].text,
                                "target": obj[0].text,
                                "type": token.text
                            })
                        except Exception as e:
                            logger.error(f"Error creating relationship: {e}")
        
        # Store in semantic memory
        embedding = self.embedding_model(text)[0][0].tolist()
        memory_id = await self.semantic_memory.store_memory(
            text,
            embedding,
            {
                "source": source,
                "entities": len(created_entities),
                "relationships": len(created_relationships)
            }
        )
        
        return {
            "entities": created_entities,
            "relationships": created_relationships,
            "memory_id": memory_id
        }
    
    def _map_entity_type(self, ner_type: str) -> EntityType:
        """Map NER types to our entity types"""
        mapping = {
            "PER": EntityType.PERSON,
            "LOC": EntityType.PLACE,
            "ORG": EntityType.OBJECT,
            "MISC": EntityType.ABSTRACT
        }
        return mapping.get(ner_type, EntityType.CONCEPT)
    
    async def query(self, query: str, search_type: str = "semantic") -> Dict[str, Any]:
        """Query the knowledge system"""
        # Generate query embedding
        query_embedding = self.embedding_model(query)[0][0].tolist()
        
        results = {
            "query": query,
            "timestamp": datetime.now().isoformat(),
            "results": []
        }
        
        if search_type == "semantic":
            # Search semantic memory
            memories = await self.semantic_memory.retrieve_similar(
                query_embedding, k=5
            )
            results["memories"] = memories
            
            # Search knowledge graph
            entities = await self.knowledge_graph.semantic_search(query, limit=5)
            results["entities"] = [e.to_dict() for e in entities]
            
        elif search_type == "graph":
            # Extract entities from query and search paths
            doc = self.knowledge_graph.nlp(query)
            entities = [ent.text for ent in doc.ents]
            
            if len(entities) >= 2:
                # Find paths between entities
                paths = await self.knowledge_graph.query_path(
                    entities[0], entities[1]
                )
                results["paths"] = paths
        
        return results

# Global instances
knowledge_manager = None

# API Models
class TextProcessRequest(BaseModel):
    text: str
    source: Optional[str] = None

class EntityRequest(BaseModel):
    name: str
    entity_type: str
    properties: Optional[Dict[str, Any]] = {}
    confidence: float = 1.0
    source: Optional[str] = None

class RelationshipRequest(BaseModel):
    source_name: str
    target_name: str
    relation_type: str
    properties: Optional[Dict[str, Any]] = {}
    confidence: float = 1.0

class QueryRequest(BaseModel):
    query: str
    search_type: str = "semantic"
    limit: int = 10

# API Endpoints
@app.on_event("startup")
async def startup():
    global knowledge_manager
    knowledge_manager = KnowledgeManager(
        os.getenv("NEO4J_URI", "bolt://neo4j:7687"),
        os.getenv("ELASTICSEARCH_URL", "http://elasticsearch:9200")
    )
    await knowledge_manager.initialize()

@app.on_event("shutdown")
async def shutdown():
    if knowledge_manager:
        await knowledge_manager.knowledge_graph.close()

@app.post("/process_text")
async def process_text(request: TextProcessRequest):
    """Process text to extract knowledge"""
    try:
        result = await knowledge_manager.process_text(
            request.text,
            request.source
        )
        return result
    except Exception as e:
        logger.error(f"Error processing text: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_entity")
async def add_entity(request: EntityRequest):
    """Add an entity to the knowledge graph"""
    try:
        entity_id = hashlib.sha256(
            f"{request.name}:{request.entity_type}".encode()
        ).hexdigest()[:16]
        
        entity = Entity(
            id=entity_id,
            name=request.name,
            entity_type=EntityType(request.entity_type),
            properties=request.properties,
            confidence=request.confidence,
            source=request.source
        )
        
        result = await knowledge_manager.knowledge_graph.create_entity(entity)
        return {"id": result, "status": "created"}
    except Exception as e:
        logger.error(f"Error adding entity: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add_relationship")
async def add_relationship(request: RelationshipRequest):
    """Add a relationship between entities"""
    try:
        # Get entity IDs
        source_id = hashlib.sha256(
            f"{request.source_name}:ENTITY".encode()
        ).hexdigest()[:16]
        target_id = hashlib.sha256(
            f"{request.target_name}:ENTITY".encode()
        ).hexdigest()[:16]
        
        relationship = Relationship(
            source_id=source_id,
            target_id=target_id,
            relation_type=RelationType(request.relation_type),
            properties=request.properties,
            confidence=request.confidence
        )
        
        result = await knowledge_manager.knowledge_graph.create_relationship(
            relationship
        )
        return {"status": "created" if result else "failed"}
    except Exception as e:
        logger.error(f"Error adding relationship: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_knowledge(request: QueryRequest):
    """Query the knowledge system"""
    try:
        result = await knowledge_manager.query(
            request.query,
            request.search_type
        )
        return result
    except Exception as e:
        logger.error(f"Error querying knowledge: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/consolidate_memory")
async def consolidate_memory(background_tasks: BackgroundTasks):
    """Trigger memory consolidation"""
    background_tasks.add_task(
        knowledge_manager.semantic_memory.consolidate_memories
    )
    return {"status": "consolidation started"}

@app.get("/stats")
async def get_stats():
    """Get knowledge system statistics"""
    # This would query Neo4j and Elasticsearch for statistics
    return {
        "status": "operational",
        "components": {
            "knowledge_graph": "connected",
            "semantic_memory": "connected"
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "knowledge-manager",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import os
    uvicorn.run(app, host="0.0.0.0", port=8000)