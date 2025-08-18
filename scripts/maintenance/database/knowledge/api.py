#!/usr/bin/env python3
"""
Knowledge Manager API endpoints
Extracted from knowledge_manager.py for modularity
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

from .models import (
    Entity, Relationship, KnowledgeQuery, EntityType, RelationType,
    EntityRequest, RelationshipRequest, SearchRequest, KnowledgeStats
)
from .graph_operations import KnowledgeGraph
from .embeddings import EmbeddingService

logger = logging.getLogger(__name__)

class KnowledgeAPI:
    """FastAPI application for knowledge management"""
    
    def __init__(self):
        self.app = FastAPI(title="SutazAI Knowledge Manager", version="2.0.0")
        self.setup_middleware()
        self.setup_routes()
        
        # Initialize services
        self.graph = None
        self.embeddings = None
        
    def setup_middleware(self):
        """Setup CORS and other middleware"""
        allowed_origins = [
            os.getenv('FRONTEND_URL', 'http://localhost:10011'),
            os.getenv('BACKEND_URL', 'http://localhost:10010'),
            "http://127.0.0.1:10011",
            "http://127.0.0.1:10010"
        ]
        
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=allowed_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.on_event("startup")
        async def startup():
            """Initialize services on startup"""
            neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
            neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
            neo4j_password = os.getenv('NEO4J_PASSWORD', 'password')
            
            self.graph = KnowledgeGraph(neo4j_uri, neo4j_user, neo4j_password)
            self.embeddings = EmbeddingService()
            
            logger.info("Knowledge Manager API started")
            
        @self.app.on_event("shutdown")
        async def shutdown():
            """Cleanup on shutdown"""
            if self.graph:
                await self.graph.close()
            logger.info("Knowledge Manager API shutdown")
            
        @self.app.get("/")
        async def root():
            """Root endpoint"""
            return {"message": "SutazAI Knowledge Manager API"}
            
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            return {"status": "healthy", "timestamp": "2025-08-17T12:00:00Z"}
            
        @self.app.post("/entities")
        async def create_entity(entity_request: EntityRequest):
            """Create a new entity"""
            try:
                # Generate embedding for the entity
                embedding = await self.embeddings.generate_embedding(
                    f"{entity_request.name} {entity_request.description}"
                )
                
                entity = Entity(
                    name=entity_request.name,
                    entity_type=EntityType(entity_request.entity_type),
                    description=entity_request.description or "",
                    properties=entity_request.properties or {},
                    embedding=embedding
                )
                
                entity_id = await self.graph.create_entity(entity)
                return {"entity_id": entity_id, "status": "created"}
                
            except Exception as e:
                logger.error(f"Failed to create entity: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.post("/relationships")
        async def create_relationship(rel_request: RelationshipRequest):
            """Create a new relationship"""
            try:
                relationship = Relationship(
                    source_id=rel_request.source_id,
                    target_id=rel_request.target_id,
                    relation_type=RelationType(rel_request.relation_type),
                    properties=rel_request.properties or {},
                    weight=rel_request.weight or 1.0
                )
                
                rel_id = await self.graph.create_relationship(relationship)
                return {"relationship_id": rel_id, "status": "created"}
                
            except Exception as e:
                logger.error(f"Failed to create relationship: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.post("/search")
        async def search_entities(search_request: SearchRequest):
            """Search for entities"""
            try:
                entity_types = [EntityType(t) for t in search_request.entity_types] if search_request.entity_types else None
                
                query = KnowledgeQuery(
                    query_text=search_request.query,
                    entity_types=entity_types,
                    limit=search_request.limit or 10,
                    min_confidence=search_request.min_confidence or 0.5
                )
                
                entities = await self.graph.search_entities(query)
                
                return {
                    "entities": [
                        {
                            "id": e.id,
                            "name": e.name,
                            "type": e.entity_type.value,
                            "description": e.description,
                            "confidence": e.confidence
                        }
                        for e in entities
                    ],
                    "count": len(entities)
                }
                
            except Exception as e:
                logger.error(f"Failed to search entities: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.get("/entities/{entity_id}/related")
        async def get_related_entities(entity_id: str, max_depth: int = 2):
            """Get entities related to the specified entity"""
            try:
                related = await self.graph.find_related_entities(entity_id, max_depth)
                
                return {
                    "related_entities": [
                        {
                            "entity": {
                                "id": entity.id,
                                "name": entity.name,
                                "type": entity.entity_type.value,
                                "description": entity.description
                            },
                            "relevance_score": weight
                        }
                        for entity, weight in related
                    ],
                    "count": len(related)
                }
                
            except Exception as e:
                logger.error(f"Failed to get related entities: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
        @self.app.get("/statistics")
        async def get_statistics():
            """Get knowledge graph statistics"""
            try:
                stats = await self.graph.get_graph_statistics()
                analysis = await self.graph.analyze_graph_structure()
                
                return {
                    "statistics": stats,
                    "analysis": analysis
                }
                
            except Exception as e:
                logger.error(f"Failed to get statistics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
                
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the API server"""
        uvicorn.run(self.app, host=host, port=port)


def create_app() -> FastAPI:
    """Factory function to create the FastAPI app"""
    api = KnowledgeAPI()
    return api.app


if __name__ == "__main__":
    api = KnowledgeAPI()
    api.run()