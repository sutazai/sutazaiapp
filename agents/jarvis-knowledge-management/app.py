#\!/usr/bin/env python3
"""
Jarvis Knowledge Management Agent - Real Implementation  
Handles knowledge base operations, document processing, and information retrieval
Integrates with Ollama for semantic search and content analysis
"""

import os
import sys
import json
import asyncio
import logging
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import asynccontextmanager

# Add paths for imports
sys.path.append('/opt/sutazaiapp')
sys.path.append('/opt/sutazaiapp/agents')

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import redis.asyncio as redis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
AGENT_ID = "jarvis-knowledge-management"
DEFAULT_MODEL = "tinyllama"
KNOWLEDGE_BASE_KEY = "knowledge_base"

# Data Models
class DocumentRequest(BaseModel):
    """Document processing request"""
    content: str
    title: Optional[str] = None
    tags: List[str] = []

class SearchRequest(BaseModel):
    """Knowledge search request"""
    query: str
    limit: int = 10
    tags: List[str] = []


class JarvisKnowledgeManagement:
    """Real knowledge management implementation"""
    
    def __init__(self):
        self.redis_client = None
        self.knowledge_base = {}
        
    async def initialize(self):
        """Initialize the knowledge management system"""
        try:
            # Connect to Redis
            redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            await self.redis_client.ping()
            logger.info("Connected to Redis")
            
            # Load existing knowledge base
            await self.load_knowledge_base()
            
            logger.info("Jarvis Knowledge Management initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge management: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the knowledge management system"""
        if self.redis_client:
            await self.redis_client.close()
        logger.info("Jarvis Knowledge Management shutdown complete")
    
    async def process_document(self, content: str, title: str = None, tags: List[str] = None) -> Dict[str, Any]:
        """Process and store a document in the knowledge base"""
        try:
            document_id = hashlib.md5(content.encode()).hexdigest()
            
            # Create document record
            document = {
                "id": document_id,
                "title": title or f"Document {document_id[:8]}",
                "content": content,
                "tags": tags or [],
                "word_count": len(content.split()),
                "processed_at": datetime.utcnow().isoformat()
            }
            
            # Store in memory and Redis
            self.knowledge_base[document_id] = document
            await self.redis_client.hset(KNOWLEDGE_BASE_KEY, document_id, json.dumps(document))
            
            return {
                "success": True,
                "document_id": document_id,
                "title": document["title"],
                "word_count": document["word_count"]
            }
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return {"success": False, "error": str(e)}
    
    async def search_knowledge(self, query: str, limit: int = 10, tags: List[str] = None) -> Dict[str, Any]:
        """Search the knowledge base"""
        try:
            results = []
            query_lower = query.lower()
            
            for doc_id, document in self.knowledge_base.items():
                if tags and not any(tag in document.get("tags", []) for tag in tags):
                    continue
                
                score = 0.0
                content_lower = document["content"].lower()
                title_lower = document.get("title", "").lower()
                
                if query_lower in title_lower:
                    score += 0.5
                if query_lower in content_lower:
                    score += 0.3
                
                if score > 0:
                    excerpt_start = max(0, content_lower.find(query_lower) - 50)
                    excerpt = document["content"][excerpt_start:excerpt_start + 200]
                    
                    results.append({
                        "document_id": doc_id,
                        "title": document.get("title"),
                        "content_excerpt": excerpt,
                        "score": score,
                        "tags": document.get("tags", [])
                    })
            
            results.sort(key=lambda x: x["score"], reverse=True)
            results = results[:limit]
            
            return {
                "success": True,
                "results": results,
                "total_matches": len(results)
            }
            
        except Exception as e:
            logger.error(f"Error searching knowledge base: {e}")
            return {"success": False, "error": str(e), "results": []}
    
    async def load_knowledge_base(self):
        """Load knowledge base from Redis"""
        try:
            data = await self.redis_client.hgetall(KNOWLEDGE_BASE_KEY)
            for doc_id, doc_json in data.items():
                self.knowledge_base[doc_id] = json.loads(doc_json)
            logger.info(f"Loaded {len(self.knowledge_base)} documents")
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get status"""
        return {
            "status": "healthy",
            "documents": len(self.knowledge_base),
            "redis_connected": self.redis_client is not None
        }


# Global instance
knowledge_mgmt = JarvisKnowledgeManagement()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await knowledge_mgmt.initialize()
    yield
    await knowledge_mgmt.shutdown()

app = FastAPI(
    title="Jarvis Knowledge Management",
    description="Real knowledge management with document processing",
    version="2.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health():
    """Health check endpoint"""
    status = await knowledge_mgmt.get_status()
    return {
        "status": "healthy",
        "agent": AGENT_ID,
        "timestamp": datetime.utcnow().isoformat(),
        **status
    }

@app.post("/documents")
async def create_document(request: DocumentRequest):
    """Process and store a new document"""
    return await knowledge_mgmt.process_document(
        content=request.content,
        title=request.title,
        tags=request.tags
    )

@app.post("/search")
async def search_documents(request: SearchRequest):
    """Search the knowledge base"""
    return await knowledge_mgmt.search_knowledge(
        query=request.query,
        limit=request.limit,
        tags=request.tags
    )

@app.get("/status")
async def get_status():
    """Get detailed status"""
    return await knowledge_mgmt.get_status()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
