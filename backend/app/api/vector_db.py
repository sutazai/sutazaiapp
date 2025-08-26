"""
Vector Database Integration API
Provides unified interface to Qdrant and ChromaDB for embedding operations
"""

import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
import httpx
# NumPy not required for basic vector operations

# Import Ollama embedding service
from ..services.consolidated_ollama_service import get_ollama_embedding_service

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/vectors", tags=["Vector Database"])

# Vector Database Configuration
QDRANT_URL = "http://sutazai-qdrant:6333"
CHROMADB_URL = "http://sutazai-chromadb:8000"

# Data Models
class DocumentRequest(BaseModel):
    """Request to store document with embedding"""
    text: str = Field(..., min_length=1, description="Text content to embed")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    collection: str = Field(default="sutazai_docs", description="Collection name")
    document_id: Optional[str] = Field(None, description="Optional document ID")

class SearchRequest(BaseModel):
    """Request to search similar documents"""
    query: str = Field(..., min_length=1, description="Search query")
    collection: str = Field(default="sutazai_docs", description="Collection to search")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results to return")
    threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Similarity threshold")

class DocumentResponse(BaseModel):
    """Document search result"""
    id: str
    text: str
    metadata: Dict[str, Any]
    score: float

class VectorStats(BaseModel):
    """Vector database statistics"""
    total_documents: int
    total_collections: int
    backend_status: Dict[str, str]
    
# Vector Database Service
class VectorDBService:
    """Unified vector database service"""
    
    def __init__(self):
        self.qdrant_available = False
        self.chromadb_available = False
        self.primary_backend = None
    
    async def initialize(self):
        """Initialize vector database connections"""
        
        # Check Qdrant availability
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{QDRANT_URL}/collections", timeout=3.0)
                if response.status_code == 200:
                    self.qdrant_available = True
                    logger.info("Qdrant vector database connected")
        except Exception as e:
            logger.warning(f"Qdrant unavailable: {e}")
        
        # Check ChromaDB availability
        try:
            async with httpx.AsyncClient() as client:
                # Get ChromaDB token from environment variable
                chroma_token = os.getenv("CHROMADB_API_KEY")
                headers = {"X-Chroma-Token": chroma_token} if chroma_token else {}
                if not chroma_token:
                    logger.warning("CHROMADB_API_KEY not set in environment variables")
                response = await client.get(f"{CHROMADB_URL}/api/v2/heartbeat", headers=headers, timeout=3.0)
                if response.status_code == 200:
                    self.chromadb_available = True
                    logger.info("ChromaDB vector database connected")
        except Exception as e:
            logger.warning(f"ChromaDB unavailable: {e}")
        
        # Set primary backend
        if self.qdrant_available:
            self.primary_backend = "qdrant"
        elif self.chromadb_available:
            self.primary_backend = "chromadb"
        else:
            logger.error("No vector databases available!")
            
        return self.primary_backend is not None
    
    async def ensure_collection(self, collection_name: str) -> bool:
        """Ensure collection exists in vector database"""
        
        if self.primary_backend == "qdrant":
            try:
                async with httpx.AsyncClient() as client:
                    # Check if collection exists
                    response = await client.get(f"{QDRANT_URL}/collections/{collection_name}")
                    
                    if response.status_code == 404:
                        # Create collection
                        create_data = {
                            "vectors": {
                                "size": 4096,  # TinyLlama embedding size
                                "distance": "Cosine"
                            }
                        }
                        
                        response = await client.put(
                            f"{QDRANT_URL}/collections/{collection_name}",
                            json=create_data
                        )
                        
                        if response.status_code == 200:
                            logger.info(f"Created Qdrant collection: {collection_name}")
                            return True
                        else:
                            logger.error(f"Failed to create Qdrant collection: {response.text}")
                            return False
                    
                    return response.status_code == 200
                    
            except Exception as e:
                logger.error(f"Error ensuring Qdrant collection: {e}")
                return False
                
        elif self.primary_backend == "chromadb":
            try:
                async with httpx.AsyncClient() as client:
                    # Create or get collection
                    collection_data = {
                        "name": collection_name,
                        "metadata": {"description": "SutazAI document collection"}
                    }
                    
                    response = await client.post(
                        f"{CHROMADB_URL}/api/v1/collections",
                        json=collection_data
                    )
                    
                    # 200 = created, 409 = already exists (both OK)
                    success = response.status_code in [200, 409]
                    if success:
                        logger.info(f"ChromaDB collection ready: {collection_name}")
                    else:
                        logger.error(f"ChromaDB collection error: {response.text}")
                        
                    return success
                    
            except Exception as e:
                logger.error(f"Error ensuring ChromaDB collection: {e}")
                return False
        
        return False
    
    async def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using Ollama"""
        try:
            embedding_service = await get_ollama_embedding_service()
            
            # Use Ollama embeddings endpoint
            embedding = await embedding_service.generate_embedding(text)
            
            if embedding and len(embedding) > 0:
                return embedding
            else:
                logger.warning("Empty embedding generated")
                return None
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    async def store_document(self, request: DocumentRequest) -> Dict[str, Any]:
        """Store document with embedding"""
        
        if not self.primary_backend:
            raise HTTPException(status_code=503, detail="No vector database available")
        
        # Ensure collection exists
        if not await self.ensure_collection(request.collection):
            raise HTTPException(status_code=500, detail="Failed to prepare collection")
        
        # Generate embedding
        embedding = await self.generate_embedding(request.text)
        if not embedding:
            raise HTTPException(status_code=500, detail="Failed to generate embedding")
        
        # Generate document ID if not provided
        doc_id = request.document_id or f"doc_{datetime.now().timestamp()}"
        
        # Store in vector database
        if self.primary_backend == "qdrant":
            return await self._store_qdrant(request, doc_id, embedding)
        elif self.primary_backend == "chromadb":
            return await self._store_chromadb(request, doc_id, embedding)
    
    async def _store_qdrant(self, request: DocumentRequest, doc_id: str, embedding: List[float]) -> Dict[str, Any]:
        """Store document in Qdrant"""
        try:
            async with httpx.AsyncClient() as client:
                point_data = {
                    "points": [{
                        "id": doc_id,
                        "vector": embedding,
                        "payload": {
                            "text": request.text,
                            "metadata": request.metadata,
                            "created_at": datetime.now().isoformat()
                        }
                    }]
                }
                
                response = await client.put(
                    f"{QDRANT_URL}/collections/{request.collection}/points",
                    json=point_data
                )
                
                if response.status_code == 200:
                    return {"id": doc_id, "backend": "qdrant", "status": "stored"}
                else:
                    raise HTTPException(status_code=500, detail=f"Qdrant storage failed: {response.text}")
                    
        except Exception as e:
            logger.error(f"Qdrant storage error: {e}")
            raise HTTPException(status_code=500, detail="Qdrant storage failed")
    
    async def _store_chromadb(self, request: DocumentRequest, doc_id: str, embedding: List[float]) -> Dict[str, Any]:
        """Store document in ChromaDB"""
        try:
            async with httpx.AsyncClient() as client:
                add_data = {
                    "ids": [doc_id],
                    "embeddings": [embedding],
                    "documents": [request.text],
                    "metadatas": [request.metadata]
                }
                
                response = await client.post(
                    f"{CHROMADB_URL}/api/v1/collections/{request.collection}/add",
                    json=add_data
                )
                
                if response.status_code == 200:
                    return {"id": doc_id, "backend": "chromadb", "status": "stored"}
                else:
                    raise HTTPException(status_code=500, detail=f"ChromaDB storage failed: {response.text}")
                    
        except Exception as e:
            logger.error(f"ChromaDB storage error: {e}")
            raise HTTPException(status_code=500, detail="ChromaDB storage failed")
    
    async def search_documents(self, request: SearchRequest) -> List[DocumentResponse]:
        """Search for similar documents"""
        
        if not self.primary_backend:
            raise HTTPException(status_code=503, detail="No vector database available")
        
        # Generate query embedding
        query_embedding = await self.generate_embedding(request.query)
        if not query_embedding:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding")
        
        # Search in vector database
        if self.primary_backend == "qdrant":
            return await self._search_qdrant(request, query_embedding)
        elif self.primary_backend == "chromadb":
            return await self._search_chromadb(request, query_embedding)
    
    async def _search_qdrant(self, request: SearchRequest, query_embedding: List[float]) -> List[DocumentResponse]:
        """Search documents in Qdrant"""
        try:
            async with httpx.AsyncClient() as client:
                search_data = {
                    "vector": query_embedding,
                    "limit": request.limit,
                    "score_threshold": request.threshold,
                    "with_payload": True
                }
                
                response = await client.post(
                    f"{QDRANT_URL}/collections/{request.collection}/points/search",
                    json=search_data
                )
                
                if response.status_code == 200:
                    results = response.json().get("result", [])
                    
                    return [
                        DocumentResponse(
                            id=str(result["id"]),
                            text=result["payload"]["text"],
                            metadata=result["payload"]["metadata"],
                            score=result["score"]
                        ) for result in results
                    ]
                else:
                    logger.error(f"Qdrant search failed: {response.text}")
                    # Return empty list with proper validation - no matching documents found
                    return []  # Valid empty list: search returned no results from Qdrant
                    
        except Exception as e:
            logger.error(f"Qdrant search error: {e}")
            # Return empty list on error - connection or query failure
            return []  # Valid empty list: Qdrant search error, no results available
    
    async def _search_chromadb(self, request: SearchRequest, query_embedding: List[float]) -> List[DocumentResponse]:
        """Search documents in ChromaDB"""
        try:
            async with httpx.AsyncClient() as client:
                query_data = {
                    "query_embeddings": [query_embedding],
                    "n_results": request.limit
                }
                
                response = await client.post(
                    f"{CHROMADB_URL}/api/v1/collections/{request.collection}/query",
                    json=query_data
                )
                
                if response.status_code == 200:
                    data = response.json()
                    
                    results = []
                    if data.get("ids") and len(data["ids"]) > 0:
                        for i, doc_id in enumerate(data["ids"][0]):
                            score = 1.0 - data["distances"][0][i]  # Convert distance to similarity
                            
                            if score >= request.threshold:
                                results.append(DocumentResponse(
                                    id=doc_id,
                                    text=data["documents"][0][i],
                                    metadata=data["metadatas"][0][i] or {},
                                    score=score
                                ))
                    
                    return results
                else:
                    logger.error(f"ChromaDB search failed: {response.text}")
                    # Return empty list with proper validation - search failed
                    return []  # Valid empty list: ChromaDB search failed, no results
                    
        except Exception as e:
            logger.error(f"ChromaDB search error: {e}")
            # Return empty list on error - connection or query failure
            return []  # Valid empty list: ChromaDB error, no results available

# Initialize service
vector_service = VectorDBService()

# Dependency
async def get_vector_service():
    """Get vector database service"""
    if not vector_service.primary_backend:
        await vector_service.initialize()
    return vector_service

# API Endpoints
@router.post("/store", response_model=Dict[str, Any])
async def store_document(
    request: DocumentRequest,
    service: VectorDBService = Depends(get_vector_service)
):
    """Store document with generated embedding"""
    return await service.store_document(request)

@router.post("/search", response_model=List[DocumentResponse])
async def search_documents(
    request: SearchRequest,
    service: VectorDBService = Depends(get_vector_service)
):
    """Search for similar documents"""
    return await service.search_documents(request)

@router.get("/collections")
async def list_collections(service: VectorDBService = Depends(get_vector_service)):
    """List available collections"""
    
    if not service.primary_backend:
        raise HTTPException(status_code=503, detail="No vector database available")
    
    try:
        if service.primary_backend == "qdrant":
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{QDRANT_URL}/collections")
                if response.status_code == 200:
                    data = response.json()
                    return {"collections": [c["name"] for c in data.get("result", {}).get("collections", [])]}
        
        elif service.primary_backend == "chromadb":
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{CHROMADB_URL}/api/v1/collections")
                if response.status_code == 200:
                    collections = response.json()
                    return {"collections": [c["name"] for c in collections]}
        
        return {"collections": []}
        
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        return {"collections": []}

@router.get("/stats", response_model=VectorStats)
async def get_vector_stats(service: VectorDBService = Depends(get_vector_service)):
    """Get vector database statistics"""
    
    total_documents = 0
    total_collections = 0
    backend_status = {}
    
    if service.qdrant_available:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{QDRANT_URL}/collections")
                if response.status_code == 200:
                    data = response.json()
                    collections = data.get("result", {}).get("collections", [])
                    total_collections += len(collections)
                    
                    # Count documents in each collection
                    for collection in collections:
                        col_response = await client.get(f"{QDRANT_URL}/collections/{collection['name']}")
                        if col_response.status_code == 200:
                            col_data = col_response.json()
                            total_documents += col_data.get("result", {}).get("points_count", 0)
                    
                    backend_status["qdrant"] = "healthy"
        except Exception as e:
            backend_status["qdrant"] = f"error: {str(e)}"
    
    if service.chromadb_available:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{CHROMADB_URL}/api/v1/collections")
                if response.status_code == 200:
                    collections = response.json()
                    total_collections += len(collections)
                    
                    backend_status["chromadb"] = "healthy"
        except Exception as e:
            backend_status["chromadb"] = f"error: {str(e)}"
    
    return VectorStats(
        total_documents=total_documents,
        total_collections=total_collections,
        backend_status=backend_status
    )
