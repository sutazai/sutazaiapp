#!/usr/bin/env python3
"""Simple FAISS service for SutazAI"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="FAISS Service", version="1.0.0")

# Simple in-memory storage for demo
vectors = {}
dimension = 128

class VectorRequest(BaseModel):
    id: str
    vector: list

class SearchRequest(BaseModel):
    vector: list
    k: int = 5

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "service": "faiss", "version": "1.0.0"}

@app.post("/index")
async def index_vector(request: VectorRequest):
    """Index a vector"""
    try:
        if len(request.vector) != dimension:
            raise ValueError(f"Vector must have {dimension} dimensions")
        
        vectors[request.id] = np.array(request.vector)
        return {"status": "indexed", "id": request.id}
    except Exception as e:
        logger.error(f"Error indexing vector: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/search")
async def search_vectors(request: SearchRequest):
    """Search for similar vectors"""
    try:
        if not vectors:
            return {"results": []}
        
        query = np.array(request.vector)
        results = []
        
        # Simple cosine similarity
        for vec_id, vec in vectors.items():
            similarity = np.dot(query, vec) / (np.linalg.norm(query) * np.linalg.norm(vec))
            results.append({"id": vec_id, "score": float(similarity)})
        
        # Sort by score and return top k
        results.sort(key=lambda x: x["score"], reverse=True)
        return {"results": results[:request.k]}
    except Exception as e:
        logger.error(f"Error searching vectors: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get service statistics"""
    return {
        "indexed_vectors": len(vectors),
        "dimension": dimension,
        "memory_usage_mb": sum(v.nbytes for v in vectors.values()) / 1024 / 1024
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)