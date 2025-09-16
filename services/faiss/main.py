"""
FAISS Service API
Fast similarity search service for AI agent memory
"""

import os
import json
import faiss
import numpy as np
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging

# Configure logging
# Handle case-insensitive log level from environment
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
# Validate log level to prevent invalid values
valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
if log_level not in valid_levels:
    log_level = "INFO"  # Default to INFO if invalid

logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="FAISS Vector Search Service",
    description="Fast approximate nearest neighbor search for AI agents",
    version="1.0.0"
)

# Configuration
INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "/app/indices")
DIMENSION = int(os.getenv("DIMENSION", "768"))
MAX_VECTORS = int(os.getenv("MAX_VECTORS", "1000000"))

# Global index storage
indices: Dict[str, faiss.IndexFlatL2] = {}

# Pydantic models
class Vector(BaseModel):
    id: str
    vector: List[float]
    metadata: Optional[Dict] = {}

class SearchRequest(BaseModel):
    collection: str
    vector: List[float]
    k: int = 10

class CreateIndexRequest(BaseModel):
    collection: str
    dimension: Optional[int] = DIMENSION
    metric: str = "L2"  # L2 or IP (inner product)

# Helper functions
def get_or_create_index(collection: str, dimension: int = DIMENSION):
    """Get existing index or create a new one"""
    if collection not in indices:
        indices[collection] = faiss.IndexFlatL2(dimension)
        logger.info(f"Created new FAISS index for collection: {collection}")
    return indices[collection]

def save_index(collection: str):
    """Save index to disk"""
    if collection in indices:
        index_file = os.path.join(INDEX_PATH, f"{collection}.index")
        faiss.write_index(indices[collection], index_file)
        logger.info(f"Saved index {collection} to {index_file}")

def load_index(collection: str):
    """Load index from disk"""
    index_file = os.path.join(INDEX_PATH, f"{collection}.index")
    if os.path.exists(index_file):
        indices[collection] = faiss.read_index(index_file)
        logger.info(f"Loaded index {collection} from {index_file}")
        return True
    return False

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Load existing indices on startup"""
    os.makedirs(INDEX_PATH, exist_ok=True)
    logger.info("FAISS Service started")
    
    # Load any existing indices
    for filename in os.listdir(INDEX_PATH):
        if filename.endswith(".index"):
            collection = filename.replace(".index", "")
            load_index(collection)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "dimension": DIMENSION,
        "max_vectors": MAX_VECTORS,
        "indices_loaded": list(indices.keys())
    }

@app.post("/index/create")
async def create_index(request: CreateIndexRequest):
    """Create a new FAISS index"""
    try:
        if request.collection in indices:
            return {"message": f"Index {request.collection} already exists"}
        
        if request.metric == "L2":
            indices[request.collection] = faiss.IndexFlatL2(request.dimension)
        elif request.metric == "IP":
            indices[request.collection] = faiss.IndexFlatIP(request.dimension)
        else:
            raise ValueError(f"Unknown metric: {request.metric}")
        
        save_index(request.collection)
        return {
            "message": f"Index {request.collection} created successfully",
            "dimension": request.dimension,
            "metric": request.metric
        }
    except Exception as e:
        logger.error(f"Error creating index: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vectors/add")
async def add_vectors(collection: str, vectors: List[Vector]):
    """Add vectors to a collection"""
    try:
        index = get_or_create_index(collection, len(vectors[0].vector))
        
        # Prepare vectors for insertion
        vector_array = np.array([v.vector for v in vectors], dtype=np.float32)
        
        # Add to index
        index.add(vector_array)
        
        # Save updated index
        save_index(collection)
        
        return {
            "message": f"Added {len(vectors)} vectors to {collection}",
            "total_vectors": index.ntotal
        }
    except Exception as e:
        logger.error(f"Error adding vectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_similar(request: SearchRequest):
    """Search for similar vectors"""
    try:
        if request.collection not in indices:
            raise HTTPException(status_code=404, detail=f"Collection {request.collection} not found")
        
        index = indices[request.collection]
        
        # Prepare query vector
        query_vector = np.array([request.vector], dtype=np.float32)
        
        # Search
        distances, indices_result = index.search(query_vector, request.k)
        
        return {
            "collection": request.collection,
            "k": request.k,
            "results": [
                {"index": int(idx), "distance": float(dist)}
                for idx, dist in zip(indices_result[0], distances[0])
                if idx >= 0  # FAISS returns -1 for not found
            ]
        }
    except Exception as e:
        logger.error(f"Error searching vectors: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
async def list_collections():
    """List all collections"""
    return {
        "collections": [
            {
                "name": name,
                "vectors": index.ntotal,
                "dimension": index.d
            }
            for name, index in indices.items()
        ]
    }

@app.delete("/collections/{collection}")
async def delete_collection(collection: str):
    """Delete a collection"""
    if collection in indices:
        del indices[collection]
        
        # Remove from disk
        index_file = os.path.join(INDEX_PATH, f"{collection}.index")
        if os.path.exists(index_file):
            os.remove(index_file)
        
        return {"message": f"Collection {collection} deleted"}
    else:
        raise HTTPException(status_code=404, detail=f"Collection {collection} not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)