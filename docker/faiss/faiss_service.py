#!/usr/bin/env python3
"""
FAISS Vector Similarity Search Service for SutazAI
Provides fast similarity search and clustering capabilities
"""

import faiss
import numpy as np
import pickle
import os
import json
import logging
import uvicorn
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="SutazAI FAISS Service",
    description="Fast Similarity Search and Vector Clustering",
    version="1.0.0"
)

class VectorData(BaseModel):
    vectors: List[List[float]]
    ids: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class SearchQuery(BaseModel):
    query_vector: List[float]
    k: int = 10
    index_name: str = "default"

class FAISSManager:
    """FAISS index management and operations"""
    
    def __init__(self, data_dir: str = "/data/faiss_indexes"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.indexes = {}
        self.metadata = {}
        self.load_existing_indexes()
    
    def load_existing_indexes(self):
        """Load existing FAISS indexes from disk"""
        for index_file in self.data_dir.glob("*.index"):
            index_name = index_file.stem
            try:
                index = faiss.read_index(str(index_file))
                self.indexes[index_name] = index
                
                # Load metadata if exists
                metadata_file = self.data_dir / f"{index_name}_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        self.metadata[index_name] = json.load(f)
                
                logger.info(f"Loaded FAISS index: {index_name}")
            except Exception as e:
                logger.error(f"Failed to load index {index_name}: {e}")
    
    def create_index(self, index_name: str, dimension: int, index_type: str = "IVFFlat"):
        """Create a new FAISS index"""
        try:
            if index_type == "IVFFlat":
                # IVF (Inverted File) with flat quantizer
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, 100)  # 100 clusters
            elif index_type == "LSH":
                # Locality Sensitive Hashing
                index = faiss.IndexLSH(dimension, 64)  # 64 bits
            elif index_type == "HNSW":
                # Hierarchical Navigable Small World
                index = faiss.IndexHNSWFlat(dimension, 32)
            else:
                # Default to flat L2 index
                index = faiss.IndexFlatL2(dimension)
            
            self.indexes[index_name] = index
            self.metadata[index_name] = {
                "dimension": dimension,
                "index_type": index_type,
                "created_at": time.time(),
                "total_vectors": 0
            }
            
            self.save_index(index_name)
            logger.info(f"Created FAISS index: {index_name} ({index_type})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create index {index_name}: {e}")
            return False
    
    def add_vectors(self, index_name: str, vectors: np.ndarray, ids: Optional[List[str]] = None):
        """Add vectors to an existing index"""
        if index_name not in self.indexes:
            raise ValueError(f"Index {index_name} does not exist")
        
        index = self.indexes[index_name]
        
        # Train index if necessary (for IVF indexes)
        if not index.is_trained and hasattr(index, 'train'):
            logger.info(f"Training index {index_name}...")
            index.train(vectors)
        
        # Add vectors
        if ids:
            # Store ID mapping in metadata
            if "id_mapping" not in self.metadata[index_name]:
                self.metadata[index_name]["id_mapping"] = {}
            
            start_id = index.ntotal
            for i, vector_id in enumerate(ids):
                self.metadata[index_name]["id_mapping"][start_id + i] = vector_id
        
        index.add(vectors)
        self.metadata[index_name]["total_vectors"] = index.ntotal
        
        self.save_index(index_name)
        logger.info(f"Added {len(vectors)} vectors to index {index_name}")
    
    def search(self, index_name: str, query_vector: np.ndarray, k: int = 10):
        """Search for similar vectors"""
        if index_name not in self.indexes:
            raise ValueError(f"Index {index_name} does not exist")
        
        index = self.indexes[index_name]
        query_vector = query_vector.reshape(1, -1)
        
        distances, indices = index.search(query_vector, k)
        
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx >= 0:  # Valid index
                result = {
                    "rank": i + 1,
                    "distance": float(distance),
                    "index": int(idx)
                }
                
                # Add ID if available
                if "id_mapping" in self.metadata[index_name]:
                    if idx in self.metadata[index_name]["id_mapping"]:
                        result["id"] = self.metadata[index_name]["id_mapping"][idx]
                
                results.append(result)
        
        return results
    
    def save_index(self, index_name: str):
        """Save index and metadata to disk"""
        try:
            index_file = self.data_dir / f"{index_name}.index"
            faiss.write_index(self.indexes[index_name], str(index_file))
            
            metadata_file = self.data_dir / f"{index_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata[index_name], f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save index {index_name}: {e}")
    
    def get_index_stats(self, index_name: str):
        """Get statistics for an index"""
        if index_name not in self.indexes:
            raise ValueError(f"Index {index_name} does not exist")
        
        index = self.indexes[index_name]
        stats = {
            "name": index_name,
            "total_vectors": index.ntotal,
            "dimension": index.d,
            "is_trained": index.is_trained,
            "metric_type": "L2",  # Default
            **self.metadata[index_name]
        }
        
        return stats

# Initialize FAISS manager
faiss_manager = FAISSManager()

@app.get("/")
async def root():
    return {"service": "SutazAI FAISS", "status": "running", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "total_indexes": len(faiss_manager.indexes)
    }

@app.post("/indexes/{index_name}")
async def create_index(index_name: str, dimension: int, index_type: str = "IVFFlat"):
    """Create a new FAISS index"""
    try:
        success = faiss_manager.create_index(index_name, dimension, index_type)
        if success:
            return {"message": f"Index {index_name} created successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to create index")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/indexes/{index_name}/vectors")
async def add_vectors(index_name: str, data: VectorData):
    """Add vectors to an existing index"""
    try:
        vectors = np.array(data.vectors, dtype=np.float32)
        faiss_manager.add_vectors(index_name, vectors, data.ids)
        return {"message": f"Added {len(vectors)} vectors to {index_name}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/search")
async def search_vectors(query: SearchQuery):
    """Search for similar vectors"""
    try:
        query_vector = np.array(query.query_vector, dtype=np.float32)
        results = faiss_manager.search(query.index_name, query_vector, query.k)
        return {"results": results, "query_time": time.time()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/indexes")
async def list_indexes():
    """List all available indexes"""
    indexes = []
    for name in faiss_manager.indexes.keys():
        stats = faiss_manager.get_index_stats(name)
        indexes.append(stats)
    return {"indexes": indexes}

@app.get("/indexes/{index_name}/stats")
async def get_index_stats(index_name: str):
    """Get statistics for a specific index"""
    try:
        stats = faiss_manager.get_index_stats(index_name)
        return stats
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.delete("/indexes/{index_name}")
async def delete_index(index_name: str):
    """Delete an index"""
    try:
        if index_name in faiss_manager.indexes:
            del faiss_manager.indexes[index_name]
            del faiss_manager.metadata[index_name]
            
            # Remove files
            index_file = faiss_manager.data_dir / f"{index_name}.index"
            metadata_file = faiss_manager.data_dir / f"{index_name}_metadata.json"
            
            if index_file.exists():
                index_file.unlink()
            if metadata_file.exists():
                metadata_file.unlink()
            
            return {"message": f"Index {index_name} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail=f"Index {index_name} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "faiss_service:app",
        host="0.0.0.0",
        port=8088,
        log_level="info",
        reload=False
    )