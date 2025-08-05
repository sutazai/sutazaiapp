#!/usr/bin/env python3
"""
SutazAI FAISS Vector Index Service
High-performance vector similarity search using FAISS
"""

import os
import json
import numpy as np
import faiss
import redis
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import pickle
import logging
from prometheus_client import Counter, Gauge, Histogram, generate_latest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
SEARCH_REQUESTS = Counter('faiss_search_requests_total', 'Total search requests')
ADD_REQUESTS = Counter('faiss_add_requests_total', 'Total add requests')
INDEX_SIZE = Gauge('faiss_index_size_total', 'Total vectors in index')
SEARCH_LATENCY = Histogram('faiss_search_latency_seconds', 'Search latency')
MEMORY_USAGE = Gauge('faiss_memory_usage_bytes', 'Memory usage in bytes')

class VectorData(BaseModel):
    id: str
    vector: List[float]
    metadata: Optional[Dict] = None

class SearchRequest(BaseModel):
    vector: List[float]
    k: int = 10
    nprobe: Optional[int] = None

class SearchResult(BaseModel):
    id: str
    score: float
    metadata: Optional[Dict] = None

class FAISSVectorService:
    def __init__(self):
        self.dimension = int(os.getenv('DIMENSION', '1536'))
        self.max_vectors = int(os.getenv('MAX_VECTORS', '1000000'))
        self.index_path = os.getenv('FAISS_INDEX_PATH', '/app/data/indexes')
        
        self.index = None
        self.id_map = {}  # Maps FAISS index position to custom IDs
        self.metadata_map = {}  # Maps custom IDs to metadata
        self.redis_client = None
        self.next_id = 0
        
        # Ensure index directory exists
        os.makedirs(self.index_path, exist_ok=True)
        
    async def initialize(self):
        """Initialize FAISS index and Redis connection"""
        try:
            # Initialize Redis
            redis_url = os.getenv('REDIS_URL', 'redis://redis:6379/3')
            self.redis_client = redis.from_url(redis_url)
            
            # Load existing index or create new one
            await self.load_or_create_index()
            
            logger.info(f"FAISS Vector Service initialized with dimension {self.dimension}")
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS service: {e}")
            raise
    
    async def load_or_create_index(self):
        """Load existing index or create a new one"""
        index_file = os.path.join(self.index_path, 'faiss.index')
        metadata_file = os.path.join(self.index_path, 'metadata.pkl')
        
        try:
            if os.path.exists(index_file) and os.path.exists(metadata_file):
                # Load existing index
                self.index = faiss.read_index(index_file)
                
                with open(metadata_file, 'rb') as f:
                    data = pickle.load(f)
                    self.id_map = data.get('id_map', {})
                    self.metadata_map = data.get('metadata_map', {})
                    self.next_id = data.get('next_id', 0)
                
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            else:
                # Create new index
                self.create_new_index()
                logger.info("Created new FAISS index")
                
            INDEX_SIZE.set(self.index.ntotal)
            
        except Exception as e:
            logger.error(f"Failed to load index, creating new one: {e}")
            self.create_new_index()
    
    def create_new_index(self):
        """Create a new FAISS index"""
        # Use IVF index for better performance with large datasets
        nlist = min(4096, max(1, int(np.sqrt(self.max_vectors))))
        quantizer = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        
        # Train the index with random vectors if it's a new index
        if not self.index.is_trained:
            # FAISS requires at least 39x the number of centroids for training
            min_training_points = max(nlist * 39, 40000)
            training_vectors = np.random.random((min_training_points, self.dimension)).astype('float32')
            # Normalize vectors for cosine similarity
            faiss.normalize_L2(training_vectors)
            self.index.train(training_vectors)
            logger.info(f"Trained FAISS index with {min_training_points} vectors for {nlist} centroids")
        
        self.index.nprobe = min(10, nlist)  # Default search scope
        
        self.id_map = {}
        self.metadata_map = {}
        self.next_id = 0
    
    async def save_index(self):
        """Save index and metadata to disk"""
        try:
            index_file = os.path.join(self.index_path, 'faiss.index')
            metadata_file = os.path.join(self.index_path, 'metadata.pkl')
            
            # Save FAISS index
            faiss.write_index(self.index, index_file)
            
            # Save metadata
            metadata = {
                'id_map': self.id_map,
                'metadata_map': self.metadata_map,
                'next_id': self.next_id
            }
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            # Cache in Redis
            if self.redis_client:
                await self.redis_client.set('faiss_stats', json.dumps({
                    'total_vectors': self.index.ntotal,
                    'last_updated': datetime.now().isoformat()
                }))
            
            logger.info(f"Saved index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
    
    async def add_vectors(self, vectors_data: List[VectorData]) -> List[int]:
        """Add vectors to the index"""
        try:
            if not vectors_data:
                return []
            
            # Prepare vectors
            vectors = np.array([v.vector for v in vectors_data], dtype=np.float32)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(vectors)
            
            # Add to index
            start_id = self.index.ntotal
            self.index.add(vectors)
            
            # Update mappings
            ids = []
            for i, vector_data in enumerate(vectors_data):
                faiss_id = start_id + i
                custom_id = vector_data.id
                
                self.id_map[faiss_id] = custom_id
                self.metadata_map[custom_id] = vector_data.metadata
                ids.append(faiss_id)
            
            # Update metrics
            INDEX_SIZE.set(self.index.ntotal)
            ADD_REQUESTS.inc(len(vectors_data))
            
            # Save periodically
            if self.index.ntotal % 1000 == 0:
                await self.save_index()
            
            logger.info(f"Added {len(vectors_data)} vectors to index")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to add vectors: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to add vectors: {e}")
    
    async def search_vectors(self, query: SearchRequest) -> List[SearchResult]:
        """Search for similar vectors"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            if self.index.ntotal == 0:
                return []
            
            # Prepare query vector
            query_vector = np.array([query.vector], dtype=np.float32)
            faiss.normalize_L2(query_vector)
            
            # Set search parameters
            if query.nprobe:
                original_nprobe = self.index.nprobe
                self.index.nprobe = min(query.nprobe, self.index.nlist)
            
            # Perform search
            k = min(query.k, self.index.ntotal)
            distances, indices = self.index.search(query_vector, k)
            
            # Restore original nprobe
            if query.nprobe:
                self.index.nprobe = original_nprobe
            
            # Prepare results
            results = []
            for i, (distance, faiss_idx) in enumerate(zip(distances[0], indices[0])):
                if faiss_idx == -1:  # Invalid result
                    continue
                
                custom_id = self.id_map.get(faiss_idx, f"unknown_{faiss_idx}")
                metadata = self.metadata_map.get(custom_id, {})
                
                results.append(SearchResult(
                    id=custom_id,
                    score=float(distance),
                    metadata=metadata
                ))
            
            # Update metrics
            search_time = asyncio.get_event_loop().time() - start_time
            SEARCH_LATENCY.observe(search_time)
            SEARCH_REQUESTS.inc()
            
            logger.info(f"Search completed in {search_time:.3f}s, found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise HTTPException(status_code=500, detail=f"Search failed: {e}")
    
    async def get_stats(self) -> Dict:
        """Get index statistics"""
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'index_type': type(self.index).__name__ if self.index else 'None',
            'nprobe': self.index.nprobe if hasattr(self.index, 'nprobe') else None,
            'is_trained': self.index.is_trained if self.index else False,
            'memory_usage_mb': self.estimate_memory_usage(),
            'last_updated': datetime.now().isoformat()
        }
    
    def estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        if not self.index:
            return 0.0
        
        # Rough estimation
        vector_size = self.index.ntotal * self.dimension * 4  # 4 bytes per float32
        metadata_size = len(str(self.metadata_map).encode('utf-8'))
        total_bytes = vector_size + metadata_size
        
        MEMORY_USAGE.set(total_bytes)
        return total_bytes / (1024 * 1024)  # Convert to MB

# Initialize service
faiss_service = FAISSVectorService()

# FastAPI app
app = FastAPI(
    title="SutazAI FAISS Vector Service",
    description="High-performance vector similarity search using FAISS",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    await faiss_service.initialize()

@app.on_event("shutdown")
async def shutdown_event():
    await faiss_service.save_index()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "faiss-vector",
        "index_ready": faiss_service.index is not None
    }

@app.get("/metrics")
async def get_prometheus_metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.post("/vectors", response_model=List[int])
async def add_vectors(vectors: List[VectorData]):
    """Add vectors to the index"""
    return await faiss_service.add_vectors(vectors)

@app.post("/search", response_model=List[SearchResult])
async def search_vectors(query: SearchRequest):
    """Search for similar vectors"""
    return await faiss_service.search_vectors(query)

@app.get("/stats")
async def get_stats():
    """Get index statistics"""
    return await faiss_service.get_stats()

@app.delete("/reset")
async def reset_index():
    """Reset the entire index (use with caution!)"""
    try:
        faiss_service.create_new_index()
        await faiss_service.save_index()
        INDEX_SIZE.set(0)
        return {"status": "Index reset successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset index: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)