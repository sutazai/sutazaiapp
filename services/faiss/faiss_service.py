from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import faiss
import numpy as np
import os
import pickle
from typing import List, Dict, Any

app = FastAPI(title="FAISS Vector Service")

# Global index and mapping
index = None
id_map = {}
dimension = 768  # Default dimension

class VectorData(BaseModel):
    id: str
    vector: List[float]

class SearchQuery(BaseModel):
    vector: List[float]
    k: int = 10

class BatchVectorData(BaseModel):
    vectors: List[VectorData]

@app.on_event("startup")
async def startup_event():
    global index, id_map, dimension
    index_path = os.getenv("FAISS_INDEX_PATH", "/data/index")
    
    # Try to load existing index
    if os.path.exists(f"{index_path}.index"):
        try:
            index = faiss.read_index(f"{index_path}.index")
            with open(f"{index_path}.map", "rb") as f:
                id_map = pickle.load(f)
            dimension = index.d
            print(f"Loaded existing index with {index.ntotal} vectors")
        except Exception as e:
            print(f"Error loading index: {e}")
            # Create new index
            index = faiss.IndexFlatL2(dimension)
    else:
        # Create new index
        index = faiss.IndexFlatL2(dimension)
        print(f"Created new index with dimension {dimension}")

@app.on_event("shutdown")
async def shutdown_event():
    # Save index on shutdown
    index_path = os.getenv("FAISS_INDEX_PATH", "/data/index")
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    
    if index and index.ntotal > 0:
        try:
            faiss.write_index(index, f"{index_path}.index")
            with open(f"{index_path}.map", "wb") as f:
                pickle.dump(id_map, f)
            print(f"Saved index with {index.ntotal} vectors")
        except Exception as e:
            print(f"Error saving index: {e}")

@app.post("/index")
async def add_vector(data: VectorData):
    """Add a single vector to the index"""
    global index, id_map
    
    try:
        vector = np.array(data.vector, dtype=np.float32).reshape(1, -1)
        
        # Verify dimension
        if vector.shape[1] != dimension:
            raise HTTPException(
                status_code=400,
                detail=f"Vector dimension {vector.shape[1]} doesn't match index dimension {dimension}"
            )
        
        # Add to index
        idx = index.ntotal
        index.add(vector)
        id_map[idx] = data.id
        
        return {"status": "indexed", "id": data.id, "index": idx}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/index/batch")
async def add_vectors_batch(data: BatchVectorData):
    """Add multiple vectors to the index"""
    global index, id_map
    
    try:
        vectors = []
        ids = []
        
        for vec_data in data.vectors:
            vector = np.array(vec_data.vector, dtype=np.float32)
            vectors.append(vector)
            ids.append(vec_data.id)
        
        vectors_array = np.array(vectors, dtype=np.float32)
        
        # Verify dimension
        if vectors_array.shape[1] != dimension:
            raise HTTPException(
                status_code=400,
                detail=f"Vector dimension {vectors_array.shape[1]} doesn't match index dimension {dimension}"
            )
        
        # Add to index
        start_idx = index.ntotal
        index.add(vectors_array)
        
        # Update mapping
        for i, vec_id in enumerate(ids):
            id_map[start_idx + i] = vec_id
        
        return {
            "status": "indexed", 
            "count": len(vectors),
            "total_vectors": index.ntotal
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_vectors(query: SearchQuery):
    """Search for similar vectors"""
    try:
        if index.ntotal == 0:
            return {"results": [], "message": "Index is empty"}
        
        vector = np.array(query.vector, dtype=np.float32).reshape(1, -1)
        
        # Verify dimension
        if vector.shape[1] != dimension:
            raise HTTPException(
                status_code=400,
                detail=f"Query dimension {vector.shape[1]} doesn't match index dimension {dimension}"
            )
        
        # Adjust k if necessary
        k = min(query.k, index.ntotal)
        
        # Search
        distances, indices = index.search(vector, k)
        
        # Build results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx in id_map:
                results.append({
                    "id": id_map[idx],
                    "distance": float(dist),
                    "rank": i,
                    "index": int(idx)
                })
        
        return {"results": results, "total_results": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/index/{vector_id}")
async def delete_vector(vector_id: str):
    """Delete a vector by ID (note: FAISS doesn't support true deletion)"""
    global id_map
    
    # Find the index for this ID
    idx_to_remove = None
    for idx, vec_id in id_map.items():
        if vec_id == vector_id:
            idx_to_remove = idx
            break
    
    if idx_to_remove is not None:
        del id_map[idx_to_remove]
        return {"status": "deleted", "id": vector_id}
    else:
        raise HTTPException(status_code=404, detail=f"Vector ID {vector_id} not found")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vectors": index.ntotal if index else 0,
        "dimension": dimension,
        "index_type": "IndexFlatL2"
    }

@app.get("/info")
async def get_info():
    """Get information about the index"""
    return {
        "total_vectors": index.ntotal if index else 0,
        "dimension": dimension,
        "index_type": "IndexFlatL2",
        "memory_usage_mb": (index.ntotal * dimension * 4) / (1024 * 1024) if index else 0
    }

@app.post("/clear")
async def clear_index():
    """Clear all vectors from the index"""
    global index, id_map
    
    index = faiss.IndexFlatL2(dimension)
    id_map = {}
    
    return {"status": "cleared", "dimension": dimension}