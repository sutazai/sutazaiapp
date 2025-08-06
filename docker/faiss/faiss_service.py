import faiss
import numpy as np
import uvicorn
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="FAISS Vector Database Service", version="1.0.0")

# Global index storage
indexes = {}
data_path = os.environ.get('FAISS_DATA_PATH', '/data/faiss_indexes')
os.makedirs(data_path, exist_ok=True)

class HealthResponse(BaseModel):
    status: str

class CreateIndexRequest(BaseModel):
    name: str
    dimension: int
    type: Optional[str] = 'IVFFlat'

class CreateIndexResponse(BaseModel):
    status: str
    name: str

class AddVectorsRequest(BaseModel):
    index: str
    vectors: List[List[float]]

class AddVectorsResponse(BaseModel):
    status: str
    total: int

class SearchRequest(BaseModel):
    index: str
    query: List[float]
    k: Optional[int] = 10

class SearchResponse(BaseModel):
    distances: List[float]
    indices: List[int]

@app.get('/health', response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy")

@app.post('/create_index', response_model=CreateIndexResponse)
async def create_index(request: CreateIndexRequest):
    index_name = request.name
    dimension = request.dimension
    index_type = request.type
    
    if index_type == 'IVFFlat':
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, 100)
    else:
        index = faiss.IndexFlatL2(dimension)
    
    indexes[index_name] = index
    return CreateIndexResponse(status="created", name=index_name)

@app.post('/add_vectors', response_model=AddVectorsResponse)
async def add_vectors(request: AddVectorsRequest):
    index_name = request.index
    vectors = np.array(request.vectors, dtype=np.float32)
    
    if index_name not in indexes:
        raise HTTPException(status_code=404, detail="Index not found")
    
    index = indexes[index_name]
    if not index.is_trained:
        index.train(vectors)
    
    index.add(vectors)
    return AddVectorsResponse(status="added", total=index.ntotal)

@app.post('/search', response_model=SearchResponse)
async def search(request: SearchRequest):
    index_name = request.index
    query_vector = np.array(request.query, dtype=np.float32).reshape(1, -1)
    k = request.k
    
    if index_name not in indexes:
        raise HTTPException(status_code=404, detail="Index not found")
    
    index = indexes[index_name]
    distances, indices = index.search(query_vector, k)
    
    return SearchResponse(
        distances=distances[0].tolist(),
        indices=indices[0].tolist()
    )

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, log_level="info")
