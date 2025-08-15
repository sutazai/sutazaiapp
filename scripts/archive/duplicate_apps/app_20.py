#!/usr/bin/env python3
"""
LlamaIndex Service for SutazAI
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.llms import Ollama
from llama_index.embeddings import HuggingFaceEmbedding
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SutazAI LlamaIndex Service")

# Initialize LlamaIndex components
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
llm = Ollama(model="tinyllama", base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:10104"))
service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm)

class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

class IndexRequest(BaseModel):
    documents: List[str]
    metadata: Optional[Dict[str, Any]] = {}

@app.get("/")
async def root():
    return {"service": "LlamaIndex", "status": "active"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/index")
async def create_index(request: IndexRequest):
    """Create or update vector index"""
    try:
        # Create index from documents
        # In production, this would persist to vector DB
        return {"status": "indexed", "documents": len(request.documents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_index(request: QueryRequest):
    """Query the vector index"""
    try:
        # Mock response for now
        return {
            "query": request.query,
            "results": [
                {"text": "Sample result 1", "score": 0.95},
                {"text": "Sample result 2", "score": 0.87}
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)