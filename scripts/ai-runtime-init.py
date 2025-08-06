#!/usr/bin/env python3
"""
Shared AI Runtime Initialization Script
Manages model loading, caching, and resource pooling for all AI services
"""

import os
import sys
import json
import logging
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime

import torch
from transformers import AutoModel, AutoTokenizer, pipeline
from langchain.llms import Ollama
from langchain.embeddings import OllamaEmbeddings
from langchain.memory import ConversationSummaryMemory
from langchain.vectorstores import Chroma, FAISS
import chromadb
from qdrant_client import QdrantClient
import redis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment configuration
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://ollama:11434')
REDIS_URL = os.getenv('REDIS_URL', 'redis://redis:6379/0')
MODELS_PATH = Path('/app/models')
SHARED_PATH = Path('/app/shared')

# Ensure directories exist
MODELS_PATH.mkdir(parents=True, exist_ok=True)
SHARED_PATH.mkdir(parents=True, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="Sutazai AI Runtime", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model registry
MODEL_REGISTRY = {}
EMBEDDINGS_CACHE = {}
VECTOR_STORES = {}

# Redis connection
redis_client = None

class ModelRequest(BaseModel):
    model_name: str
    task: str
    input_text: str
    parameters: Optional[Dict[str, Any]] = {}

class EmbeddingRequest(BaseModel):
    texts: list[str]
    model_name: str = "nomic-embed-text"

class VectorStoreRequest(BaseModel):
    store_type: str  # "chroma", "faiss", "qdrant"
    collection_name: str
    action: str  # "create", "add", "search"
    texts: Optional[list[str]] = None
    query: Optional[str] = None
    k: Optional[int] = 5

def init_redis():
    """Initialize Redis connection for caching"""
    global redis_client
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        redis_client = None

def get_or_load_model(model_name: str, task: str):
    """Load model with caching and lazy loading"""
    cache_key = f"{model_name}:{task}"
    
    if cache_key in MODEL_REGISTRY:
        logger.info(f"Using cached model: {cache_key}")
        return MODEL_REGISTRY[cache_key]
    
    logger.info(f"Loading model: {cache_key}")
    
    try:
        if task == "ollama":
            model = Ollama(
                base_url=OLLAMA_BASE_URL,
                model=model_name,
                temperature=0.7,
                num_thread=4,
                num_gpu=0  # CPU only
            )
        elif task in ["text-generation", "summarization", "question-answering"]:
            model = pipeline(
                task,
                model=model_name,
                device=-1,  # CPU
                torch_dtype=torch.float32
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            model = (model, tokenizer)
        
        MODEL_REGISTRY[cache_key] = model
        logger.info(f"Model loaded successfully: {cache_key}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model {cache_key}: {e}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

def get_or_create_embeddings(model_name: str = "nomic-embed-text"):
    """Get or create embeddings model"""
    if model_name in EMBEDDINGS_CACHE:
        return EMBEDDINGS_CACHE[model_name]
    
    embeddings = OllamaEmbeddings(
        base_url=OLLAMA_BASE_URL,
        model=model_name
    )
    EMBEDDINGS_CACHE[model_name] = embeddings
    return embeddings

def get_or_create_vector_store(store_type: str, collection_name: str):
    """Get or create vector store instance"""
    store_key = f"{store_type}:{collection_name}"
    
    if store_key in VECTOR_STORES:
        return VECTOR_STORES[store_key]
    
    embeddings = get_or_create_embeddings()
    
    if store_type == "chroma":
        client = chromadb.HttpClient(host="chromadb", port=8000)
        store = Chroma(
            client=client,
            collection_name=collection_name,
            embedding_function=embeddings
        )
    elif store_type == "faiss":
        # Create or load FAISS index
        index_path = SHARED_PATH / f"faiss_{collection_name}.index"
        if index_path.exists():
            store = FAISS.load_local(str(index_path), embeddings)
        else:
            store = FAISS.from_texts([""], embeddings)
    elif store_type == "qdrant":
        client = QdrantClient(host="qdrant", port=6333)
        store = None  # Implement Qdrant integration as needed
    else:
        raise ValueError(f"Unknown vector store type: {store_type}")
    
    VECTOR_STORES[store_key] = store
    return store

@app.on_event("startup")
async def startup_event():
    """Initialize runtime on startup"""
    logger.info("Starting Sutazai AI Runtime...")
    init_redis()
    
    # Pre-load essential models
    try:
        # Load a small model for testing
        get_or_load_model("gpt-oss", "ollama")
        logger.info("Essential models pre-loaded")
    except Exception as e:
        logger.warning(f"Failed to pre-load models: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models_loaded": len(MODEL_REGISTRY),
        "embeddings_cached": len(EMBEDDINGS_CACHE),
        "vector_stores": len(VECTOR_STORES),
        "redis_connected": redis_client is not None
    }

@app.post("/inference")
async def run_inference(request: ModelRequest):
    """Run model inference"""
    try:
        model = get_or_load_model(request.model_name, request.task)
        
        if request.task == "ollama":
            result = model(request.input_text)
        else:
            result = model(request.input_text, **request.parameters)
        
        # Cache result in Redis if available
        if redis_client:
            cache_key = f"inference:{request.model_name}:{hash(request.input_text)}"
            redis_client.setex(cache_key, 3600, json.dumps(result))
        
        return {"result": result, "model": request.model_name, "task": request.task}
        
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embeddings")
async def generate_embeddings(request: EmbeddingRequest):
    """Generate embeddings for texts"""
    try:
        embeddings_model = get_or_create_embeddings(request.model_name)
        embeddings = embeddings_model.embed_documents(request.texts)
        
        return {
            "embeddings": embeddings,
            "model": request.model_name,
            "dimensions": len(embeddings[0]) if embeddings else 0
        }
        
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vector_store")
async def manage_vector_store(request: VectorStoreRequest):
    """Manage vector store operations"""
    try:
        store = get_or_create_vector_store(request.store_type, request.collection_name)
        
        if request.action == "create":
            return {"status": "created", "store_type": request.store_type}
            
        elif request.action == "add" and request.texts:
            if request.store_type == "faiss":
                store.add_texts(request.texts)
                # Save FAISS index
                index_path = SHARED_PATH / f"faiss_{request.collection_name}.index"
                store.save_local(str(index_path))
            else:
                store.add_texts(request.texts)
            return {"status": "added", "count": len(request.texts)}
            
        elif request.action == "search" and request.query:
            results = store.similarity_search(request.query, k=request.k)
            return {"results": [doc.page_content for doc in results]}
            
        else:
            raise ValueError(f"Invalid action: {request.action}")
            
    except Exception as e:
        logger.error(f"Vector store error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/unload_model")
async def unload_model(model_name: str, task: str):
    """Unload model from memory"""
    cache_key = f"{model_name}:{task}"
    if cache_key in MODEL_REGISTRY:
        del MODEL_REGISTRY[cache_key]
        # Force garbage collection
        import gc
        gc.collect()
        return {"status": "unloaded", "model": cache_key}
    return {"status": "not_found", "model": cache_key}

@app.get("/models")
async def list_models():
    """List loaded models"""
    return {
        "models": list(MODEL_REGISTRY.keys()),
        "embeddings": list(EMBEDDINGS_CACHE.keys()),
        "vector_stores": list(VECTOR_STORES.keys())
    }

@app.get("/metrics")
async def get_metrics():
    """Get runtime metrics"""
    import psutil
    process = psutil.Process()
    
    return {
        "memory_usage_mb": process.memory_info().rss / 1024 / 1024,
        "cpu_percent": process.cpu_percent(interval=1),
        "models_loaded": len(MODEL_REGISTRY),
        "cache_size": len(EMBEDDINGS_CACHE) + len(VECTOR_STORES),
        "uptime_seconds": (datetime.now() - datetime.fromtimestamp(process.create_time())).total_seconds()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=2)