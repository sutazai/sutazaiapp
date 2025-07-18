#!/bin/bash

# SutazAI AGI/ASI Autonomous System - Complete Deployment Script V9
# Implements all required models and agents with proper integration

set -e

echo "🚀 Starting SutazAI AGI/ASI Complete Deployment V9..."

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

error_log() {
    echo -e "${RED}[ERROR $(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warning_log() {
    echo -e "${YELLOW}[WARNING $(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

# Create necessary directories
log "Creating directory structure..."
mkdir -p data/{models,vector,workspace,logs,monitoring}
mkdir -p config/{prometheus,grafana,nginx}
mkdir -p ssl

# Stop existing containers to prevent conflicts
log "Stopping existing containers..."
docker-compose down --remove-orphans 2>/dev/null || true

# Remove dangling containers and networks
log "Cleaning up Docker environment..."
docker system prune -f --volumes 2>/dev/null || true

# Install Ollama if not present
if ! command -v ollama &> /dev/null; then
    log "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    systemctl enable ollama
    systemctl start ollama
    sleep 5
fi

# Build all required Docker images
log "Building Docker images for AI services..."

# Enhanced Model Manager with DeepSeek integration
cat > docker/enhanced-model-manager/requirements.txt << EOF
fastapi==0.104.1
uvicorn==0.24.0
requests==2.31.0
torch==2.1.0
transformers==4.35.0
sentence-transformers==2.2.2
accelerate==0.24.1
bitsandbytes==0.41.1
peft==0.6.0
deepspeed==0.12.0
xformers==0.0.22
flash-attn==2.3.3
optimum==1.14.0
auto-gptq==0.5.1
datasets==2.14.6
tokenizers==0.15.0
pynvml==11.5.0
psutil==5.9.0
aiofiles==23.2.1
httpx==0.25.0
tqdm==4.66.1
huggingface-hub==0.19.0
safetensors==0.4.0
EOF

# Enhanced Model Manager service
cat > docker/enhanced-model-manager/enhanced_model_service.py << 'EOF'
#!/usr/bin/env python3
"""
Enhanced Model Manager with DeepSeek, Qwen3, and advanced model support
"""

import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional, Any
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import requests
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    AutoConfig, BitsAndBytesConfig,
    pipeline, TextGenerationPipeline
)
import psutil
from accelerate import infer_auto_device_map, dispatch_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SutazAI Enhanced Model Manager", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ModelRequest(BaseModel):
    model_name: str
    prompt: str
    max_tokens: int = 1000
    temperature: float = 0.7
    stream: bool = False

class ModelConfig(BaseModel):
    name: str
    model_path: str
    device: str = "auto"
    quantization: Optional[str] = None
    max_memory: Optional[Dict[str, str]] = None

class ModelManager:
    def __init__(self):
        self.loaded_models: Dict[str, Any] = {}
        self.model_configs = {
            "deepseek-r1:8b": {
                "path": "deepseek-ai/deepseek-r1-distill-llama-8b",
                "type": "causal_lm",
                "quantization": "8bit"
            },
            "qwen3:8b": {
                "path": "Qwen/Qwen2.5-Coder-7B-Instruct",
                "type": "causal_lm",
                "quantization": "8bit"
            },
            "deepseek-coder-33b": {
                "path": "deepseek-ai/deepseek-coder-33b-instruct",
                "type": "causal_lm",
                "quantization": "4bit"
            },
            "llama2-7b": {
                "path": "meta-llama/Llama-2-7b-chat-hf",
                "type": "causal_lm",
                "quantization": "8bit"
            }
        }
        self.ollama_models = set()
        self._initialize_ollama()

    def _initialize_ollama(self):
        """Initialize Ollama models"""
        try:
            # Pull required models through Ollama
            models_to_pull = [
                "deepseek-r1:8b",
                "qwen2.5-coder:7b",
                "llama2:7b",
                "codellama:7b"
            ]
            
            for model in models_to_pull:
                try:
                    logger.info(f"Pulling Ollama model: {model}")
                    os.system(f"ollama pull {model}")
                    self.ollama_models.add(model)
                except Exception as e:
                    logger.warning(f"Failed to pull {model}: {e}")
                    
        except Exception as e:
            logger.error(f"Ollama initialization failed: {e}")

    async def load_model(self, model_name: str) -> bool:
        """Load a model into memory"""
        if model_name in self.loaded_models:
            return True

        try:
            if model_name in self.ollama_models:
                # Model handled by Ollama
                return True
                
            config = self.model_configs.get(model_name)
            if not config:
                raise HTTPException(status_code=404, f"Model {model_name} not configured")

            logger.info(f"Loading model: {model_name}")
            
            # Configure quantization
            quantization_config = None
            if config.get("quantization") == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif config.get("quantization") == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

            # Load tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(config["path"])
            
            model = AutoModelForCausalLM.from_pretrained(
                config["path"],
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )

            # Create pipeline
            generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto"
            )

            self.loaded_models[model_name] = {
                "model": model,
                "tokenizer": tokenizer,
                "generator": generator,
                "config": config
            }

            logger.info(f"Successfully loaded model: {model_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False

    async def generate_text(self, model_name: str, prompt: str, **kwargs) -> str:
        """Generate text using specified model"""
        
        # Handle Ollama models
        if model_name in self.ollama_models:
            return await self._generate_ollama(model_name, prompt, **kwargs)
        
        # Handle local models
        if model_name not in self.loaded_models:
            await self.load_model(model_name)
        
        if model_name not in self.loaded_models:
            raise HTTPException(status_code=500, f"Failed to load model {model_name}")

        try:
            generator = self.loaded_models[model_name]["generator"]
            
            result = generator(
                prompt,
                max_length=kwargs.get("max_tokens", 1000),
                temperature=kwargs.get("temperature", 0.7),
                do_sample=True,
                pad_token_id=generator.tokenizer.eos_token_id
            )
            
            return result[0]["generated_text"][len(prompt):]
            
        except Exception as e:
            logger.error(f"Generation failed for {model_name}: {e}")
            raise HTTPException(status_code=500, f"Generation failed: {e}")

    async def _generate_ollama(self, model_name: str, prompt: str, **kwargs) -> str:
        """Generate text using Ollama API"""
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": kwargs.get("temperature", 0.7),
                        "num_predict": kwargs.get("max_tokens", 1000)
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                return response.json().get("response", "")
            else:
                raise HTTPException(status_code=response.status_code, "Ollama generation failed")
                
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise HTTPException(status_code=500, f"Ollama generation failed: {e}")

# Initialize model manager
model_manager = ModelManager()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    gpu_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if gpu_available else 0
    
    return {
        "status": "healthy",
        "models_loaded": len(model_manager.loaded_models),
        "ollama_models": len(model_manager.ollama_models),
        "gpu_available": gpu_available,
        "gpu_count": gpu_count,
        "memory_usage": psutil.virtual_memory().percent,
        "cpu_usage": psutil.cpu_percent()
    }

@app.post("/generate")
async def generate_text(request: ModelRequest):
    """Generate text using specified model"""
    try:
        result = await model_manager.generate_text(
            request.model_name,
            request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return {
            "model": request.model_name,
            "response": result,
            "prompt_tokens": len(request.prompt.split()),
            "completion_tokens": len(result.split())
        }
        
    except Exception as e:
        logger.error(f"Generation request failed: {e}")
        raise HTTPException(status_code=500, str(e))

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "loaded_models": list(model_manager.loaded_models.keys()),
        "available_models": list(model_manager.model_configs.keys()),
        "ollama_models": list(model_manager.ollama_models)
    }

@app.post("/models/{model_name}/load")
async def load_model_endpoint(model_name: str, background_tasks: BackgroundTasks):
    """Load a specific model"""
    background_tasks.add_task(model_manager.load_model, model_name)
    return {"message": f"Loading model {model_name} in background"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8090)
EOF

# Create FAISS vector service
cat > docker/faiss/faiss_service.py << 'EOF'
#!/usr/bin/env python3
"""
FAISS Vector Similarity Search Service for SutazAI
"""

import json
import logging
import numpy as np
import faiss
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from sentence_transformers import SentenceTransformer
import pickle
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SutazAI FAISS Vector Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VectorSearchRequest(BaseModel):
    query: str
    k: int = 5
    index_name: str = "default"

class VectorAddRequest(BaseModel):
    texts: List[str]
    metadata: Optional[List[Dict[str, Any]]] = None
    index_name: str = "default"

class FAISSManager:
    def __init__(self):
        self.indexes: Dict[str, faiss.Index] = {}
        self.metadata: Dict[str, List[Dict[str, Any]]] = {}
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.data_path = "/data/faiss_indexes"
        os.makedirs(self.data_path, exist_ok=True)
        self._load_indexes()

    def _load_indexes(self):
        """Load existing indexes from disk"""
        try:
            for file in os.listdir(self.data_path):
                if file.endswith('.index'):
                    index_name = file[:-6]  # Remove .index extension
                    index_path = os.path.join(self.data_path, file)
                    metadata_path = os.path.join(self.data_path, f"{index_name}.metadata")
                    
                    self.indexes[index_name] = faiss.read_index(index_path)
                    
                    if os.path.exists(metadata_path):
                        with open(metadata_path, 'rb') as f:
                            self.metadata[index_name] = pickle.load(f)
                    else:
                        self.metadata[index_name] = []
                        
                    logger.info(f"Loaded index: {index_name}")
                    
        except Exception as e:
            logger.error(f"Error loading indexes: {e}")

    def _save_index(self, index_name: str):
        """Save index to disk"""
        try:
            index_path = os.path.join(self.data_path, f"{index_name}.index")
            metadata_path = os.path.join(self.data_path, f"{index_name}.metadata")
            
            faiss.write_index(self.indexes[index_name], index_path)
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata[index_name], f)
                
            logger.info(f"Saved index: {index_name}")
            
        except Exception as e:
            logger.error(f"Error saving index {index_name}: {e}")

    def create_index(self, index_name: str, dimension: int = 384):
        """Create a new FAISS index"""
        if index_name not in self.indexes:
            # Use IVF index for better performance with large datasets
            nlist = 100  # Number of clusters
            quantizer = faiss.IndexFlatL2(dimension)
            self.indexes[index_name] = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.metadata[index_name] = []
            logger.info(f"Created new index: {index_name}")

    def add_vectors(self, index_name: str, texts: List[str], metadata: Optional[List[Dict[str, Any]]] = None):
        """Add vectors to index"""
        # Encode texts to vectors
        vectors = self.encoder.encode(texts)
        
        # Create index if it doesn't exist
        if index_name not in self.indexes:
            self.create_index(index_name, vectors.shape[1])
        
        index = self.indexes[index_name]
        
        # Train index if it's not trained (for IVF indexes)
        if not index.is_trained:
            index.train(vectors.astype('float32'))
        
        # Add vectors
        index.add(vectors.astype('float32'))
        
        # Add metadata
        if metadata:
            self.metadata[index_name].extend(metadata)
        else:
            self.metadata[index_name].extend([{"text": text} for text in texts])
        
        # Save index
        self._save_index(index_name)
        
        return len(vectors)

    def search(self, index_name: str, query: str, k: int = 5):
        """Search for similar vectors"""
        if index_name not in self.indexes:
            raise HTTPException(status_code=404, detail=f"Index {index_name} not found")
        
        # Encode query
        query_vector = self.encoder.encode([query])
        
        # Search
        distances, indices = self.indexes[index_name].search(query_vector.astype('float32'), k)
        
        # Prepare results
        results = []
        for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
            if idx != -1:  # Valid result
                metadata = self.metadata[index_name][idx] if idx < len(self.metadata[index_name]) else {}
                results.append({
                    "score": float(1.0 / (1.0 + distance)),  # Convert distance to similarity score
                    "metadata": metadata,
                    "index": int(idx)
                })
        
        return results

faiss_manager = FAISSManager()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "indexes": list(faiss_manager.indexes.keys()),
        "total_vectors": sum(index.ntotal for index in faiss_manager.indexes.values())
    }

@app.post("/search")
async def search_vectors(request: VectorSearchRequest):
    """Search for similar vectors"""
    try:
        results = faiss_manager.search(request.index_name, request.query, request.k)
        return {
            "query": request.query,
            "results": results,
            "total_found": len(results)
        }
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add")
async def add_vectors(request: VectorAddRequest):
    """Add vectors to index"""
    try:
        count = faiss_manager.add_vectors(request.index_name, request.texts, request.metadata)
        return {
            "message": f"Added {count} vectors to index {request.index_name}",
            "vectors_added": count
        }
    except Exception as e:
        logger.error(f"Add vectors failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/indexes")
async def list_indexes():
    """List all available indexes"""
    indexes_info = {}
    for name, index in faiss_manager.indexes.items():
        indexes_info[name] = {
            "total_vectors": index.ntotal,
            "dimension": index.d,
            "is_trained": index.is_trained
        }
    return indexes_info

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8088)
EOF

# Update FAISS Dockerfile
cat > docker/faiss/Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY faiss_service.py .
COPY health_check.py .

EXPOSE 8088

CMD ["python", "faiss_service.py"]
EOF

# Build all containers
log "Building enhanced Docker containers..."

# Build Enhanced Model Manager
docker build -t sutazai-enhanced-model-manager:latest docker/enhanced-model-manager/

# Build FAISS service
docker build -t sutazai-faiss:latest docker/faiss/

# Start core infrastructure
log "Starting core infrastructure..."
docker-compose up -d postgres redis qdrant chromadb

# Wait for databases to be ready
log "Waiting for databases to be ready..."
sleep 30

# Start Ollama and pull models
log "Starting Ollama and pulling models..."
docker-compose up -d ollama

# Wait for Ollama to start
sleep 15

# Pull required Ollama models
log "Pulling Ollama models..."
docker exec sutazai-ollama ollama pull deepseek-r1:8b || warning_log "Failed to pull deepseek-r1:8b"
docker exec sutazai-ollama ollama pull qwen2.5-coder:7b || warning_log "Failed to pull qwen2.5-coder:7b"  
docker exec sutazai-ollama ollama pull llama2:7b || warning_log "Failed to pull llama2:7b"
docker exec sutazai-ollama ollama pull codellama:7b || warning_log "Failed to pull codellama:7b"

# Start AI services
log "Starting AI services..."
docker-compose up -d \
    enhanced-model-manager \
    faiss \
    autogen \
    langflow \
    dify \
    pytorch \
    tensorflow \
    jax

# Start agent services
log "Starting agent services..."
docker-compose up -d \
    browser-use \
    awesome-code-ai \
    open-webui

# Start backend and frontend
log "Starting SutazAI backend and frontend..."
docker-compose up -d sutazai-backend sutazai-streamlit

# Start monitoring
log "Starting monitoring stack..."
docker-compose up -d prometheus grafana node-exporter

# Start reverse proxy
log "Starting reverse proxy..."
docker-compose up -d nginx

# Health check all services
log "Performing health checks..."
sleep 30

services=(
    "sutazai-backend:8000"
    "sutazai-streamlit:8501"
    "enhanced-model-manager:8090"
    "faiss:8088"
    "ollama:11434"
    "qdrant:6333"
    "chromadb:8000"
)

for service in "${services[@]}"; do
    name=$(echo $service | cut -d: -f1)
    port=$(echo $service | cut -d: -f2)
    
    if curl -sf http://localhost:$port/health >/dev/null 2>&1 || curl -sf http://localhost:$port/healthz >/dev/null 2>&1; then
        log "✅ $name is healthy"
    else
        warning_log "⚠️  $name health check failed"
    fi
done

# Display deployment summary
log "🎉 SutazAI AGI/ASI System Deployment Complete!"
echo
echo "🌐 Access Points:"
echo "  • Main UI: http://192.168.131.128:8501"
echo "  • Open WebUI: http://192.168.131.128:8089"
echo "  • API Backend: http://192.168.131.128:8000"
echo "  • Model Manager: http://192.168.131.128:8098"
echo "  • Ollama: http://192.168.131.128:11434"
echo "  • Vector DB (Qdrant): http://192.168.131.128:6333"
echo "  • FAISS Search: http://192.168.131.128:8096"
echo "  • Monitoring (Grafana): http://192.168.131.128:3000"
echo "  • Prometheus: http://192.168.131.128:9090"
echo
echo "🤖 Available Models:"
echo "  • DeepSeek-R1 8B (deepseek-r1:8b)"
echo "  • Qwen 2.5 Coder 7B (qwen2.5-coder:7b)"
echo "  • Llama 2 7B (llama2:7b)"
echo "  • CodeLlama 7B (codellama:7b)"
echo
echo "🔧 Agent Services:"
echo "  • AutoGen: http://192.168.131.128:8092"
echo "  • LangFlow: http://192.168.131.128:7860"
echo "  • Dify: http://192.168.131.128:5001"
echo "  • Browser Use: http://192.168.131.128:8088"
echo "  • PyTorch Service: http://192.168.131.128:8093"
echo "  • TensorFlow Service: http://192.168.131.128:8094"
echo "  • JAX Service: http://192.168.131.128:8095"
echo
echo "📊 System Status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep sutazai

log "Deployment completed successfully! 🚀"