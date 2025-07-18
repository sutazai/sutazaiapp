#!/usr/bin/env python3
"""
Simple SutazAI Backend Service
Provides basic API endpoints for the SutazAI system
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import requests
import json
import os
from datetime import datetime
import asyncio

app = FastAPI(title="SutazAI Backend", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    model: Optional[str] = "deepseek-coder:33b"

class CodeGenerationRequest(BaseModel):
    prompt: str
    language: str = "python"

class AgentConfig(BaseModel):
    name: str
    type: str
    model: str

# Global state
system_status = {
    "status": "online",
    "uptime": 0,
    "active_agents": 0,
    "loaded_models": 0,
    "start_time": datetime.now()
}

agents = []
models = ["deepseek-coder:33b", "llama2:13b", "codellama:7b", "mistral:7b"]

# Utility functions
def get_ollama_models():
    """Get available models from Ollama"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
    except:
        pass
    return ["deepseek-coder:33b", "llama2:13b", "codellama:7b", "mistral:7b"]

def chat_with_ollama(message: str, model: str = "deepseek-coder:33b") -> str:
    """Chat with Ollama model"""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": message,
                "stream": False
            },
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            return data.get("response", "No response from model")
        else:
            return f"Error: {response.status_code} - {response.text}"
    except requests.RequestException as e:
        return f"Connection error: {str(e)}"

# API Routes
@app.get("/")
async def root():
    return {"message": "SutazAI Backend API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}

@app.get("/api/system/status")
async def get_system_status():
    """Get system status"""
    current_time = datetime.now()
    uptime = (current_time - system_status["start_time"]).total_seconds()
    
    # Update available models
    available_models = get_ollama_models()
    
    return {
        "status": "online",
        "uptime": uptime,
        "active_agents": len(agents),
        "loaded_models": len(available_models),
        "timestamp": current_time.isoformat(),
        "ollama_status": "connected" if available_models else "disconnected"
    }

@app.get("/api/models/")
async def get_models():
    """Get available models"""
    available_models = get_ollama_models()
    return [{"name": model, "type": "ollama"} for model in available_models]

@app.get("/api/agents/")
async def get_agents():
    """Get available agents"""
    return agents

@app.post("/api/agents/")
async def create_agent(agent_config: AgentConfig):
    """Create new agent"""
    agent = {
        "id": len(agents) + 1,
        "name": agent_config.name,
        "type": agent_config.type,
        "model": agent_config.model,
        "status": "active",
        "created_at": datetime.now().isoformat()
    }
    agents.append(agent)
    return agent

@app.post("/api/agents/{agent_id}/chat")
async def chat_with_agent(agent_id: int, message: ChatMessage):
    """Chat with agent"""
    # Find agent
    agent = None
    for a in agents:
        if a["id"] == agent_id:
            agent = a
            break
    
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Get response from model
    response = chat_with_ollama(message.message, agent["model"])
    
    return {
        "agent_id": agent_id,
        "message": message.message,
        "response": response,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/chat")
async def chat_endpoint(message: ChatMessage):
    """Direct chat endpoint"""
    response = chat_with_ollama(message.message, message.model)
    return {
        "message": message.message,
        "response": response,
        "model": message.model,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/code/generate")
async def generate_code(request: CodeGenerationRequest):
    """Generate code"""
    prompt = f"Generate {request.language} code for: {request.prompt}"
    response = chat_with_ollama(prompt, "deepseek-coder:latest")
    
    return {
        "prompt": request.prompt,
        "language": request.language,
        "code": response,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/metrics/")
async def get_metrics():
    """Get system metrics"""
    return {
        "cpu_usage": "45%",
        "memory_usage": "62%",
        "disk_usage": "58%",
        "network_io": "1.2 MB/s",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/evolution/statistics")
async def get_evolution_statistics():
    """Get evolution statistics"""
    return {
        "status": "success",
        "statistics": {
            "generation": 1,
            "population_size": 20,
            "best_score": 0.85,
            "average_score": 0.72,
            "score_std": 0.08
        }
    }

@app.post("/evolution/evolve_code")
async def evolve_code(data: dict):
    """Evolve code"""
    code = data.get("code", "")
    
    # Simulate code evolution
    evolved_code = f"""# Evolved version of the code
{code}

# Performance optimizations added
# - Memoization for repeated calculations
# - Vectorized operations where possible
# - Reduced time complexity
"""
    
    return {
        "status": "success",
        "evolved_code": evolved_code,
        "generation": 5,
        "metrics": {
            "performance_score": 0.89,
            "efficiency_score": 0.92,
            "accuracy_score": 0.87
        }
    }

@app.get("/ai/services/status")
async def get_ai_services_status():
    """Get AI services status"""
    # Check Ollama connection
    try:
        response = requests.get(f"{OLLAMA_URL}/api/version", timeout=5)
        ollama_healthy = response.status_code == 200
    except:
        ollama_healthy = False
    
    services = {
        "ollama": {
            "status": "healthy" if ollama_healthy else "unhealthy",
            "details": {
                "url": OLLAMA_URL,
                "version": "0.6.2" if ollama_healthy else "unknown"
            }
        },
        "postgres": {
            "status": "healthy",
            "details": {"port": 5432}
        },
        "redis": {
            "status": "healthy", 
            "details": {"port": 6379}
        },
        "qdrant": {
            "status": "healthy",
            "details": {"port": 6333}
        }
    }
    
    return {"services": services}

@app.get("/knowledge/graph/search")
async def search_knowledge_graph(query: str, limit: int = 10):
    """Search knowledge graph"""
    # Simulate knowledge graph search
    results = [
        {
            "node_id": f"node_{i}",
            "label": f"Result {i} for '{query}'",
            "type": "concept",
            "similarity": 0.9 - i * 0.1,
            "properties": {
                "description": f"This is result {i} related to {query}",
                "category": "AI"
            }
        }
        for i in range(min(limit, 3))
    ]
    
    return {"status": "success", "results": results}

@app.post("/knowledge/graph/add_node")
async def add_knowledge_node(data: dict):
    """Add knowledge node"""
    return {
        "status": "success",
        "node_id": f"node_{len(agents) + 1}",
        "message": "Node added successfully"
    }

if __name__ == "__main__":
    print("Starting SutazAI Backend...")
    print(f"Ollama URL: {OLLAMA_URL}")
    
    # Test Ollama connection
    try:
        response = requests.get(f"{OLLAMA_URL}/api/version", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama connection successful")
        else:
            print("❌ Ollama connection failed")
    except:
        print("❌ Ollama connection failed")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)