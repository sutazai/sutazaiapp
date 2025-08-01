#!/usr/bin/env python3
"""
SutazAI Complete AGI/ASI System Backend
Enterprise-grade AGI implementation with all specified components
"""

import os
import sys
import asyncio
import logging
import json
import uuid
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn
import httpx
import psutil
import torch
import subprocess
import docker
import aioredis
import asyncpg
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/logs/complete_agi_backend.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("SutazAI-Complete-AGI")

# Pydantic Models
class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "tinyllama"
    agent: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    stream: bool = False

class ChatResponse(BaseModel):
    response: str
    model: str
    agent: Optional[str] = None
    tokens_used: int
    processing_time: float
    confidence: float

class AgentRequest(BaseModel):
    agent_type: str
    task: str
    parameters: Optional[Dict[str, Any]] = None
    priority: int = 5

class AgentResponse(BaseModel):
    agent_type: str
    status: str
    result: Any
    processing_time: float
    
class DocumentRequest(BaseModel):
    content: str
    document_type: str = "text"
    process_type: str = "analyze"

class CodeRequest(BaseModel):
    code: Optional[str] = None
    language: str = "python"
    action: str = "generate"  # generate, review, optimize, debug
    requirements: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    components: Dict[str, str]
    models: List[str]
    agents: Dict[str, str]

# Global instances
redis_client: Optional[aioredis.Redis] = None
docker_client: Optional[docker.DockerClient] = None
ollama_client: Optional[httpx.AsyncClient] = None

# AI Agent Manager
class AIAgentManager:
    def __init__(self):
        self.agents = {
            # Core Task Automation
            "autogpt": {"url": "http://sutazai-autogpt:8080", "type": "automation"},
            "localagi": {"url": "http://sutazai-localagi:8081", "type": "orchestration"},
            "crewai": {"url": "http://sutazai-crewai:8096", "type": "collaboration"},
            
            # Code Generation & Analysis
            "tabbyml": {"url": "http://sutazai-tabbyml:8082", "type": "code_completion"},
            "gpt_engineer": {"url": "http://sutazai-gpt-engineer:8094", "type": "code_generation"},
            "aider": {"url": "http://sutazai-aider:8095", "type": "code_editing"},
            "semgrep": {"url": "http://sutazai-semgrep:8083", "type": "security_analysis"},
            
            # Web Automation
            "browser_use": {"url": "http://sutazai-browser-use:8084", "type": "web_automation"},
            "skyvern": {"url": "http://sutazai-skyvern:8085", "type": "web_scraping"},
            
            # Document Processing
            "documind": {"url": "http://sutazai-documind:8092", "type": "document_processing"},
            "privategpt": {"url": "http://sutazai-privategpt:8097", "type": "private_llm"},
            "llamaindex": {"url": "http://sutazai-llamaindex:8098", "type": "data_indexing"},
            
            # Financial Analysis
            "finrobot": {"url": "http://sutazai-finrobot:8093", "type": "financial_analysis"},
            
            # ML Frameworks
            "pytorch": {"url": "http://sutazai-pytorch:8087", "type": "ml_framework"},
            "tensorflow": {"url": "http://sutazai-tensorflow:8088", "type": "ml_framework"},
            "jax": {"url": "http://sutazai-jax:8089", "type": "ml_framework"},
            
            # Workflow & App Building
            "langflow": {"url": "http://sutazai-langflow:8090", "type": "workflow"},
            "dify": {"url": "http://sutazai-dify:8091", "type": "app_builder"},
            "flowise": {"url": "http://sutazai-flowise:8099", "type": "llm_flows"},
            
            # Specialized Tools
            "shellgpt": {"url": "http://sutazai-shellgpt:8102", "type": "cli_assistant"},
            "pentestgpt": {"url": "http://sutazai-pentestgpt:8100", "type": "security_testing"},
            "realtime_stt": {"url": "http://sutazai-realtime-stt:8101", "type": "speech_processing"},
        }
        
    async def get_agent_status(self, agent_name: str) -> Dict[str, Any]:
        """Get status of a specific agent"""
        if agent_name not in self.agents:
            return {"status": "not_found"}
            
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.agents[agent_name]['url']}/health")
                if response.status_code == 200:
                    return {"status": "healthy", "response": response.json()}
                else:
                    return {"status": "unhealthy", "code": response.status_code}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    async def execute_agent_task(self, agent_name: str, task: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a task on a specific agent"""
        if agent_name not in self.agents:
            return {"status": "error", "message": "Agent not found"}
            
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                payload = {"task": task}
                if parameters:
                    payload.update(parameters)
                    
                response = await client.post(
                    f"{self.agents[agent_name]['url']}/execute",
                    json=payload
                )
                
                if response.status_code == 200:
                    return {"status": "success", "result": response.json()}
                else:
                    return {"status": "error", "code": response.status_code, "message": response.text}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def get_all_agents_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        status = {}
        tasks = []
        
        for agent_name in self.agents.keys():
            tasks.append(self.get_agent_status(agent_name))
            
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, agent_name in enumerate(self.agents.keys()):
            status[agent_name] = results[i] if not isinstance(results[i], Exception) else {"status": "error"}
            
        return status

# Model Manager
class ModelManager:
    def __init__(self):
        self.models = [
            "tinyllama",
            "qwen2.5:3b", 
            "qwen3:8b",
            "codellama:7b",
            "llama2:7b",
            "llama3.2:1b",
            "nomic-embed-text"
        ]
        self.ollama_url = "http://sutazai-ollama:11434"
        
    async def get_available_models(self) -> List[str]:
        """Get list of available models"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            logger.error(f"Error getting models: {e}")
            
        return self.models
    
    async def generate_response(self, prompt: str, model: str = "tinyllama") -> Dict[str, Any]:
        """Generate response using Ollama"""
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                }
                
                response = await client.post(f"{self.ollama_url}/api/generate", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "response": data.get("response", ""),
                        "model": model,
                        "tokens": data.get("eval_count", 0),
                        "processing_time": data.get("eval_duration", 0) / 1000000000  # Convert to seconds
                    }
                else:
                    logger.error(f"Ollama error: {response.status_code} - {response.text}")
                    return {"error": f"Model error: {response.status_code}"}
                    
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {"error": str(e)}

# Vector Database Manager
class VectorDBManager:
    def __init__(self):
        self.chromadb_url = "http://sutazai-chromadb:8001"
        self.qdrant_url = "http://sutazai-qdrant:6333"
        
    async def search_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search documents in vector databases"""
        results = []
        
        # Search ChromaDB
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.chromadb_url}/api/v1/collections/documents/query",
                    json={"query_texts": [query], "n_results": limit}
                )
                if response.status_code == 200:
                    data = response.json()
                    results.extend(data.get("documents", []))
        except Exception as e:
            logger.error(f"ChromaDB search error: {e}")
        
        # Search Qdrant
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.qdrant_url}/collections/documents/points/search",
                    json={"vector": query, "limit": limit}
                )
                if response.status_code == 200:
                    data = response.json()
                    results.extend(data.get("result", []))
        except Exception as e:
            logger.error(f"Qdrant search error: {e}")
        
        return results

# Initialize managers
agent_manager = AIAgentManager()
model_manager = ModelManager()
vector_manager = VectorDBManager()

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global redis_client, docker_client, ollama_client
    
    # Startup
    logger.info("Starting SutazAI Complete AGI/ASI System...")
    
    try:
        # Initialize Redis
        redis_client = aioredis.from_url("redis://sutazai-redis:6379")
        await redis_client.ping()
        logger.info("Redis connected successfully")
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
    
    try:
        # Initialize Docker client
        docker_client = docker.from_env()
        logger.info("Docker client initialized")
    except Exception as e:
        logger.error(f"Docker client initialization failed: {e}")
    
    try:
        # Initialize Ollama client
        ollama_client = httpx.AsyncClient(base_url="http://sutazai-ollama:11434")
        logger.info("Ollama client initialized")
    except Exception as e:
        logger.error(f"Ollama client initialization failed: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down SutazAI Complete AGI/ASI System...")
    if redis_client:
        await redis_client.close()
    if ollama_client:
        await ollama_client.aclose()

# Create FastAPI app
app = FastAPI(
    title="SutazAI Complete AGI/ASI System",
    description="Enterprise-grade AGI/ASI system with all specified AI agents and models",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

# GPU Detection
def detect_gpu():
    """Detect available GPU"""
    try:
        if torch.cuda.is_available():
            return {
                "available": True,
                "count": torch.cuda.device_count(),
                "current": torch.cuda.current_device(),
                "name": torch.cuda.get_device_name()
            }
    except:
        pass
    
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            return {"available": True, "name": result.stdout.strip()}
    except:
        pass
    
    return {"available": False}

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SutazAI Complete AGI/ASI System",
        "version": "2.0.0",
        "status": "operational",
        "documentation": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """System health check"""
    start_time = datetime.now()
    
    # Get system uptime
    uptime = psutil.boot_time()
    uptime_seconds = (datetime.now().timestamp() - uptime)
    
    # Check components
    components = {}
    
    # Check Ollama
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get("http://sutazai-ollama:11434/api/tags")
            components["ollama"] = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        components["ollama"] = "disconnected"
    
    # Check databases
    try:
        await redis_client.ping()
        components["redis"] = "healthy"
    except:
        components["redis"] = "disconnected"
    
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get("http://sutazai-postgres:5432")
            components["postgres"] = "healthy"
    except:
        components["postgres"] = "checking"
    
    # Check vector databases
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get("http://sutazai-chromadb:8001/api/v1/heartbeat")
            components["chromadb"] = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        components["chromadb"] = "disconnected"
    
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get("http://sutazai-qdrant:6333/health")
            components["qdrant"] = "healthy" if response.status_code == 200 else "unhealthy"
    except:
        components["qdrant"] = "disconnected"
    
    # Get available models
    models = await model_manager.get_available_models()
    
    # Get agent status
    agent_status = await agent_manager.get_all_agents_status()
    agents = {name: status.get("status", "unknown") for name, status in agent_status.items()}
    
    return HealthResponse(
        status="healthy",
        uptime_seconds=uptime_seconds,
        components=components,
        models=models,
        agents=agents
    )

@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Main chat endpoint with AI model integration"""
    start_time = datetime.now()
    
    try:
        # Route to specific agent if requested
        if request.agent:
            result = await agent_manager.execute_agent_task(
                request.agent, 
                request.message, 
                request.context or {}
            )
            
            if result["status"] == "success":
                processing_time = (datetime.now() - start_time).total_seconds()
                return ChatResponse(
                    response=str(result["result"]),
                    model=request.model,
                    agent=request.agent,
                    tokens_used=0,
                    processing_time=processing_time,
                    confidence=0.95
                )
            else:
                # Fallback to model if agent fails
                logger.warning(f"Agent {request.agent} failed, falling back to model")
        
        # Use Ollama model
        result = await model_manager.generate_response(request.message, request.model)
        
        if "error" not in result:
            processing_time = result.get("processing_time", 0)
            return ChatResponse(
                response=result["response"],
                model=result["model"],
                tokens_used=result["tokens"],
                processing_time=processing_time,
                confidence=0.9
            )
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/agents/execute", response_model=AgentResponse)
async def execute_agent(request: AgentRequest):
    """Execute task on specific agent"""
    start_time = datetime.now()
    
    try:
        result = await agent_manager.execute_agent_task(
            request.agent_type,
            request.task,
            request.parameters or {}
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return AgentResponse(
            agent_type=request.agent_type,
            status=result["status"],
            result=result.get("result", result.get("message", "")),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Agent execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/agents/status")
async def get_agents_status():
    """Get status of all agents"""
    try:
        status = await agent_manager.get_all_agents_status()
        online_count = sum(1 for s in status.values() if s.get("status") == "healthy")
        
        return {
            "total_agents": len(status),
            "online_agents": online_count,
            "offline_agents": len(status) - online_count,
            "agents": status
        }
    except Exception as e:
        logger.error(f"Agent status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/models")
async def list_models():
    """List available AI models"""
    try:
        models = await model_manager.get_available_models()
        return {"models": models, "count": len(models)}
    except Exception as e:
        logger.error(f"Models list error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/documents/process")
async def process_document(request: DocumentRequest):
    """Process document using Documind agent"""
    try:
        result = await agent_manager.execute_agent_task(
            "documind",
            request.content,
            {"document_type": request.document_type, "process_type": request.process_type}
        )
        
        return result
    except Exception as e:
        logger.error(f"Document processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/code/generate")
async def generate_code(request: CodeRequest):
    """Generate code using GPT-Engineer or Aider"""
    try:
        agent = "gpt_engineer" if request.action == "generate" else "aider"
        
        result = await agent_manager.execute_agent_task(
            agent,
            request.requirements or request.code or "",
            {
                "language": request.language,
                "action": request.action,
                "code": request.code
            }
        )
        
        return result
    except Exception as e:
        logger.error(f"Code generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/security/scan")
async def security_scan(code: str, scan_type: str = "basic"):
    """Perform security scan using Semgrep or PentestGPT"""
    try:
        agent = "semgrep" if scan_type == "code" else "pentestgpt"
        
        result = await agent_manager.execute_agent_task(
            agent,
            code,
            {"scan_type": scan_type}
        )
        
        return result
    except Exception as e:
        logger.error(f"Security scan error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/web/automate")
async def web_automation(url: str, action: str, parameters: Dict[str, Any] = None):
    """Perform web automation using Browser-Use or Skyvern"""
    try:
        agent = "browser_use" if action in ["click", "fill", "navigate"] else "skyvern"
        
        result = await agent_manager.execute_agent_task(
            agent,
            url,
            {"action": action, "parameters": parameters or {}}
        )
        
        return result
    except Exception as e:
        logger.error(f"Web automation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/financial/analyze")
async def financial_analysis(symbol: str, analysis_type: str = "overview"):
    """Perform financial analysis using FinRobot"""
    try:
        result = await agent_manager.execute_agent_task(
            "finrobot",
            symbol,
            {"analysis_type": analysis_type}
        )
        
        return result
    except Exception as e:
        logger.error(f"Financial analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/speech/transcribe")
async def speech_transcribe(audio_data: str, language: str = "en"):
    """Transcribe speech using RealtimeSTT"""
    try:
        result = await agent_manager.execute_agent_task(
            "realtime_stt",
            audio_data,
            {"language": language}
        )
        
        return result
    except Exception as e:
        logger.error(f"Speech transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/search")
async def search_knowledge(query: str, limit: int = 10):
    """Search knowledge using vector databases"""
    try:
        results = await vector_manager.search_documents(query, limit)
        return {"query": query, "results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Knowledge search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            message = message_data.get("message", "")
            model = message_data.get("model", "tinyllama")
            
            # Generate response
            result = await model_manager.generate_response(message, model)
            
            if "error" not in result:
                await websocket.send_text(json.dumps({
                    "type": "response",
                    "message": result["response"],
                    "model": result["model"],
                    "tokens": result["tokens"]
                }))
            else:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": result["error"]
                }))
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

@app.get("/api/v1/system/status")
async def system_status():
    """Get comprehensive system status"""
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get Docker containers
        containers = []
        if docker_client:
            for container in docker_client.containers.list(all=True):
                if "sutazai" in container.name:
                    containers.append({
                        "name": container.name,
                        "status": container.status,
                        "image": container.image.tags[0] if container.image.tags else "unknown"
                    })
        
        # Get GPU info
        gpu_info = detect_gpu()
        
        # Get models and agents
        models = await model_manager.get_available_models()
        agent_status = await agent_manager.get_all_agents_status()
        
        return {
            "system": {
                "cpu_usage": cpu_percent,
                "memory_usage": memory.percent,
                "memory_total": memory.total,
                "memory_available": memory.available,
                "disk_usage": disk.percent,
                "disk_total": disk.total,
                "disk_free": disk.free
            },
            "gpu": gpu_info,
            "containers": containers,
            "models": {
                "available": models,
                "count": len(models)
            },
            "agents": {
                "status": agent_status,
                "total": len(agent_status),
                "online": sum(1 for s in agent_status.values() if s.get("status") == "healthy")
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"System status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main_complete_agi:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    ) 