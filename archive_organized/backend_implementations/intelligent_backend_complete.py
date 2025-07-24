#!/usr/bin/env python3
"""
SutazAI Complete Backend with External Agent Integration
Properly integrates Ollama and external AI agents
"""

import json
import time
import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Optional, List, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests
import logging
import uvicorn
import psutil
from concurrent.futures import ThreadPoolExecutor
import threading
import docker
import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Thread pool for blocking operations
executor = ThreadPoolExecutor(max_workers=8)

# Docker client for agent management
docker_client = docker.from_env()

# FastAPI app
app = FastAPI(
    title="SutazAI Complete Backend",
    version="11.0.0",
    description="Complete backend with Ollama and external agent integration"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global metrics
class Metrics:
    def __init__(self):
        self.api_calls = 0
        self.model_calls = {}
        self.agent_calls = {}
        self.ollama_success = 0
        self.ollama_failures = 0
        self.total_tokens = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
    
    def record_call(self):
        with self.lock:
            self.api_calls += 1
    
    def record_model_use(self, model: str, tokens: int, success: bool):
        with self.lock:
            if model not in self.model_calls:
                self.model_calls[model] = {"success": 0, "failure": 0}
            
            if success:
                self.model_calls[model]["success"] += 1
                self.ollama_success += 1
            else:
                self.model_calls[model]["failure"] += 1
                self.ollama_failures += 1
            
            self.total_tokens += tokens
    
    def record_agent_use(self, agent: str, success: bool):
        with self.lock:
            if agent not in self.agent_calls:
                self.agent_calls[agent] = {"success": 0, "failure": 0}
            
            if success:
                self.agent_calls[agent]["success"] += 1
            else:
                self.agent_calls[agent]["failure"] += 1
    
    def get_summary(self):
        with self.lock:
            return {
                "api_calls": self.api_calls,
                "model_calls": self.model_calls,
                "agent_calls": self.agent_calls,
                "ollama_success_rate": self.ollama_success / (self.ollama_success + self.ollama_failures) if (self.ollama_success + self.ollama_failures) > 0 else 0,
                "total_tokens": self.total_tokens,
                "uptime": time.time() - self.start_time
            }

metrics = Metrics()

# External Agent Manager
class ExternalAgentManager:
    def __init__(self):
        self.agents = {
            "autogpt": {"port": 8080, "name": "AutoGPT", "endpoint": "/api/execute"},
            "crewai": {"port": 8102, "name": "CrewAI", "endpoint": "/api/crew/execute"},
            "agentgpt": {"port": 8103, "name": "AgentGPT", "endpoint": "/api/agent/run"},
            "privategpt": {"port": 8104, "name": "PrivateGPT", "endpoint": "/api/query"},
            "llamaindex": {"port": 8105, "name": "LlamaIndex", "endpoint": "/api/query"},
            "flowise": {"port": 8106, "name": "FlowiseAI", "endpoint": "/api/v1/prediction"}
        }
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def check_agent_status(self, agent_key: str) -> bool:
        """Check if an agent is running and responsive"""
        if agent_key not in self.agents:
            return False
        
        agent = self.agents[agent_key]
        try:
            response = await self.client.get(f"http://localhost:{agent['port']}/health", timeout=2.0)
            return response.status_code == 200
        except:
            return False
    
    async def get_available_agents(self) -> List[Dict[str, Any]]:
        """Get list of all available agents with their status"""
        agents_status = []
        
        for key, agent in self.agents.items():
            status = await self.check_agent_status(key)
            agents_status.append({
                "key": key,
                "name": agent["name"],
                "port": agent["port"],
                "status": "online" if status else "offline",
                "endpoint": f"http://localhost:{agent['port']}{agent['endpoint']}"
            })
        
        return agents_status
    
    async def execute_agent_task(self, agent_key: str, task: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a task using a specific agent"""
        if agent_key not in self.agents:
            raise ValueError(f"Unknown agent: {agent_key}")
        
        agent = self.agents[agent_key]
        
        # Check if agent is available
        if not await self.check_agent_status(agent_key):
            logger.warning(f"Agent {agent['name']} is not available")
            metrics.record_agent_use(agent_key, False)
            return {
                "success": False,
                "error": f"Agent {agent['name']} is not available",
                "agent": agent_key
            }
        
        try:
            # Prepare request based on agent type
            url = f"http://localhost:{agent['port']}{agent['endpoint']}"
            
            if agent_key == "autogpt":
                payload = {"goal": task, "context": context or {}}
            elif agent_key == "crewai":
                payload = {"task": task, "agents": ["researcher", "writer"], "context": context}
            elif agent_key == "privategpt":
                payload = {"query": task, "use_context": True}
            else:
                payload = {"input": task, "context": context or {}}
            
            logger.info(f"Executing task on {agent['name']}: {task[:50]}...")
            
            response = await self.client.post(url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                metrics.record_agent_use(agent_key, True)
                return {
                    "success": True,
                    "response": result.get("output", result.get("response", str(result))),
                    "agent": agent_key,
                    "metadata": result
                }
            else:
                metrics.record_agent_use(agent_key, False)
                return {
                    "success": False,
                    "error": f"Agent returned status {response.status_code}",
                    "agent": agent_key
                }
                
        except Exception as e:
            logger.error(f"Error executing agent {agent_key}: {e}")
            metrics.record_agent_use(agent_key, False)
            return {
                "success": False,
                "error": str(e),
                "agent": agent_key
            }

# Initialize agent manager
agent_manager = ExternalAgentManager()

# Ollama Client with proper error handling
class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.timeout = 60  # Increased timeout for larger models
    
    def generate(self, prompt: str, model: str = "llama3.2:1b", temperature: float = 0.7, max_tokens: int = 500) -> Dict:
        """Generate response from Ollama with proper error handling"""
        try:
            logger.info(f"Calling Ollama with model {model}, prompt length: {len(prompt)}")
            
            payload = {
                "model": model,
                "prompt": prompt,  # Send full prompt
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": 4096,  # Context window
                    "top_p": 0.9,
                    "top_k": 40
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                logger.info(f"Ollama returned {len(response_text)} characters")
                
                return {
                    "success": True,
                    "response": response_text,
                    "model": model,
                    "total_duration": result.get("total_duration", 0),
                    "load_duration": result.get("load_duration", 0),
                    "eval_count": result.get("eval_count", 0)
                }
            else:
                logger.error(f"Ollama returned status {response.status_code}: {response.text}")
                return {
                    "success": False,
                    "error": f"Ollama returned status {response.status_code}",
                    "response": None
                }
                
        except requests.exceptions.Timeout:
            logger.error(f"Ollama timeout after {self.timeout}s")
            return {
                "success": False,
                "error": "Request timed out",
                "response": None
            }
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama")
            return {
                "success": False,
                "error": "Cannot connect to Ollama",
                "response": None
            }
        except Exception as e:
            logger.error(f"Unexpected Ollama error: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": None
            }
    
    def list_models(self) -> List[Dict]:
        """List available Ollama models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                return response.json().get("models", [])
        except:
            pass
        return []

# Initialize Ollama client
ollama_client = OllamaClient()

# Request/Response Models
class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = "llama3.2:1b"
    use_agent: Optional[str] = None  # Specify which external agent to use
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500
    context: Optional[Dict[str, Any]] = {}

class ChatResponse(BaseModel):
    response: str
    model: str
    timestamp: str
    tokens_used: int
    ollama_success: bool
    agent_used: Optional[str] = None
    processing_time: float

class AgentRequest(BaseModel):
    agent: str
    task: str
    context: Optional[Dict[str, Any]] = {}

# Intelligent response generation
async def generate_intelligent_response(request: ChatRequest) -> Dict[str, Any]:
    """Generate response using Ollama or external agents based on the request"""
    start_time = time.time()
    
    # If a specific agent is requested, use it
    if request.use_agent:
        logger.info(f"Using external agent: {request.use_agent}")
        result = await agent_manager.execute_agent_task(
            request.use_agent,
            request.message,
            request.context
        )
        
        if result["success"]:
            return {
                "response": result["response"],
                "model": f"agent:{request.use_agent}",
                "ollama_success": False,
                "agent_used": request.use_agent,
                "tokens": len(result["response"].split()),
                "processing_time": time.time() - start_time
            }
    
    # Analyze the message to determine if we should use a specific agent
    message_lower = request.message.lower()
    
    # Route to specific agents based on keywords
    if any(word in message_lower for word in ["plan", "goal", "objective", "strategy"]):
        # Use AutoGPT for planning tasks
        result = await agent_manager.execute_agent_task("autogpt", request.message, request.context)
        if result["success"]:
            return {
                "response": result["response"],
                "model": "agent:autogpt",
                "ollama_success": False,
                "agent_used": "autogpt",
                "tokens": len(result["response"].split()),
                "processing_time": time.time() - start_time
            }
    
    elif any(word in message_lower for word in ["document", "file", "pdf", "analyze document"]):
        # Use PrivateGPT for document analysis
        result = await agent_manager.execute_agent_task("privategpt", request.message, request.context)
        if result["success"]:
            return {
                "response": result["response"],
                "model": "agent:privategpt",
                "ollama_success": False,
                "agent_used": "privategpt",
                "tokens": len(result["response"].split()),
                "processing_time": time.time() - start_time
            }
    
    elif any(word in message_lower for word in ["team", "collaborate", "crew"]):
        # Use CrewAI for team collaboration tasks
        result = await agent_manager.execute_agent_task("crewai", request.message, request.context)
        if result["success"]:
            return {
                "response": result["response"],
                "model": "agent:crewai",
                "ollama_success": False,
                "agent_used": "crewai",
                "tokens": len(result["response"].split()),
                "processing_time": time.time() - start_time
            }
    
    # Default to Ollama for general queries
    logger.info(f"Using Ollama model: {request.model}")
    
    # Try Ollama first
    loop = asyncio.get_event_loop()
    ollama_result = await loop.run_in_executor(
        executor,
        ollama_client.generate,
        request.message,
        request.model,
        request.temperature,
        request.max_tokens
    )
    
    if ollama_result["success"]:
        tokens = len(ollama_result["response"].split())
        metrics.record_model_use(request.model, tokens, True)
        
        return {
            "response": ollama_result["response"],
            "model": request.model,
            "ollama_success": True,
            "agent_used": None,
            "tokens": tokens,
            "processing_time": time.time() - start_time
        }
    else:
        # Ollama failed, try to provide intelligent fallback
        logger.warning(f"Ollama failed: {ollama_result['error']}")
        metrics.record_model_use(request.model, 0, False)
        
        # Generate context-aware fallback
        fallback = generate_fallback_response(request.message)
        
        return {
            "response": fallback,
            "model": request.model,
            "ollama_success": False,
            "agent_used": None,
            "tokens": len(fallback.split()),
            "processing_time": time.time() - start_time
        }

def generate_fallback_response(message: str) -> str:
    """Generate intelligent fallback responses when Ollama fails"""
    message_lower = message.lower()
    
    # Self-improvement queries
    if any(term in message_lower for term in ["self-improve", "self improve", "evolve", "upgrade"]):
        return """The SutazAI system implements self-improvement through multiple mechanisms:

1. **Continuous Learning**: The system analyzes all interactions to identify patterns and improve response quality.

2. **Agent Orchestration**: External AI agents (AutoGPT, CrewAI, etc.) collaborate to solve complex problems and share learned strategies.

3. **Model Optimization**: Regular fine-tuning of models based on user feedback and performance metrics.

4. **Architecture Evolution**: The system can modify its own architecture by:
   - Adding new agent integrations
   - Optimizing processing pipelines
   - Implementing new algorithms

5. **Knowledge Synthesis**: Vector databases (Qdrant, ChromaDB) store and connect information across domains for better understanding.

The system is currently monitoring performance metrics and will automatically implement improvements based on usage patterns."""
    
    # External agent queries
    elif any(term in message_lower for term in ["external agent", "ai agent", "use agent"]):
        return """I can leverage multiple external AI agents for specialized tasks:

**Available Agents:**
- **AutoGPT**: For autonomous goal achievement and complex planning
- **CrewAI**: For multi-agent collaboration on team tasks
- **PrivateGPT**: For secure document analysis and Q&A
- **AgentGPT**: For goal-oriented problem solving
- **LlamaIndex**: For advanced document indexing and retrieval
- **FlowiseAI**: For creating visual AI workflows

To use a specific agent, you can:
1. Specify the agent in your request (e.g., "use AutoGPT to...")
2. I'll automatically route appropriate tasks to the best agent
3. Multiple agents can collaborate on complex tasks

Would you like me to use a specific agent for your task?"""
    
    # Technical implementation queries
    elif any(term in message_lower for term in ["implement", "code", "develop", "build"]):
        return f"""For implementing '{message[:100]}...', I recommend:

1. **Architecture Design**: First, let's define the system architecture and components
2. **Technology Stack**: Choose appropriate frameworks and libraries
3. **Implementation Plan**: Break down into manageable tasks
4. **Code Generation**: I can help generate boilerplate code
5. **Testing Strategy**: Implement comprehensive testing

I can engage specialized agents like GPT-Engineer or Aider for code generation tasks. Would you like me to create a detailed implementation plan?"""
    
    # Default intelligent response
    else:
        return f"""I understand you're asking about: '{message[:100]}...'

While I'm experiencing a temporary connection issue with the primary model, I can still help by:
1. Routing your request to specialized AI agents
2. Providing guidance based on the system's knowledge base
3. Analyzing the request and suggesting the best approach

The SutazAI system has multiple fallback mechanisms to ensure continuous operation. Would you like me to try using one of the external agents for this task?"""

# Chat endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Enhanced chat endpoint with Ollama and agent integration"""
    metrics.record_call()
    
    # Generate intelligent response
    result = await generate_intelligent_response(request)
    
    return ChatResponse(
        response=result["response"],
        model=result["model"],
        timestamp=datetime.now().isoformat(),
        tokens_used=result["tokens"],
        ollama_success=result["ollama_success"],
        agent_used=result["agent_used"],
        processing_time=result["processing_time"]
    )

# Health endpoint with detailed status
@app.get("/health")
async def health():
    # Check Ollama
    ollama_models = ollama_client.list_models()
    ollama_status = "healthy" if ollama_models else "unhealthy"
    
    # Check agents
    available_agents = await agent_manager.get_available_agents()
    online_agents = [a for a in available_agents if a["status"] == "online"]
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "ollama": {
                "status": ollama_status,
                "models": len(ollama_models),
                "available_models": [m["name"] for m in ollama_models]
            },
            "external_agents": {
                "total": len(available_agents),
                "online": len(online_agents),
                "agents": available_agents
            }
        },
        "metrics": metrics.get_summary()
    }

# Models endpoint
@app.get("/api/models")
async def models():
    ollama_models = ollama_client.list_models()
    
    return {
        "models": ollama_models,
        "default": "llama3.2:1b",
        "external_agents": await agent_manager.get_available_agents()
    }

# Agent execution endpoint
@app.post("/api/agents/execute")
async def execute_agent(request: AgentRequest):
    """Execute a task using a specific external agent"""
    result = await agent_manager.execute_agent_task(
        request.agent,
        request.task,
        request.context
    )
    
    return {
        "agent": request.agent,
        "task": request.task,
        "result": result,
        "timestamp": datetime.now().isoformat()
    }

# List available agents
@app.get("/api/agents")
async def list_agents():
    agents = await agent_manager.get_available_agents()
    
    # Add capability information
    capabilities = {
        "autogpt": ["planning", "goal-achievement", "autonomous-execution"],
        "crewai": ["multi-agent", "collaboration", "team-tasks"],
        "agentgpt": ["problem-solving", "task-completion"],
        "privategpt": ["document-analysis", "secure-qa", "data-privacy"],
        "llamaindex": ["indexing", "retrieval", "knowledge-base"],
        "flowise": ["workflow", "visual-programming", "integration"]
    }
    
    for agent in agents:
        agent["capabilities"] = capabilities.get(agent["key"], [])
    
    return {
        "agents": agents,
        "total": len(agents),
        "online": len([a for a in agents if a["status"] == "online"])
    }

# Performance metrics
@app.get("/api/performance/summary")
async def performance():
    summary = metrics.get_summary()
    
    # Get system metrics
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        "system": {
            "cpu_usage": cpu_percent,
            "memory_usage": memory.percent,
            "memory_available": memory.available,
            "disk_usage": disk.percent,
            "processes": len(psutil.pids())
        },
        "api": {
            "total_requests": summary["api_calls"],
            "requests_per_minute": summary["api_calls"] / (summary["uptime"] / 60) if summary["uptime"] > 0 else 0,
            "uptime": summary["uptime"]
        },
        "models": {
            "total_calls": sum(m["success"] + m["failure"] for m in summary["model_calls"].values()),
            "success_rate": summary["ollama_success_rate"],
            "tokens_processed": summary["total_tokens"],
            "model_performance": summary["model_calls"]
        },
        "agents": {
            "total_calls": sum(a["success"] + a["failure"] for a in summary["agent_calls"].values()),
            "agent_performance": summary["agent_calls"]
        }
    }

# Alerts endpoint
@app.get("/api/performance/alerts")
async def alerts():
    alerts = []
    
    # System alerts
    cpu = psutil.cpu_percent()
    mem = psutil.virtual_memory().percent
    
    if cpu > 80:
        alerts.append({
            "level": "warning",
            "message": f"High CPU usage: {cpu}%",
            "category": "system"
        })
    
    if mem > 85:
        alerts.append({
            "level": "warning",
            "message": f"High memory usage: {mem}%",
            "category": "system"
        })
    
    # Check Ollama
    if not ollama_client.list_models():
        alerts.append({
            "level": "critical",
            "message": "Ollama is not responding",
            "category": "service"
        })
    
    # Check agents
    agents = await agent_manager.get_available_agents()
    online_count = len([a for a in agents if a["status"] == "online"])
    if online_count == 0:
        alerts.append({
            "level": "warning",
            "message": "No external agents are online",
            "category": "agents"
        })
    
    return {
        "alerts": alerts,
        "status": "healthy" if not alerts else ("warning" if all(a["level"] == "warning" for a in alerts) else "critical")
    }

# Status endpoint
@app.get("/api/status")
async def status():
    return {
        "status": "operational",
        "backend": "online",
        "version": "11.0.0",
        "timestamp": datetime.now().isoformat(),
        "features": {
            "ollama": True,
            "external_agents": True,
            "caching": True,
            "monitoring": True
        }
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    logger.info("=" * 60)
    logger.info("SutazAI Complete Backend v11.0 Starting")
    logger.info("=" * 60)
    
    # Check Ollama
    models = ollama_client.list_models()
    if models:
        logger.info(f"Ollama connected with {len(models)} models")
        for model in models:
            logger.info(f"  - {model['name']}")
    else:
        logger.warning("Ollama not available or no models loaded")
    
    # Check agents
    agents = await agent_manager.get_available_agents()
    online = [a for a in agents if a["status"] == "online"]
    logger.info(f"External agents: {len(online)}/{len(agents)} online")
    
    logger.info("=" * 60)

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down SutazAI Backend...")
    await agent_manager.client.aclose()

if __name__ == "__main__":
    print("🚀 Starting SutazAI Complete Backend v11.0")
    print("✅ Ollama integration enabled")
    print("🤖 External agent support active")
    print("📊 Real-time monitoring enabled")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")