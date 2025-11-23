#!/usr/bin/env python3
"""
Base Agent Wrapper for Local LLM Integration
Provides common functionality for all AI agents to work with Ollama/TinyLlama
"""

import os
import sys
import json
import logging
import asyncio
import socket
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
import uvicorn
import uuid
from datetime import datetime

# Import prometheus client for metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.warning("prometheus_client not installed - metrics endpoint will be unavailable")

# Configure logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def is_running_in_docker():
    """Detect if running inside a Docker container"""
    # Check for .dockerenv file
    if os.path.exists('/.dockerenv'):
        return True
    # Check for Docker in cgroup
    try:
        with open('/proc/self/cgroup', 'r') as f:
            return 'docker' in f.read()
    except:
        return False

def get_host_url(default_docker_host="host.docker.internal", default_local_host="localhost"):
    """Get appropriate host URL based on environment"""
    if is_running_in_docker():
        # Try to resolve host.docker.internal first (works on Docker Desktop)
        try:
            socket.gethostbyname('host.docker.internal')
            return default_docker_host
        except socket.gaierror:
            # Try gateway.docker.internal (newer Docker versions)
            try:
                socket.gethostbyname('gateway.docker.internal')
                return "gateway.docker.internal"
            except socket.gaierror:
                # Use the host IP directly - this works in Linux Docker
                # The host typically has IP .1 on the Docker network
                # Get the container's IP and derive the host IP
                try:
                    hostname = socket.gethostname()
                    host_ip = socket.gethostbyname(hostname)
                    # Convert container IP to host IP (e.g., 172.20.0.5 -> 172.20.0.1)
                    ip_parts = host_ip.split('.')
                    host_gateway = f"{ip_parts[0]}.{ip_parts[1]}.{ip_parts[2]}.1"
                    return host_gateway
                except:
                    # Final fallback - use common Docker bridge IPs
                    return "172.20.0.1"  # Common custom network gateway
    else:
        return default_local_host

# Ollama Configuration - Auto-detect environment
OLLAMA_HOST = get_host_url()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", f"http://{OLLAMA_HOST}:11434")
MODEL = os.getenv("MODEL", "tinyllama")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "120"))

# MCP Bridge Configuration - Auto-detect environment
MCP_HOST = get_host_url()
MCP_BRIDGE_URL = os.getenv("MCP_BRIDGE_URL", f"http://{MCP_HOST}:11100")
MCP_REGISTRATION_ENABLED = os.getenv("MCP_REGISTRATION_ENABLED", "true").lower() == "true"

logger.info(f"Environment detection: Docker={is_running_in_docker()}")
logger.info(f"Using Ollama URL: {OLLAMA_BASE_URL}")
logger.info(f"Using MCP Bridge URL: {MCP_BRIDGE_URL}")

class ChatMessage(BaseModel):
    """Chat message model"""
    role: str = Field(..., description="Message role (user/assistant/system)")
    content: str = Field(..., description="Message content")

class ChatRequest(BaseModel):
    """Chat completion request"""
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    model: Optional[str] = Field(default=MODEL, description="Model to use")
    temperature: Optional[float] = Field(default=0.7, description="Temperature for sampling")
    max_tokens: Optional[int] = Field(default=2048, description="Maximum tokens to generate")
    stream: Optional[bool] = Field(default=False, description="Stream the response")

class ChatResponse(BaseModel):
    """Chat completion response"""
    id: str = Field(..., description="Response ID")
    object: str = Field(default="chat.completion", description="Response type")
    created: int = Field(..., description="Creation timestamp")
    model: str = Field(..., description="Model used")
    choices: List[Dict[str, Any]] = Field(..., description="Response choices")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")

class BaseAgentWrapper:
    """Base class for AI agent wrappers with Ollama integration"""
    
    def __init__(self, agent_name: str, agent_description: str, port: int = 8000, capabilities: List[str] = None):
        self.agent_name = agent_name
        self.agent_id = agent_name.lower().replace(" ", "-")
        self.agent_description = agent_description
        self.port = port
        self.capabilities = capabilities or []
        self.app = FastAPI(
            title=f"{agent_name} Local LLM Wrapper",
            description=agent_description,
            version="1.0.0"
        )
        self.setup_cors()
        self.setup_prometheus_metrics()
        self.setup_routes()
        self.http_client = None
        self.mcp_registered = False
    
    def setup_prometheus_metrics(self):
        """Initialize Prometheus metrics collectors"""
        if PROMETHEUS_AVAILABLE:
            # Request counters
            self.requests_total = Counter(
                f'{self.agent_id}_requests_total',
                'Total requests to the agent',
                ['method', 'endpoint', 'status']
            )
            
            # Request duration histogram
            self.request_duration = Histogram(
                f'{self.agent_id}_request_duration_seconds',
                'Request duration in seconds',
                ['method', 'endpoint']
            )
            
            # Ollama request counter and duration
            self.ollama_requests_total = Counter(
                f'{self.agent_id}_ollama_requests_total',
                'Total requests to Ollama',
                ['status']
            )
            
            self.ollama_request_duration = Histogram(
                f'{self.agent_id}_ollama_request_duration_seconds',
                'Ollama request duration in seconds'
            )
            
            # Health status gauge
            self.health_status = Gauge(
                f'{self.agent_id}_health_status',
                'Agent health status (1=healthy, 0=unhealthy)'
            )
            
            # MCP registration status
            self.mcp_registered_status = Gauge(
                f'{self.agent_id}_mcp_registered',
                'MCP registration status (1=registered, 0=not registered)'
            )
            
            logger.info(f"Prometheus metrics initialized for {self.agent_name}")
        else:
            logger.warning(f"Prometheus metrics not available for {self.agent_name}")
        
    def setup_cors(self):
        """Configure CORS for the FastAPI app"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    
    def setup_routes(self):
        """Setup basic routes for the agent"""
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize HTTP client on startup"""
            self.http_client = httpx.AsyncClient(timeout=OLLAMA_TIMEOUT)
            await self.check_ollama_connection()
            if MCP_REGISTRATION_ENABLED:
                await self.register_with_mcp()
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on shutdown"""
            if self.http_client:
                await self.http_client.aclose()
        
        @self.app.get("/")
        async def root():
            """Root endpoint"""
            return {
                "agent": self.agent_name,
                "description": self.agent_description,
                "status": "running",
                "ollama_url": OLLAMA_BASE_URL,
                "model": MODEL
            }
        
        @self.app.get("/health")
        async def health():
            """Health check endpoint"""
            try:
                # Check Ollama connection
                ollama_status = await self.check_ollama_health()
                
                # Update health gauge if prometheus is available
                if PROMETHEUS_AVAILABLE:
                    self.health_status.set(1 if ollama_status else 0)
                
                return {
                    "status": "healthy" if ollama_status else "degraded",
                    "agent": self.agent_name,
                    "ollama": ollama_status,
                    "model": MODEL
                }
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                if PROMETHEUS_AVAILABLE:
                    self.health_status.set(0)
                raise HTTPException(status_code=503, detail=str(e))
        
        @self.app.get("/metrics")
        async def metrics():
            """Prometheus metrics endpoint"""
            if not PROMETHEUS_AVAILABLE:
                raise HTTPException(
                    status_code=501,
                    detail="Prometheus metrics not available - prometheus_client not installed"
                )
            
            try:
                # Generate latest metrics
                metrics_output = generate_latest(REGISTRY)
                return Response(content=metrics_output, media_type=CONTENT_TYPE_LATEST)
            except Exception as e:
                logger.error(f"Failed to generate metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/chat/completions")
        async def chat_completions(request: ChatRequest):
            """OpenAI-compatible chat completions endpoint"""
            try:
                response = await self.generate_completion(request)
                return response
            except Exception as e:
                logger.error(f"Chat completion failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/chat")
        async def chat(request: Request):
            """Simple chat endpoint for message-based interaction"""
            try:
                data = await request.json()
                messages = data.get("messages", [])
                
                if not messages:
                    raise HTTPException(status_code=400, detail="No messages provided")
                
                # Convert simple format to ChatRequest
                chat_messages = []
                for msg in messages:
                    if isinstance(msg, dict):
                        chat_messages.append(ChatMessage(
                            role=msg.get("role", "user"),
                            content=msg.get("content", "")
                        ))
                
                chat_request = ChatRequest(
                    messages=chat_messages,
                    model=data.get("model", MODEL),
                    temperature=data.get("temperature", 0.7),
                    max_tokens=data.get("max_tokens", 2048),
                    stream=data.get("stream", False)
                )
                
                response = await self.generate_completion(chat_request)
                
                # Return simple format
                return {
                    "response": response.choices[0]["message"]["content"],
                    "agent": self.agent_name,
                    "model": response.model,
                    "usage": response.usage
                }
                
            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Chat failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/generate")
        async def generate(request: Request):
            """Ollama-compatible generate endpoint"""
            try:
                data = await request.json()
                prompt = data.get("prompt", "")
                response = await self.generate_text(prompt, data)
                return response
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/mcp/receive")
        async def receive_mcp_message(request: Request):
            """Receive messages from MCP Bridge"""
            try:
                data = await request.json()
                message_type = data.get("type", "")
                payload = data.get("payload", {})
                
                # Process message based on type
                if message_type.startswith("task."):
                    # Handle task request
                    result = await self.process_task(payload)
                    return {"status": "success", "result": result}
                elif message_type == "health.check":
                    # Health check from MCP
                    return {"status": "healthy", "agent": self.agent_name}
                else:
                    # Unknown message type
                    return {"status": "unsupported", "message": f"Unknown message type: {message_type}"}
                    
            except Exception as e:
                logger.error(f"Failed to process MCP message: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    async def check_ollama_connection(self):
        """Check if Ollama is accessible"""
        try:
            response = await self.http_client.get(f"{OLLAMA_BASE_URL}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                if MODEL not in model_names and not any(MODEL in name for name in model_names):
                    logger.warning(f"Model {MODEL} not found in Ollama. Available models: {model_names}")
                else:
                    logger.info(f"Successfully connected to Ollama with model {MODEL}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False
    
    async def register_with_mcp(self):
        """Register agent with MCP Bridge"""
        try:
            # Register agent status with MCP Bridge
            register_url = f"{MCP_BRIDGE_URL}/agents/{self.agent_id}/status"
            response = await self.http_client.post(
                register_url,
                params={"status": "online"}
            )
            
            if response.status_code == 200:
                self.mcp_registered = True
                if PROMETHEUS_AVAILABLE:
                    self.mcp_registered_status.set(1)
                logger.info(f"Successfully registered {self.agent_name} with MCP Bridge")
            else:
                if PROMETHEUS_AVAILABLE:
                    self.mcp_registered_status.set(0)
                logger.warning(f"Failed to register with MCP Bridge: {response.status_code}")
                
        except Exception as e:
            if PROMETHEUS_AVAILABLE:
                self.mcp_registered_status.set(0)
            logger.error(f"Error registering with MCP Bridge: {e}")
    
    async def send_to_mcp(self, message_type: str, payload: Dict[str, Any]):
        """Send message to MCP Bridge"""
        if not self.mcp_registered:
            logger.warning("Agent not registered with MCP Bridge")
            return None
            
        try:
            message = {
                "id": str(uuid.uuid4()),
                "source": self.agent_id,
                "target": "mcp-bridge",
                "type": message_type,
                "payload": payload,
                "timestamp": datetime.now().isoformat()
            }
            
            response = await self.http_client.post(
                f"{MCP_BRIDGE_URL}/route",
                json=message
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to send message to MCP Bridge: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error sending to MCP Bridge: {e}")
            return None
    
    async def check_ollama_health(self) -> bool:
        """Check Ollama health status"""
        try:
            if not self.http_client:
                return False
            response = await self.http_client.get(f"{OLLAMA_BASE_URL}/api/tags")
            return response.status_code == 200
        except Exception:
            return False
    
    async def generate_completion(self, request: ChatRequest) -> ChatResponse:
        """Generate completion using Ollama"""
        try:
            # Convert messages to Ollama format
            messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
            
            # Prepare Ollama request
            ollama_request = {
                "model": request.model or MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens
                }
            }
            
            # Make request to Ollama
            response = await self.http_client.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json=ollama_request
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama returned status {response.status_code}: {response.text}")
            
            result = response.json()
            
            # Convert to OpenAI format
            import time
            return ChatResponse(
                id=f"chatcmpl-{self.agent_name}-{int(time.time())}",
                created=int(time.time()),
                model=request.model or MODEL,
                choices=[{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result.get("message", {}).get("content", "")
                    },
                    "finish_reason": "stop"
                }],
                usage={
                    "prompt_tokens": result.get("prompt_eval_count", 0),
                    "completion_tokens": result.get("eval_count", 0),
                    "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to generate completion: {e}")
            raise
    
    async def process_task(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task from MCP Bridge"""
        try:
            task_id = payload.get("task_id")
            description = payload.get("description")
            params = payload.get("params", {})
            
            logger.info(f"Processing task {task_id}: {description}")
            
            # Generate response using LLM
            prompt = f"Task: {description}\nParameters: {json.dumps(params)}\n\nProvide a solution:"
            
            request = ChatRequest(
                messages=[
                    ChatMessage(role="system", content=f"You are {self.agent_name}. {self.agent_description}"),
                    ChatMessage(role="user", content=prompt)
                ]
            )
            
            response = await self.generate_completion(request)
            
            result = {
                "task_id": task_id,
                "agent": self.agent_name,
                "response": response.choices[0]["message"]["content"],
                "status": "completed"
            }
            
            # Send result back to MCP Bridge
            await self.send_to_mcp("task.completed", result)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process task: {e}")
            return {"status": "error", "error": str(e)}
    
    async def generate_text(self, prompt: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text using Ollama's generate endpoint"""
        try:
            ollama_request = {
                "model": options.get("model", MODEL),
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": options.get("temperature", 0.7),
                    "num_predict": options.get("max_tokens", 2048)
                }
            }
            
            response = await self.http_client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=ollama_request
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama returned status {response.status_code}: {response.text}")
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Failed to generate text: {e}")
            raise
    
    def run(self):
        """Run the FastAPI server"""
        logger.info(f"Starting {self.agent_name} on port {self.port}")
        logger.info(f"Using Ollama at {OLLAMA_BASE_URL} with model {MODEL}")
        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info"
        )

# Helper function for quick agent creation
def create_agent_wrapper(name: str, description: str, port: int = 8000):
    """Create and run an agent wrapper"""
    agent = BaseAgentWrapper(name, description, port)
    return agent

if __name__ == "__main__":
    # Example usage
    agent = create_agent_wrapper(
        "BaseAgent",
        "Base agent wrapper for local LLM integration",
        8000
    )
    agent.run()