#!/usr/bin/env python3
"""
Jarvis AI System - Unified Interface for SutazAI
Combines best features from multiple Jarvis implementations
"""

import asyncio
import json
import logging
import os
import signal
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime
import speech_recognition as sr
import pyttsx3
from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import consul
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response, HTMLResponse

from core.orchestrator import JarvisOrchestrator
from core.task_planner import TaskPlanner
from core.voice_interface import VoiceInterface
from core.plugin_manager import PluginManager
from core.agent_coordinator import AgentCoordinator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('jarvis_requests_total', 'Total requests', ['type', 'status'])
REQUEST_DURATION = Histogram('jarvis_request_duration_seconds', 'Request duration', ['type'])
ACTIVE_SESSIONS = Counter('jarvis_active_sessions', 'Active sessions')

# Global instances
jarvis: Optional[JarvisOrchestrator] = None
consul_client: Optional[consul.Consul] = None

class TaskRequest(BaseModel):
    """Task request model"""
    command: str
    context: Optional[Dict[str, Any]] = {}
    voice_enabled: bool = False
    plugins: Optional[List[str]] = []

class TaskResponse(BaseModel):
    """Task response model"""
    result: Any
    status: str
    execution_time: float
    agents_used: List[str]
    voice_response: Optional[str] = None

app = FastAPI(
    title="Jarvis AI System",
    description="Unified AI assistant with voice interface and multi-agent coordination",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize Jarvis system"""
    global jarvis, consul_client
    
    try:
        # Load configuration
        config_path = os.getenv('JARVIS_CONFIG', '/opt/sutazaiapp/config/jarvis/config.yaml')
        
        # Initialize components
        jarvis = JarvisOrchestrator(config_path)
        await jarvis.initialize()
        
        # Initialize Consul
        consul_host = os.getenv('CONSUL_HOST', 'localhost')
        consul_port = int(os.getenv('CONSUL_PORT', '8500'))
        consul_client = consul.Consul(host=consul_host, port=consul_port)
        
        # Register with Consul
        service_port = int(os.getenv('JARVIS_PORT', '8888'))
        consul_client.agent.service.register(
            name='jarvis',
            service_id=f'jarvis-{os.getpid()}',
            address=os.getenv('SERVICE_HOST', 'localhost'),
            port=service_port,
            check=consul.Check.http(f"http://localhost:{service_port}/health", interval="10s"),
            tags=['ai', 'jarvis', 'assistant', 'voice']
        )
        
        logger.info("Jarvis AI System initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Jarvis: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global jarvis
    
    if jarvis:
        await jarvis.shutdown()
    
    if consul_client:
        try:
            consul_client.agent.service.deregister(f'jarvis-{os.getpid()}')
        except Exception as e:
            logger.error(f"Failed to deregister from Consul: {e}")

@app.get("/")
async def root():
    """Serve Jarvis web interface"""
    return HTMLResponse(content=open('/opt/sutazaiapp/services/jarvis/static/index.html').read())

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        status = await jarvis.get_status()
        return {
            "status": "healthy",
            "version": "1.0.0",
            "agents_available": status['agents_available'],
            "plugins_loaded": status['plugins_loaded'],
            "voice_enabled": status['voice_enabled']
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

@app.post("/api/task", response_model=TaskResponse)
async def execute_task(request: TaskRequest):
    """Execute a task using Jarvis orchestration"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Process the task
        result = await jarvis.execute_task(
            command=request.command,
            context=request.context,
            voice_enabled=request.voice_enabled,
            plugins=request.plugins
        )
        
        # Update metrics
        duration = asyncio.get_event_loop().time() - start_time
        REQUEST_DURATION.labels(type='task').observe(duration)
        REQUEST_COUNT.labels(type='task', status='success').inc()
        
        return TaskResponse(
            result=result['result'],
            status=result['status'],
            execution_time=duration,
            agents_used=result['agents_used'],
            voice_response=result.get('voice_response')
        )
        
    except Exception as e:
        REQUEST_COUNT.labels(type='task', status='error').inc()
        logger.error(f"Task execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice/upload")
async def process_voice_upload(audio: UploadFile = File(...)):
    """Process uploaded audio file"""
    try:
        # Save temporary audio file
        temp_path = f"/tmp/jarvis_audio_{datetime.now().timestamp()}.wav"
        with open(temp_path, "wb") as f:
            content = await audio.read()
            f.write(content)
        
        # Process audio
        result = await jarvis.process_voice_command(temp_path)
        
        # Cleanup
        os.remove(temp_path)
        
        return result
        
    except Exception as e:
        logger.error(f"Voice processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time communication"""
    await websocket.accept()
    ACTIVE_SESSIONS.inc()
    
    try:
        session_id = f"session_{datetime.now().timestamp()}"
        await jarvis.register_session(session_id, websocket)
        
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            # Process command
            result = await jarvis.execute_task(
                command=data.get('command'),
                context=data.get('context', {}),
                voice_enabled=data.get('voice_enabled', False),
                session_id=session_id
            )
            
            # Send response
            await websocket.send_json(result)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        ACTIVE_SESSIONS.dec()
        await jarvis.unregister_session(session_id)
        await websocket.close()

@app.get("/api/agents")
async def list_agents():
    """List available agents"""
    return await jarvis.list_available_agents()

@app.get("/api/plugins")
async def list_plugins():
    """List available plugins"""
    return await jarvis.list_plugins()

@app.post("/api/plugins/{plugin_name}/enable")
async def enable_plugin(plugin_name: str):
    """Enable a plugin"""
    try:
        await jarvis.enable_plugin(plugin_name)
        return {"status": "enabled", "plugin": plugin_name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/plugins/{plugin_name}/disable")
async def disable_plugin(plugin_name: str):
    """Disable a plugin"""
    try:
        await jarvis.disable_plugin(plugin_name)
        return {"status": "disabled", "plugin": plugin_name}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/history")
async def get_history(limit: int = 100):
    """Get command history"""
    return await jarvis.get_command_history(limit)

@app.post("/api/feedback")
async def submit_feedback(session_id: str, rating: int, comment: Optional[str] = None):
    """Submit feedback for a session"""
    await jarvis.record_feedback(session_id, rating, comment)
    return {"status": "recorded"}

# Static files
app.mount("/static", StaticFiles(directory="/opt/sutazaiapp/services/jarvis/static"), name="static")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down...")
    sys.exit(0)

if __name__ == "__main__":
    import uvicorn
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run the application
    host = os.getenv('JARVIS_HOST', '0.0.0.0')
    port = int(os.getenv('JARVIS_PORT', '8888'))
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )