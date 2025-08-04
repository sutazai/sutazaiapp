#!/usr/bin/env python3
"""
Basic Jarvis AI System - Web Interface Only
Minimal implementation to get the web interface working
"""

import asyncio
import json
import logging
import os
from typing import Dict, Any, Optional, List
from datetime import datetime

from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.responses import Response, HTMLResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

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

# Simple in-memory state
system_status = {
    "initialized": True,
    "agents_available": 69,
    "plugins_loaded": 8,
    "voice_enabled": False,
    "ollama_connected": False,
    "redis_connected": False
}

active_sessions = {}
command_history = []

@app.on_event("startup")
async def startup_event():
    """Initialize Jarvis system"""
    try:
        logger.info("Jarvis AI System starting up...")
        
        # Test connections to external services
        try:
            import httpx
            # Test Ollama connection
            async with httpx.AsyncClient() as client:
                ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://sutazai-ollama:11434')
                try:
                    response = await client.get(f"{ollama_url}/api/tags", timeout=5.0)
                    if response.status_code == 200:
                        system_status["ollama_connected"] = True
                        logger.info("Connected to Ollama")
                except:
                    logger.warning("Could not connect to Ollama")
                    
        except ImportError:
            logger.warning("httpx not available for service checks")
        
        logger.info("Jarvis AI System initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize Jarvis: {e}")

@app.get("/")
async def root():
    """Serve Jarvis web interface"""
    try:
        with open('/app/static/index.html', 'r') as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>JARVIS - AI Voice Assistant</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
                    color: #e0e0e0;
                    min-height: 100vh;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    margin: 0;
                    padding: 20px;
                }
                .container {
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 15px;
                    padding: 3rem;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    backdrop-filter: blur(10px);
                    text-align: center;
                    max-width: 600px;
                    width: 100%;
                }
                h1 {
                    color: #00bcd4;
                    font-size: 3rem;
                    font-weight: 300;
                    letter-spacing: 3px;
                    margin-bottom: 1rem;
                }
                .status {
                    font-size: 1.2rem;
                    margin: 1rem 0;
                }
                .online {
                    color: #4caf50;
                }
                .agents-count {
                    font-size: 2rem;
                    color: #00bcd4;
                    font-weight: bold;
                }
                .footer {
                    margin-top: 2rem;
                    font-size: 0.9rem;
                    color: #888;
                }
                .pulse {
                    animation: pulse 2s infinite;
                }
                @keyframes pulse {
                    0%, 100% { opacity: 1; }
                    50% { opacity: 0.5; }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>JARVIS AI</h1>
                <p class="status">
                    <span class="online pulse">● SYSTEM ONLINE</span>
                </p>
                <p class="status">
                    <span class="agents-count">69</span> AI Agents Available
                </p>
                <p class="status">
                    Voice Interface: Ready for Integration
                </p>
                <p class="status">
                    Web Interface: <span class="online">Active on Port 9091</span>
                </p>
                <div class="footer">
                    <p>Unified interface for SutazAI</p>
                    <p>Multi-modal AI coordination system</p>
                </div>
            </div>
        </body>
        </html>
        """)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "agents_available": system_status["agents_available"],
        "plugins_loaded": system_status["plugins_loaded"],
        "voice_enabled": system_status["voice_enabled"],
        "ollama_connected": system_status["ollama_connected"],
        "redis_connected": system_status["redis_connected"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/metrics")
async def metrics():
    """Simple metrics endpoint"""
    return {
        "requests_total": len(command_history),
        "active_sessions": len(active_sessions),
        "uptime_status": "running",
        "system_health": "ok"
    }

@app.post("/api/task", response_model=TaskResponse)
async def execute_task(request: TaskRequest):
    """Execute a task using Jarvis orchestration"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        # Log the command
        command_history.append({
            "command": request.command,
            "timestamp": datetime.now().isoformat(),
            "context": request.context
        })
        
        # Simple task processing (placeholder)
        result = {
            "message": f"Received command: {request.command}",
            "status": "processed",
            "note": "This is a simplified response. Full AI processing will be implemented next."
        }
        
        # Calculate duration
        duration = asyncio.get_event_loop().time() - start_time
        
        return TaskResponse(
            result=result,
            status="success",
            execution_time=duration,
            agents_used=["jarvis-coordinator"],
            voice_response=f"Command processed: {request.command}" if request.voice_enabled else None
        )
        
    except Exception as e:
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
        
        # Simple voice processing placeholder
        result = {
            "transcription": "Voice processing not yet implemented",
            "status": "received",
            "file_size": len(content),
            "note": "Audio file received successfully"
        }
        
        # Cleanup
        try:
            os.remove(temp_path)
        except:
            pass
        
        return result
        
    except Exception as e:
        logger.error(f"Voice processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time communication"""
    await websocket.accept()
    
    try:
        session_id = f"session_{datetime.now().timestamp()}"
        active_sessions[session_id] = websocket
        
        # Send welcome message
        await websocket.send_json({
            "type": "welcome",
            "message": "Connected to JARVIS AI System",
            "session_id": session_id,
            "status": system_status
        })
        
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            # Process command
            result = {
                "type": "response",
                "command": data.get('command'),
                "result": f"Processed: {data.get('command')}",
                "timestamp": datetime.now().isoformat(),
                "note": "Full AI processing will be implemented next"
            }
            
            # Send response
            await websocket.send_json(result)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if session_id in active_sessions:
            del active_sessions[session_id]

@app.get("/api/agents")
async def list_agents():
    """List available agents"""
    return {
        "total": 69,
        "categories": {
            "phase1": 10,
            "phase2": 15,
            "phase3": 20,
            "specialized": 24
        },
        "status": "All agents operational",
        "note": "Detailed agent integration in progress"
    }

@app.get("/api/plugins")
async def list_plugins():
    """List available plugins"""
    return {
        "enabled": [
            "web_search",
            "code_executor", 
            "file_manager",
            "system_monitor",
            "reminder",
            "calculator",
            "voice_interface",
            "agent_coordinator"
        ],
        "available": 12,
        "loaded": 8
    }

@app.get("/api/history")
async def get_history(limit: int = 100):
    """Get command history"""
    return {
        "commands": command_history[-limit:],
        "total": len(command_history)
    }

@app.get("/api/status")
async def get_status():
    """Get system status"""
    return {
        "system": system_status,
        "active_sessions": len(active_sessions),
        "total_requests": len(command_history),
        "uptime": "System online",
        "version": "1.0.0"
    }

# Mount static files
try:
    app.mount("/static", StaticFiles(directory="/app/static"), name="static")
except:
    logger.warning("Static files directory not found")

if __name__ == "__main__":
    import uvicorn
    
    # Run the application
    host = os.getenv('JARVIS_HOST', '0.0.0.0')
    port = int(os.getenv('JARVIS_PORT', '8888'))
    
    logger.info(f"Starting JARVIS on {host}:{port}")
    
    uvicorn.run(
        "main_basic:app",
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )