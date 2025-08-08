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
try:
    import speech_recognition as sr  # optional
except Exception:
    sr = None
try:
    import pyttsx3  # optional
except Exception:
    pyttsx3 = None
from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from services.jarvis.schemas import TaskRequest, TaskResponse
try:
    import consul  # optional
except Exception:
    consul = None
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest  # optional
except Exception:
    class _Counter:
        def __init__(self, *a, **k):
            pass
        def labels(self, *a, **k):
            return self
        def inc(self, *a, **k):
            pass
    class _Histogram(_Counter):
        def observe(self, *a, **k):
            pass
    class _Gauge(_Counter):
        def dec(self, *a, **k):
            pass
    def generate_latest(*a, **k):
        return b""
    Counter = _Counter
    Histogram = _Histogram
    Gauge = _Gauge
from starlette.responses import Response, HTMLResponse

# Defer heavy core imports to runtime in startup to support minimal test mode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Optional multipart support (for UploadFile)
try:
    import multipart  # type: ignore
    MULTIPART_AVAILABLE = True
except Exception:
    MULTIPART_AVAILABLE = False

# Prometheus metrics
REQUEST_COUNT = Counter('jarvis_requests_total', 'Total requests', ['type', 'status'])
REQUEST_DURATION = Histogram('jarvis_request_duration_seconds', 'Request duration', ['type'])
ACTIVE_SESSIONS = Gauge('jarvis_active_sessions', 'Active sessions')

# Global instances
jarvis: Optional[Any] = None
consul_client: Optional[Any] = None

    

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

class MinimalJarvis:
    async def initialize(self):
        return True

    async def shutdown(self):
        return True

    async def get_status(self):
        return {
            'agents_available': ['echo-agent'],
            'plugins_loaded': [],
            'voice_enabled': False
        }

    async def execute_task(self, command: str, context: Optional[Dict[str, Any]] = None, voice_enabled: bool = False, plugins: Optional[List[str]] = None, session_id: Optional[str] = None):
        return {
            'result': f"Echo: {command}",
            'status': 'completed',
            'agents_used': ['echo-agent'],
            'voice_response': None
        }

    async def process_voice_command(self, file_path: str):
        return {
            'transcript': 'test voice input',
            'result': 'Processed voice (minimal mode)',
            'confidence': 0.99
        }

    async def register_session(self, session_id: str, websocket: WebSocket):
        return True

    async def unregister_session(self, session_id: str):
        return True


@app.on_event("startup")
async def startup_event():
    """Initialize Jarvis system"""
    global jarvis, consul_client

    try:
        # Load configuration
        config_path = os.getenv('JARVIS_CONFIG', '/opt/sutazaiapp/config/jarvis/config.yaml')

        # Minimal/fake mode for local testing
        if os.getenv('JARVIS_FAKE', '0') in ('1', 'true', 'True'):
            jarvis = MinimalJarvis()
            await jarvis.initialize()
            logger.info("Jarvis started in minimal (fake) mode for testing")
            return

        # Initialize components (import core modules lazily)
        from core.orchestrator import JarvisOrchestrator
        jarvis = JarvisOrchestrator(config_path)
        await jarvis.initialize()

        # Consul (optional, enable via CONSUL_ENABLE)
        if os.getenv('CONSUL_ENABLE', '0') in ('1', 'true', 'True'):
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

# Alias for namespaced health check
@app.get("/jarvis/health")
async def jarvis_health_check():
    return await health_check()

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

@app.get("/api/agents")
async def list_agents():
    """List available agents (for static UI side panel)"""
    try:
        status = await jarvis.get_status()
        agents = status.get('agents_available', [])
        # Normalize to objects with status for UI
        return [{"name": a, "status": "available"} for a in agents]
    except Exception as e:
        logger.error(f"List agents failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to list agents")

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

async def _process_audio_content(content: bytes, filename: str = "voice_input.webm"):
    """Common audio processing for both multipart and raw uploads."""
    try:
        # Save temporary audio file (preserve extension when possible)
        _, provided_ext = os.path.splitext(filename or '')
        safe_ext = provided_ext if provided_ext in {'.wav', '.webm', '.mp3', '.m4a', '.ogg'} else '.webm'
        temp_path = f"/tmp/jarvis_audio_{datetime.now().timestamp()}{safe_ext}"
        with open(temp_path, "wb") as f:
            f.write(content)

        # Convert to WAV if needed for downstream compatibility
        processed_path = temp_path
        try:
            _, ext = os.path.splitext(temp_path)
            if ext.lower() != '.wav':
                try:
                    from pydub import AudioSegment  # Optional dependency
                    audio_seg = AudioSegment.from_file(temp_path)
                    wav_path = temp_path.rsplit('.', 1)[0] + '.wav'
                    audio_seg.export(wav_path, format='wav')
                    os.remove(temp_path)
                    processed_path = wav_path
                except Exception as conv_err:
                    logger.warning(f"Audio conversion to WAV failed or unavailable; proceeding with original file: {conv_err}")
        except Exception as e_inner:
            logger.warning(f"Audio conversion check failed: {e_inner}")

        # Process audio
        result = await jarvis.process_voice_command(processed_path)

        # Cleanup
        try:
            if os.path.exists(processed_path):
                os.remove(processed_path)
        except Exception:
            pass

        return result
    except Exception as e:
        logger.error(f"Voice processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if MULTIPART_AVAILABLE:
    @app.post("/api/voice/upload")
    async def process_voice_upload(audio: UploadFile = File(...)):
        """Process uploaded audio file (multipart)"""
        content = await audio.read()
        return await _process_audio_content(content, getattr(audio, 'filename', 'voice_input.webm'))
else:
    @app.post("/api/voice/upload")
    async def process_voice_upload_raw(request: Request):
        """Process raw uploaded audio when multipart is unavailable"""
        content = await request.body()
        return await _process_audio_content(content, 'voice_input.webm')

# Aliases to comply with IMPORTANT canonical API surface
@app.post("/jarvis/task/plan", response_model=TaskResponse)
async def jarvis_task_plan(request: TaskRequest):
    # Delegate to the same handler logic as /api/task
    return await execute_task(request)

if MULTIPART_AVAILABLE:
    @app.post("/jarvis/voice/process")
    async def jarvis_voice_process(audio: UploadFile = File(...)):
        return await process_voice_upload(audio)
else:
    @app.post("/jarvis/voice/process")
    async def jarvis_voice_process_raw(request: Request):
        return await process_voice_upload_raw(request)

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
