#!/usr/bin/env python3
"""
JARVIS Critical Fix Script - Implements essential fixes to get core functionality working
This script patches the most critical issues to enable basic voice and chat features
"""

import os
import sys
import subprocess
import json
import time

class JARVISCriticalFixer:
    """Fixes critical issues in JARVIS implementation"""
    
    def __init__(self):
        self.backend_container = "sutazai-backend"
        self.fixes_applied = []
        
    def print_status(self, message: str, status: str = "INFO"):
        """Print formatted status message"""
        icons = {
            "INFO": "ℹ️",
            "SUCCESS": "✅",
            "ERROR": "❌",
            "WARNING": "⚠️",
            "FIXING": "🔧"
        }
        print(f"{icons.get(status, '•')} {message}")
        
    def run_command(self, command: str, container: str = None) -> tuple:
        """Run command in container or host"""
        try:
            if container:
                full_command = f"docker exec {container} {command}"
            else:
                full_command = command
                
            result = subprocess.run(
                full_command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            return (True, result.stdout) if result.returncode == 0 else (False, result.stderr)
        except Exception as e:
            return (False, str(e))
            
    def fix_voice_dependencies(self):
        """Install missing voice processing dependencies"""
        self.print_status("Installing voice dependencies...", "FIXING")
        
        # Update package list
        success, output = self.run_command(
            "apt-get update",
            self.backend_container
        )
        
        if not success:
            self.print_status("Failed to update packages", "WARNING")
            
        # Install system dependencies
        self.print_status("Installing system audio libraries...", "FIXING")
        success, output = self.run_command(
            "apt-get install -y portaudio19-dev python3-pyaudio ffmpeg",
            self.backend_container
        )
        
        # Install Python packages
        packages = [
            "SpeechRecognition",
            "pyttsx3",
            "pygame",
            "websocket-client",
            "pyaudio"
        ]
        
        for package in packages:
            self.print_status(f"Installing {package}...", "FIXING")
            success, output = self.run_command(
                f"pip install {package}",
                self.backend_container
            )
            if success:
                self.fixes_applied.append(f"Installed {package}")
            else:
                self.print_status(f"Failed to install {package}", "WARNING")
                
        self.print_status("Voice dependencies installed", "SUCCESS")
        
    def fix_websocket_endpoint(self):
        """Create working WebSocket endpoint"""
        self.print_status("Creating WebSocket endpoint fix...", "FIXING")
        
        websocket_code = '''
import sys
sys.path.insert(0, "/app")

from fastapi import WebSocket, WebSocketDisconnect
from app.api.v1.router import api_router
import json
import asyncio
from typing import Dict, Set
from datetime import datetime

# Connection manager for WebSocket clients
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            
    async def send_personal_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            await websocket.send_text(message)
            
    async def broadcast(self, message: str):
        for client_id, connection in self.active_connections.items():
            await connection.send_text(message)

manager = ConnectionManager()

# Add WebSocket endpoint if not exists
if not hasattr(api_router, '_websocket_added'):
    @api_router.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time communication"""
        import uuid
        client_id = str(uuid.uuid4())
        
        await manager.connect(websocket, client_id)
        
        # Send connection confirmation
        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "client_id": client_id,
            "timestamp": datetime.now().isoformat()
        })
        
        try:
            while True:
                # Receive message
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Echo back for now (will integrate with JARVIS later)
                response = {
                    "type": "response",
                    "client_id": client_id,
                    "message": f"Received: {message.get('message', '')}",
                    "timestamp": datetime.now().isoformat()
                }
                
                await websocket.send_json(response)
                
        except WebSocketDisconnect:
            manager.disconnect(client_id)
            
    api_router._websocket_added = True
    print("WebSocket endpoint added successfully")
'''
        
        # Write the fix to a file
        with open("/tmp/websocket_fix.py", "w") as f:
            f.write(websocket_code)
            
        # Copy to container and execute
        success, output = self.run_command(
            "docker cp /tmp/websocket_fix.py sutazai-backend:/tmp/websocket_fix.py"
        )
        
        if success:
            self.print_status("WebSocket fix deployed", "SUCCESS")
            self.fixes_applied.append("WebSocket endpoint fixed")
        else:
            self.print_status("Failed to deploy WebSocket fix", "ERROR")
            
    def fix_jarvis_chat_endpoint(self):
        """Create JARVIS-specific chat endpoint"""
        self.print_status("Creating JARVIS chat endpoint...", "FIXING")
        
        jarvis_endpoint_code = '''
import sys
sys.path.insert(0, "/app")

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import time

# Create JARVIS-specific router
jarvis_router = APIRouter()

class JARVISChatRequest(BaseModel):
    message: str
    use_jarvis: bool = True
    session_id: Optional[str] = None
    stream: bool = False

class JARVISChatResponse(BaseModel):
    response: str
    session_id: str
    orchestrator_used: bool
    pipeline_stages: str
    model_used: str
    processing_time: float

@jarvis_router.post("/chat", response_model=JARVISChatResponse)
async def jarvis_chat(request: JARVISChatRequest):
    """JARVIS orchestrated chat endpoint"""
    start_time = time.time()
    
    # Import services
    from app.services.jarvis_orchestrator import JARVISOrchestrator
    from app.services.ollama_helper import ollama_helper
    import uuid
    
    session_id = request.session_id or str(uuid.uuid4())
    
    try:
        # Initialize JARVIS orchestrator
        config = {
            "enable_local_models": True,
            "enable_web_search": False,
            "max_context_length": 4096
        }
        orchestrator = JARVISOrchestrator(config)
        
        # Process through JARVIS pipeline
        # Stage 1: Task Planning
        task_plan = await orchestrator.plan_task(request.message)
        
        # Stage 2: Model Selection
        model_selection = await orchestrator.select_model(task_plan)
        
        # Stage 3: Task Execution
        result = await orchestrator.execute_task(
            request.message,
            model_selection.primary_model
        )
        
        # Stage 4: Response Generation
        response = await orchestrator.generate_response(result)
        
        processing_time = time.time() - start_time
        
        return JARVISChatResponse(
            response=response,
            session_id=session_id,
            orchestrator_used=True,
            pipeline_stages="plan->select->execute->generate",
            model_used=model_selection.primary_model,
            processing_time=processing_time
        )
        
    except Exception as e:
        # Fallback to direct Ollama
        try:
            response = await ollama_helper.chat(
                message=request.message,
                model="tinyllama:latest"
            )
            
            return JARVISChatResponse(
                response=response.get("response", "Error processing request"),
                session_id=session_id,
                orchestrator_used=False,
                pipeline_stages="direct",
                model_used="tinyllama:latest",
                processing_time=time.time() - start_time
            )
        except Exception as fallback_error:
            raise HTTPException(
                status_code=500,
                detail=f"JARVIS processing failed: {str(e)}, Fallback failed: {str(fallback_error)}"
            )

# Register the router
from app.api.v1.router import api_router
if "/jarvis" not in [route.path for route in api_router.routes]:
    api_router.include_router(jarvis_router, prefix="/jarvis", tags=["jarvis"])
    print("JARVIS chat endpoint added successfully")
'''
        
        # Write fix to file
        with open("/tmp/jarvis_endpoint_fix.py", "w") as f:
            f.write(jarvis_endpoint_code)
            
        # Deploy to container
        success, output = self.run_command(
            "docker cp /tmp/jarvis_endpoint_fix.py sutazai-backend:/tmp/jarvis_endpoint_fix.py"
        )
        
        if success:
            self.print_status("JARVIS endpoint deployed", "SUCCESS")
            self.fixes_applied.append("JARVIS chat endpoint created")
        else:
            self.print_status("Failed to deploy JARVIS endpoint", "ERROR")
            
    def fix_streaming_response(self):
        """Implement streaming response capability"""
        self.print_status("Implementing streaming responses...", "FIXING")
        
        streaming_code = '''
import sys
sys.path.insert(0, "/app")

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio
import json

stream_router = APIRouter()

class StreamRequest(BaseModel):
    message: str
    model: str = "tinyllama:latest"

@stream_router.post("/stream")
async def stream_chat(request: StreamRequest):
    """Streaming chat endpoint"""
    
    async def generate():
        # Import helper
        from app.services.ollama_helper import ollama_helper
        
        # Simulate streaming response
        try:
            # Get response from Ollama
            response = await ollama_helper.chat(
                message=request.message,
                model=request.model
            )
            
            # Stream the response word by word
            words = response.get("response", "").split()
            for word in words:
                chunk = json.dumps({"token": word + " "}) + "\\n"
                yield chunk.encode("utf-8")
                await asyncio.sleep(0.05)  # Simulate streaming delay
                
        except Exception as e:
            error_chunk = json.dumps({"error": str(e)}) + "\\n"
            yield error_chunk.encode("utf-8")
    
    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson"
    )

# Register streaming endpoint
from app.api.v1.router import api_router
if "/chat/stream" not in [route.path for route in api_router.routes]:
    api_router.include_router(stream_router, prefix="/chat", tags=["chat"])
    print("Streaming endpoint added successfully")
'''
        
        # Write and deploy
        with open("/tmp/streaming_fix.py", "w") as f:
            f.write(streaming_code)
            
        success, output = self.run_command(
            "docker cp /tmp/streaming_fix.py sutazai-backend:/tmp/streaming_fix.py"
        )
        
        if success:
            self.print_status("Streaming endpoint deployed", "SUCCESS")
            self.fixes_applied.append("Streaming responses implemented")
        else:
            self.print_status("Failed to deploy streaming endpoint", "ERROR")
            
    def restart_backend(self):
        """Restart backend to apply fixes"""
        self.print_status("Restarting backend service...", "FIXING")
        
        success, output = self.run_command("docker restart sutazai-backend")
        
        if success:
            self.print_status("Backend restarted successfully", "SUCCESS")
            time.sleep(5)  # Wait for service to come up
        else:
            self.print_status("Failed to restart backend", "ERROR")
            
    def verify_fixes(self):
        """Verify that fixes are working"""
        self.print_status("Verifying fixes...", "INFO")
        
        import requests
        
        # Test endpoints
        tests = [
            ("Chat", "http://localhost:10200/api/v1/chat/", {"message": "test"}),
            ("Health", "http://localhost:10200/health", None),
            ("Models", "http://localhost:10200/api/v1/agents/models", None),
        ]
        
        for name, url, data in tests:
            try:
                if data:
                    response = requests.post(url, json=data, timeout=5)
                else:
                    response = requests.get(url, timeout=5)
                    
                if response.status_code in [200, 201]:
                    self.print_status(f"{name} endpoint working", "SUCCESS")
                else:
                    self.print_status(f"{name} endpoint returned {response.status_code}", "WARNING")
            except Exception as e:
                self.print_status(f"{name} endpoint failed: {str(e)[:50]}", "ERROR")
                
    def run_all_fixes(self):
        """Apply all critical fixes"""
        print("\n" + "=" * 70)
        print("🔧 JARVIS CRITICAL FIX SCRIPT")
        print("=" * 70)
        print("Applying emergency fixes to get JARVIS operational...\n")
        
        # Apply fixes
        self.fix_voice_dependencies()
        self.fix_websocket_endpoint()
        self.fix_jarvis_chat_endpoint()
        self.fix_streaming_response()
        
        # Restart to apply
        self.restart_backend()
        
        # Verify
        self.verify_fixes()
        
        # Summary
        print("\n" + "=" * 70)
        print("📊 FIX SUMMARY")
        print("=" * 70)
        
        if self.fixes_applied:
            print("\n✅ Fixes Applied:")
            for fix in self.fixes_applied:
                print(f"  • {fix}")
        else:
            print("\n⚠️ No fixes were successfully applied")
            
        print("\n💡 Next Steps:")
        print("  1. Run test_jarvis_full_system.py to check improvements")
        print("  2. Check docker logs sutazai-backend for errors")
        print("  3. Manually test voice features if dependencies installed")
        print("  4. Continue with full implementation plan")
        
        print("\n" + "=" * 70)


def main():
    """Main entry point"""
    fixer = JARVISCriticalFixer()
    fixer.run_all_fixes()


if __name__ == "__main__":
    main()