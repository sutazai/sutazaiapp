"""
JARVIS WebSocket Handler
Real-time communication for voice and text interactions
Inspired by danilofalcao/jarvis WebSocket implementation
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.services.jarvis_orchestrator import JARVISOrchestrator, TaskType
from app.services.voice_pipeline import VoicePipeline, VoiceConfig, ASRProvider, TTSProvider
from app.core.database import get_db
from app.api.dependencies.auth import get_current_user
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

# WebSocket connection manager
class ConnectionManager:
    """
    Manages WebSocket connections for real-time communication
    Based on danilofalcao's multi-client architecture
    """
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        self.voice_pipelines: Dict[str, VoicePipeline] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str, user_id: Optional[str] = None):
        """Accept new WebSocket connection"""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
        # Initialize session
        self.user_sessions[client_id] = {
            "user_id": user_id,
            "connected_at": datetime.now(),
            "conversation_history": [],
            "context": {},
            "preferences": {}
        }
        
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        """Remove WebSocket connection"""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        
        if client_id in self.user_sessions:
            del self.user_sessions[client_id]
        
        if client_id in self.voice_pipelines:
            self.voice_pipelines[client_id].stop()
            del self.voice_pipelines[client_id]
        
        logger.info(f"Client {client_id} disconnected")
    
    async def send_message(self, client_id: str, message: Dict[str, Any]):
        """Send message to specific client"""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            await websocket.send_json(message)
    
    async def broadcast(self, message: Dict[str, Any], exclude: Optional[str] = None):
        """Broadcast message to all connected clients"""
        for client_id, websocket in self.active_connections.items():
            if client_id != exclude:
                await websocket.send_json(message)
    
    def get_session(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get user session data"""
        return self.user_sessions.get(client_id)
    
    def update_session(self, client_id: str, data: Dict[str, Any]):
        """Update user session data"""
        if client_id in self.user_sessions:
            self.user_sessions[client_id].update(data)


# Global connection manager
manager = ConnectionManager()

# Initialize JARVIS orchestrator
jarvis_config = {
    "enable_local_models": True,
    "enable_web_search": True,
    "max_context_length": 10000,
    "response_streaming": True
}
jarvis_orchestrator = JARVISOrchestrator(jarvis_config)


async def websocket_endpoint(
    websocket: WebSocket,
    client_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Main WebSocket endpoint for JARVIS interactions
    Handles both text and voice communications with real-time streaming
    """
    
    # Generate client ID if not provided
    if not client_id:
        client_id = str(uuid.uuid4())
    
    # Connect client
    await manager.connect(websocket, client_id)
    
    try:
        # Send welcome message
        await manager.send_message(client_id, {
            "type": "connection",
            "status": "connected",
            "client_id": client_id,
            "message": "Welcome to JARVIS. How may I assist you today?",
            "capabilities": {
                "voice": True,
                "text": True,
                "streaming": True,
                "tools": ["web_search", "calculator", "code_execution"],
                "models": ["tinyllama:latest"]
            }
        })
        
        # Main message loop
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message_type = data.get("type", "text")
            
            # Route based on message type
            if message_type == "text":
                await handle_text_message(client_id, data)
            
            elif message_type == "voice":
                await handle_voice_message(client_id, data)
            
            elif message_type == "config":
                await handle_config_update(client_id, data)
            
            elif message_type == "command":
                await handle_command(client_id, data)
            
            elif message_type == "ping":
                # Heartbeat
                await manager.send_message(client_id, {"type": "pong"})
            
            else:
                await manager.send_message(client_id, {
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                })
    
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected normally")
    
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}")
        manager.disconnect(client_id)
        
        # Try to send error message if connection still open
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Server error: {str(e)}"
            })
        except:
            pass


async def handle_text_message(client_id: str, data: Dict[str, Any]):
    """
    Handle text message from client
    Implements streaming responses inspired by danilofalcao
    """
    message = data.get("message", "")
    stream = data.get("stream", True)
    include_context = data.get("include_context", True)
    
    # Get session context
    session = manager.get_session(client_id)
    context = session.get("context", {}) if include_context else {}
    
    # Add to conversation history
    session["conversation_history"].append({
        "role": "user",
        "content": message,
        "timestamp": datetime.now().isoformat()
    })
    
    # Send processing status
    await manager.send_message(client_id, {
        "type": "status",
        "status": "processing",
        "message": "Processing your request..."
    })
    
    try:
        if stream:
            # Stream response tokens as they're generated
            response_text = ""
            async for token in jarvis_orchestrator.stream_process(message, context):
                response_text += token
                await manager.send_message(client_id, {
                    "type": "stream",
                    "token": token,
                    "partial": response_text
                })
            
            # Send final response
            response = {
                "success": True,
                "response": response_text,
                "metadata": {
                    "streamed": True,
                    "timestamp": datetime.now().isoformat()
                }
            }
        else:
            # Process without streaming
            response = await jarvis_orchestrator.process(message, context)
        
        # Add to conversation history
        session["conversation_history"].append({
            "role": "assistant",
            "content": response.get("response", ""),
            "timestamp": datetime.now().isoformat(),
            "metadata": response.get("metadata", {})
        })
        
        # Update context for next interaction
        session["context"]["last_response"] = response.get("response", "")
        session["context"]["last_model"] = response.get("metadata", {}).get("model_used", "")
        
        # Send response
        await manager.send_message(client_id, {
            "type": "response",
            "data": response
        })
    
    except Exception as e:
        logger.error(f"Error processing text message: {str(e)}")
        await manager.send_message(client_id, {
            "type": "error",
            "message": f"Failed to process message: {str(e)}"
        })


async def handle_voice_message(client_id: str, data: Dict[str, Any]):
    """
    Handle voice message from client
    Processes audio data through voice pipeline
    """
    audio_data = data.get("audio", "")  # Base64 encoded audio
    format = data.get("format", "wav")
    
    # Initialize voice pipeline if not exists
    if client_id not in manager.voice_pipelines:
        config = VoiceConfig(
            wake_word="jarvis",
            asr_provider=ASRProvider.AUTO,
            tts_provider=TTSProvider.AUTO,
            enable_wake_word=False,  # Already triggered from client
            enable_interruption=True
        )
        
        # Create callback for voice commands
        async def voice_callback(text: str) -> str:
            # Process through JARVIS
            session = manager.get_session(client_id)
            context = session.get("context", {})
            response = await jarvis_orchestrator.process(text, context)
            return response.get("response", "")
        
        manager.voice_pipelines[client_id] = VoicePipeline(config, voice_callback)
    
    pipeline = manager.voice_pipelines[client_id]
    
    try:
        # Decode and process audio
        import base64
        import io
        import wave
        
        audio_bytes = base64.b64decode(audio_data)
        
        # Convert to AudioData format
        # This is simplified - in production, handle various audio formats
        try:
            from speech_recognition import AudioData
            audio = AudioData(audio_bytes, 16000, 2)
        except ImportError:
            await manager.send_message(client_id, {
                "type": "error",
                "message": "Speech recognition not available on server"
            })
            return
        
        # Recognize speech
        text = await pipeline._recognize_speech(audio)
        
        if text:
            # Send transcription
            await manager.send_message(client_id, {
                "type": "transcription",
                "text": text
            })
            
            # Process as text message
            await handle_text_message(client_id, {"message": text})
        else:
            await manager.send_message(client_id, {
                "type": "error",
                "message": "Could not recognize speech"
            })
    
    except Exception as e:
        logger.error(f"Error processing voice message: {str(e)}")
        await manager.send_message(client_id, {
            "type": "error",
            "message": f"Failed to process voice: {str(e)}"
        })


async def handle_config_update(client_id: str, data: Dict[str, Any]):
    """
    Handle configuration updates from client
    Allows dynamic adjustment of preferences
    """
    config = data.get("config", {})
    session = manager.get_session(client_id)
    
    if session:
        # Update preferences
        session["preferences"].update(config)
        
        # Apply to voice pipeline if exists
        if client_id in manager.voice_pipelines and "voice" in config:
            pipeline = manager.voice_pipelines[client_id]
            voice_config = config["voice"]
            
            if "language" in voice_config:
                pipeline.config.language = voice_config["language"]
            if "asr_provider" in voice_config:
                pipeline.config.asr_provider = ASRProvider[voice_config["asr_provider"].upper()]
            if "tts_provider" in voice_config:
                pipeline.config.tts_provider = TTSProvider[voice_config["tts_provider"].upper()]
        
        await manager.send_message(client_id, {
            "type": "config_updated",
            "config": session["preferences"]
        })


async def handle_command(client_id: str, data: Dict[str, Any]):
    """
    Handle special commands (clear history, reset context, etc.)
    """
    command = data.get("command", "")
    session = manager.get_session(client_id)
    
    if command == "clear_history":
        session["conversation_history"] = []
        session["context"] = {}
        await manager.send_message(client_id, {
            "type": "command_response",
            "command": command,
            "status": "success",
            "message": "Conversation history cleared"
        })
    
    elif command == "get_history":
        await manager.send_message(client_id, {
            "type": "command_response",
            "command": command,
            "data": session["conversation_history"]
        })
    
    elif command == "get_metrics":
        # Get metrics from orchestrator and voice pipeline
        metrics = {
            "orchestrator": jarvis_orchestrator.metrics,
            "voice": manager.voice_pipelines[client_id].get_metrics() if client_id in manager.voice_pipelines else {}
        }
        await manager.send_message(client_id, {
            "type": "command_response",
            "command": command,
            "data": metrics
        })
    
    else:
        await manager.send_message(client_id, {
            "type": "command_response",
            "command": command,
            "status": "error",
            "message": f"Unknown command: {command}"
        })


# Additional endpoints for REST API integration

from fastapi import APIRouter

router = APIRouter()

@router.get("/jarvis/connections")
async def get_connections():
    """Get list of active WebSocket connections"""
    return {
        "connections": list(manager.active_connections.keys()),
        "count": len(manager.active_connections)
    }


@router.get("/jarvis/session/{client_id}")
async def get_session(client_id: str):
    """Get session data for specific client"""
    session = manager.get_session(client_id)
    if session:
        return session
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found for client {client_id}"
        )


@router.post("/jarvis/broadcast")
async def broadcast_message(message: Dict[str, Any]):
    """Broadcast message to all connected clients"""
    await manager.broadcast(message)
    return {"status": "broadcast sent", "recipients": len(manager.active_connections)}