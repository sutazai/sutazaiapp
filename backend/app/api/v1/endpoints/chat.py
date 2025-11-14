"""Chat and conversation endpoints with Ollama integration"""

from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Annotated
import os
import httpx
import json
import logging
from datetime import datetime
import uuid
import asyncio

from app.api.dependencies.auth import get_current_active_user, get_current_user_optional
from app.models.user import User

logger = logging.getLogger(__name__)

router = APIRouter()

# Ollama configuration - using Docker service name
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "host.docker.internal")
OLLAMA_PORT = os.getenv("OLLAMA_PORT", "11434")
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"

# In-memory session storage (replace with Redis/DB in production)
chat_sessions: Dict[str, List[Dict[str, Any]]] = {}

class ChatMessage(BaseModel):
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    model: Optional[str] = "tinyllama:latest"
    stream: Optional[bool] = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000

class ChatResponse(BaseModel):
    response: str
    session_id: str
    model: str
    status: str
    timestamp: str
    response_time: Optional[float] = None

async def call_ollama(message: str, model: str = "tinyllama:latest", temperature: float = 0.7) -> str:
    """Call Ollama API for text generation"""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            url = f"{OLLAMA_BASE_URL}/api/generate"
            
            # Ensure we use tinyllama model
            if not model or model == "tinyllama":
                model = "tinyllama:latest"
                
            payload = {
                "model": model,
                "prompt": message,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 500  # Limit response length
                }
            }
            
            logger.info(f"Calling Ollama at {url} with model {model}")
            logger.debug(f"Payload: {payload}")
            
            response = await client.post(url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "No response generated")
                logger.info(f"Got response from Ollama: {response_text[:100]}...")
                return response_text
            else:
                error_text = response.text
                logger.error(f"Ollama API error: {response.status_code} - {error_text}")
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Ollama API error: {error_text}"
                )
                    
    except httpx.ConnectError as e:
        logger.error(f"Connection error to Ollama: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Ollama service: {str(e)}"
        )
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Unexpected error calling Ollama: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(
    chat: ChatMessage,
    current_user: Optional[User] = Depends(get_current_user_optional)
) -> ChatResponse:
    """Main chat endpoint - process message through Ollama LLM (auth optional for testing)"""
    import time
    start_time = time.time()
    
    # Generate session ID if not provided
    session_id = chat.session_id or str(uuid.uuid4())
    
    # Store message in session history
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    
    chat_sessions[session_id].append({
        "role": "user",
        "content": chat.message,
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": str(current_user.id) if current_user else "anonymous"  # Use authenticated user's ID or anonymous
    })
    
    try:
        # Call Ollama for response
        response_text = await call_ollama(
            message=chat.message,
            model=chat.model or "tinyllama:latest",
            temperature=chat.temperature or 0.7
        )
        
        # Store assistant response in session
        chat_sessions[session_id].append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.utcnow().isoformat(),
            "model": chat.model
        })
        
        response_time = time.time() - start_time
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            model=chat.model,
            status="success",
            timestamp=datetime.utcnow().isoformat(),
            response_time=response_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process message: {str(e)}"
        )

@router.post("/message", response_model=ChatResponse)
async def send_message(
    chat: ChatMessage,
    current_user: Optional[User] = Depends(get_current_user_optional)
) -> ChatResponse:
    """Process chat message through Ollama LLM (legacy endpoint - auth optional for testing)"""
    import time
    start_time = time.time()
    
    # Generate session ID if not provided
    session_id = chat.session_id or str(uuid.uuid4())
    
    # Store message in session history
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    
    chat_sessions[session_id].append({
        "role": "user",
        "content": chat.message,
        "timestamp": datetime.utcnow().isoformat(),
        "user_id": str(current_user.id) if current_user else "anonymous"  # Use authenticated user's ID or anonymous
    })
    
    try:
        # Call Ollama for response
        response_text = await call_ollama(
            message=chat.message,
            model=chat.model or "tinyllama:latest",
            temperature=chat.temperature or 0.7
        )
        
        # Store assistant response in session
        chat_sessions[session_id].append({
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.utcnow().isoformat(),
            "model": chat.model
        })
        
        response_time = time.time() - start_time
        
        return ChatResponse(
            response=response_text,
            session_id=session_id,
            model=chat.model,
            status="success",
            timestamp=datetime.utcnow().isoformat(),
            response_time=response_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process message: {str(e)}"
        )

@router.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> dict:
    """Get chat session history (requires authentication)"""
    if session_id not in chat_sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )
    
    return {
        "session_id": session_id,
        "messages": chat_sessions[session_id],
        "message_count": len(chat_sessions[session_id]),
        "created_at": chat_sessions[session_id][0]["timestamp"] if chat_sessions[session_id] else None
    }

@router.get("/models")
async def list_models() -> dict:
    """List available Ollama models"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            url = f"{OLLAMA_BASE_URL}/api/tags"
            response = await client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                return {
                    "models": [
                        {
                            "name": model.get("name"),
                            "size": model.get("size"),
                            "modified": model.get("modified_at")
                        }
                        for model in models
                    ],
                    "count": len(models),
                    "ollama_url": OLLAMA_BASE_URL
                }
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail="Failed to fetch models from Ollama"
                )
    except httpx.ConnectError as e:
        logger.error(f"Cannot connect to Ollama: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Ollama service at {OLLAMA_BASE_URL}"
        )

@router.get("/health")
async def check_ollama_health() -> dict:
    """Check Ollama service health"""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            url = f"{OLLAMA_BASE_URL}/api/tags"
            response = await client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "status": "healthy",
                    "ollama_url": OLLAMA_BASE_URL,
                    "model_count": len(data.get("models", [])),
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {
                    "status": "unhealthy",
                    "ollama_url": OLLAMA_BASE_URL,
                    "error": f"HTTP {response.status_code}",
                    "timestamp": datetime.utcnow().isoformat()
                }
    except Exception as e:
        return {
            "status": "unhealthy",
            "ollama_url": OLLAMA_BASE_URL,
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.delete("/sessions/{session_id}")
async def clear_session(
    session_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> dict:
    """Clear a chat session (requires authentication)"""
    if session_id in chat_sessions:
        message_count = len(chat_sessions[session_id])
        del chat_sessions[session_id]
        return {
            "status": "success",
            "message": f"Session {session_id} cleared",
            "messages_deleted": message_count
        }
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )


class WebSocketManager:
    """Manages WebSocket connections for streaming chat"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.session_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, session_id: str):
        """Accept and track a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.session_connections[session_id] = websocket
        logger.info(f"WebSocket connected for session {session_id}")
    
    def disconnect(self, websocket: WebSocket, session_id: str):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        if session_id in self.session_connections:
            del self.session_connections[session_id]
        logger.info(f"WebSocket disconnected for session {session_id}")
    
    async def send_message(self, websocket: WebSocket, message: dict):
        """Send a JSON message to a specific WebSocket"""
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
    
    async def broadcast(self, message: dict):
        """Broadcast a message to all connected WebSockets"""
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting message: {e}")


# Global WebSocket manager
ws_manager = WebSocketManager()


async def stream_ollama_response(prompt: str, model: str = "tinyllama:latest", temperature: float = 0.7):
    """Stream responses from Ollama API"""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            url = f"{OLLAMA_BASE_URL}/api/generate"
            
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": True,  # Enable streaming
                "options": {
                    "temperature": temperature,
                    "num_predict": 500
                }
            }
            
            async with client.stream('POST', url, json=payload) as response:
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            yield data
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse JSON: {line}")
                            continue
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield {"error": str(e), "done": True}


@router.websocket("/ws")
async def websocket_chat_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time streaming chat"""
    session_id = str(uuid.uuid4())
    await ws_manager.connect(websocket, session_id)
    
    # Initialize session
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    
    try:
        # Send welcome message
        await ws_manager.send_message(websocket, {
            "type": "connection",
            "status": "connected",
            "session_id": session_id,
            "message": "WebSocket chat connected successfully"
        })
        
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # Handle different message types
            message_type = data.get("type", "chat")
            
            if message_type == "ping":
                # Heartbeat response
                await ws_manager.send_message(websocket, {
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })
                continue
            
            elif message_type == "chat":
                # Process chat message
                user_message = data.get("message", "")
                model = data.get("model", "tinyllama:latest")
                temperature = data.get("temperature", 0.7)
                stream = data.get("stream", True)
                
                if not user_message:
                    await ws_manager.send_message(websocket, {
                        "type": "error",
                        "message": "No message provided"
                    })
                    continue
                
                # Store user message
                chat_sessions[session_id].append({
                    "role": "user",
                    "content": user_message,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                # Send acknowledgment
                await ws_manager.send_message(websocket, {
                    "type": "message_received",
                    "message": user_message,
                    "timestamp": datetime.utcnow().isoformat()
                })
                
                if stream:
                    # Stream response from Ollama
                    full_response = ""
                    await ws_manager.send_message(websocket, {
                        "type": "stream_start",
                        "model": model
                    })
                    
                    async for chunk in stream_ollama_response(user_message, model, temperature):
                        if "error" in chunk:
                            await ws_manager.send_message(websocket, {
                                "type": "error",
                                "message": chunk["error"]
                            })
                            break
                        
                        if "response" in chunk:
                            full_response += chunk["response"]
                            await ws_manager.send_message(websocket, {
                                "type": "stream_chunk",
                                "content": chunk["response"],
                                "done": chunk.get("done", False)
                            })
                        
                        if chunk.get("done", False):
                            # Store assistant response
                            chat_sessions[session_id].append({
                                "role": "assistant",
                                "content": full_response,
                                "timestamp": datetime.utcnow().isoformat(),
                                "model": model
                            })
                            
                            await ws_manager.send_message(websocket, {
                                "type": "stream_end",
                                "full_response": full_response,
                                "timestamp": datetime.utcnow().isoformat()
                            })
                            break
                else:
                    # Non-streaming response
                    response_text = await call_ollama(user_message, model, temperature)
                    
                    # Store assistant response
                    chat_sessions[session_id].append({
                        "role": "assistant",
                        "content": response_text,
                        "timestamp": datetime.utcnow().isoformat(),
                        "model": model
                    })
                    
                    await ws_manager.send_message(websocket, {
                        "type": "response",
                        "content": response_text,
                        "model": model,
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            elif message_type == "get_history":
                # Send chat history
                await ws_manager.send_message(websocket, {
                    "type": "history",
                    "messages": chat_sessions.get(session_id, []),
                    "count": len(chat_sessions.get(session_id, []))
                })
            
            elif message_type == "clear_history":
                # Clear session history
                chat_sessions[session_id] = []
                await ws_manager.send_message(websocket, {
                    "type": "history_cleared",
                    "session_id": session_id
                })
                
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, session_id)
        logger.info(f"Client {session_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        try:
            await ws_manager.send_message(websocket, {
                "type": "error",
                "message": str(e)
            })
        except:
            pass
        ws_manager.disconnect(websocket, session_id)
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )