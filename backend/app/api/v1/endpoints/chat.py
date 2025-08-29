"""Chat and conversation endpoints with Ollama integration"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import httpx
import json
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

router = APIRouter()

# Ollama configuration from environment
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "sutazai-ollama")
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
        async with httpx.AsyncClient(timeout=30.0) as client:
            url = f"{OLLAMA_BASE_URL}/api/generate"
            payload = {
                "model": model,
                "prompt": message,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            
            logger.info(f"Calling Ollama at {url} with model {model}")
            
            response = await client.post(url, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No response generated")
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
    except Exception as e:
        logger.error(f"Unexpected error calling Ollama: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )

@router.post("/message", response_model=ChatResponse)
async def send_message(chat: ChatMessage) -> ChatResponse:
    """Process chat message through Ollama LLM"""
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
        "user_id": chat.user_id
    })
    
    try:
        # Call Ollama for response
        response_text = await call_ollama(
            message=chat.message,
            model=chat.model,
            temperature=chat.temperature
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
async def get_session(session_id: str) -> dict:
    """Get chat session history"""
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
async def clear_session(session_id: str) -> dict:
    """Clear a chat session"""
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