"""
JARVIS Chat API - Unified AI chat endpoint with multi-model support
Integrates the JARVIS orchestrator for intelligent model selection and fallback
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, AsyncGenerator
import json
import asyncio
import logging
from datetime import datetime
import uuid
from app.services.jarvis_orchestrator import JARVISOrchestrator, TaskType

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize JARVIS orchestrator
jarvis_config = {
    "max_retries": 3,
    "timeout": 60,
    "enable_cache": True,
    "enable_tools": True
}
jarvis = JARVISOrchestrator(jarvis_config)

# Session storage (use Redis in production)
chat_sessions: Dict[str, List[Dict[str, Any]]] = {}

class ChatRequest(BaseModel):
    """Chat request model with all options"""
    message: str = Field(..., description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation continuity")
    model: Optional[str] = Field(None, description="Specific model to use (optional)")
    task_type: Optional[str] = Field("chat", description="Task type: chat, code, analysis, creative, etc.")
    stream: bool = Field(False, description="Enable streaming response")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Response randomness")
    max_tokens: int = Field(2000, ge=100, le=8000, description="Maximum response length")
    system_prompt: Optional[str] = Field(None, description="System prompt for context")
    prefer_local: bool = Field(True, description="Prefer local models for privacy")

class ChatResponse(BaseModel):
    """Chat response model"""
    response: str
    session_id: str
    model_used: str
    task_type: str
    latency: float
    status: str
    timestamp: str

class ModelInfo(BaseModel):
    """Model information"""
    name: str
    provider: str
    capabilities: List[str]
    available: bool
    local: bool
    quality_score: float

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint with JARVIS orchestration
    Automatically selects the best model and handles fallbacks
    """
    import time
    start_time = time.time()
    
    # Generate or use session ID
    session_id = request.session_id or str(uuid.uuid4())
    
    # Get or create session history
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    
    # Add user message to history
    chat_sessions[session_id].append({
        "role": "user",
        "content": request.message,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    # Build context for JARVIS
    context = {
        "history": chat_sessions[session_id][-10:],  # Last 10 messages
        "system_prompt": request.system_prompt,
        "prefer_local": request.prefer_local,
        "session_id": session_id
    }
    
    # Override task type if specified
    if request.task_type:
        context["task_type_override"] = request.task_type
    
    # Override model if specified
    if request.model:
        context["model_override"] = request.model
    
    try:
        # Process through JARVIS orchestrator
        result = await jarvis.process(request.message, context)
        
        if result["success"]:
            response_text = result["response"]
            model_used = result["metadata"]["model_used"]
            task_type = result["metadata"]["task_type"]
            
            # Add assistant response to history
            chat_sessions[session_id].append({
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.utcnow().isoformat(),
                "model": model_used
            })
            
            latency = time.time() - start_time
            
            return ChatResponse(
                response=response_text,
                session_id=session_id,
                model_used=model_used,
                task_type=task_type,
                latency=latency,
                status="success",
                timestamp=datetime.utcnow().isoformat()
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Processing failed: {result.get('error', 'Unknown error')}"
            )
            
    except Exception as e:
        logger.error(f"Chat processing error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat processing failed: {str(e)}"
        )

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint for real-time responses
    """
    async def generate() -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        session_id = request.session_id or str(uuid.uuid4())
        
        # Add to history
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        
        chat_sessions[session_id].append({
            "role": "user",
            "content": request.message,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        context = {
            "history": chat_sessions[session_id][-10:],
            "system_prompt": request.system_prompt,
            "prefer_local": request.prefer_local,
            "session_id": session_id,
            "stream": True
        }
        
        # Start with metadata
        yield f"data: {json.dumps({'type': 'start', 'session_id': session_id})}\n\n"
        
        try:
            # Stream tokens from JARVIS
            full_response = ""
            async for token in jarvis.stream_process(request.message, context):
                full_response += token
                yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
                await asyncio.sleep(0.01)  # Small delay for smooth streaming
            
            # Add to history
            chat_sessions[session_id].append({
                "role": "assistant",
                "content": full_response,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # End signal
            yield f"data: {json.dumps({'type': 'end', 'full_response': full_response})}\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@router.websocket("/chat/ws")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for real-time bidirectional chat
    """
    await websocket.accept()
    session_id = str(uuid.uuid4())
    
    try:
        await websocket.send_json({
            "type": "connected",
            "session_id": session_id,
            "message": "JARVIS connected and ready"
        })
        
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            
            message = data.get("message", "")
            if not message:
                continue
            
            # Add to history
            if session_id not in chat_sessions:
                chat_sessions[session_id] = []
            
            chat_sessions[session_id].append({
                "role": "user",
                "content": message,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Build context
            context = {
                "history": chat_sessions[session_id][-10:],
                "system_prompt": data.get("system_prompt"),
                "prefer_local": data.get("prefer_local", True),
                "session_id": session_id
            }
            
            # Send typing indicator
            await websocket.send_json({"type": "typing", "status": "thinking"})
            
            try:
                # Process through JARVIS
                if data.get("stream", False):
                    # Stream response
                    full_response = ""
                    async for token in jarvis.stream_process(message, context):
                        full_response += token
                        await websocket.send_json({
                            "type": "token",
                            "content": token
                        })
                        await asyncio.sleep(0.01)
                    
                    # Add to history
                    chat_sessions[session_id].append({
                        "role": "assistant",
                        "content": full_response,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                    # Send completion
                    await websocket.send_json({
                        "type": "complete",
                        "response": full_response
                    })
                else:
                    # Non-streaming response
                    result = await jarvis.process(message, context)
                    
                    if result["success"]:
                        response = result["response"]
                        
                        # Add to history
                        chat_sessions[session_id].append({
                            "role": "assistant",
                            "content": response,
                            "timestamp": datetime.utcnow().isoformat(),
                            "model": result["metadata"]["model_used"]
                        })
                        
                        await websocket.send_json({
                            "type": "response",
                            "content": response,
                            "model": result["metadata"]["model_used"],
                            "task_type": result["metadata"]["task_type"]
                        })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "error": result.get("error", "Processing failed")
                        })
                        
            except Exception as e:
                logger.error(f"WebSocket processing error: {str(e)}")
                await websocket.send_json({
                    "type": "error",
                    "error": str(e)
                })
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket.close()

@router.get("/models", response_model=List[ModelInfo])
async def get_available_models() -> List[ModelInfo]:
    """
    Get list of available models with their capabilities and status
    """
    models = []
    
    for model_name, model_info in jarvis.model_registry.items():
        # Check availability (simplified - in production, actually test each provider)
        available = True
        if model_info["provider"].value == "OLLAMA":
            # Check if Ollama model is available
            import httpx
            import os
            try:
                ollama_host = os.getenv("OLLAMA_HOST", "host.docker.internal")
                ollama_port = os.getenv("OLLAMA_PORT", "11434")
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"http://{ollama_host}:{ollama_port}/api/tags")
                    if response.status_code == 200:
                        ollama_models = response.json().get("models", [])
                        model_names = [m["name"] for m in ollama_models]
                        # Map to check
                        model_mapping = {
                            "llama-3-70b": "llama3:latest",
                            "mistral-7b": "mistral:latest",
                            "tinyllama": "tinyllama:latest"
                        }
                        available = model_mapping.get(model_name, "") in model_names
            except:
                available = False
        elif model_info["provider"].value == "OPENAI":
            import os
            available = bool(os.getenv("OPENAI_API_KEY"))
        elif model_info["provider"].value == "ANTHROPIC":
            import os
            available = bool(os.getenv("ANTHROPIC_API_KEY"))
        
        models.append(ModelInfo(
            name=model_name,
            provider=model_info["provider"].value,
            capabilities=[cap.value for cap in model_info["capabilities"]],
            available=available,
            local=model_info.get("local", False),
            quality_score=model_info["quality"]
        ))
    
    return models

@router.get("/sessions/{session_id}/history")
async def get_session_history(session_id: str, limit: int = 50) -> dict:
    """
    Get chat history for a session
    """
    if session_id not in chat_sessions:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found"
        )
    
    messages = chat_sessions[session_id]
    return {
        "session_id": session_id,
        "messages": messages[-limit:] if len(messages) > limit else messages,
        "total_messages": len(messages),
        "created_at": messages[0]["timestamp"] if messages else None
    }

@router.delete("/sessions/{session_id}")
async def clear_session(session_id: str) -> dict:
    """
    Clear a chat session
    """
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

@router.get("/health")
async def health_check() -> dict:
    """
    Check JARVIS chat system health
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "orchestrator": "active",
        "sessions_active": len(chat_sessions),
        "models": {}
    }
    
    # Check each provider
    import httpx
    import os
    
    # Check Ollama
    try:
        ollama_host = os.getenv("OLLAMA_HOST", "host.docker.internal")
        ollama_port = os.getenv("OLLAMA_PORT", "11434")
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"http://{ollama_host}:{ollama_port}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                health_status["models"]["ollama"] = {
                    "status": "healthy",
                    "models_available": len(models)
                }
            else:
                health_status["models"]["ollama"] = {"status": "unhealthy"}
    except:
        health_status["models"]["ollama"] = {"status": "offline"}
    
    # Check API keys
    health_status["models"]["openai"] = {
        "status": "configured" if os.getenv("OPENAI_API_KEY") else "not_configured"
    }
    health_status["models"]["anthropic"] = {
        "status": "configured" if os.getenv("ANTHROPIC_API_KEY") else "not_configured"
    }
    
    return health_status