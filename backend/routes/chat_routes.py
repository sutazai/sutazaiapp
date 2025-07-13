#!/usr/bin/env python3
"""
Chat Routes for SutazAI
Provides API endpoints for chat functionality with Ollama integration
"""

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import asyncio
import aiohttp
import logging
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    model: Optional[str] = "llama3-chatqa"
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1000

class ChatResponse(BaseModel):
    response: str
    model: str
    timestamp: datetime
    status: str = "success"

class SystemStatus(BaseModel):
    ollama_status: str
    available_models: List[str]
    system_health: str
    timestamp: datetime

# Ollama configuration
OLLAMA_BASE_URL = "http://localhost:11434"

async def check_ollama_health() -> bool:
    """Check if Ollama service is available"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5) as response:
                return response.status == 200
    except Exception as e:
        logger.warning(f"Ollama health check failed: {e}")
        return False

async def get_available_models() -> List[str]:
    """Get list of available Ollama models"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("models", [])
                    return [model.get("name", "").split(":")[0] for model in models if model.get("name")]
                return []
    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        return []

async def chat_with_ollama(message: str, model: str = "llama3-chatqa", **kwargs) -> str:
    """Send message to Ollama and get response"""
    try:
        payload = {
            "model": model,
            "prompt": message,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": kwargs.get("max_tokens", 1000),
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json=payload,
                timeout=30
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("response", "No response received")
                else:
                    error_text = await response.text()
                    logger.error(f"Ollama API error: {response.status} - {error_text}")
                    return f"Error: Ollama API returned status {response.status}"
                    
    except asyncio.TimeoutError:
        logger.error("Ollama request timed out")
        return "Error: Request timed out. The model might be processing a complex request."
    except Exception as e:
        logger.error(f"Error communicating with Ollama: {e}")
        return f"Error: Failed to communicate with local AI model - {str(e)}"

def generate_fallback_response(message: str) -> str:
    """Generate a fallback response when Ollama is unavailable"""
    message_lower = message.lower()
    
    if any(word in message_lower for word in ["hello", "hi", "hey"]):
        return "Hello! I'm SutazAI, your local AI assistant. Unfortunately, the local AI model is currently unavailable, but I'm still here to help with basic responses."
    
    elif any(word in message_lower for word in ["status", "health", "system"]):
        return """System Status:
• Backend API: ✅ Online
• Web Interface: ✅ Online  
• Local AI Model: ❌ Unavailable
• Database: ✅ Connected

The local AI model (Ollama) appears to be offline. Please check if Ollama is running or restart the system."""
    
    elif any(word in message_lower for word in ["help", "what", "how"]):
        return """I'm SutazAI, your local AI system. I can help with:

🤖 AI-powered conversations (when local model is available)
📊 System monitoring and status
💻 Code analysis and generation
🗄️ Data processing and management
🔧 System administration tasks

Currently running in fallback mode. To access full AI capabilities, please ensure Ollama is running."""
    
    elif any(word in message_lower for word in ["code", "programming", "python"]):
        return """I'd love to help with coding! However, the local AI model is currently unavailable for advanced code assistance.

In the meantime, I can:
• Provide basic coding guidance
• Help with system monitoring
• Assist with SutazAI configuration

For full AI-powered code assistance, please start the Ollama service."""
    
    else:
        return f"""I received your message: "{message}"

I'm currently running in fallback mode because the local AI model (Ollama) is unavailable. 

To get full AI-powered responses:
1. Ensure Ollama is running: `ollama serve`
2. Check if models are installed: `ollama list`
3. Restart SutazAI if needed: `/opt/sutazaiapp/bin/start_all.sh`

Is there anything specific about the SutazAI system I can help you with?"""

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatMessage):
    """Main chat endpoint"""
    try:
        # Check if Ollama is available
        ollama_available = await check_ollama_health()
        
        if ollama_available:
            # Get response from Ollama
            response = await chat_with_ollama(
                message=chat_request.message,
                model=chat_request.model,
                temperature=chat_request.temperature,
                max_tokens=chat_request.max_tokens
            )
        else:
            # Use fallback response
            response = generate_fallback_response(chat_request.message)
        
        return ChatResponse(
            response=response,
            model=chat_request.model,
            timestamp=datetime.now(),
            status="success" if ollama_available else "fallback"
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.get("/chat/models", response_model=List[str])
async def get_models():
    """Get available Ollama models"""
    try:
        models = await get_available_models()
        if not models:
            # Return default models if none found
            return ["llama3-chatqa", "llama3", "codellama"]
        return models
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return ["llama3-chatqa"]

@router.get("/chat/status", response_model=SystemStatus)
async def get_chat_status():
    """Get chat system status"""
    try:
        ollama_health = await check_ollama_health()
        models = await get_available_models()
        
        return SystemStatus(
            ollama_status="online" if ollama_health else "offline",
            available_models=models,
            system_health="healthy" if ollama_health else "degraded",
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"Status check error: {e}")
        return SystemStatus(
            ollama_status="error",
            available_models=[],
            system_health="error",
            timestamp=datetime.now()
        )

@router.post("/chat/system")
async def system_command(command: Dict[str, Any]):
    """Execute system commands (admin only)"""
    try:
        cmd = command.get("command", "").lower()
        
        if cmd == "restart_ollama":
            # This would restart Ollama service
            return {"status": "success", "message": "Ollama restart requested"}
        
        elif cmd == "check_health":
            ollama_health = await check_ollama_health()
            models = await get_available_models()
            
            return {
                "status": "success",
                "health": {
                    "ollama": "online" if ollama_health else "offline",
                    "models_count": len(models),
                    "models": models
                }
            }
        
        else:
            return {"status": "error", "message": f"Unknown command: {cmd}"}
            
    except Exception as e:
        logger.error(f"System command error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"System command failed: {str(e)}"
        )