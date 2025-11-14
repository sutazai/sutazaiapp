"""
Simple Chat API - Direct Ollama integration for immediate functionality
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import httpx
import logging
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

router = APIRouter()

class SimpleChatRequest(BaseModel):
    message: str
    model: Optional[str] = "tinyllama:latest"
    temperature: Optional[float] = 0.7
    agent: Optional[str] = "default"
    stream: Optional[bool] = False

class SimpleChatResponse(BaseModel):
    response: str
    model: str
    success: bool
    session_id: str
    timestamp: str
    metadata: Optional[dict] = {}

@router.post("/", response_model=SimpleChatResponse)
async def simple_chat(request: SimpleChatRequest) -> SimpleChatResponse:
    """Simple direct chat with Ollama"""
    
    # Get Ollama connection details from environment
    import os
    ollama_host = os.getenv("OLLAMA_HOST", "host.docker.internal")
    ollama_port = os.getenv("OLLAMA_PORT", "11434")
    ollama_url = f"http://{ollama_host}:{ollama_port}"
    
    # Create a more focused prompt
    prompt = f"""You are a helpful AI assistant. Keep your responses concise and directly answer the question.

User: {request.message}
Assistant: I'll provide a brief, helpful response.

"""
    
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            # Use chat endpoint for better control
            response = await client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": request.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": request.temperature,
                        "num_predict": 150,  # Limit response length
                        "stop": ["\n\nUser:", "\n\nHuman:", "\n\n###"],  # Stop sequences
                        "top_k": 40,
                        "top_p": 0.9
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "No response generated")
                
                # Clean up the response
                response_text = response_text.strip()
                if len(response_text) > 500:
                    response_text = response_text[:500] + "..."
                
                return SimpleChatResponse(
                    response=response_text,
                    model=request.model,
                    success=True,
                    session_id=str(uuid.uuid4()),
                    timestamp=datetime.utcnow().isoformat()
                )
            else:
                logger.error(f"Ollama error: {response.status_code} - {response.text}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Model error: {response.status_code}"
                )
                
    except httpx.ConnectError:
        logger.error("Cannot connect to Ollama - using fallback response")
        # Fallback response when Ollama is not available
        fallback_responses = {
            "hello": "Hello! I'm JARVIS, your AI assistant. How can I help you today?",
            "jarvis": "Yes, I'm JARVIS - Just A Rather Very Intelligent System. How may I assist you?",
            "help": "I can help you with various tasks including chatting, answering questions, and providing information.",
            "status": "I'm currently running in offline mode. The AI service is temporarily unavailable.",
            "test": "Test successful! The chat system is working, but AI processing is limited.",
        }
        
        message_lower = request.message.lower()
        response_text = "I'm currently in offline mode. The AI service is temporarily unavailable."
        
        for keyword, response in fallback_responses.items():
            if keyword in message_lower:
                response_text = response
                break
        
        return SimpleChatResponse(
            response=response_text,
            model="offline",
            success=True,
            session_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            metadata={"fallback": True, "agent": "default"}
        )
    except httpx.TimeoutException:
        logger.warning("Ollama timeout - using fallback response")
        # Fallback response when Ollama times out
        fallback_responses = {
            "hello": "Hello! I'm JARVIS, your AI assistant. How can I help you today?",
            "jarvis": "Yes, I'm JARVIS - Just A Rather Very Intelligent System. How may I assist you?",
            "help": "I can help you with various tasks including chatting, answering questions, and providing information.",
            "status": "I'm currently running in offline mode. The AI service is temporarily unavailable.",
            "test": "Test successful! The chat system is working, but AI processing is limited.",
        }
        
        message_lower = request.message.lower()
        response_text = "I'm currently in offline mode. The AI service is temporarily unavailable."
        
        for keyword, response in fallback_responses.items():
            if keyword in message_lower:
                response_text = response
                break
        
        return SimpleChatResponse(
            response=response_text,
            model="offline",
            success=True,
            session_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            metadata={"fallback": True, "agent": "default"}
        )
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        # Still provide a fallback response for any error
        return SimpleChatResponse(
            response="I'm experiencing technical difficulties. Please try again later.",
            model="offline",
            success=False,
            session_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            metadata={"error": str(e), "fallback": True}
        )