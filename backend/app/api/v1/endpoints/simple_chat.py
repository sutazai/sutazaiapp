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

class SimpleChatResponse(BaseModel):
    response: str
    model: str
    success: bool
    session_id: str
    timestamp: str

@router.post("/simple", response_model=SimpleChatResponse)
async def simple_chat(request: SimpleChatRequest) -> SimpleChatResponse:
    """Simple direct chat with Ollama"""
    
    ollama_url = "http://sutazai-ollama:11434"
    
    # Create a more focused prompt
    prompt = f"""You are a helpful AI assistant. Keep your responses concise and directly answer the question.

User: {request.message}
Assistant: I'll provide a brief, helpful response.

"""
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
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
        logger.error("Cannot connect to Ollama")
        raise HTTPException(
            status_code=503,
            detail="AI service unavailable"
        )
    except Exception as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Chat failed: {str(e)}"
        )