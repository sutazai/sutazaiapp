"""
Chat endpoint for SutazAI
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Dict, Optional
from app.services.model_manager import ModelManager
from app.core.dependencies import get_model_manager
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048

class ChatResponse(BaseModel):
    response: str
    model: str
    tokens_used: Optional[int] = None

@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Chat with an AI model
    """
    try:
        logger.info(f"Chat request received: {request.message[:50]}...")
        
        # Use chat format for better responses
        messages = [
            {"role": "system", "content": "You are SutazAI, an advanced AGI assistant. Be helpful, accurate, and concise."},
            {"role": "user", "content": request.message}
        ]
        
        response = await model_manager.chat(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
            options={"num_predict": request.max_tokens}
        )
        
        if not response:
            # Fallback to simple generation if chat fails
            response = await model_manager.generate(
                prompt=request.message,
                model=request.model,
                temperature=request.temperature,
                options={"num_predict": request.max_tokens}
            )
        
        return ChatResponse(
            response=response or "I apologize, but I couldn't generate a response. Please try again.",
            model=request.model or model_manager.default_model
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))