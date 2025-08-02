"""
Chat endpoint for SutazAI with comprehensive XSS protection
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, validator
from typing import List, Dict, Optional
from app.services.model_manager import ModelManager
from app.core.dependencies import get_model_manager
from app.core.security import xss_protection
import logging
import html

logger = logging.getLogger(__name__)

router = APIRouter()

class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    
    @validator('message')
    def validate_message(cls, v):
        """Validate and sanitize chat message for XSS protection"""
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
            
        # Use XSS protection to validate input
        try:
            sanitized = xss_protection.validator.validate_input(v, "chat_message")
            return sanitized
        except ValueError as e:
            raise ValueError(f"Invalid message content: {str(e)}")
    
    @validator('model')
    def validate_model(cls, v):
        """Validate model name"""
        if v is not None:
            # Basic sanitization for model name
            v = v.strip()
            # Only allow alphanumeric characters, dots, hyphens, underscores, and colons
            import re
            if not re.match(r'^[a-zA-Z0-9._:-]+$', v):
                raise ValueError("Invalid model name format")
        return v

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
    Chat with an AI model - XSS protected endpoint
    """
    try:
        # Log sanitized request (first 50 chars only for security)
        sanitized_preview = html.escape(request.message[:50])
        logger.info(f"Chat request received: {sanitized_preview}...")
        
        # Additional validation to ensure message is clean
        if not request.message or len(request.message.strip()) == 0:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Use chat format for better responses
        # Note: The system message is predefined and safe
        messages = [
            {"role": "system", "content": "You are SutazAI, an advanced automation assistant. Be helpful, accurate, and concise."},
            {"role": "user", "content": request.message}  # Already sanitized by validator
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
        
        # Sanitize the AI response before returning
        safe_response = response or "I apologize, but I couldn't generate a response. Please try again."
        try:
            # Additional sanitization of the AI response
            safe_response = xss_protection.validator.validate_input(safe_response, "text")
        except ValueError:
            # If sanitization fails, use a safe fallback
            safe_response = "Response content was filtered for security reasons."
            logger.warning("AI response was blocked by XSS protection")
        
        return ChatResponse(
            response=safe_response,
            model=request.model or model_manager.default_model
        )
        
    except ValueError as e:
        # Handle validation errors specifically
        logger.warning(f"Chat validation error: {e}")
        raise HTTPException(status_code=400, detail=f"Input validation failed: {str(e)}")
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="An error occurred while processing your request")