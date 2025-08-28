"""Chat and conversation endpoints"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

class ChatMessage(BaseModel):
    message: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None

@router.post("/message")
async def send_message(chat: ChatMessage) -> dict:
    """Process chat message through AI system"""
    return {
        "response": f"Echo: {chat.message}",
        "session_id": chat.session_id or "default",
        "model": "tinyllama",
        "status": "success"
    }

@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    """Get chat session history"""
    return {
        "session_id": session_id,
        "messages": [],
        "created_at": "2025-08-27T00:00:00Z"
    }