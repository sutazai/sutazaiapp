"""Agent management endpoints"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any, Annotated
from app.api.dependencies.auth import get_current_active_user, get_current_user_optional
from app.models.user import User

router = APIRouter()

@router.get("/")
async def list_agents() -> List[Dict[str, Any]]:
    """List all available AI agents"""
    return [
        {"id": "1", "name": "Letta", "status": "pending", "type": "memory"},
        {"id": "2", "name": "AutoGPT", "status": "pending", "type": "autonomous"},
        {"id": "3", "name": "CrewAI", "status": "pending", "type": "collaborative"}
    ]

@router.post("/create")
async def create_agent(
    agent_type: str, 
    name: str,
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> Dict[str, Any]:
    """Create a new agent instance (requires authentication)"""
    return {
        "id": "new_agent_id",
        "name": name,
        "type": agent_type,
        "status": "initializing",
        "created_by": current_user.username
    }