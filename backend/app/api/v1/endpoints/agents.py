"""Agents endpoint stub"""
from fastapi import APIRouter
router = APIRouter()

@router.get("/")
async def list_agents():
    return {"agents": [], "active_count": 0, "total_count": 0}
EOF < /dev/null
