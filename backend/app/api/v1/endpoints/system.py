"""System endpoint stub"""
from fastapi import APIRouter
router = APIRouter()

@router.get("/")
async def system_info():
    return {"status": "ok", "version": "1.0.0"}
EOF < /dev/null
