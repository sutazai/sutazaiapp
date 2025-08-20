"""
System endpoint - MINIMAL IMPLEMENTATION

Provides basic system status information.
TODO: Enhance with real system metrics:
- CPU/Memory usage statistics
- Active connections count
- Database connection pool status
- Cache hit rates
- Error rates and health checks
"""
from fastapi import APIRouter
import platform
import os

router = APIRouter()

@router.get("/")
async def system_info():
    """
    Get system information - MINIMAL IMPLEMENTATION
    
    Currently returns basic status.
    TODO: Add comprehensive system metrics and health monitoring
    """
    return {
        "status": "ok",
        "version": "1.0.0",
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "message": "Full system monitoring not yet implemented"
    }
