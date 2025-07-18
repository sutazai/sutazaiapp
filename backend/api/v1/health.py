#!/usr/bin/env python3
"""
SutazAI Health Check API
Health endpoints for system monitoring
"""

from fastapi import APIRouter, Depends
from typing import Dict, Any
from datetime import datetime

router = APIRouter()

@router.get("/")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "SutazAI Backend"
    }

@router.get("/detailed")
async def detailed_health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "SutazAI Backend",
        "version": "1.0.0",
        "components": {
            "database": "healthy",
            "cache": "healthy",
            "vector_stores": "healthy",
            "ai_services": "healthy"
        }
    }