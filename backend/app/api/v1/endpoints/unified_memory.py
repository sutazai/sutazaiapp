"""
Unified Memory Service API Endpoints
TDD Implementation - Minimal code to pass tests (GREEN phase)
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
import httpx
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Unified Memory Service URL
UNIFIED_MEMORY_URL = "http://localhost:3009"

@router.get("/health")
async def health_check():
    """Health check for unified memory service"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{UNIFIED_MEMORY_URL}/health")
            if response.status_code == 200:
                data = response.json()
                # Add backend integration info
                data["backend_integration"] = True
                data["endpoint"] = "/api/v1/mcp/unified-memory"
                return data
            else:
                logger.warning(f"Unified memory service returned {response.status_code}")
                raise HTTPException(status_code=503, detail=f"Service returned {response.status_code}")
    except httpx.ConnectError as e:
        logger.error(f"Cannot connect to unified memory service: {e}")
        raise HTTPException(status_code=503, detail="Unified memory service unavailable")
    except httpx.TimeoutException as e:
        logger.error(f"Timeout connecting to unified memory service: {e}")
        raise HTTPException(status_code=504, detail="Service timeout")
    except Exception as e:
        logger.error(f"Unexpected health check error: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")

@router.post("/store")
async def store_memory(request: dict):
    """Store memory via unified memory service"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{UNIFIED_MEMORY_URL}/memory/store",
                json=request
            )
            if response.status_code == 200:
                return response.json()
            raise HTTPException(status_code=response.status_code, detail="Store failed")
    except Exception as e:
        logger.error(f"Store memory failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/retrieve/{key}")
async def retrieve_memory(key: str, namespace: str = Query(default="default")):
    """Retrieve memory via unified memory service"""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{UNIFIED_MEMORY_URL}/memory/retrieve/{key}",
                params={"namespace": namespace}
            )
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                raise HTTPException(status_code=404, detail=f"Memory '{key}' not found in namespace '{namespace}'")
            else:
                logger.warning(f"Retrieve failed with status {response.status_code}: {response.text}")
                raise HTTPException(status_code=response.status_code, detail=f"Retrieve failed: {response.text}")
    except HTTPException:
        raise
    except httpx.ConnectError as e:
        logger.error(f"Cannot connect to unified memory service: {e}")
        raise HTTPException(status_code=503, detail="Memory service unavailable")
    except httpx.TimeoutException as e:
        logger.error(f"Timeout retrieving memory: {e}")
        raise HTTPException(status_code=504, detail="Service timeout")
    except Exception as e:
        logger.error(f"Unexpected retrieve error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/search")
async def search_memory(
    query: str = Query(..., description="Search query"),
    namespace: str = Query(default="default", description="Namespace to search"),
    limit: int = Query(default=10, description="Maximum results")
):
    """Search memory via unified memory service"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{UNIFIED_MEMORY_URL}/memory/search",
                params={"query": query, "namespace": namespace, "limit": limit}
            )
            if response.status_code == 200:
                return response.json()
            raise HTTPException(status_code=response.status_code, detail="Search failed")
    except Exception as e:
        logger.error(f"Search memory failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/stats")
async def memory_stats():
    """Get memory statistics via unified memory service"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{UNIFIED_MEMORY_URL}/memory/stats")
            if response.status_code == 200:
                return response.json()
            raise HTTPException(status_code=response.status_code, detail="Stats failed")
    except Exception as e:
        logger.error(f"Memory stats failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.delete("/delete/{key}")
async def delete_memory(key: str, namespace: str = Query(default="default")):
    """Delete memory via unified memory service"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"{UNIFIED_MEMORY_URL}/memory/delete/{key}",
                params={"namespace": namespace}
            )
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                raise HTTPException(status_code=404, detail="Memory not found")
            raise HTTPException(status_code=response.status_code, detail="Delete failed")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete memory failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# Legacy endpoints for backward compatibility
@router.get("/extended-memory/load_contexts", deprecated=True)
async def legacy_extended_memory_load_contexts():
    """Legacy endpoint - deprecated, use unified memory instead"""
    return {
        "status": "deprecated", 
        "message": "Use /api/v1/mcp/unified-memory/search instead",
        "redirect": "/api/v1/mcp/unified-memory/search"
    }

@router.get("/memory-bank-mcp/contexts", deprecated=True)
async def legacy_memory_bank_contexts():
    """Legacy endpoint - deprecated, use unified memory instead"""
    return {
        "status": "deprecated",
        "message": "Use /api/v1/mcp/unified-memory/stats instead", 
        "redirect": "/api/v1/mcp/unified-memory/stats"
    }

# Migration endpoints
@router.post("/migration/extended-memory-to-unified")
async def migrate_extended_memory():
    """Migrate extended-memory data to unified memory service"""
    # Minimal implementation for GREEN phase
    return {"success": True, "migrated_count": 0, "message": "Migration endpoint ready"}

@router.post("/migration/memory-bank-to-unified")
async def migrate_memory_bank():
    """Migrate memory-bank-mcp data to unified memory service"""
    # Minimal implementation for GREEN phase
    return {"success": True, "migrated_count": 0, "message": "Migration endpoint ready"}