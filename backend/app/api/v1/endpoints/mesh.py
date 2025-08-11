"""
Lightweight Mesh API (Redis Streams)

Endpoints:
- POST /api/v1/mesh/enqueue {topic, task} -> {id}
- GET  /api/v1/mesh/results?topic=...&count=N -> latest results
- GET  /api/v1/mesh/agents -> list of alive agents (if any have registered)
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Query
import httpx
import os
import hashlib

from app.mesh.redis_bus import enqueue_task, tail_results, list_agents, get_redis
from app.services.rate_limiter import default_ollama_bucket


router = APIRouter()


class EnqueueRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-z0-9:_-]+$")
    task: Dict[str, Any]


class EnqueueResponse(BaseModel):
    id: str


@router.post("/enqueue", response_model=EnqueueResponse)
def enqueue(req: EnqueueRequest) -> EnqueueResponse:
    try:
        msg_id = enqueue_task(req.topic, req.task)
        return EnqueueResponse(id=msg_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results")
def get_results(topic: str = Query(..., pattern=r"^[a-z0-9:_-]+$"), count: int = Query(10, ge=1, le=100)) -> List[Dict[str, Any]]:
    try:
        items = tail_results(topic, count)
        return [{"id": mid, "data": data} for (mid, data) in items]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/agents")
def get_agents() -> Dict[str, Any]:
    try:
        agents = list_agents()
        return {"count": len(agents), "agents": agents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
def health() -> Dict[str, Any]:
    """Mesh health check: verifies Redis connectivity and basic keys.
    Returns OK plus counts to aid monitoring.
    """
    try:
        r = get_redis()
        pong = r.ping()
        agents = list_agents()
        return {
            "status": "ok" if pong else "degraded",
            "redis": bool(pong),
            "agents_count": len(agents),
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"mesh unhealthy: {e}")


class GenerateRequest(BaseModel):
    model: str = Field(default_factory=lambda: os.environ.get("OLLAMA_DEFAULT_MODEL", "tinyllama"))
    prompt: str
    options: Optional[Dict[str, Any]] = None


@router.post("/ollama/generate")
async def ollama_generate(req: GenerateRequest) -> Dict[str, Any]:
    """Generate text using Ollama with connection pooling and caching"""
    from app.core.connection_pool import get_pool_manager
    from app.core.cache import get_cache_service, cache_model_data
    
    # Enforce simple prompt size limit to protect RAM/CPU
    if len(req.prompt.encode("utf-8")) > 32 * 1024:
        raise HTTPException(status_code=400, detail="prompt too large (32KB limit)")

    bucket = default_ollama_bucket()
    allowed, wait_ms = bucket.try_acquire(1)
    if not allowed:
        # Communicate backpressure to caller
        raise HTTPException(status_code=429, detail={"retry_after_ms": wait_ms})
    
    # Check cache first for repeated prompts
    cache_service = await get_cache_service()
    cache_key = f"ollama:generate:{req.model}:{hashlib.md5(req.prompt.encode()).hexdigest()[:16]}"
    
    cached_response = await cache_service.get(cache_key, force_redis=True)
    if cached_response:
        return cached_response
    
    # Use connection pool for Ollama requests
    pool_manager = await get_pool_manager()
    
    payload = {"model": req.model, "prompt": req.prompt, "stream": False}
    if req.options:
        payload["options"] = req.options
    
    try:
        async with pool_manager.get_http_client("ollama") as client:
            resp = await client.post("/api/generate", json=payload)
            resp.raise_for_status()
            result = resp.json()
            
            # Cache successful responses
            await cache_service.set(cache_key, result, ttl=1800, redis_priority=True)
            
            return result
            
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ollama connection error: {e}")
