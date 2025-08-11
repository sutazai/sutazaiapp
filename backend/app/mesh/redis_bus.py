"""
Lightweight Mesh: Redis Streams Bus

Provides minimal helpers for enqueueing tasks, consuming results, and a simple
agent registry using Redis. Designed to be hardware‑friendly and avoid heavy
infrastructure (no Kong/Consul/RabbitMQ).
"""
from __future__ import annotations

import json
import os
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import redis
import redis.asyncio as redis_async


# Global connection pool instance - shared across all requests
_redis_pool = None
_redis_async_pool = None


def _redis_url() -> str:
    return os.environ.get("REDIS_URL", "redis://redis:6379/0")


def get_redis() -> "redis.Redis":
    """Get Redis client with connection pooling for better performance"""
    global _redis_pool
    if _redis_pool is None:
        # Create connection pool with optimized settings
        _redis_pool = redis.ConnectionPool.from_url(
            _redis_url(),
            decode_responses=True,
            max_connections=50,
            socket_connect_timeout=5,
            socket_timeout=5,
            socket_keepalive=True,
            socket_keepalive_options={
                1: 1,  # TCP_KEEPIDLE
                2: 3,  # TCP_KEEPINTVL  
                3: 5,  # TCP_KEEPCNT
            },
            health_check_interval=30
        )
    return redis.Redis(connection_pool=_redis_pool)


async def get_redis_async() -> "redis_async.Redis":
    """Get async Redis client with connection pooling"""
    global _redis_async_pool
    if _redis_async_pool is None:
        _redis_async_pool = redis_async.ConnectionPool.from_url(
            _redis_url(),
            decode_responses=True,
            max_connections=50,
            socket_connect_timeout=5,
            socket_timeout=5,
            socket_keepalive=True,
            socket_keepalive_options={
                1: 1,  # TCP_KEEPIDLE
                2: 3,  # TCP_KEEPINTVL
                3: 5,  # TCP_KEEPCNT
            },
            health_check_interval=30
        )
    return redis_async.Redis(connection_pool=_redis_async_pool)


# Stream keys
def task_stream(topic: str) -> str:
    return f"stream:tasks:{topic}"


def result_stream(topic: str) -> str:
    return f"stream:results:{topic}"


def dead_stream(topic: str) -> str:
    return f"stream:dead:{topic}"


# Agent registry (optional usage)
def agent_key(agent_id: str) -> str:
    return f"mesh:agent:{agent_id}"


def register_agent(agent_id: str, agent_type: str, ttl_seconds: int = 60, meta: Optional[Dict[str, Any]] = None) -> None:
    r = get_redis()
    data = {"agent_id": agent_id, "agent_type": agent_type, "meta": meta or {}}
    r.set(agent_key(agent_id), json.dumps(data), ex=ttl_seconds)


def heartbeat_agent(agent_id: str, ttl_seconds: int = 60) -> None:
    r = get_redis()
    k = agent_key(agent_id)
    if r.exists(k):
        r.expire(k, ttl_seconds)


def list_agents() -> List[Dict[str, Any]]:
    """List agents with optimized batch fetching"""
    r = get_redis()
    
    # Use SCAN instead of KEYS for production safety
    cursor = 0
    keys = []
    while True:
        cursor, batch = r.scan(cursor, match="mesh:agent:*", count=100)
        keys.extend(batch)
        if cursor == 0:
            break
    
    if not keys:
        return []
    
    # Batch fetch all values using pipeline
    with r.pipeline() as pipe:
        for k in keys:
            pipe.get(k)
        values = pipe.execute()
    
    agents: List[Dict[str, Any]] = []
    for val in values:
        if val:
            try:
                agents.append(json.loads(val))
            except Exception:
                continue
    return agents


def enqueue_task(topic: str, payload: Dict[str, Any], maxlen: int = 10000) -> str:
    """Enqueue task with connection pooling and caching"""
    r = get_redis()
    stream_key = task_stream(topic)
    
    # Cache the stream key existence check to reduce Redis calls
    cache_key = f"stream_exists:{stream_key}"
    if not hasattr(enqueue_task, '_stream_cache'):
        enqueue_task._stream_cache = {}
    
    if cache_key not in enqueue_task._stream_cache:
        enqueue_task._stream_cache[cache_key] = True
        # Ensure stream exists with consumer group
        try:
            r.xgroup_create(stream_key, "default", id="$", mkstream=True)
        except redis.exceptions.ResponseError:
            pass  # Group already exists
    
    # Store as a single JSON field for flexibility
    msg_id = r.xadd(stream_key, {"json": json.dumps(payload)}, maxlen=maxlen, approximate=True)
    return msg_id


def tail_results(topic: str, count: int = 10) -> List[Tuple[str, Dict[str, Any]]]:
    """Tail results with optimized batch reading"""
    r = get_redis()
    stream_key = result_stream(topic)
    
    # Use pipeline for batch operations
    with r.pipeline() as pipe:
        pipe.xrevrange(stream_key, count=count)
        raw = pipe.execute()[0]
    
    out: List[Tuple[str, Dict[str, Any]]] = []
    for msg_id, fields in raw:
        try:
            data = json.loads(fields.get("json", "{}"))
        except Exception:
            data = fields
        out.append((msg_id, data))
    return out


def create_consumer_group(topic: str, group: str) -> None:
    r = get_redis()
    try:
        r.xgroup_create(task_stream(topic), group, id="$", mkstream=True)
    except redis.exceptions.ResponseError as e:
        # Group may already exist
        if "BUSYGROUP" in str(e):
            return
        raise


def read_group(topic: str, group: str, consumer: str, count: int = 1, block_ms: int = 1000) -> List[Tuple[str, Dict[str, Any]]]:
    r = get_redis()
    resp = r.xreadgroup(group, consumer, {task_stream(topic): ">"}, count=count, block=block_ms)
    messages: List[Tuple[str, Dict[str, Any]]] = []
    for _stream, entries in resp:
        for msg_id, fields in entries:
            try:
                data = json.loads(fields.get("json", "{}"))
            except Exception:
                data = fields
            messages.append((msg_id, data))
    return messages


def ack(topic: str, group: str, msg_id: str) -> int:
    r = get_redis()
    return r.xack(task_stream(topic), group, msg_id)


def move_to_dead(topic: str, msg_id: str, payload: Dict[str, Any]) -> str:
    r = get_redis()
    return r.xadd(dead_stream(topic), {"json": json.dumps({"id": msg_id, "payload": payload})}, maxlen=10000, approximate=True)

