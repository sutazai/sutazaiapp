"""
Lightweight Mesh: Redis Streams Bus

Provides minimal helpers for enqueueing tasks, consuming results, and a simple
agent registry using Redis. Designed to be hardware‑friendly and avoid heavy
infrastructure (no Kong/Consul/RabbitMQ).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import redis


def _redis_url() -> str:
    return os.environ.get("REDIS_URL", "redis://redis:6379/0")


def get_redis() -> "redis.Redis":
    return redis.from_url(_redis_url(), decode_responses=True)


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
    r = get_redis()
    keys = r.keys("mesh:agent:*")
    agents: List[Dict[str, Any]] = []
    for k in keys:
        try:
            val = r.get(k)
            if val:
                agents.append(json.loads(val))
        except Exception:
            continue
    return agents


def enqueue_task(topic: str, payload: Dict[str, Any], maxlen: int = 10000) -> str:
    r = get_redis()
    # Store as a single JSON field for flexibility
    msg_id = r.xadd(task_stream(topic), {"json": json.dumps(payload)}, maxlen=maxlen, approximate=True)
    return msg_id


def tail_results(topic: str, count: int = 10) -> List[Tuple[str, Dict[str, Any]]]:
    r = get_redis()
    # XREVRANGE returns latest first
    raw = r.xrevrange(result_stream(topic), count=count)
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

