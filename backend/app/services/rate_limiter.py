"""
Redis-backed Token Bucket Rate Limiter

Used to protect Ollama on limited hardware. Implements a simple token bucket
with atomic Lua script. Default keys are namespaced under `ratelimit:`.
"""
from __future__ import annotations

import os
import time
from typing import Tuple

from app.mesh.redis_bus import get_redis


_LUA = """
-- KEYS[1] = tokens key
-- KEYS[2] = ts key
-- ARGV[1] = capacity
-- ARGV[2] = refill_per_sec
-- ARGV[3] = now_ms
-- ARGV[4] = requested tokens

local capacity = tonumber(ARGV[1])
local rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local req = tonumber(ARGV[4])

local tokens = tonumber(redis.call('GET', KEYS[1]))
if tokens == nil then tokens = capacity end
local last_ts = tonumber(redis.call('GET', KEYS[2]))
if last_ts == nil then last_ts = now end

local elapsed = (now - last_ts) / 1000.0
local refill = elapsed * rate
tokens = math.min(capacity, tokens + refill)

local allowed = 0
local wait_ms = 0
if tokens >= req then
  tokens = tokens - req
  allowed = 1
else
  local deficit = req - tokens
  wait_ms = math.ceil((deficit / rate) * 1000)
end

redis.call('SET', KEYS[1], tokens)
redis.call('SET', KEYS[2], now)

return {allowed, math.floor(tokens), wait_ms}
"""


class TokenBucket:
    def __init__(self, name: str, capacity: float, refill_per_sec: float):
        self.name = name
        self.capacity = float(capacity)
        self.refill = float(refill_per_sec)
        self._sha = None

    def _keys(self):
        base = f"ratelimit:{self.name}"
        return [f"{base}:tokens", f"{base}:ts"]

    def try_acquire(self, tokens: float = 1.0) -> Tuple[bool, int]:
        """Attempt to acquire tokens. Returns (allowed, retry_after_ms)."""
        r = get_redis()
        if not self._sha:
            self._sha = r.script_load(_LUA)
        now_ms = int(time.time() * 1000)
        allowed, remaining, wait_ms = r.evalsha(
            self._sha, 2, *self._keys(), str(self.capacity), str(self.refill), str(now_ms), str(tokens)
        )
        return bool(allowed), int(wait_ms)


def default_ollama_bucket() -> TokenBucket:
    capacity = float(os.environ.get("OLLAMA_BUCKET_CAPACITY", "10"))
    refill = float(os.environ.get("OLLAMA_BUCKET_REFILL_PER_SEC", "2"))
    return TokenBucket("ollama", capacity=capacity, refill_per_sec=refill)

