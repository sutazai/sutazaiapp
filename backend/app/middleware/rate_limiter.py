"""
Redis-based rate limiting middleware
Implements per-user rate limiting with sliding window algorithm
"""

import time
import logging
from typing import Optional, Callable, Awaitable
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp
from app.services.connections import service_connections

logger = logging.getLogger(__name__)


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using Redis backend
    
    Features:
    - Per-user rate limiting
    - Sliding window algorithm
    - Configurable limits per endpoint
    - Returns 429 Too Many Requests when limit exceeded
    """
    
    def __init__(
        self,
        app: ASGIApp,
        requests_per_minute: int = 100,
        burst_size: int = 20,
        enabled: bool = True
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.enabled = enabled
        self.window_size = 60  # 1 minute window in seconds
        
    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        """Process request with rate limiting"""
        
        if not self.enabled:
            return await call_next(request)
        
        # Skip rate limiting for health check endpoints
        if request.url.path in ["/health", "/api/v1/health", "/api/v1/health/", "/api/v1/health/services"]:
            return await call_next(request)
        
        # Get user identifier from JWT token or IP address as fallback
        user_id = await self._get_user_identifier(request)
        
        # Check rate limit
        is_allowed, remaining, reset_time = await self._check_rate_limit(user_id, request.url.path)
        
        # Add rate limit headers to response
        if is_allowed:
            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(reset_time)
            return response
        else:
            # Rate limit exceeded
            logger.warning(f"Rate limit exceeded for user {user_id} on {request.url.path}")
            raise HTTPException(
                status_code=429,
                detail={
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {self.requests_per_minute} requests per minute",
                    "retry_after": reset_time - int(time.time()),
                    "limit": self.requests_per_minute,
                    "remaining": 0,
                    "reset": reset_time
                }
            )
    
    async def _get_user_identifier(self, request: Request) -> str:
        """Get user identifier from JWT token or IP address"""
        try:
            # Try to get user from Authorization header
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header.split(" ")[1]
                # Simple token parsing - extract user ID from token
                # In production, this should validate the token properly
                from jose import jwt
                try:
                    payload = jwt.decode(
                        token,
                        options={"verify_signature": False}  # Just for user ID extraction
                    )
                    user_id = payload.get("sub")
                    if user_id:
                        return f"user:{user_id}"
                except Exception:
                    pass
        except Exception:
            pass
        
        # Fallback to IP address
        client_ip = request.client.host if request.client else "unknown"
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        return f"ip:{client_ip}"
    
    async def _check_rate_limit(
        self, user_id: str, endpoint: str
    ) -> tuple[bool, int, int]:
        """
        Check rate limit using Redis sliding window
        
        Returns:
            tuple: (is_allowed, remaining_requests, reset_timestamp)
        """
        redis_client = service_connections.redis_client
        
        if not redis_client:
            # If Redis is unavailable, allow request but log warning
            logger.warning("Redis unavailable - rate limiting disabled")
            return True, self.requests_per_minute, int(time.time() + self.window_size)
        
        try:
            current_time = time.time()
            window_start = current_time - self.window_size
            
            # Create unique key for this user and endpoint
            key = f"ratelimit:{user_id}:{endpoint}"
            
            # Use Redis sorted set for sliding window
            # Remove old entries outside the window
            await redis_client.zremrangebyscore(key, 0, window_start)
            
            # Count requests in current window
            request_count = await redis_client.zcard(key)
            
            # Check if limit exceeded
            if request_count >= self.requests_per_minute:
                # Get oldest request timestamp to calculate reset time
                oldest = await redis_client.zrange(key, 0, 0, withscores=True)
                if oldest:
                    reset_time = int(oldest[0][1] + self.window_size)
                else:
                    reset_time = int(current_time + self.window_size)
                
                return False, 0, reset_time
            
            # Add current request to sorted set
            await redis_client.zadd(key, {str(current_time): current_time})
            
            # Set expiration on key (cleanup)
            await redis_client.expire(key, self.window_size * 2)
            
            # Calculate remaining requests and reset time
            remaining = self.requests_per_minute - (request_count + 1)
            reset_time = int(current_time + self.window_size)
            
            return True, remaining, reset_time
            
        except Exception as e:
            # On error, allow request but log the issue
            logger.error(f"Rate limit check error: {e}", exc_info=True)
            return True, self.requests_per_minute, int(time.time() + self.window_size)


class BurstRateLimiter:
    """
    Helper class for burst rate limiting
    Allows short bursts above the average rate
    """
    
    def __init__(
        self,
        redis_client,
        max_burst: int = 20,
        refill_rate: float = 1.0,
    ):
        self.redis_client = redis_client
        self.max_burst = max_burst
        self.refill_rate = refill_rate  # tokens per second
    
    async def allow_request(self, user_id: str) -> tuple[bool, int]:
        """
        Token bucket algorithm for burst limiting
        
        Returns:
            tuple: (is_allowed, tokens_remaining)
        """
        key = f"burst:{user_id}"
        current_time = time.time()
        
        try:
            # Get current bucket state
            bucket_data = await self.redis_client.get(key)
            
            if bucket_data:
                tokens, last_refill = eval(bucket_data)
            else:
                tokens = self.max_burst
                last_refill = current_time
            
            # Calculate tokens to add based on time elapsed
            time_elapsed = current_time - last_refill
            tokens_to_add = time_elapsed * self.refill_rate
            tokens = min(self.max_burst, tokens + tokens_to_add)
            
            if tokens >= 1.0:
                # Allow request and consume token
                tokens -= 1.0
                await self.redis_client.setex(
                    key,
                    300,  # 5 minute expiration
                    str((tokens, current_time))
                )
                return True, int(tokens)
            else:
                # Not enough tokens
                return False, 0
                
        except Exception as e:
            logger.error(f"Burst rate limit error: {e}")
            return True, self.max_burst  # Fail open
