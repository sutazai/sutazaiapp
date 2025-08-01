#!/usr/bin/env python3
"""
Enterprise Rate Limiting System
Advanced rate limiting with Redis backend and security monitoring
"""

import time
import logging
import redis
from typing import Dict, Optional, Tuple
from fastapi import Request, HTTPException
from functools import wraps
import asyncio
from datetime import datetime, timedelta
import json
import hashlib

logger = logging.getLogger("RateLimiter")

class AdvancedRateLimiter:
    """Enterprise-grade rate limiter with Redis backend and security features"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        try:
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            self.redis_client.ping()  # Test connection
            logger.info("Connected to Redis for rate limiting")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            # Fallback to in-memory rate limiting
            self.redis_client = None
            self.memory_cache = {}
            logger.warning("Using in-memory rate limiting (not recommended for production)")
    
    def _get_client_identifier(self, request: Request) -> str:
        """Get unique client identifier"""
        # Priority order for client identification:
        # 1. User ID from JWT token (if authenticated)
        # 2. API key (if provided)
        # 3. IP address
        
        client_id = None
        
        # Check for authenticated user
        if hasattr(request.state, 'user_id') and request.state.user_id:
            client_id = f"user:{request.state.user_id}"
        
        # Check for API key
        elif 'x-api-key' in request.headers:
            api_key_hash = hashlib.sha256(request.headers['x-api-key'].encode()).hexdigest()[:16]
            client_id = f"api:{api_key_hash}"
        
        # Fallback to IP address
        else:
            # Handle proxy headers for real IP
            client_ip = (
                request.headers.get('x-forwarded-for', '').split(',')[0].strip() or
                request.headers.get('x-real-ip', '') or
                request.client.host
            )
            client_id = f"ip:{client_ip}"
        
        return client_id
    
    def _parse_rate_limit(self, limit_str: str) -> Tuple[int, int]:
        """Parse rate limit string like '100/minute' into (count, window_seconds)"""
        try:
            count, period = limit_str.split('/')
            count = int(count)
            
            period_map = {
                'second': 1,
                'minute': 60,
                'hour': 3600,
                'day': 86400
            }
            
            window_seconds = period_map.get(period, 60)  # Default to minute
            return count, window_seconds
        except Exception as e:
            logger.error(f"Failed to parse rate limit '{limit_str}': {e}")
            return 100, 60  # Default fallback
    
    async def check_rate_limit(self, request: Request, limit: str, endpoint: str = None) -> bool:
        """Check if request is within rate limit"""
        client_id = self._get_client_identifier(request)
        count, window_seconds = self._parse_rate_limit(limit)
        
        # Create unique key for this client/endpoint combination
        key = f"rate_limit:{client_id}:{endpoint or 'default'}"
        
        current_time = int(time.time())
        window_start = current_time - window_seconds
        
        if self.redis_client:
            # Redis-based rate limiting with sliding window
            try:
                # Use Redis transaction for atomic operations
                pipe = self.redis_client.pipeline()
                
                # Remove old entries outside the window
                pipe.zremrangebyscore(key, 0, window_start)
                
                # Count current requests in window
                pipe.zcard(key)
                
                # Add current request
                pipe.zadd(key, {str(current_time): current_time})
                
                # Set expiration
                pipe.expire(key, window_seconds + 1)
                
                results = pipe.execute()
                current_count = results[1] + 1  # +1 for the request we just added
                
            except Exception as e:
                logger.error(f"Redis rate limiting error: {e}")
                # Fallback to allowing the request
                return True
                
        else:
            # In-memory fallback (not recommended for production)
            if key not in self.memory_cache:
                self.memory_cache[key] = []
            
            # Clean old entries
            self.memory_cache[key] = [
                ts for ts in self.memory_cache[key] 
                if ts > window_start
            ]
            
            # Add current request
            self.memory_cache[key].append(current_time)
            current_count = len(self.memory_cache[key])
        
        # Check if limit exceeded
        if current_count > count:
            logger.warning(f"Rate limit exceeded for {client_id} on {endpoint}: {current_count}/{count}")
            await self._log_rate_limit_violation(client_id, endpoint, current_count, count)
            return False
        
        return True
    
    async def _log_rate_limit_violation(self, client_id: str, endpoint: str, current: int, limit: int):
        """Log rate limit violations for security monitoring"""
        violation = {
            "timestamp": datetime.utcnow().isoformat(),
            "client_id": client_id,
            "endpoint": endpoint,
            "current_requests": current,
            "limit": limit,
            "violation_type": "rate_limit_exceeded"
        }
        
        # Log to security monitoring system
        logger.warning(f"SECURITY_EVENT: Rate limit violation: {json.dumps(violation)}")
        
        # Store in Redis for security analysis (if available)
        if self.redis_client:
            try:
                self.redis_client.lpush("security:rate_limit_violations", json.dumps(violation))
                self.redis_client.ltrim("security:rate_limit_violations", 0, 1000)  # Keep last 1000
            except Exception as e:
                logger.error(f"Failed to log rate limit violation: {e}")

# Global rate limiter instance
rate_limiter = AdvancedRateLimiter()

def rate_limit(limit: str, endpoint: str = None):
    """Decorator for rate limiting endpoints"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Find the request object in arguments
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
            
            if not request:
                # If no request found, allow the call (might be internal)
                logger.warning(f"No request object found for rate limiting in {func.__name__}")
                return await func(*args, **kwargs)
            
            # Check rate limit
            if not await rate_limiter.check_rate_limit(request, limit, endpoint or func.__name__):
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Rate limit exceeded",
                        "message": f"Too many requests. Limit: {limit}",
                        "retry_after": 60  # Suggest retry after 1 minute
                    }
                )
            
            return await func(*args, **kwargs)
        return wrapper
    return decorator

# Specific rate limiters for common use cases
def chat_rate_limit(func):
    """Rate limiter for chat endpoints (10/minute)"""
    return rate_limit("10/minute", "chat")(func)

def auth_rate_limit(func):
    """Rate limiter for authentication endpoints (5/minute)"""
    return rate_limit("5/minute", "auth")(func)

def upload_rate_limit(func):
    """Rate limiter for file upload endpoints (5/minute)"""
    return rate_limit("5/minute", "upload")(func)

def model_inference_rate_limit(func):
    """Rate limiter for model inference endpoints (20/minute)"""
    return rate_limit("20/minute", "model_inference")(func)

def admin_rate_limit(func):
    """Rate limiter for admin endpoints (50/minute)"""
    return rate_limit("50/minute", "admin")(func)

# FastAPI middleware for automatic rate limiting
class RateLimitMiddleware:
    """FastAPI middleware for automatic rate limiting"""
    
    def __init__(self, app, default_limit: str = "100/minute"):
        self.app = app
        self.default_limit = default_limit
    
    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        
        # Create request object
        from fastapi import Request
        request = Request(scope, receive)
        
        # Apply rate limiting
        endpoint = scope.get("path", "unknown")
        if not await rate_limiter.check_rate_limit(request, self.default_limit, endpoint):
            # Send rate limit exceeded response
            response = {
                "status_code": 429,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"retry-after", b"60")
                ],
                "body": json.dumps({
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Limit: {self.default_limit}",
                    "retry_after": 60
                }).encode()
            }
            
            await send({
                "type": "http.response.start",
                "status": 429,
                "headers": response["headers"]
            })
            await send({
                "type": "http.response.body",
                "body": response["body"]
            })
            return
        
        await self.app(scope, receive, send)

if __name__ == "__main__":
    # Test rate limiter
    import asyncio
    from unittest.mock import Mock
    
    async def test_rate_limiter():
        # Create mock request
        request = Mock()
        request.client.host = "127.0.0.1"
        request.headers = {}
        request.state = Mock()
        request.state.user_id = None
        
        print("Testing rate limiter...")
        
        # Test multiple requests
        for i in range(15):
            result = await rate_limiter.check_rate_limit(request, "10/minute", "test")
            print(f"Request {i+1}: {'ALLOWED' if result else 'BLOCKED'}")
            if i < 12:  # Add small delay for first few requests
                await asyncio.sleep(0.1)
    
    asyncio.run(test_rate_limiter())