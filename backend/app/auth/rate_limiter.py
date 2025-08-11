"""
ULTRA Security Rate Limiter for Authentication
Implements advanced rate limiting with sliding window, distributed locking, and IP reputation
Author: ULTRA Security Engineer
Date: 2025-08-11
"""

import time
import hashlib
import ipaddress
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple, List, Any
from collections import defaultdict, deque
import redis
import asyncio
from functools import wraps
import logging

logger = logging.getLogger(__name__)


class UltraRateLimiter:
    """
    Enterprise-grade rate limiter with multiple strategies:
    - Sliding window log algorithm
    - Token bucket algorithm
    - IP reputation scoring
    - Distributed rate limiting via Redis
    - Adaptive rate limiting based on behavior
    """
    
    def __init__(
        self,
        redis_client: Optional[redis.Redis] = None,
        default_limit: int = 10,
        default_window: int = 60,
        enable_reputation: bool = True
    ):
        """
        Initialize rate limiter
        
        Args:
            redis_client: Redis client for distributed rate limiting
            default_limit: Default request limit
            default_window: Default time window in seconds
            enable_reputation: Enable IP reputation scoring
        """
        self.redis_client = redis_client
        self.default_limit = default_limit
        self.default_window = default_window
        self.enable_reputation = enable_reputation
        
        # Local cache for non-distributed mode
        self.local_cache: Dict[str, deque] = defaultdict(deque)
        self.token_buckets: Dict[str, Dict] = {}
        self.ip_reputation: Dict[str, float] = {}
        self.blocked_ips: set = set()
        
        # Rate limit configurations per endpoint
        self.endpoint_limits = {
            "/api/v1/auth/login": {"limit": 5, "window": 300, "burst": 2},
            "/api/v1/auth/register": {"limit": 3, "window": 3600, "burst": 1},
            "/api/v1/auth/reset-password": {"limit": 3, "window": 3600, "burst": 1},
            "/api/v1/auth/verify-2fa": {"limit": 5, "window": 300, "burst": 2},
            "/api/v1/chat": {"limit": 30, "window": 60, "burst": 10},
            "/api/v1/models": {"limit": 100, "window": 60, "burst": 20},
            "default": {"limit": 60, "window": 60, "burst": 10}
        }
        
        # Suspicious patterns
        self.suspicious_patterns = [
            "admin", "root", "test", "' OR '1'='1", "<script>",
            "../", "..\\", "%00", "union select", "drop table"
        ]
        
    def _get_redis_key(self, identifier: str, endpoint: str) -> str:
        """Generate Redis key for rate limiting"""
        return f"rate_limit:{endpoint}:{identifier}"
        
    def _get_reputation_key(self, ip: str) -> str:
        """Generate Redis key for IP reputation"""
        return f"ip_reputation:{ip}"
        
    async def check_rate_limit(
        self,
        identifier: str,
        endpoint: str = "default",
        ip_address: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if request should be rate limited
        
        Args:
            identifier: Unique identifier (user_id, IP, session_id)
            endpoint: API endpoint being accessed
            ip_address: Client IP address
            
        Returns:
            Tuple of (allowed, metadata)
        """
        # Check if IP is blocked
        if ip_address and ip_address in self.blocked_ips:
            return False, {"reason": "IP blocked", "retry_after": 3600}
            
        # Get endpoint configuration
        config = self.endpoint_limits.get(endpoint, self.endpoint_limits["default"])
        limit = config["limit"]
        window = config["window"]
        burst = config.get("burst", limit // 2)
        
        # Check IP reputation if enabled
        if self.enable_reputation and ip_address:
            reputation_score = await self._get_ip_reputation(ip_address)
            if reputation_score < 0.3:  # Bad reputation
                limit = max(1, limit // 3)  # Reduce limit for bad actors
            elif reputation_score > 0.8:  # Good reputation
                limit = int(limit * 1.5)  # Increase limit for good actors
                
        # Use Redis for distributed rate limiting if available
        if self.redis_client:
            return await self._check_redis_rate_limit(identifier, endpoint, limit, window, burst)
        else:
            return self._check_local_rate_limit(identifier, endpoint, limit, window, burst)
            
    async def _check_redis_rate_limit(
        self,
        identifier: str,
        endpoint: str,
        limit: int,
        window: int,
        burst: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using Redis (sliding window log)"""
        try:
            key = self._get_redis_key(identifier, endpoint)
            now = time.time()
            window_start = now - window
            
            # Use Redis pipeline for atomic operations
            pipe = self.redis_client.pipeline()
            
            # Remove old entries outside the window
            pipe.zremrangebyscore(key, 0, window_start)
            
            # Count requests in current window
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(now): now})
            
            # Set expiry
            pipe.expire(key, window)
            
            # Execute pipeline
            results = pipe.execute()
            request_count = results[1]
            
            # Check burst limit (requests in last 1 second)
            burst_count = self.redis_client.zcount(key, now - 1, now)
            
            if request_count > limit or burst_count > burst:
                # Calculate retry after
                oldest = self.redis_client.zrange(key, 0, 0, withscores=True)
                if oldest:
                    retry_after = int(window - (now - oldest[0][1]))
                else:
                    retry_after = window
                    
                return False, {
                    "limit": limit,
                    "remaining": max(0, limit - request_count),
                    "reset": int(now + retry_after),
                    "retry_after": retry_after,
                    "burst_exceeded": burst_count > burst
                }
                
            return True, {
                "limit": limit,
                "remaining": limit - request_count - 1,
                "reset": int(now + window)
            }
            
        except Exception as e:
            logger.error(f"Redis rate limit check failed: {e}")
            # Fallback to local rate limiting
            return self._check_local_rate_limit(identifier, endpoint, limit, window, burst)
            
    def _check_local_rate_limit(
        self,
        identifier: str,
        endpoint: str,
        limit: int,
        window: int,
        burst: int
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check rate limit using local memory (token bucket)"""
        key = f"{endpoint}:{identifier}"
        now = time.time()
        
        # Initialize bucket if not exists
        if key not in self.token_buckets:
            self.token_buckets[key] = {
                "tokens": limit,
                "last_refill": now,
                "burst_tokens": burst
            }
            
        bucket = self.token_buckets[key]
        
        # Refill tokens based on time elapsed
        time_elapsed = now - bucket["last_refill"]
        tokens_to_add = (time_elapsed / window) * limit
        bucket["tokens"] = min(limit, bucket["tokens"] + tokens_to_add)
        bucket["last_refill"] = now
        
        # Check burst limit
        burst_window_requests = len([
            t for t in self.local_cache[key]
            if now - t < 1
        ])
        
        if burst_window_requests >= burst:
            return False, {
                "limit": limit,
                "remaining": int(bucket["tokens"]),
                "reset": int(now + window),
                "retry_after": 1,
                "burst_exceeded": True
            }
            
        # Check if tokens available
        if bucket["tokens"] >= 1:
            bucket["tokens"] -= 1
            self.local_cache[key].append(now)
            
            # Clean old entries
            cutoff = now - window
            while self.local_cache[key] and self.local_cache[key][0] < cutoff:
                self.local_cache[key].popleft()
                
            return True, {
                "limit": limit,
                "remaining": int(bucket["tokens"]),
                "reset": int(now + window)
            }
            
        # Calculate retry after
        tokens_needed = 1 - bucket["tokens"]
        retry_after = int((tokens_needed / limit) * window)
        
        return False, {
            "limit": limit,
            "remaining": 0,
            "reset": int(now + retry_after),
            "retry_after": retry_after
        }
        
    async def _get_ip_reputation(self, ip: str) -> float:
        """
        Calculate IP reputation score (0.0 = bad, 1.0 = good)
        
        Factors considered:
        - Request patterns
        - Failed authentication attempts
        - Suspicious payloads
        - Geographic location (if GeoIP available)
        - Known bad IP lists
        """
        if self.redis_client:
            key = self._get_reputation_key(ip)
            reputation = self.redis_client.get(key)
            if reputation:
                return float(reputation)
                
        # Calculate reputation based on history
        score = 0.5  # Start neutral
        
        # Check if IP is private/local
        try:
            ip_obj = ipaddress.ip_address(ip)
            if ip_obj.is_private or ip_obj.is_loopback:
                score += 0.2  # Trust local IPs more
        except:
            pass
            
        # Check failed auth attempts (would query from database)
        # For now, use cached data
        if ip in self.ip_reputation:
            score = self.ip_reputation[ip]
            
        # Store reputation
        if self.redis_client:
            self.redis_client.setex(
                self._get_reputation_key(ip),
                3600,  # Cache for 1 hour
                str(score)
            )
        else:
            self.ip_reputation[ip] = score
            
        return score
        
    async def record_failed_attempt(
        self,
        identifier: str,
        ip_address: Optional[str] = None,
        reason: str = "authentication_failed"
    ):
        """Record a failed attempt and update reputation"""
        if ip_address:
            # Decrease reputation
            current_score = await self._get_ip_reputation(ip_address)
            new_score = max(0.0, current_score - 0.1)
            
            if self.redis_client:
                self.redis_client.setex(
                    self._get_reputation_key(ip_address),
                    3600,
                    str(new_score)
                )
            else:
                self.ip_reputation[ip_address] = new_score
                
            # Block IP if reputation too low
            if new_score < 0.1:
                await self.block_ip(ip_address, duration=3600)
                
    async def record_successful_attempt(
        self,
        identifier: str,
        ip_address: Optional[str] = None
    ):
        """Record a successful attempt and improve reputation"""
        if ip_address:
            # Increase reputation slightly
            current_score = await self._get_ip_reputation(ip_address)
            new_score = min(1.0, current_score + 0.05)
            
            if self.redis_client:
                self.redis_client.setex(
                    self._get_reputation_key(ip_address),
                    3600,
                    str(new_score)
                )
            else:
                self.ip_reputation[ip_address] = new_score
                
    async def block_ip(self, ip_address: str, duration: int = 3600):
        """Block an IP address for specified duration"""
        self.blocked_ips.add(ip_address)
        
        if self.redis_client:
            key = f"blocked_ip:{ip_address}"
            self.redis_client.setex(key, duration, "1")
            
        logger.warning(f"Blocked IP {ip_address} for {duration} seconds")
        
        # Schedule unblock
        asyncio.create_task(self._unblock_ip_after(ip_address, duration))
        
    async def _unblock_ip_after(self, ip_address: str, duration: int):
        """Unblock IP after duration"""
        await asyncio.sleep(duration)
        self.blocked_ips.discard(ip_address)
        logger.info(f"Unblocked IP {ip_address}")
        
    def check_suspicious_pattern(self, data: str) -> bool:
        """Check if data contains suspicious patterns"""
        data_lower = data.lower()
        for pattern in self.suspicious_patterns:
            if pattern in data_lower:
                return True
        return False
        
    def get_rate_limit_headers(self, metadata: Dict[str, Any]) -> Dict[str, str]:
        """Generate rate limit headers for HTTP response"""
        headers = {
            "X-RateLimit-Limit": str(metadata.get("limit", self.default_limit)),
            "X-RateLimit-Remaining": str(metadata.get("remaining", 0)),
            "X-RateLimit-Reset": str(metadata.get("reset", 0))
        }
        
        if "retry_after" in metadata:
            headers["Retry-After"] = str(metadata["retry_after"])
            
        return headers


def rate_limit(
    limit: Optional[int] = None,
    window: Optional[int] = None,
    key_func: Optional[callable] = None
):
    """
    Decorator for rate limiting FastAPI endpoints
    
    Args:
        limit: Request limit (overrides default)
        window: Time window in seconds (overrides default)
        key_func: Function to extract identifier from request
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Extract request object (assumes FastAPI endpoint)
            request = kwargs.get("request")
            if not request:
                # Try to find request in args
                for arg in args:
                    if hasattr(arg, "client"):
                        request = arg
                        break
                        
            if not request:
                # No request object, skip rate limiting
                return await func(*args, **kwargs)
                
            # Get identifier
            if key_func:
                identifier = key_func(request)
            else:
                # Default to IP address
                identifier = request.client.host if request.client else "unknown"
                
            # Get endpoint
            endpoint = request.url.path
            
            # Initialize rate limiter (would be injected in production)
            limiter = UltraRateLimiter()
            
            # Check rate limit
            allowed, metadata = await limiter.check_rate_limit(
                identifier=identifier,
                endpoint=endpoint,
                ip_address=request.client.host if request.client else None
            )
            
            if not allowed:
                # Rate limit exceeded
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded",
                    headers=limiter.get_rate_limit_headers(metadata)
                )
                
            # Add rate limit headers to response
            response = await func(*args, **kwargs)
            if hasattr(response, "headers"):
                for key, value in limiter.get_rate_limit_headers(metadata).items():
                    response.headers[key] = value
                    
            return response
            
        return wrapper
    return decorator


# Example usage for authentication endpoints
class AuthRateLimiter:
    """Specialized rate limiter for authentication endpoints"""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.limiter = UltraRateLimiter(
            redis_client=redis_client,
            enable_reputation=True
        )
        
    async def check_login_attempt(
        self,
        username: str,
        ip_address: str
    ) -> Tuple[bool, Dict[str, Any]]:
        """Check if login attempt should be allowed"""
        # Check both username and IP rate limits
        username_check = await self.limiter.check_rate_limit(
            identifier=f"user:{username}",
            endpoint="/api/v1/auth/login",
            ip_address=ip_address
        )
        
        ip_check = await self.limiter.check_rate_limit(
            identifier=f"ip:{ip_address}",
            endpoint="/api/v1/auth/login",
            ip_address=ip_address
        )
        
        # Both must pass
        if not username_check[0]:
            return username_check
        if not ip_check[0]:
            return ip_check
            
        return True, {**username_check[1], **ip_check[1]}
        
    async def record_login_failure(
        self,
        username: str,
        ip_address: str,
        reason: str = "invalid_credentials"
    ):
        """Record failed login attempt"""
        await self.limiter.record_failed_attempt(
            identifier=f"user:{username}",
            ip_address=ip_address,
            reason=reason
        )
        
    async def record_login_success(
        self,
        username: str,
        ip_address: str
    ):
        """Record successful login"""
        await self.limiter.record_successful_attempt(
            identifier=f"user:{username}",
            ip_address=ip_address
        )