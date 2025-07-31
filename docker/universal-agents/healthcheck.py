#!/usr/bin/env python3
"""
Health Check Script for Universal Agent System
=============================================

This script performs health checks on the universal agent system
to ensure all components are functioning properly.
"""

import asyncio
import sys
import time
from typing import Dict, Any

try:
    import aioredis
    import httpx
except ImportError:
    print("Required packages not available")
    sys.exit(1)


async def check_redis_connection() -> bool:
    """Check Redis connection"""
    try:
        redis = aioredis.from_url("redis://universal-agent-redis:6379", decode_responses=True)
        await redis.ping()
        await redis.close()
        return True
    except Exception as e:
        print(f"Redis health check failed: {e}")
        return False


async def check_ollama_connection() -> bool:
    """Check Ollama connection"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://universal-agent-ollama:11434/api/tags",
                timeout=10.0
            )
            return response.status_code == 200
    except Exception as e:
        print(f"Ollama health check failed: {e}")
        return False


async def check_system_health() -> bool:
    """Check overall system health"""
    try:
        # Check if we can import the core system
        sys.path.insert(0, '/app')
        from backend.ai_agents.core import UniversalAgentSystem
        return True
    except Exception as e:
        print(f"System health check failed: {e}")
        return False


async def main():
    """Main health check function"""
    print("Starting health check...")
    
    checks = [
        ("Redis", check_redis_connection()),
        ("Ollama", check_ollama_connection()),
        ("System", check_system_health())
    ]
    
    results = {}
    
    for name, check_coro in checks:
        try:
            result = await check_coro
            results[name] = result
            status = "✓" if result else "✗"
            print(f"{status} {name}: {'OK' if result else 'FAILED'}")
        except Exception as e:
            results[name] = False
            print(f"✗ {name}: ERROR - {e}")
    
    # Overall health status
    all_healthy = all(results.values())
    
    if all_healthy:
        print("🟢 All health checks passed")
        sys.exit(0)
    else:
        failed_checks = [name for name, result in results.items() if not result]
        print(f"🔴 Health check failed: {', '.join(failed_checks)}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())