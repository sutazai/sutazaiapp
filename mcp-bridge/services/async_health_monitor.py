#!/usr/bin/env python3
"""
Async health check module for MCP Bridge
Replaces synchronous health checks with concurrent async checks
"""

import asyncio
import httpx
import logging
from typing import Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class AsyncHealthMonitor:
    """Asynchronous health monitoring for agents and services"""
    
    def __init__(self, timeout: float = 2.0, cache_ttl: int = 10):
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self.health_cache: Dict[str, Dict[str, Any]] = {}
        self.http_client = None
    
    async def __aenter__(self):
        self.http_client = httpx.AsyncClient(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.http_client:
            await self.http_client.aclose()
    
    async def check_agent_health(self, agent_id: str, port: int) -> Dict[str, Any]:
        """Check single agent health with caching"""
        
        # Check cache first
        cache_key = f"agent:{agent_id}"
        if cache_key in self.health_cache:
            cached = self.health_cache[cache_key]
            if cached["expires"] > datetime.now():
                return cached["status"]
        
        # Perform health check
        status = {
            "agent_id": agent_id,
            "port": port,
            "status": "unknown",
            "last_check": datetime.now().isoformat(),
            "response_time": None,
            "error": None
        }
        
        try:
            start_time = asyncio.get_event_loop().time()
            response = await self.http_client.get(
                f"http://localhost:{port}/health"
            )
            response_time = asyncio.get_event_loop().time() - start_time
            
            if response.status_code == 200:
                status["status"] = "online"
                status["response_time"] = round(response_time * 1000, 2)  # ms
                data = response.json()
                status["details"] = data
            else:
                status["status"] = "degraded"
                status["error"] = f"HTTP {response.status_code}"
                
        except asyncio.TimeoutError:
            status["status"] = "timeout"
            status["error"] = f"Timeout after {self.timeout}s"
        except httpx.ConnectError:
            status["status"] = "offline"
            status["error"] = "Connection refused"
        except Exception as e:
            status["status"] = "error"
            status["error"] = str(e)
        
        # Cache the result
        self.health_cache[cache_key] = {
            "status": status,
            "expires": datetime.now() + timedelta(seconds=self.cache_ttl)
        }
        
        return status
    
    async def check_all_agents(self, agents: Dict[str, Dict]) -> Dict[str, Any]:
        """Check all agents concurrently"""
        
        tasks = []
        for agent_id, agent_info in agents.items():
            port = agent_info.get("port")
            if port:
                task = self.check_agent_health(agent_id, port)
                tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        health_summary = {
            "timestamp": datetime.now().isoformat(),
            "total_agents": len(agents),
            "online": 0,
            "offline": 0,
            "degraded": 0,
            "agents": {}
        }
        
        for i, (agent_id, agent_info) in enumerate(agents.items()):
            if i < len(results):
                result = results[i]
                if isinstance(result, Exception):
                    health_summary["agents"][agent_id] = {
                        "status": "error",
                        "error": str(result)
                    }
                    health_summary["offline"] += 1
                else:
                    health_summary["agents"][agent_id] = result
                    status = result.get("status", "unknown")
                    if status == "online":
                        health_summary["online"] += 1
                    elif status == "offline":
                        health_summary["offline"] += 1
                    else:
                        health_summary["degraded"] += 1
        
        return health_summary

# Integration function for MCP Bridge
async def integrate_async_health_monitor(agent_registry: Dict):
    """Integration point for MCP Bridge server"""
    
    async with AsyncHealthMonitor() as monitor:
        return await monitor.check_all_agents(agent_registry)
