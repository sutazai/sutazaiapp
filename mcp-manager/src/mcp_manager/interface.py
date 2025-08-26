"""
Unified MCP Interface

Provides a single, consistent interface for interacting with all MCP servers,
handling routing, load balancing, and failover automatically.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger

from .connection import ConnectionManager
from .discovery import ServerDiscoveryEngine
from .health import HealthMonitor
from .models import (
    HealthCheckResult,
    HealthStatus,
    MCPCapability,
    ServerConfig,
    ServerState,
    ServerStatus,
)


class UnifiedMCPInterface:
    """
    Unified interface for all MCP servers providing:
    - Automatic server selection and load balancing
    - Failover and retry logic
    - Capability aggregation and routing
    - Request throttling and rate limiting
    - Caching and performance optimization
    """
    
    def __init__(
        self,
        connection_manager: ConnectionManager,
        discovery_engine: ServerDiscoveryEngine,
        health_monitor: HealthMonitor
    ) -> None:
        self.connection_manager = connection_manager
        self.discovery_engine = discovery_engine
        self.health_monitor = health_monitor
        
        # Capability routing
        self._capability_registry: Dict[str, List[str]] = {}  # capability -> server names
        self._server_capabilities: Dict[str, List[MCPCapability]] = {}
        self._capability_cache_time: Dict[str, float] = {}
        self._capability_cache_ttl = 300.0  # 5 minutes
        
        # Load balancing and routing
        self._server_load: Dict[str, int] = {}  # server -> active requests
        self._round_robin_index: Dict[str, int] = {}  # capability -> index
        
        # Performance tracking
        self._request_stats: Dict[str, Dict[str, Any]] = {}
        self._response_times: Dict[str, List[float]] = {}
        
        # Configuration
        self._max_retries = 3
        self._retry_delay = 1.0
        self._request_timeout = 60.0
        self._enable_caching = True
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._cache_ttl = 60.0  # 1 minute default cache TTL
    
    async def refresh_capabilities(self, force: bool = False) -> None:
        """Refresh capability registry from all connected servers"""
        try:
            logger.debug("Refreshing capability registry")
            
            # Get all connection states
            connection_states = self.connection_manager.get_all_connection_states()
            
            # Clear old capabilities if forcing refresh
            if force:
                self._capability_registry.clear()
                self._server_capabilities.clear()
                self._capability_cache_time.clear()
            
            # Fetch capabilities from each connected server
            for server_name in connection_states.keys():
                try:
                    # Check if we need to refresh this server's capabilities
                    last_cache_time = self._capability_cache_time.get(server_name, 0)
                    if not force and time.time() - last_cache_time < self._capability_cache_ttl:
                        continue
                    
                    # Get server health
                    health = self.health_monitor.get_server_health(server_name)
                    if health and health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
                        # Fetch capabilities
                        capabilities = await self.connection_manager.list_tools(
                            server_name,
                            timeout=10.0
                        )
                        
                        # Update registry
                        self._server_capabilities[server_name] = capabilities
                        self._capability_cache_time[server_name] = time.time()
                        
                        # Update capability to server mapping
                        for capability in capabilities:
                            if capability.name not in self._capability_registry:
                                self._capability_registry[capability.name] = []
                            
                            if server_name not in self._capability_registry[capability.name]:
                                self._capability_registry[capability.name].append(server_name)
                        
                        logger.debug(
                            f"Updated capabilities for {server_name}: "
                            f"{len(capabilities)} capabilities"
                        )
                    
                except Exception as e:
                    logger.error(f"Failed to fetch capabilities from {server_name}: {e}")
            
            logger.info(f"Capability registry updated: {len(self._capability_registry)} unique capabilities")
            
        except Exception as e:
            logger.error(f"Error refreshing capabilities: {e}")
    
    async def list_all_capabilities(self) -> Dict[str, List[MCPCapability]]:
        """List all available capabilities across all servers"""
        await self.refresh_capabilities()
        return self._server_capabilities.copy()
    
    async def find_servers_for_capability(self, capability_name: str) -> List[str]:
        """Find all servers that provide a specific capability"""
        await self.refresh_capabilities()
        return self._capability_registry.get(capability_name, [])
    
    async def get_capability_info(self, capability_name: str) -> Optional[MCPCapability]:
        """Get detailed information about a capability"""
        servers = await self.find_servers_for_capability(capability_name)
        
        if not servers:
            return None
        
        # Return capability info from the first available server
        for server_name in servers:
            capabilities = self._server_capabilities.get(server_name, [])
            for capability in capabilities:
                if capability.name == capability_name:
                    return capability
        
        return None
    
    def _select_server_for_capability(self, capability_name: str, exclude_servers: Optional[List[str]] = None) -> Optional[str]:
        """Select the best server for a capability using load balancing"""
        available_servers = self._capability_registry.get(capability_name, [])
        
        if not available_servers:
            return None
        
        # Filter out excluded servers
        if exclude_servers:
            available_servers = [s for s in available_servers if s not in exclude_servers]
        
        if not available_servers:
            return None
        
        # Filter by health status
        healthy_servers = []
        for server_name in available_servers:
            health = self.health_monitor.get_server_health(server_name)
            if health and health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
                healthy_servers.append(server_name)
        
        if not healthy_servers:
            logger.warning(f"No healthy servers available for capability: {capability_name}")
            return None
        
        # Load balancing: select server with lowest load
        best_server = None
        min_load = float('inf')
        
        for server_name in healthy_servers:
            load = self._server_load.get(server_name, 0)
            if load < min_load:
                min_load = load
                best_server = server_name
        
        return best_server
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        server_preference: Optional[str] = None,
        timeout: Optional[float] = None,
        enable_retry: bool = True,
        enable_cache: Optional[bool] = None
    ) -> Any:
        """
        Call a tool on the best available server.
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
            server_preference: Preferred server name (if available)
            timeout: Request timeout
            enable_retry: Enable automatic retry on failure
            enable_cache: Enable response caching (None = use default)
            
        Returns:
            Tool execution result
            
        Raises:
            ValueError: If tool is not available
            ConnectionError: If no servers are available
            TimeoutError: If request times out
        """
        timeout = timeout or self._request_timeout
        enable_cache = enable_cache if enable_cache is not None else self._enable_caching
        
        # Check cache first
        if enable_cache:
            cache_key = self._generate_cache_key(tool_name, arguments)
            cached_result, cache_time = self._cache.get(cache_key, (None, 0))
            
            if cached_result is not None and time.time() - cache_time < self._cache_ttl:
                logger.debug(f"Returning cached result for {tool_name}")
                return cached_result
        
        # Refresh capabilities if needed
        await self.refresh_capabilities()
        
        # Find available servers
        available_servers = await self.find_servers_for_capability(tool_name)
        if not available_servers:
            raise ValueError(f"Tool '{tool_name}' is not available on any server")
        
        # Select server
        target_server = None
        
        # Use preference if available and healthy
        if server_preference and server_preference in available_servers:
            health = self.health_monitor.get_server_health(server_preference)
            if health and health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
                target_server = server_preference
        
        # Otherwise use load balancing
        if not target_server:
            target_server = self._select_server_for_capability(tool_name)
        
        if not target_server:
            raise ConnectionError(f"No healthy servers available for tool '{tool_name}'")
        
        # Execute with retry logic
        last_error = None
        excluded_servers = []
        
        for attempt in range(self._max_retries if enable_retry else 1):
            try:
                # Update load tracking
                self._server_load[target_server] = self._server_load.get(target_server, 0) + 1
                
                start_time = time.time()
                
                # Call the tool
                result = await self.connection_manager.call_tool(
                    target_server,
                    tool_name,
                    arguments,
                    timeout=timeout
                )
                
                # Track performance
                execution_time = time.time() - start_time
                self._track_performance(target_server, tool_name, execution_time, True)
                
                # Cache result if enabled
                if enable_cache:
                    cache_key = self._generate_cache_key(tool_name, arguments)
                    self._cache[cache_key] = (result, time.time())
                
                logger.debug(
                    f"Successfully called {tool_name} on {target_server} "
                    f"(attempt {attempt + 1}, {execution_time:.3f}s)"
                )
                
                return result
                
            except Exception as e:
                execution_time = time.time() - start_time
                self._track_performance(target_server, tool_name, execution_time, False)
                
                last_error = e
                logger.warning(
                    f"Failed to call {tool_name} on {target_server} "
                    f"(attempt {attempt + 1}): {e}"
                )
                
                # Don't retry on the same server
                excluded_servers.append(target_server)
                
                # Select different server for retry
                if enable_retry and attempt < self._max_retries - 1:
                    target_server = self._select_server_for_capability(tool_name, excluded_servers)
                    
                    if not target_server:
                        logger.error(f"No more servers available for retry of {tool_name}")
                        break
                    
                    # Brief delay before retry
                    await asyncio.sleep(self._retry_delay * (attempt + 1))
                
            finally:
                # Update load tracking
                if target_server:
                    self._server_load[target_server] = max(0, self._server_load.get(target_server, 1) - 1)
        
        # All retries failed
        raise last_error or ConnectionError(f"Failed to execute tool '{tool_name}' after {self._max_retries} attempts")
    
    def _generate_cache_key(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Generate cache key for tool call"""
        import hashlib
        import json
        
        # Create deterministic key from tool name and arguments
        data = {"tool": tool_name, "args": arguments}
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.md5(json_str.encode()).hexdigest()
    
    def _track_performance(self, server_name: str, tool_name: str, execution_time: float, success: bool) -> None:
        """Track performance metrics for server and tool"""
        # Initialize stats if needed
        if server_name not in self._request_stats:
            self._request_stats[server_name] = {
                "total_requests": 0,
                "successful_requests": 0,
                "failed_requests": 0,
                "total_execution_time": 0.0,
                "tools": {}
            }
        
        stats = self._request_stats[server_name]
        
        # Update server-level stats
        stats["total_requests"] += 1
        stats["total_execution_time"] += execution_time
        
        if success:
            stats["successful_requests"] += 1
        else:
            stats["failed_requests"] += 1
        
        # Update tool-level stats
        if tool_name not in stats["tools"]:
            stats["tools"][tool_name] = {
                "calls": 0,
                "successes": 0,
                "failures": 0,
                "total_time": 0.0,
                "min_time": float('inf'),
                "max_time": 0.0
            }
        
        tool_stats = stats["tools"][tool_name]
        tool_stats["calls"] += 1
        tool_stats["total_time"] += execution_time
        tool_stats["min_time"] = min(tool_stats["min_time"], execution_time)
        tool_stats["max_time"] = max(tool_stats["max_time"], execution_time)
        
        if success:
            tool_stats["successes"] += 1
        else:
            tool_stats["failures"] += 1
        
        # Track response times for server
        if server_name not in self._response_times:
            self._response_times[server_name] = []
        
        self._response_times[server_name].append(execution_time)
        
        # Keep only recent response times (last 100)
        if len(self._response_times[server_name]) > 100:
            self._response_times[server_name].pop(0)
    
    async def get_server_performance_stats(self, server_name: str) -> Dict[str, Any]:
        """Get performance statistics for a server"""
        stats = self._request_stats.get(server_name, {})
        response_times = self._response_times.get(server_name, [])
        
        if not stats:
            return {"error": "No statistics available"}
        
        # Calculate averages
        avg_execution_time = (
            stats["total_execution_time"] / stats["total_requests"]
            if stats["total_requests"] > 0 else 0
        )
        
        success_rate = (
            stats["successful_requests"] / stats["total_requests"] * 100
            if stats["total_requests"] > 0 else 0
        )
        
        # Calculate response time percentiles
        if response_times:
            sorted_times = sorted(response_times)
            p50 = sorted_times[len(sorted_times) // 2]
            p95 = sorted_times[int(len(sorted_times) * 0.95)]
            p99 = sorted_times[int(len(sorted_times) * 0.99)]
        else:
            p50 = p95 = p99 = 0
        
        return {
            "total_requests": stats["total_requests"],
            "successful_requests": stats["successful_requests"],
            "failed_requests": stats["failed_requests"],
            "success_rate_percent": success_rate,
            "average_execution_time": avg_execution_time,
            "response_time_p50": p50,
            "response_time_p95": p95,
            "response_time_p99": p99,
            "current_load": self._server_load.get(server_name, 0),
            "tools": stats.get("tools", {})
        }
    
    async def get_capability_summary(self) -> Dict[str, Any]:
        """Get summary of all capabilities and their availability"""
        await self.refresh_capabilities()
        
        summary = {
            "total_capabilities": len(self._capability_registry),
            "total_servers": len(self._server_capabilities),
            "capabilities": {}
        }
        
        for capability_name, server_list in self._capability_registry.items():
            healthy_servers = []
            
            for server_name in server_list:
                health = self.health_monitor.get_server_health(server_name)
                if health and health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
                    healthy_servers.append(server_name)
            
            summary["capabilities"][capability_name] = {
                "total_servers": len(server_list),
                "healthy_servers": len(healthy_servers),
                "server_list": server_list,
                "healthy_server_list": healthy_servers,
                "available": len(healthy_servers) > 0
            }
        
        return summary
    
    def clear_cache(self, tool_name: Optional[str] = None) -> None:
        """Clear response cache"""
        if tool_name:
            # Clear cache entries for specific tool
            keys_to_remove = []
            for key in self._cache.keys():
                if tool_name in key:  # Simple heuristic
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self._cache[key]
            
            logger.debug(f"Cleared {len(keys_to_remove)} cache entries for {tool_name}")
        else:
            # Clear all cache
            self._cache.clear()
            logger.debug("Cleared all cache entries")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "cache_entries": len(self._cache),
            "cache_enabled": self._enable_caching,
            "cache_ttl_seconds": self._cache_ttl
        }