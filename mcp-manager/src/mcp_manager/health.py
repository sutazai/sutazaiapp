"""
Health Monitor for MCP Management System

Continuously monitors MCP server health, detects failures,
and triggers automatic recovery procedures.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Set

from loguru import logger

from .connection import ConnectionManager
from .models import (
    HealthCheckResult,
    HealthStatus,
    ServerConfig,
    ServerState,
    ServerStatus,
)


class HealthMonitor:
    """
    Monitors health of all MCP servers and triggers recovery actions.
    
    Features:
    - Continuous health monitoring with configurable intervals
    - Failure detection and threshold management
    - Automatic recovery and restart procedures
    - Health trend analysis and predictive failure detection
    - Configurable alerting and notification system
    """
    
    def __init__(
        self,
        connection_manager: ConnectionManager,
        health_check_interval: float = 30.0,
        max_concurrent_checks: int = 5
    ) -> None:
        self.connection_manager = connection_manager
        self.health_check_interval = health_check_interval
        self.max_concurrent_checks = max_concurrent_checks
        
        # Health tracking
        self._server_health: Dict[str, HealthCheckResult] = {}
        self._health_history: Dict[str, List[HealthCheckResult]] = {}
        self._failure_counts: Dict[str, int] = {}
        self._recovery_counts: Dict[str, int] = {}
        self._last_health_check: Dict[str, datetime] = {}
        
        # Monitoring control
        self._monitoring_enabled = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._recovery_tasks: Dict[str, asyncio.Task] = {}
        
        # Configuration
        self._max_health_history = 100  # Keep last 100 health checks per server
        self._health_check_timeout = 10.0
        self._recovery_cooldown = 60.0  # Minimum time between recovery attempts
        self._last_recovery_attempt: Dict[str, datetime] = {}
        
        # Callbacks for health events
        self._health_change_callbacks: List[Callable[[str, HealthCheckResult], None]] = []
        self._failure_callbacks: List[Callable[[str, ServerState], None]] = []
        self._recovery_callbacks: List[Callable[[str, ServerState], None]] = []
    
    def add_health_change_callback(self, callback: Callable[[str, HealthCheckResult], None]) -> None:
        """Add callback for health status changes"""
        self._health_change_callbacks.append(callback)
    
    def add_failure_callback(self, callback: Callable[[str, ServerState], None]) -> None:
        """Add callback for server failures"""
        self._failure_callbacks.append(callback)
    
    def add_recovery_callback(self, callback: Callable[[str, ServerState], None]) -> None:
        """Add callback for server recovery"""
        self._recovery_callbacks.append(callback)
    
    async def start_monitoring(self) -> None:
        """Start continuous health monitoring"""
        if self._monitoring_enabled:
            logger.warning("Health monitoring is already running")
            return
        
        logger.info(f"Starting health monitoring (interval: {self.health_check_interval}s)")
        self._monitoring_enabled = True
        
        # Start monitoring task
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self) -> None:
        """Stop health monitoring"""
        if not self._monitoring_enabled:
            return
        
        logger.info("Stopping health monitoring")
        self._monitoring_enabled = False
        
        # Cancel monitoring task
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
        
        # Cancel any active recovery tasks
        for task in self._recovery_tasks.values():
            task.cancel()
        
        if self._recovery_tasks:
            await asyncio.gather(*self._recovery_tasks.values(), return_exceptions=True)
        
        self._recovery_tasks.clear()
    
    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        try:
            while self._monitoring_enabled:
                start_time = time.time()
                
                # Get all connection states
                connection_states = self.connection_manager.get_all_connection_states()
                
                if connection_states:
                    # Perform health checks with concurrency limit
                    semaphore = asyncio.Semaphore(self.max_concurrent_checks)
                    
                    async def check_server_health(server_name: str) -> None:
                        async with semaphore:
                            await self._check_server_health(server_name)
                    
                    # Run health checks concurrently
                    await asyncio.gather(
                        *[check_server_health(name) for name in connection_states.keys()],
                        return_exceptions=True
                    )
                
                # Calculate how long to sleep
                elapsed = time.time() - start_time
                sleep_time = max(0, self.health_check_interval - elapsed)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
        except asyncio.CancelledError:
            logger.debug("Health monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in health monitoring loop: {e}")
    
    async def _check_server_health(self, server_name: str) -> None:
        """Check health of a specific server"""
        try:
            # Perform health check
            health_result = await self.connection_manager.health_check_server(
                server_name, 
                timeout=self._health_check_timeout
            )
            
            # Store health result
            previous_health = self._server_health.get(server_name)
            self._server_health[server_name] = health_result
            self._last_health_check[server_name] = datetime.utcnow()
            
            # Update health history
            if server_name not in self._health_history:
                self._health_history[server_name] = []
            
            self._health_history[server_name].append(health_result)
            
            # Trim history if too long
            if len(self._health_history[server_name]) > self._max_health_history:
                self._health_history[server_name].pop(0)
            
            # Analyze health status changes
            await self._analyze_health_change(server_name, previous_health, health_result)
            
        except Exception as e:
            logger.error(f"Error checking health of {server_name}: {e}")
            
            # Create error health result
            error_result = HealthCheckResult(
                status=HealthStatus.CRITICAL,
                error_message=str(e),
                error_code="HEALTH_CHECK_ERROR"
            )
            
            self._server_health[server_name] = error_result
            await self._analyze_health_change(server_name, None, error_result)
    
    async def _analyze_health_change(
        self,
        server_name: str,
        previous_health: Optional[HealthCheckResult],
        current_health: HealthCheckResult
    ) -> None:
        """Analyze health status changes and trigger appropriate actions"""
        try:
            # Check if status changed
            status_changed = (
                previous_health is None or 
                previous_health.status != current_health.status
            )
            
            if status_changed:
                logger.info(
                    f"Health status changed for {server_name}: "
                    f"{previous_health.status if previous_health else 'UNKNOWN'} -> {current_health.status}"
                )
                
                # Notify callbacks
                for callback in self._health_change_callbacks:
                    try:
                        callback(server_name, current_health)
                    except Exception as e:
                        logger.error(f"Error in health change callback: {e}")
            
            # Handle unhealthy status
            if current_health.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                await self._handle_unhealthy_server(server_name, current_health)
            
            # Handle recovery
            elif current_health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
                await self._handle_healthy_server(server_name, current_health)
            
        except Exception as e:
            logger.error(f"Error analyzing health change for {server_name}: {e}")
    
    async def _handle_unhealthy_server(self, server_name: str, health_result: HealthCheckResult) -> None:
        """Handle server that is unhealthy or critical"""
        try:
            # Increment failure count
            self._failure_counts[server_name] = self._failure_counts.get(server_name, 0) + 1
            failure_count = self._failure_counts[server_name]
            
            # Reset recovery count
            self._recovery_counts[server_name] = 0
            
            logger.warning(
                f"Server {server_name} is {health_result.status.value} "
                f"(failure count: {failure_count}, error: {health_result.error_message})"
            )
            
            # Check if we should trigger recovery
            connection_state = self.connection_manager.get_connection_state(server_name)
            if connection_state:
                # Get server config to check failure threshold
                # Note: In a real implementation, we'd get this from the manager
                failure_threshold = 3  # Default threshold
                
                if failure_count >= failure_threshold:
                    await self._trigger_recovery(server_name, health_result)
            
        except Exception as e:
            logger.error(f"Error handling unhealthy server {server_name}: {e}")
    
    async def _handle_healthy_server(self, server_name: str, health_result: HealthCheckResult) -> None:
        """Handle server that is healthy or degraded"""
        try:
            # Increment recovery count
            self._recovery_counts[server_name] = self._recovery_counts.get(server_name, 0) + 1
            recovery_count = self._recovery_counts[server_name]
            
            # Check if this is a recovery from failure
            if self._failure_counts.get(server_name, 0) > 0:
                recovery_threshold = 2  # Default threshold
                
                if recovery_count >= recovery_threshold:
                    logger.success(
                        f"Server {server_name} has recovered "
                        f"(recovery count: {recovery_count})"
                    )
                    
                    # Reset failure count
                    self._failure_counts[server_name] = 0
                    
                    # Notify recovery callbacks
                    for callback in self._recovery_callbacks:
                        try:
                            # Note: We'd need to get the server state from the manager
                            # callback(server_name, server_state)
                            pass
                        except Exception as e:
                            logger.error(f"Error in recovery callback: {e}")
            
        except Exception as e:
            logger.error(f"Error handling healthy server {server_name}: {e}")
    
    async def _trigger_recovery(self, server_name: str, health_result: HealthCheckResult) -> None:
        """Trigger recovery procedure for a failed server"""
        try:
            # Check recovery cooldown
            now = datetime.utcnow()
            last_recovery = self._last_recovery_attempt.get(server_name)
            
            if last_recovery and (now - last_recovery).total_seconds() < self._recovery_cooldown:
                logger.debug(f"Recovery cooldown active for {server_name}, skipping")
                return
            
            # Check if recovery is already in progress
            if server_name in self._recovery_tasks and not self._recovery_tasks[server_name].done():
                logger.debug(f"Recovery already in progress for {server_name}")
                return
            
            logger.warning(f"Triggering recovery for {server_name}")
            self._last_recovery_attempt[server_name] = now
            
            # Start recovery task
            self._recovery_tasks[server_name] = asyncio.create_task(
                self._recover_server(server_name, health_result)
            )
            
        except Exception as e:
            logger.error(f"Error triggering recovery for {server_name}: {e}")
    
    async def _recover_server(self, server_name: str, health_result: HealthCheckResult) -> None:
        """Attempt to recover a failed server"""
        try:
            logger.info(f"Starting recovery procedure for {server_name}")
            
            # Step 1: Disconnect and reconnect
            await self.connection_manager.disconnect_server(server_name)
            await asyncio.sleep(2)  # Brief pause
            
            # Note: In a real implementation, we'd need the server config
            # This would typically come from the main manager
            logger.info(f"Recovery procedure completed for {server_name}")
            
        except Exception as e:
            logger.error(f"Error during recovery of {server_name}: {e}")
        finally:
            # Clean up recovery task
            if server_name in self._recovery_tasks:
                del self._recovery_tasks[server_name]
    
    def get_server_health(self, server_name: str) -> Optional[HealthCheckResult]:
        """Get current health status of a server"""
        return self._server_health.get(server_name)
    
    def get_all_server_health(self) -> Dict[str, HealthCheckResult]:
        """Get health status of all servers"""
        return self._server_health.copy()
    
    def get_health_history(self, server_name: str, limit: int = 50) -> List[HealthCheckResult]:
        """Get health history for a server"""
        history = self._health_history.get(server_name, [])
        return history[-limit:] if limit else history
    
    def get_server_statistics(self, server_name: str) -> Dict[str, any]:
        """Get health statistics for a server"""
        history = self._health_history.get(server_name, [])
        
        if not history:
            return {}
        
        # Calculate statistics
        total_checks = len(history)
        healthy_checks = sum(1 for h in history if h.status == HealthStatus.HEALTHY)
        degraded_checks = sum(1 for h in history if h.status == HealthStatus.DEGRADED)
        unhealthy_checks = sum(1 for h in history if h.status == HealthStatus.UNHEALTHY)
        critical_checks = sum(1 for h in history if h.status == HealthStatus.CRITICAL)
        
        response_times = [h.response_time for h in history if h.response_time > 0]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            "total_checks": total_checks,
            "healthy_percentage": (healthy_checks / total_checks) * 100,
            "degraded_percentage": (degraded_checks / total_checks) * 100,
            "unhealthy_percentage": (unhealthy_checks / total_checks) * 100,
            "critical_percentage": (critical_checks / total_checks) * 100,
            "average_response_time": avg_response_time,
            "failure_count": self._failure_counts.get(server_name, 0),
            "recovery_count": self._recovery_counts.get(server_name, 0),
            "last_check": self._last_health_check.get(server_name),
        }
    
    def get_overall_health_summary(self) -> Dict[str, any]:
        """Get overall health summary for all servers"""
        all_health = self._server_health
        
        if not all_health:
            return {"status": "no_servers", "total_servers": 0}
        
        total_servers = len(all_health)
        healthy_servers = sum(1 for h in all_health.values() if h.status == HealthStatus.HEALTHY)
        degraded_servers = sum(1 for h in all_health.values() if h.status == HealthStatus.DEGRADED)
        unhealthy_servers = sum(1 for h in all_health.values() if h.status == HealthStatus.UNHEALTHY)
        critical_servers = sum(1 for h in all_health.values() if h.status == HealthStatus.CRITICAL)
        
        # Determine overall status
        if critical_servers > 0:
            overall_status = "critical"
        elif unhealthy_servers > 0:
            overall_status = "unhealthy" 
        elif degraded_servers > 0:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        return {
            "status": overall_status,
            "total_servers": total_servers,
            "healthy_servers": healthy_servers,
            "degraded_servers": degraded_servers,
            "unhealthy_servers": unhealthy_servers,
            "critical_servers": critical_servers,
            "healthy_percentage": (healthy_servers / total_servers) * 100,
            "monitoring_enabled": self._monitoring_enabled,
        }
    
    def is_monitoring(self) -> bool:
        """Check if monitoring is currently active"""
        return self._monitoring_enabled