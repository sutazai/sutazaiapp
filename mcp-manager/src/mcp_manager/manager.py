"""
Main MCP Manager

Central orchestrator that coordinates all MCP management components
providing a unified API for server lifecycle management.
"""

import asyncio
import json
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from pydantic import ValidationError

from .connection import ConnectionManager
from .discovery import ServerDiscoveryEngine
from .health import HealthMonitor
from .interface import UnifiedMCPInterface
from .models import (
    MCPManagerConfig,
    ServerConfig,
    ServerState,
    ServerStatus,
    ServerMetrics,
    HealthCheckResult,
)


class MCPManager:
    """
    Main MCP Management System coordinating all components.
    
    Provides:
    - Centralized server lifecycle management
    - Automated discovery and configuration
    - Health monitoring and recovery
    - Unified interface for all MCP operations
    - State persistence and recovery
    - Performance monitoring and metrics
    """
    
    def __init__(self, config: Optional[MCPManagerConfig] = None) -> None:
        # Load configuration
        self.config = config or MCPManagerConfig()
        
        # Initialize components
        self.connection_manager = ConnectionManager()
        self.discovery_engine = ServerDiscoveryEngine(self.config.config_directories)
        self.health_monitor = HealthMonitor(
            self.connection_manager,
            self.config.global_health_check_interval,
            self.config.max_concurrent_health_checks
        )
        self.unified_interface = UnifiedMCPInterface(
            self.connection_manager,
            self.discovery_engine,
            self.health_monitor
        )
        
        # State management
        self._server_states: Dict[str, ServerState] = {}
        self._is_running = False
        self._background_tasks: List[asyncio.Task] = []
        
        # Setup logging
        self._configure_logging()
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Setup health monitor callbacks
        self._setup_health_callbacks()
    
    def _configure_logging(self) -> None:
        """Configure logging based on configuration"""
        logger.remove()  # Remove default handler
        
        # Console handler
        logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=self.config.log_level,
            colorize=True
        )
        
        # File handler if configured
        if self.config.log_file:
            logger.add(
                self.config.log_file,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                level=self.config.log_level,
                rotation="10 MB",
                retention="1 week",
                compression="gz"
            )
    
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful shutdown"""
        try:
            for sig in [signal.SIGINT, signal.SIGTERM]:
                signal.signal(sig, self._signal_handler)
        except (AttributeError, ValueError):
            # Signal handling not available (e.g., on Windows)
            pass
    
    def _signal_handler(self, signum: int, frame) -> None:
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        asyncio.create_task(self.stop())
    
    def _setup_health_callbacks(self) -> None:
        """Setup callbacks for health monitoring events"""
        self.health_monitor.add_health_change_callback(self._on_health_change)
        self.health_monitor.add_failure_callback(self._on_server_failure)
        self.health_monitor.add_recovery_callback(self._on_server_recovery)
    
    async def start(self) -> None:
        """Start the MCP Manager and all its components"""
        if self._is_running:
            logger.warning("MCP Manager is already running")
            return
        
        logger.info("Starting MCP Manager")
        
        try:
            # Load persisted state
            await self._load_state()
            
            # Start server discovery
            logger.info("Discovering MCP servers...")
            discovered_servers = await self.discovery_engine.discover_servers(force_refresh=True)
            logger.info(f"Discovered {len(discovered_servers)} MCP servers")
            
            # Initialize server states
            for server_name, server_config in discovered_servers.items():
                if server_config.enabled:
                    self._server_states[server_name] = ServerState(
                        config=server_config,
                        status=ServerStatus.STOPPED,
                        health=HealthCheckResult(status=HealthStatus.UNHEALTHY),
                        metrics=ServerMetrics()
                    )
            
            # Start enabled servers
            await self._start_enabled_servers()
            
            # Start health monitoring
            await self.health_monitor.start_monitoring()
            
            # Start background tasks
            await self._start_background_tasks()
            
            # Mark as running
            self._is_running = True
            
            logger.success("MCP Manager started successfully")
            
        except Exception as e:
            logger.error(f"Failed to start MCP Manager: {e}")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the MCP Manager and all components"""
        if not self._is_running:
            logger.debug("MCP Manager is not running")
            return
        
        logger.info("Stopping MCP Manager")
        
        try:
            # Mark as stopping
            self._is_running = False
            
            # Stop background tasks
            await self._stop_background_tasks()
            
            # Stop health monitoring
            await self.health_monitor.stop_monitoring()
            
            # Stop discovery engine file watching
            await self.discovery_engine.stop_watching()
            
            # Disconnect all servers
            await self._stop_all_servers()
            
            # Close all connections
            await self.connection_manager.close_all_connections()
            
            # Save state
            await self._save_state()
            
            logger.success("MCP Manager stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping MCP Manager: {e}")
    
    async def _start_enabled_servers(self) -> None:
        """Start all enabled servers"""
        enabled_servers = [
            name for name, state in self._server_states.items()
            if state.config.enabled
        ]
        
        if not enabled_servers:
            logger.info("No enabled servers to start")
            return
        
        logger.info(f"Starting {len(enabled_servers)} servers")
        
        # Start servers with concurrency limit
        semaphore = asyncio.Semaphore(self.config.max_startup_concurrency)
        
        async def start_server(server_name: str) -> None:
            async with semaphore:
                await self.start_server(server_name)
        
        # Start all servers concurrently
        tasks = [start_server(name) for name in enabled_servers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log results
        successful = sum(1 for r in results if not isinstance(r, Exception))
        failed = len(results) - successful
        
        logger.info(f"Server startup completed: {successful} successful, {failed} failed")
    
    async def _stop_all_servers(self) -> None:
        """Stop all running servers"""
        running_servers = [
            name for name, state in self._server_states.items()
            if state.status in [ServerStatus.RUNNING, ServerStatus.STARTING]
        ]
        
        if not running_servers:
            return
        
        logger.info(f"Stopping {len(running_servers)} servers")
        
        # Stop servers concurrently
        tasks = [self.stop_server(name) for name in running_servers]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _start_background_tasks(self) -> None:
        """Start background maintenance tasks"""
        # Periodic server discovery
        if self.config.auto_discovery:
            discovery_task = asyncio.create_task(self._periodic_discovery())
            self._background_tasks.append(discovery_task)
        
        # Periodic state saving
        save_task = asyncio.create_task(self._periodic_state_save())
        self._background_tasks.append(save_task)
        
        # Metrics collection
        metrics_task = asyncio.create_task(self._periodic_metrics_collection())
        self._background_tasks.append(metrics_task)
        
        logger.debug(f"Started {len(self._background_tasks)} background tasks")
    
    async def _stop_background_tasks(self) -> None:
        """Stop all background tasks"""
        if not self._background_tasks:
            return
        
        logger.debug(f"Stopping {len(self._background_tasks)} background tasks")
        
        # Cancel all tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self._background_tasks.clear()
    
    async def _periodic_discovery(self) -> None:
        """Periodic server discovery task"""
        try:
            while self._is_running:
                await asyncio.sleep(self.config.discovery_interval)
                
                if self._is_running:  # Check again after sleep
                    try:
                        discovered = await self.discovery_engine.discover_servers()
                        await self._update_discovered_servers(discovered)
                    except Exception as e:
                        logger.error(f"Error in periodic discovery: {e}")
                        
        except asyncio.CancelledError:
            logger.debug("Periodic discovery task cancelled")
    
    async def _periodic_state_save(self) -> None:
        """Periodic state saving task"""
        try:
            while self._is_running:
                await asyncio.sleep(300)  # Save every 5 minutes
                
                if self._is_running:
                    try:
                        await self._save_state()
                        logger.debug("Periodic state save completed")
                    except Exception as e:
                        logger.error(f"Error in periodic state save: {e}")
                        
        except asyncio.CancelledError:
            logger.debug("Periodic state save task cancelled")
    
    async def _periodic_metrics_collection(self) -> None:
        """Periodic metrics collection task"""
        try:
            while self._is_running:
                await asyncio.sleep(self.config.metrics_collection_interval)
                
                if self._is_running:
                    try:
                        await self._collect_metrics()
                    except Exception as e:
                        logger.error(f"Error in metrics collection: {e}")
                        
        except asyncio.CancelledError:
            logger.debug("Metrics collection task cancelled")
    
    async def _collect_metrics(self) -> None:
        """Collect performance metrics from all servers"""
        for server_name, server_state in self._server_states.items():
            try:
                if server_state.status == ServerStatus.RUNNING:
                    # Get performance stats from unified interface
                    perf_stats = await self.unified_interface.get_server_performance_stats(server_name)
                    
                    # Update metrics
                    server_state.metrics.total_requests = perf_stats.get("total_requests", 0)
                    server_state.metrics.successful_requests = perf_stats.get("successful_requests", 0)
                    server_state.metrics.failed_requests = perf_stats.get("failed_requests", 0)
                    server_state.metrics.avg_response_time = perf_stats.get("average_execution_time", 0.0)
                    server_state.metrics.active_connections = perf_stats.get("current_load", 0)
                    
                    # Calculate uptime
                    if server_state.start_time:
                        uptime = (datetime.utcnow() - server_state.start_time).total_seconds()
                        server_state.metrics.uptime_seconds = uptime
                    
            except Exception as e:
                logger.error(f"Error collecting metrics for {server_name}: {e}")
    
    async def _update_discovered_servers(self, discovered_servers: Dict[str, ServerConfig]) -> None:
        """Update server states with newly discovered servers"""
        # Add new servers
        for server_name, server_config in discovered_servers.items():
            if server_name not in self._server_states and server_config.enabled:
                logger.info(f"New server discovered: {server_name}")
                
                server_state = ServerState(
                    config=server_config,
                    status=ServerStatus.STOPPED,
                    health=HealthCheckResult(status=HealthStatus.UNHEALTHY),
                    metrics=ServerMetrics()
                )
                
                self._server_states[server_name] = server_state
                
                # Auto-start if enabled
                if server_config.enabled:
                    await self.start_server(server_name)
        
        # Remove servers that are no longer discovered (optional)
        # This is commented out for safety - we don't want to auto-remove servers
        # discovered_names = set(discovered_servers.keys())
        # current_names = set(self._server_states.keys())
        # removed_servers = current_names - discovered_names
        # 
        # for server_name in removed_servers:
        #     logger.info(f"Server no longer discovered: {server_name}")
        #     await self.stop_server(server_name)
        #     del self._server_states[server_name]
    
    async def start_server(self, server_name: str) -> Tuple[bool, Optional[str]]:
        """Start a specific MCP server"""
        if server_name not in self._server_states:
            return False, f"Server '{server_name}' not found"
        
        server_state = self._server_states[server_name]
        
        if server_state.status == ServerStatus.RUNNING:
            return True, f"Server '{server_name}' is already running"
        
        logger.info(f"Starting server: {server_name}")
        
        try:
            # Update status
            server_state.status = ServerStatus.STARTING
            server_state.start_time = datetime.utcnow()
            
            # Connect to server
            success, error = await self.connection_manager.connect_server(
                server_state.config,
                force_reconnect=True
            )
            
            if success:
                server_state.status = ServerStatus.RUNNING
                server_state.connection_established = datetime.utcnow()
                server_state.connection_failures = 0
                
                # Refresh capabilities
                await self.unified_interface.refresh_capabilities()
                
                logger.success(f"Server '{server_name}' started successfully")
                return True, None
            else:
                server_state.status = ServerStatus.ERROR
                server_state.connection_failures += 1
                server_state.last_connection_error = error
                
                logger.error(f"Failed to start server '{server_name}': {error}")
                return False, error
                
        except Exception as e:
            error_msg = f"Error starting server '{server_name}': {str(e)}"
            logger.error(error_msg)
            
            server_state.status = ServerStatus.ERROR
            server_state.last_connection_error = error_msg
            
            return False, error_msg
    
    async def stop_server(self, server_name: str) -> Tuple[bool, Optional[str]]:
        """Stop a specific MCP server"""
        if server_name not in self._server_states:
            return False, f"Server '{server_name}' not found"
        
        server_state = self._server_states[server_name]
        
        if server_state.status == ServerStatus.STOPPED:
            return True, f"Server '{server_name}' is already stopped"
        
        logger.info(f"Stopping server: {server_name}")
        
        try:
            # Update status
            server_state.status = ServerStatus.STOPPING
            
            # Disconnect from server
            success = await self.connection_manager.disconnect_server(server_name)
            
            # Update status regardless of disconnect result
            server_state.status = ServerStatus.STOPPED
            server_state.connection_established = None
            server_state.process_id = None
            
            logger.success(f"Server '{server_name}' stopped")
            return True, None
            
        except Exception as e:
            error_msg = f"Error stopping server '{server_name}': {str(e)}"
            logger.error(error_msg)
            
            server_state.status = ServerStatus.ERROR
            return False, error_msg
    
    async def restart_server(self, server_name: str) -> Tuple[bool, Optional[str]]:
        """Restart a specific MCP server"""
        logger.info(f"Restarting server: {server_name}")
        
        # Stop server first
        stop_success, stop_error = await self.stop_server(server_name)
        if not stop_success:
            return False, f"Failed to stop server for restart: {stop_error}"
        
        # Brief delay
        await asyncio.sleep(2)
        
        # Start server
        return await self.start_server(server_name)
    
    def _on_health_change(self, server_name: str, health_result: HealthCheckResult) -> None:
        """Handle health status changes"""
        if server_name in self._server_states:
            self._server_states[server_name].health = health_result
            
            # Update last activity
            self._server_states[server_name].last_activity = datetime.utcnow()
    
    def _on_server_failure(self, server_name: str, server_state: ServerState) -> None:
        """Handle server failure events"""
        logger.warning(f"Server failure detected: {server_name}")
        
        # Update status if not already in error state
        if server_state.status not in [ServerStatus.ERROR, ServerStatus.CRASHED]:
            server_state.status = ServerStatus.ERROR
            server_state.metrics.restart_count += 1
            server_state.metrics.last_restart = datetime.utcnow()
        
        # Trigger auto-recovery if enabled
        if server_state.config.auto_restart and self.config.global_auto_recovery:
            asyncio.create_task(self._auto_recover_server(server_name))
    
    def _on_server_recovery(self, server_name: str, server_state: ServerState) -> None:
        """Handle server recovery events"""
        logger.success(f"Server recovery detected: {server_name}")
        
        # Update status if was in error state
        if server_state.status in [ServerStatus.ERROR, ServerStatus.CRASHED]:
            server_state.status = ServerStatus.RUNNING
    
    async def _auto_recover_server(self, server_name: str) -> None:
        """Attempt automatic server recovery"""
        if server_name not in self._server_states:
            return
        
        server_state = self._server_states[server_name]
        
        logger.info(f"Attempting auto-recovery for {server_name}")
        
        try:
            # Check restart limits
            if server_state.metrics.restart_count >= server_state.config.max_restart_attempts:
                logger.error(
                    f"Max restart attempts reached for {server_name} "
                    f"({server_state.metrics.restart_count})"
                )
                return
            
            # Delay before restart
            await asyncio.sleep(server_state.config.restart_delay)
            
            # Attempt restart
            success, error = await self.restart_server(server_name)
            
            if success:
                logger.success(f"Auto-recovery successful for {server_name}")
            else:
                logger.error(f"Auto-recovery failed for {server_name}: {error}")
                
        except Exception as e:
            logger.error(f"Error in auto-recovery for {server_name}: {e}")
    
    async def _load_state(self) -> None:
        """Load persisted state from file"""
        try:
            if self.config.state_file and self.config.state_file.exists():
                with open(self.config.state_file, 'r') as f:
                    state_data = json.load(f)
                
                # Restore server states (simplified for now)
                logger.debug(f"Loaded state from {self.config.state_file}")
            else:
                logger.debug("No state file found, starting fresh")
                
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
    
    async def _save_state(self) -> None:
        """Save current state to file"""
        try:
            if self.config.state_file:
                # Create directory if needed
                self.config.state_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Prepare state data (simplified for now)
                state_data = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "server_count": len(self._server_states),
                    "running_servers": [
                        name for name, state in self._server_states.items()
                        if state.status == ServerStatus.RUNNING
                    ]
                }
                
                with open(self.config.state_file, 'w') as f:
                    json.dump(state_data, f, indent=2)
                
                logger.debug(f"Saved state to {self.config.state_file}")
                
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    # Public API methods
    
    def get_server_states(self) -> Dict[str, ServerState]:
        """Get all server states"""
        return self._server_states.copy()
    
    def get_server_state(self, server_name: str) -> Optional[ServerState]:
        """Get state of a specific server"""
        return self._server_states.get(server_name)
    
    async def add_server(self, server_config: ServerConfig) -> Tuple[bool, Optional[str]]:
        """Add a new server configuration"""
        try:
            if server_config.name in self._server_states:
                return False, f"Server '{server_config.name}' already exists"
            
            # Validate configuration
            # This is automatically done by Pydantic
            
            # Create server state
            server_state = ServerState(
                config=server_config,
                status=ServerStatus.STOPPED,
                health=HealthCheckResult(status=HealthStatus.UNHEALTHY),
                metrics=ServerMetrics()
            )
            
            self._server_states[server_config.name] = server_state
            
            logger.info(f"Added server configuration: {server_config.name}")
            
            # Auto-start if enabled
            if server_config.enabled:
                await self.start_server(server_config.name)
            
            return True, None
            
        except ValidationError as e:
            return False, f"Configuration validation failed: {str(e)}"
        except Exception as e:
            return False, f"Error adding server: {str(e)}"
    
    async def remove_server(self, server_name: str) -> Tuple[bool, Optional[str]]:
        """Remove a server configuration"""
        if server_name not in self._server_states:
            return False, f"Server '{server_name}' not found"
        
        try:
            # Stop server first
            await self.stop_server(server_name)
            
            # Remove from state
            del self._server_states[server_name]
            
            logger.info(f"Removed server: {server_name}")
            return True, None
            
        except Exception as e:
            return False, f"Error removing server: {str(e)}"
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        total_servers = len(self._server_states)
        running_servers = sum(
            1 for state in self._server_states.values()
            if state.status == ServerStatus.RUNNING
        )
        healthy_servers = sum(
            1 for state in self._server_states.values()
            if state.is_healthy
        )
        
        return {
            "manager_running": self._is_running,
            "total_servers": total_servers,
            "running_servers": running_servers,
            "healthy_servers": healthy_servers,
            "monitoring_active": self.health_monitor.is_monitoring(),
            "active_connections": self.connection_manager.get_active_connections_count(),
            "health_summary": self.health_monitor.get_overall_health_summary(),
        }
    
    # Unified interface delegation
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any], **kwargs) -> Any:
        """Call a tool through the unified interface"""
        return await self.unified_interface.call_tool(tool_name, arguments, **kwargs)
    
    async def list_capabilities(self) -> Dict[str, Any]:
        """List all available capabilities"""
        return await self.unified_interface.get_capability_summary()
    
    def is_running(self) -> bool:
        """Check if the manager is running"""
        return self._is_running