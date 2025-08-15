#!/usr/bin/env python3
"""
MCP Central Orchestration Service

Master orchestrator that coordinates all MCP automation components including
update management, testing, cleanup, and monitoring. Provides unified control
plane with event-driven architecture and comprehensive lifecycle management.

Author: Claude AI Assistant (ai-agent-orchestrator)
Created: 2025-08-15 11:50:00 UTC
Version: 1.0.0
"""

import asyncio
import signal
import sys
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Set, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
from contextlib import asynccontextmanager

# Add parent path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import get_config, MCPAutomationConfig
from mcp_update_manager import MCPUpdateManager, UpdateStatus
from cleanup.cleanup_manager import CleanupManager, CleanupStatus
from version_manager import VersionManager

from .workflow_engine import WorkflowEngine, WorkflowDefinition, WorkflowStatus
from .service_registry import ServiceRegistry, ServiceInfo, ServiceStatus
from .event_manager import EventManager, Event, EventType, EventPriority
from .state_manager import StateManager, SystemState
from .policy_engine import PolicyEngine, PolicyViolation
from .api_gateway import create_api_app


class OrchestrationMode(Enum):
    """Orchestration operational modes."""
    AUTOMATIC = "automatic"
    SEMI_AUTOMATIC = "semi_automatic"
    MANUAL = "manual"
    MAINTENANCE = "maintenance"
    EMERGENCY = "emergency"


class OrchestratorStatus(Enum):
    """Orchestrator lifecycle status."""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"


@dataclass
class OrchestrationMetrics:
    """Orchestration performance metrics."""
    workflows_executed: int = 0
    workflows_succeeded: int = 0
    workflows_failed: int = 0
    events_processed: int = 0
    policy_violations: int = 0
    service_failures: int = 0
    average_workflow_time: float = 0.0
    uptime_seconds: float = 0.0
    last_error: Optional[str] = None
    last_success: Optional[datetime] = None


@dataclass
class OrchestrationContext:
    """Context for orchestration operations."""
    mode: OrchestrationMode
    user: Optional[str] = None
    source: str = "system"
    priority: int = 5
    metadata: Dict[str, Any] = field(default_factory=dict)
    dry_run: bool = False
    force: bool = False
    timeout: Optional[int] = None


class MCPOrchestrator:
    """
    Central orchestration service for MCP automation.
    
    Coordinates all MCP automation components with unified lifecycle
    management, event handling, and policy enforcement.
    """
    
    def __init__(
        self,
        config: Optional[MCPAutomationConfig] = None,
        mode: OrchestrationMode = OrchestrationMode.AUTOMATIC
    ):
        """Initialize orchestrator with configuration."""
        self.config = config or get_config()
        self.mode = mode
        self.status = OrchestratorStatus.INITIALIZING
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Core components
        self.state_manager = StateManager(config=self.config)
        self.event_manager = EventManager(config=self.config)
        self.service_registry = ServiceRegistry(config=self.config)
        self.workflow_engine = WorkflowEngine(config=self.config)
        self.policy_engine = PolicyEngine(config=self.config)
        
        # Automation components
        self.update_manager: Optional[MCPUpdateManager] = None
        self.cleanup_manager: Optional[CleanupManager] = None
        self.version_manager: Optional[VersionManager] = None
        
        # Metrics and monitoring
        self.metrics = OrchestrationMetrics()
        self.start_time = datetime.now(timezone.utc)
        
        # Control flags
        self._shutdown_event = asyncio.Event()
        self._pause_event = asyncio.Event()
        self._tasks: Set[asyncio.Task] = set()
        
        # API server
        self.api_app = None
        self.api_server = None
        
    def _setup_logging(self) -> logging.Logger:
        """Configure structured logging."""
        logger = logging.getLogger("mcp.orchestrator")
        logger.setLevel(self.config.log_level.value.upper())
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    async def initialize(self) -> None:
        """Initialize all orchestration components."""
        try:
            self.logger.info("Initializing MCP Orchestrator...")
            
            # Initialize state manager
            await self.state_manager.initialize()
            
            # Restore previous state if exists
            previous_state = await self.state_manager.get_state("orchestrator")
            if previous_state:
                self.logger.info(f"Restored previous state: {previous_state}")
                self.metrics = OrchestrationMetrics(**previous_state.get("metrics", {}))
            
            # Initialize event manager
            await self.event_manager.initialize()
            self.event_manager.subscribe(
                EventType.SYSTEM,
                self._handle_system_event
            )
            
            # Initialize service registry
            await self.service_registry.initialize()
            await self._discover_services()
            
            # Initialize workflow engine
            await self.workflow_engine.initialize()
            self.workflow_engine.set_event_handler(self.event_manager.publish)
            
            # Initialize policy engine
            await self.policy_engine.initialize()
            await self.policy_engine.load_policies()
            
            # Initialize automation components
            await self._initialize_automation_components()
            
            # Start API server
            await self._start_api_server()
            
            # Update status
            self.status = OrchestratorStatus.READY
            await self._publish_status_event()
            
            self.logger.info("MCP Orchestrator initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize orchestrator: {e}")
            self.status = OrchestratorStatus.ERROR
            self.metrics.last_error = str(e)
            raise
            
    async def _initialize_automation_components(self) -> None:
        """Initialize automation subsystems."""
        try:
            # Initialize update manager
            self.update_manager = MCPUpdateManager(config=self.config)
            await self.service_registry.register_service(
                ServiceInfo(
                    name="mcp-update-manager",
                    type="automation",
                    version="1.0.0",
                    endpoint="internal://update-manager",
                    capabilities=["update", "rollback", "version-check"],
                    status=ServiceStatus.READY
                )
            )
            
            # Initialize cleanup manager
            self.cleanup_manager = CleanupManager(config=self.config)
            await self.service_registry.register_service(
                ServiceInfo(
                    name="mcp-cleanup-manager",
                    type="automation",
                    version="1.0.0",
                    endpoint="internal://cleanup-manager",
                    capabilities=["cleanup", "retention", "audit"],
                    status=ServiceStatus.READY
                )
            )
            
            # Initialize version manager
            self.version_manager = VersionManager(config=self.config)
            
            self.logger.info("Automation components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize automation components: {e}")
            raise
            
    async def _discover_services(self) -> None:
        """Discover available MCP services."""
        try:
            # Load MCP configuration
            mcp_config_path = Path("/opt/sutazaiapp/.mcp.json")
            if mcp_config_path.exists():
                with open(mcp_config_path) as f:
                    mcp_config = json.load(f)
                    
                for name, server_config in mcp_config.get("mcpServers", {}).items():
                    service_info = ServiceInfo(
                        name=f"mcp-{name}",
                        type="mcp-server",
                        version="unknown",
                        endpoint=server_config.get("command", ""),
                        capabilities=[],
                        metadata=server_config,
                        status=ServiceStatus.UNKNOWN
                    )
                    await self.service_registry.register_service(service_info)
                    
            self.logger.info(f"Discovered {len(self.service_registry.services)} services")
            
        except Exception as e:
            self.logger.error(f"Service discovery failed: {e}")
            
    async def _start_api_server(self) -> None:
        """Start REST API server."""
        try:
            # Create FastAPI app with orchestrator reference
            self.api_app = create_api_app(orchestrator=self)
            
            # Start server in background
            import uvicorn
            config = uvicorn.Config(
                app=self.api_app,
                host="0.0.0.0",
                port=10500,  # MCP automation API port
                log_level=self.config.log_level.value.lower()
            )
            server = uvicorn.Server(config)
            
            # Run server in background task
            task = asyncio.create_task(server.serve())
            self._tasks.add(task)
            
            self.logger.info("API server started on port 10500")
            
        except Exception as e:
            self.logger.error(f"Failed to start API server: {e}")
            
    async def start(self) -> None:
        """Start orchestration service."""
        try:
            if self.status != OrchestratorStatus.READY:
                await self.initialize()
                
            self.logger.info(f"Starting orchestrator in {self.mode.value} mode")
            self.status = OrchestratorStatus.RUNNING
            
            # Start background tasks
            self._tasks.add(asyncio.create_task(self._monitor_services()))
            self._tasks.add(asyncio.create_task(self._process_workflows()))
            self._tasks.add(asyncio.create_task(self._handle_events()))
            self._tasks.add(asyncio.create_task(self._update_metrics()))
            
            # Publish startup event
            await self.event_manager.publish(
                Event(
                    type=EventType.SYSTEM,
                    source="orchestrator",
                    data={"status": "started", "mode": self.mode.value},
                    priority=EventPriority.HIGH
                )
            )
            
            # Wait for shutdown signal
            await self._shutdown_event.wait()
            
        except Exception as e:
            self.logger.error(f"Orchestrator error: {e}")
            self.status = OrchestratorStatus.ERROR
            self.metrics.last_error = str(e)
            raise
            
    async def stop(self) -> None:
        """Stop orchestration service gracefully."""
        try:
            self.logger.info("Stopping orchestrator...")
            self.status = OrchestratorStatus.STOPPING
            
            # Cancel background tasks
            for task in self._tasks:
                task.cancel()
                
            # Wait for tasks to complete
            await asyncio.gather(*self._tasks, return_exceptions=True)
            
            # Save state
            await self._save_state()
            
            # Shutdown components
            await self.workflow_engine.shutdown()
            await self.event_manager.shutdown()
            await self.service_registry.shutdown()
            await self.state_manager.shutdown()
            
            if self.update_manager:
                await self.update_manager.shutdown()
            if self.cleanup_manager:
                await self.cleanup_manager.shutdown()
                
            self.status = OrchestratorStatus.STOPPED
            self._shutdown_event.set()
            
            self.logger.info("Orchestrator stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            
    async def pause(self) -> None:
        """Pause orchestration operations."""
        self.logger.info("Pausing orchestrator...")
        self.status = OrchestratorStatus.PAUSED
        self._pause_event.set()
        await self._publish_status_event()
        
    async def resume(self) -> None:
        """Resume orchestration operations."""
        self.logger.info("Resuming orchestrator...")
        self.status = OrchestratorStatus.RUNNING
        self._pause_event.clear()
        await self._publish_status_event()
        
    async def execute_workflow(
        self,
        workflow_name: str,
        context: Optional[OrchestrationContext] = None
    ) -> Dict[str, Any]:
        """Execute a named workflow."""
        try:
            context = context or OrchestrationContext(
                mode=self.mode,
                source="orchestrator"
            )
            
            # Check policy compliance
            violations = await self.policy_engine.check_workflow(
                workflow_name,
                context.metadata
            )
            if violations and not context.force:
                raise PolicyViolation(f"Policy violations: {violations}")
                
            # Get workflow definition
            workflow = await self.workflow_engine.get_workflow(workflow_name)
            if not workflow:
                raise ValueError(f"Unknown workflow: {workflow_name}")
                
            # Execute workflow
            self.logger.info(f"Executing workflow: {workflow_name}")
            result = await self.workflow_engine.execute(
                workflow,
                context=asdict(context)
            )
            
            # Update metrics
            self.metrics.workflows_executed += 1
            if result.get("status") == WorkflowStatus.COMPLETED.value:
                self.metrics.workflows_succeeded += 1
                self.metrics.last_success = datetime.now(timezone.utc)
            else:
                self.metrics.workflows_failed += 1
                
            # Publish completion event
            await self.event_manager.publish(
                Event(
                    type=EventType.WORKFLOW,
                    source="orchestrator",
                    data={
                        "workflow": workflow_name,
                        "result": result,
                        "context": asdict(context)
                    }
                )
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            self.metrics.workflows_failed += 1
            self.metrics.last_error = str(e)
            raise
            
    async def trigger_update_check(
        self,
        servers: Optional[List[str]] = None,
        force: bool = False
    ) -> Dict[str, Any]:
        """Trigger MCP server update check."""
        try:
            if not self.update_manager:
                raise RuntimeError("Update manager not initialized")
                
            context = OrchestrationContext(
                mode=self.mode,
                source="orchestrator",
                metadata={"servers": servers, "force": force}
            )
            
            # Execute update workflow
            return await self.execute_workflow("mcp-update-check", context)
            
        except Exception as e:
            self.logger.error(f"Update check failed: {e}")
            raise
            
    async def trigger_cleanup(
        self,
        dry_run: bool = True,
        types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Trigger cleanup operation."""
        try:
            if not self.cleanup_manager:
                raise RuntimeError("Cleanup manager not initialized")
                
            context = OrchestrationContext(
                mode=self.mode,
                source="orchestrator",
                dry_run=dry_run,
                metadata={"types": types or ["all"]}
            )
            
            # Execute cleanup workflow
            return await self.execute_workflow("mcp-cleanup", context)
            
        except Exception as e:
            self.logger.error(f"Cleanup operation failed: {e}")
            raise
            
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        try:
            # Gather component statuses
            services = await self.service_registry.get_all_services()
            workflows = await self.workflow_engine.get_active_workflows()
            
            uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            self.metrics.uptime_seconds = uptime
            
            return {
                "orchestrator": {
                    "status": self.status.value,
                    "mode": self.mode.value,
                    "uptime_seconds": uptime,
                    "version": "1.0.0"
                },
                "services": {
                    "total": len(services),
                    "healthy": sum(1 for s in services if s.status == ServiceStatus.READY),
                    "unhealthy": sum(1 for s in services if s.status == ServiceStatus.ERROR),
                    "details": [asdict(s) for s in services]
                },
                "workflows": {
                    "active": len(workflows),
                    "details": workflows
                },
                "metrics": asdict(self.metrics),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            raise
            
    async def _monitor_services(self) -> None:
        """Monitor service health continuously."""
        while self.status == OrchestratorStatus.RUNNING:
            try:
                if not self._pause_event.is_set():
                    # Check all services
                    unhealthy = await self.service_registry.check_health()
                    
                    if unhealthy:
                        self.logger.warning(f"Unhealthy services detected: {unhealthy}")
                        
                        # Publish health alert
                        await self.event_manager.publish(
                            Event(
                                type=EventType.ALERT,
                                source="orchestrator",
                                data={"unhealthy_services": unhealthy},
                                priority=EventPriority.HIGH
                            )
                        )
                        
                        # Trigger recovery workflow if configured
                        if self.mode == OrchestrationMode.AUTOMATIC:
                            for service in unhealthy:
                                await self.execute_workflow(
                                    "service-recovery",
                                    OrchestrationContext(
                                        mode=self.mode,
                                        metadata={"service": service}
                                    )
                                )
                                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Service monitoring error: {e}")
                await asyncio.sleep(60)
                
    async def _process_workflows(self) -> None:
        """Process workflow queue continuously."""
        while self.status == OrchestratorStatus.RUNNING:
            try:
                if not self._pause_event.is_set():
                    # Process pending workflows
                    pending = await self.workflow_engine.get_pending_workflows()
                    
                    for workflow_id in pending:
                        # Check if we can execute
                        if await self.workflow_engine.can_execute(workflow_id):
                            asyncio.create_task(
                                self.workflow_engine.execute_by_id(workflow_id)
                            )
                            
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Workflow processing error: {e}")
                await asyncio.sleep(10)
                
    async def _handle_events(self) -> None:
        """Process events continuously."""
        while self.status == OrchestratorStatus.RUNNING:
            try:
                # Get next event
                event = await self.event_manager.get_next_event(timeout=5)
                
                if event and not self._pause_event.is_set():
                    self.metrics.events_processed += 1
                    
                    # Handle event based on type
                    if event.type == EventType.SYSTEM:
                        await self._handle_system_event(event)
                    elif event.type == EventType.WORKFLOW:
                        await self._handle_workflow_event(event)
                    elif event.type == EventType.ALERT:
                        await self._handle_alert_event(event)
                        
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Event handling error: {e}")
                
    async def _handle_system_event(self, event: Event) -> None:
        """Handle system events."""
        try:
            self.logger.debug(f"System event: {event.data}")
            
            # Handle specific system events
            if event.data.get("action") == "shutdown":
                await self.stop()
            elif event.data.get("action") == "pause":
                await self.pause()
            elif event.data.get("action") == "resume":
                await self.resume()
                
        except Exception as e:
            self.logger.error(f"System event handling error: {e}")
            
    async def _handle_workflow_event(self, event: Event) -> None:
        """Handle workflow events."""
        try:
            self.logger.debug(f"Workflow event: {event.data}")
            
            # Update workflow metrics
            if event.data.get("status") == "completed":
                workflow_time = event.data.get("execution_time", 0)
                self.metrics.average_workflow_time = (
                    (self.metrics.average_workflow_time * 
                     (self.metrics.workflows_executed - 1) + workflow_time) /
                    self.metrics.workflows_executed
                )
                
        except Exception as e:
            self.logger.error(f"Workflow event handling error: {e}")
            
    async def _handle_alert_event(self, event: Event) -> None:
        """Handle alert events."""
        try:
            self.logger.warning(f"Alert event: {event.data}")
            
            # Take action based on alert severity
            if event.priority == EventPriority.CRITICAL:
                # Trigger emergency response
                if self.mode == OrchestrationMode.AUTOMATIC:
                    await self.execute_workflow(
                        "emergency-response",
                        OrchestrationContext(
                            mode=OrchestrationMode.EMERGENCY,
                            metadata=event.data
                        )
                    )
                    
        except Exception as e:
            self.logger.error(f"Alert event handling error: {e}")
            
    async def _update_metrics(self) -> None:
        """Update metrics periodically."""
        while self.status == OrchestratorStatus.RUNNING:
            try:
                # Calculate uptime
                uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
                self.metrics.uptime_seconds = uptime
                
                # Save metrics to state
                await self.state_manager.set_state(
                    "orchestrator_metrics",
                    asdict(self.metrics)
                )
                
                await asyncio.sleep(60)  # Update every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Metrics update error: {e}")
                await asyncio.sleep(60)
                
    async def _save_state(self) -> None:
        """Save orchestrator state."""
        try:
            state = {
                "status": self.status.value,
                "mode": self.mode.value,
                "metrics": asdict(self.metrics),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            await self.state_manager.set_state("orchestrator", state)
            self.logger.debug("State saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")
            
    async def _publish_status_event(self) -> None:
        """Publish orchestrator status event."""
        await self.event_manager.publish(
            Event(
                type=EventType.SYSTEM,
                source="orchestrator",
                data={
                    "status": self.status.value,
                    "mode": self.mode.value,
                    "metrics": asdict(self.metrics)
                }
            )
        )
        
    def handle_signal(self, signum: int, frame: Any) -> None:
        """Handle system signals."""
        self.logger.info(f"Received signal {signum}")
        asyncio.create_task(self.stop())


async def main():
    """Main entry point for orchestrator."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="MCP Orchestration Service")
    parser.add_argument(
        "--mode",
        choices=["automatic", "semi_automatic", "manual", "maintenance"],
        default="automatic",
        help="Orchestration mode"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration file path"
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Create orchestrator
    mode = OrchestrationMode[args.mode.upper()]
    orchestrator = MCPOrchestrator(mode=mode)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, orchestrator.handle_signal)
    signal.signal(signal.SIGTERM, orchestrator.handle_signal)
    
    try:
        # Start orchestrator
        await orchestrator.start()
        
    except KeyboardInterrupt:
        print("\nShutdown requested...")
        await orchestrator.stop()
        
    except Exception as e:
        print(f"Orchestrator failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())