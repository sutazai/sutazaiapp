#!/usr/bin/env python3
"""
SutazAI Core System Implementation
Main system orchestration, configuration, and lifecycle management
"""

import asyncio
import json
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, Callable
from concurrent.futures import ThreadPoolExecutor
import psutil

# Core system imports
from .config import ConfigManager
from .lifecycle import SystemLifecycle
from .errors import SutazAIError, SystemError

# Component imports (will be implemented)
from ..models import ModelManager
from ..neural import NeuralProcessor
from ..agents import AgentOrchestrator
from ..knowledge import KnowledgeEngine
from ..security import SecurityManager
from ..monitoring import SystemMonitor
from ..utils import Logger, CacheManager, ValidationEngine

logger = logging.getLogger(__name__)

class SystemStatus(Enum):
    """System status enumeration"""
    INITIALIZING = "initializing"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"

class ComponentStatus(Enum):
    """Component status enumeration"""
    INACTIVE = "inactive"
    STARTING = "starting"
    ACTIVE = "active"
    ERROR = "error"
    STOPPING = "stopping"

@dataclass
class SystemMetrics:
    """System performance metrics"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    gpu_usage: float = 0.0
    network_in: float = 0.0
    network_out: float = 0.0
    active_models: int = 0
    active_agents: int = 0
    active_sessions: int = 0
    requests_per_second: float = 0.0
    average_response_time: float = 0.0
    error_rate: float = 0.0
    uptime: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class ComponentInfo:
    """Component information"""
    name: str
    version: str
    status: ComponentStatus
    config: Dict[str, Any]
    metrics: Dict[str, Any]
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class SystemConfig:
    """System configuration"""
    # Core settings
    system_name: str = "SutazAI"
    version: str = "1.0.0"
    environment: str = "production"
    debug_mode: bool = False
    
    # Directories
    base_dir: str = "/opt/sutazaiapp"
    data_dir: str = "/opt/sutazaiapp/data"
    models_dir: str = "/opt/sutazaiapp/models"
    logs_dir: str = "/opt/sutazaiapp/logs"
    cache_dir: str = "/opt/sutazaiapp/cache"
    
    # Component configurations
    models_config: Dict[str, Any] = field(default_factory=dict)
    neural_config: Dict[str, Any] = field(default_factory=dict)
    agents_config: Dict[str, Any] = field(default_factory=dict)
    knowledge_config: Dict[str, Any] = field(default_factory=dict)
    security_config: Dict[str, Any] = field(default_factory=dict)
    monitoring_config: Dict[str, Any] = field(default_factory=dict)
    
    # Performance settings
    max_workers: int = 4
    timeout_seconds: int = 300
    max_memory_mb: int = 8192
    enable_gpu: bool = True
    
    # Feature flags
    enable_neural_processing: bool = True
    enable_agent_orchestration: bool = True
    enable_knowledge_management: bool = True
    enable_web_learning: bool = True
    enable_self_modification: bool = False
    
    # Security settings
    enable_security: bool = True
    enable_audit: bool = True
    enable_encryption: bool = True
    
    # Monitoring settings
    enable_monitoring: bool = True
    metrics_interval: int = 60
    health_check_interval: int = 30

class SutazAICore:
    """
    Main SutazAI system orchestrator
    Manages all system components and provides unified interface
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[SystemConfig] = None):
        """Initialize SutazAI Core System"""
        self.config = config or SystemConfig()
        self.config_path = config_path
        self.status = SystemStatus.INITIALIZING
        self.start_time = datetime.now(timezone.utc)
        
        # System components
        self.components: Dict[str, Any] = {}
        self.component_status: Dict[str, ComponentStatus] = {}
        self.component_info: Dict[str, ComponentInfo] = {}
        
        # Core utilities
        self.logger = Logger(log_dir=self.config.logs_dir)
        self.config_manager = ConfigManager(config_path)
        self.cache_manager = CacheManager(cache_dir=self.config.cache_dir)
        self.validation_engine = ValidationEngine()
        
        # Performance monitoring
        self.metrics = SystemMetrics()
        self.metrics_history: List[SystemMetrics] = []
        
        # Thread safety
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        
        # Event system
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Initialize system
        self._initialize_system()
        
        logger.info(f"SutazAI Core System initialized - Version {self.config.version}")
    
    def _initialize_system(self):
        """Initialize core system components"""
        try:
            # Load configuration
            if self.config_path:
                self.config = self.config_manager.load_config(self.config_path)
            
            # Create directories
            self._create_directories()
            
            # Initialize components
            self._initialize_components()
            
            # Setup event handlers
            self._setup_event_handlers()
            
            # Start background tasks
            self._start_background_tasks()
            
            self.status = SystemStatus.STOPPED
            logger.info("System initialization completed")
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            self.status = SystemStatus.ERROR
            raise SystemError(f"Failed to initialize system: {e}")
    
    def _create_directories(self):
        """Create required directories"""
        directories = [
            self.config.data_dir,
            self.config.models_dir,
            self.config.logs_dir,
            self.config.cache_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _initialize_components(self):
        """Initialize system components"""
        try:
            # Model Management
            self.components["models"] = ModelManager(
                config=self.config.models_config,
                storage_dir=self.config.models_dir
            )
            self.component_status["models"] = ComponentStatus.INACTIVE
            
            # Neural Processing
            if self.config.enable_neural_processing:
                self.components["neural"] = NeuralProcessor(
                    config=self.config.neural_config
                )
                self.component_status["neural"] = ComponentStatus.INACTIVE
            
            # Agent Orchestration
            if self.config.enable_agent_orchestration:
                self.components["agents"] = AgentOrchestrator(
                    config=self.config.agents_config
                )
                self.component_status["agents"] = ComponentStatus.INACTIVE
            
            # Knowledge Management
            if self.config.enable_knowledge_management:
                self.components["knowledge"] = KnowledgeEngine(
                    config=self.config.knowledge_config,
                    data_dir=self.config.data_dir
                )
                self.component_status["knowledge"] = ComponentStatus.INACTIVE
            
            # Security Manager
            if self.config.enable_security:
                self.components["security"] = SecurityManager(
                    config=self.config.security_config
                )
                self.component_status["security"] = ComponentStatus.INACTIVE
            
            # System Monitor
            if self.config.enable_monitoring:
                self.components["monitor"] = SystemMonitor(
                    config=self.config.monitoring_config
                )
                self.component_status["monitor"] = ComponentStatus.INACTIVE
            
            # Update component info
            self._update_component_info()
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise SystemError(f"Failed to initialize components: {e}")
    
    def _setup_event_handlers(self):
        """Setup system event handlers"""
        self.event_handlers = {
            "system_start": [],
            "system_stop": [],
            "component_start": [],
            "component_stop": [],
            "error": [],
            "metrics_update": []
        }
    
    def _start_background_tasks(self):
        """Start background monitoring tasks"""
        # Metrics collection
        self._executor.submit(self._metrics_collector)
        
        # Health monitoring
        self._executor.submit(self._health_monitor)
        
        # Component status monitor
        self._executor.submit(self._component_monitor)
    
    async def start(self) -> bool:
        """Start the SutazAI system"""
        try:
            if self.status != SystemStatus.STOPPED:
                logger.warning(f"System already running or in transition state: {self.status}")
                return False
            
            logger.info("Starting SutazAI system...")
            self.status = SystemStatus.STARTING
            
            # Fire start event
            await self._fire_event("system_start")
            
            # Start components in order
            start_order = ["security", "models", "neural", "knowledge", "agents", "monitor"]
            
            for component_name in start_order:
                if component_name in self.components:
                    await self._start_component(component_name)
            
            # Verify all components are running
            if not await self._verify_system_health():
                raise SystemError("System health check failed")
            
            self.status = SystemStatus.RUNNING
            logger.info("SutazAI system started successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            self.status = SystemStatus.ERROR
            await self._fire_event("error", {"error": str(e)})
            return False
    
    async def stop(self) -> bool:
        """Stop the SutazAI system"""
        try:
            if self.status == SystemStatus.STOPPED:
                logger.info("System already stopped")
                return True
            
            logger.info("Stopping SutazAI system...")
            self.status = SystemStatus.STOPPING
            
            # Fire stop event
            await self._fire_event("system_stop")
            
            # Stop components in reverse order
            stop_order = ["monitor", "agents", "knowledge", "neural", "models", "security"]
            
            for component_name in stop_order:
                if component_name in self.components:
                    await self._stop_component(component_name)
            
            # Signal shutdown
            self._shutdown_event.set()
            
            self.status = SystemStatus.STOPPED
            logger.info("SutazAI system stopped successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop system: {e}")
            self.status = SystemStatus.ERROR
            return False
    
    async def restart(self) -> bool:
        """Restart the SutazAI system"""
        try:
            logger.info("Restarting SutazAI system...")
            
            if not await self.stop():
                return False
            
            # Brief pause before restart
            await asyncio.sleep(2)
            
            return await self.start()
            
        except Exception as e:
            logger.error(f"Failed to restart system: {e}")
            return False
    
    async def pause(self) -> bool:
        """Pause the SutazAI system"""
        try:
            if self.status != SystemStatus.RUNNING:
                logger.warning("System not running, cannot pause")
                return False
            
            logger.info("Pausing SutazAI system...")
            self.status = SystemStatus.PAUSED
            
            # Pause components
            for component_name in self.components:
                component = self.components[component_name]
                if hasattr(component, 'pause'):
                    await component.pause()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to pause system: {e}")
            return False
    
    async def resume(self) -> bool:
        """Resume the SutazAI system"""
        try:
            if self.status != SystemStatus.PAUSED:
                logger.warning("System not paused, cannot resume")
                return False
            
            logger.info("Resuming SutazAI system...")
            
            # Resume components
            for component_name in self.components:
                component = self.components[component_name]
                if hasattr(component, 'resume'):
                    await component.resume()
            
            self.status = SystemStatus.RUNNING
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume system: {e}")
            return False
    
    async def _start_component(self, component_name: str) -> bool:
        """Start a specific component"""
        try:
            if component_name not in self.components:
                logger.error(f"Component not found: {component_name}")
                return False
            
            logger.info(f"Starting component: {component_name}")
            self.component_status[component_name] = ComponentStatus.STARTING
            
            component = self.components[component_name]
            
            # Start component
            if hasattr(component, 'start'):
                await component.start()
            elif hasattr(component, 'initialize'):
                await component.initialize()
            
            self.component_status[component_name] = ComponentStatus.ACTIVE
            
            # Fire component start event
            await self._fire_event("component_start", {"component": component_name})
            
            logger.info(f"Component started: {component_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start component {component_name}: {e}")
            self.component_status[component_name] = ComponentStatus.ERROR
            await self._fire_event("error", {"component": component_name, "error": str(e)})
            return False
    
    async def _stop_component(self, component_name: str) -> bool:
        """Stop a specific component"""
        try:
            if component_name not in self.components:
                logger.error(f"Component not found: {component_name}")
                return False
            
            logger.info(f"Stopping component: {component_name}")
            self.component_status[component_name] = ComponentStatus.STOPPING
            
            component = self.components[component_name]
            
            # Stop component
            if hasattr(component, 'stop'):
                await component.stop()
            elif hasattr(component, 'shutdown'):
                await component.shutdown()
            
            self.component_status[component_name] = ComponentStatus.INACTIVE
            
            # Fire component stop event
            await self._fire_event("component_stop", {"component": component_name})
            
            logger.info(f"Component stopped: {component_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop component {component_name}: {e}")
            self.component_status[component_name] = ComponentStatus.ERROR
            return False
    
    async def _verify_system_health(self) -> bool:
        """Verify system health"""
        try:
            # Check component status
            for component_name, status in self.component_status.items():
                if status == ComponentStatus.ERROR:
                    logger.error(f"Component {component_name} in error state")
                    return False
            
            # Check system resources
            if self.metrics.memory_usage > 90:
                logger.warning(f"High memory usage: {self.metrics.memory_usage}%")
            
            if self.metrics.cpu_usage > 90:
                logger.warning(f"High CPU usage: {self.metrics.cpu_usage}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def _metrics_collector(self):
        """Background metrics collection"""
        while not self._shutdown_event.is_set():
            try:
                # Collect system metrics
                self.metrics.cpu_usage = psutil.cpu_percent(interval=1)
                self.metrics.memory_usage = psutil.virtual_memory().percent
                self.metrics.disk_usage = psutil.disk_usage('/').percent
                
                # Network metrics
                net_io = psutil.net_io_counters()
                self.metrics.network_in = net_io.bytes_recv
                self.metrics.network_out = net_io.bytes_sent
                
                # System uptime
                self.metrics.uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
                
                # Component metrics
                self.metrics.active_models = len([c for c in self.components.values() 
                                                if hasattr(c, 'get_loaded_models')])
                
                # Store metrics history
                self.metrics_history.append(self.metrics)
                
                # Keep only last 1000 metrics
                if len(self.metrics_history) > 1000:
                    self.metrics_history.pop(0)
                
                # Fire metrics update event
                asyncio.create_task(self._fire_event("metrics_update", {"metrics": self.metrics}))
                
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
            
            time.sleep(self.config.metrics_interval)
    
    def _health_monitor(self):
        """Background health monitoring"""
        while not self._shutdown_event.is_set():
            try:
                # Check component health
                for component_name, component in self.components.items():
                    if hasattr(component, 'health_check'):
                        health = component.health_check()
                        if not health:
                            logger.warning(f"Component {component_name} health check failed")
                            self.component_status[component_name] = ComponentStatus.ERROR
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
            
            time.sleep(self.config.health_check_interval)
    
    def _component_monitor(self):
        """Background component status monitoring"""
        while not self._shutdown_event.is_set():
            try:
                self._update_component_info()
                
            except Exception as e:
                logger.error(f"Component monitoring error: {e}")
            
            time.sleep(30)  # Update every 30 seconds
    
    def _update_component_info(self):
        """Update component information"""
        for component_name, component in self.components.items():
            try:
                info = ComponentInfo(
                    name=component_name,
                    version=getattr(component, 'version', '1.0.0'),
                    status=self.component_status.get(component_name, ComponentStatus.INACTIVE),
                    config=getattr(component, 'config', {}),
                    metrics=getattr(component, 'metrics', {})
                )
                self.component_info[component_name] = info
                
            except Exception as e:
                logger.error(f"Error updating component info for {component_name}: {e}")
    
    async def _fire_event(self, event_type: str, data: Dict[str, Any] = None):
        """Fire system event"""
        try:
            if event_type in self.event_handlers:
                for handler in self.event_handlers[event_type]:
                    await handler(data or {})
        except Exception as e:
            logger.error(f"Event handling error: {e}")
    
    def add_event_handler(self, event_type: str, handler: Callable):
        """Add event handler"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def remove_event_handler(self, event_type: str, handler: Callable):
        """Remove event handler"""
        if event_type in self.event_handlers:
            try:
                self.event_handlers[event_type].remove(handler)
            except ValueError:
                pass
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "status": self.status.value,
            "uptime": self.metrics.uptime,
            "components": {name: status.value for name, status in self.component_status.items()},
            "metrics": self.metrics,
            "version": self.config.version,
            "environment": self.config.environment
        }
    
    def get_component_info(self, component_name: str = None) -> Union[ComponentInfo, Dict[str, ComponentInfo]]:
        """Get component information"""
        if component_name:
            return self.component_info.get(component_name)
        return self.component_info
    
    def get_metrics_history(self, limit: int = 100) -> List[SystemMetrics]:
        """Get metrics history"""
        return self.metrics_history[-limit:]
    
    async def execute_command(self, command: str, **kwargs) -> Any:
        """Execute system command"""
        try:
            # Route commands to appropriate components
            if command.startswith("model."):
                return await self.components["models"].execute_command(command[6:], **kwargs)
            elif command.startswith("agent."):
                return await self.components["agents"].execute_command(command[6:], **kwargs)
            elif command.startswith("knowledge."):
                return await self.components["knowledge"].execute_command(command[10:], **kwargs)
            elif command.startswith("neural."):
                return await self.components["neural"].execute_command(command[7:], **kwargs)
            else:
                # System commands
                if command == "status":
                    return self.get_system_status()
                elif command == "restart":
                    return await self.restart()
                elif command == "pause":
                    return await self.pause()
                elif command == "resume":
                    return await self.resume()
                else:
                    raise ValueError(f"Unknown command: {command}")
                    
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise SystemError(f"Failed to execute command '{command}': {e}")
    
    def __del__(self):
        """Cleanup system resources"""
        try:
            if hasattr(self, '_shutdown_event'):
                self._shutdown_event.set()
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# Factory function
def create_sutazai_system(config_path: str = None) -> SutazAICore:
    """Create SutazAI system instance"""
    return SutazAICore(config_path=config_path)