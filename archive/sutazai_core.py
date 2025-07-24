#!/usr/bin/env python3
"""
SutazAI Core System
Enhanced system orchestration and management for the SutazAI AGI/ASI platform
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import threading
import psutil

# Import existing components
from .ai_agents.model_manager import ModelManager
from .ai_agents.agent_framework import AgentFramework
from .ai_agents.agent_manager import AgentManager
from .ethics.ethical_verifier import EthicalVerifier
from .sandbox.code_sandbox import CodeSandbox
from .vector_db import VectorDB
from .monitoring.monitoring import SystemMonitor
from .utils.logging_setup import get_api_logger
from .neural_engine import NeuralProcessor, create_neural_processor

logger = get_api_logger()

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
    base_dir: str = "/opt/sutazaiapp/backend"
    data_dir: str = "/opt/sutazaiapp/backend/data"
    models_dir: str = "/opt/sutazaiapp/backend/data/models"
    logs_dir: str = "/opt/sutazaiapp/backend/logs"
    cache_dir: str = "/opt/sutazaiapp/backend/data/cache"
    config_dir: str = "/opt/sutazaiapp/backend/config"
    
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
    Enhanced SutazAI system orchestrator
    Manages all system components and provides unified interface
    """
    
    def __init__(self, config: Optional[SystemConfig] = None):
        """Initialize SutazAI Core System"""
        self.config = config or SystemConfig()
        self.status = SystemStatus.INITIALIZING
        self.start_time = datetime.now(timezone.utc)
        
        # System components
        self.components: Dict[str, Any] = {}
        self.component_status: Dict[str, ComponentStatus] = {}
        self.component_info: Dict[str, ComponentInfo] = {}
        
        # Performance monitoring
        self.metrics = SystemMetrics()
        self.metrics_history: List[SystemMetrics] = []
        
        # Thread safety
        self._lock = threading.RLock()
        self._shutdown_event = threading.Event()
        
        # Event system
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Initialize system
        self._initialize_system()
        
        logger.info(f"SutazAI Core System initialized - Version {self.config.version}")
    
    def _initialize_system(self):
        """Initialize core system components"""
        try:
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
            raise RuntimeError(f"Failed to initialize system: {e}")
    
    def _create_directories(self):
        """Create required directories"""
        directories = [
            self.config.data_dir,
            self.config.models_dir,
            self.config.logs_dir,
            self.config.cache_dir,
            self.config.config_dir
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def _initialize_components(self):
        """Initialize system components"""
        try:
            # Model Management
            self.components["model_manager"] = ModelManager(
                models_dir=self.config.models_dir,
                config_path=f"{self.config.config_dir}/models.json",
                cache_dir=f"{self.config.cache_dir}/models",
                max_memory_gb=self.config.max_memory_mb // 1024,
                device="auto"
            )
            self.component_status["model_manager"] = ComponentStatus.INACTIVE
            
            # Agent Framework
            if self.config.enable_agent_orchestration:
                self.components["agent_framework"] = AgentFramework(
                    model_manager=self.components["model_manager"]
                )
                self.component_status["agent_framework"] = ComponentStatus.INACTIVE
                
                # Agent Manager
                self.components["agent_manager"] = AgentManager(
                    agent_framework=self.components["agent_framework"]
                )
                self.component_status["agent_manager"] = ComponentStatus.INACTIVE
            
            # Ethical Verifier
            if self.config.enable_security:
                self.components["ethical_verifier"] = EthicalVerifier()
                self.component_status["ethical_verifier"] = ComponentStatus.INACTIVE
            
            # Code Sandbox
            self.components["code_sandbox"] = CodeSandbox()
            self.component_status["code_sandbox"] = ComponentStatus.INACTIVE
            
            # Neural Processing Engine
            if self.config.enable_neural_processing:
                self.components["neural_processor"] = create_neural_processor(
                    config=getattr(self.config, 'neural_config', {})
                )
                self.component_status["neural_processor"] = ComponentStatus.INACTIVE
            
            # Vector Database
            if self.config.enable_knowledge_management:
                self.components["vector_db"] = VectorDB(
                    storage_path=f"{self.config.data_dir}/vectors"
                )
                self.component_status["vector_db"] = ComponentStatus.INACTIVE
            
            # Knowledge Graph Engine
            if self.config.enable_knowledge_management:
                from knowledge_graph import KnowledgeGraphEngine
                self.components["knowledge_graph"] = KnowledgeGraphEngine(
                    storage_path=f"{self.config.data_dir}/knowledge_graph.db"
                )
                self.component_status["knowledge_graph"] = ComponentStatus.INACTIVE
            
            # Self-Evolution Engine
            if getattr(self.config, 'enable_self_evolution', True):
                from self_evolution import SelfEvolutionEngine
                self.components["self_evolution"] = SelfEvolutionEngine(
                    workspace_path=f"{self.config.data_dir}/evolution"
                )
                self.component_status["self_evolution"] = ComponentStatus.INACTIVE
            
            # Web Learning Pipeline
            if self.config.enable_web_learning:
                from web_learning import LearningPipeline
                self.components["web_learning"] = LearningPipeline()
                self.component_status["web_learning"] = ComponentStatus.INACTIVE
            
            # System Monitor
            if self.config.enable_monitoring:
                self.components["system_monitor"] = SystemMonitor()
                self.component_status["system_monitor"] = ComponentStatus.INACTIVE
            
            # Update component info
            self._update_component_info()
            
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            raise RuntimeError(f"Failed to initialize components: {e}")
    
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
        # Start metrics collection in background thread
        def metrics_thread():
            while not self._shutdown_event.is_set():
                try:
                    self._collect_metrics()
                    time.sleep(self.config.metrics_interval)
                except Exception as e:
                    logger.error(f"Metrics collection error: {e}")
                    time.sleep(60)  # Wait before retrying
        
        # Start health monitoring in background thread
        def health_thread():
            while not self._shutdown_event.is_set():
                try:
                    self._check_health()
                    time.sleep(self.config.health_check_interval)
                except Exception as e:
                    logger.error(f"Health check error: {e}")
                    time.sleep(60)  # Wait before retrying
        
        threading.Thread(target=metrics_thread, daemon=True).start()
        threading.Thread(target=health_thread, daemon=True).start()
    
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
            
            # Start components in dependency order
            start_order = [
                "model_manager",
                "neural_processor",
                "vector_db", 
                "knowledge_graph",
                "self_evolution",
                "web_learning",
                "ethical_verifier",
                "code_sandbox",
                "agent_framework",
                "agent_manager",
                "system_monitor"
            ]
            
            for component_name in start_order:
                if component_name in self.components:
                    await self._start_component(component_name)
            
            # Verify system health
            if not await self._verify_system_health():
                raise RuntimeError("System health check failed")
            
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
            stop_order = [
                "system_monitor",
                "agent_manager",
                "agent_framework",
                "code_sandbox",
                "ethical_verifier",
                "web_learning",
                "self_evolution",
                "knowledge_graph",
                "vector_db",
                "neural_processor",
                "model_manager"
            ]
            
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
    
    async def _start_component(self, component_name: str) -> bool:
        """Start a specific component"""
        try:
            if component_name not in self.components:
                logger.error(f"Component not found: {component_name}")
                return False
            
            logger.info(f"Starting component: {component_name}")
            self.component_status[component_name] = ComponentStatus.STARTING
            
            component = self.components[component_name]
            
            # Start component based on its interface
            if hasattr(component, 'start'):
                if asyncio.iscoroutinefunction(component.start):
                    await component.start()
                else:
                    component.start()
            elif hasattr(component, 'initialize'):
                if asyncio.iscoroutinefunction(component.initialize):
                    await component.initialize()
                else:
                    component.initialize()
            
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
            
            # Stop component based on its interface
            if hasattr(component, 'stop'):
                if asyncio.iscoroutinefunction(component.stop):
                    await component.stop()
                else:
                    component.stop()
            elif hasattr(component, 'cleanup'):
                if asyncio.iscoroutinefunction(component.cleanup):
                    await component.cleanup()
                else:
                    component.cleanup()
            elif hasattr(component, 'shutdown'):
                if asyncio.iscoroutinefunction(component.shutdown):
                    await component.shutdown()
                else:
                    component.shutdown()
            
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
            if self.metrics.memory_usage > 95:
                logger.error(f"Critical memory usage: {self.metrics.memory_usage}%")
                return False
            
            if self.metrics.cpu_usage > 95:
                logger.error(f"Critical CPU usage: {self.metrics.cpu_usage}%")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    def _collect_metrics(self):
        """Collect system metrics"""
        try:
            # System metrics
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
            if "model_manager" in self.components:
                try:
                    model_manager = self.components["model_manager"]
                    self.metrics.active_models = len(model_manager.loaded_models)
                except Exception as e:
                    logger.warning(f"Failed to get model metrics: {e}")
            
            if "agent_manager" in self.components:
                try:
                    agent_manager = self.components["agent_manager"]
                    if hasattr(agent_manager, 'get_active_agents'):
                        self.metrics.active_agents = len(agent_manager.get_active_agents())
                except Exception as e:
                    logger.warning(f"Failed to get agent metrics: {e}")
            
            # Store metrics history
            self.metrics.timestamp = datetime.now(timezone.utc)
            self.metrics_history.append(self.metrics)
            
            # Keep only last 1000 metrics
            if len(self.metrics_history) > 1000:
                self.metrics_history.pop(0)
            
        except Exception as e:
            logger.error(f"Metrics collection error: {e}")
    
    def _check_health(self):
        """Check system health"""
        try:
            # Check component health
            for component_name, component in self.components.items():
                if hasattr(component, 'health_check'):
                    try:
                        health = component.health_check()
                        if not health:
                            logger.warning(f"Component {component_name} health check failed")
                            self.component_status[component_name] = ComponentStatus.ERROR
                    except Exception as e:
                        logger.warning(f"Health check error for {component_name}: {e}")
                        self.component_status[component_name] = ComponentStatus.ERROR
            
            # Update component info
            self._update_component_info()
            
        except Exception as e:
            logger.error(f"Health monitoring error: {e}")
    
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
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data or {})
                    else:
                        handler(data or {})
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
            "metrics": {
                "cpu_usage": self.metrics.cpu_usage,
                "memory_usage": self.metrics.memory_usage,
                "disk_usage": self.metrics.disk_usage,
                "active_models": self.metrics.active_models,
                "active_agents": self.metrics.active_agents,
                "uptime": self.metrics.uptime
            },
            "version": self.config.version,
            "environment": self.config.environment
        }
    
    def get_component_info(self, component_name: str = None) -> Union[ComponentInfo, Dict[str, ComponentInfo]]:
        """Get component information"""
        if component_name:
            return self.component_info.get(component_name)
        return self.component_info
    
    def get_metrics_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get metrics history"""
        history = self.metrics_history[-limit:]
        return [
            {
                "timestamp": m.timestamp.isoformat(),
                "cpu_usage": m.cpu_usage,
                "memory_usage": m.memory_usage,
                "disk_usage": m.disk_usage,
                "active_models": m.active_models,
                "active_agents": m.active_agents,
                "uptime": m.uptime
            }
            for m in history
        ]
    
    async def execute_command(self, command: str, **kwargs) -> Any:
        """Execute system command"""
        try:
            # Route commands to appropriate components
            if command.startswith("model."):
                return await self._execute_model_command(command[6:], **kwargs)
            elif command.startswith("agent."):
                return await self._execute_agent_command(command[6:], **kwargs)
            elif command.startswith("neural."):
                return await self._execute_neural_command(command[7:], **kwargs)
            elif command.startswith("system."):
                return await self._execute_system_command(command[7:], **kwargs)
            else:
                # Direct system commands
                if command == "status":
                    return self.get_system_status()
                elif command == "restart":
                    return await self.restart()
                elif command == "start":
                    return await self.start()
                elif command == "stop":
                    return await self.stop()
                else:
                    raise ValueError(f"Unknown command: {command}")
                    
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise RuntimeError(f"Failed to execute command '{command}': {e}")
    
    async def _execute_model_command(self, command: str, **kwargs) -> Any:
        """Execute model-related command"""
        if "model_manager" not in self.components:
            raise RuntimeError("Model manager not available")
        
        model_manager = self.components["model_manager"]
        
        if command == "list":
            return model_manager.list_models()
        elif command == "load":
            model_name = kwargs.get("model_name")
            if not model_name:
                raise ValueError("model_name required")
            return await model_manager.load_model(model_name)
        elif command == "unload":
            model_name = kwargs.get("model_name")
            if not model_name:
                raise ValueError("model_name required")
            return await model_manager.unload_model(model_name)
        elif command == "status":
            return model_manager.get_model_status()
        else:
            raise ValueError(f"Unknown model command: {command}")
    
    async def _execute_agent_command(self, command: str, **kwargs) -> Any:
        """Execute agent-related command"""
        if "agent_framework" not in self.components:
            raise RuntimeError("Agent framework not available")
        
        agent_framework = self.components["agent_framework"]
        
        if command == "list":
            return agent_framework.list_agents()
        elif command == "create":
            agent_type = kwargs.get("agent_type")
            if not agent_type:
                raise ValueError("agent_type required")
            return await agent_framework.create_agent(agent_type)
        elif command == "execute":
            instance_id = kwargs.get("instance_id")
            task = kwargs.get("task")
            if not instance_id or not task:
                raise ValueError("instance_id and task required")
            return await agent_framework.execute_task(instance_id, task)
        else:
            raise ValueError(f"Unknown agent command: {command}")
    
    async def _execute_neural_command(self, command: str, **kwargs) -> Any:
        """Execute neural processing command"""
        if "neural_processor" not in self.components:
            raise RuntimeError("Neural processor not available")
        
        neural_processor = self.components["neural_processor"]
        
        if command == "process":
            input_data = kwargs.get("input_data")
            if input_data is None:
                raise ValueError("input_data required for neural processing")
            return await neural_processor.process_input(input_data)
        elif command == "train":
            training_data = kwargs.get("training_data")
            if training_data is None:
                raise ValueError("training_data required for neural training")
            return await neural_processor.train(training_data)
        elif command == "status":
            return neural_processor.get_status()
        elif command == "metrics":
            return neural_processor.get_metrics()
        else:
            raise ValueError(f"Unknown neural command: {command}")
    
    async def _execute_system_command(self, command: str, **kwargs) -> Any:
        """Execute system-related command"""
        if command == "health":
            return await self._verify_system_health()
        elif command == "metrics":
            return self.get_metrics_history(kwargs.get("limit", 100))
        elif command == "config":
            return self.config.__dict__
        else:
            raise ValueError(f"Unknown system command: {command}")
    
    def get_component(self, name: str) -> Any:
        """Get a specific component"""
        return self.components.get(name)
    
    def __del__(self):
        """Cleanup system resources"""
        try:
            if hasattr(self, '_shutdown_event'):
                self._shutdown_event.set()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")

# Factory function
def create_sutazai_system(config: Optional[SystemConfig] = None) -> SutazAICore:
    """Create SutazAI system instance"""
    return SutazAICore(config=config)

# Global system instance
_system_instance: Optional[SutazAICore] = None

def get_system_instance() -> Optional[SutazAICore]:
    """Get the global system instance"""
    return _system_instance

def initialize_system(config: Optional[SystemConfig] = None) -> SutazAICore:
    """Initialize the global system instance"""
    global _system_instance
    if _system_instance is None:
        _system_instance = create_sutazai_system(config)
    return _system_instance