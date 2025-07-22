#!/usr/bin/env python3
"""
System Manager for SutazAI
Provides centralized system management and integration with FastAPI
"""

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict

# Import core system
from .sutazai_core import SutazAICore, SystemConfig, SystemStatus, ComponentStatus
from .utils.logging_setup import get_api_logger

logger = get_api_logger()

@dataclass
class SystemStats:
    """System statistics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    uptime_seconds: int = 0
    models_loaded: int = 0
    agents_active: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    disk_usage_percent: float = 0.0

class SystemManager:
    """
    Centralized system manager for SutazAI
    Integrates with FastAPI and provides unified system management
    """
    
    def __init__(self, config_path: str = None):
        """Initialize system manager"""
        self.config_path = config_path or "/opt/sutazaiapp/backend/config/system.json"
        self.config = self._load_config()
        
        # Initialize core system
        self.core = SutazAICore(config=self.config)
        
        # System statistics
        self.stats = SystemStats()
        
        # Request tracking
        self.request_times: List[float] = []
        self.max_request_history = 1000
        
        logger.info("SystemManager initialized")
    
    def _load_config(self) -> SystemConfig:
        """Load system configuration"""
        try:
            config_file = Path(self.config_path)
            if not config_file.exists():
                logger.warning(f"Config file not found: {self.config_path}, using defaults")
                return SystemConfig()
            
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Convert config data to SystemConfig
            return SystemConfig(
                system_name=config_data.get("system_name", "SutazAI"),
                version=config_data.get("version", "1.0.0"),
                environment=config_data.get("environment", "production"),
                debug_mode=config_data.get("debug_mode", False),
                
                # Directories
                base_dir=config_data.get("directories", {}).get("base_dir", "/opt/sutazaiapp/backend"),
                data_dir=config_data.get("directories", {}).get("data_dir", "/opt/sutazaiapp/backend/data"),
                models_dir=config_data.get("directories", {}).get("models_dir", "/opt/sutazaiapp/backend/data/models"),
                logs_dir=config_data.get("directories", {}).get("logs_dir", "/opt/sutazaiapp/backend/logs"),
                cache_dir=config_data.get("directories", {}).get("cache_dir", "/opt/sutazaiapp/backend/data/cache"),
                config_dir=config_data.get("directories", {}).get("config_dir", "/opt/sutazaiapp/backend/config"),
                
                # Performance
                max_workers=config_data.get("performance", {}).get("max_workers", 4),
                timeout_seconds=config_data.get("performance", {}).get("timeout_seconds", 300),
                max_memory_mb=config_data.get("performance", {}).get("max_memory_mb", 8192),
                enable_gpu=config_data.get("performance", {}).get("enable_gpu", True),
                
                # Features
                enable_neural_processing=config_data.get("features", {}).get("enable_neural_processing", True),
                enable_agent_orchestration=config_data.get("features", {}).get("enable_agent_orchestration", True),
                enable_knowledge_management=config_data.get("features", {}).get("enable_knowledge_management", True),
                enable_web_learning=config_data.get("features", {}).get("enable_web_learning", True),
                enable_self_modification=config_data.get("features", {}).get("enable_self_modification", False),
                
                # Security
                enable_security=config_data.get("security", {}).get("enable_security", True),
                enable_audit=config_data.get("security", {}).get("enable_audit", True),
                enable_encryption=config_data.get("security", {}).get("enable_encryption", True),
                
                # Monitoring
                enable_monitoring=config_data.get("monitoring", {}).get("enable_monitoring", True),
                metrics_interval=config_data.get("monitoring", {}).get("metrics_interval", 60),
                health_check_interval=config_data.get("monitoring", {}).get("health_check_interval", 30)
            )
            
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return SystemConfig()
    
    async def initialize(self) -> bool:
        """Initialize the system"""
        try:
            logger.info("Initializing SutazAI system...")
            
            # Start the core system
            success = await self.core.start()
            if success:
                logger.info("SutazAI system initialized successfully")
                
                # Setup event handlers
                self._setup_event_handlers()
                
                return True
            else:
                logger.error("Failed to initialize SutazAI system")
                return False
                
        except Exception as e:
            logger.error(f"System initialization error: {e}")
            return False
    
    def _setup_event_handlers(self):
        """Setup system event handlers"""
        # Request tracking
        self.core.add_event_handler("request_start", self._on_request_start)
        self.core.add_event_handler("request_end", self._on_request_end)
        
        # Error handling
        self.core.add_event_handler("error", self._on_error)
        
        # Metrics updates
        self.core.add_event_handler("metrics_update", self._on_metrics_update)
    
    async def _on_request_start(self, data: Dict[str, Any]):
        """Handle request start"""
        self.stats.total_requests += 1
    
    async def _on_request_end(self, data: Dict[str, Any]):
        """Handle request end"""
        response_time = data.get("response_time", 0)
        success = data.get("success", False)
        
        if success:
            self.stats.successful_requests += 1
        else:
            self.stats.failed_requests += 1
        
        # Update response time tracking
        self.request_times.append(response_time)
        if len(self.request_times) > self.max_request_history:
            self.request_times.pop(0)
        
        # Calculate average response time
        if self.request_times:
            self.stats.average_response_time = sum(self.request_times) / len(self.request_times)
    
    async def _on_error(self, data: Dict[str, Any]):
        """Handle system errors"""
        error = data.get("error", "Unknown error")
        component = data.get("component", "Unknown")
        logger.error(f"System error in {component}: {error}")
    
    async def _on_metrics_update(self, data: Dict[str, Any]):
        """Handle metrics updates"""
        metrics = data.get("metrics", {})
        self.stats.memory_usage_mb = metrics.get("memory_usage", 0) * self.config.max_memory_mb / 100
        self.stats.cpu_usage_percent = metrics.get("cpu_usage", 0)
        self.stats.disk_usage_percent = metrics.get("disk_usage", 0)
        self.stats.uptime_seconds = int(metrics.get("uptime", 0))
        self.stats.models_loaded = metrics.get("active_models", 0)
        self.stats.agents_active = metrics.get("active_agents", 0)
    
    async def shutdown(self) -> bool:
        """Shutdown the system"""
        try:
            logger.info("Shutting down SutazAI system...")
            return await self.core.stop()
        except Exception as e:
            logger.error(f"System shutdown error: {e}")
            return False
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        core_status = self.core.get_system_status()
        
        # Add additional stats
        core_status.update({
            "statistics": asdict(self.stats),
            "configuration": {
                "system_name": self.config.system_name,
                "version": self.config.version,
                "environment": self.config.environment,
                "debug_mode": self.config.debug_mode,
                "features": {
                    "neural_processing": self.config.enable_neural_processing,
                    "agent_orchestration": self.config.enable_agent_orchestration,
                    "knowledge_management": self.config.enable_knowledge_management,
                    "web_learning": self.config.enable_web_learning,
                    "self_modification": self.config.enable_self_modification
                }
            }
        })
        
        return core_status
    
    def get_component_status(self, component_name: str = None) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Get component status"""
        if component_name:
            component_info = self.core.get_component_info(component_name)
            if component_info:
                return {
                    "name": component_info.name,
                    "version": component_info.version,
                    "status": component_info.status.value,
                    "config": component_info.config,
                    "metrics": component_info.metrics,
                    "last_updated": component_info.last_updated.isoformat()
                }
            else:
                return {"error": f"Component {component_name} not found"}
        else:
            # Return all components
            components = []
            for name, info in self.core.get_component_info().items():
                components.append({
                    "name": info.name,
                    "version": info.version,
                    "status": info.status.value,
                    "config": info.config,
                    "metrics": info.metrics,
                    "last_updated": info.last_updated.isoformat()
                })
            return components
    
    def get_system_metrics(self, limit: int = 100) -> Dict[str, Any]:
        """Get system metrics"""
        return {
            "current": asdict(self.stats),
            "history": self.core.get_metrics_history(limit),
            "request_times": self.request_times[-limit:] if self.request_times else []
        }
    
    async def execute_command(self, command: str, **kwargs) -> Any:
        """Execute system command"""
        try:
            # Add request tracking
            await self.core._fire_event("request_start", {"command": command})
            
            start_time = asyncio.get_event_loop().time()
            
            try:
                result = await self.core.execute_command(command, **kwargs)
                
                # Track successful request
                end_time = asyncio.get_event_loop().time()
                response_time = end_time - start_time
                
                await self.core._fire_event("request_end", {
                    "command": command,
                    "response_time": response_time,
                    "success": True
                })
                
                return result
                
            except Exception as e:
                # Track failed request
                end_time = asyncio.get_event_loop().time()
                response_time = end_time - start_time
                
                await self.core._fire_event("request_end", {
                    "command": command,
                    "response_time": response_time,
                    "success": False,
                    "error": str(e)
                })
                
                raise
                
        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            raise
    
    # Convenience methods for common operations
    
    async def load_model(self, model_name: str) -> Any:
        """Load a model"""
        return await self.execute_command("model.load", model_name=model_name)
    
    async def unload_model(self, model_name: str) -> Any:
        """Unload a model"""
        return await self.execute_command("model.unload", model_name=model_name)
    
    async def list_models(self) -> Any:
        """List available models"""
        return await self.execute_command("model.list")
    
    async def create_agent(self, agent_type: str, **kwargs) -> Any:
        """Create an agent"""
        return await self.execute_command("agent.create", agent_type=agent_type, **kwargs)
    
    async def execute_agent_task(self, instance_id: str, task: Dict[str, Any]) -> Any:
        """Execute an agent task"""
        return await self.execute_command("agent.execute", instance_id=instance_id, task=task)
    
    async def get_health_check(self) -> Any:
        """Get system health check"""
        return await self.execute_command("system.health")
    
    def get_config(self) -> Dict[str, Any]:
        """Get system configuration"""
        return asdict(self.config)
    
    def update_config(self, config_updates: Dict[str, Any]) -> bool:
        """Update system configuration"""
        try:
            # Load current config
            with open(self.config_path, 'r') as f:
                current_config = json.load(f)
            
            # Update with new values
            def deep_update(base_dict, update_dict):
                for key, value in update_dict.items():
                    if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                        deep_update(base_dict[key], value)
                    else:
                        base_dict[key] = value
            
            deep_update(current_config, config_updates)
            
            # Save updated config
            with open(self.config_path, 'w') as f:
                json.dump(current_config, f, indent=2)
            
            logger.info("Configuration updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return False
    
    def get_component(self, name: str) -> Any:
        """Get a specific component"""
        return self.core.get_component(name)
    
    def is_running(self) -> bool:
        """Check if system is running"""
        return self.core.status == SystemStatus.RUNNING
    
    def is_healthy(self) -> bool:
        """Check if system is healthy"""
        try:
            # Check if all critical components are active
            critical_components = ["model_manager", "agent_framework"]
            for component in critical_components:
                if component in self.core.component_status:
                    if self.core.component_status[component] != ComponentStatus.ACTIVE:
                        return False
            
            # Check resource usage
            if self.stats.memory_usage_mb > self.config.max_memory_mb * 0.9:
                return False
            
            if self.stats.cpu_usage_percent > 95:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False

# Global system manager instance
_system_manager: Optional[SystemManager] = None

def get_system_manager() -> Optional[SystemManager]:
    """Get the global system manager instance"""
    return _system_manager

def initialize_system_manager(config_path: str = None) -> SystemManager:
    """Initialize the global system manager"""
    global _system_manager
    if _system_manager is None:
        _system_manager = SystemManager(config_path)
    return _system_manager

async def startup_system_manager() -> bool:
    """Startup the system manager"""
    system_manager = get_system_manager()
    if system_manager:
        return await system_manager.initialize()
    return False

async def shutdown_system_manager() -> bool:
    """Shutdown the system manager"""
    system_manager = get_system_manager()
    if system_manager:
        return await system_manager.shutdown()
    return False