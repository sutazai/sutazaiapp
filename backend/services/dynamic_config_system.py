"""
Dynamic Configuration Update System
Advanced system for managing and updating system configurations in real-time
"""

import asyncio
import logging
import json
import yaml
import time
import os
import shutil
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
import hashlib
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logger = logging.getLogger(__name__)

class ConfigChangeType(str, Enum):
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    RELOAD = "reload"

class ConfigLevel(str, Enum):
    SYSTEM = "system"
    USER = "user"
    AGENT = "agent"
    SERVICE = "service"
    RUNTIME = "runtime"

@dataclass
class ConfigChange:
    """Configuration change record"""
    id: str
    timestamp: float
    change_type: ConfigChangeType
    config_level: ConfigLevel
    config_path: str
    old_value: Any
    new_value: Any
    user_id: str
    reason: str
    validated: bool = False
    applied: bool = False
    rollback_available: bool = False

@dataclass
class ConfigSchema:
    """Configuration schema definition"""
    name: str
    schema: Dict[str, Any]
    validation_rules: List[Dict[str, Any]]
    dependencies: List[str]
    impact_level: str  # low, medium, high, critical
    requires_restart: bool
    hot_reloadable: bool

class ConfigFileHandler(FileSystemEventHandler):
    """File system event handler for configuration changes"""
    
    def __init__(self, config_system):
        self.config_system = config_system
        
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(('.yml', '.yaml', '.json')):
            asyncio.create_task(self.config_system._handle_file_change(event.src_path))

class DynamicConfigSystem:
    """
    Dynamic Configuration Update System
    Manages real-time configuration updates with validation and rollback
    """
    
    # Authorized user for critical changes
    AUTHORIZED_USER = "chrissuta01@gmail.com"
    
    def __init__(self, config_dir: str = "/opt/sutazaiapp/config"):
        self.config_dir = Path(config_dir)
        self.data_dir = Path("/opt/sutazaiapp/data/config_management")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration state
        self.current_configs = {}
        self.config_schemas = {}
        self.config_history = []
        self.pending_changes = {}
        
        # Validation and safety
        self.validation_rules = {}
        self.rollback_points = {}
        self.change_callbacks = {}
        
        # Performance metrics
        self.performance_baselines = {}
        self.config_impact_metrics = {}
        
        # File watching
        self.file_observer = None
        
        # Initialize
        self._load_config_schemas()
        self._load_current_configs()
        self._setup_file_watching()
        self._create_initial_rollback_point()
    
    def _load_config_schemas(self):
        """Load configuration schemas"""
        try:
            schema_dir = self.config_dir / "schemas"
            if schema_dir.exists():
                for schema_file in schema_dir.glob("*.json"):
                    with open(schema_file, 'r') as f:
                        schema_data = json.load(f)
                        schema = ConfigSchema(**schema_data)
                        self.config_schemas[schema.name] = schema
            
            # Create default schemas if none exist
            if not self.config_schemas:
                self._create_default_schemas()
                
            logger.info(f"✅ Loaded {len(self.config_schemas)} configuration schemas")
            
        except Exception as e:
            logger.error(f"Failed to load config schemas: {e}")
            self._create_default_schemas()
    
    def _create_default_schemas(self):
        """Create default configuration schemas"""
        default_schemas = {
            "main_config": ConfigSchema(
                name="main_config",
                schema={
                    "type": "object",
                    "properties": {
                        "app": {"type": "object"},
                        "database": {"type": "object"},
                        "agents": {"type": "object"},
                        "logging": {"type": "object"}
                    }
                },
                validation_rules=[
                    {"rule": "required_fields", "fields": ["app", "database"]},
                    {"rule": "valid_log_level", "field": "logging.level", "values": ["DEBUG", "INFO", "WARNING", "ERROR"]}
                ],
                dependencies=[],
                impact_level="high",
                requires_restart=True,
                hot_reloadable=False
            ),
            "agent_config": ConfigSchema(
                name="agent_config",
                schema={
                    "type": "object",
                    "properties": {
                        "enabled": {"type": "boolean"},
                        "config": {"type": "object"},
                        "capabilities": {"type": "array"}
                    }
                },
                validation_rules=[
                    {"rule": "valid_capabilities", "field": "capabilities"}
                ],
                dependencies=["main_config"],
                impact_level="medium",
                requires_restart=False,
                hot_reloadable=True
            )
        }
        
        self.config_schemas.update(default_schemas)
    
    def _load_current_configs(self):
        """Load current configuration files"""
        try:
            config_files = {
                "main_config": self.config_dir / "config.yml",
                "agent_config": self.config_dir / "agents.json",
                "logging_config": self.config_dir / "logging.yml"
            }
            
            for config_name, config_path in config_files.items():
                if config_path.exists():
                    self.current_configs[config_name] = self._load_config_file(config_path)
                else:
                    self.current_configs[config_name] = {}
            
            logger.info(f"✅ Loaded {len(self.current_configs)} configuration files")
            
        except Exception as e:
            logger.error(f"Failed to load current configs: {e}")
    
    def _load_config_file(self, file_path: Path) -> Dict[str, Any]:
        """Load a single configuration file"""
        try:
            with open(file_path, 'r') as f:
                if file_path.suffix.lower() in ['.yml', '.yaml']:
                    return yaml.safe_load(f) or {}
                elif file_path.suffix.lower() == '.json':
                    return json.load(f) or {}
                else:
                    return {}
        except Exception as e:
            logger.error(f"Failed to load config file {file_path}: {e}")
            return {}
    
    def _setup_file_watching(self):
        """Setup file system watching for configuration changes"""
        try:
            self.file_observer = Observer()
            event_handler = ConfigFileHandler(self)
            self.file_observer.schedule(event_handler, str(self.config_dir), recursive=True)
            self.file_observer.start()
            
            logger.info("✅ Configuration file watching enabled")
            
        except Exception as e:
            logger.error(f"Failed to setup file watching: {e}")
    
    def _create_initial_rollback_point(self):
        """Create initial system rollback point"""
        try:
            rollback_id = str(uuid.uuid4())
            rollback_data = {
                "id": rollback_id,
                "timestamp": time.time(),
                "configs": self.current_configs.copy(),
                "description": "Initial system state",
                "performance_baseline": self._capture_performance_baseline()
            }
            
            self.rollback_points[rollback_id] = rollback_data
            
            # Save to disk
            rollback_file = self.data_dir / f"rollback_{rollback_id}.json"
            with open(rollback_file, 'w') as f:
                json.dump(rollback_data, f, indent=2, default=str)
            
            logger.info(f"✅ Created initial rollback point: {rollback_id}")
            
        except Exception as e:
            logger.error(f"Failed to create initial rollback point: {e}")
    
    def _capture_performance_baseline(self) -> Dict[str, Any]:
        """Capture current performance metrics as baseline"""
        try:
            # This would integrate with monitoring systems
            baseline = {
                "response_time": 100,  # ms
                "memory_usage": 512,   # MB
                "cpu_usage": 10,       # %
                "error_rate": 0.01,    # %
                "timestamp": time.time()
            }
            
            return baseline
            
        except Exception as e:
            logger.error(f"Failed to capture performance baseline: {e}")
            return {}
    
    async def update_config(self, config_name: str, config_path: str, new_value: Any, user_id: str, reason: str = "") -> Dict[str, Any]:
        """Update a configuration value with validation and safety checks"""
        try:
            # Authorization check for critical changes
            if await self._is_critical_change(config_name, config_path, new_value):
                if user_id != self.AUTHORIZED_USER:
                    return {
                        "success": False,
                        "error": "Unauthorized: Only the authorized user can make critical configuration changes",
                        "required_authorization": self.AUTHORIZED_USER
                    }
            
            # Get current value
            old_value = self._get_config_value(config_name, config_path)
            
            # Validate the change
            validation_result = await self._validate_config_change(config_name, config_path, new_value)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": f"Validation failed: {validation_result['error']}",
                    "validation_details": validation_result
                }
            
            # Create change record
            change = ConfigChange(
                id=str(uuid.uuid4()),
                timestamp=time.time(),
                change_type=ConfigChangeType.UPDATE,
                config_level=self._determine_config_level(config_name),
                config_path=f"{config_name}.{config_path}",
                old_value=old_value,
                new_value=new_value,
                user_id=user_id,
                reason=reason,
                validated=True
            )
            
            # Create rollback point before applying change
            rollback_id = await self._create_rollback_point(f"Before {change.id}")
            change.rollback_available = True
            
            # Apply the change
            success = await self._apply_config_change(change)
            
            if success:
                change.applied = True
                self.config_history.append(change)
                
                # Monitor performance impact
                await self._monitor_change_impact(change)
                
                # Trigger callbacks
                await self._trigger_change_callbacks(config_name, config_path, new_value)
                
                return {
                    "success": True,
                    "change_id": change.id,
                    "rollback_id": rollback_id,
                    "applied_at": change.timestamp,
                    "hot_reloadable": self._is_hot_reloadable(config_name)
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to apply configuration change",
                    "change_id": change.id
                }
                
        except Exception as e:
            logger.error(f"Failed to update config {config_name}.{config_path}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _is_critical_change(self, config_name: str, config_path: str, new_value: Any) -> bool:
        """Determine if a configuration change is critical"""
        try:
            schema = self.config_schemas.get(config_name)
            if schema and schema.impact_level == "critical":
                return True
            
            # Check for specific critical paths
            critical_paths = [
                "database.password",
                "security.secret_key",
                "app.environment",
                "agents.*.enabled"  # Wildcard for agent enablement
            ]
            
            full_path = f"{config_name}.{config_path}"
            for critical_path in critical_paths:
                if self._path_matches(full_path, critical_path):
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to determine if change is critical: {e}")
            return True  # Err on the side of caution
    
    def _path_matches(self, actual_path: str, pattern: str) -> bool:
        """Check if a config path matches a pattern with wildcards"""
        actual_parts = actual_path.split('.')
        pattern_parts = pattern.split('.')
        
        if len(actual_parts) != len(pattern_parts):
            return False
        
        for actual, pattern_part in zip(actual_parts, pattern_parts):
            if pattern_part != '*' and actual != pattern_part:
                return False
        
        return True
    
    def _get_config_value(self, config_name: str, config_path: str) -> Any:
        """Get current configuration value"""
        try:
            config = self.current_configs.get(config_name, {})
            path_parts = config_path.split('.')
            
            current = config
            for part in path_parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return None
            
            return current
            
        except Exception as e:
            logger.error(f"Failed to get config value: {e}")
            return None
    
    async def _validate_config_change(self, config_name: str, config_path: str, new_value: Any) -> Dict[str, Any]:
        """Validate a configuration change"""
        try:
            schema = self.config_schemas.get(config_name)
            if not schema:
                return {"valid": True, "warnings": ["No schema found for validation"]}
            
            validation_errors = []
            warnings = []
            
            # Apply validation rules
            for rule in schema.validation_rules:
                rule_result = await self._apply_validation_rule(rule, config_name, config_path, new_value)
                if not rule_result["valid"]:
                    validation_errors.append(rule_result["error"])
                if rule_result.get("warnings"):
                    warnings.extend(rule_result["warnings"])
            
            # Check dependencies
            dependency_result = await self._check_dependencies(schema, config_name, config_path, new_value)
            if not dependency_result["valid"]:
                validation_errors.extend(dependency_result["errors"])
            
            # Type validation
            type_result = await self._validate_type(config_path, new_value, schema.schema)
            if not type_result["valid"]:
                validation_errors.append(type_result["error"])
            
            return {
                "valid": len(validation_errors) == 0,
                "error": "; ".join(validation_errors) if validation_errors else None,
                "warnings": warnings,
                "validation_details": {
                    "schema_name": schema.name,
                    "rules_applied": len(schema.validation_rules),
                    "dependencies_checked": len(schema.dependencies)
                }
            }
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {"valid": False, "error": f"Validation error: {str(e)}"}
    
    async def _apply_validation_rule(self, rule: Dict[str, Any], config_name: str, config_path: str, new_value: Any) -> Dict[str, Any]:
        """Apply a specific validation rule"""
        try:
            rule_type = rule.get("rule")
            
            if rule_type == "required_fields":
                # Check if required fields are present
                return {"valid": True}  # Simplified
            
            elif rule_type == "valid_log_level":
                if config_path == rule.get("field", "").split(".", 1)[-1]:
                    valid_levels = rule.get("values", [])
                    if str(new_value).upper() not in valid_levels:
                        return {
                            "valid": False,
                            "error": f"Invalid log level. Must be one of: {valid_levels}"
                        }
                return {"valid": True}
            
            elif rule_type == "range_check":
                min_val = rule.get("min")
                max_val = rule.get("max")
                if min_val is not None and new_value < min_val:
                    return {"valid": False, "error": f"Value must be >= {min_val}"}
                if max_val is not None and new_value > max_val:
                    return {"valid": False, "error": f"Value must be <= {max_val}"}
                return {"valid": True}
            
            else:
                return {"valid": True, "warnings": [f"Unknown validation rule: {rule_type}"]}
                
        except Exception as e:
            return {"valid": False, "error": f"Rule validation error: {str(e)}"}
    
    async def _check_dependencies(self, schema: ConfigSchema, config_name: str, config_path: str, new_value: Any) -> Dict[str, Any]:
        """Check configuration dependencies"""
        try:
            errors = []
            
            for dependency in schema.dependencies:
                if dependency not in self.current_configs:
                    errors.append(f"Missing dependency: {dependency}")
                    continue
                
                # Check specific dependency requirements
                if dependency == "main_config" and config_name == "agent_config":
                    # Ensure main config allows agent changes
                    main_config = self.current_configs["main_config"]
                    if not main_config.get("agents", {}).get("dynamic_updates", True):
                        errors.append("Dynamic agent updates are disabled in main configuration")
            
            return {
                "valid": len(errors) == 0,
                "errors": errors
            }
            
        except Exception as e:
            return {"valid": False, "errors": [f"Dependency check error: {str(e)}"]}
    
    async def _validate_type(self, config_path: str, new_value: Any, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate value type against schema"""
        try:
            # Simplified type validation
            expected_type = schema.get("type")
            
            if expected_type == "string" and not isinstance(new_value, str):
                return {"valid": False, "error": "Expected string value"}
            elif expected_type == "integer" and not isinstance(new_value, int):
                return {"valid": False, "error": "Expected integer value"}
            elif expected_type == "boolean" and not isinstance(new_value, bool):
                return {"valid": False, "error": "Expected boolean value"}
            elif expected_type == "array" and not isinstance(new_value, list):
                return {"valid": False, "error": "Expected array value"}
            elif expected_type == "object" and not isinstance(new_value, dict):
                return {"valid": False, "error": "Expected object value"}
            
            return {"valid": True}
            
        except Exception as e:
            return {"valid": False, "error": f"Type validation error: {str(e)}"}
    
    def _determine_config_level(self, config_name: str) -> ConfigLevel:
        """Determine the configuration level"""
        if config_name in ["main_config", "logging_config"]:
            return ConfigLevel.SYSTEM
        elif config_name == "agent_config":
            return ConfigLevel.AGENT
        else:
            return ConfigLevel.SERVICE
    
    async def _create_rollback_point(self, description: str) -> str:
        """Create a rollback point"""
        try:
            rollback_id = str(uuid.uuid4())
            rollback_data = {
                "id": rollback_id,
                "timestamp": time.time(),
                "configs": self.current_configs.copy(),
                "description": description,
                "performance_baseline": self._capture_performance_baseline()
            }
            
            self.rollback_points[rollback_id] = rollback_data
            
            # Save to disk
            rollback_file = self.data_dir / f"rollback_{rollback_id}.json"
            with open(rollback_file, 'w') as f:
                json.dump(rollback_data, f, indent=2, default=str)
            
            # Keep only last 50 rollback points
            if len(self.rollback_points) > 50:
                oldest_id = min(self.rollback_points.keys(), 
                              key=lambda x: self.rollback_points[x]["timestamp"])
                del self.rollback_points[oldest_id]
                (self.data_dir / f"rollback_{oldest_id}.json").unlink(missing_ok=True)
            
            logger.info(f"✅ Created rollback point: {rollback_id}")
            return rollback_id
            
        except Exception as e:
            logger.error(f"Failed to create rollback point: {e}")
            return ""
    
    async def _apply_config_change(self, change: ConfigChange) -> bool:
        """Apply a configuration change"""
        try:
            config_name, config_path = change.config_path.split('.', 1)
            
            # Update in-memory configuration
            config = self.current_configs.get(config_name, {})
            self._set_nested_value(config, config_path, change.new_value)
            self.current_configs[config_name] = config
            
            # Write to file if applicable
            await self._write_config_to_file(config_name)
            
            # Hot reload if supported
            if self._is_hot_reloadable(config_name):
                await self._hot_reload_config(config_name)
            
            logger.info(f"✅ Applied config change: {change.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply config change: {e}")
            return False
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any):
        """Set a nested configuration value"""
        path_parts = path.split('.')
        current = config
        
        for part in path_parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[path_parts[-1]] = value
    
    async def _write_config_to_file(self, config_name: str):
        """Write configuration to file"""
        try:
            config_files = {
                "main_config": self.config_dir / "config.yml",
                "agent_config": self.config_dir / "agents.json",
                "logging_config": self.config_dir / "logging.yml"
            }
            
            config_file = config_files.get(config_name)
            if not config_file:
                return
            
            config_data = self.current_configs[config_name]
            
            # Create backup
            backup_file = config_file.with_suffix(f".backup.{int(time.time())}")
            if config_file.exists():
                shutil.copy2(config_file, backup_file)
            
            # Write new configuration
            with open(config_file, 'w') as f:
                if config_file.suffix.lower() in ['.yml', '.yaml']:
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                elif config_file.suffix.lower() == '.json':
                    json.dump(config_data, f, indent=2)
            
            logger.info(f"✅ Updated config file: {config_file}")
            
        except Exception as e:
            logger.error(f"Failed to write config file: {e}")
    
    def _is_hot_reloadable(self, config_name: str) -> bool:
        """Check if configuration supports hot reloading"""
        schema = self.config_schemas.get(config_name)
        return schema.hot_reloadable if schema else False
    
    async def _hot_reload_config(self, config_name: str):
        """Perform hot reload of configuration"""
        try:
            # Trigger reload callbacks
            if config_name in self.change_callbacks:
                for callback in self.change_callbacks[config_name]:
                    await callback(self.current_configs[config_name])
            
            logger.info(f"✅ Hot reloaded config: {config_name}")
            
        except Exception as e:
            logger.error(f"Failed to hot reload config: {e}")
    
    async def _monitor_change_impact(self, change: ConfigChange):
        """Monitor the impact of a configuration change"""
        try:
            # Wait a bit for change to take effect
            await asyncio.sleep(5)
            
            # Capture new performance metrics
            new_metrics = self._capture_performance_baseline()
            
            # Store impact data
            impact_data = {
                "change_id": change.id,
                "timestamp": time.time(),
                "metrics_before": self.performance_baselines.get("current", {}),
                "metrics_after": new_metrics,
                "impact_detected": False
            }
            
            # Simple impact detection
            if self.performance_baselines.get("current"):
                before = self.performance_baselines["current"]
                
                # Check for significant changes
                response_time_change = abs(new_metrics.get("response_time", 0) - before.get("response_time", 0))
                memory_change = abs(new_metrics.get("memory_usage", 0) - before.get("memory_usage", 0))
                
                if response_time_change > 50 or memory_change > 100:  # Thresholds
                    impact_data["impact_detected"] = True
                    impact_data["impact_details"] = {
                        "response_time_change": response_time_change,
                        "memory_change": memory_change
                    }
            
            self.config_impact_metrics[change.id] = impact_data
            self.performance_baselines["current"] = new_metrics
            
            if impact_data["impact_detected"]:
                logger.warning(f"⚠️ Performance impact detected for change {change.id}")
            
        except Exception as e:
            logger.error(f"Failed to monitor change impact: {e}")
    
    async def _trigger_change_callbacks(self, config_name: str, config_path: str, new_value: Any):
        """Trigger callbacks for configuration changes"""
        try:
            callbacks = self.change_callbacks.get(config_name, [])
            for callback in callbacks:
                await callback(config_path, new_value)
                
        except Exception as e:
            logger.error(f"Failed to trigger change callbacks: {e}")
    
    def register_change_callback(self, config_name: str, callback: Callable):
        """Register a callback for configuration changes"""
        if config_name not in self.change_callbacks:
            self.change_callbacks[config_name] = []
        self.change_callbacks[config_name].append(callback)
    
    async def rollback_to_point(self, rollback_id: str, user_id: str) -> Dict[str, Any]:
        """Rollback to a specific configuration state"""
        try:
            # Authorization check
            if user_id != self.AUTHORIZED_USER:
                return {
                    "success": False,
                    "error": "Unauthorized: Only the authorized user can perform rollbacks"
                }
            
            rollback_data = self.rollback_points.get(rollback_id)
            if not rollback_data:
                return {"success": False, "error": "Rollback point not found"}
            
            # Create new rollback point before applying rollback
            current_rollback_id = await self._create_rollback_point("Before rollback")
            
            # Apply rollback
            self.current_configs = rollback_data["configs"].copy()
            
            # Write configurations to files
            for config_name in self.current_configs:
                await self._write_config_to_file(config_name)
            
            # Record rollback in history
            rollback_change = ConfigChange(
                id=str(uuid.uuid4()),
                timestamp=time.time(),
                change_type=ConfigChangeType.RELOAD,
                config_level=ConfigLevel.SYSTEM,
                config_path="system.rollback",
                old_value=f"current_state",
                new_value=f"rollback_to_{rollback_id}",
                user_id=user_id,
                reason=f"Rollback to {rollback_data['description']}",
                validated=True,
                applied=True,
                rollback_available=True
            )
            
            self.config_history.append(rollback_change)
            
            logger.info(f"✅ Rolled back to point: {rollback_id}")
            
            return {
                "success": True,
                "rollback_id": rollback_id,
                "description": rollback_data["description"],
                "rollback_timestamp": rollback_data["timestamp"],
                "new_rollback_point": current_rollback_id
            }
            
        except Exception as e:
            logger.error(f"Failed to rollback: {e}")
            return {"success": False, "error": str(e)}
    
    async def _handle_file_change(self, file_path: str):
        """Handle external file changes"""
        try:
            file_path = Path(file_path)
            config_name = self._get_config_name_from_file(file_path)
            
            if config_name:
                # Reload configuration from file
                new_config = self._load_config_file(file_path)
                old_config = self.current_configs.get(config_name, {})
                
                if new_config != old_config:
                    # Create change record for external modification
                    change = ConfigChange(
                        id=str(uuid.uuid4()),
                        timestamp=time.time(),
                        change_type=ConfigChangeType.UPDATE,
                        config_level=self._determine_config_level(config_name),
                        config_path=f"{config_name}.external_update",
                        old_value=old_config,
                        new_value=new_config,
                        user_id="external",
                        reason="External file modification",
                        validated=False,
                        applied=True
                    )
                    
                    self.current_configs[config_name] = new_config
                    self.config_history.append(change)
                    
                    # Trigger callbacks
                    await self._trigger_change_callbacks(config_name, "external_update", new_config)
                    
                    logger.info(f"📁 External config file change detected: {file_path}")
                    
        except Exception as e:
            logger.error(f"Failed to handle file change: {e}")
    
    def _get_config_name_from_file(self, file_path: Path) -> Optional[str]:
        """Get configuration name from file path"""
        mapping = {
            "config.yml": "main_config",
            "agents.json": "agent_config",
            "logging.yml": "logging_config"
        }
        return mapping.get(file_path.name)
    
    async def get_config_status(self) -> Dict[str, Any]:
        """Get comprehensive configuration status"""
        try:
            recent_changes = [
                asdict(change) for change in self.config_history[-10:]
            ]
            
            # Calculate change frequency
            now = time.time()
            recent_changes_count = len([
                change for change in self.config_history
                if now - change.timestamp < 3600  # Last hour
            ])
            
            return {
                "total_configs": len(self.current_configs),
                "total_schemas": len(self.config_schemas),
                "recent_changes": recent_changes,
                "recent_changes_count": recent_changes_count,
                "rollback_points": len(self.rollback_points),
                "hot_reloadable_configs": [
                    name for name, schema in self.config_schemas.items()
                    if schema.hot_reloadable
                ],
                "pending_changes": len(self.pending_changes),
                "file_watching_active": self.file_observer.is_alive() if self.file_observer else False,
                "performance_monitoring": bool(self.config_impact_metrics),
                "last_change": self.config_history[-1].timestamp if self.config_history else None
            }
            
        except Exception as e:
            logger.error(f"Failed to get config status: {e}")
            return {"error": str(e)}
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            if self.file_observer:
                self.file_observer.stop()
                self.file_observer.join()
            
            logger.info("✅ Dynamic config system cleaned up")
            
        except Exception as e:
            logger.error(f"Failed to cleanup: {e}")

# Global instance
dynamic_config_system = DynamicConfigSystem()

# Convenience functions
async def update_config(config_name: str, config_path: str, new_value: Any, user_id: str, reason: str = "") -> Dict[str, Any]:
    """Update configuration"""
    return await dynamic_config_system.update_config(config_name, config_path, new_value, user_id, reason)

async def rollback_to_point(rollback_id: str, user_id: str) -> Dict[str, Any]:
    """Rollback to configuration point"""
    return await dynamic_config_system.rollback_to_point(rollback_id, user_id)

async def get_config_status() -> Dict[str, Any]:
    """Get configuration status"""
    return await dynamic_config_system.get_config_status()

def register_change_callback(config_name: str, callback: Callable):
    """Register change callback"""
    dynamic_config_system.register_change_callback(config_name, callback)