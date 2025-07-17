#!/usr/bin/env python3
"""
Configuration Management for SutazAI System
Handles loading, validation, and management of system configuration
"""

import json
import os
import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type
import logging
from jsonschema import validate, ValidationError

from .errors import ConfigError

logger = logging.getLogger(__name__)

class ConfigValidator:
    """Configuration validation engine"""
    
    # JSON Schema for system configuration
    SYSTEM_CONFIG_SCHEMA = {
        "type": "object",
        "properties": {
            "system_name": {"type": "string"},
            "version": {"type": "string"},
            "environment": {"type": "string", "enum": ["development", "staging", "production"]},
            "debug_mode": {"type": "boolean"},
            "base_dir": {"type": "string"},
            "data_dir": {"type": "string"},
            "models_dir": {"type": "string"},
            "logs_dir": {"type": "string"},
            "cache_dir": {"type": "string"},
            "max_workers": {"type": "integer", "minimum": 1, "maximum": 32},
            "timeout_seconds": {"type": "integer", "minimum": 1},
            "max_memory_mb": {"type": "integer", "minimum": 512},
            "enable_gpu": {"type": "boolean"},
            "enable_neural_processing": {"type": "boolean"},
            "enable_agent_orchestration": {"type": "boolean"},
            "enable_knowledge_management": {"type": "boolean"},
            "enable_web_learning": {"type": "boolean"},
            "enable_self_modification": {"type": "boolean"},
            "enable_security": {"type": "boolean"},
            "enable_audit": {"type": "boolean"},
            "enable_encryption": {"type": "boolean"},
            "enable_monitoring": {"type": "boolean"},
            "metrics_interval": {"type": "integer", "minimum": 1},
            "health_check_interval": {"type": "integer", "minimum": 1}
        },
        "required": ["system_name", "version", "environment"]
    }
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> bool:
        """Validate configuration against schema"""
        try:
            validate(instance=config, schema=cls.SYSTEM_CONFIG_SCHEMA)
            return True
        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e}")
            raise ConfigError(f"Invalid configuration: {e}")
    
    @classmethod
    def validate_directories(cls, config: Dict[str, Any]) -> bool:
        """Validate directory paths exist or can be created"""
        dir_keys = ["base_dir", "data_dir", "models_dir", "logs_dir", "cache_dir"]
        
        for key in dir_keys:
            if key in config:
                path = Path(config[key])
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    if not path.exists():
                        raise ConfigError(f"Cannot create directory: {path}")
                except Exception as e:
                    raise ConfigError(f"Directory validation failed for {key}: {e}")
        
        return True
    
    @classmethod
    def validate_resources(cls, config: Dict[str, Any]) -> bool:
        """Validate resource limits"""
        import psutil
        
        # Check memory limits
        if "max_memory_mb" in config:
            available_memory = psutil.virtual_memory().available // (1024 * 1024)
            if config["max_memory_mb"] > available_memory:
                logger.warning(f"Requested memory ({config['max_memory_mb']}MB) exceeds available ({available_memory}MB)")
        
        # Check worker limits
        if "max_workers" in config:
            cpu_count = psutil.cpu_count()
            if config["max_workers"] > cpu_count * 2:
                logger.warning(f"High worker count ({config['max_workers']}) for CPU count ({cpu_count})")
        
        return True

class ConfigManager:
    """Configuration manager for SutazAI system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config_cache: Dict[str, Any] = {}
        self.validator = ConfigValidator()
        
        # Default configuration paths
        self.default_paths = [
            "/opt/sutazaiapp/config/system.json",
            "/opt/sutazaiapp/config/system.yaml",
            os.path.expanduser("~/.sutazai/config.json"),
            os.path.expanduser("~/.sutazai/config.yaml")
        ]
    
    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            path = config_path or self.config_path
            
            if not path:
                # Try default paths
                for default_path in self.default_paths:
                    if os.path.exists(default_path):
                        path = default_path
                        break
            
            if not path:
                logger.info("No configuration file found, using defaults")
                return self._get_default_config()
            
            config_file = Path(path)
            if not config_file.exists():
                raise ConfigError(f"Configuration file not found: {path}")
            
            # Load based on file extension
            if config_file.suffix.lower() == '.json':
                config = self._load_json_config(config_file)
            elif config_file.suffix.lower() in ['.yaml', '.yml']:
                config = self._load_yaml_config(config_file)
            else:
                raise ConfigError(f"Unsupported config file format: {config_file.suffix}")
            
            # Validate configuration
            self.validator.validate_config(config)
            self.validator.validate_directories(config)
            self.validator.validate_resources(config)
            
            # Cache configuration
            self.config_cache[path] = config
            
            logger.info(f"Configuration loaded from: {path}")
            return config
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise ConfigError(f"Configuration loading failed: {e}")
    
    def _load_json_config(self, config_file: Path) -> Dict[str, Any]:
        """Load JSON configuration"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ConfigError(f"Invalid JSON configuration: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to read JSON config: {e}")
    
    def _load_yaml_config(self, config_file: Path) -> Dict[str, Any]:
        """Load YAML configuration"""
        try:
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML configuration: {e}")
        except Exception as e:
            raise ConfigError(f"Failed to read YAML config: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "system_name": "SutazAI",
            "version": "1.0.0",
            "environment": "production",
            "debug_mode": False,
            "base_dir": "/opt/sutazaiapp",
            "data_dir": "/opt/sutazaiapp/data",
            "models_dir": "/opt/sutazaiapp/models",
            "logs_dir": "/opt/sutazaiapp/logs",
            "cache_dir": "/opt/sutazaiapp/cache",
            "max_workers": 4,
            "timeout_seconds": 300,
            "max_memory_mb": 8192,
            "enable_gpu": True,
            "enable_neural_processing": True,
            "enable_agent_orchestration": True,
            "enable_knowledge_management": True,
            "enable_web_learning": True,
            "enable_self_modification": False,
            "enable_security": True,
            "enable_audit": True,
            "enable_encryption": True,
            "enable_monitoring": True,
            "metrics_interval": 60,
            "health_check_interval": 30,
            "models_config": {
                "storage_type": "local",
                "max_models": 10,
                "auto_unload": True,
                "quantization": "auto"
            },
            "neural_config": {
                "enable_biological_modeling": True,
                "enable_neuromorphic": True,
                "optimization_level": "balanced"
            },
            "agents_config": {
                "max_agents": 50,
                "agent_timeout": 300,
                "enable_communication": True,
                "enable_collaboration": True
            },
            "knowledge_config": {
                "storage_type": "vector",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "max_documents": 100000,
                "enable_search": True
            },
            "security_config": {
                "authentication": True,
                "authorization": True,
                "encryption": "AES-256",
                "audit_level": "full"
            },
            "monitoring_config": {
                "enable_metrics": True,
                "enable_alerts": True,
                "enable_logging": True,
                "log_level": "INFO"
            }
        }
    
    def save_config(self, config: Dict[str, Any], config_path: Optional[str] = None) -> bool:
        """Save configuration to file"""
        try:
            path = config_path or self.config_path
            if not path:
                raise ConfigError("No configuration path specified")
            
            config_file = Path(path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Validate before saving
            self.validator.validate_config(config)
            
            # Save based on file extension
            if config_file.suffix.lower() == '.json':
                with open(config_file, 'w') as f:
                    json.dump(config, f, indent=2)
            elif config_file.suffix.lower() in ['.yaml', '.yml']:
                with open(config_file, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
            else:
                raise ConfigError(f"Unsupported config file format: {config_file.suffix}")
            
            # Update cache
            self.config_cache[path] = config
            
            logger.info(f"Configuration saved to: {path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise ConfigError(f"Configuration saving failed: {e}")
    
    def get_component_config(self, component_name: str, 
                           config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get component-specific configuration"""
        if config is None:
            config = self.load_config()
        
        component_key = f"{component_name}_config"
        return config.get(component_key, {})
    
    def update_component_config(self, component_name: str, 
                              component_config: Dict[str, Any],
                              config_path: Optional[str] = None) -> bool:
        """Update component-specific configuration"""
        try:
            # Load current config
            config = self.load_config(config_path)
            
            # Update component config
            component_key = f"{component_name}_config"
            config[component_key] = component_config
            
            # Save updated config
            return self.save_config(config, config_path)
            
        except Exception as e:
            logger.error(f"Failed to update component config: {e}")
            raise ConfigError(f"Component config update failed: {e}")
    
    def merge_configs(self, base_config: Dict[str, Any], 
                     override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration dictionaries"""
        def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result
        
        return deep_merge(base_config, override_config)
    
    def validate_config_change(self, old_config: Dict[str, Any], 
                             new_config: Dict[str, Any]) -> List[str]:
        """Validate configuration changes and return warnings"""
        warnings = []
        
        # Check for critical changes
        critical_keys = ["base_dir", "data_dir", "models_dir"]
        for key in critical_keys:
            if old_config.get(key) != new_config.get(key):
                warnings.append(f"Critical path change: {key}")
        
        # Check for feature disabling
        feature_keys = [
            "enable_neural_processing", "enable_agent_orchestration",
            "enable_knowledge_management", "enable_security", "enable_monitoring"
        ]
        for key in feature_keys:
            if old_config.get(key, True) and not new_config.get(key, True):
                warnings.append(f"Feature disabled: {key}")
        
        # Check for resource changes
        resource_keys = ["max_workers", "max_memory_mb", "timeout_seconds"]
        for key in resource_keys:
            old_val = old_config.get(key, 0)
            new_val = new_config.get(key, 0)
            if new_val < old_val * 0.5:  # 50% reduction
                warnings.append(f"Significant resource reduction: {key}")
        
        return warnings
    
    def get_environment_config(self, environment: str) -> Dict[str, Any]:
        """Get environment-specific configuration"""
        env_configs = {
            "development": {
                "debug_mode": True,
                "enable_monitoring": True,
                "log_level": "DEBUG",
                "metrics_interval": 30
            },
            "staging": {
                "debug_mode": False,
                "enable_monitoring": True,
                "log_level": "INFO",
                "metrics_interval": 60
            },
            "production": {
                "debug_mode": False,
                "enable_monitoring": True,
                "log_level": "WARNING",
                "metrics_interval": 300
            }
        }
        
        return env_configs.get(environment, {})
    
    def apply_environment_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment-specific configuration"""
        environment = config.get("environment", "production")
        env_config = self.get_environment_config(environment)
        return self.merge_configs(config, env_config)
    
    def get_cached_config(self, config_path: str) -> Optional[Dict[str, Any]]:
        """Get cached configuration"""
        return self.config_cache.get(config_path)
    
    def clear_cache(self):
        """Clear configuration cache"""
        self.config_cache.clear()
    
    def list_config_files(self) -> List[str]:
        """List available configuration files"""
        configs = []
        for path in self.default_paths:
            if os.path.exists(path):
                configs.append(path)
        return configs