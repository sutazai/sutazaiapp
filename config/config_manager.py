#!/usr/bin/env python3
"""
SutazAI Comprehensive Configuration Management System

Provides advanced, multi-environment configuration handling with
security, validation, and dynamic loading capabilities.
"""

import os
import sys
import yaml
import json
from typing import Dict, Any, Optional
from functools import lru_cache
from dataclasses import dataclass, asdict
import jsonschema
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='/opt/sutazai_project/SutazAI/logs/config_management.log'
)
logger = logging.getLogger('SutazAI.ConfigManager')

@dataclass
class ConfigurationProfile:
    """
    Represents a comprehensive configuration profile
    with advanced validation and security features.
    """
    environment: str
    debug_mode: bool = False
    security_level: int = 2
    allowed_modules: Optional[Dict[str, bool]] = None
    rate_limits: Optional[Dict[str, int]] = None

class ConfigurationManager:
    """
    Advanced configuration management with multi-environment support,
    dynamic loading, and comprehensive validation.
    """
    
    _BASE_CONFIG_PATH = '/opt/sutazai_project/SutazAI/config'
    _ENVIRONMENT_CONFIGS = {
        'development': 'development.yml',
        'production': 'production.yml',
        'testing': 'testing.yml'
    }
    
    def __init__(
        self, 
        environment: str = 'development', 
        config_dir: Optional[str] = None
    ):
        """
        Initialize configuration manager with environment-specific settings.
        
        Args:
            environment (str): Target environment configuration
            config_dir (Optional[str]): Custom configuration directory
        """
        self.environment = environment
        self.config_dir = config_dir or self._BASE_CONFIG_PATH
        self._validate_environment()
    
    def _validate_environment(self):
        """
        Validate and ensure the selected environment is supported.
        """
        if self.environment not in self._ENVIRONMENT_CONFIGS:
            raise ValueError(f"Unsupported environment: {self.environment}")
    
    @lru_cache(maxsize=32)
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration with caching and comprehensive validation.
        
        Returns:
            Validated configuration dictionary
        """
        config_path = os.path.join(
            self.config_dir, 
            self._ENVIRONMENT_CONFIGS[self.environment]
        )
        
        try:
            with open(config_path, 'r') as config_file:
                config = yaml.safe_load(config_file)
            
            # Validate configuration
            self._validate_config(config)
            
            logger.info(f"Loaded {self.environment} configuration successfully")
            return config
        
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"YAML parsing error: {e}")
            raise
    
    def _validate_config(self, config: Dict[str, Any]):
        """
        Perform comprehensive configuration validation.
        
        Args:
            config (Dict): Configuration dictionary to validate
        """
        required_keys = ['system', 'security', 'modules']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration section: {key}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a configuration value with optional default.
        
        Args:
            key (str): Configuration key to retrieve
            default (Any): Default value if key is not found
        
        Returns:
            Configuration value or default
        """
        config = self.load_config()
        return config.get(key, default)
    
    def create_profile(self) -> ConfigurationProfile:
        """
        Create a configuration profile based on current environment.
        
        Returns:
            ConfigurationProfile instance
        """
        config = self.load_config()
        return ConfigurationProfile(
            environment=self.environment,
            debug_mode=config.get('system', {}).get('debug', False),
            security_level=config.get('security', {}).get('level', 2),
            allowed_modules=config.get('modules', {}),
            rate_limits=config.get('rate_limits', {})
        )

def main():
    """
    Demonstration of configuration management capabilities.
    """
    config_manager = ConfigurationManager(environment='development')
    profile = config_manager.create_profile()
    
    print("Configuration Profile:")
    print(yaml.dump(asdict(profile), default_flow_style=False))

if __name__ == '__main__':
    main()