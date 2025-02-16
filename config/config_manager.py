from typing import Dict, Any
import os
import yaml
from dotenv import load_dotenv
import logging

class ConfigManager:
    _instance = None
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.config: Dict[str, Any] = {}
        self._load_env()
        self._load_yaml_configs()
        self._validate_config()
    
    def _load_env(self):
        load_dotenv()
        for key, value in os.environ.items():
            self.config[key] = value
    
    def _load_yaml_configs(self):
        config_files = [
            'config/base.yaml',
            f'config/{os.getenv("DEPLOYMENT_ENV", "local")}.yaml'
        ]
        
        for config_file in config_files:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    self.config.update(yaml_config)
    
    def _validate_config(self):
        required_keys = [
            'DB_HOST', 'DB_PORT', 'DB_USER', 
            'SECRET_KEY', 'API_V1_STR'
        ]
        
        for key in required_keys:
            if key not in self.config:
                logging.error(f"Missing required configuration: {key}")
                raise ValueError(f"Missing required configuration: {key}")
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)
    
    def reload(self):
        self._initialize()

config_manager = ConfigManager() 