from typing import Any, Dict, Optional
import os
import json
import yaml
from dataclasses import dataclass, asdict
import logging

@dataclass
class ConfigSettings:
    debug: bool = False
    host: str = 'localhost'
    port: int = 8000
    log_level: str = 'INFO'

class AdvancedConfigManager:
    _instance = None
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(
        self, 
        config_path: Optional[str] = None, 
        debug: bool = False, 
        port: int = 8000
    ):
        self.config_path = config_path or self._default_config_path()
        self.debug = bool(debug)
        self.port = int(port)
        self.config = self._load_config()
    
    def _default_config_path(self) -> str:
        return os.path.join(os.getcwd(), 'config.json')

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r') as config_file:
                return json.load(config_file)
        except FileNotFoundError:
            logging.warning(f"Config file not found at {self.config_path}")
            return {}
        except json.JSONDecodeError:
            logging.error(f"Invalid JSON in config file: {self.config_path}")
            return {}
    
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self.config, key, default)

def get_config_settings() -> ConfigSettings:
    return AdvancedConfigManager().config 