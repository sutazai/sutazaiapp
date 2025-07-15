"""
Configuration Settings
System configuration management
"""

import json
import logging
from typing import Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class Settings:
    """System settings manager"""
    
    def __init__(self, config_path: str = "/opt/sutazaiapp/config/settings.json"):
        self.config_path = config_path
        self.settings = self._load_settings()
        
    def _load_settings(self) -> Dict[str, Any]:
        """Load settings from file"""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    return json.load(f)
            else:
                return self._get_default_settings()
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            return self._get_default_settings()
    
    def _get_default_settings(self) -> Dict[str, Any]:
        """Get default settings"""
        return {
            "database": {
                "type": "sqlite",
                "path": "/opt/sutazaiapp/data/sutazai.db"
            },
            "api": {
                "host": "0.0.0.0",
                "port": 8000
            },
            "security": {
                "enabled": True,
                "authorized_user": "chrissuta01@gmail.com"
            },
            "neural_network": {
                "default_nodes": 100,
                "learning_rate": 0.01
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get setting value"""
        keys = key.split('.')
        value = self.settings
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def dict(self) -> Dict[str, Any]:
        """Get all settings as dictionary"""
        return self.settings.copy()