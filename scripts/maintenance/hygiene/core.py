#!/usr/bin/env python3
"""
Core hygiene orchestration classes and data models
Extracted from hygiene_orchestrator.py for modularity
"""

import os
import json
import time
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from collections import deque, defaultdict
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class ViolationPattern:
    """Represents a detected hygiene violation pattern"""
    pattern_type: str
    severity: str
    confidence: float
    description: str
    file_path: str
    line_number: int
    suggested_fix: str
    auto_fixable: bool = False
    dependencies: List[str] = field(default_factory=list)

@dataclass
class HygieneMetrics:
    """Performance and health metrics for hygiene operations"""
    total_files_scanned: int = 0
    violations_found: int = 0
    violations_fixed: int = 0
    scan_duration: float = 0.0
    fix_duration: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    
@dataclass
class SystemHealth:
    """Overall system health status"""
    status: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.now)
    components: Dict[str, bool] = field(default_factory=dict)
    metrics: HygieneMetrics = field(default_factory=HygieneMetrics)
    alerts: List[str] = field(default_factory=list)

class HygieneConfig:
    """Configuration management for hygiene operations"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.getenv('HYGIENE_CONFIG', 'hygiene_config.json')
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file or environment"""
        default_config = {
            "scan_directories": ["/opt/sutazaiapp"],
            "exclude_patterns": ["node_modules", ".git", "__pycache__"],
            "file_size_limit": 500,
            "violation_thresholds": {
                "critical": 10,
                "high": 25,
                "medium": 50
            },
            "auto_fix_enabled": False,
            "notification_enabled": True
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}")
        
        return default_config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration"""
        self.config.update(updates)
        self._save_config()
    
    def _save_config(self) -> None:
        """Save configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save config to {self.config_path}: {e}")