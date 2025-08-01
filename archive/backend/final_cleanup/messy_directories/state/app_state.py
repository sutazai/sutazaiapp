#!/usr/bin/env python3
"""
SutazAI Application State Management
Provides centralized state management for the SutazAI system
"""

import threading
from typing import Dict, Any, Optional
from datetime import datetime


class AppState:
    """Centralized application state manager"""
    
    def __init__(self):
        self._state: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._initialized = False
        self._startup_time = datetime.now()
    
    def initialize(self) -> None:
        """Initialize the application state"""
        with self._lock:
            if not self._initialized:
                self._state.update({
                    "startup_time": self._startup_time,
                    "version": "1.0.0",
                    "services": {},
                    "agents": {},
                    "metrics": {}
                })
                self._initialized = True
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the state"""
        with self._lock:
            return self._state.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a value in the state"""
        with self._lock:
            self._state[key] = value
    
    def update(self, data: Dict[str, Any]) -> None:
        """Update multiple values in the state"""
        with self._lock:
            self._state.update(data)
    
    def is_initialized(self) -> bool:
        """Check if the state is initialized"""
        return self._initialized
    
    def get_all(self) -> Dict[str, Any]:
        """Get all state data (copy)"""
        with self._lock:
            return self._state.copy()
    
    def cleanup(self) -> None:
        """Cleanup state on shutdown"""
        with self._lock:
            self._state.clear()
            self._initialized = False