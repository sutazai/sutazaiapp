import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseComponent(ABC):
    """
    Advanced abstract base class for all system components.
    Provides a comprehensive interface for initialization, 
    configuration, lifecycle management, and autonomous operation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the base component with advanced tracking and validation.

        Args:
            config: Optional configuration parameters.
        """
        self._id = str(uuid.uuid4())  # Unique identifier for each component instance
        self._logger = logging.getLogger(f"{self.__class__.__name__}[{self._id[:8]}]")
        
        self._config = config or {}
        self._state = {
            "creation_timestamp": logging.Formatter('%(asctime)s').formatTime(logging.LogRecord('', 0, '', 0, '', (), None)),
            "last_updated": None,
            "initialization_attempts": 0
        }
        self._initialized = False
        
        # Log component creation
        self._logger.info(f"Component instantiated: {self.__class__.__name__}")
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the component with its specific setup logic.
        Must be implemented by subclasses with comprehensive error handling.
        
        Raises:
            ComponentInitializationError: If initialization fails.
        """
        pass
    
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Update the component's configuration dynamically with comprehensive logging.
        
        Args:
            config (Dict[str, Any]): New configuration parameters.
        """
        old_config = self._config.copy()
        self._config.update(config)
        self._state['last_updated'] = logging.Formatter('%(asctime)s').formatTime(logging.LogRecord('', 0, '', 0, '', (), None))
        
        # Log configuration changes
        changed_keys = set(config.keys()) - set(old_config.keys())
        self._logger.info(f"Configuration updated: {list(changed_keys)}")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Retrieve a comprehensive status of the component.
        
        Returns:
            Dict[str, Any]: Detailed component status and state.
        """
        return {
            "id": self._id,
            "class": self.__class__.__name__,
            "initialized": self._initialized,
            "config_keys": list(self._config.keys()),
            "state": self._state,
            "logger_name": self._logger.name
        }
    
    def log(self, message: str, level: str = 'info') -> None:
        """
        Advanced logging method with multiple severity levels.
        
        Args:
            message (str): Log message.
            level (str, optional): Logging level. Defaults to 'info'.
        """
        log_method = getattr(self._logger, level.lower(), self._logger.info)
        log_method(message)
    
    @abstractmethod
    def validate(self) -> bool:
        """
        Validate the component's current state and configuration.
        Provides a comprehensive validation mechanism.
        
        Returns:
            bool: Whether the component is in a valid state.
        """
        pass 
    
    def __repr__(self) -> str:
        """
        Provide a detailed string representation of the component.
        
        Returns:
            str: Detailed component representation.
        """
        return f"{self.__class__.__name__}(id={self._id[:8]}, initialized={self._initialized})"