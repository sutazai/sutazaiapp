"""Error Handler for SutazAI"""
import logging
import traceback
from typing import Dict, Any
from functools import wraps

logger = logging.getLogger(__name__)

class ErrorHandler:
    def __init__(self):
        self.error_count = 0
        self.error_history = []
    
    def handle_error(self, error: Exception, context: str = None):
        """Handle errors with logging"""
        self.error_count += 1
        
        error_info = {
            "error": str(error),
            "type": type(error).__name__,
            "context": context,
            "timestamp": time.time(),
            "traceback": traceback.format_exc()
        }
        
        self.error_history.append(error_info)
        logger.error(f"Error in {context}: {error}")
        
        return error_info
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        return {
            "total_errors": self.error_count,
            "recent_errors": self.error_history[-5:]
        }

# Global instance
error_handler = ErrorHandler()

def handle_errors(context: str = None):
    """Decorator for error handling"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler.handle_error(e, context or func.__name__)
                raise
        return wrapper
    return decorator
