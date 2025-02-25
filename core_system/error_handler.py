import logging
import traceback
from functools import wraps
from typing import Callable


class SutazAiErrorHandler:
    @staticmethod
    def handle_system_errors(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"System error in {func.__name__}: {e}")
                logging.error(traceback.format_exc())
                # Potentially trigger error recovery or notification
                raise

        return wrapper

    @staticmethod
    def log_error(error: Exception, context: dict = None):
        """Advanced error logging with context"""
        logging.error(f"Error: {error}")
        if context:
            logging.error(f"Context: {context}")

        # Optional: Send error to monitoring system
        # self._send_error_to_monitoring(error, context)
