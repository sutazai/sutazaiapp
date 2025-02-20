import os
import sys
import logging
import traceback
from typing import Any, Callable, Optional, Dict

class SutazAIErrorHandler:
    """
    Centralized error handling and logging framework for SutazAI.
    Provides advanced error tracking, reporting, and remediation capabilities.
    """

    def __init__(self, log_dir: str = 'logs', log_level: int = logging.INFO):
        """
        Initialize the error handler with configurable logging.

        Args:
            log_dir (str): Directory to store log files
            log_level (int): Logging level (default: logging.INFO)
        """
        self.log_dir = os.path.abspath(log_dir)
        os.makedirs(self.log_dir, exist_ok=True)

        # Configure main logger
        self.logger = logging.getLogger('SutazAI')
        self.logger.setLevel(log_level)

        # File handler for persistent logging
        file_handler = logging.FileHandler(
            os.path.join(self.log_dir, 'sutazai_errors.log')
        )
        file_handler.setLevel(log_level)

        # Console handler for immediate visibility
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Formatter for detailed logging
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log_error(
        self, 
        message: str, 
        error: Optional[Exception] = None, 
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log an error with optional exception and context.

        Args:
            message (str): Error message
            error (Optional[Exception]): Exception object
            context (Optional[Dict[str, Any]]): Additional context information
        """
        error_details = {
            'message': message,
            'context': context or {}
        }

        if error:
            error_details.update({
                'exception_type': type(error).__name__,
                'exception_message': str(error),
                'traceback': traceback.format_exc()
            })

        self.logger.error(str(error_details))

    def safe_execute(
        self, 
        func: Callable[..., Any], 
        *args: Any, 
        **kwargs: Any
    ) -> Optional[Any]:
        """
        Execute a function with comprehensive error handling.

        Args:
            func (Callable): Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function

        Returns:
            Optional result of the function or None if an error occurs
        """
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.log_error(
                f"Error executing {func.__name__}", 
                error=e, 
                context={
                    'args': args,
                    'kwargs': kwargs
                }
            )
            return None

    def generate_error_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive error report.

        Returns:
            Dict[str, Any]: Detailed error report
        """
        try:
            with open(os.path.join(self.log_dir, 'sutazai_errors.log'), 'r') as f:
                log_contents = f.readlines()

            return {
                'total_errors': len(log_contents),
                'recent_errors': log_contents[-10:] if log_contents else [],
                'log_path': os.path.join(self.log_dir, 'sutazai_errors.log')
            }
        except Exception as e:
            self.log_error("Could not generate error report", error=e)
            return {}

def global_exception_handler(exc_type, exc_value, exc_traceback):
    """
    Global exception handler for unhandled exceptions.

    Args:
        exc_type: Exception type
        exc_value: Exception value
        exc_traceback: Traceback object
    """
    error_handler = SutazAIErrorHandler()
    error_handler.log_error(
        "Unhandled exception",
        error=exc_value,
        context={
            'type': exc_type.__name__,
            'traceback': traceback.format_tb(exc_traceback)
        }
    )

# Set global exception handler
sys.excepthook = global_exception_handler

def main():
    error_handler = SutazAIErrorHandler()
    report = error_handler.generate_error_report()
    print("Error Report:", report)

if __name__ == '__main__':
    main() 