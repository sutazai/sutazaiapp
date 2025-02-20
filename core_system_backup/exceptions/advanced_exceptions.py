from typing import Optional, Type, TypeVar, Generic, Any, Dict
import traceback
import logging

ExceptionT = TypeVar('ExceptionT', bound=BaseException)

class AdvancedSystemException(Exception):
    """Base exception for system-wide error handling"""
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        original_exception: Optional[Exception] = None
    ):
        self.message = message
        self.error_code = error_code
        self.original_exception = original_exception
        self.traceback = traceback.format_exc() if original_exception else None
        
        super().__init__(self.message)
        self._log_exception()
    
    def _log_exception(self):
        logging.critical(
            f"System Exception: {self.message}\n"
            f"Error Code: {self.error_code}\n"
            f"Traceback: {self.traceback}"
        )

class ExceptionHandler(Generic[ExceptionT]):
    """Advanced exception handling and management"""
    @classmethod
    def handle(
        cls, 
        exception_type: Type[ExceptionT], 
        message: Optional[str] = None
    ) -> AdvancedSystemException:
        """Standardized exception handling"""
        return AdvancedSystemException(
            message or f"Unhandled {exception_type.__name__}",
            error_code=exception_type.__name__
        )

class ComprehensiveException(Exception):
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.error_code = error_code
        self.context = context or {}
        self._log_exception()

    def _log_exception(self):
        logging.error(
            f"Error Code: {self.error_code}\n"
            f"Message: {str(self)}\n"
            f"Context: {self.context}\n"
            f"Traceback: {traceback.format_exc()}"
        ) 