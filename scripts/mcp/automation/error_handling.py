#!/usr/bin/env python3
"""
MCP Automation Error Handling

Comprehensive error handling, retry logic, and recovery mechanisms for the
MCP automation system. Provides structured error tracking, automatic recovery,
and detailed audit trails following Enforcement Rules compliance.

Author: Claude AI Assistant (python-architect.md)
Created: 2025-08-15 11:45:00 UTC
Version: 1.0.0
"""

import asyncio
import functools
import traceback
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Type, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from config import get_config, MCPAutomationConfig


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    SECURITY = "security"


class ErrorCategory(Enum):
    """Error category types."""
    NETWORK = "network"
    FILESYSTEM = "filesystem"
    VALIDATION = "validation"
    SECURITY = "security"
    CONFIGURATION = "configuration"
    EXTERNAL_DEPENDENCY = "external_dependency"
    INTERNAL_LOGIC = "internal_logic"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    TIMEOUT = "timeout"
    PERMISSION = "permission"


class RecoveryAction(Enum):
    """Possible recovery actions."""
    RETRY = "retry"
    FALLBACK = "fallback"
    ROLLBACK = "rollback"
    SKIP = "skip"
    ESCALATE = "escalate"
    MANUAL_INTERVENTION = "manual_intervention"
    SYSTEM_RESTART = "system_restart"


@dataclass
class ErrorRecord:
    """Structured error record for tracking and analysis."""
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    function_name: str
    error_type: str
    error_message: str
    stack_trace: str
    context: Dict[str, Any]
    recovery_action: Optional[RecoveryAction] = None
    retry_count: int = 0
    resolved: bool = False
    resolution_timestamp: Optional[datetime] = None
    resolution_notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['severity'] = self.severity.value
        result['category'] = self.category.value
        if self.recovery_action:
            result['recovery_action'] = self.recovery_action.value
        if self.resolution_timestamp:
            result['resolution_timestamp'] = self.resolution_timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ErrorRecord':
        """Create from dictionary."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['severity'] = ErrorSeverity(data['severity'])
        data['category'] = ErrorCategory(data['category'])
        if data.get('recovery_action'):
            data['recovery_action'] = RecoveryAction(data['recovery_action'])
        if data.get('resolution_timestamp'):
            data['resolution_timestamp'] = datetime.fromisoformat(data['resolution_timestamp'])
        return cls(**data)


class MCPError(Exception):
    """Base exception for MCP automation system."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 category: ErrorCategory = ErrorCategory.INTERNAL_LOGIC,
                 context: Optional[Dict[str, Any]] = None,
                 recovery_action: Optional[RecoveryAction] = None):
        super().__init__(message)
        self.severity = severity
        self.category = category
        self.context = context or {}
        self.recovery_action = recovery_action


class MCPConfigurationError(MCPError):
    """Configuration-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.CONFIGURATION, **kwargs)


class MCPNetworkError(MCPError):
    """Network-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.NETWORK, **kwargs)


class MCPSecurityError(MCPError):
    """Security-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, 
                        severity=ErrorSeverity.SECURITY,
                        category=ErrorCategory.SECURITY, 
                        **kwargs)


class MCPValidationError(MCPError):
    """Validation-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.VALIDATION, **kwargs)


class MCPTimeoutError(MCPError):
    """Timeout-related errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, category=ErrorCategory.TIMEOUT, **kwargs)


class MCPResourceError(MCPError):
    """Resource exhaustion errors."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(message, 
                        severity=ErrorSeverity.HIGH,
                        category=ErrorCategory.RESOURCE_EXHAUSTION, 
                        **kwargs)


class ErrorTracker:
    """
    Comprehensive error tracking and analysis system.
    
    Tracks all errors, provides analytics, and manages recovery operations
    with detailed audit trails and pattern recognition.
    """
    
    def __init__(self, config: Optional[MCPAutomationConfig] = None):
        """
        Initialize error tracker.
        
        Args:
            config: Optional configuration override
        """
        self.config = config or get_config()
        self.logger = self._setup_logging()
        
        # Error storage
        self.error_records: List[ErrorRecord] = []
        self.error_patterns: Dict[str, int] = {}
        
        # Recovery statistics
        self.recovery_stats = {
            'total_errors': 0,
            'successful_recoveries': 0,
            'failed_recoveries': 0,
            'manual_interventions': 0,
            'system_restarts': 0
        }
        
        # Load existing error history
        self._load_error_history()
        
        self.logger.info("Error tracker initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup error tracking logging."""
        logger = logging.getLogger(f"{__name__}.ErrorTracker")
        logger.setLevel(getattr(logging, self.config.log_level.value))
        
        if not logger.handlers:
            # Create error logs directory
            error_log_dir = self.config.paths.logs_root / "errors"
            error_log_dir.mkdir(parents=True, exist_ok=True)
            
            # File handler for error logs
            log_file = error_log_dir / f"error_tracker_{datetime.now().strftime('%Y%m%d')}.log"
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
            
            # Console handler for critical errors
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.ERROR)
            console_formatter = logging.Formatter(
                '%(asctime)s - ERROR - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
        
        return logger
    
    def record_error(self, error: Exception, component: str, function_name: str,
                    context: Optional[Dict[str, Any]] = None,
                    severity: Optional[ErrorSeverity] = None,
                    category: Optional[ErrorCategory] = None) -> str:
        """
        Record an error with comprehensive details.
        
        Args:
            error: Exception that occurred
            component: Component where error occurred
            function_name: Function where error occurred
            context: Additional context information
            severity: Error severity (auto-detected if None)
            category: Error category (auto-detected if None)
            
        Returns:
            Error ID for tracking
        """
        try:
            # Generate error ID
            error_id = f"error_{int(datetime.now().timestamp())}_{hash(str(error)) % 10000:04d}"
            
            # Auto-detect severity and category if not provided
            if isinstance(error, MCPError):
                detected_severity = error.severity
                detected_category = error.category
                error_context = error.context
            else:
                detected_severity = self._detect_severity(error)
                detected_category = self._detect_category(error)
                error_context = {}
            
            final_severity = severity or detected_severity
            final_category = category or detected_category
            
            # Merge context
            full_context = {**(context or {}), **error_context}
            
            # Create error record
            error_record = ErrorRecord(
                error_id=error_id,
                timestamp=datetime.now(timezone.utc),
                severity=final_severity,
                category=final_category,
                component=component,
                function_name=function_name,
                error_type=type(error).__name__,
                error_message=str(error),
                stack_trace=traceback.format_exc(),
                context=full_context
            )
            
            # Store error record
            self.error_records.append(error_record)
            
            # Update statistics
            self.recovery_stats['total_errors'] += 1
            
            # Track error patterns
            pattern_key = f"{component}:{type(error).__name__}"
            self.error_patterns[pattern_key] = self.error_patterns.get(pattern_key, 0) + 1
            
            # Log error
            self.logger.error(
                f"Error recorded: {error_id} - {final_severity.value.upper()} - "
                f"{component}.{function_name}: {str(error)}"
            )
            
            # Save error history
            self._save_error_history()
            
            # Check for critical errors requiring immediate action
            if final_severity in [ErrorSeverity.CRITICAL, ErrorSeverity.SECURITY]:
                self._handle_critical_error(error_record)
            
            return error_id
        
        except Exception as e:
            # Don't let error tracking itself fail
            self.logger.critical(f"Failed to record error: {e}")
            return f"error_tracking_failed_{int(datetime.now().timestamp())}"
    
    def mark_error_resolved(self, error_id: str, resolution_notes: str):
        """Mark an error as resolved."""
        try:
            for record in self.error_records:
                if record.error_id == error_id:
                    record.resolved = True
                    record.resolution_timestamp = datetime.now(timezone.utc)
                    record.resolution_notes = resolution_notes
                    
                    self.logger.info(f"Error {error_id} marked as resolved: {resolution_notes}")
                    self._save_error_history()
                    break
        
        except Exception as e:
            self.logger.error(f"Failed to mark error as resolved: {e}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        try:
            total_errors = len(self.error_records)
            if total_errors == 0:
                return {'total_errors': 0}
            
            # Count by severity
            severity_counts = {}
            for severity in ErrorSeverity:
                severity_counts[severity.value] = len([
                    r for r in self.error_records if r.severity == severity
                ])
            
            # Count by category
            category_counts = {}
            for category in ErrorCategory:
                category_counts[category.value] = len([
                    r for r in self.error_records if r.category == category
                ])
            
            # Recent errors (last 24 hours)
            recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            recent_errors = [r for r in self.error_records if r.timestamp > recent_cutoff]
            
            # Resolution rate
            resolved_errors = [r for r in self.error_records if r.resolved]
            resolution_rate = (len(resolved_errors) / total_errors * 100) if total_errors > 0 else 0
            
            return {
                'total_errors': total_errors,
                'severity_distribution': severity_counts,
                'category_distribution': category_counts,
                'recent_errors_24h': len(recent_errors),
                'resolution_rate_percent': round(resolution_rate, 2),
                'error_patterns': dict(sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)[:10]),
                'recovery_statistics': self.recovery_stats,
                'most_problematic_components': self._get_problematic_components()
            }
        
        except Exception as e:
            self.logger.error(f"Failed to generate error statistics: {e}")
            return {'error': 'Failed to generate statistics'}
    
    def _detect_severity(self, error: Exception) -> ErrorSeverity:
        """Auto-detect error severity based on exception type."""
        if isinstance(error, (PermissionError, OSError)):
            return ErrorSeverity.HIGH
        elif isinstance(error, (ConnectionError, TimeoutError)):
            return ErrorSeverity.MEDIUM
        elif isinstance(error, (ValueError, TypeError)):
            return ErrorSeverity.LOW
        elif isinstance(error, KeyboardInterrupt):
            return ErrorSeverity.LOW
        else:
            return ErrorSeverity.MEDIUM
    
    def _detect_category(self, error: Exception) -> ErrorCategory:
        """Auto-detect error category based on exception type."""
        if isinstance(error, (ConnectionError, OSError)):
            return ErrorCategory.NETWORK
        elif isinstance(error, (PermissionError,)):
            return ErrorCategory.PERMISSION
        elif isinstance(error, (TimeoutError, asyncio.TimeoutError)):
            return ErrorCategory.TIMEOUT
        elif isinstance(error, (ValueError, TypeError)):
            return ErrorCategory.VALIDATION
        elif isinstance(error, (FileNotFoundError, IsADirectoryError)):
            return ErrorCategory.FILESYSTEM
        else:
            return ErrorCategory.INTERNAL_LOGIC
    
    def _handle_critical_error(self, error_record: ErrorRecord):
        """Handle critical errors requiring immediate attention."""
        try:
            self.logger.critical(
                f"CRITICAL ERROR DETECTED: {error_record.error_id} - "
                f"{error_record.component}.{error_record.function_name}: {error_record.error_message}"
            )
            
            # Could integrate with alerting systems here
            # For example: send webhook, email, or Slack notification
            
            # Log to separate critical error file
            critical_log_file = self.config.paths.logs_root / "critical_errors.log"
            with open(critical_log_file, 'a') as f:
                f.write(f"{datetime.now().isoformat()} - {json.dumps(error_record.to_dict(), indent=2)}\\n")
        
        except Exception as e:
            self.logger.error(f"Failed to handle critical error: {e}")
    
    def _get_problematic_components(self) -> List[Dict[str, Any]]:
        """Get components with the most errors."""
        try:
            component_errors = {}
            for record in self.error_records:
                component = record.component
                if component not in component_errors:
                    component_errors[component] = {
                        'total_errors': 0,
                        'critical_errors': 0,
                        'recent_errors': 0
                    }
                
                component_errors[component]['total_errors'] += 1
                
                if record.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.SECURITY]:
                    component_errors[component]['critical_errors'] += 1
                
                # Check if error is recent (last 24 hours)
                recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
                if record.timestamp > recent_cutoff:
                    component_errors[component]['recent_errors'] += 1
            
            # Sort by total errors
            sorted_components = sorted(
                component_errors.items(),
                key=lambda x: x[1]['total_errors'],
                reverse=True
            )
            
            return [
                {'component': comp, **stats}
                for comp, stats in sorted_components[:5]  # Top 5
            ]
        
        except Exception as e:
            self.logger.error(f"Failed to get problematic components: {e}")
            return []
    
    def _load_error_history(self):
        """Load error history from storage."""
        try:
            error_file = self.config.paths.automation_root / "error_history.json"
            if error_file.exists():
                with open(error_file, 'r') as f:
                    data = json.load(f)
                
                self.error_records = [ErrorRecord.from_dict(record) for record in data.get('error_records', [])]
                self.error_patterns = data.get('error_patterns', {})
                self.recovery_stats = data.get('recovery_stats', self.recovery_stats)
                
                self.logger.debug(f"Loaded {len(self.error_records)} error records from history")
        
        except Exception as e:
            self.logger.warning(f"Failed to load error history: {e}")
            self.error_records = []
            self.error_patterns = {}
    
    def _save_error_history(self):
        """Save error history to storage."""
        try:
            error_file = self.config.paths.automation_root / "error_history.json"
            
            # Keep only last 1000 error records
            recent_records = self.error_records[-1000:] if len(self.error_records) > 1000 else self.error_records
            
            data = {
                'error_records': [record.to_dict() for record in recent_records],
                'error_patterns': self.error_patterns,
                'recovery_stats': self.recovery_stats,
                'last_save': datetime.now(timezone.utc).isoformat()
            }
            
            # Atomic write
            temp_file = error_file.with_suffix('.tmp')
            with open(temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            temp_file.replace(error_file)
        
        except Exception as e:
            self.logger.error(f"Failed to save error history: {e}")


def with_error_handling(component: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                       category: ErrorCategory = ErrorCategory.INTERNAL_LOGIC,
                       retry_count: int = 0, retry_delay: float = 1.0,
                       fallback_value: Any = None):
    """
    Decorator for comprehensive error handling with retry logic.
    
    Args:
        component: Component name for error tracking
        severity: Default error severity
        category: Default error category
        retry_count: Number of retry attempts
        retry_delay: Delay between retries in seconds
        fallback_value: Value to return on failure
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            config = get_config()
            error_tracker = ErrorTracker(config)
            
            for attempt in range(retry_count + 1):
                try:
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
                
                except Exception as e:
                    error_id = error_tracker.record_error(
                        error=e,
                        component=component,
                        function_name=func.__name__,
                        context={
                            'attempt': attempt + 1,
                            'max_attempts': retry_count + 1,
                            'args': str(args)[:200],  # Truncate for readability
                            'kwargs': str(kwargs)[:200]
                        },
                        severity=severity,
                        category=category
                    )
                    
                    # If this is the last attempt, raise the error
                    if attempt == retry_count:
                        if fallback_value is not None:
                            logger = logging.getLogger(component)
                            logger.warning(f"Returning fallback value after {retry_count + 1} failed attempts: {error_id}")
                            return fallback_value
                        else:
                            raise
                    
                    # Wait before retry
                    if retry_delay > 0:
                        await asyncio.sleep(retry_delay * (attempt + 1))  # Exponential backoff
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            config = get_config()
            error_tracker = ErrorTracker(config)
            
            for attempt in range(retry_count + 1):
                try:
                    return func(*args, **kwargs)
                
                except Exception as e:
                    error_id = error_tracker.record_error(
                        error=e,
                        component=component,
                        function_name=func.__name__,
                        context={
                            'attempt': attempt + 1,
                            'max_attempts': retry_count + 1,
                            'args': str(args)[:200],
                            'kwargs': str(kwargs)[:200]
                        },
                        severity=severity,
                        category=category
                    )
                    
                    # If this is the last attempt, raise the error
                    if attempt == retry_count:
                        if fallback_value is not None:
                            logger = logging.getLogger(component)
                            logger.warning(f"Returning fallback value after {retry_count + 1} failed attempts: {error_id}")
                            return fallback_value
                        else:
                            raise
                    
                    # Wait before retry
                    if retry_delay > 0:
                        import time
                        time.sleep(retry_delay * (attempt + 1))
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# Global error tracker instance
_error_tracker_instance: Optional[ErrorTracker] = None


def get_error_tracker(config: Optional[MCPAutomationConfig] = None) -> ErrorTracker:
    """Get global error tracker instance (singleton pattern)."""
    global _error_tracker_instance
    
    if _error_tracker_instance is None:
        _error_tracker_instance = ErrorTracker(config)
    
    return _error_tracker_instance


if __name__ == "__main__":
    # Error handling testing
    async def test_error_handling():
        tracker = get_error_tracker()
        
        # Test error recording
        try:
            raise ValueError("Test error for demonstration")
        except Exception as e:
            error_id = tracker.record_error(e, "test_component", "test_function")
            print(f"Recorded error: {error_id}")
        
        # Test statistics
        stats = tracker.get_error_statistics()
        print(f"Error statistics: {json.dumps(stats, indent=2)}")
    
    asyncio.run(test_error_handling())