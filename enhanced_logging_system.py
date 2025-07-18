#!/usr/bin/env python3
"""
Enhanced Logging System for SutazAI
Comprehensive real-time logging and monitoring with UI integration
"""

import logging
import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import threading
import queue
import sys
from contextlib import contextmanager
import streamlit as st
from functools import wraps

class LogLevel:
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

class EnhancedLogger:
    """Advanced logging system with real-time UI integration"""
    
    def __init__(self, name: str = "SutazAI"):
        self.name = name
        self.log_queue = queue.Queue()
        self.log_history = []
        self.max_history = 1000
        self.session_id = f"session_{int(time.time())}"
        
        # Create log directory
        self.log_dir = Path("/opt/sutazaiapp/logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Initialize loggers
        self._setup_loggers()
        
        # Start background log processor
        self._start_log_processor()
        
    def _setup_loggers(self):
        """Setup multiple specialized loggers"""
        
        # Main application logger
        self.app_logger = logging.getLogger(f"{self.name}.app")
        self.app_logger.setLevel(logging.DEBUG)
        
        # API/Backend logger
        self.api_logger = logging.getLogger(f"{self.name}.api")
        self.api_logger.setLevel(logging.DEBUG)
        
        # Frontend/UI logger
        self.ui_logger = logging.getLogger(f"{self.name}.ui")
        self.ui_logger.setLevel(logging.DEBUG)
        
        # System/Performance logger
        self.sys_logger = logging.getLogger(f"{self.name}.system")
        self.sys_logger.setLevel(logging.DEBUG)
        
        # Error/Exception logger
        self.error_logger = logging.getLogger(f"{self.name}.error")
        self.error_logger.setLevel(logging.WARNING)
        
        # Setup file handlers
        self._setup_file_handlers()
        
        # Setup console handlers
        self._setup_console_handlers()
    
    def _setup_file_handlers(self):
        """Setup file handlers for different log types"""
        
        # Formatter with detailed info
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s'
        )
        
        # JSON formatter for structured logs
        json_formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "logger": "%(name)s", "level": "%(levelname)s", '
            '"function": "%(funcName)s", "line": %(lineno)d, "message": "%(message)s"}'
        )
        
        # Application log file
        app_handler = logging.FileHandler(self.log_dir / "sutazai_app.log")
        app_handler.setFormatter(detailed_formatter)
        self.app_logger.addHandler(app_handler)
        
        # API log file
        api_handler = logging.FileHandler(self.log_dir / "sutazai_api.log")
        api_handler.setFormatter(detailed_formatter)
        self.api_logger.addHandler(api_handler)
        
        # UI log file
        ui_handler = logging.FileHandler(self.log_dir / "sutazai_ui.log")
        ui_handler.setFormatter(detailed_formatter)
        self.ui_logger.addHandler(ui_handler)
        
        # System log file
        sys_handler = logging.FileHandler(self.log_dir / "sutazai_system.log")
        sys_handler.setFormatter(detailed_formatter)
        self.sys_logger.addHandler(sys_handler)
        
        # Error log file
        error_handler = logging.FileHandler(self.log_dir / "sutazai_errors.log")
        error_handler.setFormatter(detailed_formatter)
        self.error_logger.addHandler(error_handler)
        
        # Structured JSON log for analysis
        json_handler = logging.FileHandler(self.log_dir / "sutazai_structured.jsonl")
        json_handler.setFormatter(json_formatter)
        
        # Add JSON handler to all loggers
        for logger in [self.app_logger, self.api_logger, self.ui_logger, self.sys_logger, self.error_logger]:
            logger.addHandler(json_handler)
    
    def _setup_console_handlers(self):
        """Setup console handlers with colors"""
        
        class ColoredFormatter(logging.Formatter):
            COLORS = {
                'DEBUG': '\033[36m',     # Cyan
                'INFO': '\033[32m',      # Green
                'WARNING': '\033[33m',   # Yellow
                'ERROR': '\033[31m',     # Red
                'CRITICAL': '\033[35m'   # Magenta
            }
            RESET = '\033[0m'
            
            def format(self, record):
                log_color = self.COLORS.get(record.levelname, self.RESET)
                record.levelname = f"{log_color}{record.levelname}{self.RESET}"
                return super().format(record)
        
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        
        # Add console handler to main logger only
        self.app_logger.addHandler(console_handler)
    
    def _start_log_processor(self):
        """Start background thread to process logs for UI"""
        def process_logs():
            while True:
                try:
                    log_entry = self.log_queue.get(timeout=1)
                    self.log_history.append(log_entry)
                    
                    # Maintain history limit
                    if len(self.log_history) > self.max_history:
                        self.log_history = self.log_history[-self.max_history:]
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"Log processor error: {e}")
        
        processor_thread = threading.Thread(target=process_logs, daemon=True)
        processor_thread.start()
    
    def log(self, level: str, message: str, category: str = "app", **kwargs):
        """Main logging method"""
        timestamp = datetime.now().isoformat()
        
        # Create log entry
        log_entry = {
            "timestamp": timestamp,
            "level": level.upper(),
            "category": category,
            "message": message,
            "session_id": self.session_id,
            "thread_id": threading.get_ident(),
            **kwargs
        }
        
        # Add to queue for UI
        try:
            self.log_queue.put_nowait(log_entry)
        except queue.Full:
            pass  # Drop log if queue is full
        
        # Route to appropriate logger
        logger_map = {
            "app": self.app_logger,
            "api": self.api_logger,
            "ui": self.ui_logger,
            "system": self.sys_logger,
            "error": self.error_logger
        }
        
        logger = logger_map.get(category, self.app_logger)
        level_map = {
            "trace": 5,
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
            "critical": logging.CRITICAL
        }
        
        log_level = level_map.get(level.lower(), logging.INFO)
        logger.log(log_level, message, extra=kwargs)
    
    def trace(self, message: str, category: str = "app", **kwargs):
        self.log("trace", message, category, **kwargs)
    
    def debug(self, message: str, category: str = "app", **kwargs):
        self.log("debug", message, category, **kwargs)
    
    def info(self, message: str, category: str = "app", **kwargs):
        self.log("info", message, category, **kwargs)
    
    def warning(self, message: str, category: str = "app", **kwargs):
        self.log("warning", message, category, **kwargs)
    
    def error(self, message: str, category: str = "error", **kwargs):
        self.log("error", message, category, **kwargs)
    
    def critical(self, message: str, category: str = "error", **kwargs):
        self.log("critical", message, category, **kwargs)
    
    def log_exception(self, exc: Exception, context: str = "", category: str = "error"):
        """Log exception with full traceback"""
        exc_info = {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "context": context
        }
        
        self.error(
            f"Exception in {context}: {type(exc).__name__}: {str(exc)}",
            category=category,
            **exc_info
        )
    
    def get_recent_logs(self, limit: int = 100, level_filter: Optional[str] = None) -> List[Dict]:
        """Get recent log entries for UI display"""
        logs = self.log_history[-limit:] if limit else self.log_history
        
        if level_filter:
            logs = [log for log in logs if log["level"].lower() == level_filter.lower()]
        
        return logs
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get logging statistics"""
        if not self.log_history:
            return {"total": 0, "by_level": {}, "by_category": {}}
        
        by_level = {}
        by_category = {}
        
        for log in self.log_history:
            level = log["level"]
            category = log["category"]
            
            by_level[level] = by_level.get(level, 0) + 1
            by_category[category] = by_category.get(category, 0) + 1
        
        return {
            "total": len(self.log_history),
            "by_level": by_level,
            "by_category": by_category,
            "session_id": self.session_id
        }

# Global logger instance
sutazai_logger = EnhancedLogger("SutazAI")

# Convenience functions
def trace(message: str, category: str = "app", **kwargs):
    sutazai_logger.trace(message, category, **kwargs)

def debug(message: str, category: str = "app", **kwargs):
    sutazai_logger.debug(message, category, **kwargs)

def info(message: str, category: str = "app", **kwargs):
    sutazai_logger.info(message, category, **kwargs)

def warning(message: str, category: str = "app", **kwargs):
    sutazai_logger.warning(message, category, **kwargs)

def error(message: str, category: str = "error", **kwargs):
    sutazai_logger.error(message, category, **kwargs)

def critical(message: str, category: str = "error", **kwargs):
    sutazai_logger.critical(message, category, **kwargs)

def log_exception(exc: Exception, context: str = "", category: str = "error"):
    sutazai_logger.log_exception(exc, context, category)

# Decorators for automatic logging
def log_function_calls(category: str = "app", log_args: bool = False):
    """Decorator to log function calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"
            
            # Log function entry
            entry_msg = f"Entering function: {func_name}"
            if log_args and (args or kwargs):
                entry_msg += f" with args={args[:3]}, kwargs={list(kwargs.keys())}"
            
            debug(entry_msg, category=category, function=func_name)
            
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log successful completion
                debug(
                    f"Function {func_name} completed successfully in {duration:.3f}s",
                    category=category,
                    function=func_name,
                    duration=duration
                )
                
                return result
                
            except Exception as exc:
                duration = time.time() - start_time
                
                # Log exception
                log_exception(
                    exc,
                    context=f"Function {func_name} after {duration:.3f}s",
                    category="error"
                )
                raise
        
        return wrapper
    return decorator

def log_api_calls(category: str = "api"):
    """Decorator specifically for API calls"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_name = f"{func.__module__}.{func.__qualname__}"
            
            info(f"API call started: {func_name}", category=category, api_function=func_name)
            
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                info(
                    f"API call completed: {func_name} in {duration:.3f}s",
                    category=category,
                    api_function=func_name,
                    duration=duration,
                    success=True
                )
                
                return result
                
            except Exception as exc:
                duration = time.time() - start_time
                
                error(
                    f"API call failed: {func_name} after {duration:.3f}s - {str(exc)}",
                    category="error",
                    api_function=func_name,
                    duration=duration,
                    success=False,
                    error_type=type(exc).__name__
                )
                raise
        
        return wrapper
    return decorator

@contextmanager
def log_context(message: str, category: str = "app", **kwargs):
    """Context manager for logging operations"""
    start_time = time.time()
    debug(f"Starting: {message}", category=category, **kwargs)
    
    try:
        yield
        duration = time.time() - start_time
        debug(f"Completed: {message} in {duration:.3f}s", category=category, duration=duration, **kwargs)
    except Exception as exc:
        duration = time.time() - start_time
        log_exception(exc, context=f"{message} after {duration:.3f}s", category="error")
        raise

# Streamlit UI Components for real-time logging
def display_log_viewer():
    """Display real-time log viewer in Streamlit"""
    st.subheader("🔍 Real-Time System Logs")
    
    # Log controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        log_limit = st.selectbox("Show entries", [50, 100, 200, 500], index=1)
    
    with col2:
        level_filter = st.selectbox(
            "Filter by level", 
            ["All", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        )
    
    with col3:
        category_filter = st.selectbox(
            "Filter by category",
            ["All", "app", "api", "ui", "system", "error"]
        )
    
    # Get filtered logs
    level_filter = None if level_filter == "All" else level_filter
    recent_logs = sutazai_logger.get_recent_logs(limit=log_limit, level_filter=level_filter)
    
    if category_filter != "All":
        recent_logs = [log for log in recent_logs if log["category"] == category_filter]
    
    # Display logs
    if recent_logs:
        # Log statistics
        stats = sutazai_logger.get_log_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Logs", stats["total"])
        col2.metric("Errors", stats["by_level"].get("ERROR", 0))
        col3.metric("Warnings", stats["by_level"].get("WARNING", 0))
        col4.metric("Session", stats["session_id"][-8:])
        
        # Log entries
        for log_entry in reversed(recent_logs[-50:]):  # Show most recent first
            level = log_entry["level"]
            
            # Color coding
            if level == "ERROR":
                st.error(f"🔴 **{log_entry['timestamp']}** | {log_entry['category'].upper()} | {log_entry['message']}")
            elif level == "WARNING":
                st.warning(f"🟡 **{log_entry['timestamp']}** | {log_entry['category'].upper()} | {log_entry['message']}")
            elif level == "INFO":
                st.info(f"🔵 **{log_entry['timestamp']}** | {log_entry['category'].upper()} | {log_entry['message']}")
            else:
                st.text(f"⚪ **{log_entry['timestamp']}** | {log_entry['category'].upper()} | {log_entry['message']}")
    else:
        st.info("No logs available with current filters")
    
    # Auto-refresh
    if st.button("🔄 Refresh Logs"):
        st.rerun()

def display_log_stats():
    """Display logging statistics"""
    stats = sutazai_logger.get_log_stats()
    
    if stats["total"] > 0:
        st.subheader("📊 Logging Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**By Level:**")
            for level, count in stats["by_level"].items():
                st.write(f"- {level}: {count}")
        
        with col2:
            st.write("**By Category:**")
            for category, count in stats["by_category"].items():
                st.write(f"- {category}: {count}")

# Initialize logging for the current module
info("Enhanced logging system initialized", category="system")