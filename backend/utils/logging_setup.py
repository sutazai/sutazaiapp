"""
Logging Setup Module for SutazAI

This module provides standardized logging configuration across the SutazAI system,
with support for structured logging, metrics collection, and integration with
monitoring systems like ELK Stack and Prometheus.
"""

import os
import sys
import json
import time
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
import socket
import threading
import atexit
from typing import Dict, Any, Optional, Union

# Try to import optional dependencies
try:
    import structlog
    from structlog.contextvars import (
        bind_contextvars,
        bound_contextvars,
        clear_contextvars,
        merge_contextvars,
        unbind_contextvars,
    )

    STRUCTLOG_AVAILABLE = True
except ImportError:
    STRUCTLOG_AVAILABLE = False

try:
    import prometheus_client
    from prometheus_client import Counter, Histogram, Gauge, push_to_gateway, REGISTRY

    PROMETHEUS_AVAILABLE = True
except ImportError:
    prometheus_client = None
    PROMETHEUS_AVAILABLE = False

try:
    import sentry_sdk
    from sentry_sdk.integrations.logging import LoggingIntegration
    from sentry_sdk.integrations.fastapi import FastApiIntegration
    from sentry_sdk.integrations.sqlalchemy import SqlalchemyIntegration

    SENTRY_AVAILABLE = True
except ImportError:
    sentry_sdk = None
    SENTRY_AVAILABLE = False

# Define log directory
LOG_DIR = os.environ.get("LOG_DIR", "/opt/sutazaiapp/logs")
Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

# Define metrics directory
METRICS_DIR = os.environ.get("METRICS_DIR", "/opt/sutazaiapp/metrics")
Path(METRICS_DIR).mkdir(parents=True, exist_ok=True)

# Environment and security settings
ENV = os.environ.get("ENVIRONMENT", "development")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
PUSH_GATEWAY_URL = os.environ.get("PUSH_GATEWAY_URL", None)
SENTRY_DSN = os.environ.get("SENTRY_DSN", None)
HOSTNAME = socket.gethostname()
REDACT_FIELDS = ["password", "token", "api_key", "secret", "authorization"]

# Prometheus metrics
if PROMETHEUS_AVAILABLE:
    # Define common metrics
    log_events = Counter(
        "sutazai_log_events_total", "Total number of log events", ["component", "level"]
    )

    request_duration = Histogram(
        "sutazai_request_duration_seconds",
        "Request duration in seconds",
        ["component", "endpoint"],
    )

    model_inference_duration = Histogram(
        "sutazai_model_inference_duration_seconds",
        "Model inference duration in seconds",
        ["model_name", "operation"],
    )

    model_error_counter = Counter(
        "sutazai_model_errors_total",
        "Total number of model errors",
        ["model_name", "error_type"],
    )

    system_memory_usage = Gauge(
        "sutazai_memory_usage_bytes", "Memory usage in bytes", ["component"]
    )

# Initialize Sentry if available
if SENTRY_AVAILABLE and SENTRY_DSN:
    sentry_sdk.init(
        dsn=SENTRY_DSN, environment=ENV, traces_sample_rate=0.1, send_default_pii=False
    )

# Sentry integrations list
SENTRY_INTEGRATIONS = []
if SENTRY_AVAILABLE:
    SENTRY_INTEGRATIONS.extend([
        LoggingIntegration(event_level=logging.WARNING),
        FastApiIntegration(),
        SqlalchemyIntegration(),
    ])

# Helper functions
def _sanitize_log_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Redact sensitive information from log data."""
    if not isinstance(data, dict):
        # If data is not a dict, return it as is (or handle differently if needed)
        return data # Ensure non-dict data is returned

    sanitized: Dict[str, Any] = {}
    for key, value in data.items():
        if any(field in key.lower() for field in REDACT_FIELDS):
            sanitized[key] = "***REDACTED***"
        elif isinstance(value, dict):
            sanitized[key] = _sanitize_log_data(value)
        else:
            sanitized[key] = value
    return sanitized


def _get_log_level(level_name: str) -> int:
    """Convert string log level to logging module constant."""
    return getattr(logging, level_name.upper(), logging.INFO)


def setup_structlog() -> None:
    """Configure structlog if available."""
    if not STRUCTLOG_AVAILABLE:
        return

    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ]

    structlog.configure(
        processors=shared_processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


# Setup structlog if available
if STRUCTLOG_AVAILABLE:
    setup_structlog()

# Setup periodic metrics reporting
if PROMETHEUS_AVAILABLE and PUSH_GATEWAY_URL:

    def _push_metrics():
        try:
            push_to_gateway(PUSH_GATEWAY_URL, job="sutazai", registry=REGISTRY)
        except Exception as e:
            # Can't use our logger here to avoid circular logging
            print(f"Error pushing metrics: {e}")

    # Set up periodic push (every 15 seconds)
    def _periodic_push():
        while True:
            _push_metrics()
            time.sleep(15)

    # Start push thread
    push_thread = threading.Thread(target=_periodic_push, daemon=True)
    push_thread.start()

    # Register push at exit
    atexit.register(_push_metrics)


def setup_logger(
    name: str, log_file: Optional[str] = None, level: Optional[Union[int, str]] = None
) -> logging.Logger:
    """
    Set up and configure a logger with console and file handlers.

    Args:
        name: Logger name
        log_file: File to write logs to
        level: Logging level

    Returns:
        Configured logger
    """
    if isinstance(level, str):
        level = _get_log_level(level)
    elif level is None:
        level = _get_log_level(LOG_LEVEL)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatters
    simple_formatter = logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Custom JSON formatter
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_data = {
                "timestamp": datetime.now().isoformat(),
                "logger": record.name,
                "level": record.levelname,
                "message": record.getMessage(),
                "host": HOSTNAME,
                "environment": ENV,
                "path": record.pathname,
                "line": record.lineno,
            }
            if record.exc_info:
                log_data["exception"] = self.formatException(record.exc_info)
            return json.dumps(log_data)

    json_formatter = JsonFormatter()

    # Console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(simple_formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        if not log_file.startswith("/"):
            log_file = os.path.join(LOG_DIR, log_file)

        # Regular rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10485760,  # 10MB
            backupCount=10,
        )
        file_handler.setFormatter(simple_formatter)
        logger.addHandler(file_handler)

        # JSON file handler for ELK integration
        json_log_file = log_file.replace(".log", ".json.log")
        json_handler = logging.handlers.RotatingFileHandler(
            json_log_file,
            maxBytes=10485760,  # 10MB
            backupCount=10,
        )
        json_handler.setFormatter(json_formatter)
        logger.addHandler(json_handler)

    # Add a Sentry handler for errors in production
    if SENTRY_AVAILABLE and SENTRY_DSN and ENV == "production":

        class SentryHandler(logging.Handler):
            def emit(self, record):
                if record.levelno >= logging.ERROR:
                    sentry_sdk.capture_exception()

        logger.addHandler(SentryHandler())

    # Add Prometheus metrics
    if PROMETHEUS_AVAILABLE:

        class PrometheusHandler(logging.Handler):
            def emit(self, record):
                log_events.labels(component=record.name, level=record.levelname).inc()

        logger.addHandler(PrometheusHandler())

    return logger


def get_structured_logger(name: str, log_file: Optional[str] = None) -> Any:
    """
    Get a structured logger if structlog is available, otherwise a standard logger.

    Args:
        name: Logger name
        log_file: File to write logs to

    Returns:
        A structured logger or standard logger
    """
    if STRUCTLOG_AVAILABLE:
        # Set up the underlying stdlib logger
        setup_logger(name, log_file)
        # Create and return a structlog logger
        return structlog.get_logger(name)
    else:
        # Fall back to standard logger
        return setup_logger(name, log_file)


# Convenience functions for different component loggers
def get_app_logger():
    """Get the main application logger."""
    return setup_logger("sutazai_app", "app.log")


def get_model_logger(model_name):
    """Get a logger for a specific model."""
    return setup_logger(f"model.{model_name}", "model.log")


def get_agent_logger(agent_id):
    """Get a logger for a specific agent."""
    return setup_logger(f"agent.{agent_id}", "agent.log")


def get_api_logger():
    """Get the API logger."""
    return setup_logger("sutazai_api", "api.log")


def get_security_logger():
    """Get the security logger."""
    return setup_logger("sutazai_security", "security.log", level=logging.WARNING)


def log_metrics(component, metrics):
    """
    Log metrics to a JSON file and update Prometheus metrics if available.

    Args:
        component: Component name
        metrics: Dictionary of metric values
    """
    logger = setup_logger(f"metrics.{component}", "metrics.log")
    metrics_data = {
        "timestamp": datetime.now().isoformat(),
        "component": component,
        "metrics": metrics,
    }
    logger.info(json.dumps(metrics_data))

    # If Prometheus is available, update relevant metrics
    if PROMETHEUS_AVAILABLE:
        for key, value in metrics.items():
            # For certain known metrics, update Prometheus
            if key == "memory_usage_bytes" and isinstance(value, (int, float)):
                system_memory_usage.labels(component=component).set(value)


def log_request(component, endpoint, duration, status_code=200, error=None):
    """
    Log API request information and update metrics.

    Args:
        component: Component name
        endpoint: API endpoint
        duration: Request duration in seconds
        status_code: HTTP status code
        error: Error message if any
    """
    logger = get_api_logger()

    log_data = {
        "component": component,
        "endpoint": endpoint,
        "duration": duration,
        "status_code": status_code,
    }

    if error:
        log_data["error"] = str(error)
        logger.error(f"API request error: {json.dumps(log_data)}")
    else:
        logger.info(f"API request completed: {json.dumps(log_data)}")

    # Update Prometheus metrics if available
    if PROMETHEUS_AVAILABLE:
        request_duration.labels(component=component, endpoint=endpoint).observe(
            duration
        )


def log_model_inference(model_name, operation, duration, error=None):
    """
    Log model inference information and update metrics.

    Args:
        model_name: Name of the model
        operation: Operation performed (e.g., 'generate', 'embed')
        duration: Inference duration in seconds
        error: Error message if any
    """
    logger = get_model_logger(model_name)

    log_data = {"model": model_name, "operation": operation, "duration": duration}

    if error:
        log_data["error"] = str(error)
        logger.error(f"Model inference error: {json.dumps(log_data)}")

        # Update error metrics if Prometheus is available
        if PROMETHEUS_AVAILABLE:
            model_error_counter.labels(
                model_name=model_name, error_type=type(error).__name__
            ).inc()
    else:
        logger.info(f"Model inference completed: {json.dumps(log_data)}")

    # Update duration metrics if Prometheus is available
    if PROMETHEUS_AVAILABLE:
        model_inference_duration.labels(
            model_name=model_name, operation=operation
        ).observe(duration)


def capture_exception(exc, context=None):
    """
    Capture an exception for logging and monitoring.

    Args:
        exc: The exception
        context: Additional context information
    """
    # Log the exception
    logger = get_app_logger()

    ctx_str = ""
    if context:
        sanitized_context = _sanitize_log_data(context)
        ctx_str = f" Context: {json.dumps(sanitized_context)}"

    logger.exception(f"Exception: {str(exc)}.{ctx_str}")

    # Send to Sentry if available
    if SENTRY_AVAILABLE and SENTRY_DSN:
        if context:
            with sentry_sdk.push_scope() as scope:
                for key, value in sanitized_context.items():
                    scope.set_extra(key, value)
                sentry_sdk.capture_exception(exc)
        else:
            sentry_sdk.capture_exception(exc)
