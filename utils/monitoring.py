"""
Monitoring Module for SutazAI

This module provides monitoring middleware and utilities for the SutazAI system.
"""

import time
import logging
import os
from typing import Callable, Dict, Any, Optional

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

# Try to import optional dependencies
try:
    from prometheus_client import Counter, Histogram, Gauge, Summary

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Import our logging setup
from utils.logging_setup import log_request, capture_exception

# Prometheus metrics if available
if PROMETHEUS_AVAILABLE:
    REQUEST_COUNT = Counter(
        "sutazai_http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "status_code"],
    )

    REQUEST_LATENCY = Histogram(
        "sutazai_http_request_duration_seconds",
        "HTTP request latency in seconds",
        ["method", "endpoint"],
    )

    REQUEST_SIZE = Summary(
        "sutazai_http_request_size_bytes",
        "HTTP request size in bytes",
        ["method", "endpoint"],
    )

    RESPONSE_SIZE = Summary(
        "sutazai_http_response_size_bytes",
        "HTTP response size in bytes",
        ["method", "endpoint", "status_code"],
    )

    if PSUTIL_AVAILABLE:
        PROCESS_MEMORY = Gauge(
            "sutazai_process_memory_bytes", "Memory usage of the process in bytes"
        )

        PROCESS_CPU_SECONDS = Counter(
            "sutazai_process_cpu_seconds_total",
            "CPU time spent by the process in seconds",
        )


# System metrics gathering
def gather_system_metrics() -> Dict[str, Any]:
    """Gather system metrics like CPU and memory usage."""
    metrics = {}

    if PSUTIL_AVAILABLE:
        process = psutil.Process(os.getpid())

        # Memory usage
        memory_info = process.memory_info()
        metrics["memory_usage_bytes"] = memory_info.rss
        metrics["memory_vms_bytes"] = memory_info.vms

        # CPU usage
        metrics["cpu_percent"] = process.cpu_percent()

        # System-wide metrics
        metrics["system_cpu_percent"] = psutil.cpu_percent()
        metrics["system_memory_percent"] = psutil.virtual_memory().percent

        # Update Prometheus metrics if available
        if PROMETHEUS_AVAILABLE:
            PROCESS_MEMORY.set(memory_info.rss)
            PROCESS_CPU_SECONDS.inc(
                process.cpu_times().user + process.cpu_times().system
            )

    return metrics


class MonitoringMiddleware(BaseHTTPMiddleware):
    """
    Middleware for request/response monitoring.

    This middleware tracks:
    - Request latency
    - Request/response sizes
    - Status codes
    - Exceptions

    And exports them as Prometheus metrics.
    """

    def __init__(self, app: ASGIApp, exclude_paths: Optional[list] = None):
        super().__init__(app)
        self.exclude_paths = exclude_paths or []
        self.logger = logging.getLogger("sutazai_monitoring")

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Skip monitoring for excluded paths
        path = request.url.path
        if any(path.startswith(excluded) for excluded in self.exclude_paths):
            return await call_next(request)

        # Record request start time
        start_time = time.time()

        # Get request size
        try:
            request_size = int(request.headers.get("content-length", 0))
        except (ValueError, TypeError):
            request_size = 0

        # Process request
        status_code = 500  # Default in case of exception
        response = None
        exception = None
        component = "api"
        endpoint = path
        method = request.method

        try:
            # Execute the request
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            # Log and re-raise the exception
            exception = e
            capture_exception(
                e,
                {
                    "method": method,
                    "path": path,
                    "query_params": str(request.query_params),
                    "client": request.client.host if request.client else "unknown",
                },
            )
            raise
        finally:
            # Calculate duration
            duration = time.time() - start_time

            # Log request info
            log_request(
                component=component,
                endpoint=endpoint,
                duration=duration,
                status_code=status_code,
                error=exception,
            )

            # Update Prometheus metrics if available
            if PROMETHEUS_AVAILABLE:
                REQUEST_COUNT.labels(
                    method=method, endpoint=endpoint, status_code=status_code
                ).inc()

                REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(
                    duration
                )

                REQUEST_SIZE.labels(method=method, endpoint=endpoint).observe(
                    request_size
                )

                # Only record response size if we have a response
                if response and "content-length" in response.headers:
                    try:
                        response_size = int(response.headers["content-length"])
                        RESPONSE_SIZE.labels(
                            method=method, endpoint=endpoint, status_code=status_code
                        ).observe(response_size)
                    except (ValueError, TypeError):
                        pass

        return response


def setup_monitoring(app: FastAPI, exclude_paths: Optional[list] = None) -> None:
    """
    Set up monitoring for a FastAPI application.

    Args:
        app: FastAPI application
        exclude_paths: List of paths to exclude from monitoring
    """
    # Add monitoring middleware
    app.add_middleware(
        MonitoringMiddleware,
        exclude_paths=exclude_paths or ["/metrics", "/health", "/static"],
    )

    # If Prometheus is available, add metrics endpoint
    if PROMETHEUS_AVAILABLE:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        from fastapi.responses import Response

        @app.get("/metrics")
        async def metrics():
            # Gather process metrics before returning
            if PSUTIL_AVAILABLE:
                gather_system_metrics()

            return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

        # Add process metrics updating as background task
        @app.on_event("startup")
        async def start_metrics_gathering():
            if PSUTIL_AVAILABLE:
                gather_system_metrics()
