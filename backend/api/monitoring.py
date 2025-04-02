"""
SutazAI Monitoring API

This module provides endpoints for monitoring the SutazAI system.
"""

import time
import psutil
import platform
from typing import Dict, Any
from datetime import datetime
from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# Define Prometheus metrics
REQUEST_COUNT = Counter(
    "sutazai_http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "sutazai_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
)

SYSTEM_MEMORY_USAGE = Gauge(
    "sutazai_system_memory_usage_bytes", "System memory usage in bytes"
)

SYSTEM_CPU_USAGE = Gauge(
    "sutazai_system_cpu_usage_percent", "System CPU usage percentage"
)

# Create router
router = APIRouter(prefix="/monitoring", tags=["monitoring"])


# Middleware to track metrics
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)

    # Record metrics
    duration = time.time() - start_time
    REQUEST_COUNT.labels(
        method=request.method, endpoint=request.url.path, status=response.status_code
    ).inc()

    REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(
        duration
    )

    return response


# Metrics endpoint
@router.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Expose Prometheus metrics."""
    # Update system metrics
    SYSTEM_MEMORY_USAGE.set(psutil.virtual_memory().used)
    SYSTEM_CPU_USAGE.set(psutil.cpu_percent())

    return PlainTextResponse(
        content=generate_latest().decode(), media_type=CONTENT_TYPE_LATEST
    )


# Health check endpoint
@router.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# System info endpoint
@router.get("/system")
async def system_info() -> Dict[str, Any]:
    """Get system information."""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage("/")

    return {
        "hostname": platform.node(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
        },
        "cpu": {
            "count": psutil.cpu_count(),
            "usage_percent": psutil.cpu_percent(),
        },
        "memory": {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent,
        },
        "disk": {
            "total": disk.total,
            "free": disk.free,
            "used": disk.used,
            "percent": disk.percent,
        },
        "timestamp": datetime.now().isoformat(),
    }
