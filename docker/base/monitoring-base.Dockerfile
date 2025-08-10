# SutazAI Monitoring Base Image - ULTRA CONSOLIDATION
# Consolidates all monitoring, metrics, and observability containers
# Author: System Reorganization Expert - ULTRA Deduplication Operation
# Date: August 10, 2025
# Purpose: Single base for Prometheus exporters, metrics collectors, health checkers

FROM alpine:3.19 as base

# Install comprehensive monitoring tools and dependencies
RUN apk add --no-cache \
    # Core essentials for monitoring services
    curl \
    wget \
    bash \
    jq \
    # Network monitoring tools
    netcat-openbsd \
    ping \
    traceroute \
    tcpdump \
    # System monitoring
    htop \
    procps \
    lsof \
    # SSL/TLS for secure metrics collection
    ca-certificates \
    openssl \
    # Python for custom metrics scripts
    python3 \
    python3-dev \
    py3-pip \
    # Build tools for Python packages
    gcc \
    musl-dev \
    libffi-dev \
    # Time utilities for metrics timestamps
    tzdata \
    # Text processing for log analysis
    awk \
    sed \
    grep

# Install monitoring Python packages
RUN pip3 install --no-cache-dir --break-system-packages \
    # Prometheus client for metrics export
    prometheus-client==0.19.0 \
    # HTTP client for health checks
    requests==2.31.0 \
    # Fast HTTP client
    httpx==0.25.2 \
    # Structured logging
    structlog==23.2.0 \
    # System metrics
    psutil==5.9.6 \
    # Configuration management
    pyyaml==6.0.1 \
    # Environment variables
    python-dotenv==1.0.0 \
    # Async capabilities
    aiohttp==3.9.1

# SECURITY: Create monitoring user with appropriate permissions
RUN addgroup -g 1000 monitor && \
    adduser -D -u 1000 -G monitor monitor && \
    mkdir -p /app /app/data /app/logs /app/metrics /app/config \
    && chown -R monitor:monitor /app

# Production stage - ULTRA OPTIMIZED for monitoring services
FROM base as production

WORKDIR /app

# COMPREHENSIVE environment variables for monitoring services
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Service configuration
    SERVICE_PORT=9090 \
    METRICS_PORT=9090 \
    HEALTH_PORT=9091 \
    # Monitoring configuration
    SCRAPE_INTERVAL=15s \
    EVALUATION_INTERVAL=15s \
    # Performance settings
    GOMAXPROCS=2 \
    # Logging configuration
    LOG_LEVEL=info \
    LOG_FORMAT=json \
    # Timezone
    TZ=UTC

# Create standard monitoring directories
RUN mkdir -p \
    /app/metrics \
    /app/logs \
    /app/config \
    /app/data \
    && chown -R monitor:monitor /app

# FLEXIBLE health check for monitoring services
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${HEALTH_PORT:-9091}/health || \
        curl -f http://localhost:${METRICS_PORT:-9090}/metrics || \
        curl -f http://localhost:${SERVICE_PORT:-9090}/ || exit 1

# SECURITY: Switch to non-root monitoring user
USER monitor

# Default ports for monitoring services
EXPOSE 9090 9091

# FLEXIBLE default command for monitoring services
CMD ["python3", "-u", "monitor.py"]