# SutazAI Production Base Image - ULTRA CONSOLIDATION
# Multi-stage optimized builds for production deployments
# Author: System Reorganization Expert - ULTRA Deduplication Operation
# Date: August 10, 2025
# Purpose: Minimal attack surface, maximum security, production-ready base

FROM python:3.12.8-slim-bookworm as builder

# Install build dependencies only in builder stage
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    make \
    # Development headers
    python3-dev \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    # Git for dependency installation
    git \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment for cleaner Python installation
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install comprehensive production Python packages
COPY base-requirements.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /tmp/base-requirements.txt

# PRODUCTION STAGE - Minimal runtime image
FROM python:3.12.8-slim-bookworm as production

# Install only essential runtime dependencies
RUN apt-get update && apt-get install -y \
    # Minimal runtime essentials
    curl \
    # SSL certificates for HTTPS
    ca-certificates \
    # Process monitoring
    procps \
    # Network utilities
    netcat-openbsd \
    # Cleanup to minimize attack surface
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# MAXIMUM SECURITY: Create dedicated production user with minimal privileges
RUN groupadd -r -g 1000 sutazai && \
    useradd -r -g sutazai -u 1000 -s /bin/false -M sutazai && \
    mkdir -p /app /app/data /app/logs && \
    chown -R sutazai:sutazai /app && \
    # Remove shell access for security
    usermod -s /sbin/nologin sutazai && \
    # Lock the account
    usermod -L sutazai

WORKDIR /app

# PRODUCTION environment variables - security and performance optimized
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Security settings
    PYTHONHASHSEED=random \
    # Performance optimization
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    # Service configuration
    SERVICE_PORT=8080 \
    # Logging for production
    LOG_LEVEL=warning \
    LOG_FORMAT=json \
    # Resource limits
    MALLOC_ARENA_MAX=2 \
    # Security headers
    SECURE_SSL_REDIRECT=true \
    SECURE_HSTS_SECONDS=31536000

# Production health check with shorter intervals
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${SERVICE_PORT:-8080}/health || exit 1

# MAXIMUM SECURITY: Switch to non-root user with no shell
USER sutazai

# Production-ready command
CMD ["python", "-u", "app.py"]