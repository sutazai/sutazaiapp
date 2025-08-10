# SutazAI Python Base Image - Master Consolidated Version
# Consolidates Python-based service patterns into secure, optimized base
# Author: Dockerfile Consolidation Specialist
# Date: August 10, 2025
# Purpose: Single source of truth for all Python services

FROM python:3.12.8-slim-bookworm as base

# System dependencies for core Python services
RUN apt-get update && apt-get install -y \
    # Core essentials
    curl \
    wget \
    git \
    unzip \
    # Build tools for compiled packages
    build-essential \
    gcc \
    g++ \
    make \
    # System utilities
    procps \
    htop \
    vim \
    # Network tools
    netcat-openbsd \
    iputils-ping \
    # Development libraries
    python3-dev \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    # ML/AI dependencies
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    gfortran \
    # Clean up in one layer
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Install core Python packages
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# SECURITY: Create secure non-root user
RUN groupadd -r appuser && useradd -r -g appuser -s /bin/bash appuser \
    && mkdir -p /app /app/data /app/logs /app/workspace /app/models /app/temp \
    && chown -R appuser:appuser /app

# Production stage
FROM base as production

WORKDIR /app

# Core environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SERVICE_PORT=8080 \
    PYTHONIOENCODING=utf-8 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Health check template
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${SERVICE_PORT:-8080}/health || curl -f http://localhost:${SERVICE_PORT:-8080}/ || exit 1

# Switch to non-root user
USER appuser

# Default command
CMD ["python", "-u", "app.py"]