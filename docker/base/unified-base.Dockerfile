# Unified Base Image for SutazAI
# Combines Python, Node.js, AI/ML, Security, and Monitoring capabilities
FROM python:3.11-slim-bookworm AS base

# Metadata
LABEL maintainer="SutazAI Team"
LABEL version="1.0.0"
LABEL description="Unified base image for all SutazAI services"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive \
    NODE_ENV=production \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    TZ=UTC

# Install system dependencies and Node.js
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    # Network and system tools
    curl \
    wget \
    gnupg \
    ca-certificates \
    git \
    openssh-client \
    # Security tools
    openssl \
    libssl-dev \
    # Database clients
    postgresql-client \
    redis-tools \
    # Monitoring tools
    htop \
    iotop \
    sysstat \
    procps \
    # Python development
    python3-dev \
    # AI/ML dependencies
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    # Image processing
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Compression
    zip \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20.x
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g npm@latest \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r sutazai && useradd -r -g sutazai -m -s /bin/bash sutazai

# Create necessary directories
RUN mkdir -p /app /data /logs /tmp/cache \
    && chown -R sutazai:sutazai /app /data /logs /tmp/cache

# Install Python base packages
COPY --chown=sutazai:sutazai requirements-base.txt /tmp/
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r /tmp/requirements-base.txt \
    && rm /tmp/requirements-base.txt

# Install Node.js global packages for MCP support
RUN npm install -g \
    @modelcontextprotocol/sdk \
    typescript \
    ts-node \
    nodemon \
    pm2

# Security hardening
RUN chmod 700 /app \
    && chmod 750 /data /logs

# Health check script
COPY --chown=sutazai:sutazai healthcheck.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/healthcheck.sh

# Switch to non-root user
USER sutazai
WORKDIR /app

# Default health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ["/usr/local/bin/healthcheck.sh"]

# Multi-stage capability - can be extended by child images
ONBUILD COPY --chown=sutazai:sutazai requirements.txt /tmp/
ONBUILD RUN pip install --user --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Expose common ports (can be overridden in child images)
EXPOSE 8000 8080 3000 5000

# Default command (override in child images)
CMD ["python", "--version"]