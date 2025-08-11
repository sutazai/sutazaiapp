# SutazAI Node.js Base Image - Master Consolidated Version
# Consolidates Node.js-based service patterns into secure, optimized base
# Author: Dockerfile Consolidation Specialist
# Date: August 10, 2025
# Purpose: Single source of truth for all Node.js services

FROM node:18-slim as base

# System dependencies for Node.js services
RUN apt-get update && apt-get install -y \
    # Core essentials
    curl \
    wget \
    git \
    unzip \
    # Build tools for native modules
    build-essential \
    gcc \
    g++ \
    make \
    # Python integration (many Node.js services use Python for AI)
    python3 \
    python3-pip \
    python3-dev \
    # System utilities
    procps \
    htop \
    vim \
    # Network tools
    netcat-openbsd \
    iputils-ping \
    # Development libraries
    libffi-dev \
    libssl-dev \
    # Clean up
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && apt-get autoremove -y

# Install global Node.js packages
RUN npm install -g \
    pm2 \
    typescript \
    ts-node \
    nodemon \
    @types/node

# SECURITY: Create secure non-root user
RUN groupadd -r appuser && useradd -r -g appuser -s /bin/bash appuser \
    && mkdir -p /app /app/data /app/logs /app/workspace /app/node_modules \
    && chown -R appuser:appuser /app

# Production stage
FROM base as production

WORKDIR /app

# Core environment variables
ENV NODE_ENV=production \
    NPM_CONFIG_CACHE=/tmp/.npm \
    PATH=/app/node_modules/.bin:$PATH \
    SERVICE_PORT=3000 \
    NODE_OPTIONS="--max-old-space-size=1024" \
    UV_THREADPOOL_SIZE=4

# Health check template
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${SERVICE_PORT:-3000}/health || curl -f http://localhost:${SERVICE_PORT:-3000}/ || exit 1

# Switch to non-root user
USER appuser

# Default command
CMD ["node", "index.js"]