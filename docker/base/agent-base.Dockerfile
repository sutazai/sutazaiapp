# Multi-stage AI Agent base image
# Stage 1: Build environment  
FROM python:3.12.8-slim-bookworm as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install AI/ML dependencies
COPY requirements-agent.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements-agent.txt

# Stage 2: Runtime environment
FROM python:3.12.8-slim-bookworm

# Security: Create non-root user for agents
RUN groupadd -r agent && useradd -r -g agent -s /bin/false agent

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Agent-specific environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AGENT_ENV=production \
    LOG_LEVEL=INFO \
    HEALTH_PORT=8080

# Create agent directories with proper permissions
RUN mkdir -p /app /app/shared /app/data /app/logs /app/config && \
    chown -R agent:agent /app

WORKDIR /app

# Copy shared agent components (these will be mounted as volumes)
# COPY shared/ ./shared/

# Switch to non-root user
USER agent

# Default health check for agents
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${HEALTH_PORT:-8080}/health || exit 1

# Default command (to be overridden)
CMD ["python", "-c", "print('Agent base image - override CMD in specific agent')"]