# MCP Manager - Container orchestration interface
FROM python:3.11-alpine3.19

# Security: Create non-root user and docker group for socket access
# Note: GID 2375 must match the docker group in the DinD orchestrator
RUN addgroup -g 2375 docker && \
    addgroup -g 1001 mcpmanager && \
    adduser -D -u 1001 -G mcpmanager mcpmanager && \
    adduser mcpmanager docker

# Install Docker client and dependencies
RUN apk add --no-cache \
    docker-cli \
    curl \
    bash \
    jq \
    && rm -rf /var/cache/apk/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ .
COPY scripts/ /scripts/

# Make scripts executable
RUN chmod +x /scripts/*.sh

# Create necessary directories
RUN mkdir -p /var/log/manager && \
    chown -R mcpmanager:mcpmanager /app /var/log/manager /scripts

# Switch to non-root user
USER mcpmanager

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Expose management port
EXPOSE 8081

# Default command
CMD ["python", "manager.py"]