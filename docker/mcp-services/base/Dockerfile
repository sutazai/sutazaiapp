# Base MCP Service Container
# Provides common HTTP wrapper functionality for MCP services

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    netcat-openbsd \
    dnsutils \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    consul-python \
    prometheus_client \
    structlog

# Create application directory
WORKDIR /opt/mcp

# Copy base MCP HTTP server
COPY mcp_http_server.py /opt/mcp/
COPY health_check.py /opt/mcp/
COPY consul_registration.py /opt/mcp/

# Create non-root user
RUN useradd -r -u 1000 -m -d /opt/mcp -s /bin/bash mcp
RUN chown -R mcp:mcp /opt/mcp
USER mcp

# Default environment variables
ENV MCP_SERVICE_NAME=mcp-service
ENV MCP_SERVICE_PORT=11100
ENV CONSUL_AGENT=mcp-consul-agent:8500
ENV LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python3 health_check.py

# Default command
CMD ["python3", "mcp_http_server.py"]