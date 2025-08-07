# Optimized Agent Base Image
# Pre-installs all common dependencies to eliminate startup package installation
FROM python:3.11-slim

LABEL maintainer="SutazAI Platform"
LABEL description="Optimized base image for SutazAI agents with pre-installed dependencies"
LABEL version="1.0"

# Set working directory
WORKDIR /app

# Set environment variables for optimization
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements file
COPY agent-requirements.txt /tmp/

# Install Python dependencies in one layer for optimal caching
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/agent-requirements.txt && \
    rm /tmp/agent-requirements.txt

# Create agent user for security
RUN groupadd -r agent && useradd -r -g agent -s /bin/bash agent

# Create necessary directories and set permissions
RUN mkdir -p /app/logs /app/data /app/config && \
    chown -R agent:agent /app

# Create health check script
RUN echo '#!/bin/bash\ncurl -f http://localhost:8080/health || exit 1' > /usr/local/bin/health-check.sh && \
    chmod +x /usr/local/bin/health-check.sh

# Set default health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /usr/local/bin/health-check.sh

# Switch to non-root user
USER agent

# Expose default port
EXPOSE 8080

# Default command - to be overridden by specific agents
CMD ["python", "container_startup.py"]