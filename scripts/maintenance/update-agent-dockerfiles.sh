#!/bin/bash

# Strict error handling
set -euo pipefail

# Purpose: Update all agent Dockerfiles to use proper base image
# Usage: ./update-agent-dockerfiles.sh
# Requires: bash, find, sed


# Signal handlers for graceful shutdown
cleanup_and_exit() {
    local exit_code="${1:-0}"
    echo "Script interrupted, cleaning up..." >&2
    # Clean up any background processes
    jobs -p | xargs -r kill 2>/dev/null || true
    exit "$exit_code"
}

trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM
trap 'cleanup_and_exit 1' ERR

echo "Updating agent Dockerfiles..."

# Find all Dockerfiles in agent directories that use the wrong base image
find /opt/sutazaiapp/agents -name "Dockerfile" -type f | while read dockerfile; do
    if grep -q "FROM sutazai/python-agent:latest" "$dockerfile"; then
        echo "Updating: $dockerfile"
        # Replace the FROM line
        sed -i '1s|FROM sutazai/python-agent:latest|FROM python:3.11-slim|' "$dockerfile"
        
        # Add the full content after the WORKDIR line
        sed -i '/WORKDIR \/app/a\
\
# Install system dependencies\
RUN apt-get update && apt-get install -y \\\
    curl \\\
    && rm -rf /var/lib/apt/lists/*\
\
# Install base Python packages\
RUN pip install --no-cache-dir \\\
    fastapi==0.104.1 \\\
    uvicorn==0.24.0 \\\
    pydantic==2.5.0 \\\
    httpx==0.25.2 \\\
    python-dotenv==1.0.0 \\\
    redis==5.0.1 \\\
    prometheus-client==0.19.0 \\\
    psutil==5.9.6 \\\
    structlog==23.2.0' "$dockerfile"
        
        # Update the user creation section
        sed -i 's|USER root|# Create non-root user\
RUN groupadd -r agent \&\& useradd -r -g agent agent|' "$dockerfile"
        
        # Add health check before the ENV section
        sed -i '/ENV AGENT_NAME=/i\
# Health check\
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \\\
    CMD curl -f http://localhost:8080/health || exit 1\
\
# Set environment variables\
ENV PYTHONUNBUFFERED=1' "$dockerfile"
        
        # Add EXPOSE before CMD
        sed -i '/CMD \[/i\
EXPOSE 8080\
' "$dockerfile"
    fi
done

echo "Dockerfile updates complete!"