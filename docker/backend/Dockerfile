# SutazAI Backend Service - Rule 11 Compliant
# Secure, non-root, pinned versions
# Last Updated: 2025-01-15
FROM python:3.11.9-alpine3.19

# Security: Create non-root user
RUN addgroup -g 1000 -S appgroup && \
    adduser -u 1000 -S appuser -G appgroup -h /home/appuser -s /bin/sh

# Install system dependencies and security updates
RUN apk update && \
    apk upgrade && \
    apk add --no-cache \
        build-base \
        curl \
        git \
        postgresql-client \
        ca-certificates \
        linux-headers \
        gcc \
        musl-dev \
        libffi-dev && \
    rm -rf /var/cache/apk/*

WORKDIR /app

# Copy requirements as root for installation
COPY requirements.txt /tmp/backend-requirements.txt

# Install Python dependencies as root (required for system packages)
RUN pip install --no-cache-dir -r /tmp/backend-requirements.txt && \
    rm /tmp/backend-requirements.txt

# Switch to non-root user before copying app code
USER appuser

# Copy application code with proper ownership
COPY --chown=appuser:appgroup . /app/

# Override base environment variables for backend service
ENV SERVICE_PORT=8000
ENV AGENT_ID=sutazai-backend
ENV AGENT_NAME="SutazAI Backend Service"
ENV DEBUG=false

# Override health check for backend service
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health').raise_for_status()"

# Expose backend port
EXPOSE 8000

# Ensure we're running as non-root user (already set above, but explicit for clarity)
USER appuser

# Set PYTHONPATH to ensure app module can be found
ENV PYTHONPATH="/app:$PYTHONPATH"

# Set working directory
WORKDIR /app

# Start backend service
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
