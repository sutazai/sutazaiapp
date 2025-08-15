# SutazAI Backend Service - ULTRAFIX Dockerfile Consolidation
# Migrated to use sutazai-python-agent-master base image
# Date: August 10, 2025 - CRITICAL DOCKERFILE DEDUPLICATION
# Security Fix: 2025-08-16 - Removed root user elevation for package installation
FROM sutazai-python-agent-master:v1.0.0

# Copy requirements with proper ownership (avoid root elevation)
# The base image already has appuser:appgroup configured
COPY --chown=appuser:appgroup requirements.txt /tmp/backend-requirements.txt

# Install backend-specific requirements as non-root user
# Using --user flag to install packages in user directory
USER appuser
RUN pip install --user --no-cache-dir -r /tmp/backend-requirements.txt && rm /tmp/backend-requirements.txt

# Copy application code with proper ownership
COPY --chown=appuser:appgroup . .

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

# Start backend service
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
