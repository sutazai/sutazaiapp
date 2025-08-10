# SutazAI Backend Service - ULTRAFIX Dockerfile Consolidation
# Migrated to use sutazai-python-agent-master base image
# Date: August 10, 2025 - CRITICAL DOCKERFILE DEDUPLICATION
FROM sutazai-python-agent-master:latest

# Install backend-specific requirements on top of base
COPY requirements.txt /tmp/backend-requirements.txt
RUN pip install --no-cache-dir -r /tmp/backend-requirements.txt && rm /tmp/backend-requirements.txt

# Copy application code
COPY . .

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

# Switch to non-root user (inherited from base)
USER appuser

# Start backend service
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]