# SutazAI Frontend Service - ULTRAFIX Dockerfile Consolidation  
# Migrated to use sutazai-python-agent-master base image
# Date: August 10, 2025 - CRITICAL DOCKERFILE DEDUPLICATION
FROM sutazai-python-agent-master:latest

# Ensure we can install Python packages system-wide
USER root

# Install frontend-specific requirements on top of base  
# Note: Copy from context root, not relative path
# Use local optimized requirements within frontend context
COPY requirements_optimized.txt /tmp/frontend-requirements.txt
RUN pip install --no-cache-dir -r /tmp/frontend-requirements.txt && rm /tmp/frontend-requirements.txt

# Copy application and components
COPY app.py .
COPY components/ ./components/
COPY pages/ ./pages/

# Override base environment variables for frontend service
ENV SERVICE_PORT=8501
ENV AGENT_ID=sutazai-frontend
ENV AGENT_NAME="SutazAI Frontend Service"
ENV PORT=8501

# Override health check for frontend service
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/health || exit 1

# Expose frontend port
EXPOSE 8501

# Switch back to non-root user (inherited from base)
USER appuser

# Start frontend service
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
