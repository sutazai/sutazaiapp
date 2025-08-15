# SutazAI Frontend Service - ULTRAFIX Dockerfile Consolidation  
# Migrated to use sutazai-python-agent-master base image
# Date: August 10, 2025 - CRITICAL DOCKERFILE DEDUPLICATION
# Security Fix: 2025-08-16 - Removed root user elevation for package installation
FROM sutazai-python-agent-master:v1.0.0

# Copy requirements with proper ownership (avoid root elevation)
# The base image already has appuser:appgroup configured
COPY --chown=appuser:appgroup requirements_optimized.txt /tmp/frontend-requirements.txt

# Install frontend-specific requirements as non-root user
# Using --user flag to install packages in user directory
USER appuser
RUN pip install --user --no-cache-dir -r /tmp/frontend-requirements.txt && rm /tmp/frontend-requirements.txt

# Copy application and components with proper ownership
COPY --chown=appuser:appgroup app.py .
COPY --chown=appuser:appgroup components/ ./components/
COPY --chown=appuser:appgroup pages/ ./pages/

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

# Ensure we're running as non-root user (already set above, but explicit for clarity)
USER appuser

# Start frontend service
CMD ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
