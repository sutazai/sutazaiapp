# SutazAI Skyvern Service - ULTRAFIX Dockerfile Consolidation  
# Migrated to use sutazai-python-agent-master base image
# Date: August 10, 2025 - CRITICAL DOCKERFILE DEDUPLICATION
FROM sutazai-python-agent-master:latest

# Install Playwright + Chromium on top of base
RUN pip install --no-cache-dir playwright && \
    python -m playwright install --with-deps chromium

# Clone and install Skyvern from GitHub
RUN git clone https://github.com/Skyvern-AI/skyvern.git /tmp/skyvern && \
    cd /tmp/skyvern && \
    pip install --no-cache-dir -e . && \
    cp -r . /app/ && \
    rm -rf /tmp/skyvern

# Override base environment variables for this service
ENV SERVICE_PORT=8080
ENV AGENT_ID=sutazai-skyvern
ENV AGENT_NAME="SutazAI Skyvern Service"

# Override health check for this service
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || curl -f http://localhost:8080/ || exit 1

# Expose service port
EXPOSE 8080

# Switch to non-root user (inherited from base)
USER appuser

# Use default Python entry point (services can override with specific commands)
CMD ["python", "-u", "app.py"]