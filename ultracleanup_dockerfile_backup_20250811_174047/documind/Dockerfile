# SutazAI Documind Service - ULTRAFIX Dockerfile Consolidation
# Migrated to use sutazai-nodejs-agent-master base image
# Date: August 10, 2025 - CRITICAL DOCKERFILE DEDUPLICATION
FROM sutazai-nodejs-agent-master:latest

# Clone Documind from GitHub with proper permissions
RUN git clone --depth 1 https://github.com/DocumindHQ/documind.git /tmp/documind && \
    cd /tmp/documind && \
    chown -R appuser:appuser /tmp/documind /tmp/.npm 2>/dev/null || true && \
    npm ci --legacy-peer-deps --cache /tmp/npm-cache && \
    cp -r . /app/ && \
    chown -R appuser:appuser /app && \
    rm -rf /tmp/documind /tmp/npm-cache && \
    npm cache clean --force

# Override base environment variables for this service
ENV SERVICE_PORT=3000
ENV AGENT_ID=sutazai-documind
ENV AGENT_NAME="SutazAI Documind Service"

# Expose service port
EXPOSE 3000

# Switch to non-root user (inherited from base)
USER appuser

# Development service - runs indefinitely
CMD ["sleep", "infinity"]