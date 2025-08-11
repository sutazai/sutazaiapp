# SutazAI BigAGI Service - ULTRAFIX Dockerfile Consolidation
# Migrated to use sutazai-nodejs-agent-master base image
# Date: August 10, 2025 - CRITICAL DOCKERFILE DEDUPLICATION
FROM sutazai-nodejs-agent-master:latest

# Clone BigAGI from GitHub
RUN git clone https://github.com/enricoros/big-AGI.git /tmp/bigagi && \
    cd /tmp/bigagi && \
    npm install && \
    npm run build && \
    cp -r . /app/ && \
    rm -rf /tmp/bigagi && \
    npm cache clean --force

# Create startup script
RUN echo '#!/bin/bash\nnpm start' > /app/start.sh && chmod +x /app/start.sh

# Override base environment variables for this service
ENV SERVICE_PORT=3000
ENV AGENT_ID=sutazai-bigagi
ENV AGENT_NAME="SutazAI BigAGI Service"

# Expose service port
EXPOSE 3000

# Switch to non-root user (inherited from base)
USER appuser

# Use startup script
CMD ["./start.sh"]
