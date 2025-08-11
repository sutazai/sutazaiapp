# SutazAI Docker Service - ULTRAFIX Dockerfile Consolidation
# Migrated to use sutazai-nodejs-agent-master base image
# Date: August 10, 2025 - CRITICAL DOCKERFILE DEDUPLICATION
FROM sutazai-nodejs-agent-master:latest

# Copy service-specific package.json if it exists
COPY package*.json ./

# Install dependencies including Playwright
RUN npm install && \
    npx playwright install && \
    npx playwright install-deps && \
    npm cache clean --force

# Copy application files
COPY . .

# Override base environment variables for this service
ENV SERVICE_PORT=3000
ENV AGENT_ID=sutazai-docker-service
ENV AGENT_NAME="SutazAI Docker Service"

# Override health check for this service
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:3000/health || curl -f http://localhost:3000/ || exit 1

# Expose service port
EXPOSE 3000

# Switch to non-root user (inherited from base)
USER appuser

# Use Node.js entry point
CMD ["node", "index.js"]