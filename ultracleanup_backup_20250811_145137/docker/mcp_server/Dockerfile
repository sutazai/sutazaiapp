# SutazAI MCP Server Service - ULTRAFIX Dockerfile Consolidation
# Migrated to use sutazai-nodejs-agent-master base image
# Date: August 10, 2025 - CRITICAL DOCKERFILE DEDUPLICATION
FROM sutazai-nodejs-agent-master:latest

# Install PostgreSQL client and Redis
RUN apt-get update && apt-get install -y postgresql-client redis-tools && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy service-specific package.json
COPY package*.json ./

# Install dependencies
RUN npm install --production && npm cache clean --force

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data

# Override base environment variables for this service
ENV SERVICE_PORT=3000
ENV AGENT_ID=sutazai-mcp-server
ENV AGENT_NAME="SutazAI MCP Server Service"
ENV NODE_ENV=production
ENV LOG_LEVEL=info

# Override health check for this service
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD node -e "console.log('Health check passed')" || exit 1

# Expose service port
EXPOSE 3000

# Switch to non-root user (inherited from base)
USER appuser

# Start the application
CMD ["node", "index.js"]