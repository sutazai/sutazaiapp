# SutazAI MCP Server - ULTRAFIX Dockerfile Consolidation
# Migrated to use sutazai-nodejs-agent-master base image
# Date: August 10, 2025 - CRITICAL DOCKERFILE DEDUPLICATION
FROM sutazai-nodejs-agent-master:latest

# Install Docker client (note: requires proper socket mounting)
RUN apt-get update && apt-get install -y docker.io && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy service-specific package.json
COPY package*.json ./

# Install Node.js dependencies
RUN npm install --omit=dev && npm cache clean --force

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p /app/data /app/logs

# Override base environment variables for this service
ENV SERVICE_PORT=3030
ENV AGENT_ID=sutazai-mcp-server-root
ENV AGENT_NAME="SutazAI MCP Server Root"
ENV MCP_HTTP_PORT=3030

# Override health check for this service
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD node -e "fetch('http://localhost:'+(process.env.MCP_HTTP_PORT||3030)+'/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

# Expose HTTP health port
EXPOSE 3030

# Switch to non-root user (inherited from base)
USER appuser

# Start the MCP server
CMD ["node", "index.js"]