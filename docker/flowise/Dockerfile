# SutazAI Flowise Service - ULTRAFIX Dockerfile Consolidation
# Migrated to use sutazai-nodejs-agent-master base image
# Date: August 10, 2025 - CRITICAL DOCKERFILE DEDUPLICATION
FROM sutazai-nodejs-agent-master:latest

# Copy service-specific package.json
COPY package.json ./

# Install service dependencies (Alpine packages through apt)
RUN apt-get update && apt-get install -y python3 make g++ && \
    npm install --production && \
    npm cache clean --force && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy service files
COPY flowise_service.js .

# Override base environment variables for this service
ENV SERVICE_PORT=3000
ENV AGENT_ID=sutazai-flowise
ENV AGENT_NAME="SutazAI Flowise Service"
ENV FLOWISE_USERNAME=admin
ENV FLOWISE_PASSWORD=sutazai123
ENV PORT=3000

# Expose service port
EXPOSE 3000

# Switch to non-root user (inherited from base)
USER appuser

# Use npm start command
CMD ["npm", "start"]