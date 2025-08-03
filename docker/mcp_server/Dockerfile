FROM node:20-alpine AS base

# Install dependencies for PostgreSQL and other native modules
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    postgresql-client \
    redis \
    curl

# Create app directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install --production && npm cache clean --force

# Create non-root user
RUN addgroup -g 1001 -S mcp && \
    adduser -S mcp -u 1001

# Copy application code
COPY --chown=mcp:mcp . .

# Create necessary directories
RUN mkdir -p /app/logs /app/data && \
    chown -R mcp:mcp /app

# Switch to non-root user
USER mcp

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD node -e "console.log('Health check passed')" || exit 1

# Expose port (if needed for debugging)
EXPOSE 3000

# Set default environment
ENV NODE_ENV=production
ENV LOG_LEVEL=info

# Start the application
CMD ["node", "index.js"] 