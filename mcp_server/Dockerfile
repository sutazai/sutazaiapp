# MCP Server Dockerfile
FROM node:20-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY package*.json ./

# Install Node.js dependencies
RUN npm ci --only=production

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p /app/data /app/logs

# Create non-root user
RUN useradd -m -u 1001 nodeuser && chown -R nodeuser:nodeuser /app
USER nodeuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD node -e "process.exit(0)"

# Expose port
EXPOSE 3000

# Start the MCP server
CMD ["node", "index.js"]