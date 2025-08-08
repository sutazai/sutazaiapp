# MCP Server Dockerfile
FROM node:20-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY package*.json ./

# Install Node.js dependencies (without requiring a lockfile)
RUN npm install --omit=dev

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p /app/data /app/logs

# Create non-root user
RUN useradd -m -u 1001 nodeuser && chown -R nodeuser:nodeuser /app
USER nodeuser

# Health check hits HTTP health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD node -e "fetch('http://localhost:'+(process.env.MCP_HTTP_PORT||3030)+'/health').then(r=>process.exit(r.ok?0:1)).catch(()=>process.exit(1))"

# Expose HTTP health port
EXPOSE 3030

# Start the MCP server
CMD ["node", "index.js"]
