# Unified Development Service Dockerfile
# Consolidates ultimatecoder, language-server, and sequentialthinking
# Target: 512MB memory usage, multi-language support

FROM node:18-alpine AS base

# Install system dependencies for multi-language support
RUN apk add --no-cache \
    python3 \
    python3-dev \
    py3-pip \
    go \
    git \
    curl \
    bash \
    build-base \
    linux-headers

# Create app directory and user
WORKDIR /opt/mcp
RUN addgroup -g 1001 -S mcp && adduser -S mcp -u 1001 -G mcp

# Setup Python environment
RUN python3 -m venv /opt/mcp/python && \
    /opt/mcp/python/bin/pip install --no-cache-dir --upgrade pip setuptools wheel

# Setup Go environment
ENV GOPATH=/opt/mcp/go
ENV PATH=$PATH:$GOPATH/bin:/usr/local/go/bin
RUN mkdir -p $GOPATH/bin

# Create wrapper scripts directory
RUN mkdir -p /opt/mcp/wrappers

# Copy package files
COPY package.json ./
RUN npm install --production --no-optional && \
    npm cache clean --force

# Copy application source
COPY src/ ./src/
COPY config/ ./config/
COPY scripts/ ./scripts/

# Create required directories
RUN mkdir -p /var/lib/mcp /var/log/mcp && \
    chown -R mcp:mcp /opt/mcp /var/lib/mcp /var/log/mcp

# Health check script
COPY health-check.sh /opt/mcp/health-check.sh

RUN chmod +x /opt/mcp/health-check.sh

# Set resource limits and environment
ENV NODE_ENV=production
ENV NODE_OPTIONS="--max-old-space-size=512"
ENV MCP_SERVICE=unified-dev
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=4000
ENV PYTHON_PATH=/opt/mcp/python
ENV GO_PATH=/opt/mcp/go
ENV MAX_INSTANCES=3

# Expose port
EXPOSE 4000

# Switch to non-root user
USER mcp

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD /opt/mcp/health-check.sh

# Start command
CMD ["node", "src/unified-dev-server.js"]