# SutazAI MCP Services - Consolidated Configuration
# Unified MCP server container for all services
# Last Updated: 2025-08-20

FROM node:18-alpine AS builder

# Install build dependencies
RUN apk add --no-cache \
    python3-dev \
    py3-pip \
    go \
    git \
    build-base \
    linux-headers

# Setup Go environment
ENV GOPATH=/tmp/go
ENV PATH=$PATH:$GOPATH/bin:/usr/local/go/bin

WORKDIR /build

# Copy all MCP service sources
COPY package*.json ./
RUN npm ci --only=production

# Copy service implementations
COPY src/ ./src/
COPY wrappers/ ./wrappers/

# Production stage
FROM node:18-alpine

# Install runtime dependencies
RUN apk add --no-cache \
    python3 \
    py3-pip \
    go \
    bash \
    curl \
    tini

# Create non-root user
RUN addgroup -g 1001 -S mcp && \
    adduser -S mcp -u 1001 -G mcp

# Create required directories
WORKDIR /opt/mcp
RUN mkdir -p \
    /opt/mcp/python \
    /opt/mcp/go/bin \
    /opt/mcp/wrappers \
    /var/lib/mcp \
    /var/log/mcp && \
    chown -R mcp:mcp /opt/mcp /var/lib/mcp /var/log/mcp

# Setup Python virtual environment
RUN python3 -m venv /opt/mcp/python && \
    /opt/mcp/python/bin/pip install --no-cache-dir --upgrade pip

# Copy application from builder
COPY --from=builder --chown=mcp:mcp /build/node_modules ./node_modules/
COPY --from=builder --chown=mcp:mcp /build/src ./src/
COPY --from=builder --chown=mcp:mcp /build/wrappers ./wrappers/
COPY --from=builder --chown=mcp:mcp /build/package.json ./

# Health check script
COPY --chown=mcp:mcp <<'EOF' /opt/mcp/health-check.sh
#!/bin/bash
set -e

# Check if service is responding
if ! curl -f -s http://localhost:${MCP_PORT:-4000}/health >/dev/null 2>&1; then
    echo "Health check failed"
    exit 1
fi

# Check memory usage
MEMORY_USAGE=$(cat /proc/meminfo | awk '/MemAvailable/ {print int($2/1024)}')
if [ "$MEMORY_USAGE" -lt 50 ]; then
    echo "Low memory: ${MEMORY_USAGE}MB"
    exit 1
fi

echo "Health check passed"
EOF

RUN chmod +x /opt/mcp/health-check.sh /opt/mcp/wrappers/*.sh

# Environment configuration
ENV NODE_ENV=production \
    NODE_OPTIONS="--max-old-space-size=512 --gc-interval=100" \
    MCP_SERVICE=unified \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=4000 \
    PYTHON_PATH=/opt/mcp/python \
    GO_PATH=/opt/mcp/go \
    MAX_INSTANCES=3

# Switch to non-root user
USER mcp

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD /opt/mcp/health-check.sh

# Expose port
EXPOSE 4000

# Use tini for proper signal handling
ENTRYPOINT ["/sbin/tini", "--"]

# Start unified MCP server
CMD ["node", "src/unified-server.js"]