# Unified Memory Service - Docker Container
# Consolidates extended-memory and memory-bank-mcp functionality
FROM python:3.11-slim

# Set working directory
WORKDIR /opt/memory

# Install system dependencies
RUN apt-get update && apt-get install -y \
    sqlite3 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create MCP directories
RUN mkdir -p /opt/mcp/wrappers /var/lib/mcp /var/log/mcp

# Copy Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy unified memory service
COPY unified-memory-service.py .
COPY unified-memory-wrapper.sh /opt/mcp/wrappers/
RUN chmod +x /opt/mcp/wrappers/unified-memory-wrapper.sh

# Create non-root user for security
RUN groupadd -r mcp && useradd -r -g mcp mcp
RUN chown -R mcp:mcp /opt/memory /var/lib/mcp /var/log/mcp

# Switch to non-root user
USER mcp

# Set environment variables
ENV PYTHONPATH=/opt/memory
ENV MCP_SERVICE=unified-memory
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=3009
ENV SQLITE_PATH=/var/lib/mcp/unified_memory.db

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:3009/health || exit 1

# Expose port
EXPOSE 3009

# Start service
CMD ["python", "unified-memory-service.py"]